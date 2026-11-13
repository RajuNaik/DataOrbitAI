"""
DataOrbitAI - Transformers Explained, Part 2
CAPTURE SCRIPT: runs the real demo steps with distilgpt2 and saves every
actual result to  outputs/execution_results.json

Run:  python capture_outputs.py
Then upload outputs/execution_results.json.

Nothing in the JSON is typed by hand: every value is read from the model,
the tokenizer, or the tensors produced by this run.
"""

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = "outputs"
OUT_FILE = os.path.join(OUT_DIR, "execution_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


def to_list(t, decimals=6):
    """Tensor -> nested Python lists (rounded)."""
    return torch.round(t.detach().cpu().double() * 10**decimals).div(10**decimals).tolist()


R = {}  # everything we capture goes in here

# ---------------------------------------------------------------- environment
R["environment"] = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "device": "cpu",
    "cuda_available": bool(torch.cuda.is_available()),
}

# ----------------------------------------------------------- 1. load the model
model_name = "distilgpt2"
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
model.eval()
load_seconds = round(time.time() - t0, 2)

total_params = sum(p.numel() for p in model.parameters())
R["model"] = {
    "model_name": model_name,
    "model_class": type(model).__name__,
    "tokenizer_class": type(tokenizer).__name__,
    "load_seconds": load_seconds,
    "attn_implementation": getattr(model.config, "_attn_implementation", None),
    "total_parameters": total_params,
    "wte_parameters": model.transformer.wte.weight.numel(),
    "wpe_parameters": model.transformer.wpe.weight.numel(),
    "one_block_parameters": sum(p.numel() for p in model.transformer.h[0].parameters()),
    "config_printed": str(model.config),
    "config_dict": model.config.to_dict(),
    "n_layer": model.config.n_layer,
    "n_head": model.config.n_head,
    "n_embd": model.config.n_embd,
    "vocab_size": model.config.vocab_size,
    "n_positions": model.config.n_positions,
}

# ------------------------------------------------------------- 2. tokenization
text = "Transformers are powerful models for language understanding."
tokens = tokenizer.tokenize(text)
input_ids = tokenizer(text, return_tensors="pt")

R["tokenization"] = {
    "text": text,
    "tokens": tokens,
    "num_tokens": len(tokens),
    "token_ids": input_ids["input_ids"].tolist(),
    "token_ids_shape": list(input_ids["input_ids"].shape),
    "attention_mask": input_ids["attention_mask"].tolist(),
    "id_to_token": [
        {"id": i, "token": tokenizer.convert_ids_to_tokens(i)}
        for i in input_ids["input_ids"][0].tolist()
    ],
    "printed_token_ids": str(input_ids["input_ids"]),
    "printed_attention_mask": str(input_ids["attention_mask"]),
}

# ------------------------------------------------------------- 3. embeddings
with torch.no_grad():
    token_embeddings = model.transformer.wte(input_ids["input_ids"])

position_ids = torch.arange(input_ids["input_ids"].size(1)).unsqueeze(0)
with torch.no_grad():
    positional_embeddings = model.transformer.wpe(position_ids)

input_representation = token_embeddings + positional_embeddings

R["embeddings"] = {
    "token_embedding_shape": list(token_embeddings.shape),
    "positional_embedding_shape": list(positional_embeddings.shape),
    "input_representation_shape": list(input_representation.shape),
    "position_ids": position_ids.tolist(),
    # First 8 dimensions of every token's vectors (full vectors are 768 numbers)
    "token_embedding_first8_dims": to_list(token_embeddings[0, :, :8]),
    "positional_embedding_first8_dims": to_list(positional_embeddings[0, :, :8]),
    "input_representation_first8_dims": to_list(input_representation[0, :, :8]),
    "token_embedding_l2_norms": to_list(token_embeddings[0].norm(dim=-1)),
    "positional_embedding_l2_norms": to_list(positional_embeddings[0].norm(dim=-1)),
}

# ------------------------------------------------------ 4. architecture in code
R["architecture"] = {
    "n_layer": model.config.n_layer,
    "n_head": model.config.n_head,
    "n_embd": model.config.n_embd,
    "head_dim": model.config.n_embd // model.config.n_head,
    "mlp_inner_dim": model.config.n_inner or 4 * model.config.n_embd,
    "printed_blocks_full": str(model.transformer.h),
    "printed_block_0": str(model.transformer.h[0]),
    "top_level_modules": [name for name, _ in model.named_children()],
    "transformer_children": [name for name, _ in model.transformer.named_children()],
}

# ------------------------------------------------ 5. hidden states + attention
with torch.no_grad():
    outputs = model(**input_ids, output_attentions=True, output_hidden_states=True)

attentions = outputs.attentions
hidden_states = outputs.hidden_states
attn_available = attentions is not None and all(a is not None for a in attentions)

R["hidden_states_and_attention"] = {
    "attentions_returned": attn_available,
    "num_hidden_state_outputs": len(hidden_states),
    "num_attention_outputs": len(attentions) if attentions is not None else 0,
    "hidden_state_shapes": [list(h.shape) for h in hidden_states],
    "first_attention_tensor_shape": list(attentions[0].shape) if attn_available else None,
    "all_attention_tensor_shapes": [list(a.shape) for a in attentions] if attn_available else None,
    # Mean L2 norm of the token vectors after each layer (index 0 = embeddings)
    "hidden_state_mean_l2_norm_per_layer": [
        round(h[0].norm(dim=-1).mean().item(), 4) for h in hidden_states
    ],
    # Sanity check: hidden_states[0] should equal token + positional embedding
    "hidden_state0_equals_wte_plus_wpe_max_abs_diff": float(
        (hidden_states[0] - input_representation).abs().max().item()
    ),
}

if attn_available:
    stacked = torch.stack([a[0] for a in attentions])  # (layers, heads, seq, seq)
    seq = stacked.size(-1)
    upper = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    R["hidden_states_and_attention"].update({
        "stacked_attention_shape_layers_heads_seq_seq": list(stacked.shape),
        # Causal check: weights above the diagonal (future tokens) should be 0
        "causal_check_max_weight_above_diagonal": float(stacked[..., upper].max().item()),
        # Softmax check: every row should sum to 1
        "row_sum_min": float(stacked.sum(-1).min().item()),
        "row_sum_max": float(stacked.sum(-1).max().item()),
        "attention_tokens": tokens,
        # Full real attention weights: [layer][head][query_token][key_token]
        "attention_weights": to_list(stacked, decimals=6),
        # Layer 0 head 0, shown as in the script printout
        "layer0_head0_printed": str(attentions[0][0, 0].numpy().round(2)),
        # Layer-averaged-over-heads maps, handy for one clear heatmap per layer
        "mean_over_heads_per_layer": to_list(stacked.mean(dim=1), decimals=6),
    })

# ------------------------------------------------- 6. logits + next token
with torch.no_grad():
    outputs2 = model(**input_ids)
logits = outputs2.logits
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)
probabilities = torch.softmax(next_token_logits, dim=-1)
top_p, top_i = torch.topk(probabilities, k=10, dim=-1)
top_l = next_token_logits[0, top_i[0]]

R["logits"] = {
    "logits_shape": list(logits.shape),
    "next_token_logits_shape": list(next_token_logits.shape),
    "predicted_next_token_id": int(next_token_id.item()),
    "predicted_next_token": tokenizer.decode(next_token_id),
    "predicted_next_token_logit": round(float(next_token_logits[0, next_token_id.item()]), 4),
    "predicted_next_token_probability": round(float(probabilities[0, next_token_id.item()]), 6),
    "probabilities_sum": float(probabilities.sum().item()),
    "top10": [
        {
            "rank": r + 1,
            "token_id": int(top_i[0, r]),
            "token": tokenizer.decode(top_i[0, r]),
            "logit": round(float(top_l[r]), 4),
            "probability": round(float(top_p[0, r]), 6),
        }
        for r in range(10)
    ],
}

# ------------------------------------------------- 7. generation (as in video)
prompt = "Data engineering is evolving because"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
n_prompt = inputs["input_ids"].size(1)

R["generation"] = {
    "prompt": prompt,
    "prompt_token_ids": inputs["input_ids"].tolist(),
    "prompt_tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()),
    "max_new_tokens": 40,
    "do_sample": False,
    "generated_token_ids": generated[0].tolist(),
    "num_prompt_tokens": n_prompt,
    "num_total_tokens": int(generated.size(1)),
    "generated_text_full": generated_text,
    "generated_text_new_only": tokenizer.decode(generated[0][n_prompt:], skip_special_tokens=True),
}

# ------------------------------------------------- 8. manual autoregressive loop
# Same greedy process as generate(), written out token by token. Records the
# top-3 candidates at each step so the video can show real decisions.
steps = []
current = inputs["input_ids"]
for step in range(1, 41):
    with torch.no_grad():
        step_logits = model(input_ids=current).logits[:, -1, :]
    step_probs = torch.softmax(step_logits, dim=-1)
    new_id = torch.argmax(step_logits, dim=-1, keepdim=True)
    tp, ti = torch.topk(step_probs, k=3, dim=-1)
    steps.append({
        "step": step,
        "context_length": int(current.size(1)),
        "new_token_id": int(new_id.item()),
        "new_token": tokenizer.decode(new_id[0]),
        "new_token_probability": round(float(step_probs[0, new_id.item()]), 6),
        "top3": [
            {"token": tokenizer.decode(ti[0, k]), "token_id": int(ti[0, k]),
             "probability": round(float(tp[0, k]), 6)}
            for k in range(3)
        ],
    })
    current = torch.cat([current, new_id], dim=-1)

R["autoregressive_steps"] = {
    "steps": steps,
    "manual_text": tokenizer.decode(current[0], skip_special_tokens=True),
    "manual_matches_generate": bool(torch.equal(current[0], generated[0])),
}

# ------------------------------------------------------------------ save
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False, default=str)

size_kb = os.path.getsize(OUT_FILE) / 1024
print(f"Saved {OUT_FILE} ({size_kb:.0f} KB)")
print("Model:", model_name, "| attention returned:", attn_available)
print("Tokens:", tokens)
print("Predicted next token:", repr(R["logits"]["predicted_next_token"]))
print("Generated:", generated_text)
print("Manual loop matches generate():", R["autoregressive_steps"]["manual_matches_generate"])
print("\nDone. Please upload", OUT_FILE)
