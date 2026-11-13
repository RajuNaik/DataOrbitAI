"""
DataOrbitAI - Transformers Explained, Part 2
Practical Python implementation with Hugging Face (model: distilgpt2)

Run:  python transformer_part2_demo.py
Runs on CPU. The first run downloads the model (~350 MB) and caches it.
"""

# ---------------------------------------------------------------------------
# 1. Load the tokenizer and the pretrained model
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# attn_implementation="eager" makes sure attention weights can be returned.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
)

model.eval()  # inference mode: turns off dropout

print("=" * 70)
print("1. MODEL CONFIGURATION")
print("=" * 70)
print(model.config)

# ---------------------------------------------------------------------------
# 2. Tokenization: text -> tokens -> token IDs
#    (done by the tokenizer, NOT by the neural network)
# ---------------------------------------------------------------------------
text = "Transformers are powerful models for language understanding."

tokens = tokenizer.tokenize(text)

input_ids = tokenizer(
    text,
    return_tensors="pt"
)

print("\n" + "=" * 70)
print("2. TOKENIZATION")
print("=" * 70)
print("Tokens:", tokens)
print("Token IDs:", input_ids["input_ids"])
print("Attention mask:", input_ids["attention_mask"])

# ---------------------------------------------------------------------------
# 3. Token embeddings and positional embeddings
#    (GPT-2 family uses LEARNED positional embeddings; other Transformer
#     architectures can use different positional schemes.)
# ---------------------------------------------------------------------------
with torch.no_grad():
    token_embeddings = model.transformer.wte(
        input_ids["input_ids"]
    )

print("\n" + "=" * 70)
print("3. EMBEDDINGS")
print("=" * 70)
print("Token embedding shape:", token_embeddings.shape)

position_ids = torch.arange(
    input_ids["input_ids"].size(1)
).unsqueeze(0)

with torch.no_grad():
    positional_embeddings = model.transformer.wpe(
        position_ids
    )

print("Positional embedding shape:", positional_embeddings.shape)

# Input representation = Token embedding + Positional embedding
input_representation = token_embeddings + positional_embeddings
print("Input representation shape:", input_representation.shape)

# ---------------------------------------------------------------------------
# 4. Inspect the Transformer architecture in code
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. TRANSFORMER ARCHITECTURE")
print("=" * 70)
print("Number of Transformer layers:", model.config.n_layer)
print("Number of attention heads:", model.config.n_head)
print("Hidden dimension:", model.config.n_embd)

print(model.transformer.h)

# ---------------------------------------------------------------------------
# 5. Hidden states and attention weights
# ---------------------------------------------------------------------------
with torch.no_grad():
    outputs = model(
        **input_ids,
        output_attentions=True,
        output_hidden_states=True
    )

print("\n" + "=" * 70)
print("5. HIDDEN STATES AND ATTENTION")
print("=" * 70)
print(
    "Number of hidden-state outputs:",
    len(outputs.hidden_states)
)

print(
    "Number of attention outputs:",
    len(outputs.attentions)
)

print(
    "First attention tensor shape:",
    outputs.attentions[0].shape
)
# Shape = (batch size, number of heads, sequence length, sequence length)

# Small text view of one attention head (layer 0, head 0).
# Row = token doing the "looking", column = token being looked at.
# Everything above the diagonal is 0: causal attention hides future tokens.
attn = outputs.attentions[0][0, 0]
print("\nLayer 0, head 0 attention weights (rows sum to 1):")
print(attn.numpy().round(2))

# ---------------------------------------------------------------------------
# 6. Logits and next-token prediction
# ---------------------------------------------------------------------------
with torch.no_grad():
    outputs = model(**input_ids)

logits = outputs.logits

print("\n" + "=" * 70)
print("6. LOGITS AND NEXT-TOKEN PREDICTION")
print("=" * 70)
print("Logits shape:", logits.shape)
# Shape = (batch size, sequence length, vocabulary size)

next_token_logits = logits[:, -1, :]

next_token_id = torch.argmax(
    next_token_logits,
    dim=-1
)

print(
    "Predicted next token:",
    tokenizer.decode(next_token_id)
)

# Logits are raw scores. Softmax turns them into probabilities.
probabilities = torch.softmax(next_token_logits, dim=-1)
top_probs, top_ids = torch.topk(probabilities, k=5, dim=-1)
print("\nTop 5 candidates for the next token:")
for prob, token_id in zip(top_probs[0], top_ids[0]):
    print(
        f"  id={token_id.item():>6}  "
        f"token={tokenizer.decode(token_id)!r:<14} "
        f"probability={prob.item():.4f}"
    )

# ---------------------------------------------------------------------------
# 7. Text generation (greedy decoding)
# ---------------------------------------------------------------------------
prompt = "Data engineering is evolving because"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "=" * 70)
print("7. TEXT GENERATION")
print("=" * 70)
print(
    tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )
)

# ---------------------------------------------------------------------------
# 8. Autoregressive generation, one token at a time (what generate() does)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("8. AUTOREGRESSIVE LOOP: ONE TOKEN AT A TIME")
print("=" * 70)

current_ids = inputs["input_ids"]
for step in range(1, 11):
    with torch.no_grad():
        step_logits = model(input_ids=current_ids).logits
    new_id = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
    current_ids = torch.cat([current_ids, new_id], dim=-1)
    print(f"Step {step:>2}: new token = {tokenizer.decode(new_id[0])!r}")

print("\nSequence after 10 manual steps:")
print(tokenizer.decode(current_ids[0], skip_special_tokens=True))

# Greedy decoding is deterministic, so the first 10 new tokens should match.
n_prompt = inputs["input_ids"].size(1)
print(
    "Matches generate() for these 10 tokens:",
    torch.equal(current_ids[0], generated[0][: n_prompt + 10]),
)
