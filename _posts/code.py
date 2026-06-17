"""
NanoGPT + KV Cache — Prefill/Decode Split.

Adds key-value caching to the baseline transformer. During inference, the
prefill phase processes the full prompt once, and subsequent decode steps
feed only the new token while attending over the cached K/V tensors.
Eliminates O(n²) redundant recomputation.

Builds on: nanogpt.py
Key additions:
    - Per-head key_cache / value_cache tensors
    - Prefill phase (full prompt) → decode phase (single token)
    - start_pos parameter for correct positional embeddings during decode
    - clear_kv_cache() for cache lifecycle management

Benchmark: ~2.6× throughput improvement over no-cache generation.

Run:
    python nanogpt-kv-cache.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from benchmarks.kv_cache_baseline_benchmark_runs import (
    run_kv_cache_baseline_benchmark_suite,
)

# hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# ------------

# hyperparameters for testing

batch_size = 8          # smaller training batches
block_size = 64        # keep same for now so your benchmark assumptions hold
max_iters = 1000         # much faster than 5000
eval_interval = 200
learning_rate = 1e-3
device = 'cpu'          # force CPU
eval_iters = 10         # much faster validation
n_embd = 32             # was 64
n_head = 4              # 32 / 4 = 8 dim per head
n_layer = 4             # was 4
dropout = 0.0

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(): #evaluates average loss over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def clear_kv_cache(model):
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.key_cache = None
            module.value_cache = None

class CausalSelfAttention(nn.Module):
    """
    Implement self-attention with fused QKV projection and KV cache.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()  
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attn_proj = nn.Linear(n_embd, n_embd)
        self.num_heads = num_heads
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.key_cache = None
        self.value_cache = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x) # (B, T, 3 * n_embd)
        q, k, v = qkv.split(n_embd, dim=2) # (B, T, n_embd)
        # reshape to (B, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # can also use F.scaled_dot_product_attention here
        if not self.training:
            if self.key_cache is not None:
                self.key_cache   = torch.cat([self.key_cache, k], dim=2)   # dim=2 is T now
                self.value_cache = torch.cat([self.value_cache, v], dim=2)
            else:
                self.key_cache, self.value_cache = k, v

            scale = self.head_size ** -0.5
            attn = (q @ self.key_cache.transpose(-2, -1)) * scale  # (B, n_head, T_q, T_cache)
            attn = F.softmax(attn, dim=-1)
            out  = attn @ self.value_cache  # (B, n_head, T_q, head_size)
        else:
            scale = self.head_size ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)
            attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # ← missing
            attn = F.softmax(attn, dim=-1)
            out  = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.attn_proj(out)
        return out   

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos=0):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(start_pos, start_pos + T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def generate_kv_cache(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)
    
    # Prefill: process the initial context all at once
    logits, _ = model(idx)
    
    for _ in range(max_new_tokens):
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        # Forward pass with ONLY the new token. We pass start_pos to get the right position embeddings.
        logits, _ = model(idx_next, start_pos=idx.shape[1] - 1)
        
    model.train()
    return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = block_size - context.shape[1] # total capacity - space initial prompt takes up
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# ── non-cached generate (forces full-context recompute every step) ────────────
def generate_no_cache(model, idx, max_new_tokens):
    """Runs in train mode so the KV cache branch is never entered."""
    model.train()                          # disables KV cache path
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ── cached generate (your existing path, one token fed at a time) ─────────────
def generate_with_cache(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Feed only the LAST token so the cache does the rest of the work
            logits, _ = model(idx[:, -1:])   # (B, 1, vocab_size)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx
