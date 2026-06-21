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
from benchmarks.cuda_graph_benchmark_runs import (
    run_cuda_graph_benchmark_suite,
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
device = 'cuda'         # CUDA graphs require GPU
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
            module.key_cache.zero_()
            module.value_cache.zero_()

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

        # Pre-allocated KV cache — fixed address, never reallocated
        self.key_cache = torch.zeros(1, num_heads, block_size, head_size, device=device)
        self.value_cache = torch.zeros(1, num_heads, block_size, head_size, device=device)

        # Static buffer for decode masking — avoids torch.arange in the graph
        # kv_indices[j] = j, so "kv_indices <= cache_pos" gives the valid mask
        self.register_buffer('kv_indices', torch.arange(block_size))

    def forward(self, x, cache_pos=None):
        """Used for training and eager-mode prefill/decode. NOT graph-captured."""
        B, T, C = x.shape
        qkv = self.qkv(x) # (B, T, 3 * n_embd)
        q, k, v = qkv.split(n_embd, dim=2) # (B, T, n_embd)
        # reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if not self.training and cache_pos is not None:
            # ── Inference path: write into pre-allocated cache ──
            self.key_cache[:, :, cache_pos:cache_pos + T, :] = k
            self.value_cache[:, :, cache_pos:cache_pos + T, :] = v

            # ── Attend over FULL cache, mask unfilled + causal ──
            scale = self.head_size ** -0.5
            attn = (q @ self.key_cache.transpose(-2, -1)) * scale  # (B, n_head, T, block_size)

            # Build mask: query at absolute position (cache_pos + i) can see
            # KV positions 0..cache_pos+i (causal) and nothing beyond.
            q_positions  = torch.arange(cache_pos, cache_pos + T, device=x.device)  # (T,)
            kv_positions = torch.arange(block_size, device=x.device)                 # (block_size,)
            mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)              # (T, block_size)
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn = F.softmax(attn, dim=-1)
            out  = attn @ self.value_cache  # (B, n_head, T, head_size)
        else:
            # ── Training path: standard causal attention, no cache ──
            scale = self.head_size ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)
            attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out  = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.attn_proj(out)
        return out

    def decode_cached(self, x, cache_pos):
        """
        Graph-safe decode: T is always 1, no torch.arange, no conditionals.

        cache_pos: scalar tensor (shape ()) — the slot to write into.
                   This is a static buffer whose VALUE changes but ADDRESS doesn't.
        """
        B, T, C = x.shape  # T is always 1
        qkv = self.qkv(x)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)  # (B, n_head, 1, hs)
        k = k.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)

        # ── Write new K/V into the cache at the correct slot ──
        # index_copy_ is in-place and graph-safe: copies k into key_cache
        # at the position(s) specified by cache_pos along dim 2
        self.key_cache.index_copy_(2, cache_pos.view(1), k)
        self.value_cache.index_copy_(2, cache_pos.view(1), v)

        # ── Attend over full cache with static mask ──
        scale = self.head_size ** -0.5
        attn = (q @ self.key_cache.transpose(-2, -1)) * scale  # (B, n_head, 1, block_size)

        # Mask: kv_indices is [0,1,2,...,63] (registered buffer, fixed address)
        # cache_pos is a scalar tensor. This comparison is a pure GPU op
        # that the graph captures — on replay, cache_pos has a new value,
        # so the mask changes, but the operation and tensor addresses are the same.
        mask = self.kv_indices <= cache_pos  # (block_size,)
        attn = attn.masked_fill(~mask.view(1, 1, 1, block_size), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out  = attn @ self.value_cache  # (B, n_head, 1, head_size)

        out = out.transpose(1, 2).contiguous().view(B, 1, C)
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

    def forward(self, x, cache_pos=None):
        x = x + self.sa(self.ln1(x), cache_pos=cache_pos)
        x = x + self.ffwd(self.ln2(x))
        return x

    def decode_cached(self, x, cache_pos):
        """Graph-safe path — delegates to CausalSelfAttention.decode_cached."""
        x = x + self.sa.decode_cached(self.ln1(x), cache_pos)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # ModuleList so we can pass cache_pos through each block
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # ── Static buffers for CUDA graph replay (Hint 2) ──
        # These tensors are allocated ONCE. Their addresses never change.
        # Before each graph.replay(), we .copy_() / .fill_() new values in.
        self.static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_position  = torch.zeros(1, dtype=torch.long, device=device)
        self.static_cache_pos = torch.zeros(1, dtype=torch.long, device=device)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos=0, cache_pos=None):
        """Used for training and eager-mode prefill/decode. NOT graph-captured."""
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(start_pos, start_pos + T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, cache_pos=cache_pos) # (B,T,C)
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

    def decode_one_token(self):
        """
        Graph-safe decode: reads from static buffers, no dynamic shapes.

        This method has NO parameters — it reads from self.static_input_ids,
        self.static_position, and self.static_cache_pos. Before calling this
        (or replaying the graph that captured it), you .copy_()/.fill_() the
        actual values into those buffers.

        Every operation here is a fixed-shape GPU operation:
        - No torch.arange (uses static_position buffer instead)
        - No if/else branches
        - No Python integers as tensor indices (uses scalar tensors)
        """
        # ── Embedding lookup from static buffers ──
        tok_emb = self.token_embedding_table(self.static_input_ids)   # (1, 1, n_embd)
        pos_emb = self.position_embedding_table(self.static_position) # (1, n_embd)
        x = tok_emb + pos_emb  # broadcasts: (1, 1, n_embd) + (1, n_embd) → (1, 1, n_embd)

        # ── Run through all blocks using the graph-safe decode path ──
        for block in self.blocks:
            x = block.decode_cached(x, self.static_cache_pos)

        x = self.ln_f(x)            # (1, 1, n_embd)
        logits = self.lm_head(x)    # (1, 1, vocab_size)
        return logits

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
    
    T_prompt = idx.shape[1]
    
    # Prefill: process the initial context all at once
    # cache_pos=0 means "write these T_prompt tokens starting at slot 0"
    logits, _ = model(idx, cache_pos=0)
    
    cache_pos = T_prompt  # next token goes into this slot
    
    for _ in range(max_new_tokens):
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        # Forward pass with ONLY the new token.
        # start_pos = cache_pos for correct position embedding
        # cache_pos = cache_pos for writing into the correct cache slot
        logits, _ = model(idx_next, start_pos=cache_pos, cache_pos=cache_pos)
        cache_pos += 1
        
    model.train()
    return idx

def generate_cuda_graph(model, idx, max_new_tokens):
    """
    Generate tokens using CUDA graph replay for the decode loop.

    Three phases:
      1. Prefill  — eager mode (variable-length prompt, runs once)
      2. Capture  — record one decode step as a CUDA graph
      3. Decode   — replay the captured graph for each new token
    """
    model.eval()
    clear_kv_cache(model)

    T_prompt = idx.shape[1]

    # ════════════════════════════════════════════════════════
    # Phase 1: PREFILL (eager — not graph-captured)
    # ════════════════════════════════════════════════════════
    # Prefill has variable sequence length, so we run it normally.
    # This populates KV cache slots 0..T_prompt-1.
    logits, _ = model(idx, cache_pos=0)

    # Sample the first decode token from prefill output
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
    idx = torch.cat((idx, idx_next), dim=1)

    cache_pos = T_prompt  # the first decode token goes into this slot

    # ════════════════════════════════════════════════════════
    # Phase 2: WARMUP + CAPTURE
    # ════════════════════════════════════════════════════════
    # Step 2a: Load real values into static buffers for the warmup run
    model.static_input_ids.copy_(idx_next)
    model.static_position.fill_(cache_pos)
    model.static_cache_pos.fill_(cache_pos)

    # Step 2b: Warmup — run decode_one_token once WITHOUT capturing.
    # This forces PyTorch to allocate all intermediate tensors.
    # If allocations happen during capture, replay would try to
    # re-allocate and crash.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        static_output = model.decode_one_token()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Step 2c: Capture — record the decode step as a CUDA graph.
    # Every kernel launch inside decode_one_token() is recorded.
    # The graph remembers the exact tensor addresses it reads/writes.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        static_output = model.decode_one_token()

    # static_output now lives at a fixed address. After each replay(),
    # it contains the logits for whatever token was in static_input_ids.

    cache_pos += 1  # warmup already wrote into cache_pos, advance

    # ════════════════════════════════════════════════════════
    # Phase 3: DECODE LOOP (graph replay)
    # ════════════════════════════════════════════════════════
    for _ in range(max_new_tokens - 1):  # -1 because we already sampled one token
        # Step 3a: Load real values into static buffers
        # The graph will read from these exact addresses on replay.
        model.static_input_ids.copy_(idx_next)
        model.static_position.fill_(cache_pos)
        model.static_cache_pos.fill_(cache_pos)

        # Step 3b: Replay the captured graph.
        # This runs all ~50 kernels back-to-back in a single GPU command.
        # No Python loop, no CPU-GPU sync per kernel. Cost: ~5μs.
        graph.replay()

        # Step 3c: Read logits from the static output tensor.
        # static_output is the SAME tensor that was created during capture —
        # replay() overwrote its contents with results for the new input.
        logits = static_output[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        cache_pos += 1

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

# ── Generate: eager vs CUDA graph comparison ──
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = block_size - context.shape[1]

print("\n── Eager (KV cache) ──")
torch.manual_seed(42)
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))

print("\n── CUDA Graph ──")
torch.manual_seed(42)
print(decode(generate_cuda_graph(m, context, max_gen)[0].tolist()))
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
    
run_cuda_graph_benchmark_suite(
    m,
    train_data=train_data,
    clear_cache_fn=clear_kv_cache,
    device=device,
    block_size=block_size,
)
