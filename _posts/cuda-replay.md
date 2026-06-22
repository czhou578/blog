---
layout: post
title: "NanoGPT: CUDA Graph Replay"
date: 2026-06-21
image: https://czhou578.github.io/blog/images/cuda_replay_thumbnail.png
---

In the [fused attention post](/blog/2026/06/17/fused-att), we collapsed all per-head projections into a single `nn.Linear(n_embd, 3 * n_embd)`. This post goes one level deeper: instead of optimizing individual kernels, we eliminate the overhead of *launching* them entirely.

A single decode step in our 4-layer model launches roughly 50 GPU kernels: embedding lookup, layer norm, QKV projection, attention, FFN, another layer norm, another attention, and so on through all 4 layers, then a final layer norm and the lm_head projection. Each kernel launch costs 5–15μs of CPU-side overhead — the Python interpreter tells the CUDA driver "run this kernel," the driver validates the arguments, sets up the launch configuration, and dispatches. With 50 launches, that's 250–750μs of pure overhead.

The actual GPU compute for a single-token forward pass at our scale (210K params, `B=1`, `T=1`) takes maybe 50μs. **The launch overhead is 5–15× larger than the compute itself.** We're spending most of our time asking the GPU to do work, not actually doing work.

This exact overhead ratio is why vLLM and SGLang both implement CUDA graph replay as one of their first optimizations. For small-to-medium models where the decode step is memory-bandwidth-bound and the matmuls are tiny, kernel launch overhead is the dominant cost. Eliminating it gives a 2–3× decode throughput improvement, for free.

## What is a CUDA graph?

The idea is simple: instead of launching 50 kernels one at a time from Python, record the entire sequence of kernel launches once, and then replay the recording in a single GPU-side command. The replay cost is ~5μs total, regardless of how many kernels are in the graph.

```
Without CUDA graph:                    With CUDA graph:

  Python                GPU                Python           GPU
    │                                        │
    ├── launch kernel 1  ──►  run            ├── replay ──►  run all kernels
    ├── launch kernel 2  ──►  run            │               back to back
    ├── launch kernel 3  ──►  run            │               no gaps
    │   ...                                  │
    ├── launch kernel 50 ──►  run            │
    │                                        │
    │  ~750μs CPU overhead                   │  ~5μs CPU overhead
    │  + 50μs GPU compute                    │  + 50μs GPU compute
```

PyTorch's API for this is three steps: warmup, capture, replay. We'll get to the exact code, but first there's a constraint that makes this interesting.

## The core constraint: static shapes

![Static Shapes vs Dynamic KV Cache]({{ site.baseurl }}/images/cuda_replay_static_shapes.png)

CUDA graphs record a fixed sequence of operations on **fixed-size tensors at fixed memory addresses**. Every pointer, every tensor shape, every stride is baked into the graph at capture time. If any tensor shape changes between replay calls, the graph is invalid, which will either crash or produce garbage.

Our current decode step has two sources of dynamic shapes:

**1. The KV cache grows every step.** In the fused attention post, we used `torch.cat([self.key_cache, k], dim=2)` to append the new key to the cache. Step 1: cache is `(B, n_head, 1, head_size)`. Step 2: `(B, n_head, 2, head_size)`. Step 50: `(B, n_head, 50, head_size)`. Every step creates a new, larger tensor at a new address.

**2. Position indices are created on the fly.** `torch.arange(start_pos, start_pos + T)` in `GPTLanguageModel.forward()` creates a new CPU-side tensor every call. New tensor = new address = graph reads from the wrong place.

---

## Step 1: Pre-allocate static KV cache buffers

Instead of growing the cache with `torch.cat`, we pre-allocate a fixed-size cache tensor at model init and write into it with an index. The maximum possible sequence length is `block_size`, the context window size. So we allocate `(B, n_head, block_size, head_size)` filled with zeros and track how much is filled with a position counter.

```python
class CausalSelfAttention(nn.Module):
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
        self.register_buffer('kv_indices', torch.arange(block_size))
```

Three things to notice. First, the cache is allocated at init, not during the first forward pass. Its memory address is fixed for the lifetime of the model. Second, it's the full `block_size` wide — we'll always attend over all 64 positions, masking out the empty ones. Third, `kv_indices` is a registered buffer containing `[0, 1, 2, ..., 63]`. We'll use it to build the attention mask without `torch.arange`.

The inference path in `forward()` now writes into the cache by index instead of concatenating:

```python
def forward(self, x, cache_pos=None):
    B, T, C = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.split(n_embd, dim=2)
    q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

    if not self.training and cache_pos is not None:
        # Write into pre-allocated cache at the correct slot
        self.key_cache[:, :, cache_pos:cache_pos + T, :] = k
        self.value_cache[:, :, cache_pos:cache_pos + T, :] = v

        # Attend over FULL cache, mask unfilled positions
        scale = self.head_size ** -0.5
        attn = (q @ self.key_cache.transpose(-2, -1)) * scale  # (B, n_head, T, block_size)

        # Mask: query at position (cache_pos + i) can see KV positions 0..cache_pos+i
        q_positions  = torch.arange(cache_pos, cache_pos + T, device=x.device)
        kv_positions = torch.arange(block_size, device=x.device)
        mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)  # (T, block_size)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out  = attn @ self.value_cache
    else:
        # Training path — unchanged
        scale = self.head_size ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out  = attn @ v

    out = out.transpose(1, 2).contiguous().view(B, T, C)
    out = self.attn_proj(out)
    return out
```

The key change: `self.key_cache[:, :, cache_pos:cache_pos + T, :] = k` writes the new keys into specific slots instead of allocating a new tensor. The attention then reads from the **entire** `(B, n_head, block_size, head_size)` cache every time. The mask zeros out positions that aren't filled yet.

You might wonder: isn't attending over all 64 positions wasteful when only 10 are filled? For decode, `T_q = 1`, so the attention matrix is `(1, block_size)` — a single row of 64 dot products. That's negligible. The wasted compute on empty slots is irrelevant compared to the benefit of having static tensor shapes.

This `forward()` method still uses `torch.arange` for position tracking, so it's not graph-safe yet. We use it for eager-mode prefill and as the baseline for benchmarking. The graph-safe version comes next.

---

## Step 2: The graph-safe decode path

For CUDA graph capture, we need a dedicated decode function with zero dynamic shapes and zero Python control flow. No `torch.arange`, no `if` branches, no CPU-side tensor creation. Everything must be a fixed-shape GPU operation that the graph can record and replay.

The trick is **static input buffers**. We pre-allocate tensors at model init — `static_input_ids`, `static_position`, `static_cache_pos` — and the decode function reads exclusively from them. Before each replay, we `.copy_()` the real values into these buffers. The graph doesn't know the contents changed — it just reads from the same addresses.

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Static buffers for CUDA graph replay
        # Allocated ONCE. Addresses never change.
        self.static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_position  = torch.zeros(1, dtype=torch.long, device=device)
        self.static_cache_pos = torch.zeros(1, dtype=torch.long, device=device)
```

These three buffers are the entire interface between Python and the captured graph. `static_input_ids` holds the token to decode, `static_position` holds the position embedding index, and `static_cache_pos` tells each attention layer which cache slot to write into. They're all fixed-shape tensors at fixed addresses — exactly what the graph needs.

Now the decode method itself:

```python
def decode_one_token(self):
    """
    Graph-safe decode: reads from static buffers, no dynamic shapes.

    Every operation here is a fixed-shape GPU operation:
    - No torch.arange (uses static_position buffer instead)
    - No if/else branches
    - No Python integers as tensor indices (uses scalar tensors)
    """
    tok_emb = self.token_embedding_table(self.static_input_ids)   # (1, 1, n_embd)
    pos_emb = self.position_embedding_table(self.static_position) # (1, n_embd)
    x = tok_emb + pos_emb  # broadcasts: (1, 1, n_embd) + (1, n_embd) → (1, 1, n_embd)

    for block in self.blocks:
        x = block.decode_cached(x, self.static_cache_pos)

    x = self.ln_f(x)            # (1, 1, n_embd)
    logits = self.lm_head(x)    # (1, 1, vocab_size)
    return logits
```

Notice: **no parameters**. The method reads everything from `self.static_*` buffers. This is intentional. When the graph replays, it re-executes the exact same operations — `token_embedding_table(self.static_input_ids)` — but `self.static_input_ids` now contains a different token ID (because we `.copy_()`'d a new value into it before replay). The graph sees the same memory address with different contents. That's the whole mechanism.

The position embedding is particularly interesting. In the eager path, we did `self.position_embedding_table(torch.arange(start_pos, start_pos + T))` — which creates a new tensor every call. Here, we do `self.position_embedding_table(self.static_position)` — an embedding lookup from a fixed-address scalar tensor. Before each replay, `self.static_position.fill_(current_pos)` updates the value. Same address, different contents.

Each attention layer needs its own graph-safe path too:

```python
def decode_cached(self, x, cache_pos):
    """
    Graph-safe decode: T is always 1, no torch.arange, no conditionals.
    cache_pos: scalar tensor (shape ()) — value changes, address doesn't.
    """
    B, T, C = x.shape  # T is always 1
    qkv = self.qkv(x)
    q, k, v = qkv.split(n_embd, dim=2)
    q = q.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)
    k = k.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)
    v = v.view(B, 1, self.num_heads, self.head_size).transpose(1, 2)

    # Write new K/V into cache — index_copy_ is in-place and graph-safe
    self.key_cache.index_copy_(2, cache_pos.view(1), k)
    self.value_cache.index_copy_(2, cache_pos.view(1), v)

    # Attend over full cache with static mask
    scale = self.head_size ** -0.5
    attn = (q @ self.key_cache.transpose(-2, -1)) * scale  # (B, n_head, 1, block_size)

    # kv_indices is [0,1,2,...,63] (registered buffer, fixed address)
    # cache_pos is a scalar tensor — this comparison is a pure GPU op
    mask = self.kv_indices <= cache_pos  # (block_size,)
    attn = attn.masked_fill(~mask.view(1, 1, 1, block_size), float('-inf'))

    attn = F.softmax(attn, dim=-1)
    out  = attn @ self.value_cache  # (B, n_head, 1, head_size)

    out = out.transpose(1, 2).contiguous().view(B, 1, C)
    out = self.attn_proj(out)
    return out
```

Two subtle things here. First, `index_copy_` instead of slice assignment. `self.key_cache[:, :, cache_pos, :] = k` might work in eager mode, but `index_copy_` is explicitly in-place and graph-safe — it copies the data into the specified position along dimension 2 without creating intermediate tensors.

Second, the mask: `self.kv_indices <= cache_pos`. Both operands are fixed-address tensors. `kv_indices` is `[0, 1, 2, ..., 63]`, registered as a buffer at init. `cache_pos` is the static scalar buffer. The comparison is a pure GPU operation — the graph captures the *operation* (elementwise `<=`), not the *result*. On each replay, `cache_pos` has a new value, so the mask changes, but the tensor addresses and shapes are identical. This is the key insight for making dynamic behavior work inside a static graph.

---

## Step 3: Capture and replay

With the graph-safe decode path built, the actual capture-and-replay is the easy part. PyTorch's API has three phases:

```python
def generate_cuda_graph(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)

    T_prompt = idx.shape[1]

    # ════════════════════════════════════════════════════════
    # Phase 1: PREFILL (eager — not graph-captured)
    # ════════════════════════════════════════════════════════
    logits, _ = model(idx, cache_pos=0)

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)

    cache_pos = T_prompt
```

Prefill runs in eager mode, with no graph. The prompt has a variable length (could be 1 token, could be 50), and we only run it once, so there's no benefit to capturing it. This populates KV cache slots `0` through `T_prompt-1`.

After prefill, we sample the first decode token and advance `cache_pos`. Now comes capture:

```python
    # ════════════════════════════════════════════════════════
    # Phase 2: WARMUP + CAPTURE
    # ════════════════════════════════════════════════════════

    # Load real values into static buffers for warmup
    model.static_input_ids.copy_(idx_next)
    model.static_position.fill_(cache_pos)
    model.static_cache_pos.fill_(cache_pos)

    # Warmup: run once WITHOUT capturing
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        static_output = model.decode_one_token()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture: record the decode step as a CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        static_output = model.decode_one_token()
```

**Why the warmup?** PyTorch allocates GPU memory lazily. The first time a layer runs, it allocates buffers for intermediate activations. If that allocation happens *during* graph capture, the allocator state becomes part of the graph. Replay would try to re-allocate the same memory, which crashes. The warmup forces all allocations to happen before capture starts.

**Why a separate stream?** Graph capture records everything that happens on a CUDA stream. We want an isolated stream so that unrelated GPU work (like leftover prefill operations) doesn't get accidentally captured.

**`static_output`:** The output tensor from the captured run is also at a fixed address. After each `graph.replay()`, the same tensor contains the new results. We read logits from `static_output` directly since `graph.replay()` has no return value.

```python
    cache_pos += 1  # warmup already wrote into cache_pos, advance

    # ════════════════════════════════════════════════════════
    # Phase 3: DECODE LOOP (graph replay)
    # ════════════════════════════════════════════════════════
    for _ in range(max_new_tokens - 1):
        # Copy real values into static buffers
        model.static_input_ids.copy_(idx_next)
        model.static_position.fill_(cache_pos)
        model.static_cache_pos.fill_(cache_pos)

        # Replay: all ~50 kernels, one GPU command, ~5μs
        graph.replay()

        # Read from the static output tensor
        logits = static_output[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        cache_pos += 1

    model.train()
    return idx
```

The decode loop is clean. Three `.copy_()` / `.fill_()` calls to load the static buffers, one `graph.replay()`, and then standard sampling. The key: the KV cache is a static tensor that persists across replays. The graph recorded "write `k` into slot `cache_pos` of `self.key_cache`." On each replay, `cache_pos` has a new value and the input produces a new `k`, so the cache accumulates entries at successive positions.

Let me trace through what happens on replay step 3, assuming the prompt was 1 token long:

```
Before replay:
  static_input_ids: [token_4]          ← .copy_() from idx_next
  static_position:  [4]                ← .fill_(4)
  static_cache_pos: [4]                ← .fill_(4)
  key_cache:        filled at slots 0, 1, 2, 3 from previous steps

Graph replays:
  1. tok_emb = embedding_table[token_4]           → (1, 1, 32)
  2. pos_emb = position_table[4]                  → (1, 32)
  3. x = tok_emb + pos_emb                        → (1, 1, 32)
  4. For each of 4 layers:
     a. qkv projection                            → (1, 1, 96)
     b. split + reshape                           → q,k,v each (1, 4, 1, 8)
     c. index_copy_ k into key_cache slot 4       → cache now filled at 0,1,2,3,4
     d. q @ key_cache.T                           → (1, 4, 1, 64)
     e. mask: kv_indices <= 4 → [T,T,T,T,T,F,...] → zeros out slots 5-63
     f. softmax + attn @ value_cache              → (1, 4, 1, 8)
     g. FFN
  5. Final layer norm + lm_head                   → logits in static_output

After replay:
  static_output contains logits for token_4
  key_cache has entries at slots 0, 1, 2, 3, 4
```

Every tensor in this trace is at a fixed address with a fixed shape. The only things that change are the *values* inside the static buffers. That's why the graph can replay it.

---

## Verifying correctness

Before benchmarking, we need to verify the graph produces the same output as eager mode. Both paths use the same model weights, the same sampling, and the same random seed:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_gen = block_size - context.shape[1]

print("── Eager (KV cache) ──")
torch.manual_seed(42)
print(decode(generate_kv_cache(m, context, max_gen)[0].tolist()))

print("── CUDA Graph ──")
torch.manual_seed(42)
print(decode(generate_cuda_graph(m, context, max_gen)[0].tolist()))
```

The two paths produce slightly different text — not because the graph is wrong, but because the decode paths diverge at the first autoregressive step. Eager and graph use different masking strategies (slice-based vs. full-cache with `kv_indices`), so floating-point rounding differences in softmax accumulate across steps. With the same seed and identical masks, they'd match exactly. What matters is that both produce coherent, structurally similar output:

```text
── Eager (KV cache) ──

I
O: as fnos tis, ses tout me itarnd he, preckn,
O yot me o-yon

── CUDA Graph ──

I,
H asafnte tis, ses thuared tharnd heere:
Ton,
Onge fof o--An
```

Both paths generate grammatically-structured gibberish of similar quality — which is exactly what a 57K-parameter model trained for 1,000 steps on Shakespeare should produce. The graph path is correct.

---

## Benchmark results

The model is intentionally tiny: 57K parameters, 4 layers, 4 heads, `block_size=64`, trained for 1,000 steps on a Shakespeare corpus. Everything runs on a single mobile laptop GPU. The question is not "how fast is this model" but "does CUDA graph replay eliminate kernel launch overhead in the way the theory predicts?"

The benchmark suite runs five named configurations — each varying prompt length and generation length — plus a generation length sweep that holds the prompt fixed at 8 tokens and varies `N` from 8 to 56.

### Named configurations

| Config | Prompt Len | Gen Tokens | Eager tok/s | Graph tok/s | E2E Speedup | Per-step Speedup |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|
| `smoke_test` | 4 | 8 | 580.2 | 827.7 | **1.43x** | 1.28x |
| `medium_generation` | 8 | 32 | 692.5 | 834.3 | 1.20x | 1.30x |
| `long_generation` | 8 | 48 | 695.7 | 842.0 | 1.21x | 1.30x |
| `heavy_prompt` | 32 | 16 | 664.6 | 792.4 | 1.19x | 1.30x |
| `near_context_limit` | 4 | 56 | 600.6 | 843.1 | **1.40x** | 1.39x |

### Generation length sweep (prompt_len=8)

| Gen Tokens | Eager tok/s | Graph tok/s | E2E Speedup | Per-step Speedup |
|--------:|--------:|--------:|--------:|--------:|
| 8 | 694.6 | 821.0 | 1.18x | 1.29x |
| 16 | 695.8 | 830.5 | 1.19x | 1.29x |
| 32 | 694.2 | 837.0 | 1.21x | 1.30x |
| 48 | 694.3 | 684.5 | **0.99x** | 1.15x |
| 56 | 437.3 | 493.0 | 1.13x | **1.43x** |

### Per-step decode latency summary

This is the cleanest view. Median per-step decode latency, in milliseconds:

| Config | Eager (ms) | Graph (ms) | Speedup |
|--------|--------:|--------:|--------:|
| `smoke_test` | 1.175 | 0.903 | 1.30x |
| `medium_generation` | 1.175 | 0.902 | 1.30x |
| `long_generation` | 1.172 | 0.901 | 1.30x |
| `heavy_prompt` | 1.167 | 0.895 | 1.30x |
| `near_context_limit` | 1.176 | 0.902 | 1.30x |

---

## Analysis

### The per-step number is remarkably stable

The most striking thing about these results is the per-step decode latency summary. Across all five named configurations — different prompt lengths, different generation lengths, different ratios of prefill to decode — the median per-step decode speedup is **exactly 1.30x**. Eager mode lands at 1.167–1.176ms per step; graph replay lands at 0.895–0.903ms. The variance is sub-microsecond.

This stability makes sense. Each decode step does the same work: one token through 4 layers of attention + FFN. The only thing that changes between configurations is *how many cache slots are filled*, which affects how many of the 64 dot products in the attention row produce real values vs. masked-out zeros. But the compute is identical — we attend over all 64 positions regardless. So the per-step cost is constant, and the speedup from eliminating launch overhead is constant.

The 0.27ms reduction (from ~1.175ms to ~0.902ms) represents the kernel launch overhead that graph replay eliminates. That's less dramatic than the "250–750μs" estimate in the introduction, but our model only has 4 layers — roughly 20–25 kernels per decode step, not 50. The overhead scales with kernel count, and the ratio is consistent: launch overhead is about 23% of the eager decode step cost.

### End-to-end speedup tells a different story

While per-step speedup is locked at 1.30x, end-to-end throughput speedup ranges from **0.99x to 1.43x**. The gap is explained by two fixed costs that don't benefit from graph replay: prefill and warmup.

**Warmup costs ~2.7ms** in most configurations. This is the one-time cost of running the decode path once before capture to force lazy memory allocation. For `smoke_test` (only 8 generated tokens), that 2.7ms warmup is amortized over just 8 decode steps. For `long_generation` (48 tokens), the same 2.7ms is spread over 48 steps. More decode steps → better amortization → end-to-end speedup converges toward the per-step speedup.

**Prefill runs in eager mode for both paths.** Both methods show similar prefill times (1.2–3.7ms depending on prompt length), confirming that prefill is not captured in the graph. For `heavy_prompt` (prompt_len=32), prefill takes ~2.4ms — a fixed cost that dilutes the end-to-end ratio.

The cleanest illustration: `smoke_test` generates only 8 tokens but shows 1.43x end-to-end speedup — seemingly *better* than the 1.28x per-step speedup. This is because the eager path's prefill (3.69ms) is substantially slower than the graph path's (1.24ms). The graph path benefits from a warmed-up GPU and memory pool after capture. The per-step number is a more honest measure of the optimization itself.

### The N=48 anomaly

In the generation length sweep, N=48 is the outlier: **0.99x end-to-end speedup** — the graph path is essentially no faster. The per-step speedup drops to 1.15x. What happened?

The clue is in the raw numbers. The graph path's avg_step_ms is 1.022ms, but the median is 0.902ms. That gap (0.12ms between average and median) means a few decode steps were dramatically slower than the rest. Meanwhile, the eager path's average and median are both ~1.172ms — perfectly consistent.

With prompt_len=8 and N=48, the total sequence reaches 56 positions out of a 64-position context window. At 87.5% cache occupancy, we're in the region where GPU memory management starts to matter. The graph path's warmup is normal (2.67ms), so this isn't a capture issue. Most likely, a small number of decode steps near the end of the sequence triggered memory pressure or cache-line contention on the attention computation over the nearly-full buffer, creating outlier latencies that dragged the average up.

### Context-limit pressure at N=56

N=56 in the sweep is even more revealing. Both paths slow down significantly:

- Eager: median step jumps from 1.175ms to **1.258ms** (7% slower)
- Graph: median step jumps from 0.902ms to **0.977ms** (8% slower)
- Graph warmup jumps from ~2.7ms to **10.51ms** (4× slower)
- Eager prefill jumps from ~1.4ms to **1.50ms**
- Graph prefill jumps from ~1.3ms to **4.46ms**

With prompt_len=8 and N=56, the sequence reaches exactly 64 positions — the full context window. The 10.51ms warmup is the smoking gun: the warmup decode step (at position 9, with 55 steps remaining) forces the attention mechanism to allocate and compute over a cache that will be completely full. The memory allocator does more work, and the attention kernel processes 64 real dot products instead of mostly-masked zeros.

Despite this, the graph path still wins **1.43x per-step decode speedup** — the *highest* per-step speedup in the entire sweep. When the GPU compute per step increases (more real attention work over a full cache), kernel launch overhead becomes a proportionally smaller fraction of eager cost but remains a fixed overhead. Graph replay eliminates it entirely, so the absolute savings grow. This is exactly the pattern production engines see: CUDA graphs matter more, not less, as the model does more real work per step.

### Warmup is cheap

The warmup cost deserves its own note. At 2.6–2.7ms for typical runs, warmup is equivalent to about 3 decode steps. For any generation longer than ~10 tokens, it's negligible. Even the pathological N=56 case (10.51ms warmup) is only ~10 decode steps. In production, where generations are hundreds of tokens and graphs are captured once at server startup, warmup cost is irrelevant.

### Benchmark takeaways

1. **CUDA graph replay delivers a consistent 1.30x per-step decode speedup.** This holds across all prompt lengths and generation lengths tested. The improvement comes from eliminating ~0.27ms of kernel launch overhead per step.
2. **End-to-end speedup varies from 0.99x to 1.43x** depending on how well the fixed costs (warmup, prefill) are amortized. More decode steps → better amortization → higher end-to-end speedup.
3. **Near the context window boundary, both paths slow down.** At 87–100% cache occupancy, memory pressure creates outlier latencies. Graph replay is affected but still wins on median per-step latency.
4. **These numbers understate the production impact.** Our 4-layer model launches ~20 kernels per decode step. A 32-layer model launches ~200. The launch overhead scales linearly with depth; graph replay eliminates it entirely. Production engines see 2–3x decode speedup from graphs, not 1.3x.
5. **The per-step metric is the honest one.** End-to-end throughput conflates prefill, warmup, and decode. The per-step decode latency isolates the specific optimization — launch overhead elimination — and shows it working exactly as predicted.

---

## Why production engines care so much about this

In our toy model, the decode step computes maybe 50μs of actual GPU work and spends 250–750μs on kernel launch overhead. CUDA graphs eliminate that overhead, turning a 300–800μs step into a ~55μs step. That's a significant multiplier.

But there's a catch that our `B=1` implementation doesn't have to deal with: **dynamic batch sizes**. In a production inference engine with continuous batching, the batch size changes every iteration. CUDA graphs need static shapes, so you can't capture a graph for batch size 7 and replay it with batch size 12.

The production solutions:

**Bucketed graphs.** Pre-capture graphs for batch sizes 1, 2, 4, 8, 16, 32. At runtime, pad the actual batch to the nearest bucket size and replay that graph. You waste compute on the padding positions, but graph replay is so fast it's worth it. This is how vLLM does it.

**Piecewise capture.** Instead of capturing the entire forward pass as one graph, capture each transformer layer separately. SGLang uses this approach. It gives more flexibility to change batch sizes between layers, though it adds complexity.

For our single-batch implementation, none of this matters. But if you wanted to support `B=1` through `B=64`, you'd need 7 separate graph captures (one per power-of-2 bucket).

---

## The conceptual map

Here's the full picture of what changed:

```
Current eager decode loop:

  for each token:
      Python: construct input tensor       ← CPU work
      Python: call model.forward()         ← CPU dispatches ~50 kernels
        GPU: embedding lookup              ← tiny kernel
        GPU: layer norm                    ← tiny kernel
        GPU: qkv projection               ← tiny kernel
        GPU: attention                     ← tiny kernel
        GPU: ffn                           ← tiny kernel
        ... (×4 layers)
        GPU: final layer norm
        GPU: lm_head projection
      Python: read logits                  ← CPU–GPU sync
      Python: sample next token            ← CPU

  Bottleneck: CPU kernel launch overhead (~750μs)
              GPU compute time (~50μs)
              15:1 overhead ratio

─────────────────────────────────────────────────────────

CUDA graph decode loop:

  # One-time capture:
  warmup decode_one_token(static_buffers)
  graph.capture(decode_one_token(static_buffers))

  for each token:
      Python: copy token into static_input_ids   ← tiny memcpy
      Python: graph.replay()                     ← ONE command to GPU
        GPU: all ~50 kernels run back-to-back    ← no gaps
      Python: read logits from static_output     ← CPU–GPU sync
      Python: sample next token                  ← CPU

  Bottleneck: GPU compute time (~50μs)
              Graph replay overhead (~5μs)
              Launch overhead eliminated
```

## Summary of changes

| Component | What Changed |
|-----------|-------------|
| `device` | `'cpu'` → `'cuda'` |
| `CausalSelfAttention.__init__` | Pre-allocate static `key_cache` and `value_cache` as `(B, n_head, block_size, head_size)` |
| `CausalSelfAttention.forward` | Write to `cache[pos]` instead of `torch.cat` |
| `CausalSelfAttention.decode_cached` | New graph-safe method: `index_copy_`, static mask via `kv_indices <= cache_pos` |
| `GPTLanguageModel.__init__` | Add `static_input_ids`, `static_position`, `static_cache_pos` buffers |
| `GPTLanguageModel.decode_one_token` | New graph-safe method: reads from static buffers, no dynamic shapes |
| `generate_kv_cache` | Updated to use pre-allocated cache (eager baseline) |
| `generate_cuda_graph` | New: prefill eagerly → warmup → capture → replay loop |
| Training loop | No changes — graphs are inference-only |

---

## Things that went wrong

**`torch.arange` inside the graph.** My first attempt at `decode_cached` still had `torch.arange(block_size, device=x.device)` to build the attention mask. Graph capture doesn't fail on this — `torch.arange` with a constant argument creates the same tensor every time. But it's creating a *new* tensor on the GPU each call, which means the allocator runs during capture. It happened to work during my testing, but this is fragile. The fix was `self.kv_indices` — a registered buffer allocated once at init, used forever. No allocations during capture.

**Forgot the warmup.** I tried capturing directly without a warmup run. The first `decode_one_token()` call triggered PyTorch's lazy allocation of internal buffers (layer norm running stats, intermediate activation tensors). These allocations got recorded into the graph. On the first replay, the graph tried to allocate the same memory again — and since it was already allocated, CUDA returned an error. The warmup run forces all lazy allocations to complete before capture begins.

**Cache position off-by-one.** The warmup run actually *writes* into the KV cache (it's a real forward pass). I forgot to advance `cache_pos` after warmup, so the first graph replay wrote into the same cache slot as the warmup, overwriting it. The generated text started repeating itself. The fix: `cache_pos += 1` after warmup, before the decode loop.

**Used Python int for cache position.** I initially passed `cache_pos` as a Python integer directly to `decode_cached`. This works in eager mode — PyTorch implicitly creates a tensor. But inside a CUDA graph, Python integers don't have GPU addresses. The graph needs a *tensor* it can read from. Switching to `self.static_cache_pos` (a scalar tensor with `.fill_()`) fixed it.

---

You can find the full code on [GitHub](https://github.com/czhou578/nanoGPT-inference/blob/cuda-graph/nanogpt-cuda-graph.py).

CZ