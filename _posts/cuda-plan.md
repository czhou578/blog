# CUDA Graph Decode Playback — Implementation Plan & Hints

## Base File: `nanogpt-fused-attention.py`

**Output file:** `nanogpt-cuda-graph.py`

### Why this base file?

`nanogpt-fused-attention.py` is the right starting point because:

- **Fused QKV projection** — All heads run in a single batched matmul. CUDA graphs capture at the kernel level, so fewer kernel launches = simpler graph topology. The old per-head architecture would have captured 12+ separate kernels per layer.
- **KV cache already works** — The prefill/decode split is implemented. You only want to graph-capture the decode step, not prefill (prefill has variable sequence length).
- **`device = 'cpu'` needs to change** — CUDA graphs don't exist on CPU. This is your first real GPU-required file. Change to `device = 'cuda'`.

### What is a CUDA graph?

Every time you call `model(idx_next, start_pos=...)`, Python tells the CUDA driver to launch dozens of kernels: embedding lookup, layer norm, QKV projection, attention, FFN, etc. Each kernel launch has **CPU-side overhead** — typically 5–15μs on modern hardware. With a 4-layer model, you might have ~50 kernel launches per decode step, costing ~250–750μs just in launch overhead.

The actual GPU compute for a single-token forward pass at 210K params takes maybe 50μs. **The launch overhead is 5–15× larger than the compute.**

CUDA graphs solve this by recording the entire sequence of kernel launches once, then replaying the recorded graph in a single GPU-side command. No Python loop, no CPU–GPU synchronization per kernel. The replay cost is ~5μs total regardless of how many kernels are in the graph.

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

### Why this is how vLLM and SGLang get their decode throughput

In production, the decode step (generate one token) is the bottleneck — it runs hundreds of times per request. The model is memory-bandwidth-bound (reading weights from HBM for a single token), and the actual matmul compute is tiny. CPU launch overhead becomes the dominant cost. CUDA graphs eliminate that overhead entirely.

Both vLLM and SGLang capture decode forward passes as CUDA graphs and replay them each step. This is one of their most impactful optimizations — often a 2–3× decode throughput improvement for small-to-medium models.

---

## The Core Constraint: Static Shapes

This is the educational gold in this exercise. CUDA graphs record a fixed sequence of operations on **fixed-size tensors at fixed memory addresses**. If any tensor shape changes between replay calls, the graph is invalid — it will either crash or produce garbage.

Your current decode step has **two sources of dynamic shapes**:

1. **The KV cache grows every step.** `torch.cat([self.key_cache, k], dim=2)` creates a new, larger tensor each time. Step 1: `(B, n_head, 1, head_size)`. Step 2: `(B, n_head, 2, head_size)`. Step 50: `(B, n_head, 50, head_size)`. This breaks CUDA graphs.

2. **`torch.arange(start_pos, start_pos + T)` in `GPTLanguageModel.forward()`** — the position index changes each step. This is a CPU-side tensor creation that generates new addresses.

**This is the constraint every production engine solves**, and understanding how they solve it is the whole point.

---

## Hint 1: Pre-allocate Static KV Cache Buffers

Instead of growing the cache with `torch.cat` each step, **pre-allocate a fixed-size cache tensor** at model init and write into it with an index.

Think about:
- What is the maximum possible sequence length? (`block_size` — it's your context window.)
- If you pre-allocate `key_cache` as `(B, n_head, block_size, head_size)` filled with zeros, how do you track how much of it is filled?
- During each decode step, instead of `torch.cat`, you would do `self.key_cache[:, :, pos, :] = k.squeeze(2)` — writing the new key into slot `pos`.
- For attention, you'd attend over `self.key_cache[:, :, :pos+1, :]` — but that's a **slice**, which creates a view with a different shape each step. Does this break the graph?

**The fix:** Attend over the **full** `(B, n_head, block_size, head_size)` cache every time, but use an **attention mask** to zero out positions that aren't filled yet. The tensor shapes are now 100% static — only the mask values change, not the tensor shapes.

**Key question to think about:** If you always attend over `block_size` positions even when only 10 are filled, you're wasting compute on the 54 zero-padded positions. At production scale, does this matter? (Hint: for decode, `T_q = 1`, so the attention matrix is `(1, block_size)` — negligible regardless of `block_size`.)

---

## Hint 2: Static Input Buffers

The input to the decode step — the token ID and the position index — must also be static-shape tensors at fixed memory addresses.

Think about:
- Pre-allocate `static_input_ids = torch.zeros((B, 1), dtype=torch.long, device='cuda')` once.
- Pre-allocate `static_position = torch.zeros((1,), dtype=torch.long, device='cuda')` once.
- Before each graph replay, **copy** the actual values into these buffers: `static_input_ids.copy_(idx_next)` and `static_position.fill_(current_pos)`.
- The graph was captured reading from these exact memory addresses, so when you overwrite the contents and replay, it picks up the new values.

**The deeper insight:** This copy-then-replay pattern is how vLLM's `CUDAGraphRunner` works. The runner owns a set of "placeholder" tensors. Before each replay, it copies the real inputs into the placeholders. The graph doesn't know or care that the contents changed — it just reads from the same addresses.

**Question to think about:** `self.position_embedding_table(torch.arange(...))` creates a new tensor on every call. How do you make this graph-safe? (Hint: pre-compute the position embedding lookup as an index into the static position buffer.)

---

## Hint 3: Refactor `forward()` for Graph Capture

Your current `GPTLanguageModel.forward()` handles both training and inference, prefill and decode, all in one function. For CUDA graphs, you need a **dedicated decode function** that:

1. Accepts only the static buffers as input (no Python-side shape variation)
2. Contains no Python control flow that changes between steps (no `if self.key_cache is not None`)
3. Contains no CPU-side tensor creation (`torch.arange`, etc.)

Think about:
- Write a new method `decode_one_token(self, input_ids, position, cache_position)` that:
  - Looks up token embedding from `input_ids`
  - Looks up position embedding from `position` (pre-computed, not `torch.arange`)
  - Runs through all transformer blocks
  - Each block's attention writes to `key_cache[:, :, cache_position, :]` instead of `torch.cat`
  - Each block's attention attends over the full static cache with a mask
  - Returns logits
- `cache_position` is a single integer tensor — it tells each layer where to write the new K/V.

**Important:** The `if self.key_cache is not None` branch in `CausalSelfAttention` must be eliminated. During graph capture, both branches would be traced, and the graph can't handle conditional execution. The decode path must be unconditional: always write at `cache_position`, always attend over the full cache.

---

## Hint 4: The Capture-and-Replay Pattern

PyTorch's CUDA graph API is straightforward:

```python
# Phase 1: Warmup (run the function once without capturing, to trigger allocations)
with torch.cuda.stream(s):
    decode_one_token(static_input, static_pos, static_cache_pos)
torch.cuda.synchronize()

# Phase 2: Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=s):
    static_output = decode_one_token(static_input, static_pos, static_cache_pos)

# Phase 3: Replay (called many times)
static_input.copy_(real_token)
static_pos.fill_(real_position)
static_cache_pos.fill_(real_cache_position)
graph.replay()
# static_output now contains the result for real_token
```

Think about:
- **Why the warmup?** PyTorch allocates GPU memory lazily. If the first allocation happens *during* graph capture, the allocator state becomes part of the graph, and replay would try to re-allocate (and crash). The warmup forces all allocations to happen before capture.
- **Why a separate stream?** Graph capture records everything that happens on a CUDA stream. You want an isolated stream so that unrelated GPU work doesn't get captured.
- **`static_output`:** The output tensor created during capture is also at a fixed address. After replay, it contains the new results. You read logits from `static_output` directly — no return value from `graph.replay()`.

**Question:** What happens to `self.key_cache` after capture? The graph recorded the operation "write `k` into slot `cache_position` of `self.key_cache`". On each replay, it writes the new `k` (from the new input) into the slot indicated by `cache_position`. The cache is a static tensor — it persists across replays and accumulates K/V values. This is exactly what you want.

---

## Hint 5: The Generate Loop with Graph Replay

Your new `generate_cuda_graph()` function should:

1. **Prefill** — Run normally (no graph). Variable sequence length, runs once. This populates the KV cache for positions 0..T_prompt-1.
2. **Capture** — After prefill, capture one decode step as a graph.
3. **Decode loop** — For each new token:
   - Copy `idx_next` into `static_input_ids`
   - Set `static_cache_pos` to the current position
   - `graph.replay()`
   - Read logits from `static_output`
   - Sample next token

Think about:
- Prefill still uses `torch.cat` or writes to cache positions 0..T_prompt-1 in a loop. It doesn't need to be graph-captured (it runs once, and variable-length prefill would require separate graphs per length).
- The graph is captured after prefill, with the cache already partially filled. Replays continue filling subsequent positions.
- You need a `cache_position` counter that increments each step.

---

## Hint 6: Why Dynamic Shapes Break Graphs (and How Production Engines Solve It)

Your model has `B=1` during generation, so batch size is static. But in production engines with continuous batching, the batch size changes every step (requests arrive and complete). This is the central tension:

**CUDA graphs need static shapes. Continuous batching produces dynamic shapes.**

Production solutions:

1. **Bucketed graphs** — Pre-capture graphs for batch sizes 1, 2, 4, 8, 16, 32, ... At runtime, pad the actual batch to the nearest bucket size and replay that graph. Wasted compute on padding positions, but graph replay is so fast it's worth it.

2. **Piecewise capture (SGLang)** — Instead of capturing the entire forward pass as one graph, capture each transformer layer separately. Layers have the same shapes regardless of what comes before/after, so you can mix and match.

You don't need to implement bucketed graphs for this exercise — your `B=1` is fixed. But understanding *why* the bucket approach exists is the conceptual goal.

**Question:** If you wanted to support `B=1` and `B=4` generation, how many separate graph captures would you need? What about `B=1` through `B=64`? This is why production engines use powers-of-2 buckets instead of capturing every possible batch size.

---

## Hint 7: Benchmark — Proving the Speedup

Design benchmarks that isolate the graph replay benefit:

### Benchmark 1: Decode Step Latency
Time a single decode step (one token, KV cache already populated). Compare:
- Eager mode (no graph) — your current `generate_kv_cache`
- CUDA graph replay — your new `generate_cuda_graph`

The difference is pure launch overhead. At 210K params, expect ~2-5× speedup per decode step.

### Benchmark 2: End-to-End Throughput
Generate 64 tokens. Measure total time and tokens/second. The graph version should be faster because every decode step saves launch overhead.

### Benchmark 3: Warmup Cost
Measure how long the initial graph capture takes. This is a one-time cost — typically 50-200ms. Report it separately from per-step decode latency to show the amortization.

### Benchmark 4: Graph vs Eager (Large Batch)
If you implement batch support, compare at B=1 vs B=8 vs B=32. The graph speedup should be relatively constant (it's eliminating a fixed CPU overhead), but the eager mode overhead becomes a smaller fraction of total time at large batch sizes.

**Important:** Use `torch.cuda.synchronize()` before and after timing to ensure you're measuring GPU time, not just CPU dispatch time. Use `torch.cuda.Event` for precise GPU timing:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... your code ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

---

## Summary of Changes vs. `nanogpt-fused-attention.py`

| Component | What Changes |
|-----------|-------------|
| `device` | `'cpu'` → `'cuda'` |
| `CausalSelfAttention.__init__` | Pre-allocate static `key_cache` and `value_cache` as `(B, n_head, block_size, head_size)` |
| `CausalSelfAttention.forward` | Add a `decode_mode` path that writes to `cache[pos]` instead of `torch.cat` |
| `GPTLanguageModel` | Add `decode_one_token()` method — graph-safe, no dynamic shapes |
| `generate_kv_cache` | **Keep as-is** — baseline for benchmark comparison |
| New: `generate_cuda_graph` | Prefill eagerly → capture graph → replay loop |
| New: static buffers | `static_input_ids`, `static_position`, `static_cache_pos` |
| Training loop | **No changes** — graphs are inference-only |

---

## Recommended Implementation Order

1. **Step 1: Copy `nanogpt-fused-attention.py` → `nanogpt-cuda-graph.py`**
   - Change `device = 'cpu'` to `device = 'cuda'`.
   - Verify the model still trains and generates correctly on GPU.

2. **Step 2: Refactor KV cache to pre-allocated static tensors (Hint 1)**
   - Replace `torch.cat` with index-based writes.
   - Add `cache_position` tracking.
   - Verify generation still works in eager mode (no graphs yet).

3. **Step 3: Create `decode_one_token()` method (Hint 3)**
   - Eliminate all dynamic shapes and conditionals.
   - Replace `torch.arange` with a static position buffer.
   - Verify it produces the same logits as the current `forward()`.

4. **Step 4: Add static input buffers (Hint 2)**
   - Pre-allocate `static_input_ids`, `static_position`, `static_cache_pos`.
   - Wire `decode_one_token()` to read from these buffers.

5. **Step 5: Implement graph capture and replay (Hint 4)**
   - Write the warmup → capture → replay pattern.
   - Wire into `generate_cuda_graph()` (Hint 5).
   - Verify output matches eager-mode generation.

6. **Step 6: Benchmark (Hint 7)**
   - Decode step latency: graph vs eager.
   - End-to-end tokens/sec.
   - Report warmup cost separately.

---

## The Conceptual Map

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
