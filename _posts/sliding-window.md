---
layout: post
title: "Sliding Window KV Cache Eviction"
date: 2026-06-11
---

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>

In the [previous post](/blog/2026/06/09/radix-tree), we built a radix tree for prefix caching — a structural upgrade that protects shared prefixes from blind eviction. But there's a separate memory problem we haven't addressed: KV caches that grow without bound.

Every active request appends one KV entry per decode step. A request generating 200 tokens holds 200 key-value pairs per layer per head, even though the model's attention pattern may only meaningfully attend to the last 30–50 positions. For long-running generations, the cache grows linearly and eventually forces preemptions, even when most of the cached entries are contributing almost nothing to the output.

Sliding window eviction fixes this by capping each request's KV cache to the last *W* entries. After every decode step, any entries older than position `T - W` are trimmed. The model loses access to distant context, but gains a hard memory ceiling per request, which means more concurrent requests, fewer preemptions, and predictable memory usage.

This is the same mechanism used in Mistral's sliding window attention (SWA), where the window size is baked into the architecture itself. Here, we're applying it as an eviction policy on top of standard full attention, a scheduler-level optimization rather than an architectural change.

## The Problem: Unbounded Cache Growth

Here's what happens to three concurrent requests under our existing scheduler without any window:

<div class="mermaid">
graph LR
    subgraph "Step 0"
        A0["Req 0: 8 KV entries"]
        B0["Req 1: 6 KV entries"]
    end
    subgraph "Step 15"
        A15["Req 0: 23 KV entries"]
        B15["Req 1: 21 KV entries"]
        C15["Req 2: 18 KV entries"]
    end
    subgraph "Step 30"
        A30["Req 0: 38 KV entries"]
        B30["Req 1: 36 KV entries"]
        C30["Req 2: 33 KV entries"]
    end

    A0 --> A15 --> A30
    B0 --> B15 --> B30
    C15 --> C30
</div>

Each request's cache grows by one entry every step. With three requests generating 30 tokens each, and prompt lengths of 6–8 tokens, the total KV memory at step 30 is `38 + 36 + 33 = 107` tokens. Under a tight `max_kv_tokens` budget, this triggers preemptions — the scheduler evicts the lowest-priority request, clears its cache, and forces it to re-prefill from scratch when memory frees up. That re-prefill is pure wasted GPU work.

With a sliding window of `W = 20`, those same three requests would cap at `20 + 20 + 20 = 60` tokens total. No preemption, no re-prefill, no wasted work.

## The Eviction Function

The core logic is a single function that trims a request's KV cache to the last `W` entries:

```python
def evict_kv_cache(request, window_size):
    """
    Trim the request's KV cache to keep only the last `window_size` entries.
    
    Before: kv_cache[(layer, head)] = (k, v) with shape (1, T, hs)
    After:  kv_cache[(layer, head)] = (k, v) with shape (1, min(T, W), hs)
    """
    if window_size is None:
        return  # no window configured

    for (layer, head), (k, v) in request.kv_cache.items():
        T = k.shape[1]
        if T > window_size:
            request.kv_cache[(layer, head)] = (
                k[:, -window_size:, :],   # keep the LAST W entries
                v[:, -window_size:, :],
            )
```

The slice `k[:, -window_size:, :]` takes the last `W` entries along the sequence dimension (dim=1). This is a view into the existing tensor — no copy, no allocation. The old entries get garbage-collected when nothing else references them.

A few things to notice:

- **It operates on per-request caches**, not batched tensors. We call it *after* `disassemble_batch_cache` has split the batched output back into per-request KV pairs.
- **It's called every decode step**, not just when memory is tight. The window is a hard cap, not a reactive threshold. This is what makes the memory usage predictable.
- **The slice direction matters.** We keep the *last* W entries, not the first. Early tokens (the prompt) get evicted first, which makes sense — in a language model, recent context is generally more informative than distant context for next-token prediction.

## Integration with the Scheduler

The scheduler gains a `sliding_window` parameter that flows through to two places: the `_effective_kv_tokens` accounting method, and the eviction call in the decode loop.

### Memory Accounting

The scheduler needs to know how much KV memory each request *actually uses* (not how much it *would use* without eviction) when deciding whether to admit new requests:

```python
def _effective_kv_tokens(self, req):
    """How many KV entries this request actually holds (after eviction)."""
    total = len(req.prompt_tokens) + req.num_generated
    if self.sliding_window is not None:
        return min(total, self.sliding_window)
    return total
```

Without this, the scheduler would think each request uses `prompt_len + num_generated` KV entries, overestimating memory usage and refusing to admit new requests even though the caches are being trimmed. This is a subtle but important detail — the admission control and the eviction policy must agree on the accounting.

### Calling Eviction in the Decode Loop

The eviction call sits right after `disassemble_batch_cache`, before the next step's token sampling:

```python
if decode_reqs:
    batch_tokens = torch.cat([req._last_token for req in scheduler.active], dim=0)
    batch_positions = torch.tensor(
        [[len(req.tokens_so_far) - 1] for req in scheduler.active], device=device
    )

    past_kvs, attn_mask, pad_lengths = assemble_batch_cache(scheduler.active)
    logits, _, new_kvs = model(batch_tokens, pos=batch_positions, past_kvs=past_kvs, attn_mask=attn_mask)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)

    disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)

    # NEW: trim KV caches to sliding window size
    if scheduler.sliding_window is not None:
        for req in scheduler.active:
            evict_kv_cache(req, scheduler.sliding_window)

    for i, req in enumerate(scheduler.active):
        req.generated_tokens.append(idx_next[i].item())
        req._last_token = idx_next[i : i + 1]
```

The ordering is important: we evict *after* the model has used the full cache for this step's attention computation, but *before* the next step. This means the model always sees up to `W + 1` entries during attention (the window plus the just-appended token), and we trim back to `W` before storage.

### Why Not Evict Before Attention?

You might wonder: if we're going to evict anyway, why not trim the cache *before* feeding it to the model? The answer is that the current step's attention computation has already been paid for — the full cache was assembled and padded. Evicting before attention would require re-padding the batch (since different requests might have different post-eviction lengths), adding complexity for no benefit. Evicting after attention is simpler and produces the same result for the *next* step.

## The Attention Mask Interaction

There's a subtlety with how the sliding window interacts with `assemble_batch_cache`. When we batch multiple requests for a single forward pass, we left-pad shorter caches so all requests have the same sequence length:

```python
def assemble_batch_cache(requests):
    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_t = max(lengths)

    pad_lengths = [max_t - t for t in lengths]

    attn_mask = torch.zeros((B, 1, max_t), dtype=torch.bool, device=device)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, :, pad:] = True
    ...
```

The `attn_mask` marks padded positions as `False` so the attention layer masks them with `-inf` before softmax. With the sliding window, all active requests trend toward the same cache length (`W`), which means less padding overhead. Without the window, a request that started early might have 50 cached entries while a recently-admitted one has 12 — requiring 38 padding positions. With `W = 20`, the gap is at most a few entries.

This is a secondary benefit of the sliding window: it reduces the padding waste in batched attention.

## Benchmark Results

The benchmark suite runs four scenarios comparing full cache against windowed cache:

### Benchmark 1: Memory Savings (W=20)

Three requests with 30-token generations and moderate prompts:

| Config | Requests | Done | Gen Tokens | Preemptions | Avg Peak KV | Max Peak KV | Max Total KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_cache | 3 | 3 | 90 | 0 | 35.0 | 37 | 107 |
| window=20 | 3 | 3 | 90 | 0 | 20.0 | 20 | 60 |

Peak KV reduction: **45.9%**. Each request caps at 20 KV entries instead of growing to 35–37. The total memory high-water mark drops from 107 to 60 tokens — nearly half.

### Benchmark 2: Preemption Reduction Under Tight Memory (W=20)

Five requests competing for a tight `max_kv_tokens=50` budget with mixed priorities:

| Config | Requests | Done | Gen Tokens | Preemptions | Max Total KV |
|---|---:|---:|---:|---:|---:|
| full_cache | 5 | 5 | 80 | ≥1 | 50 |
| window=20 | 5 | 5 | 80 | 0 | ≤50 |

This is where the sliding window has its biggest practical impact. Without the window, the growing caches exceed the budget and force preemptions — the scheduler evicts a low-priority request, clears its cache, and re-prefills it later. With the window, caches stop growing at 20 entries, staying comfortably within the 50-token budget. Zero preemptions means zero wasted re-prefill work.

### Benchmark 3: Quality vs. Window Size

This is the trade-off. We generate 52 tokens from a single request under different window sizes and compare the output token-by-token against the full-cache baseline:

| Window | Agreement | Gen Tokens |
|---:|---:|---:|
| 8 | ~40-55% | 52 |
| 16 | ~55-70% | 52 |
| 32 | ~80-90% | 52 |
| 48 | ~95-100% | 52 |
| full | 100.0% | 52 |

Smaller windows diverge more from the baseline because the model loses access to distant context. At `W = 8`, the model can only "see" the last 8 tokens, so its predictions drift significantly. At `W = 48` (close to the full sequence length), the output is nearly identical to the baseline.

The practical question is: how much quality loss is acceptable for the memory savings? For many workloads — chatbots, code completion, streaming responses — a window of 32–64 tokens captures the vast majority of useful context. For tasks requiring long-range reasoning (document summarization, multi-step math), a larger window or full cache is necessary.

### Benchmark 4: Batch Capacity (W=16)

Eight small requests under a fixed `max_kv_tokens=80` budget:

| Config | Requests | Done | Avg Batch Size | Max Batch Size |
|---|---:|---:|---:|---:|
| full_cache | 8 | 8 | ~2.5 | 4 |
| window=16 | 8 | 8 | ~3.5 | 6 |

With the window, each request uses at most 16 KV entries instead of 26+, so more requests fit within the 80-token budget simultaneously. Higher batch sizes mean better GPU utilization — the model processes more tokens per forward pass, amortizing the fixed overhead of kernel launches, memory transfers, and Python dispatch.

## Why This Works (and When It Doesn't)

The sliding window exploits a property of autoregressive language models: attention weights are heavily skewed toward recent tokens. In practice, the attention distribution follows a roughly exponential decay — the model attends strongly to the last few tokens and weakly to distant ones. Trimming the oldest entries removes the weakest attention contributions.

<div class="mermaid">
graph TD
    subgraph "Full Cache (T=50)"
        F1["Positions 0-10: weak attention"]
        F2["Positions 11-30: moderate attention"]
        F3["Positions 31-50: strong attention"]
    end
    subgraph "Window=20"
        W1["Positions 31-50: strong attention ✓"]
    end

    F3 -.-> W1
</div>

This breaks down in specific cases:

- **Long-range dependencies.** "In the first paragraph, the author mentions X. Based on that..." — if X was 100 tokens ago and the window is 50, the model can't reference it.
- **Structured output.** JSON, code, or tabular output where opening brackets/tags from the beginning of the sequence constrain the closing structure.
- **Repetition control.** Models use attention to earlier parts of the sequence to avoid repeating themselves. A small window increases the risk of degenerate repetition loops.

For production systems, Mistral addresses this architecturally with a fixed window size chosen during training. Our approach is more flexible — the window is a runtime parameter — but the model wasn't trained with a window, so there's an inherent mismatch between what the attention heads expect (full context) and what they get (truncated context).

## Comparison with Production Approaches

| Approach | Where It Runs | Window Fixed? | Quality Impact |
|---|---|---|---|
| **Our implementation** | Scheduler (eviction) | Runtime parameter | Some — model expects full context |
| **Mistral SWA** | Architecture (attention mask) | Training-time constant | Minimal — model trained with window |
| **vLLM PagedAttention** | Memory manager (block-level) | N/A — uses paging instead | None — full cache, paged storage |
| **StreamingLLM** | Attention sink + window | Runtime parameter | Moderate — keeps first few + last W tokens |

Our implementation is closest to a simplified StreamingLLM without the attention sink (the observation that keeping the very first token's KV entry alongside the window significantly improves quality). Adding an attention sink would be a natural extension — keep positions 0–3 plus the last W entries, rather than just the last W.

## Summary

The sliding window is a simple idea with meaningful impact: cap each request's KV cache to the last *W* entries, and the system gains predictable memory usage, fewer preemptions, and higher batch concurrency — at the cost of losing access to distant context.

Three lines do the heavy lifting: the `evict_kv_cache` function that slices the tensors, the `_effective_kv_tokens` method that updates the scheduler's memory accounting, and the eviction call after `disassemble_batch_cache` in the decode loop. Everything else — the model, the attention heads, the batch assembly — is unchanged.

The full code can be found here: [https://github.com/czhou578/multimodal-inference-visualizer](https://github.com/czhou578/multimodal-inference-visualizer)

CZ