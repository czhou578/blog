---
layout: post
title: "Adding Prefix Caching to NanoGPT"
date: 2026-05-17
---

In the previous post, we discussed how to quantize NanoGPT. Since we didn't achieve significant improvements in performance due to the limitations of static quantization on our toy model, we're going to pivot to a different optimization technique: prefix caching. 

## Problem

So far, we have a scheduling system in NanoGPT that can handle multiple requests at a time. However, we are still recomputing the KV cache for each request every time we generate a new token. If the user were to send prompts that had a common prefix, we would be recomputing the KV cache for that common prefix every time. This is inefficient.

**Prefix caching** stores completed KV blocks in a content-addressed cache. When Request B arrives and its prompt starts with the same tokens as Request A, the scheduler finds the cached KV blocks, skips the prefill for those tokens, and only computes the **suffix** (e.g. `"Goodbye"`). This directly reduces TTFT or time to first token.

In production (vLLM's Automatic Prefix Caching), this cuts prefill compute by 50–90% for
workloads with shared system prompts — which is the vast majority of API deployments.

## Why This Matters Even at 210K Params

We won't see a meaningful wall-clock improvement on nanoGPT — the model is too small and the prompts too short for the cache lookup overhead to pay for itself. But the concepts are exactly what vLLM implements:

1. **Content-addressed hashing** — KV blocks are keyed by their token content, not by request ID or position.
2. **Chained hashes** — each block's hash includes its parent's hash, so the entire prefix history is captured transitively.
3. **LRU eviction** — when memory is full, the least-recently-used cached blocks are evicted
   to make room for new ones.
4. **The scheduler integrates cache hits** — cached tokens are subtracted from the work to do,
   so a fully-cached prefix means near-zero prefill cost.

The goal is to learn the architecture, not hit a perf number.

## Thinking in Blocks

Right now, our KV cache is per-request and per-(layer, head). 

For prefix caching, we are going to have think in terms of **fixed-size blocks** of tokens. 

The reason is that caching will be easier. If we had a request wih 100 tokens, storing each token in the cache would mean 100 lookups. 

In addition, memory management becomes easier, since we now have a natural unit of allocation. Each cached entry is a fixed size chunk per layer, and there is no fragementation of memory that you would otherweise have with per token storage. 

Finally, the semantics become very clear for the developer. If we see a block is in the cache, we can be certain that its hash is meaningful.

Choose a block size
(e.g. `BLOCK_SIZE = 4` — small enough to see the mechanics at nanoGPT scale). A prompt of
12 tokens becomes 3 blocks:

```
Block 0: tokens[0:4]   → KV for positions 0, 1, 2, 3
Block 1: tokens[4:8]   → KV for positions 4, 5, 6, 7
Block 2: tokens[8:12]  → KV for positions 8, 9, 10, 11
```

Each block stores a fixed-size KV chunk: `(1, BLOCK_SIZE, head_size)` per (layer, head).
Only **full** blocks (exactly `BLOCK_SIZE` tokens) are eligible for caching. The trailing
partial block is never cached — it changes with every new decode token.

**Question to ask yourself:** Why can't you cache partial blocks?

## Content Addressed Hashing with Parent Chains

The cache key for a block is **not** just its token IDs. It's a hash of:

```python
block_hash = hash((parent_block_hash, tuple(block_token_ids)))
```

**Why the parent hash?** Because KV values are context-dependent. Consider:

```
Request A: ["The", "cat", "sat", "on"] ["the", "mat", ".", "!"]
Request B: ["The", "dog", "sat", "on"] ["the", "mat", ".", "!"]
```

Block 1 (`["the", "mat", ".", "!"]`) has the **same token IDs** in both requests. But the
KV tensors are numerically different — in Request A, every token in Block 1 attended to
`"The cat sat on"`, while in Request B it attended to `"The dog sat on"`. The K and V
projections produce different values because the input `x` to the attention layer is different
(it was contextualized by a different prefix).

By chaining the parent hash, Block 1's hash in Request A encodes the full history through
Block 0 (`["The", "cat", "sat", "on"]`), which differs from Block 0 in Request B
(`["The", "dog", "sat", "on"]`). The two Block 1 hashes are therefore different, and the
cache correctly treats them as distinct entries.

**The transitive property:** if block `k` matches, it implies all blocks `0..k-1` also match.
A cache hit at any block guarantees the entire prefix up to that block is identical.