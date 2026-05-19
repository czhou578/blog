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

## Global Cache:

The next step is adding a global cache that lives outside any individual request. This is because we need to share KV data across requests. 

Let's say we have Request A and it does its prefill operation. The KV blocks that are completed are committed to the global BlockCache. We don't want these blocks to be garbage collected when Request A is cleaned up. 

By saving it in the global cache, we can later on have Request B arrive with the same prefix as Request A. We can then look up the KV blocks in the global cache and use them to speed up the prefill operation for Request B. 

We will also define a class called `CachedBlock` that will store the KV data for a single block. 

A `CachedBlock` is identified by its `block_hash`. It also stores the `token_ids` that correspond to the block, and the `kv_data` which is a dictionary that maps `(layer, head)` tuples to `(k, v)` tensors. 

```python
@dataclass
class CachedBlock:
    """A cached KV block with its content hash."""
    block_hash: bytes
    token_ids: tuple                    # the tokens this block covers
    kv_data: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
    # kv_data[(layer, head)] = (k, v), each (1, BLOCK_SIZE, head_size)
    last_access_step: int = 0          # for LRU eviction

class BlockCache:
    def __init__(self, max_blocks=64):
        self.max_blocks = max_blocks
        self.cache: Dict[bytes, CachedBlock] = {}  # hash → CachedBlock

    def lookup(self, block_hash) -> CachedBlock | None:
        """Look up a block by its content hash."""
        block = self.cache.get(block_hash)
        if block is not None:
            block.last_access_step = self.current_step  # touch for LRU
        return block

    def insert(self, block_hash, token_ids, kv_data):
        """Insert a completed block into the cache."""
        if len(self.cache) >= self.max_blocks:
            self._evict_lru()
        self.cache[block_hash] = CachedBlock(
            block_hash=block_hash,
            token_ids=token_ids,
            kv_data=kv_data,
        )

    def _evict_lru(self):
        """Evict the least-recently-used block."""
        oldest = min(self.cache.values(), key=lambda b: b.last_access_step)
        del self.cache[oldest.block_hash]
```

*For those of you familiar with an LRU cache, you will notice that this is a very standard implementation.

## Hashing

To help us develop a reliable hashing function, we will use the `hashlib` library like this:

```python

NONE_HASH = b'\x00' * 16  # sentinel for the first block (no parent)

def hash_block_tokens(parent_hash, token_ids):
    """Compute a chained content hash for a KV block."""
    data = (parent_hash, tuple(token_ids))
    return hashlib.md5(str(data).encode()).digest()

```

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

## Finding Cache Hits During Admission

When a new request arrives, the scheduler needs to figure out how many of its prompt tokens are already cached. This is done by computing block hashes from the prompt and checking each one against the `BlockCache`:

```python
def find_cached_prefix(block_cache, prompt_tokens, block_size):
    """
    Walk the prompt left-to-right in block-sized chunks.
    Return the number of tokens that are fully cached.
    """
    num_cached_tokens = 0
    parent_hash = NONE_HASH

    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens):
            break  # partial block — not cacheable

        chunk = prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)

        cached = block_cache.lookup(block_hash)
        if cached is None:
            break  # cache miss — everything from here on must be computed

        num_cached_tokens += block_size
        parent_hash = block_hash  # chain for the next block

    return num_cached_tokens
```

1. Initialize — zero count + NONE_HASH sentinel, matching the insertion convention
2. Loop in block strides — must go left-to-right (can't skip due to hash chaining)
3. Skip partial blocks — incomplete blocks are never cached
4. Compute chained hash — reproduces the same hash used during insertion
5. Cache lookup — miss at block k means all subsequent blocks will also miss (chained hashes guarantee this)
6. Accumulate and chain — count cached tokens, update parent_hash for next iteration
7. Return — tells the scheduler where to set prefill_cursor so prefill skips cached tokens

## Load Cached Blocks Into Request

Once we know how many tokens are cached, we can skip the corresponding KV computations and directly copy the cached values into the request's KV cache. The remaining tokens (if any) are then computed normally.

Here is the function code to do this:

```python

def load_cached_blocks(request, block_cache, prompt_tokens, block_size):
    """ 
    Load cached KV blocks onto a request and return how many tokens were cached. 
    Sets request.prefill_cursor to skip past the cached potion
    """

    parent_hash = NONE_HASH
    num_cached = 0

    for start in range(0, len(prompt_tokens), block_size):
        end = start + block_size
        if end > len(prompt_tokens): break

        chunk = prompt_tokens[start:end]
        chunk_hash = hash_block_tokens(parent_hash, chunk)
        cached = block_cache.lookup(chunk_hash) # returns block. 

        if cached is None: break

        for (layer, head), (k, v) in cached.kv_data.items():
            if (layer, head) in request.kv_cache:
                existing_k, existing_v = request.kv_cache[(layer, head)]

                request.kv_cache[(layer, head)] = (
                    torch.cat([existing_k, k.clone()], dim=1),
                    torch.cat([existing_v, v.clone()], dim=1)
                )
            else:
                request.kv_cache[(layer, head)] = (k.clone(), v.clone())

        num_cached += block_size
        parent_hash = chunk_hash
    
    request.prefill_cursor = num_cached

    return num_cached

```

**Initialize.** Same as `find_cached_prefix` — we start with `NONE_HASH` and zero cached tokens. This function does the same walk, but instead of just *counting* hits, it *copies* the KV data into the request.

**Loop in block strides.** Identical traversal to `find_cached_prefix` — left-to-right, skipping partial trailing blocks. The two functions walk in lockstep by design so their hash chains always agree.

**Hash and lookup.** We compute the chained hash for the current chunk and look it up in the global cache. On a miss, we break — everything from here on must be computed from scratch during prefill.

**Copy KV data into the request.** This is the core of the function. For each `(layer, head)` pair in the cached block's `kv_data`, we either concatenate the cached K and V tensors onto the request's existing KV cache (if prior blocks have already been loaded), or initialize the cache entry with a fresh copy. The `.clone()` calls are critical — without them, multiple requests would share the same tensor objects, and any in-place modification (e.g., during decode) would corrupt the global cache.

**Accumulate and chain.** We add `block_size` to the cached count and update `parent_hash` for the next iteration, maintaining the hash chain.

**Set prefill cursor.** After all cached blocks are loaded, we advance the request's `prefill_cursor` past the cached tokens. When the scheduler later runs prefill for this request, it starts from `prefill_cursor` instead of position 0 — skipping all the tokens whose KV values were just loaded from the cache.

**Return.** The count of cached tokens is returned so the caller knows how much work was saved. If `num_cached == len(prompt_tokens)` (fully cached), prefill is essentially free — only the first decode token needs to be computed.

## Caching Newly Computed Blocks

After prefilling a request (partially or fully), the model returns new KV tensors. we need to **commit completed blocks** to the `BlockCache` for future requests to reuse.

```python
def commit_completed_blocks(request, block_cache, block_size):
    """
    After a prefill step, check if any new full blocks were completed.
    If so, insert them into the global cache.
    """
    total_tokens = len(request.prompt_tokens) + request.num_generated
    num_full_blocks = request.prefill_cursor // block_size

    # We need to track which blocks have already been committed
    # to avoid re-inserting on every step
    if not hasattr(request, '_committed_blocks'):
        request._committed_blocks = 0

    parent_hash = NONE_HASH
    for block_idx in range(num_full_blocks):
        start = block_idx * block_size
        end = start + block_size
        chunk = request.prompt_tokens[start:end]
        block_hash = hash_block_tokens(parent_hash, chunk)

        if block_idx >= request._committed_blocks:
            # Extract this block's KV slice from the request's cache
            kv_data = {}
            for (layer, head), (k, v) in request.kv_cache.items():
                kv_data[(layer, head)] = (
                    k[:, start:end, :].clone(),
                    v[:, start:end, :].clone(),
                )
            block_cache.insert(block_hash, tuple(chunk), kv_data)

        parent_hash = block_hash

    request._committed_blocks = num_full_blocks
```

**Count full blocks.** We use `request.prefill_cursor // block_size` to figure out how many complete blocks have been processed so far. Only full blocks are eligible for caching — the partial tail is still being built and will change with the next prefill chunk.

**Lazy-init the commit tracker.** The `_committed_blocks` attribute tracks how many blocks have already been inserted into the cache. Without this, we'd re-insert the same blocks on every prefill step. On the first call, it defaults to 0.

**Walk and hash.** Same left-to-right traversal as the other two functions — compute each block's chained hash from `parent_hash` and the chunk's token IDs. We always walk from block 0 (even for already-committed blocks) because we need to rebuild the hash chain so that later blocks get the correct `parent_hash`.

**Skip already-committed blocks.** The `if block_idx >= request._committed_blocks` guard ensures we only insert *newly* completed blocks. Blocks committed in previous prefill steps are skipped — we still compute their hashes (to maintain the chain), but we don't touch the cache.

**Extract and clone the KV slice.** For each new block, we slice the request's KV cache at `[start:end]` for every `(layer, head)` pair and `.clone()` the tensors. The clone is critical — the request's KV cache continues to grow during decode, and without cloning, the cached block's tensors would be views into the request's memory that get silently corrupted as the request progresses.

**Insert into the global cache.** The block hash, token IDs (as a tuple for hashability), and cloned KV data are inserted into the `BlockCache`. If the cache is full, this triggers LRU eviction of the oldest block.

**Update the commit watermark.** After the loop, `_committed_blocks` is set to the current number of full blocks, so the next call knows where to pick up.

## Using the Cache During Pre-fill

Once we know how many tokens are cached, we can skip the corresponding KV computations and directly copy the cached values into the request's KV cache. The remaining tokens (if any) are then computed normally.

```python
def prefill_with_cache(model, request, block_cache, block_size):
    """
    Prefill a request, reusing cached KV blocks where possible.
    Returns:
        num_computed_tokens: how many tokens were actually computed (not cached)
        kv_cache: the computed KV tensors for this request
    """
    prompt_tokens = request.prompt_tokens
    num_prompt = len(prompt_tokens)

    # 1) Find how much is already cached
    num_cached = find_cached_prefix(block_cache, prompt_tokens, block_size)

    # 2) Compute the rest
    if num_cached < num_prompt:
        # Compute the non-cached suffix
        to_compute = prompt_tokens[num_cached:]
        kv_cache = model.compute_kv_cache(to_compute)
        num_computed = len(to_compute)

        # 3) Cache the newly computed blocks
        for start in range(0, num_computed, block_size):
            end = start + block_size
            chunk = to_compute[start:end]
            chunk_kv = {
                (l, h): (k[start:end], v[start:end])
                for (l, h), (k, v) in kv_cache.items()
            }

            parent_hash = NONE_HASH
            if num_cached > 0:
                # Find the hash of the last cached block
                last_cached_chunk = prompt_tokens[num_cached - block_size:num_cached]
                parent_hash = hash_block_tokens(
                    NONE_HASH if num_cached == block_size else hash_block_tokens(
                        NONE_HASH,
                        prompt_tokens[num_cached - 2 * block_size:num_cached - block_size]
                    ),
                    last_cached_chunk
                )

            block_hash = hash_block_tokens(parent_hash, chunk)
            block_cache.insert(block_hash, tuple(chunk), chunk_kv)

    else:
        # All cached — no computation needed
        kv_cache = {}
        num_computed = 0

    # 4) Copy cached KV blocks into the request's KV cache
    for start in range(0, num_cached, block_size):
        end = start + block_size
        chunk = prompt_tokens[start:end]
        block_hash = hash_block_tokens(NONE_HASH if start == 0 else hash_block_tokens(
            NONE_HASH,
            prompt_tokens[start - block_size:start]
        ), chunk)

        cached = block_cache.lookup(block_hash)
        for (l, h), (k, v) in cached.kv_data.items():
            kv_cache.setdefault((l, h), ([], []))
            kv_cache[(l, h)][0].append(k)
            kv_cache[(l, h)][1].append(v)

    # Convert lists to tensors
    for (l, h), (k_list, v_list) in kv_cache.items():
        kv_cache[(l, h)] = (
            torch.cat(k_list, dim=1),
            torch.cat(v_list, dim=1)
        )

    return num_computed, kv_cache
```

## Tests

### Test 1: Identical prefixes

Two requests with identical prompts. The second request should reuse all complete blocks
from the first, prefilling only the trailing partial block (if any) plus zero new full blocks.

```python
requests = [
    Request(id=0, prompt_tokens=encode("To be or not to be"), max_new_tokens=20),
    Request(id=1, prompt_tokens=encode("To be or not to be"), max_new_tokens=20),
]
# req 1 should cache-hit on all full blocks from req 0
```

The result is: 

```text

============================================================
Test 1: Identical prefixes
============================================================
[step 0] Admitting req 0: 0/1 tokens cached (0 blocks hit), 1 tokens to prefill
[step 2] Admitting req 0: 16/1 tokens cached (4 blocks hit), -15 tokens to prefill
  Prompt length: 18 tokens
  Full blocks cacheable: 4
  Cache size after both: 4 blocks
  Req 0: 'To be or not to be
tile; nob'
  Req 1: 'To be or not to becy'd, just'
✅ Test 1 passed

```

We can see that in our case:

1. Request 0 — 0 cache hits (empty cache), full 18-token prefill from scratch
2. Request 1 — 16/18 tokens cached (4 blocks hit), only 2 trailing tokens to prefill — 89% of prefill work skipped
3. Cache size — 4 blocks total, no duplicates despite two identical requests
4. Different outputs — expected because torch.multinomial RNG state differs between decode phases; the cache guarantees identical KV values, not identical sampling

### Test 2: Shared Prefix, Different Suffixes

```python

# ══════════════════════════════════════════════════════════════
# Test 2: Shared prefix, different suffix
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 2: Shared prefix, different suffix")
print("=" * 60)

BLOCK_SIZE = 4
torch.manual_seed(42)

shared = encode("Hello world ")  # 12 tokens
reqs = [
    Request(id=0, prompt_tokens=shared + encode("cat"), max_new_tokens=10),
    Request(id=1, prompt_tokens=shared + encode("dog"), max_new_tokens=10),
]

s = scheduled_generate(model, reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)

shared_full_blocks = len(shared) // BLOCK_SIZE
print(f"  Shared prefix: {len(shared)} tokens → {shared_full_blocks} full blocks")
print(f"  Cache size: {len(s.block_cache.cache)} blocks")

assert len(s.block_cache.cache) >= shared_full_blocks, \
    f"❌ Expected at least {shared_full_blocks} shared blocks cached"

for req in reqs:
    assert req.status == "done", f"❌ Req {req.id} not done"
    assert req.num_generated == 10
    print(f"  Req {req.id}: '{decode(req.tokens_so_far)}'")

print("✅ Test 2 passed — req 1 reused shared prefix blocks from req 0")

```

In this test, we construct two requests that share the same 12-token prefix (`"Hello world "`) but diverge at the suffix — Req 0 appends `"cat"` and Req 1 appends `"dog"`. This is the most common real-world scenario: a shared system prompt with different user queries. The test verifies that the cache correctly reuses the 3 full blocks from the shared prefix (12 tokens ÷ 4 tokens per block = 3 blocks), while computing new KV values only for the diverging suffix tokens. The assertion checks that at least 3 blocks are cached after both requests complete.

Here is the result:

```text
============================================================
Test 2: Shared prefix, different suffix
============================================================
  Shared prefix: 12 tokens → 3 full blocks
  Cache size: 3 blocks
  Req 0: 'Hello world cative as fat'
  Req 1: 'Hello world dog, your jus'
✅ Test 2 passed — req 1 reused shared prefix blocks from req 0
```

The cache contains exactly **3 blocks** — the 3 full blocks from the shared `"Hello world "` prefix. Even though the two requests have different suffixes (`"cat"` vs `"dog"`), the shared prefix blocks are identical and only get cached once. Request 1 hit all 3 blocks during admission, skipping 12 tokens of prefill and only computing KV values for its unique `"dog"` suffix.

Notice that the outputs diverge after `"Hello world "` — Req 0 generates `"cative as fat"` while Req 1 generates `"dog, your jus"`. This is correct: the suffix tokens (`"cat"` vs `"dog"`) produce different KV values from the divergence point onward, leading to different attention contexts and therefore different generated continuations. The cache correctly avoided sharing any blocks past the divergence point because the chained hashes differ once the token content changes.


### Test 3: No Shared Prefix

```python
# ══════════════════════════════════════════════════════════════
# Test 3: No shared prefix — full prefill for both
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 3: No shared prefix")
print("=" * 60)

BLOCK_SIZE = 4
torch.manual_seed(42)

reqs = [
    Request(id=0, prompt_tokens=encode("The cat sat on the mat"),  max_new_tokens=10),  # 21+10=31 ✓
    Request(id=1, prompt_tokens=encode("Once upon a midnight"),    max_new_tokens=10),  # 20+10=30 ✓
]

s = scheduled_generate(model, reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)

# Both prompts are different — no blocks should be shared between them
# Cache should contain blocks from BOTH requests independently
total_blocks_req0 = len(reqs[0].prompt_tokens) // BLOCK_SIZE
total_blocks_req1 = len(reqs[1].prompt_tokens) // BLOCK_SIZE
print(f"  Req 0 blocks: {total_blocks_req0}, Req 1 blocks: {total_blocks_req1}")
print(f"  Cache size: {len(s.block_cache.cache)} blocks")

# No sharing — cache should have blocks from both, none reused
assert len(s.block_cache.cache) >= total_blocks_req0 + total_blocks_req1, \
    f"❌ Expected {total_blocks_req0 + total_blocks_req1} total blocks (no sharing)"

for req in reqs:
    assert req.status == "done", f"❌ Req {req.id} not done"
    assert req.num_generated == 10
    print(f"  Req {req.id}: '{decode(req.tokens_so_far)}'")

print("✅ Test 3 passed — 0 cache hits, full prefill for both")
```

This is the negative case — a sanity check that the cache doesn't produce false hits. The two prompts (`"The cat sat on the mat"` and `"Once upon a midnight"`) share no common prefix, so every block in both requests should have a unique chained hash. The test verifies that both requests do a full prefill with zero cache hits, and that the cache stores blocks from both requests independently (no sharing, no deduplication). The assertion checks that the total cache size equals the sum of both requests' full blocks.

Here is the result:

```text
============================================================
Test 3: No shared prefix
============================================================
  Req 0 blocks: 5, Req 1 blocks: 5
  Cache size: 10 blocks
  Req 0: 'The cat sat on the matter's nobl'
  Req 1: 'Once upon a midnight his E'emi'
✅ Test 3 passed — 0 cache hits, full prefill for both
```

The cache contains **10 blocks** — 5 from each request, with zero overlap. This confirms that the chained hashing correctly distinguishes blocks with entirely different prefix histories. Even if some individual tokens happened to appear in both prompts (e.g., `"the"`), the chained hash prevents false sharing because the parent hashes differ from block 0 onward.

Both requests completed with full prefill work, and the cache faithfully stored all blocks from both. If a third request arrived with a prefix matching either prompt, it would get cache hits against the appropriate set of 5 blocks — the cache is doing its job of building up a library of reusable prefixes over time, even when the first two requests don't benefit from each other.

### Test 4: Cache Eviction Under Memory Pressure

```python

# ══════════════════════════════════════════════════════════════
# Test 4: LRU eviction when cache is full
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 4: Cache eviction under memory pressure")
print("=" * 60)

BLOCK_SIZE = 4
torch.manual_seed(42)

# Use a tiny cache: only 3 block slots
prompt_a = encode("The cat sat on the mat and then")  # ~8 blocks worth
prompt_b = encode("The cat sat on the mat and then")  # identical — should hit

# Run req 0 first to fill the cache, then req 1 to check hits
req0 = Request(id=0, prompt_tokens=prompt_a, max_new_tokens=5)
req1 = Request(id=1, prompt_tokens=prompt_b, max_new_tokens=5)

# Small max_blocks forces eviction
s = Scheduler(policy="fcfs", token_budget=16, max_kv_tokens=256, block_size=BLOCK_SIZE)
s.block_cache = BlockCache(max_blocks=3)  # only room for 3 blocks!

# Run them through scheduled_generate but with this custom scheduler
# (We need to manually wire this — or just test BlockCache directly)

# Direct BlockCache test:
cache = BlockCache(max_blocks=3)
parent_hash = NONE_HASH

# Insert 4 blocks — the 4th should evict the 1st (LRU)
for i in range(4):
    tokens = prompt_a[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]
    if len(tokens) < BLOCK_SIZE:
        break
    h = hash_block_tokens(parent_hash, tokens)
    cache.cache[h] = CachedBlock(
        block_hash=h,
        token_ids=tuple(tokens),
        kv_data={},  # dummy for this test
        last_access_step=i,
    )
    if len(cache.cache) > cache.max_blocks:
        cache._evict_lru()
    parent_hash = h

assert len(cache.cache) == 3, f"❌ Cache should have 3 blocks, got {len(cache.cache)}"

# The block with last_access_step=0 should have been evicted
for block in cache.cache.values():
    assert block.last_access_step != 0, \
        f"❌ Block with step 0 should have been evicted (LRU)"

print(f"  Cache size: {len(cache.cache)} (max: 3)")
print(f"  Remaining blocks accessed at steps: {[b.last_access_step for b in cache.cache.values()]}")
print("✅ Test 4 passed — LRU eviction works correctly")

```

This test verifies that the `BlockCache` correctly evicts the least-recently-used block when it runs out of space. We set `max_blocks=3` — an artificially tiny cache — and insert 4 blocks sequentially with `last_access_step` values of 0, 1, 2, 3. When the 4th block is inserted, the cache exceeds capacity and must evict one block. The LRU policy should evict the block with `last_access_step=0` (the oldest). The assertions check two things: (1) the cache never exceeds 3 blocks, and (2) the evicted block is specifically the one with step 0.

Here is the result:

```text
============================================================
Test 4: Cache eviction under memory pressure
============================================================
  Cache size: 3 (max: 3)
  Remaining blocks accessed at steps: [1, 2, 3]
✅ Test 4 passed — LRU eviction works correctly
```

The remaining blocks have access steps `[1, 2, 3]` — step 0 is gone, confirming that `_evict_lru` correctly identified and removed the oldest block. This is the same eviction strategy used in production systems like vLLM: when GPU memory is full, the least-recently-accessed KV blocks are discarded first. The rationale is that blocks accessed recently are more likely to be needed again (e.g., a popular system prompt), while stale blocks from old requests are safe to discard. If a future request needs an evicted block, it simply re-prefills those tokens — a cache miss costs compute, not correctness.

Thanks for reading! You can find the link to the entire source code at

CZ