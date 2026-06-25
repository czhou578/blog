---
layout: post
title: "Testing Correctness Across Every Inference Optimization"
date: 2026-06-05
---

The scariest bugs in inference optimization are the ones that don't crash.

A wrong positional embedding doesn't throw an error - the model just produces slightly worse text. A KV cache that silently drops one entry doesn't segfault - the attention scores shift by a fraction and the output drifts in a way that looks like the model is being "creative." A broken attention mask in a fused batch doesn't fail loudly - it lets one request peek at another's KV entries, and the outputs look plausible but are subtly contaminated.

Over the course of this blog series, I've implemented KV caching, continuous batching, paged attention, prefix caching, chunked prefill, speculative decoding, and fused interleaved inference. Each one assumed it didn't break the model. Each one had plenty of opportunities to introduce exactly the kind of silent corruption described above.

This post describes 10 correctness equivalence tests that check each optimization against the simplest possible baseline. The structure is the same every time: run the optimized path, run the naive path, compare outputs. If they diverge, the optimization has a bug.

The key testing technique: **most of these tests use greedy (argmax) decoding**. Greedy decoding eliminates randomness entirely. There is exactly one correct next token at every step. If the optimized path produces a different token than the baseline path, that is not a flaky test or a sampling artifact - it is a real bug in the code. The two tests that do involve sampling use statistical methods instead.

All 10 tests were run against the [trigram speculative decoding](/blog/2026/06/05/speculative-trigram) NanoGPT file, which exercises the full stack - KV caching, batching, paged attention, prefix caching, speculative decoding with both bigram and trigram draft models, chunked prefill, and fused interleaved inference. Every test passed.

## 1. Recompute vs. KV Cache

**What it tests:** the [KV cache](/blog/2026/05/10/adding-kv-cache-to-nanogpt) produces the same logits as recomputing the full sequence from scratch.

This is the most fundamental invariant in the entire codebase. Every other optimization builds on KV caching. If the cache silently drifts from a full recompute, everything downstream is wrong.

The test runs two paths:

```text
Cached path:
  prefill [prompt]  →  decode token_0  →  decode token_1  →  ...  →  decode token_N
  (each step appends to the KV cache)

Recompute path:
  forward([prompt, token_0, token_1, ..., token_N])
  (one big forward pass, no cache)
```

Then it compares logits at each decode position:

```python
for i in range(len(cached_logits)):
    recompute_logit = full_logits[0, prompt_len - 1 + i, :]
    if not torch.allclose(cached_logits[i], recompute_logit, atol=1e-5):
        all_close = False
```

The `atol=1e-5` tolerance is necessary. Float32 arithmetic is not associative - the order of additions in attention can produce slightly different results between the cached and recomputed paths. But `1e-5` is tight enough that any real bug (wrong position index, missing cache entry, transposed dimensions) will blow past it.

**What bugs this catches:** wrong positional embeddings in the cached path, off-by-one errors in cache length tracking, incorrect KV concatenation order.

## 2. Single Request vs. Batched Requests

**What it tests:** decoding each request alone produces the same tokens as decoding all requests together in a [continuous batch](/blog/2026/05/11/adding-continuous-batching).

The concern here is subtle. When you stack multiple requests into one forward pass using `_stack_kvs`, each request's attention computation must be completely independent. A bug in the stacking - say, a request accidentally attending to another request's KV entries - would silently corrupt outputs. You wouldn't see an error. The model would just produce slightly wrong tokens.

```text
Unbatched:
  Request A alone:  prefill → decode → decode → decode → [tokens_A]
  Request B alone:  prefill → decode → decode → decode → [tokens_B]
  Request C alone:  prefill → decode → decode → decode → [tokens_C]

Batched:
  Prefill each individually, then decode all together:
  step 1: forward([A_tok, B_tok, C_tok]) → next tokens for all three
  step 2: forward([A_tok, B_tok, C_tok]) → next tokens for all three
```

Greedy decoding makes this a hard pass/fail. If request B produces a different token sequence when batched with A and C than when decoded alone, the batching is broken.

**What bugs this catches:** cross-request attention leakage in stacked KV caches, incorrect position computation in batched forward passes, shape mismatches in `_stack_kvs` / `_unstack_kvs`.

## 3. Contiguous KV vs. Paged KV

**What it tests:** writing KV entries into a [paged block pool](/blog/2026/05/24/paged-att) and gathering them back produces the same decode output as a normal contiguous cache.

Paged attention adds a layer of indirection. Instead of one contiguous tensor per request, KV entries live in fixed-size blocks scattered across a shared pool. The concern is that the scatter-gather logic - `_write_kvs_to_pool` and `_gather_paged_kv` - might introduce off-by-one errors at block boundaries.

The test is deliberate about this. It runs multiple prompt lengths - 3, 4, 5, 7, 8 - specifically chosen to land on and off block boundaries (with `page_block_size=4`):

```python
prompt_lens = [pl for pl in [3, 4, 5, 7, 8]
               if pl + decode_steps <= block_size]
```

Prompt length 4 fills exactly one block. Prompt length 5 fills one block and spills one slot into a second block. Prompt length 7 fills one block and spills three slots. If the code computes `num_filled_slots // page_block_size` wrong, one of these lengths will catch it.

**What bugs this catches:** off-by-one errors in block boundary arithmetic, incorrect slot indexing within blocks, misaligned writes during decode appends.

## 4. Prefix-Cached vs. Normal Prefill

**What it tests:** loading KV blocks from a [prefix cache](/blog/2026/05/22/prefix-caching) and prefilling only the remaining suffix produces the same output as prefilling the entire prompt from scratch.

This test builds three prompts that share the same prefix but have different suffixes:

```python
prefix = torch.randint(0, vocab_size, (shared_prefix_len,)).tolist()
suffixes = [torch.randint(0, vocab_size, (unique_suffix_len,)).tolist()
            for _ in range(3)]
prompts = [prefix + s for s in suffixes]
```

The first request populates the prefix cache. The second and third requests should hit that cache, skip the shared prefix during prefill, and only run the suffix through the model. If the hash-chaining, block-slicing, or cache-loading logic has a bug, the cached path will produce different logits than the full-prefill path.

The test also verifies that the cache is actually being used:

```python
if i > 0 and req.cached_prefix_tokens == 0:
    print(f"    Request {i} didn't hit prefix cache (expected >0)")
```

A test that silently falls back to full prefill and still passes is not testing prefix caching at all.

**What bugs this catches:** incorrect hash chaining across prefix blocks, wrong KV slicing when loading cached blocks, cache misses due to block-size alignment errors.

## 5. Speculative Greedy vs. Autoregressive Greedy

**What it tests:** under greedy decoding, [speculative decoding](/blog/2026/05/26/spec-decode) produces the exact same tokens as standard autoregressive decoding.

This is a property of the math, not an implementation choice. Under argmax:

```text
If draft_token == target_argmax → accept_prob = p/q ≥ 1 → always accept
If draft_token != target_argmax → reject → resample from max(0, target - draft)
                                         → argmax of residual = target_argmax
```

So greedy speculative decoding is a no-op. The draft model's guesses don't matter - every accepted token is the target's argmax, and every resampled token after rejection is also the target's argmax. If the implementation produces different tokens, the accept/reject logic has a bug.

The test checks both the bigram and trigram draft models:

```python
bi_tokens = _greedy_spec_decode(prompt, bigram, is_trigram=False)
if auto_tokens[i] != bi_tokens:
    all_match = False

tri_tokens = _greedy_spec_decode(prompt, trigram, is_trigram=True)
if auto_tokens[i] != tri_tokens:
    all_match = False
```

The comparison is exact - not `allclose`, not within tolerance. Identical token sequences or failure.

**What bugs this catches:** incorrect accept/reject probability computation, wrong KV cache trimming after rejection, off-by-one in the verify input sequence, broken rolling context in the trigram draft model.

## 6. Speculative Sampling Distribution

**What it tests:** under temperature sampling, speculative decoding preserves the target model's output distribution.

This is the test that cannot use argmax. The entire point of speculative decoding under sampling is that the accept/reject math produces the *same distribution* as the target model, not the same deterministic sequence. So we need a statistical test.

The approach: generate a single token from the same prompt many times via (a) direct sampling from the target distribution and (b) the speculative accept/reject algorithm. Then compare the two histograms:

```python
def _sample_spec(seed, draft_model, is_trigram):
    for _ in range(num_samples):
        # Draft one candidate
        draft_tok = torch.multinomial(draft_probs, 1, generator=gen).item()

        # Accept/reject against known target distribution
        accept_prob = (p / q).clamp(max=1.0)
        if draw < accept_prob:
            output_token = draft_tok
        else:
            adjusted = torch.clamp(target_dist - draft_probs, min=0)
            output_token = torch.multinomial(adjusted / adjusted.sum(), 1).item()
```

Only one token is generated per trial. This is critical - if we generated multiple tokens, the RNG consumption would differ between the two paths (the spec-decode path sometimes draws an extra random number for rejection), causing the sequences to diverge for reasons unrelated to correctness.

The comparison uses a chi-squared test with a significance threshold of `p > 0.01`. Because chi-squared tests have an inherent ~1% false-positive rate, the test runs three independent seeds and requires at least two to pass:

```python
seeds = [100, 200, 300]
pass_count = 0
for seed in seeds:
    auto_counts = _sample_auto(seed)
    spec_counts = _sample_spec(seed + 1, trigram, is_trigram=True)
    if _chi_squared_ok(auto_counts, spec_counts):
        pass_count += 1
ok = pass_count >= 2
```

**What bugs this catches:** incorrect rejection probability formula, wrong adjusted distribution after rejection, broken temperature scaling in the draft model.

## 7. KV Cache Trim Consistency

**What it tests:** after trimming rejected candidates from the KV cache, the remaining cache is identical to a full recompute of the accepted sequence.

This test simulates the exact scenario that happens after every speculative decoding step:

```text
1. Prefill a prompt, decode 2 tokens → build a cache
2. Run a verify pass with 4 draft candidates → cache grows by 5 entries
3. "Accept" only 2 of 4 → trim the cache
4. Decode 1 more token from the trimmed cache → logits_A
5. Full recompute of (prompt + 2 decoded + 2 accepted) → logits_B
6. Assert logits_A ≈ logits_B
```

The test checks both the cache shape and the logit values:

```python
expected_cache_len = cache_len_before + 1 + accept_count
actual_cache_len = trimmed[0][0][0].shape[1]
shape_ok = actual_cache_len == expected_cache_len
```

A shape mismatch means `_trim_kv_cache` is keeping the wrong number of entries. A logit mismatch means the kept entries are correct in number but wrong in content - perhaps the trimming sliced along the wrong dimension, or the "keep" count was off by one.

**What bugs this catches:** off-by-one in the trim boundary, wrong slice dimension, incorrect `keep_new_tokens` accounting.

## 8. Draft Model Distribution Sanity

**What it tests:** the bigram and trigram draft models produce valid probability distributions - normalized, non-zero, and peaked on the right tokens.

This is the only test that doesn't need the target model at all. It constructs a tiny known corpus and checks that the draft model learns the obvious pattern:

```python
corpus = [0, 1, 2, 0, 1, 2, 0, 1, 2]
tri = TrigramDraftModel(corpus, vocab_size=3, device=device)
probs_01 = tri.get_probs(0, 1)

# After context (0, 1), token 2 always followed. With smoothing, P(2|0,1) > 0.5
assert probs_01[2] > 0.5
```

It also checks normalization across a sweep of contexts:

```python
for prev in range(0, 20, 5):
    for cur in range(0, 20, 5):
        s = tri_big.get_probs(prev, cur).sum().item()
        assert abs(s - 1.0) < 1e-5
```

A distribution that doesn't sum to 1.0 would silently break `torch.multinomial` sampling and corrupt the accept/reject probability ratios.

**What bugs this catches:** incorrect Laplace smoothing, unnormalized rows in the probability table, dtype precision loss during normalization.

## 9. Chunked Prefill vs. Full Prefill

**What it tests:** prefilling a prompt in small chunks (accumulating the KV cache across multiple forward calls) produces the same logits as prefilling the entire prompt in one shot.

[Chunked prefill](/blog/2026/05/13/adding-chunked-prefill) is a scheduling optimization - it breaks a long prompt into pieces so that active decode requests aren't starved. But the final logits must be identical regardless of how the prompt was split.

```python
# Full prefill: one forward pass
logits_full, _, kvs_full = model(prompt_t, pos=pos_full)

# Chunked prefill: multiple forward passes
kvs_chunked = None
for start in range(0, prompt_len, chunk_size):
    end = min(start + chunk_size, prompt_len)
    chunk = torch.tensor([prompt[start:end]], ...)
    pos = torch.arange(start, end, ...)
    logits_chunked, _, kvs_chunked = model(chunk, pos=pos, past_kvs=kvs_chunked)
```

The test compares last-position logits, KV cache shapes, and the first greedy-decoded token. All three must match.

**What bugs this catches:** incorrect position threading across chunk boundaries, KV cache concatenation errors, off-by-one in `prefill_cursor` logic.

## 10. Fused Interleaved vs. Sequential

**What it tests:** packing a decode token and a prefill chunk into one [fused batched forward pass](/blog/2026/05/29/interleave) (with left-padding and attention masks) produces the same logits as running them as separate forward passes.

This is the most complex test because fused interleaved batching is the most complex optimization. The scenario:

```text
Request A: already prefilled, has a KV cache, is decoding one token
Request B: new arrival, needs a prefill chunk of 4 tokens

Sequential:  run A's decode alone, run B's prefill alone
Fused:       pack both into one (2, T_max) forward pass with masking
```

The fused batch uses left-padding for the shorter decode row, an attention mask to prevent B from seeing A's KV cache, and an input mask to mark which positions in each row are real:

```python
attn_mask = torch.zeros((2, 1, cache_len_a), dtype=torch.bool, device=device)
attn_mask[0, :, :] = True   # A can see its entire cache
# attn_mask[1] stays False - B has no real cached positions

input_mask = torch.zeros((2, t_max), dtype=torch.bool, device=device)
input_mask[0, -1] = True            # only last position is real for decode
input_mask[1, :chunk_size] = True   # first chunk_size positions are real for prefill
```

If the attention mask, input mask, or left-padding logic is wrong, one of the two rows will produce different logits than its sequential counterpart. The test checks both independently.

**What bugs this catches:** incorrect attention masking in fused batches, wrong position assignment for left-padded decode rows, cross-request attention leakage through improperly zeroed KV entries.

## What Each Test Guards Against

| Test | Optimization | Baseline | Method | Bug Class |
|---|---|---|---|---|
| 1 | KV cache | Full recompute | Logit comparison | Wrong positions, bad concatenation |
| 2 | Continuous batching | Unbatched decode | Token comparison | Cross-request leakage, stacking errors |
| 3 | Paged attention | Contiguous cache | Token comparison | Block boundary off-by-one |
| 4 | Prefix caching | Full prefill | Token comparison | Hash chain errors, wrong KV slicing |
| 5 | Speculative decode | Autoregressive | Exact token match | Accept/reject math, trim errors |
| 6 | Speculative decode | Direct sampling | Chi-squared test | Distribution preservation |
| 7 | KV trim | Full recompute | Logit comparison | Trim boundary errors |
| 8 | Draft models | Known corpus | Normalization check | Smoothing, normalization bugs |
| 9 | Chunked prefill | Full prefill | Logit comparison | Chunk boundary threading |
| 10 | Fused batching | Sequential forward | Logit comparison | Mask and padding errors |

## Running The Tests

I ran the full suite against [nanogpt-trigram-spec-decode.py](https://github.com/czhou578/multimodal-inference-visualizer/blob/main/nanogpt-trigram-spec-decode.py), the most complete version of the NanoGPT inference engine that includes every optimization from this blog series. The tests finish in a few seconds on CPU after training:

```python
from benchmarks.test_correctness_equivalence import run_all_correctness_tests
run_all_correctness_tests(model, vocab_size=vocab_size, device=device,
                          block_size=block_size, train_data=train_data,
                          val_data=val_data)
```

```text
============================================================
  Correctness Equivalence Tests
============================================================
  ✅ PASS: recompute logits == kv-cached logits
  ✅ PASS: unbatched == batched output (argmax)
  ✅ PASS: contiguous KV == paged KV (argmax)
  ✅ PASS: prefix-cached == normal prefill (argmax)
  ✅ PASS: speculative greedy == autoregressive greedy (argmax)
  ✅ PASS: spec-decode distribution ≈ target distribution (chi²)
  ✅ PASS: KV cache trim consistency
  ✅ PASS: draft model distributions are valid
  ✅ PASS: chunked prefill == full prefill
  ✅ PASS: fused interleaved == sequential

  10/10 tests passed.
============================================================
```

These tests don't prove the optimizations are fast. The [benchmark posts](/blog/2026/05/29/benchmarks-one) handle that. These tests prove the optimizations are correct. That is the prerequisite.

You can find the full test code here: [https://github.com/czhou578/multimodal-inference-visualizer/blob/main/benchmarks/test_correctness_equivalence.py](https://github.com/czhou578/multimodal-inference-visualizer/blob/main/benchmarks/test_correctness_equivalence.py)

CZ
