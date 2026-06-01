---
layout: post
title: "Benchmarking Speculative Decoding"
date: 2026-05-29
---

Previously, for all of the posts in the series, we have been focused on implementing optimizations to NanoGPT.

But we haven't done a comprehensive testing and benchmarking article / suite for this massive upgrade. 

In this article, which will be a part of a series, we will progressively benchmark and measure the impact of each optimization.

## Setup

We want to create a BenchMarkConfig class, which will hold all the configuration for our benchmarks.

In addition, we want to adjust the hyperparameters in nanoGPT to accelerate iteration. For me, I was using my Intel CPU and thus had to reduce the number of iterations and evaluation steps significantly. It doesn't actually matter whether or not we are getting comprehensible output, because we are only interested in the time it takes to generate tokens.*

*For the rest of this series, we will ignore the quality of the outputs in order to focus on benchmarking the actual inference optimizations. 

## Baseline Generation vs KV Cached Generation

For these set of experiments, I defined a baseline where we have no KV cache implemented. This implementation would repeatedly recalculate all the key-value matrices from scratch for each and every single decoding step. This is in contrast to the optimized version where we store all the key-value matrices, and only calculate the new one at each step. 

The output reports:

| Metric | Meaning |
|---|---|
| `tokens` | Number of new tokens generated. |
| `wall_time_s` | Total elapsed time for the generation run. Lower is better. |
| `tokens_per_s` | Generated tokens divided by wall time. Higher is better. |
| `ttft_ms` | Time to first token in milliseconds. Lower is better. |
| `KV-cache throughput speedup` | `kv_cache tokens_per_s / no_cache tokens_per_s`. Higher is better. |

The most important metric here is tokens per second, because this benchmark is focused on decode throughput. TTFT is still useful, but in this particular harness it is affected by the difference between the no-cache first step and the cached prefill step, so it should be interpreted more carefully.

These are the main benchmark cases:

| Case | Prompt Len | Generated Tokens | No Cache Tok/s | KV Cache Tok/s | Speedup | Wall Time Reduction |
|---|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 16 | 126.67 | 185.93 | 1.47x | 31.8% |
| `medium_generation` | 16 | 32 | 102.20 | 265.23 | 2.60x | 61.5% |
| `longer_generation` | 16 | 48 | 143.56 | 276.58 | 1.93x | 48.1% |
| `heavier_prompt` | 32 | 32 | 119.99 | 274.34 | 2.29x | 56.3% |
| `near_context_limit` | 8 | 56 | 209.39 | 233.16 | 1.11x | 10.2% |


### Overall Trend

The named runs show a strong throughput advantage for KV caching. The largest speedup appears in `medium_generation`, where cached generation reaches **265.23 tokens/sec** compared with **102.20 tokens/sec** for the no-cache baseline. That is a **2.60x throughput improvement** and cuts wall time by about **61.5%**.

The `heavier_prompt` result is also important. It uses a longer prompt of 32 tokens and generates 32 new tokens. The KV-cache path reaches **274.34 tokens/sec**, while the no-cache path reaches **119.99 tokens/sec**, giving a **2.29x speedup**. This matches the expected behavior: as the prompt gets longer, recomputing the full context becomes more wasteful, so caching previous keys and values becomes more valuable.

## Generation Length Sweep

The generation length sweep holds the prompt length fixed at 8 tokens and varies the number of generated tokens:

| Generated Tokens | No Cache Tok/s | KV Cache Tok/s | Speedup | Wall Time Reduction |
|---:|---:|---:|---:|---:|
| 8 | 222.60 | 275.18 | 1.24x | 18.9% |
| 16 | 109.83 | 193.64 | 1.76x | 43.3% |
| 32 | 225.47 | 227.41 | 1.01x | 0.8% |
| 48 | 130.52 | 233.10 | 1.79x | 44.0% |
| 56 | 188.92 | 275.38 | 1.46x | 31.4% |

### Sweep Trend

The sweep mostly confirms that KV caching improves decode throughput as generation length increases. The strongest sweep result is at **N=48**, where KV caching reaches **233.10 tokens/sec** compared with **130.52 tokens/sec** for no-cache generation, a **1.79x speedup**.

The **N=16** case is also strong, with a **1.76x speedup**. The **N=56** case remains meaningfully faster at **1.46x**.

The unusual row is **N=32**, where KV caching is only **1.01x faster**. That does not mean the KV cache stopped working. It is more likely a measurement artifact caused by the small CPU workload. The total wall times are only about **0.14 seconds** for both methods, so a small amount of scheduler noise, allocator behavior, Python overhead, or CPU frequency variation can move the result noticeably.

## TTFT Interpretation

The TTFT results are mixed:

| Case | No Cache TTFT | KV Cache TTFT | Change |
|---|---:|---:|---:|
| `small_smoke_test` | 5.26 ms | 7.15 ms | KV cache slower by 1.89 ms |
| `medium_generation` | 4.37 ms | 6.56 ms | KV cache slower by 2.19 ms |
| `longer_generation` | 3.84 ms | 3.64 ms | KV cache faster by 0.20 ms |
| `heavier_prompt` | 4.42 ms | 3.61 ms | KV cache faster by 0.81 ms |
| `near_context_limit` | 3.86 ms | 3.58 ms | KV cache faster by 0.28 ms |

This variation is expected. In the cached path, TTFT includes the prefill pass over the prompt plus sampling the first output token. In the no-cache path, the first decode step also processes the prompt, but the exact measured path differs because the benchmark toggles between train and eval behavior and uses different attention branches.

For serving systems, TTFT usually separates into:

1. Queueing and scheduling delay.
2. Prefill time.
3. Time to sample and emit the first token.

This benchmark does not model queueing, batching, network transfer, streaming, or scheduler delay. So TTFT here should be treated as a local implementation sanity check, not a complete user-facing latency measurement.

The throughput numbers are the clearer signal.

## Why The Speedup Is Not Perfectly Monotonic

In theory, KV-cache benefits should become more obvious as the generated sequence gets longer or as the prompt gets longer. The benchmark broadly shows that, but the numbers are not perfectly monotonic. There are several reasons.

### 1. The Model Is Tiny

At **0.056769M parameters**, this model is small enough that framework overhead can be a large fraction of total runtime. In a larger transformer, attention and MLP compute dominate more of the timing, so avoiding repeated work tends to show up more cleanly.

### 2. The Benchmark Runs On CPU

CPU timing is sensitive to operating system scheduling, cache locality, thread behavior, and frequency scaling. Since many runs complete in under half a second, small timing fluctuations can noticeably change tokens/sec.

### 3. The KV Cache Uses Tensor Concatenation

The implementation stores `key_cache` and `value_cache` inside each attention head and appends with `torch.cat` on every decode step. This is simple and educational, but it is not how production inference engines usually manage KV memory.

Each append can allocate and copy tensors. As the cache grows, that overhead can partially offset the benefit of avoiding recomputation. Production systems usually preallocate KV buffers or use paged/block-based KV cache layouts to avoid repeated copying.

### 4. Sequence Length Is Capped At 64

The benchmark uses `block_size=64`, so the no-cache baseline never processes more than 64 tokens per step. This limits how bad the no-cache path can get. With longer context windows, the cost of recomputing the full context would grow, making the KV-cache advantage more dramatic.

### 5. The No-Cache Path Uses A Different Mode

The benchmark uses `model.train()` to force the no-cache path and `model.eval()` to enable the cache path. Dropout is set to `0.0`, so this should not introduce dropout randomness, but the model still takes different branches through the attention implementation. This is fine for demonstrating the optimization, but it is worth remembering when interpreting very small timing differences.

## Row-by-Row Interpretation

### `small_smoke_test`

This is a short sanity test with an 8-token prompt and 16 generated tokens. KV caching improves throughput from **126.67 tokens/sec** to **185.93 tokens/sec**, a **1.47x speedup**.

This confirms that the cached path is functional. Even on a tiny workload, avoiding repeated context processing saves enough time to show up.

### `medium_generation`

This is the strongest result: a 16-token prompt and 32 generated tokens. KV caching improves throughput from **102.20 tokens/sec** to **265.23 tokens/sec**, a **2.60x speedup**.

This is the clearest evidence that the benchmark is capturing the intended optimization. The generation length is long enough for repeated no-cache recomputation to accumulate, while the cached path can reuse previous keys and values.

### `longer_generation`

With a 16-token prompt and 48 generated tokens, KV caching gives a **1.93x speedup**. Wall time drops from **0.3344s** to **0.1735s**.

The speedup is lower than `medium_generation`, even though generation is longer. That is a reminder that this microbenchmark is noisy. Still, nearly doubling throughput is a strong result.

### `heavier_prompt`

This case increases prompt length to 32 while generating 32 tokens. KV caching gives a **2.29x speedup**.

This result is especially aligned with theory. A longer prompt means the no-cache path has more previous context to recompute at every step. The cached path pays for that prompt once, then reuses the cached keys and values.

### `near_context_limit`

This case generates 56 tokens with an 8-token prompt, filling the 64-token context. KV caching gives only a **1.11x speedup**.

This is lower than expected in a pure compute model, but understandable in this implementation. As generation approaches the context limit, the internal cache grows, and repeated `torch.cat` operations become more expensive. The no-cache path is also capped by `block_size=64`, so it never grows beyond that fixed context size. The result still favors KV caching, but the simple cache implementation leaves performance on the table.

## Main Conclusions

1. **KV caching works in this benchmark.** Every named benchmark case shows cached generation outperforming no-cache generation.

2. **The largest gains appear when repeated context work matters most.** `medium_generation` and `heavier_prompt` both exceed **2x speedup**.

3. **The current cache implementation is educational, not production-optimized.** Appending to cache tensors with `torch.cat` is simple but can introduce copying overhead.

4. **TTFT is not the main story here.** The TTFT numbers are mixed and measured over very short CPU timings. Throughput is the more reliable signal for this benchmark.

5. **The results likely understate the importance of KV caching for real LLMs.** Larger models, longer contexts, and optimized GPU kernels typically make the cost of recomputation much more expensive than it appears in this tiny CPU setup.

Here is the raw unedited result of my benchmarks:

```text
0.056769 M parameters
step 0: train loss 4.1800, val loss 4.1795
step 20: train loss 3.6072, val loss 3.6477
step 40: train loss 3.3308, val loss 3.3400
step 60: train loss 3.1136, val loss 3.1371
step 80: train loss 2.9638, val loss 2.9796
step 100: train loss 2.8327, val loss 2.8783
step 119: train loss 2.7903, val loss 2.8249

XJad, f zotg pyreds hosn avtfZISp.Tit iuominerl O? umr,n nu, d 

=== KV Cache Baseline Benchmark Suite ===
block_size=64, device=cpu


=== small_smoke_test ===
{'prompt_len': 8, 'N': 16}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 16     | 0.1263      | 126.67       | 5.26   
kv_cache | 16     | 0.0861      | 185.93       | 7.15   

KV-cache throughput speedup: 1.47x


=== medium_generation ===
{'prompt_len': 16, 'N': 32}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 32     | 0.3131      | 102.20       | 4.37   
kv_cache | 32     | 0.1206      | 265.23       | 6.56   

KV-cache throughput speedup: 2.60x


=== longer_generation ===
{'prompt_len': 16, 'N': 48}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 48     | 0.3344      | 143.56       | 3.84   
kv_cache | 48     | 0.1735      | 276.58       | 3.64   

KV-cache throughput speedup: 1.93x


=== heavier_prompt ===
{'prompt_len': 32, 'N': 32}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 32     | 0.2667      | 119.99       | 4.42   
kv_cache | 32     | 0.1166      | 274.34       | 3.61   

KV-cache throughput speedup: 2.29x


=== near_context_limit ===
{'prompt_len': 8, 'N': 56}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 56     | 0.2674      | 209.39       | 3.86   
kv_cache | 56     | 0.2402      | 233.16       | 3.58   

KV-cache throughput speedup: 1.11x


=== Generation Length Sweep ===


=== N=8 ===
{'prompt_len': 8, 'N': 8}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 8      | 0.0359      | 222.60       | 7.60   
kv_cache | 8      | 0.0291      | 275.18       | 3.29   

KV-cache throughput speedup: 1.24x


=== N=16 ===
{'prompt_len': 8, 'N': 16}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 16     | 0.1457      | 109.83       | 4.93   
kv_cache | 16     | 0.0826      | 193.64       | 10.41  

KV-cache throughput speedup: 1.76x


=== N=32 ===
{'prompt_len': 8, 'N': 32}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 32     | 0.1419      | 225.47       | 4.32   
kv_cache | 32     | 0.1407      | 227.41       | 3.41   

KV-cache throughput speedup: 1.01x


=== N=48 ===
{'prompt_len': 8, 'N': 48}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 48     | 0.3678      | 130.52       | 3.62   
kv_cache | 48     | 0.2059      | 233.10       | 5.75   

KV-cache throughput speedup: 1.79x


=== N=56 ===
{'prompt_len': 8, 'N': 56}
method   | tokens | wall_time_s | tokens_per_s | ttft_ms
---------+--------+-------------+--------------+--------
no_cache | 56     | 0.2964      | 188.92       | 4.47   
kv_cache | 56     | 0.2034      | 275.38       | 3.39   

KV-cache throughput speedup: 1.46x

```

![alt text]({{ site.baseurl }}/images/kv_cache_benchmark.png)

The code can be found here: 

## Single Request vs Continuous Batching

## What Was Benchmarked

The benchmark compares two serving strategies:

| Method | Behavior | Why It Matters |
|---|---|---|
| `single_request_sequential` | Serves each request to completion before starting the next request. Each request uses KV-cache generation. | This is the simple baseline. It minimizes per-request interference but cannot share work across concurrent requests. |
| `continuous_batching` | Maintains active requests, admits arrivals by scheduler step, prefills admitted requests, and batches active decode tokens into one forward pass. | This models the core idea used by LLM serving systems: every decode step can process one token for many requests at once. |

The benchmark uses uniform workloads:

- Same prompt length within each run.
- Same generation length within each run.
- `arrival_gap=0` for all runs in the result file, so all requests are immediately available.
- KV-cache decode is used by both strategies.

The uniform setup is intentional. Batched cached decode is easiest when active requests have aligned KV-cache lengths. Mixed prompt lengths, mixed output lengths, and real arrival timing would require padding, masks, or a more complete scheduler.

## Named Benchmark Results

| Case | Requests Configured | Max Batch | Seq Tok/s | Batched Tok/s | Speedup | Avg Lat Ratio | Avg TTFT Ratio | Batched Completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 4 | 271.52 | 524.11 | 1.93x | 1.37x | 1.94x | 5/8 |
| `more_requests_small_batch` | 16 | 4 | 266.90 | 476.62 | 1.79x | 1.25x | 1.74x | 5/16 |
| `more_requests_larger_batch` | 16 | 8 | 241.15 | 850.88 | 3.53x | 1.58x | 2.85x | 9/16 |
| `longer_generations` | 16 | 8 | 265.04 | 956.53 | 3.61x | 1.51x | 3.98x | 9/16 |
| `heavier_prompt` | 16 | 8 | 277.84 | 492.91 | 1.77x | 3.41x | 7.83x | 9/16 |
| `stress_batch_capacity` | 32 | 8 | 246.66 | 671.97 | 2.72x | 1.97x | 3.21x | 9/32 |

![alt text]({{ site.baseurl }}/images/contin_batching_benchmark.png)

## Row-by-Row Interpretation

### `small_smoke_test`

This run uses 8 configured requests, an 8-token prompt, 16 generated tokens, and `max_batch_size=4`. Continuous batching reports **524.11 tokens/sec** compared with **271.52 tokens/sec** for sequential serving, a **1.93x speedup**.

The batched path completes 5 of the 8 configured requests. Within that subset, the benchmark shows the intended batching effect: average batch size rises to **2.50**, and throughput nearly doubles.

### `more_requests_small_batch`

This run increases the configured workload to 16 requests while keeping `max_batch_size=4`. Throughput improves by **1.79x**, from **266.90 tokens/sec** to **476.62 tokens/sec**.

Because the batch cap is still 4, the completed batched subset remains 5 requests. The result shows that simply adding more queued requests does not help unless the scheduler can keep and admit them correctly.

### `more_requests_larger_batch`

This run raises `max_batch_size` from 4 to 8. Throughput jumps to **850.88 tokens/sec**, a **3.53x speedup**.

This is the cleanest comparison against `more_requests_small_batch`: larger batches create more opportunity to amortize each forward pass. Average latency and TTFT also rise, showing the cost of waiting and sharing.

### `longer_generations`

This run keeps `max_batch_size=8` but doubles generation length to 48 tokens. It produces the strongest named speedup: **3.61x**.

Longer generations make continuous batching more useful because there are more decode steps over which the active batch can remain full. Once prefill is done, the engine gets many chances to run efficient batched decode.

### `heavier_prompt`

This run increases prompt length to 16 and generates 32 tokens. Throughput still improves by **1.77x**, but latency gets much worse.

Average TTFT rises from **5.17 ms** to **40.45 ms**. This is the most important latency warning in the file. The benchmark handles prefill one request at a time, so heavier prompts delay first-token emission and reduce the relative benefit of batched decode.

### `stress_batch_capacity`

This run configures 32 requests with `max_batch_size=8`. Continuous batching reports a **2.72x throughput speedup**, but completes only 9 of 32 configured requests.

The result still shows that batched decode is faster than sequential decode, but this row should not be interpreted as "the batched engine served all 32 requests quickly." It served a subset quickly due to the admission-loop limitation.

## Main Conclusions

1. **Continuous batching increases throughput when the batch size is large enough.** The sweep shows a clear scaling pattern from **0.68x** at batch size 1 to **5.99x** at batch size 16.

2. **Batching has overhead.** With `max_batch_size=1` and `max_batch_size=2`, the continuous batching path is slower or roughly equal to sequential serving because it adds scheduling, KV stacking, and bookkeeping without enough parallel work to amortize that overhead.

3. **Latency and TTFT worsen in this scaffold.** This is expected: the engine is optimizing system throughput, not single-request latency. The effect is especially strong when prompt prefill is heavier.

4. **Prefill handling matters.** The `heavier_prompt` case shows that prefill can dominate first-token latency. This motivates chunked prefill and prefill/decode scheduling.

5. **The benchmark currently under-completes the workload in the continuous path.** The request-count mismatch must be fixed before treating the ratios as a rigorous end-to-end serving comparison.

6. **The result still demonstrates the core principle.** Batched decode can emit many more tokens per second than serving one request at a time.


## No Prefix Cache vs Prefix Cache

![alt text]({{ site.baseurl }}/images/chunked_prefill_benchmark.png)

## The Wall-Clock Surprise

Even though prefix caching reduces prefill work, it does not improve generated-token throughput in this benchmark:

| Case | No Cache Gen Tok/s | Prefix Cache Gen Tok/s | Ratio |
|---|---:|---:|---:|
| `shared_prefix_basic` | 89.22 | 84.44 | 0.95x |
| `high_reuse_many_requests` | 80.64 | 75.48 | 0.94x |
| `multi_prefix_groups` | 81.10 | 76.43 | 0.94x |
| `low_reuse_control` | 89.53 | 78.98 | 0.88x |
| `eviction_pressure` | 81.30 | 68.75 | 0.85x |

This is the most important nuance in the file. The cache is reducing model prefill tokens, but the benchmark still gets slower overall.

The likely reasons are:

1. **The model is tiny.** With only `0.056769M` parameters, prefill compute is cheap. There is not much expensive model work to skip.
2. **Cache operations are expensive relative to model compute.** The benchmark clones KV tensors, slices blocks, concatenates cached blocks, hashes block tokens, and updates cache metadata.
3. **Requests are short.** Prompts are 24 to 28 tokens long. Prefix caching usually matters more when shared prefixes are much longer.
4. **Decode still dominates much of the run.** The benchmark reports generated-token throughput, and every request still decodes the same number of output tokens.
5. **The implementation is educational.** Production systems optimize KV memory layout, cache metadata, hashing, and block movement much more aggressively.

So the benchmark proves the mechanism, but not a wall-clock speedup.

## TTFT Behavior

Prefix caching increases TTFT in the shared-prefix cases:

| Case | No Cache Avg TTFT | Prefix Cache Avg TTFT | Ratio |
|---|---:|---:|---:|
| `shared_prefix_basic` | 2.79 ms | 7.96 ms | 2.85x |
| `high_reuse_many_requests` | 2.70 ms | 9.43 ms | 3.49x |
| `multi_prefix_groups` | 2.78 ms | 6.64 ms | 2.39x |
| `low_reuse_control` | 2.79 ms | 2.71 ms | 0.97x |
| `eviction_pressure` | 2.76 ms | 2.69 ms | 0.98x |

In a larger production system, prefix caching often improves TTFT for repeated long prompts because it avoids recomputing the shared prefix. Here, the opposite happens in the reuse cases because cache loading overhead is larger than the skipped compute.

The control and eviction rows have TTFT ratios near **1.0x**, which suggests the huge TTFT penalty in shared-prefix rows is tied to the cache-load path rather than normal request processing.

## Control Case: `low_reuse_control`

This workload uses unique prompts, so the prefix cache has nothing useful to reuse.

Results:

- Cached tokens: **0**
- Hit rate: **0.0%**
- Prefill reduction: **0.0%**
- Throughput ratio: **0.88x**
- Final blocks: **48**

This is a useful sanity check. The cache fills with blocks, but no later request has the same prefix chain, so there are no hits. The prefix-cache path is slower because it pays cache insertion and lookup overhead without any reuse benefit.

This row teaches an important serving lesson:

> Prefix caching only helps when prompts actually share prefixes.

If traffic has mostly unique prompts, prefix caching can add overhead without reducing prefill.

## Eviction Case: `eviction_pressure`

This workload has shared-prefix groups, but the cache is intentionally tiny:

```text
max_cache_blocks=8
num_groups=6
shared_prefix_len=16
prefix_block_size=4
```

Each shared prefix needs 4 blocks. Six groups need many more blocks than the cache can hold. The result:

- Cached tokens: **0**
- Hit rate: **0.0%**
- Evictions: **136**
- Throughput ratio: **0.85x**

This is the clearest failure mode. The cache constantly evicts blocks before they can be reused. That creates pure overhead: hashing, insertion, eviction, and metadata churn with no prefill-token reduction.

The lesson:

> Prefix caching needs enough capacity to keep hot prefixes resident.

If the cache is too small relative to the number of active prefix groups, it can perform worse than no cache.

## Why `prefill_tok/s` Drops

At first glance, `prefill_tok/s` looks worse for prefix caching. For example, in `high_reuse_many_requests`, it drops from **564.51** to **94.34**.

That metric is based on **actual prefill tokens**, not logical prompt tokens. Since prefix caching intentionally reduces actual prefill tokens from **672** to **120**, the denominator gets much smaller. The lower `prefill_tok/s` is not by itself a failure. It mostly says:

- fewer prefill tokens were actually run,
- but the total run still includes cache operations and decode work,
- so actual prefill tokens per wall second is not the clearest success metric for prefix caching.

For this benchmark, the clearer cache success metric is **actual prefill-token reduction**.

## Row-by-Row Interpretation

### `shared_prefix_basic`

Prefix caching avoids **112** of **192** prompt tokens, a **58.3%** prefill reduction. The cache hit rate is **77.8%**, and no blocks are evicted.

This confirms the basic mechanism works. However, throughput drops slightly from **89.22** to **84.44 generated tokens/sec**, and average TTFT rises from **2.79 ms** to **7.96 ms**. The cache saves prefill work, but the overhead is larger than the compute saved.

### `high_reuse_many_requests`

This is the strongest reuse case. Prefix caching avoids **552** of **672** prompt tokens, an **82.1%** prefill reduction. Hit rate reaches **85.2%**.

Despite that, throughput drops from **80.64** to **75.48 generated tokens/sec**. This shows how small-model benchmarks can hide the benefits of compute-saving optimizations: the skipped prefill is real, but not expensive enough to overcome cache overhead.

### `multi_prefix_groups`

With 4 prefix groups, prefix caching avoids **320** of **576** prompt tokens, a **55.6%** reduction. This is lower than the single-group high-reuse case because each group needs its own cache warmup.

Throughput again drops slightly, from **81.10** to **76.43 generated tokens/sec**. The mechanism works, but the wall-clock result is still overhead-bound.

### `low_reuse_control`

There are no shared prefixes, so prefix caching avoids **0** tokens and has a **0.0%** hit rate.

The prefix-cache path is slower: **78.98** vs **89.53 generated tokens/sec**. This is expected. Cache bookkeeping with no hits is pure overhead.

### `eviction_pressure`

The workload has shared-prefix groups, but the cache is too small. It ends with only **8** blocks and performs **136 evictions**.

The cache hit rate is **0.0%**, cached tokens are **0**, and throughput drops to **0.85x** of the no-cache baseline. This is the most important warning row: insufficient cache capacity can eliminate the benefit of prefix caching.

## Main Conclusions

1. **Prefix caching works mechanically.** In high-reuse workloads, it reduces actual prefill tokens by **55.6% to 82.1%**.

2. **This benchmark does not show wall-clock speedup.** Generated-token throughput is lower in every row, ranging from **0.85x to 0.95x** of the no-cache path.

3. **Overhead dominates at this scale.** The model, prompts, and benchmark windows are too small for saved prefill compute to outweigh cache-management overhead.

4. **Reuse pattern matters.** The low-reuse control gets no hits and becomes slower.

5. **Cache capacity matters.** The eviction-pressure case has shared prefixes but still gets no hits because blocks are evicted too aggressively.

6. **The most meaningful success metric here is prefill-token reduction.** Throughput will become more meaningful after scaling the model, prompt length, and cache implementation.

## Significance For LLM Serving

Prefix caching is important in real LLM serving because many workloads repeat large prompt prefixes:

- System prompts reused across many conversations.
- Tool instructions or policy text prepended to every request.
- Retrieval-augmented prompts with shared document headers.
- Multi-turn conversations that reuse a long chat history.
- Agents repeatedly calling a model with the same instruction scaffold.

In those settings, recomputing the shared prefix for every request wastes compute. Prefix caching stores KV blocks for the shared prefix and lets later requests resume from the cached state.

The benchmark demonstrates the core serving idea:

```text
first request:
    prefill shared prefix + unique suffix
    cache shared prefix blocks

later request:
    load shared prefix KV blocks
    prefill only unique suffix
```

In production, this can reduce TTFT, reduce prefill GPU load, and increase serving capacity. But the cache has to be implemented efficiently and sized for the traffic pattern.

## Caveats

These results should be interpreted as an educational microbenchmark:

- The model is extremely small.
- Prompt lengths are short.
- Prefix blocks are only 4 tokens.
- The cache stores cloned tensors and reconstructs prefixes with Python-level operations.
- The benchmark serves requests sequentially rather than through a full continuous batching scheduler.
- It reports one run, not repeated statistics.
- CUDA timings for tiny workloads can be noisy.

The qualitative result is still useful: cache hits and prefill-token savings are real, but cache overhead must be controlled for prefix caching to improve wall-clock latency.