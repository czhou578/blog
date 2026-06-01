---
layout: post
title: "Benchmarking NanoGPT Optimizations - Part 1"
date: 2026-05-29
---

So far in this series, we have focused on implementing inference optimizations in NanoGPT. This post switches from implementation to measurement.

The goal is to benchmark three serving optimizations: KV caching, continuous batching, and chunked prefill. I am not measuring output quality here. The model is intentionally tiny and runs on CPU so the experiments are easy to iterate on. The useful signals are decode throughput, time to first token, prefill behavior, and the overhead introduced by each optimization.

## Setup

The benchmark harness uses a `BenchmarkConfig` class to hold the configuration for each run.

I also reduced the NanoGPT hyperparameters to keep iteration fast on an Intel CPU. The generated text does not need to be coherent for these tests, because the benchmark is measuring how long it takes to generate tokens.

For the rest of this series, I will ignore output quality and focus on the performance behavior of the inference optimizations.

## Baseline Generation vs KV Cached Generation

### What Was Benchmarked

For this set of experiments, I defined a no-cache baseline. That version repeatedly recalculates all key and value matrices from scratch at every decode step. The optimized version stores previous keys and values, then calculates only the new token's KV entries at each step.

The output reports:

| Metric | Meaning |
|---|---|
| `tokens` | Number of new tokens generated. |
| `wall_time_s` | Total elapsed time for the generation run. Lower is better. |
| `tokens_per_s` | Generated tokens divided by wall time. Higher is better. |
| `ttft_ms` | Time to first token in milliseconds. Lower is better. |
| `KV-cache throughput speedup` | `kv_cache tokens_per_s / no_cache tokens_per_s`. Higher is better. |

The most important metric here is tokens per second, because this benchmark is focused on decode throughput. TTFT is still useful, but in this particular harness it is affected by the difference between the no-cache first step and the cached prefill step, so it should be interpreted more carefully.

### Results

These are the main benchmark cases:

| Case | Prompt Len | Generated Tokens | No Cache Tok/s | KV Cache Tok/s | Speedup | Wall Time Reduction |
|---|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 16 | 126.67 | 185.93 | 1.47x | 31.8% |
| `medium_generation` | 16 | 32 | 102.20 | 265.23 | 2.60x | 61.5% |
| `longer_generation` | 16 | 48 | 143.56 | 276.58 | 1.93x | 48.1% |
| `heavier_prompt` | 32 | 32 | 119.99 | 274.34 | 2.29x | 56.3% |
| `near_context_limit` | 8 | 56 | 209.39 | 233.16 | 1.11x | 10.2% |


![KV cache benchmark results]({{ site.baseurl }}/images/kv_cache_benchmark.png)

### Interpretation

#### Overall Trend

The named runs show a strong throughput advantage for KV caching. The largest speedup appears in `medium_generation`, where cached generation reaches **265.23 tokens/sec** compared with **102.20 tokens/sec** for the no-cache baseline. That is a **2.60x throughput improvement** and cuts wall time by about **61.5%**.

The `heavier_prompt` result is also important. It uses a longer prompt of 32 tokens and generates 32 new tokens. The KV-cache path reaches **274.34 tokens/sec**, while the no-cache path reaches **119.99 tokens/sec**, giving a **2.29x speedup**. This matches the expected behavior: as the prompt gets longer, recomputing the full context becomes more wasteful, so caching previous keys and values becomes more valuable.

#### Generation Length Sweep

The generation length sweep holds the prompt length fixed at 8 tokens and varies the number of generated tokens:

| Generated Tokens | No Cache Tok/s | KV Cache Tok/s | Speedup | Wall Time Reduction |
|---:|---:|---:|---:|---:|
| 8 | 222.60 | 275.18 | 1.24x | 18.9% |
| 16 | 109.83 | 193.64 | 1.76x | 43.3% |
| 32 | 225.47 | 227.41 | 1.01x | 0.8% |
| 48 | 130.52 | 233.10 | 1.79x | 44.0% |
| 56 | 188.92 | 275.38 | 1.46x | 31.4% |

#### Sweep Trend

The sweep mostly confirms that KV caching improves decode throughput as generation length increases. The strongest sweep result is at **N=48**, where KV caching reaches **233.10 tokens/sec** compared with **130.52 tokens/sec** for no-cache generation, a **1.79x speedup**.

The **N=16** case is also strong, with a **1.76x speedup**. The **N=56** case remains meaningfully faster at **1.46x**.

The unusual row is **N=32**, where KV caching is only **1.01x faster**. That does not mean the KV cache stopped working. It is more likely a measurement artifact caused by the small CPU workload. The total wall times are only about **0.14 seconds** for both methods, so a small amount of scheduler noise, allocator behavior, Python overhead, or CPU frequency variation can move the result noticeably.

#### TTFT Interpretation

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

#### Row-by-Row Notes

##### `small_smoke_test`

This is a short sanity test with an 8-token prompt and 16 generated tokens. KV caching improves throughput from **126.67 tokens/sec** to **185.93 tokens/sec**, a **1.47x speedup**.

This confirms that the cached path is functional. Even on a tiny workload, avoiding repeated context processing saves enough time to show up.

##### `medium_generation`

This is the strongest result: a 16-token prompt and 32 generated tokens. KV caching improves throughput from **102.20 tokens/sec** to **265.23 tokens/sec**, a **2.60x speedup**.

This is the clearest evidence that the benchmark is capturing the intended optimization. The generation length is long enough for repeated no-cache recomputation to accumulate, while the cached path can reuse previous keys and values.

##### `longer_generation`

With a 16-token prompt and 48 generated tokens, KV caching gives a **1.93x speedup**. Wall time drops from **0.3344s** to **0.1735s**.

The speedup is lower than `medium_generation`, even though generation is longer. That is a reminder that this microbenchmark is noisy. Still, nearly doubling throughput is a strong result.

##### `heavier_prompt`

This case increases prompt length to 32 while generating 32 tokens. KV caching gives a **2.29x speedup**.

This result is especially aligned with theory. A longer prompt means the no-cache path has more previous context to recompute at every step. The cached path pays for that prompt once, then reuses the cached keys and values.

##### `near_context_limit`

This case generates 56 tokens with an 8-token prompt, filling the 64-token context. KV caching gives only a **1.11x speedup**.

This is lower than expected in a pure compute model, but understandable in this implementation. As generation approaches the context limit, the internal cache grows, and repeated `torch.cat` operations become more expensive. The no-cache path is also capped by `block_size=64`, so it never grows beyond that fixed context size. The result still favors KV caching, but the simple cache implementation leaves performance on the table.

### Caveats

#### Why The Speedup Is Not Perfectly Monotonic

In theory, KV-cache benefits should become more obvious as the generated sequence gets longer or as the prompt gets longer. The benchmark broadly shows that, but the numbers are not perfectly monotonic. There are several reasons.

#### 1. The Model Is Tiny

At **0.056769M parameters**, this model is small enough that framework overhead can be a large fraction of total runtime. In a larger transformer, attention and MLP compute dominate more of the timing, so avoiding repeated work tends to show up more cleanly.

#### 2. The Benchmark Runs On CPU

CPU timing is sensitive to operating system scheduling, cache locality, thread behavior, and frequency scaling. Since many runs complete in under half a second, small timing fluctuations can noticeably change tokens/sec.

#### 3. The KV Cache Uses Tensor Concatenation

The implementation stores `key_cache` and `value_cache` inside each attention head and appends with `torch.cat` on every decode step. This is simple and educational, but it is not how production inference engines usually manage KV memory.

Each append can allocate and copy tensors. As the cache grows, that overhead can partially offset the benefit of avoiding recomputation. Production systems usually preallocate KV buffers or use paged/block-based KV cache layouts to avoid repeated copying.

#### 4. Sequence Length Is Capped At 64

The benchmark uses `block_size=64`, so the no-cache baseline never processes more than 64 tokens per step. This limits how bad the no-cache path can get. With longer context windows, the cost of recomputing the full context would grow, making the KV-cache advantage more dramatic.

#### 5. The No-Cache Path Uses A Different Mode

The benchmark uses `model.train()` to force the no-cache path and `model.eval()` to enable the cache path. Dropout is set to `0.0`, so this should not introduce dropout randomness, but the model still takes different branches through the attention implementation. This is fine for demonstrating the optimization, but it is worth remembering when interpreting very small timing differences.

### Takeaway

1. **KV caching works in this benchmark.** Every named benchmark case shows cached generation outperforming no-cache generation.

2. **The largest gains appear when repeated context work matters most.** `medium_generation` and `heavier_prompt` both exceed **2x speedup**.

3. **The current cache implementation is educational, not production-optimized.** Appending to cache tensors with `torch.cat` is simple but can introduce copying overhead.

4. **TTFT is not the main story here.** The TTFT numbers are mixed and measured over very short CPU timings. Throughput is the more reliable signal for this benchmark.

5. **The results likely understate the importance of KV caching for real LLMs.** Larger models, longer contexts, and optimized GPU kernels typically make the cost of recomputation much more expensive than it appears in this tiny CPU setup.

The raw KV-cache benchmark log is included in the appendix.

## Single Request vs Continuous Batching

### What Was Benchmarked

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

### Results

| Case | Requests Configured | Max Batch | Seq Tok/s | Batched Tok/s | Speedup | Avg Lat Ratio | Avg TTFT Ratio | Batched Completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 4 | 271.52 | 524.11 | 1.93x | 1.37x | 1.94x | 5/8 |
| `more_requests_small_batch` | 16 | 4 | 266.90 | 476.62 | 1.79x | 1.25x | 1.74x | 5/16 |
| `more_requests_larger_batch` | 16 | 8 | 241.15 | 850.88 | 3.53x | 1.58x | 2.85x | 9/16 |
| `longer_generations` | 16 | 8 | 265.04 | 956.53 | 3.61x | 1.51x | 3.98x | 9/16 |
| `heavier_prompt` | 16 | 8 | 277.84 | 492.91 | 1.77x | 3.41x | 7.83x | 9/16 |
| `stress_batch_capacity` | 32 | 8 | 246.66 | 671.97 | 2.72x | 1.97x | 3.21x | 9/32 |

![Continuous batching benchmark results]({{ site.baseurl }}/images/contin_batching_benchmark.png)

### Interpretation

#### Row-by-Row Notes

##### `small_smoke_test`

This run uses 8 configured requests, an 8-token prompt, 16 generated tokens, and `max_batch_size=4`. Continuous batching reports **524.11 tokens/sec** compared with **271.52 tokens/sec** for sequential serving, a **1.93x speedup**.

The batched path completes 5 of the 8 configured requests. Within that subset, the benchmark shows the intended batching effect: average batch size rises to **2.50**, and throughput nearly doubles.

##### `more_requests_small_batch`

This run increases the configured workload to 16 requests while keeping `max_batch_size=4`. Throughput improves by **1.79x**, from **266.90 tokens/sec** to **476.62 tokens/sec**.

Because the batch cap is still 4, the completed batched subset remains 5 requests. The result shows that simply adding more queued requests does not help unless the scheduler can keep and admit them correctly.

##### `more_requests_larger_batch`

This run raises `max_batch_size` from 4 to 8. Throughput jumps to **850.88 tokens/sec**, a **3.53x speedup**.

This is the cleanest comparison against `more_requests_small_batch`: larger batches create more opportunity to amortize each forward pass. Average latency and TTFT also rise, showing the cost of waiting and sharing.

##### `longer_generations`

This run keeps `max_batch_size=8` but doubles generation length to 48 tokens. It produces the strongest named speedup: **3.61x**.

Longer generations make continuous batching more useful because there are more decode steps over which the active batch can remain full. Once prefill is done, the engine gets many chances to run efficient batched decode.

##### `heavier_prompt`

This run increases prompt length to 16 and generates 32 tokens. Throughput still improves by **1.77x**, but latency gets much worse.

Average TTFT rises from **5.17 ms** to **40.45 ms**. This is the most important latency warning in the file. The benchmark handles prefill one request at a time, so heavier prompts delay first-token emission and reduce the relative benefit of batched decode.

##### `stress_batch_capacity`

This run configures 32 requests with `max_batch_size=8`. Continuous batching reports a **2.72x throughput speedup**, but completes only 9 of 32 configured requests.

The result still shows that batched decode is faster than sequential decode, but this row should not be interpreted as "the batched engine served all 32 requests quickly." It served a subset quickly due to the admission-loop limitation.

### Caveats

The continuous batching benchmark is useful for understanding batched decode, but it is not a full end-to-end serving benchmark yet. The batched path under-completes some configured workloads, such as `9/16` or `9/32`, so the ratios should be read as evidence of the batching mechanism rather than final production throughput numbers.

### Takeaway

1. **Continuous batching increases throughput when the batch size is large enough.** The sweep shows a clear scaling pattern from **0.68x** at batch size 1 to **5.99x** at batch size 16.

2. **Batching has overhead.** With `max_batch_size=1` and `max_batch_size=2`, the continuous batching path is slower or roughly equal to sequential serving because it adds scheduling, KV stacking, and bookkeeping without enough parallel work to amortize that overhead.

3. **Latency and TTFT worsen in this scaffold.** This is expected: the engine is optimizing system throughput, not single-request latency. The effect is especially strong when prompt prefill is heavier.

4. **Prefill handling matters.** The `heavier_prompt` case shows that prefill can dominate first-token latency. This motivates chunked prefill and prefill/decode scheduling.

5. **The benchmark currently under-completes the workload in the continuous path.** The request-count mismatch must be fixed before treating the ratios as a rigorous end-to-end serving comparison.

6. **The result still demonstrates the core principle.** Batched decode can emit many more tokens per second than serving one request at a time.


## No Chunked Prefill vs Chunked Prefill

### What Was Benchmarked

This benchmark compares two prefill scheduling policies:

- **Normal prefill:** when a request arrives, process its entire prompt in one forward pass.
- **Chunked prefill:** split prompt processing into smaller chunks and interleave those chunks with decode work under a token budget.

Chunked prefill splits large prompt processing into smaller pieces so that long-prefill requests do not block decoding and other requests. In production serving systems, this can improve scheduling fairness, streaming smoothness, and GPU utilization.

### Results

In these results, **normal prefill is faster in every configuration**. Chunked prefill achieves only **0.74x to 0.91x** of normal-prefill generated-token throughput, averaging about **0.86x**. It also has much worse average TTFT, ranging from **4.30x to 13.09x** slower than normal prefill.

That may look surprising because chunked prefill is usually introduced as an optimization. The key is that chunked prefill is not primarily about making a tiny CPU microbenchmark faster. It is a serving policy for managing interference between long prompt prefill and active decode streams, especially on larger GPU-backed systems. In this small NanoGPT benchmark, chunking creates extra forward passes and scheduler overhead without getting the production benefits of GPU utilization, batched decode, or many concurrent requests.

The important result is not "chunked prefill is bad." The better reading is:

> In this implementation and workload, chunked prefill trades throughput and first-token latency for slightly smoother decode gaps in some pressure cases.

![Chunked prefill benchmark results]({{ site.baseurl }}/images/chunked_prefill_benchmark.png)

### Interpretation

#### Why Chunked Prefill Is Slower Here

##### 1. Chunking Creates More Forward Passes

Normal prefill processes a prompt in one model call:

```text
long prompt: [32 tokens] -> one prefill forward
```

Chunked prefill splits that same prompt:

```text
long prompt: [8 tokens] -> forward
long prompt: [8 tokens] -> forward
long prompt: [8 tokens] -> forward
long prompt: [8 tokens] -> forward
```

That is easier to schedule around active decode, but it adds repeated Python, PyTorch, position setup, cache update, and sampling overhead. On a tiny CPU model, that overhead is very visible.

##### 2. The Benchmark Does Not Batch Decode

The benchmark intentionally decodes active requests one at a time. That is useful for isolating prefill policy, but it removes one of the main reasons chunked prefill is useful in production.

In a real serving engine, chunked prefill is often paired with continuous batching. Decode tokens from many requests can be batched together, and leftover token budget can be used for prompt chunks. This benchmark does not get that throughput benefit.

##### 3. Decode-First Scheduling Delays Long Requests' First Token

The chunked policy decodes active requests first, then spends remaining budget on prefill chunks. That protects existing streams, but new long-prompt requests may need several scheduler steps before their full prompt is processed and their first token can be emitted.

That is why TTFT gets much worse:

| Case | Normal Avg TTFT | Chunked Avg TTFT | Increase |
|---|---:|---:|---:|
| `small_smoke_test` | 6.13 ms | 34.40 ms | +28.27 ms |
| `more_long_prompts` | 8.48 ms | 82.81 ms | +74.33 ms |
| `smaller_chunks_smoother_decode` | 7.71 ms | 100.92 ms | +93.21 ms |
| `larger_chunks_less_overhead` | 9.84 ms | 42.31 ms | +32.47 ms |
| `decode_heavy_pressure` | 11.27 ms | 119.16 ms | +107.89 ms |
| `late_long_prompt_interruptions` | 11.92 ms | 129.57 ms | +117.65 ms |

This is a real policy tradeoff: protecting existing decode streams can make newly arrived long prompts wait longer for their first token.

##### 4. The Model And Context Are Tiny

With only **0.056769M parameters** and `block_size=64`, full-prompt prefill is not expensive enough to create massive stalls. Splitting the prompt into chunks can cost more than it saves.

With larger models, longer prompts, GPU execution, and many concurrent decode streams, the tradeoff can look different.

#### Streaming Smoothness: The More Interesting Signal

Chunked prefill is meant to reduce decode interruptions. The best metric for that in this file is `max_gap_ms`, the worst inter-token gap.

| Case | Normal Max Gap | Chunked Max Gap | Ratio | Interpretation |
|---|---:|---:|---:|---|
| `small_smoke_test` | 16.10 ms | 20.09 ms | 1.25x | Chunked worse |
| `more_long_prompts` | 24.46 ms | 29.98 ms | 1.23x | Chunked worse |
| `smaller_chunks_smoother_decode` | 24.02 ms | 23.35 ms | 0.97x | Slightly smoother |
| `larger_chunks_less_overhead` | 25.93 ms | 30.05 ms | 1.16x | Chunked worse |
| `decode_heavy_pressure` | 35.75 ms | 32.25 ms | 0.90x | Smoother |
| `late_long_prompt_interruptions` | 33.94 ms | 31.78 ms | 0.94x | Smoother |

This is where chunked prefill starts to show its intended purpose. In the decode-heavy pressure cases, chunking reduces the worst inter-token gap:

- `decode_heavy_pressure`: max gap improves from **35.75 ms** to **32.25 ms**.
- `late_long_prompt_interruptions`: max gap improves from **33.94 ms** to **31.78 ms**.
- `smaller_chunks_smoother_decode`: max gap improves slightly from **24.02 ms** to **23.35 ms**.

These improvements are modest, but they point in the expected direction: smaller chunks can prevent long prefill work from blocking decode for too long.

The average gap does not improve, though. Chunked prefill has slightly higher average gaps in every row. So in this benchmark, chunking mainly helps the worst stall in some pressure cases, not the overall streaming cadence.

#### Chunk Size And Token Budget

The clearest chunk-size comparison is between these two rows:

| Case | Chunk Size | Token Budget | Chunked Gen Tok/s | Chunked Avg TTFT | Chunked Max Gap |
|---|---:|---:|---:|---:|---:|
| `smaller_chunks_smoother_decode` | 4 | 16 | 310.58 | 100.92 ms | 23.35 ms |
| `larger_chunks_less_overhead` | 16 | 32 | 359.99 | 42.31 ms | 30.05 ms |

Smaller chunks produce the smoothest worst-case decode gap, but they are expensive:

- More chunks means more forward calls.
- More forward calls means more overhead.
- More chunks also mean long prompts wait through more steps before first token.

Larger chunks reduce overhead and improve TTFT, but they can create larger decode stalls because each chunk occupies a bigger piece of a scheduler step.

This is the core chunked-prefill tuning problem:

> Smaller chunks protect streaming smoothness. Larger chunks protect throughput and TTFT.

The right value depends on the serving goal. A chat system may prefer smoother streaming. A batch/offline system may prefer throughput.

#### Row-by-Row Notes

##### `small_smoke_test`

This is a light workload: 4 short requests and 2 long requests. Chunked prefill reaches **384.01 generated tokens/sec** compared with **423.55** for normal prefill, a **0.91x ratio**.

This is the least alarming slowdown because the workload is small and the chunk size is moderate. But TTFT is already much worse: **34.40 ms** vs **6.13 ms**.

##### `more_long_prompts`

This increases long requests from 2 to 4. Chunked throughput falls to **0.74x** of normal prefill, the worst ratio in the file.

The extra long prompts create more chunked-prefill work, and each long prompt takes several chunks before it can emit its first token. That pushes average TTFT to **82.81 ms**, almost **10x** normal prefill.

##### `smaller_chunks_smoother_decode`

This keeps the same workload as `more_long_prompts` but reduces chunk size from 8 to 4. The smaller chunks slightly improve max inter-token gap: **23.35 ms** vs **24.02 ms** for normal prefill.

But the cost is high. Average TTFT rises to **100.92 ms**, the throughput ratio remains only **0.79x**, and total wall time increases by about **27.2%**.

This row captures the smoothness-throughput tradeoff most clearly.

##### `larger_chunks_less_overhead`

This increases chunk size to 16 and token budget to 32. Chunked prefill becomes much more competitive, reaching **0.91x** of normal throughput.

Average TTFT is still worse than normal prefill, but much better than the small-chunk case: **42.31 ms** instead of **100.92 ms**. The tradeoff is that max inter-token gap becomes worse than normal prefill: **30.05 ms** vs **25.93 ms**.

##### `decode_heavy_pressure`

This workload has 8 short decode-heavy requests and 4 long prompt-heavy requests. Chunked throughput is **0.90x** of normal throughput, but max inter-token gap improves from **35.75 ms** to **32.25 ms**.

This is one of the rows where chunked prefill behaves according to its intended purpose: it reduces the worst decode stall while active decode-heavy streams are running. It still pays for that with worse TTFT and lower total throughput.

##### `late_long_prompt_interruptions`

This is similar to `decode_heavy_pressure`, but long prompts arrive later and are longer at 40 tokens. Chunked throughput is **0.89x** of normal throughput, and max gap improves from **33.94 ms** to **31.78 ms**.

The TTFT penalty is the largest in the file: chunked prefill averages **129.57 ms** vs **11.92 ms** for normal prefill. Late long prompts wait behind ongoing decode-first work and require multiple chunks before they emit a first token.

### Caveats

This benchmark isolates prefill scheduling, so it does not include the full production setting where chunked prefill is usually useful. There is no batched decode, no GPU utilization pressure, no large model, and no very long context. That makes the overhead of smaller chunks much more visible than it would be in a larger serving system.

### Takeaway

1. **Normal prefill wins on raw throughput in this benchmark.** Chunked prefill is consistently slower, with an average generated-token throughput ratio of about **0.86x**.

2. **Chunked prefill heavily penalizes TTFT here.** Average TTFT is roughly **4.30x to 13.09x** worse because long prompts must complete multiple chunks before emitting their first token.

3. **Chunking can improve worst-case decode stalls in pressure cases.** The max inter-token gap improves in `smaller_chunks_smoother_decode`, `decode_heavy_pressure`, and `late_long_prompt_interruptions`.

4. **Smaller chunks are smoother but more expensive.** Chunk size 4 gives the best max-gap behavior, while chunk size 16 gives better throughput and TTFT.

5. **This benchmark does not include the full production setting where chunked prefill shines.** There is no batched decode, no GPU utilization pressure, no large model, and no very long context.

6. **The current result is still useful.** It demonstrates the real scheduling tradeoff: prefill work can either be processed efficiently in large blocks or spread out to protect streaming decode.

## Appendix: Raw KV Cache Benchmark Log

Here is a sample of the raw benchmark log that I generated when doing the KV cache benchmark test. You can find the other logs in my repo link here: 

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

You can find the rest of the code here:

CZ
