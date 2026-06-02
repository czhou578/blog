---
layout: post
title: "Adding Interleaving to NanoGPT"
date: 2026-05-29
---

In the last post, we benchmarked and intensively analyzed the performance of several inference techniques: KV Caching, continuous batching, and chunked prefill.

This time, we will focus our attention on scheduling, prefix caching, and interleaving prefill.


## Scheduling 

### Preface

As a primer, scheduling in nanoGPT was built to solve the problem of FCFS (First Come First Serve) behavior. 

Before we added scheduling, our NanoGPT model processed requests in the order that they arrive. Prefilling requests block later arrivals from being admitted until they're done.

In a real server, if a very long low priority batch job arrives first and hogs the token budget for many steps, a short high priority request would be forced to wait possibly a long time before processing.

Having a scheduler can preempt the low-priority job, serve the high-priority one immediately, and resume the evicted request when resources free up. This is exactly how vLLM's scheduler manages competing requests under memory pressure and what our implementation attempts to achieve.

For reference, the entire source code is here: 

he benchmark compares two scheduling policies:

- **FCFS**, which serves requests in first-come, first-served order.
- **Priority scheduling**, which admits lower priority-number requests first and can preempt lower-priority active requests when a higher-priority request is blocked.

The headline result is:

> Priority scheduling greatly improves high-priority request latency, while total throughput stays roughly similar or slightly worse when preemption causes recomputation.

This is exactly the tradeoff a scheduler is supposed to expose. Scheduling is not mainly a model-speed optimization. It is a policy layer that decides **who gets served first** when token budget, batch size, and KV-cache capacity are limited.

Across the benchmark:

- Priority improves high-priority latency in the meaningful priority workloads.
- Priority has little to no effect in the equal-priority control case.
- Priority can slightly reduce throughput when it preempts active work and has to recompute it later.
- All benchmark cases complete the same number of requests and generated tokens under both policies, so the comparisons are clean.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CUDA
- **Context length:** `block_size=64`
- **Benchmark target:** scheduling behavior, not model quality

The training log before the benchmark shows the model learning:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8684 |
| 119 | 2.7762 | 2.7998 |

The generated sample is still noisy, which is expected for a tiny model trained briefly. The important part is that the same model and same workloads are used for both scheduling policies.

## What Was Measured

The benchmark runs four scheduling scenarios:

| Case | Purpose |
|---|---|
| `priority_inversion_serial` | A high-priority short request arrives behind lower-priority work, with `max_batch_size=1`. This makes priority inversion easy to see. |
| `priority_mix_small_batch` | Several requests with mixed priorities run under a small batched decode setup. |
| `memory_pressure_preemption` | A tight KV-token budget forces priority scheduling to preempt lower-priority requests. |
| `equal_priority_control` | All requests have equal priority. FCFS and priority should behave the same. |

Both policies use cached generation, chunked prefill, and batched decode where the workload allows it. The key difference is admission order and whether priority is allowed to preempt lower-priority active requests.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Total requests in the workload. |
| `done` | Requests completed. This matches `reqs` in every row. |
| `gen_tok` | Generated tokens completed by the policy. |
| `wall_s` | Total wall-clock time for the run. |
| `tok/s` | Generated tokens per second. |
| `avg_ttft_ms` | Average time to first token. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency from arrival to completion. |
| `p95_lat_ms` | 95th percentile request latency. |
| `hi_lat_ms` | Average latency for high-priority requests. |
| `low_lat_ms` | Average latency for lower-priority requests. |
| `preempt` | Number of recompute preemptions. |
| `avg_batch` | Average decode batch size. |
| `max_batch` | Largest decode batch size. |
| `forward_s` | Time spent in measured model forward work. |

The most important metrics are `hi_lat_ms`, `avg_lat_ms`, `tok/s`, and `preempt`.

## Results Summary

| Case | FCFS Tok/s | Priority Tok/s | Throughput Ratio | FCFS High-Priority Latency | Priority High-Priority Latency | High-Priority Latency Ratio | Preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|
| `priority_inversion_serial` | 83.04 | 79.77 | 0.96x | 308.77 ms | 112.88 ms | 0.37x | 1 |
| `priority_mix_small_batch` | 166.15 | 169.10 | 1.02x | 235.69 ms | 108.55 ms | 0.46x | 0 |
| `memory_pressure_preemption` | 127.67 | 119.76 | 0.94x | 183.62 ms | 120.07 ms | 0.65x | 2 |
| `equal_priority_control` | 165.27 | 165.35 | 1.00x | 210.86 ms | 210.66 ms | 1.00x | 0 |

The strongest and most useful trend is the high-priority latency improvement:

- In `priority_inversion_serial`, high-priority latency drops from **308.77 ms** to **112.88 ms**.
- In `priority_mix_small_batch`, high-priority latency drops from **235.69 ms** to **108.55 ms**.
- In `memory_pressure_preemption`, high-priority latency drops from **183.62 ms** to **120.07 ms**.
- In the equal-priority control, latency is unchanged, which validates that priority ordering is not changing behavior when there is no priority difference.

## Why Priority Scheduling Helps

FCFS is fair in arrival order, but not necessarily fair in user impact. If a long, low-priority request arrives first, a short high-priority request may wait behind it even though serving the high-priority request first would improve responsiveness.

Priority scheduling changes the admission rule:

```text
FCFS:
    serve earlier request first

Priority:
    serve higher-priority request first
    use arrival order only as a tie-breaker
```

When resources are constrained, priority scheduling can also preempt lower-priority active requests. In this benchmark, preemption is recompute-based: the evicted request loses its KV cache and must prefill again later. That improves high-priority latency, but it can hurt throughput and low-priority latency.

## Case-By-Case Interpretation

### `priority_inversion_serial`

Configuration:

```text
max_batch_size=1
token_budget=12
prefill_chunk_size=10
max_kv_tokens=40
```

This is the cleanest priority-inversion test. With `max_batch_size=1`, only one request can be active at a time. FCFS lets the early lower-priority request occupy the engine. Priority scheduling performs **1 preemption**, allowing the high-priority requests to move ahead.

Key results:

| Metric | FCFS | Priority | Interpretation |
|---|---:|---:|---|
| Throughput | 83.04 tok/s | 79.77 tok/s | Priority is slightly slower. |
| Avg TTFT | 199.97 ms | 130.38 ms | Priority improves first-token responsiveness. |
| Avg latency | 323.95 ms | 253.90 ms | Overall average latency improves. |
| High-priority latency | 308.77 ms | 112.88 ms | Major win for important requests. |
| Low-priority latency | 339.14 ms | 394.92 ms | Lower-priority work waits longer. |
| Preemptions | 0 | 1 | Priority pays recompute cost. |

This row captures the main scheduling tradeoff beautifully. Priority scheduling reduces high-priority latency to about **37%** of the FCFS value, but low-priority latency gets worse and throughput drops slightly to **0.96x**.

### `priority_mix_small_batch`

Configuration:

```text
max_batch_size=4
token_budget=16
prefill_chunk_size=8
max_kv_tokens=64
```

This workload has 12 requests with a mix of high-priority and low-priority work. Because batch size and KV budget are less restrictive, priority scheduling does not need to preempt.

Key results:

| Metric | FCFS | Priority | Interpretation |
|---|---:|---:|---|
| Throughput | 166.15 tok/s | 169.10 tok/s | Essentially unchanged, slightly better for priority. |
| Avg TTFT | 178.68 ms | 162.93 ms | Priority improves first-token delay. |
| Avg latency | 344.37 ms | 330.97 ms | Priority slightly improves average latency. |
| High-priority latency | 235.69 ms | 108.55 ms | High-priority requests complete much faster. |
| Low-priority latency | 380.60 ms | 405.12 ms | Low-priority requests pay the cost. |
| Preemptions | 0 | 0 | Admission order alone creates the benefit. |

This is the most attractive result for priority scheduling. It cuts high-priority latency by more than half, with no preemption and no meaningful throughput penalty.

### `memory_pressure_preemption`

Configuration:

```text
max_batch_size=3
token_budget=12
prefill_chunk_size=8
max_kv_tokens=32
```

This workload intentionally creates KV-cache pressure. Priority scheduling performs **2 preemptions**, which lets high-priority work move forward but forces lower-priority requests to recompute.

Key results:

| Metric | FCFS | Priority | Interpretation |
|---|---:|---:|---|
| Throughput | 127.67 tok/s | 119.76 tok/s | Priority is slower due to recompute. |
| Avg TTFT | 75.94 ms | 90.15 ms | Average TTFT gets worse. |
| Avg latency | 225.58 ms | 249.03 ms | Average latency gets worse. |
| High-priority latency | 183.62 ms | 120.07 ms | High-priority latency still improves. |
| Low-priority latency | 253.56 ms | 335.00 ms | Low-priority latency worsens substantially. |
| Preemptions | 0 | 2 | Recompute cost is visible. |

This is the sharpest demonstration of the cost side of priority scheduling. High-priority latency improves to about **65%** of the FCFS value, but total throughput falls to **0.94x** and average latency gets worse.

That is not a bug. It is the expected cost of recompute preemption: the scheduler discards lower-priority KV cache state to make room, then pays to rebuild it later.

### `equal_priority_control`

Configuration:

```text
max_batch_size=4
token_budget=16
prefill_chunk_size=8
max_kv_tokens=64
```

This is the control case. All requests have equal priority, so priority scheduling should reduce to FCFS behavior.

Key results:

| Metric | FCFS | Priority |
|---|---:|---:|
| Throughput | 165.27 tok/s | 165.35 tok/s |
| Avg TTFT | 60.42 ms | 60.22 ms |
| Avg latency | 210.86 ms | 210.66 ms |
| High-priority latency | 210.86 ms | 210.66 ms |
| Preemptions | 0 | 0 |

The results are effectively identical. This is important because it validates the benchmark. Priority scheduling changes behavior only when priority information matters.

## Main Trends

### 1. Priority Scheduling Improves High-Priority Latency

This is the main success criterion. In every workload with meaningful priority differences, high-priority latency improves:

| Case | High-Priority Latency Improvement |
|---|---:|
| `priority_inversion_serial` | 63.4% lower |
| `priority_mix_small_batch` | 53.9% lower |
| `memory_pressure_preemption` | 34.6% lower |

For serving systems, this is often exactly what you want. Interactive requests, paid-tier users, short chat completions, or latency-sensitive requests can avoid getting stuck behind slow background work.

### 2. Throughput Is Not The Main Win

Priority throughput is close to FCFS:

| Case | Priority Throughput Ratio |
|---|---:|
| `priority_inversion_serial` | 0.96x |
| `priority_mix_small_batch` | 1.02x |
| `memory_pressure_preemption` | 0.94x |
| `equal_priority_control` | 1.00x |

The scheduler does not make the model inherently faster. It changes request ordering. Throughput only changes indirectly through batching shape, memory pressure, and recomputation.

### 3. Low-Priority Requests Pay The Cost

Priority scheduling improves important requests by making less important requests wait:

| Case | FCFS Low-Priority Latency | Priority Low-Priority Latency |
|---|---:|---:|
| `priority_inversion_serial` | 339.14 ms | 394.92 ms |
| `priority_mix_small_batch` | 380.60 ms | 405.12 ms |
| `memory_pressure_preemption` | 253.56 ms | 335.00 ms |

This is expected. A scheduler is a policy tool. It does not remove work; it decides which work absorbs waiting time.

### 4. Preemption Is Useful But Expensive

Preemption appears in two priority runs:

| Case | Preemptions | Effect |
|---|---:|---|
| `priority_inversion_serial` | 1 | Big high-priority latency win, slight throughput loss. |
| `memory_pressure_preemption` | 2 | High-priority latency win, average latency and throughput worsen. |

In this benchmark, preemption uses a recompute strategy. When a request is preempted:

1. Its KV cache is discarded.
2. Its generated state is reset.
3. It re-enters the waiting queue.
4. It must prefill again later.

That is simple and memory-efficient, but it wastes compute. Production systems often add more sophisticated options, such as swapping KV cache blocks to CPU memory or using paged KV memory to reduce the need for preemption.

### 5. The Control Case Confirms The Policy Logic

The equal-priority run has nearly identical FCFS and priority results. That is exactly what should happen. If every request has the same priority, the priority key falls back to arrival order, so the behavior becomes FCFS.

## Significance For LLM Serving

Scheduling matters because LLM serving is resource-constrained:

- KV cache memory is finite.
- Decode is sequential per request.
- Batch slots are limited.
- Prefill and decode compete for token budget.
- Not all requests have equal importance.

Without scheduling, a server can accidentally let long, low-value work delay short, high-value work. Priority scheduling fixes that by explicitly encoding service policy.

In real systems, this can support:

- Interactive requests over background batch jobs.
- Paid-tier traffic over free-tier traffic.
- Short latency-sensitive requests over long offline generations.
- System or moderation requests over normal traffic.
- Deadline-aware or SLA-aware request handling.

The benchmark shows that this does not come for free. Better high-priority latency can mean worse low-priority latency, extra recomputation, and occasionally lower throughput. That is the central scheduler tradeoff.

## Prefix Caching

### Preface

**Prefix caching** stores completed KV blocks in a content-addressed cache. When Request B arrives and its prompt starts with the same tokens as Request A, the scheduler finds the cached KV blocks, skips the prefill for those tokens, and only computes the **suffix** (e.g. `"Goodbye"`). This directly reduces TTFT or time to first token.

## Executive Summary

The benchmark shows the core tradeoff behind continuous batching:

- **Throughput improves substantially** when multiple active requests share decode forward passes.
- **Average latency and TTFT get worse** in this simple harness because requests wait behind batch formation, batched decode steps, and individually handled prefill.
- **Larger batch sizes produce larger throughput gains**, but also increase user-facing waiting time.

Across the named benchmark cases, continuous batching reports throughput speedups from **1.77x to 3.61x**, averaging about **2.56x**. In the batch-size sweep, speedup grows from **0.68x** at `max_batch_size=1` to **5.99x** at `max_batch_size=16`.

The most important caveat is that the current continuous batching scaffold does **not complete the same number of requests as the sequential baseline**. For example, in `stress_batch_capacity`, the sequential path completes **32 requests**, while the continuous batching path reports only **9 completed requests**. This means the throughput ratios are best read as "tokens/sec while the batched engine is active" rather than as a fully fair end-to-end comparison over the same workload.

That caveat does not invalidate the trend. It does mean the benchmark should be treated as an educational microbenchmark for batched decode, not yet as a production-quality serving benchmark.

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

## Important Harness Caveat: Request Counts Differ

The `reqs` column reveals a major limitation:

| Case | Requested Workload | Sequential Completed | Continuous Completed |
|---|---:|---:|---:|
| `small_smoke_test` | 8 | 8 | 5 |
| `more_requests_small_batch` | 16 | 16 | 5 |
| `more_requests_larger_batch` | 16 | 16 | 9 |
| `longer_generations` | 16 | 16 | 9 |
| `heavier_prompt` | 16 | 16 | 9 |
| `stress_batch_capacity` | 32 | 32 | 9 |

This happens because the continuous batching implementation admits all requests that have arrived for the current scheduler step into a temporary `newly_arrived` list. If the active batch is full, it requeues only the current overflow request and breaks out of the loop. Later requests already removed from `pending` are not requeued.

The practical result is that the continuous batching path often processes only `max_batch_size + 1` requests:

- `max_batch_size=4` tends to complete 5 requests.
- `max_batch_size=8` tends to complete 9 requests.
- `max_batch_size=16` tends to complete 17 requests.

Because of this, the `wall_s` values are not directly comparable as "time to finish the same workload." The safer comparison is the reported **tokens/sec for completed tokens**, while remembering that the batched path is operating on a smaller subset of the intended workload.

Before using this benchmark as a rigorous end-to-end serving comparison, the admission loop should preserve all overflow arrivals.

## Metrics

The output reports:

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests completed by that method. |
| `tokens` | Total generated tokens emitted by completed work. |
| `wall_s` | Total measured wall-clock time for the run. |
| `tok/s` | Generated tokens divided by wall time. Higher is better. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. Lower is better. |
| `avg_lat_ms` | Average request latency from arrival to completion. Lower is better. |
| `p95_lat_ms` | 95th percentile request latency. Lower is better. |
| `avg_batch` | Average recorded non-empty active batch size. |
| `max_batch` | Largest recorded active batch size. |
| `forward_s` | Total time spent inside measured model forward work. |

The main engine-facing metric is **tokens/sec**. The main user-facing metrics are **TTFT** and **request latency**.

One extra detail: in this implementation, `avg_batch` is computed from the `StepMetrics.batch_size` field, which is recorded after each scheduler step has updated active requests. It is a useful signal, but it is not a perfect trace of the exact batch size passed into every forward call.

## Training Context

Before the benchmark runs, the file shows a short training log:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8319 | 2.8681 |
| 119 | 2.7760 | 2.7998 |

The model is learning: both training and validation loss decrease steadily. As with the KV-cache benchmark, generation quality is not the point here. The model is tiny and trained briefly so the benchmark can focus on inference mechanics.

The run uses:

- **0.056769M parameters**
- **CPU**
- **`block_size=64`**
- A very small NanoGPT-style model

These conditions make the benchmark fast and educational, but also noisy. Python overhead, scheduler bookkeeping, cache stacking/unstacking, and CPU timing variation are all large relative to the model's actual compute.

## Named Benchmark Results

| Case | Requests Configured | Max Batch | Seq Tok/s | Batched Tok/s | Speedup | Avg Lat Ratio | Avg TTFT Ratio | Batched Completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 4 | 271.52 | 524.11 | 1.93x | 1.37x | 1.94x | 5/8 |
| `more_requests_small_batch` | 16 | 4 | 266.90 | 476.62 | 1.79x | 1.25x | 1.74x | 5/16 |
| `more_requests_larger_batch` | 16 | 8 | 241.15 | 850.88 | 3.53x | 1.58x | 2.85x | 9/16 |
| `longer_generations` | 16 | 8 | 265.04 | 956.53 | 3.61x | 1.51x | 3.98x | 9/16 |
| `heavier_prompt` | 16 | 8 | 277.84 | 492.91 | 1.77x | 3.41x | 7.83x | 9/16 |
| `stress_batch_capacity` | 32 | 8 | 246.66 | 671.97 | 2.72x | 1.97x | 3.21x | 9/32 |

### Overall Trend

Continuous batching improves generated-token throughput in every named case. The most favorable named case is `longer_generations`, where throughput increases from **265.04 tokens/sec** to **956.53 tokens/sec**, a **3.61x speedup**.

The second strongest named case is `more_requests_larger_batch`, where increasing the batch cap to 8 raises throughput to **850.88 tokens/sec**, a **3.53x speedup** over sequential serving.

The pattern is exactly what we expect: when the engine can decode several requests in the same forward pass, it amortizes model overhead across more emitted tokens.

## Why Continuous Batching Helps

Autoregressive decode produces one token per request per step. Without batching, a serving loop might do:

```text
request A: decode token 1
request A: decode token 2
request A: decode token 3
...
request B: decode token 1
request B: decode token 2
...
```

Continuous batching instead tries to do:

```text
step 1: decode one token for A, B, C, D together
step 2: decode one token for A, B, C, D together
step 3: decode one token for A, B, C, D together
...
```

Each request still gets only one new token per decode step, but the model forward pass is shared across requests. This is especially important for GPUs, where larger batches can improve utilization. Even on CPU in this tiny benchmark, the effect is visible.

## Latency Tradeoff

The throughput gains come with worse latency metrics:

| Case | Sequential Avg Latency | Batched Avg Latency | Sequential Avg TTFT | Batched Avg TTFT |
|---|---:|---:|---:|---:|
| `small_smoke_test` | 58.93 ms | 80.78 ms | 5.01 ms | 9.70 ms |
| `more_requests_small_batch` | 89.92 ms | 112.57 ms | 5.27 ms | 9.15 ms |
| `more_requests_larger_batch` | 99.52 ms | 157.11 ms | 5.57 ms | 15.84 ms |
| `longer_generations` | 181.10 ms | 272.66 ms | 4.32 ms | 17.21 ms |
| `heavier_prompt` | 115.17 ms | 392.75 ms | 5.17 ms | 40.45 ms |
| `stress_batch_capacity` | 129.73 ms | 255.02 ms | 5.69 ms | 18.29 ms |

This is the central serving tradeoff:

- Sequential serving gives each request exclusive attention, so a single request can finish quickly.
- Continuous batching improves system throughput by making requests share model steps.
- Sharing model steps can increase per-request wait time, especially for TTFT.

The `heavier_prompt` case makes this especially clear. Throughput still improves by **1.77x**, but average latency worsens by **3.41x** and average TTFT worsens by **7.83x**. The reason is that prefill is handled individually in this scaffold. Longer prompts increase the amount of prefill work that must happen before requests can join the decode batch, so first-token latency suffers.

## Batch Size Sweep

The batch-size sweep uses 32 configured requests, prompt length 8, and 32 generated tokens per request. It varies only `max_batch_size`.

| Max Batch Size | Seq Tok/s | Batched Tok/s | Speedup | Avg Batch | Avg Lat Ratio | Avg TTFT Ratio | Batched Completion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 221.53 | 149.76 | 0.68x | 1.00 | 1.48x | 1.86x | 2/32 |
| 2 | 255.75 | 246.86 | 0.97x | 1.50 | 1.53x | 1.40x | 3/32 |
| 4 | 273.32 | 607.05 | 2.22x | 2.50 | 1.28x | 1.74x | 5/32 |
| 8 | 235.78 | 651.10 | 2.76x | 4.50 | 1.77x | 3.76x | 9/32 |
| 16 | 245.28 | 1469.02 | 5.99x | 8.50 | 1.96x | 5.25x | 17/32 |

### Sweep Trend

The sweep shows a clear throughput scaling curve:

- At `max_batch_size=1`, continuous batching is slower than the sequential baseline: **0.68x**.
- At `max_batch_size=2`, it is roughly break-even: **0.97x**.
- At `max_batch_size=4`, it becomes meaningfully faster: **2.22x**.
- At `max_batch_size=8`, it reaches **2.76x**.
- At `max_batch_size=16`, it reaches **5.99x**.

This is the strongest evidence that the batched decode path is doing real work. With batch size 1, the continuous batching loop adds scheduler and cache-management overhead without getting much batching benefit. As the allowed batch size grows, more requests share each model forward pass, so throughput rises sharply.

The latency trend moves in the opposite direction. At `max_batch_size=16`, throughput is excellent, but average TTFT is **5.25x** worse than sequential serving. This is the classic throughput-latency tension in inference serving.

## Significance For LLM Inference

Continuous batching is one of the most important techniques in LLM serving because decode is inherently step-by-step. Each request can only produce the next token after the previous token is known. That means a server with many concurrent users needs a way to keep hardware busy while respecting each request's sequential dependency.

Continuous batching solves this by batching across requests rather than across time:

- Each request contributes one token position to the current decode step.
- Finished requests leave the batch.
- Newly ready requests can join future steps.
- The engine maintains high utilization even when individual requests start and finish at different times.

In production systems, this is the basis for serving many streaming chat completions concurrently. It also connects directly to:

- **KV caching**, because each request needs its own cached history.
- **Paged attention**, because many concurrent KV caches need efficient memory management.
- **Scheduling**, because the engine must decide which requests enter each step.
- **Chunked prefill**, because long prompts can otherwise delay decode work.
- **Prefix caching**, because shared prompt prefixes can reduce prefill cost.

This benchmark demonstrates the basic throughput mechanism, but it does not yet model all the surrounding serving-engine complexity.

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

## Chunked Prefill

### Preface

**Decode-prefill interleaving** merges both types of work into one `model()` call per step. The token budget constrains total tokens per step, decode requests get first priority (they're cheap and already own KV memory), and the remaining budget goes to prefill chunks.

## Executive Summary

This benchmark compares two prompt-processing paths:

- **No prefix cache:** every request prefills its full prompt from scratch.
- **Prefix cache:** block-aligned prompt prefixes can be reused from earlier requests, so later requests only prefill the uncached suffix.

The benchmark shows that prefix caching is **mechanically working**: in shared-prefix workloads it avoids a large amount of prompt prefill work.

The strongest examples:

- `high_reuse_many_requests` reduces actual prefill from **672 tokens** to **120 tokens**, an **82.1% prefill-token reduction**.
- `shared_prefix_basic` reduces actual prefill from **192 tokens** to **80 tokens**, a **58.3% reduction**.
- `multi_prefix_groups` reduces actual prefill from **576 tokens** to **256 tokens**, a **55.6% reduction**.

However, wall-clock throughput does **not** improve in this tiny CUDA benchmark. Prefix caching reports generated-token throughput ratios between **0.85x and 0.95x**, meaning it is slower end-to-end in every row.

That is not as contradictory as it first looks. The benchmark successfully demonstrates cache reuse, but the model and prompts are small enough that the overhead of cache lookup, KV cloning, slicing, concatenation, insertion, and eviction dominates the compute saved by skipping prefill tokens.

The right reading is:

> Prefix caching clearly reduces repeated prefill work, but this small educational implementation does not yet convert that saved work into faster wall-clock throughput.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CUDA
- **Context length:** `block_size=64`
- **Prefix block size:** `4` tokens
- **Benchmark target:** prefix-cache reuse behavior, not model quality

The training log before the benchmark shows the tiny model learning:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8684 |
| 119 | 2.7762 | 2.7998 |

The generated text is noisy, which is expected. The benchmark is about inference mechanics: whether shared prompt prefixes reduce repeated prefill work.

## What Was Benchmarked

The benchmark uses two policies:

| Method | Behavior |
|---|---|
| `no_prefix_cache` | Prefills the full prompt for every request. |
| `prefix_cache` | Checks whether block-aligned prefix blocks already exist in a global cache, loads matching KV blocks, and prefills only the remaining suffix. |

Every shared-prefix workload includes a unique suffix. This matters because the benchmark avoids the special case where an entire prompt is cached and no suffix forward pass is available to produce first-token logits.

The prefix cache is block-based:

```text
prompt = [cached prefix blocks] + [unique suffix]
```

For example, with `prefix_block_size=4` and a 16-token shared prefix:

```text
shared prefix = 4 cached blocks
unique suffix = freshly prefilled
```

The cache uses chained hashes so a block is identified by both its own tokens and all previous prefix blocks. This prevents the same 4-token block from being reused incorrectly when it appears under a different prefix.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests served. |
| `prompt_tok` | Total logical prompt tokens across all requests. |
| `actual_prefill` | Prompt tokens actually passed through model prefill after cache reuse. Lower is better. |
| `cached_tok` | Prompt tokens skipped by loading cached KV blocks. Higher means more reuse. |
| `gen_tok` | Generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `prefill_tok/s` | Actual prefill tokens per second. This uses actual prefill tokens, not logical prompt tokens. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. Lower is better. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `hit_rate` | Cache hits divided by cache lookups. |
| `blocks` | Number of blocks left in the cache at the end. |
| `evict` | Number of cache block evictions. |
| `forward_s` | Time spent in measured model forward calls. |

The most important metrics here are `actual_prefill`, `cached_tok`, `hit_rate`, `evict`, and `gen_tok/s`.

## Results Summary

| Case | Requests | Prompt Tokens | Cached Tokens | Actual Prefill | Prefill Reduction | Hit Rate | Throughput Ratio | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shared_prefix_basic` | 8 | 192 | 112 | 80 | 58.3% | 77.8% | 0.95x | 0 |
| `high_reuse_many_requests` | 24 | 672 | 552 | 120 | 82.1% | 85.2% | 0.94x | 0 |
| `multi_prefix_groups` | 24 | 576 | 320 | 256 | 55.6% | 76.9% | 0.94x | 0 |
| `low_reuse_control` | 8 | 192 | 0 | 192 | 0.0% | 0.0% | 0.88x | 0 |
| `eviction_pressure` | 24 | 576 | 0 | 576 | 0.0% | 0.0% | 0.85x | 136 |

The first three rows show the cache doing useful work. The last two rows show what happens when there is no reuse or when the cache is too small to retain useful prefixes.

## Main Trend: Cache Reuse Works

In the shared-prefix cases, prefix caching dramatically reduces actual prefill tokens.

### `shared_prefix_basic`

The workload has 8 requests, one shared prefix group, a 16-token shared prefix, and an 8-token unique suffix.

Results:

- Logical prompt tokens: **192**
- Actual prefill tokens with cache: **80**
- Cached tokens: **112**
- Hit rate: **77.8%**
- Prefill reduction: **58.3%**

This is exactly the intended behavior. The first request populates the cache. Later requests reuse shared prefix blocks and only process their unique suffix.

### `high_reuse_many_requests`

This is the strongest cache-reuse result. It has 24 requests sharing one 24-token prefix.

Results:

- Logical prompt tokens: **672**
- Actual prefill tokens with cache: **120**
- Cached tokens: **552**
- Hit rate: **85.2%**
- Prefill reduction: **82.1%**

This is the best demonstration that prefix caching scales with repeated shared context. Once the shared blocks are cached, almost every later request avoids most of its prompt prefill.

### `multi_prefix_groups`

This workload has 24 requests split across 4 shared-prefix groups.

Results:

- Logical prompt tokens: **576**
- Actual prefill tokens with cache: **256**
- Cached tokens: **320**
- Hit rate: **76.9%**
- Prefill reduction: **55.6%**

The reduction is smaller than `high_reuse_many_requests` because reuse is spread across multiple groups. Each group needs to warm its own prefix blocks before later requests can hit.

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


