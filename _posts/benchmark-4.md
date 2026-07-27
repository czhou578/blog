---
layout: post
title: "Developing my own Benchmark Pt. 4 (The Concurrency Test)"
date: 2026-07-26
---

In Part 3 of my series, I covered the latency and decode benchmarks that measure single-request performance. But a model that performs well in isolation doesn't tell the whole story. In production, multiple users hit the model simultaneously, and the GPU's resources get contested. This is where the concurrency benchmark comes in: it measures throughput and latency degradation as request concurrency increases.

The code lives in `concurrency.py` and answers a simple question:

> How does aggregate throughput change when I fire 1, 4, 16, or 32 requests simultaneously? At what point does the GPU bottleneck show up?

## Purpose and Approach

For each concurrency level, the benchmark fires `requests_per_level` requests at the same time using Python's `ThreadPoolExecutor`. All requests share the same prompt and parameters, so the only variable is how many are competing for the GPU at once.

The default concurrency levels are `[1, 2, 4, 8, 16]`, covering the range from single-user to heavy multi-user load. The default `requests_per_level` is 16, which is enough to get a statistically meaningful result. 

## The Core Loop

Each concurrency level runs in its own loop iteration. Inside, requests are dispatched via a thread pool:

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
    futures = [executor.submit(_run_one, i) for i in range(requests_per_level)]
    for f in concurrent.futures.as_completed(futures):
        request_results.append(f.result())
```

The `as_completed` loop is critical here. We don't wait for all 16 requests to finish in submission order. Instead, we collect results as they arrive, which is more natural for a streaming API where requests complete at different times. The wall time for the entire batch is measured with `time.time()` before and after the loop.

## Why ThreadPoolExecutor?

There are a few options for concurrent Python: `threading`, `multiprocessing`, and `asyncio`. `ThreadPoolExecutor` is the right choice here for three reasons:

1. **I/O-bound work.** The actual work is waiting for the vLLM server to process requests over HTTP. Threads are the right tool for I/O-bound concurrency because the GIL doesn't block network sockets.

2. **Simplicity.** The OpenAI-compatible client used in `core_runner.py` is synchronous. Converting the benchmark to async would require an async-compatible HTTP client. ThreadPoolExecutor works directly with the existing synchronous client.

3. **Controlled fan-out.** `max_workers=level` ensures exactly the right number of threads are alive for the concurrency level being tested. At level 4, exactly 4 threads. At level 16, exactly 16.

## The Request Worker

Each thread runs `_run_one`, a small function that takes one request, calls `client.generate()`, and returns a result dict. Error handling is per-request, not per-batch:

```python
def _run_one(idx: int) -> dict[str, Any]:
    try:
        gen = client.generate(CONCURRENCY_PROMPT, max_tokens=max_tokens, temperature=temperature)
        return {
            "success": True,
            "index": idx,
            "prompt_tokens": gen.prompt_tokens,
            "output_tokens": gen.output_tokens,
            "ttft_s": gen.ttft_s,
            "total_time_s": gen.total_time_s,
        }
    except Exception as e:
        return {"success": False, "index": idx, "error": str(e)}
```

If a request fails (e.g., the server rejects it under load), the error is captured and the batch continues. This is important because in production, not every request succeeds under heavy load, and we need to know which ones failed and why without crashing the entire experiment.

## The Prompt

Every request uses the same prompt: the creative-writing prompt from `core_runner.py`, built to exactly 256 tokens.

- **Determinism**: the prompt is fixed across all concurrency levels, so any difference in results comes from concurrency, not from different prompts.
- **Fixed context size**: 256 tokens is long enough for the prefill phase to be meaningful but short enough that it doesn't dominate wall time.
- **No repetition traps**: the prompt uses the structured-diversity approach from `build_prompt_of_length()`, so the model doesn't produce meta-commentary and terminate early.

## What Gets Measured

For each concurrency level, the benchmark computes two categories of metrics:

### Aggregate Metrics

- **`wall_time_s`**: how long the entire batch took (from first request to last completion)
- **`total_output_tokens`**: sum of output tokens from all successful requests
- **`aggregate_throughput_tok_s`**: total_output / wall_time, the key scaling metric
- **`n_requests`, `n_success`, `n_failed`, `success_rate`**: request counts and reliability

The aggregate throughput is what you want to see scale with concurrency. At level 1, if you get 50 tok/s, at level 4 you should ideally get ~200 tok/s. If you get 50 tok/s at level 1 and 60 at level 4, the GPU is bottlenecked.

### Per-Request Statistics

For every successful request, the benchmark collects the TTFT and total time, then passes them to `_stat_summary()`:

```python
results["per_concurrency_level"][str(level)] = {
    "ttft": _stat_summary(ttfts),
    "total_time_s": _stat_summary(latencies),
    # ...
}
```

`_stat_summary` returns mean, median, p95, p99, min, and max. Percentiles are critical because under concurrency, the average TTFT can look fine while the p99 is terrible. You might see an average TTFT of 100ms at concurrency 16 but a p99 of 2000ms -- that means most requests are fast but a few are stuck in queue.

### Individual Request Details

Beyond aggregates, every individual request appears in `individual_requests`:

```python
"individual_requests": [
    {
        "index": r["index"],
        "success": r["success"],
        "prompt_tokens": r["prompt_tokens"],
        "output_tokens": r["output_tokens"],
        "ttft_s": round(r["ttft_s"], 4),
        "total_time_s": round(r["total_time_s"], 4),
        "error": r.get("error", ""),
    }
    for r in request_results
]
```

Aggregate numbers hide outliers. Individual requests expose them. A single request with a TTFT 10x the others reveals a scheduling hiccup. A request with zero output tokens but a non-zero TTFT reveals a server error. Aggregate averages would smooth all of this away.

## Design Tradeoffs

### Fixed Parameters

Temperature is 0.0, max_tokens is 256, and the prompt is always the same. This minimizes variance. The benchmark isn't testing whether the model behaves differently at different temperatures; it's testing whether the server handles concurrent load.

### 256 Token Prompt

Why not use a longer prompt? The answer is that the concurrency benchmark is about scheduling, not prefill. A 32K-token prompt would make prefill the dominant cost. At 256 tokens, the prefill is fast (5-30ms) and the decode dominates. This isolates the scheduling and GPU-contention behavior.

### Number of Requests Per Level

The default is 16 requests per level. This is a practical choice:

- Too few (4) and a single slow request skews the averages.
- Too many (64) and the benchmark runtime explodes. Each level runs sequentially, so 64 requests at concurrency 16 = 64 simultaneous requests that all finish before the next level starts.
- 16 gives a good signal-to-noise ratio while keeping the benchmark fast.

### No Cache Isolation

The concurrency benchmark doesn't pass a cache salt. This means prefix caching is active: if a previous request at the same or lower concurrency already computed some KV blocks, later requests can reuse them. This makes the benchmark measure "real world" performance, where caching helps.

## Integration with the Core Runner

`concurrency.py` is imported lazily by `core_runner.py` inside the main orchestration loop. The lazy import ensures the concurrency module is only loaded when the user hasn't passed `--skip-concurrency`. The results are saved to `concurrency.json` alongside the latency, decode, and reasoning results, giving a complete picture of serving scalability.

## Why This Matters

The latency benchmark (Part 3) tells you how fast your model serves one request. The concurrency benchmark tells you how fast it serves *your* workload. In production, you'll have concurrent users, API calls, and background jobs. The concurrency benchmark reveals:

- **Where the GPU bottleneck lives**. Throughput plateaus when the GPU can no longer parallelize across requests. That plateau tells you the maximum throughput of your model at its current configuration.
- **How latency degrades under load**. At concurrency 1, the average TTFT might be 100ms. At concurrency 16, it might be 500ms. A 5x increase in concurrency causing a 5x increase in TTFT is normal (linear queuing). A 5x increase causing 50x TTFT means the scheduler is contending.
- **Where your system breaks**. Some requests will fail under high concurrency. The success rate and individual error messages tell you whether failures are timeouts, OOM errors, or HTTP 500s. 

You can find the whole code here: [https://github.com/czhou578/model-benchmarks/blob/main/benchmarks/concurrency.py](https://github.com/czhou578/model-benchmarks/blob/main/benchmarks/concurrency.py)

CZ