---
layout: post
title: "Speculative Decoding Benchmarking"
date: 2026-05-29
---

## Speculative Decoding Benchmarks (Bigram)

The annoying thing about autoregressive decoding is that it is very serial. You ask the model for one token, sample it, append it, and only then can you ask for the next one. KV caching makes each step cheaper, but it does not remove the loop.

Speculative decoding is one way to bend that loop a little. A cheap draft model guesses a short continuation. The real model then verifies the guess in one larger forward pass. If the guess is good enough, we get to emit multiple tokens while paying for only one target-model call.

This benchmark compares two decoding paths:

- **`kv_decode`**: normal autoregressive decoding with a target-model KV cache. The target model produces one generated token per decode forward pass.
- **`spec_decode`**: a cheap bigram draft model proposes multiple candidate tokens, then the target model verifies those candidates in a larger forward pass. If the candidates are accepted, the request emits multiple tokens from one target-model verification step.

The headline result is simple: speculative decoding is faster in every workload in this file.

- Throughput improves by **1.54x to 1.78x**.
- Average latency drops by roughly **35% to 44%**.
- Target forward calls drop to **40% to 49%** of the KV baseline.
- Average TTFT is slightly better in every row, although TTFT is not the main benefit of speculative decoding.

The most important nuance is that speculative decoding does **not** reduce the number of target tokens evaluated. In fact, it evaluates **13% to 48% more target tokens** than the baseline. The speedup comes from reducing the number of separate target-model forward calls by verifying multiple candidate tokens at once.

The core takeaway is the whole trick in one sentence:

> Speculative decoding wins here because it trades many tiny one-token decode forwards for fewer, fatter verification forwards. Even with a crude bigram draft model and imperfect acceptance, cutting the number of target-model calls is enough to improve end-to-end throughput.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CPU
- **Context length:** `block_size=64`
- **Prompt length:** `24` tokens in every benchmark row
- **Draft model:** cheap bigram model built from training-token transition counts
- **Benchmark target:** speculative decoding mechanics, not output quality

Before the benchmark, the tiny model trains briefly:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8682 |
| 119 | 2.7759 | 2.7995 |

The generated text sample is noisy, as you would expect from a tiny character-level model trained briefly. That is fine. This benchmark is not trying to prove the model is Shakespeare. It is about serving mechanics: how many target forwards are needed, how often draft tokens are accepted, and how throughput changes.

## What Was Benchmarked

The benchmark compares `kv_decode` against `spec_decode` over the same request workload.

| Method | Behavior |
|---|---|
| `kv_decode` | Prefill the prompt once, sample the first token, then decode one token per target-model forward call using the KV cache. |
| `spec_decode` | Prefill the prompt once, sample the first token, then repeatedly ask a bigram draft model for `K` candidate tokens and verify those candidates with the target model in one forward call. |

Speculative decoding has three important phases:

1. **Draft:** a small draft model proposes several future tokens.
2. **Verify:** the target model scores the current token plus the proposed candidates in a single forward pass.
3. **Accept or reject:** accepted draft tokens are emitted immediately; if a candidate is rejected, the benchmark samples a corrected token from the target distribution and trims the KV cache to the accepted prefix.

The draft model is intentionally almost silly. It is not a second neural network. It is just a bigram table:

```text
P(next_token | current_token)
```

That simplicity is useful because it makes the benchmark easy to run and easy to reason about. It also puts a ceiling on the result: this draft model is much weaker than the target model, so many proposed tokens will be rejected.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests served. |
| `prompt_tok` | Total prompt tokens processed during prefill. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `target_calls` | Number of target-model forward calls. Lower is usually better. |
| `target_tok` | Number of tokens evaluated by the target model, including prompt tokens and speculative verification tokens. |
| `tgt_tok/gen` | Target tokens evaluated per generated token. Lower means less target work per output token. |
| `avg_verify` | Average emitted tokens per speculative verification step. Higher is better. |
| `accept` | Fraction of proposed draft tokens accepted. Higher is better. |
| `draft_tok` | Number of draft tokens proposed. |
| `bonus` | Number of bonus tokens sampled after all candidates in a step were accepted. |
| `resample` | Number of corrected tokens sampled after a rejection. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `forward_s` | Time spent in measured target-model forward work. |

For this benchmark, the most important metrics are `gen_tok/s`, `target_calls`, `target_tok`, `accept`, `avg_verify`, and `avg_lat_ms`.

## Results Summary

| Case | Requests | Generated Tokens | K | Draft Noise | KV Tok/s | Spec Tok/s | Throughput Ratio | Target Call Ratio | Target Token Ratio | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k2_bigram_draft` | 8 | 128 | 2 | 0.0 | 896.05 | 1379.86 | 1.54x | 0.49x | 1.13x | 63.6% |
| `k4_bigram_draft` | 8 | 128 | 4 | 0.0 | 895.85 | 1549.35 | 1.73x | 0.43x | 1.29x | 48.1% |
| `k6_bigram_draft` | 8 | 128 | 6 | 0.0 | 938.86 | 1616.56 | 1.72x | 0.43x | 1.48x | 34.7% |
| `k4_noisy_draft` | 8 | 128 | 4 | 0.5 | 957.25 | 1707.07 | 1.78x | 0.40x | 1.26x | 52.2% |
| `longer_outputs_k4` | 6 | 144 | 4 | 0.0 | 893.93 | 1589.45 | 1.78x | 0.41x | 1.40x | 46.2% |

Speculative decoding improves throughput in every case. Across the five rows, the average throughput ratio is about **1.71x**. For a draft model this primitive, that is a surprisingly healthy win.

## Main Trend: Fewer Target Calls Beats More Target Tokens

The central pattern is:

- `spec_decode` makes far fewer target-model forward calls.
- `spec_decode` often evaluates more total target tokens.
- Despite evaluating more target tokens, it still runs faster.

This can look contradictory at first. Why is the model evaluating more tokens and still finishing sooner? The answer is that the shape of the work matters, not just the count.

Normal KV decoding is sequential:

```text
target forward -> 1 token
target forward -> 1 token
target forward -> 1 token
...
```

Speculative decoding groups work:

```text
draft proposes K tokens
target verifies K+1 positions in one forward
emit several tokens if accepted
```

Each speculative verification forward is larger, but there are many fewer of them. In this benchmark, target forward calls drop from **128** to **51-63** for the 8-request, 128-token workloads. That is the main source of the speedup.

The target-token count tells the other side of the tradeoff:

| Case | KV Target Tokens | Spec Target Tokens | Increase |
|---|---:|---:|---:|
| `k2_bigram_draft` | 312 | 354 | +13% |
| `k4_bigram_draft` | 312 | 401 | +29% |
| `k6_bigram_draft` | 312 | 461 | +48% |
| `k4_noisy_draft` | 312 | 392 | +26% |
| `longer_outputs_k4` | 282 | 394 | +40% |

Speculative decoding is doing extra target-token work, but it packages that work into fewer calls. On this CPU microbenchmark, reducing Python/PyTorch forward-call overhead and giving each call a little more useful work outweighs the extra token evaluations.

## Speculation Depth: K=2 vs K=4 vs K=6

The cleanest comparison is the three bigram-draft rows with the same workload:

| Case | K | Spec Tok/s | Target Calls | Target Tokens | Avg Verify | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| `k2_bigram_draft` | 2 | 1379.86 | 63 | 354 | 2.18 | 63.6% |
| `k4_bigram_draft` | 4 | 1549.35 | 55 | 401 | 2.55 | 48.1% |
| `k6_bigram_draft` | 6 | 1616.56 | 55 | 461 | 2.55 | 34.7% |

Increasing `K` from 2 to 4 helps:

- Throughput rises from **1379.86** to **1549.35 tok/s**.
- Target calls fall from **63** to **55**.
- Average emitted tokens per verification rises from **2.18** to **2.55**.

This is the happy path for speculation: propose more than one token, verify them together, and sometimes move the sequence forward by several tokens from one target call.

Increasing `K` from 4 to 6 is more mixed:

- Throughput rises only slightly: **1549.35** to **1616.56 tok/s**.
- Target calls do not improve: both use **55** target calls.
- Target tokens evaluated jump from **401** to **461**.
- Acceptance falls sharply from **48.1%** to **34.7%**.

This is the classic speculative-decoding tuning curve. Larger `K` gives you a chance to accept more tokens per verification, but it also asks the draft model to predict farther into the future. Weak draft models get worse as the chain gets longer.

The result:

> K=4 is the better balanced setting in this benchmark. K=6 does not meaningfully reduce target calls, but it does increase verification-token work.

## Acceptance Rate Matters

Acceptance rate measures how often draft tokens survive target verification.

| Case | Draft Tokens Proposed | Accepted Draft Tokens | Acceptance |
|---|---:|---:|---:|
| `k2_bigram_draft` | 107 | 68 | 63.6% |
| `k4_bigram_draft` | 162 | 78 | 48.1% |
| `k6_bigram_draft` | 222 | 77 | 34.7% |
| `k4_noisy_draft` | 157 | 82 | 52.2% |
| `longer_outputs_k4` | 197 | 91 | 46.2% |

The acceptance rate falls as speculation depth increases:

- At `K=2`, the draft only needs to be right for short spans.
- At `K=4`, it proposes farther ahead, so more chains break.
- At `K=6`, many later candidates become wasted verification work.

Acceptance does not need to be perfect for speculative decoding to help. The `k4_bigram_draft` row accepts less than half of draft tokens and still reaches **1.73x** KV throughput. The reason is that a verification step does not have to be perfect to be useful. Even a partial accept can emit more than one token.

However, acceptance still sets the ceiling. A stronger draft model would likely improve throughput by:

- raising `avg_verify`,
- reducing resampling,
- reducing wasted target-token evaluations,
- allowing larger `K` values to remain useful.

## Bonus And Resampled Tokens

Speculative decoding has two non-obvious token counters:

| Case | Bonus Tokens | Resampled Tokens |
|---|---:|---:|
| `k2_bigram_draft` | 28 | 24 |
| `k4_bigram_draft` | 13 | 29 |
| `k6_bigram_draft` | 6 | 37 |
| `k4_noisy_draft` | 13 | 25 |
| `longer_outputs_k4` | 13 | 34 |

**Bonus tokens** happen when all draft candidates in a verification step are accepted. The target has already produced the next distribution after those candidates, so the benchmark can sample one extra token.

**Resampled tokens** happen when a draft candidate is rejected. The benchmark samples a replacement token from an adjusted target distribution, emits it, and stops accepting further candidates from that speculative batch.

The trend is useful:

- `K=2` has the most bonus tokens because shorter speculative chains are easier to accept completely.
- `K=6` has fewer bonus tokens and more resampled tokens because long chains are more likely to reject somewhere.

This again shows why larger speculation depth is not automatically better.

## Latency And TTFT

Speculative decoding improves average latency in every row:

| Case | KV Avg Latency | Spec Avg Latency | Reduction |
|---|---:|---:|---:|
| `k2_bigram_draft` | 17.86 ms | 11.59 ms | 35.1% |
| `k4_bigram_draft` | 17.86 ms | 10.33 ms | 42.2% |
| `k6_bigram_draft` | 17.04 ms | 9.90 ms | 41.9% |
| `k4_noisy_draft` | 16.71 ms | 9.37 ms | 43.9% |
| `longer_outputs_k4` | 26.85 ms | 15.10 ms | 43.8% |

The latency improvement tracks the reduction in target calls. Each request needs fewer decode iterations before it reaches its output length.

TTFT improves only slightly:

| Case | KV Avg TTFT | Spec Avg TTFT |
|---|---:|---:|
| `k2_bigram_draft` | 1.55 ms | 1.36 ms |
| `k4_bigram_draft` | 1.60 ms | 1.35 ms |
| `k6_bigram_draft` | 1.53 ms | 1.27 ms |
| `k4_noisy_draft` | 1.48 ms | 1.42 ms |
| `longer_outputs_k4` | 1.69 ms | 1.45 ms |

This is expected. Both methods do the same prompt prefill before producing the first sampled token. Speculative decoding mostly accelerates the continuation after the first token, so its biggest benefit appears in request latency and generated-token throughput rather than TTFT.

## The Noisy Draft Result

`k4_noisy_draft` uses the same `K=4` depth as `k4_bigram_draft`, but blends the draft distribution with uniform noise:

```text
draft_noise=0.5
```

Surprisingly, it is the fastest 128-token row:

- Throughput: **1707.07 tok/s**
- Throughput ratio: **1.78x**
- Target calls: **51**, lower than **55** for the clean K=4 run
- Acceptance: **52.2%**, slightly higher than **48.1%**

I would not read this as "noise improves drafts." The benchmark is stochastic: token sampling, acceptance draws, and the tiny model's uncertain distributions can make one run look unusually favorable. With only 8 requests and 128 generated tokens, randomness has room to move the result.

The more conservative interpretation is:

> The K=4 speculative path is robust in this small benchmark. Even with a noisier draft distribution, it still reduces target calls and improves throughput.

For a stronger conclusion about draft quality, this benchmark should be repeated over multiple seeds and larger request counts.

## Longer Output Run

The `longer_outputs_k4` row changes the workload to:

```text
num_requests=6
prompt_len=24
max_new_tokens=24
speculation_len=4
```

Compared with the 16-token output rows, this gives each request more decode work after the initial prompt prefill.

Results:

- KV throughput: **893.93 tok/s**
- Speculative throughput: **1589.45 tok/s**
- Throughput ratio: **1.78x**
- Target call ratio: **0.41x**
- Average latency: **26.85 ms** down to **15.10 ms**

This row matters because speculative decoding is mostly a decode-phase optimization. Longer outputs mean more of the request is spent in the serial token loop. That gives speculation more opportunities to collapse one-token forwards into fewer verification forwards, and the speedup remains strong.

## Forward Time

The `forward_s` column shows time spent in measured target-model forward work:

| Case | KV Forward Time | Spec Forward Time | Reduction |
|---|---:|---:|---:|
| `k2_bigram_draft` | 0.1364 s | 0.0807 s | 40.8% |
| `k4_bigram_draft` | 0.1364 s | 0.0703 s | 48.5% |
| `k6_bigram_draft` | 0.1302 s | 0.0663 s | 49.1% |
| `k4_noisy_draft` | 0.1274 s | 0.0635 s | 50.2% |
| `longer_outputs_k4` | 0.1535 s | 0.0764 s | 50.2% |

This column is a useful sanity check. The speculative path is not merely moving time outside the measured forward loop. It substantially reduces measured target-forward time.

One caveat: this is a tiny CPU model. On a larger GPU model, the exact balance can change. Speculative decoding usually matters most when the target model is expensive and the draft model is much cheaper.

## Row-by-Row Interpretation

### `k2_bigram_draft`

This is the conservative speculation setting:

```text
num_requests=8
prompt_len=24
max_new_tokens=16
speculation_len=2
draft_noise=0.0
```

The draft model proposes at most 2 tokens per verification step.

Results:

- Throughput improves from **896.05** to **1379.86 tok/s**.
- Target calls drop from **128** to **63**.
- Acceptance is **63.6%**, the highest in the suite.
- Average latency drops from **17.86 ms** to **11.59 ms**.

This is the cautious setting. K=2 has high acceptance and relatively little wasted verification work, but it also leaves some batching benefit on the table.

### `k4_bigram_draft`

This is the most balanced clean-draft setting:

```text
speculation_len=4
draft_noise=0.0
```

Results:

- Throughput improves to **1549.35 tok/s**, a **1.73x** speedup.
- Target calls drop to **55**, or **43%** of baseline.
- Average emitted tokens per verification rises to **2.55**.
- Acceptance falls to **48.1%**, but the run is still much faster.

This is the strongest clean demonstration that speculative decoding can tolerate imperfect drafts. Less than half of candidate tokens are accepted, but enough verification steps emit multiple tokens to beat the one-token-at-a-time loop.

### `k6_bigram_draft`

This row pushes speculation deeper:

```text
speculation_len=6
draft_noise=0.0
```

Results:

- Throughput is **1616.56 tok/s**, slightly above K=4.
- Target calls stay at **55**, the same as K=4.
- Target tokens rise to **461**, much higher than K=4's **401**.
- Acceptance falls to **34.7%**.

K=6 is not a clean improvement. It asks the target model to verify more speculative positions, but it does not reduce target calls beyond K=4. The draft model is too weak to reliably support the longer chain.

### `k4_noisy_draft`

This row tests whether the speculative path still works with a lower-quality draft distribution:

```text
speculation_len=4
draft_noise=0.5
```

Results:

- Throughput is **1707.07 tok/s**, the fastest 128-token row.
- Target calls drop to **51**.
- Acceptance is **52.2%**.
- Average latency drops to **9.37 ms**.

The result is positive, but likely includes sampling noise. The safe conclusion is not that noisy drafts are better. It is that this speculative decoding implementation remains effective even when the draft distribution is imperfect.

### `longer_outputs_k4`

This row keeps `K=4` but increases each request from 16 to 24 generated tokens:

```text
num_requests=6
prompt_len=24
max_new_tokens=24
```

Results:

- Throughput improves from **893.93** to **1589.45 tok/s**.
- Target calls drop from **144** to **59**.
- Average latency drops from **26.85 ms** to **15.10 ms**.
- Acceptance is **46.2%**.

This row confirms the benefit scales into longer decode workloads. More output tokens means more chances to replace one-token target forwards with multi-token verification forwards.

## Why Speculative Decoding Matters

The broader reason this matters is that autoregressive decoding is hard to parallelize. Each next token depends on the previous token. KV caching removes repeated attention work over the prefix, but it does not remove the sequential loop:

```text
sample token 1
sample token 2
sample token 3
...
```

Speculative decoding attacks that loop by using a cheap model to guess several future tokens. The target model still controls correctness: it verifies the draft tokens and rejects or corrects them when needed.

That is why speculative decoding is attractive: the target model still owns the final distribution, but the serving loop can sometimes advance by several tokens per target call.

In this benchmark, the effect is visible even with a very small model:

- The target model is called much less often.
- Multiple tokens are often emitted per verification.
- Average latency falls substantially.
- Throughput rises across every workload.

## Caveats

These results are useful, but they are still a toy benchmark. The caveats matter.

1. **The model is tiny.** With only `0.056769M` parameters, overheads and sampling noise are large relative to model compute.
2. **The draft model is a bigram table.** Real speculative decoding usually uses a smaller neural draft model or specialized draft heads.
3. **The run is CPU-only.** GPU behavior can differ because larger verification forwards may utilize hardware differently.
4. **The sample size is small.** Most rows generate only 128 tokens, so random sampling can affect acceptance and throughput.
5. **The benchmark is single-run.** Multiple seeds would make the `k4_noisy_draft` result easier to interpret.
6. **Quality is not evaluated.** The benchmark measures mechanics and speed, not whether generated text is good.

## Practical Takeaways

1. **Speculative decoding is working in this implementation.** Every run completes the same generated-token count and improves throughput over KV decoding.
2. **The main win is fewer target forwards.** Target calls fall to about **40% to 49%** of baseline.
3. **More speculation is not always better.** K=6 evaluates many more target tokens and has much lower acceptance than K=4.
4. **Acceptance rate is a key health metric.** Higher acceptance means more emitted tokens per verification and less wasted target work.
5. **Longer decode workloads are a better fit.** The `longer_outputs_k4` row shows speculation remains strong when more time is spent in decode.
6. **Draft quality should be tested over multiple seeds.** The noisy-draft row is encouraging, but too small to prove that noise helps.


## Speculative Decoding Benchmarks (Trigram)

The bigram draft above is about as cheap as a draft model can get: look at the current token, guess the next one. The trigram draft gives the guesser one more token of context. It is still tiny, still table-based, and still nowhere near a neural draft model. But it lets us ask a useful question: does a slightly richer draft change the shape of the benchmark?

This benchmark compares two decoding paths:

- **`kv_decode`**: normal autoregressive decoding with the target model and a KV cache. After prefill, each target forward produces one generated token.
- **`trigram_spec_decode`**: a cheap trigram draft model proposes several future tokens, and the target model verifies those candidates in a larger forward pass.

The trigram speculative path is faster in every workload:

- Throughput improves by **1.40x to 2.13x**.
- Average request latency drops by about **29% to 53%**.
- Target forward calls fall to **47% to 62%** of the KV baseline.
- TTFT is slightly better in every row.

The main tradeoff is that speculative decoding evaluates more target tokens:

- Target token work rises to **1.29x to 1.70x** of the KV baseline.
- Acceptance is modest, ranging from **25.7% to 39.9%**.
- Larger speculation depths reduce target calls, but they also create more rejected or wasted speculative positions.

The core result is the same pattern as before:

> Trigram speculative decoding improves throughput because it greatly reduces the number of target-model forward calls, even though it evaluates more total target tokens than normal KV decoding.

So the story has not changed. We still have fewer serial target calls, larger verification calls, and an acceptance rate that decides how much of the speculative work turns into useful output.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CPU
- **Context length:** `block_size=64`
- **Prompt length:** `24` tokens in every row
- **Draft model:** smoothed trigram table built from training-token transition counts
- **Benchmark target:** speculative decoding mechanics, not output quality

The tiny model trains briefly before the benchmark:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8682 |
| 119 | 2.7759 | 2.7995 |

The generated sample text is noisy, again as expected for a tiny character-level model trained for a short run. The benchmark is focused on serving behavior: target forward calls, draft acceptance, verification size, latency, and throughput.

## What Was Benchmarked

The benchmark runs the same request workload through two policies.

| Method | Behavior |
|---|---|
| `kv_decode` | Prefill the prompt once, sample the first token, then decode one token per target-model forward using the KV cache. |
| `trigram_spec_decode` | Prefill the prompt once, sample the first token, then use a trigram draft model to propose `K` future tokens. The target model verifies `[current_token] + candidates` in one forward pass. |

The trigram draft estimates:

```text
P(next_token | token_{t-2}, token_{t-1})
```

So it conditions on the previous two tokens instead of only the previous one. It is still extremely cheap compared with a neural draft model, but it has more local context than the bigram table.

The speculative loop has three phases:

1. **Draft:** propose up to `K` candidate tokens from the trigram table.
2. **Verify:** run the target model on the current token plus all candidates.
3. **Accept/reject:** accept matching draft tokens, sample a corrected token on rejection, or sample a bonus token if the full draft chain is accepted.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests served. |
| `prompt_tok` | Total prompt tokens processed during prefill. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `target_calls` | Number of target-model forward calls. Lower is usually better. |
| `target_tok` | Total tokens evaluated by the target model, including prompt and verification tokens. |
| `tgt_tok/gen` | Target tokens evaluated per generated token. |
| `avg_verify` | Average emitted tokens per speculative verification step. Higher is better. |
| `accept` | Fraction of proposed draft tokens accepted by target verification. |
| `draft_tok` | Number of draft tokens proposed. |
| `bonus` | Number of bonus tokens sampled when every draft candidate in a step is accepted. |
| `resample` | Number of corrected tokens sampled after rejection. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `forward_s` | Time spent in measured target-model forward work. |

For this benchmark, the most important metrics are `gen_tok/s`, `target_calls`, `target_tok`, `avg_verify`, `accept`, and `avg_lat_ms`.

## Results Summary

| Case | Requests | Generated Tokens | K | Draft Noise | KV Tok/s | Trigram Spec Tok/s | Throughput Ratio | Target Call Ratio | Target Token Ratio | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k2_trigram_draft` | 8 | 128 | 2 | 0.0 | 545.44 | 1162.72 | 2.13x | 0.62x | 1.29x | 39.9% |
| `k4_trigram_draft` | 8 | 128 | 4 | 0.0 | 877.26 | 1232.47 | 1.40x | 0.54x | 1.52x | 28.1% |
| `k6_trigram_draft` | 8 | 128 | 6 | 0.0 | 959.98 | 1361.51 | 1.42x | 0.48x | 1.70x | 25.7% |
| `k4_noisy_trigram` | 8 | 128 | 4 | 0.5 | 941.86 | 1318.78 | 1.40x | 0.51x | 1.47x | 31.9% |
| `longer_outputs_k4_trigram` | 6 | 144 | 4 | 0.0 | 958.46 | 1431.82 | 1.49x | 0.47x | 1.58x | 33.5% |

Speculative decoding improves throughput in every row. The average throughput ratio is about **1.57x**. The win is real, but a bit less clean than the bigram section because the individual baselines vary more.

One caution: the `k2_trigram_draft` KV baseline is much slower than the other KV baselines in this section. That makes its **2.13x** ratio look extra dramatic. In absolute terms, K=2 is not the fastest speculative row; K=6 and the longer-output K=4 run generate more tokens per second.

## Main Trend: The Trigram Draft Reduces Target Calls

The cleanest success signal is target forward calls.

| Case | KV Target Calls | Trigram Target Calls | Calls Saved | Reduction |
|---|---:|---:|---:|---:|
| `k2_trigram_draft` | 128 | 79 | 49 | 38.3% |
| `k4_trigram_draft` | 128 | 69 | 59 | 46.1% |
| `k6_trigram_draft` | 128 | 61 | 67 | 52.3% |
| `k4_noisy_trigram` | 128 | 65 | 63 | 49.2% |
| `longer_outputs_k4_trigram` | 144 | 68 | 76 | 52.8% |

This is exactly the point of speculative decoding. Normal KV decode has a serial loop:

```text
target forward -> one generated token
target forward -> one generated token
target forward -> one generated token
...
```

Trigram speculative decoding changes that shape:

```text
draft proposes several tokens
target verifies several positions in one call
emit one or more tokens
```

Even when many draft tokens are rejected, each verification step can still emit a corrected token. When some tokens are accepted, the request advances by more than one token from one target call.

## The Tradeoff: More Target Tokens Evaluated

Again, the speedup does not come from reducing the total number of target tokens evaluated. It comes from reducing the number of separate target calls.

| Case | KV Target Tokens | Trigram Target Tokens | Increase |
|---|---:|---:|---:|
| `k2_trigram_draft` | 312 | 401 | +28.5% |
| `k4_trigram_draft` | 312 | 474 | +51.9% |
| `k6_trigram_draft` | 312 | 529 | +69.6% |
| `k4_noisy_trigram` | 312 | 459 | +47.1% |
| `longer_outputs_k4_trigram` | 282 | 445 | +57.8% |

This is the central speculative-decoding exchange:

- **Cost:** target verification evaluates extra candidate positions.
- **Benefit:** those positions are evaluated in fewer, larger forward calls.

On this CPU run, fewer calls win. The target token count rises substantially, but wall-clock time still drops. This is the benchmark saying: call overhead and serial stepping are expensive enough that batching the verification work pays off.

## Speculation Depth: K=2 vs K=4 vs K=6

The three clean trigram rows compare speculation depth on the same 8-request, 128-token workload.

| Case | K | Spec Tok/s | Target Calls | Target Tokens | Avg Verify | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| `k2_trigram_draft` | 2 | 1162.72 | 79 | 401 | 1.69 | 39.9% |
| `k4_trigram_draft` | 4 | 1232.47 | 69 | 474 | 1.97 | 28.1% |
| `k6_trigram_draft` | 6 | 1361.51 | 61 | 529 | 2.26 | 25.7% |

Increasing `K` reduces target calls:

- K=2 uses **79** target calls.
- K=4 uses **69** target calls.
- K=6 uses **61** target calls.

It also increases average emitted tokens per verification:

- K=2 emits **1.69** tokens per verify step.
- K=4 emits **1.97** tokens per verify step.
- K=6 emits **2.26** tokens per verify step.

But this comes with two costs:

- Acceptance drops from **39.9%** to **25.7%**.
- Target tokens evaluated rise from **401** to **529**.

The best absolute throughput among these three rows is K=6 at **1361.51 tok/s**, but K=6 also does the most extra target-token work. This is the same speculation-depth tradeoff again:

> Larger K can reduce serial target calls, but weak later predictions create more rejected candidate work.

## Acceptance Rate Is Modest

The trigram draft's acceptance rates are not high:

| Case | Draft Tokens Proposed | Accepted Draft Tokens | Acceptance |
|---|---:|---:|---:|
| `k2_trigram_draft` | 138 | 55 | 39.9% |
| `k4_trigram_draft` | 221 | 62 | 28.1% |
| `k6_trigram_draft` | 284 | 73 | 25.7% |
| `k4_noisy_trigram` | 210 | 67 | 31.9% |
| `longer_outputs_k4_trigram` | 239 | 80 | 33.5% |

This means most proposed draft tokens are not directly accepted. The speculative path still wins because it often emits more than one token per target verification, and because even rejection steps still produce a corrected token.

Still, the acceptance rate sets the ceiling. A stronger draft model would likely:

- increase `avg_verify`,
- reduce `resample`,
- reduce target tokens evaluated per generated token,
- make larger `K` values more efficient.

The low acceptance also tells us the trigram draft is not a very strong approximation of the target model. It is useful as a cheap educational draft, but it is not close to a production neural draft model.

## Bonus And Resampled Tokens

Two counters explain how speculative steps end.

| Case | Bonus Tokens | Resampled Tokens |
|---|---:|---:|
| `k2_trigram_draft` | 15 | 50 |
| `k4_trigram_draft` | 4 | 54 |
| `k6_trigram_draft` | 2 | 45 |
| `k4_noisy_trigram` | 7 | 46 |
| `longer_outputs_k4_trigram` | 6 | 52 |

**Bonus tokens** happen when all draft candidates in a verification step are accepted. The target has already computed the next distribution, so the benchmark can sample one extra token.

**Resampled tokens** happen when a draft candidate is rejected. The benchmark emits a corrected token and stops using the rest of that speculative batch.

The pattern is intuitive:

- K=2 has the most bonus tokens because shorter speculative chains are easier to accept completely.
- K=4 and K=6 have fewer bonus tokens because a longer chain is more likely to reject somewhere.
- Resampling is common in every trigram run, which reflects the modest draft quality.

## Latency And TTFT

Trigram speculative decoding improves average latency in every row:

| Case | KV Avg Latency | Trigram Avg Latency | Reduction |
|---|---:|---:|---:|
| `k2_trigram_draft` | 29.33 ms | 13.76 ms | 53.1% |
| `k4_trigram_draft` | 18.24 ms | 12.98 ms | 28.8% |
| `k6_trigram_draft` | 16.67 ms | 11.75 ms | 29.5% |
| `k4_noisy_trigram` | 16.99 ms | 12.13 ms | 28.6% |
| `longer_outputs_k4_trigram` | 25.04 ms | 16.76 ms | 33.1% |

Latency improves because each request needs fewer target-model iterations to reach its output length.

TTFT also improves slightly:

| Case | KV Avg TTFT | Trigram Avg TTFT |
|---|---:|---:|
| `k2_trigram_draft` | 2.00 ms | 1.30 ms |
| `k4_trigram_draft` | 1.72 ms | 1.37 ms |
| `k6_trigram_draft` | 1.51 ms | 1.29 ms |
| `k4_noisy_trigram` | 1.65 ms | 1.33 ms |
| `longer_outputs_k4_trigram` | 1.63 ms | 1.36 ms |

TTFT is not the main speculative-decoding benefit because both methods still prefill the prompt before emitting the first sampled token. The larger win is in post-first-token decode latency.

## The Noisy Trigram Result

`k4_noisy_trigram` uses the same K=4 depth as `k4_trigram_draft`, but blends the draft distribution with uniform noise:

```text
draft_noise=0.5
```

Results:

- Throughput rises from **1232.47** to **1318.78 tok/s** compared with clean K=4.
- Target calls fall from **69** to **65**.
- Target tokens fall from **474** to **459**.
- Acceptance rises from **28.1%** to **31.9%**.

This looks like noise helped, but I would be careful. These are short, stochastic runs, and sampling variation can easily move acceptance and latency. The safer reading is:

> The K=4 trigram speculative path is robust to a noisier draft distribution in this run.

To prove that noise truly improves the draft, this benchmark would need multiple seeds, larger sample sizes, and direct comparison against held-out next-token accuracy.

## Longer Output Run

The `longer_outputs_k4_trigram` row changes the workload to:

```text
num_requests=6
prompt_len=24
max_new_tokens=24
speculation_len=4
```

This creates more decode work per request. That matters because speculative decoding is mainly a decode-phase optimization.

Results:

- KV throughput: **958.46 tok/s**
- Trigram speculative throughput: **1431.82 tok/s**
- Throughput ratio: **1.49x**
- Target calls drop from **144** to **68**
- Average latency drops from **25.04 ms** to **16.76 ms**

This is one of the most meaningful rows. Longer outputs give speculative decoding more opportunities to collapse many one-token decode forwards into fewer verification forwards. If the optimization is real, this is where we expect it to show up.

## Forward Time

The measured forward time drops in every row:

| Case | KV Forward Time | Trigram Forward Time | Reduction |
|---|---:|---:|---:|
| `k2_trigram_draft` | 0.2227 s | 0.0947 s | 57.5% |
| `k4_trigram_draft` | 0.1387 s | 0.0871 s | 37.2% |
| `k6_trigram_draft` | 0.1269 s | 0.0774 s | 39.0% |
| `k4_noisy_trigram` | 0.1295 s | 0.0819 s | 36.8% |
| `longer_outputs_k4_trigram` | 0.1430 s | 0.0839 s | 41.3% |

This is another useful sanity check. The speedup is not just outside the model. The speculative path reduces synchronized target-forward time despite evaluating more target tokens, because it uses far fewer separate target calls.

## Row-by-Row Interpretation

### `k2_trigram_draft`

This is the conservative trigram setting:

```text
speculation_len=2
draft_noise=0.0
```

Results:

- Throughput improves from **545.44** to **1162.72 tok/s**.
- Target calls drop from **128** to **79**.
- Acceptance is **39.9%**, the highest in the suite.
- Average latency drops from **29.33 ms** to **13.76 ms**.

This row has the largest throughput ratio, but the KV baseline is unusually slow compared with the other KV rows. The result still shows speculative decoding working, but I would not over-weight the ratio.

### `k4_trigram_draft`

This row increases speculation depth to 4:

```text
speculation_len=4
draft_noise=0.0
```

Results:

- Throughput improves from **877.26** to **1232.47 tok/s**.
- Target calls drop from **128** to **69**.
- Target tokens rise from **312** to **474**.
- Acceptance falls to **28.1%**.

K=4 is a more aggressive setting. It saves more target calls than K=2, but it also verifies many more candidate positions and accepts a smaller fraction of them.

### `k6_trigram_draft`

This row pushes speculation depth to 6:

```text
speculation_len=6
draft_noise=0.0
```

Results:

- Throughput reaches **1361.51 tok/s**, the fastest 128-token clean trigram run.
- Target calls drop to **61**, the lowest among the 128-token clean rows.
- Target tokens rise to **529**, the highest among the same rows.
- Acceptance falls to **25.7%**.

K=6 wins absolute throughput here, but it is not a free win. It does much more target-token verification work. On a larger model or different hardware, that extra work might become the bottleneck.

### `k4_noisy_trigram`

This row adds noise to the K=4 trigram draft:

```text
speculation_len=4
draft_noise=0.5
```

Results:

- Throughput is **1318.78 tok/s**.
- Target calls drop to **65**.
- Acceptance is **31.9%**.
- Average latency is **12.13 ms**.

The noisy row performs better than the clean K=4 row in this single run, but the likely explanation is sampling variance. The stronger conclusion is that the trigram speculative mechanism remains effective even when the draft distribution is imperfect.

### `longer_outputs_k4_trigram`

This row tests K=4 on longer outputs:

```text
num_requests=6
max_new_tokens=24
```

Results:

- Throughput improves from **958.46** to **1431.82 tok/s**.
- Target calls drop from **144** to **68**.
- Average emitted tokens per verification is **2.23**.
- Average latency drops by about **33%**.

This row confirms that trigram speculative decoding remains useful when decode work is a larger share of the request.

## Why Trigram Speculative Decoding Matters

KV caching avoids recomputing the full prompt at every decode step, but it does not remove the serial autoregressive loop. A normal decoder still asks the target model for one token at a time. That is the thing speculation is trying to relax.

Speculative decoding uses a cheap draft model to guess several future tokens, then asks the target model to verify those guesses in one call. The target model still controls the final distribution, but the serving loop may advance by multiple tokens per verification.

This benchmark shows that even a simple trigram draft can provide useful speedups:

- target calls are roughly halved,
- latency drops,
- generated-token throughput improves in every run,
- longer decode workloads benefit clearly.

## Caveats

These results are useful, but they are still a toy benchmark. The caveats matter.

1. **The model is tiny.** With only `0.056769M` parameters, Python and PyTorch overhead can strongly affect timing.
2. **The run is CPU-only.** GPU behavior may differ, especially for larger target models and larger verification batches.
3. **The draft is a smoothed trigram table.** It is cheap, but not a strong approximation of the target model.
4. **Acceptance is low.** Most proposed draft tokens are rejected, so a better draft could improve the result substantially.
5. **The benchmark is stochastic.** Single-run differences, especially the noisy-draft row and the slow K=2 baseline, should be interpreted carefully.
6. **Quality is not measured.** The benchmark measures speed and mechanics, not generated text quality.

## Practical Takeaways

1. **The trigram speculative path works.** It completes the same generated-token count and improves throughput in every case.
2. **The main win is fewer target calls.** Target calls fall to **47% to 62%** of normal KV decoding.
3. **Target-token work increases.** The speculative path evaluates **29% to 70%** more target tokens.
4. **Acceptance is the limiting factor.** Acceptance ranges only **25.7% to 39.9%**, which leaves plenty of room for a stronger draft.
5. **Larger K is a tradeoff.** K=6 has the fastest clean trigram throughput, but also the most extra target-token verification.
6. **Longer outputs are a natural fit.** The longer-output row shows a clear speedup when decode work dominates more of the request.