---
layout: post
title: "Adding Trigram to Speculative Decoding"
date: 2026-05-29
---

Previously, we added speculative decoding to nanoGPT using a bigram draft model.

The bigram model was intentionally simple:

```text
P(next_token | current_token)
```

It could look at one token of context, sample a few cheap draft tokens, and let the real transformer verify them in one forward pass.

In this post, we will upgrade the draft model from a bigram to a trigram:

```text
P(next_token | previous_token, current_token)
```

The hope is simple. If the draft model sees a little more context, its guesses should be closer to the target model. If the guesses are closer, speculative decoding accepts more tokens. If it accepts more tokens, we get more generated tokens per expensive target-model forward pass.

## Why Trigram?

A bigram table is a tiny lookup table of shape:

```text
(vocab_size, vocab_size)
```

Row `a` stores the distribution of tokens that usually follow token `a`.

This is useful, but it is also obviously limited. Suppose the current token is `"t"`. What comes next?

```text
" t" -> likely "h" or "o"
"ht" -> probably rare
"st" -> maybe " " or another letter
```

The next token depends a lot on the previous token. A bigram only sees `"t"`, so it has to average all of these situations together.

A trigram table keeps one more token of context. It asks:

```text
given (previous_token, current_token), what usually comes next?
```

This is still a very weak language model. It knows nothing about syntax, meaning, attention, or long-range dependencies. But for speculative decoding, the draft model does not need to be smart in the transformer sense. It needs to be:

- cheap
- plausible
- able to return the probability of whatever it sampled

A trigram table does all three.

## The Table

The core object is a 3D count tensor:

```text
counts[a, b, c] = number of times token c followed the context (a, b)
```

For a character-level Shakespeare vocabulary of about 65 tokens, this table has:

```text
65 * 65 * 65 = 274,625 entries
```

That sounds large compared to the bigram table, but in absolute terms it is tiny. It is just a few hundred thousand floats. No matmuls, no attention, no layers, just indexing.

Here is the draft model:

```python
class TrigramDraftModel:
    def __init__(self, token_ids, vocab_size, device, fallback_bigram=None, min_context_count=2):
        counts = torch.zeros(vocab_size, vocab_size, vocab_size, dtype=torch.float32)
        ids = torch.as_tensor(token_ids, dtype=torch.long).flatten().cpu()

        for a, b, c in zip(ids[:-2].tolist(), ids[1:-1].tolist(), ids[2:].tolist()):
            if 0 <= a < vocab_size and 0 <= b < vocab_size and 0 <= c < vocab_size:
                counts[a, b, c] += 1.0

        self.context_counts = counts.sum(dim=-1)

        counts += 1.0
        self.probs = counts / counts.sum(dim=-1, keepdim=True)

        self.probs = self.probs.to(device)
        self.context_counts = self.context_counts.to(device)
        self.fallback_bigram = fallback_bigram
        self.min_context_count = min_context_count
```

Let's unpack it.

```python
counts = torch.zeros(vocab_size, vocab_size, vocab_size, dtype=torch.float32)
```

This creates the raw histogram. The three axes are:

```text
previous token, current token, next token
```

So `counts[a, b]` is a vector of length `vocab_size`. It stores the next-token histogram after seeing the two-token context `(a, b)`.

Then we scan the training data with a sliding window:

```python
for a, b, c in zip(ids[:-2].tolist(), ids[1:-1].tolist(), ids[2:].tolist()):
    counts[a, b, c] += 1.0
```

Visually:

```text
tokens:  [x0, x1, x2, x3, x4, ...]
window:  (x0, x1, x2)
window:      (x1, x2, x3)
window:          (x2, x3, x4)
```

For each triple `(a, b, c)`, we increment the count saying that `c` followed `(a, b)`.

Before smoothing, we save the amount of real evidence for each context:

```python
self.context_counts = counts.sum(dim=-1)
```

This produces a `(vocab_size, vocab_size)` tensor:

```text
context_counts[a, b] = number of times context (a, b) appeared
```

This matters because trigram tables are sparse. Some contexts appear thousands of times. Some appear once. Some never appear. We should not trust all rows equally.

Finally, we smooth and normalize:

```python
counts += 1.0
self.probs = counts / counts.sum(dim=-1, keepdim=True)
```

After this, every `self.probs[a, b]` is a valid next-token probability distribution:

```text
self.probs[a, b, c] = P(c | a, b)
```

The `+1` is Laplace smoothing. It prevents zero probabilities, which is important because speculative decoding compares the target and draft distributions using a ratio:

```text
target_prob[token] / draft_prob[token]
```

If the draft probability is zero, the math becomes numerically awkward. Smoothing keeps every token legal.

## Probabilities

The most important method is `get_probs`. It returns the exact distribution that the draft model will sample from.

```python
def get_probs(self, prev_token_id, token_id, temperature=1.0):
    if prev_token_id is None:
        probs = self.fallback_bigram.get_probs(token_id)
    else:
        prev_token_id = int(prev_token_id)
        token_id = int(token_id)

        if (
            self.fallback_bigram is not None
            and self.context_counts[prev_token_id, token_id] < self.min_context_count
        ):
            probs = self.fallback_bigram.get_probs(token_id)
        else:
            probs = self.probs[prev_token_id, token_id]

    if temperature == 1.0:
        return probs

    scaled = probs.clamp_min(1e-12).pow(1.0 / temperature)
    return scaled / scaled.sum()
```

The normal trigram path is just:

```python
probs = self.probs[prev_token_id, token_id]
```

The shape goes from:

```text
self.probs                (vocab_size, vocab_size, vocab_size)
self.probs[prev, current] (vocab_size,)
```

That one row is the full distribution over the next token.

There are two fallback cases.

First, sometimes we only have one token of context:

```python
if prev_token_id is None:
    probs = self.fallback_bigram.get_probs(token_id)
```

A trigram needs `(prev, current)`. At the very beginning of generation, we may only have `current`. In that case, the natural thing is to fall back to the bigram:

```text
P(next | current)
```

Second, sometimes the trigram context exists but is too rare:

```python
self.context_counts[prev_token_id, token_id] < self.min_context_count
```

This is subtle but important. Laplace smoothing makes every row valid, but it does not make every row good. If a context never appeared in the training data, its smoothed trigram row is mostly just a weak prior. A bigram row may be much more useful because it has seen the current token many times.

So the policy is:

```text
If the trigram context has enough evidence, use P(next | prev, current).
Otherwise, fall back to P(next | current).
```

This keeps the trigram sharp when it has evidence and conservative when it does not.

### Temperature

The draft model also supports temperature:

```python
scaled = probs.clamp_min(1e-12).pow(1.0 / temperature)
return scaled / scaled.sum()
```

We apply temperature after choosing either the trigram row or the fallback bigram row. This keeps the code compatible with the original `BigramDraftModel`, whose `get_probs` method only takes `token_id`.

Temperature changes the shape of a probability distribution:

```text
temperature < 1.0  -> sharper, more greedy
temperature = 1.0  -> unchanged
temperature > 1.0  -> flatter, more random
```

The key rule in speculative decoding is that the returned `draft_probs[i]` must match the distribution that actually produced `candidates[i]`.

If we sample a candidate from the temperature-scaled distribution but return the unscaled distribution, the accept/reject math is wrong. The algorithm no longer preserves the target model distribution exactly.

The `clamp_min(1e-12)` is a small numerical guard. In theory, Laplace smoothing should already make probabilities non-zero. In practice, after device moves, dtype changes, or future modifications, it is cheap to protect the exponentiation from exact zeros.

## Sampling

Sampling is now just a wrapper around `get_probs`:

```python
def sample(self, prev_token_id, token_id, *, temperature=1.0, generator=None):
    probs = self.get_probs(prev_token_id, token_id, temperature=temperature)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return next_token, probs
```

Notice that we return both:

```text
next_token -> the sampled candidate
probs      -> the distribution that produced it
```

That second return value is not optional. The speculative accept/reject step needs the draft probability of the sampled token.

## Drafting Candidates

The draft phase changes slightly because the trigram needs two tokens of rolling context.

```python
def draft_tokens(draft_model, prev_token, current_token, K, temperature=1.0):
    candidates = []
    draft_probs = []

    a = prev_token
    b = current_token

    for _ in range(K):
        next_token, probs = draft_model.sample(a, b, temperature=temperature)
        candidates.append(next_token)
        draft_probs.append(probs)

        a, b = b, next_token

    return candidates, draft_probs
```

The chain looks like:

```text
(prev_token, current_token)  -> candidate_0
(current_token, candidate_0) -> candidate_1
(candidate_0, candidate_1)   -> candidate_2
```

This rolling context is internal to the draft model. The target model does not need to know that the draft used a trigram.

## The Speculative Loop

The target verification step stays almost the same as before.

This is the part that is easy to get wrong. Because the draft model uses `prev_token`, it is tempting to feed `prev_token` into the target model during verification too.

Do not do that.

The target model already has everything before `current_token` in its KV cache. If the cache ends after `prev_token`, then verification starts at `current_token`.

```text
wrong:
[prev_token, current_token, candidate_0, candidate_1, ...]

right:
[current_token, candidate_0, candidate_1, ...]
```

The rule is:

```text
verification input starts exactly where the KV cache ends
```

Here is the loop with the trigram state added:

```python
@torch.no_grad()
def speculative_generate(target_model, draft_model, prompt_tokens, max_new_tokens, K=4):
    target_model.eval()
    generated = []

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
    logits, _, past_kvs = target_model(input_ids, pos=positions)

    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(current_token)

    prev_token = prompt_tokens[-1]

    while len(generated) < max_new_tokens:
        cache_len = past_kvs[0][0][0].shape[1]
        k = min(K, max_new_tokens - len(generated))

        candidates, draft_probs = draft_tokens(
            draft_model,
            prev_token,
            current_token,
            k,
        )

        target_probs, new_kvs = verify_candidates(
            target_model,
            current_token,
            candidates,
            past_kvs,
        )

        accepted = accept_reject(candidates, draft_probs, target_probs)
        past_kvs = trim_kv_cache(new_kvs, len(accepted), cache_len)

        generated.extend(accepted)

        for token in accepted:
            prev_token, current_token = current_token, token

    return generated[:max_new_tokens]
```

Only two pieces of state are new:

```python
prev_token = prompt_tokens[-1]
```

and:

```python
for token in accepted:
    prev_token, current_token = current_token, token
```

The draft model needs the last two emitted tokens. After every accepted token, we roll the pair forward.

One more detail: if your `accept_reject` function can emit a resampled token after rejection, make sure that token is included in `accepted` or otherwise used to update `(prev_token, current_token)`. The next draft context must be based on the actual output sequence, not the rejected candidate chain.

## Gotchas

The main gotchas are all about keeping the probability bookkeeping exact.

**The draft distribution must match the sampled token.** If `candidate_i` was sampled from `P(next | a, b)`, then `draft_probs[i]` must be that same row, with the same temperature. Recomputing with the wrong rolling context breaks rejection sampling.

**Do not feed `prev_token` into verification.** It is already represented in the target model's KV cache. Verification still starts from `[current_token] + candidates`.

**Sparse trigram rows can hurt.** A smoothed trigram row from a context seen once may be worse than a bigram row seen thousands of times. This is why `min_context_count` exists.

**Dense trigram storage only works because the vocabulary is tiny.** For character-level nanoGPT, `(65, 65, 65)` is fine. For a 50k-token LLM vocabulary, a dense trigram tensor is impossible. Production n-gram proposers use sparse maps, suffix arrays, or retrieval-style methods over observed spans.

**State must update after every emitted token.** The draft context is always the last two real output tokens. If rejection produces a resampled token, that token becomes part of the real sequence.

## Summary

The trigram upgrade is a small change with a useful lesson.

The target model is unchanged. Verification is unchanged. The KV cache rule is unchanged.

Only the draft model got a little more context:

```text
bigram:  P(next | current)
trigram: P(next | previous, current)
```

That extra token of context should make the draft guesses more realistic, which should increase acceptance rate and improve throughput. At the same time, fallback to bigram keeps the model from trusting rare trigram contexts too much.

The whole point is not that trigram is a great language model. It is not. The point is that speculative decoding rewards cheap models that are just good enough to guess ahead.
