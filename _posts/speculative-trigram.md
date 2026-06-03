---
layout: post
title: "Adding Trigram to Speculative Decoding"
date: 2026-05-29
---

Previously, we implemented speculative decoding using a bigram draft model for our nanoGPT implementation. 

In this article, we will upgrade from a bigram draft model to a trigram to see how much it improves performance. The goal is to improve the draft acceptance rate now that there is more context being fed into predictions, which should also improve throughput and lower latency. 

## Why Trigram?

A trigram allows us to speculate the next token based on both the current token and the previous token. 

## Using Bigram as Fallback

The fallback can reuse the existing `BigramDraftModel`, or the trigram model can build its
own bigram table. Reusing the existing class is the smallest implementation.

```python
def get_probs(self, prev_token_id, token_id, temperature=1.0):
    prev_token_id = int(prev_token_id)
    token_id = int(token_id)

    # Tune this threshold. 1 means "use trigram if the context ever appeared".
    min_context_count = 2
    if self.context_counts[prev_token_id, token_id] < min_context_count:
        probs = self.fallback_bigram.get_probs(token_id, temperature=temperature)
    else:
        probs = self.probs[prev_token_id, token_id]

    if temperature == 1.0:
        return probs

    scaled = probs.clamp_min(1e-12).pow(1.0 / temperature)
    return scaled / scaled.sum()

why do we need temperature: The draft model needs to support temperature because the rejection sampler requires that draft_probs[i] exactly matches the distribution that produced candidates[i]. If you sample at temperature 0.8 but return unscaled probs, the acceptance math is wrong and the target distribution is no longer preserved.

why do we need to clamp_min. what is the behavior of this?

def sample(self, prev_token_id, token_id, *, temperature=1.0, generator=None):
    probs = self.get_probs(prev_token_id, token_id, temperature=temperature)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return next_token, probs
```

**Hint:** make `BigramDraftModel.get_probs()` accept `temperature` first if the benchmark
version already does. In `nanogpt-spec-decode.py`, it currently does not, so either add the
optional parameter there or keep the fallback call temperature-free in that file.

## TrigramDraftModel Class

The first thing to do is to add the new class representing the `TrigramDraftModel`.

Recall what the bigram draft model was doing. It had a table of shape `(vocab_size, vocab_size)`, where row `a` stored the empirical distribution:

```text
P(next_token | current_token = a)
```

So if the current token was `"t"`, the bigram table looked at the row for `"t"` and sampled whatever usually follows `"t"` in the training data. This is already useful, but it is also a very blunt instrument. The token `"t"` can be followed by `"h"` in `"the"`, but it can also be followed by `"o"` in `"to"`, `" "` after the end of a word, or many other things depending on the previous character. A bigram has no way to distinguish those situations because it only remembers one token of context.

A trigram is the next tiny step up the ladder. Instead of conditioning on one token, it conditions on two:

```text
P(next_token = c | previous_token = a, current_token = b)
```

Now the model can learn that `"t"` after a space behaves differently from `"t"` after `"h"`. In character-level Shakespeare, this matters a lot. The context `" t"` is very likely to continue with `"h"` or `"o"`, while `"ht"` is probably rare or weird. The table is still extremely small compared to a transformer, but it is noticeably more informed than the bigram table.

Concretely, the trigram model stores a 3D tensor:

```text
counts[a, b, c] = how many times token c followed the pair (a, b)
```

If `vocab_size = 65`, then the full table has shape `(65, 65, 65)`, or about 275k entries. That sounds big compared to the bigram's 4k entries, but it is still tiny in absolute terms. It is just a lookup table. No attention, no MLP, no matmuls, no layers. For speculative decoding, this is exactly the kind of thing we want: a draft model that is cheap enough to run several times while the expensive target model runs once.

```python
    def __init__(self, token_ids, vocab_size, device, fallback_bigram=None):
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
```

Let's go line by line.

```python
counts = torch.zeros(vocab_size, vocab_size, vocab_size, dtype=torch.float32)
```

This creates the raw frequency table. The three axes correspond to `(previous_token, current_token, next_token)`. Said another way, `counts[a, b]` is a full vector of length `vocab_size`, and that vector stores the next-token histogram after seeing the two-token context `(a, b)`.

So if we had a toy vocabulary:

```text
0 = " "
1 = "t"
2 = "h"
3 = "e"
4 = "o"
```

Then `counts[0, 1, 2]` would mean: how many times did `"h"` appear after the context `" t"`?

```python
ids = torch.as_tensor(token_ids, dtype=torch.long).flatten().cpu()
```

Here we normalize the training data into a simple 1D tensor of token IDs on CPU. This is intentional. Building the table is a one-time preprocessing step, and the loop below is ordinary Python bookkeeping. There is no reason to involve the GPU while incrementing individual count cells one by one. The finished probability table gets moved to `device` later.

```python
for a, b, c in zip(ids[:-2].tolist(), ids[1:-1].tolist(), ids[2:].tolist()):
    if 0 <= a < vocab_size and 0 <= b < vocab_size and 0 <= c < vocab_size:
        counts[a, b, c] += 1.0
```

This is the entire "training loop" for the trigram. We slide a length-3 window over the corpus:

```text
tokens:  [x0, x1, x2, x3, x4, ...]
window:  (x0, x1, x2)
window:      (x1, x2, x3)
window:          (x2, x3, x4)
```

For every triple `(a, b, c)`, we increment the count that says: after seeing `(a, b)`, we observed `c` as the next token.

The bounds check is defensive. In a clean character-level setup every token ID should already be inside `[0, vocab_size)`, but it is cheap to guard against bad data. If some malformed token sneaks in, we skip it instead of indexing outside the tensor.

```python
self.context_counts = counts.sum(dim=-1)
```

Before smoothing, we save how much real evidence we have for each two-token context. `counts.sum(dim=-1)` collapses the next-token axis, producing a `(vocab_size, vocab_size)` table:

```text
context_counts[a, b] = total number of times context (a, b) appeared
```

This is useful because trigram tables are sparse. Many token pairs will appear frequently, but many others will barely appear or never appear at all. If the context `(a, b)` appeared 500 times, we can trust the trigram distribution. If it appeared 0 times, the trigram distribution after smoothing is basically just a uniform-ish guess. In that case, it can be better to fall back to a bigram.

```python
counts += 1.0
```

This is Laplace smoothing. We add one fake observation to every possible transition. Without smoothing, unseen transitions would get probability 0. That is dangerous in speculative decoding because the acceptance ratio uses:

```text
target_prob[token] / draft_prob[token]
```

If `draft_prob[token]` is zero, the math becomes annoying and numerically fragile. More importantly, `torch.multinomial` also needs a valid non-zero distribution to sample from. Smoothing gives every token at least a tiny chance.

There is a trade-off. Smoothing makes the distribution less sharp, especially for rare contexts. But for a draft model, this is a good trade. The draft model does not need to be perfect. It needs to be cheap, plausible, and mathematically well-behaved.

```python
self.probs = counts / counts.sum(dim=-1, keepdim=True)
```

Now we normalize the counts into probabilities. For every context `(a, b)`, the vector `counts[a, b, :]` is divided by its total count, producing a categorical distribution over the next token:

```text
self.probs[a, b, c] = P(c | a, b)
```

After this line, `self.probs[a, b]` sums to 1 across the vocabulary.

```python
self.probs = self.probs.to(device)
self.context_counts = self.context_counts.to(device)
self.fallback_bigram = fallback_bigram
```

Finally, we move the lookup tables to the same device as the rest of inference. The draft model is just indexing and sampling, so keeping `self.probs` on the target device avoids little CPU/GPU transfers inside the decoding loop. We also store an optional `fallback_bigram`, which lets the trigram gracefully degrade to the simpler model when it does not have enough evidence for a two-token context.

### Get probabilities

The main method we need from the draft model is `get_probs`. Given the two most recent tokens, it should return a `(vocab_size,)` probability distribution for the next token.

Conceptually:

```text
previous token: a
current token:  b
return:         P(next | a, b)
```

In code, it looks like this:

```python
    def get_probs(self, prev_token_id, token_id):
        if prev_token_id is None:
            return self.fallback_bigram.get_probs(token_id)

        if self.context_counts[prev_token_id, token_id] == 0 and self.fallback_bigram is not None:
            return self.fallback_bigram.get_probs(token_id)

        return self.probs[prev_token_id, token_id]
```

The happy path is the final line:

```python
return self.probs[prev_token_id, token_id]
```

This is the trigram lookup. We index into the first two dimensions of the table using the two-token context, and the result is the full next-token distribution. The shape is:

```text
self.probs                  (vocab_size, vocab_size, vocab_size)
self.probs[prev, current]   (vocab_size,)
```

That returned vector is exactly what `torch.multinomial` needs for sampling, and it is exactly what the speculative decoding accept/reject step needs later as the draft distribution `q`.

The first edge case is the beginning of generation:

```python
if prev_token_id is None:
    return self.fallback_bigram.get_probs(token_id)
```

A trigram needs two tokens of context. But at the very beginning of a sequence, or sometimes at the boundary of our speculative loop, we may only have one token available. If there is no previous token, we cannot form `(prev, current)`. So we fall back to the bigram distribution:

```text
P(next | current)
```

This is not a hack so much as the natural hierarchy of n-gram models:

```text
trigram if we have 2 tokens of context
bigram  if we have 1 token of context
unigram if we have 0 tokens of context
```

The second edge case is an unseen context:

```python
if self.context_counts[prev_token_id, token_id] == 0 and self.fallback_bigram is not None:
    return self.fallback_bigram.get_probs(token_id)
```

Remember that `context_counts` was saved before smoothing. So if `context_counts[prev, current] == 0`, it means the pair `(prev, current)` never occurred in the training data. Yes, `self.probs[prev, current]` is technically still valid because Laplace smoothing made it non-zero. But it is not very informative. It is mostly the smoothing prior.

In that situation, the bigram can actually be better because it may have seen `current_token` many times, even if it never saw the exact pair `(prev_token, current_token)`. For example, maybe the pair `"Qz"` never appeared, but `"z"` did appear enough times for the bigram row to know something about what usually follows `"z"`.

So the fallback policy is:

```text
If the trigram context has evidence, use it.
If the trigram context has no evidence, ask the bigram.
```

This gives us the best of both worlds. Frequent contexts get sharper, more local predictions. Rare or impossible contexts do not collapse into a meaningless smoothed trigram row.

This matters for speculative decoding because acceptance depends on how close the draft distribution is to the target distribution. If the trigram guesses more like the target model, more proposed tokens get accepted, and each target forward pass buys us more generated tokens. But if the trigram gets overconfident in bad places, acceptance falls. The fallback keeps the draft model boring in exactly the places where boring is good.

The returned `draft_probs[i]` still has shape `(vocab_size,)`, so `accept_reject()` does not
need to know whether the distribution came from a bigram or trigram table.

### Why Not Just Smooth Everything?

Laplace smoothing makes every trigram context legal, but it also makes unseen two-token
contexts nearly uniform. A uniform draft is safe, but weak. For sparse contexts, a bigram
fallback is usually better:

```text
If (prev, current) was seen often: use P(next | prev, current)
If (prev, current) was rare/unseen: use P(next | current)
```

This keeps the trigram model from becoming overconfident on tiny counts.

### Sampling

For the sampling, we simply call the `get_probs` function as we explained above and then use `torch.multinomial` to sample the next item from the distribution and return that along with the probabilities.

## Speculative Loop

In the loop, we make several modifictions

```python
@torch.no_grad()
def speculative_generate(target_model, draft_model, prompt_tokens, max_new_tokens, K=4):
    target_model.eval()
    generated = []

    # 1. Prefill: run the full prompt through the target model
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), device=device).unsqueeze(0)
    logits, _, past_kvs = target_model(input_ids, pos=positions)    

    # Sample the first token from the prefill output
    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(current_token)    

    prev_token = prompt_tokens[-1] # 

    # 2. Speculative decode loop
    while len(generated) < max_new_tokens:
        cache_len = past_kvs[0][0][0].shape[1]  # current KV cache length

        # How many tokens to speculate (don't overshoot max_new_tokens)
        k = min(K, max_new_tokens - len(generated))

        # DRAFT
        candidates, draft_probs = draft_tokens(draft_model, prev_token, current_token, k)

        # VERIFY
        target_probs, new_kvs = verify_candidates(
            target_model, current_token, candidates, past_kvs
        )

        # ACCEPT/REJECT
        accepted = accept_reject(candidates, draft_probs, target_probs)

        # TRIM KV CACHE (keep only accepted tokens)
        past_kvs = trim_kv_cache(new_kvs, len(accepted), cache_len)

        # Update state
        generated.extend(accepted)

        # Update prev_token and current_token for next iteration
        for tok in accepted:
            prev_token, current_token = current_token, tok

    return generated[:max_new_tokens] 
```

In the loop, we simply create a variable to hold the value of the previous token in `prompt_tokens`. Then, we call the `draft_tokens` function with the previous token passed in. 

At the end of the loop, we take the accepted tokens, and update the `prev_token` and `current_token` variables in anticipation for the next iteration.

**Things to be aware**:

## What the draft model needs

The trigram draft needs two tokens of context to *propose* the next candidate:

```
prev_token, current_token  →  candidate_0
current_token, candidate_0 →  candidate_1
...
```

That's purely internal to `draft_tokens`. The target model never sees this.

---

## What the target model needs

The target model runs a **single forward pass** over:

```
[current_token, candidate_0, candidate_1, ..., candidate_K-1]
```

Its KV cache already holds everything up to and including `prev_token`. So when you feed it `current_token`, it effectively "sees" `(prev_token, current_token)` through the cached attention — you don't need to re-feed `prev_token`.

---

## The mental slip

The natural (wrong) instinct when upgrading to trigram is: *"the draft uses `prev_token`, so maybe I need to also feed it into verification."*

**No.** `prev_token` is already in the KV cache. Feeding it again would shift the sequence:

```
# WRONG — shifts everything by 1, KV cache alignment breaks
verify_input = [prev_token, current_token, candidate_0, ...]

# CORRECT — KV cache already covers everything before current_token
verify_input = [current_token, candidate_0, ...]
```

The rule is simple: **the verification input starts from where the KV cache ends.** The KV cache ends after `prev_token`, so verification starts at `current_token`. The extra context `prev_token` that the *draft* needed is irrelevant to the *verifier* — it's already cached.

## Gotchas

1. **The draft distribution must match the sampled token.** If candidate `c_i` was sampled
   from `P(next | a, b)`, then `draft_probs[i]` must be that same row. Recomputing it with
   the wrong rolling context breaks rejection sampling.

2. **Do not feed `prev_token` into verification.** It is already in the target KV cache.
   Verification still starts from `[current_token] + candidates`.

3. **Sparse trigram rows can hurt.** A smoothed row from a context seen once may be worse
   than a bigram row seen thousands of times. Use `min_context_count`.

4. **Dense trigram storage is okay here, but not always.** Character vocab is tiny. For a
   50k-token LLM vocabulary, a dense trigram tensor would be impossible. Production n-gram
   proposers use sparse maps, suffix arrays, or retrieval over observed prompt/document
   spans.

5. **Keep probabilities strictly non-zero.** Rejection sampling divides by `q`. Smoothing,
   fallback, and `clamp_min(1e-12)` in accept/reject are all useful guardrails.

6. **Update state after every emitted token.** A rejection can emit a resampled token that
   was not in the candidate chain. The next draft context must use the actual emitted token,
   not the rejected candidate.


Why do we need temperature setting?




