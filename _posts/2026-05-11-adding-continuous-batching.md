---
layout: post
title: "Adding Continuous Batching to NanoGPT"
date: 2026-05-11
---

In a [previous post](/blog/2026/05/10/adding-kv-cache-to-nanogpt), I added a KV cache to NanoGPT — Karpathy's from-scratch GPT trained on Shakespeare. That let us avoid recomputing past keys and values, cutting decode from O(n²) to O(n) and giving us a moderate speedup in tokens per second. But we were still serving one request at a time. `B` in the batch dimension was always 1 during generation.

A real inference server doesn't do that. It packs multiple requests into a single batch and lets them arrive and leave independently — some requests are still generating while new ones join and finished ones depart. This is called continuous batching, and in this post I'm going to build it from scratch on top of our toy model.

The problem is straightforward: without batching, the GPU sits mostly idle. One request at a time means one row in the batch dimension, which is nowhere near enough to saturate the hardware. Worse, different requests have different sequence lengths, so naive static batching forces short requests to wait for long ones. Continuous batching solves both: requests flow in and out of the batch every decode step, and the GPU stays busy.

## Moving the KV cache out of the model

The current `generate_with_cache` function has the KV cache baked into the `Head` class as a single tensor for the whole batch. That worked fine for one request, but now we need multiple requests with different sequence lengths sharing the same forward pass. The cache has to move out of the model and into each request.

Each request carries the state it needs: the original prompt tokens, how many tokens it still wants to generate, and the tokens produced so far. The `status` field tracks its lifecycle — `"waiting"` means it hasn't been prefilled yet, `"active"` means it's in the decode batch, and `"done"` means it's finished. Most importantly, each request owns its own KV cache: a dictionary keyed by `(layer_idx, head_idx)` holding the cached key and value tensors for that request's history. This is the key architectural change — the cache moves from inside the model to inside the request.

```python

@dataclass
class Request:
    """Each in-flight generation carries its own state and KV cache."""
    id: int
    prompt_tokens: List[int]          # the original encoded prompt
    max_new_tokens: int               # how many tokens this request wants
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"           # "waiting" -> "active" -> "done"

    # Per-request KV cache, keyed by (layer_idx, head_idx)
    # Each value is a (key_tensor, value_tensor) tuple of shape (1, T_i, head_size)
    # T_i grows by 1 each decode step — different requests have different T_i
    kv_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    @property
    def tokens_so_far(self) -> List[int]:
        """Full sequence: prompt + everything generated."""
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_done(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    def clear_cache(self):
        self.kv_cache.clear()
```

The old `Head.key_cache` had shape `(B, T, hs)` — one contiguous tensor for the whole batch. Now each request gets its own `(1, T_i, hs)` tensor, and `T_i` can differ across requests. The question becomes: how do you stitch these together for a batched forward pass when the sequence lengths don't match? We'll get to that.

## Per-request generation

Before tackling the full scheduler, it helps to build a simpler stepping stone: a function that generates for a single `Request` object, but with the cache living on the request instead of inside the model.

```python
# ── Per-Request generation ────────────────────────────────────────────────────
def generate_request(model, request: Request):
    """
    Generate for a single Request object.
    The KV cache lives on the Request, not inside the model.

    This is the building block for the continuous batching scheduler.
    Each request independently owns its cache, so different requests
    can have different sequence lengths and lifetimes.
    """
    model.eval()
    with torch.no_grad():
        # Convert prompt to tensor
        prompt = torch.tensor(
            [request.prompt_tokens], dtype=torch.long, device=device
        )  # (1, T_prompt)

        # ── Prefill ──
        logits, _, new_kvs = model(prompt)

        # Store the cache on the request object
        # new_kvs[layer_idx][head_idx] = (key_tensor, value_tensor)
        for layer_idx, block_kv in enumerate(new_kvs):
            for head_idx, (k, v) in enumerate(block_kv):
                request.kv_cache[(layer_idx, head_idx)] = (k, v)

        request.status = "active"

        # ── Decode loop ──
        while not request.is_done:
            logits = logits[:, -1, :]           # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)

            request.generated_tokens.append(idx_next.item())

            if request.is_done:
                break

            # Rebuild past_kvs from request's per-request cache
            past_kvs = []
            for layer_idx in range(n_layer):
                block_kv = []
                for head_idx in range(n_head):
                    block_kv.append(request.kv_cache[(layer_idx, head_idx)])
                past_kvs.append(block_kv)

            curr_pos = torch.tensor(
                [[len(request.tokens_so_far) - 1]], device=device
            )  # (1, 1)
            logits, _, new_kvs = model(idx_next, pos=curr_pos, past_kvs=past_kvs)

            # Update the request's cache with the new K/V
            for layer_idx, block_kv in enumerate(new_kvs):
                for head_idx, (k, v) in enumerate(block_kv):
                    request.kv_cache[(layer_idx, head_idx)] = (k, v)

        request.status = "done"

```

**Prefill.** We convert the prompt into a tensor of shape `(1, T_prompt)` and run it through the model in one shot. The model returns logits and a fresh KV cache for every layer and head. We store that cache on the request object — keyed by `(layer_idx, head_idx)` — so the request now owns all the cached state it needs for future decode steps. Since we have multiple layers and each layer has multiple heads, that tuple is the natural index into the 16 separate K/V caches per request (4 layers × 4 heads in our model).

**Decode loop.** Each iteration samples one token from the last position's logits, appends it to the request, then feeds just that single token back through the model along with the cached past. The position index `curr_pos` tells the model where this token sits in the full sequence — without it, the model would always use position 0 for every decode step, producing garbage because the positional embeddings would be wrong and the attention scores would be computed with corrupted signals.

**Cache update.** After each forward pass, the model returns updated K/V tensors (the old cache with the new token's key and value concatenated). We write these back to the request, overwriting the old cache.

## Packing caches into a batch

But there's a problem. Different requests have different KV cache lengths. Request A might have 50 cached positions while Request B has 15. The model expects a uniform tensor of shape `(B, T, hs)`, not a ragged list. We need to pack all the per-request caches into a single batch before the forward pass.

The solution is left-padding. We pad shorter caches with zeros on the left so they all match the length of the longest cache. New tokens always land at the right edge, so the padding stays inert on the left. During the forward pass, an attention mask tells the model to ignore the padding positions — they get filled with `-inf` before softmax, which drives their attention weight to zero.

```python

def assemble_batch_cache(requests):
    """
    Gather per-request KV caches into batched tensors.
    LEFT-pads shorter caches so new tokens always land at the right edge.

    Big problem: You have 3 active requests. Each owns its own KV cache. You need to feed them to the model as one
    batched tensor. But their caches have different lengths:

    Returns:
        past_kvs:    batched cache structure  [layer][head] = (B, T_max, hs)
        attn_mask:   (B, 1, T_max) bool — True = valid, False = padding
        pad_lengths: list of int — how many pad positions per request (for disassembly)
    """

    B = len(requests)
    lengths = [req.kv_cache[(0, 0)][0].shape[1] for req in requests]
    max_t = max(lengths)

    pad_lengths = [max_t - t for t in lengths] # pad lengths for every position in t

    attn_mask = torch.zeros(B, 1, max_t, device=device, dtype=torch.bool)

    for i, pad in enumerate(pad_lengths):
        attn_mask[i, 0, pad:] = True

    past_kvs = []

    for layer_idx in range(n_layer):
        block_kv = []

        for head_idx in range(n_head):
            keys, values = [], []

            for i, req in enumerate(requests):
                k, v = req.kv_cache[(layer_idx, head_idx)]
                if pad_lengths[i] > 0:
                    hs = k.shape[2]
                    pad = torch.zeros(1, pad_lengths[i], hs, device=device)
                    k = torch.cat([pad, k], dim=1)
                    v = torch.cat([pad, v], dim=1)

                keys.append(k)
                values.append(v)

            block_kv.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        past_kvs.append(block_kv)

    return past_kvs, attn_mask, pad_lengths

```

We want to access the KV cache of every layer and head for every request. Then, we want to pad the left side with zeroes using `torch.zeros`. Then, we iterate over every layer and head, and for each layer and head, we iterate over every request and append the key and value tensors to the `keys` and `values` lists. Finally, we concatenate the `keys` and `values` lists to get the batched KV cache. The values that we return are the batched KV cache, the attention mask, and the pad lengths. The attention mask will be passed into the model as we will see later, and the pad_lengths will be used to disassemble the batch cache, which will be in the next section. 

## Unpacking after the forward pass

After the model's forward pass, the KV cache comes back as one big batched tensor of shape `(B, T_max + 1, hs)` — the old cache plus the new token's entry. But that batched format still includes the left-padding we added during assembly. We need to strip it back out so each request gets only its own real history.

```python

def disassemble_batch_cache(requests, new_kvs, pad_lengths):
    """
    Scatter batched KV cache back to per-request storage.
    After Head's torch.cat, each row is (T_max + 1) — strip the left-padding.
    """
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):
                pad = pad_lengths[i]
                req.kv_cache[(layer_idx, head_idx)] = (
                    batched_k[i : i + 1, pad:, :],      # (1, T_i + 1, hs)
                    batched_v[i : i + 1, pad:, :],
                )

```

After the model's forward pass, the KV cache comes back as one big batched tensor of shape (B, T_max + 1, hs) — but that batched format includes the left-padding we added during assembly, which is meaningless filler that doesn't belong to any request's real history. 

The disassemble_batch_cache function reverses the assembly step by slicing each request's row out of the batch and stripping the left-padding using the saved pad_lengths, so each request gets back only its own real KV entries of shape (1, T_i + 1, hs). Without this step, the padding zeros would accumulate in each request's cache on every decode iteration, steadily corrupting attention scores and eventually causing the model to attend over garbage positions. In short, assemble_batch_cache packs requests together for an efficient batched forward pass, and disassemble_batch_cache unpacks them so each request's cache stays clean and correctly sized for the next step. 

Now, we are ready to tackle the most difficult part of this, which is the actual continuous batching loop itself!

## Continuous Batching Loop

The current loop runs a fixed number of steps for one request. The continuous batching loop looks more like:

```
while there are active requests OR the waiting queue is non-empty:
    1. Check the waiting queue — can any new requests join the batch?
    2. Build the input tensor from ALL active requests (each contributes 1 token)
    3. Forward pass → get logits for all active requests at once
    4. Sample next token for each request
    5. Check: did any request hit its max_new_tokens? → remove it, emit its result
    6. Go to 1
```

The key insight: steps 1 and 5 happen every iteration, not just at the start and end. Requests flow in and out continuously.

The function takes three arguments: the trained model, a `request_queue` of `(arrival_step, Request)` pairs sorted by arrival time, and a `max_batch_size` cap. It returns the list of completed requests.

The outer `while` loop iterates as long as there are requests to serve. A `queue_idx` variable tracks which request we're pulling next, and a `step` counter tracks the current time step — this matters because requests can arrive at different times.

**Prefill.** At the top of each iteration, we check whether new requests have arrived (their `arrival_step` ≤ the current `step`) and whether there's room in the batch. For each new arrival, we run the full prompt through the model, store the resulting KV cache on the request, and sample its first token:

```python
    for li, bkv in enumerate(new_kvs):
        for hi, (k, v) in enumerate(bkv):
            req.kv_cache[(li, hi)] = (k, v)

    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)

    req.generated_tokens.append(idx_next.item())
    req.status = "active"
    req._last_token = idx_next

    if req.is_done:
        req.status = "done"
        completed_requests.append(req)
    else:
        active_requests.append(req)

    queue_idx += 1

    print(f"  [step {step}] Admitted request {req.id} "
            f"(prompt={len(req.prompt_tokens)}, "
            f"max_new={req.max_new_tokens})")
```
This code handles what happens immediately after a new request is prefilled. First, it stores the freshly computed KV cache from the prefill into the request's per-request dictionary, keyed by (layer_index, head_index), so that the request now owns all the cached key/value state it needs for future decode steps. 

Then it samples the first generated token from the prefill logits — taking only the last position's logits, converting to probabilities via softmax, and drawing a token via multinomial sampling and stashes both the token and the raw tensor (_last_token) on the request so the decode loop knows what to feed in next. 

Finally, it checks whether that single token was enough to satisfy max_new_tokens (edge case where max_new_tokens=1); if so, the request is immediately marked done, otherwise it's pushed into active_requests to join the shared decode batch on the next iteration.

Now, the main logic for the decode step:

```python

        if not active_requests:
            step += 1
            continue

        B_active = len(active_requests)

        batch_tokens = torch.cat([req._last_token for req in active_requests])

        batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in active_requests], device=device)

        past_kvs, attn_mask, pad_lengths = assemble_batch_cache(active_requests)

        logits, _, new_kvs = model(
            batch_tokens,
            pos=batch_positions,
            past_kvs=past_kvs,
            attn_mask=attn_mask
        )

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        disassemble_batch_cache(active_requests, new_kvs, pad_lengths)

        for i, req in enumerate(active_requests):
            req.generated_tokens.append(idx_next[i].item())
            req._last_token = idx_next[i : i + 1]

        still_active = []
        for req in active_requests:
            if req.is_done:
                req.status = "done"
                completed_requests.append(req)
                print(f"  [step {step}] Completed request {req.id} "
                        f"({req.num_generated} tokens)")

            else:
                still_active.append(req)

        active_requests = still_active

        step += 1

return completed_requests
```

Let's walk through each piece:

**Batch assembly.** We gather the last generated token from each active request into a single tensor of shape `(B, 1)`, and similarly stack their position indices. Then `assemble_batch_cache` pads all the per-request KV caches to the same length and produces one batched cache the model can consume in a single forward pass.

**Forward pass.** The model sees all active requests simultaneously — it doesn't know or care that they are independent sequences at different stages of generation. It returns logits for every request and updated KV caches that now include the new token's key and value.

**Model forward pass.** This is the same as before. We pass the batch_tokens, batch_positions, past_kvs, and attn_mask to the model, and get the logits, loss, and new_kvs. We will then take the new kv cache from the model and store it in the request's kv cache using the disassemble_batch_cache function. In addition, we have to add the token's that were newly generated for the active requests to the request's generated tokens list.

**Completed requests.** Finally, we check if any of the requests in the active requests list are done. If they are, we remove them from the active requests list and add them to the completed requests list, and then we increment the time step.

## Training

The training loop is the same as the previous post — nothing about continuous batching changes the training procedure. The model is identical; only the inference-time serving logic is different.

```python
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss, _ = model(xb, yb)  # _ discards the cache during training
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Quick sanity check with the original no-cache generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
```

The loss goes from ~4.2 (random) down to ~1.65 over 5,000 steps. The model learns Shakespeare's character-level patterns — not well enough to fool anyone, but well enough to test our batching logic:

```text
0.209729 M parameters
step 0: train loss 4.1959, val loss 4.1962
step 100: train loss 2.6229, val loss 2.6166
step 200: train loss 2.4555, val loss 2.4488
step 300: train loss 2.3810, val loss 2.3928
step 400: train loss 2.3202, val loss 2.3223
step 500: train loss 2.2364, val loss 2.2541
step 600: train loss 2.1812, val loss 2.2234
step 700: train loss 2.1326, val loss 2.1583
step 800: train loss 2.0932, val loss 2.1352
step 900: train loss 2.0499, val loss 2.0987
step 1000: train loss 2.0349, val loss 2.0819
step 1100: train loss 1.9994, val loss 2.0706
step 1200: train loss 1.9857, val loss 2.0726
step 1300: train loss 1.9638, val loss 2.0401
step 1400: train loss 1.9386, val loss 2.0296
step 1500: train loss 1.9028, val loss 1.9947
step 1600: train loss 1.8821, val loss 1.9914
step 1700: train loss 1.8785, val loss 1.9813
step 1800: train loss 1.8746, val loss 1.9878
step 1900: train loss 1.8451, val loss 1.9599
step 2000: train loss 1.8294, val loss 1.9525
step 2100: train loss 1.8333, val loss 1.9716
step 2200: train loss 1.8067, val loss 1.9342
step 2300: train loss 1.8009, val loss 1.9301
step 2400: train loss 1.7751, val loss 1.9097
step 2500: train loss 1.7753, val loss 1.9112
step 2600: train loss 1.7572, val loss 1.8935
step 2700: train loss 1.7596, val loss 1.9044
step 2800: train loss 1.7485, val loss 1.8818
step 2900: train loss 1.7349, val loss 1.8797
step 3000: train loss 1.7420, val loss 1.8873
step 3100: train loss 1.7331, val loss 1.8899
step 3200: train loss 1.7262, val loss 1.8832
step 3300: train loss 1.7109, val loss 1.8544
step 3400: train loss 1.7205, val loss 1.8765
step 3500: train loss 1.7111, val loss 1.8623
step 3600: train loss 1.7134, val loss 1.8649
step 3700: train loss 1.6996, val loss 1.8583
step 3800: train loss 1.6962, val loss 1.8547
step 3900: train loss 1.6856, val loss 1.8406
step 4000: train loss 1.6803, val loss 1.8274
...

And they brid write, is not the die;
Though we art One my day hangs:
Wart he us hath bury, dills ane away, my feanst,
Anzing heavens, tofultien me milen's
Whines is eye, hain latise, drovets, and Will
```

## Testing

Now we simulate three requests arriving at different times with different prompt lengths. If the scheduler is correct, requests 0 and 1 start together at step 0, request 2 joins later at step 3, and they all finish independently:

```python

# Simulate 3 requests arriving at different times with different lengths
request_queue = [
    (0,  Request(id=0, prompt_tokens=encode("O Romeo, "),     max_new_tokens=17)),
    (0,  Request(id=1, prompt_tokens=encode("To be or "),     max_new_tokens=22)),
    (3,  Request(id=2, prompt_tokens=encode("KING HENRY:\n"), max_new_tokens=15)),
]

print("=" * 60)
print("Continuous Batching — Simulated Arrivals")
print("=" * 60)

completed = continuous_batching_generate(model, request_queue, max_batch_size=4)

# Print results
for req in sorted(completed, key=lambda r: r.id):
    print(f"\n{'─'*40}")
    print(f"Request {req.id}  |  {req.num_generated} tokens  |  status: {req.status}")
    print(f"{'─'*40}")
    print(decode(req.tokens_so_far))

# Verify correctness
for req in completed:
    k, _ = req.kv_cache[(0, 0)]
    expected_T = len(req.prompt_tokens) + req.num_generated - 1
    assert k.shape[1] == expected_T, f"Req {req.id}: cache T={k.shape[1]}, expected {expected_T}"
    assert req.status == "done"
    assert req.num_generated == req.max_new_tokens

print("\n✓ All requests completed with correct cache shapes!")

```

And the output:

```text
============================================================
Continuous Batching — Simulated Arrivals
============================================================
  [step 0] Admitted request 0 (prompt=9, max_new=17)
  [step 0] Admitted request 1 (prompt=9, max_new=22)
  [step 3] Admitted request 2 (prompt=12, max_new=15)
  [step 15] Completed request 0 (17 tokens)
  [step 16] Completed request 2 (15 tokens)
  [step 20] Completed request 1 (22 tokens)

────────────────────────────────────────
Request 0  |  17 tokens  |  status: done
────────────────────────────────────────
O Romeo, moor and see-spea

────────────────────────────────────────
Request 1  |  22 tokens  |  status: done
────────────────────────────────────────
To be or head he true, let not 

────────────────────────────────────────
Request 2  |  15 tokens  |  status: done
────────────────────────────────────────
KING HENRY:
She thout to He

✓ All requests completed with correct cache shapes!
```

Request 0 finishes first at step 15, request 2 (which arrived late at step 3) finishes at step 16, and request 1 — which wanted the most tokens — finishes last at step 20. The batch size shrinks as requests complete. This is exactly the behavior we want.

## Odds and ends

**Could we use `torch.ones` instead of `torch.zeros` for the mask?** Either works. The mask starts as all-False (zeros), and we flip valid positions to True. Starting with all-True (ones) and flipping padding to False gives the exact same tensor. It's a convention.

**What does `~full_mask` mean?** `~` is Python's bitwise NOT operator. On a bool tensor, it flips every value: `[False, False, True, True, True]` becomes `[True, True, False, False, False]`. We use it so that "everywhere the mask says False (= padding), replace the attention score with `-inf`."

**Why `(layer, head)` as the cache key?** The model has 4 layers, each with 4 attention heads, and every single head computes its own independent K and V projections. Head 0 in layer 0 learns completely different projections than head 2 in layer 3 — they produce different K/V tensors even for the same input token. So we need one cache per head per layer: 4 × 4 = 16 separate K/V caches per request. The `(layer_idx, head_idx)` tuple is just the natural key:

```python
req.kv_cache[(0, 0)]  # Layer 0, Head 0 → (key_tensor, value_tensor)
req.kv_cache[(0, 1)]  # Layer 0, Head 1 → different K/V
req.kv_cache[(2, 3)]  # Layer 2, Head 3 → different K/V
```

Each value is a `(key_tensor, value_tensor)` tuple where both tensors have shape `(1, T, 16)` — one sequence, `T` cached positions, 16 dimensions per head.

**Why build a `still_active` list instead of removing from `active_requests` directly?** Deleting items from a list while iterating over it is a classic Python bug. When you remove an item, the indices shift and the loop skips the next element. Building a fresh list avoids the problem entirely. Within the scheduler's `while` loop, `active_requests` acts as global state: at the start of each iteration you read who's in the batch, at the end you overwrite it with whoever's still going.

## From here to production

Between this toy implementation and a production system like vLLM, there's a long list of engineering that changes. Paged attention replaces our naive left-padding with a virtual memory system for the KV cache. Preemption policies handle what happens when memory gets tight and you need to evict a request mid-generation. Chunked prefill breaks long prompts into pieces so they don't starve decode requests. Speculative decoding generates multiple candidate tokens per step to reduce latency.

None of these change the core loop. Admit requests, prefill them, decode in a shared batch, evict when done — that's exactly what we built here, just with 200 lines of Python instead of 200,000.

You can find the entire code at this link: [https://github.com/czhou578/multimodal-inference-visualizer/blob/main/nanogpt_cont_batching.ipynb](https://github.com/czhou578/multimodal-inference-visualizer/blob/main/nanogpt_cont_batching.ipynb)

CZ