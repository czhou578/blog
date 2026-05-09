---
layout: post
title: "Adding Continuous Batching to NanoGPT"
date: 2026-05-09
---

*This post requires an in-depth understanding of transformers and attention mechanisms in the context of Andrej Karpathy's NanoGPT repository. 

I have been getting my feet wet in ML inference systems recently, and decided to try implementing KV caching in NanoGPT. 

Just as brief context, NanoGPT is a repository by Andrej Karpathy that implements a GPT model from scratch, stripping away all the abstractions and optimizations to deliver the most simplistic language model that models ChatGPT.

For context, I highly recommend watching his Makemore series on YouTube and also how to build a GPT from scratch. The following will make more sense if you have a basic understanding of transformers and attention mechanisms.

## Context:

In the previous post, I talked about how I added KV caching to NanoGPT. We were able to manually add a KV cache to the `Head` class, which is responsible for calculating the attention scores for a specific token in the input sequence. We were able to reduce the time complexity of the attention mechanism from O(n^2) to O(n), where n is the length of the input sequence, as well as achiee a moderate speedup to the tok/sec generation in inference. 

In this article, we will add Continuous Batching to NanoGPT. 

## Problem:

Right now, even witih the KV optimization, we are only serving one request at a time. For real life production systems, this is nowhere near enough since we want to maximize the GPU's utilization. In addition, we are not taking into account the fact that different requests will have different sequence lengths. Without some kind of way to evenly distribut the load, we will have cases where one request will be waiting for other requests to finish, even if it could have been finished earlier. 

This bottleneck will only get worse as we add more and more requests to the system. 


## Steps

The current `generate_with_cache` serves **one request at a time**. `B` in your batch dim is always 1 during generation. Continuous batching means multiple independent requests share that batch dimension, but they can **arrive and finish at different times**.

The first thing we need to do is to implement a Request class. The reason is that we now have to keep track of multiple requests at the same time. 

Each request will have:
- Its token IDs generated so far
- How many tokens it still needs to produce (`max_new_tokens`)
- Its own KV cache entries (currently baked into `Head` as a single tensor for the whole batch). The reason is that from a code perspective, it is easier to manage the KV cache outside of the Head class when there are multiple requests.

KV Cache needs to be per batch, not per head. 

```python

@dataclass
class Request:
    """Each in-flight generation carries its own state and KV cache."""
    id: int
    prompt_tokens: List[int]          # the original encoded prompt
    max_new_tokens: int               # how many tokens this request wants
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "waiting"           # "waiting" -> "active" -> "done"

    # Hint 2: Per-request KV cache, keyed by (layer_idx, head_idx)
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

For this code, we define several helper functions that will help us manage the requests, as well as the attributes that each request needs to track. 

## Attributes

1. With the `prompt_tokens` attribute, we keep track of the original encoded prompt. 

2. `max_new_tokens` is the number of tokens that we want to generate for the request. 

3. `generated_tokens` is the list of tokens that have been generated so far. 

4. `status` is the status of the request, which can be "waiting", "active", or "done". The three states are the best way to designate the state of the request as we push them into queues as shown later in the code. 

5. `kv_cache` is the KV cache for the request, which is a dictionary of key-value tensors for each layer and head. We 


Your `Head.key_cache` has shape `(B, T, hs)` — one contiguous tensor for the whole batch. With continuous batching, different requests have different sequence lengths. You have two options to think through:

1. **Per-request caches** (dict keyed by request ID) — each request gets its own `(1, T_i, hs)` tensor, and you `torch.cat` them along dim=0 before the attention math. What do you do when the `T_i` values differ across requests?
2. **Padded batch cache** — pre-allocate `(max_batch, max_seq_len, hs)` and track how many tokens each slot has actually used. How do you mask out the padding in attention?

## Generation Request Loop

We need now a generation request loop to serve as a stepping stone to becoming a continuous batching system. 

```python
# ── 3. Per-Request generation (Hint 1 + 2 combined) ──────────────────────────
def generate_request(model, request: Request):
    """
    Generate for a single Request object.
    The KV cache lives on the Request, not inside the model.

    This is the building block for the continuous batching scheduler (Hint 3).
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

In the beginning, we want to convert the prompt tokens into a tensor and pass it through the model. The model expects a tensor of shape (B, T) where B is the batch size and T is the sequence length. In this case, B = 1 and T = len(request.prompt_tokens).

Then, we run the model once wit the pormpts and get the logits for the last token in the sequence. We store the KV cache for each layer in the request and mark the request as 'active'. Keep in mind that we are storing the KV cache entry in a dictionary in each Request where the key is the head and layer index and the value is a tuple of (key_tensor, value_tensor).*

*Since we have multiple layers and each layer can have multiple heads, it is important tht we keep track of this arrangement using the index of the head and the layer.



## Scheduler Loop

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

The key insight: **step 1 and step 5 happen every iteration**, not just at the start and end. Requests flow in and out continuously. 

The code looks something like this:

## Training

So my training loop for this run is very similar to the one before. 

The loop is like this:

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

Here was the result: 

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
step 4100: train loss 1.6742, val loss 1.8410
step 4200: train loss 1.6706, val loss 1.8450
step 4300: train loss 1.6722, val loss 1.8282
step 4400: train loss 1.6620, val loss 1.8330
step 4500: train loss 1.6661, val loss 1.8250
step 4600: train loss 1.6471, val loss 1.8433
step 4700: train loss 1.6588, val loss 1.8178
step 4800: train loss 1.6478, val loss 1.8191
step 4900: train loss 1.6431, val loss 1.8307
step 4999: train loss 1.6503, val loss 1.8342

And they brid write, is not the die;
Though we art One my day hangs:
Wart he us hath bury, dills ane away, my feanst,
Anzing heavens, tofultien me milen's
Whines is eye, hain latise, drovets, and Will

```

After getting the training to work, we can implement several generation functions as before. 



So we now are going to have a scheduler loop that will take in requests and process them in parallel. This will replace the old method. 