---
layout: post
title: "Adding KV Cache to NanoGPT"
date: 2026-04-18
---

*This post requires an in-depth understanding of transformers and attention mechanisms in the context of Andrej Karpathy's NanoGPT repository. 

I have been getting my feet wet in ML inference systems recently, and decided to try implementing KV caching in NanoGPT. 

Just as brief context, NanoGPT is a repository by Andrej Karpathy that implements a GPT model from scratch, stripping away all the abstractions and optimizations to deliver the most simplistic language model that models ChatGPT.

For context, I highly recommend watching his Makemore series on YouTube and also how to build a GPT from scratch. The following will make more sense if you have a basic understanding of transformers and attention mechanisms.

**1. The Problem**

In a standard GPT model, the attention mechanism calculates the attention scores for all tokens in the input sequence. This is done by calculating the dot product of the query and key matrices, and then applying a softmax function to get the attention scores. 

The problem is that for long sequences, this becomes very computationally expensive. It would be wrong to calculate the key and value matrices for all the tokens every time we want to generate a new token, since we have already calculated them for the previous tokens. This is quadratic time complexity for a sequence generation and is where the idea of a cache comes in. 

**2. The Solution**

The solution is to cache the key and value matrices for each token in the input sequence. This way, we don't have to recalculate them every time we want to generate a new token. This reduces the time complexity of the attention mechanism from O(n^2) to O(n), where n is the length of the input sequence. 

**3. Implementation in NanoGPT**

In NanoGPT, we have to first identify the place where the KV Cache will live. In this case, it is at the most basic unit of the implementation, which is the `Head` class. We have defined the `Head` as the class that handles one head of self attention. 

As a recap, each head of self attention is responsible for calculating the attention scores for a specific token in the input sequence. 

I had to ask myself several questions:

1. What data structure should we use for the KV Cache? 
2. Does Q @ K^T still work if Q has shape (B, 1, hs) and K has shape (B, T, hs)? What does that mean?
3. Does masking even make sense anymore?
4. How should the forward method deal with inference vs training?

For the KV Cache, I initially thought that it would be some sort of a hashmap, where the keys are the token id's and the values are the key and value matrices for that token. But after thinking about it, I realized that it really is just a regular tensor of shape None initially that will hold the key entries which are just (B, 1, hs), all concatenated along the -2 axis. That way, each row will contain the key entries for a specific token. 

In actuality, you don't want to mix the key and value entries in one data structure, because the whole point of the cache is that you can directly multiply the keys with the values. Interleaving them would mean you have to specifically extract out the keys and values at every step, which defeats the purposes of caching. 

Now, we have to carefully consider that in the original implementation of NanoGPT, we were masking the future tokens of a sequence at a timestamp with negative infinity, which prevented the model from calculating attention scores for tokens it hadn't seen yet. But that was when we were recalculating attention for every single token in a sequence at every forward pass. 

Now that we are generating one token at a time and using a cache, it doesn't require this masking. So we can remove it!

Next, we have to consider how we are calculating the weights or "wei". Before, we were taking the query matrix and doing dot product with the key matrix. Now, since we have a key cache, we can instead dot product with the key matrix, making sure to transpose it so that the shapes can work out. 

The softmax and the dropout can stay the same, but we have to then dot product the value cache with the wei matrix to get the final result, which is the variable `out`. 

Now, we run into trouble since we only want to calculate the KV cache values during inference. How do we prevent the caches from being populated at the wrong time? Thankfully, PyTorch implicitly has a `self.training` flag that every single submodule from `nn.Module` inherits which has a boolean value showing whether training is active or not. We can just have an if-else condition that guards the training code from the inference code like so: 

Lastly, I had to add an if else condition to the concatenation logic, since if this is the very first time we are running forward, then the key and value caches would be None, so we need to set it to the first key / value tensors that were generated. 

THe final code looks like this:

```python

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.key_cache = None
        self.value_cache = None
        self.dropout = nn.Dropout(dropout)

    # KV Cache lives here.
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,1,hs)
        q = self.query(x) # (B,1,hs)
        v = self.value(x)

        # use the accumulated kv_cache here

        """
            (b, 1, hs)
            interleaving k and v in one tensor is wrong
                - you would have extract them back out every step, which defeats purpose of easy access.
            
            you need to priotize density and memory access patterns (locality and caches)

            Q should be attending over the full cache, all past tokens plus the current one.

            cache should be None

            set k to self.key_cache if initially none

            self.training = False is set on every submodule when you call model.eval().

        """

        if not self.training:
            if self.key_cache is not None:
                self.key_cache = torch.cat([self.key_cache, k], dim=-2) # (B, num_tokens_seen, hs)
                self.value_cache = torch.cat([self.value_cache, v], dim=-2) # (B, num_tokens_seen, hs)
            else:
                self.key_cache = k
                self.value_cache = v

            wei = q @ torch.transpose(self.key_cache, 1, 2) * self.key_cache.shape[-1]**-0.5

            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            out = wei @ self.value_cache # (B, 1, T) @ (B, T, hs) -> (B, 1, hs)
            return out
        else:
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the 
            
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out  

```

Now, let's write a generation function that runs during evaluation time and actually gives back the tokens that we want to see in the result! This is again somewhat similar to the real generation function from the original NanoGPT repository, but taking into account the KV Cache implementation.

Here is what it looks like: 
```python
def generate_kv_cache(model, idx, max_num_tokens):
    model.eval()
    clear_kv_cache(model)

    model(idx)

    with torch.no_grad():
        for step in range(max_num_tokens):
            curr_pos = idx.shape[1]

            logits, _ = model(idx[:, -1:], pos=torch.tensor[curr_pos], device=device) # (B, 1, C)
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    
    return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(generate_kv_cache(m, context, max_num_tokens=500)[0].tolist()))


```
In this function, we are setting the model to evaluation mode, and making sure to clear the kv cache for the model. 

Now, we run `model(idx)` once since that is how we prefill the KV cache before the next token is generated. Then, we have a for loop that iterates until the max number of new tokens we want, and grab the logits for the specific index, run softmax over the logits to get the probabilities, and then sample the next index. The index is added to the running sequence of indexes, which will then be decoded into the correct letters at the final step.

## Positional Encoding

A transformer processes all tokens at the same time. The problem is that this means the model has no sense of position. The sentence "A cat is big" and "A big is cat" would be treated as the same sentence, which is wrong. We need someway to encode the positional information of each token and feed that into the model so it knows this.

The model will use the position embedding table to learn the positions of the tokens. 

During a normal forward pass, if we were to have a sequence of length 17, then we can just easily look up the first 17 positions of the position embedding table and add it to the token embeddings. 

But since we now have the KV Cache, we aren't feeding the entire sequence into the model, but rather feeding one token at a time into the forward pass. If we didn't pass `pos` into the model, then the model would treat every token as the first token, which is wrong. 

I am assigning `curr_pos` to be `idx.shape[1]` (which is the number of tokens we have seen so far). 

Here is a walkthrough:

Let's say that my prompt is the string "O Romeo, " which encodes to 9 tokens: [15, 23, 6, 18, 14, 5, 12, 0, 3]

idx = [[15, 23, 6, 18, 14, 5, 12, 0, 3]]   # shape (1, 9)
         ↑   ↑   ↑   ↑   ↑   ↑   ↑  ↑  ↑
        pos0 pos1 ... ... ... ... ... ... pos8

Step 0: The width of the first row is 9. The model generates 42. We append it. 

idx = [[15, 23, 6, 18, 14, 5, 12, 0, 3, 42]]   # shape (1, 10)

Step 1: The width of the first row is 10. The model generates 10. We append it. 

idx = [[15, 23, 6, 18, 14, 5, 12, 0, 3, 42, 10]]   # shape (1, 11)

Step 2: The width of the first row is 11. The model generates 19. We append it. 

idx = [[15, 23, 6, 18, 14, 5, 12, 0, 3, 42, 10, 19]]   # shape (1, 12)

In conclusion, using idx.shape[1] allows us to immediatley know how long the total sequence is. This tells us where the next position the next token should be. 

## Result

After running the result, we now get 

```text
And they brid write, is not the die;
Though we art One my day hangs:
Wart he us hath bury, dills ane away, my feanst,
Anzing heavens, tofultien me milen's
Whines is eye, hain latise, drovets, and Will.

Downerabs!
Alhin the courtius, onceivy:
Supplain's twoy. Hence's norfole,
Against my lows thee again Willo when evicks eye myself?
ETo husing stroops: the resheper my brupt for treign the flows.
Tale oftenceful in thy offery your
Hasting is a aday Was happesty:
if courty.

ANGCIO:
Say, from care,

```

Now, let's do a shapes check to verify things:

GPTLanguageModel.forward(idx, pos=None)
  idx:     (1, 9)       # (B, T)

  tok_emb: (1, 9, 64)   # token_embedding_table lookup
  pos_emb: (9, 64)      # position_embedding_table(arange(9))
  x:       (1, 9, 64)   # tok_emb + pos_emb — broadcast adds fine

  ↓ Into Head.forward(x):
    x: (1, 9, 64)
    k = self.key(x):     (1, 9, 16)   # Linear(64 → 16)
    q = self.query(x):   (1, 9, 16)
    v = self.value(x):   (1, 9, 16)

    # key_cache is None → set it directly
    self.key_cache:    (1, 9, 16)
    self.value_cache:  (1, 9, 16)

    wei = q @ key_cache.T:  (1,9,16) @ (1,16,9) → (1, 9, 9)
    wei (after softmax):    (1, 9, 9)
    out = wei @ value_cache: (1,9,9) @ (1,9,16) → (1, 9, 16)


Phase 2: Decode Step 0:

GPTLanguageModel.forward(idx[:, -1:], pos=9)
  idx:     (1, 1)       # only the last token
  tok_emb: (1, 1, 64)
  pos_emb: (1, 64)      # position_embedding_table(tensor([9]))
  x:       (1, 1, 64)

  ↓ Into Head.forward(x):
    x: (1, 1, 64)
    k = self.key(x):   (1, 1, 16)
    q = self.query(x): (1, 1, 16)
    v = self.value(x): (1, 1, 16)

    # key_cache exists — concatenate!
    self.key_cache:   cat[(1,9,16), (1,1,16)] → (1, 10, 16)
    self.value_cache: cat[(1,9,16), (1,1,16)] → (1, 10, 16)

    wei = q @ key_cache.T:   (1,1,16) @ (1,16,10) → (1, 1, 10)
    wei (after softmax):     (1, 1, 10)
    out = wei @ value_cache: (1,1,10) @ (1,10,16) → (1, 1, 16)

The best way to definitely check this is to do an output equivalence test. In other words, if the KV cache is mathematically correct, then it should produce the exact same tokens as the no-cache version given the same random seed and prompt. 

I added this test cell to my notebook:

```python
torch.manual_seed(42)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Run without cache
out_no_cache = generate_no_cache(model, context.clone(), max_new_tokens=20)

# Run with cache (same seed, same prompt)
torch.manual_seed(42)
out_with_cache = generate_with_cache(model, context.clone(), max_new_tokens=20)

# Check token-by-token equality
assert torch.equal(out_no_cache, out_with_cache), \
    f"MISMATCH!\nNo cache:   {out_no_cache}\nWith cache: {out_with_cache}"

print("✓ Cache output matches no-cache output exactly!")
print(decode(out_with_cache[0].tolist()))
```

What this does is run the same prompt through both the no-cache and cache versions of the model, and checks if the output is the same. If it is, then we know that the KV cache is working as expected. 

## Benchmarks

Now, let's see how much faster the KV cache makes the model, since that is what matters right? 

```python

# ── non-cached generate (forces full-context recompute every step) ────────────
def generate_no_cache(model, idx, max_new_tokens):
    """Runs in train mode so the KV cache branch is never entered."""
    model.train()                          # disables KV cache path
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ── cached generate (your existing path, one token fed at a time) ─────────────
def generate_with_cache(model, idx, max_new_tokens):
    model.eval()
    clear_kv_cache(model)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Feed only the LAST token so the cache does the rest of the work
            logits, _ = model(idx[:, -1:])   # (B, 1, vocab_size)
            logits = logits[:, -1, :]
            probs  = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ── benchmark ─────────────────────────────────────────────────────────────────
N_TOKENS   = 200
N_RUNS     = 3       # average over multiple runs for stability
context    = torch.zeros((1, 1), dtype=torch.long, device=device)

# warm-up (avoids cold-start CUDA overhead skewing results)
_ = generate_no_cache(model, context.clone(), 10)
clear_kv_cache(model)
_ = generate_with_cache(model, context.clone(), 10)

# --- No KV cache ---
times_no_cache = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    generate_no_cache(model, context.clone(), N_TOKENS)
    if device == 'cuda':
        torch.cuda.synchronize()
    times_no_cache.append(time.perf_counter() - t0)

# --- With KV cache ---
times_cache = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    generate_with_cache(model, context.clone(), N_TOKENS)
    if device == 'cuda':
        torch.cuda.synchronize()
    times_cache.append(time.perf_counter() - t0)

avg_no_cache = sum(times_no_cache) / N_RUNS
avg_cache    = sum(times_cache)    / N_RUNS

print(f"Tokens generated : {N_TOKENS}")
print(f"No KV cache      : {avg_no_cache:.3f}s  ({N_TOKENS/avg_no_cache:.1f} tok/s)")
print(f"With KV cache    : {avg_cache:.3f}s  ({N_TOKENS/avg_cache:.1f} tok/s)")
print(f"Speedup          : {avg_no_cache/avg_cache:.2f}×")

```

In this block of code, we first define the no-cache and cache versions of the generate function. The no-cache version is the original generate function, which is used to generate text from the model. The cache version is the same as the no-cache version, but it uses the KV cache to generate text from the model. 

Then, we define the benchmark function, which is used to benchmark the no-cache and cache versions of the generate function. The benchmark function first generates text from the model using the no-cache version, and then from the model using the cache version. Finally, it prints the speedup of the cache version over the no-cache version.

Running this code, we get:

```text
Tokens generated : 200
No KV cache      : 1.305s  (153.3 tok/s)
With KV cache    : 1.172s  (170.7 tok/s)
Speedup          : 1.11×

```

The main bottleneck at this point is because the model itself right now is so small to the point where the Python runtime overhead of running the model is the main bottleneck. But we can see that from a numerical standpoint, the KV cache is speeding up the tokens per second being generated. 

## Conclusion

In conclusion, we were able to achieve a small speedup in token generation speed by using a small KV cache optimization. By preventing the unnecessary recomputation of past tokens, we can achieve speedup in token generation. 

## Errors I encountered:

1. Previously, I was doing the estimate loss loop very frequently, which greatly slowed down the training process on the free Google Colab GPU that I had access to. I ended up changing the code so that it ran every 500 iterations instead of every 100 iterations. This greatly sped things up.

2. I tried to use `torch.compile` on the model but it turned out that with my custom implemented key and value caches, the behavior of this command doesn't often play well. Torch compile creates a computated graph that it replays during inference, but since the KV cache is mutable and frequently changing, it can corrupt the graph. 

3. There was a time when my validating loss wasn't going down. I found that in my estimate loss function, since I wasn't using dropout, there was no difference between having `model.train()` and `model.eval()` called. Once I removed this from the estimation loss loop, the displayed numbers went down quite fast. 

4. After my training finished, I encountered this:

```text

AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_2203/257733165.py in <cell line: 0>()
    351 # generate from the model
    352 context = torch.zeros((1, 1), dtype=torch.long, device=device)
--> 353 print(decode(generate_kv_cache(m, context, max_num_tokens=500)[0].tolist()))
    354 # print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

/tmp/ipykernel_2203/257733165.py in generate_kv_cache(model, idx, max_num_tokens)
    313             curr_pos = idx.shape[1]
    314 
--> 315             logits, _ = model(idx[:, -1:], pos=torch.tensor([[curr_pos]], device=device))
    316             logits = logits[:, -1, :]
    317 

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```

The cause of this was that my positional embeddings table only had 32 entries, so when I was generating over 500 tokens initially, the out of bound error was being thrown. In order to fix this, I had to limit the number of generated tokens to be less than the block size, so a max of 31. 

This is a real limitation worth mentioning — with a fixed-size learned position embedding table and a simple KV cache, you can only generate up to block_size total tokens (prompt + generated). Production models solve this with either RoPE (rotary positional embeddings, which don't have a table), or a sliding window cache that evicts old entries and reuses positions.

Thanks for reading! Hopefully it helps!

CZ