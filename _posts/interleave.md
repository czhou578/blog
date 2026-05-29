---
layout: post
title: "Adding Interleaving to NanoGPT"
date: 2026-05-29
---

* This is a post that was supposed to have been published after the chunked prefill post, but got delayed due to time constraints. 

## Problem

The issue is that chunked prefill as we currently have it is doing 2 forward passes per step, one for the prefill requests and one for the decode requests. 

On a real GPU, each forward pass has fixed overhead (kernel launches, memory transfers). A production inference engine (vLLM) **interleaves** decode and prefill tokens into a **single forward pass**, because the model doesn't care whether a token in the batch is a decode token or a prefill token — it just sees an input tensor of shape `(B, T)`.

**Decode-prefill interleaving** merges both types of work into one `model()` call per step. The token budget constrains total tokens per step, decode requests get first priority (they're cheap and already own KV memory), and the remaining budget goes to prefill chunks.

So far, we have the following:

- ✅ Token budget (`token_budget` parameter)
- ✅ Decode-first priority (`remaining_budget = token_budget - len(active_requests)`)
- ✅ Chunked prefill with `prefill_cursor`
- ✅ `assemble_batch_cache` / `disassemble_batch_cache` for batching decode requests
- ✅ Per-request KV cache on the `Request` object

What's missing: the prefill chunk and decode tokens go through **separate** `model()` calls. Your goal is to fuse them into **one** call.

## Visualizing the the Fused Input Tensor

We need to first get a visual understanding of what we want in the fused tensor with both prefill and decode tokens. 

Imagine step 5 has 3 active decode requests + 1 prefilling request getting an 8-token chunk. The fused input looks like:

```
token_budget = 16
decode requests: A (position 22), B (position 15), C (position 9)  → 3 tokens
prefill chunk:   D (tokens at positions 0..7)                       → 8 tokens
total: 11 tokens ≤ 16 ✓

T_max = max(1, 8) = 8   (longest row in the batch)

batch_tokens (B=4, T=8):
Row 0 (decode A): [ PAD, PAD, PAD, PAD, PAD, PAD, PAD, tok_A ]  ← 1 token, left-padded
Row 1 (decode B): [ PAD, PAD, PAD, PAD, PAD, PAD, PAD, tok_B ]  ← 1 token, left-padded
Row 2 (decode C): [ PAD, PAD, PAD, PAD, PAD, PAD, PAD, tok_C ]  ← 1 token, left-padded
Row 3 (prefill D):[ D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7 ]  ← 8 tokens, no padding

batch_positions (B=4, T=8):
Row 0: [ 0, 0, 0, 0, 0, 0, 0, 22 ]   ← position 22 for decode token
Row 1: [ 0, 0, 0, 0, 0, 0, 0, 15 ]
Row 2: [ 0, 0, 0, 0, 0, 0, 0,  9 ]
Row 3: [ 0, 1, 2, 3, 4, 5, 6,  7 ]   ← positions 0-7 for prefill chunk
``` 

**Each row can have a different number of "real" tokens.** Decode rows have T=1 (left-padded), the prefill row has T=chunk_size. They're all padded to the same `T_max` so they fit in a single tensor.

**Question to ask yourself:** Why left-pad instead of right-pad? Because model logits for the "next token" are always taken from the **last position** (`logits[:, -1, :]`). Left-padding keeps all real tokens right-aligned, so the last position is always meaningful.

## Assemble the Batch

Right now, the `assemble_batch_cache` only handles decode requests (all with T=1). 

But we now need to assemble a cache that includes:

- **Decode requests:** have a populated `kv_cache` — same as before
- **Prefilling request (continuation chunk):** may have a partial `kv_cache` from earlier chunks, OR no cache at all (first chunk)

When a prefill request has `past_k` of shape `(1, T_past, hs)` and its chunk has `T_chunk` tokens, the model will output `new_k` of shape `(1, T_past + T_chunk, hs)`. But decode requests output `new_k` of shape `(1, T_past_i + 1, hs)`. After the fused forward pass, we need to strip padding from **each row independently** during disassembly.

We will extend the `assemble_batch_cache` to accept a mixed list of requests where one request may contribute more than 1 token. 

```python

def assemble_fused_batch(decode_reqs: List[Request], prefill_req, chunk_size):
    """
    Build a single (B, T_max) input tensor + batched cache for the fused forward pass.

    Args:
        decode_reqs:  list of active Request objects (each contributes 1 token)
        prefill_req:  the request being prefilled (contributes chunk_size tokens), or None
        chunk_size:   number of prefill tokens this step

    Returns:
        batch_tokens:   (B, T_max) input tensor
        batch_positions: (B, T_max) position indices
        past_kvs:       batched cache [layer][head] = (B, T_max_cache, hs)
        attn_mask:      (B, 1, T_max_cache) bool mask for cached positions
        pad_info:       dict with per-row metadata for disassembly
    """

    num_new_tokens = []
    all_reqs = []

    for req in decode_reqs:
        all_reqs.append(req)
        num_new_tokens.append(1)
    
    if prefill_req:
        all_reqs.append(prefill_req)
        num_new_tokens.append(chunk_size)

    B = len(all_reqs)
    T_max = max(num_new_tokens)

    batch_tokens = []
    batch_positions = []

    for req in decode_reqs:
        pos_val = len(req.tokens_so_far) - 1
        row = [0] * (T_max - 1) + [pos_val]
        batch_positions.append(row)

        token_row = [0] * (T_max - 1) + [req.tokens_so_far[-1]]
        batch_tokens.append(token_row)
    
    if prefill_req:
        cursor = prefill_req.prefill_cursor

        chunk_positions = list(range(cursor, cursor + chunk_size))

        padding = [0] * (T_max - chunk_size)

        batch_positions.append(padding + chunk_positions)

        chunk = prefill_req.prompt_tokens[cursor: cursor + chunk_size]
        pad = [0] * (T_max - chunk_size)

        batch_tokens.append(pad + chunk)

    batch_positions = torch.tensor(batch_positions, device=device)        
    batch_tokens = torch.tensor(batch_tokens, dtype=torch.long, device=device)  
    # Assemble KV cache 

    if prefill_req and not prefill_req.kv_cache:
        head_size = n_embd // n_head

        for li in range(n_layer):
            for hi in range(n_head):
                prefill_req.kv_cache[(li, hi)] = (
                    torch.empty(1, 0, head_size, device=device),
                    torch.empty(1, 0, head_size, device=device)
                )
        
    past_kvs, attn_mask, pad_lengths = assemble_batch_cache(all_reqs)
    
    return batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths

```

In this new function, we first create two empty lists, `all_reqs` and `num_new_tokens` to keep track of the requests and how many tokens each will contribute to the batch. 

Each decode request is added to the `all_reqs` list and its token count is recorded as 1. A reminder that decode requests always contribute exactly one token per step — they're already done with prefill and are just generating the next token autoregressively.  

If there is an active prefilling request, it's appended last to `all_reqs` and its contribution is `chunk_size` tokens (chunked prefill!).

Next, we want to build the token and position rows. For each decode request, `tokens_so_far` holds every token the request has generated so far (including the prompt). The last valid position index is `len - 1`, i.e., the position of the token the model is about to process. For example, if the request has produced 22 tokens, `pos_val = 22`. 

```python

row = [0] * (T_max - 1) + [pos_val]
batch_positions.append(row)

```

We do this because we are left padding the batch with zeros and we want the actual positions to be on the right side of the tensor so that the model can attend to them. The same thing for the actual token ID's.

For the prefill request, we have the `prefill_req.prefill_cursor` to tell us where we are in the prompt. The chunk we are processing this step is `prefill_req.prompt_tokens[cursor: cursor + chunk_size]`.

The absolute position indices for the tokens are calculated the same way as before.

Then, the Python lists-of-lists are converted into `(B, T_max)` PyTorch tensors and moved onto the appropriate device (CPU/GPU). `dtype=torch.long` is required for token IDs since they're used as indices into the embedding table.

Next, we want to check if there's a prefill request and that it has no KV Cache, which is true for the very first chunk. For every layer `li` and every head `hi`, it initializes the KV cache as a pair of empty tensors of shape `(1, 0, head_size)`. The `T=0` dimension means "no cached tokens yet." This is the canonical "empty past" representation — assemble_batch_cache downstream expects every request to have a `kv_cache` dict, and empty tensors serve as a valid zero-length cache.

After calling our previous `assemble_batch_cache` function, we can now proceed to return everything that we need for a single fused forward pass of the model: 

- `batch_tokens` — what to feed the embedding layer
- `batch_positions` — for positional embeddings
- `past_kvs` — the batched KV context for attention
- `attn_mask` — which cache positions to attend to
- `pad_lengths` — bookkeeping so `disassemble_fused_batch` can correctly unpack each row's new KV cache after the forward pass

## Disassemble Fused Cache

For the disassemble part, we can reuse our existing `assemble_batch_cache` function and just modify it slightly to handle the fused cache. 

```python

def disassemble_fused_cache(requests, new_kvs, num_new_tokens_per_req):
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
            for i, req in enumerate(requests):

                t_new = num_new_tokens_per_req[i]

                k_new_valid = batched_k[i : i + 1, -t_new:, :]
                v_new_valid = batched_v[i : i + 1, -t_new:, :]

                if (layer_idx, head_idx) in req.kv_cache:
                    k_old, v_old = req.kv_cache[(layer_idx, head_idx)]
                    req.kv_cache[(layer_idx, head_idx)] = (
                        torch.cat([k_old, k_new_valid], dim=1),
                        torch.cat([v_old, v_new_valid], dim=1)
                    )
                else:
                    req.kv_cache[(layer_idx, head_idx)] = (k_new_valid, v_new_valid)

```

In this function, we take the new KV cache outputs from the model's forward pass and add them to each request's persistent KV cache. Because the model processes padded input sequences, we must carefully slice out only the *valid* new KV cache entries for each request, discarding the padding.

```python
    for layer_idx, block_kv in enumerate(new_kvs):
        for head_idx, (batched_k, batched_v) in enumerate(block_kv):
```

Here, we iterate through every layer and every attention head's output. `new_kvs` contains the new Key and Value tensors produced by the model during the forward pass for this specific step. The shape of `batched_k` and `batched_v` is `(B, T_max, head_size)`.

```python
            for i, req in enumerate(requests):
                t_new = num_new_tokens_per_req[i]
```

For each layer and head, we loop over each request in the batch. The `requests` list matches the row order of the batch. `num_new_tokens_per_req[i]` tells us exactly how many real tokens this specific request contributed to the forward pass (e.g., 1 for decode requests, `chunk_size` for the prefill request).

```python
                k_new_valid = batched_k[i : i + 1, -t_new:, :]
                v_new_valid = batched_v[i : i + 1, -t_new:, :]
```

This is the most critical part of the function: **stripping the padding**.
Because we used **left-padding** when assembling the batch, the real tokens are always at the end of the sequence dimension.
- `i : i + 1` selects the specific row for this request, keeping the batch dimension `(1, ...)`.
- `-t_new:` slices exactly the last `t_new` tokens from the sequence dimension, ignoring the zeros prepended as padding.
Now, `k_new_valid` has the shape `(1, t_new, head_size)`, containing only the real KV cache updates.

```python
                if (layer_idx, head_idx) in req.kv_cache:
                    k_old, v_old = req.kv_cache[(layer_idx, head_idx)]
                    req.kv_cache[(layer_idx, head_idx)] = (
                        torch.cat([k_old, k_new_valid], dim=1),
                        torch.cat([v_old, v_new_valid], dim=1)
                    )
```

If the request already has an existing KV cache (which is true for all decode requests and subsequent prefill chunks), we fetch its old Key and Value tensors. We then concatenate the old cache (`k_old`) with the new valid cache (`k_new_valid`) along the sequence dimension (`dim=1`), and update the request's `kv_cache` dictionary. 

```python
                else:
                    req.kv_cache[(layer_idx, head_idx)] = (k_new_valid, v_new_valid)
```

If this is the first chunk of a new prefill request, the `kv_cache` might not have this layer/head initialized yet (or it might have been initialized to empty tensors during assembly). In this case, we just set the new valid cache directly as the request's cache for this layer and head.

## Head Modifications

Previously, the `Head.forward()` had two branches:

- `past_k is not None` → decode path (no causal mask, just attend to full cache + new token)
- `past_k is None` → prefill path (causal mask)

But now, we want to handle both cases in one function. So we modify the `Head.forward()` function to handle both cases in one function.

```python

class Head(nn.Module):
    """One head of self-attention — now STATELESS (no internal cache)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None, attn_mask=None):
        """
        Args:
            x:      (B, T, C)       input embeddings
            past_k: (B, T_past, hs) cached keys, or None
            past_v: (B, T_past, hs) cached values, or None
        Returns:
            out:   (B, T, hs)           attention output
            new_k: (B, T_past+T, hs)    updated key cache   (None during training)
            new_v: (B, T_past+T, hs)    updated value cache  (None during training)
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        if not self.training:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

            T_full = k.shape[1]

            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T_full)

            causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)

            if T > 1:
                new_token_mask = self.tril[:T, :T]
                causal_mask[:, -T:] = new_token_mask

            causal_mask = causal_mask.unsqueeze(0).expand(B, -1,- 1)
        
            if attn_mask is not None:
                new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
                full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)
                causal_mask = causal_mask & full_pad_mask

            wei = wei.masked_fill(~causal_mask, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v

            return out, k, v

        else:
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, None, None   

```

Here's a breakdown of the new fused `Head.forward` logic:

```python
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)
```

First, we project the new input tokens `x` (which could be padded single tokens or a full chunk) into their Query, Key, and Value representations.

```python
        if not self.training:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

            T_full = k.shape[1]
```

During inference, we take the batched KV cache (`past_k` and `past_v` provided by `assemble_fused_batch`) and concatenate it with the new `k` and `v`. `T_full` represents the total sequence length (past history + new tokens).

```python
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T_full)
```

We compute the raw attention scores. The queries of the *new* tokens are dotted with all keys (past and present).

```python
            causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
```

We initialize a causal mask with all `True` values, meaning every query can attend to every key by default. This is exactly what we want for decode requests (T=1), which just need to attend to all past tokens and their own new token.

```python
            if T > 1:
                new_token_mask = self.tril[:T, :T]
                causal_mask[:, -T:] = new_token_mask
```

If `T > 1`, it means we have a prefill chunk in our batch. For these new prefill tokens, they must not look ahead at future tokens within the chunk. So we take the lower-triangular mask (`self.tril`) and apply it to the last `T` columns of our causal mask (the columns corresponding to the new tokens).

```python
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1,- 1)
```

We add a batch dimension and expand the causal mask so it has the shape `(B, T, T_full)`.

```python
            if attn_mask is not None:
                new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
                full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)
                causal_mask = causal_mask & full_pad_mask
```

This handles the **padding**. `attn_mask` (from `assemble_fused_batch`) tells us which tokens in the *past* KV cache are real and which are just left-padding zeros. We create a `new_valid` mask of all `True` for the new tokens (the left padding in `x` means those padded query positions will generate garbage that we ignore, but we still need the tensor shapes to align). We concatenate them to get `full_pad_mask`, and logically AND it with our causal mask to ensure no query ever attends to a padding token.

```python
            wei = wei.masked_fill(~causal_mask, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v

            return out, k, v
```

Finally, we apply our unified mask, compute the softmax probabilities, multiply by the values `v`, and return the output along with the fully updated `k` and `v` caches. These new caches will then be unpacked by `disassemble_fused_cache`.

Here is a step-by-step graphical visualization of how the tensors transform. 

Let's use a concrete example where:
* **`T_past = 4`** (We have 4 tokens previously processed in the KV cache)
* **`T = 3`** (We are processing a chunk of 3 new tokens in the current step)
* **`T_full = 7`** (`T_past + T`)
* **`B = 2`** (Batch size of 2)

I will use `██` for **True** (allowed to attend) and `░░` for **False** (masked out/prevented from attending).

### Step 1: Initialize all-True mask
```python
causal_mask = torch.ones(T, T_full, device=x.device, dtype=torch.bool)
```
We start with a shape of `(3, 7)`. Initially, every new token is allowed to look at every token in the entire `T_full` sequence.
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
New Token 1   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
New Token 2   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
```

### Step 2: Grab the lower-triangular mask
```python
new_token_mask = self.tril[:T, :T] 
```
We grab a `(3, 3)` square matrix. This enforces the rule: "Token 0 can only see Token 0. Token 1 can see Tokens 0 & 1. Token 2 can see Tokens 0, 1, & 2."
```text
                 New Tokens (T)
                 0    1    2
New Token 0   [ ██ , ░░ , ░░ ]
New Token 1   [ ██ , ██ , ░░ ]
New Token 2   [ ██ , ██ , ██ ]
```

### Step 3: Apply the lower-triangular mask to the right side
```python
causal_mask[:, -T:] = new_token_mask 
```
We take the `(3, 3)` matrix from Step 2 and paste it over the rightmost 3 columns (`-T:`) of the `causal_mask` from Step 1. The past tokens remain untouched (all `True`).
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [ ██ , ██ , ██ , ██    |   ██ , ░░ , ░░ ]
New Token 1   [ ██ , ██ , ██ , ██    |   ██ , ██ , ░░ ]
New Token 2   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]
```
*Notice how Token 0 can look at all past tokens + itself, but is blocked from looking at future tokens 1 and 2.*

### Step 4: Expand for the Batch Dimension
```python
causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
```
Right now, `causal_mask` is just a 2D grid of shape `(T, T_full)`. But our attention weights `wei` have a shape of `(B, T, T_full)`. 

1. **`unsqueeze(0)`**: Adds a batch dimension to the front, changing the shape to `(1, 3, 7)`.
2. **`expand(B, -1, -1)`**: Broadcasts that single mask across the entire batch (size `B=2`), creating a final shape of `(2, 3, 7)`. It effectively duplicates the mask so every sequence in the batch uses it.

```text
Batch Item 0 (e.g., Sequence A)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [ ██ , ██ , ██ , ██    |   ██ , ░░ , ░░ ]
New Token 1   [ ██ , ██ , ██ , ██    |   ██ , ██ , ░░ ]
New Token 2   [ ██ , ██ , ██ , ██    |   ██ , ██ , ██ ]

Here is a graphical explanation of how the padding mask (`attn_mask`) is integrated with the `causal_mask`. 

This step handles the fact that in a batch, some sequences might be shorter than others and have been padded with empty tokens in their KV cache history. The model must be prevented from paying attention to these meaningless padded tokens.

Let's build on the previous example:
* **`T_past = 4`**
* **`T = 3`**
* **`B = 2`**

Imagine our batch has two sequences with different past histories:
* **Sequence A** (Batch Item 0): Only has **2 real tokens** in the past cache, and **2 padding tokens** (left-padded).
* **Sequence B** (Batch Item 1): Has **4 real tokens** in the past cache (no padding).

### Step 1: The Input `attn_mask`
The `attn_mask` passed into the forward pass only covers the *past* tokens (`T_past`). It has shape `(B, 1, T_past)`.
```text
                 Past Tokens (T_past)
                 0    1    2    3    
Sequence A    [  F ,  F ,  T ,  T ]   <-- Positions 0 and 1 are padding!
Sequence B    [  T ,  T ,  T ,  T ]   <-- All valid tokens
```

### Step 2: Create `new_valid`
```python
new_valid = torch.ones(B, 1, T, device=x.device, dtype=torch.bool)
```
The code assumes all new tokens being passed in are valid for the purpose of this specific mask step. We create an all-`True` mask of shape `(2, 1, 3)` for the new chunk.
```text
                 New Tokens (T)
                 0    1    2
Sequence A    [  T ,  T ,  T ]
Sequence B    [  T ,  T ,  T ]
```

### Step 3: Concatenate to make `full_pad_mask`
```python
full_pad_mask = torch.cat([attn_mask, new_valid], dim=-1)  # (B, 1, T_full)
```
We stick them together horizontally to get a mask that covers the entire `T_full` (7 tokens).
```text
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
Sequence A    [  F ,  F ,  T ,  T    |    T ,  T ,  T ]
Sequence B    [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
```

### Step 4: Logical AND with the `causal_mask`
```python
causal_mask = causal_mask & full_pad_mask
```
Now, we overlay (Logical AND) this padding mask onto the `causal_mask` we built earlier. 

* If a position is `T` in both, it stays `T`. 
* If a position is `F` in either mask, it becomes `F`.

Because `full_pad_mask` has a dimension of `1` in the middle `(B, 1, T_full)`, PyTorch automatically broadcasts (copies) it downwards across all `T` rows of the causal mask.

Here is the final result:

```text
Batch Item 0 (Sequence A - Has Padding)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  F ,  F ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  F ,  F ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  F ,  F ,  T ,  T    |    T ,  T ,  T ]
                ^^^^^^^^
           (Padding is completely blocked from attention)

-----------------------------------------------------------
Batch Item 1 (Sequence B - No Padding)
                 Past Tokens (T_past)       New Tokens (T)
                 0    1    2    3         0    1    2
New Token 0   [  T ,  T ,  T ,  T    |    T ,  F ,  F ]
New Token 1   [  T ,  T ,  T ,  T    |    T ,  T ,  F ]
New Token 2   [  T ,  T ,  T ,  T    |    T ,  T ,  T ]
                ^^^^^^^^
           (All past tokens remain attendable)
```

**Why this matters:** Without this step, `New Token 0` in Sequence A would average in the values of the empty padding tokens at indices 0 and 1, which would completely corrupt the model's predictions.

## Scheduler Loop (Interleave Generate)

Now, we can write our now simplified scheduling loop. 

```python

def interleaved_generate(model, requests, policy="fcfs", token_budget=16, max_kv_tokens=256):
    scheduler = Scheduler(policy, token_budget=token_budget, max_kv_tokens=max_kv_tokens)

    step = 0

    for req in requests:
        req.arrival_time = step
        scheduler.add_request(req)

    model.eval()

    with torch.no_grad():
        while not scheduler.is_done():
            prefill_req, decode_reqs = scheduler.schedule(step)

            chunk_size = 0
            remaining_budget = token_budget - len(decode_reqs)

            if remaining_budget > 0 and prefill_req is not None:
                tokens_left = len(prefill_req.prompt_tokens) - prefill_req.prefill_cursor

                chunk_size = min(remaining_budget, tokens_left)

            if chunk_size == 0 and not decode_reqs:
                step += 1
                continue

            # 3. ── SINGLE FUSED MODEL CALL ──
            # Use your already-written helper to build the batched inputs
            batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths = assemble_fused_batch(
                decode_reqs, 
                prefill_req if chunk_size > 0 else None, 
                chunk_size
            )

            logits, _, new_kvs = model(
                batch_tokens,
                pos=batch_positions,
                past_kvs=past_kvs,
                attn_mask=attn_mask
            )

            ## DISASSEMBLY

            all_reqs = decode_reqs[:]
            num_new_tokens_per_req = [1] * len(decode_reqs)

            if chunk_size > 0:
                all_reqs.append(prefill_req)
                num_new_tokens_per_req.append(chunk_size)
                
            disassemble_fused_cache(all_reqs, new_kvs, num_new_tokens_per_req)
            
            # 5. ── POST-PROCESSING ──
            # Handle decode requests (they are the first N rows in the batch)

            if len(decode_reqs) > 0:
                logits_decode = logits[:len(decode_reqs), -1, :]
                probs = F.softmax(logits_decode, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                for i, req in enumerate(decode_reqs):
                    req.generated_tokens.append(idx_next[i].item())
                    req._last_token = idx_next[i : i + 1]
                    if req.is_done:
                        scheduler.complete(req)
            
            if chunk_size > 0:
                prefill_req.prefill_cursor += chunk_size
            
                if prefill_req.is_fully_prefilled:
                    prefill_logits = logits[-1:, -1, :]
                    probs = F.softmax(prefill_logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    prefill_req.generated_tokens.append(idx_next.item())
                    prefill_req._last_token = idx_next
                    commit_completed_blocks(prefill_req, scheduler.block_cache, BLOCK_SIZE)
                    scheduler.promote(prefill_req)

            step += 1

    return scheduler                    
```

Let's walk through the key changes in the `interleaved_generate` scheduling loop compared to our previous chunked prefill implementation:

```python
            chunk_size = min(remaining_budget, tokens_left)
```

Just like before, we calculate the remaining budget after scheduling all active decode requests. This tells us exactly how many tokens of the prefill request we can process this step.

```python
            # 3. ── SINGLE FUSED MODEL CALL ──
            batch_tokens, batch_positions, past_kvs, attn_mask, pad_lengths = assemble_fused_batch(
                decode_reqs, 
                prefill_req if chunk_size > 0 else None, 
                chunk_size
            )

            logits, _, new_kvs = model(
                batch_tokens,
                pos=batch_positions,
                past_kvs=past_kvs,
                attn_mask=attn_mask
            )
```

**This is the core difference.** Instead of calling `model()` twice (once for decode requests and once for the prefill chunk), we use our new `assemble_fused_batch` function to package them into a single unified tensor. The model executes one single, highly efficient forward pass for both decode and prefill workloads.

```python
            ## DISASSEMBLY
            all_reqs = decode_reqs[:]
            num_new_tokens_per_req = [1] * len(decode_reqs)

            if chunk_size > 0:
                all_reqs.append(prefill_req)
                num_new_tokens_per_req.append(chunk_size)
                
            disassemble_fused_cache(all_reqs, new_kvs, num_new_tokens_per_req)
```

Because the KV caches for the decode requests (T=1) and the prefill chunk (T=chunk_size) are packed into a single padded output tensor `new_kvs`, we must carefully slice them back out. We reconstruct the exact order of requests (`all_reqs`) and how many tokens they contributed (`num_new_tokens_per_req`), and hand them off to `disassemble_fused_cache` to strip the padding and append to the individual requests' KV cache state.

```python
            if len(decode_reqs) > 0:
                logits_decode = logits[:len(decode_reqs), -1, :]
```

During assembly, we deliberately placed all decode requests at the *beginning* of the batch tensor. Therefore, to sample the next tokens for our decode requests, we simply slice out the first `len(decode_reqs)` rows from the `logits`.

```python
            if chunk_size > 0:
                prefill_req.prefill_cursor += chunk_size
            
                if prefill_req.is_fully_prefilled:
                    prefill_logits = logits[-1:, -1, :]
```

If we processed a prefill chunk and it reached the end of its prompt (`is_fully_prefilled`), we need to generate its first output token. Because the prefill request was placed at the *end* of the batch tensor during assembly, its logits are simply the very last row (`logits[-1:]`). We sample from it, update the state, and promote the request to active decoding for the next step!

# Gotchas

Here are a couple of things worth pointing out:

## Gotchas

1. **KV cache shape mismatch during assembly.** Decode requests have `T_past_i` tokens in their cache. The prefill request may have `T_past_prefill` tokens (from earlier chunks) or 0 tokens (first chunk). `assemble_batch_cache`'s padding needs to handle this varying `T_past` across rows.

2. **Causal mask interaction with padding.** When applying the causal mask within the prefill chunk, make sure the mask also zeroes out pad positions on the left. Otherwise the model attends to zero-valued pad tokens, which subtly corrupts the softmax distribution.

3. **Disassembly: different numbers of new tokens per row.** After the fused forward pass, decode row `i` produced `(T_past_i + 1)` cache entries. The prefill row produced `(T_past_prefill + chunk_size)` entries. Both need their left-padding stripped. Track `pad_lengths` per row during assembly and use them during disassembly.

4. **Logit extraction.** `logits[:, -1, :]` gives you the last-position logits for every row. For decode rows, this is the next token prediction (correct). For the prefill row, this is the logit after the last token in the chunk — which is only the "first generated token" if the chunk completes the prefill. If the chunk is partial, you don't sample from it; you just cache the KV and move on. Be careful not to sample from a partial prefill row.

5. **Empty decode batch.** When there are no active decode requests (only a prefilling request), the batch is just the prefill chunk. This degenerates to a standard prefill call. Make sure your code handles `len(decode_reqs) == 0` gracefully.

The full code can be found here:

Thanks for reading!

CZ