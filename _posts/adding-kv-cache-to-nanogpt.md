---
layout: post
title: "Adding KV Cache to NanoGPT"
date: 2026-04-18
---

*This post requires an in-depth understanding of transformers and attention mechanisms in the context of Andrej Karpathy's NanoGPT repository. 

I have been getting my feet wet in ML inference systems recently, and decided to try implementing KV caching in NanoGPT. 

Just as brief context, NanoGPT is a repository by Andrej Karpathy that implements a GPT model from scratch. 

**1. The Problem**

In a standard GPT model, the attention mechanism calculates the attention scores for all tokens in the input sequence. This is done by calculating the dot product of the query and key matrices, and then applying a softmax function to get the attention scores. 

The problem is that for long sequences, this becomes very computationally expensive. It would be wrong to calculate the key and value matrices for all the tokens every time we want to generate a new token, since we have already calculated them for the previous tokens. This is quadratic time complexity for a sequence generation. This is where the idea of a cache comes in.

**2. The Solution**

The solution is to cache the key and value matrices for each token in the input sequence. This way, we don't have to recalculate them every time we want to generate a new token. This reduces the time complexity of the attention mechanism from O(n^2) to O(n), where n is the length of the input sequence. 

**3. Implementation in NanoGPT**

In NanoGPT, we have to first identify the place where the KV Cache will live. In this case, it is at the most basic unit of the implementation, which is the `Head` class. We have defined the `Head` as the class that handles one head of self attention.

I had to ask myself several questions:

1. What data structure should we use for the KV Cache? 
2. Does Q @ K^T still work if Q has shape (B, 1, hs) and K has shape (B, T, hs)? What does that mean?
3. Does masking even make sense anymore?
4. How should the forward method deal with inference vs training?

For the KV Cache, I initially thought that it would be some sort of a hashmap, where the keys are _ and the values are _. But after thinking about it, I realized that it really is just a regular tensor of shape None initially that will hold the key entries which are just (B, 1, hs), all concatenated along the -2 axis. That way, each row will contain the key entries for a specific token. 

Now, we have to carefully consider that in the original implementation of NanoGPT, we were masking the future tokens of a sequence at a timestamp with negative infinity, which prevented the model from calculating attention scores for tokens it hadn't seen yet. But that was when we were recalculating attention for every single token in a sequence at every forward pass. 

Now that we are generating one token at a time and using a cache, it doesn't require this masking. So we can remove it!

Now, we run into trouble since we only want to calculate the KV cache values during inference. How do we prevent the caches from being populated at the wrong time? Thankfully, PyTorch implicitly has a `self.training` flag that every single submodule from `nn.Module` inherits which has a boolean value showing whether training is active or not. We can just have an if-else condition that guards the training code from the inference code like so: 

Lastly, I had to add an if else condition to the concatenation logic, since if this is the very first time we are running forward, then the key and value caches would be None, so we need to set it to the first key / value tensors that were generated. 

Now, let's write a generation function that runs during evaluation time and actually gives back the tokens that we want to see in the result!

Here is what it looks like: 
```python
def generate_kv_cache(model, idx, max_num_tokens):
    model.eval()
    clear_kv_cache(model)

    model(idx)

    with torch.no_grad():
        for step in range(max_num_tokens):
            curr_pos = step + idx.shape(1) 

            logits, _ = model(idx[:, -1:], pos=torch.tensor[curr_pos], device=device) # (B, 1, C)
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    
    return idx

```
In this function, we are setting the model to evaluation mode, and making sure to clear the kv cache for the model. 

Now, we run `model(idx)` once since that is how we prefill the KV cache before the next token is generated. Then, we have a for loop that iterates until the max number of new tokens we want, and grab the logits for the specific index, run softmax over the logits to get the probabilities, and then sample the next index. The index is added to the running sequence of indexes, which will then be decoded into the correct letters at the final step.

But what is the 

Now, let's do a shapes check to verify things:




