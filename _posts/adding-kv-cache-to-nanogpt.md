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


