---
layout: post
title: "Adding Chunked Prefill to NanoGPT"
date: 2026-05-09
---

*This post requires an in-depth understanding of transformers and attention mechanisms in the context of Andrej Karpathy's NanoGPT repository. 

I have been getting my feet wet in ML inference systems recently, and decided to try implementing KV caching in NanoGPT. 

Just as brief context, NanoGPT is a repository by Andrej Karpathy that implements a GPT model from scratch, stripping away all the abstractions and optimizations to deliver the most simplistic language model that models ChatGPT.

For context, I highly recommend watching his Makemore series on YouTube and also how to build a GPT from scratch. The following will make more sense if you have a basic understanding of transformers and attention mechanisms.

## Context:

In the previous post, I talked about how I added KV caching to NanoGPT. We were able to manually add a KV cache to the `Head` class, which is responsible for calculating the attention scores for a specific token in the input sequence. We were able to reduce the time complexity of the attention mechanism from O(n^2) to O(n), where n is the length of the input sequence, as well as achiee a moderate speedup to the tok/sec generation in inference. 

In this article, we will add Chunked Prefill to NanoGPT. 

## Problem:

The problem that we now have with the current implementation is that for very long sequences, the prefill stage can hog up the GPU's and starve other requests that are coming in. If we have multiple decode and prefill requests at the same time, we don't want the decode requests to be starved. In addition, there is the risk of running out of VRAM if the sequence is too long, since it can spike the memory usage. 

## Solution:

The solution is to chunk the prefill stage into smaller chunks, and process them in parallel. 

## Steps:

We are going to make modifications to 
