---
layout: post
title: "Benchmarking Speculative Decoding"
date: 2026-05-29
---

Previously, for all of the posts in the series, we have been focused on implementing optimizations to NanoGPT.

But we haven't done a comprehensive testing and benchmarking article / suite for this massive upgrade. 

In this article, which will be a part of a series, we will progressively benchmark and measure the impact of each optimization.

## Setup

We want to create a BenchMarkConfig class, which will hold all the configuration for our benchmarks.

```python

class BenchMarkConfig:
    def __init__(self):
        self.prompt_tokens = 128
        self.max_new_tokens = 256
        self.K = 4 #speculative
        self.batch_size = 4
        self.chunk_size = 16
        self.continuous_batching = True
        self.prefix_caching = True
        self.paged_kv = True
        self.priority_scheduling = True
        self.target_quantization = 'fp32'
        self.draft_quantization = 'fp32'
        self.token_budget = 16
        self.schedule = 'FCFS'
        self.block_size = 16
        self.kv_cache_size = 128

    def get_config(self):
        return self.__dict__    

class BenchmarkMetrics:
    def __init__(self):
        self.total_wall_time = 0
        self.total_generated_tokens = 0
        self.tokens_per_sec = 0
        self.request_latency = 0
        self.time_to_first_token = 0
        self.inter_token_latency = 0
        self.prefill_tokens_per_sec = 0
        self.decode_tokens_per_sec = 0
        self.batch_size_per_step = 0
        self.tokens_per_forward = 0
        self.kv_cache_tokens = 0
        self.allocated_kv_blocks = 0
        self.prefix_cache_hits = 0
        self.speculative_acceptance_rate = 0
        self.target_forwards_avoided = 0

    def __str__(self):
        return f"BenchmarkMetrics({self.__dict__})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict__

```

## Baseline Generation vs KV Cached Generation

For this, we want to 


## Single Request vs Continuous Batching

## No Prefix Cache vs Prefix Cache