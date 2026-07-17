---
layout: post
title: "Developing my own Benchmark!"
date: 2026-07-20
---

In this post, I would like to explain how I was able to create my own custom benchmark for personal open source model testing. It was a bunch of twists and turns in order to produce something comprehensive on both hardware and software wise. 

Since I got my DGX Spark, I've been downloading and trying out various open source models on my machine, but testing each one out of the box is time intensive and can waste a lot of time. My coding workflows are quite diverse and it can be difficult to accurately measure the true performance in terms of hardware utilization and decode speed from just blind testing. 

This is why I decided to invest in building my own testing harness that takes inspiration from other benchmarks out there. 

## Folder and Architecture

I did long sessions with ChatGPT and other cloud LLM's to map out an ideal architecture and folder structure for my benchmarking harness.

Eventually, I settled on this:

# High-level Architecture

```text

benchmark-suite/
│
├── models/
│   ├── qwen3-35b.yaml
│   ├── llama4.yaml
│   └── ...
│
├── benchmarks/
│   ├── latency/
│   ├── coding/
│   ├── reasoning/
│   ├── swe/
│   ├── hle/
│   └── system/
│
├── datasets/
│   ├── humaneval/
│   ├── mbpp/
│   ├── swebench/
│   ├── hle/
│   └── prompts/
│
├── runners/
│   ├── vllm_runner.py
│   ├── llama_runner.py
│   ├── sglang_runner.py
│   └── api_runner.py
│
├── metrics/
│   ├── latency.py
│   ├── memory.py
│   ├── power.py
│   ├── code.py
│   └── reasoning.py
│
├── reports/
│
└── benchmark.py
```

The benchmarking will be driven by config (.yaml) files. The yaml files will define the model name, metadata, testing prompt lengths, and the actual prompts that are going to be launched. The goal of doing this is to make it as model agnostic as possible. 

If there is a new model that needs to be added, it is just one config file and run the `benchmark.py` file. The results of the run will populate a subfolder in the `results/` folder with a collection of `.json` files that contain the metrics.

### What To Measure

The following metrics need to be measured for our benchmark:

First-token latency (TTFT) 

-  This is defined as the time between ```POST /generate ``` and ```first generated token```. We are going to use multiple prompt lengths from 32 to 16384 tokens. We want a final graph between prompt length and latency. This will reveal the KV-cache behavior. 

Decode speed 

- This is defined as ```Input tokens / prefill time``` to give tokens/sec. 

Output

```
tokens/sec
```

prompt processing speed 
reasoning token count 
code correctness 
memory usage 
power consumption

We want a way to measure how fast the models can respond with tokens while also measuring the backend performance and hardware utilization. Ideally, it will give us a bigger picture of how the hardware and software are coexisting together within this system. 

We also have to make sure we standardize the environment, which includes the CUDA version, PyTorch version, CPU, RAM, GPU, etc. Benchmarks are supposed to be replicable and having constantly changing hardware would make running them useless.




