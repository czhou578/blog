---
layout: post
title: "Pruning Qwen3.6-35B-A3B: Fitting a second model beside DeepSeek"
date: 2026-11-25
image: https://czhou578.github.io/blog/images/pruning_qwen_thumbnail.png
---

![Pruning Qwen3.6-35B-A3B: about 100 tok/s, a fast second model in memory beside DeepSeek on one DGX Spark]({{ site.baseurl }}/images/pruning_qwen_thumbnail.png)

In my [last post]({{ site.baseurl }}{% post_url 2026-11-11-pruning-deepseek-cold-experts %}), I cut DeepSeek V4 Flash 0731 down to 69.79 GiB of weights, under the 72 GiB cap that I had set for it. The cap had one purpose: to leave room for a second model on the same DGX Spark.

DeepSeek is strong, but it decodes at about 35 tok/s on my machine. Many of my requests do not need it: a chat, a quick lookup, a coding subagent that reads files. For those, I wanted a fast model in memory at the same time, so that neither model has to stop for the other.

This post is about how I fit Qwen3.6-35B-A3B into the room that was left. It took a model search, an expert prune with a quality cost that I chose to accept, a quantized draft, and a few hidden memory costs in vLLM. In its current form, Qwen decodes at about 100 tok/s beside DeepSeek.


## The budget

Here is the DeepSeek server after the MD prune, measured on 2026-11-13:

| Part | GiB |
|---|---|
| Weights of the target | 66.54 |
| Draft model (64 experts) | 3.25 |
| KV cache, pinned, 256k context | 7.25 |
| Runtime memory of the GPU process beyond these | 5.03 |
| Host memory of the server processes | 3.9 |
| **Available beside the idle server** | **about 29** |

The DGX Spark has 121 GiB of memory, shared by the CPU and the GPU. A second model had to fit in about 29 GiB, with the weights, the KV cache and the runtime. I also wanted a margin of at least 4 GiB for my Claude Code sessions, the bot and the page cache.

## Picking the second model

My first candidate was a dense model, `nvidia/Qwen3.8-27B-NVFP4`. A test window measured it alone, with DeepSeek stopped. At a 256k context, it used 34 GiB at the peak with no draft. It decoded my bot's speed test at 11.1 tok/s with no draft, and at 30.8 tok/s with its MTP draft. Both rates are slower than DeepSeek.

The decode on this machine is bound by the bytes that each token reads. A dense model of 27B parameters reads all of its weights for each token. The fast model had to be a MoE, which reads only its few active parameters.

A second window measured two of them, each at a 256k context with one sequence:

| Point | Weights | Used at the peak | Bot decode | Against DeepSeek (35.3 tok/s) | 250k prompt |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B, no draft | 19.65 | 28.81 | 57.9 | 1.6x | 138 s |
| Qwen3.6-35B-A3B, MTP draft | 21.22 | 32.29 | 99.3 | 2.8x | 144 s |
| Nemotron 3.5 Lightning 30B-A3B, no draft | 17.82 | 25.51 | 74.1 | 2.1x | 82 s |
| Nemotron 3.5 Lightning 30B-A3B, DSpark draft | 19.37 | 28.78 | 127.2 | 3.6x | 77 s |

*Memory in GiB, decode in tok/s. Each rate is one run of 5 bot sessions, with prompts of 30,000 to 73,000 tokens.*

On these numbers, Nemotron won: it was faster, and it needed 3.3 to 3.5 GiB less. I still chose Qwen: by online consensus, it has a slight edge over Nemotron in reasoning, and it was fast enough. The price of that choice: Qwen with its draft used 32.3 GiB at 256k, 3.3 GiB more than the room that was left.

One number from this window surprised me. At every point, vLLM used 6.4 to 7.9 GiB beyond the weights and the KV cache. The dense 27B model had 6.7 GiB, in the same range. The overhead is a cost of the vLLM process on this machine, not of the model. I come back to it below.

## The gap, and the lever

A shorter context saved little. Only 10 of Qwen's 40 layers use full attention, so its KV cache is small. At 128k with its draft, Qwen used 29.9 GiB at the peak. With about 29 GiB available, that left no margin at all.

Qwen3.6-35B-A3B has 256 routed experts in each of its 40 layers, and each token uses 8 of them. The routed experts hold 16.88 GiB, 77% of the checkpoint. As on DeepSeek, the experts were the lever.

## The same method, with fewer patches

I reused the steps of my DeepSeek prune: record the experts that my own traffic uses, rank them, remove the cold ones, and gate the cut against the full model. Three parts were simpler for Qwen:

- Each expert is a set of plain NVFP4 tensors. `qwen_prune.py` copies the kept ones with no new quantization, in about 75 seconds.
- vLLM 0.33 returns the routed experts of each token with a flag, so the capture needed no server patch.
- One expert count for all layers needed no loader patch.

**The data.** The traffic of Qwen is not the traffic of my bot. Qwen mainly gets coding, web search and tool calls. The calibration data came from three sources:

| Source | Calibration | Held-out |
|---|---|---|
| My Claude Code transcripts | 33 sessions from 13 projects | 13 sessions |
| My bot's recorded sessions | 22 sessions | 18 sessions |
| Repository files | 42 files of 7 repositories | 6 files of `nanoGPT-inference`, a repository that the calibration never sees |

The transcripts can hold secrets, so a pattern replaced 18 strings, and the data stays on my machine. The capture covered 99 segments and 2.44M prompt tokens.

**The coverage.** For each layer, I computed the share of the routed slots that its top K experts carry:

| Point | Mean over the layers | Lowest layer |
|---|---|---|
| Qwen K 208 of 256 | About 96% | About 88% |
| Qwen K 192 of 256 | 93.9% | 82.8% |
| Qwen K 176 of 256 | 91.0% | 77.7% |
| DeepSeek K 160 of 216 (passed) | 95.4% | 86.5% |
| DeepSeek K 144 of 216 (failed) | 92.5% | 81.9% |

The routing of Qwen was spread much more evenly than the routing of DeepSeek, and every expert of every layer was used at least once. K 176, my first goal, kept less of the routing than the DeepSeek cut that had failed. That was the first sign that Qwen had less to give.

![Coverage of the routing against the share of experts kept, for Qwen and DeepSeek, with the pass and fail points of the NLL gate]({{ site.baseurl }}/images/qwen-deepseek-coverage.svg)

## The gate windows

Each cut ran against the full model in the same window:

| Gate | Measure | Limit |
|---|---|---|
| G1 | The full model writes 60 held-out turns, and the pruned model scores them | At most +3% summed NLL |
| G2 | The tool names and the decisions of the same turns | Tool names at least 98%; decisions down by at most 3 |
| G3 | A tool-calling suite of 85 tasks | At most 3 tasks fewer |
| G5 | The decode speed with the draft | Report |
| G6 | The peak memory at 128k with the draft | A margin of at least 4 GiB beside DeepSeek |

**A fault in vLLM.** In the first gate window, K 208 and K 192 did not boot: "IndexError: index 208 is out of bounds for dimension 0 with size 208". vLLM does not report the routing of the draft layer, so I had no rank for the draft's experts, and the draft kept all 256. But vLLM 0.33 builds the MTP draft with the config of the target, so it tried to load 256 draft experts into tensors of 208 rows. A small patch, `qwen_mtp_patch/sitecustomize.py`, gives the draft a copy of the config with its own expert count. It changes nothing when the two counts are equal.

The second window:

| Point | Loaded | G1: NLL change | G2: decisions, tool names | G3: suite | G5: decode | G6: peak |
|---|---|---|---|---|---|---|
| Full model, window 1 (the reference) | 21.22 GiB | | 60 of 60, 100% | 52 of 85 | 97.8 tok/s | 31.69 GiB |
| Full model, window 2 | 21.22 GiB | −0.10% | 60 of 60, 100% | | 98.5 tok/s | 31.14 GiB |
| K 208 | 18.09 GiB | **+4.15%** | 60 of 60, 100% | 50 of 85 | 101.2 tok/s | 28.85 GiB |
| K 192 | 17.00 GiB | **+8.04%** | 60 of 60, 100% | 51 of 85 | 101.4 tok/s | 27.55 GiB |
| K 176 | 15.98 GiB | **+11.11%** | 60 of 60, 100% | 51 of 85 | 94.2 tok/s | 26.56 GiB |

*The peaks include the memory of the test clients.*

Every cut failed G1. The full model repeated within 0.1% between the two windows, so +4.15% was about 40 times the noise. The loss was also broad: at K 208, 37 of the 60 turns got worse, and most kinds of work lost 4% to 6%, the held-out repository included.

Qwen lost about twice as much as DeepSeek at the same coverage. K 208 kept about 96% of the routing and lost 4.2%. The DeepSeek K 160 prune kept 95.4% and lost 1.9%. My reading: Qwen's experts are small (1.69 MiB each), and each token mixes 8 of them from a more even routing, so each expert carries more.

The decisions, the tool names and the tool-call suite did not move. The loss was in the wording of the turns, not in the choice of a tool.

G1 held only above about K 224, and K 224 left a margin of about 1.2 GiB. No point passed both G1 and G6, so my first decision was: no prune.

![NLL change against the share of experts kept, for Qwen and DeepSeek, with the +3% limit]({{ site.baseurl }}/images/qwen-deepseek-nll.svg)

The 1.9% of DeepSeek also includes a second cut, from 6 to 5 experts per token. With the expert cut alone, DeepSeek K 160 lost only 1.4%, so Qwen lost about three times as much at the same coverage.

## Can the vLLM overhead go?

Before I gave up on the full model, I tested the overhead of vLLM itself. Of the 29.9 GiB that the full model used at 128k, about 6.6 GiB was neither weights nor KV cache. If I could free 4 GiB of it, the full model would fit with no quality loss.

The overhead window tested four levers on the full model with its draft, each with a prompt of 120,000 tokens and the bot decode test:

| Point | Change | Peak | Long prompt | Decode |
|---|---|---|---|---|
| o0 | None | 29.64 GiB | 46.1 s | 101.8 tok/s |
| o1 | Prefill budget 512 (from 2,048) | 29.60 GiB | 59.0 s | 98.9 tok/s |
| o2 | `--enforce-eager` | 29.62 GiB | 51.6 s | 72.6 tok/s |
| o3 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 29.71 GiB | 46.1 s | 96.5 tok/s |
| o4 | o1 and o3 | 29.57 GiB | 59.6 s | 99.7 tok/s |
| k0 | K 208, with the settings of o0 | **26.51 GiB** | 45.5 s | 97.7 tok/s |

No lever moved the peak by more than 0.07 GiB, and eager mode cost 29% of the decode speed. The overhead is a fixed cost in two parts:

- **About 2.2 GiB on the GPU**, beyond the weights and the KV cache: the CUDA context, the kernel workspaces and the CUDA graphs.
- **About 4.4 GiB of host memory** in the two processes of `vllm serve`, the API server and the engine core: Python, PyTorch, the CUDA libraries and the tokenizer.

`vllm serve` always runs the engine core in its own process, so I could not merge the two. On a machine with shared memory, the host memory of the server processes comes out of the same pool as the GPU memory.

## Does +4% NLL show?

G1 measures how far a pruned model moves from the full model. It does not measure whether I would notice. I ran two more tests on K 208 to find out.

**80 rare facts, with no search.** I took questions from SimpleQA. On the 80 questions that both models answered in their time:

| | Full model | K 208 |
|---|---|---|
| Correct | 15 | 10 |
| Incorrect | 43 | 46 |
| Not attempted | 22 | 24 |

K 208 lost a third of its correct answers. Five of the 7 answers that it lost were science and technology, and some losses were confident inventions: "Dolly the sheep was cloned into three identical copies" (she gave birth to triplets). The sample was too small to prove a loss (p = 0.18), but its direction agreed with G1. The full model itself was wrong on 3 of 4 of these questions.

**A blind judge on my own work.** On 30 held-out turns, DeepSeek compared two new turns with the recorded turn of the full model: a turn of K 208, and a new sample of the full model as a control. Each pair ran in both orders. K 208 scored better than the control on 8 turns, worse on 10, and the same on 12 (p = 0.81): no difference that the judge could see.

I chose K 208, with its G1 cost. It decodes as fast as the full model, and its tool calls did not change. Its weakness is rare facts with no search, and my agents can search the web.

## A quantized draft

K 208 used 26.51 GiB at the peak: a margin of about 2.3 to 3.0 GiB. The next lever was the MTP draft. Its 256 experts were BF16, and they held 1.50 GiB.

`qwen_draft_quant.py` converts them to NVFP4 for the weights only. Speculation works the same way: the target checks every draft token, so the output does not change, and only the acceptance can fall. Three details of note:

- vLLM builds the draft with the quant config of the target. The build therefore patches the target's ignore list, and it adds a group for the draft experts.
- The MoE backend of the target, `flashinfer_cutlass`, has no weight-only mode. The draft runs its MoE on Marlin instead.
- A weight-only format needs no activation scales, so the draft needed no calibration.

The draft experts went from 1.50 GiB to 0.42 GiB.

**The first test said that it did not work.** The load fell by 1.08 GiB, but the peak fell by only 0.07 GiB. The cause was a cold compile. The patched quant config changed the graph, so torch.compile ran from scratch (17.6 s, against 0.4 s from the cache), and the compile held about 1 GiB more for a minute. A second boot with a warm cache:

| | BF16 draft | NVFP4 draft, cold compile | NVFP4 draft, warm cache |
|---|---|---|---|
| Model loading | 18.09 GiB | 17.01 GiB | 17.01 GiB |
| Peak | 26.51 GiB | 26.44 GiB | **25.42 GiB** |
| Decode | 97.65 tok/s | 97.92 tok/s | 101.78 tok/s |
| Tokens accepted per draft step | 2.24 | 2.19 | 2.31 |

With a warm cache, the NVFP4 draft saved 1.09 GiB at the same speed. Since then, I boot each new setup once alone to fill the compile cache, before I boot it beside DeepSeek.

## Both servers at once

Until then, every test had run with DeepSeek stopped. The real test was Qwen beside the live DeepSeek, with no downtime. Two scripts ran it:

- `pair_watch.sh` reads the available memory every second. Below 2 GiB, it kills Qwen, and never DeepSeek. DeepSeek also shows on the host as a vLLM process, so the watch matches the Qwen venv and port 8889, not a process name.
- `pair_run.sh` checks that at least 28,600 MB is available, boots Qwen beside DeepSeek, and runs load tests on both.

One vLLM detail helped. With a pinned KV cache (`--kv-cache-memory-bytes`), vLLM uses `--gpu-memory-utilization` only for a free-memory check at the boot. Beside DeepSeek, I set it to 0.005, so that the check passes.

The first run, text only, on 2026-11-14:

| Test | Lowest available | Result |
|---|---|---|
| Both idle | 9,852 MB | |
| Qwen: 119,745 tokens, while DeepSeek decodes | 8,220 MB | Qwen 52.7 s (45 s alone) |
| DeepSeek: 127,837 tokens, while Qwen decodes | 8,477 MB | DeepSeek 245 s |
| Bot decode test, each model alone | 7,517 MB | Qwen 99.1 tok/s, DeepSeek 36.9 tok/s |
| Bot decode test, both at once | 7,459 MB | Qwen 46.8 tok/s, DeepSeek 19.3 tok/s |

**The 7.3 GiB was not headroom.** My first reading was that Qwen cost much less memory beside DeepSeek than in the windows. That was wrong. One hour after the boot, 7.8 GiB of my 16 GiB swap file was in use, and at least 4.6 GiB of it belonged to the two servers. When Qwen booted, the kernel moved idle host pages of both servers to swap. `MemAvailable` does not fall until the kernel has swapped, so it hid the pressure. Without swap, the lowest available memory would have been about 0 to 2.7 GiB, close to my projection. Since then, every test logs the free swap next to the available memory.

**The two models share the memory bandwidth.** Each model alone keeps its speed. When both decode at once, each runs at about half speed. A router that sends one request at a time keeps the full speed.

![Available memory and swap in use during the first run of Qwen beside DeepSeek, 2026-11-14]({{ site.baseurl }}/images/qwen-pair-swap.svg)

## Vision on

My rank counted text tokens only, so it never saw the experts that image tokens use. I had to test vision before I turned it on. I rendered screenshots of code, charts, UI screens and diagrams with known answers, and DeepSeek graded each answer. K 208 answered 51 of 51 questions, as the full model did, and 15 of them were hard: a full-HD screenshot of dense code, a table of 16 rows, a chart with no value labels. Both models reached the ceiling, so this test rules out a gross loss, not a small one.

Vision cost 2.66 GiB at the peak, three times the weights of the vision tower. One part was hidden: vLLM keeps a cache of processed images, 4 GiB by default, in the API server and again in the engine core. With the default, the server can grow by up to 8 GiB as images arrive. Two settings cut the cost to 2.36 GiB and stopped the growth:

- `--mm-processor-cache-gb 0`: no cache of processed images.
- `--mm-processor-kwargs '{"max_pixels":3686400}'`: at most 2560 × 1440 pixels per image. A larger screenshot is scaled down.

The live test with vision beside DeepSeek passed: the lowest available memory was 5,827 MB, and the free swap stayed above 6.6 GiB. I left vision on.

Two hours later, the watch fired. At 19:48 on 2026-11-14, the available memory fell to 1,958 MB while new Claude Code processes started, and each of them holds about 0.3 to 0.5 GB. The watch stopped Qwen, DeepSeek stayed healthy, and no process was killed for memory. The short tests had left 5.8 GB, but two hours of normal use took the margin under 2 GiB.

## Four sequences, vision off, and a larger cache

Claude Code sends work to Qwen through subagents. With `--max-num-seqs 1`, parallel subagents waited in line. I tested more sequences:

| `--max-num-seqs` | Qwen, one stream | Qwen, all streams | Lowest available |
|---|---|---|---|
| 1 | 103.0 tok/s | | 6,578 MB |
| 2 | 98.4 tok/s | 160.6 tok/s (1.63x) | 6,661 MB |
| 4 | 100.9 tok/s | 285.6 tok/s (2.83x) | 6,214 MB |

One stream did not get slower. Each step cost about 0.3 to 0.5 GB of available memory at idle. Beside four Qwen streams, DeepSeek decoded at 15.2 tok/s, against 18.2 beside one stream and 35.7 alone. This happens only when four subagents and the bot run at the same time. I kept four sequences.

On 2026-11-15, I made two more changes:

- **The sampling.** With vLLM's default `--generation-config vllm`, the server used vLLM's own sampling defaults, not the values of the model. Qwen then sampled from its full vocabulary, and one subagent turn ran to 20,072 tokens. The server now uses `--generation-config auto` (temperature 1.0, top-p 0.95, top-k 20) plus a presence penalty of 1.5: the values of the model card for thinking mode. My gateway also caps a request to Qwen at 8,192 output tokens.
- **Vision off, and a larger KV cache.** The four sequences shared a cache of 132,987 tokens: enough for four subagents of about 30k tokens each. With more, vLLM stops a request and computes it again later. I turned vision off and gave its memory to the KV cache: 241,038 tokens (3.33 GiB, against 1.83 GiB). One request can still use at most 128k tokens, but four subagents of about 60k tokens each now fit at once.

The load test of this setup beside DeepSeek gave a lowest available memory of 5,998 MB and 95.5 tok/s on one stream.

## The current form

| Setting | Value |
|---|---|
| Model | Qwen3.6-35B-A3B NVFP4, K 208 of 256 routed experts per layer |
| Draft | MTP, 3 tokens, 256 experts in NVFP4 (weights only), MoE on Marlin |
| Vision | Off (`--language-model-only`) |
| Context of one request | 128,000 tokens |
| KV cache | 241,038 tokens, FP8, pinned, shared by up to 4 sequences |
| Sampling | The model card values, presence penalty 1.5 |
| Weights loaded | 17.01 GiB (DeepSeek: 69.79 GiB) |
| Decode, one stream | 95 to 101 tok/s |
| Lowest available memory in the load tests | 5,998 MB, with swap in use |
| Guard | The watch stops Qwen below 2 GiB available |

**How I use the two models.** A small router serves one model name, `auto`, to my Hermes agent. A prompt of more than 100k tokens, or with a code block, goes to DeepSeek. For any other prompt, Qwen sorts the request with one word, CODE or CHAT, in about 0.13 s: code goes to DeepSeek, and chat and quick lookups go to Qwen. The router keeps the answer for each user message, so a tool loop stays on one model. The sort was right on 30 of 30 tuning prompts and 20 of 20 held-out prompts. In Claude Code, two subagents send work to one model each.

## What is still open

- **The margin leans on swap.** The load tests leave about 6 GB available, but only because several GiB of idle server pages sit in swap. The watch is the guard against a hard reset.
- **A durable fix is on the DeepSeek side.** The cold cut of DeepSeek is at its limit, and no layer can go. The next lever there is a narrower draft model.

## The biggest lessons

1. **Budget the runtime before the model.** vLLM used 6.4 to 7.9 GiB beyond the weights and the KV cache with every model that I tried, and no flag freed it. On a budget of 29 GiB, that is about a quarter.
2. **Check the swap before you report a margin.** `MemAvailable` stays high while the kernel swaps idle pages. Log `SwapFree` next to it.
3. **Measure memory on a warm compile cache.** A cold torch.compile held 1 GiB more, and it almost made me reject a change that worked.
4. **Read the defaults.** Two vLLM defaults cost me: a cache of processed images of 4 GiB in each of two processes, and sampling settings that ignore the model card.
5. **A failed gate is not always a visible loss.** K 208 failed my NLL gate by a wide margin, but a blind judge saw no loss on my work. The rare facts showed the loss. Know which of the two your use needs.

Thanks for reading! If you have any questions or suggestions, feel free to reach out.

My code is in this repo: [czhou578/spark-moe-specialize](https://github.com/czhou578/spark-moe-specialize#files). [`QWEN.md`](https://github.com/czhou578/spark-moe-specialize/blob/main/QWEN.md) is the summary of the Qwen prune. The tools of this post are `qwen_data.py` (the data), `capture.py` (the capture), `qwen_prune.py` (the prune), `qwen_gate.py` and `qwen_facts.py` (the gates and the facts test), `qwen_draft_quant.py` (the NVFP4 draft), [`qwen_mtp_patch/sitecustomize.py`](https://github.com/czhou578/spark-moe-specialize/blob/main/qwen_mtp_patch/sitecustomize.py) (the draft patch for vLLM 0.33), `qwen_vision.py` (the image test), and `pair_run.sh` and `pair_watch.sh` (both servers at once).

CZ
