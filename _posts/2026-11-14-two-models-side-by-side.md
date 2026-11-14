---
layout: post
title: "Two Local Models Side By Side On One DGX Spark: DeepSeek V4 Flash And Qwen3.6-35B-A3B"
date: 2026-11-14
---

In my [first pruning post]({{ site.baseurl }}{% post_url 2026-11-07-pruning-deepseek %}), I cut DeepSeek V4 Flash 0731 down so that a second model could run beside it on my single DGX Spark. After one more round of cuts, the live DeepSeek server loads 69.79 GiB and leaves about 29 GiB of memory free.

This post is about the second model. I wanted Qwen3.6-35B-A3B with its own draft model and a 128k context, as the fast model of my router. It needed 29.6 GiB at its peak, and I had 28.8 to 29.5 GiB. It did not fit.

It runs now. Both servers are up at the same time: DeepSeek at 37 tok/s for hard reasoning, and Qwen at 99 tok/s for everything else. To get there, I pruned Qwen, patched vLLM, and quantized the draft model's experts. Several ideas did not work, and this post covers those too.

![Peak memory of Qwen in three steps, against the memory free beside DeepSeek]({{ site.baseurl }}/images/qwen-memory-steps.svg)

*The peak memory of Qwen alone, with a 120k-token prompt. The red band is the memory that is free beside the live DeepSeek server. The gap between a bar and the band is the margin. The full model has none. The overhead is vLLM's own memory: I could not shrink it.*

The work took one evening and the next day: eight test windows with DeepSeek stopped, 5 hours and 44 minutes of downtime in all.

| Window | Time | DeepSeek downtime |
|---|---|---|
| The capture | 2026-11-13, 20:20 to 21:39 | 79 min |
| Gate window 1 | 21:50 to 22:23 | 33 min |
| Gate window 2 | 22:30 to 23:56 | 85 min |
| vLLM's overhead | 2026-11-14, 00:08 to 00:56 | 48 min |
| Facts and side-by-side | 09:48 to 10:59 | 70 min |
| The floor of my own engine | 13:21 to 13:29 | 7 min |
| The 4-bit draft | 13:48 to 13:57 | 9 min |
| The 4-bit draft again | 14:02 to 14:15 | 13 min |
| Both servers together | 14:25 to 14:48 | None |

## The second model

Here is the model, read from its files:

| Item | Value |
|---|---|
| Model | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`: a MoE with about 3 B active parameters |
| Layers | 40: 30 Gated DeltaNet layers, and full attention in every fourth layer |
| **Routed experts (K)** | **256 per layer** |
| **Experts per token (top-k)** | **8 routed + 1 shared** |
| One expert | 1.69 MiB in NVFP4 (4 bits), in 12 tensors |
| Size on disk | 22.0 GiB, of which the routed experts are 18.1 GB |
| Draft model | One MTP layer with its own 256 experts, stored in BF16: 1.5 GiB |
| KV cache | About 10 KB per token: only 10 of the 40 layers use attention |
| Server | vLLM 0.33, one sequence, a 128k context |

I picked it after two earlier test windows. Qwen 27B did not fit at all, and it decoded slower than DeepSeek. Qwen3.6-35B-A3B decoded at 58.8 tok/s alone and at 98.2 tok/s with its MTP draft, which got 2.24 tokens accepted per step. An `ngram` draft, which needs no weights, reached only 62.7 tok/s. So I wanted the MTP draft, and the MTP draft made the memory problem worse: with it, the model used 29.9 GiB at the peak.

The plan was the one that had worked on DeepSeek: find the experts that my own work rarely routes to, and cut them from the files.

## The gates

Before I cut anything, I wrote the gates. A pruned model had to pass all of them against the full model, measured in the same window:

| Gate | Test | Limit |
|---|---|---|
| G1 | The NLL of 60 held-out turns that the full model wrote | At most +3% |
| G2 | On the same turns: the tool names and the first answer token | Tool names at least 98%; decisions down by no more than 3 |
| G3 | A tool-call suite of 85 tasks | No more than 3 tasks below the full model |
| G4 | A smoke test: a short request and a tool call | Pass |
| G5 | The decode speed with the draft | Report |
| G6 | The memory beside DeepSeek | A margin of at least 4 GiB |

G1 came from my DeepSeek work. A broad NLL loss shows up in long sessions before it shows up in a short test suite, so it went first.

## Calibration data from my own work

On DeepSeek, I ranked the experts by the traffic of my stock-research bot. The second model has a wider job: coding, web search and tool calls. So I built the calibration set from three sources:

- my Claude Code transcripts,
- the sessions of my bot,
- eight of my own repositories.

The transcripts needed work. My agents will run under Hermes, so the tool rewrote each Claude Code tool call as its Hermes equivalent: `Bash` became `terminal`, `Read` became `read_file`, `Edit` became `patch`, and so on. It replaced every key-like string with `[REDACTED]`, and it cut each session into segments of about 160k characters that start with a user message.

I held out 15% of the sessions, and one whole repository, for the gates. The data never left the machine.

## The capture window

vLLM can return the routing of every token: the flag `--enable-return-routed-experts` adds an array of shape (tokens, 40 layers, 8 experts) to each reply. The capture sent every calibration segment through the full model and counted how often each expert was picked.

The first run failed on every request. The Qwen chat template refuses a message list with no user message, and my capture tokenized the shared prefix as a system message alone. DeepSeek's template had accepted that. I added a dummy user message and ran the capture again on the same boot.

The result: 99 segments and 2.44 M counted tokens. I also let Qwen write 256 tokens after each segment, so that the rank saw Qwen's own style and not only the style of the transcripts.

Then I looked at the coverage: the share of the routing that the top K experts of a layer carry.

| Point | Mean over the layers | Lowest layer |
|---|---|---|
| Qwen, K 208 of 256 | About 96% | About 88% |
| Qwen, K 192 | 93.9% | 82.8% |
| Qwen, K 176 | 91.0% | 77.7% |
| DeepSeek, K 160 of 216 (my live prune: +1.9% NLL) | 95.4% | 86.5% |
| DeepSeek, K 144 (failed: over +3%) | 92.5% | 81.9% |

This was a warning. Qwen spreads my traffic more evenly than DeepSeek does: every expert of every layer was used at least once. K 176, my first target, kept less than the DeepSeek point that had failed. So I added K 208 as a safer point.

The capture had one more gap: vLLM does not report the routing of the draft layer. I had no rank for the draft's 256 experts, so I decided to keep the draft whole.

## The prune tool

The checkpoint stores each expert as its own 12 tensors, so the prune is a byte copy, as on DeepSeek. For each layer, the tool keeps the top K experts, gives them the ids 0 to K − 1, cuts the same rows out of the router, and writes the new count into the config. It does no new quantization, and a check compares every kept tensor with its source, byte for byte. One build takes about 75 seconds.

## Gate window 1: the boot fails

The full model gave the reference: an NLL of 0.3144 per token, 60 of 60 decisions, 52 of 85 tool-call tasks, and 97.8 tok/s.

Then K 208 did not boot:

```
IndexError: index 208 is out of bounds for dimension 0 with size 208
```

I had pointed the draft at the original folder, with the `model` field of vLLM's speculative config. But vLLM 0.33 builds the draft with the config of the *target*. The field changes only where the weights come from. So vLLM built a draft with 208 experts and then tried to load 256 into it.

The fix is a small patch that I load through `sitecustomize.py` on the server's `PYTHONPATH`. It wraps the constructor of the draft class: when the draft folder has another expert count than the target, the draft gets a copy of the config with its own count. It changes nothing else.

## Gate window 2: every size fails

With the patch, all three sizes booted with the full draft.

![NLL change of Qwen against the number of experts kept]({{ site.baseurl }}/images/qwen-prune-nll.svg)

| Point | NLL change (limit +3%) | Decisions, tool names | Tool-call suite | Speed with the draft | Peak |
|---|---|---|---|---|---|
| K 256 (full) | | 60 of 60, 100% | 52 of 85 | 97.8 tok/s | 31.69 GiB |
| K 256 again | −0.10% | 60 of 60, 100% | | 98.5 tok/s | 31.14 GiB |
| K 208 | **+4.15%** | 60 of 60, 100% | 50 of 85 | 101.2 tok/s | 28.85 GiB |
| K 192 | **+8.04%** | 60 of 60, 100% | 51 of 85 | 101.4 tok/s | 27.55 GiB |
| K 176 | **+11.11%** | 60 of 60, 100% | 51 of 85 | 94.2 tok/s | 26.56 GiB |

*The peaks here include the test clients. The clean peak of the full model is 29.6 GiB.*

Every size failed G1. Three things stood out:

- **The loss was real.** I scored the full model a second time at the end of the window. It moved by 0.10%.
- **The loss was broad.** At K 208, 37 of the 60 turns got worse, and most kinds of work lost 4% to 6%, the held-out repository included. No single weak area caused it, so better calibration data would not fix it.
- **Qwen loses about twice as much as DeepSeek.** K 208 keeps 96% of the routing and loses 4.15%. My DeepSeek prune keeps 95.4% and loses 1.9%. Qwen is a smaller model, and it mixes 8 experts per token from an even routing, so each expert carries more.

The tool choices did not move at all. The loss was in the wording of Qwen's turns.

The gate held only above about K 224, and K 224 would have left a margin of about 1 GiB. No size passed both G1 and G6. By my own rules, the answer was: no prune.

## Is vLLM's own memory the problem?

The prune was never the biggest cost. Of the 29.6 GiB, 21.2 GiB were weights and 1.8 GiB was the KV cache. The other 6.6 GiB was vLLM itself. If I could free 3 GiB there, the full model would fit with no quality loss.

One window tested four levers on the full model:

| Setup | Peak | Decode |
|---|---|---|
| Normal settings | 29.64 GiB | 101.8 tok/s |
| A smaller prefill budget (512 tokens) | 29.60 GiB | 98.9 tok/s |
| Eager mode (no CUDA graphs, no `torch.compile`) | 29.62 GiB | 72.6 tok/s |
| The allocator's expandable segments | 29.71 GiB | 96.5 tok/s |
| Smaller budget + expandable segments | 29.57 GiB | 99.7 tok/s |

Nothing moved the peak by more than 0.07 GiB. The overhead is a fixed cost in two parts:

- **About 2.2 GiB on the GPU:** the CUDA context, the kernel workspaces and the CUDA graphs.
- **About 4.4 GiB of host memory:** `vllm serve` always starts two processes. The API server held 1.9 GB and the engine held 3.1 GB, for Python, PyTorch, the CUDA libraries and the tokenizer.

## How bad is +4.15%?

K 208 had failed only one gate, and that gate was a number that I had set myself. Before I loosened it, I wanted to know what the loss looks like. The earlier tests measured no facts, so I ran two more.

**Rare facts.** Both models answered questions from SimpleQA, a public set of short questions with one known answer, with no web search. DeepSeek graded the answers. On the 80 questions that both models answered:

| | Full model | K 208 |
|---|---|---|
| Correct | 15 | 10 |
| Wrong (a hallucination) | 43 | 46 |
| No answer | 22 | 24 |

The extra hallucinations (+3) were inside the noise. The clearer loss was knowledge: the full model knew 7 answers that K 208 missed, and K 208 knew 2 that the full model missed. Some of the lost answers were confident inventions. K 208 said that Dolly the sheep "was cloned into three identical copies"; she gave birth to triplets. Still, the full model was wrong on 3 of every 4 rare facts that it tried. Neither model should answer such questions without a search.

**My own work.** Each model wrote a new reply to 30 held-out turns. DeepSeek judged each reply blind against the full model's recorded reply, in both orders.

| New reply from | Better | Worse | Tie |
|---|---|---|---|
| The full model, a second sample (the noise) | 28 | 17 | 14 |
| K 208 | 24 | 24 | 11 |

Turn by turn, K 208 did better than the full model on 8 turns, worse on 10, and the same on 12. The judge could not tell them apart.

So for coding, tool calls and research with a search, the loss was not visible. For stored knowledge of rare things, K 208 knows about a third less. I accepted that, and I chose K 208.

## A side trip: my own inference engine

I also have my own inference engine, colinfer, and I wondered whether a port of Qwen to it would save the memory that vLLM wastes. Before 200 hours of work, I measured the engine's fixed memory with the model that it already serves.

It came out at 5.2 GB: less than vLLM's 6.6 GiB, but an empty process already used 2.5 GB. With every cut that I could name, the full Qwen would have landed at about the memory of K 208 under vLLM. That is worth having for the quality, but it would not give me the margin. I put it aside.

## Quantize the draft's experts

K 208 with the draft used 26.5 GiB: a margin of 2.3 to 3.0 GiB, still short of my 4 GiB. But one part of the model was still in 16 bits: the 256 experts of the draft, 1.5 GiB in BF16.

Speculative decoding in vLLM is exact: the target checks every draft token. So a worse draft can never change the output. It can only get fewer tokens accepted. That made the draft the safest thing to quantize.

The tool rewrote the draft's experts in NVFP4 with the scales and the packing of the `compressed-tensors` library: 1.50 GiB became 0.42 GiB. Three details of vLLM mattered:

- **The target's quant config decides.** vLLM builds the draft with the quantization config of the target folder, so the patch of the config goes into the target, not into the draft folder. The checkpoint's config ignores the whole draft; mine ignores everything in the draft but its experts.
- **Weights only.** The target's experts quantize their inputs too, and that needs calibrated scales. For the draft I used weight-only NVFP4, which needs none.
- **Another kernel.** The target's MoE backend has no weight-only mode, so the draft runs its experts on Marlin: `"moe_backend": "marlin"` in the speculative config.

The first test was a disappointment. The model loaded 1.08 GiB less, the speed held, and the peak fell by only 0.07 GiB.

I guessed that the allocator kept freed blocks, and I wrote a hook to release them. The second window proved the guess wrong: the hook never ran, because it patched a class that vLLM 0.33 does not use in this setup, and the memory fell anyway. The real cause was a cold compile. My new config had changed the model graph, so the first boot ran `torch.compile` for 17.6 seconds, and the compile held about 1 GiB. The second boot loaded the compiled graph from the cache in 0.4 seconds.

| | BF16 draft | NVFP4 draft, warm cache |
|---|---|---|
| Peak, 120k-token prompt | 26.51 GiB | **25.42 GiB** |
| Decode with the draft | 97.65 tok/s | 101.78 tok/s |
| Tokens accepted per draft step | 2.24 | 2.31 |

The 4-bit draft cost nothing in speed and freed 1.09 GiB. The margin was now 3.4 to 4.1 GiB.

## Both servers at the same time

Out of memory, the Spark can do a hard reset. So the last step had its own guard: a watch that reads the available memory every second and kills Qwen below 2 GiB. It kills only the process group whose command line is the vLLM 0.33 server on port 8889. DeepSeek also shows on the host as a vLLM process, so a match by name would have been unsafe.

The boot needed one trick. vLLM checks the free GPU memory at boot, and on this machine that number leaves out the page cache: torch saw about 1 GiB free while 28 GiB was available. With a pinned KV cache, vLLM uses `--gpu-memory-utilization` only for that check, so I set it to 0.005.

Qwen booted beside DeepSeek in three minutes. Then six tests ran:

| Test | Lowest available memory | Result |
|---|---|---|
| Both idle | 9,852 MB | |
| Qwen: a 120k-token prompt, while DeepSeek decodes | 8,220 MB | 52.7 s (45 s alone) |
| DeepSeek: a 128k-token prompt, while Qwen decodes | 8,477 MB | 245 s |
| Decode test, Qwen alone | 8,327 MB | 99.1 tok/s |
| Decode test, DeepSeek alone | 7,517 MB | 36.9 tok/s |
| Decode test, both at once | 7,459 MB | Qwen 46.8 tok/s, DeepSeek 19.3 tok/s |

The watch never fired, both servers stayed healthy, and Linux killed no process.

Two notes on these numbers.

**The speed.** A model that decodes alone keeps its full speed. When both decode at once, they share the memory bandwidth, and each drops to about half. A router that sends one request at a time to one model keeps the full speed.

**The memory.** My first reading was that 7.3 GiB stayed free, twice my projection. That was wrong. An hour later I looked at the swap: 7.8 GiB of my 16 GiB swap file was in use, and at least 4.6 GiB of that belonged to the two servers. When Qwen booted, the kernel moved idle host pages of both servers to swap, and the available memory stayed high. Without the swap, the lowest point would have been about 0 to 2.7 GiB, which matches the projection. No speed loss showed, but the available memory alone hides this, so I now log the swap as well.

I had also planned to cap DeepSeek's context at 160k for more room. I skipped it. I had written that the cap frees 1.1 GiB, and that number was stale: it was the saving against an older KV setting. Against the live setting, it frees about 0.35 GiB. That was not worth the loss of DeepSeek's 256k context.

## Where things stand

| | DeepSeek V4 Flash | Qwen3.6-35B-A3B |
|---|---|---|
| Job | Hard reasoning | Fast replies, coding, tool calls |
| Experts per layer | 140 to 160 of 216 | 208 of 256 |
| Draft model | DSpark, 64 experts | MTP, 256 experts in NVFP4 |
| Context | 256k | 128k |
| Weights loaded | 69.79 GiB | 17.01 GiB |
| Decode alone | 36.9 tok/s | 99.1 tok/s |
| Port | 8888 | 8889 |

What comes next:

- **A 24-hour gate.** Both servers stay up through a day of normal use, with the watch on. It passes if the watch never fires, no process is killed, and the swap stays under control.
- **The router.** Nothing sends requests to Qwen yet. The router is the next project.
- **The swap.** I want to know how much of the swapped memory comes back under real traffic, and whether a floor on the free swap belongs in the watch.

## The biggest lessons

1. **A proxy from one model does not transfer to the next.** 96% routing coverage passed on DeepSeek with room to spare. On Qwen it failed. Measure the new model; do not carry the threshold over.
2. **Measure a fixed cost before you tune it.** Four settings and 48 minutes showed that vLLM's 6.6 GiB does not move. That closed the question, and it sent me to the weights that I could still shrink.
3. **Quantize first what cannot change the output.** Exact speculation makes the draft model a free target: a worse draft costs speed at most, never quality. Here it cost neither.
4. **Boot twice before you trust a memory number.** A cold compile held 1 GiB and almost made me drop a change that worked. My explanation for it was wrong, too. The repeat showed both.
5. **Available memory hides the swap.** The run looked twice as good as the projection. A result that beats the arithmetic by that much needs a second look before it goes into a table.
6. **Check where your own numbers come from.** I planned around a 1.1 GiB saving that was really 0.35 GiB, because I copied it from an older plan.

Thanks for reading! If you have any questions or suggestions, feel free to reach out.

My code is in this repo: [czhou578/dsv4-specialize](https://github.com/czhou578/dsv4-specialize#files). The plan with every number is [`plans/PLAN_QWEN_PRUNE.md`](https://github.com/czhou578/dsv4-specialize/blob/main/plans/PLAN_QWEN_PRUNE.md). The tools of this post are `qwen_data.py` (the calibration data), `capture.py` (the routing counts), `qwen_prune.py` (the prune), `qwen_gate.py` (the replay gate), `qwen_facts.py` (the facts and the side-by-side test), `qwen_draft_quant.py` (the 4-bit draft) and [`qwen_mtp_patch/sitecustomize.py`](https://github.com/czhou578/dsv4-specialize/blob/main/qwen_mtp_patch/sitecustomize.py) (the vLLM patch). The window script is `qwen_fit.sh`, and the scripts for both servers are `pair_watch.sh` and `pair_run.sh`.

CZ
