---
layout: post
title: "prune-moe-kit: One pipeline from two prunes by hand"
date: 2026-11-26
image: https://czhou578.github.io/blog/images/prune_moe_kit_thumbnail.png
---

![prune-moe-kit: 1 pipeline, two prunes by hand made into one tool, and a 41-minute probe that agrees with the full gate on Inkling-2-Mini]({{ site.baseurl }}/images/prune_moe_kit_thumbnail.png)

In my last three posts, I pruned two models by hand. DeepSeek V4 Flash 0731 went from 95.39 GiB of loaded weights to 69.79 GiB, in [two]({{ site.baseurl }}{% post_url 2026-11-07-pruning-deepseek %}) [steps]({{ site.baseurl }}{% post_url 2026-11-11-pruning-deepseek-cold-experts %}). Then [Qwen3.6-35B-A3B]({{ site.baseurl }}{% post_url 2026-11-16-pruning-qwen-second-model %}) went from 21.22 GiB to 17.01 GiB. Both models now run at the same time on one DGX Spark.

Each prune was a project of its own, but the steps were the same both times: record the experts that my work uses, rank them, remove the cold ones, and test the cut against the full model.

This post is about prune-moe-kit, the tool that I made from those steps. For testing, it had to give the results of my two prunes again. I then had it prune the newest Inkling-2-Mini model.

## The results in short

- **The same bytes.** `prune-moe-kit build` made the K 208 folder of my Qwen prune again: 101,316 tensors and 18.84 GiB, equal byte for byte.
- **The same verdict.** On Qwen, the gates gave the result of the prune by hand: K 208 and K 192 fail the NLL limit, and three other gates show no loss.
- **A third model.** Inkling-2-Mini ran through the pipeline in four windows. K 112 frees 1.82 GiB for +1.97% NLL and passes the `nll` gate. K 104 fails at +4.24%.
- **A short check gives the first answer.** A probe of 41 minutes estimated K 108 as the deepest cut that passes. The full gate of 103 minutes gave K 109.

## Terms

| Term | Meaning |
|---|---|
| Routed expert | An expert that the router of a layer can choose for a token. A layer has many of them, and a token uses only a few |
| K | The number of routed experts that a cut keeps in each layer. Qwen has 256 per layer, so K 208 removes 48 |
| Depth | How far a cut goes. A deeper cut keeps a smaller K |
| Cold expert | An expert that my work almost never uses |
| Cold cut | The second step of the DeepSeek prune: it removed 486 more cold experts, a different number in each layer |
| Replay | The full model writes held-out turns of my work. Then each cut scores the same tokens |
| NLL | The negative log-likelihood per token: the mean of −ln p over the tokens of a replay. p is the probability that the model gives to the token |
| NLL change | `100 × (NLL of the cut / NLL of the full model − 1)`, in percent. The `nll` gate allows at most +3% |
| Gate | A test with a limit that a cut must pass |
| Point | One model folder that a window tests: the full model or one cut |
| Worker | One of my two live model servers: DeepSeek and Qwen |
| Window | A time in which a worker is down, so that a test server can use its memory |
| Draft model | A small model for speculative decoding: it proposes tokens, and the full model checks them |
| KV pin | The memory that a server reserves at its start for the KV cache of its context |

## What the two prunes shared

I started with a descriptive table. Each row is a stage of both prunes, and the last column shows where the work was specific to one model.

| Stage | DeepSeek V4 Flash | Qwen3.6-35B-A3B | Shared? |
|---|---|---|---|
| 1. Data | Bot sessions, 1.75M tokens, 34 segments | Transcripts, bot sessions and repositories, 2.44M tokens, 99 segments | Shared, with one data format |
| 2. Capture | A server patch | One flag of vLLM 0.33 | Specific: the server |
| 3. Rank | `capture.py --rank` | The same tool | Shared |
| 4. Screen | A mask of the router, built in a minute | Direct builds: a prune is a copy of bytes | Specific: the layout |
| 5. Prune | `prune_experts.py`, EXL3 tensors, new ids | `qwen_prune.py`, plain tensors | Specific: the layout |
| 6. Gates | Replay NLL, tool names, speed, full runs, judge, blind review | The same gates, plus a facts test and an image test | Shared |
| 7. Decision | I chose K 160, then the cold cut | I chose K 208, against the NLL gate | Mine |

Only three parts were not shared: the layout of the checkpoint, the server, and the chat template. In prune-moe-kit, these three parts are an adapter for each model, and all the other code is the same for each model.

The chat template comprised most of the real adapter work. In the capture window of Qwen, each request failed at first since the Qwen template refuses a message list with no user message, and the DeepSeek template accepts it.

The two prunes also gave rules. I wrote them down as a checklist, [`BLUEPRINT.md`](https://github.com/czhou578/prune-moe-kit/blob/main/BLUEPRINT.md). Each rule comes from a result:

- **Use the workload's own rank.** With the public rank, DeepSeek K 192 failed at +3.31%. With the rank of my bot's own traffic, K 176 cost +0.29%.
- **Measure the reference in the same window.** One replay moves by up to 0.56% between runs.
- **Read a judge result as "no loss", not as a gain.** Most wins against an old baseline come from newer data.
- **Test the speed with the draft model.** A smaller K lowered the acceptance of the DeepSeek draft, and the decode got slower.
- **Write a new folder.** A weight tool never changes its source.
- **Hold out whole sessions.** The calibration set never sees a held-out session or a held-out project.

## The design

One YAML file describes one pruning workflow. This is the file of the Qwen prune, the fast coding route:

```yaml
name: coding-fast-route
data:
  sources:
    - {kind: gateway_records, route: second-model}     # ~/.local/share/model-gateway/records
    - {kind: claude_code, projects: ["*"]}             # ~/.claude/projects
    - {kind: repo, paths: [~/Projects/model-benchmarks]}
  held_out: {share: 0.15, projects: [nanoGPT-inference]}
  min_calibration_tokens: 1000000
memory: {peak_gib: 26.5, context: 128000, margin_gib: 4}
gates:
  nll:   {limit_pct: 3.0, turns: 60}
  tools: {names_min_pct: 98, decisions_max_drop: 3}
  tasks: {sets: [dev, heldout], seeds: 3}
  judge: {turns: 30, model: deepseek}
points: auto            # or a list of K values
```

The file holds the data sources, the memory goal, and the limits of the gates. The limits come from my two prunes.

The stages are commands:

```
0  preflight  prune-moe-kit preflight    How much memory does each cut free, and can the pipeline run on this model?
1  data       prune-moe-kit data         Which recorded sessions are the workflow, and which sessions are held out?
2  capture    prune-moe-kit capture      Which experts does the model route the workflow to?
3  rank       prune-moe-kit rank         Which experts of each layer does the workflow use least?
4  screen     by hand                    Which cuts pass a short replay?
5  prune      prune-moe-kit build        Does the pruned folder hold the kept experts, byte for byte?
6  gates      prune-moe-kit gate         Does the point pass the replay, the tools, the tasks, the judge, the speed and the memory?
7  decision   by hand                    Which point do you choose?
```

By design, prune-moe-kit does not do these things:

- It prunes only the routed experts of a MoE model. It does not prune a dense model.
- It does not train or quantize.
- It does not deploy, and does not choose the cut. It ends with a pruned folder and a report.

## Level 0: the preflight

The first check reads `config.json` and the headers of the weight files. It loads no weights and runs in less than a second beside the live servers. This was its output for Inkling-2-Mini on 2026-11-24, before prune-moe-kit had any code for that model:

```
type      inkling2_loop: MoE, stacked layout, NVFP4
weights   18.78 GiB in the folder; draft 0.59 GiB
experts   128 per layer, 8 per token, in 24 layers; 4.86 MiB per expert
routed    14.59 GiB (78% of the weights)

cut       freed        of the weights
K 120      0.91 GiB    5%
K 112      1.82 GiB    10%
K 104      2.74 GiB    15%
K 96       3.65 GiB    19%

can the pipeline run?
  routed experts  yes      128 per layer, 8 per token
  prune tool      no       prune-moe-kit has no prune for the stacked layout yet
  capture         unknown  not proven for this model type: test the capture flag of its server
  mask            info     possible: the router has a bias
  draft model     info     dense: no expert count to keep; test its acceptance
  loop            info     layers 6 to 17 run 3 times: the counts of such a layer are the sum of the passes
  memory          info     the full model loads 18.78 GiB, and 4.1 GiB is available: needs a window
verdict   GO WITH WORK: prune tool, capture
```

The verdict is `GO`, `GO WITH WORK` with the parts that need work, or `NO-GO`. A dense model gets `NO-GO` and a list of the other levers.

The preflight answers one question: how much memory does each cut free? It cannot answer the second question: how many experts can the model lose before its answers get worse? The plan for the Qwen prune made a guess: "A likely result is that 192 passes and 160 fails." The guess was wrong. In the test, K 208 failed the NLL limit of +3%, and K 208 keeps 16 more experts per layer than K 192.

The files also do not hold the other levers. The server of Inkling pins 20 GiB of KV cache for a context of 262,144 tokens. A smaller context can free more memory than the deepest cut of this table.

## Three checks against my own record

My two prunes gave me results that I trust, and three checks compared the tool with them before it touched a new model.

**Check A: the numbers.** From the file headers alone, the preflight had to give the recorded sizes. For Qwen, K 208 frees 3.16 GiB (my record rounds it to 3.17). For DeepSeek, K 160 frees 21.27 GiB in 28,896 tensors, and the cold cut frees 4.29 GiB more.

**Check B: the bytes.** `prune-moe-kit build` made K 208 from the full Qwen, with the rank file of the prune by hand. I then had a script compare the result with the folder of my live server. The five weight files and the index were equal, byte for byte: 101,316 tensors, 18.84 GiB. Only `config.json` differed, in the quantization keys that I had changed later for the 4-bit draft.

**Check C: the scorer.** The new replay scorer had to send the same requests as the old one. All 121 requests were equal. Then I had both scorers run on the live Qwen, on the 60 recorded turns of my Qwen prune:

| Run | NLL per token |
|---|---|
| The record of the prune by hand | 0.3274 |
| The old scorer, on 2026-11-25 | 0.3282 |
| The new scorer, run 1 | 0.3303 |
| The new scorer, run 2 | 0.3283 |

My first limit for this check was 0.2%. Two runs of the same scorer differed by 0.62%. The limit was below the noise, so I changed the check: the new scorer must agree with the old one inside the noise of two runs.

## Level 1: the probe

The second check loads the model once in one window and does the following:

1. Boot the full model with the capture of the routed experts.
2. Capture a sample of the calibration data. The default is 400,000 tokens.
3. Record 20 held-out turns of the full model, and score them. This is the reference.
4. Rank the sample, and build two or three cuts from that rank.
5. Boot each cut, and score the same 20 turns.
6. Stop the test server, and restore the workers.

A full gate uses the whole capture and 60 turns. The probe uses a small part of the tokens and a third of the turns, so its result has more noise. The report shows that noise as a band: a 90% bootstrap interval. The probe draws a new sample of the turns 1,000 times, with replacement. Each draw keeps the two scores of a turn together: the score of the full model and the score of the cut. The band is the range from the 5% point to the 95% point of the NLL change over these draws. A cut is `pass` or `fail` only when its whole band is on one side of the limit. Each other cut is `near the limit`.

The report also fits a line to the measured cuts, by least squares: the NLL change against the share of the routing that a cut removes. The line goes through zero, because the full model has no cut and no cost. The line gives an estimate of the deepest K that holds the gate. The estimate itself is not a gate result.

## The first real windows

Each measurement needs the full model in memory. With both workers up, my machine has 3 to 8 GiB available. The full Qwen uses 29 to 31 GiB at the peak, and about 32 GiB is free with Qwen down. That leaves no margin beside DeepSeek, so each window stopped both workers.

A window runner does this work. Its rules come from the shell scripts of the two prunes:

- It stops a worker only when I name the worker on the command line, and only when the worker has no running and no waiting requests.
- The three deadlines have no defaults: the start, the last boot, and the restore.
- A boot that is not healthy in 15 minutes goes down.
- A watch reads the available memory and the free swap. Below a limit, it stops the client.
- The window always ends with the restore of each worker that it stopped.

The tests of the runner had passed, but they were dry runs: they started no server. The first real window, on 2026-11-25, found three faults:

| | Window 1 (10:37 to 11:15) | Window 2 (11:17 to 11:36) |
|---|---|---|
| Runner | The first real run. Two faults needed my help | The repaired runner. No help, exit code 0 |
| Capture | 9 segments, 405,534 tokens, 40 layers, no error | The capture and the turns of window 1 |
| K 224 | Lost: the old server still answered at its boot | +5.23% NLL (band +2.34% to +8.46%): near the limit |
| K 208 | +8.32% (band +3.64% to +13.26%): fail | |
| K 192 | +10.66% (band +4.12% to +16.49%): fail | |
| Restore | DeepSeek at once. Qwen at the second start, by hand | Both at the first try |

- **A server that stops still answers.** A vLLM server ignores SIGTERM while it loads its weights, and the stop itself needs time. The health check of the next cut got its answer from the old server, and the result of K 224 was lost. The runner now stops the whole process group, and it waits until the health URL is silent.
- **The page cache blocked the restore.** The start of Qwen failed with a CUDA out-of-memory error. The page cache of the new cut folders held 25 GiB. On a machine with unified memory, a server start needs that memory free. The runner now asks the kernel to drop the cached pages of its folders before each boot and before the restore.
- **`sh` is not bash.** The stop command of Qwen was not valid in `sh`. For this reason, the first try of window 1 failed safely, before DeepSeek stopped.

The numbers of the probe were higher than the numbers of my full gate by hand, which gave +4.15% at K 208 and +8.04% at K 192. But both recorded values are inside the bands of the probe, and the verdict is the same: K 208 and K 192 fail.

The rank of the sample was good. At K 208, 87% of its dropped experts were also dropped by the full rank. Most of the difference came from the 20 turns, not from the smaller capture.

## All the gates in one window

A third window (14:48 to 16:05) ran each gate on the full Qwen, on K 208 and on K 192. It used the rank of the prune by hand and 30 held-out turns. The runner needed no help.

| Gate | Full model | K 208 | K 192 | The prune by hand |
|---|---|---|---|---|
| `nll` (limit +3%) | Reference | +5.04% (band +3.53% to +6.74%): fail | +6.36% (band +4.67% to +7.97%): fail | +4.15% and +8.04%, with 60 turns |
| `tools` | Reference | Pass | Pass | |
| `tasks` (85 tasks, at most 3 fewer) | 54 | 52: pass | 52: pass | 52, 50 and 51 |
| `judge` (30 blind pairs, DeepSeek) | Reference | Better 10, worse 9, equal 11 (p = 1.00): no loss | Better 12, worse 10, equal 8 (p = 0.83): no loss | |
| `speed` (no limit) | 110.2 tok/s | 104.4 tok/s | 113.2 tok/s | |
| `memory` (goal 26.5 GiB) | 30.9 GiB | 28.0 GiB: fail | 26.4 GiB: pass | |

The bands of the `nll` gate are the bootstrap intervals of the probe, here with 30 turns. The p values of the `judge` gate come from a two-sided sign test on the pairs that are not equal.

The result of my Qwen post holds: both cuts fail the `nll` gate, and the gates `tools`, `tasks` and `judge` show no loss. That is the case in which I chose K 208, and the report has a name for it:

| Verdict | Meaning |
|---|---|
| `pass` | Each gate that has a result passes, and `nll` passes |
| `your decision` | Only `nll` does not pass, and `tasks` and `judge` pass |
| `fail` | A gate of `tools`, `tasks`, `memory` or `judge` fails |
| `open` | `nll` does not pass, and `tasks` or `judge` has no result yet |

Two gates taught me something about the gates themselves:

- **The `speed` gate showed no change.** The three prompts of one point differed by up to 20%, and the points differed by less. The gate now has eight prompts, and it shows the change for the same prompts.
- **The `memory` gate depends on the test server.** K 208 failed the goal with 28.0 GiB. The cause was the draft model of the test server: it was the MTP draft of the full snapshot, not my 4-bit draft. The test server also had the larger KV pin of the live server. A window that must decide the `memory` gate needs the flags of the server that will run the cut.

Each point needed 24 minutes. The window stopped Qwen for 77 minutes and DeepSeek for 74 minutes. The judge ran on DeepSeek after the window, for about 20 minutes per cut.

## The third model: Inkling-2-Mini

The real test of a general tool is a model that it was not made from. I chose Inkling-2-Mini because it was on my disk, and because it differs from both earlier models:

| | Qwen3.6-35B-A3B | Inkling-2-Mini |
|---|---|---|
| Layers | 40 | 24, and the middle 12 run three times |
| Experts per layer, per token | 256, 8 | 128, 8 |
| Layout | One tensor set per expert | One stacked tensor set per layer |
| Server | vLLM 0.33 in a venv | Its own vLLM Docker image |
| Draft | MTP with experts of its own | Dense MTP, in its own file |

Before the first window, I read the vLLM code in its server image, and I ran its chat renderer on the CPU. This gave three facts for the adapter with no downtime:

- The server gives one capture row for each pass of a looped layer: 48 rows for 24 layers.
- It renders a chat with its own code.
- It accepts all 154 segments of my coding data.

Then four windows stopped both workers, because Inkling needs 45 GiB with its KV pin:

| Window | Downtime | Result |
|---|---|---|
| 1 | 17:08 to 17:15 | No boot: the capture flag of this server needs the prefix cache, and my profile turned it off |
| 2 | 17:16 to 17:25 | No boot: 128,000 tokens need a KV pin of 15.3 GiB, and my profile gave 12 GiB |
| 3 | 17:25 to 17:55 | The full model ran each step. No cut loaded: the prune wrote the draft tensors into the weight index |
| 4 | 17:56 to 20:26 | 8 points with no fault: the probe, then the full gate, with no restore between them |

Three windows failed. The runner restored both workers at the first try after each one, so a failed try was cheap and safe.

Window 4 ran the probe and the full gate on the same three cuts:

![The probe against the full gate on Inkling-2-Mini: the NLL change of three cuts, with the 90% band of each value and the +3% limit]({{ site.baseurl }}/images/prune-moe-kit-probe-full-gate.svg)

| Cut | Freed | Probe: 450,383 tokens, 20 turns | Band (90%) of the probe | Full gate: 3,974,504 tokens, 60 turns | Gates of the full gate |
|---|---|---|---|---|---|
| K 120 | 0.91 GiB | +0.72% | +0.24% to +1.21% | +0.69% | `nll` and `tools` pass |
| K 112 | 1.82 GiB | +2.01% | +1.14% to +3.29% | +1.97% | `nll` and `tools` pass |
| K 104 | 2.74 GiB | +3.75% | +2.42% to +5.53% | +4.24% | `nll` fails, `tools` passes |

**The probe gave the same answer as the full gate.** Each value of the full gate is inside the band of the probe. The probe estimated K 108 as the deepest cut that holds the `nll` gate, and the line of the full gate gave K 109. The probe needed 41 minutes, and the full gate needed 103 minutes.

**The passes of a looped layer use different experts.** A looped layer runs three times with the same weights, so one keep list must serve each pass. The three passes shared only 52% of their drop lists. I ranked the experts by the sum of the passes, and this cost each pass little. At K 112, a pass lost 1.99% of its routing with the rank of the sum, and 1.55% with its own rank. The worst pass lost 4.43%.

**The depth was a surprise again.** A token uses 6.25% of a layer of Inkling (8 of 128), and 3.1% of a layer of Qwen (8 of 256). My open question was whether this share predicts the depth. The +3% limit is at about 15% of the experts for Inkling, and at about 12% to 15% for Qwen. For these two models, the answer is no.

**This result has limits.** 13 of the 60 turns stopped at the limit of 800 tokens, inside the reasoning, and such a turn has no decision to score. The gates `tasks`, `speed` and `judge` did not run on Inkling, and the test server had no draft model. I also did not deploy a pruned Inkling: the run was a test of the tool.

After window 4, the preflight of Inkling gives `GO`: its lines `prune tool` and `capture` now say `yes`.

## What is still open

- **prune-moe-kit does not build a DeepSeek prune.** The preflight reads the EXL3 checkpoint of DeepSeek, but `prune-moe-kit build` works on safetensors checkpoints only. The DeepSeek tools stay in their own repo.
- **The last repairs did not run in a real window.** After the windows, I changed the runner so that Ctrl-C also restores the workers. I also gave the probe an option that adds the full gate to the same window. The report gives the Inkling numbers again from the files, but no real window ran with these changes.
- **Two stages are still by hand:** the screen and the decision.
- **The estimate of the depth comes from three models.** Each model that runs through prune-moe-kit adds a data point.
- **The test was not cheap.** The windows of 2026-11-25 stopped my two workers for about five and a half hours in total.
- **A general tool costs more than one prune by hand.** If I prune only one more model, the checklist and the preflight give most of the value. The rest pays off only from the model after that.

## The biggest lessons

1. **Write the method down before the tool.** The table of the shared stages showed that only three parts depend on the model. The checklist is useful even with no tool.
2. **Test a new tool against a result that you already have.** A byte-for-byte copy of my live folder was a stronger test than any unit test.
3. **Measure the noise before you set a limit.** My first limit for the scorer was about a third of the noise of two runs.
4. **Give a short check a band.** With a band, 20 turns gave the same answer as 60 turns, and the report said when it could not decide.
5. **The first real run finds the faults that the tests do not.** A server that still answers, a full page cache and a wrong shell did not show in the dry runs. The restore must work when everything else fails.
6. **Do not predict the depth.** My forecast was wrong for Qwen, and the one rule that I tested failed on Inkling.

Thanks for reading! If you have any questions or suggestions, feel free to reach out.

The code of prune-moe-kit is in this repo: [czhou578/prune-moe-kit](https://github.com/czhou578/prune-moe-kit).

The code of the two prunes by hand is in this repo: [czhou578/spark-moe-specialize](https://github.com/czhou578/spark-moe-specialize#files).

CZ
