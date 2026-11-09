---
layout: post
title: "Pruning DeepSeek: A Comprehensive Guide"
date: 2026-11-07
---

I have been working with DeepSeek v1 Flash to run my stock-research bot on my single DGX Spark. 

I have always wanted to create a model router that can intelligently select the best model for a given task. My idea was to use DeepSeek to serve as a high reasoning model while a smaller model like Qwen could handle the simpler tasks.

The problem is that DeepSeek is a very large model which takes up most of my Spark's 128gb of memory. For a 27b model to fit in memory with enough KV cache room to be useful, there had to be drastic reductions in space usage. 


Since DeepSeek is a mixture of experts model, I was curious as to whether it was possible to prune the experts of the model to retain only the experts that were the most used & activated.

Further quantization made no sense since I was already using a heavily quantized EXL3 model (the original model was 284 billion parameters).

Here are the original model specs:

| Item | Value |
|---|---|
| Base model | `deepseek-ai/DeepSeek-V4-Flash` 0731: 284 B parameters; 240.7 B after the REAP prune |
| Layers | 43 MoE layers, hidden size 4096, plus 3 `mtp` layers |
| **Routed experts (K)** | **216 per layer (256 before the REAP prune)** |
| **Experts per token (top-k)** | **6 routed + 1 shared** |
| Routing, layers 0 to 2 | Hash: a fixed token-to-expert table (`ffn.gate.tid2eid`) |
| Routing, layers 3 to 42 | Learned router: `sqrtsoftplus` score + `ffn.gate.bias`, top-6, weights normalized |
| One expert | 3 × 4096 × 2048 = 25.2 M parameters = 9.47 MB at 3.0 bpw |
| Expert storage | EXL3 3.0 bpw trellis; 12 tensors per expert; one 2.05 GB file per layer |
| Other weights | Attention, shared experts, embeddings, head: FP8/BF16, about 7 B parameters, 18 GB |
| Size on disk | 106.8 GB (99.5 GiB) |
| Draft model | DSpark: a separate folder with 64 MTP experts |

*REAP pruning (Router-weighted Expert Activation Pruning) is a one-shot method from Cerebras. It runs a calibration dataset through the model and gives each expert a score: the router's gate weight times the size of the expert's output, averaged over the tokens that the router sends to that expert. REAP then removes the lowest-scoring experts in each layer, and the model needs no retraining. Here, it removed 40 of the 256 experts in each layer.*

*DSpark is a speculative decoding method from DeepSeek. A small draft model proposes a block of tokens in one parallel pass, and a light "Markov head" links each draft token to the token before it. The full model then checks the whole block in one pass and keeps the tokens that it agrees with.*

The interesting thing to note is that EXL3 actually makes further pruning cheap. 

Each expert has its own 12 tensors: three matrices (w1, w2, w3), each in four parts (trellis, suh, svh, mcg). One file holds the experts of one layer.

![The EXL3 weight files]({{ site.baseurl }}/images/exl3-weight-files.svg)

*The weight files. The width of each file in the top strip is to scale. The 43 layer files hold the routed experts, about 88 GB of the 106 GB. Each step down zooms in: one layer file, then one expert.*

The draft model we started with was DSpark, which has 64 MTP experts that reads target layers 40 to 42. 

Each decode step checks 6 tokens: 5 draft tokens and 1 more.

Those 6 tokens per step matter for the key idea of this project: an MoE model has two separate levers.

- **K**, the number of routed experts in each layer, controls memory. The server loads all K experts, so each expert that I remove from every layer frees about 0.4 GiB. It saves almost no time, because each token still reads 6 experts.
- **Top-k**, the number of experts that each token uses, controls speed. Memory bandwidth (about 273 GB/s) limits the decode on the Spark, so fewer bytes per token give more tokens per second. It saves no memory, because all K experts stay loaded.

The math shows why. In each layer, a decode step reads every expert that any of its 6 tokens picks. At K 216 and top-k 6, that is about 34 experts per layer: 13.7 GB per step, plus about 7 GB of static weights. If I cut K to 160, the count only drops to about 33, because 6 tokens × 6 experts can never read more than 36. If I cut top-k to 5, it drops to about 28.

So to free memory without losing speed, I would need to pull both levers.

## Building the baseline

I started by building a harness that tells a real loss from run to run noise at three levels of cost. 

![The three levels of the measuring stick]({{ site.baseurl }}/images/measuring-stick-levels.svg)

*The three levels. The left side gives the time per test point, and the right side gives the noise between two runs of the same model. A change counts only if it is larger than the noise of its level.*

The harness reads my bot's own history: 57 recorded sessions with 2,747 messages. Most of the text is tool output, such as SEC filings, web pages and price tables. A script tagged each session with its skill and held out about 30% of the sessions for the tests. The rest became calibration data for the expert rank of Part 2.

### Level 1:

The replay screen sends a recorded context up to an assistant turn, with that recorded turn as the last message:

```text
system     You are a stock-research assistant for my portfolio.
user       Give me an update on ACME after its earnings.
assistant  [tool call] get_filing(ticker="ACME", form="10-Q")
tool       ...Revenue was $4.1 billion, up 12% from a year ago...
assistant  ACME's revenue rose 12% to $4.1 billion.    <- the recorded turn: the screen scores this one
```

*A simplified example. The company, the tool and the numbers are made up.*

The model writes nothing new here. vLLM returns the log-probability of every token in this prompt, and the screen keeps only the tokens of the last turn. For each token, the question is: after everything before it, how likely does the model find the token that the stock model actually wrote?

The screen turns these probabilities into one score: the NLL (negative log-likelihood) per token, which is the mean of −ln p over the tokens of the turn. A token that the model expects (p near 1) adds almost nothing, and a surprising token (p near 0) adds a lot. So a lower NLL means that the model finds the recorded reply more likely.

| Token in the recorded turn | p, stock model | p, damaged model |
|---|---|---|
| ` rose` | 0.80 | 0.75 |
| ` 12` | 0.97 | 0.70 |
| ` 4.1` | 0.95 | 0.60 |
| **NLL per token (mean of −ln p)** | **0.10** | **0.39** |

The stock model copies "12" and "4.1" from the tool result with almost no doubt. A model that lost that skill is less sure, so its NLL goes up. The real changes are much smaller than in this toy: the gate is +3% on the total over 45,690 tokens.

Another target from the same session ends at the tool call. There, the screen also checks if the model ranks the recorded tool name, `get_filing`, as its first choice.

![Replay noise: the same model, run twice]({{ site.baseurl }}/images/replay-noise.svg)

*How I measured the noise floor. Each dot is one target: the change of its NLL from run 1 to run 2 of the same model. Single targets move by up to 37% (median 1.6%), but the sum over all 60 targets moves by only 0.22%. So the gate uses the total, never a single target or skill. The official floor of 0.23% comes from the rounded totals.*

### Level 2:

The speed tests measure the decode speed in tokens/s. 

The bot test sends the context of 10 held-out sessions up to the final report turn and decodes up to 1,024 tokens.

The code test is the server repo's own benchmark: 5 fixed code prompts of 512 tokens, with thinking off. Each test does one warm-up run and then measured runs. The rate of a run is all decoded tokens over all decode time, and the result is the median rate.

I ran both tests 5 times on the stock model. The bot test gave 33.64 tok/s and the code test 38.86 tok/s. The spread was 1.42 and 1.65 tok/s: 4.2% for both.

The tests also read the draft acceptance from the vLLM metrics: 2.25 accepted tokens per draft on the bot test and 2.67 on the code test.

### Level 3:

The full runs answer whether the bot still does its job. The e2e (end to end) runner starts one docker container per prompt and runs the bot's own agent loop with live tools.

The container sees copies of the bot's data and no Discord token. A guard compares the hashes of the real data before and after each batch. The prompts are the first messages of the 18 held-out sessions, plus the daily-report cron job: 19 bot prompts (the first runs used 15). One run of 19 prompts took about 4 hours.

Each run then goes through five automatic checks. The first four check that the bot still follows its own rules, and the last one checks that it does not make up numbers:

1. **Tool-call validity.** Every tool call that the model writes must parse. The server's `deepseek_v4` parser reads each call, and one bad call breaks a step of the agent loop.
2. **Call order.** The `stock-update` skill must send its two writes and then the save, all in one message. This shows that the model still follows a multi-step instruction exactly.
3. **Report marker.** The final reply must end with `[[report:FILE]]`, and the file that it names must exist.
4. **Takes JSON.** The daily report writes its takes to a JSON file. That file must pass the bot's own validator.
5. **Number fidelity.** Each number in the final report must appear in a tool result of the same session. For a stock bot, this is the check that matters most: a pruned model must not invent a price or a revenue figure.

A pairwise LLM judge reads the prompt and two final reports, in random order and without the model names. It answers A, B or tie. It judges each pair twice with the order swapped, and two different answers count as a tie. 

### The gates

Here are the gates of all three levels. A test point must pass the gates of one level before it goes to the next:

| Level | Metric | Gate |
|---|---|---|
| **Replay screen** | **NLL per token, 60 targets** | **At most +3%: 13 times the noise floor** |
| **Replay screen** | **Next tool name** | **At least 98% of the baseline: 76 of 83 passes, 75 of 83 fails** |
| Speed tests | Decode tok/s, bot test and code test | A change counts only above 4.2% |
| Full runs | Tool-call validity | 100%, no new failures |
| Full runs | Call order, report marker | No new failures |
| Full runs | Takes JSON | No missing or invalid file |
| Full runs | Number fidelity | Run share at most 1.5 points above the baseline; one prompt at most 5 points |
| Full runs | LLM judge | No gate: it reports only |

Things I learned:

1. **The boot KV margin.** Non-torch memory grew from 3.2 to 4.81 GiB. At `GPU_MEMORY_UTILIZATION` 0.92, vLLM then had no room for a 256k-token KV cache, so it refused every boot, the restore too. The bot was down overnight, and I moved to 0.93.
2. **Claude Code's memory check.** It stopped a replay at 58 of 60 targets. Now I start long runs with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
3. **Core dumps filled the disk.** A headless browser crashed in the sandbox and wrote four core dumps, the largest 1.22 TB. Now each sandbox container gets `--ulimit core=0`, `--shm-size 1g` and `--memory 3g`.
4. **The memory watch worked.** Free memory fell from 2,185 MB to 380 MB in one minute. The watch stopped the replay, and the script restored the live server.

**Lesson:** Keep the way back separate from the thing that can fail, and write the safety rules in code.

## Finding the right size

I tested each size with a mask before I changed a single weight file. 

The mask is an abstraction that hides some experts from the model router, but doesn't delete them. 

A real prune would immediately be hours of work and would rewrite about 43 files, each 2GB.

Changing only the small routing tensors allows us to test many sizes like K = 192, 176, 160 etc very cheaply. 

The weight files retain the experts and the model behaves exactly like if someone had pruned them, at the cost of loading all 216 original experts every time at boot. 

### How it works in layers 3 to 42:

Each dropped expert gets a very large negative bias: `gate.bias = -1e4`. The router adds the bias only to choose the top-6, so it never picks a dropped expert. The weights come from the scores without the bias. 

![How the mask changes the router in the two kinds of layer]({{ site.baseurl }}/images/mask-two-layer-kinds.svg)

*The mask in the two kinds of layer. Left: a toy router with 10 experts shows how a bias of −1e4 removes e1 from the top-6 and lets e7 in. The kept experts keep their scores, so their weights do not change. Right: a toy row of the hash table. The dropped expert 205 moves to the most similar kept expert that passes the two rules. The ids and scores are examples. The counts at the bottom come from `MASK.json` of the bot K 160 mask.*

### How it works in layers 0 to 2:

DeepSeek V4 doesn't use the score test in its first 3 layers: it uses a fixed table, `ffn.gate.tid2eid`, that maps each token id to 6 experts.

```text
tid2eid, layer 0 (made-up ids)
  token 4516  →  [  3, 190,  44,  71, 158,  20]
  token 4517  →  [ 12,  87, 140, 205,  63,   9]
  token 4518  →  [101,   6, 177,  39, 212, 130]
  ...
```

For token 4517, layer 0 always runs these specific experts in every sentence. 

The bias trick from earlier doesn't work here because the table will send a token's work to that expert regardless.

The tool edits this table. Each entry for a dropped expert moves to the kept expert with the most similar router row. 

There are two rules that must be followed:

That expert must not already be in the same row.

The load of the new expert must stay below twice the mean. Otherwise a few "central" experts take the slots of many dropped experts. Without this rule, one expert got 12 times the median load at K 160.

```text
before:  token 4517  →  [12, 87, 140, 205, 63, 9]     205 is dropped
         candidates: 140 (already in the row), 31 (load too high), 118 (OK)
after:   token 4517  →  [12, 87, 140, 118, 63, 9]
```


### Exactness Check:

A mask only helps if it really blocks the dropped experts. I read the vLLM code: the router adds the bias only to choose the experts, and it holds the bias in FP32, so −1e4 fits.

But one part of the routing runs purely in a CUDA kernel with no source code in the server image, so I tested the routing directly.

1. Boot the mask with routed-expert capture on, so that each reply lists the experts that each token used in each layer.

2. Send four short prompts.

3. Pass only if no dropped expert appears in layers 3 to 42, and the rows of layers 0 to 2 match the mask's table.

On the bot K 160 mask, the router made 0 picks of a dropped expert over 71,583 tokens in all 43 layers.

### Results:

For the first masks, I used the public REAP rank: the same expert scores that made the original prune from 256 to 216 experts. For a mask with K experts, the tool drops the 216 − K lowest-ranked experts of each layer. The top 32 experts of each of the two tool-call categories in the rank never drop.

Then I ran each mask through the replay screen. The NLL measures how surprised the model is by the replies that the stock model actually wrote. If a mask removes experts that the bot needs, the NLL goes up. A mask passes only if its NLL rises by at most 3%.

![NLL change of the two REAP masks against the gate]({{ site.baseurl }}/images/reap-masks-nll.svg)

*Both REAP masks fail the NLL gate. The replay screen on 60 held-out targets. K 176 ran on 58 of 60 targets, and its baseline was cut to the same 58. Sources: `mask-k192-compare.txt`, `mask-k176-compare.txt`.*

Even the smallest cut failed. K 192 drops 24 experts per layer, and its NLL rose from 0.4440 to 0.4587: +3.31%, just over the gate. K 176 drops 40 experts per layer, and its NLL rose by 6.31%, about twice as much. The change at K 192 is 14 times the 0.23% noise floor, so it is a real loss, not noise.

The tool names held at both sizes: 78 of 83 at K 192, and 74 of 80 at K 176 (its 58 targets had 80 tool calls). So the masked model still picked the right tools. It only found the bot's replies less likely.

The loss doubled from K 192 to K 176, so I stopped there. K 160, the size that I needed, was out of reach with this rank.

Why did REAP do so badly? Its rank came from code, math, science and tool-call data. My bot reads earnings reports, SEC filings and web pages, so the REAP rank dropped some experts that my bot actually uses. The next step was to rank the experts on my bot's own traffic.






