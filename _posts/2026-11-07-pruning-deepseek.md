---
layout: post
title: "Pruning DeepSeek: A Comprehensive Guide"
date: 2026-11-07
---

I have been working the past few months with DeepSeek v1 Flash 0731 to run my stock-research bot on my single DGX Spark. It has been amazing to automate the previously tedious research in this process and have well researched stock reports deposited into my custom Discord channel every morning.

Despite this, I have always considered DeepSeek to be the very heavy reasoning model that I can run. Ultimately my DGX Spark is capable of running multiple local models at once, why shouldn't it? 

I have always wanted to create a model router that can intelligently select the best local model for a given task. My idea was to use DeepSeek to serve as a high reasoning model while a smaller model like Qwen could handle simpler tasks. This could also unlock ways to drive Qwen with DeepSeek and so much more.

The problem is that DeepSeek is a very large model which takes up most of my Spark's 128gb of memory. For a 27b model to fit in memory with enough KV cache room to be useful, there had to be drastic reductions in space usage. 

Since DeepSeek is a mixture of experts model, I was curious as to whether it was possible to prune the experts of the model to retain only the experts that were the most used & activated. Further quantization made no sense since I was already using a heavily quantized EXL3 model (the original model was 284 billion parameters).

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

In each layer, a decode step reads every expert that any of its 6 tokens picks. At K 216 and top-k 6, that is about 34 experts per layer: 13.7 GB per step, plus about 7 GB of static weights. If I cut K to 160, the count only drops to about 33, because 6 tokens × 6 experts can never read more than 36. If I cut top-k to 5, it drops to about 28.

So to free memory without losing speed, I would need to adjust both levers.

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

vLLM returns the log-probability of every token in this prompt, and the screen keeps only the tokens of the last turn. For each token, the question is: after everything before it, how likely does the model find the token that the stock model actually wrote?

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

## Step 6: Rank Experts on my bot's traffic

The REAP rank measures the experts that the model uses on a general-purpose dataset. 

I wanted to measure the experts that my bot actually uses, so I built a new rank. 

The first step was to boot `compose-capture.yml`, which starts vLLM in a special mode for the capture.

The goal is to see which experts the unchanged model uses on your bot's traffic. So all 216 experts must be available to the router.

vLLM's Model Runner V1 allows us to define a capture flag, which is `--capture-return-routed-experts` which keeps each choice of experts for each token in each layer, and sends it back in the reply. 

There is no DSpark draft in this step. 

Next, I have a custom Python script that sends the 57 recorded sessions to the model, and collects the routed experts for each token in each layer.

The script uses the 34 segments of the calibration split. These are the sessions that the replay screen does not use. If the rank and the test used the same sessions, the replay screen would grade the rank on the data that made it. The test would then look better than the truth.

Each request holds the recorded context up to the last target turn, as in the replay screen. That is the system prompt, the user messages, the tool calls, the tool results (SEC filings, web pages, price tables) and the assistant turns.

The server writes only one new token and stops. Before that token, it must read the full prompt through all 43 layers. During this read, the router picks 6 experts for every prompt token in every layer, and the capture records all of those picks. The script then ignores the one generated token.

Reading a prompt (prefill) processes all of its tokens in parallel, so it is much faster than writing tokens one at a time. The routing during the read is also the same routing that the model uses on that text in real use. The router's choice for a token depends only on the tokens before it, and the read sees exactly those tokens. Thus 34 requests give the expert choices for 2,023,990 tokens of real bot traffic, and this took no generation at all.

Then the script decodes the routing from each reply. The field `choices[0].routed_experts` holds a long string of text. JSON can carry only text, so the server encodes the binary data as base64, and the script decodes it back into bytes. Those bytes are a NumPy `.npy` file, and `np.load` turns them into an array with the shape (prompt tokens, layers, top-k).

Each entry is one expert id. For example, `routed[1000, 5]` might be `[12, 87, 140, 118, 63, 9]`: the 6 experts that the router picked for token 1000 in layer 5. A 48k-token prompt gives 48,000 × 43 × 6 ≈ 12 million ids, which is why the server sends them as binary. The script also checks that the array has exactly one row for each prompt token.

Next, the script counts how often each of the 216 experts appears in each layer, which gives 43 rows of 216 counts. Two details keep the counts honest:

1. **The shared prefix counts only once.** Every request starts with the same system prompt and tool definitions. If the script counted them in all 34 requests, the experts that read the system prompt would get 34 times their real weight. So the script asks the server's `/tokenize` endpoint for the tokens of the system prompt and the tools, finds how many of the first prompt tokens match, and counts those rows only the first time that this prefix appears.
2. **Unfilled rows do not count.** A real router never picks the same expert twice for one token, so a row of six equal ids marks a token that the capture did not fill. The script skips those rows. My run had none.

This is how 2,023,990 prompt tokens became 1.75 M counted tokens. The difference is the repeated prefix.

Finally, the script writes everything to `expert_rank_bot.json`:

- **Per layer:** all 216 experts, most used first, each with its id, its count and its share of the picks in that layer. One expert gets 1/216 ≈ 0.46% of the picks on average.
- **Per skill:** the same counts, split by the bot skill of each session. These show whether an expert is rare overall but important for one skill.
- **Totals:** the prompt tokens, the counted tokens and the unfilled rows.

The mask tool reads this file the same way it read the REAP rank: for a mask with K experts, it drops the 216 − K experts at the bottom of each layer's list.

The capture boot takes my live bot offline, so I made the script easy to resume. Each segment writes one line of counts to an output file, and a rerun skips the segments that are already done. A separate pass then adds up the lines into the JSON file. Later, I also extended the script to rank the experts by router weight, but this step used the plain count.

Before I booted any new mask, I compared the two ranks on the capture data. The question was simple: how much of my bot's routing goes to the experts that each mask drops? The less traffic the dropped experts carry, the less the bot should notice that they are gone.

![Per layer, the share of the bot traffic that the dropped experts carry]({{ site.baseurl }}/images/rank-drop-share.svg)

*The REAP rank drops more of the bot's traffic with 24 experts than the bot rank does with 56. For each learned-router layer, the share of the bot's routed tokens that goes to the dropped experts of each mask. The dotted line marks the share of any 24 experts (24 of 216 = 11.1%).*

Each dot is one of the 40 learned-router layers (3 to 42). Its height is the share of my bot's routed tokens that go to the experts that the mask drops in that layer. Lower is better.

The REAP mask at K 192 drops only 24 experts per layer, but they carry 10.0% of my bot's routed tokens on average, and 23.2% in layer 37. Any 24 of the 216 experts would carry 11.1% (24 / 216). So for my bot, the REAP rank is barely better than a random choice.

The bot rank does much better. Its 40 least used experts carry 2.4% of the routing, and no layer goes above 5%. Even its 56 least used experts carry only 4.8% on average (7.3% at most), although 17 of the 40 layers go above 5%. So the bot rank drops more than twice as many experts as REAP, and still removes less than half as much of my bot's traffic.

The two ranks also disagree on which experts to drop. At K 160, they share only about 23 of the 56 dropped experts per layer.

### Results:

Then I built the two bot-rank masks, K 176 and K 160, and ran them through the replay screen. The table also shows the baseline and the REAP mask at K 192. "Answer top-1" is the share of answer tokens that the model ranks first: how often its first choice is the token that the stock model wrote.

| Rank | K | NLL | NLL change | Answer top-1 | Tool names | Replay gates |
|---|---|---|---|---|---|---|
| none | 216 | 0.4440 | 0 | 91.40% | 77 of 83 | Baseline |
| REAP | 192 | 0.4587 | +3.31% | 91.08% | 78 of 83 | Fail |
| bot | 176 | 0.4453 | +0.29% | 91.24% | 77 of 83 | Pass |
| **bot** | **160** | **0.4501** | **+1.37%** | **91.23%** | **78 of 83** | **Pass** |

![Replay NLL change against K for the REAP rank and the bot rank]({{ site.baseurl }}/images/nll-vs-k.svg)

*The bot rank moves the whole curve down. Replay NLL change against K for both ranks, at top-k 6. The bot rank at K 160 drops 56 experts and costs less than half of the REAP rank at K 192, which drops 24.*

Both bot-rank masks pass the replay screen. K 176 costs +0.29%, close to the 0.23% noise floor. K 160 costs +1.37%, less than half of the +3% gate, and it is the size that I needed. The bot rank at K 176 drops 40 experts and costs about a tenth of what the REAP rank costs with only 24.

The answer top-1 and the tool names barely moved at either size. The top-1 share fell by less than 0.2 points (91.40% to 91.23%), so the masked models still make almost the same first choices as the stock model. They also still pick the right tools.

A pass here is a pass of the replay screen only. The replay targets come from the same weeks of traffic as the capture, so traffic of a new kind can use other experts. 

## Step 7: Fewer experts per token

The problem now is that the DSpark draft was designed to work with the full K = 216 model, not with anything pruned.

When I ran the bot K 160 mask with the draft, the acceptance dropped from 2.25 to 2.09 tokens in the bot test and from 2.67 to 2.45 in the code test.

A smaller K cannot win this back. The decode reads the weights from memory, and each token still reads 6 routed experts, whatever K is. A smaller K saves memory, not bytes per token. Only fewer experts per token cut the bytes per token.

Here is how I fixed this:

I cut top-k, the number of routed experts that each token uses. First I built a tool that makes a top-k copy of a mask, and then I tested which value to use.

### The top-k tool

My tool `topk_checkpoint.py` copies a mask folder and changes the number of experts per token. It changes no expert weights. vLLM reads top-k from one number in `config.json`, `num_experts_per_tok`, so the tool sets that number to the new top-k. In every layer, top-k 5 means the same thing: each token loses the weakest of its 6 experts.

In layers 3 to 42, the config change is enough. The router now takes its top 5 in place of its top 6, so it drops the expert that was 6th. vLLM then renormalizes the weights of the 5 kept experts, so they add up to the same total as the 6 did. The mask bias still keeps the dropped experts out.

Layers 0 to 2 need more work. vLLM expects the `tid2eid` table to have one column for each expert of a token, so (129,280, 5) now. The file still holds 6 columns, so the boot would fail with a shape mismatch. The tool must drop one id from each row.

The column position means nothing, so the tool drops the id with the smallest router weight, the same rule as in the other layers. But the real weight depends on the input to the layer, and attention mixes the earlier tokens into that input. So one token id gets a different input in every sentence, but the table has only one row for it. The tool uses an estimate: it puts the token's own embedding in place of the real input, and drops the id with the lowest score.

```text
token "price", layer 0 (made-up numbers)
row before:  [ 17,   92,  140,    5,  201,   63 ]
est. score:  [1.4,  0.9,  1.6,  0.5,  1.2,  1.1 ]
row after:   [ 17,   92,  140,       201,   63 ]   <- id 5 has the lowest score
```

By the same estimate, the dropped expert carried about 12 to 13% of a token's routing weight in these layers. With 6 equal weights, each expert would carry 16.7%. So the tool drops a weaker expert than average, but not one with zero weight.

Then the tool writes the new folder:

1. It rewrites the one file that holds the 3 tables (4.28 GB) with the new table shapes. The other 495 tensors in that file stay byte for byte the same.
2. It links the other files, such as the 43 expert files of about 88 GB, in place of a copy. So one folder takes only about 35 seconds.
3. It runs a self-check: each new row holds only ids from the old row, and every other tensor has the same bytes as before.

```bash
python topk_checkpoint.py --source tp1-botmask-k160 --top-k 5 --out tp1-botmask5-k160
```

Each boot loaded the new tables with no error, and the EXL3 runtime line showed `topk=5`. But layers 0 to 2 use an estimate, so only the replay screen can show the real cost.

### Testing potential top-k values

Top-k takes only whole numbers, so I had three values to test: 6, 5 and 4. Before any test, I estimated that top-k 5 would decode about 9% faster than top-k 6, and top-k 4 about 21% faster. But each expert that a token loses also costs some quality, and I could not predict how much. So I measured it.

I wrote down the rules for the pick before the test, so that the results could not change them:

1. Drop each point that fails a gate.
2. From the rest, take the fastest point. Two speeds within the 4.2% noise count as equal.
3. If two points have equal speed, take the smaller one.

Then a script tested three points in one night, from 2026-11-04 to 2026-11-05: bot K 160 at top-k 5, bot K 160 at top-k 4, and K 216 at top-k 5 with no mask. For each point, it booted the server, ran the replay screen and then the speed tests (the median of 3 runs per test). At the end, it restored my live server.

### Results:

The table shows the three points of the night next to the stock model and bot K 160 at top-k 6. The speed changes compare each point with the stock model. "Acceptance" is the number of draft tokens that the full model accepts per draft, in the bot test and in the code test.

| Rank | K | Top-k | NLL change | Tool names | Bot test | Code test | Acceptance (bot / code) | Result |
|---|---|---|---|---|---|---|---|---|
| none | 216 | 6 | 0 | 77 of 83 | 33.64 tok/s | 38.86 tok/s | 2.25 / 2.67 | Baseline |
| bot | 160 | 6 | +1.37% | 78 of 83 | 31.50 (−6.4%) | 35.67 (−8.2%) | 2.09 / 2.45 | Slower |
| **bot** | **160** | **5** | **+2.27%** | **79 of 83** | **33.98 (+1.0%)** | **42.07 (+8.3%)** | **2.10 / 2.76** | **The pick** |
| bot | 160 | 4 | +6.05% | 79 of 83 | 36.05 (+7.2%) | 42.78 (+10.1%) | 2.00 / 2.51 | NLL fail |
| none | 216 | 5 | +1.29% | 49 of 54 | – | – | – | Pass (35 targets) |

Top-k 4 is the fastest point, but its NLL change of +6.05% is twice the gate. So rule 1 drops it.

Top-k 5 passes with room to spare: +2.27% against the +3% gate, and it still picks the right tools. Compared with top-k 6 at the same K 160, it decodes 7.9% faster in the bot test and 17.9% faster in the code test, for only +0.87% more NLL. Compared with the stock model, the code test gains 8.3%, above the noise. The bot test gains 1.0%, inside the noise, so it is as fast as the stock model. Top-k 5 won back all the speed that the mask lost.

The acceptance column shows where that speed came from. In the bot test, the acceptance hardly changed from top-k 6 (2.09 to 2.10), so the gain there comes from fewer bytes per token. In the code test, the acceptance also rose from 2.45 to 2.76, even above the 2.67 of the stock model, so the code test gained more.

So 5 is the smallest top-k that stays inside the quality gate, and the rules picked bot K 160 at top-k 5.

The last row, K 216 at top-k 5 with no mask, answers one more question: does the mask make the top-k cut worse? The memory watch stopped this replay at 35 of 60 targets, and the speed tests did not run. So I compared the points on those 35 targets only. There, top-k 5 alone cost +1.29%, and the mask alone cost +1.24%. Both together cost +2.50%, about the sum of the two. So the two costs simply add up, and neither change makes the other one worse.

**Lesson:** The number of experts per layer sets the size. The number of experts per token sets the speed. A pruned MoE model needs both changes to be smaller and faster.

The number of experts per layer sets the size. The number of experts per token sets the speed. A pruned MoE model needs both changes to be smaller and faster, and a draft model for speculative decoding loses acceptance when the target changes.

## Step 8: Full runs and the final review

Before I cut any experts out of the files, the pick (bot K 160 at top-k 5) had to pass two more tests: the full runs, and a final review of its answers.

The full run on 2026-11-05 sent 15 bot prompts through the real agent. All 15 finished, and the boot log showed `topk=5`. Every check passed except the report marker on one prompt: the model wrote the full report in its reply, but saved no file. The stock model also skipped that save on 1 of 3 runs of the same prompt, so I counted it as a problem of the prompt, not of the model.

### The final review: 100 judge pairs and 20 blind pairs

The replay screen and the checks catch broken behavior, but they do not answer the main question: are the answers still as good? So the plan asked for two reviews of the final reports:

- **100 judge pairs.** The LLM judge compares a report of the stock model with a report of the pick for the same prompt. It does not see the model names, and it judges each pair twice with A and B swapped. Two different answers count as a tie. I ran the stock model 3 times and the pick 2 times on all 19 held-out prompts, so the 100 pairs grew to 3 × 2 × 19 = 114.
- **20 blind pairs that I reviewed myself.** A review page showed each pair as A and B, with no run names and in a random order. For each pair, I chose the better report or a tie.

The judge is the stock model itself, so it can prefer its own answers. So I also ran 57 control pairs: each stock run against each other stock run. The same model is on both sides, so a control shows how often the judge prefers one side by chance.

If the two models were equal, each pair with a winner would be a coin flip. A sign test checks if the pick lost more pairs than chance allows (p < 0.05). The pick fails if the judge or my own review finds a loss. No control may fail the same test.

![LLM judge: wins, ties and losses of the pick, and the three controls]({{ site.baseurl }}/images/judge-pairs.svg)

*The judge finds no loss, and the controls find no bias. Each bar shows the share of pairs, and the numbers are pair counts. Against the two older baselines, the pick has 34 wins and 10 losses. Against baseline-3, which ran on the same day as the new runs, it has 10 wins and 9 losses. In a control, a win is a win of the later baseline.*

The pick passed all three rules:

- **LLM judge, 114 pairs:** 44 wins, 51 ties and 19 losses for the pick. The fail line was about 38 losses, so the judge found no loss.
- **Controls, 57 pairs:** No control failed the test in either direction, so the judge showed no bias toward one side.
- **My blind review, 20 pairs:** I preferred the pick in 9 pairs and the stock model in 6, with 5 ties. The pick would fail only if I had preferred the stock model in 12 of the 15 decided pairs.

The judge and I agreed on only 9 of the 20 pairs. So the judge is a good check for "no loss" over many pairs, but not for a single pair.

I also read the answers for hard faults. Wrong numbers came up on both sides, so they do not separate the two models. Two faults of the pick looked like effects of the mask: a literal `</s>` in a broken sentence, and a call to a tool that does not exist. On 2026-11-06, I accepted these faults, and the pick passed.

**Lesson:** Read the result as "no loss", not as a gain. Most of the judge wins come from newer market data in the new runs, not from a better model. Against baseline-3, which ran on the same day, the judge gave only 10 wins and 9 losses.

## Step 9: Cut the experts out of the files

The mask hides 56 experts per layer from the router, but vLLM still loads all 216. So the mask model still loads 95.39 GiB. To free that memory, I had to remove the dropped experts from the weight files. My tool `prune_experts.py` reads a mask folder and writes a new folder. It never changes its source.

Three facts made the job simple:

1. Each expert is 12 separate tensors in the file of its layer, so the tool can take them out one by one.
2. The tool copies those tensors byte for byte, with no new quantization. So the kept experts are exactly the experts that the mask tests measured.
3. The DSpark draft is a separate folder, and vLLM skips the MTP tensors when it loads the target model. So the prune does not touch the draft, and the stock draft still works.

### New ids for the kept experts

vLLM expects the experts of a layer to have the ids 0 to K − 1, with no gaps. So the 160 kept experts of each layer get new ids: in ascending order, they become 0 to 159. Every tensor that finds an expert by its position must follow the same order:

- **The 43 expert files:** For each layer, the tool writes a new file with the 12 × 160 kept tensors, under their new ids.
- **The router rows:** `gate.weight` and `gate.bias` keep only the rows of the kept experts, in the same order. The masked −1e4 values leave with the dropped rows.
- **The hash tables (layers 0 to 2):** Each value in `tid2eid` changes to its new id. The tool stops if it finds a dropped id, because the mask must have moved all of them already.

![New expert ids in layer 3]({{ site.baseurl }}/images/prune-new-ids.svg)

*New ids for the kept experts, from the real keep map of layer 3. The kept ids keep their ascending order and close the gaps: old 3 becomes new 2, old 11 becomes new 7, and old 215 becomes new 159. The router row of each expert (blue bar) moves with it, so the router still scores the same experts. The tool applies one keep map per layer to all 43 layers.*

The tool also sets the expert count to 160 in `config.json` and `quantization_config.json`, because vLLM stops at load if the counts do not match. It also rewrites the weight index and the manifest, which the server checks at every boot.

### The build

Root owns the 43 stock expert files, and only root can read them. So `prune_run.sh` runs the tool in a root container from the server image, with no network, an 8 GB memory limit and the stock folder mounted read-only. The tool streams the tensor bytes and never holds a whole file in memory, so my bot stayed up during the build.

![The prune tool, from the mask folder to the pruned folder]({{ site.baseurl }}/images/prune-tool-pipeline.svg)

*The prune tool. The mask folder rewrote only the router files and links to the stock folder for the expert files, so the tool reads the expert bytes from `data/tp1`. Only root can read them, so the tool runs in a container. Green files hold changed tensor data, and blue JSON files hold new counts and checksums. The dashed box holds the files that did not change: they stay links.*

The build ran on 2026-11-06 from 18:51 to 19:02, about 11 minutes. It rewrote 46 files: the 43 expert files and the 3 files with the router tensors. It removed 28,896 tensors (43 layers × 56 experts × 12 tensors), or 21.27 GiB of tensor data.

### The self-check

A rewrite of this size must be exact. At the end of the build, the tool reads the new folder and the source again, and confirms four points:

1. Each of the 82,560 kept expert tensors (43 × 160 × 12) has the same bytes as its source tensor.
2. The 5,463 other tensors are also byte-equal to the source.
3. The router tensors hold the source rows in the keep order, and each new `tid2eid` value maps back to its old value.
4. The manifest matches every file, and the manifest check from the server image passes.

The self-check passed on the first build. Before the real run, six unit tests ran the tool on a small fake checkpoint. They also confirmed that the tool stops on three faults: a dropped id in `tid2eid`, a bias that is not masked, and one changed byte.

## Step 10: Test boot of the pruned model

The pruned model had one job: to behave exactly like the mask, with the same replay result and the same speed. It needed no new full runs or final review, because the mask had already passed them. On 2026-11-06 at 19:05, a script stopped my live server and booted the pruned model with the stock draft model. It ran the replay screen and the two speed tests, and then it restored the live server.

| | Stock model | Mask (the pick) | Pruned model |
|---|---|---|---|
| **Weights loaded** | 95.39 GiB | 95.39 GiB | **74.09 GiB** |
| **Replay NLL change** | 0 | +2.27% | **+1.89%** |
| Tool names | 77 of 83 | 79 of 83 | 78 of 83 |
| Bot test | 33.64 tok/s | 33.98 tok/s | 33.13 tok/s |
| Code test | 38.86 tok/s | 42.07 tok/s | 40.10 tok/s |
| **KV cache** | 7.36 GiB (269,629 tokens) | – | **29.76 GiB (1,089,707 tokens)** |

**21.30 GiB less memory.** vLLM printed "Model loading took 74.09 GiB", against 95.39 GiB for the stock model. My estimate from the exact tensor sizes was 74.1 GiB. My cap was about 72 GiB, to leave room for the second model, so the pruned model is 2.09 GiB over it. I accepted that on the same day.

**The same routing.** The pruned model passes the replay screen at +1.89%, even 0.38% lower than the mask. That difference is only 0.15 points over the 0.23% noise floor, and in the better direction. The median change per target (1.92%) is close to the spread between two runs of the stock model (1.64%). A wrong expert map would make the NLL clearly worse: the failed masks cost +3% to +6%. One cause of the small difference is that the SparkInfer kernels choose their tile size from the expert count, so the order of the sums changes.

**The same speed.** Both speed tests are inside the 4.2% noise of the stock model: −1.5% in the bot test and +3.2% in the code test. The code test is 4.7% slower than the mask, but one trial causes that. On the mask, the model wrote the same `cuda_cpp` answer in all 4 runs, and the draft model predicts that text well. The pruned model wrote a different answer in each run. So the text changed, not the speed of the compute.

**The freed memory went to the KV cache.** I kept `GPU_MEMORY_UTILIZATION` at 0.93, so vLLM filled the same memory budget as before. It gave all of the freed memory to the KV cache, which grew from 7.36 GiB to 29.76 GiB: about 4 times as many tokens. That is correct when DeepSeek runs alone. But the freed memory is meant for the second model, and the next step fixes that.

## Step 11: Deploy and pin the KV cache

I deployed the pruned model on 2026-11-06. The bot was down for only 3 min 19 s, and I kept the stock server intact, so a rollback takes three commands.

After the deploy, the KV cache took the memory that the prune freed: 29.8 GiB, room for 1,091,301 tokens. The whole machine had only 2.4 GB left. My bot does not need a cache that large, and the second model needs that memory. So the server had to get smaller in memory, not only on disk.

### The first try: a lower memory budget

My first idea was a lower `GPU_MEMORY_UTILIZATION`. vLLM multiplies this value by the 121.7 GiB of the Spark to get its memory budget. Everything that is left after the weights and the fixed overhead becomes the KV cache. A test boot at 0.73, with a 128k context, failed two times in a row:

```text
6.54 GiB KV cache is needed, which is larger than the available KV cache memory (5.82 GiB)
```

At 0.73, the budget is 88.8 GiB. The weights take 74.09 GiB and the fixed overhead (draft model, CUDA graphs, activations, non-torch memory) takes about 8.9 GiB, so only 5.82 GiB is left for the KV cache. vLLM refuses to boot unless the KV cache can hold at least one request of the full context length.

A shorter context does not solve this. At 128k, vLLM needs 6.54 GiB, and at 256k, it needs 6.99 GiB: only 0.45 GiB more. So I kept the 256k context and looked for a different control.

### The fix: set the KV cache size directly

The flag `--kv-cache-memory-bytes` sets the size of the KV cache directly. I added `--kv-cache-memory-bytes 8589934592` to the server flags. vLLM then reserves exactly 8 GiB for the KV cache and skips its own memory estimate.

| | After the deploy | After the pin |
|---|---|---|
| **KV cache** | 29.8 GiB | **8 GiB** |
| **KV cache tokens** | 1,091,301 | **292,952** |
| Memory used on the machine | 120.6 GB | 100.9 GB |
| **Memory available** | 2.4 GB | **23.7 GB** |

![The live server before and after the KV cache pin]({{ site.baseurl }}/images/kv-cache-pin.svg)

*The live server before and after `--kv-cache-memory-bytes`. Only the KV cache changed: 29.8 GiB became 8 GiB. The weights stay at 74.09 GiB, and the runtime overhead (draft model, activations, CUDA graphs, non-torch memory) is about 12 GiB after the pin.*

8 GiB is enough for my bot. Its largest session is about 100k tokens, and the 292,952 tokens hold 1.14 requests of the full 256k context. The pin gave back 21.8 GiB, and the machine now has 23.7 GB available for the second model.

With the KV cache pinned, `GPU_MEMORY_UTILIZATION` no longer sets the KV cache size, but vLLM still checks it at boot. vLLM starts only if the free memory is at least that value × 121.69 GiB. On 2026-11-07, more host processes ran, and a restart at 0.93 failed this check. I set the value to 0.90, and the server booted with the same 8 GiB KV cache and 292,952 tokens.

## Step 12: A new draft model

The pruned model still used the stock DSpark draft. The draft model only proposes tokens: the target model checks each one and writes every token that the user sees. So a better draft model changes no output. It only raises the acceptance, the number of draft tokens that the target model accepts per draft. A higher acceptance means fewer passes of the large model per token.

### Why a new draft model

The DSpark draft has its own MTP layer with 64 of the 216 experts. The stock draft chose those 64 by the public REAP rank, and that was a poor fit for my bot. Only 16 of the 64 are among my bot's 64 most used experts in layer 42, and those top 64 carry 68% of the routing weight. The mask had also lowered the acceptance in the bot test, from 2.25 to 2.09.

So I built a new draft model, `dspark-draft-k64-bot`. It keeps the 64 MTP experts with the highest bot rank in layer 42. It has the same size as the stock draft model, 3.0 GB.

### Results

On 2026-11-07, a script booted the live pruned model with the new draft model and ran both speed tests. Then it restored the live server, after 30 minutes of downtime.

| Speed test | Stock draft model | Bot-rank draft model | Speed change | Acceptance change |
|---|---|---|---|---|
| Bot test | 33.13 tok/s, 2.05 per draft | 34.50 tok/s, 2.14 per draft | **+4.1%** | **+4.7%** |
| Code test | 40.10 tok/s, 2.52 per draft | 40.69 tok/s, 2.61 per draft | +1.5% | +3.5% |

![Stock draft against the bot-rank draft on the pruned model]({{ site.baseurl }}/images/draft-test.svg)

*The draft test of 2026-11-07, on the live pruned model. Left: the bot gain (+4.1%) is just inside the grey noise band, and the code gain (+1.5%) is well inside it. Middle and right: the acceptance rose in both tests and at every draft position. So a real, small gain is the most likely result.*

**A small gain, just inside the noise.** The bot test gained 1.37 tok/s, just under the 4.2% noise floor (1.42 tok/s). The code test gain is well inside the noise. My own rule says that a change counts only above the noise, so by that rule, the new draft model does not qualify.

**But the gain is most likely real.** Both tests moved in the same direction. The acceptance also rose at every draft position. The right panel shows the share of drafts in which the target model accepts the token at each position. It rose from 0.78 to 0.80 at position 1, and from 0.22 to 0.24 at position 4.

**And the new draft model costs nothing.** Speculative decoding is lossless, so the new draft model changes no output. It has the same size as the stock draft model. The switch is one mount line in the compose file and a short reboot.

So I switched on 2026-11-07. The bot-rank draft model has been live since 09:17, after 2 min 34 s of downtime. The server came back with the same 74.09 GiB of weights and 292,952 KV cache tokens.

## The final memory savings

I removed 56 of the 216 routed experts in each layer, and I cut the experts per token from 6 to 5. Here is the stock model next to the live model of today:

| | Stock model | Pruned model, live |
|---|---|---|
| Routed experts per layer (K) | 216 | 160 |
| Experts per token (top-k) | 6 | 5 |
| **Weights loaded** | **95.39 GiB** | **74.09 GiB (−21.30 GiB)** |
| KV cache | 7.36 GiB (269,629 tokens) | 8 GiB, pinned (292,952 tokens) |
| Replay NLL change | 0 | +1.89% |
| Bot test | 33.64 tok/s | 34.50 tok/s |
| Code test | 38.86 tok/s | 40.69 tok/s |

When I started, only 4 GiB stayed free with the stock model loaded. Now the whole server uses about 94 GiB, and 23.7 GB of host memory stays available. The bot shows no loss that I notice, and the decode speed stayed the same or got faster.

That memory is for the second model of my router. The host (the OS, the gateway, a browser and the CLIs) needs about 5 GiB, and I also want a free margin. That leaves about 12 GiB for the second model with a 10 GiB margin, or about 18 GiB with a 4 GiB margin.

![The 121.7 GiB pool with the pruned server and a second model]({{ site.baseurl }}/images/second-model-room.svg)

*The room for a second model, drawn to scale, with two sizes of free margin. The share of the second model must hold its weights, its KV cache and its CUDA graphs. The values are estimates, and a boot with two models will measure the real ones.*

This is less than the 22 GiB that my first plan assumed, for two reasons. The pruned model is 2.09 GiB over my 72 GiB cap, and the live server needs more runtime memory than its boot profile showed. So a MoE model or a small dense model now fits better than a 4-bit 27B dense model.

## The biggest lessons

1. **A public expert rank is not your traffic.** The REAP rank failed at K 192 (+3.31%). The rank from my bot's own traffic cost only +0.29% at K 176. Rank the experts on the traffic that the model will serve.
2. **Test with a mask before you rewrite any weights.** A mask hides experts from the router with a bias of −1e4, and it takes about a minute to build. The pruned model then matched the mask within the noise.
3. **K saves memory, and top-k saves time.** K 160 at top-k 6 decoded 6.4% slower in the bot test, because each token still ran 6 experts. Top-k 5 won the speed back, and top-k 4 cost too much quality (+6.05%).
4. **A pruned model needs a new look at its draft model.** The mask lowered the bot acceptance from 2.25 to 2.09. A draft model ranked on my bot's traffic won part of it back (+4.7%).
5. **Measure the noise floor before you read a result.** Two runs of the same model differ by 0.23% in NLL and 4.2% in speed. Without those numbers, I could not read the test boot or the draft test.
6. **Check the LLM judge with controls.** Three runs of the stock model against itself gave no significant result. Only then did 44 wins and 19 losses mean "no loss".
7. **Pin the KV cache when two models share memory.** vLLM gives all of its free budget to the KV cache. The flag `--kv-cache-memory-bytes` fixed it at 8 GiB and freed 21.3 GB of host memory.
