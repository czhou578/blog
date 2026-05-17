---
layout: post
title: "Adding Scheduling to NanoGPT"
date: 2026-05-17
---

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>

In the previous post, we added chunked prefill to Andrej Karpathy's NanoGPT. We will continue in our quest to optimize the inference layer of our NanoGPT model by adding a scheduler. 

## Problem

Right now, our NanoGPT model is operating in a FCFS (First Come First Serve) manner. This means that requests are processed in the order that they arrive. Prefilling requests block later arrivals from being admitted until they're done.

In a real server, if a very long low priority batch job arrives first and hogs the token budget for many steps, a short high priority request would be forced to wait possibly a long time before processing.

Having a scheduler can preempt the low-priority job, serve the high-priority one immediately, and resume the evicted request when resources free up. This is exactly how vLLM's scheduler manages competing requests under memory pressure.

Here's the request lifecycle with scheduling. The key addition compared to our chunked prefill post is the preemption arrow — an active request can be evicted back to waiting if the system runs out of KV memory:

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> waiting
    waiting --> prefilling : _maybe_admit
    prefilling --> active : is_fully_prefilled
    active --> done : is_done
    active --> waiting : _maybe_preempt
    done --> [*]

    note right of waiting : Heap-ordered by\n(priority, arrival_time)
    note right of active : Decoding 1 token/step\nKV cache grows each step
    note left of waiting : Preempted requests\nre-enter here with\ncache cleared
</div>

## The Request Dataclass

A reminder that the Request dataclass right now contains everything that we need to know about a single request. In order to have a scheduler, we need to introduce two more pieces of data; a priority and arrival_time. 

```python
@dataclass
class Request:
    ...
    priority: int = 0         # 0 = highest priority
    arrival_time: int = 0     # set when admitted to the scheduler
```

We need a data structure that would make ordering by (priority, arrival_time) efficient. Let's hope your Leetcoding skills are sharp, because we're going to need to use a **min-heap**!

Here is a question for you: When two requests have the same priority, which should be served first — the one that arrived earlier or the one with fewer tokens left?

## Scheduler Class

We could theoretically leave the scheduler logic inside the `generate` function, but it would not be modular. Instead, we can extract that into a separate class called `Scheduler`. This will also make it easy to swap out different scheduling policies later on.

There will be a `schedule` function that runs once per step and returns which requests are doing what this step. 

Here is the `schedule` function:

```python

    def schedule(self, step: int):
        """
        Returns:
            prefill_req:  Request | None  — one request getting a prefill chunk (or None)
            decode_reqs:  List[Request]   — all requests currently being decoded (active)

        """
        self._maybe_admit(step)       # promote waiting → prefilling if memory allows
        self._maybe_preempt()         # evict if over memory budget

        prefill_req = self.prefilling[0] if self.prefilling else None
        decode_reqs = list(self.active)

        return prefill_req, decode_reqs

```

And here are the two helper methods that do the heavy lifting:

```python
    def _maybe_admit(self, step):
        if self.prefilling:
            return
        
        if not self.waiting:
            return

        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        _, _, _, candidate = self.waiting[0]
        candidate_kv = len(candidate.prompt_tokens)

        if kv_used + candidate_kv > self.max_kv_tokens:
            return
        
        if len(self.active) + len(self.prefilling) >= self.max_batch_size: return

        heapq.heappop(self.waiting)
        candidate.arrival_time = step
        candidate.status = "prefilling"
        self.prefilling.append(candidate)
    
    def _maybe_preempt(self):
        kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)

        while self.active and kv_used > self.max_kv_tokens:
            victim = max(self.active, key=lambda r: (r.priority, -r.arrival_time))
            self.active.remove(victim)
            victim.clear_cache()
            victim.prefill_cursor = 0
            victim.status = "waiting"
            self.preempted.append(victim)

            key = self._sort_key(victim)
            heapq.heappush(self.waiting, (*key, victim.id, victim))
            kv_used = sum(len(req.prompt_tokens) + req.num_generated for req in self.active + self.prefilling)
```

`_maybe_admit` is the gatekeeper: it decides whether a new request can enter the system. It peeks at the highest-priority candidate in the waiting heap and checks two constraints before letting it through — first, whether the total KV memory currently consumed by active and prefilling requests, plus the candidate's full prompt length, would exceed `max_kv_tokens`, and second, whether the batch is already at capacity. If either constraint fails, the candidate stays in the heap and we try again next step. Notice the early return when `self.prefilling` is non-empty — we only allow one request to be mid-prefill at a time (matching our chunked prefill design), so a new request can't be admitted until the current one finishes prefilling and graduates to active.

### When to Preempt

This is the real problem that scheduling solves. We only have a limited token budget, which is `max_kv_tokens`. If admitting a new prefilling request will push the total kv_used over the limit, then we cannot just admit it.

We would have to do the following: 

1. Pick the lowest-priority active request (the one you'd sacrifice first).
2. Call `req.clear_cache()` to free its KV memory.
3. Reset `req.prefill_cursor = 0` and `req.status = "waiting"`.
4. Move it to `self.preempted` (a separate list so you don't lose it).

When the memory frees up again, the preempted request re-enters the waiting queue and must re-prefill from scratch. This is the recompute preemption strategy.

Question to ask yourself: Should preempted requests go to the front of the waiting queue (preserving their original priority) or to the back?

Here's a concrete example. Two requests share a tight KV budget of 22 tokens. Request A (low priority) is actively decoding, Request B (high priority) arrives and needs prefilling. As B prefills and both caches grow, the system hits the memory limit — so the scheduler preempts A, clears its cache, and lets B finish. Once B completes and frees its memory, A re-enters and re-prefills from scratch:

<div class="mermaid">
sequenceDiagram
    participant W as Waiting Queue
    participant S as Scheduler
    participant P as Prefilling
    participant A as Active (Decoding)
    participant D as Done

    Note over W: Req A (pri=2), Req B (pri=0)

    rect rgb(230, 245, 255)
        Note right of S: Step 0-1: Admit &amp; prefill A
        S->>W: _maybe_admit → pop A
        W->>P: A moves to prefilling
        P->>A: A fully prefilled → active
    end

    rect rgb(230, 245, 255)
        Note right of S: Step 2: Admit B, A decoding
        S->>W: _maybe_admit → pop B
        W->>P: B moves to prefilling
        Note over A: A decodes (KV grows)
        Note over P: B prefill chunk 1
    end

    rect rgb(255, 235, 235)
        Note right of S: Step 3: KV exceeds budget!
        Note over S: kv_used &gt; max_kv_tokens
        S->>A: _maybe_preempt → evict A
        A->>W: A cache cleared, cursor=0
        Note over W: A re-enters heap
        Note over P: B prefill chunk 2
    end

    rect rgb(230, 255, 230)
        Note right of S: Step 4-5: B finishes
        P->>A: B fully prefilled → active
        A->>D: B generates all tokens → done
        Note over D: B completed ✓
    end

    rect rgb(230, 255, 230)
        Note right of S: Step 6+: A re-enters
        S->>W: _maybe_admit → pop A
        W->>P: A re-prefills from scratch
        P->>A: A fully prefilled → active
        A->>D: A generates all tokens → done
        Note over D: A completed ✓
    end
</div>

`_maybe_preempt` is the safety valve that fires when the system is *already* over budget — something `_maybe_admit` tries to prevent, but can't always guarantee because active requests grow their KV caches by one token every decode step. It loops until memory usage drops below `max_kv_tokens`, each iteration picking the worst victim: the request with the highest priority number (lowest importance) and, among ties, the one that arrived most recently. The victim's KV cache is cleared, its `prefill_cursor` is reset to zero, and it's pushed back into the waiting heap — meaning it will have to re-prefill from scratch when it's eventually re-admitted. This is the **recompute preemption** strategy: we trade future GPU work (re-prefilling) for immediate memory relief.

Together, these two functions enforce the core invariant of the scheduler: the system never exceeds its KV memory budget for more than a single step. `_maybe_admit` prevents overcommitment on the way in; `_maybe_preempt` corrects it if the system drifts over budget due to ongoing decode growth. Without `_maybe_admit`, we'd blindly admit requests and constantly trigger expensive preemptions. Without `_maybe_preempt`, a slow accumulation of decode tokens across many active requests could silently blow past the memory limit and crash the system.

## Adding the Scheduler to the Generate Function

Now, we can add the scheduler to the scheduled_generate function. 

Here is the code:

```python

def scheduled_generate(model, requests, policy="fcfs", token_budget=16, max_kv_tokens=256):
    scheduler = Scheduler(policy, token_budget=token_budget, max_kv_tokens=max_kv_tokens)

    step = 0

    for req in requests:
        req.arrival_time = step
        scheduler.add_request(req)
    
    model.eval()

    with torch.no_grad():
        while not scheduler.is_done():
            print(f"[step {step}] prefill={prefill_req.id if prefill_req else None} "
                f"decode={[r.id for r in decode_reqs]} "
                f"waiting={[x[3].id for x in scheduler.waiting]}")

            prefill_req, decode_reqs = scheduler.schedule(step)

            if prefill_req:
            
                prefill_chunk_tokens = []
                
                remaining_budget = token_budget - len(scheduler.active)

                if remaining_budget > 0 and scheduler.prefilling:
                    p_req = scheduler.prefilling[0]

                    tokens_left = len(p_req.prompt_tokens) - p_req.prefill_cursor
                    chunk_size = min(remaining_budget, tokens_left)

                    chunk_start = p_req.prefill_cursor 

                    chunk_tokens = p_req.prompt_tokens[chunk_start: chunk_start + chunk_size]

                    prefill_chunk_tokens = torch.tensor([chunk_tokens], dtype=torch.long, device=device)

                    p_req.prefill_cursor += chunk_size

                if not prefill_chunk_tokens and not scheduler.active:
                    step += 1
                    continue

                if prefill_chunk_tokens:
                    pos = torch.arange(chunk_start, chunk_start + chunk_size).unsqueeze(0)

                    if p_req.kv_cache:
                        #  This format is wrong 
                        # logits, _, new_kvs = model(prefill_chunk_tokens, past_kvs=req.kv_cache)
                        # list[list[(k, v)]] is shape
                        
                        past_kvs = []
                        for layer_idx in range(n_layer):
                            block_kv = [(p_req.kv_cache[(layer_idx, hi)]) for hi in range(n_head)] 
                            past_kvs.append(block_kv)
                        
                        logits, _, new_kvs = model(prefill_chunk_tokens, past_kvs, pos=pos)
                    else:
                        logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos)

                    for li, bkv in enumerate(new_kvs):
                        for hi, (k, v) in enumerate(bkv):
                            p_req.kv_cache[(li, hi)] = (k, v)


                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    if prefill_req.is_fully_prefilled:
                        prefill_req.generated_tokens.append(idx_next.item())
                        prefill_req._last_token = idx_next
                        scheduler.promote(prefill_req)

            if decode_reqs:
                B_active = len(scheduler.active)

                batch_tokens = torch.cat([req._last_token for req in scheduler.active])

                batch_positions = torch.tensor([[len(req.tokens_so_far) - 1] for req in scheduler.active], device=device)

                past_kvs, attn_mask, pad_lengths = assemble_batch_cache(scheduler.active)

                logits, _, new_kvs = model(
                    batch_tokens,
                    pos=batch_positions,
                    past_kvs=past_kvs,
                    attn_mask=attn_mask
                )

                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                disassemble_batch_cache(scheduler.active, new_kvs, pad_lengths)

                for i, req in enumerate(decode_reqs):
                    req.generated_tokens.append(idx_next[i].item())
                    req._last_token = idx_next[i : i + 1]
                
                for req in decode_reqs:
                    if req.is_done:
                        scheduler.complete(req)
            
            step += 1
        
        return scheduler
```

The main difference is that the Scheduler instance is being created up top witih the policy, token budget, and max kv tokens. 

In the beginning, for each request, we simply add it to the scheduler using its `add_request` method. 

Then, inside the while loop, we call `scheduler.schedule(step)` which returns the prefill request and the decode requests. 

If there is a prefill request, we prefill it for one step. This logic is the exact same compared to the previous post. 

If there are decode requests, we decode them for one step. This logic is also the exact same compared to the previous post. 

Finally, we check if any requests are done and remove them from the scheduler. This is a change from the chunked prefill implementation, where instead of manually keeping track of a still active requests list, we simply let the scheduler handle it for us (by calling the `scheduler.complete()` method).

* For a more detailed explanation of the prefill and decode logic, refer to my previous posts here. 

## Tests

As always, we want to run some simulated tests to show that our logic is sound until different situations. 

### FCFS Correctness

In this case, all requests have the same priority. Verify they complete in arrival order and output matches chunked-prefill notebook for the same prompts.

```python

# ══════════════════════════════════════════════════════════════
# Test 1: FCFS — same priority, all complete, valid output
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: FCFS correctness")
print("=" * 60)
torch.manual_seed(42)
reqs = [
    Request(id=0, prompt_tokens=encode("O Romeo, "),     max_new_tokens=10),
    Request(id=1, prompt_tokens=encode("To be or "),     max_new_tokens=10),
    Request(id=2, prompt_tokens=encode("KING HENRY:\n"), max_new_tokens=10),
]
s = scheduled_generate(model, reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)
for req in reqs:
    assert req.status == "done", f"❌ Req {req.id} not done"
    assert req.num_generated == 10, f"❌ Req {req.id}: got {req.num_generated} tokens, expected 10"
    print(f"  Req {req.id}: '{decode(req.tokens_so_far)}'")
print("✅ Test 1 passed")

```

Here is the result:

```text
============================================================
Test 1: FCFS correctness
============================================================
  Req 0: 'O Romeo, dings; nob'
  Req 1: 'To be or than thy's'
  Req 2: 'KING HENRY:
Hast nowes'
✅ Test 1 passed
```

From this result, we can see that the requests are completed in the order they were added, and the output is valid.

### Priority Correctness

One low-priority long request + one high-priority short request. Verify the short one completes first under Priority policy, but second under FCFS. This will demonstrate the ability to use multiple types of policies.

```python

# ══════════════════════════════════════════════════════════════
# Test 2: Same requests under FCFS vs Priority — admission order flips
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 2: Priority jumping the queue")
print("=" * 60)

# ── FCFS: req 0 (lower id) admitted first ──
print("\n── FCFS run ──")
torch.manual_seed(42)
fcfs_reqs = [
    Request(id=0, prompt_tokens=encode("A " * 15), max_new_tokens=3, priority=2),
    Request(id=1, prompt_tokens=encode("B " * 3),  max_new_tokens=3, priority=0),
]
s_fcfs = scheduled_generate(model, fcfs_reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)

# ── Priority: req 1 (priority=0) admitted first ──
print("\n── Priority run ──")
torch.manual_seed(42)
prio_reqs = [
    Request(id=0, prompt_tokens=encode("A " * 15), max_new_tokens=3, priority=2),
    Request(id=1, prompt_tokens=encode("B " * 3),  max_new_tokens=3, priority=0),
]
s_prio = scheduled_generate(model, prio_reqs, policy="priority", token_budget=16, max_kv_tokens=256)

for req in fcfs_reqs + prio_reqs:
    assert req.status == "done", f"❌ Req {req.id} not done"
    assert req.num_generated == 3

# Check step logs above:
#   FCFS     → [step 0] prefill=0  (req 0 admitted first, long prompt hogs budget)
#   Priority → [step 0] prefill=1  (req 1 jumps queue despite arriving at same time)
print("\n✅ Test 2 passed — verify from logs: FCFS admits req 0 first, Priority admits req 1 first")

```

Here is the result:

```text
============================================================
Test 2: Priority jumping the queue
============================================================

── FCFS run ──

── Priority run ──

✅ Test 2 passed — verify from logs: FCFS admits req 0 first, Priority admits req 1 first
```

From this result, we can see that the FCFS policy admits req 0 first, while the Priority policy admits req 1 first. This is because the Priority policy prioritizes requests with lower priority values, and in this case, req 1 has a lower priority value than req 0.

### Preemption Correctness

Set max_kv_tokens low enough to force at least one preemption. Verify:

The evicted request's cache is cleared.
It re-prefills correctly and produces valid output.
Total output tokens match what you'd get without preemption (just slower).

```python

# ══════════════════════════════════════════════════════════════
# Test 3: Tight KV budget forces preemption
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 3: Preemption under memory pressure")
print("=" * 60)

torch.manual_seed(42)
reqs = [
    Request(id=0, prompt_tokens=encode("A " * 5), max_new_tokens=10, priority=2),  # low pri → victim
    Request(id=1, prompt_tokens=encode("B " * 5), max_new_tokens=5,  priority=0),  # high pri → stays
]
# 2 requests × 10-token prompt = 20 KV at admission. After a couple decode steps → exceeds 22.
s = scheduled_generate(model, reqs, policy="priority", token_budget=16, max_kv_tokens=22)

# Preemption must have fired
preempted_ids = [r.id for r in s.preempted]
assert len(s.preempted) > 0, "❌ No preemption occurred — lower max_kv_tokens"
assert 0 in preempted_ids, f"❌ Expected req 0 (low priority) to be preempted, got {preempted_ids}"

# Both requests still finish
for req in reqs:
    assert req.status == "done", f"❌ Req {req.id} stuck in status '{req.status}'"
    print(f"  Req {req.id}: {req.num_generated} tokens | '{decode(req.tokens_so_far)}'")

print(f"  Preempted IDs: {preempted_ids}")
print("✅ Test 3 passed")

```

Here is the result:

```text
============================================================
Test 3: Preemption under memory pressure
============================================================
  Req 0: 10 tokens | 'A A A A A cCafpier,-'
  Req 1: 5 tokens | 'B B B B B Plunk'
  Preempted IDs: [0]
✅ Test 3 passed
```

From this result, we can see that req 0 was preempted, while req 1 was not. This is because req 0 has a lower priority value than req 1, and req 1 has a shorter prompt than req 0. This is a good sign that our preemption logic is working correctly.

###  Preempted request re-enters correctly

After preemption, the re-admitted request should produce identical output to if it had never been preempted (same random seed). This validates that your clear_cache + full re-prefill is correct.

```python

# ══════════════════════════════════════════════════════════════
# Test 4: After preemption + re-prefill, KV cache is consistent
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 4: Preempted request re-enters correctly")
print("=" * 60)

# ── Baseline: req 0 alone, no preemption ──
torch.manual_seed(42)
baseline = Request(id=0, prompt_tokens=encode("A " * 5), max_new_tokens=10)
scheduled_generate(model, [baseline], policy="fcfs", token_budget=16, max_kv_tokens=256)

# ── Stress: req 0 gets preempted, must re-prefill ──
torch.manual_seed(42)
req0 = Request(id=0, prompt_tokens=encode("A " * 5), max_new_tokens=10, priority=2)
req1 = Request(id=1, prompt_tokens=encode("B " * 5), max_new_tokens=5,  priority=0)
s = scheduled_generate(model, [req0, req1], policy="priority", token_budget=16, max_kv_tokens=22)

assert req0.status == "done"
assert req0.num_generated == req0.max_new_tokens

# KEY CHECK: KV cache length must equal prompt + generated tokens.
# If generated_tokens wasn't reset on preemption, old tokens have no KV backing
# and this assertion will fail — exposing the bug.
expected_kv_len = len(req0.prompt_tokens) + req0.num_generated - 1
actual_kv_len = req0.kv_cache[(0, 0)][0].shape[1]
assert actual_kv_len == expected_kv_len, \
    f"❌ KV cache has {actual_kv_len} entries, expected {expected_kv_len}. " \
    f"Hint: generated_tokens should be reset to [] on preemption."

print(f"  Baseline:  '{decode(baseline.tokens_so_far)}'")
print(f"  Stressed:  '{decode(req0.tokens_so_far)}'")
print(f"  KV cache length: {actual_kv_len} ✓")
print("✅ Test 4 passed")

```

Here is the result:

```text
============================================================
Test 4: Preempted request re-enters correctly
============================================================
  Baseline:  'A A A A A drie as fa'
  Stressed:  'A A A A A Cafpter,-''
  KV cache length: 19 ✓
✅ Test 4 passed
```

From this result, we can see that the KV cache length is 19, which is equal to the number of prompt tokens plus the number of generated tokens minus 1. This is a good sign that our preemption logic is working correctly. 

## Errors that I Encountered:

Here are some errors I encountered while implementing this:

Errors:

```text

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_1412/2400345122.py in <cell line: 0>()
     12     Request(id=2, prompt_tokens=encode("KING HENRY:\n"), max_new_tokens=10),
     13 ]
---> 14 s = scheduled_generate(model, reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)
     15 
     16 for req in reqs:

/tmp/ipykernel_1412/2884244380.py in scheduled_generate(model, requests, policy, token_budget, max_kv_tokens)
    103                     p_req.prefill_cursor += chunk_size
    104 
--> 105                 if not prefill_chunk_tokens and not scheduler.active:
    106                     step += 1
    107                     continue

RuntimeError: Boolean value of Tensor with more than one value is ambiguous

```

In order to fix this, we need to chek if the length of the prefill_chunk_tokens is greater than 0 before we try to prefill it. The reason for this error is that Python doesn't know how to evaluate a tensor with more than one value as a boolean.

```text
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_1412/2400345122.py in <cell line: 0>()
     12     Request(id=2, prompt_tokens=encode("KING HENRY:\n"), max_new_tokens=10),
     13 ]
---> 14 s = scheduled_generate(model, reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)
     15 
     16 for req in reqs:

/tmp/ipykernel_1412/3168524644.py in scheduled_generate(model, requests, policy, token_budget, max_kv_tokens)
    122                         logits, _, new_kvs = model(prefill_chunk_tokens, past_kvs, pos=pos)
    123                     else:
--> 124                         logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos)
    125 
    126                     for li, bkv in enumerate(new_kvs):

/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1774             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775         else:
-> 1776             return self._call_impl(*args, **kwargs)
   1777 
   1778     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1785                 or _global_backward_pre_hooks or _global_backward_hooks
...
-> 2567     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
   2568 
   2569 

RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method 

```

In order to fix this, we need to make sure that all tensors are on the same device. We can do this by using the .to(device) method on the tensor. Simple fix!

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_1412/2036090065.py in <cell line: 0>()
     13     Request(id=1, prompt_tokens=encode("B " * 3),  max_new_tokens=3, priority=0),
     14 ]
---> 15 s_fcfs = scheduled_generate(model, fcfs_reqs, policy="fcfs", token_budget=16, max_kv_tokens=256)
     16 
     17 # ── Priority: req 1 (priority=0) admitted first ──

/tmp/ipykernel_1412/3878423562.py in scheduled_generate(model, requests, policy, token_budget, max_kv_tokens)
    120                             past_kvs.append(block_kv)
    121 
--> 122                         logits, _, new_kvs = model(prefill_chunk_tokens, past_kvs, pos=pos)
    123                     else:
    124                         logits, _, new_kvs = model(prefill_chunk_tokens, pos=pos)

/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1774             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775         else:
-> 1776             return self._call_impl(*args, **kwargs)
   1777 
   1778     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1785                 or _global_backward_pre_hooks or _global_backward_hooks
...
--> 195             targets = targets.view(B*T)
    196             loss = F.cross_entropy(logits, targets)
    197 

AttributeError: 'list' object has no attribute 'view'

```

The error here is that I was passing in the arguments to model() incorrectly. Once I changed the positions of the past_kvs and pos arguments, the error went away.

```text

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_1412/2036090065.py in <cell line: 0>()
     22     Request(id=1, prompt_tokens=encode("B " * 3),  max_new_tokens=3, priority=0),
     23 ]
---> 24 s_prio = scheduled_generate(model, prio_reqs, policy="priority", token_budget=16, max_kv_tokens=256)
     25 
     26 for req in fcfs_reqs + prio_reqs:

/tmp/ipykernel_1412/1488154673.py in scheduled_generate(model, requests, policy, token_budget, max_kv_tokens)
     71     for req in requests:
     72         req.arrival_time = step
---> 73         scheduler.add_request(req)
     74 
     75     model.eval()

/tmp/ipykernel_1412/3680500861.py in add_request(self, req)
     26     def add_request(self, req):
     27         key = self._sort_key(req)
---> 28         heapq.heappush(self.waiting, (*key, req.id, req))
     29 
     30     def is_done(self):

TypeError: Value after * must be an iterable, not NoneType

```

In order to fix this, I had to account for the possibility of the `priority` policy in my Scheduler. I had to add a method called `_sort_key` that checks how to return an request based upon the specified scheduling policy passed in. 

The method looks like this:

```python

def _sort_key(self, req):
    if self.policy == "fcfs":
        return (0, req.arrival_time)
    elif self.policy == "priority":
        return (req.priority, req.arrival_time)
```

```text
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_1412/122799207.py in <cell line: 0>()
     25 expected_kv_len = len(req0.prompt_tokens) + req0.num_generated
     26 actual_kv_len = req0.kv_cache[(0, 0)][0].shape[1]
---> 27 assert actual_kv_len == expected_kv_len, \
     28     f"❌ KV cache has {actual_kv_len} entries, expected {expected_kv_len}. " \
     29     f"Hint: generated_tokens should be reset to [] on preemption."

AssertionError: ❌ KV cache has 18 entries, expected 20. Hint: generated_tokens should be reset to [] on preemption

```
In order to fix this, we need to reset `generated_tokens` to an empty list when a request is preempted. When `_maybe_preempt` clears the KV cache and resets `prefill_cursor`, the old generated tokens still linger — but after re-prefill, those orphaned tokens have no KV backing, creating the 2-entry mismatch. Adding `victim.generated_tokens = []` right after `victim.prefill_cursor = 0` in `_maybe_preempt` fixes the core bug. After that fix, the assertion is still off by 1 because the last generated token is sampled from logits but never fed back to the model, so the KV cache is always 1 shorter than the full sequence. The correct expected length is `len(req0.prompt_tokens) + req0.num_generated - 1`.

The full code can be found here: [https://github.com/czhou578/multimodal-inference-visualizer/blob/main/nanogpt_scheduling.ipynb](https://github.com/czhou578/multimodal-inference-visualizer/blob/main/nanogpt_scheduling.ipynb)

CZ