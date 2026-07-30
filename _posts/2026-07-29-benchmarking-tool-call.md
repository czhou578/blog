---
layout: post
title: "The tool-calling benchmark I built to understand what LLMs can and can't do"
date: 2026-07-28
---

Every model marketing page today claims "excellent tool calling." The problem? That claim is impossible to compare across models because there's no standardized way to measure it. Is a model that calls the right tool but fills in wrong params "good"? Is a model that chains three tools together perfectly but uses the wrong one "better"?

I built a benchmark to answer these questions. Here's what it does, why it matters, and what I learned building it.

![Toolcalling Benchmark]({{ site.baseurl }}/images/benchmark-toolcalling.png)

## What tool-calling actually means

When an LLM "calls a tool," the model receives a description of available tools (their names, what they do, what parameters they expect), and it must select the right one and fill in the parameters correctly.

In practice, tools form *chains*, where the output of one feeds into the next. Stock price → currency converter → spending calculator. Weather lookup → search query → email summary. And when a tool returns an error? The model should recognize it, adjust, and try again.

## The benchmark design

The benchmark I built covers five task categories across three phases, each with its own scoring logic:

**Phase 1: Single-Tool Invocation**
The simplest case: given a task, does the model call the right tool with correct parameters? This tests basic tool selection and parameter validation. A prompt like "What's the current price of NVDA?" should call `get_stock_price` with `symbol: "NVDA"`.

**Phase 2: Multi-Tool Chaining**
Now the model must orchestrate multiple tools across turns. The first tool's output feeds into the second. A task like "What's the 20% tip in USD on a £50 meal in the UK?" requires: get the exchange rate, convert the currency, then calculate the tip. The benchmark checks three things: correct tool order, correct data flow between turns, and knowing when to stop.

**Phase 3: Schema Compliance & Error Recovery**
The model must respect strict JSON Schema constraints: string length limits, enum values, regex patterns, required fields, and nested objects. When a tool call returns an error (e.g., invalid stock symbol), the model should self-correct rather than repeating the same call with the same wrong params.

Each category has its own overlay scoring logic because the pass/fail criteria are fundamentally different. A refusal task (where the correct answer is *not* calling any tool) is scored completely differently from a schema compliance task.

## Key design decisions

### Deterministic mock responses

The benchmark doesn't call real APIs. Instead, it uses a mock tool executor that returns deterministic responses for common inputs. `get_stock_price("NVDA")` always returns `{"price": 150.0, "symbol": "NVDA", "change_pct": 2.3}`. `get_weather("Tokyo")` returns `{"temp": 22, "condition": "sunny"}`.

This means every run is reproducible. If a model passes a test today, it passes tomorrow.

The mock executor also feeds results back into the conversation for multi-turn tasks. When a task asks "What's 20% tip on a £50 meal?" the model first calls `convert_currency` with `from_currency: "GBP", to_currency: "USD"`. The mock returns `{"result": 63.5}`. Then the model's next call to `calculate` uses `expression: "63.5 * 0.20"`. The benchmark checks that the model actually used the previous result (63.5) rather than just calling the calculator with hardcoded numbers.

### Composite scoring, not just pass/fail

Instead of a binary correct/incorrect, each task produces a *composite score*: `tool_correct × params_complete × params_valid`. A model that calls the right tool but misses one required parameter still gets partial credit (e.g., 0.5). A model that calls the wrong tool gets zero regardless of parameter quality.

This granularity matters. Most benchmarks only track pass/fail, which hides a lot of signal.

### Multi-turn orchestration scoring

For chained tasks, the benchmark tracks:
- **orchestration_correct**: Did the model call the right tools in the right order?
- **data_flow_correct**: Did the model use the previous tool's output as input?
- **turns_correct**: Did the model stop at the right turn?

A model that perfectly chains three tools but continues unnecessarily still fails the turns check. A model that calls tools in the right order but doesn't use the previous result fails data flow. These are independently tracked and combined into a multi_score.

### Ambiguous task handling

Not all prompts have one correct answer. "Which tool would help me..." is fundamentally different from "Look up the weather." The benchmark handles three ambiguous strategies:

1. **no_tool**: The model should not call any tool at all.
2. **underspecified**: The model should ask for clarification. But if it makes a reasonable assumption instead (calling a tool with no hallucinated params), that's also accepted.
3. **assumption**: The model may make a reasonable default call.

For assumption tasks, the key metric is *no hallucinated params*: the model shouldn't invent values the prompt didn't provide. Here's what that looks like. A task like "What's the current weather?" provides no city, so the model must choose: ask for clarification, or make a reasonable assumption. If it calls `get_weather` with `location: "London"` when the prompt never mentioned a city, the benchmark flags that as a hallucinated param. If it instead asks "which city should I look up?" the benchmark accepts that too. For assumption tasks like "How much would a $100 dinner cost in Japan?", the prompt gives enough context for a reasonable call (convert currency). The benchmark checks that the model used the actual numbers from the prompt, not fabricated ones.

## Why we should care

Here's what I think is missing from the LLM ecosystem:

**Tool calling is the gatekeeper to utility.** An LLM that writes code but can't reliably call tools is a parrot. An LLM that can call tools reliably is an agent. The gap between those two capabilities is measured by tool calling, but most evaluations skip that measurement entirely.

**Current "leaderboards" don't test what matters.** Most benchmarks use 2-4 tools with simple, well-defined parameters. Real-world tool schemas are complex: nested objects with required fields, regex-validated strings, enums with specific values, arrays with constrained items. A model that passes a toy benchmark may fail catastrophically on real schemas.

**Multi-turn chains expose a real weakness.** I've found that models that are great at single-tool calls often fail at chains. They'll call the right first tool, but then either (a) repeat it, (b) call the wrong second tool, or (c) not use the previous result in their next call. This is the difference between a model that *understands* the task and one that just *recognizes patterns*.

**Error recovery tests true intelligence.** When a tool returns an error, most models just repeat the same call with the same wrong params. A model that actually reads the error, understands what went wrong, and produces a corrected call in the next turn, hitting the right number of steps with valid parameters, is operating on a fundamentally different level. This is the difference between a tool and an agent.

## The technical architecture

The benchmark is structured as a CLI tool that works with vLLM endpoints:

```
python -m benchmarks.tool_calling --tasks datasets/tool_calling_tasks.yaml --base-url http://localhost:8000/v1 --model <model-name>
```

Tasks are defined in YAML, which makes it easy to add new tests without touching code. Each task specifies the prompt, expected tool, expected parameters, tools available, and scoring overlay.

Here's what happens when a single task runs:

1. **Build the request**: The task prompt becomes the user message. All available tool definitions are attached so the model knows what options it has. `temperature: 0.0` ensures deterministic output, and `tool_choice: "auto"` lets the model decide whether to call a tool or not.
2. **Send to vLLM**: A POST request hits the `/chat/completions` endpoint. The response contains the model's decision: either a text reply, or a list of tool calls with names and parameters.
3. **Parse the response**: The benchmark extracts extracts tool calls by reading the `tool_calls` array from the response. For single-tool tasks, only the first tool call matters. For multi-tool tasks, all calls in a turn are collected.
4. **Execute tools**: Single-tool tasks stop here. The mock response is recorded. Multi-tool tasks continue: each tool's response is fed back into the conversation as a `tool` message, and the loop repeats.
5. **Score against the expected outcome**: Did the model call the right tool? Did it fill in the right parameters? For multi-tool tasks, was the sequence correct and did data flow between turns?
6. **Apply category overlay**: If the task is ambiguous, schema-compliance, or error-recovery, an additional scoring layer runs on top of the baseline scoring. This is where refusal tasks, underspecified prompts, and error self-correction get their specific pass/fail criteria.

A single-tool task follows steps 1 through 6 and returns. A multi-tool task repeats steps 2 through 4 in a loop until the model stops calling tools (or hits a turn limit), then scores the full sequence.

The scoring engine has three layers:
1. **Baseline scoring**: tool match, parameter completeness, parameter validity, composite score
2. **Overlay scoring**: category-specific logic for ambiguous, schema compliance, and error recovery tasks
3. **Aggregation**: category-level statistics, failure mode tracking, weighted composite

The weighted composite score combines six components: one measure from each major dimension of tool-calling:
- **Tool accuracy** (Phase 1): pass rate on single-tool selection
- **Parameter completeness** (Phase 1): fraction of required params filled
- **Parameter correctness** (Phase 1): fraction of valid parameter values
- **Multi-tool score** (Phase 2): orchestration, data flow, and turn correctness
- **Schema compliance** (Phase 3): strict constraint satisfaction
- **Refusal correctness** (ambiguous category): handling of "no tool" tasks

Each component gets equal weight (1.0), so no single dimension dominates the final score. The composite pulls its value from the corresponding category score: for example, tool accuracy comes from the `single_tool` category's `tool_accuracy` field. If a category is missing entirely, its component defaults to 0.0.

## What the output tells you

The benchmark produces a detailed JSON report with per-task results, category scores, and failure modes. The failure modes tracking is particularly useful: an illustrative example of what the output tells you:

```json
{
  "wrong_tool_selected": 3,
  "missing_required_param": 5,
  "wrong_tool_sequence": 7,
  "data_flow_error": 8,
  "wrong_turn_count": 4,
  "schema_constraint_violation": 6,
  "error_no_retry": 9,
  "hallucinated_param_value": 2
}
```

This tells you *what* kind of behavior the model is exhibiting, not just *that* it failed. A model with many `data_flow_error` failures is fundamentally not connecting information between turns. A model with many `error_no_retry` failures doesn't adapt after errors. These are different behavioral patterns that point to different root causes.

## Building this

The most challenging part was the multi-turn orchestration. Each turn is a full API call to the model, and the conversation state (previous tool calls, responses, intermediate results) must be managed correctly. The mock tool executor needs to handle parameterized lookups: not just "this tool returns this response" but "this tool with *these specific params* returns this response."

The scoring overlays were the other challenge. Each category has fundamentally different pass/fail criteria, and the code needed to handle them without becoming a tangle of if/else chains. The overlay system uses separate functions for ambiguous, schema compliance, and error recovery scoring, keeping each category's logic isolated and testable.

## Parsing Helpers

To help the harness, I wrote a couple of parsing helpers, which mainly serve to extract tool calls from vLLM's responses, and to check whether a value matches the expected JSON schema type. In addition, the scoring of the parameters, which gives back the completeness_ratio and validity ratios are all calculated by a helper function. 

I delegate helper functions only when an operation can be repeatedly used, which is a good practice for general software development.

---

*This post is part of my series on developing benchmarks for AI systems. Previous posts cover general benchmarking methodology and my approach to measuring model capabilities.*

The full code can be found here: [https://github.com/czhou578/model-benchmarks/blob/main/benchmarks/tool_calling.py](https://github.com/czhou578/model-benchmarks/blob/main/benchmarks/tool_calling.py)

CZ
