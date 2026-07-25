---
layout: post
title: "Developing my own Benchmark! Part 2"
date: 2026-07-20
---

This is Part 2 of my effort to develop a more comprehensive benchmark for measuring the performance of large language models. Today, we will be focusing on building the `core_runner.py` file, which will serve as the foundation for our benchmarking framework.

This file will be responsible for orchestration and running all the benchmarking tasks through command line arguments.

## Helper Functions

## GPU Monitor

This function is the GPU monitor that is responsible for monitoring the GPU and collecting data. It is a background thread that runs continuously and collects a basket of metrics like utilization, memory usage, and power usage. 

We track the metrics in a dictionary and store them in a list. The monitor samples continuously and the final numbers are calculated from the list of samples.

We also measure within a window within a parameter of user specified length. This makes sure that we can isolate the metrics to a specific time period during different phases of the benchmark suite.

This class has a few helper functions that controls the monitor lifecycle, especially the start and stop methods. The stop method is responsible for stopping the monitor and saving the samples to a file as well as doing minor calculations of the stats to present it in a summary format.

```python

class GpuMonitor:
    """Background thread sampling GPU memory + power draw once per second."""

    FIELDS = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    # Keys that map to the CSV fields above
    KEYS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_power_w")

    def __init__(self, out_dir: Path, interval_s: float = 1.0):
        self.out_dir = out_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[dict[str, Any]] = []
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        # Per-length window tracking
        self._windows: dict[str, list[dict[str, Any]]] = {}
        self._idle_samples: list[dict[str, Any]] = []

    def _parse_row(self, out: str) -> dict[str, Any]:
        """Parse one nvidia-smi CSV line into a row dict."""
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        row: dict[str, Any] = {}
        for k, v in zip(self.KEYS, parts):
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                row[k] = None
        return row

    def _sample_once(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": time.time()}
        if self.has_nvidia_smi:
            out = _run([
                "nvidia-smi",
                f"--query-gpu={self.FIELDS}",
                "--format=csv,noheader,nounits",
            ])
            if out:
                parsed = self._parse_row(out)
                row.update(parsed)
        return row

    def _loop(self):
        while not self._stop.is_set():
            sample = self._sample_once()
            self.samples.append(sample)
            # Also record in every active window
            for window in self._windows.values():
                window.append(sample)
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Window management for per-length telemetry
    # ------------------------------------------------------------------ #

    def start_idle(self) -> None:
        """Begin recording an idle baseline before benchmark work."""
        self._idle_samples = []

    def record_idle(self) -> None:
        """Capture one sample into the idle baseline (caller must hold lock or schedule)."""
        sample = self._sample_once()
        self._idle_samples.append(sample)

    def start_window(self, name: str) -> None:
        """Start a new telemetry window by name."""
        self._windows[name] = []

    def stop_window(self, name: str) -> dict[str, Any] | None:
        """Stop the named window and return a summary.  Returns ``None`` if the
        window is empty (e.g. the benchmark loop was short and no sample was
        taken)."""
        window = self._windows.pop(name, None)
        if not window:
            return None
        return _window_summary(window, self.interval_s, self._idle_samples)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

        csv_path = self.out_dir / "gpu_samples.csv"
        if self.samples:
            keys = sorted({k for row in self.samples for k in row.keys()})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.samples)

        powers = [s["gpu_power_w"] for s in self.samples if s.get("gpu_power_w") is not None]
        mem = [s["gpu_mem_used_mib"] for s in self.samples if s.get("gpu_mem_used_mib") is not None]
        util = [s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None]
        mem_util = [s["gpu_mem_util_pct"] for s in self.samples if s.get("gpu_mem_util_pct") is not None]
        summary: dict[str, Any] = {
            "samples_csv": str(csv_path) if self.samples else None,
            "num_samples": len(self.samples),
            "gpu_util_avg_pct": round(statistics.mean(util), 2) if util else None,
            "gpu_util_peak_pct": round(max(util), 2) if util else None,
            "gpu_power_avg_w": round(statistics.mean(powers), 2) if powers else None,
            "gpu_power_peak_w": round(max(powers), 2) if powers else None,
            "gpu_mem_used_avg_mib": round(statistics.mean(mem), 1) if mem else None,
            "gpu_mem_used_peak_mib": round(max(mem), 1) if mem else None,
            "gpu_mem_util_avg_pct": round(statistics.mean(mem_util), 2) if mem_util else None,
            "gpu_mem_util_peak_pct": round(max(mem_util), 2) if mem_util else None,
        }
        if powers:
            duration_h = (len(powers) * self.interval_s) / 3600.0
            summary["energy_wh"] = round(statistics.mean(powers) * duration_h, 4)
        return summary

```

## Model Client

THe model client is the main class that handles the communication with the model server. 

In the beginning, it is responsible for waiting for the model to be ready. This is a dedicated function that waits for the vLLM server to finish intializing, and then send a dummy chat completion request to the server to ensure that it is ready to serve requests.

We send a dummy message because the server may take some time to initialize, and we want to make sure that it is ready before we start the benchmark. 

There is also a function called `generate()` that takes in temperature, prompt, max_tokens, and cache salt. In this function, we send a request with the given parameters to the model server and return the response. 

The cache salt is used to prevent the model server from caching the response, which would cause the prefix caching benchmark to be biased in terms of the latency of the model server.

Of course, not every possible server configuration may accept a cache salt, so we have to have a separate function called `preflight_cache_salt()` that tries to send a request with a cache salt and returns `True` if the request is successful.

## Deep Context + Building Prompts

We have a function here called `run_deep_context()` that takes in the model client, the prompt, and the number of tokens to generate. 

This function is responsible for measuring the time to first token (TTFT) and prefill throughput at extended context lengths. 

We store all the results in a dictionary. For every context length, we build the prompt. 

The prompt building is done by the `build_prompt_of_length()` function at the top of the file. On first glance, this doesn't seem to be necessary until we understand how LLM's truly behave.

In earlier versions of this benchmark, I simply used a repetitive prompt that generated up to the maximum token length to send to the model. But after doing this many times, the model actually realized that the prompt was repetitive and started generating meta-commentary instead of creative content (ie. "I notice that you are repeating the same sentence over and over again.").

This meta commentary was repeatedly hitting a wall at around 500 tokens and immediately stopping the stream after, ensuring that my decode-speed and energy consumption numbers remained low and wrong.


The solution is structured diversity, where the function defines 5 distinct creative-writing blocks about related topics (underwater civilization, marine life, trade routes, etc.). It then loops through them, appending blocks one at a time until the prompt reaches the target token count. This gives you:

Exact length control: you know you're testing a 16K-token prompt, not a 15,800-token one by accident

Substantive content: the model has real material to work with across each block, producing long, coherent generations

No repetition loops: the model doesn't try to "complete" a pattern because each block is a distinct creative direction

Deterministic fallback: after concatenation, it trims/cuts at an exact token boundary using the tokenizer, so the final prompt is precisely the requested length

Without this, every benchmark parameter sweep (latency at 32, 128, 512, 2K, 8K, 16K tokens) would either produce gibberish or get truncated early, making the whole measurement useless.

In the end, the results are sent to the vLLM server via the `generate()` function and returned to the user.

The `run_latency_sweep()` function is responsible for measuring the prefill throughput and latency at different prompt lengths. It works relatively the same way. 


## Decode Speed

We have a benchmark function called `run_decode_speed()` that takes in the model client, the output lengths, and the fixed prompt tokens. 

This function is responsible for measuring the decode speed at different output lengths. 

It generates a single prompt that serves as context. It then loops through the output lengths and generates a response for each length. 

It captures the total time, decode time, and token per second rate for each response. The total time can be split into the TTFT and decode time through the equation: 

decode_time = gen.total_time_s - gen.ttft_s

We also want to compute per-token gaps, to measure the wall-clock gap between consecutive tokens. We can do this by using the numbers collected in the `per_token_times` attribute of the `GenerationResult` object.

The fastest single token, slowest single token, and average token time are all very important since even if the GPU stutters, the min/max spread can be used to determine consistency.

## Reasoning Token Count

This benchmark function is responsible for measuring the reasoning token count at different output lengths.

We ideally want to measure this because it can give us an idea of how much of the model's output is actually reasoning, and how much is just the final answer. 

The function handles three types of format for reasoning tokens: 

1. Qwen3 XML tags:  <antThinking>...thinking...</antThinking>answer...
2. Anthropic XML tags: <thinking>...thinking...</thinking>answer...
3. Plain-text reasoning: common reasoning markers followed by structured answer ie:

```text

    reasoning_markers = [
        "here's a thinking",
        "here is a thinking",
        "let me think",
        "let me analyze",
        "let's think",
        "step by step",
        "first,",
        "firstly,",
        "to solve",
        "to analyze",
        "break this down",
        "breaking this down",
    ]

```

For each case, it identifies the thinking tokens opening and closing tags, and the answer tokens.

It then calculates the ratio of thinking tokens to answer tokens, and the exact number of thinking/answer tokens and returns the result.

In the `run_reasoning_benchmark()` function, we call this `analyze_reasoning_tokens()` function to get the reasoning token count. This is done for each prompt in the list of prompts passed in.

## Speculative Decode Comparison

For the speculative decode benchmark, the setup is that we run a model using vLLM with speculative decoding enabled, and then run it again with speculative decoding disabled. We then compare the results of the two runs to see how much faster the speculative decoding is.

This requires us to have a single function that compare the results of the two runs.

It takes three arguments: the results of the speculative decoding run, the results of the baseline run, and the output lengths. 

It then loops through the output lengths and compares the results of the two runs. 

For each output length, it calculates the token per second rate for both runs and compares them. 

It also calculates the time to first token (TTFT) for both runs and compares them, and also the TPS (token per second) improvement. 

`core_runner.py` runs the two variants (it starts/stops the vLLM server twice with different flags, lines 1339-1380), but it delegates the comparison logic to this function. That keeps the core runner focused on orchestration and lets the comparison be tested/reused independently. The raw results are also saved individually (`spec_enabled.json`, `spec_disabled.json`) alongside the comparison 

## Orchestration

In this section, we have a list of wrapper functions that perform various operations related to the runner.

This includes loading the model configuration, creating a run directory, and saving the results to a JSON file.

## Argument Parsing

At the bottom, we have the big function called `main()` that parses the command-line arguments and runs the benchmarks. Based upon which flags you pass in, it will run the appropriate benchmark or skip it. 

All the result files are saved in the `results` directory.

The lifecycle is that in the beginning, the .yaml config file is loaded. Then, a run directory is created. Then, the model is loaded. Then, the benchmark is run. Then, the results are saved. Then, the model is unloaded.

I will be posting a follow up describing the individual benchmark details.

Stay tuned! The code can be found here: [https://github.com/czhou578/model-benchmarks/blob/main/core_runner.py](https://github.com/czhou578/model-benchmarks/blob/main/core_runner.py)

CZ








