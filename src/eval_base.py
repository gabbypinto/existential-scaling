"""Shared infrastructure for all benchmark evals."""
import json
import os
import threading
import time
from pathlib import Path

import requests
import yaml

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _vram_monitor(stop_event: threading.Event, peak_mb: list) -> None:
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        while not stop_event.is_set():
            used_mb = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles) / 1024 ** 2
            if used_mb > peak_mb[0]:
                peak_mb[0] = used_mb
            stop_event.wait(0.5)
    except Exception:
        pass


def build_prompt(problem: str, cfg: dict) -> str:
    pre = cfg.get("pre_prompt", "").strip()
    post = cfg.get("post_prompt", "").strip()
    parts = [p for p in [pre, problem, post] if p]
    return "\n\n".join(parts)


def query(problem: str, cfg: dict, temperature: float) -> tuple[str, str, dict]:
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    enable_thinking = cfg.get("enable_thinking", False)
    thinking_budget = cfg.get("thinking_budget", 0)
    max_output_tokens = cfg.get("max_output_tokens", 2048)
    timeout = cfg.get("request_timeout", 600)

    prompt = build_prompt(problem, cfg)

    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": thinking_budget + max_output_tokens,
        "temperature": temperature,
        "top_p": cfg.get("top_p", 1.0),
        "repetition_penalty": cfg.get("repetition_penalty", 1.0),
        "chat_template_kwargs": {"enable_thinking": enable_thinking, "thinking_budget": thinking_budget},
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    peak_mb: list[float] = [0.0]
    stop_event = threading.Event()
    if NVML_AVAILABLE:
        threading.Thread(target=_vram_monitor, args=(stop_event, peak_mb), daemon=True).start()

    thinking = ""
    answer = ""
    buf = ""
    usage: dict = {}
    in_think = False
    showed_think_header = False
    showed_resp_header = False

    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = line.decode("utf-8")
            if not data.startswith("data: ") or data[6:] == "[DONE]":
                continue
            chunk = json.loads(data[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            buf += delta.get("content") or ""

            while True:
                if not in_think:
                    idx = buf.find(THINK_OPEN)
                    if idx == -1:
                        safe_len = max(0, len(buf) - len(THINK_OPEN) + 1)
                        chunk_text = buf[:safe_len]
                        if chunk_text:
                            if not showed_resp_header:
                                if showed_think_header:
                                    print("\n------- END THINKING ----\n")
                                print("------- RESPONSE --------")
                                showed_resp_header = True
                            answer += chunk_text
                            print(chunk_text, end="", flush=True)
                        buf = buf[safe_len:]
                        break
                    else:
                        pre = buf[:idx]
                        if pre:
                            if not showed_resp_header:
                                print("------- RESPONSE --------")
                                showed_resp_header = True
                            answer += pre
                            print(pre, end="", flush=True)
                        buf = buf[idx + len(THINK_OPEN):]
                        in_think = True
                        if not showed_think_header:
                            print("------- THINKING --------")
                            showed_think_header = True
                else:
                    idx = buf.find(THINK_CLOSE)
                    if idx == -1:
                        safe_len = max(0, len(buf) - len(THINK_CLOSE) + 1)
                        chunk_text = buf[:safe_len]
                        if chunk_text:
                            thinking += chunk_text
                            print(chunk_text, end="", flush=True)
                        buf = buf[safe_len:]
                        break
                    else:
                        chunk_text = buf[:idx]
                        thinking += chunk_text
                        print(chunk_text, end="", flush=True)
                        buf = buf[idx + len(THINK_CLOSE):]
                        in_think = False

    if buf:
        if not showed_resp_header:
            if showed_think_header:
                print("\n------- END THINKING ----\n")
            print("------- RESPONSE --------")
            showed_resp_header = True
        answer += buf
        print(buf, end="", flush=True)

    if showed_resp_header:
        print("\n------- END RESPONSE ----")
    print()

    stop_event.set()

    metrics = {
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens":      usage.get("total_tokens"),
        "peak_vram_mb":      round(peak_mb[0], 1) if NVML_AVAILABLE else None,
    }
    return thinking.strip(), answer.strip(), metrics


def _avg(key: str, trials: list[dict]):
    vals = [r[key] for r in trials if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def run_eval(benchmark, cfg: dict) -> None:
    if "model" not in cfg:
        cfg["model"] = os.environ.get("MODEL", "")
    if "port" not in cfg:
        cfg["port"] = int(os.environ.get("PORT", ""))

    if not cfg.get("model"):
        raise ValueError("model not set — add to model.yaml or set MODEL env var")
    if not cfg.get("port"):
        raise ValueError("port not set — add to model.yaml or set PORT env var")

    problems = benchmark.load_problems(cfg)

    limit = cfg.get("limit")
    if limit is not None:
        problems = problems[:limit]

    benchmark_name = cfg.get("benchmark", "eval")
    model_name = cfg["model"]
    model_short = model_name.split("/")[-1].lower()

    log_dir = cfg.get("log_dir") or f"logs/{benchmark_name}/{model_short}"
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    num_rounds = cfg.get("num_rounds", 1)
    temperature = (
        cfg.get("thinking_temp", 0.6)
        if cfg.get("enable_thinking", False)
        else cfg.get("nonthinking_temp", 0.0)
    )

    config_block = {
        "model":             cfg["model"],
        "port":              cfg["port"],
        "benchmark":         benchmark_name,
        "context_window":    cfg.get("context_window"),
        "enable_thinking":   cfg.get("enable_thinking", False),
        "temperature":       temperature,
        "thinking_budget":   cfg.get("thinking_budget", 0),
        "max_output_tokens": cfg.get("max_output_tokens", 2048),
        "top_p":             cfg.get("top_p"),
        "repetition_penalty": cfg.get("repetition_penalty"),
        "num_rounds":        num_rounds,
        "log_dir":           log_dir,
        "pre_prompt":        cfg.get("pre_prompt", "").strip(),
        "post_prompt":       cfg.get("post_prompt", "").strip(),
    }

    print(f"Benchmark: {benchmark_name}")
    print(f"Model: {cfg['model']}")
    print(f"Thinking: {cfg.get('enable_thinking')} | Temperature: {temperature}")
    print(f"Running {num_rounds} rounds x {len(problems)} problems\n")

    all_results: dict[int, dict] = {idx: {} for idx in range(1, len(problems) + 1)}

    for rnd in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{num_rounds}")
        print(f"{'='*60}\n")

        round_results = {}

        for idx, row in enumerate(problems, start=1):
            label = benchmark.get_label(row)
            print(f"[Round {rnd} | {idx:02d}/{len(problems)}] {label}\n")

            t0 = time.time()
            thinking, answer, metrics = query(benchmark.get_question_text(row), cfg, temperature)
            elapsed = time.time() - t0

            completion_tokens = metrics.get("completion_tokens") or 0
            tokens_per_sec = round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens else None
            metrics["tokens_per_sec"] = tokens_per_sec

            result = benchmark.build_result(row, thinking, answer, metrics, elapsed)
            result["elapsed_s"] = round(elapsed, 2)
            result["tokens_per_sec"] = tokens_per_sec
            result["prompt_tokens"] = metrics.get("prompt_tokens")
            result["completion_tokens"] = metrics.get("completion_tokens")
            result["total_tokens"] = metrics.get("total_tokens")
            result["peak_vram_mb"] = metrics.get("peak_vram_mb")

            correct = result.get("correct", result.get("passed_all_tests", False))
            print(f"\n({elapsed:.1f}s | {tokens_per_sec} tok/s | {'PASS' if correct else 'FAIL'} | peak VRAM {metrics.get('peak_vram_mb')} MB)\n")

            round_results[f"question_{idx}"] = result
            all_results[idx][rnd] = result

            round_file = log_path / f"round-{rnd}_results.json"
            round_file.write_text(json.dumps({"config": config_block, "results": round_results}, indent=2, ensure_ascii=False))

        print(f"Round {rnd} complete -> {round_file}")

    # pass@1: question passes if at least 1 round is correct
    per_question = {}
    for idx in range(1, len(problems) + 1):
        trials = all_results[idx]
        passing_rounds = [
            r for r, res in trials.items()
            if res.get("correct", res.get("passed_all_tests", False))
        ]
        entry = benchmark.build_summary_entry(problems[idx - 1], passing_rounds)
        entry["pass_at_1"] = len(passing_rounds) >= 1
        entry["num_correct"] = len(passing_rounds)
        entry["passing_rounds"] = passing_rounds
        per_question[f"question_{idx}"] = entry

    num_pass = sum(1 for q in per_question.values() if q["pass_at_1"])
    overall_pass_at_1 = num_pass / len(problems) if problems else 0.0

    all_trials = [res for trials in all_results.values() for res in trials.values()]

    summary = {
        "config":             config_block,
        "overall_pass_at_1":  overall_pass_at_1,
        "questions_passed":   num_pass,
        "total_questions":    len(problems),
        "avg_elapsed_s":      _avg("elapsed_s", all_trials),
        "avg_prompt_tokens":  _avg("prompt_tokens", all_trials),
        "avg_completion_tokens": _avg("completion_tokens", all_trials),
        "avg_total_tokens":   _avg("total_tokens", all_trials),
        "avg_tokens_per_sec": _avg("tokens_per_sec", all_trials),
        "avg_peak_vram_mb":   _avg("peak_vram_mb", all_trials),
        "per_question":       per_question,
    }

    summary_file = log_path / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nDone!")
    print(f"Pass@1: {num_pass}/{len(problems)} = {overall_pass_at_1:.1%}")
    print(f"Summary saved -> {summary_file}")
