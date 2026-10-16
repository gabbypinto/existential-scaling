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

# llama.cpp's server default. The eval never sends min_p, so llama.cpp runs sample
# with 0.05; vLLM requests send it explicitly so both engines sample the same way.
LLAMACPP_DEFAULT_MIN_P = 0.05

_print_lock = threading.Lock()
_engine_cache: dict[int, str] = {}
_engine_lock = threading.Lock()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def log(*args) -> None:
    """print() that keeps lines from concurrent requests from interleaving."""
    with _print_lock:
        print(*args, flush=True)


def detect_engine(cfg: dict) -> str:
    """'vllm' or 'llamacpp': cfg/ENGINE override, else the server's /v1/models owned_by."""
    engine = cfg.get("engine") or os.environ.get("ENGINE")
    if engine:
        return engine
    port = cfg["port"]
    with _engine_lock:
        if port not in _engine_cache:
            engine = "llamacpp"
            try:
                r = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
                models = r.json().get("data", [])
                if models and models[0].get("owned_by") == "vllm":
                    engine = "vllm"
            except Exception:
                pass
            _engine_cache[port] = engine
        return _engine_cache[port]


def budget_enforced(cfg: dict) -> bool:
    return bool(
        cfg.get("enable_thinking", False)
        and cfg.get("thinking_budget", 0) > 0
        and cfg.get("enforce_thinking_budget", True)
    )


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


def query(problem: str, cfg: dict, temperature: float, quiet: bool = False) -> tuple[str, str, dict]:
    """Send one problem and stream the reply. quiet=True prints nothing (concurrent runs)."""
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    thinking_budget = cfg.get("thinking_budget", 0)
    max_output_tokens = cfg.get("max_output_tokens", 2048)
    timeout = cfg.get("request_timeout", 600)
    engine = detect_engine(cfg)

    def say(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    prompt = build_prompt(problem, cfg)

    messages = []
    experimental = cfg.get("system_prompt", "").strip()
    if experimental:
        messages.append({"role": "system", "content": experimental})
    messages.append({"role": "user", "content": prompt})

    context_window = cfg.get("context_window", 32768)

    # Enforced budget, llama.cpp: phase 1 may only think for thinking_budget tokens;
    # if it runs out, _force_answer() closes </think> and generates the answer with
    # up to max_output_tokens. vLLM does the same server-side via thinking_token_budget.
    # Not enforced: one request shares the whole budget.
    enforce_budget = budget_enforced(cfg)
    two_phase = enforce_budget and engine != "vllm"
    max_tokens = thinking_budget if two_phase else thinking_budget + max_output_tokens

    if engine == "vllm":
        # Mirror what llama.cpp applies: its min_p default, and no repetition_penalty
        # (llama.cpp only reads repeat_penalty, so the one we send has no effect there).
        sampling = {
            "temperature": temperature,
            "top_p": cfg.get("top_p", 1.0),
            "top_k": cfg.get("top_k", 0),
            "presence_penalty": cfg.get("presence_penalty", 0.0),
            "min_p": LLAMACPP_DEFAULT_MIN_P,
        }
    else:
        sampling = {
            "temperature": temperature,
            "top_p": cfg.get("top_p", 1.0),
            "top_k": cfg.get("top_k", 0),  # llama.cpp: 0 = disabled
            "presence_penalty": cfg.get("presence_penalty", 0.0),
            "repetition_penalty": cfg.get("repetition_penalty", 1.0),
        }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": min(max_tokens, context_window),
        **sampling,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if engine == "vllm" and enforce_budget:
        payload["thinking_token_budget"] = thinking_budget

    peak_mb: list[float] = [0.0]
    stop_event = threading.Event()
    if NVML_AVAILABLE and not quiet:
        threading.Thread(target=_vram_monitor, args=(stop_event, peak_mb), daemon=True).start()

    thinking = ""
    answer = ""
    buf = ""
    usage: dict = {}
    timings: dict = {}
    finish_reason: str | None = None
    in_think = False
    showed_think_header = False
    showed_resp_header = False
    ttft: float | None = None
    seen_reasoning_content = False

    # printing out log metrics in case initial prompt procesing takes a while
    first_token_event = threading.Event()
    def _prefill_heartbeat():
        t = 0
        while not first_token_event.wait(timeout=10):
            t += 10
            print(f"  [prefill: {t}s elapsed...]", flush=True)
    if not quiet:
        threading.Thread(target=_prefill_heartbeat, daemon=True).start()

    say("  Querying model...", flush=True)
    t_request_start = time.time()

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
            # llama.cpp sends timings at the top level of the chunk, not in usage
            if chunk.get("timings"):
                timings = chunk["timings"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta", {})
            # vLLM calls the field "reasoning" (formerly "reasoning_content", as llama.cpp does)
            reasoning_token = delta.get("reasoning") or delta.get("reasoning_content") or ""
            token = delta.get("content") or ""

            # Primary path: the server separates thinking from the answer
            # (llama.cpp --reasoning-format deepseek-legacy, vLLM --reasoning-parser).
            if reasoning_token:
                seen_reasoning_content = True
                if not first_token_event.is_set():
                    first_token_event.set()
                    ttft = time.time() - t_request_start
                    say(f"  First token: {ttft:.1f}s", flush=True)
                if not showed_think_header:
                    say("------- THINKING --------", flush=True)
                    showed_think_header = True
                thinking += reasoning_token
                say(reasoning_token, end="", flush=True)

            if token:
                if not first_token_event.is_set():
                    first_token_event.set()
                    ttft = time.time() - t_request_start
                    say(f"  First token: {ttft:.1f}s", flush=True)
                # deepseek-legacy also echoes <think> tags in content — strip
                # them to avoid double-counting when reasoning_content is active.
                if seen_reasoning_content:
                    token = token.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
                buf += token

            while True:
                if not in_think:
                    idx_open  = buf.find(THINK_OPEN)
                    idx_close = buf.find(THINK_CLOSE)

                    # Implicit thinking: model emits </think> before any <think>
                    # (e.g. DeepSeek R1 outputs reasoning without an opening tag)
                    if idx_close != -1 and (idx_open == -1 or idx_close < idx_open):
                        chunk_text = buf[:idx_close]
                        thinking = answer + chunk_text  # retroactively reclassify
                        answer = ""
                        showed_resp_header = False
                        buf = buf[idx_close + len(THINK_CLOSE):]
                        in_think = False
                        showed_think_header = True
                        say("\n------- END THINKING ----\n", flush=True)
                        continue

                    idx = idx_open
                    if idx == -1:
                        safe_len = max(0, len(buf) - max(len(THINK_OPEN), len(THINK_CLOSE)) + 1)
                        chunk_text = buf[:safe_len]
                        if chunk_text:
                            if not showed_resp_header:
                                if showed_think_header:
                                    say("\n------- END THINKING ----\n")
                                say("------- RESPONSE --------")
                                showed_resp_header = True
                            answer += chunk_text
                            say(chunk_text, end="", flush=True)
                        buf = buf[safe_len:]
                        break
                    else:
                        pre = buf[:idx]
                        if pre:
                            if not showed_resp_header:
                                say("------- RESPONSE --------")
                                showed_resp_header = True
                            answer += pre
                            say(pre, end="", flush=True)
                        buf = buf[idx + len(THINK_OPEN):]
                        in_think = True
                        if not showed_think_header:
                            say("------- THINKING --------")
                            showed_think_header = True
                else:
                    idx = buf.find(THINK_CLOSE)
                    if idx == -1:
                        safe_len = max(0, len(buf) - len(THINK_CLOSE) + 1)
                        chunk_text = buf[:safe_len]
                        if chunk_text:
                            thinking += chunk_text
                            say(chunk_text, end="", flush=True)
                        buf = buf[safe_len:]
                        break
                    else:
                        chunk_text = buf[:idx]
                        thinking += chunk_text
                        say(chunk_text, end="", flush=True)
                        buf = buf[idx + len(THINK_CLOSE):]
                        in_think = False

    first_token_event.set()  # stop heartbeat if response ended without content

    # cut off mid-thought: the held-back tail belongs to the thinking, not the answer
    if in_think and buf:
        thinking += buf
        say(buf, end="", flush=True)
        buf = ""

    if buf:
        if not showed_resp_header:
            if showed_think_header:
                say("\n------- END THINKING ----\n")
            say("------- RESPONSE --------")
            showed_resp_header = True
        answer += buf
        say(buf, end="", flush=True)

    completion_tokens = usage.get("completion_tokens")
    prompt_ms = timings.get("prompt_ms")
    generation_ms = timings.get("predicted_ms")
    tokens_per_sec_llama = timings.get("predicted_per_second")
    budget_forced = False

    if two_phase and finish_reason == "length":
        still_thinking = in_think or not answer.strip()
        if still_thinking:
            budget_forced = True
            say("\n------- END THINKING (budget reached, forcing answer) ----\n", flush=True)
            say("------- RESPONSE --------", flush=True)
            showed_resp_header = True
        extra, final = _force_answer(
            cfg, messages, sampling, thinking, answer, max_output_tokens, timeout, quiet,
        )
        answer += extra
        if final:
            finish_reason = "length" if final.get("stop_type") == "limit" else "stop"
            p2 = final.get("timings", {})
            completion_tokens = (completion_tokens or 0) + (final.get("tokens_predicted") or 0)
            prompt_ms = (prompt_ms or 0) + (p2.get("prompt_ms") or 0)
            generation_ms = (generation_ms or 0) + (p2.get("predicted_ms") or 0)
            tokens_per_sec_llama = (
                round(completion_tokens / (generation_ms / 1000), 2) if generation_ms else None
            )

    t_end = time.time()
    if showed_resp_header:
        say("\n------- END RESPONSE ----")
    say()

    stop_event.set()

    # Client-side timing, for both engines (vLLM sends no timings): decode speed
    # from first token to end of stream, and the server-side fields as a fallback.
    decode_s = t_end - t_request_start - ttft if ttft is not None else None
    decode_tok_s = (
        round((completion_tokens - 1) / decode_s, 2)
        if decode_s and decode_s > 0 and completion_tokens and completion_tokens > 1 else None
    )
    if prompt_ms is None and ttft is not None:
        prompt_ms = round(ttft * 1000, 1)
    if generation_ms is None and decode_s is not None:
        generation_ms = round(decode_s * 1000, 1)

    prompt_tokens = usage.get("prompt_tokens")
    metrics = {
        "engine":               engine,
        "prompt_tokens":        prompt_tokens,
        "completion_tokens":    completion_tokens,
        "total_tokens":         (prompt_tokens or 0) + (completion_tokens or 0) if usage else None,
        "peak_vram_mb":         round(peak_mb[0], 1) if NVML_AVAILABLE and not quiet else None,
        "prompt_ms":            prompt_ms,
        "generation_ms":        generation_ms,
        "tokens_per_sec_llama": tokens_per_sec_llama,
        "decode_tok_s":         decode_tok_s,
        "ttft_s":               round(ttft, 2) if ttft is not None else None,
        "finish_reason":        finish_reason,
        "budget_forced":        budget_forced,
    }
    return thinking.strip(), answer.strip(), metrics


def _force_answer(
    cfg: dict,
    messages: list,
    sampling: dict,
    thinking: str,
    answer: str,
    max_output_tokens: int,
    timeout: int,
    quiet: bool = False,
) -> tuple[str, dict]:
    """Phase 2 of an enforced thinking budget on llama.cpp: continue from a raw prompt.

    Closes the thinking with a bare </think> (no injected instruction text) and
    continues the answer, empty if the model was still thinking, with whatever
    is left of max_output_tokens. Returns the extra answer text and the final
    /completion chunk (timings, tokens_predicted, stop_type).
    """
    port = cfg["port"]
    r = requests.post(f"http://localhost:{port}/apply-template", json={"messages": messages}, timeout=30)
    r.raise_for_status()
    prompt = r.json()["prompt"]
    if not prompt.rstrip().endswith(THINK_OPEN):
        prompt += THINK_OPEN + "\n"

    prompt += thinking + "\n" + THINK_CLOSE + "\n\n" + answer
    n_predict = max_output_tokens - (count_tokens(answer, port) or 0)
    if n_predict <= 0:
        return "", {}

    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        **sampling,
        "stream": True,
        "cache_prompt": True,  # reuse phase 1's KV cache for the shared prefix
    }
    text = ""
    final: dict = {}
    with requests.post(f"http://localhost:{port}/completion", json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = line.decode("utf-8")
            if not data.startswith("data: ") or data[6:] == "[DONE]":
                continue
            chunk = json.loads(data[6:])
            token = chunk.get("content") or ""
            if token:
                text += token
                if not quiet:
                    print(token, end="", flush=True)
            if chunk.get("stop"):
                final = chunk
    return text, final


def _avg(key: str, trials: list[dict]):
    vals = [r[key] for r in trials if r.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _pct(key: str, trials: list[dict], value=True):
    vals = [r[key] for r in trials if r.get(key) is not None]
    return round(100 * sum(1 for v in vals if v == value) / len(vals), 2) if vals else None


def question_pass_fields(passing_rounds: list, num_rounds: int) -> dict:
    """Per-question fields: pass_at_1 = fraction of rounds correct, pass_at_k = any round correct."""
    num_correct = len(passing_rounds)
    return {
        "pass_at_1":      num_correct / num_rounds if num_rounds else 0.0,
        "pass_at_k":      num_correct >= 1,
        "num_correct":    num_correct,
        "num_rounds":     num_rounds,
        "passing_rounds": passing_rounds,
    }


def accuracy_metrics(per_question: dict) -> dict:
    """Overall accuracy from entries built by question_pass_fields().

    overall_pass_at_1 is the standard pass@1: mean accuracy over all rounds.
    overall_pass_at_k is the fraction of questions solved in at least one of
    the k = num_rounds rounds (what overall_pass_at_1 used to mean).
    """
    entries = list(per_question.values())
    n = len(entries)
    solved = sum(1 for e in entries if e["pass_at_k"])
    return {
        "overall_pass_at_1": sum(e["pass_at_1"] for e in entries) / n if n else 0.0,
        "overall_pass_at_k": solved / n if n else 0.0,
        "num_rounds":        max((e["num_rounds"] for e in entries), default=0),
        "questions_passed":  solved,
        "total_questions":   n,
    }


def count_tokens(text: str, port: int, engine: str = "llamacpp", model: str | None = None) -> int | None:
    if not text:
        return 0
    try:
        if engine == "vllm":
            payload = {"model": model, "prompt": text, "add_special_tokens": False}
        else:
            payload = {"content": text}
        r = requests.post(f"http://localhost:{port}/tokenize", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["count"] if "count" in data else len(data["tokens"])
    except Exception:
        return None


def resolve_model_port(cfg: dict) -> None:
    """Fill cfg model/port from the MODEL/PORT env vars when the yaml doesn't set them."""
    if "model" not in cfg:
        cfg["model"] = os.environ.get("MODEL", "")
    if "port" not in cfg:
        cfg["port"] = int(os.environ.get("PORT", "0") or 0)

    if not cfg.get("model"):
        raise ValueError("model not set — add to model.yaml or set MODEL env var")
    if not cfg.get("port"):
        raise ValueError("port not set — add to model.yaml or set PORT env var")


def sampling_temperature(cfg: dict) -> float:
    return (
        cfg.get("thinking_temp", 0.6)
        if cfg.get("enable_thinking", False)
        else cfg.get("nonthinking_temp", 0.0)
    )


def make_config_block(cfg: dict, temperature: float, num_rounds: int, log_dir: str) -> dict:
    """The "config" block written to every round file and summary.json."""
    engine = detect_engine(cfg)
    return {
        "model":             cfg["model"],
        "port":              cfg["port"],
        "benchmark":         cfg.get("benchmark", "eval"),
        "engine":            engine,
        "concurrency":       cfg.get("concurrency", 1),
        "context_window":    cfg.get("context_window"),
        "enable_thinking":   cfg.get("enable_thinking", False),
        "temperature":       temperature,
        "thinking_budget":   cfg.get("thinking_budget"),
        "enforce_thinking_budget": cfg.get("enforce_thinking_budget", True),
        "max_output_tokens": cfg.get("max_output_tokens"),
        "top_p":             cfg.get("top_p"),
        "top_k":             cfg.get("top_k"),
        "min_p":             LLAMACPP_DEFAULT_MIN_P,  # llama.cpp default; sent explicitly to vLLM
        "presence_penalty":  cfg.get("presence_penalty"),
        "repetition_penalty": cfg.get("repetition_penalty"),  # not applied by either engine
        "num_rounds":        num_rounds,
        "log_dir":           log_dir,
        "pre_prompt":        cfg.get("pre_prompt", "").strip(),
        "post_prompt":       cfg.get("post_prompt", "").strip(),
        "prompt_key":        cfg.get("prompt_key", ""),
        "system_prompt": cfg.get("system_prompt", ""),
    }


def run_one(benchmark, row: dict, cfg: dict, temperature: float, quiet: bool = False) -> dict:
    """Query one problem once and return its graded result dict (one round entry)."""
    t0 = time.time()
    thinking, answer, metrics = query(benchmark.get_question_text(row), cfg, temperature, quiet=quiet)
    elapsed = time.time() - t0

    completion_tokens = metrics.get("completion_tokens") or 0
    tokens_per_sec = round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens else None
    metrics["tokens_per_sec"] = tokens_per_sec

    engine = metrics.get("engine", "llamacpp")
    thinking_tokens = count_tokens(thinking, cfg["port"], engine, cfg["model"])
    if engine == "vllm" and budget_enforced(cfg) and thinking_tokens is not None:
        # vLLM ends the thinking itself and doesn't report it; infer from the length
        metrics["budget_forced"] = thinking_tokens >= cfg.get("thinking_budget", 0) - 5

    result = benchmark.build_result(row, thinking, answer, metrics, elapsed)
    result["elapsed_s"]            = round(elapsed, 2)
    result["tokens_per_sec"]       = tokens_per_sec
    result["prompt_tokens"]        = metrics.get("prompt_tokens")
    result["completion_tokens"]    = metrics.get("completion_tokens")
    result["total_tokens"]         = metrics.get("total_tokens")
    result["peak_vram_mb"]         = metrics.get("peak_vram_mb")
    result["thinking_tokens"]      = thinking_tokens
    result["prompt_ms"]            = metrics.get("prompt_ms")
    result["generation_ms"]        = metrics.get("generation_ms")
    result["tokens_per_sec_llama"] = metrics.get("tokens_per_sec_llama")
    result["decode_tok_s"]         = metrics.get("decode_tok_s")
    result["finish_reason"]        = metrics.get("finish_reason")
    result["budget_forced"]        = metrics.get("budget_forced")

    if not quiet:
        correct = result.get("correct", result.get("passed_all_tests", False))
        print(f"\n({elapsed:.1f}s | {tokens_per_sec} tok/s | {'PASS' if correct else 'FAIL'} | peak VRAM {metrics.get('peak_vram_mb')} MB)\n")
    return result


def is_correct(result: dict) -> bool:
    return bool(result.get("correct", result.get("passed_all_tests", False)))


def build_summary(benchmark, problems: list, all_results: dict, config_block: dict, total_elapsed_s: float) -> dict:
    """summary.json contents. all_results maps question idx (1-based) -> {round: result}."""
    per_question = {}
    for idx in range(1, len(problems) + 1):
        trials = all_results[idx]
        passing_rounds = sorted(r for r, res in trials.items() if is_correct(res))
        entry = benchmark.build_summary_entry(problems[idx - 1], passing_rounds)
        entry.update(question_pass_fields(passing_rounds, len(trials)))
        per_question[f"question_{idx}"] = entry

    acc = accuracy_metrics(per_question)

    all_trials = [res for trials in all_results.values() for res in trials.values()]

    return {
        "config":                    config_block,
        **acc,
        "pct_budget_forced":         _pct("budget_forced", all_trials),
        "pct_truncated":             _pct("finish_reason", all_trials, "length"),
        "total_elapsed_s":           round(total_elapsed_s, 1),
        "total_elapsed_h":           round(total_elapsed_s / 3600, 4),
        "avg_elapsed_s":             _avg("elapsed_s", all_trials),
        "avg_prompt_tokens":         _avg("prompt_tokens", all_trials),
        "avg_completion_tokens":     _avg("completion_tokens", all_trials),
        "avg_total_tokens":          _avg("total_tokens", all_trials),
        "avg_tokens_per_sec":        _avg("tokens_per_sec", all_trials),
        "avg_tokens_per_sec_llama":  _avg("tokens_per_sec_llama", all_trials),
        "avg_decode_tok_s":          _avg("decode_tok_s", all_trials),
        "avg_thinking_tokens":       _avg("thinking_tokens", all_trials),
        "avg_prompt_ms":             _avg("prompt_ms", all_trials),
        "avg_generation_ms":         _avg("generation_ms", all_trials),
        "avg_peak_vram_mb":          _avg("peak_vram_mb", all_trials),
        "per_question":              per_question,
    }


def run_eval(benchmark, cfg: dict) -> None:
    resolve_model_port(cfg)

    benchmark_name = cfg.get("benchmark", "eval")
    model_name = cfg["model"]
    model_short = model_name.split("/")[-1].lower()

    log_dir = cfg.get("log_dir") or f"logs/{benchmark_name}/{model_short}"
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    num_rounds = cfg.get("num_rounds", 1)
    temperature = sampling_temperature(cfg)

    print(f"Benchmark: {benchmark_name}", flush=True)
    print(f"Model: {cfg['model']}", flush=True)
    print(f"Thinking: {cfg.get('enable_thinking')} | Temperature: {temperature}", flush=True)
    print(f"Loading dataset...", flush=True)

    problems = benchmark.load_problems(cfg)

    limit = cfg.get("limit")
    if limit is not None:
        problems = problems[:limit]

    print(f"Loaded {len(problems)} problems.\n", flush=True)

    config_block = make_config_block(cfg, temperature, num_rounds, log_dir)

    all_results: dict[int, dict] = {idx: {} for idx in range(1, len(problems) + 1)}
    total_elapsed_s = 0.0

    for rnd in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{num_rounds}")
        print(f"{'='*60}\n", flush=True)

        round_results = {}
        rnd_t0 = time.time()

        for idx, row in enumerate(problems, start=1):
            label = benchmark.get_label(row)
            print(f"[Round {rnd} | {idx:02d}/{len(problems)}] {label}\n")

            result = run_one(benchmark, row, cfg, temperature)

            round_results[f"question_{idx}"] = result
            all_results[idx][rnd] = result

            round_file = log_path / f"round-{rnd}_results.json"
            round_file.write_text(json.dumps({"config": config_block, "results": round_results}, indent=2, ensure_ascii=False))

        rnd_elapsed_s = time.time() - rnd_t0
        total_elapsed_s += rnd_elapsed_s
        rnd_elapsed_h = round(rnd_elapsed_s / 3600, 4)
        # patch round_elapsed_h into the round file
        round_data = json.loads(round_file.read_text())
        round_data["round_elapsed_h"] = rnd_elapsed_h
        round_file.write_text(json.dumps(round_data, indent=2, ensure_ascii=False))

        print(f"Round {rnd} complete ({rnd_elapsed_h:.3f}h) -> {round_file}")

    summary = build_summary(benchmark, problems, all_results, config_block, total_elapsed_s)

    summary_file = log_path / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nDone!")
    print(f"Pass@1 (mean over {num_rounds} rounds): {summary['overall_pass_at_1']:.1%}")
    print(f"Pass@{num_rounds} (any round): {summary['questions_passed']}/{len(problems)} = {summary['overall_pass_at_k']:.1%}")
    print(f"Total time: {total_elapsed_s/3600:.3f}h")
    print(f"Summary saved -> {summary_file}")
