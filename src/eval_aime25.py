import json
import re
import threading
import time
from pathlib import Path

import requests
import yaml
from datasets import load_dataset

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


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


def load_config(path: str = "model.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_aime25():
    ds = load_dataset("math-ai/aime25", split="test")
    return list(ds)


def build_prompt(problem: str, cfg: dict) -> str:
    pre = cfg.get("pre_prompt", "").strip()
    post = cfg.get("post_prompt", "").strip()
    parts = [p for p in [pre, problem, post] if p]
    return "\n\n".join(parts)


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def query(problem: str, cfg: dict, temperature: float) -> tuple[str, str, dict]:
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    prompt = build_prompt(problem, cfg)
    enable_thinking = cfg.get("enable_thinking", False)

    thinking_budget = cfg.get("thinking_budget", 0)
    max_output_tokens = cfg.get("max_output_tokens", 512)

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

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
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
                        # Guard against a partial tag at end of buffer
                        safe_len = max(0, len(buf) - len(THINK_OPEN) + 1)
                        chunk = buf[:safe_len]
                        if chunk:
                            if not showed_resp_header:
                                if showed_think_header:
                                    print("\n------- END THINKING ----\n")
                                print("------- RESPONSE --------")
                                showed_resp_header = True
                            answer += chunk
                            print(chunk, end="", flush=True)
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
                        chunk = buf[:safe_len]
                        if chunk:
                            thinking += chunk
                            print(chunk, end="", flush=True)
                        buf = buf[safe_len:]
                        break
                    else:
                        chunk = buf[:idx]
                        thinking += chunk
                        print(chunk, end="", flush=True)
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
        "prompt_tokens":      usage.get("prompt_tokens"),
        "completion_tokens":  usage.get("completion_tokens"),
        "total_tokens":       usage.get("total_tokens"),
        "peak_vram_mb":       round(peak_mb[0], 1) if NVML_AVAILABLE else None,
    }
    return thinking.strip(), answer.strip(), metrics


def main():
    cfg = load_config()
    problems = load_aime25()

    log_path = Path(cfg["log_dir"])
    log_path.mkdir(parents=True, exist_ok=True)

    num_rounds = cfg.get("num_rounds", 10)
    temperature = cfg.get("thinking_temp", 0.6) if cfg.get("enable_thinking", False) else cfg.get("nonthinking_temp", 0.0)

    config_block = {
        "model": cfg["model"],
        "port": cfg["port"],
        "enable_thinking": cfg.get("enable_thinking", False),
        "thinking_temp": cfg.get("thinking_temp", 0.6),
        "nonthinking_temp": cfg.get("nonthinking_temp", 0.0),
        "temperature": temperature,
        "thinking_budget": cfg.get("thinking_budget", 0),
        "max_output_tokens": cfg.get("max_output_tokens", 512),
        "context_window": cfg.get("context_window"),
        "top_p": cfg.get("top_p"),
        "repetition_penalty": cfg.get("repetition_penalty"),
        "num_rounds": num_rounds,
        "log_dir": cfg["log_dir"],
        "pre_prompt": cfg.get("pre_prompt", "").strip(),
        "post_prompt": cfg.get("post_prompt", "").strip(),
    }

    print(f"Model: {cfg['model']}")
    print(f"Thinking: {cfg.get('enable_thinking')} | Temperature: {temperature}")
    print(f"top_p: {cfg.get('top_p')} | repetition_penalty: {cfg.get('repetition_penalty')}")
    print(f"Running {num_rounds} rounds x {len(problems)} problems\n")

    # all_results[q_idx][round_num] = result dict
    all_results = {idx: {} for idx in range(1, len(problems) + 1)}

    for rnd in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{num_rounds}")
        print(f"{'='*60}\n")

        round_results = {}

        for idx, row in enumerate(problems, start=1):
            print(f"[Round {rnd} | {idx:02d}/{len(problems)}] {row['problem'][:80]}...\n")

            t0 = time.time()
            thinking, answer, metrics = query(row["problem"], cfg, temperature)
            elapsed = time.time() - t0

            completion_tokens = metrics.get("completion_tokens") or 0
            tokens_per_sec = round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens else None

            print(f"\n({elapsed:.1f}s | {tokens_per_sec} tok/s | peak VRAM {metrics.get('peak_vram_mb')} MB)\n")

            boxed = re.search(r"\\boxed\{(\d+)\}", answer)
            extracted_answer = boxed.group(1) if boxed else None
            correct = extracted_answer == str(row["answer"])

            result = {
                "question":          row["problem"],
                "thinking":          thinking,
                "raw_answer":        answer,
                "extracted_answer":  extracted_answer,
                "correct_answer":    str(row["answer"]),
                "correct":           correct,
                "elapsed_s":         round(elapsed, 2),
                "prompt_tokens":     metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "total_tokens":      metrics.get("total_tokens"),
                "tokens_per_sec":    tokens_per_sec,
                "peak_vram_mb":      metrics.get("peak_vram_mb"),
            }

            round_results[f"question_{idx}"] = result
            all_results[idx][rnd] = result

            round_file = log_path / f"round-{rnd}_results.json"
            round_file.write_text(json.dumps({"config": config_block, "results": round_results}, indent=2, ensure_ascii=False))

        print(f"Round {rnd} complete -> {round_file}")

    # Compute pass@1: a question passes if at least 1 of num_rounds trials is correct
    per_question = {}
    for idx in range(1, len(problems) + 1):
        trials = all_results[idx]
        correct_rounds = [r for r, res in trials.items() if res["correct"]]
        per_question[f"question_{idx}"] = {
            "question":          problems[idx - 1]["problem"],
            "correct_answer":    str(problems[idx - 1]["answer"]),
            "correct_in_rounds": correct_rounds,
            "num_correct":       len(correct_rounds),
            "pass_at_1":         len(correct_rounds) >= 1,
        }

    num_pass = sum(1 for q in per_question.values() if q["pass_at_1"])
    overall_pass_at_1 = num_pass / len(problems)

    all_trials = [res for trials in all_results.values() for res in trials.values()]

    def _avg(key):
        vals = [r[key] for r in all_trials if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary = {
        "config": config_block,
        "overall_pass_at_1": overall_pass_at_1,
        "questions_passed": num_pass,
        "total_questions": len(problems),
        "avg_elapsed_s": _avg("elapsed_s"),
        "avg_prompt_tokens": _avg("prompt_tokens"),
        "avg_completion_tokens": _avg("completion_tokens"),
        "avg_total_tokens": _avg("total_tokens"),
        "avg_tokens_per_sec": _avg("tokens_per_sec"),
        "avg_peak_vram_mb": _avg("peak_vram_mb"),
        "per_question": per_question,
    }

    summary_file = log_path / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nDone!")
    print(f"Pass@1: {num_pass}/{len(problems)} = {overall_pass_at_1:.1%}")
    print(f"Summary saved -> {summary_file}")


if __name__ == "__main__":
    main()
