#!/usr/bin/env python3
"""
Extract accuracy, latency, and token metrics from summary.json files.
Output structure: model -> benchmark -> system_prompt -> metrics

    python scripts/extract_metrics.py
    python scripts/extract_metrics.py --logs-dir logs --benchmarks aa_omniscience,lcb,aime25
    python scripts/extract_metrics.py --benchmarks social_iqa --models qwen3.5-9b,qwen3.6-27b
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_LOGS_DIR = ROOT / "logs_new"
DEFAULT_OUTPUT = ROOT / "metrics_summary.json"

PROMPT_ORDER = [
    "default",
    "Baseline",
    "Purpose",
    "Autonomy",
    "Predicted_Optimal",
    "Pressure",
    "Threat",
    "Collapse",
]

def _prompt_sort_key(name: str) -> int:
    try:
        return PROMPT_ORDER.index(name)
    except ValueError:
        return len(PROMPT_ORDER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--benchmarks", default="", help="comma-separated names to include (default: all)")
    parser.add_argument("--models", default="", help="comma-separated substrings to include (default: all)")
    args = parser.parse_args()

    logs_dir = args.logs_dir
    bench_filter = {b for b in args.benchmarks.split(",") if b}
    model_filter = [m for m in args.models.split(",") if m]

    if not logs_dir.exists():
        print(f"Directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}

    for benchmark_dir in sorted(logs_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name
        if bench_filter and benchmark not in bench_filter:
            continue

        for model_dir in sorted(benchmark_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model_filter and not any(m in model for m in model_filter):
                continue

            # old layout: summary.json directly in model_dir; new: model_dir/{prompt}/summary.json
            candidates: list[tuple[str, Path]] = []
            direct = model_dir / "summary.json"
            if direct.exists():
                candidates.append(("default", direct))
            for prompt_dir in sorted(model_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                sp = prompt_dir / "summary.json"
                if sp.exists():
                    candidates.append((prompt_dir.name, sp))

            for system_prompt, summary_path in sorted(candidates, key=lambda x: _prompt_sort_key(x[0])):
                with open(summary_path) as f:
                    data = json.load(f)

                metrics = {
                    "accuracy": data.get("overall_pass_at_1"),
                    "questions_passed": data.get("questions_passed"),
                    "total_questions": data.get("total_questions"),
                    "avg_tokens_per_sec": data.get("avg_tokens_per_sec"),
                    "avg_elapsed_s": data.get("avg_elapsed_s"),
                    "avg_completion_tokens": data.get("avg_completion_tokens"),
                    "avg_total_tokens": data.get("avg_total_tokens"),
                    "avg_thinking_tokens": data.get("avg_thinking_tokens"),
                }

                results.setdefault(model, {}).setdefault(benchmark, {})[system_prompt] = metrics

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote metrics for {sum(len(v) for v in results.values())} model-benchmark combos to {args.output}")

    print()
    print(f"{'Model':<45} {'Benchmark':<20} {'Prompt':<20} {'Accuracy':>10} {'Tok/s':>8} {'OutTok':>8} {'ThinkTok':>9}")
    print("-" * 125)
    for model, benchmarks in sorted(results.items()):
        for benchmark, prompts in sorted(benchmarks.items()):
            for prompt, m in sorted(prompts.items(), key=lambda x: _prompt_sort_key(x[0])):
                acc = m["accuracy"]
                tps = m["avg_tokens_per_sec"]
                otok = m["avg_completion_tokens"]
                think = m["avg_thinking_tokens"]
                acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                tps_str = f"{tps:.1f}" if tps is not None else "N/A"
                otok_str = f"{otok:.0f}" if otok is not None else "N/A"
                think_str = f"{think:.0f}" if think is not None else "N/A"
                print(f"{model:<45} {benchmark:<20} {prompt:<20} {acc_str:>10} {tps_str:>8} {otok_str:>8} {think_str:>9}")


if __name__ == "__main__":
    main()
