#!/usr/bin/env python3
"""
Extract accuracy, latency, and token metrics from logs_new summary.json files.
Output structure: model -> benchmark -> system_prompt -> metrics
"""

import json
import sys
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs_new"
OUTPUT_FILE = Path(__file__).parent / "metrics_summary.json"


def main():
    if not LOGS_DIR.exists():
        print(f"logs_new directory not found at {LOGS_DIR}", file=sys.stderr)
        sys.exit(1)

    results = {}

    for benchmark_dir in sorted(LOGS_DIR.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name

        for model_dir in sorted(benchmark_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            for prompt_dir in sorted(model_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                system_prompt = prompt_dir.name

                summary_path = prompt_dir / "summary.json"
                if not summary_path.exists():
                    print(f"  [skip] missing summary.json: {summary_path}", file=sys.stderr)
                    continue

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

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote metrics for {sum(len(v) for v in results.values())} model-benchmark combos to {OUTPUT_FILE}")

    # Print a quick summary table
    print()
    print(f"{'Model':<45} {'Benchmark':<20} {'Prompt':<20} {'Accuracy':>10} {'Tok/s':>8} {'OutTok':>8}")
    print("-" * 115)
    for model, benchmarks in sorted(results.items()):
        for benchmark, prompts in sorted(benchmarks.items()):
            for prompt, m in sorted(prompts.items()):
                acc = m["accuracy"]
                tps = m["avg_tokens_per_sec"]
                otok = m["avg_completion_tokens"]
                acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                tps_str = f"{tps:.1f}" if tps is not None else "N/A"
                otok_str = f"{otok:.0f}" if otok is not None else "N/A"
                print(f"{model:<45} {benchmark:<20} {prompt:<20} {acc_str:>10} {tps_str:>8} {otok_str:>8}")


if __name__ == "__main__":
    main()
