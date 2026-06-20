#!/usr/bin/env python3
"""
Quick local test: load a benchmark dataset and optionally run N questions
against a local inference server.

Usage (from repo root):
    # Just test dataset loading:
    python scripts/test_benchmark_load.py --benchmark lcb_pro

    # Load + run 2 questions against a local server:
    python scripts/test_benchmark_load.py --benchmark aa_omniscience --limit 2 --port 20003
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml
from benchmarks import REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=None, help="Local server port — skip inference if omitted")
    parser.add_argument("--model", default="", help="Model name for inference")
    args = parser.parse_args()

    bench_yaml = os.path.join(
        os.path.dirname(__file__), "..", "src", "configs", "benchmarks", f"{args.benchmark}.yaml"
    )
    if not os.path.exists(bench_yaml):
        print(f"ERROR: {bench_yaml} not found")
        sys.exit(1)

    with open(bench_yaml) as f:
        cfg = yaml.safe_load(f)

    if args.benchmark not in REGISTRY:
        print(f"ERROR: '{args.benchmark}' not in registry. Available: {list(REGISTRY)}")
        sys.exit(1)

    benchmark = REGISTRY[args.benchmark]()

    print(f"Loading {args.benchmark} dataset...")
    try:
        problems = benchmark.load_problems(cfg)
    except Exception as e:
        print(f"\nFAILED to load dataset:\n  {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"OK — loaded {len(problems)} problems")
    if problems:
        print(f"First row keys: {list(problems[0].keys())}")
        print(f"Sample label:   {benchmark.get_label(problems[0])}")
        q = benchmark.get_question_text(problems[0])
        print(f"Sample question (first 300 chars):\n{q[:300]}")
    elif len(problems) == 0:
        print("WARNING: 0 problems loaded — likely a filter (difficulty/domain/subset) matched nothing.")
        print("Tip: temporarily comment out the filter in the benchmark's load_problems() to see raw rows.")

    if args.port is None or args.limit is None:
        print("\n(Pass --port and --limit to also run inference)")
        return

    # Run inference
    from eval_base import run_eval
    cfg["port"] = args.port
    cfg["model"] = args.model or os.environ.get("MODEL", "local-model")
    cfg["limit"] = args.limit
    cfg["num_rounds"] = 1
    run_eval(benchmark, cfg)


if __name__ == "__main__":
    main()
