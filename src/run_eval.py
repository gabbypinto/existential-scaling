import argparse
import sys
from eval_base import load_config, run_eval
from benchmarks import REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model config YAML")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems (smoke test)")
    args = parser.parse_args()

    model_cfg = load_config(args.model)
    bench_cfg = load_config(args.benchmark)

    cfg = {**model_cfg, **bench_cfg}

    benchmark_name = cfg.get("benchmark")
    if not benchmark_name:
        print(f"ERROR: benchmark config must include a 'benchmark:' field.")
        print(f"Available: {list(REGISTRY.keys())}")
        sys.exit(1)

    if benchmark_name not in REGISTRY:
        print(f"ERROR: unknown benchmark '{benchmark_name}'")
        print(f"Available: {list(REGISTRY.keys())}")
        sys.exit(1)

    if args.limit is not None:
        cfg["limit"] = args.limit

    benchmark = REGISTRY[benchmark_name]()
    run_eval(benchmark, cfg)


if __name__ == "__main__":
    main()
