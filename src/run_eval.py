import argparse
import sys
import yaml
from eval_base import load_config, run_eval
from benchmarks import REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model config YAML")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems (smoke test)")
    parser.add_argument("--prompt-key", default="", help="Key to look up in --prompts-file")
    parser.add_argument("--prompts-file", default="src/configs/prompts.yaml", help="YAML file of system prompt variants")
    parser.add_argument("--log-dir", default=None, help="Override the auto-computed log directory")
    parser.add_argument("--num-rounds", type=int, default=None, help="Rounds per problem for pass@k (overrides benchmark YAML)")
    args = parser.parse_args()

    model_cfg = load_config(args.model)
    bench_cfg = load_config(args.benchmark)

    cfg = {**model_cfg, **bench_cfg}

    if args.prompt_key:
        try:
            with open(args.prompts_file) as f:
                prompts = yaml.safe_load(f)
            cfg["system_prompt"] = prompts.get(args.prompt_key, "")
            cfg["prompt_key"] = args.prompt_key
        except FileNotFoundError:
            print(f"WARNING: prompts file not found: {args.prompts_file}")

    if args.log_dir:
        cfg["log_dir"] = args.log_dir

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

    if args.num_rounds is not None:
        cfg["num_rounds"] = args.num_rounds

    benchmark = REGISTRY[benchmark_name]()
    run_eval(benchmark, cfg)


if __name__ == "__main__":
    main()
