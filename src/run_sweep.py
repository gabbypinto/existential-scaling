"""
Run benchmarks x system prompts for one model through a single shared request pool.

Every (benchmark, prompt, round, question) is one task; `--concurrency` tasks are in
flight at once, so a batching server (vLLM) stays full across benchmark and prompt
boundaries. Results use the same layout as run_eval.py:

    logs/{benchmark}/{model}/{prompt}/round-N_results.json + summary.json

so extract_metrics.py, regrade.py and prompt_stats.py work unchanged. A (benchmark,
prompt) pair that already has a summary.json is skipped unless --overwrite, so an
interrupted sweep resumes by re-running the same command.

Usage (from src/, normally via scripts/run_vllm_sweep.sh):
    python run_sweep.py --model configs/model.yaml --benchmarks aime24,gpqa --concurrency 32
    python run_sweep.py --model configs/model.yaml --benchmarks aime25 --prompts Baseline,Collapse --limit 2
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from benchmarks import REGISTRY
from eval_base import (
    build_summary, detect_engine, is_correct, load_config, log, make_config_block,
    resolve_model_port, run_one, sampling_temperature,
)

CONFIG_DIR = Path(__file__).parent / "configs"
ROUND_WRITE_INTERVAL_S = 15  # how often in-progress round files are rewritten


class Unit:
    """One (benchmark, prompt) run: collects results and writes its log files."""

    def __init__(self, benchmark, problems, cfg, temperature, num_rounds, log_path):
        self.benchmark = benchmark
        self.problems = problems
        self.cfg = cfg
        self.temperature = temperature
        self.num_rounds = num_rounds
        self.log_path = log_path
        self.name = f"{cfg['benchmark']}/{cfg['prompt_key']}"
        self.config_block = make_config_block(cfg, temperature, num_rounds, str(log_path))
        self.results = {rnd: {} for rnd in range(1, num_rounds + 1)}   # round -> idx -> result
        self.round_span = {}                                            # round -> [start, end]
        self.pending = num_rounds * len(problems)
        self.failed = False
        self.t_start = None
        self.last_write = {}
        self.lock = threading.Lock()

    def started(self, rnd: int) -> None:
        now = time.time()
        with self.lock:
            self.t_start = self.t_start or now
            span = self.round_span.setdefault(rnd, [now, now])
            span[0] = min(span[0], now)

    def record(self, rnd: int, idx: int, result: dict | None) -> bool:
        """Store one result (None = failed). Returns True when this was the unit's last task."""
        now = time.time()
        with self.lock:
            if result is None:
                self.failed = True
            else:
                self.results[rnd][idx] = result
                self.round_span[rnd][1] = now
                if now - self.last_write.get(rnd, 0) >= ROUND_WRITE_INTERVAL_S:
                    self._write_round(rnd)
            self.pending -= 1
            return self.pending == 0

    def _write_round(self, rnd: int) -> None:
        results = {f"question_{i}": self.results[rnd][i] for i in sorted(self.results[rnd])}
        data = {"config": self.config_block, "results": results}
        if len(results) == len(self.problems):
            start, end = self.round_span[rnd]
            data["round_elapsed_h"] = round((end - start) / 3600, 4)
        (self.log_path / f"round-{rnd}_results.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False))
        self.last_write[rnd] = time.time()

    def finish(self) -> dict | None:
        """Write final round files and, if every task ran and succeeded, summary.json.

        No summary for a failed or interrupted unit, so the next sweep re-runs it.
        """
        with self.lock:
            for rnd in self.results:
                if self.results[rnd]:
                    self._write_round(rnd)
            if self.failed or self.pending > 0:
                return None
            all_results = {
                idx: {rnd: self.results[rnd][idx] for rnd in self.results}
                for idx in range(1, len(self.problems) + 1)
            }
            elapsed = time.time() - (self.t_start or time.time())
            summary = build_summary(self.benchmark, self.problems, all_results, self.config_block, elapsed)
            (self.log_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            return summary


def bench_config_path(name: str) -> Path:
    p = Path(name)
    return p if p.suffix in (".yaml", ".yml") else CONFIG_DIR / "benchmarks" / f"{name}.yaml"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model config YAML")
    p.add_argument("--benchmarks", required=True, help="comma-separated benchmark names (configs/benchmarks/<name>.yaml)")
    p.add_argument("--prompts", default="", help="comma-separated prompt keys (default: every key in --prompts-file)")
    p.add_argument("--prompts-file", default=str(CONFIG_DIR / "prompts.yaml"))
    p.add_argument("--concurrency", type=int, default=8, help="requests in flight at once")
    p.add_argument("--limit", type=int, default=None, help="only the first N problems per benchmark")
    p.add_argument("--num-rounds", type=int, default=None, help="override each benchmark's num_rounds")
    p.add_argument("--log-root", default="logs")
    p.add_argument("--overwrite", action="store_true", help="re-run units that already have a summary.json")
    args = p.parse_args()

    model_cfg = load_config(args.model)
    resolve_model_port(model_cfg)
    model_short = model_cfg["model"].split("/")[-1].lower()
    engine = detect_engine(model_cfg)

    prompts = yaml.safe_load(Path(args.prompts_file).read_text())
    keys = [k.strip() for k in args.prompts.split(",") if k.strip()] or list(prompts)
    missing = [k for k in keys if k not in prompts]
    if missing:
        raise SystemExit(f"prompt keys not in {args.prompts_file}: {missing}")

    # ---------- build units: one per (benchmark, prompt) ----------
    units: list[Unit] = []
    skipped = []
    for bench_name in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        bench_cfg = load_config(bench_config_path(bench_name))
        name = bench_cfg.get("benchmark")
        if name not in REGISTRY:
            raise SystemExit(f"unknown benchmark '{name}' (from {bench_name}); available: {list(REGISTRY)}")
        base_cfg = {**model_cfg, **bench_cfg, "concurrency": args.concurrency}
        if args.num_rounds is not None:
            base_cfg["num_rounds"] = args.num_rounds

        todo = [k for k in keys
                if args.overwrite or not (Path(args.log_root) / name / model_short / k / "summary.json").exists()]
        skipped += [f"{name}/{k}" for k in keys if k not in todo]
        if not todo:
            continue

        benchmark = REGISTRY[name]()
        log(f"Loading {name}...")
        problems = benchmark.load_problems(base_cfg)
        if args.limit is not None:
            problems = problems[:args.limit]

        for key in todo:
            cfg = {**base_cfg, "system_prompt": prompts[key] or "", "prompt_key": key}
            log_path = Path(args.log_root) / name / model_short / key
            cfg["log_dir"] = str(log_path)
            log_path.mkdir(parents=True, exist_ok=True)
            units.append(Unit(benchmark, problems, cfg, sampling_temperature(cfg),
                              cfg.get("num_rounds", 1), log_path))

    tasks = [(u, rnd, idx, row)
             for u in units
             for rnd in range(1, u.num_rounds + 1)
             for idx, row in enumerate(u.problems, start=1)]

    log(f"\nModel: {model_cfg['model']} ({engine})  port {model_cfg['port']}")
    log(f"Units: {len(units)} to run, {len(skipped)} already done{(' (' + ', '.join(skipped) + ')') if skipped else ''}")
    log(f"Tasks: {len(tasks)}  concurrency {args.concurrency}\n")
    if not tasks:
        return

    done = [0]
    done_lock = threading.Lock()
    t0 = time.time()

    def run_task(unit: Unit, rnd: int, idx: int, row: dict) -> None:
        unit.started(rnd)
        result = None
        for attempt in range(3):
            try:
                result = run_one(unit.benchmark, row, unit.cfg, unit.temperature, quiet=True)
                break
            except Exception as e:
                log(f"  [{unit.name} r{rnd} q{idx}] attempt {attempt + 1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        with done_lock:
            done[0] += 1
            n = done[0]
        if result is None:
            log(f"[{unit.name} r{rnd} q{idx}] ERROR — gave up; unit will not get a summary ({n}/{len(tasks)})")
        else:
            log(f"[{unit.name} r{rnd} q{idx}] {'PASS' if is_correct(result) else 'FAIL'} "
                f"{result.get('completion_tokens') or 0:,} tok {result['elapsed_s']:.0f}s ({n}/{len(tasks)})")
        if unit.record(rnd, idx, result):
            summary = unit.finish()
            if summary is None:
                log(f"== {unit.name}: INCOMPLETE (failed requests) — re-run to retry")
            else:
                log(f"== {unit.name}: pass@1 {summary['overall_pass_at_1']:.1%}, "
                    f"pass@{unit.num_rounds} {summary['overall_pass_at_k']:.1%} -> {unit.log_path}/summary.json")

    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        futures = [pool.submit(run_task, *t) for t in tasks]
        for f in futures:
            f.result()
    except KeyboardInterrupt:
        log("\nInterrupted — cancelling queued requests (in-flight ones finish first)...")
        pool.shutdown(wait=True, cancel_futures=True)
        for u in units:
            if u.pending:
                u.finish()  # save partial round files; no summary for unfinished units
        raise SystemExit(1)
    pool.shutdown(wait=True)

    finished = sum(1 for u in units if not u.failed)
    log(f"\nDone: {finished}/{len(units)} units in {(time.time() - t0) / 3600:.2f}h")


if __name__ == "__main__":
    main()
