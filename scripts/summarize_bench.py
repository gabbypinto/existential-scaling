#!/usr/bin/env python3
"""
Summarize scripts/bench_concurrency.sh results: one table per model, engines side by side.

Reads logs/bench_concurrency/<engine>/<name>/{meta.json,conc_<N>.json} (written by
`vllm bench serve --save-result`) and pairs runs of the same model by their tokenizer
(the HF model id), so a vLLM run and a llama.cpp GGUF run of one model share a table.

Columns per engine:
  out tok/s  total output tokens per second across all concurrent requests
  per-req    decode speed each request sees (1000 / median TPOT ms)
  TTFT ms    median time to first token

Usage:
  python3 scripts/summarize_bench.py
  python3 scripts/summarize_bench.py --dir logs/bench_concurrency --csv summary.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_DIR = ROOT / "logs" / "bench_concurrency"


def load_runs(base: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(base.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        for f in meta_path.parent.glob("conc_*.json"):
            r = json.loads(f.read_text())
            conc = r.get("max_concurrency") or int(re.search(r"conc_(\d+)", f.name).group(1))
            tpot = r.get("median_tpot_ms")
            rows.append({
                "model":            meta.get("tokenizer") or meta.get("name"),
                "engine":           meta.get("engine"),
                "name":             meta.get("name"),
                "concurrency":      int(conc),
                "output_tok_s":     r.get("output_throughput"),
                "per_request_tok_s": round(1000 / tpot, 1) if tpot else None,
                "median_ttft_ms":   r.get("median_ttft_ms"),
                "median_tpot_ms":   tpot,
                "completed":        r.get("completed"),
                "total_output_tokens": r.get("total_output_tokens"),
            })
    return rows


def fmt(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f}"


def print_tables(rows: list[dict]) -> None:
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        # one column group per run, so e.g. an F16 and a Q8 llama.cpp run sit side by side
        runs = sorted({(r["engine"], r["name"]) for r in mrows})
        per_engine = {e: sum(1 for re_, _ in runs if re_ == e) for e, _ in runs}
        # e.g. "llamacpp-f16" / "llamacpp-q8_0": engine plus the last part of the run name
        label = {run: run[0] if per_engine[run[0]] == 1 else f"{run[0]}-{run[1].rsplit('-', 1)[-1]}" for run in runs}
        by = {(r["engine"], r["name"], r["concurrency"]): r for r in mrows}
        levels = sorted({r["concurrency"] for r in mrows})
        # throughput ratio vs each llama.cpp run, when there is exactly one vLLM run
        vllm_runs = [run for run in runs if run[0] == "vllm"]
        ratio_runs = [run for run in runs if run[0] == "llamacpp"] if len(vllm_runs) == 1 else []

        print(f"\n== {model} ==")
        for run in runs:
            print(f"  {label[run]}: {run[0]} / {run[1]}")
        head = f"{'conc':>5}"
        for run in runs:
            head += f" | {label[run] + ' tok/s':>22} {'per-req':>8} {'TTFT ms':>8}"
        for run in ratio_runs:
            head += f" | {'vllm/' + label[run]:>20}"
        print(head)
        print("-" * len(head))
        for c in levels:
            line = f"{c:>5}"
            for run in runs:
                r = by.get((*run, c), {})
                line += f" | {fmt(r.get('output_tok_s')):>22} {fmt(r.get('per_request_tok_s')):>8} {fmt(r.get('median_ttft_ms'), 0):>8}"
            for run in ratio_runs:
                v = by.get((*vllm_runs[0], c), {}).get("output_tok_s")
                l = by.get((*run, c), {}).get("output_tok_s")
                line += f" | {(f'{v / l:.2f}x' if v and l else '-'):>20}"
            print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--csv", type=Path, default=None, help="CSV output (default: <dir>/summary.csv)")
    args = p.parse_args()

    rows = load_runs(args.dir)
    if not rows:
        raise SystemExit(f"no results under {args.dir} (run scripts/bench_concurrency.sh first)")

    print_tables(rows)

    out = args.csv or args.dir / "summary.csv"
    rows.sort(key=lambda r: (r["model"], r["engine"], r["concurrency"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
