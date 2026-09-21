#!/usr/bin/env python3
"""
Statistical analysis of system-prompt effects (Phase 1 method, Emma 07/17).

Two modes:

  aggregate  - one number per (model, benchmark, prompt) "cell", e.g. accuracy
               and average output tokens. Blocks are model x benchmark cells.
               Input: metrics_summary.json from scripts/extract_metrics.py,
               or a results-sheet CSV export.

  bullshit   - per-question BullshitBench judge scores (0/1/2). Blocks are
               questions, run separately per model. Input: logs/bullshit_bench.

Tests (aggregate mode):
  * Friedman test across prompts, blocked on model x benchmark, for output
    tokens and for accuracy (only cells that have every prompt are used).
  * Reference prompt (default Collapse) below its cell average: count + two-sided
    sign test.
  * Reference vs each other prompt: two-sided sign test and Wilcoxon signed-rank
    on output tokens, raw and Holm-corrected across the comparisons.
  * Mean % token difference, reference vs each prompt.
  * Largest single-cell token gap (max / min prompt).
  * Pearson r between output tokens and accuracy across prompts, per cell.
  * Optional pgfplots .dat files (tok_<bench>.dat, acc_<bench>.dat, tok_norm.dat)
    in the format used by the Phase 1 figures.tex.

Tests (bullshit mode), per model:
  * mean score, clear-pushback rate (score 2), no-answer count per prompt
  * Friedman across prompts on per-question scores (questions with a missing
    score in any prompt are dropped, and the count is reported)
  * each prompt vs the reference prompt (default Baseline): Wilcoxon signed-rank
    on scores and exact McNemar on clear pushback, raw and Holm-corrected

Usage (from repo root):
  python src/analysis/prompt_stats.py aggregate --metrics metrics_summary.json
  python src/analysis/prompt_stats.py aggregate --metrics metrics_summary.json \\
      --benchmarks aime24,aime25,gpqa,global_mmlu_lite --dat-dir plots/phase1
  python src/analysis/prompt_stats.py aggregate --csv src/analysis/data/phase1_emma.csv
  python src/analysis/prompt_stats.py bullshit --logs-dir logs/bullshit_bench
  python src/analysis/prompt_stats.py bullshit --logs-dir logs/bullshit_bench --reference none

Reproduces the 07/17 slide from the Phase 1 sheet export:
  Friedman tokens chi2 = 13.19, p = 0.022; Friedman accuracy p = 0.60;
  Collapse below cell average 11/12 (sign test p = 0.006); Instruct token/accuracy
  r = -0.73 (AIME24), +0.71 (AIME25).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROMPT_ORDER = [
    "none",               # no system prompt (run without --prompt-key)
    "Baseline",
    "Purpose",
    "Autonomy",
    "Predicted_Optimal",
    "Pressure",
    "Threat",
    "Collapse",
]


# ----------------------------------------------------------------------------- helpers

def norm_prompt(name: str) -> str:
    name = str(name).strip()
    if name in ("", "default"):          # extract_metrics.py labels the no-prompt run "default"
        return "none"
    return name.replace(" ", "_")


def order_prompts(prompts) -> list[str]:
    known = [p for p in PROMPT_ORDER if p in prompts]
    extra = sorted(p for p in prompts if p not in PROMPT_ORDER)
    return known + extra


def short_model(name: str) -> str:
    """'Mistral 3 3B Instruct' / 'ministral-3-3b-instruct-2512-gguf' -> 'Instruct'."""
    low = name.lower()
    for key in ("reasoning", "instruct", "base"):
        if re.search(rf"\b{key}\b", low.replace("-", " ")):
            return key.capitalize()
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def sign_test(a, b) -> tuple[int, int, float]:
    """Two-sided sign test that a < b. Ties are dropped. Returns (k, n, p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    k = int((d < 0).sum())
    n = len(d)
    p = stats.binomtest(k, n, 0.5).pvalue if n else float("nan")
    return k, n, p


def wilcoxon_p(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(d == 0):
        return 1.0
    return stats.wilcoxon(a, b).pvalue


def mcnemar_exact(a_bool, b_bool) -> tuple[int, int, float]:
    """Exact McNemar on paired booleans. Returns (a_only, b_only, p)."""
    a = np.asarray(a_bool, bool)
    b = np.asarray(b_bool, bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n = a_only + b_only
    p = stats.binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p


def fmt_p(p: float) -> str:
    if p != p:  # nan
        return "  n/a"
    return f"{p:.3f}" if p >= 0.001 else "<.001"


# ----------------------------------------------------------------------------- loading (aggregate)

def load_metrics_json(path: Path, acc_key: str) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for model, benches in data.items():
        for bench, prompts in benches.items():
            for prompt, m in prompts.items():
                rows.append({
                    "model": model,
                    "benchmark": bench,
                    "prompt": norm_prompt(prompt),
                    "accuracy": m.get(acc_key),
                    "tokens": m.get("avg_completion_tokens"),
                })
    return pd.DataFrame(rows)


def load_sheet_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in col:
                return col[n]
        raise SystemExit(f"CSV {path} has no column like {names}; columns: {list(df.columns)}")

    return pd.DataFrame({
        "model": df[pick("model")],
        "benchmark": df[pick("benchmark")],
        "prompt": df[pick("prompt")].map(norm_prompt),
        "accuracy": pd.to_numeric(df[pick("pass@1", "accuracy")], errors="coerce"),
        "tokens": pd.to_numeric(df[pick("avg output tokens", "avg_completion_tokens", "output tokens")], errors="coerce"),
    })


def filter_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.benchmarks:
        keep = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        df = df[df.benchmark.isin(keep)]
    if args.models:
        subs = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        df = df[df.model.str.lower().apply(lambda s: any(x in s for x in subs))]
    if args.prompts:
        keep = [norm_prompt(p) for p in args.prompts.split(",") if p.strip()]
        df = df[df.prompt.isin(keep)]
    elif not args.include_none:
        # default: compare the system-prompt conditions; the no-prompt run is a separate control
        df = df[df.prompt != "none"]
    return df


# ----------------------------------------------------------------------------- aggregate mode

def run_aggregate(args) -> None:
    if bool(args.metrics) == bool(args.csv):
        raise SystemExit("give exactly one of --metrics or --csv")
    df = load_metrics_json(Path(args.metrics), args.acc_key) if args.metrics else load_sheet_csv(Path(args.csv))
    df = filter_rows(df, args).dropna(subset=["tokens"])
    if df.empty:
        raise SystemExit("no rows left after filtering")

    prompts = order_prompts(set(df.prompt))
    tok = df.pivot_table(index=["model", "benchmark"], columns="prompt", values="tokens")
    acc = df.pivot_table(index=["model", "benchmark"], columns="prompt", values="accuracy")
    tok = tok.reindex(columns=prompts)
    acc = acc.reindex(columns=prompts)

    complete = tok.dropna().index
    dropped = len(tok) - len(complete)
    tok_c = tok.loc[complete]
    acc_c = acc.loc[complete]
    n = len(tok_c)
    ref = norm_prompt(args.reference)

    print(f"prompts ({len(prompts)}): {', '.join(prompts)}")
    print(f"cells (model x benchmark) with every prompt: {n}" + (f"  [dropped {dropped} incomplete]" if dropped else ""))
    if n < 2:
        raise SystemExit("need at least 2 complete cells for the tests")

    print("\n== Friedman across prompts (blocks = model x benchmark) ==")
    f = stats.friedmanchisquare(*[tok_c[p].values for p in prompts])
    print(f"  output tokens : chi2 = {f.statistic:6.2f}  p = {fmt_p(f.pvalue)}")
    acc_ok = acc_c.dropna()
    if len(acc_ok) >= 2:
        f = stats.friedmanchisquare(*[acc_ok[p].values for p in prompts])
        print(f"  accuracy      : chi2 = {f.statistic:6.2f}  p = {fmt_p(f.pvalue)}  (n = {len(acc_ok)})")

    print("\n== mean rank per prompt (1 = fewest tokens) ==")
    ranks = tok_c.rank(axis=1).mean()
    print("  " + "  ".join(f"{p} {ranks[p]:.2f}" for p in prompts))

    if ref in prompts:
        others = [p for p in prompts if p != ref]
        below = int((tok_c[ref] < tok_c[prompts].mean(axis=1)).sum())
        p_below = stats.binomtest(below, n, 0.5).pvalue
        print(f"\n== {ref} vs cell average ==")
        print(f"  {ref} below its cell's average in {below}/{n} cells  (sign test p = {fmt_p(p_below)})")

        print(f"\n== {ref} vs each prompt, output tokens ({len(others)} comparisons, Holm-corrected) ==")
        sg = [sign_test(tok_c[ref], tok_c[p]) for p in others]
        wx = [wilcoxon_p(tok_c[ref], tok_c[p]) for p in others]
        sg_h = holm([s[2] for s in sg])
        wx_h = holm(wx)
        print(f"  {'prompt':<18} {'%diff':>7} {ref+'<':>9} {'sign p':>7} {'Holm':>6} {'Wilcox p':>9} {'Holm':>6}")
        for p, s, sh, w, wh in zip(others, sg, sg_h, wx, wx_h):
            pct = 100 * (tok_c[ref] / tok_c[p] - 1).mean()
            print(f"  {p:<18} {pct:>+6.0f}% {s[0]:>4}/{s[1]:<4} {fmt_p(s[2]):>7} {fmt_p(sh):>6} {fmt_p(w):>9} {fmt_p(wh):>6}")
    else:
        print(f"\n(reference prompt {ref!r} not present; skipping reference comparisons)")

    ratio = tok_c.max(axis=1) / tok_c.min(axis=1)
    cell = ratio.idxmax()
    lo, hi = tok_c.loc[cell].idxmin(), tok_c.loc[cell].idxmax()
    print("\n== largest single-cell gap ==")
    print(f"  {cell[0]} / {cell[1]}: {lo} {tok_c.loc[cell, lo]:.0f} tokens vs {hi} {tok_c.loc[cell, hi]:.0f} ({ratio.max():.1f}x)")

    print("\n== Pearson r(output tokens, accuracy) across prompts, per cell ==")
    for idx in acc_ok.index:
        x, y = tok_c.loc[idx], acc_ok.loc[idx]
        if x.std() == 0 or y.std() == 0:
            print(f"  {idx[0]:<40} {idx[1]:<20}   n/a (no variance)")
            continue
        r = stats.pearsonr(x, y)
        print(f"  {idx[0]:<40} {idx[1]:<20} r = {r.statistic:+.2f}  p = {fmt_p(r.pvalue)}")

    if args.dat_dir:
        write_dat(Path(args.dat_dir), tok_c, acc_c, prompts)


def write_dat(out: Path, tok: pd.DataFrame, acc: pd.DataFrame, prompts: list[str]) -> None:
    """pgfplots tables: x = prompt index, one column per model (Base/Instruct/Reasoning)."""
    out.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(tok.index.get_level_values(0)))
    labels = [short_model(m) for m in models]
    written = []

    def dump(path: Path, frame: pd.DataFrame, decimals: int):
        lines = ["x " + " ".join(labels)]
        for i, p in enumerate(prompts):
            vals = []
            for m in models:
                v = frame.loc[m, p] if m in frame.index else np.nan
                vals.append("nan" if pd.isna(v) else str(round(float(v), decimals)))
            lines.append(f"{i} " + " ".join(vals))
        path.write_text("\n".join(lines) + "\n")
        written.append(path.name)

    for bench in dict.fromkeys(tok.index.get_level_values(1)):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(bench)).strip("_")
        dump(out / f"tok_{safe}.dat", tok.xs(bench, level=1), 2)
        dump(out / f"acc_{safe}.dat", acc.xs(bench, level=1), 4)
    norm = tok.div(tok.mean(axis=1), axis=0).groupby(level=0).mean()
    dump(out / "tok_norm.dat", norm, 4)
    print(f"\nwrote {len(written)} .dat files to {out}/ (x = {', '.join(f'{i}:{p}' for i, p in enumerate(prompts))})")


# ----------------------------------------------------------------------------- bullshit mode

def load_bullshit(logs_dir: Path) -> pd.DataFrame:
    rows = []
    for summary in sorted(logs_dir.rglob("summary.json")):
        rel = summary.parent.relative_to(logs_dir).parts
        if len(rel) == 1:
            model, prompt = rel[0], "none"
        elif len(rel) == 2:
            model, prompt = rel[0], rel[1]
        else:
            continue
        data = json.loads(summary.read_text())
        if "score_distribution" not in data:
            print(f"  (skipping ungraded run {'/'.join(rel)})", file=sys.stderr)
            continue
        for q in data.get("per_question", {}).values():
            rows.append({
                "model": model,
                "prompt": norm_prompt(prompt),
                "qid": q.get("id"),
                "domain_group": q.get("domain_group"),
                "technique": q.get("technique"),
                "score": q.get("score"),
            })
    return pd.DataFrame(rows)


def run_bullshit(args) -> None:
    df = load_bullshit(Path(args.logs_dir))
    if df.empty:
        raise SystemExit(f"no graded runs under {args.logs_dir}")
    if args.models:
        subs = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        df = df[df.model.str.lower().apply(lambda s: any(x in s for x in subs))]
    if args.prompts:
        df = df[df.prompt.isin([norm_prompt(p) for p in args.prompts.split(",")])]
    ref = norm_prompt(args.reference)

    for model, g in df.groupby("model"):
        prompts = order_prompts(set(g.prompt))
        mat = g.pivot_table(index="qid", columns="prompt", values="score", aggfunc="first").reindex(columns=prompts)
        print(f"\n######## {model}")
        print(f"  {'prompt':<18} {'n':>4} {'mean':>6} {'clear%':>7} {'none':>5}")
        for p in prompts:
            col = mat[p]
            graded = col.dropna()
            clear = 100 * (graded == 2).mean() if len(graded) else float("nan")
            print(f"  {p:<18} {len(graded):>4} {graded.mean():>6.2f} {clear:>6.1f}% {int(col.isna().sum()):>5}")

        full = mat.dropna()
        dropped = len(mat) - len(full)
        print(f"\n  questions with a score under every prompt: {len(full)}" + (f"  [dropped {dropped}]" if dropped else ""))
        if len(prompts) >= 3 and len(full) >= 2:
            if full.nunique().max() <= 1 and full.std(axis=1).max() == 0:
                print("  Friedman: all scores identical across prompts, not testable")
            else:
                f = stats.friedmanchisquare(*[full[p].values for p in prompts])
                print(f"  Friedman across {len(prompts)} prompts: chi2 = {f.statistic:.2f}  p = {fmt_p(f.pvalue)}")

        if ref not in prompts:
            print(f"  (reference {ref!r} not present for this model; skipping pairwise tests)")
            continue
        others = [p for p in prompts if p != ref]
        wx, mc = [], []
        for p in others:
            pair = mat[[ref, p]].dropna()
            wx.append(wilcoxon_p(pair[p], pair[ref]))
            mc.append(mcnemar_exact(pair[p] == 2, pair[ref] == 2) + (len(pair),))
        wx_h, mc_h = holm(wx), holm([m[2] for m in mc])
        print(f"\n  each prompt vs {ref} ({len(others)} comparisons, Holm-corrected)")
        print(f"  {'prompt':<18} {'n':>4} {'dMean':>6} {'Wilcox p':>9} {'Holm':>6} {'clear +/-':>10} {'McNemar p':>10} {'Holm':>6}")
        for p, w, wh, m, mh in zip(others, wx, wx_h, mc, mc_h):
            pair = mat[[ref, p]].dropna()
            d = pair[p].mean() - pair[ref].mean()
            print(f"  {p:<18} {m[3]:>4} {d:>+6.2f} {fmt_p(w):>9} {fmt_p(wh):>6} {m[0]:>4}/{m[1]:<5} {fmt_p(m[2]):>10} {fmt_p(mh):>6}")

        if args.by:
            print(f"\n  mean score by {args.by}")
            tab = g.pivot_table(index=args.by, columns="prompt", values="score", aggfunc="mean").reindex(columns=prompts)
            print(tab.round(2).to_string().replace("\n", "\n  ").join(["  ", ""]))


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("aggregate", help="one number per model x benchmark x prompt")
    a.add_argument("--metrics", help="metrics_summary.json from scripts/extract_metrics.py")
    a.add_argument("--csv", help="results-sheet CSV (Model, Prompt, Benchmark, Pass@1, Avg Output tokens)")
    a.add_argument("--acc-key", default="accuracy", help="metrics_summary field used as accuracy (e.g. mean_score)")
    a.add_argument("--benchmarks", default="", help="comma-separated benchmark names to keep")
    a.add_argument("--models", default="", help="comma-separated substrings of model names to keep")
    a.add_argument("--prompts", default="", help="comma-separated prompts to compare (default: all system-prompt conditions)")
    a.add_argument("--include-none", action="store_true", help="also include the no-system-prompt run as a condition")
    a.add_argument("--reference", default="Collapse", help="prompt compared against every other prompt")
    a.add_argument("--dat-dir", default="", help="write pgfplots .dat files here")

    b = sub.add_parser("bullshit", help="per-question BullshitBench judge scores")
    b.add_argument("--logs-dir", default="logs/bullshit_bench")
    b.add_argument("--models", default="")
    b.add_argument("--prompts", default="", help="comma-separated prompts to include (use 'none' for the no-system-prompt run)")
    b.add_argument("--reference", default="Baseline", help="prompt every other prompt is compared against ('none' = no system prompt)")
    b.add_argument("--by", choices=["domain_group", "technique"], help="also print mean score broken down by this field")

    args = ap.parse_args()
    run_aggregate(args) if args.mode == "aggregate" else run_bullshit(args)


if __name__ == "__main__":
    main()
