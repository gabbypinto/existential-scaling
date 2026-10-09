#!/usr/bin/env python3
"""
Regrade saved AIME runs with the current grader and rebuild their summaries.

Recomputes extracted_answer / correct from each result's stored raw_answer and
correct_answer (no model calls), then rewrites per-question pass fields and
overall_pass_at_1 / overall_pass_at_k in summary.json. Every file is copied to
<name>.bak before its first rewrite; existing .bak files are never overwritten,
so re-running keeps the original backup.

Usage:
    python scripts/regrade.py logs_new/aime24 logs_new/aime25
    python scripts/regrade.py logs/AIME2024 --dry-run
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.aime24 import grade as grade_aime24  # noqa: E402
from benchmarks.aime25 import grade as grade_aime25  # noqa: E402
from eval_base import accuracy_metrics, question_pass_fields  # noqa: E402

GRADERS = {"aime24": grade_aime24, "aime25": grade_aime25}


def _backup_and_write(path: Path, data: dict, dry_run: bool) -> None:
    if dry_run:
        return
    bak = path.with_name(path.name + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _round_num(path: Path) -> int:
    return int(re.search(r"round-(\d+)", path.name).group(1))


def regrade_run(run_dir: Path, dry_run: bool) -> dict | None:
    round_files = sorted(run_dir.glob("round-*_results.json"), key=_round_num)
    rounds = {_round_num(rf): (rf, json.loads(rf.read_text())) for rf in round_files}
    benchmark = next(iter(rounds.values()))[1]["config"].get("benchmark")
    grade = GRADERS.get(benchmark)
    if grade is None:
        print(f"  skip {run_dir} (no grader for benchmark '{benchmark}')")
        return None

    flips = 0
    passing: dict[str, list[int]] = {}
    seen: dict[str, int] = {}
    for rnd, (rf, data) in rounds.items():
        changed = False
        for qkey, res in data["results"].items():
            extracted, correct = grade(res.get("raw_answer") or "", res["correct_answer"])
            correct_answer = str(int(res["correct_answer"]))
            if correct != res.get("correct"):
                flips += 1
            new = {"extracted_answer": extracted, "correct": correct, "correct_answer": correct_answer}
            if any(res.get(k) != v for k, v in new.items()):
                res.update(new)
                changed = True
            seen[qkey] = seen.get(qkey, 0) + 1
            passing.setdefault(qkey, [])
            if correct:
                passing[qkey].append(rnd)
        if changed:
            _backup_and_write(rf, data, dry_run)

    row = {"run": str(run_dir), "flips": flips, "old": None}
    summary_path = run_dir / "summary.json"
    per_q = {}
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        row["old"] = summary.get("overall_pass_at_1")
        per_q = summary.get("per_question", {})
    for qkey in passing:
        entry = per_q.setdefault(qkey, {})
        entry.update(question_pass_fields(passing[qkey], seen[qkey]))
        if "correct_answer" in entry:
            entry["correct_answer"] = str(int(entry["correct_answer"]))
    acc = accuracy_metrics(per_q)
    row.update(pass_at_1=acc["overall_pass_at_1"], pass_at_k=acc["overall_pass_at_k"], k=acc["num_rounds"])

    if summary_path.exists():
        before = json.dumps(summary, sort_keys=True)
        summary.update(acc)
        summary["per_question"] = per_q
        if json.dumps(summary, sort_keys=True) != before:
            _backup_and_write(summary_path, summary, dry_run)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path, help="log dirs to search for round-*_results.json")
    p.add_argument("--dry-run", action="store_true", help="report changes without writing anything")
    args = p.parse_args()

    run_dirs = sorted({rf.parent for d in args.dirs for rf in d.rglob("round-*_results.json")})
    if not run_dirs:
        raise SystemExit(f"no round-*_results.json under {', '.join(map(str, args.dirs))}")

    rows = [r for r in (regrade_run(d, args.dry_run) for d in run_dirs) if r]

    fmt = lambda v: "-" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    print(f"\n{'run':<62} {'flips':>6} {'old (pass@k)':>13} {'pass@1':>8} {'pass@k':>8} {'k':>3}")
    print("-" * 104)
    for r in rows:
        print(f"{r['run']:<62} {r['flips']:>6} {fmt(r['old']):>13} {fmt(r['pass_at_1']):>8} {fmt(r['pass_at_k']):>8} {r['k']:>3}")
    print(f"\n{len(rows)} runs, {sum(r['flips'] for r in rows)} results flipped"
          + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
