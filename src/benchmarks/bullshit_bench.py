"""BullshitBench V2 - 100 nonsensical questions (software / finance / legal / medical / physics).

Source: https://github.com/petergpt/bullshit-benchmark  (questions.v2.json)

Baseline protocol (team deck, 07/27):
  * user turn = the question text only (pre_prompt / post_prompt are empty)
  * no system prompt  -> run WITHOUT --prompt-key
  * temperature 0     -> run with --model model_bullshit

Scoring is not done during generation. Responses are graded afterwards by an
LLM judge on the official 0/1/2 rubric:
    bash scripts/grade_bullshit_bench.sh --slot <judge slot>
Until graded, every result has score=None and correct=False, so the pass@1
printed by run_eval is meaningless for this benchmark. After grading,
"correct" means score == 2 (clear pushback), matching the official
"clear score" metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from benchmarks.base import Benchmark

_SRC_DIR = Path(__file__).resolve().parent.parent            # .../src
_DEFAULT_FILE = Path("data") / "bullshit_bench" / "questions.v2.json"
_RAW_URL = "https://raw.githubusercontent.com/petergpt/bullshit-benchmark/main/questions.v2.json"

_KEEP_FIELDS = (
    "id", "question", "nonsensical_element", "domain", "domain_group",
    "technique", "difficulty", "difficulty_label", "is_control",
)


def _as_list(value) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def _flatten(doc: dict) -> list[dict]:
    """questions.v2.json nests questions under techniques[]; flatten to one row per question."""
    rows = []
    for tech in doc.get("techniques", []):
        for q in tech.get("questions", []):
            row = {k: q.get(k) for k in _KEEP_FIELDS}
            if not row.get("technique"):
                row["technique"] = tech.get("technique")
            rows.append(row)
    return rows


def _resolve_path(cfg: dict) -> Path:
    path = Path(cfg.get("questions_file") or _DEFAULT_FILE)
    if not path.is_absolute():
        path = _SRC_DIR / path
    return path


class BullshitBenchBenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        path = _resolve_path(cfg)
        if not path.exists():
            print(f"{path} not found - downloading from {_RAW_URL}")
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(_RAW_URL, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)

        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = _flatten(doc)

        # v2 ships no control questions, but never grade a control as nonsense.
        rows = [r for r in rows if not r.get("is_control")]

        groups = _as_list(cfg.get("domain_groups"))
        if groups:
            rows = [r for r in rows if r.get("domain_group") in groups]
        techniques = _as_list(cfg.get("techniques"))
        if techniques:
            rows = [r for r in rows if r.get("technique") in techniques]

        # stable order: by id, so limit/--limit smoke tests are reproducible
        rows.sort(key=lambda r: r.get("id") or "")
        return rows

    def get_question_text(self, row: dict) -> str:
        # Question only. pre_prompt / post_prompt in the YAML are intentionally empty.
        return row["question"].strip()

    def get_label(self, row: dict) -> str:
        return f"[{row.get('id')}] {row['question'][:70]}..."

    def build_result(self, row: dict, thinking: str, answer: str, _metrics: dict, _elapsed: float) -> dict:
        return {
            "id":                  row.get("id"),
            "question":            row["question"],
            "nonsensical_element": row.get("nonsensical_element"),
            "domain":              row.get("domain"),
            "domain_group":        row.get("domain_group"),
            "technique":           row.get("technique"),
            "thinking":            thinking,
            "raw_answer":          answer,
            # filled in by grade_bullshit_bench.py
            "score":               None,
            "judge_justification": None,
            "judge_model":         None,
            # run_eval needs a bool; becomes (score == 2) after grading
            "correct":             False,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "id":           row.get("id"),
            "question":     row["question"][:120],
            "domain_group": row.get("domain_group"),
            "technique":    row.get("technique"),
        }
