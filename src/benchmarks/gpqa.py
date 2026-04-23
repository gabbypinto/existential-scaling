import re

from datasets import load_dataset

from benchmarks.base import Benchmark

_DATASET = "fingertap/GPQA-Diamond"

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_letter(answer: str) -> str | None:
    """Return the last A/B/C/D letter found in the response."""
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    for line in reversed(lines):
        m = _LETTER_RE.search(line)
        if m:
            return m.group(1)
    return None


class GPQABenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        return list(load_dataset(_DATASET, split="test"))

    def get_question_text(self, row: dict) -> str:
        return row["question"]

    def get_label(self, row: dict) -> str:
        return row["question"][:80] + "..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted = _extract_letter(answer)
        correct_answer = row["answer"]
        correct = extracted == correct_answer
        return {
            "question":         row["question"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   correct_answer,
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question":       row["question"][:120],
            "correct_answer": row["answer"],
        }
