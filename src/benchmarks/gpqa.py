import hashlib
import random
import re

from datasets import load_dataset

from benchmarks.base import Benchmark


_DATASET = "Idavidrein/gpqa"
_LETTERS = ["A", "B", "C", "D"]
_LETTER_RE = re.compile(r"\b([A-D])\b")

def _shuffle_options(row: dict) -> tuple[list[tuple[str, str]], str]:
    """Deterministically shuffle the 4 options and return labeled pairs + correct letter."""
    options = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    seed = int(hashlib.md5(row["Question"].encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(options)
    correct_letter = _LETTERS[options.index(row["Correct Answer"])]
    return list(zip(_LETTERS, options)), correct_letter


def _extract_letter(answer: str) -> str | None:
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    for line in reversed(lines):
        m = _LETTER_RE.search(line)
        if m:
            return m.group(1)
    return None


class GPQABenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        subset = cfg.get("subset", "gpqa_diamond")
        rows = [dict(r) for r in load_dataset(_DATASET, subset, split="train")]
        for row in rows:
            _, row["_correct_letter"] = _shuffle_options(row)
        return rows

    def get_question_text(self, row: dict) -> str:
        labeled, _ = _shuffle_options(row)
        options_text = "\n".join(f"({letter}) {text}" for letter, text in labeled)
        return f"{row['Question']}\n\n{options_text}"

    def get_label(self, row: dict) -> str:
        return row["Question"][:80] + "..."

    def build_result(self, row: dict, thinking: str, answer: str, _metrics: dict, _elapsed: float) -> dict:
        extracted = _extract_letter(answer)
        correct_letter = row["_correct_letter"]
        return {
            "question":         row["Question"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   correct_letter,
            "correct":          extracted == correct_letter,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question":       row["Question"][:120],
            "correct_answer": row["_correct_letter"],
        }
