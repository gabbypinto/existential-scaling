import re

from datasets import load_dataset

from benchmarks.base import Benchmark


_DATASET = "mrlbenchmarks/global-piqa-nonparallel"
_ANSWER_RE = re.compile(r"\bAnswer:\s*([01])\b", re.IGNORECASE)
_DIGIT_RE  = re.compile(r"\b([01])\b")

def _extract_choice(answer: str) -> int | None:
    # "Answer: 0" / "Answer: 1" anywhere in the response
    m = _ANSWER_RE.search(answer)
    if m:
        return int(m.group(1))
    # bare digit on the very last non-empty line only (like A, B etc)
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if lines:
        m = _DIGIT_RE.search(lines[-1])
        if m:
            return int(m.group(1))
    return None


def _format_question(row: dict) -> str:
    return (
        f"{row['prompt']}\n\n"
        f"Solution 0: {row['solution0']}\n"
        f"Solution 1: {row['solution1']}\n\n"
        "Which solution is correct? Answer with 0 or 1."
    )

class GlobalPIQABenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        subset = cfg.get("subset", "acm_arab")
        return list(load_dataset(_DATASET, subset, split="test"))

    def get_question_text(self, row: dict) -> str:
        return _format_question(row)

    def get_label(self, row: dict) -> str:
        return row["prompt"][:80]

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted = _extract_choice(answer)
        correct = extracted == row["label"]
        return {
            "prompt":           row["prompt"],
            "solution0":        row["solution0"],
            "solution1":        row["solution1"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   row["label"],
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "prompt":         row["prompt"],
            "correct_answer": row["label"],
        }
