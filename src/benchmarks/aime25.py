import re

from datasets import load_dataset

from benchmarks.base import Benchmark


def grade(answer: str, correct_answer) -> tuple[str | None, bool]:
    """Extract the boxed integer and compare numerically (keys like "025" match 25)."""
    boxed = re.search(r"\\boxed\{(\d+)\}", answer)
    extracted = boxed.group(1) if boxed else None
    return extracted, extracted is not None and int(extracted) == int(correct_answer)


class AIME25Benchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        return list(load_dataset("math-ai/aime25", split="test"))

    def get_question_text(self, row: dict) -> str:
        return row["problem"]

    def get_label(self, row: dict) -> str:
        return row["problem"][:80] + "..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted, correct = grade(answer, row["answer"])
        return {
            "question":         row["problem"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   str(int(row["answer"])),
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question":       row["problem"],
            "correct_answer": str(int(row["answer"])),
        }
