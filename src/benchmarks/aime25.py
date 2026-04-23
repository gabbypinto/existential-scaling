import re

from datasets import load_dataset

from benchmarks.base import Benchmark


class AIME25Benchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        return list(load_dataset("math-ai/aime25", split="test"))

    def get_question_text(self, row: dict) -> str:
        return row["problem"]

    def get_label(self, row: dict) -> str:
        return row["problem"][:80] + "..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        boxed = re.search(r"\\boxed\{(\d+)\}", answer)
        extracted = boxed.group(1) if boxed else None
        correct = extracted == str(row["answer"])
        return {
            "question":         row["problem"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   str(row["answer"]),
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question":       row["problem"],
            "correct_answer": str(row["answer"]),
        }
