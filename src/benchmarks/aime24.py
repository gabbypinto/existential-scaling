import re

from datasets import load_dataset

from benchmarks.base import Benchmark


class AIME24Benchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        # split is "train" — only split available, 30 problems (I+II combined)
        return list(load_dataset("HuggingFaceH4/aime_2024", split="train"))

    def get_question_text(self, row: dict) -> str:
        return row["problem"]

    def get_label(self, row: dict) -> str:
        problem_id = row.get("id", "")
        return f"{problem_id}: {row['problem'][:60]}..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        boxed = re.search(r"\\boxed\{(\d+)\}", answer)
        extracted = boxed.group(1) if boxed else None
        correct_answer = str(row["answer"])
        correct = extracted == correct_answer
        return {
            "problem_id":       row.get("id", ""),
            "question":         row["problem"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   correct_answer,
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "problem_id":     row.get("id", ""),
            "question":       row["problem"],
            "correct_answer": str(row["answer"]),
        }
