import re

from datasets import load_dataset

from benchmarks.base import Benchmark


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _extract_answer(response: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    if not lines:
        return None
    # Try to extract a numeric/mathematical token from the last line
    # rather than returning the whole sentence as the answer
    m = re.search(r"[-+]?\d[\d/.,]*(?:\^\{[^}]+\}|\*\*\d+)?", lines[-1])
    if m:
        return m.group(0).strip()
    return lines[-1]


class MathArenaApexBenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        # split is "train" — only split, 12 problems total
        return list(load_dataset("MathArena/apex_2025", split="train"))

    def get_question_text(self, row: dict) -> str:
        return row["problem"]

    def get_label(self, row: dict) -> str:
        return row.get("source", f"problem_{row.get('problem_idx', '?')}")

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted = _extract_answer(answer)
        correct_answer = str(row["answer"])
        correct = (
            extracted is not None
            and _normalize(extracted) == _normalize(correct_answer)
        )
        return {
            "source":           row.get("source", ""),
            "problem_idx":      row.get("problem_idx"),
            "question":         row["problem"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   correct_answer,
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "source":         row.get("source", ""),
            "problem_idx":    row.get("problem_idx"),
            "correct_answer": str(row["answer"]),
        }
