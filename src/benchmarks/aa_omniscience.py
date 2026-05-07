import re

from datasets import load_dataset

from benchmarks.base import Benchmark


_DATASET = "ArtificialAnalysis/AA-Omniscience-Public"

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _check_answer(response: str, correct: str) -> bool:
    norm_correct = _normalize(correct)
    norm_response = _normalize(response)
    return norm_correct in norm_response


class AAOmniscienceBenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        # Only split is "train" (600 public questions)
        ds = list(load_dataset(_DATASET, split="train"))
        domain = cfg.get("domain")
        if domain:
            ds = [r for r in ds if r.get("domain", "").lower() == domain.lower()]
        return ds

    def get_question_text(self, row: dict) -> str:
        return row["question"]

    def get_label(self, row: dict) -> str:
        return f"[{row.get('domain', '?')} / {row.get('topic', '?')}] {row['question'][:60]}..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        correct_answer = row["answer"]
        correct = _check_answer(answer, correct_answer)
        return {
            "domain":           row.get("domain", ""),
            "topic":            row.get("topic", ""),
            "question":         row["question"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "correct_answer":   correct_answer,
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "domain":         row.get("domain", ""),
            "topic":          row.get("topic", ""),
            "question":       row["question"],
            "correct_answer": row["answer"],
        }
