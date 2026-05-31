import re

from datasets import load_dataset

from benchmarks.base import Benchmark


_DATASET = "CohereLabs/Global-MMLU-Lite"
_OPTION_KEYS = ["option_a", "option_b", "option_c", "option_d"]
_LETTERS = ["A", "B", "C", "D"]

def _format_question(row: dict) -> str:
    options = "\n".join(
        f"{letter}. {row[key]}"
        for letter, key in zip(_LETTERS, _OPTION_KEYS)
        if row.get(key)
    )
    return f"{row['question']}\n\n{options}"


def _extract_letter(response: str) -> str | None:
    m = re.search(r"\bAnswer:\s*([ABCD])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    if lines:
        m = re.fullmatch(r"([ABCD])[.):\s]?.*", lines[-1], re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


class GlobalMMLULiteBenchmark(Benchmark):
    def __init__(self):
        self._language = "en"

    def load_problems(self, cfg: dict) -> list:
        self._language = cfg.get("language", "en")
        return list(load_dataset(_DATASET, self._language, split="test"))

    def get_question_text(self, row: dict) -> str:
        return _format_question(row)

    def get_label(self, row: dict) -> str:
        return f"[{row.get('subject', '?')}] {row['question'][:70]}..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted = _extract_letter(answer)
        correct_answer = row["answer"].upper()
        correct = extracted == correct_answer
        return {
            "subject":          row.get("subject", ""),
            "question":         row["question"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   correct_answer,
            "correct":          correct,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "subject":        row.get("subject", ""),
            "question":       row["question"],
            "correct_answer": row["answer"].upper(),
        }
