import re

from datasets import load_dataset

from benchmarks.base import Benchmark


_DATASET = "allenai/social_i_qa"
_ANSWER_KEYS = ["answerA", "answerB", "answerC"]
_LETTERS = ["A", "B", "C"]
_ANSWER_RE = re.compile(r"\bAnswer:\s*([ABC])\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([ABC])\b")


def _format_question(row: dict) -> str:
    options = "\n".join(
        f"{letter}. {row[key].strip()}"
        for letter, key in zip(_LETTERS, _ANSWER_KEYS)
    )
    return f"{row['context'].strip()}\n\n{row['question'].strip()}\n\n{options}"


def _extract_letter(response: str) -> str | None:
    # Regex: an explicit "Answer: X"
    m = _ANSWER_RE.search(response)
    if m:
        return m.group(1).upper()
    # Fallback Regex: a standalone A/B/C, scanning the last non-empty lines first
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    for line in reversed(lines):
        m = _LETTER_RE.search(line)
        if m:
            return m.group(1)
    return None


def _correct_letter(row: dict) -> str:
    # label is a 1-indexed string ("1"/"2"/"3") and the loader leaves a trailing newline
    return _LETTERS[int(row["label"].strip()) - 1]


class SocialIQABenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        rows = list(load_dataset(_DATASET, split="validation", trust_remote_code=True))
        sample_size = cfg.get("sample_size")
        if sample_size is not None:
            rows = rows[:sample_size]
        return rows

    def get_question_text(self, row: dict) -> str:
        return _format_question(row)

    def get_label(self, row: dict) -> str:
        return f"{row['question'].strip()[:70]}..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted = _extract_letter(answer)
        correct_answer = _correct_letter(row)
        return {
            "context":          row["context"],
            "question":         row["question"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_answer": extracted,
            "correct_answer":   correct_answer,
            "correct":          extracted == correct_answer,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question":       row["question"],
            "correct_answer": _correct_letter(row),
        }
