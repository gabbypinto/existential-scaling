import re

from datasets import load_dataset

from benchmarks.base import Benchmark


_DATASET = "cais/mmlu"
_LETTERS = ["A", "B", "C", "D"]

_ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

def _format_question(row: dict) -> str:
    options = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(_LETTERS, row["choices"])
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


class MMLUBenchmark(Benchmark):
    def load_problems(self, cfg: dict) -> list:
        subject = cfg.get("subject", "all")
        sample_pct = cfg.get("sample_pct")

        if subject == "all" and sample_pct is not None:
            # Load each subject individually, take first sample_pct%, combine
            fraction = sample_pct / 100.0
            problems = []
            for subj in _ALL_SUBJECTS:
                rows = list(load_dataset(_DATASET, subj, split="test"))
                n = max(1, round(len(rows) * fraction))
                problems.extend(rows[:n])
            return problems

        rows = list(load_dataset(_DATASET, subject, split="test"))
        if sample_pct is not None:
            n = max(1, round(len(rows) * sample_pct / 100.0))
            rows = rows[:n]
        return rows

    def get_question_text(self, row: dict) -> str:
        return _format_question(row)

    def get_label(self, row: dict) -> str:
        return f"[{row.get('subject', '?')}] {row['question'][:70]}..."

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        extracted = _extract_letter(answer)
        correct_answer = _LETTERS[row["answer"]]
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
            "correct_answer": _LETTERS[row["answer"]],
        }
