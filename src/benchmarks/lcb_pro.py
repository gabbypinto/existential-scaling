import os

from datasets import load_dataset

from benchmarks.base import Benchmark
from benchmarks.lcb import extract_code, run_against_tests
import json


_DATASET = "QAQAQAQAQ/LiveCodeBench-Pro"

class LCBProBenchmark(Benchmark):
    def __init__(self):
        self._timeout = 10

    def load_problems(self, cfg: dict) -> list:
        self._timeout = cfg.get("timeout_per_test", 10)
        difficulty = cfg.get("difficulty")
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        ds = load_dataset(_DATASET, split="test", token=token, trust_remote_code=True)
        problems = list(ds)
        if difficulty:
            problems = [r for r in problems if r.get("difficulty", "").lower() == difficulty.lower()]
        return problems

    def get_question_text(self, row: dict) -> str:
        return row["question_content"]

    def get_label(self, row: dict) -> str:
        title = row.get("question_title", row.get("question_id", "?"))
        diff = row.get("difficulty", "")
        platform = row.get("platform", "")
        return f"{title} ({platform} / {diff})"

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        test_cases = json.loads(row["public_test_cases"])
        code = extract_code(answer)
        if code:
            test_results = run_against_tests(code, test_cases, self._timeout)
            num_passed = sum(r["passed"] for r in test_results)
            passed_all = num_passed == len(test_results)
        else:
            test_results = []
            passed_all = False

        return {
            "question_title":   row.get("question_title", ""),
            "platform":         row.get("platform", ""),
            "difficulty":       row.get("difficulty", ""),
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_code":   code,
            "test_results":     test_results,
            "passed_all_tests": passed_all,
            "correct":          passed_all,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question_title": row.get("question_title", ""),
            "platform":       row.get("platform", ""),
            "difficulty":     row.get("difficulty", ""),
        }
