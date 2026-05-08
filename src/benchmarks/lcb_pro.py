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
        splits = cfg.get("splits", ["biannual_2025_1_6"])
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if isinstance(splits, str):
            splits = [splits]
        problems = []
        seen = set()
        for split in splits:
            ds = load_dataset(_DATASET, split=split, token=token, trust_remote_code=True)
            for row in ds:
                key = row.get("question_id") or row.get("question_title", "")
                if key not in seen:
                    seen.add(key)
                    problems.append(dict(row))
        if difficulty:
            problems = [r for r in problems if r.get("difficulty", "").lower() == difficulty.lower()]
        return problems

    def get_question_text(self, row: dict) -> str:
        return row["problem_statement"]

    def get_label(self, row: dict) -> str:
        return f"{row.get('problem_title', row.get('problem_id', '?'))} ({row.get('platform', '')} / {row.get('difficulty', '')})"

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        code = extract_code(answer)
        raw_test_cases = row.get("public_test_cases")
        if code and raw_test_cases:
            test_cases = json.loads(raw_test_cases) if isinstance(raw_test_cases, str) else raw_test_cases
            test_results = run_against_tests(code, test_cases, self._timeout)
            passed_all = all(r["passed"] for r in test_results)
        else:
            test_results = []
            # NOTE: LCB Pro main dataset has no test cases — test cases are in the
            # separate gated repo QAQAQAQAQ/LiveCodeBench-Pro-Testcase.
            # Without them, correct is always False.
            passed_all = False

        return {
            "problem_id":       row.get("problem_id", ""),
            "problem_title":    row.get("problem_title", ""),
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
            "problem_id":    row.get("problem_id", ""),
            "problem_title": row.get("problem_title", ""),
            "platform":      row.get("platform", ""),
            "difficulty":    row.get("difficulty", ""),
        }
