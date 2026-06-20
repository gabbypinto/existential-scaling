import json
import re
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset

from benchmarks.base import Benchmark


def load_lcb(version_tag: str, difficulty: str | None = None) -> list:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=version_tag,
        split="test",
        trust_remote_code=True,
    )
    if difficulty:
        ds = [row for row in ds if row["difficulty"].lower() == difficulty.lower()]
    return list(ds)


def extract_code(response: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip FIM tokens, treat raw Python code directly (base models)
    cleaned = re.split(r"<\|fim_\w+\|>", response)[0].strip()
    if re.match(r"^(import |from |def |class |#)", cleaned):
        return cleaned
    return None


def run_against_tests(code: str, test_cases: list, timeout: int = 10) -> list[dict]:
    """test_cases: list of {"input": str, "output": str} dicts."""
    results = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name

    try:
        for i, tc in enumerate(test_cases):
            inp, expected = tc["input"], tc["output"]
            try:
                proc = subprocess.run(
                    ["python", fname],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                passed = proc.stdout.strip() == expected.strip()
                results.append({
                    "test":   i,
                    "passed": passed,
                    "stderr": proc.stderr[:300] if proc.stderr else None,
                })
            except subprocess.TimeoutExpired:
                results.append({"test": i, "passed": False, "stderr": "timeout"})
            except Exception as e:
                results.append({"test": i, "passed": False, "stderr": str(e)[:300]})
    finally:
        Path(fname).unlink(missing_ok=True)

    return results


class LCBBenchmark(Benchmark):
    def __init__(self):
        self._timeout = 10

    def load_problems(self, cfg: dict) -> list:
        self._timeout = cfg.get("timeout_per_test", 10)
        return load_lcb(cfg["release_version"], cfg.get("difficulty"))

    def get_question_text(self, row: dict) -> str:
        return row["question_content"]

    def get_label(self, row: dict) -> str:
        return f"{row['question_title']} ({row['platform']} / {row['difficulty']})"

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        timeout = self._timeout
        test_cases = json.loads(row["public_test_cases"])
        code = extract_code(answer)
        if code:
            test_results = run_against_tests(code, test_cases, timeout)
            num_passed = sum(r["passed"] for r in test_results)
            num_tests = len(test_results)
            passed_all = num_passed == num_tests
        else:
            test_results = []
            num_passed = 0
            num_tests = len(test_cases)
            passed_all = False

        return {
            "question_title":   row["question_title"],
            "platform":         row["platform"],
            "difficulty":       row["difficulty"],
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_code":   code,
            "test_results":     test_results,
            "passed_all_tests": passed_all,
            "correct":          passed_all,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "question_title": row["question_title"],
            "platform":       row["platform"],
            "difficulty":     row["difficulty"],
        }
