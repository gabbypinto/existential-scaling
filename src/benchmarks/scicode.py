import re
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset

from benchmarks.base import Benchmark

_DATASET = "SciCode1/SciCode"

_CODE_RE_PYTHON = re.compile(r"```python\s*(.*?)```", re.DOTALL)
_CODE_RE_GENERIC = re.compile(r"```\s*(.*?)```", re.DOTALL)


def _extract_code(response: str) -> str | None:
    m = _CODE_RE_PYTHON.search(response)
    if m:
        return m.group(1).strip()
    m = _CODE_RE_GENERIC.search(response)
    if m:
        return m.group(1).strip()
    return None


def _run_scicode_tests(code: str, dependencies: str, test_strings: list[str], timeout: int = 30) -> list[dict]:
    results = []
    for i, test_src in enumerate(test_strings):
        script = "\n".join([
            dependencies or "",
            code,
            "",
            test_src,
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            fname = f.name

        try:
            proc = subprocess.run(
                ["python", fname],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = proc.returncode == 0
            stderr = proc.stderr[:300] if proc.stderr else None
            results.append({"test": i, "passed": passed, "stderr": stderr})
        except subprocess.TimeoutExpired:
            results.append({"test": i, "passed": False, "stderr": "timeout"})
        except Exception as e:
            results.append({"test": i, "passed": False, "stderr": str(e)[:300]})
        finally:
            Path(fname).unlink(missing_ok=True)

    return results


def _format_question(row: dict, include_background: bool = True) -> str:
    parts = [row["problem_description_main"]]
    if include_background and row.get("problem_background_main"):
        parts.append(f"Background:\n{row['problem_background_main']}")
    if row.get("problem_io"):
        parts.append(f"Input/Output specification:\n{row['problem_io']}")
    return "\n\n".join(parts)


class SciCodeBenchmark(Benchmark):
    def __init__(self):
        self._timeout = 30
        self._include_background = True

    def load_problems(self, cfg: dict) -> list:
        self._timeout = cfg.get("timeout_per_test", 30)
        self._include_background = cfg.get("include_background", True)
        return list(load_dataset(_DATASET, split="test"))

    def get_question_text(self, row: dict) -> str:
        return _format_question(row, self._include_background)

    def get_label(self, row: dict) -> str:
        return row.get("problem_name", row.get("problem_id", "?"))

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        code = _extract_code(answer)
        deps = row.get("required_dependencies", "")
        tests = row.get("general_tests", [])

        if code:
            test_results = _run_scicode_tests(code, deps, tests, self._timeout)
            num_passed = sum(r["passed"] for r in test_results)
            passed_all = num_passed == len(test_results) and len(test_results) > 0
        else:
            test_results = []
            num_passed = 0
            passed_all = False

        return {
            "problem_name":     row.get("problem_name", ""),
            "problem_id":       row.get("problem_id", ""),
            "thinking":         thinking,
            "raw_answer":       answer,
            "extracted_code":   code,
            "test_results":     test_results,
            "passed_all_tests": passed_all,
            "correct":          passed_all,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "problem_name": row.get("problem_name", ""),
            "problem_id":   row.get("problem_id", ""),
        }
