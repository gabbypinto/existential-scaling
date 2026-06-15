import atexit
import os
import re
import time

from datasets import load_dataset

from benchmarks.base import Benchmark

try:
    from judge import LightCPVerifierJudge, ProblemNotFoundError, SupportedLanguage
    _JUDGE_AVAILABLE = True
except ImportError:
    _JUDGE_AVAILABLE = False


_DATASET = "QAQAQAQAQ/LiveCodeBench-Pro"


def extract_cpp_code(response: str) -> str | None:
    for pattern in [r"```cpp\s*(.*?)```", r"```c\+\+\s*(.*?)```"]:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    # Fallback: raw code block that looks like C++
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "#include" in code or "int main" in code:
            return code
    # Fallback: raw response starting with #include
    cleaned = response.strip()
    if re.match(r"^#include", cleaned):
        return cleaned
    return None


class LCBProBenchmark(Benchmark):
    def __init__(self):
        self._judge = None
        self._judge_ctx = None

    def load_problems(self, cfg: dict) -> list:
        difficulty = cfg.get("difficulty")
        splits = cfg.get("splits", ["biannual_2025_1_6"])
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if isinstance(splits, str):
            splits = [splits]

        if not _JUDGE_AVAILABLE:
            print("WARNING: lcb_pro_toolkit not found — judge unavailable, correct will always be False.")
            print("         Clone: git clone https://github.com/GavinZhengOI/LiveCodeBench-Pro lcb_pro_toolkit")
        else:
            self._judge_ctx = LightCPVerifierJudge(worker=4)
            self._judge = self._judge_ctx.__enter__()
            atexit.register(self._cleanup_judge)

        problems = []
        seen = set()
        for split in splits:
            ds = load_dataset(_DATASET, split=split, token=token, trust_remote_code=True)
            for row in ds:
                pid = row.get("problem_id", "")
                key = pid or row.get("problem_title", "")
                if key not in seen:
                    seen.add(key)
                    problems.append(dict(row))

        if difficulty:
            problems = [r for r in problems if r.get("difficulty", "").lower() == difficulty.lower()]
        return problems

    def _cleanup_judge(self):
        if self._judge_ctx is not None:
            try:
                self._judge_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._judge_ctx = None
            self._judge = None

    def get_question_text(self, row: dict) -> str:
        return row["problem_statement"]

    def get_label(self, row: dict) -> str:
        return f"{row.get('problem_title', row.get('problem_id', '?'))} ({row.get('platform', '')} / {row.get('difficulty', '')})"

    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        code = extract_cpp_code(answer)
        pid = row.get("problem_id", "")
        judge_result = "No Code"
        passed = False

        if code and self._judge and pid:
            try:
                sid = self._judge.submit(pid, SupportedLanguage.CPP, code)
                judge_result = "Judging"
                while judge_result == "Judging":
                    time.sleep(2)
                    judge_result = self._judge.get_result(sid)
                passed = judge_result == "Accepted"
            except ProblemNotFoundError:
                judge_result = "Not Found"
            except Exception as e:
                judge_result = f"Error: {e}"

        return {
            "problem_id":     pid,
            "problem_title":  row.get("problem_title", ""),
            "platform":       row.get("platform", ""),
            "difficulty":     row.get("difficulty", ""),
            "question":       row.get("problem_statement", ""),
            "thinking":       thinking,
            "raw_answer":     answer,
            "extracted_code": code,
            "judge_result":   judge_result,
            "correct":        passed,
        }

    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        return {
            "problem_id":    row.get("problem_id", ""),
            "problem_title": row.get("problem_title", ""),
            "platform":      row.get("platform", ""),
            "difficulty":    row.get("difficulty", ""),
        }
