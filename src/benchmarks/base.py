from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def load_problems(self, cfg: dict) -> list:
        """Return list of problem dicts from the dataset."""

    @abstractmethod
    def get_question_text(self, row: dict) -> str:
        """Return the raw problem text to be wrapped by pre/post prompt."""

    @abstractmethod
    def get_label(self, row: dict) -> str:
        """Return a short display label for logging progress (e.g. question title)."""

    @abstractmethod
    def build_result(self, row: dict, thinking: str, answer: str, metrics: dict, elapsed: float) -> dict:
        """Return a result dict for one problem/round. Must include 'correct' or 'passed_all_tests' bool."""

    @abstractmethod
    def build_summary_entry(self, row: dict, passing_rounds: list) -> dict:
        """Return per-question summary fields for the final summary.json."""
