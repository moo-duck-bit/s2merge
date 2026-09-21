"""Running evaluation metric."""

from __future__ import annotations


class Accuracy:
    """Fraction of answered items that were correct."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, is_correct: bool) -> None:
        self.correct += int(bool(is_correct))
        self.total += 1

    @property
    def value(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.value * 100:.2f}% ({self.correct}/{self.total})"
