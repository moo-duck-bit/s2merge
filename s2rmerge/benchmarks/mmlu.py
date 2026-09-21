"""MMLU: four-option multiple choice across 57 subjects."""

from __future__ import annotations

import csv
from typing import List, Optional

from s2rmerge.benchmarks.base import Benchmark, Record
from s2rmerge.benchmarks.io import dataset_path

_SHUFFLE_SEED = 888
_COLUMNS = ("question", "A", "B", "C", "D", "correct_answer")


class MMLU(Benchmark):
    name = "mmlu"
    domain = "mmlu"
    agent_type = "AnalyzeAgent"
    decision_method = "FinalRefer"
    rounds = 1
    merge_iterations = 2

    _SPLIT_DIRS = {"train": "val", "test": "test"}

    def load(self, split: str) -> List[Record]:
        import numpy as np

        directory = dataset_path("mmlu", self._SPLIT_DIRS[split])
        records: List[Record] = []
        for path in sorted(directory.glob("*.csv")):
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.reader(handle):
                    if len(row) < len(_COLUMNS):
                        continue
                    question, a, b, c, d, correct = row[: len(_COLUMNS)]
                    task = (
                        f"{question}\n"
                        f"Option A: {a}\n"
                        f"Option B: {b}\n"
                        f"Option C: {c}\n"
                        f"Option D: {d}\n"
                    )
                    records.append(
                        Record(task=task, answer=correct.strip(), meta={"subject": path.stem})
                    )

        # Fixed permutation so every method sees the subjects in the same order.
        order = np.random.default_rng(_SHUFFLE_SEED).permutation(len(records))
        return [records[i] for i in order]

    def is_correct(self, response: Optional[str], record: Record) -> bool:
        if not isinstance(response, str) or not response:
            return False
        # Agents are told to open their reply with the option letter.
        return response.strip()[0].upper() == record.answer.upper()
