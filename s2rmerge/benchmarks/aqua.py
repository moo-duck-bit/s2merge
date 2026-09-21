"""AQuA-RAT: algebraic word problems with five lettered options."""

from __future__ import annotations

from typing import List, Optional

from s2rmerge.answers import extract_choice
from s2rmerge.benchmarks.base import Benchmark, Record
from s2rmerge.benchmarks.io import dataset_path, read_json


class AQuA(Benchmark):
    name = "aqua"
    domain = "aqua"
    agent_type = "MathSolver"
    decision_method = "FinalRefer"
    rounds = 2

    _FILES = {"train": "train.json", "test": "test.json"}

    def load(self, split: str) -> List[Record]:
        rows = read_json(dataset_path("aqua", self._FILES[split]))
        records = []
        for row in rows:
            options = " ".join(row["options"])
            records.append(
                Record(
                    task=f"{row['question']} Choices: {options}",
                    answer=row["correct"],
                    meta={"rationale": row.get("rationale", "")},
                )
            )
        return records

    def is_correct(self, response: Optional[str], record: Record) -> bool:
        return extract_choice(response) == record.answer
