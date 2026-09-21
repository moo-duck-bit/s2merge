"""Math word-problem benchmarks: GSM8K, MultiArith and SVAMP.

All three ask for a single number and share the GSM8K role set, so they differ
only in how their files are laid out on disk.
"""

from __future__ import annotations

from typing import List, Optional

from s2rmerge.answers import extract_numeric
from s2rmerge.benchmarks.base import Benchmark, Record
from s2rmerge.benchmarks.io import dataset_path, read_json, read_jsonl


class _NumericAnswerBenchmark(Benchmark):
    domain = "gsm8k"
    agent_type = "MathSolver"
    decision_method = "FinalRefer"
    rounds = 2

    def is_correct(self, response: Optional[str], record: Record) -> bool:
        predicted = extract_numeric(response)
        if predicted is None:
            return False
        try:
            return float(predicted) == float(record.answer)
        except (TypeError, ValueError):
            return str(predicted).strip() == str(record.answer).strip()


class GSM8K(_NumericAnswerBenchmark):
    name = "gsm8k"

    _FILES = {"train": "train.jsonl", "test": "test.jsonl"}

    def load(self, split: str) -> List[Record]:
        rows = read_jsonl(dataset_path("gsm8k", self._FILES[split]))
        records = []
        for row in rows:
            # GSM8K answers are "<worked solution>\n#### <number>".
            answer = row["answer"].split("\n####")[-1].replace(",", "").strip()
            records.append(Record(task=row["question"], answer=answer))
        return records


class MultiArith(_NumericAnswerBenchmark):
    name = "multiarith"
    merge_iterations = 2

    _FILES = {"train": "train.json", "test": "test.json"}

    def load(self, split: str) -> List[Record]:
        rows = read_json(dataset_path("multiarith", self._FILES[split]))
        return [
            Record(task=row["question"], answer=str(row["answer"]))
            for row in rows
        ]


class SVAMP(_NumericAnswerBenchmark):
    name = "svamp"
    merge_iterations = 2

    _FILES = {"train": "train.json", "test": "test.json"}

    def load(self, split: str) -> List[Record]:
        rows = read_json(dataset_path("svamp", self._FILES[split]))
        return [
            Record(task=f"{row['Body']} {row['Question']}", answer=str(row["Answer"]))
            for row in rows
        ]
