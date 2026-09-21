"""HumanEval: Python function synthesis scored by the released unit tests."""

from __future__ import annotations

from typing import List, Optional

from s2rmerge.benchmarks.base import Benchmark, Record
from s2rmerge.benchmarks.io import dataset_path, read_jsonl

_TEST_TIMEOUT = 100


class HumanEval(Benchmark):
    name = "humaneval"
    domain = "humaneval"
    agent_type = "CodeWriting"
    decision_method = "FinalWriteCode"
    rounds = 2
    metric_name = "pass@1"

    has_train_split = False
    """HumanEval ships a single split; graph optimisation reuses the test set."""

    def load(self, split: str) -> List[Record]:
        rows = read_jsonl(dataset_path("humaneval", "humaneval-py.jsonl"))
        return [
            Record(
                task=row["prompt"],
                answer=row["test"],
                meta={"task_id": row["task_id"], "entry_point": row["entry_point"]},
            )
            for row in rows
        ]

    def is_correct(self, response: Optional[str], record: Record) -> bool:
        if not isinstance(response, str):
            return False

        from s2rmerge.mas.executor.python_executor import PyExecutor

        code = response.removeprefix("```python").strip().removesuffix("```").strip()
        try:
            passed, _, _ = PyExecutor().execute(code, [record.answer], timeout=_TEST_TIMEOUT)
        except Exception:
            return False
        return bool(passed)
