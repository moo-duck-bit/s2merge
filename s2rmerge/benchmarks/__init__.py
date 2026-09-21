"""The six benchmarks reported in the paper."""

from __future__ import annotations

from typing import Dict, List, Type

from s2rmerge.benchmarks.aqua import AQuA
from s2rmerge.benchmarks.base import Benchmark, Record
from s2rmerge.benchmarks.humaneval import HumanEval
from s2rmerge.benchmarks.math_word import GSM8K, MultiArith, SVAMP
from s2rmerge.benchmarks.metrics import Accuracy
from s2rmerge.benchmarks.mmlu import MMLU

_BENCHMARKS: Dict[str, Type[Benchmark]] = {
    benchmark.name: benchmark
    for benchmark in (GSM8K, MultiArith, SVAMP, AQuA, MMLU, HumanEval)
}


def available() -> List[str]:
    return list(_BENCHMARKS)


def get_benchmark(name: str) -> Benchmark:
    try:
        return _BENCHMARKS[name]()
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; choose from {available()}") from None


__all__ = ["Accuracy", "Benchmark", "Record", "available", "get_benchmark"]
