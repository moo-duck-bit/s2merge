"""Fetch the six benchmarks from Hugging Face into ``data/``.

    python scripts/download_datasets.py            # everything
    python scripts/download_datasets.py --only gsm8k mmlu

Each dataset is skipped if its files are already present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s2rmerge.paths import DATA_DIR  # noqa: E402

CACHE_DIR = DATA_DIR / ".hf_cache"


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f"  {path.relative_to(DATA_DIR.parent)}: {len(rows)} rows")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"  {path.relative_to(DATA_DIR.parent)}: {count} rows")


def _load(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, cache_dir=str(CACHE_DIR), **kwargs)


def download_gsm8k() -> None:
    target = DATA_DIR / "gsm8k"
    if (target / "test.jsonl").exists():
        print("gsm8k: already present")
        return

    print("gsm8k: downloading")
    dataset = _load("openai/gsm8k", "main")
    for split, filename in (("train", "train.jsonl"), ("test", "test.jsonl")):
        _write_jsonl(
            target / filename,
            ({"question": row["question"], "answer": row["answer"]} for row in dataset[split]),
        )


def download_multiarith() -> None:
    target = DATA_DIR / "multiarith"
    if (target / "test.json").exists():
        print("multiarith: already present")
        return

    print("multiarith: downloading")
    dataset = _load("ChilleD/MultiArith")
    for split in ("train", "test"):
        _write_json(
            target / f"{split}.json",
            [
                {
                    "question": row["question"],
                    "answer": row.get("final_ans", row.get("answer", "")),
                }
                for row in dataset[split]
            ],
        )


def download_svamp() -> None:
    target = DATA_DIR / "svamp"
    if (target / "test.json").exists():
        print("svamp: already present")
        return

    print("svamp: downloading")
    dataset = _load("ChilleD/SVAMP")
    for split in ("train", "test"):
        _write_json(
            target / f"{split}.json",
            [
                {"Body": row["Body"], "Question": row["Question"], "Answer": row["Answer"]}
                for row in dataset[split]
            ],
        )


def download_aqua() -> None:
    target = DATA_DIR / "aqua"
    if (target / "test.json").exists():
        print("aqua: already present")
        return

    print("aqua: downloading")
    dataset = _load("deepmind/aqua_rat", "raw")
    for split, source in (("train", "train"), ("test", "test")):
        _write_json(
            target / f"{split}.json",
            [
                {
                    "question": row["question"],
                    "options": row["options"],
                    "rationale": row["rationale"],
                    "correct": row["correct"],
                }
                for row in dataset[source]
            ],
        )


def download_humaneval() -> None:
    target = DATA_DIR / "humaneval"
    if (target / "humaneval-py.jsonl").exists():
        print("humaneval: already present")
        return

    print("humaneval: downloading")
    dataset = _load("openai_humaneval")
    _write_jsonl(
        target / "humaneval-py.jsonl",
        (
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            for row in dataset["test"]
        ),
    )


def download_mmlu() -> None:
    target = DATA_DIR / "mmlu"
    if (target / "test").exists() and any((target / "test").iterdir()):
        print("mmlu: already present")
        return

    print("mmlu: downloading")
    dataset = _load("cais/mmlu", "all")
    letters = ("A", "B", "C", "D")

    # "validation" stands in for the training split; MMLU's own train split is
    # far larger than the handful of batches graph optimisation needs.
    for source, split in (("validation", "val"), ("test", "test")):
        directory = target / split
        directory.mkdir(parents=True, exist_ok=True)

        by_subject: Dict[str, List[List[str]]] = {}
        for row in dataset[source]:
            by_subject.setdefault(row["subject"], []).append(
                [row["question"], *row["choices"], letters[row["answer"]]]
            )

        for subject, rows in by_subject.items():
            with open(directory / f"{subject}.csv", "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
        print(f"  {split}: {len(by_subject)} subjects, {len(dataset[source])} questions")


DOWNLOADERS = {
    "gsm8k": download_gsm8k,
    "multiarith": download_multiarith,
    "svamp": download_svamp,
    "aqua": download_aqua,
    "humaneval": download_humaneval,
    "mmlu": download_mmlu,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(DOWNLOADERS), default=None)
    args = parser.parse_args()

    for name in args.only or DOWNLOADERS:
        DOWNLOADERS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
