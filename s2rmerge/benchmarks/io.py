"""Reading the on-disk dataset files produced by ``scripts/download_datasets.py``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from s2rmerge.paths import DATA_DIR


def dataset_path(*parts: str) -> Path:
    """Path under the project's ``data/`` root, checked for existence."""
    path = DATA_DIR.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/download_datasets.py` first."
        )
    return path


def read_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
