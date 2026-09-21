"""Filesystem locations used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
PROTOTYPE_DIR = PROJECT_ROOT / "prototypes"
RESULT_DIR = PROJECT_ROOT / "results"
