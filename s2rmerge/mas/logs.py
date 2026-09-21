"""Transcript of what each agent said, written alongside a run's results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

_SEPARATOR = "=" * 80
_DIVIDER = "-" * 80


def open_transcript(path: Path, title: str) -> Path:
    """Start a fresh transcript file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{_SEPARATOR}\n{title}\n{_SEPARATOR}\n\n")
    return path


def log_agent_turn(
    path: Optional[Path],
    agent: str,
    round_index: int,
    task: str,
    output: Any,
) -> None:
    """Append one agent turn. Does nothing when transcripts are switched off."""
    if path is None:
        return

    if isinstance(output, list):
        rendered = "\n".join("None" if item is None else str(item) for item in output)
    else:
        rendered = "None" if output is None else str(output)

    entry = (
        f"\n{_SEPARATOR}\n"
        f"Agent: {agent} | Round: {round_index}\n"
        f"{_DIVIDER}\n"
        f"Task:\n{task}\n"
        f"{_DIVIDER}\n"
        f"Output:\n{rendered}\n"
        f"{_SEPARATOR}\n"
    )

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError as error:
        # A transcript is a convenience; losing it must not abort a long run.
        print(f"could not write transcript to {path}: {error}")


def log_note(path: Optional[Path], text: str) -> None:
    """Append a free-form note, such as the router's decision for a query."""
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"\n{_SEPARATOR}\n{text}\n{_SEPARATOR}\n")
    except OSError as error:
        print(f"could not write transcript to {path}: {error}")
