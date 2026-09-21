"""Parsing final answers out of free-form agent responses.

Agents are instructed to end with ``The answer is <x>``, but they do not always
comply, so extraction falls back to the last boxed expression and then to the
last numeral or option letter in the text. The LaTeX normalisation below follows
the convention used by the MATH/GSM8K evaluation harnesses.

This module deliberately has no project imports: both the prompt sets and the
benchmark loaders depend on it.
"""

from __future__ import annotations

import re
from typing import Optional

ANSWER_MARKERS = ("The answer is ", "the answer is ")
_NUMBER_RE = re.compile(r"-?\d*\.?\d+")
_DIGITS_RE = re.compile(r"\d+")
_UPPERCASE_RE = re.compile(r"[A-Z]")


def _fix_sqrt(text: str) -> str:
    """Rewrite ``\\sqrt3`` as ``\\sqrt{3}``."""
    if "\\sqrt" not in text:
        return text
    head, *tails = text.split("\\sqrt")
    out = head
    for tail in tails:
        if tail and tail[0] != "{":
            out += "\\sqrt{" + tail[0] + "}" + tail[1:]
        else:
            out += "\\sqrt" + tail
    return out


def _fix_fracs(text: str) -> str:
    """Rewrite ``\\frac12`` as ``\\frac{1}{2}``."""
    head, *tails = text.split("\\frac")
    out = head
    for tail in tails:
        out += "\\frac"
        if tail.startswith("{"):
            out += tail
        elif len(tail) >= 2:
            numerator, denominator, rest = tail[0], tail[1], tail[2:]
            if denominator == "{":
                out += "{" + numerator + "}" + denominator + rest
            else:
                out += "{" + numerator + "}{" + denominator + "}" + rest
        else:
            return text
    return out


def _fix_a_slash_b(text: str) -> str:
    """Rewrite ``3/4`` as ``\\frac{3}{4}`` when both sides are integers."""
    parts = text.split("/")
    if len(parts) != 2:
        return text
    try:
        numerator, denominator = int(parts[0]), int(parts[1])
    except ValueError:
        return text
    if text != f"{numerator}/{denominator}":
        return text
    return "\\frac{" + str(numerator) + "}{" + str(denominator) + "}"


def _drop_trailing_units(text: str) -> str:
    """Drop a trailing ``\\text{ ...}`` unit annotation."""
    if "\\text{ " not in text:
        return text
    return text.split("\\text{ ")[0]


def normalize_latex(text: str) -> str:
    """Canonicalise LaTeX-flavoured answer strings so they compare equal."""
    text = text.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    text = text.replace("tfrac", "frac").replace("dfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "")
    text = _drop_trailing_units(text)
    text = text.replace("\\%", "").replace("%", "")
    text = text.replace(" .", " 0.").replace("{.", "{0.")

    if not text:
        return text
    if text[0] == ".":
        text = "0" + text

    # Strip a short variable assignment such as "x = 5".
    sides = text.split("=")
    if len(sides) == 2 and len(sides[0]) <= 2:
        text = sides[1]

    text = _fix_sqrt(text).replace(" ", "")
    text = _fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    return _fix_a_slash_b(text)


def _last_boxed(text: str) -> Optional[str]:
    """Contents of the last ``\\boxed{...}`` group, if there is one."""
    if "boxed" not in text:
        return None
    tail = text.split("boxed")[-1]
    if not tail:
        return None
    if tail[0] != "{":
        return tail.split("$")[0].strip()

    depth, out = 1, ""
    for char in tail[1:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        out += char
    return out


def _answer_span(text: str) -> Optional[str]:
    """Text following the last ``The answer is`` marker."""
    for marker in ANSWER_MARKERS:
        if marker in text:
            return text.split(marker)[-1].strip()
    return None


def _candidate(text: str) -> str:
    """Narrow a response down to the span most likely to hold the answer."""
    span = _answer_span(text)
    if span is None:
        span = _last_boxed(text)
    if span is None:
        return text

    span = span.rstrip("./")
    span = normalize_latex(span)

    boxed = _last_boxed(span)
    return normalize_latex(boxed) if boxed is not None else span


def extract_numeric(response: Optional[str]) -> Optional[str]:
    """Final numeric answer, as a digit string. ``None`` if the response is not text."""
    if not isinstance(response, str):
        return None

    candidate = _candidate(response)
    if candidate.isdigit():
        return candidate

    digits = _DIGITS_RE.findall(candidate)
    if digits:
        return digits[-1]

    numbers = _NUMBER_RE.findall(response)
    return numbers[-1] if numbers else "0"


def extract_choice(response: Optional[str]) -> Optional[str]:
    """Final multiple-choice letter. ``'None.'`` if no letter can be found."""
    if not isinstance(response, str):
        return None

    candidate = _candidate(response)
    if len(candidate) == 1 and candidate.isupper():
        return candidate

    letters = _UPPERCASE_RE.findall(candidate)
    return letters[-1] if letters else "None."
