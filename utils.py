"""Shared utilities — JSON parsing and safe file loading."""

import re
from pathlib import Path


def parser_json(text: str) -> str:
    """Strip markdown code fences from *text*, returning clean JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json|python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def loader_safe_file(file_path: str | Path) -> str:
    """Read *file_path* safely, returning its content or an empty string."""
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, Exception):
        return ""
