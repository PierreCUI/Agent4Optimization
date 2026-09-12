"""Shared utilities — JSON parsing and safe file loading."""

import json
import re
from pathlib import Path


def parser_json(text: str) -> str:
    """Strip markdown code fences and leading prose from *text*.

    Models often reply ``"Here is the JSON:\n```json {...}```"``, wrap
    the payload in a `` ```text `` fence, or fence a narrative FIRST and
    put the JSON in a second `` ```json `` block — all previously broke
    ``json.loads`` and silently triggered fail-open fallbacks.  Rules:

    1. A `` ```json `` block is preferred, wherever it appears.
    2. Otherwise the first fenced block (any language) is used, whether
       the payload follows the marker line (`` ```text\n{...}``) or sits
       on it (`` ```{"a": 1}``` ``).
    3. Without a fence, trailing wrapper backticks are stripped and the
       string is returned (the caller's json.loads decides).

    A block is CLOSED only by a `` ``` `` that starts its own line — a
    ``` inside a JSON string value can never close the block.  Payloads
    are required to be non-empty, and empty fenced blocks are dropped
    (a stray `` ```lang\n``` `` with the real JSON on the next line must
    not shadow it, and its label must never become "the payload").
    """
    # CRLF: models stream \\r\\n and this repo's files are CRLF —
    # normalize once so every fence rule sees plain \\n.
    text = text.replace("\r\n", "\n").strip()
    # 0. Drop EMPTY fenced blocks (opener line, optional blank line,
    #    closer line) — and inline empty fences (```lang   ```).  The
    #    opener's label may only be word characters, so a REAL payload
    #    (```{"a": 1}``` or ```-1.5```) can never match.  An inline
    #    word that PARSES as JSON (```true```) is a payload, not a
    #    label — keep it.  The closer must END its line (lookahead): a
    #    close-fence immediately followed by the next block's open-fence
    #    is TWO blocks, not one empty block.
    text = re.sub(r"```[A-Za-z0-9_+-]*\n(?:[ \t]*\n)?```(?=\n|$)", "", text)

    def _keep_json_label(m) -> str:
        try:
            json.loads(m.group(1))
        except ValueError:
            return ""  # not JSON — an empty block with a language label
        return m.group(0)  # a JSON scalar payload on the marker line

    text = re.sub(r"```([A-Za-z0-9_+-]*)[ \t]*```(?=\n|$)", _keep_json_label, text)
    text = text.strip()

    def _clean_payload(s: str) -> str | None:
        """Non-empty payload, minus a trailing line-anchored closer."""
        s = re.sub(r"\n```[ \t]*$", "", s).strip()
        return s or None

    # 1. Prefer an explicit json fence (searching the whole text).
    #    1a. closer on its own line (trailing prose after it is fine)
    m = re.search(r"```json[^\n]*\n([\s\S]+?)\n```", text, re.IGNORECASE)
    p = _clean_payload(m.group(1)) if m else None
    if p:
        return p
    #    1b. unclosed at the end of the text
    m = re.search(r"```json[^\n]*\n([\s\S]+)$", text, re.IGNORECASE)
    if m:
        p = _clean_payload(m.group(1))
        if p:
            return p
    # 2. First fenced block whatever its language label.
    #    2a. label on the marker line, payload below: ```text\n{...}```
    m = re.search(r"```[^\n]*\n([\s\S]+?)\n```", text)
    p = _clean_payload(m.group(1)) if m else None
    if p:
        return p
    #    2b. unclosed at the end of the text
    m = re.search(r"```[^\n]*\n([\s\S]+)$", text)
    if m:
        p = _clean_payload(m.group(1))
        if p:
            return p
    #    2c. payload on the marker line: ```{"a": 1}``` — only accept a
    #        candidate that PARSES as JSON, so a bare language label can
    #        never become "the payload".
    m = re.search(r"```[^\n]*?([\s\S]+?)```", text)
    if m:
        cand = m.group(1).strip()
        try:
            json.loads(cand)
        except ValueError:
            pass
        else:
            return cand
    # 3. No fence: plain JSON (possibly trailing backticks from wrappers).
    return re.sub(r"\s*(?:`{1,3})\s*$", "", text).strip()


def code_masked_positions(text: str) -> set[int]:
    """Character positions inside comments or string literals of *text*.

    Quote-aware (backslash escapes and triple-quoted strings handled),
    so position-sensitive scans (finditer over code shapes) can skip
    text that is not real code.  Every call site that needs
    ``setup_optimization``/``Lagrangian`` spans uses this to reject
    comment/docstring mentions.
    """
    masked = set()
    i, n = 0, len(text)
    quote = None
    while i < n:
        ch = text[i]
        if quote:
            masked.add(i)
            if ch == "\\" and len(quote) == 1:
                if i + 1 < n:
                    masked.add(i + 1)
                    i += 2
                else:
                    i += 1
                continue
            if text.startswith(quote, i):
                masked.update(range(i, min(i + len(quote), n)))
                i += len(quote)
                quote = None
                continue
            i += 1
        elif ch in ("'", '"'):
            quote = next((q for q in ('"""', "'''", '"', "'") if text.startswith(q, i)), ch)
            masked.add(i)
            i += 1
        elif ch == "#":
            while i < n and text[i] != "\n":
                masked.add(i)
                i += 1
        else:
            i += 1
    return masked


def loader_safe_file(file_path: str | Path) -> str:
    """Read *file_path* safely, returning its content or an empty string."""
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return ""
