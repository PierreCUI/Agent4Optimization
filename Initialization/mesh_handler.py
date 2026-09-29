"""
Mesh-spec side of the Initialization phase (item 11).

The extraction/design work itself now lives in
``Initialization.spec_generation`` (the generate -> review -> repair
loop producing BOTH spec JSONs together).  This module keeps the
heuristic mesh-content detection and the ``handle_mesh`` adapter that
returns the mesh half of the loop's output (``{"extracted", "designed"}``)
for callers expecting the classic interface.
"""

import re
from typing import Optional

from Initialization.spec_generation import run_spec_generation

# ======================================================================
# detection — does any input file contain mesh-related content?
# ======================================================================

# Heuristic patterns that suggest a file contains FreeFEM mesh *code*.
_MESH_CODE_PATTERNS = [
    r"\bborder\s+\w+\s*\(",  # border C01(t=0,1){...}
    r"\bbuildmesh\s*\(",  # buildmesh(...)
    r"\bsquare\s*\(",  # square(n,m) or square(n,m,[...])
    r"\bmesh\s+Th\s*=\s*",  # mesh Th = ...
    r"\bmovemesh\b",  # movemesh
    r"\badaptmesh\b",  # adaptmesh
    r"\bsavemesh\b",  # savemesh
]

# Heuristic patterns that suggest a *geometric description* (not code).
# Cartesian products appear as "[a,b]x[c,d]", "×", "\times" (LaTeX),
# or the word "times" — all must match so an explicit domain rectangle
# is never misclassified as "absent".
_GEOMETRY_DESCRIPTION_PATTERNS = [
    r"(?:rectangle|rectangular)\s+(?:domain|region).*?(?:\d)",
    r"(?:square|circular|annular|L-shaped)\s+(?:domain|region)",
    r"domain\s+(?:is|:).*?(?:rectangle|square|circle|disc)",
    r"\[-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\]\s*(?:x|×|\\times|times)\s*\[-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\]",  # noqa: E501
]


def _scan_for_mesh_code(text: str) -> bool:
    """Return True if *text* looks like it contains FreeFEM mesh code."""
    for pat in _MESH_CODE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _scan_for_geometry_description(text: str) -> bool:
    """Return True if *text* contains a geometric description (not code)."""
    for pat in _GEOMETRY_DESCRIPTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def detect_mesh_source(input_files: dict) -> dict:
    """
    Scan all input files and classify what mesh information is available.

    Returns
    -------
    dict with keys:
      category:   "code" | "description" | "absent"
      files:      list of filenames that contain mesh content
    """
    code_files = []
    desc_files = []

    for fname, content in input_files.items():
        if _scan_for_mesh_code(content):
            code_files.append(fname)
        elif _scan_for_geometry_description(content):
            desc_files.append(fname)

    if code_files:
        return {"category": "code", "files": code_files}
    elif desc_files:
        return {"category": "description", "files": desc_files}
    else:
        return {"category": "absent", "files": []}


# ======================================================================
# adapter — mesh half of the one-loop generation
# ======================================================================


def handle_mesh(agent, input_files: dict, problem_context: Optional[dict] = None) -> dict:
    """Run the spec-generation loop and return the mesh half.

    Returns ``{"extracted": ..., "designed": ...}``.  *problem_context*
    is accepted for signature compatibility but unused — the loop
    generates the problem and mesh specs TOGETHER.  When the loop exceeds
    its repair budget, the returned state carries ``_passed: False`` (its
    designed parts may still be non-empty) — the caller's guard must
    stop the pipeline.
    """
    if not input_files:
        return {}
    return run_spec_generation(agent, input_files)["mesh"]
