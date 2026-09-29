"""
Problem-spec side of the Initialization phase.

The extraction/design work itself now lives in
``Initialization.spec_generation`` (the generate -> review -> repair
loop producing BOTH spec JSONs together).  This module keeps the input
collection and the ``parse_problem`` adapter that returns the problem
half of the loop's output (``{"extracted", "designed"}``) for callers
expecting the classic interface.
"""

from pathlib import Path
from typing import Optional

from utils import loader_safe_file
from Initialization.spec_generation import run_spec_generation

# ======================================================================
# helpers
# ======================================================================


# Artifacts produced by a previous pipeline/tool run on the same folder.
# They are NOT problem inputs — feeding them back to the LLM would let a
# previous generation supervise the next one.  ``mesh.txt`` is DELIBERATELY
# absent: a demo's mesh.txt is an INPUT (the reference mesh the problem is
# defined on) and is collected — the generate prompt is instructed to keep
# the active mesh verbatim (source "extracted") when its syntax is valid.
_ARTIFACT_NAMES = {
    "initial.json",
    "mesh.json",
    "run_result.json",
    "reflection.txt",
    "main.py",
    "brief.txt",  # human-readable summary only — never an LLM input
}


def collect_input_files(problem_dir: Path) -> dict:
    """Return ``{filename: content}`` for every **input** file in *problem_dir*.

    Excluded: ``.py`` files, ``brief.txt`` (human-readable summary only),
    and pipeline artifacts written by earlier runs (initial.json, mesh.json,
    run_result.json, reflection.txt).  ``mesh.txt`` is collected — the
    problem's reference mesh.
    """
    problem_dir = Path(problem_dir)
    if not problem_dir.exists() or not problem_dir.is_dir():
        return {}
    files = {}
    for p in sorted(problem_dir.rglob("*")):
        if not p.is_file() or p.suffix == ".py":
            continue
        if p.name in _ARTIFACT_NAMES:
            continue
        content = loader_safe_file(p)
        if content:
            files[str(p.relative_to(problem_dir))] = content
    return files


# ======================================================================
# adapter — problem half of the one-loop generation
# ======================================================================


def parse_problem(agent, input_files: dict, problem_name: Optional[str] = None) -> dict:
    """Run the spec-generation loop and return the problem half.

    Returns ``{"extracted": ..., "designed": ...}``.  When the loop
    exceeds its repair budget, the returned state carries ``_passed:
    False`` (its designed parts may still be non-empty) — the caller's
    guard must stop the pipeline.
    """
    if not input_files:
        return {}
    return run_spec_generation(agent, input_files, problem_name)["initial"]
