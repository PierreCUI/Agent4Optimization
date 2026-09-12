"""
Mesh Handling — detect, validate, repair, or design mesh code.

Mesh data can arrive in several forms:
  (a) FreeFEM border / buildmesh / square / cube code in a file.
  (b) A geometric description embedded in the LaTeX problem text.
  (c) Implicit — nothing explicitly stated; must be designed from context.

This module:
  1. Scans all input files for mesh-related content.
  2. Classifies what was found: "code", "description", or "absent".
  3. Phase A — faithfully extracts whatever was found (immutable).
  4. Phase B — produces valid, runnable FreeFEM mesh code.
     - code given & valid      → copy as-is, mark "extracted".
     - code given & has errors → repair syntax only, mark "extracted_repaired".
     - description given       → generate code following the description,
                                 mark "designed_from_description".
     - nothing given           → design from problem context,
                                 mark "designed".
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import MAX_FILE_CHARS  # noqa: E402
from utils import parser_json  # noqa: E402

# ======================================================================
# detection — does any input file contain mesh-related content?
# ======================================================================

# Heuristic patterns that suggest a file contains FreeFEM mesh *code*.
_MESH_CODE_PATTERNS = [
    r"\bborder\s+\w+\s*\(",  # border C01(t=0,1){...}
    r"\bbuildmesh\s*\(",  # buildmesh(...)
    r"\bsquare\s*\(",  # square(n,m) or square(n,m,[...])
    r"\bcube\s*\(",  # cube(n,m,l)
    r"\bmesh\s+Th\s*=\s*",  # mesh Th = ...
    r"\bmesh3\s+Th\s*=\s*",  # mesh3 Th = ...
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
    r"\[-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\]\s*(?:x|×|\\times|times)\s*\[-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\]",
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
# Phase A — faithful extraction (immutable)
# ======================================================================

_EXTRACT_SYSTEM = """\
    You are a precise FreeFEM mesh code transcriber.  You read input files and
    extract ONLY the mesh-related content exactly as given.
    
    ## Your task
    
    1. Identify any FreeFEM mesh code (border, buildmesh, square, cube, etc.)
       present in the input files.
    
    2. Return a JSON object with keys:
       category: "code" | "description" | "absent"
       mesh_code: the complete mesh code fragment, verbatim (or "")
       geometry_description: any geometric description found in text (or "")
       labels_found: array of label strings e.g. ["1","2"]
       has_syntax_errors: true or false
       syntax_error_details: description of error (or "")
    
    ## Rules
    - If FreeFEM border/buildmesh/square/cube code is present, copy it VERBATIM
      into mesh_code.  Do NOT modify, prettify, or "fix" anything.
    - Extract ALL numeric/string labels that appear in border label=... statements
      or in buildmesh/square arguments.
    - If the mesh is only described geometrically (e.g. "a rectangle [0,2]x[0,1]"),
      set category="description" and put the description text in geometry_description.
    - If nothing mesh-related is found, set category="absent".
    - Return ONLY the JSON object.  No markdown, no code fences.
"""

_EXTRACT_USER = """\
    Input files:
    {files_block}
    
    Return the mesh extraction JSON.
"""


def extract_mesh(agent, input_files: dict) -> dict:
    """Phase A — faithfully extract mesh content from input files."""
    # Quick heuristic pre-scan
    detected = detect_mesh_source(input_files)

    if detected["category"] == "absent":
        return {
            "category": "absent",
            "mesh_code": "",
            "geometry_description": "",
            "labels_found": [],
            "has_syntax_errors": False,
            "syntax_error_details": "",
        }

    files_block = ""
    for fname, content in input_files.items():
        max_len = MAX_FILE_CHARS
        truncated = (
            content if len(content) <= max_len else content[:max_len] + "\n\n... [truncated]"
        )
        files_block += f"=== FILE: {fname} ===\n{truncated}\n\n"

    prompt_list = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": _EXTRACT_USER},
    ]
    agent.set_prompt_list(prompt_list)
    raw = agent.invoke({"files_block": files_block})

    try:
        return json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"extract_mesh JSON parse failed: {e}")
        return {"_raw": raw, "_error": str(e)}


# ======================================================================
# Phase B — validate, repair, or design mesh code (mutable)
# ======================================================================

_DESIGN_SYSTEM = """\
    You are a FreeFEM mesh expert.  You receive an extraction result describing
    what mesh information was found in the problem input.  Your job is to produce
    valid, runnable FreeFEM mesh code.
    
    ## Cases
    
    ### Case 1: category = "code" (mesh code was explicitly provided)
    
    - If has_syntax_errors is false → copy mesh_code verbatim.
      Set source="extracted".  The code is immutable ground truth.
    
    - If has_syntax_errors is true → repair ONLY the syntax errors.
      Do NOT change geometry, labels, or structure.  Fix things like:
      missing semicolons, unmatched parentheses, wrong parameter names.
      Set source="extracted_repaired" and add a rationale describing the fix.
      The ORIGINAL geometry/labels are still immutable.
    
    ### Case 2: category = "description" (only a geometric description exists)
    
    - Generate FreeFEM mesh code that faithfully follows the description.
    - If the description includes specific dimensions → use them exactly.
    - If labels are mentioned → use them exactly.
    - Set source="designed_from_description" and cite the description.
    
    ### Case 3: category = "absent" (nothing about the mesh was stated)
    
    - DESIGN a reasonable mesh from the problem context.
    - Use your knowledge of continuum mechanics:
      * Cantilever beam → rectangle [0,2]x[0,1], left edge clamped (label 4),
        right edge loaded (label 2).
      * Bridge → long rectangle [0,4]x[0,1], both ends clamped (labels 2,4),
        load on top/bottom (label 1 or 3).
      * Thermal fin → rectangle with appropriate aspect ratio.
      * Fluid-structure → channel with obstacle.
    - Set source="designed" with rationale.
    
    ## Output format

    Return a JSON object with these keys:
      mesh_code: complete runnable FreeFEM mesh code string
      mesh_code_source: one of "extracted", "extracted_repaired", "designed_from_description", "designed"
      mesh_code_rationale: explanation if not purely extracted, else ""
      labels_used: array of label strings
      dimension: 2 or 3
      domain_description: brief English description of the geometry
    
    ## FreeFEM mesh code requirements
    
    - Use `border C01(t=0, L){{x=...; y=...; label=N;}}` syntax for 2D.
    - Use `mesh Th = buildmesh(C01(n) + C02(n) + ...);` to finalise.
    - Use `square(n, m, [x,y])` or `cube(n, m, l)` only if appropriate.
    - Use `mesh3 Th = ...` for 3D.
    - Parameterise each edge in the counter-clockwise direction.
    - The mesh resolution parameter `n` should be defined: `int n = 50;`.
    - Every border MUST have a label.
    - Labels MUST be integer literals, not variables (unless the original code
      already uses variables — then preserve them).
    - The label-10 boundary (material boundary — holes / the solid-void
      boundary of the level set inside the domain) MUST use label 10:
      the material boundary label is ALWAYS 10 (a FreeFEM requirement).
      If the problem needs a pre-cut hole, assign its border `label=10`
      by hand.  (Holes carved by the level-set function during
      optimisation are labelled 10 automatically and need no mesh-side
      handling.)  The label-10 boundary is NOT a non-10 boundary — it
      must not be confused with the labelled non-10 edges (label 1..4
      etc.).
    - Multi-domain meshes (problems with two or more computational
      domains): interface edges — the shared boundaries BETWEEN two
      computational domains — are non-10 boundaries and need their own
      mesh labels, distinct from the other non-10 boundary labels and
      from label 10.  Add them only when the problem actually has
      multiple computational domains; single-domain problems have no
      interfaces.
    - The final line must produce a mesh variable named `Th`.
    
    ## Output rules
    - Return ONLY the JSON object.  No markdown, no code fences.
    - All strings in English.
"""

_DESIGN_USER = """\
    Extraction result:
    {extracted_json}
    
    Problem context (from problem_parser output, if available):
    {problem_context}
    
    Return the mesh design JSON.
"""


def design_mesh(agent, extracted: dict, problem_context: Optional[dict] = None) -> dict:
    """Phase B — produce valid, runnable FreeFEM mesh code."""
    extracted_json = json.dumps(extracted, indent=2, ensure_ascii=False)
    ctx_json = json.dumps(problem_context or {}, indent=2, ensure_ascii=False)

    prompt_list = [
        {"role": "system", "content": _DESIGN_SYSTEM},
        {"role": "user", "content": _DESIGN_USER},
    ]
    agent.set_prompt_list(prompt_list)
    raw = agent.invoke({"extracted_json": extracted_json, "problem_context": ctx_json})

    try:
        return json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"design_mesh JSON parse failed: {e}")
        return {"_raw": raw, "_error": str(e)}


# ======================================================================
# orchestrator
# ======================================================================


def handle_mesh(agent, input_files: dict, problem_context: Optional[dict] = None) -> dict:
    """Run Phase A + Phase B, return {"extracted": ..., "designed": ...}."""
    if not input_files:
        warnings.warn("handle_mesh: empty input_files")
        return {}

    extracted = extract_mesh(agent, input_files)
    if "_error" in extracted:
        return {"_error": f"Phase A failed: {extracted['_error']}"}

    designed = design_mesh(agent, extracted, problem_context)
    if "_error" in designed:
        return {"_error": f"Phase B failed: {designed['_error']}", "extracted": extracted}

    return {"extracted": extracted, "designed": designed}


# ======================================================================
# Local test — 3 diverse cases
# ======================================================================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from API import DeepSeekCodeAgentV4Flash
    from Initialization import collect_input_files

    agent = DeepSeekCodeAgentV4Flash()
    demos_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "demos"

    test_cases = sorted([d.name for d in demos_dir.iterdir() if d.is_dir()])

    print("=" * 60)
    print(f"Mesh Handler — {len(test_cases)} demos")
    print("=" * 60)
    summary_rows = []

    for case_name in test_cases:
        case_dir = demos_dir / case_name
        if not case_dir.exists():
            print(f"\n[SKIP] {case_name}")
            continue

        input_files = collect_input_files(case_dir)

        result = handle_mesh(agent, input_files)

        # Save raw result to results/<name>/mesh.json
        out_dir = Path("results") / case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "mesh.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if "_error" in result:
            print(f"  [{case_name}] ERROR: {result['_error']}")
            summary_rows.append((case_name, "ERROR", "", "", "", "", ""))
            continue

        ex = result.get("extracted", {})
        ds = result.get("designed", {})

        cat = ex.get("category", "?")
        src = ds.get("mesh_code_source", "?")
        dim = ds.get("dimension", "?")
        # `or []` guards None from a malformed LLM response — iterating
        # None would abort the demo runner.
        labels = ", ".join(str(l) for l in (ds.get("labels_used") or []))
        domain = ds.get("domain_description", "")[:60]
        syntax_ok = "OK" if not ex.get("has_syntax_errors") else "FIXED"

        summary_rows.append((case_name, cat, src, dim, labels, domain, syntax_ok))
        print(
            f"  [{case_name}] cat={cat}, src={src}, dim={dim}D, "
            f'labels=[{labels}], syntax={syntax_ok}, domain="{domain}"'
        )

    # ── summary table ──
    print(f"\n{'=' * 100}")
    print("SUMMARY TABLE — All Demos")
    print(f"{'=' * 100}")
    print(
        f"| {'Problem':<42} | {'Cat':<12} | {'Source':<22} | {'Dim':<4} | "
        f"{'Labels':<18} | {'Syntax':<6} | {'Domain':<30} |"
    )
    print(f"|{'-' * 42}|{'-' * 12}|{'-' * 22}|{'-' * 4}|{'-' * 18}|{'-' * 6}|{'-' * 30}|")
    for name, cat, src, dim, labels, domain, syntax_ok in summary_rows:
        print(
            f"| {name:<42} | {cat:<12} | {src:<22} | {str(dim):<4} | {labels:<18} | {syntax_ok:<6} | {domain:<30} |"
        )

    print(f"\n{'=' * 60}")
    print("Test complete.")
