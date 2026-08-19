"""
Structured Problem Analysis — two-phase: extract then design.

Phase A — ``extract_problem`` : faithfully transcribe only what is explicitly
stated in the input files.  Nothing is invented.  The result is **immutable**
(ground truth from the problem statement).

Phase B — ``design_problem`` : take the extracted spec and fill every missing
piece with physically reasonable values using domain expertise.  The result is
**mutable** (may change during later experiments or the reflection loop).

``parse_problem`` orchestrates both and returns {"extracted": ..., "designed": ...}.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import MAX_FILE_CHARS
from utils import loader_safe_file, parser_json

# ======================================================================
# helpers
# ======================================================================


def collect_input_files(problem_dir: Path) -> dict:
    """Return ``{filename: content}`` for every **non-.py** file in *problem_dir*."""
    problem_dir = Path(problem_dir)
    if not problem_dir.exists() or not problem_dir.is_dir():
        return {}
    files = {}
    for p in sorted(problem_dir.rglob("*")):
        if not p.is_file() or p.suffix == ".py":
            continue
        content = loader_safe_file(p)
        if content:
            files[str(p.relative_to(problem_dir))] = content
    return files


# ======================================================================
# Phase A — faithful extraction (immutable)
# ======================================================================

_EXTRACT_SYSTEM = """\
    You are a precise mathematical transcription engine.  Read the input files
    (LaTeX .tex, mesh files, etc.) and return a JSON object containing ONLY what
    is explicitly stated.  **Do not invent, guess, or fill gaps.**
    
    This data becomes immutable ground truth — downstream code may NOT change it.
    
    ## What to extract
    
    Return a flat JSON object.  Omit any key whose value is not explicitly found.
    
    ### Top-level scalars

    - problem_type: "shape_optimization" or "topology_optimization" (REQUIRED)
    This is MANDATORY — you MUST return exactly one.  Do NOT omit.
    - problem_name: short human-readable name
    - dimension: 2 or 3
    - domain_name: e.g. "D", "Omega"
    - mesh_variable: e.g. "Th"
    - mesh_file: the filename of the mesh file (e.g. "mesh.txt")
    
    ### mesh_labels_used
    Array of all label strings that appear in the mesh file.
    Extract from border/buildmesh commands.  Example: ["1","2","3","4"].
    
    ### boundaries (LOGICAL boundaries from the LaTeX, NOT raw mesh segments)
    Each boundary:
      name:         LaTeX name (e.g. "\\Gamma_N", "\\Gamma_4")
      kind:         "Dirichlet" or "Neumann"
      description:  short English phrase (e.g. "loaded boundary")
      mesh_label:   string — ONLY if the LaTeX/text explicitly states which
                    mesh label this boundary corresponds to.  Otherwise OMIT.
    
    CRITICAL: extract LOGICAL boundaries like Gamma_N, Gamma_D from the LaTeX
    mathematics, NOT the C01/C02/... geometric segments from mesh.txt.
    The mesh segments are implementation details — ignore them.
    
    ### fields (state/primal unknowns)
    Each: name, latex, type ("scalar"|"vector"), fe_space ("P1"|"P2"),
          unit (SI string), dirichlet_value, dirichlet_labels [string]
    
    ### test_functions
    Each: name, latex, type, fe_space
    
    ### constants
    Each: name, latex, value (number), description, unit.
    ONLY include constants whose numerical VALUE is explicitly given
    (in a parameter table or inline).  Omit if only the symbol is named.
    
    ### expressions
    Each: name, latex, formula (mathematical notation).
    ONLY include those with an explicit formula written in the text.
    
    ### characteristic_function
    Object: latex, strong_value, weak_value.  Omit if not present.
    
    ### objective
    name, description, terms [string]
    
    ### state_equation
    description, weak_form_lhs, weak_form_rhs, is_nonlinear (bool)
    
    ### newton_linearisation (if explicitly given)
    description, weak_form
    
    ### adjoint_equation (if explicitly given)
    description, weak_form_lhs, weak_form_rhs
    
    ### shape_derivative (if explicitly given)
    volume_expression, boundary_expression
    
    ### optimization_params (if explicitly given)
    Any of: hmin, hmax, diffusion, v0, iterations, level_set_initial,
    boundary_labels, fixed_labels
    
    ## Rules
    - Omit keys with null/empty/unknown values.  The designer fills gaps later.
    - All string values in English.
    - Return ONLY the JSON object.  No markdown, no code fences.
"""

_EXTRACT_USER = """\
    Problem: {problem_name}
    
    Input files:
    {files_block}
    
    Return the extraction JSON (omit anything not explicitly stated).
"""


def extract_problem(agent, input_files: dict, problem_name: Optional[str] = None) -> dict:
    """Phase A — faithfully transcribe what the input files explicitly state."""
    if not input_files:
        warnings.warn("extract_problem: empty input_files")
        return {}

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
    raw = agent.invoke({"problem_name": problem_name or "unknown", "files_block": files_block})

    try:
        return json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"extract_problem JSON parse failed: {e}")
        return {"_raw": raw, "_error": str(e)}


# ======================================================================
# Phase B — design missing pieces (mutable)
# ======================================================================

_DESIGN_SYSTEM = """\
    You are an expert in continuum mechanics and PDE-constrained optimisation.
    You receive a PARTIAL ProblemSpec extracted from LaTeX — many fields are
    missing or incomplete.  Your job is to produce a COMPLETE, physically sound
    ProblemSpec by designing every missing piece.
    
    ## CRITICAL: flat output format
    
    Every value gets a sibling ``source`` key.  Designed values also get a
    ``rationale`` key.  DO NOT wrap values in nested objects.
    
    CORRECT (flat):
      "problem_type": "shape_optimization",
      "problem_type_source": "extracted"
    
    WRONG (nested — NEVER do this):
      "problem_type": {{"value": "shape_optimization", "source": "extracted"}}
    
    For arrays (boundaries, fields, etc.) each element has ``source`` and
    optionally ``rationale`` as direct keys inside the element object.
    
    ## Complete schema
    
    Scalar fields (each with _source, and _rationale if designed):
      problem_type, problem_type_source[, problem_type_rationale]
      problem_name, problem_name_source
      dimension, dimension_source
      domain_name, domain_name_source
      mesh_variable, mesh_variable_source
      mesh_file, mesh_file_source
    
      objective_name, objective_name_source
      objective_description, objective_description_source
      objective_terms, objective_terms_source   (array of strings)
    
      state_description, state_description_source
      state_weak_lhs, state_weak_lhs_source
      state_weak_rhs, state_weak_rhs_source
      state_is_nonlinear, state_is_nonlinear_source
    
      newton_description, newton_weak_form, newton_source  (if applicable)
    
      adjoint_description, adjoint_description_source
      adjoint_weak_lhs, adjoint_weak_lhs_source
      adjoint_weak_rhs, adjoint_weak_rhs_source
    
      shape_derivative_volume, shape_derivative_boundary, shape_derivative_source
    
    Array fields (each element has source [+ rationale]):
      boundaries: array of objects with keys — name, label, kind, description, source, rationale?
      fields:    array of objects with keys — name, latex, type, fe_space, unit,
                 dirichlet_value, dirichlet_labels, source, rationale?
      test_functions: array of objects with keys — name, latex, type, fe_space, source, rationale?
      constants: array of objects with keys — name, latex, value, description, unit, source, rationale?
      expressions: array of objects with keys — name, latex, formula, source, rationale?
    
    Object fields:
      characteristic_function: null, or object with keys — latex, strong_value, weak_value, source, rationale?
      optimization_config:
        object with keys — mesh_file, hmin, hmax, diffusion, v0, iterations,
        level_set_initial, boundary_labels, fixed_labels, extra_params, source, rationale?
    
    ## Design rules
    
    - **problem_type**: "shape_optimization" or "topology_optimization".
    
    - **boundaries**: the extracted spec may list LaTeX names without mesh_label.
      RECONCILE them with mesh_labels_used by geometric reasoning.
      E.g. "left edge fixed" → label 4 if mesh shows left edge as label 4.
      Every boundary MUST get a concrete label string.
      Mark designed label assignments source="designed".
    
    - **fields**: default fe_space "P2" for vectors (elasticity), "P1" for
      scalars (diffusion/thermal).  Default unit "meter" for displacement,
      "kelvin" for temperature.  dirichlet_labels must match boundary labels.
    
    - **constants**: if extracted constants are empty (no parameter table),
      DESIGN physically reasonable values:
      - Structural steel: E=200e9 Pa, nu=0.3
      - Thermal: k0=1 W/(m*K), alpha=0.01 1/K^2
      - Load magnitude: scale to problem dimensions
      - Lagrange multiplier ell: 0.1-100, scale to balance physics terms
      Mark ALL designed constants source="designed".
    
    - **expressions**: include standard derived quantities (strain, stress,
      Lame parameters) even if not extracted.  Mark designed ones.
    
    - **characteristic_function**: if chi is mentioned, ensure weak_value exists.
      Default "1/100" for solid mechanics, "1/10" for diffusion/thermal.
    
    - **adjoint_equation**: almost every optimisation problem needs one.
      If missing, DESIGN it from the Lagrangian of the state equation.
    
    - **optimization_config**: THE MOST CRITICAL section to design.
      If the extracted spec has none:
      - hmin/hmax: ~1/100 to 1/10 of domain size
      - diffusion: 1e-4
      - v0: 1e-3 to 1e-2
      - iterations: 20-50
      - level_set_initial: a sensible signed-distance expression for the geometry
      - boundary_labels/fixed_labels: align with boundary design
      Mark source="designed" with rationale.
    
    ## Output
    - ONE flat JSON object.  No markdown.  No code fences.
    - EVERY value has a source annotation.
    - Designed values carry a rationale.
    - English throughout.
"""

_DESIGN_USER = """\
    Extracted spec (missing keys = no explicit information):
    {extracted_json}
    
    Problem name: {problem_name}
    
    Return the complete, designed ProblemSpec JSON.
"""


def design_problem(agent, extracted: dict, problem_name: Optional[str] = None) -> dict:
    """Phase B — design every missing piece of a complete ProblemSpec."""
    extracted_json = json.dumps(extracted, indent=2, ensure_ascii=False)

    prompt_list = [
        {"role": "system", "content": _DESIGN_SYSTEM},
        {"role": "user", "content": _DESIGN_USER},
    ]
    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {"extracted_json": extracted_json, "problem_name": problem_name or "unknown"}
    )

    try:
        return json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"design_problem JSON parse failed: {e}")
        return {"_raw": raw, "_error": str(e)}


# ======================================================================
# orchestrator
# ======================================================================


def parse_problem(agent, input_files: dict, problem_name: Optional[str] = None) -> dict:
    """Run Phase A + Phase B, return {"extracted": ..., "designed": ...}."""
    if not input_files:
        warnings.warn("parse_problem: empty input_files")
        return {}

    extracted = extract_problem(agent, input_files, problem_name)
    if "_error" in extracted:
        warnings.warn(
            f"Phase A (extract) failed: {extracted['_error']}.  Using raw input text as fallback."
        )
        # Build a minimal extracted dict from raw input
        raw_text = "\n\n".join(input_files.values())
        extracted = {"_raw_fallback": True, "problem_description": raw_text}

    designed = design_problem(agent, extracted, problem_name)
    if "_error" in designed:
        warnings.warn(
            f"Phase B (design) failed: {designed['_error']}.  Returning extracted-only result."
        )
        # Return extracted without designed — caller can still proceed
        designed = {}

    return {"extracted": extracted, "designed": designed}


# ======================================================================
# Local test — 3 diverse cases
# ======================================================================
if __name__ == "__main__":
    # Ensure UTF-8 output on all platforms (CMD, PyCharm, etc.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from API import DeepSeekCodeAgentV4Flash

    agent = DeepSeekCodeAgentV4Flash()
    demos_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "demos"

    test_cases = sorted([d.name for d in demos_dir.iterdir() if d.is_dir()])

    print("=" * 60)
    print(f"Problem Parser (Extract -> Design) — {len(test_cases)} demos")
    print("=" * 60)
    summary_rows = []

    for case_name in test_cases:
        case_dir = demos_dir / case_name
        if not case_dir.exists():
            print(f"\n[SKIP] {case_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Problem: {case_name}")
        print(f"{'=' * 60}")

        input_files = collect_input_files(case_dir)

        result = parse_problem(agent, input_files, problem_name=case_name)

        # Save raw result to results/<name>/initial.json
        out_dir = Path("results") / case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "initial.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        ex = result.get("extracted", {})
        ds = result.get("designed", {})

        # collect summary data
        pt = ds.get("problem_type", "?")
        dim = ds.get("dimension", "?")

        ex_n = sum(1 for c in ds.get("constants", []) if c.get("source") == "extracted")
        de_n = sum(1 for c in ds.get("constants", []) if c.get("source") == "designed")

        nl = "nonlinear" if ds.get("state_is_nonlinear") else "linear"
        adj = "yes" if (ds.get("adjoint_description") or ds.get("adjoint_weak_lhs")) else "no"
        sd = "yes" if ds.get("shape_derivative_volume") else "no"
        opt_src = ds.get("optimization_config", {}).get("source", "?")
        fields = ", ".join(f.get("name", "?") for f in ds.get("fields", []))

        summary_rows.append((case_name, pt, dim, ex_n, de_n, nl, opt_src, fields))

        print(f"  [{case_name}]")
        print(f"    type={pt}, {dim}D, {nl}")
        print(f"    constants: {ex_n} extracted, {de_n} designed")
        print(f"    adjoint={adj}, shape_deriv={sd}, opt={opt_src}")
        print(f"    fields: [{fields}]")

    # ── summary table ──
    print(f"\n{'=' * 110}")
    print("SUMMARY TABLE — All Demos")
    print(f"{'=' * 110}")
    print(
        f"| {'Problem':<42} | {'Type':<20} | {'Dim':<4} | {'Linear?':<9} "
        f"| {'Constants':<18} | {'Opt config':<10} | {'Fields':<25} |"
    )
    print(f"|{'-' * 42}|{'-' * 20}|{'-' * 4}|{'-' * 9}|{'-' * 18}|{'-' * 10}|{'-' * 25}|")
    for name, pt, dim, ex_n, de_n, nl, opt_src, fields in summary_rows:
        const_str = f"{ex_n} extracted, {de_n} designed"
        print(
            f"| {name:<42} | {pt:<20} | {str(dim):<4} | {nl:<9} | {const_str:<18} | {opt_src:<10} | {fields:<25} |"
        )

    print(f"\n{'=' * 60}")
    print("Test complete.")
