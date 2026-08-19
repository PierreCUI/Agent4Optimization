"""
Static Check & Auto-Fix — catch errors before running FreeFEM.

``static_check`` performs a battery of static analyses on generated code
(no execution).  ``auto_fix`` calls an LLM to repair the detected issues.

Checks performed (see ``_CHECKS`` for the complete list):
  1. import         — ``from autofreefem import *`` present
  2. syntax         — Python AST parse succeeds
  3. boundary_args  — ``Boundary(..., 'ThL', ...)`` not ``'Th'``
  4. afac_naming    — ``Afac`` variable used consistently with ``\\ell`` LaTeX
  5. chi_latex      — ``CharacteristicFunction`` has full LaTeX decoration
  6. lagrangian     — ``Lagrangian([`` present
  7. setup_opt      — ``setup_optimization`` or ``lag.run(`` present
  8. mesh_ref       — ``mesh=`` argument + labels in code exist in mesh spec
  9. dsx_labels     — dsx(...Th, label) labels are valid (mesh labels or Boundary indices)
 10. dimension_3d   — ``dimensions=3`` present for 3D problems
 11. labels_consistency — boundaryLabels/fixedLabels match ProblemSpec boundary kinds
"""

import ast
import inspect
import json
import os
import re
import sys
import warnings
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Generation.label_conventions import AUTO_FIX_LABEL_INSTRUCTIONS
from utils import parser_json

# ======================================================================
# check registry — each check returns a list of error dicts
# ======================================================================


def _check_import(code: str) -> list[dict]:
    """Ensure ``from autofreefem import *`` is present."""
    if "from autofreefem import" not in code:
        return [
            {
                "check": "import",
                "severity": "error",
                "message": "Missing 'from autofreefem import *'.",
            }
        ]
    return []


def _check_syntax(code: str) -> list[dict]:
    """Ensure valid Python syntax via AST parse."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return [
            {
                "check": "syntax",
                "severity": "error",
                "message": f"Python syntax error at line {e.lineno}: {e.msg}",
                "line": e.lineno,
            }
        ]
    return []


def _check_boundary_args(code: str) -> list[dict]:
    """Boundary() second argument must be 'ThL', not 'Th'."""
    pattern = re.compile(r"Boundary\([^)]+,\s*'Th'(?:\s*,|\s*\))")
    matches = pattern.findall(code)
    if matches:
        return [
            {
                "check": "boundary_args",
                "severity": "warning",
                "message": f"Boundary() uses 'Th' instead of 'ThL' ({len(matches)} occurrence(s)).",
            }
        ]
    return []


def _check_afac_naming(code: str) -> list[dict]:
    """If a Lagrange multiplier constant uses LaTeX ``\\ell``, the Python
    variable should be ``Afac`` and bare ``ell`` must not appear in expressions."""
    has_latex_ell = re.search(r"(\w+)\s*=\s*Constant\(r'\\ell'", code)
    if has_latex_ell:
        var_name = has_latex_ell.group(1)
        # Warn if Python variable is 'ell' instead of the preferred 'Afac'
        if var_name == "ell":
            return [
                {
                    "check": "afac_naming",
                    "severity": "warning",
                    "message": "Lagrange multiplier variable is 'ell'; "
                    "consider renaming to 'Afac' for consistency.",
                }
            ]
        # Error if bare 'ell' used in expressions elsewhere
        if re.search(r"\bell\b", code):
            lines_with_ell = [
                l.strip()
                for l in code.split("\n")
                if re.search(r"\bell\b", l) and "Constant" not in l and "r'\\ell'" not in l
            ]
            if lines_with_ell:
                return [
                    {
                        "check": "afac_naming",
                        "severity": "error",
                        "message": f"Variable '{var_name}' defined for Lagrange "
                        f"multiplier but 'ell' used in expressions.",
                    }
                ]
    return []


def _check_chi_latex(code: str) -> list[dict]:
    """CharacteristicFunction must have full LaTeX: ``\\raisebox{\\depth}{$\\chi$}``."""
    if "CharacteristicFunction" not in code:
        return []
    # Match CharacteristicFunction( ... ) and inspect its first argument
    m = re.search(r"CharacteristicFunction\(\s*(r'.+?'|\".+?\")\s*,", code)
    if m:
        arg = m.group(1)
        if "raisebox" not in arg and "\\depth" not in arg:
            return [
                {
                    "check": "chi_latex",
                    "severity": "warning",
                    "message": f"CharacteristicFunction LaTeX should use "
                    f"r'\\raisebox{{\\depth}}{{$\\chi$}}' not {arg}.",
                }
            ]
    return []


def _check_lagrangian(code: str) -> list[dict]:
    """Ensure Lagrangian is instantiated."""
    if "Lagrangian([" not in code:
        return [
            {
                "check": "lagrangian",
                "severity": "error",
                "message": "Missing Lagrangian instantiation.",
            }
        ]
    return []


def _check_setup_opt(code: str) -> list[dict]:
    """Ensure setup_optimization or lag.run is called."""
    if "setup_optimization" not in code and "lag.run(" not in code:
        return [
            {
                "check": "setup_opt",
                "severity": "error",
                "message": "Missing setup_optimization() or lag.run() call.",
            }
        ]
    return []


def _check_mesh_ref(code: str, mesh_spec: dict = None) -> list[dict]:
    """Ensure mesh= argument references a reasonable filename."""
    mesh_match = re.search(r"mesh\s*=\s*'([^']+)'", code)
    if not mesh_match:
        mesh_match = re.search(r'mesh\s*=\s*"([^"]+)"', code)
    if not mesh_match:
        return [
            {
                "check": "mesh_ref",
                "severity": "warning",
                "message": "No mesh= argument found in setup_optimization.",
            }
        ]
    mesh_name = mesh_match.group(1)
    if not mesh_name.endswith((".txt", ".mesh", ".msh")):
        return [
            {
                "check": "mesh_ref",
                "severity": "warning",
                "message": f"Mesh file '{mesh_name}' has unusual extension.",
            }
        ]
    # If mesh_spec is available, check labels consistency
    if mesh_spec:
        ds = mesh_spec.get("designed", {})
        mesh_labels = set(str(l) for l in ds.get("labels_used", []))
        if mesh_labels:
            # Extract labels from boundaryLabels/fixedLabels in code
            bl_match = re.search(r"boundaryLabels\s*=\s*\[([^\]]+)\]", code)
            fl_match = re.search(r"fixedLabels\s*=\s*\[([^\]]+)\]", code)
            code_labels = set()
            for m in (bl_match, fl_match):
                if m:
                    code_labels.update(
                        lbl
                        for lbl in (x.strip().strip("'\"") for x in m.group(1).split(","))
                        if lbl and lbl != "999"
                    )
            unknown = code_labels - mesh_labels
            if unknown:
                return [
                    {
                        "check": "mesh_ref",
                        "severity": "error",
                        "message": f"Labels {sorted(unknown)} used in code but "
                        f"not found in mesh ({sorted(mesh_labels)}).",
                    }
                ]
    return []


def _check_dimension_3d(code: str, problem_spec: dict = None) -> list[dict]:
    """If problem is 3D, ensure dimensions=3 is passed to Lagrangian."""
    if problem_spec:
        ds = problem_spec.get("designed", {})
        if ds.get("dimension") == 3:
            if "dimensions=3" not in code:
                return [
                    {
                        "check": "dimension_3d",
                        "severity": "error",
                        "message": "3D problem detected but 'dimensions=3' missing from Lagrangian().",
                    }
                ]
    return []


def _check_dsx_labels(code: str, mesh_spec: dict = None) -> list[dict]:
    """
    Validate that dsx(..., Th, label) labels exist in the mesh.

    AutoFreeFEM's dsx label is an INDEX into Domain's Boundary list
    when Boundaries are declared, or a raw mesh label otherwise.
    This check verifies that raw integer labels (no Boundaries)
    correspond to actual labels in the mesh file.
    """
    if not mesh_spec:
        return []

    ds_info = mesh_spec.get("designed", {})
    mesh_labels = set()
    for src in [ds_info.get("labels_used", []), ds_info.get("all_labels", [])]:
        if src:
            for l in src:
                try:
                    mesh_labels.add(int(l) if not isinstance(l, (int, float)) else l)
                except (ValueError, TypeError):
                    pass  # skip non-numeric labels like '2/4' or named labels

    if not mesh_labels:
        return []

    # Check if Domain() has Boundary objects
    has_boundary = bool(re.search(r"=\s*Boundary\(", code))
    boundary_count = len(re.findall(r"=\s*Boundary\(", code))

    # Extract dsx labels with paren-counting (handles nested calls like
    # dsx(inner(u, testu), Th, 3)).  The regex fallback catches trivial cases.
    dsx_labels_raw = []
    dsx_labels_str = []
    for m in re.finditer(r"\bdsx\(", code):
        start = m.end()
        depth = 1
        i = start
        while i < len(code) and depth > 0:
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
            i += 1
        call_text = code[start : i - 1]  # text between dsx( and the matching )
        # Extract the label: last argument after the last comma
        # Pattern: ..., Th, LABEL
        label_match = re.search(r",\s*Th\s*,\s*(\d+)\s*$", call_text)
        if label_match:
            dsx_labels_raw.append(label_match.group(1))
        else:
            label_match = re.search(r",\s*Th\s*,\s*'(\d+)'\s*$", call_text)
            if label_match:
                dsx_labels_str.append(label_match.group(1))

    errors = []

    if has_boundary and boundary_count > 0:
        # Labels are boundary indices; verify they are in range
        for lbl_str in dsx_labels_raw + dsx_labels_str:
            lbl = int(lbl_str)
            if lbl == 999:
                continue  # label 999 is always the level-set interface, exempt
            if lbl >= boundary_count:
                errors.append(
                    {
                        "check": "dsx_labels",
                        "severity": "error",
                        "message": (
                            f"dsx label {lbl} is a Boundary index, but Domain() has "
                            f"only {boundary_count} Boundary object(s).  "
                            f"Valid indices: 0..{boundary_count - 1}.  "
                            f"Use index 0 for the first Boundary, 1 for the second, etc."
                        ),
                    }
                )
    else:
        # No Boundaries — labels are raw mesh labels; verify they exist in mesh
        for lbl_str in dsx_labels_raw + dsx_labels_str:
            lbl = int(lbl_str)
            if lbl not in mesh_labels and lbl != 999:
                errors.append(
                    {
                        "check": "dsx_labels",
                        "severity": "error",
                        "message": (
                            f"dsx uses label {lbl}, but the mesh only has labels "
                            f"{sorted(mesh_labels)}.  Label {lbl} will silently "
                            f"evaluate to 0 in FreeFEM, likely causing wrong results.  "
                            f"Either declare Boundary objects and use indices, "
                            f"or use one of the existing mesh labels."
                        ),
                    }
                )

    return errors


def _check_labels_consistency(code: str, problem_spec: dict = None) -> list[dict]:
    """
    Validate label allocation against boundary types from ProblemSpec.

    Rules:
    - Dirichlet labels → must be in fixedLabels (shape-frozen), NOT in
      boundaryLabels (shape-optimisable).
    - Neumann labels → must be in boundaryLabels, NOT in fixedLabels.
    - No label may appear in BOTH lists.
    """
    if not problem_spec:
        return []

    ds = problem_spec.get("designed", {})
    boundaries = ds.get("boundaries", [])
    if not boundaries:
        return []

    # Classify labels by boundary kind
    dirichlet_labels = set()
    neumann_labels = set()
    for b in boundaries:
        label = str(b.get("label", ""))
        kind = (b.get("kind", "") or "").lower()
        if not label:
            continue
        if "dirichlet" in kind:
            dirichlet_labels.add(label)
        elif "neumann" in kind or "natural" in kind or "free" in kind:
            neumann_labels.add(label)
        elif "robin" in kind:
            # Robin (mixed) BC: shape-optimisable like Neumann, but
            # flag as a warning since the BC form depends on position.
            neumann_labels.add(label)

    # Parse boundaryLabels and fixedLabels from code
    bl_match = re.search(r"boundaryLabels\s*=\s*\[([^\]]+)\]", code)
    fl_match = re.search(r"fixedLabels\s*=\s*\[([^\]]+)\]", code)

    code_boundary = set()
    code_fixed = set()
    if bl_match:
        for lbl in bl_match.group(1).split(","):
            lbl = lbl.strip().strip("'\"")
            if lbl:
                code_boundary.add(lbl)
    if fl_match:
        for lbl in fl_match.group(1).split(","):
            lbl = lbl.strip().strip("'\"")
            if lbl:
                code_fixed.add(lbl)

    errors = []

    # Rule 1: Dirichlet labels must NOT be in boundaryLabels
    dirichlet_in_boundary = dirichlet_labels & code_boundary
    if dirichlet_in_boundary:
        errors.append(
            {
                "check": "labels_consistency",
                "severity": "error",
                "message": (
                    f"Dirichlet label(s) {sorted(dirichlet_in_boundary)} found in "
                    f"boundaryLabels.  Dirichlet boundaries are shape-FIXED — "
                    f"move them to fixedLabels instead."
                ),
            }
        )

    # Rule 2: Neumann labels should be in boundaryLabels (warning only)
    neumann_not_in_boundary = neumann_labels - code_boundary
    if neumann_labels and neumann_not_in_boundary:
        errors.append(
            {
                "check": "labels_consistency",
                "severity": "warning",
                "message": (
                    f"Neumann label(s) {sorted(neumann_not_in_boundary)} not in "
                    f"boundaryLabels.  Neumann boundaries are shape-OPTIMISABLE."
                ),
            }
        )

    # Rule 3: No label in both lists
    overlap = code_boundary & code_fixed
    if overlap:
        errors.append(
            {
                "check": "labels_consistency",
                "severity": "error",
                "message": (
                    f"Label(s) {sorted(overlap)} appear in BOTH boundaryLabels and "
                    f"fixedLabels.  A label cannot be both optimisable and fixed."
                ),
            }
        )

    return errors


# ── registry ──────────────────────────────────────────────────────────

_CHECKS = [
    _check_import,
    _check_syntax,
    _check_boundary_args,
    _check_afac_naming,
    _check_chi_latex,
    _check_lagrangian,
    _check_setup_opt,
    _check_mesh_ref,
    _check_dsx_labels,
    _check_dimension_3d,
    _check_labels_consistency,
]


# ======================================================================
# public API
# ======================================================================


def static_check(code: str, mesh_spec: dict = None, problem_spec: dict = None) -> list[dict]:
    """
    Run all static checks on *code* and return a list of error dicts.

    Each error dict has keys: ``check``, ``severity``, ``message``,
    and optionally ``line``.

    Parameters
    ----------
    code : str
        The generated AutoFreeFEM Python source.
    mesh_spec : dict or None
        Output of ``handle_mesh`` (for label / dimension validation).
    problem_spec : dict or None
        Output of ``parse_problem`` (for dimension validation).

    Returns
    -------
    list[dict]
        Errors found, ordered by severity ("error" before "warning").
    """
    all_errors = []
    for check_fn in _CHECKS:
        try:
            sig = inspect.signature(check_fn)
            kwargs = {}
            if "mesh_spec" in sig.parameters:
                kwargs["mesh_spec"] = mesh_spec
            if "problem_spec" in sig.parameters:
                kwargs["problem_spec"] = problem_spec
            all_errors.extend(check_fn(code, **kwargs))
        except Exception as e:
            warnings.warn(f"Static check {check_fn.__name__} raised: {e}")

    # Sort: errors first, then warnings
    all_errors.sort(key=lambda e: 0 if e["severity"] == "error" else 1)
    return all_errors


# ======================================================================
# auto-fix via LLM
# ======================================================================

_FIXER_SYSTEM = (
    """\
    You are an expert AutoFreeFEM code repairer.  You receive a piece of generated
    AutoFreeFEM Python code and a list of static-check errors.  Your job is to
    fix every error while keeping the code otherwise unchanged.
    
    ## Rules
    
    - Fix ONLY the reported issues.  Do not refactor, restructure, or "improve"
      unrelated parts.
    - If the error is about ``Boundary(..., 'Th')`` → change to ``'ThL'``.
    - If the error is about Afac/ell naming → rename consistently.
    - If the error is about CharacteristicFunction LaTeX → add the full
      ``\\raisebox{{\\depth}}{{$\\chi$}}`` decoration.
    - If the error is about missing imports → add ``from autofreefem import *``.
    - If the error is about Python syntax → fix exactly the syntax issue.
    - If the error is about missing dimensions=3 → add it to Lagrangian().
    """
    + AUTO_FIX_LABEL_INSTRUCTIONS
    + """
    
    ## Output
    
    Return exactly one JSON object:
    {{"main": "<complete corrected Python code>", "mesh": ""}}
    No markdown.  No code fences.  No extra text.
"""
)

_FIXER_USER = """\
    Current code:
    {code}
    
    Errors to fix:
    {errors_json}
    
    Return the corrected code JSON.
"""


def auto_fix(agent, code: str, errors: list[dict]) -> dict:
    """Use an LLM to repair *errors* in *code*.

    Returns ``{"main": "<corrected code>", "mesh": ""}``.
    """
    if not errors:
        return {"main": code, "mesh": ""}

    prompt_list = [
        {"role": "system", "content": _FIXER_SYSTEM},
        {"role": "user", "content": _FIXER_USER},
    ]
    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {"code": code, "errors_json": json.dumps(errors, indent=2, ensure_ascii=False)}
    )

    try:
        return json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"auto_fix JSON parse failed: {e}")
        return {"main": code, "mesh": "", "_error": str(e)}


# ======================================================================
# Local test — run on all generated main.py files in results/
# ======================================================================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from API import DeepSeekCodeAgentV4Flash

    agent = DeepSeekCodeAgentV4Flash()
    results_dir = Path("results")

    py_files = sorted(results_dir.glob("*/main.py"))
    if not py_files:
        print("No results/*/main.py found. Run coder first.")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"Static Check — {len(py_files)} generated files")
    print(f"{'=' * 70}")

    summary_rows = []

    for py_file in py_files:
        name = py_file.parent.name
        code = py_file.read_text(encoding="utf-8", errors="replace")

        # Load optional context
        mesh_spec = None
        mesh_file = py_file.parent / "mesh.json"
        if mesh_file.exists():
            with open(mesh_file, "r", encoding="utf-8") as f:
                mesh_spec = json.load(f)

        problem_spec = None
        spec_file = py_file.parent / "initial.json"
        if spec_file.exists():
            with open(spec_file, "r", encoding="utf-8") as f:
                problem_spec = json.load(f)

        errors = static_check(code, mesh_spec, problem_spec)
        err_count = len([e for e in errors if e["severity"] == "error"])
        warn_count = len([e for e in errors if e["severity"] == "warning"])

        status = "OK" if not errors else f"{err_count}E+{warn_count}W"
        print(f"\n  [{name}] {status}")

        for e in errors:
            sev = e["severity"].upper()
            print(f"    [{sev}] {e['check']}: {e['message']}")

        summary_rows.append((name, status, errors))

    # ── summary table ──
    print(f"\n{'=' * 70}")
    print("STATIC CHECK SUMMARY")
    print(f"{'=' * 70}")
    print(f"| {'Problem':<42} | {'Status':<10} | Details")
    print(f"|{'-' * 42}|{'-' * 10}|{'-' * 40}")
    for name, status, errors in summary_rows:
        detail = "; ".join(f"{e['check']}" for e in errors[:3])
        if len(errors) > 3:
            detail += f", ... (+{len(errors) - 3})"
        print(f"| {name:<42} | {status:<10} | {detail:<40}")

    # ── auto-fix test on first file with errors ──
    files_with_errors = [(n, s, e) for n, s, e in summary_rows if e]
    if files_with_errors:
        name, _, errors = files_with_errors[0]
        py_file = results_dir / name / "main.py"
        code = py_file.read_text(encoding="utf-8", errors="replace")
        print(f"\n{'=' * 70}")
        print(f"Auto-Fix test on: {name}")
        print(f"{'=' * 70}")
        fixed = auto_fix(agent, code, errors)
        if "_error" in fixed:
            print(f"  Fix ERROR: {fixed['_error']}")
        else:
            new_errors = static_check(fixed.get("main", ""))
            print(f"  Errors before: {len(errors)}  →  after: {len(new_errors)}")
            for e in new_errors:
                print(f"    [!] {e['check']}: {e['message']}")
    else:
        print(f"\nAll files clean — auto_fix test skipped.")

    print(f"\nDone.")
