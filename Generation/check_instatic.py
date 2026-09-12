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
 11. labels_consistency — explicit design_role in the ProblemSpec matches
     the code; no label in both lists.  (Kind-based heuristics are NOT
     enforced: the reference corpus intentionally mixes physical kinds and
     design roles.)
 12. internal_boundary_labels — label 10 (the label-10 material boundary)
     never appears in boundaryLabels/fixedLabels
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

from Generation.label_conventions import AUTO_FIX_LABEL_INSTRUCTIONS  # noqa: E402
from utils import code_masked_positions, parser_json  # noqa: E402

# ======================================================================
# shared helpers
# ======================================================================


def _call_content(code: str, m) -> str | None:
    """Return the text between the parens of the call whose opening paren
    is at *m.end()* (m = a regex match of ``<func>(``).

    Quote-aware: parentheses inside string arguments (e.g. a Boundary
    LaTeX ``r'\\Gamma(\\alpha)'``) must not desync the depth counter.
    Returns None when the call is unbalanced (nested scan of the same
    code should then just skip it).
    """
    start = m.end()
    depth = 1
    i = start
    quote = None
    while i < len(code) and depth > 0:
        ch = code[i]
        if quote:
            if ch == quote and code[i - 1] != "\\":
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    return code[start : i - 1] if depth == 0 else None


def _strip_comments(text: str) -> str:
    """Return *text* with ``#`` line comments removed, strings intact.

    Generated code can contain commented-out examples (the coder prompt
    shows setup_optimization/Boundary/Lagrangian shapes) — substring
    checks must not take comment text for real code.

    Line offsets are preserved (each removed comment becomes exactly one
    newline); backslash escapes and triple-quoted strings are handled so
    a ``#`` inside a string is never taken for a comment.
    """
    out = []
    quote = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and len(quote) == 1:
                out.append(ch)
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if text.startswith(quote, i):
                out.extend(text[i : i + len(quote)])
                i += len(quote)
                quote = None
                continue
            out.append(ch)
            i += 1
        elif ch in ("'", '"'):
            quote = next((q for q in ('"""', "'''", '"', "'") if text.startswith(q, i)), ch)
            out.append(ch)
            i += 1
        elif ch == "#":
            # comment: discard through the end of the line, newline
            # included (append exactly one "\n" in its place)
            while i < n and text[i] != "\n":
                i += 1
            if i < n:
                i += 1
            out.append("\n")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _top_level_split(text: str) -> list[str]:
    """Split *text* on commas at the TOP level (not inside parentheses
    or quotes) — used to parse Domain(...)/Boundary(...) call arguments."""
    parts, cur, depth = [], [], 0
    quote = None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _parse_label_lists(code: str) -> tuple[set, set]:
    """Extract ``boundaryLabels=[...]`` / ``fixedLabels=[...]`` from *code*.

    Returns ``(code_boundary, code_fixed)`` as sets of stripped label
    strings.  Both regexes match the canonical multi-line list format
    produced by the coder.
    """
    code_boundary = set()
    code_fixed = set()
    for kw, target in (("boundaryLabels", code_boundary), ("fixedLabels", code_fixed)):
        m = re.search(rf"{kw}\s*=\s*\[([^\]]+)\]", code)
        if m:
            target.update(
                lbl.strip().strip("'\"") for lbl in m.group(1).split(",") if lbl.strip()
            )
    return code_boundary, code_fixed


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
    """Boundary() second argument must be 'ThL', not 'Th'.

    Paren-counting argument split: a LaTeX argument like
    ``r'\\Gamma(0.5)'`` contains balanced parens and either quote style
    is accepted, so no Boundary call is missed by a flat regex.  Comments
    are stripped first — a commented-out wrong example is not code.
    """
    matches = 0
    code_nc = _strip_comments(code)
    for m in re.finditer(r"\bBoundary\(", code_nc):
        call_text = _call_content(code_nc, m)
        if call_text is None:
            continue
        args = _top_level_split(call_text)
        if len(args) >= 2 and args[1].strip() in ("'Th'", '"Th"'):
            matches += 1
    if matches:
        return [
            {
                "check": "boundary_args",
                "severity": "warning",
                "message": f"Boundary() uses 'Th' instead of 'ThL' ({matches} occurrence(s)).",
            }
        ]
    return []


def _find_latex_ell_var(code: str) -> str | None:
    """Return the variable name assigned ``= Constant(<latex>\\ell)``.

    AST-based, so both quote styles (``r'\\ell'`` / ``r"\\ell"``),
    whitespace variants and comments are handled; on unparseable code
    (the syntax check reports that anyway) fall back to a tolerant regex.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        m = re.search(
            r"(\w+)\s*=\s*Constant\s*\(\s*r?\s*['\"]\\ell['\"]", code
        )
        return m.group(1) if m else None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Constant"
        ):
            a0 = node.value.args[0] if node.value.args else None
            if isinstance(a0, ast.Constant) and a0.value == "\\ell":
                return node.targets[0].id
    return None


def _check_afac_naming(code: str) -> list[dict]:
    """If a Lagrange multiplier constant uses LaTeX ``\\ell``, the Python
    variable should be ``Afac`` and bare ``ell`` must not appear in expressions."""
    has_latex_ell = _find_latex_ell_var(code)
    if has_latex_ell is not None:
        var_name = has_latex_ell
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
        # Error if bare 'ell' used in expressions elsewhere — only the
        # *identifier* counts: a comment ("# ell is the multiplier") or a
        # string (r'\ell') must not trigger a repair round.  AST-based so
        # the check sees Names only; on unparseable code fall back to a
        # quote/comment-aware line scan (the syntax check reports the
        # real problem anyway).
        ell_lines = []
        try:
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Name) and node.id == "ell":
                    ell_lines.append(node.lineno)
        except SyntaxError:
            for i, l in enumerate(code.split("\n")):
                l_code = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "", l)
                l_code = re.sub(r"#.*", "", l_code)
                if re.search(r"\bell\b", l_code):
                    ell_lines.append(i + 1)
        if ell_lines:
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
    """The *contrast* CharacteristicFunction must have full LaTeX:
    ``\\raisebox{\\depth}{$\\chi$}``.

    CharacteristicFunction is also used for material-property fields
    with plain names (``'E'``, ``'k'``, ``r'\\alpha_T'``) — only a LaTeX
    string that actually denotes chi needs the decoration.

    The first argument may be a quoted literal (``'...'`` / ``r'...'`` /
    ``"..."`` / ``r"..."``) or a variable name whose value is assigned in
    the code (e.g. ``chi = r'\\chi'``); both are inspected, so an
    unquoted variable cannot silently bypass the check.
    """
    if "CharacteristicFunction" not in code:
        return []

    def _is_full_latex(s: str) -> bool:
        return "raisebox" in s and "\\depth" in s

    matches = []  # latex strings found (or None when unresolved)
    for m in re.finditer(r"CharacteristicFunction\(\s*((?:r)?(['\"]).*?\2)\s*,", code):
        arg = m.group(1)
        # Drop an optional r prefix FIRST, then strip the quotes —
        # dropping the first char instead would remove the opening quote
        # of a bare '\chi' / "\chi" literal and skip the check.
        lit = arg[1:] if arg[:1] == "r" else arg
        if lit[:1] in ("'", '"') and lit[-1:] == lit[:1]:
            matches.append(lit[1:-1])
    # Unquoted-variable first arguments: resolve the assignment for the
    # variable name (e.g. `chi = r'\chi'` -> inspect that string).
    # Only look at the code BEFORE the call (a later assignment would be
    # a NameError anyway) and require a line-anchored assignment so an
    # unrelated earlier `chi = '...'` string is not misattributed.
    # Single-letter names are variables here just as much as longer ones.
    for m in re.finditer(r"CharacteristicFunction\(\s*(\w+)\s*,", code):
        var = m.group(1)
        am = re.search(rf"(?m)^\s*{var}\s*=\s*((?:r)?(['\"]).*?\2)", code[: m.start()])
        if am:
            lit = am.group(1)[1:] if am.group(1)[:1] == "r" else am.group(1)
            if lit[:1] in ("'", '"') and lit[-1:] == lit[:1]:
                matches.append(lit[1:-1])

    for latex in matches:
        # Only the chi-carrying contrast function: property fields use
        # plain names ('E', 'k', '\\alpha_T') and need no decoration.
        if "\\chi" in latex and not _is_full_latex(latex):
            return [
                {
                    "check": "chi_latex",
                    "severity": "warning",
                    "message": f"CharacteristicFunction LaTeX should use "
                    f"r'\\raisebox{{\\depth}}{{$\\chi$}}' not {latex!r}.",
                }
            ]
    return []


def _check_lagrangian(code: str) -> list[dict]:
    """Ensure Lagrangian is instantiated (comment text does not count)."""
    if not re.search(r"Lagrangian\s*\(\s*\[", _strip_comments(code)):
        return [
            {
                "check": "lagrangian",
                "severity": "error",
                "message": "Missing Lagrangian instantiation.",
            }
        ]
    return []


def _check_setup_opt(code: str) -> list[dict]:
    """Ensure setup_optimization or lag.run is called.

    Call-form regexes on comment-stripped code, so a comment mentioning
    ``setup_optimization(...)`` cannot satisfy the check for a missing call.
    """
    code_nc = _strip_comments(code)
    if not re.search(r"\bsetup_optimization\s*\(", code_nc) and not re.search(
        r"\blag\.run\s*\(", code_nc
    ):
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
    # Comments stripped — a commented-out mesh='...' example must not
    # fake a match (or fake its absence), like every other substring
    # check in this module.
    code_nc = _strip_comments(code)
    mesh_match = re.search(r"mesh\s*=\s*'([^']+)'", code_nc)
    if not mesh_match:
        mesh_match = re.search(r'mesh\s*=\s*"([^"]+)"', code_nc)
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
        mesh_labels = set(str(l) for l in (ds.get("labels_used") or []))
        if mesh_labels:
            code_boundary, code_fixed = _parse_label_lists(code)
            code_labels = (code_boundary | code_fixed) - {"999"}
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
    """If problem is 3D, ensure dimensions=3 is passed to Lagrangian.

    Scoped to the Lagrangian call text (comments stripped) so a comment
    mentioning ``dimensions=3`` or a kwarg in another call cannot satisfy
    the check for a missing flag.
    """
    if not problem_spec:
        return []
    ds = problem_spec.get("designed", {})
    if ds.get("dimension") != 3:
        return []
    # Only REAL Lagrangian calls count: a Lagrangian(...)-shaped example
    # inside a docstring/string or comment must not satisfy the check
    # (the code mask rejects matches whose start is masked).
    masked = code_masked_positions(code)
    for m in re.finditer(r"\bLagrangian\s*\(", code):
        if m.start() in masked:
            continue
        call_text = _call_content(code, m)
        if call_text is not None and re.search(r"\bdimensions\s*=\s*3\b", call_text):
            return []
    return [
        {
            "check": "dimension_3d",
            "severity": "error",
            "message": "3D problem detected but 'dimensions=3' missing from Lagrangian().",
        }
    ]


def _check_dsx_labels(code: str, mesh_spec: dict = None) -> list[dict]:
    """
    Validate that dsx(..., <mesh_var>, label) labels are consistent.

    autofreefem's dsx semantics (``integraloperators.py`` ``dsx._sympystr``):
    *label* < number of declared Boundaries  → INDEX into the Domain's
    Boundary list;
    *label* >= that count (or none declared) → RAW mesh label.
    Only the raw-label mode can be wrong (an unknown raw label silently
    evaluates to 0 in FreeFEM); an index always resolves to a declared
    Boundary.  The semantics are decided PER DOMAIN: a Boundary object
    declared elsewhere (another Domain, or not attached at all) must NOT
    flip a different domain's valid raw mesh label into an index check.
    """
    if not mesh_spec:
        return []

    ds_info = mesh_spec.get("designed", {})
    mesh_labels = set()
    for key in ("labels_used", "all_labels"):
        src = ds_info.get(key)
        if not isinstance(src, (list, tuple, set)):
            continue
        for lbl in src:
            try:
                mesh_labels.add(int(lbl) if not isinstance(lbl, (int, float)) else lbl)
            except (ValueError, TypeError):
                pass  # skip non-numeric labels like '2/4' or named labels

    if not mesh_labels:
        return []

    # All scans below run on comment-stripped code: a commented-out
    # ``bndN = Boundary(...)`` declaration would inflate the per-Domain
    # count and flip a genuinely invalid RAW label into the index mode.
    # (No positions are reported back, so stripping is safe here.)
    code = _strip_comments(code)

    # --- per-Domain Boundary counts: mesh variable -> n_boundaries ---
    domain_info = {}
    for m in re.finditer(r"\bDomain\(", code):
        call_text = _call_content(code, m)
        if call_text is None:
            continue
        args = _top_level_split(call_text)
        mesh_var = ""
        if len(args) >= 2:
            mesh_var = args[1].strip().strip("'\"")
        # Boundaries bound to THIS domain: inline Boundary(...) calls in
        # the argument list PLUS declarations referenced by bare name
        # (canonical coder pattern: bndN = Boundary(...); ...;
        # Domain('D', 'Th', bndN)).
        n_bound = len(re.findall(r"\bBoundary\(", call_text))
        for a in args[2:]:
            name = a.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", name):
                n_bound += len(re.findall(rf"\b{name}\s*=\s*Boundary\(", code))
        if mesh_var:
            # A declaration with boundaries (re)defines the semantics;
            # absent boundaries keep the raw-label semantics.
            if n_bound > 0 or mesh_var not in domain_info:
                domain_info[mesh_var] = n_bound

    # --- dsx calls: (mesh_var, label) via paren-counting ---
    dsx_pairs = []
    for m in re.finditer(r"\bdsx\(", code):
        call_text = _call_content(code, m)
        if call_text is None:
            continue
        # Pattern: ..., <mesh_var>, LABEL — the label need NOT be the
        # final argument (extra kwargs like weight=... are allowed).
        lm = re.search(r",\s*(\w+)\s*,\s*(\d+)\s*(?:,|$)", call_text)
        if lm:
            dsx_pairs.append((lm.group(1), int(lm.group(2))))
        else:
            lm = re.search(r",\s*(\w+)\s*,\s*'(\d+)'\s*(?:,|$)", call_text)
            if lm:
                dsx_pairs.append((lm.group(1), int(lm.group(2))))

    errors = []
    for var, lbl in dsx_pairs:
        if lbl == 999:
            continue  # label 999 is always the level-set boundary, exempt
        count = domain_info.get(var, 0)
        if count > 0 and lbl < count:
            # INDEX mode — the package resolves to the label-th declared
            # Boundary (label 0 = first, 1 = second, ...); always valid.
            continue
        # RAW-label mode (no Boundaries declared, or label >= count with
        # Boundaries declared — the package falls through to int1d(Th, lbl)).
        if lbl not in mesh_labels:
            errors.append(
                {
                    "check": "dsx_labels",
                    "severity": "error",
                    "message": (
                        f"dsx({var}) uses label {lbl}, but the mesh only has "
                        f"labels {sorted(mesh_labels)}.  "
                        f"With {count} Boundary object(s) declared, labels "
                        f"0..{count - 1} are indices and labels >= {count} are "
                        f"raw mesh labels.  Label {lbl} will silently evaluate "
                        f"to 0 in FreeFEM, likely causing wrong results — use "
                        f"an index below {count} or one of the existing mesh labels."
                    ),
                }
            )

    return errors


def _check_internal_boundary_labels(code: str) -> list[dict]:
    """Label 10 (the label-10 MATERIAL boundary — a hole or the solid/void
    boundary of the level set within ONE domain) must not appear in
    boundaryLabels or fixedLabels.  The label-10 boundary is design by
    default (it moves with the level set) and carries no physical
    condition, and both lists are for NON-10 boundaries only.

    NOTE: label 10 is NOT the interface of a multi-domain problem — an
    interface separates TWO computational domains, is a non-10 boundary
    with its own design role (design or fixed) and a continuity
    condition, and uses its own mesh label (never 10).  This check
    applies to label 10 only."""
    errors = []
    code_boundary, code_fixed = _parse_label_lists(code)
    for kw, labels in (("boundaryLabels", code_boundary), ("fixedLabels", code_fixed)):
        if "10" in labels:
            errors.append(
                    {
                        "check": "internal_boundary_labels",
                        "severity": "error",
                        "message": (
                            f"Label 10 (the label-10 material boundary) appears in "
                            f"{kw}.  Label 10 marks the material boundary "
                            f"(a hole / solid-void boundary) — it is NOT a "
                            f"non-10 boundary.  The label-10 boundary is design "
                            f"by default and never belongs in boundaryLabels or "
                            f"fixedLabels — remove it."
                        ),
                    }
                )
    return errors


def _check_labels_consistency(code: str, problem_spec: dict = None) -> list[dict]:
    """
    Validate the DESIGN role (boundaryLabels vs fixedLabels) of the non-10
    boundaries against an EXPLICIT design_role from the ProblemSpec.

    Non-10 boundaries carry two independent aspects:
    - design role: design (movable → boundaryLabels) or fixed (frozen → fixedLabels);
    - physical condition: Dirichlet / Neumann / Robin.

    Explicit design_role in the ProblemSpec is ground truth and the code
    must match it.  WITHOUT an explicit role the physical-kind
    "convention" is deliberately NOT enforced: the pipeline's reference
    corpus mixes kinds and roles on purpose (a clamped Dirichlet edge
    that participates in the design lives in boundaryLabels; a loaded
    Neumann edge frozen by the device layout lives in fixedLabels), and
    reported findings would otherwise trigger an auto-fix round that
    inverts a working configuration.

    Structural invariants:
    - No label may appear in BOTH lists.
    - The label-10 boundary (material boundary) carries no physical
      kind and is handled by _check_internal_boundary_labels.
    """
    if not problem_spec:
        return []

    ds = problem_spec.get("designed", {})
    boundaries = ds.get("boundaries", [])
    if not boundaries:
        return []

    design_role = {}  # label -> "design" | "fixed"
    for b in boundaries:
        label = str(b.get("label", ""))
        role = (b.get("design_role", "") or "").lower()
        if not label or label == "10":
            # The label-10 material boundary is handled exclusively by
            # _check_internal_boundary_labels, never classified here.
            continue
        if role in ("design", "fixed"):
            design_role[label] = role

    # Parse boundaryLabels and fixedLabels from code (shared helper)
    code_boundary, code_fixed = _parse_label_lists(code)

    errors = []

    # Rule 0: explicit design_role in the ProblemSpec is ground truth —
    # the code must match it, regardless of the physical kind.
    for lbl, role in design_role.items():
        if role == "fixed" and lbl in code_boundary:
            errors.append(
                {
                    "check": "labels_consistency",
                    "severity": "error",
                    "message": (
                        f"Label {lbl} has explicit design_role='fixed' in the "
                        f"ProblemSpec but appears in boundaryLabels.  A fixed "
                        f"(frozen) boundary must go in fixedLabels."
                    ),
                }
            )
        elif role == "design" and lbl in code_fixed:
            errors.append(
                {
                    "check": "labels_consistency",
                    "severity": "error",
                    "message": (
                        f"Label {lbl} has explicit design_role='design' in the "
                        f"ProblemSpec but appears in fixedLabels.  A design "
                        f"(movable) boundary must go in boundaryLabels."
                    ),
                }
            )
        elif lbl not in code_boundary and lbl not in code_fixed:
            # Absence is as much a violation as misplacement: the
            # explicit role must show up in the code somewhere.
            errors.append(
                {
                    "check": "labels_consistency",
                    "severity": "error",
                    "message": (
                        f"Label {lbl} has explicit design_role='{role}' in the "
                        f"ProblemSpec but appears in NEITHER boundaryLabels nor "
                        f"fixedLabels.  The design role must be assigned where it "
                        f"is needed — put it in boundaryLabels (design/movable) or "
                        f"fixedLabels (fixed/frozen) accordingly."
                    ),
                }
            )

    # Rule 2: No label in both lists
    overlap = code_boundary & code_fixed
    if overlap:
        errors.append(
            {
                "check": "labels_consistency",
                "severity": "error",
                "message": (
                    f"Label(s) {sorted(overlap)} appear in BOTH boundaryLabels and "
                    f"fixedLabels.  A boundary has exactly one design role — it "
                    f"cannot be both movable and fixed."
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
    _check_internal_boundary_labels,
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
        parsed = json.loads(parser_json(raw))
        if not isinstance(parsed, dict):
            warnings.warn(f"auto_fix returned non-object JSON ({type(parsed).__name__}).")
            return {"main": code, "mesh": "", "_error": "non-object JSON response"}
        return parsed
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

        summary_rows.append((name, status, errors, mesh_spec, problem_spec))

    # ── summary table ──
    print(f"\n{'=' * 70}")
    print("STATIC CHECK SUMMARY")
    print(f"{'=' * 70}")
    print(f"| {'Problem':<42} | {'Status':<10} | Details")
    print(f"|{'-' * 42}|{'-' * 10}|{'-' * 40}")
    for name, status, errors, _, _ in summary_rows:
        detail = "; ".join(f"{e['check']}" for e in errors[:3])
        if len(errors) > 3:
            detail += f", ... (+{len(errors) - 3})"
        print(f"| {name:<42} | {status:<10} | {detail:<40}")

    # ── auto-fix test on first file with errors ──
    files_with_errors = [row for row in summary_rows if row[2]]
    if files_with_errors:
        name, _, errors, mesh_spec, problem_spec = files_with_errors[0]
        py_file = results_dir / name / "main.py"
        code = py_file.read_text(encoding="utf-8", errors="replace")
        print(f"\n{'=' * 70}")
        print(f"Auto-Fix test on: {name}")
        print(f"{'=' * 70}")
        fixed = auto_fix(agent, code, errors)
        if "_error" in fixed:
            print(f"  Fix ERROR: {fixed['_error']}")
        else:
            # Verify with the SAME specs THAT FILE was checked with (the
            # loop variables are stale by now) — otherwise the
            # dsx_labels/labels_consistency/dimension_3d checks silently
            # disappear during the local test.
            new_errors = static_check(fixed.get("main", ""), mesh_spec, problem_spec)
            print(f"  Errors before: {len(errors)}  ->  after: {len(new_errors)}")
            for e in new_errors:
                print(f"    [!] {e['check']}: {e['message']}")
    else:
        print("\nAll files clean — auto_fix test skipped.")

    print("\nDone.")
