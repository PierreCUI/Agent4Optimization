"""
Static Check & Repair — Stage 5 of the pipeline (the merged static
check + parameter review).

``static_check`` runs a battery of deterministic analyses on the
generated code (no execution).  ``run_static_repair`` is the
generate-once -> check -> repair loop: every round runs the
deterministic checks AND the merged LLM review, then an LLM repair
fixes ONLY the reported issues — the code is never rewritten from
scratch.  At most ``configs.MAX_STATIC_FIXES`` repairs are attempted;
a repair history (per-round errors, what each round removed / added /
modified) is maintained exactly like the Initialization spec loop and,
when a history_log path is given, recorded in full into history.txt;
the repair prompt forbids repeating a mistake already listed there.

The checks follow the 14-item code checklist; every item that CAN be
hard-checked IS hard-checked deterministically, the rest is bundled
into ONE LLM review (``_REVIEW_SYSTEM`` uses the same numbering):

   1 import         — ``from autofreefem import *`` present
   2 name           — the ``name`` variable is defined
   3 (removed)      — Boundary objects are no longer written: boundary
                     terms use raw mesh labels directly
   4 domain         — every Domain( has exactly (name, mesh variable);
                     spec domains all declared
   5 fields         — every spec field declared with its fe_space,
                     domain, dirichlet value and labels
   6 test functions — every spec test function declared with the tested
                     field's fe_space, same domain, NO extras
   7 multi-domain   — covered by 4 (spec domains) + LLM (interfaces)
   8 constants      — every spec constant declared; extracted values
                     appear; pen = 1e15 when used
   9 characteristic — full latex decoration; strong_value = 1;
                     weak_value given
  10 expressions    — the objective (J) declared
  11 lagrangian     — exists; state/test lists match the spec; third
                     argument is the objective; check_units=True
  12 setup_optimization — exists; first argument is ``name``; all
                     required parameters present (mesh/hmin/hmax/
                     diffusion/v0/iterations/phi/boundaryLabels/
                     fixedLabels); only API-accepted kwargs (never
                     maxiter); mesh='mesh.txt'; iterations/phi/labels
                     equal the spec EXACTLY; 0 < hmin < hmax
  13 mesh           — mesh.txt defines a mesh (``mesh Th = ...``);
                     label-10 hole borders require the region-3 mark
                     (``change(Th, reft=re)``)
  14 no_comments    — the generated code contains NO comments
Plus: syntax (AST), structure order, phi sanity (parses, NEGATIVE
somewhere in the domain), and the TWO-PART symmetry check (see
Initialization.phi_rules): part 1 decides whether the INPUT problem is
symmetric; when it is NOT, the check ends there.  Part 2 — when the
problem IS symmetric — checks that the RESULT (phi) is symmetric about
the same axes.

Deterministic enforcement (never left to the LLM):
  - ``show_weak_material`` is forced to ``False`` before every round and
    at the end.
  - comments are stripped from the final code (rule e).
  - a still-asymmetric phi on a symmetric problem falls back to the
    canonical symmetric pattern.
"""

import ast
import difflib
import inspect
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import configs  # noqa: E402
from history import record as record_history, set_label  # noqa: E402
from Initialization.phi_rules import (  # noqa: E402
    _get_sympy,
    canonical_symmetric_phi,
    detect_symmetry,
    is_phi_symmetric,
    mesh_box,
    mesh_dims,
    symmetry_issue,
)
from utils import code_masked_positions, parser_json  # noqa: E402

# ======================================================================
# shared helpers
# ======================================================================


def _call_content(code: str, m) -> str | None:
    """Return the text between the parens of the call whose opening paren
    is at *m.end()* (m = a regex match of ``<func>(``).

    Quote-aware: parentheses inside string arguments (e.g. a LaTeX name
    ``r'\\Gamma(\\alpha)'``) must not desync the depth counter.
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
    shows setup_optimization/Lagrangian shapes) — substring checks must
    not take comment text for real code.

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


def _find_comment_lines(text: str) -> list:
    """Line numbers of ``#`` comments in *text* (strings never count).

    Same quote awareness as ``_strip_comments`` but reports positions;
    triple-quoted strings are matched with their real length so a
    comment after a docstring is still found.
    """
    lines = []
    quote = None
    i, n = 0, len(text)
    line = 1
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and len(quote) == 1:
                if i + 1 < n and text[i + 1] == "\n":
                    line += 1
                i += 2 if i + 1 < n else 1
                continue
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            if ch == "\n":
                line += 1
            i += 1
        elif ch in ("'", '"'):
            quote = next((q for q in ('"""', "'''", '"', "'") if text.startswith(q, i)), ch)
            i += len(quote)
        elif ch == "#":
            lines.append(line)
            while i < n and text[i] != "\n":
                i += 1
        elif ch == "\n":
            line += 1
            i += 1
        else:
            i += 1
    return lines


def _top_level_split(text: str) -> list:
    """Split *text* on commas at the TOP level (not inside parentheses,
    brackets, braces or quotes) — so a list argument like ``[T, u]``
    stays one argument instead of splitting at its commas."""
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
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _find_setup_optimization(code: str):
    """Return the (start, end) character span of the ``setup_optimization``
    call, or None when the call is absent.

    Takes the LAST balanced REAL call: the coder prompt shows a
    ``setup_optimization``-shaped example, so comment/docstring mentions
    before OR after the real call are common and must not hijack the
    span.  Comments and strings are rejected via the shared code mask.

    The parenthesis scan is string-aware: brackets inside string
    arguments (e.g. the phi expression) are not counted."""
    masked = code_masked_positions(code)
    last = None
    for m in re.finditer(r"setup_optimization\s*\(", code):
        if m.start() in masked:
            continue  # comment/string mention — never a real call
        start = m.end()
        depth = 1
        i = start
        while i < len(code) and depth > 0:
            c = code[i]
            if c in ("'", '"'):
                # skip the whole string literal
                j = i + 1
                while j < len(code):
                    if code[j] == "\\":
                        j += 2
                        continue
                    if code[j] == c:
                        break
                    j += 1
                i = j + 1
                continue
            if c == "#":
                # line comment — skip to end of line: an apostrophe
                # inside a comment must not enter string mode and
                # swallow the real closing parenthesis
                j = code.find("\n", i)
                i = len(code) if j == -1 else j
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        if depth == 0:
            last = (m.start(), i)
    return last


def _param_span(call_text: str, param: str):
    """Return the (start, end) span of the ``param=...`` argument inside a
    setup_optimization call text, or None when the parameter is absent.

    List arguments (``boundaryLabels=['1','2']``) are scanned with
    bracket balancing; string arguments (``phi='min(x, 1.0-x)'``) with
    quote-aware scanning — the naive ``[^,)]`` match would stop at the
    comma or parenthesis inside the value.  Comment mentions of the
    parameter (``# iterations = 50``) are skipped via the code mask."""
    masked = code_masked_positions(call_text)
    m = next(
        (
            mm
            for mm in re.finditer(rf"\b{param}\s*=\s*", call_text)
            if mm.start() not in masked
        ),
        None,
    )
    if not m:
        return None
    rest = call_text[m.end():]
    stripped = rest.lstrip()
    value_start = m.end() + (len(rest) - len(stripped))
    if stripped.startswith("["):
        depth = 0
        i = value_start
        while i < len(call_text):
            c = call_text[i]
            if c in ("'", '"'):
                # skip string elements inside the list
                j = i + 1
                while j < len(call_text):
                    if call_text[j] == "\\":
                        j += 2
                        continue
                    if call_text[j] == c:
                        break
                    j += 1
                i = j + 1
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return m.start(), i + 1
            i += 1
        return None
    if stripped.startswith(("'", '"')):
        quote = stripped[0]
        j = value_start + 1
        while j < len(call_text):
            if call_text[j] == "\\":
                j += 2
                continue
            if call_text[j] == quote:
                return m.start(), j + 1
            j += 1
        return None
    if stripped.startswith("("):
        # parenthesised value (hmin=(0.01)) — paren-balanced scan so the
        # replacement never leaves a dangling ')'
        depth = 0
        i = value_start
        while i < len(call_text):
            c = call_text[i]
            if c in ("'", '"'):
                j = i + 1
                while j < len(call_text):
                    if call_text[j] == "\\":
                        j += 2
                        continue
                    if call_text[j] == c:
                        break
                    j += 1
                i = j + 1
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return m.start(), i + 1
            i += 1
        return None
    m2 = re.match(r"[^,\n()#]+", rest)  # '#' keeps a trailing comment out,
    # '(' keeps a parenthesised sub-expression out of the value span
    if not m2:
        return None
    end = m.end() + m2.end()
    # A value containing '(' right after the matched span (an unquoted
    # expression like v0=-0.4-cos(...)) would be truncated into a
    # mangled fragment by the plain match — refuse to rewrite it.
    if end < len(call_text) and call_text[end:].lstrip().startswith("("):
        warnings.warn(
            f"check_static: refusing to replace {param} — value contains "
            f"parentheses and is not quoted"
        )
        return None
    # keep trailing whitespace OUT of the span so a following comment
    # (``True  # old value``) keeps its separating space after replacement
    while end > m.end() and call_text[end - 1] in " \t":
        end -= 1
    return m.start(), end


def _extract_phi(code: str) -> str:
    """Extract the phi string literal from the setup_optimization call.

    Scoped to the call span so a commented-out ``phi='...'`` example
    above or below the real call can never hijack the value (the value
    feeds the symmetry and solid-region checks, where a wrong phi
    triggers spurious correction rounds)."""
    span = _find_setup_optimization(code)
    if span is None:
        return ""
    call_text = code[span[0]: span[1]]
    arg = _param_span(call_text, "phi")
    if arg is None:
        return ""
    value_text = call_text[arg[0]: arg[1]]
    m = re.search(r"['\"]([^'\"]*)['\"]", value_text[value_text.find("="):])
    return m.group(1) if m else ""


def _setup_param_value(code: str, param: str):
    """The numeric value of a setup_optimization parameter, or None."""
    span = _find_setup_optimization(code)
    if span is None:
        return None
    call_text = code[span[0]: span[1]]
    arg = _param_span(call_text, param)
    if arg is None:
        return None
    text = call_text[arg[0]: arg[1]]
    m = re.search(r"=\s*([-+0-9.eE]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _insert_before_closing(call_text: str, text: str) -> str:
    """Insert *text* as a new trailing argument BEFORE the closing
    parenthesis of a setup_optimization call text."""
    inner = call_text.rstrip()
    body = inner[:-1].rstrip()  # drop the closing paren
    needs_comma = not body.endswith(",")
    base = "\n    " if "\n" in body else " "
    return body + ("," if needs_comma else "") + base + text + ")"


def force_show_weak_material_false(code: str) -> str:
    """Deterministically set ``show_weak_material=False`` in the
    setup_optimization call — inserting it when absent.

    Goes through ``_param_span`` so the replacement touches ONLY the
    parameter's value span: a trailing comment on the old value survives,
    and comment/string mentions inside the call cannot hijack it."""
    span = _find_setup_optimization(code)
    if span is None:
        return code
    start, end = span
    call_text = code[start:end]

    arg = _param_span(call_text, "show_weak_material")
    if arg is not None:
        new_call = (
            call_text[: arg[0]] + "show_weak_material=False" + call_text[arg[1]:]
        )
        return code[:start] + new_call + code[end:]

    # not present — insert before the closing parenthesis
    new_call = _insert_before_closing(call_text, "show_weak_material=False")
    return code[:start] + new_call + code[end:]


def _set_phi(code: str, phi: str) -> str:
    """Replace (or insert) the phi parameter of setup_optimization —
    the deterministic symmetric-phi fallback.  Rule-based enforcement
    only: apart from this fallback, phi is transcribed from the spec
    and never rewritten."""
    span = _find_setup_optimization(code)
    if span is None:
        return code
    call_text = code[span[0]: span[1]]
    arg = _param_span(call_text, "phi")
    value_text = repr(phi)
    if arg is not None:
        new_call = call_text[:arg[0]] + f"phi={value_text}" + call_text[arg[1]:]
    else:
        # parameter absent — insert before the closing parenthesis
        new_call = _insert_before_closing(call_text, f"phi={value_text}")
    return code[:span[0]] + new_call + code[span[1]:]


def strip_ff_comments(text: str) -> str:
    """Remove FreeFEM ``//`` line comments (strings kept intact).

    The generated main and mesh carry NO comments (rule e / item 14) —
    the mesh side is enforced here, deterministically, when mesh.txt is
    written.
    """
    out = []
    quote = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == "\\":
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ======================================================================
# check registry — each check returns a list of error dicts
# ======================================================================


def _check_import(code: str) -> list:
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


def _check_syntax(code: str) -> list:
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


# ======================================================================
# deterministic per-call argument check — every DSL function's arguments
# are validated: count, positional placement (a parameter passed BOTH
# positionally and by keyword is a runtime TypeError), argument types,
# and missing/extra arguments.  This is what turns the BaseUnit /
# duplicate-dirichlet_labels class of runtime crashes into static
# errors the repair loop can actually see.
# ======================================================================

# name -> (parameter names in order, variadic positional allowed,
#          minimum required positional slots, allowed keyword names)
# A keyword NOT in params is checked against the allowed-keyword set
# (empty = no keywords allowed).  ``None`` allowed-keywords means the
# function takes any keyword (skip that part — e.g. setup_optimization,
# which has its own kwargs whitelist check).
_CALL_SIGNATURES = {
    "ScalarField": (
        ("name", "fe_space", "domain", "dirichlet_value", "dirichlet_labels", "unit"),
        False, 3, None,
    ),
    "VectorField": (
        ("name", "fe_space", "domain", "dirichlet_value", "dirichlet_labels", "unit"),
        False, 3, None,
    ),
    "Domain": (("name", "mesh_variable"), True, 2, set()),
    "Constant": (("latex", "value", "description", "unit"), False, 2, None),
    "CharacteristicFunction": (
        ("latex", "strong_value", "weak_value"), False, 3, None,
    ),
    "Parameter": (("name", "expression", "unit"), False, 3, None),
    "Expression": (("latex", "expression"), False, 2, None),
    # EXACTLY five elements: [states], [tests], J, the weak-form
    # expression, check_units=True — the coupling terms of multi-field
    # problems are summed INSIDE the one weak-form expression.
    "Lagrangian": (
        ("states", "tests", "objective", "weak_form"), False, 4, {"check_units"},
    ),
    # method call — receiver varies (lag.*), keywords whitelisted by the
    # dedicated setup-optimization check (exactly name + the 11 kwargs)
    "setup_optimization": (("name",), False, 1, None),
}


def _err_call_args(line: int, func: str, message: str) -> dict:
    return {
        "check": "call_args",
        "severity": "error",
        "message": f"{func}(): {message}",
        "line": line,
    }


def _arg_is_name(node) -> bool:
    return isinstance(node, ast.Name)


def _arg_is_number(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    # unary minus on a number (e.g. -10.0)
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    )


def _arg_is_int(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) \
        and not isinstance(node.value, bool)


def _arg_is_label_list(node) -> bool:
    """int, digit string, or list/tuple of ints (dirichlet_labels form)."""
    if _arg_is_int(node):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str) \
            and node.value.isdigit():
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_arg_is_int(e) or (
            isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.isdigit()
        ) for e in node.elts)
    return False


def _arg_is_str(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _arg_is_fe_space(node) -> bool:
    if _arg_is_str(node):
        return node.value in ("P1", "P2")
    return isinstance(node, ast.Name) and node.id in ("P1", "P2")


def _check_call_args(code: str) -> list:
    """Deterministic argument validation of every known DSL call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # the syntax check reports that — nothing to inspect here
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = None
        if isinstance(node.func, ast.Name):
            if node.func.id in _CALL_SIGNATURES:
                func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) \
                and node.func.attr == "setup_optimization":
            func_name = "setup_optimization"
        if func_name is None:
            continue
        params, variadic, min_pos, allowed_kws = _CALL_SIGNATURES[func_name]
        pos = node.args
        kws = node.keywords
        # conservative: *args/**kwargs splats make the call opaque
        if any(isinstance(a, ast.Starred) for a in pos):
            continue
        if any(k.arg is None for k in kws):  # **kwargs splat
            continue

        # 1. too many positional arguments (fixed-parameter functions)
        if not variadic and len(pos) > len(params):
            errors.append(_err_call_args(
                node.lineno, func_name,
                f"too many positional arguments ({len(pos)} given, "
                f"at most {len(params)} accepted)",
            ))
        # 2. a parameter passed BOTH positionally and by keyword — a
        # runtime TypeError ("got multiple values")
        kw_names = {k.arg for k in kws}
        for i, pname in enumerate(params):
            if i < len(pos) and pname in kw_names:
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    f"argument {pname!r} is passed BOTH positionally "
                    f"(slot {i + 1}) and by keyword — runtime TypeError "
                    f"'got multiple values'",
                ))
        # 3. unknown keyword arguments
        if allowed_kws is not None:
            for k in kw_names:
                if k not in params and k not in allowed_kws:
                    errors.append(_err_call_args(
                        node.lineno, func_name,
                        f"unexpected keyword argument {k!r}",
                    ))
        # 4. missing required arguments
        for i in range(min_pos):
            if i >= len(pos) and params[i] not in kw_names:
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    f"missing required argument {params[i]!r}",
                ))
                break  # one missing-arg error is enough per call
        # 5. argument TYPE checks (literal positions only)
        def _slot(i: int):
            if i < len(pos):
                return pos[i]
            for k in kws:
                if k.arg == params[i]:
                    return k.value
            return None

        if func_name in ("ScalarField", "VectorField"):
            fe = _slot(1)
            if fe is not None and not _arg_is_fe_space(fe):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    f"fe_space must be P1 or P2 (bare or quoted), got "
                    f"{ast.dump(fe)[:40]}",
                ))
            dv = _slot(3)
            if dv is not None and not (_arg_is_number(dv) or _arg_is_str(dv)):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "dirichlet_value must be a value (number/string) — a "
                    "bare name here is almost always a unit misplaced in "
                    "the dirichlet slot (use the unit= keyword instead)",
                ))
            dl = _slot(4)
            if dl is not None and not _arg_is_label_list(dl):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "dirichlet_labels must be a bare label (int or digit "
                    "string) or a list of labels",
                ))
            unit = _slot(5)
            if unit is not None and _arg_is_number(unit):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "unit must be a unit name (meter/kelvin/...), not a number",
                ))
        elif func_name == "Constant":
            latex = _slot(0)
            if latex is not None and not _arg_is_str(latex):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "the first argument must be the latex string",
                ))
        elif func_name == "Lagrangian":
            for si in (0, 1):
                states = _slot(si)
                if states is not None and not isinstance(states, ast.List):
                    errors.append(_err_call_args(
                        node.lineno, func_name,
                        f"{params[si]} must be a list of fields",
                    ))
            # the FIFTH element — check_units=True is mandatory
            cu = None
            for k in kws:
                if k.arg == "check_units":
                    cu = k.value
            if cu is None:
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "check_units=True is required (the fifth element of "
                    "the call)",
                ))
            elif not (isinstance(cu, ast.Constant) and cu.value is True):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "check_units must be True",
                ))
        elif func_name == "setup_optimization":
            first = pos[0] if pos else None
            if first is not None and not _arg_is_name(first):
                errors.append(_err_call_args(
                    node.lineno, func_name,
                    "the first argument must be the problem-name variable",
                ))
    return errors


def _check_lagrangian(code: str, problem_spec: dict = None) -> list:
    """Item 11 — Lagrangian is instantiated; the state/test lists match
    the spec; the third argument is the objective; check_units=True."""
    code_nc = _strip_comments(code)
    m = re.search(r"\bLagrangian\s*\(", code_nc)
    if not m:
        return [
            {
                "check": "lagrangian",
                "severity": "error",
                "message": "Missing Lagrangian instantiation.",
            }
        ]
    errors = []
    call_text = _call_content(code_nc, m)
    args = _top_level_split(call_text) if call_text is not None else []
    if problem_spec:
        ds = problem_spec.get("designed") or {}
        for key, slot, label in (
            ("fields", 0, "state field"),
            ("test_functions", 1, "test function"),
        ):
            names = [
                str(x.get("name") or "").strip()
                for x in (ds.get(key) or [])
                if isinstance(x, dict) and x.get("name")
                and not (key == "fields" and _is_adjoint_field(x))
            ]
            if names and len(args) > slot:
                for name in names:
                    # the list holds VARIABLE names — match the spec
                    # entry through its declared variable (by spec name
                    # or by its latex first argument)
                    x = next(
                        (xx for xx in (ds.get(key) or [])
                         if isinstance(xx, dict)
                         and str(xx.get("name") or "") == name),
                        None,
                    )
                    var = _field_var_name(
                        code_nc, name, str((x or {}).get("latex") or "")
                    ) or name
                    if not re.search(rf"\b{re.escape(var)}\b", args[slot]):
                        errors.append(
                            {
                                "check": "lagrangian",
                                "severity": "error",
                                "message": (
                                    f"{label} '{name}' (variable '{var}') is "
                                    f"missing from the Lagrangian's {key} "
                                    f"list."
                                ),
                            }
                        )
        obj = ds.get("objective_name")
        if not obj and isinstance(ds.get("objective"), dict):
            obj = ds["objective"].get("name")
        if obj and len(args) > 2 and not re.search(
            rf"\b{re.escape(str(obj))}\b", args[2]
        ):
            errors.append(
                {
                    "check": "lagrangian",
                    "severity": "error",
                    "message": (
                        f"the Lagrangian's third argument must be the "
                        f"objective {obj}."
                    ),
                }
            )
    if call_text is not None and "check_units=True" not in call_text:
        errors.append(
            {
                "check": "lagrangian",
                "severity": "error",
                "message": "The Lagrangian must set check_units=True.",
            }
        )
    return errors


def _check_lagrangian_calls(code: str) -> list:
    """The Lagrangian API is EXACTLY two calls — nothing else:

    ``lag = Lagrangian([fields], [test_fields], J, weak_form,
    check_units=True)`` then ``lag.setup_optimization(name, ...)``.
    No second constructor, no second setup call, and NO other method
    calls on the Lagrangian variable (lag.run(...) etc.).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    errors = []

    def _err(message: str) -> dict:
        return {"check": "lagrangian_calls", "severity": "error", "message": message}

    constructor = None  # the single Lagrangian(...) Call node
    assigned_var = None  # the variable it is assigned to
    setup_calls = []  # (Call node, receiver) of setup_optimization(...)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "Lagrangian":
            if constructor is None:
                constructor = node
            else:
                errors.append(_err(
                    f"more than one Lagrangian(...) call — exactly one is "
                    f"allowed (line {node.lineno})"
                ))
        elif isinstance(node.func, ast.Attribute) \
                and node.func.attr == "setup_optimization":
            setup_calls.append((node, node.func.value))

    if constructor is not None:
        # the constructor must be assigned to a single variable
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is constructor:
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    assigned_var = node.targets[0].id
                break
        if assigned_var is None:
            errors.append(_err(
                "Lagrangian(...) must be assigned to a single variable "
                "(lag = Lagrangian(...))"
            ))
        # no OTHER method calls on that variable
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == assigned_var \
                    and node.func.attr != "setup_optimization":
                errors.append(_err(
                    f"unexpected call lag.{node.func.attr}(...) (line "
                    f"{node.lineno}) — the only Lagrangian API calls are "
                    f"lag = Lagrangian(...) and lag.setup_optimization(...)"
                ))

    if not setup_calls:
        errors.append(_err("missing lag.setup_optimization(...) call"))
    elif len(setup_calls) > 1:
        errors.append(_err(
            f"more than one setup_optimization(...) call "
            f"({len(setup_calls)}) — exactly one is allowed"
        ))
    elif assigned_var is not None:
        recv = setup_calls[0][1]
        if not (isinstance(recv, ast.Name) and recv.id == assigned_var):
            errors.append(_err(
                f"setup_optimization must be called on the variable "
                f"assigned from Lagrangian ('{assigned_var}')"
            ))
    return errors


def _check_setup_opt(code: str) -> list:
    """Ensure setup_optimization is called.

    Call-form regex on comment-stripped code, so a comment mentioning
    ``setup_optimization(...)`` cannot satisfy the check for a missing call.
    """
    code_nc = _strip_comments(code)
    if not re.search(r"\bsetup_optimization\s*\(", code_nc):
        return [
            {
                "check": "setup_opt",
                "severity": "error",
                "message": "Missing setup_optimization() call.",
            }
        ]
    return []


# The ONLY keyword arguments setup_optimization() may take — the
# pipeline's canonical call is exactly name + these 11 (nothing else).
# Anything else raises "TypeError: got an unexpected keyword argument"
# at runtime.  The coder prompt asks for the nonlinear solver's
# iteration budget, which landed as ``maxiter=`` on this call in BOTH
# nonlinear demos (the bridge and the cantilever finite-strain case),
# and the Stage 5 repair loop then spent a full LLM-driven round
# repairing it — a static check costs nothing.
_SETUP_OPT_KWARGS = {
    "mesh", "hmin", "hmax", "diffusion", "v0", "iterations", "phi",
    "boundaryLabels", "fixedLabels", "show_weak_material",
}

_SETUP_OPT_KWARG_HINTS = {
    "maxiter": (
        " The nonlinear solver's iteration budget is NOT a "
        "setup_optimization argument — drop it (the pipeline does not "
        "pass solver settings through setup_optimization)."
    ),
}


def _check_setup_opt_kwargs(code: str) -> list:
    """Flags keyword arguments the setup_optimization API does not accept.

    Only REAL calls are scanned (comments stripped) and the argument text
    is split with the shared paren/quote-aware splitter, so a value
    containing commas (``boundaryLabels=['1','2']``) cannot fake a
    keyword.  Positional arguments are skipped — the API's first two
    (name, mesh) are positional by design.
    """
    code_nc = _strip_comments(code)
    errors = []
    for m in re.finditer(r"\bsetup_optimization\s*\(", code_nc):
        call_text = _call_content(code_nc, m)
        if call_text is None:
            continue
        for arg in _top_level_split(call_text):
            km = re.match(r"\s*(\w+)\s*=(?!=)", arg)
            if not km or km.group(1) in _SETUP_OPT_KWARGS:
                continue
            kw = km.group(1)
            errors.append(
                {
                    "check": "setup_opt_kwargs",
                    "severity": "error",
                    "message": (
                        f"setup_optimization() does not accept the keyword "
                        f"argument '{kw}' — the call fails at runtime with "
                        f"'got an unexpected keyword argument'.{_SETUP_OPT_KWARG_HINTS.get(kw, '')}"
                    ),
                }
            )
    return errors


def _check_no_comments(code: str) -> list:
    """The generated code must contain NO comments (rule e / item 14)."""
    lines = _find_comment_lines(code)
    if lines:
        shown = lines[:8]
        return [
            {
                "check": "no_comments",
                "severity": "error",
                "message": (
                    f"the generated code must contain NO comments — '#' "
                    f"comments found at line(s) {shown}"
                    f"{' ...' if len(lines) > 8 else ''}.  Remove them."
                ),
            }
        ]
    return []


def _check_params(code: str) -> list:
    """Deterministic parameter sanity: 0 < hmin < hmax, both finite."""
    hmin = _setup_param_value(code, "hmin")
    hmax = _setup_param_value(code, "hmax")
    if hmin is None or hmax is None:
        return []  # presence/magnitude is the merged review's territory
    if not (math.isfinite(hmin) and math.isfinite(hmax)
            and hmin > 0 and hmax > 0 and hmin < hmax):
        return [
            {
                "check": "params",
                "severity": "error",
                "message": (
                    f"hmin={hmin!r} / hmax={hmax!r} violate "
                    f"0 < hmin < hmax (both finite and positive)."
                ),
            }
        ]
    return []


def _check_phi(code: str, mesh_txt: str = "") -> list:
    """Deterministic phi checks: parses, and is NEGATIVE somewhere on
    the domain box (phi < 0 = solid — a non-negative phi everywhere
    starts from an EMPTY design)."""
    phi = _extract_phi(code)
    if not phi:
        return []  # absence is the merged review's territory
    try:
        expr = _get_sympy().sympify(phi)
    except Exception:
        return [
            {
                "check": "phi_syntax",
                "severity": "error",
                "message": f"phi '{phi}' is not a parseable expression.",
            }
        ]
    errors = []
    box = mesh_box(mesh_txt)
    if box is not None:
        x0, x1, y0, y1 = box
        x, y = _get_sympy().symbols("x y")
        best = None
        for i in range(16):
            for j in range(16):
                px = x0 + (x1 - x0) * i / 15.0
                py = y0 + (y1 - y0) * j / 15.0
                try:
                    v = complex(expr.subs({x: px, y: py})).real
                except Exception:
                    continue
                if math.isnan(v):
                    continue
                best = v if best is None else min(best, v)
        if best is not None and best >= 0:
            errors.append(
                {
                    "check": "phi_solid",
                    "severity": "error",
                    "message": (
                        f"phi is non-negative everywhere on the domain "
                        f"[{x0},{x1}]x[{y0},{y1}] — the design starts EMPTY "
                        f"(phi < 0 = solid).  Use a phi with a solid region."
                    ),
                }
            )
    return errors


def _check_symmetry(code: str, mesh_txt: str = "") -> list:
    """The TWO-PART symmetry check (Initialization.phi_rules.symmetry_issue):

    Part 1 — the INPUT problem: the mesh is analysed; when the problem
    is NOT symmetric the check ends here (no symmetry requirement).
    Part 2 — the RESULT: when the problem IS symmetric, phi must be
    symmetric about the same axes.
    """
    return symmetry_issue(_extract_phi(code), mesh_txt)


def _parse_label_lists(code: str):
    """Extract ``boundaryLabels=[...]`` / ``fixedLabels=[...]`` from *code*.

    Returns ``(code_boundary, code_fixed)`` as sets of stripped label
    strings (the canonical multi-line list format produced by the coder)."""
    code_boundary = set()
    code_fixed = set()
    for kw, target in (("boundaryLabels", code_boundary), ("fixedLabels", code_fixed)):
        m = re.search(rf"{kw}\s*=\s*\[([^\]]+)\]", code)
        if m:
            target.update(
                lbl.strip().strip("'\"") for lbl in m.group(1).split(",") if lbl.strip()
            )
    return code_boundary, code_fixed


def _check_structure(code: str) -> list:
    """Rule b — structure order: the import is the first statement and
    setup_optimization comes AFTER the Lagrangian instantiation."""
    code_nc = _strip_comments(code).lstrip()
    errors = []
    if code_nc and not (code_nc.startswith("from autofreefem")
                        or code_nc.startswith("import")):
        errors.append(
            {
                "check": "structure",
                "severity": "error",
                "message": "The first statement of the code must be the "
                "import (structure order: imports first).",
            }
        )
    lag_pos = re.search(r"\bLagrangian\s*\(", code_nc)
    setup_pos = re.search(r"\bsetup_optimization\s*\(", code_nc)
    if lag_pos and setup_pos and setup_pos.start() < lag_pos.start():
        errors.append(
            {
                "check": "structure",
                "severity": "error",
                "message": "setup_optimization() appears BEFORE the "
                "Lagrangian instantiation — structure order violated "
                "(Lagrangian comes before the optimization setup).",
            }
        )
    return errors


def _check_name(code: str) -> list:
    """Item 2 — the ``name`` variable is defined (autofreefem needs it
    for include filenames)."""
    code_nc = _strip_comments(code)
    if not re.search(r"\bname\s*=\s*(['\"])([^'\"]*)\1", code_nc):
        return [
            {
                "check": "name",
                "severity": "error",
                "message": "The `name` variable is not defined — "
                "autofreefem uses it to generate include filenames.",
            }
        ]
    return []


def _check_spec_iterations(code: str, problem_spec: dict = None) -> list:
    """Rule d — the code's iterations must equal the spec's
    optimization_config.iterations (Initialization designed it)."""
    if not problem_spec:
        return []
    opt = (problem_spec.get("designed") or {}).get("optimization_config")
    if not isinstance(opt, dict):
        return []
    try:
        spec_it = float(opt.get("iterations"))
    except (TypeError, ValueError):
        return []
    code_it = _setup_param_value(code, "iterations")
    if code_it is None or code_it == spec_it:
        return []
    return [
        {
            "check": "spec_iterations",
            "severity": "error",
            "message": (
                f"iterations={code_it} in the code differs from the spec's "
                f"optimization_config.iterations={spec_it} — use the spec value."
            ),
        }
    ]


def _check_spec_mesh_file(code: str) -> list:
    """Rule d — ``mesh='mesh.txt'`` (the spec's mesh_file, hard-coded)."""
    span = _find_setup_optimization(code)
    if span is None:
        return []
    call_text = code[span[0]: span[1]]
    arg = _param_span(call_text, "mesh")
    if arg is None:
        return [
            {
                "check": "spec_mesh_file",
                "severity": "error",
                "message": "setup_optimization() has no mesh= argument — "
                "it must be mesh='mesh.txt' (the spec's mesh_file).",
            }
        ]
    value_text = call_text[arg[0]: arg[1]]
    m = re.search(r"['\"]([^'\"]*)['\"]", value_text)
    mesh_name = m.group(1) if m else ""
    if mesh_name != "mesh.txt":
        return [
            {
                "check": "spec_mesh_file",
                "severity": "error",
                "message": (
                    f"mesh={mesh_name!r} — must be 'mesh.txt' (the spec's mesh_file)."
                ),
            }
        ]
    return []


def _check_spec_labels(code: str, problem_spec: dict = None) -> list:
    """Rule d — boundaryLabels/fixedLabels must match the spec's
    optimization_config EXACTLY (labels are Initialization ground truth);
    label 10 never appears in either list."""
    if not problem_spec:
        return []
    opt = (problem_spec.get("designed") or {}).get("optimization_config")
    if not isinstance(opt, dict):
        return []
    spec_b = {str(x) for x in (opt.get("boundary_labels") or [])}
    spec_f = {str(x) for x in (opt.get("fixed_labels") or [])}
    if not spec_b and not spec_f:
        return []
    code_b, code_f = _parse_label_lists(code)
    errors = []
    for kw, spec, code_l in (
        ("boundaryLabels", spec_b, code_b),
        ("fixedLabels", spec_f, code_f),
    ):
        if code_l != spec:
            errors.append(
                {
                    "check": "spec_labels",
                    "severity": "error",
                    "message": (
                        f"{kw} in the code ({sorted(code_l)}) does NOT match "
                        f"the spec's optimization_config ({sorted(spec)}) — "
                        f"labels are ground truth, copy them EXACTLY."
                    ),
                }
            )
    for kw, code_l in (("boundaryLabels", code_b), ("fixedLabels", code_f)):
        if "10" in code_l:
            errors.append(
                {
                    "check": "spec_labels",
                    "severity": "error",
                    "message": (
                        f"label 10 appears in {kw} — the material boundary "
                        f"never belongs in either list."
                    ),
                }
            )
    return errors


def _check_spec_phi(code: str, problem_spec: dict = None) -> list:
    """Rule d — the code's phi must be the spec's optimization_config.phi
    EXACTLY (Initialization designed it under its phi rules)."""
    if not problem_spec:
        return []
    opt = (problem_spec.get("designed") or {}).get("optimization_config")
    if not isinstance(opt, dict):
        return []
    spec_phi = opt.get("phi")
    if not isinstance(spec_phi, str) or not spec_phi.strip():
        return []
    code_phi = _extract_phi(code)
    if not code_phi:
        return []
    if code_phi.strip() != spec_phi.strip():
        return [
            {
                "check": "spec_phi",
                "severity": "error",
                "message": (
                    f"phi in the code differs from the spec's "
                    f"optimization_config.phi — Initialization designed it, "
                    f"use it EXACTLY (code: {code_phi.strip()[:80]!r}; "
                    f"spec: {spec_phi.strip()[:80]!r})."
                ),
            }
        ]
    return []


def _check_characteristic(code: str, problem_spec: dict = None) -> list:
    """Item 9 — the *contrast* CharacteristicFunction: full LaTeX
    decoration (``\\raisebox{\\depth}{$\\chi$}``), strong_value (the
    larger side) = 1, weak_value (the smaller side) given.

    CharacteristicFunction is also used for material-property fields
    with plain names (``'E'``, ``'k'``, ``r'\\alpha_T'``) — only a LaTeX
    string that actually denotes chi is the contrast function and must
    carry the decoration.  The first argument may be a quoted literal or
    a variable name whose value is assigned in the code; both are
    inspected, so an unquoted variable cannot silently bypass the check.
    """
    errors = []
    spec_has = False
    if problem_spec:
        spec_has = bool((problem_spec.get("designed") or {}).get("characteristic_function"))
    code_nc = _strip_comments(code)

    def _is_full_latex(s: str) -> bool:
        return "raisebox" in s and "\\depth" in s

    calls = []  # (latex, args) pairs
    for m in re.finditer(r"CharacteristicFunction\s*\(", code_nc):
        call_text = _call_content(code_nc, m)
        if call_text is None:
            continue
        args = _top_level_split(call_text)
        latex = ""
        if args:
            a0 = args[0].strip()
            lit = None
            if len(a0) >= 2 and a0[:1] in ("'", '"') and a0[-1:] == a0[:1]:
                lit = a0[1:-1]
            elif len(a0) >= 3 and a0[:1] == "r" and a0[1:2] in ("'", '"') \
                    and a0[-1:] == a0[1:2]:
                lit = a0[2:-1]
            if lit is None:
                # Unquoted-variable first argument: resolve the
                # assignment for the variable name (only look at the
                # code BEFORE the call and require a line-anchored
                # assignment).
                vm = re.fullmatch(r"(\w+)", a0)
                if vm:
                    am = re.search(
                        rf"(?m)^\s*{vm.group(1)}\s*=\s*((?:r)?(['\"]).*?\2)",
                        code_nc[: m.start()],
                    )
                    if am:
                        raw = am.group(1)
                        lit = raw[2:-1] if raw[:1] == "r" else raw[1:-1]
            latex = lit or ""
        calls.append((latex, args))

    chi_calls = [(latex, args) for latex, args in calls if latex and "\\chi" in latex]
    if spec_has and not chi_calls:
        errors.append(
            {
                "check": "chi_declared",
                "severity": "error",
                "message": "The spec defines a characteristic function but "
                "the code declares no chi-carrying CharacteristicFunction.",
            }
        )
        return errors
    for latex, args in chi_calls:
        if not _is_full_latex(latex):
            errors.append(
                {
                    "check": "chi_latex",
                    "severity": "error",
                    "message": f"CharacteristicFunction LaTeX must use "
                    f"r'\\raisebox{{\\depth}}{{$\\chi$}}' not {latex!r}.",
                }
            )
        strong = args[1].strip() if len(args) > 1 else ""
        if strong not in ("1", "1.0"):
            errors.append(
                {
                    "check": "chi_values",
                    "severity": "error",
                    "message": (
                        f"the characteristic function's strong_value (the "
                        f"larger side) must be 1 — got {strong!r}."
                    ),
                }
            )
        if len(args) < 3 or not args[2].strip():
            errors.append(
                {
                    "check": "chi_values",
                    "severity": "error",
                    "message": "the characteristic function's weak_value "
                    "(the smaller side) MUST be given.",
                }
            )
    return errors


def _check_pen(code: str) -> list:
    """Rule d — when ``pen`` is used in the code, its Constant must be
    declared with the spec's value 1e15 (never tgv)."""
    code_nc = _strip_comments(code)
    if not re.search(r"\bpen\b", code_nc):
        return []
    m = re.search(r"\bpen\s*=\s*Constant\s*\(", code_nc)
    if not m:
        return [
            {
                "check": "pen",
                "severity": "error",
                "message": "'pen' is used in the code but no "
                "pen = Constant(...) is declared — declare it with the "
                "spec's value 1e15 (the Dirichlet penalty).",
            }
        ]
    call = _call_content(code_nc, m)
    if call is None:
        return [
            {
                "check": "pen",
                "severity": "error",
                "message": "The pen = Constant(...) call is unparseable.",
            }
        ]
    args = _top_level_split(call)
    value = None
    if len(args) >= 2:
        try:
            value = float(args[1].strip())
        except (TypeError, ValueError):
            value = None
    if value is None:
        vm = re.search(r"\bvalue\s*=\s*([-+0-9.eE]+)", call)
        if vm:
            try:
                value = float(vm.group(1))
            except ValueError:
                value = None
    if value != 1e15:
        return [
            {
                "check": "pen",
                "severity": "error",
                "message": (
                    f"pen = Constant(..., {value!r}) — the Dirichlet penalty "
                    f"must be the spec's value 1e15 (never tgv)."
                ),
            }
        ]
    return []


def _check_domain(code: str, problem_spec: dict = None) -> list:
    """Items 4 + 7 — every Domain( has exactly (name, mesh variable);
    every spec domain must be declared (Boundary objects are no longer
    written — boundary terms use raw mesh labels)."""
    code_nc = _strip_comments(code)
    errors = []
    domain_calls = []  # argument lists of every Domain(...) call
    for m in re.finditer(r"\bDomain\s*\(", code_nc):
        call_text = _call_content(code_nc, m)
        if call_text is None:
            continue
        args = _top_level_split(call_text)
        domain_calls.append(args)
        if len(args) < 2:
            errors.append(
                {
                    "check": "domain",
                    "severity": "error",
                    "message": "Domain(...) needs at least (name, mesh variable).",
                }
            )
            continue
        if not args[0].strip().strip("'\""):
            errors.append(
                {
                    "check": "domain",
                    "severity": "error",
                    "message": "Domain(...) is missing its name.",
                }
            )
        # Boundary objects are no longer written — Domain takes exactly
        # (name, mesh variable); any extra argument is a leftover
        for a in args[2:]:
            errors.append(
                {
                    "check": "domain",
                    "severity": "error",
                    "message": (
                        f"Domain(...) takes exactly (name, mesh variable) "
                        f"— unexpected extra argument {a.strip()!r} "
                        f"(Boundary objects are no longer written; use "
                        f"raw mesh labels in dsx instead)."
                    ),
                }
            )
    # item 7 — every spec domain declared (multi-domain problems)
    if problem_spec:
        doms = (problem_spec.get("designed") or {}).get("domains") or []
        for d in doms:
            if not isinstance(d, dict):
                continue
            dname = str(d.get("name") or "").strip()
            if dname and not any(
                args[0].strip().strip("'\"") == dname for args in domain_calls
            ):
                errors.append(
                    {
                        "check": "domain",
                        "severity": "error",
                        "message": (
                            f"spec domain '{dname}' is not declared in the "
                            f"code — multi-domain problems declare one "
                            f"Domain per region."
                        ),
                    }
                )
    return errors


def _quoted(s: str):
    """The string content of a quoted literal ('...' / "..." / r'...'),
    else None."""
    s = s.strip()
    if len(s) >= 2 and s[:1] in ("'", '"') and s[-1:] == s[:1]:
        return s[1:-1]
    if len(s) >= 3 and s[:1] == "r" and s[1:2] in ("'", '"') and s[-1:] == s[1:2]:
        return s[2:-1]
    return None


def _domain_meshvars(code_nc: str) -> dict:
    """Map Domain name -> its mesh variable, parsed from the code."""
    mapping = {}
    for m in re.finditer(r"\bDomain\s*\(", code_nc):
        call_text = _call_content(code_nc, m)
        if call_text is None:
            continue
        args = _top_level_split(call_text)
        if len(args) >= 2 and _quoted(args[0]):
            mapping[_quoted(args[0])] = _quoted(args[1]) or args[1].strip()
    return mapping


def _values_equal(a, b) -> bool:
    """Numeric-tolerant equality ('0' == '0.') with a string fallback."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _is_adjoint_field(f: dict) -> bool:
    """The ADJOINT state (u_tilde) is auto-derived by the DSL for linear
    problems — the code never declares it, so the declaration check
    skips it (its Dirichlet conditions live in the adjoint weak form,
    the LLM review's territory)."""
    return "tilde" in str(f.get("name") or "").lower() \
        or "\\tilde" in str(f.get("latex") or "")


def _test_latex_candidates(name: str) -> list:
    """LaTeX first-argument candidates for a test-function NAME: the
    source symbol delta_T / δT / Δu corresponds to the code declaration
    ``testT = ScalarField(r'\\delta T', ...)`` — bridge name -> latex.
    A name that is ALREADY latex (starts with a backslash, e.g.
    ``\\delta T``) is its own candidate."""
    s = str(name or "").strip()
    if s.startswith("\\"):
        # the name IS the latex symbol — the declaration's first
        # argument carries it verbatim
        return [s]
    if s.lower().startswith("delta"):
        rest = s[len("delta"):].lstrip("_ ").replace("_", " ")
        return [f"\\delta {rest}"] if rest else []
    if s.startswith(("δ", "Δ")):
        rest = s[1:].lstrip("_ ").replace("_", " ")
        return [f"\\delta {rest}"] if rest else []
    return []


def _norm_latex(s) -> str:
    """Normalise a latex symbol for COMPARISON: collapse the braces of
    single-character sub/superscripts (``\\alpha_{T}`` == ``\\alpha_T``,
    ``E_{0}`` == ``E_0`` — identical rendering)."""
    if not isinstance(s, str):
        return s
    return re.sub(r"([_^])\{([A-Za-z0-9])\}", r"\1\2", s)


def _match_field_call(code_nc: str, name: str, latex: str):
    """The argument list of the field/test declaration matching *name*
    (variable declaration) or *latex* (first argument), else None."""
    if name and name.isidentifier():
        m = re.search(
            rf"\b{re.escape(name)}\s*=\s*(?:ScalarField|VectorField)\s*\(", code_nc
        )
        if m:
            call_text = _call_content(code_nc, m)
            if call_text is not None:
                return _top_level_split(call_text)
    for cand in ([latex] if latex else []) + _test_latex_candidates(name):
        for m in re.finditer(r"\b(?:ScalarField|VectorField)\s*\(", code_nc):
            call_text = _call_content(code_nc, m)
            if call_text is None:
                continue
            args = _top_level_split(call_text)
            if args and _norm_latex(_quoted(args[0])) == _norm_latex(cand):
                return args
    return None


def _field_var_name(code_nc: str, name: str, latex: str):
    """The Python variable name bound to the field/test declaration
    matching *name* or *latex* — for the Lagrangian list check, which
    works on variable names."""
    if name and name.isidentifier() and re.search(
        rf"\b{re.escape(name)}\s*=\s*(?:ScalarField|VectorField)\s*\(", code_nc
    ):
        return name
    for cand in ([latex] if latex else []) + _test_latex_candidates(name):
        for m in re.finditer(r"\b(\w+)\s*=\s*(?:ScalarField|VectorField)\s*\(", code_nc):
            call_text = _call_content(code_nc, m)
            if call_text is None:
                continue
            args = _top_level_split(call_text)
            if args and _norm_latex(_quoted(args[0])) == _norm_latex(cand):
                return m.group(1)
    return None


def _check_fields(code: str, problem_spec: dict = None) -> list:
    """Item 5 — every PHYSICAL spec field is declared with its name,
    fe_space, domain (the spec's domain name or its mesh variable),
    dirichlet value and dirichlet labels.  The adjoint state (u_tilde)
    is skipped — the DSL auto-derives it."""
    if not problem_spec:
        return []
    fields = (problem_spec.get("designed") or {}).get("fields")
    if not isinstance(fields, list):
        return []
    code_nc = _strip_comments(code)
    meshvars = _domain_meshvars(code_nc)
    errors = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        if _is_adjoint_field(f):
            # the ADJOINT field is auto-derived by the DSL — declaring
            # it in the code is itself an error (a duplicate u_tilde
            # would conflict with the auto-derived adjoint)
            name = str(f.get("name") or "").strip()
            if _match_field_call(code_nc, name, str(f.get("latex") or "")) \
                    is not None:
                errors.append(
                    {
                        "check": "fields",
                        "severity": "error",
                        "message": (
                            f"the ADJOINT field '{name}' is declared in the "
                            f"code but the DSL auto-derives it for linear "
                            f"problems — remove its declaration."
                        ),
                    }
                )
            continue
        name = str(f.get("name") or "").strip()
        if not name:
            continue
        args = _match_field_call(code_nc, name, str(f.get("latex") or ""))
        if args is None:
            errors.append(
                {
                    "check": "fields",
                    "severity": "error",
                    "message": (
                        f"field '{name}' is not declared "
                        f"({name} = ScalarField/VectorField(...))."
                    ),
                }
            )
            continue
        fe = str(f.get("fe_space") or "")
        if fe and (len(args) < 2 or args[1].strip().strip("'\"") != fe):
            errors.append(
                {
                    "check": "fields",
                    "severity": "error",
                    "message": f"field '{name}' fe_space must be {fe} "
                    f"(the spec's value).",
                }
            )
        dom = str(f.get("domain") or "")
        if dom and (len(args) < 3
                    or (args[2].strip() != dom
                        and meshvars.get(dom) != args[2].strip())):
            errors.append(
                {
                    "check": "fields",
                    "severity": "error",
                    "message": f"field '{name}' domain must be {dom} "
                    f"(the spec's value, or its mesh variable).",
                }
            )
        dv = f.get("dirichlet_value")
        if dv not in (None, ""):
            code_dv = _quoted(args[3]) if len(args) > 3 else None
            if code_dv is None or not _values_equal(code_dv, dv):
                errors.append(
                    {
                        "check": "fields",
                        "severity": "error",
                        "message": (
                            f"field '{name}' Dirichlet value must be "
                            f"{dv} (the spec's value)."
                        ),
                    }
                )
        dl = f.get("dirichlet_labels")
        if isinstance(dl, list) and dl and args is not None:
            call_text = ",".join(args)
            missing = [str(l) for l in dl if str(l) not in call_text]
            if missing:
                errors.append(
                    {
                        "check": "fields",
                        "severity": "error",
                        "message": (
                            f"field '{name}' Dirichlet labels {missing} "
                            f"are missing from its declaration."
                        ),
                    }
                )
        # a SINGLE Dirichlet label is passed bare (3), never wrapped in
        # a one-element list ([3])
        if args is not None:
            call_text = ",".join(args)
            for lm in re.finditer(r"\[\s*['\"]?(\d+)['\"]?\s*\]", call_text):
                errors.append(
                    {
                        "check": "fields",
                        "severity": "error",
                        "message": (
                            f"field '{name}': a single Dirichlet label "
                            f"must be bare ({lm.group(1)}), not "
                            f"[{lm.group(1)}]."
                        ),
                    }
                )
    return errors


def _check_test_functions(code: str, problem_spec: dict = None) -> list:
    """Item 6 — every spec test function is declared with the tested
    field's fe_space, the same domain, and NO extras (no unit, no
    Dirichlet values)."""
    if not problem_spec:
        return []
    tests = (problem_spec.get("designed") or {}).get("test_functions")
    if not isinstance(tests, list):
        return []
    code_nc = _strip_comments(code)
    meshvars = _domain_meshvars(code_nc)
    errors = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        args = _match_field_call(code_nc, name, str(t.get("latex") or ""))
        if args is None:
            errors.append(
                {
                    "check": "test_functions",
                    "severity": "error",
                    "message": (
                        f"test function '{name}' is not declared "
                        f"({name} = ScalarField/VectorField(...))."
                    ),
                }
            )
            continue
        fe = str(t.get("fe_space") or "")
        if fe and (len(args) < 2 or args[1].strip().strip("'\"") != fe):
            errors.append(
                {
                    "check": "test_functions",
                    "severity": "error",
                    "message": (
                        f"test function '{name}' fe_space must be {fe} — "
                        f"it MUST match the physical field it tests."
                    ),
                }
            )
        dom = str(t.get("domain") or "")
        if dom and (len(args) < 3
                    or (args[2].strip() != dom
                        and meshvars.get(dom) != args[2].strip())):
            errors.append(
                {
                    "check": "test_functions",
                    "severity": "error",
                    "message": f"test function '{name}' domain must be {dom}.",
                }
            )
        if len(args) > 3:
            errors.append(
                {
                    "check": "test_functions",
                    "severity": "error",
                    "message": (
                        f"test function '{name}' carries extra arguments "
                        f"({args[3].strip()}) — test functions take ONLY "
                        f"(name, fe_space, domain), never a unit or "
                        f"Dirichlet values."
                    ),
                }
            )
    return errors


def _check_constants(code: str, problem_spec: dict = None) -> list:
    """Item 8 — every spec constant is declared.  Matching is by the
    Python variable name OR by the constant's first-argument latex (a
    spec name that is a Python reserved word, e.g. "lambda", must be
    declared under another variable like ``lam``).  Values/units are
    covered by the extracted-value check, the pen check and the LLM
    review."""
    if not problem_spec:
        return []
    ds = problem_spec.get("designed") or {}
    consts = ds.get("constants")
    if not isinstance(consts, list):
        consts = (problem_spec.get("extracted") or {}).get("constants")
    if not isinstance(consts, list):
        return []
    code_nc = _strip_comments(code)
    latex_args = set()
    for m in re.finditer(r"\bConstant\s*\(", code_nc):
        call_text = _call_content(code_nc, m)
        if call_text is None:
            continue
        args = _top_level_split(call_text)
        if args:
            lit = _quoted(args[0])
            if lit:
                latex_args.add(_norm_latex(lit))
    errors = []
    for c in consts:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        declared_by_name = bool(
            name.isidentifier()
            and re.search(rf"\b{re.escape(name)}\s*=\s*Constant\s*\(", code_nc)
        )
        latex = str(c.get("latex") or "").strip()
        declared_by_latex = bool(latex and _norm_latex(latex) in latex_args)
        if not (declared_by_name or declared_by_latex):
            errors.append(
                {
                    "check": "constants",
                    "severity": "error",
                    "message": (
                        f"constant '{name}' is not declared "
                        f"({name} = Constant(...), or a Constant whose "
                        f"first argument is its latex symbol)."
                    ),
                }
            )
    return errors


def _check_expressions(code: str, problem_spec: dict = None) -> list:
    """Item 10 — the OBJECTIVE (J) must be declared as an Expression;
    the remaining spec expressions (derivative formulas, derived
    quantities) are embedded in the weak forms and are NOT checked
    here — the LLM review checks their mathematical equivalence."""
    if not problem_spec:
        return []
    ds = problem_spec.get("designed") or {}
    obj = ds.get("objective_name")
    if not obj and isinstance(ds.get("objective"), dict):
        obj = ds["objective"].get("name")
    if not obj:
        for e in (ds.get("expressions") or []):
            if isinstance(e, dict) and str(e.get("name") or "").strip() == "J":
                obj = "J"
                break
    code_nc = _strip_comments(code)
    errors = []
    if obj and not re.search(
        rf"\b{re.escape(str(obj))}\s*=\s*Expression\s*\(", code_nc
    ):
        errors.append(
            {
                "check": "expressions",
                "severity": "error",
                "message": (
                    f"the objective '{obj}' is not declared "
                    f"({obj} = Expression(...))."
                ),
            }
        )
    # (only the objective is hard-checked — the remaining spec
    # expressions, derivative formulas included, are checked for
    # mathematical equivalence by the LLM review)
    return errors


def _check_extracted_constants(code: str, problem_spec: dict = None) -> list:
    """Rule a — every extracted constant's numeric value must appear in
    the code (extracted values are ground truth; a value that appears
    nowhere was dropped or changed)."""
    if not problem_spec:
        return []
    consts = (problem_spec.get("extracted") or {}).get("constants")
    if not isinstance(consts, list):
        return []
    code_nc = _strip_comments(code)
    literals = [
        float(x)
        for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", code_nc)
        if x not in ("", ".")
    ]
    errors = []
    for c in consts:
        if not isinstance(c, dict):
            continue
        name = c.get("name", "?")
        try:
            v = float(c.get("value"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if any(abs(l - v) <= 1e-9 * max(1.0, abs(v)) for l in literals):
            continue
        errors.append(
            {
                "check": "extracted_constants",
                "severity": "error",
                "message": (
                    f"extracted constant {name} = {v} does not appear in "
                    f"the code — extracted values are GROUND TRUTH: declare "
                    f"and use them exactly."
                ),
            }
        )
    return errors


def _check_setup_args(code: str) -> list:
    """Item 12 — the first positional argument is ``name`` and every
    required parameter is present (content correctness is the other
    spec-driven checks' job).  The canonical call carries EXACTLY name
    + the 11 parameters below — nothing else (the kwargs whitelist
    check flags extras)."""
    span = _find_setup_optimization(code)
    if span is None:
        return []
    call_text = code[span[0]: span[1]]
    # drop the leading ``setup_optimization(`` and the closing ``)`` so
    # the top-level split sees the ARGUMENTS only
    call_text = call_text[call_text.index("(") + 1: call_text.rindex(")")]
    args = _top_level_split(call_text)
    errors = []
    if not args or args[0].strip() != "name":
        errors.append(
            {
                "check": "setup_args",
                "severity": "error",
                "message": (
                    "setup_optimization's first positional argument must "
                    "be the `name` variable."
                ),
            }
        )
    for param in ("mesh", "hmin", "hmax", "diffusion", "v0",
                  "iterations", "phi", "boundaryLabels", "fixedLabels",
                  "show_weak_material"):
        if _param_span(call_text, param) is None:
            errors.append(
                {
                    "check": "setup_args",
                    "severity": "error",
                    "message": f"setup_optimization() is missing {param}=.",
                }
            )
    return errors


def _check_mesh_defined(code: str, mesh_txt: str = "") -> list:
    """Item 13 — the mesh file defines a mesh (``mesh Th = ...``);
    label/resolution adequacy is the Initialization spec's job and the
    LLM review's."""
    if not (mesh_txt or "").strip():
        return [
            {
                "check": "mesh_defined",
                "severity": "error",
                "message": "The mesh file (mesh.txt) is empty.",
            }
        ]
    if not re.search(r"\bmesh\s+\w+\s*=", mesh_txt):
        return [
            {
                "check": "mesh_defined",
                "severity": "error",
                "message": (
                    "The mesh file does not define a mesh — no "
                    "`mesh Th = ...` statement found."
                ),
            }
        ]
    return []


def _check_mesh_region3(code: str, mesh_txt: str = "") -> list:
    """Item 13 — a mesh whose BORDERS carry label 10 (manually-defined
    pre-cut holes) must mark the hole as region 3 (void) right after
    the mesh definition: ``int[int] re = [0, 3]; Th = change(Th,
    reft=re);``."""
    if not mesh_txt:
        return []
    if not re.search(r"label\s*=\s*10\b", mesh_txt):
        return []
    if not re.search(r"\bchange\s*\(\s*Th\s*,\s*reft\s*=\s*re\s*\)", mesh_txt):
        return [
            {
                "check": "mesh_region3",
                "severity": "error",
                "message": (
                    "the mesh defines label-10 hole borders but does NOT "
                    "mark the hole as region 3 — add right after the mesh "
                    "definition: int[int] re = [0, 3]; "
                    "Th = change(Th, reft=re);"
                ),
            }
        ]
    return []


# ── registry ──────────────────────────────────────────────────────────

_CHECKS = [
    _check_import,             # item 1
    _check_syntax,
    _check_call_args,          # deterministic per-call argument validation
    _check_structure,
    _check_name,               # item 2
    _check_domain,             # items 4 + 7
    _check_fields,             # item 5
    _check_test_functions,     # item 6
    _check_constants,          # item 8
    _check_extracted_constants,  # item 8
    _check_pen,                # item 8 (giant penalty)
    _check_characteristic,     # item 9
    _check_expressions,        # item 10
    _check_lagrangian,         # item 11
    _check_lagrangian_calls,   # item 11 (exactly the two Lagrangian calls)
    _check_setup_opt,          # item 12
    _check_setup_opt_kwargs,   # item 12 (never maxiter)
    _check_setup_args,         # item 12
    _check_params,             # item 12
    _check_spec_iterations,    # item 12
    _check_spec_mesh_file,     # item 12
    _check_spec_labels,        # item 12
    _check_spec_phi,           # item 12
    _check_phi,                # item 12 (phi sanity)
    _check_symmetry,           # phi (two-part)
    _check_mesh_defined,       # item 13
    _check_mesh_region3,       # item 13 (pre-cut holes -> region 3)
    _check_no_comments,        # item 14
]


def static_check(code: str, problem_spec: dict = None, mesh_txt: str = "") -> list:
    """
    Run all deterministic static checks on *code* and return a list of
    error dicts.

    Each error dict has keys: ``check``, ``severity``, ``message``,
    and optionally ``line``.

    Parameters
    ----------
    code : str
        The generated AutoFreeFEM Python source.
    problem_spec : dict or None
        Output of ``parse_problem`` (the spec-driven checks).
    mesh_txt : str
        The mesh file content (phi domain box, symmetry detection).

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
            if "problem_spec" in sig.parameters:
                kwargs["problem_spec"] = problem_spec
            if "mesh_txt" in sig.parameters:
                kwargs["mesh_txt"] = mesh_txt
            all_errors.extend(check_fn(code, **kwargs))
        except Exception as e:
            warnings.warn(f"Static check {check_fn.__name__} raised: {e}")

    # Sort: errors first, then warnings
    all_errors.sort(key=lambda e: 0 if e["severity"] == "error" else 1)
    return all_errors


# ======================================================================
# merged LLM review (source annotations, structure, parameters, phi, ...)
# ======================================================================

_JSON_REPAIR_SYSTEM = """\
Your previous answer was NOT valid JSON.  Return THE SAME CONTENT —
unchanged in meaning — as exactly ONE valid JSON object.
No markdown, no code fences, no explanations.
"""

_JSON_REPAIR_USER = """\
The previous (invalid) answer:
{raw_text}

Return the same content as one valid JSON object.
"""

_REVIEW_SYSTEM = """\
You are the reviewer of a generated AutoFreeFEM main.py.  You receive the
problem's two specs (initial.json + mesh.json), the mesh file, the
problem description, the selected reference examples, and the generated
code.  Initialization has ALREADY designed everything in the specs
(labels, phi, units, weak forms, parameters); the code must TRANSCRIBE
the spec, never re-design it.  The deterministic checks hard-check the
mechanical parts of the 14-item checklist below — you review the
SEMANTIC parts (same numbering) and return only issues (the pipeline
repairs them).

## Review checklist

1. IMPORTS — hard-checked.  (Nothing to review.)
2. NAME — hard-checked (defined).  Review: the name string reflects
   the problem (spaces, no underscores preferred).
3. BOUNDARY — REMOVED: Boundary() objects are never written — boundary
   terms in the weak form and BCs use RAW mesh labels directly.  If a
   Boundary(...) call appears anywhere, report it (the repair must
   remove it and use the raw label in dsx instead).
4. DOMAIN — shape hard-checked.  Review: multi-domain
   problems declare one Domain per spec region with the right mesh
   variables.
5. FIELDS — declaration/fe_space/domain/dirichlet hard-checked.
   Review: the UNIT argument matches the spec's unit (AutoFreeFEM
   CAPITALIZED syntax — Newton, Watt, Pascal, meter, kelvin;
   dimensionless '-'), and the latex names match the spec; the
   ADJOINT field (u_tilde) is auto-derived by the DSL — declaring it
   in the code is an issue.
6. TEST FUNCTIONS — hard-checked (fe_space matching the tested field,
   no extras).  (Nothing more to review.)
7. MULTI-DOMAIN — declarations hard-checked.  Review: the interface
   CONTINUITY conditions (value / normal-flux) are implemented between
   the domains per the spec's interfaces.
8. CONSTANTS — declarations/extracted values/pen hard-checked.
   Review: designed values are used as-is or refined ONLY when the
   code could not run otherwise; descriptions and units match the
   spec; a pen penalty exists ONLY where the spec's weak forms carry
   it (the FreeFEM workaround for multi-valued Dirichlet) — its value
   is 1e15, never tgv, never added by the code on its own.
9. CHARACTERISTIC FUNCTION — hard-checked (full latex decoration,
   strong=1, weak given).  Review: the weak value matches the spec's
   weak_value ('1/100' solid mechanics, '1/10' pure thermal).
10. EXPRESSIONS — declarations hard-checked.  Review: every formula
    (including J) is MATHEMATICALLY EQUIVALENT to the spec's formula
    (equivalent notation counts, textual identity does not); no
    placeholders, no dropped terms; the objective J matches the
    spec's objective_name/description.
11. LAGRANGIAN — structure hard-checked (lists/objective/check_units).
    Review: the weak forms implement the spec's Lagrangian (objective,
    state, adjoint, shape derivative) with the spec's SIGNS and
    concrete boundary labels; Dirichlet conditions come from the
    FIELD declarations; pen penalty terms appear ONLY where the
    spec's weak forms carry them.
12. SETUP_OPTIMIZATION — hard-checked (name/mesh='mesh.txt'/iterations/
    phi/labels equal the spec EXACTLY; all parameters present; no
    illegal kwargs; hmin<hmax).  Parameter MAGNITUDES are NOT
    reviewed — the spec's designed values are used as-is.
    show_weak_material must be False (enforced deterministically).
13. MESH — existence and the region-3 mark hard-checked (label-10
    hole borders require ``int[int] re = [0, 3]; Th = change(Th,
    reft=re);`` right after the mesh definition).  Review against the
    spec's mesh: the resolution n resolves the geometric features
    (several elements per feature), the labels match the spec's edges
    (names/roles), and the code only uses labels the mesh has.
14. COMMENTS — hard-checked.  (Nothing to review.)
Return ONE JSON:
{{"ok": true, "issues": []}}
or
{{"ok": false, "issues": [{{"item": "<item number:what>",
"severity": "error"|"warning", "problem": "<one sentence>"}}]}}

No markdown.  No code fences.
"""

_REVIEW_USER = """\
Problem description (the paper — source of extracted values):
{problem_description}

Specs (initial.json + mesh.json):
{specs_json}

Mesh file (the real domain — check hmin/hmax against it):
{mesh_txt}

Current main.py:
```python
{code}
```

Selected reference examples (parameter ranges to consult):
{examples_text}

Return the review JSON.
"""


def _review(agent, code, problem_spec, mesh_spec, problem_description,
            mesh_txt, examples) -> dict:
    """The merged LLM review — fail-open: a broken reviewer answer
    blocks nothing (the deterministic checks still gate the loop)."""
    examples_text = ""
    for exd in examples or []:
        if isinstance(exd, dict):
            examples_text += f"=== EXAMPLE: {exd.get('name', '?')} ===\n{exd.get('py', '')}\n\n"
    specs_json = json.dumps(
        {"initial": problem_spec or {}, "mesh": mesh_spec or {}},
        indent=2, ensure_ascii=False,
    )
    agent.set_prompt_list([
        {"role": "system", "content": _REVIEW_SYSTEM},
        {"role": "user", "content": _REVIEW_USER},
    ])
    payload = {
        "problem_description": problem_description or "(not available)",
        "specs_json": specs_json,
        "mesh_txt": mesh_txt or "(not available)",
        "code": code,
        "examples_text": examples_text or "(no examples selected)",
    }
    # A broken reviewer answer is redone up to configs.MAX_GENERATE_RETRIES
    # times, then fail-open (the deterministic checks still gate the loop).
    last_raw = ""
    for _attempt in range(configs.MAX_GENERATE_RETRIES + 1):
        raw = agent.invoke(payload)
        last_raw = raw
        try:
            result = json.loads(parser_json(raw))
        except Exception as e:
            warnings.warn(f"check_static review JSON parse failed (fail-open): {e}")
            continue
        if not isinstance(result, dict):
            warnings.warn(
                f"check_static review returned non-object JSON "
                f"({type(result).__name__}) — fail-open."
            )
            continue
        return result
    # ONE JSON-repair round on the LAST invalid text — the model must
    # return the same content as valid JSON; only then fail open
    if last_raw:
        try:
            set_label(agent, "Generation/run_static_repair (review JSON repair)")
            agent.set_prompt_list([
                {"role": "system", "content": _JSON_REPAIR_SYSTEM},
                {"role": "user", "content": _JSON_REPAIR_USER},
            ])
            repaired = agent.invoke({"raw_text": last_raw})
            result = json.loads(parser_json(repaired))
            if isinstance(result, dict):
                return result
        except Exception as e:
            warnings.warn(f"check_static review JSON repair failed (fail-open): {e}")
    return {}


# ======================================================================
# targeted repair via LLM (fix only, never rewrite)
# ======================================================================

_REPAIR_SYSTEM = """\
You are an expert AutoFreeFEM code repairer.  You receive a generated
main.py and a list of issues found by the checks and the review.  Your
job is to FIX ONLY the reported issues while keeping the code otherwise
EXACTLY as it is — do not refactor, restructure, or "improve" unrelated
parts, and do not regenerate the code.

Initialization has ALREADY designed everything in the specs (labels,
phi, units, weak forms, parameters).  The code must TRANSCRIBE the
spec, never re-design it:
- Values the spec marks extracted are GROUND TRUTH — never change them.
- Labels, phi, iterations and the mesh filename come from the spec's
  optimization_config — fix the CODE to match the spec, never the
  other way (phi must be the spec's phi EXACTLY — never replace or
  rewrite it).  The two lists are the THREE-way boundary treatment:
  fixedLabels = FROZEN, boundaryLabels = NORMAL-LOCKED, a boundary in
  NEITHER list = FREE — two empty lists mean every non-10 boundary is
  FREE; never expand a FREE boundary into an explicit list.
- Dirichlet conditions come from the FIELD declarations; pen penalty
  terms appear ONLY where the spec's weak forms carry them (the
  FreeFEM workaround for multi-valued Dirichlet) — fix the code to
  match the spec, never add or remove a penalty on your own; the
  penalty value is 1e15, never tgv.
- The code must contain NO comments.
- NEVER repeat a mistake already made in a previous round — the repair
  history in the user message lists the earlier errors and what each
  earlier repair removed / added / modified.  Learn from it and change
  strategy accordingly.
- PARTIAL REPAIR: fix the issues you CAN fix; when you are NOT SURE
  how to fix an issue correctly, LEAVE it UNCHANGED rather than
  guessing — the next round will address it.  Issues the repair history
  marks FAILED were not fixed by the previous repair: fix them now with
  a DIFFERENT approach.  Issues marked FIXED must STAY fixed (never
  reintroduce them).

## Output

Return exactly one JSON object:
{{"main": "<complete corrected Python code>"}}
No markdown.  No code fences.  No extra text.
"""

_REPAIR_USER = """\
Current code:
{code}

Issues to fix (severity | check | message):
{issues_text}

Repair history:
{history_text}

Return the corrected code JSON.
"""


def _repair(agent, code: str, errors: list, history_text: str) -> str | None:
    """LLM targeted repair; None when the answer is unusable (the round
    is consumed and the current code carries into the next round)."""
    issues_text = "\n".join(
        f"[{e.get('severity', 'error')}] {e.get('check', 'issue')}: {e.get('message', '')}"
        for e in errors
    )
    agent.set_prompt_list([
        {"role": "system", "content": _REPAIR_SYSTEM},
        {"role": "user", "content": _REPAIR_USER},
    ])
    raw = agent.invoke(
        {
            "code": code,
            "issues_text": issues_text,
            "history_text": history_text or "(no previous repairs)",
        }
    )
    try:
        parsed = json.loads(parser_json(raw))
    except Exception as e:
        warnings.warn(f"check_static repair JSON parse failed: {e}")
        return None
    if not isinstance(parsed, dict):
        warnings.warn(
            f"check_static repair returned non-object JSON "
            f"({type(parsed).__name__})."
        )
        return None
    main = parsed.get("main")
    if not isinstance(main, str) or not main.strip():
        warnings.warn("check_static repair returned no usable main code.")
        return None
    return main


# ======================================================================
# repair history (kept in memory, and recorded in full into history.txt
# via the history_log parameter — same maintenance style as Initialization)
# ======================================================================


def _diff_code(prev: str, curr: str, cap: int = 8) -> dict:
    """Compact line diff between two code versions.

    Returns ``{"removed": [...], "added": [...], "modified": [...]}`` —
    each list capped to *cap* entries; modified entries are shown as
    ``"old line -> new line"`` pairs."""
    removed, added, modified = [], [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, prev.splitlines(), curr.splitlines()
    ).get_opcodes():
        a = prev.splitlines()[i1:i2]
        b = curr.splitlines()[j1:j2]
        if tag == "delete":
            removed.extend(a)
        elif tag == "insert":
            added.extend(b)
        elif tag == "replace":
            modified.extend(f"{x} -> {y}" for x, y in zip(a, b))
    return {
        "removed": removed[:cap],
        "added": added[:cap],
        "modified": modified[:cap],
    }


def _history_text(history: list) -> str:
    """One compact block per previous repair round: errors, what the
    round FIXED, what it FAILED to fix, and the changes."""
    lines = []
    for h in history:
        errs = "; ".join(str(e)[:100] for e in h["errors"][:5])
        fixed = "; ".join(str(x)[:60] for x in h.get("fixed", [])[:4])
        failed = "; ".join(str(x)[:60] for x in h.get("failed", [])[:4])
        lines.append(
            f"round {h['round']}: errors=[{errs}] fixed=[{fixed}] "
            f"failed=[{failed}] removed={h['removed'][:8]} "
            f"added={h['added'][:8]} modified={h['modified'][:8]}"
        )
    return "\n".join(lines) if lines else "(no previous rounds)"


# ======================================================================
# deterministic post-passes (applied even when the loop fails)
# ======================================================================


def _apply_symmetry_fallback(code: str, mesh_txt: str) -> str:
    """When the problem IS symmetric and phi is still not, replace phi
    with the canonical symmetric pattern (the two-part check's final
    deterministic fallback)."""
    axes = detect_symmetry(mesh_txt)
    if not axes:
        return code
    L, H = mesh_dims(mesh_txt)
    if L is None or H is None:
        return code
    phi = _extract_phi(code)
    if not phi or is_phi_symmetric(phi, L, H, axes):
        return code
    warnings.warn(
        f"check_static: phi still asymmetric for axes {axes} — using "
        f"the canonical symmetric phi."
    )
    return _set_phi(code, canonical_symmetric_phi(L, H))


# ======================================================================
# main entry point — the check -> review -> repair loop
# ======================================================================


def run_static_repair(
    agent,
    code: str,
    problem_spec: dict = None,
    mesh_spec: dict = None,
    problem_description: str = "",
    mesh_txt: str = "",
    examples: list = None,
    max_fixes: int = None,
    history_log=None,
) -> dict:
    """Check -> review -> targeted repair loop for the generated main.py.

    The code is generated ONCE by the coder; this loop only REPAIRS the
    issues reported by each round (deterministic checks + merged LLM
    review), never rewrites the code from scratch.  At most
    ``configs.MAX_STATIC_FIXES`` repairs are attempted.

    A repair history is maintained exactly like the Initialization spec
    loop: every round records its errors and what it removed / added /
    modified relative to the previous code, and the repair prompt
    forbids repeating a mistake already listed there.  When
    *history_log* is given, each round's reported issues are also
    appended as full records to that operation log
    (results/<problem>/history.txt).

    Deterministic enforcement runs before every round and at the end
    (never left to the LLM): ``show_weak_material=False``, comment
    stripping (rule e / item 14), and the canonical symmetric-phi
    fallback for a still-asymmetric phi on a symmetric problem.

    Returns ``{"main", "_passed", "history", "warnings"}`` — ``main`` is
    the last state (repaired when the loop passed, best-effort when the
    budget was exhausted); ``history`` the in-memory repair record.
    """
    max_fixes = configs.MAX_STATIC_FIXES if max_fixes is None else int(max_fixes)
    history = []
    passed = False
    final_warnings = []

    for attempt in range(max_fixes + 1):
        code = force_show_weak_material_false(code)
        det = static_check(code, problem_spec, mesh_txt)
        round_label = f"Generation/run_static_repair round {attempt + 1}/{max_fixes + 1}"
        set_label(agent, f"{round_label} (review)")
        review = _review(
            agent, code, problem_spec, mesh_spec, problem_description,
            mesh_txt, examples,
        )
        issues = (review.get("issues") or []) if isinstance(review, dict) else []
        # keep only dict entries that actually say something (an empty
        # {"severity": "warning"} must not surface as "[W] None: None")
        issues = [i for i in issues if isinstance(i, dict)
                  and (i.get("item") or i.get("problem"))]
        # review issues carry item/problem but no "message" — synthesise
        # a readable message so the console AND the repair prompt never
        # feed the repair LLM an empty issue line
        for i in issues:
            if not i.get("message"):
                item = str(i.get("item") or "")
                problem = str(i.get("problem") or "")
                i["message"] = f"{item}: {problem}".strip(": ")
        errors = [e for e in det if e.get("severity") == "error"]
        errors += [i for i in issues if i.get("severity") == "error"]
        final_warnings = [e for e in det if e.get("severity") != "error"]
        final_warnings += [i for i in issues if i.get("severity") != "error"]
        if not errors:
            passed = True
            break
        if attempt == max_fixes:
            break
        print(f"  [static repair {attempt + 1}/{max_fixes}] "
              f"{len(errors)} error(s) — repairing", flush=True)
        for e in errors[:6]:
            print(f"    - {e.get('severity', '?')} | {e.get('check', 'issue')} | "
                  f"{str(e.get('message', ''))[:140]}", flush=True)
        # full record of the round's INPUT to the repair (the reported
        # issues) — the LLM record above carries the prompt values
        if history_log is not None:
            record_history(
                history_log,
                "run_static_repair (deterministic checks)",
                label=round_label,
                output_text="\n".join(
                    f"- {e.get('severity', '?')} | {e.get('check', 'issue')} | "
                    f"{e.get('message', '')}"
                    for e in errors
                ),
            )
        prev = code
        set_label(agent, f"{round_label} (repair)")
        fixed = _repair(agent, code, errors, _history_text(history))
        if fixed is None:
            # a garbage repair answer is NOT retried — it consumes the
            # round (at most max_fixes rounds total) and the next round
            # re-reports the same errors with this failure in the history
            print("  [static repair] repair answer unusable — moving to "
                  "the next round", flush=True)
            history.append(
                {
                    "round": attempt + 1,
                    "errors": [
                        f"{e.get('check', 'issue')}: {str(e.get('message', ''))[:160]}"
                        for e in errors
                    ],
                    "fixed": [],
                    "failed": [
                        f"{e.get('check', 'issue')}: {str(e.get('message', ''))[:160]}"
                        for e in errors if e.get("check")
                    ],
                    "unverified": [
                        f"{e.get('item', 'issue')}: {str(e.get('problem', ''))[:160]}"
                        for e in errors if not e.get("check")
                    ],
                    "removed": [], "added": [], "modified": [],
                }
            )
            continue
        code = fixed
        # PARTIAL REPAIR — classify which of the round's errors the
        # repair actually fixed: re-run the deterministic checks on the
        # repaired code; an error still present is recorded FAILED (it is
        # skipped, not blocking — the next round addresses it), a
        # disappeared one FIXED.  LLM-review issues cannot be re-verified
        # deterministically and are recorded UNVERIFIED (the next review
        # re-reports them if they persist).
        after = static_check(code, problem_spec, mesh_txt)
        after_keys = {(e.get("check"), e.get("message")) for e in after}
        fixed_list, failed_list, unverified = [], [], []
        for e in errors:
            if e.get("check"):
                label = f"{e.get('check')}: {str(e.get('message', ''))[:140]}"
                if (e.get("check"), e.get("message")) in after_keys:
                    failed_list.append(label)
                else:
                    fixed_list.append(label)
            else:
                unverified.append(
                    f"{e.get('item', 'issue')}: {str(e.get('problem', ''))[:140]}"
                )
        history.append(
            {
                "round": attempt + 1,
                "errors": [
                    f"{e.get('check', 'issue')}: {str(e.get('message', ''))[:160]}"
                    for e in errors
                ],
                "fixed": fixed_list,
                "failed": failed_list,
                "unverified": unverified,
                **_diff_code(prev, code),
            }
        )

    # ── deterministic final enforcement (even on failure) ────────────
    code = force_show_weak_material_false(code)
    code = _strip_comments(code)  # rule e / item 14: main carries no comments
    code = _apply_symmetry_fallback(code, mesh_txt)

    return {
        "main": code,
        "_passed": passed,
        "history": history,
        "warnings": final_warnings,
    }


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
    print(f"Static Check & Repair — {len(py_files)} generated files")
    print(f"{'=' * 70}")

    for py_file in py_files:
        name = py_file.parent.name
        code = py_file.read_text(encoding="utf-8", errors="replace")

        problem_spec = None
        spec_file = py_file.parent / "initial.json"
        if spec_file.exists():
            problem_spec = json.loads(spec_file.read_text(encoding="utf-8"))

        mesh_spec = None
        mesh_json = py_file.parent / "mesh.json"
        if mesh_json.exists():
            mesh_spec = json.loads(mesh_json.read_text(encoding="utf-8"))

        mesh_txt = ""
        mesh_file = py_file.parent / "mesh.txt"
        if mesh_file.exists():
            mesh_txt = mesh_file.read_text(encoding="utf-8", errors="replace")

        print(f"\n  [{name}]", end=" ", flush=True)
        result = run_static_repair(
            agent, code,
            problem_spec=problem_spec,
            mesh_spec=mesh_spec,
            mesh_txt=mesh_txt,
        )
        history = result.get("history") or []
        if result.get("_passed"):
            print(f"OK ({len(history)} repair(s))")
        else:
            print(f"FAILED — repair budget exhausted ({len(history)} repair(s))")
        for w in result.get("warnings") or []:
            print(f"    [W] {w.get('check')}: {w.get('message')}")

    print("\nDone.")
