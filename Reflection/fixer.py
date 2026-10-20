"""
Fix strategies for the Reflection module.

Fix types (the classification side has 5 error categories):
- syntax_fix   — reuses Generation's rule set (``run_static_repair`` in
  ``Generation/check_static.py``); the reflector calls it directly.
- formula_fix  — formula-related code (intermediate formulas, J, the
  Lagrangian formulas, nonlinear formulas).  Only DESIGNED formula
  parts may change.
- design_fix   — design VALUES: physical-parameter numbers, Afac (ell),
  CharacteristicFunction values, phi, the three-way boundary labels,
  boundary physical properties (including the ScalarField/VectorField
  Dirichlet conditions and the pen penalty implementation — pen's
  EXISTENCE is a design matter, its VALUE is the fixed convention
  1e15), and the mesh border definitions (the mesh part only when the
  mesh is NOT a provided/frozen one).  The boundary is DETERMINISTIC:
  ``design_enforce`` filters the response.
- fe_param_fix — hmin/hmax/diffusion/v0, the mesh resolution, and the
  field precision (P1/P2).  ``fe_param_enforce`` filters the response.
- free_fix     — single free-text repair (the fallback).

All modes return {"main": ..., "mesh": ..., "rationale": ...} (or
{"_error": ...} on parse failure).
"""

import ast
import json
import math
import re
import warnings

from utils import parser_json

from .classifier import (
    _constant_replace,
    _constant_value,
    _extracted_constant_names,
    _source_of,
)

# The setup_optimization parameters the fe_param_fix may write.
_FE_PARAM_KEYS = ("hmin", "hmax", "diffusion", "v0")

# "no real change" tolerance for the enforcers' value comparisons
# (relative, with an absolute floor in the chi weak-value check)
_VALUE_CHANGE_EPS = 1e-15


def _match_field_call(code_nc: str, name: str, latex: str):
    """The POSITIONAL argument texts of the ScalarField/VectorField
    declaration matching *name* (the bound variable) or *latex* (the
    first argument), else None.

    Local reimplementation of the helper formerly imported from
    Generation.check_static (removed there with the declaration
    hard-checks).  Only build_design_permission uses it.
    """
    if not (name or latex):
        return None
    try:
        tree = ast.parse(code_nc)
    except SyntaxError:
        return None
    candidates = {str(latex)} if latex else set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if getattr(call.func, "id", None) not in ("ScalarField", "VectorField"):
            continue
        args = call.args
        if not args:
            continue
        target = node.targets[0]
        if name and isinstance(target, ast.Name) and target.id == name:
            return [ast.unparse(a) for a in args]
        first = args[0]
        if candidates and isinstance(first, ast.Constant) \
                and isinstance(first.value, str) and first.value in candidates:
            return [ast.unparse(a) for a in args]
    return None


def apply_param_updates(code: str, updates: dict) -> str:
    """Mechanically write parameter VALUES back into the
    setup_optimization call.

    *updates* maps allowed parameter names (hmin/hmax/diffusion/v0 —
    the fe parameters, plus phi for the design fix) to their new
    values.  Everything else in the program is untouched by
    construction — the LLM response is only the carrier of these
    values."""
    from Generation.check_static import (  # lazy — keeps module loading light
        _find_setup_optimization,
        _insert_before_closing,
        _param_span,
    )
    for param in ("hmin", "hmax", "diffusion", "v0", "phi"):
        if param not in updates:
            continue
        value = updates[param]
        # type gate: numeric parameters must be finite numbers,
        # phi must be a non-empty string
        if param == "phi":
            if not isinstance(value, str) or not value.strip():
                warnings.warn(f"fixer: ignoring non-string/empty phi value: {value!r}")
                continue
            value_text = repr(value)
        else:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                warnings.warn(f"fixer: ignoring non-numeric {param} value: {value!r}")
                continue
            value_text = str(value)

        span = _find_setup_optimization(code)
        if span is None:
            return code
        call_text = code[span[0]:span[1]]
        arg_span = _param_span(call_text, param)
        if arg_span is not None:
            new_call = call_text[:arg_span[0]] + f"{param}={value_text}" + call_text[arg_span[1]:]
            code = code[:span[0]] + new_call + code[span[1]:]
        else:
            new_call = _insert_before_closing(call_text, f"{param}={value_text}")
            code = code[:span[0]] + new_call + code[span[1]:]
    return code


# ======================================================================
# permission builders (deterministic, from the spec's source annotations)
# ======================================================================


def build_design_permission(spec: dict, main_py: str, mesh_frozen: bool) -> dict:
    """The design_fix permission set.

    ``{"constants": {PyName: current}, "phi": str|None,
    "labels": {"boundary_labels": set, "fixed_labels": set},
    "chi": bool, "mesh": bool, "pen": bool,
    "dirichlet": {field: {"name", "latex", "value", "value_editable",
    "labels", "labels_editable"}}}`` — only values whose spec source is
    DESIGNED are included (extracted values are ground truth).  The pen
    VALUE is fixed content (the project convention 1e15 — never a
    tuning target); only its EXISTENCE (the penalty implementation) is
    a design matter, granted when a boundary or a Dirichlet condition
    is designed.
    """
    permission = {
        "constants": {}, "phi": None, "labels": {}, "chi": False,
        "mesh": False, "pen": False, "dirichlet": {},
    }
    if not isinstance(spec, dict):
        return permission
    designed = spec.get("designed") or {}
    opt = designed.get("optimization_config") or {}

    extracted_names = _extracted_constant_names(spec)
    for c in designed.get("constants") or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("source", "designed")) != "designed":
            continue
        name = c.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in extracted_names or c.get("latex") in extracted_names:
            continue
        if name.lower() == "pen":
            # pen's VALUE is the fixed project convention 1e15 — only its
            # existence may be touched (the pen flag below)
            permission["pen"] = True
            continue
        current = _constant_value(main_py, name)
        if current is not None:
            permission["constants"][name] = current
    # Afac is a design value when ell is not extracted
    if not (extracted_names & {"ell", "Afac", "\\ell"}):
        current = _constant_value(main_py, "Afac")
        if current is not None:
            permission["constants"]["Afac"] = current

    from Generation.check_static import _extract_phi  # lazy

    for key in ("phi", "level_set_initial"):
        if key in opt and _source_of(opt, key) == "designed":
            current = _extract_phi(main_py)
            if current:
                permission["phi"] = current
            break

    from Generation.check_static import _parse_label_lists  # lazy

    code_boundary, code_fixed = _parse_label_lists(main_py)
    for key, code_set in (("boundary_labels", code_boundary),
                          ("fixed_labels", code_fixed)):
        if key in opt and _source_of(opt, key) == "designed":
            if isinstance(opt.get(key), list):
                permission["labels"][key] = code_set

    chi = designed.get("characteristic_function")
    if isinstance(chi, dict) and str(chi.get("source", "designed")) == "designed":
        permission["chi"] = True

    # Dirichlet conditions of the declared ScalarField/VectorField are
    # boundary physical properties — a DESIGNED one may be corrected
    # (extracted ones are ground truth).  The current declaration is the
    # enforcement baseline.
    from Generation.check_static import (  # lazy
        _quoted,
        _strip_comments,
    )

    code_nc = _strip_comments(main_py)
    for f in designed.get("fields") or []:
        if not isinstance(f, dict):
            continue
        entry = {
            "name": str(f.get("name") or ""),
            "latex": str(f.get("latex") or ""),
            "value": None, "value_editable": False,
            "labels": None, "labels_editable": False,
        }
        args = None
        if entry["name"] or entry["latex"]:
            args = _match_field_call(code_nc, entry["name"], entry["latex"])
        dv = f.get("dirichlet_value")
        if dv not in (None, "") and _source_of(f, "dirichlet_value") == "designed":
            entry["value_editable"] = True
            if args is not None and len(args) > 3:
                entry["value"] = _quoted(args[3])
        dl = f.get("dirichlet_labels")
        if isinstance(dl, list) and dl and _source_of(f, "dirichlet_labels") == "designed":
            entry["labels_editable"] = True
            if args is not None and len(args) > 4:
                entry["labels"] = sorted(
                    {int(x) for x in re.findall(r"\d+", args[4])}
                )
        if entry["value_editable"] or entry["labels_editable"]:
            permission["dirichlet"][entry["name"] or entry["latex"]] = entry
            # a designed Dirichlet condition may need the penalty
            # restructure (pen existence) to be fixed
            permission["pen"] = True

    # designed boundaries grant the pen-existence repair as well (a
    # design-Dirichlet boundary must be a penalty Robin, never a field
    # Dirichlet)
    if any(
        isinstance(b, dict) and str(b.get("source", "designed")) == "designed"
        for b in (designed.get("domain") or {}).get("boundaries", []) or []
    ):
        permission["pen"] = True

    permission["mesh"] = not mesh_frozen
    return permission


def build_fe_param_permission(spec: dict, main_py: str, mesh_frozen: bool) -> dict:
    """The fe_param_fix permission set.

    ``{"setup": {name: current}, "resolution": bool, "fe_space": bool}``.
    """
    permission = {"setup": {}, "resolution": False, "fe_space": False}
    if not isinstance(spec, dict):
        return permission
    designed = spec.get("designed") or {}
    opt = designed.get("optimization_config") or {}

    from Generation.check_static import _setup_param_value  # lazy

    for key in _FE_PARAM_KEYS:
        if key in opt and _source_of(opt, key) == "designed":
            current = _setup_param_value(main_py, key)
            if current is not None:
                permission["setup"][key] = current

    permission["resolution"] = not mesh_frozen
    for f in designed.get("fields") or []:
        if isinstance(f, dict) and f.get("fe_space") \
                and _source_of(f, "fe_space") == "designed":
            permission["fe_space"] = True
            break
    return permission


def permission_text(kind: str, permission: dict) -> str:
    """Render a permission set for prompts (deterministic)."""
    if not permission:
        return ""
    lines = [
        "## Repair Permission (deterministic — ONLY these values may change)"
    ]
    if kind == "design_fix":
        for name in sorted(permission.get("constants") or {}):
            lines.append(f"- constant {name}= (current: {permission['constants'][name]})")
        if permission.get("phi") is not None:
            lines.append(f"- setup_optimization phi= (current: {permission['phi']})")
        for key in sorted(permission.get("labels") or {}):
            kw = "boundaryLabels" if key == "boundary_labels" else "fixedLabels"
            lines.append(f"- setup_optimization {kw}= (current: {sorted(permission['labels'][key])})")
        if permission.get("chi"):
            lines.append("- CharacteristicFunction strong/weak values")
        for field in sorted(permission.get("dirichlet") or {}):
            entry = permission["dirichlet"][field]
            if entry.get("value_editable"):
                lines.append(
                    f"- field '{field}' Dirichlet value "
                    f"(current: {entry.get('value')})"
                )
            if entry.get("labels_editable"):
                lines.append(
                    f"- field '{field}' Dirichlet labels "
                    f"(current: {entry.get('labels')})"
                )
        if permission.get("pen"):
            lines.append("- the pen penalty implementation may be ADDED or REMOVED "
                         "(design-Dirichlet / multi-valued Dirichlet restructure) "
                         "— the pen VALUE is always 1e15 (fixed content, never "
                         "a tuning target)")
        if permission.get("mesh"):
            lines.append("- mesh.txt definitions — borders AND square()/buildmesh count "
                         "as borders (the mesh is NOT a provided/frozen one)")
    elif kind == "fe_param_fix":
        for key in sorted(permission.get("setup") or {}):
            lines.append(f"- setup_optimization {key}= (current: {permission['setup'][key]})")
        if permission.get("resolution"):
            lines.append("- mesh.txt resolution (int n)")
        if permission.get("fe_space"):
            lines.append("- field precision P1/P2 (designed fields only)")
    lines.append(
        "Everything else in main.py / mesh.txt must stay byte-identical "
        "(extracted values are ground truth)."
    )
    return "\n".join(lines)


# ======================================================================
# deterministic enforcement — the LLM response is only the carrier of
# the permitted values
# ======================================================================

_CHI_RE = re.compile(
    r"CharacteristicFunction\(\s*(r?['\"][^'\"]*['\"])\s*,\s*([-+0-9.eE]+)\s*,"
    r"\s*([^)]*)\)"
)


def _chi_parts(code: str):
    """(latex, strong, weak) of the FIRST CharacteristicFunction call."""
    from utils import code_masked_positions

    masked = code_masked_positions(code)
    for m in _CHI_RE.finditer(code):
        if m.start() in masked:
            continue
        return m.group(1), m.group(2), m.group(3).strip()
    return None


def _chi_replace(code: str, strong=None, weak=None) -> str:
    """Replace the strong/weak values of the first CharacteristicFunction."""
    from utils import code_masked_positions

    masked = code_masked_positions(code)
    out = []
    pos = 0
    for m in _CHI_RE.finditer(code):
        if m.start() in masked:
            continue
        out.append(code[pos:m.start(2)])
        out.append(f"{strong:g}" if strong is not None else m.group(2))
        pos = m.end(2)
        if weak is not None:
            out.append(code[pos:m.start(3)])
            out.append(weak)
            pos = m.end(3)
        break
    out.append(code[pos:])
    return "".join(out)


def _parse_numeric_or_fraction(text: str):
    """Parse '1/100', 0, 0.5, ... — the weak-value forms."""
    text = text.strip().strip("'\"")
    m = re.fullmatch(r"([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)", text)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _write_labels(out: str, key: str, resp_set: set) -> tuple:
    from Generation.check_static import (  # lazy
        _find_setup_optimization,
        _param_span,
    )

    kw = "boundaryLabels" if key == "boundary_labels" else "fixedLabels"
    span = _find_setup_optimization(out)
    if span is None:
        return out, False
    call_text = out[span[0]:span[1]]
    arg = _param_span(call_text, kw)
    if arg is None:
        return out, False
    new_lit = f"{kw}={sorted(resp_set)!r}"
    new_call = call_text[:arg[0]] + new_lit + call_text[arg[1]:]
    return out[:span[0]] + new_call + out[span[1]:], True


def _field_call_span(code: str, name: str, latex: str):
    """(start, end) span of the ScalarField/VectorField call for *name*
    (variable declaration) or *latex* (first argument), else None.

    Quote-aware balanced scan (mirrors ``check_static._call_content``);
    comment/string positions are masked so a mention never matches.
    """
    from Generation.check_static import (  # lazy
        _call_content,
        _norm_latex,
        _quoted,
        _top_level_split,
    )
    from utils import code_masked_positions

    masked = code_masked_positions(code)
    if name and name.isidentifier():
        for m in re.finditer(
            rf"\b{re.escape(name)}\s*=\s*(?:ScalarField|VectorField)\s*\(", code
        ):
            if m.start() in masked:
                continue
            body = _call_content(code, m)
            if body is not None:
                return m.start(), m.end() + len(body) + 1
    if latex:
        for m in re.finditer(r"\b(?:ScalarField|VectorField)\s*\(", code):
            if m.start() in masked:
                continue
            body = _call_content(code, m)
            if body is None:
                continue
            args = _top_level_split(body)
            if args and _norm_latex(_quoted(args[0])) == _norm_latex(latex):
                return m.start(), m.end() + len(body) + 1
    return None


def _split_arg_spans(text: str) -> list:
    """(start, end) of each TOP-LEVEL comma-separated argument in *text*
    (quote/paren aware) — the span-level mirror of
    ``check_static._top_level_split``."""
    spans = []
    depth = 0
    quote = None
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            spans.append((start, i))
            start = i + 1
        i += 1
    spans.append((start, len(text)))
    return spans


def _replace_field_arg(code: str, name: str, latex: str, arg_idx: int,
                       new_text: str) -> str:
    """Replace the arg_idx-th (0-based) argument of the field call with
    *new_text*; no-op when the call or the argument position is absent."""
    span = _field_call_span(code, name, latex)
    if span is None:
        return code
    start, end = span
    call_text = code[start:end]
    body_start = call_text.find("(") + 1
    body = call_text[body_start:-1]
    spans = _split_arg_spans(body)
    if len(spans) <= arg_idx:
        return code
    a0, a1 = spans[arg_idx]
    # keep the original separator spacing (the span includes the
    # whitespace after the previous comma)
    lead = ""
    j = a0
    while j < a1 and body[j] in " \t":
        lead += body[j]
        j += 1
    new_body = body[:a0] + lead + new_text + body[a1:]
    return code[:start] + call_text[:body_start] + new_body + ")" + code[end:]


def _format_dirichlet_labels(labels) -> str:
    """The DSL form of the labels: a SINGLE label is bare (``3``), two
    or more are a list (``[1, 2]``) — per check_static rule."""
    labels = sorted({int(x) for x in labels})
    if len(labels) == 1:
        return str(labels[0])
    return "[" + ", ".join(str(x) for x in labels) + "]"


def design_enforce(main_py: str, resp_main: str, resp_mesh: str,
                   permission: dict) -> tuple:
    """Keep ONLY the permitted design values from the response.

    Returns ``(main, mesh, changes)`` — *mesh* is "" unless the border
    definitions are mutable.  When the pen EXISTENCE is permitted the
    response's structure is the base (the penalty implementation may be
    added/removed — LLM-check territory, per the static-where-possible
    rule); the pen VALUE is then pinned to the project convention 1e15
    (fixed content, never a tuning target).
    """
    from Generation.check_static import _parse_label_lists  # lazy

    out = resp_main if permission.get("pen") else main_py
    if permission.get("pen"):
        out = _constant_replace(out, "pen", 1e15)
    changes = {}

    for name in sorted(permission.get("constants") or {}):
        value = _constant_value(resp_main, name)
        if value is None or not math.isfinite(value):
            continue
        current = permission["constants"].get(name)
        if current is not None and abs(value - current) <= abs(current) * _VALUE_CHANGE_EPS:
            continue
        out = _constant_replace(out, name, value)
        changes[name] = value

    if permission.get("phi") is not None:
        from Generation.check_static import _extract_phi

        phi = _extract_phi(resp_main)
        if phi and phi != permission["phi"]:
            out = apply_param_updates(out, {"phi": phi})
            changes["phi"] = phi

    resp_boundary, resp_fixed = _parse_label_lists(resp_main)
    for key in sorted(permission.get("labels") or {}):
        resp_set = resp_boundary if key == "boundary_labels" else resp_fixed
        if not resp_set:
            continue
        current = permission["labels"][key]
        if resp_set == current:
            continue
        out, ok = _write_labels(out, key, resp_set)
        if ok:
            changes[key] = sorted(resp_set)

    if permission.get("chi"):
        parts = _chi_parts(resp_main)
        if parts is not None:
            _latex, strong, weak = parts
            cur = _chi_parts(out)
            if cur is not None:
                strong_ok = math.isfinite(float(strong)) \
                    and strong.strip() != cur[1]
                weak_val = _parse_numeric_or_fraction(weak)
                weak_cur = _parse_numeric_or_fraction(cur[2])
                weak_ok = weak_val is not None and weak_cur is not None \
                    and abs(weak_val - weak_cur) > max(
                        _VALUE_CHANGE_EPS, abs(weak_cur) * _VALUE_CHANGE_EPS)
                if strong_ok or weak_ok:
                    out = _chi_replace(
                        out,
                        strong=float(strong) if strong_ok else None,
                        weak=weak if weak_ok else None,
                    )
                    changes["chi"] = (strong, weak)

    # Dirichlet conditions of the declared fields (boundary physical
    # properties): the response is only the carrier of the permitted
    # value/labels — the write happens into the enforced base code.
    if permission.get("dirichlet"):
        from Generation.check_static import (  # lazy
            _quoted,
            _strip_comments,
        )

        resp_nc = _strip_comments(resp_main)
        for field, entry in permission["dirichlet"].items():
            resp_args = _match_field_call(
                resp_nc, entry["name"], entry["latex"],
            )
            if resp_args is None:
                continue
            if entry["value_editable"] and len(resp_args) > 3:
                new_v = _quoted(resp_args[3])
                if new_v is not None and str(new_v) != str(entry["value"]):
                    out = _replace_field_arg(
                        out, entry["name"], entry["latex"], 3, repr(new_v),
                    )
                    changes[f"dirichlet:{field}:value"] = new_v
            if entry["labels_editable"] and len(resp_args) > 4:
                new_labels = sorted(
                    {int(x) for x in re.findall(r"\d+", resp_args[4])}
                )
                if new_labels and new_labels != (entry["labels"] or []):
                    out = _replace_field_arg(
                        out, entry["name"], entry["latex"], 4,
                        _format_dirichlet_labels(new_labels),
                    )
                    changes[f"dirichlet:{field}:labels"] = new_labels

    mesh = resp_mesh if permission.get("mesh") else ""
    return out, mesh, changes


def fe_param_enforce(main_py: str, resp_main: str, resp_mesh: str,
                     permission: dict) -> tuple:
    """Keep ONLY the permitted fe-parameter values from the response.

    The mesh resolution (``int n = N;``) crosses into mesh.txt only
    when the mesh is not frozen.  The field precision (P1/P2) is
    permitted by the prompt but re-verified by the static checks
    against the spec, so it is not filtered here.
    """
    out = main_py
    changes = {}
    for param in sorted(permission.get("setup") or {}):
        from Generation.check_static import _find_setup_optimization, _param_span

        span = _find_setup_optimization(resp_main)
        if span is None:
            continue
        call_text = resp_main[span[0]:span[1]]
        arg = _param_span(call_text, param)
        if arg is None:
            continue
        m = re.search(r"=\s*([-+0-9.eE]+)", call_text[arg[0]:arg[1]])
        if m is None:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        if not math.isfinite(value):
            warnings.warn(f"fixer: ignoring non-finite {param} value in response")
            continue
        current = permission["setup"].get(param)
        if current is not None and abs(value - current) <= abs(current) * _VALUE_CHANGE_EPS:
            continue
        out = apply_param_updates(out, {param: value})
        changes[param] = value

    mesh = ""
    if permission.get("resolution") and resp_mesh:
        m = re.search(r"\bint\s+n\s*=\s*([0-9]+)\s*;", resp_mesh)
        if m and m.group(1) != re.search(r"\bint\s+n\s*=\s*([0-9]+)\s*;", main_py or ""):
            mesh = resp_mesh
            changes["resolution"] = int(m.group(1))
    return out, mesh, changes


# ======================================================================
# fix mode entry points
# ======================================================================


def _fix_payload(
    problem_description, main_py, mesh_txt, run_result_json,
    objective_data, convergence_data, edp_code, repair_records,
    doc_text, analysis_artifacts, coarse, image_section,
    fix_direction="", permission_block="",
) -> dict:
    """The shared invoke payload of the fix modes."""
    return {
        "problem_description": problem_description,
        "main_py": main_py,
        "mesh_txt": mesh_txt,
        "run_result_json": run_result_json,
        "objective_data": objective_data or "(not available)",
        "convergence_data": convergence_data or "(not available)",
        "edp_code": edp_code or "(not available)",
        "repair_records": repair_records or "(no repair records yet)",
        "doc_text": doc_text,
        "analysis_artifacts": analysis_artifacts or "(not available)",
        "coarse": coarse,
        "fix_direction": fix_direction or "",
        "permission": permission_block or "",
        "image_section": image_section,
    }


def formula_fix(
    agent, *, problem_description, main_py, mesh_txt, run_result_json,
    objective_data, convergence_data, edp_code, repair_records,
    doc_text, analysis_artifacts, coarse, images=None, image_section="",
    fix_direction="",
) -> dict:
    """One formula-repair candidate (one direction)."""
    from .prompts import FIX_FORMULA_SYSTEM, FIX_FORMULA_USER

    agent.set_prompt_list([
        {"role": "system", "content": FIX_FORMULA_SYSTEM},
        {"role": "user", "content": FIX_FORMULA_USER},
    ])
    raw = agent.invoke_multimodal(
        _fix_payload(
            problem_description, main_py, mesh_txt, run_result_json,
            objective_data, convergence_data, edp_code, repair_records,
            doc_text, analysis_artifacts, coarse, image_section,
            fix_direction=fix_direction,
        ),
        images=images,
    )
    return _parse_fix_response(raw)


def design_fix(
    agent, *, problem_description, main_py, mesh_txt, run_result_json,
    objective_data, convergence_data, edp_code, repair_records,
    doc_text, analysis_artifacts, coarse, permission=None,
    images=None, image_section="", fix_direction="",
) -> dict:
    """One design-value candidate (one direction), enforced to the
    deterministic permission set."""
    from .prompts import FIX_DESIGN_SYSTEM, FIX_DESIGN_USER

    permission = permission or {}
    agent.set_prompt_list([
        {"role": "system", "content": FIX_DESIGN_SYSTEM},
        {"role": "user", "content": FIX_DESIGN_USER},
    ])
    raw = agent.invoke_multimodal(
        _fix_payload(
            problem_description, main_py, mesh_txt, run_result_json,
            objective_data, convergence_data, edp_code, repair_records,
            doc_text, analysis_artifacts, coarse, image_section,
            fix_direction=fix_direction,
            permission_block=permission_text("design_fix", permission),
        ),
        images=images,
    )
    parsed = _parse_fix_response(raw)
    if not parsed.get("_error"):
        parsed["main"], parsed["mesh"], _ = design_enforce(
            main_py, parsed.get("main") or "", parsed.get("mesh") or "", permission,
        )
    return parsed


def fe_param_fix(
    agent, *, problem_description, main_py, mesh_txt, run_result_json,
    objective_data, convergence_data, edp_code, repair_records,
    doc_text, analysis_artifacts, coarse, permission=None,
    images=None, image_section="", fix_direction="",
) -> dict:
    """One fe-parameter candidate (one direction), enforced to the
    deterministic permission set."""
    from .prompts import FIX_FE_PARAM_SYSTEM, FIX_FE_PARAM_USER

    permission = permission or {}
    agent.set_prompt_list([
        {"role": "system", "content": FIX_FE_PARAM_SYSTEM},
        {"role": "user", "content": FIX_FE_PARAM_USER},
    ])
    raw = agent.invoke_multimodal(
        _fix_payload(
            problem_description, main_py, mesh_txt, run_result_json,
            objective_data, convergence_data, edp_code, repair_records,
            doc_text, analysis_artifacts, coarse, image_section,
            fix_direction=fix_direction,
            permission_block=permission_text("fe_param_fix", permission),
        ),
        images=images,
    )
    parsed = _parse_fix_response(raw)
    if not parsed.get("_error"):
        parsed["main"], parsed["mesh"], _ = fe_param_enforce(
            main_py, parsed.get("main") or "", parsed.get("mesh") or "", permission,
        )
    return parsed


def free_fix(
    agent, *, problem_description, main_py, mesh_txt, run_result_json,
    objective_data, convergence_data, edp_code, repair_records,
    doc_text, analysis_artifacts, coarse, images=None, image_section="",
) -> dict:
    """Single free-text repair (the fallback)."""
    from .prompts import FIX_FREE_SYSTEM, FIX_FREE_USER

    agent.set_prompt_list([
        {"role": "system", "content": FIX_FREE_SYSTEM},
        {"role": "user", "content": FIX_FREE_USER},
    ])
    raw = agent.invoke_multimodal(
        _fix_payload(
            problem_description, main_py, mesh_txt, run_result_json,
            objective_data, convergence_data, edp_code, repair_records,
            doc_text, analysis_artifacts, coarse, image_section,
        ),
        images=images,
    )
    return _parse_fix_response(raw)


def _parse_fix_response(raw: str) -> dict:
    """Parse LLM JSON response, with fallback on failure.

    A response carrying NEITHER ``main`` NOR ``mesh`` code (wrong key
    names, schema drift, empty output) is treated as a parse failure —
    the reflector retries instead of permanently giving up.
    """
    try:
        result = json.loads(parser_json(raw))
        if not isinstance(result, dict):
            warnings.warn(f"fixer returned non-object JSON ({type(result).__name__}).")
            return {"main": "", "mesh": "", "rationale": "", "_error": "non-object JSON response"}

        main = result.get("main")
        mesh = result.get("mesh")
        if main is not None and not isinstance(main, str):
            return {"main": "", "mesh": "", "rationale": "", "_error": "main is not a string"}
        if mesh is not None and not isinstance(mesh, str):
            return {"main": "", "mesh": "", "rationale": "", "_error": "mesh is not a string"}
        has_code = (
            (isinstance(main, str) and main.strip() != "")
            or (isinstance(mesh, str) and mesh.strip() != "")
        )
        if not has_code:
            warnings.warn("fixer response carries neither main nor mesh code.")
            return {"main": "", "mesh": "", "rationale": "", "_error": "no main/mesh in response"}
        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"fixer JSON parse failed: {e}")
        return {"main": "", "mesh": "", "rationale": "", "_error": str(e)}
