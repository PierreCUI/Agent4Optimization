"""
Initial-Parameter Review — Stage 5.5 of the pipeline, between the static
check and the execution.

Reviews the initial optimisation parameters of ``setup_optimization``
and corrects them before any computation starts:

* ``show_weak_material`` is FORCED to ``False`` — enforced
  deterministically, never left to the LLM.
* ``hmin``, ``hmax``, ``diffusion``, ``v0``, ``phi`` are reviewed by the
  LLM.  Reference priority: the optimisation parameters explicitly stated
  in the problem description (the "paper") first, then the parameter
  ranges of the selected few-shot examples, then the designed defaults.
* The LLM is STRICTLY LIMITED to returning parameter VALUES — it never
  returns code.  The values are written back into ``setup_optimization``
  mechanically by ``apply_param_updates``, so name/mesh/iterations/
  boundaryLabels/fixedLabels and everything else in the program are
  physically untouchable.
"""

import json
import math
import os
import re
import sys
from typing import Optional
import warnings
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Generation.coder import phi_style_hint  # noqa: E402
from Generation.phi_conventions import PHI_GUIDANCE  # noqa: E402
from utils import code_masked_positions, parser_json  # noqa: E402

# ======================================================================
# deterministic enforcement
# ======================================================================

# The ONLY setup_optimization parameters the LLM may return values for.
_ALLOWED_PARAM_KEYS = ("hmin", "hmax", "diffusion", "v0", "phi")


def _extract_phi(code: str) -> str:
    """Extract the phi string literal from the setup_optimization call.

    Scoped to the call span so a commented-out ``phi='...'`` example
    above or below the real call can never hijack the value (the value
    feeds the symmetry check, where a wrong phi triggers spurious
    correction rounds)."""
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


def _find_setup_optimization(code: str) -> tuple[int, int] | None:
    """Return the (start, end) character span of the ``setup_optimization``
    call, or None when the call is absent.

    Takes the LAST balanced REAL call: the coder prompt shows a
    ``setup_optimization``-shaped example, so comment/docstring mentions
    before OR after the real call are common and must not hijack the
    span (a balanced span inside a comment would silently swallow every
    Stage 5.5 parameter update).  Comments and strings are rejected via
    the shared code mask.

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


def _param_span(call_text: str, param: str) -> tuple[int, int] | None:
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
            f"check_params: refusing to replace {param} — value contains "
            f"parentheses and is not quoted"
        )
        return None
    # keep trailing whitespace OUT of the span so a following comment
    # (``True  # old value``) keeps its separating space after replacement
    while end > m.end() and call_text[end - 1] in " \t":
        end -= 1
    return m.start(), end


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


def apply_param_updates(code: str, updates: dict) -> str:
    """Mechanically write the reviewed parameter VALUES back into the
    setup_optimization call.

    *updates* maps allowed parameter names (hmin/hmax/diffusion/v0/phi)
    to their new values.  Everything else in the program is untouched by
    construction — this is the permission boundary of the LLM pass.
    """
    for param in _ALLOWED_PARAM_KEYS:
        if param not in updates:
            continue
        value = updates[param]
        # type gate: numeric parameters must be finite numbers,
        # phi must be a non-empty string
        if param == "phi":
            if not isinstance(value, str) or not value.strip():
                warnings.warn(f"check_params: ignoring non-string/empty phi value: {value!r}")
                continue
            value_text = repr(value)
        else:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                # json.loads accepts NaN/Infinity — str(nan) would poison
                # the code with an undefined name
                warnings.warn(f"check_params: ignoring non-numeric {param} value: {value!r}")
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
            # parameter absent — insert before the closing parenthesis
            new_call = _insert_before_closing(call_text, f"{param}={value_text}")
            code = code[:span[0]] + new_call + code[span[1]:]
    return code


# ======================================================================
# system prompt
# ======================================================================

_PARAM_SYSTEM = """\
You are an AutoFreeFEM initial-parameter reviewer.  You receive a generated
main.py and must review ONLY the initial optimisation parameters of the
``setup_optimization`` call: ``hmin``, ``hmax``, ``diffusion``, ``v0``,
and ``phi``.  Correct them when they are unreasonable for the problem.

## Reference priority

1. **The paper (problem description) first**: optimisation parameters
   explicitly stated in the problem description are GROUND TRUTH — if the
   paper gives hmin/hmax/diffusion/v0/level_set_initial, use them exactly.
2. **Selected examples second**: if the paper gives none, take the
   parameter values AND the phi construction style used by the selected
   reference examples (successful runs of similar problems) and adapt
   them to this problem's domain size and geometry.
3. **Designed defaults last**: only when neither source provides values.

## Review rules

- hmin/hmax must match the REAL domain size read from the mesh file:
  hmin ≈ 1/1000 .. 1/100 of the domain length, hmax ≈ 1/100 .. 1/10 of
  the domain length, and hmin < hmax.
- diffusion (level-set regularisation): order 1e-6 .. 1e-2.
- v0 (shape advection speed): order 1e-7 .. 1e-2.
- phi (initial level set) must produce a CLEAR SOLID REGION: it must be
  negative somewhere inside the domain (a phi that is non-negative
  everywhere starts from an EMPTY design).  phi must use the REAL domain
  coordinates — never mix normalised coordinates (e.g. assuming the
  domain is [0,1]x[0,1]) with the actual mesh coordinates.
- OTHERWISE FOLLOW THE PHI RULES AT THE END OF THIS PROMPT: a phi
  prescribed by the paper wins; without one, pick a seed from the PHI
  TEMPLATE LIBRARY (solid trunk shapes, sparse holes, checkerboard,
  combinations) and adapt it to the problem's PHYSICS (a connected
  load-to-sink path, solid contact on the fixed load/sink boundaries —
  their material contact is frozen during the level-set evolution —
  features several times hmax, smooth and low-frequency), with the
  reference examples contributing only the construction style.
- SYMMETRY: if the mesh geometry and boundary conditions are symmetric
  about an axis (mirror segments carry the same labels), the initial phi
  MUST be symmetric about the same axis: phi(x, y) == phi(L - x, y) for
  a vertical mirror axis at x = L/2, phi(x, y) == phi(x, H - y) for a
  horizontal axis.  Use cos-type oscillatory terms about the mirror
  axis — cos(2*pi*x/L) is symmetric about x = L/2 while
  sin(2*pi*x/L) is NOT.

## Hard rules

- Change ONLY hmin, hmax, diffusion, v0, phi.
- NEVER change name, mesh, iterations, boundaryLabels, fixedLabels.
- show_weak_material must be False — it is enforced outside; do not touch it.
- You CANNOT return code — the pipeline applies your values mechanically
  to the setup_optimization call and nothing else is reachable.

""" + PHI_GUIDANCE + """

## Output

Return exactly one JSON object with the parameter VALUES you want to
change (omit any parameter you leave unchanged):
{{"hmin": 0.0001, "hmax": 0.001, "diffusion": 0.0001, "v0": 0.001,
 "phi": "y - 0.02", "rationale": "<one sentence on what was changed and why>"}}
hmin/hmax/diffusion/v0 are numbers, phi is a string containing the
expression.  No markdown.  No code fences.  No extra text.
"""

_PARAM_USER = """\
Problem description (the paper — source of extracted optimisation parameters):
{problem_description}

Analysis artifacts (extracted vs designed optimisation parameters):
{optimization_params_json}

Mesh file (the real domain — check phi and hmin/hmax against it):
{mesh_txt}

Current main.py:
```python
{code}
```

Selected reference examples (parameter ranges to consult when the paper gives nothing):
{examples_text}

{phi_style_hint}

Review hmin, hmax, diffusion, v0, phi.  Return JSON.
"""


# ======================================================================
# main entry point
# ======================================================================


def check_and_fix_params(
    agent,
    code: str,
    problem_spec: Optional[dict] = None,
    problem_description: str = "",
    mesh_txt: str = "",
    examples: Optional[list[dict]] = None,
) -> dict:
    """
    Review and correct the initial optimisation parameters of *code*.

    The LLM may ONLY return parameter VALUES (hmin/hmax/diffusion/v0/
    phi); ``apply_param_updates`` writes them mechanically back into the
    setup_optimization call.  Everything else in the program — name,
    mesh, iterations, boundaryLabels, fixedLabels, the whole code body —
    is unreachable by construction.  show_weak_material is additionally
    forced to False.

    Reference priority for the review: the paper (extracted optimisation
    parameters), then the selected examples, then the designed defaults.

    Returns ``{"main": ..., "rationale": ...}``; on LLM parse failure
    returns the deterministically-fixed code with an ``_error`` note.
    """
    # ── deterministic: force show_weak_material=False ────────────────
    code = force_show_weak_material_false(code)

    problem_spec = problem_spec or {}
    ex = problem_spec.get("extracted", {})
    ds = problem_spec.get("designed", {})

    opt_params = {
        "extracted_optimization_params": ex.get("optimization_params", {}),
        "designed_optimization_config": ds.get("optimization_config", {}),
    }

    examples_text = ""
    for exd in examples or []:
        # Defensive: callers may pass malformed example dicts
        if not isinstance(exd, dict):
            continue
        examples_text += f"=== EXAMPLE: {exd.get('name', '?')} ===\n{exd.get('py', '')}\n\n"

    prompt_list = [
        {"role": "system", "content": _PARAM_SYSTEM},
        {"role": "user", "content": _PARAM_USER},
    ]
    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_description": problem_description or "(not available)",
            "optimization_params_json": json.dumps(opt_params, indent=2, ensure_ascii=False),
            "mesh_txt": mesh_txt or "(not available)",
            "code": code,
            "examples_text": examples_text or "(no examples selected)",
            "phi_style_hint": phi_style_hint(examples) or "(no phi style signal)",
        }
    )

    try:
        result = json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"check_and_fix_params JSON parse failed: {e}")
        return {"main": code, "rationale": "", "_error": str(e)}

    if not isinstance(result, dict):
        warnings.warn(f"check_and_fix_params unexpected result type: {type(result)}")
        return {"main": code, "rationale": "", "_error": "not a dict"}

    # ── permission boundary: keep only the allowed parameter keys ────
    updates = {k: result[k] for k in _ALLOWED_PARAM_KEYS if k in result}
    code = apply_param_updates(code, updates)

    # ── deterministic: symmetric problems require a symmetric phi ────
    from .symmetry import (  # local import keeps module loading light
        canonical_symmetric_phi,
        detect_symmetry,
        is_phi_symmetric,
        mesh_dims,
    )

    axes = detect_symmetry(mesh_txt) if mesh_txt else []
    if axes:
        L, H = mesh_dims(mesh_txt)
        phi_cur = _extract_phi(code)
        if not is_phi_symmetric(phi_cur, L, H, axes):
            feedback = (
                "\n\nSYMMETRY CORRECTION REQUIRED: the mesh is symmetric "
                f"about {axes} but your phi '{phi_cur}' is NOT symmetric on "
                "these axes.  Return a symmetric phi (cos-type terms about "
                "the mirror axis) and nothing else wrong."
            )
            raw2 = agent.invoke(
                {
                    "problem_description": problem_description or "(not available)",
                    "optimization_params_json": json.dumps(opt_params, indent=2, ensure_ascii=False),
                    "mesh_txt": mesh_txt or "(not available)",
                    "code": code,
                    "examples_text": examples_text or "(no examples selected)",
                    "phi_style_hint": (phi_style_hint(examples) or "(no phi style signal)") + feedback,
                }
            )
            try:
                result2 = json.loads(parser_json(raw2))
                if isinstance(result2, dict):
                    updates2 = {k: result2[k] for k in _ALLOWED_PARAM_KEYS if k in result2}
                    code = apply_param_updates(code, updates2)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            phi_cur = _extract_phi(code)
            if not is_phi_symmetric(phi_cur, L, H, axes):
                # deterministic fallback: a canonical symmetric pattern
                warnings.warn(
                    f"check_params: LLM phi '{phi_cur}' still asymmetric for "
                    f"axes {axes} — using canonical symmetric phi."
                )
                code = apply_param_updates(
                    code, {"phi": canonical_symmetric_phi(L, H, axes)}
                )

    # rationale is returned to the caller for display — normalise to str
    # so downstream `.replace(...)[:80]` formatting cannot crash on a
    # non-string value.
    rationale = result.get("rationale", "")
    if not isinstance(rationale, str):
        warnings.warn(f"check_params: non-string rationale ignored: {rationale!r}")
        rationale = ""

    return {"main": code, "rationale": rationale}


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
    print(f"Parameter Check — {len(py_files)} generated files")
    print(f"{'=' * 70}")

    for py_file in py_files:
        name = py_file.parent.name
        code = py_file.read_text(encoding="utf-8", errors="replace")

        problem_spec = None
        spec_file = py_file.parent / "initial.json"
        if spec_file.exists():
            with open(spec_file, "r", encoding="utf-8") as f:
                problem_spec = json.load(f)

        mesh_txt = ""
        mesh_file = py_file.parent / "mesh.txt"
        if mesh_file.exists():
            mesh_txt = mesh_file.read_text(encoding="utf-8", errors="replace")

        print(f"\n  [{name}]", end=" ", flush=True)
        result = check_and_fix_params(agent, code, problem_spec, mesh_txt=mesh_txt)
        if "_error" in result:
            print(f"ERROR: {result['_error']}")
            continue

        with open(py_file, "w", encoding="utf-8") as f:
            f.write(result["main"])
        print(f"-> {result.get('rationale', '')[:80]}")

    print("\nDone. Corrected code written back to results/*/main.py")
