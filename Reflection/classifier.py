"""Coarse error classifier + deterministic fix-eligibility gates.

Error taxonomy (the reflection loop's five categories):
  1. syntax_error       — imports, call signatures/arguments, spelling
                          and capitalisation, fixed contents
                          (check_units / show_weak_material /
                          iterations), the extracted/designed rules,
                          and any other Python/FreeFEM-reported syntax
                          problem.
  2. formula_error      — every FORMULA: intermediate formulas, the J
                          formula, the Lagrangian formulas, and the
                          nonlinear formulas.
  3. design_error       — design VALUES: physical-parameter numbers,
                          Afac (ell), CharacteristicFunction values,
                          phi, the three-way boundary labels, each
                          boundary's physical properties (including
                          the ScalarField/VectorField Dirichlet
                          conditions and the pen penalty implementation
                          — pen's EXISTENCE is design, its VALUE is the
                          fixed convention 1e15), and the mesh border
                          definitions.
  4. fe_parameter_error — finite-element parameters: hmin/hmax/
                          diffusion/v0, the mesh resolution, and each
                          field's precision (P1/P2).
  5. other_error        — the fallback.

The model judges the category from the ERROR BEHAVIOUR (crash,
non-convergence, oscillation, wrong result, degenerate result) plus ALL
other information — there is NO fixed symptom-to-category wiring.
Before the first repair round the reflector pre-computes which fix
types are USABLE (have repair room under the extracted/designed rules)
and the classification prompt shows ONLY the usable types.
"""

import re
import warnings
from typing import Optional

# Shared by the reflector (which appends the header to the context) and
# the classifier (which uses its presence as a hard filter) — one
# definition, owned HERE so the dependency-free classifier stays
# importable without meshio (the renderer imports this module, never
# the other way around).
MINTH_RENDER_SUMMARY_HEADER = "=== MinTh.vtu Render Summary ==="

# ======================================================================
# shared label normalization
# ======================================================================


def _canonical_label(text: str) -> str:
    """Extract the single canonical keyword from an LLM label answer.

    Accepts benign variants ("pass.", "Pass ✓", '"formula_error"');
    a prose/refusal answer maps to "unknown" so the caller's warning
    path fires instead of silently faking a value.
    """
    t = re.sub(r"[^a-z]+", "", text.strip().lower())
    # Pass verdict checked FIRST so a passing run whose justification
    # mentions a failure word is never inverted into an error category.
    if t == "pass" or t.startswith("pass"):
        return "pass"
    # A CANONICAL LABEL WORD wins over any explanatory prose.
    if "wrongresult" in t:
        return "wrong_result"  # quality-check contract label
    if "syntaxerror" in t:
        return "syntax_error"
    if "formulaerror" in t:
        return "formula_error"
    if "designerror" in t:
        return "design_error"
    if "feparametererror" in t or "feparameter" in t:
        return "fe_parameter_error"
    if "othererror" in t:
        return "other_error"
    # NO prose heuristics — no keyword-to-category wiring: a reply
    # without a canonical label word is "unknown" and the callers fall
    # back to other_error uniformly.
    return "unknown"


# ======================================================================
# coarse classification
# ======================================================================


def coarse_classify(
    agent, problem_name: str, context_text: str, images=None,
    image_section: str = "", available_types: str = "",
) -> str:
    """
    Classify the current problem state into one of 6 categories.

    *available_types* is the pre-computed list of USABLE error types
    (the model must pick from it — the deterministic gates already
    removed the types with no repair room).
    """
    from .prompts import CLASSIFY_COARSE_SYSTEM, CLASSIFY_COARSE_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_COARSE_SYSTEM},
        {"role": "user", "content": CLASSIFY_COARSE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke_multimodal(
        {
            "problem_name": problem_name,
            "context_text": context_text,
            "image_section": image_section,
            "available_types": available_types or "(all types usable)",
        },
        images=images,
    )

    label = _canonical_label(raw)
    valid = [
        "pass",
        "syntax_error",
        "formula_error",
        "design_error",
        "fe_parameter_error",
        "other_error",
    ]

    if label not in valid:
        warnings.warn(
            f"coarse_classify: unknown label '{label}' (raw: {raw[:60]!r}). "
            f"Falling back to 'other_error'."
        )
        label = "other_error"

    return label


# ======================================================================
# deterministic fix-eligibility gates (no LLM calls)
#
# Before the FIRST repair round the reflector computes, for each fix
# type, whether it has REPAIR ROOM under the extracted/designed rules:
#   * syntax_fix / free_fix — always usable (the code itself is the
#     subject; the fallback may touch anything).
#   * formula_fix — usable while at least one formula part is DESIGNED
#     (extracted formulas are ground truth).
#   * design_fix — usable while at least one design VALUE is designed:
#     physical constants, Afac (when ell is not extracted), the
#     CharacteristicFunction values, phi, the boundary labels, a
#     boundary's physical properties (incl. a field's designed
#     Dirichlet condition and the pen-existence repair), or (when the
#     mesh is not a provided/frozen one) the border definitions.
#   * fe_param_fix — usable while at least one of hmin/hmax/diffusion/
#     v0 is designed, or the mesh resolution is mutable, or a field's
#     fe_space is designed.
# ======================================================================

# Designed keys whose NAME starts with one of these prefixes are
# formula/model parts (they carry a <key>_source sibling).
_FORMULA_PREFIXES = (
    "objective",
    "state_",
    "adjoint",
    "shape_derivative",
    "newton",
    "expression",
    "characteristic",
    "Lagrangian",   # canonical capitalised key (item 7)
    "lagrangian",   # tolerated LLM casing variant (normalize renames it)
    "is_nonlinear",
)

# Numeric setup_optimization parameters that fe_param_fix may write.
_FE_PARAM_KEYS = ("hmin", "hmax", "diffusion", "v0")

# The phi key: current-format specs use "phi", older on-disk specs use
# "level_set_initial".
_PHI_KEYS = ("phi", "level_set_initial")


def _source_of(mapping: dict, key: str) -> str:
    """The source annotation of *key* in *mapping*.

    Accepts the current per-key ``<key>_source`` sibling and the older
    object-level ``source`` (both appear in on-disk specs).
    """
    return str(mapping.get(f"{key}_source", mapping.get("source", "designed")))


def _extracted_constant_names(spec: dict) -> set:
    """Names (and latex names) of extracted constants — immutable."""
    extracted = spec.get("extracted") or {}
    names = set()
    for c in extracted.get("constants") or []:
        if not isinstance(c, dict):
            continue
        for k in ("name", "latex"):
            if isinstance(c.get(k), str) and c[k]:
                names.add(c[k])
    return names


def _designed_formula_part(spec: dict) -> bool:
    """Whether any formula/model part is DESIGNED (replaceable)."""
    designed = spec.get("designed") or {}
    if not isinstance(designed, dict):
        return False
    for key, value in designed.items():
        if key.endswith(("_source", "_rationale")):
            continue
        if value and key.startswith(_FORMULA_PREFIXES):
            if _source_of(designed, key) == "designed":
                return True
    return False


def available_fix_types(spec: dict, main_py: str, mesh_frozen: bool) -> dict:
    """The pre-computed usable fix types (before the first repair round).

    Returns ``{fix_type: (usable, reason)}`` — *reason* is a short
    human-readable explanation (empty when usable).
    """
    out = {
        "syntax_fix": (True, ""),
        "free_fix": (True, ""),
        "formula_fix": (False, ""),
        "design_fix": (False, ""),
        "fe_param_fix": (False, ""),
    }
    if not isinstance(spec, dict):
        return out
    designed = spec.get("designed") or {}
    opt = designed.get("optimization_config") or {}

    # ── formula_fix ────────────────────────────────────────────────
    if _designed_formula_part(spec):
        out["formula_fix"] = (True, "")

    # ── design_fix ─────────────────────────────────────────────────
    reasons = []
    hit = False
    consts = designed.get("constants") or []
    extracted_names = _extracted_constant_names(spec)
    designed_const = False
    for c in consts:
        if not isinstance(c, dict):
            continue
        if str(c.get("source", "designed")) != "designed":
            continue
        name = c.get("name")
        if name in extracted_names or c.get("latex") in extracted_names:
            continue
        designed_const = True
        break
    if designed_const:
        hit = True
    afac_free = not (extracted_names & {"ell", "Afac", "\\ell"}) \
        and bool(re.search(r"\bAfac\s*=\s*Constant\s*\(", main_py))
    if afac_free:
        hit = True
    chi = designed.get("characteristic_function")
    if isinstance(chi, dict) and str(chi.get("source", "designed")) == "designed":
        hit = True
    for k in _PHI_KEYS:
        if k in opt and _source_of(opt, k) == "designed":
            hit = True
    for k in ("boundary_labels", "fixed_labels"):
        if k in opt and _source_of(opt, k) == "designed":
            hit = True
    for b in (designed.get("domain") or {}).get("boundaries", []) or []:
        if isinstance(b, dict) and str(b.get("source", "designed")) == "designed":
            hit = True
            break
    # a designed Dirichlet condition of a declared field is a boundary
    # physical property — design repair room (the pen-existence repair
    # rides on it too)
    for f in designed.get("fields") or []:
        if not isinstance(f, dict):
            continue
        if (f.get("dirichlet_value") not in (None, "")
                and _source_of(f, "dirichlet_value") == "designed") \
                or (isinstance(f.get("dirichlet_labels"), list)
                    and f.get("dirichlet_labels")
                    and _source_of(f, "dirichlet_labels") == "designed"):
            hit = True
            break
    if not mesh_frozen:
        hit = True  # the border definitions are mutable
    if hit:
        out["design_fix"] = (True, "")
    else:
        reasons.append("no designed design values (constants/Afac/chi/phi/"
                       "labels/boundaries/Dirichlet/mesh are all extracted "
                       "or frozen)")

    # ── fe_param_fix ───────────────────────────────────────────────
    fe_hit = False
    for k in _FE_PARAM_KEYS:
        if k in opt and _source_of(opt, k) == "designed":
            fe_hit = True
    if not mesh_frozen:
        fe_hit = True  # the resolution is mutable
    for f in designed.get("fields") or []:
        if isinstance(f, dict) and f.get("fe_space") \
                and _source_of(f, "fe_space") == "designed":
            fe_hit = True
    if fe_hit:
        out["fe_param_fix"] = (True, "")
    else:
        reasons.append("no designed fe parameters (hmin/hmax/diffusion/v0/"
                       "resolution/fe_space are all extracted or frozen)")
    return out


def available_types_text(available: dict) -> str:
    """Render the usable fix types for the classification prompt: ONLY
    the usable error types and their repair methods are shown."""
    label = {
        "syntax_fix": "syntax_error — repaired by syntax_fix (code repair with the text model)",
        "formula_fix": "formula_error — repaired by formula_fix (formula-related code)",
        "design_fix": "design_error — repaired by design_fix (design values: physical-parameter numbers, Afac, CharacteristicFunction values, phi, the boundary labels, boundary physical properties incl. the fields' Dirichlet conditions, the pen penalty implementation — existence only, its value is always 1e15, mesh definitions — borders and square()/buildmesh count as borders)",
        "fe_param_fix": "fe_parameter_error — repaired by fe_param_fix (hmin/hmax/diffusion/v0, mesh resolution, field precision P1/P2)",
        "free_fix": "other_error — repaired by free_fix (free repair, the fallback)",
    }
    lines = ["## Usable error types (choose ONLY among these)"]
    for fix_type, (usable, _reason) in available.items():
        if usable:
            lines.append(f"- {label[fix_type]}")
    return "\n".join(lines)


def fix_type_for(coarse: str, available: dict) -> tuple:
    """Map a coarse category to the fix type using the PRE-COMPUTED
    availability (no implicit remapping: an unusable type falls back to
    free_fix deterministically)."""
    mapping = {
        "syntax_error": ("syntax_fix", ""),
        "formula_error": ("formula_fix", "formula_fix has no repair room (formulas extracted) — falling back to free_fix"),
        "design_error": ("design_fix", "design_fix has no repair room (design values extracted/frozen) — falling back to free_fix"),
        "fe_parameter_error": ("fe_param_fix", "fe_param_fix has no repair room (fe parameters extracted/frozen) — falling back to free_fix"),
        "other_error": ("free_fix", ""),
    }
    fix_type, note = mapping.get(coarse, ("free_fix", ""))
    if fix_type in available and available[fix_type][0]:
        return fix_type, ""
    return "free_fix", note


# ======================================================================
# quality check — used when MinTh.vtu exists
# ======================================================================


def quality_check(agent, problem_name: str, context_text: str, images=None,
                  image_section: str = "") -> str:
    """
    Check whether a MinTh.vtu result is genuinely successful.

    Args:
        agent: CodeAgent for LLM calls.
        problem_name: Name of the problem.
        context_text: Full working directory contents.
        images: Optional PNG paths (one per design snapshot).
        image_section: Text describing the attached image(s).

    Returns:
        "pass" or "wrong_result".
    """
    from .prompts import QUALITY_CHECK_SYSTEM, QUALITY_CHECK_USER

    prompt_list = [
        {"role": "system", "content": QUALITY_CHECK_SYSTEM},
        {"role": "user", "content": QUALITY_CHECK_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke_multimodal(
        {"problem_name": problem_name, "context_text": context_text,
         "image_section": image_section},
        images=images,
    )

    label = _canonical_label(raw)
    if label == "pass":
        return "pass"
    if label == "wrong_result":
        return "wrong_result"
    warnings.warn(
        f"quality_check: unknown label '{label}' (raw: {raw[:60]!r}). "
        f"Falling back to 'wrong_result'."
    )
    return "wrong_result"


# ======================================================================
# generic Constant(...) value read/replace — shared by the permission
# builders and the fix enforcers (both in fixer.py).  Mask-aware so a
# comment/string mention never matches.
# ======================================================================

_CONSTANT_RE_TMPL = (
    r"\b{name}\s*=\s*Constant\(\s*r?['\"][^'\"]*['\"]\s*,\s*([-+0-9.eE]+)"
)


def _constant_value(code: str, name: str) -> Optional[float]:
    """The numeric value of ``name = Constant(...)`` in *code*."""
    from utils import code_masked_positions

    masked = code_masked_positions(code)
    for m in re.finditer(_CONSTANT_RE_TMPL.format(name=re.escape(name)), code):
        if m.start() in masked:
            continue
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None


def _constant_replace(code: str, name: str, value: float) -> str:
    """Replace the numeric value of ``name = Constant(...)`` (mask-aware)."""
    from utils import code_masked_positions

    masked = code_masked_positions(code)
    new = f"{value:g}"
    out = []
    pos = 0
    for m in re.finditer(_CONSTANT_RE_TMPL.format(name=re.escape(name)), code):
        if m.start() in masked:
            continue
        out.append(code[pos:m.start(1)])
        out.append(new)
        pos = m.end(1)
    out.append(code[pos:])
    return "".join(out)
