"""
Static checks for the Initialization phase — problem spec and mesh spec.

Check strategy (per project spec):

- ``syntax`` level — DETERMINISTIC ONLY, reading ONLY the generated
  content (the spec JSONs themselves, never the problem input files):
  JSON shape, required keys, types, enums, source annotations, numeric
  sanity, and the structural consistency invariants between generated
  parts (label lists disjoint, no label 10 in the lists, roles matching
  the lists, edges matching the code, verbatim immutability).

- ``extraction`` and ``design`` levels — LLM ONLY: the merged review in
  ``Initialization.spec_generation`` (transcription fidelity against the
  sources, design rules, physical plausibility, information omissions —
  judged in ONE review pass).
  Problem file formats are often complex or messy, so no deterministic
  text matching is attempted at these levels.

Every check returns a list of flat error dicts::

    {"check": <name>, "level": "syntax"|"extraction"|"design",
     "severity": "error"|"warning", "message": <actionable English sentence>}

Hard errors (severity=error) drive the generate -> review -> repair loop
of ``Initialization.spec_generation`` (bounded by
``configs.MAX_SPEC_FIXES``); warnings never block.

The deterministic normalizers (``normalize_*``) run BEFORE the checks and
mechanically enforce the format conventions (Lagrange-multiplier rule,
enum spelling, label stringification, source backfill), so the repairs
stay reserved for real semantic failures.
"""

import json
import math
import re
import sys
import warnings
from pathlib import Path

# Allow local testing without importing the full package.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Initialization.phi_rules import _get_sympy  # noqa: E402 (lazy sympy)

# ──────────────────────────────────────────────────────────────────────
# error-dict helpers
# ──────────────────────────────────────────────────────────────────────


def _err(check, level, message, severity="error"):
    return {"check": check, "level": level, "severity": severity, "message": message}


def hard_errors(checks):
    """Only the severity=error checks (warnings never block)."""
    return [c for c in checks if c.get("severity") == "error"]


def format_errors(checks):
    """One line per check, for feedback blocks and console output."""
    return "\n".join(f"- {c['level']} | {c['check']} | {c['message']}" for c in checks)


def files_block(input_files, max_chars=None):
    """The shared ``=== FILE: x ===`` block sent to the LLM prompts."""
    if max_chars is None:
        from configs import MAX_FILE_CHARS

        max_chars = MAX_FILE_CHARS
    parts = []
    for fname, content in (input_files or {}).items():
        truncated = (
            content if len(content) <= max_chars
            else content[:max_chars] + "\n\n... [truncated]"
        )
        parts.append(f"=== FILE: {fname} ===\n{truncated}")
    return "\n\n".join(parts)


def _to_float(v):
    """Numeric tolerance for LLM JSON quirks: numbers may arrive as numeric
    strings ("0.01").  Returns a float, or None when *v* is not numeric."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _to_int(v):
    """Like :func:`_to_float` but integral only (returns an int or None)."""
    f = _to_float(v)
    if f is None or not f.is_integer():
        return None
    return int(f)


def _as_bool(v):
    """Boolean tolerance: accepts True/False and "true"/"false" strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "false"}:
            return s == "true"
    return None


# ──────────────────────────────────────────────────────────────────────
# enums and shape constants (mirror the prompt schemas)
# ──────────────────────────────────────────────────────────────────────

_PROBLEM_TYPES = {"shape_optimization", "topology_optimization"}
_KINDS = {"Dirichlet", "Neumann", "Robin"}
_ROLES = {"design", "fixed"}
_FE_SPACES = {"P1", "P2"}
_FIELD_TYPES = {"scalar", "vector"}
_INTERFACE_PHYSICALS = {"value_continuity", "flux_continuity"}
_SOURCES = {"extracted", "designed"}
_MESH_CATEGORIES = {"code", "description", "absent"}
_MESH_SOURCES = {"extracted", "designed"}

# scalar keys of the extracted spec that carry a "<key>_source" sibling
_EXTRACTED_SCALAR_KEYS = (
    "problem_type", "problem_name", "dimension", "domain_name",
    "domain_description", "is_nonlinear", "mesh_variable", "mesh_file",
    "mesh_labels_used",
)
# list keys whose items carry an in-item "source"
_EXTRACTED_LIST_KEYS = ("boundaries", "fields", "test_functions", "constants",
                        "expressions", "interfaces", "domains",
                        "extra_requirements")
# sub-object keys that carry an object-level "source"
_EXTRACTED_OBJECT_KEYS = ("objective", "state_equation", "newton_linearisation",
                          "adjoint_equation", "shape_derivative",
                          "characteristic_function", "optimization_params")

# scalar keys of the designed spec that carry a "<key>_source" sibling
_DESIGNED_SCALAR_KEYS = (
    "problem_type", "problem_name", "dimension", "domain_name",
    "domain_description", "objective_name", "objective_description",
    "state_description", "state_weak_lhs", "state_weak_rhs",
    "state_is_nonlinear", "check_units", "adjoint_description",
    "adjoint_weak_lhs", "adjoint_weak_rhs",
)
_DESIGNED_LIST_KEYS = ("boundaries", "interfaces", "fields", "test_functions",
                       "constants", "expressions", "domains")


def _norm_enum(v, canonical, aliases):
    if v in canonical:
        return v
    if isinstance(v, str):
        s = v.strip()
        if s in canonical:
            return s
        if s.lower() in aliases:
            return aliases[s.lower()]
    return v


_KIND_ALIASES = {"d": "Dirichlet", "n": "Neumann", "r": "Robin"}
_ROLE_ALIASES = {"movable": "design", "frozen": "fixed",
                 "non-design": "fixed", "non_design": "fixed"}
_TYPE_ALIASES = {"displacement": "vector", "velocity": "vector", "temperature": "scalar",
                 "scalar field": "scalar", "vector field": "vector"}
_PHYSICAL_ALIASES = {"value": "value_continuity", "flux": "flux_continuity",
                     "value continuity": "value_continuity", "flux continuity": "flux_continuity"}


def _norm_fe_space(v):
    if v in _FE_SPACES:
        return v
    if isinstance(v, str):
        s = v.strip()
        if s in _FE_SPACES:
            return s
        if s.upper() in _FE_SPACES:
            return s.upper()
        # the convention is a SCALAR space P1/P2 (the DSL expands it
        # per component) — a LIST form like "[P2, P2]" or "P2, P2"
        # collapses to the scalar space
        m = re.fullmatch(r"\[\s*(P[12])\s*(?:,\s*\1\s*)*\]", s, re.I)
        if m:
            return m.group(1).upper()
        m = re.fullmatch(r"(P[12])\s*(?:,\s*\1\s*)+", s, re.I)
        if m:
            return m.group(1).upper()
    return v


def _norm_labels(lst):
    """Every label as a string (LLMs sometimes emit JSON numbers, a
    single "1" / "1,2" string instead of an array, or structured
    entries like {"label": "1", "source": ...} — dict entries are
    flattened to their "label" value)."""
    if isinstance(lst, str):
        lst = lst.replace(" ", "").split(",")
    if not isinstance(lst, list):
        return lst
    out = []
    for x in lst:
        if isinstance(x, dict):
            x = x.get("label", x)
        out.append(str(x))
    return out


# ──────────────────────────────────────────────────────────────────────
# deterministic post-parse normalization (LLM-output tolerance)
#
# Mechanical format/convention items are enforced HERE, deterministically,
# instead of relying on the repair retries: enum spelling, label
# stringification, missing source annotations, and the physics
# conventions (fe_space <-> type, design_role from the extracted ground
# truth or the Dirichlet->fixed convention).  The checks remain the
# auditors; the normalizers make the phases robust so retries stay for
# real semantic failures.
# ──────────────────────────────────────────────────────────────────────


def _backfill_item_sources(items, fallback):
    """Give every array item without a ``source`` the *fallback* source —
    a single missing item-source must not burn a repair retry."""
    if not isinstance(items, list):
        return items
    for it in items:
        if isinstance(it, dict) and it.get("source") not in _SOURCES:
            it["source"] = fallback
    return items


def _backfill_scalar_sources(d, keys, source_fn):
    """Give every present scalar without a ``<key>_source`` sibling the
    source *source_fn(key)* returns."""
    for k in keys:
        if k in d and d.get(f"{k}_source") not in _SOURCES:
            d[f"{k}_source"] = source_fn(k)
    return d


def normalize_extracted(ex):
    """Normalize a parsed extraction result (problem spec)."""
    if not isinstance(ex, dict):
        return ex
    # LLM key alias: the prompt section is named after the KEY, but the LLM
    # sometimes echoes the section topic instead — rename deterministically
    if "optimization_params" not in ex and isinstance(ex.get("setup_optimization"), dict):
        warnings.warn("normalize: renamed LLM key 'setup_optimization' to 'optimization_params'")
        ex["optimization_params"] = ex.pop("setup_optimization")
    # LLM key alias: the canonical key is CAPITALISED ('Lagrangian') —
    # a lowercase variant is renamed deterministically (the LLM
    # fluctuates between runs; downstream reads the capital form)
    if "lagrangian" in ex and "Lagrangian" not in ex:
        warnings.warn("normalize: renamed LLM key 'lagrangian' to 'Lagrangian' (canonical capitalisation)")
        ex["Lagrangian"] = ex.pop("lagrangian")
    # an EMPTY optimization_params object is noise (nothing was stated in
    # the sources) — drop it instead of carrying a hollow annotated object
    op = ex.get("optimization_params")
    if isinstance(op, dict) and not any(
        v is not None and v != "" for k, v in op.items() if k != "source"
    ):
        ex.pop("optimization_params", None)
    # every extraction entry is ground truth by definition — backfill the
    # scalar _source siblings deterministically
    _backfill_scalar_sources(ex, _EXTRACTED_SCALAR_KEYS, lambda k: "extracted")
    if "mesh_labels_used" in ex:
        ex["mesh_labels_used"] = _norm_labels(ex.get("mesh_labels_used"))
    for k in _EXTRACTED_LIST_KEYS:
        if k in ex:
            _backfill_item_sources(ex[k], "extracted")
    for k in _EXTRACTED_OBJECT_KEYS:
        obj = ex.get(k)
        if isinstance(obj, dict) and obj.get("source") != "extracted":
            obj["source"] = "extracted"
    for b in ex.get("boundaries") or []:
        if isinstance(b, dict):
            b["kind"] = _norm_enum(b.get("kind"), _KINDS, _KIND_ALIASES)
            b["design_role"] = _norm_enum(b.get("design_role"), _ROLES, _ROLE_ALIASES)
            if b.get("mesh_label") is not None:
                b["mesh_label"] = str(b["mesh_label"])
    # dirichlet_labels may arrive as LaTeX boundary names (\\Gamma_D) — map
    # them to the mesh labels via the extracted boundaries, falling back to
    # the digits in the entry when they match a known mesh label
    name_to_label = {
        str(b.get("name")): str(b.get("mesh_label"))
        for b in (ex.get("boundaries") or [])
        if isinstance(b, dict) and b.get("name") is not None and b.get("mesh_label") is not None
    }
    known_mesh = (
        {str(l) for l in ex.get("mesh_labels_used")}
        if isinstance(ex.get("mesh_labels_used"), list) else set()
    )
    # a boundary named like Gamma_4 whose digit is a known mesh label gets
    # its mesh_label filled deterministically (the 编号 reconciliation)
    for b in (ex.get("boundaries") or []):
        if isinstance(b, dict) and b.get("mesh_label") is None:
            m = re.search(r"\d+", str(b.get("name") or ""))
            if m and m.group(0) in known_mesh:
                b["mesh_label"] = m.group(0)
    for f in ex.get("fields") or []:
        if isinstance(f, dict):
            f["type"] = _norm_enum(f.get("type"), _FIELD_TYPES, _TYPE_ALIASES)
            f["fe_space"] = _norm_fe_space(f.get("fe_space"))
            dl = _norm_labels(f.get("dirichlet_labels"))
            if isinstance(dl, list):
                mapped = []
                for l in dl:
                    s = str(l)
                    if s in name_to_label:
                        mapped.append(name_to_label[s])
                    else:
                        m = re.search(r"\d+", s)
                        mapped.append(m.group(0) if m and m.group(0) in known_mesh else s)
                f["dirichlet_labels"] = mapped
    # test functions carry ONLY name/latex/type/fe_space/domain — strip the
    # extras on the extracted side too, and on fields that are actually
    # test functions misplaced under `fields` (delta-prefixed names)
    for tf in (ex.get("test_functions") or []):
        if isinstance(tf, dict):
            for extra in ("unit", "dirichlet_value", "dirichlet_labels"):
                tf.pop(extra, None)
    for f in (ex.get("fields") or []):
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "")
        if name.lower().startswith(("delta", "δ")):
            for extra in ("latex", "type", "unit", "dirichlet_value",
                          "dirichlet_labels"):
                f.pop(extra, None)
    for it in ex.get("interfaces") or []:
        if isinstance(it, dict):
            it["physical"] = _norm_enum(it.get("physical"), _INTERFACE_PHYSICALS, _PHYSICAL_ALIASES)
    return ex


def normalize_designed(ds, extracted=None):
    """Normalize a parsed design result (problem spec)."""
    if not isinstance(ds, dict):
        return ds
    extracted = extracted or {}
    # LLM key alias: the prompt section is named after the KEY, but the LLM
    # sometimes echoes the section topic instead — rename deterministically
    if "optimization_config" not in ds and isinstance(ds.get("setup_optimization"), dict):
        warnings.warn("normalize: renamed LLM key 'setup_optimization' to 'optimization_config'")
        ds["optimization_config"] = ds.pop("setup_optimization")
    # LLM key alias: the canonical key is CAPITALISED ('Lagrangian') —
    # a lowercase variant is renamed deterministically (the LLM
    # fluctuates between runs; downstream reads the capital form)
    if "lagrangian" in ds and "Lagrangian" not in ds:
        warnings.warn("normalize: renamed LLM key 'lagrangian' to 'Lagrangian' (canonical capitalisation)")
        ds["Lagrangian"] = ds.pop("lagrangian")
    # FLATTEN nested {"value": X, "source": S} scalars — the format rule is
    # flat, but the LLM sometimes wraps a value; unwrap deterministically
    for k in list(ds.keys()):
        v = ds[k]
        if (isinstance(v, dict) and not k.endswith("_source")
                and "value" in v and v.get("source") in _SOURCES):
            ds[k] = v["value"]
            if f"{k}_source" not in ds:
                ds[f"{k}_source"] = v["source"]

    # flat designed keys that live in NESTED extracted sub-objects
    _NESTED_MAP = {
        "objective_name": ("objective", "name"),
        "objective_description": ("objective", "description"),
        "state_description": ("state_equation", "description"),
        "state_weak_lhs": ("state_equation", "weak_form_lhs"),
        "state_weak_rhs": ("state_equation", "weak_form_rhs"),
        "adjoint_description": ("adjoint_equation", "description"),
        "adjoint_weak_lhs": ("adjoint_equation", "weak_form_lhs"),
        "adjoint_weak_rhs": ("adjoint_equation", "weak_form_rhs"),
    }

    def _ex_val(k):
        if k == "state_is_nonlinear":
            return extracted.get("is_nonlinear")
        if k in _NESTED_MAP:
            obj_key, field = _NESTED_MAP[k]
            obj = extracted.get(obj_key)
            return obj.get(field) if isinstance(obj, dict) else None
        return extracted.get(k)

    # backfill scalar _source siblings: equal to the extracted value ->
    # carried-over ground truth, anything else was designed
    _backfill_scalar_sources(
        ds, _DESIGNED_SCALAR_KEYS,
        lambda k: "extracted" if k in extracted and ds[k] == _ex_val(k) else "designed",
    )
    # DETERMINISTIC IMMUTABILITY: an "extracted"-marked scalar is VERBATIM
    # ground truth — restore the extracted value where the LLM embellished
    # it (also resolves the is_nonlinear -> state_is_nonlinear naming)
    for k in _DESIGNED_SCALAR_KEYS:
        if ds.get(f"{k}_source") == "extracted" and _ex_val(k) is not None:
            ds[k] = _ex_val(k)
    if "objective_terms" in ds and isinstance(ds.get("objective_terms"), list):
        ex_obj = extracted.get("objective")
        ex_terms = ex_obj.get("terms") if isinstance(ex_obj, dict) else None
        srcs = ds.get("objective_terms_source")
        if not isinstance(srcs, list) or len(srcs) != len(ds["objective_terms"]):
            ds["objective_terms_source"] = [
                "extracted" if isinstance(ex_terms, list) and t in ex_terms else "designed"
                for t in ds["objective_terms"]
            ]
    # backfill per-item sources: an item matching an extracted one by name
    # is carried-over ground truth, anything else was designed
    ex_names = {
        k: {str(it.get("name")) for it in (extracted.get(k) or [])
            if isinstance(it, dict) and it.get("name")}
        for k in _DESIGNED_LIST_KEYS
    }
    for k in _DESIGNED_LIST_KEYS:
        items = ds.get(k)
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict) and it.get("source") not in _SOURCES:
                it["source"] = "extracted" if str(it.get("name")) in ex_names[k] else "designed"
    # DETERMINISTIC IMMUTABILITY: an "extracted"-marked item is VERBATIM
    # ground truth — restore the extracted content where the LLM
    # embellished it, and relabel invented items mislabeled "extracted"
    _copy_keys = {
        "boundaries": ("name", "definition", "kind", "design_role"),
        "fields": ("name", "latex", "type", "fe_space", "unit",
                   "dirichlet_value", "dirichlet_labels"),
        "test_functions": ("name", "latex", "type", "fe_space"),
        "constants": ("name", "latex", "value", "description", "unit"),
        "expressions": ("name", "latex", "formula"),
        "interfaces": ("name", "domains", "physical", "definition"),
    }
    for k, keys in _copy_keys.items():
        ex_items = {
            str(it.get("name")): it
            for it in (extracted.get(k) or [])
            if isinstance(it, dict) and it.get("name") is not None
        }
        items = ds.get(k)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or it.get("source") != "extracted":
                continue
            ex_it = ex_items.get(str(it.get("name")))
            if ex_it is None:
                it["source"] = "designed"  # invented entry, mislabeled
                continue
            for f in keys:
                if f in ex_it and ex_it[f] is not None:
                    it[f] = ex_it[f]
            if k == "boundaries" and ex_it.get("mesh_label") is not None:
                it["label"] = str(ex_it["mesh_label"])
    # every extracted expression is ground truth and must appear in the
    # designed expressions too — carry over missing ones verbatim
    ex_expr = {
        str(it.get("name")): it
        for it in (extracted.get("expressions") or [])
        if isinstance(it, dict) and it.get("name") is not None
    }
    ds_expr = ds.get("expressions")
    if isinstance(ds_expr, list) and ex_expr:
        have = {str(it.get("name")) for it in ds_expr if isinstance(it, dict)}
        for name, it in ex_expr.items():
            if name not in have:
                ds_expr.append(dict(it, source="extracted"))
    # extra_requirements items carry 'requirement' (no 'name') — backfill
    # their sources by content match against the extracted requirements
    xr = ds.get("extra_requirements")
    if isinstance(xr, list):
        ex_reqs = {
            str(it.get("requirement"))
            for it in (extracted.get("extra_requirements") or [])
            if isinstance(it, dict)
        }
        for it in xr:
            if not isinstance(it, dict):
                continue
            if it.get("source") not in _SOURCES:
                it["source"] = "extracted" if str(it.get("requirement")) in ex_reqs else "designed"
            elif it.get("source") == "extracted" and str(it.get("requirement")) not in ex_reqs:
                it["source"] = "designed"  # invented entry mislabeled extracted
    # explicit extracted design_role is ground truth; the convention
    # (Dirichlet -> fixed, Neumann/Robin -> design) fills the rest
    ex_roles = {
        str(eb.get("mesh_label")): eb.get("design_role")
        for eb in (extracted.get("boundaries") or [])
        if isinstance(eb, dict) and eb.get("mesh_label") is not None
        and eb.get("design_role") in _ROLES
    }
    for b in ds.get("boundaries") or []:
        if not isinstance(b, dict):
            continue
        b["kind"] = _norm_enum(b.get("kind"), _KINDS, _KIND_ALIASES)
        b["design_role"] = _norm_enum(b.get("design_role"), _ROLES, _ROLE_ALIASES)
        if b.get("label") is not None:
            b["label"] = str(b["label"])
        if b.get("design_role") not in _ROLES:
            key = str(b.get("label")) if b.get("label") is not None else None
            if key in ex_roles:
                b["design_role"] = ex_roles[key]
            elif b.get("kind") == "Dirichlet":
                b["design_role"] = "fixed"
            elif b.get("kind") in ("Neumann", "Robin"):
                b["design_role"] = "design"
        # the label-10 (material boundary) entry carries NO design role —
        # it never enters boundary_labels / fixed_labels
        if str(b.get("label")) == "10":
            b.pop("design_role", None)
    for f in ds.get("fields") or []:
        if not isinstance(f, dict):
            continue
        t, fs = f.get("type"), f.get("fe_space")
        if t not in _FIELD_TYPES and fs in _FE_SPACES:
            f["type"] = "vector" if fs == "P2" else "scalar"
        elif fs not in _FE_SPACES and t in _FIELD_TYPES:
            f["fe_space"] = "P2" if t == "vector" else "P1"
        f["type"] = _norm_enum(f.get("type"), _FIELD_TYPES, _TYPE_ALIASES)
        f["fe_space"] = _norm_fe_space(f.get("fe_space"))
        f["dirichlet_labels"] = _norm_labels(f.get("dirichlet_labels"))
        if f.get("domain") is not None:
            f["domain"] = str(f["domain"])
        # the physics convention for units: displacement in meter,
        # temperature in kelvin (enforced deterministically, per the
        # design prompt's convention)
        if f.get("unit") is None and f.get("type") in _FIELD_TYPES:
            f["unit"] = "meter" if f.get("type") == "vector" else "kelvin"
            f["unit_source"] = "designed"
        # a Dirichlet field with labels but no value is clamped to zero
        # (the standard homogeneous Dirichlet convention)
        if f.get("dirichlet_value") is None and f.get("dirichlet_labels"):
            f["dirichlet_value"] = "0"
            f["dirichlet_value_source"] = "designed"
    for tf in (ds.get("test_functions") or []):
        if not isinstance(tf, dict):
            continue
        # a test function carries ONLY name/fe_space/domain
        for extra in ("latex", "type", "unit", "dirichlet_value",
                      "dirichlet_labels"):
            tf.pop(extra, None)
    # test functions sometimes land in `fields` by mistake (names like
    # "delta T" / "delta_u") — MOVE them deterministically to
    # test_functions with only name/fe_space/domain (plus their source)
    # (a half-stripped entry left in fields would oscillate the
    # field_type check forever)
    moved = []
    kept = []
    for f in (ds.get("fields") or []):
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "")
        if name.lower().startswith(("delta", "δ")):
            moved.append({
                "name": f.get("name"),
                "fe_space": f.get("fe_space"),
                "domain": f.get("domain"),
                "source": f.get("source") or "extracted",
            })
        else:
            kept.append(f)
    if moved:
        ds["fields"] = kept
        tf = ds.get("test_functions")
        if not isinstance(tf, list):
            tf = []
        ds["test_functions"] = [t for t in tf if isinstance(t, dict)] + moved
    # a Dirichlet enforced by a PENALTY term is not declared on the
    # field (the penalty IS the enforcement — a ScalarField Dirichlet
    # would double-enforce).  The decision is SEMANTIC, never by the
    # constant's NAME: only a weak-form term of the shape
    # ``<c> * ( <field> ± <value> ) * ...`` — a prescribed-value
    # penalty (the FreeFEM workaround for a MULTI-VALUED Dirichlet) —
    # strips the field's declaration.  A spring term of the shape
    # ``<c> * inner(<field>, test)`` prescribes no value and never
    # strips.  A single-valued Dirichlet therefore always keeps its
    # declaration.
    weak_text = json.dumps(ds)
    for f in (ds.get("fields") or []):
        if not isinstance(f, dict):
            continue
        dl = f.get("dirichlet_labels")
        if not (isinstance(dl, list) and dl):
            continue
        fname = str(f.get("name") or "")
        if not fname:
            continue
        if re.search(rf"\(\s*{re.escape(fname)}\s*[+-]\s*[^)]*\)\s*\*", weak_text):
            for k in ("dirichlet_value", "dirichlet_value_source",
                      "dirichlet_labels", "dirichlet_labels_source"):
                f.pop(k, None)
    # designed mesh labels (deterministic, mirrors the extraction mapping)
    bnd_name_to_label = {
        str(b.get("name")): str(b.get("label"))
        for b in (ds.get("boundaries") or [])
        if isinstance(b, dict) and b.get("name") is not None and b.get("label") is not None
    }
    # names arrive with or without the leading LaTeX backslash
    bnd_name_to_label.update(
        {
            k.lstrip("\\"): v
            for k, v in bnd_name_to_label.items()
        }
    )
    known_mesh_ds = {
        str(l) for l in (extracted.get("mesh_labels_used") or [])
    } if isinstance(extracted.get("mesh_labels_used"), list) else set()
    # the extraction may omit mesh_labels_used — the DESIGNED boundary
    # labels are the authoritative numeric universe for the fallback
    designed_labels = set(bnd_name_to_label.values())
    for f in (ds.get("fields") or []):
        if not isinstance(f, dict):
            continue
        dl = f.get("dirichlet_labels")
        if isinstance(dl, list):
            mapped = []
            for l in dl:
                s = str(l)
                if s in bnd_name_to_label:
                    mapped.append(bnd_name_to_label[s])
                else:
                    m = re.search(r"\d+", s)
                    digit = m.group(0) if m else None
                    if digit is not None and (
                        digit in known_mesh_ds or digit in designed_labels
                    ):
                        mapped.append(digit)
                    else:
                        mapped.append(s)
            f["dirichlet_labels"] = mapped
    for it in ds.get("interfaces") or []:
        if isinstance(it, dict):
            it["physical"] = _norm_enum(it.get("physical"), _INTERFACE_PHYSICALS, _PHYSICAL_ALIASES)
            if it.get("label") is not None:
                it["label"] = str(it["label"])
    chi = ds.get("characteristic_function")
    if isinstance(chi, dict) and chi.get("source") not in _SOURCES:
        chi["source"] = (
            "extracted" if chi == extracted.get("characteristic_function")
            else "designed"
        )
    if isinstance(chi, dict) and chi.get("source") == "extracted" \
            and isinstance(extracted.get("characteristic_function"), dict):
        ex_chi = extracted["characteristic_function"]
        for f in ("latex", "strong_value", "weak_value"):
            if ex_chi.get(f) is not None:
                chi[f] = ex_chi[f]
    # deterministic defaults for the contrast values: the LARGER side is 1;
    # the smaller (weak) side follows the physics convention (1/100 for
    # solid mechanics when a vector field exists, 1/10 for pure
    # thermal/scalar problems)
    if isinstance(chi, dict):
        if not chi.get("strong_value"):
            chi["strong_value"] = "1"
        if not chi.get("weak_value"):
            any_vector = any(
                isinstance(f, dict) and f.get("type") == "vector"
                for f in (ds.get("fields") or [])
            )
            chi["weak_value"] = "1/100" if any_vector else "1/10"
    # the source's shape derivative is ground truth: when the design left
    # only a note/placeholder, carry the extracted formulas over verbatim
    ex_sd = extracted.get("shape_derivative")
    if isinstance(ex_sd, dict) \
            and not (ds.get("shape_derivative_volume") or ds.get("shape_derivative_boundary")):
        for k, key in (("volume_expression", "shape_derivative_volume"),
                       ("boundary_expression", "shape_derivative_boundary")):
            if ex_sd.get(k) is not None:
                ds[key] = ex_sd[k]
        if ex_sd.get("volume_expression") or ex_sd.get("boundary_expression"):
            ds["shape_derivative_source"] = "extracted"
    # the Dirichlet penalty constant pen = 1e15 must be LISTED whenever
    # the weak forms use it (the project convention, enforced here) —
    # scan the WHOLE designed block: the weak forms live in nested
    # structures whose key names evolved over the schema revisions.
    # pen is a FreeFEM WORKAROUND used ONLY when a field enforces two
    # or more DIFFERENT Dirichlet values — never backfill it for a
    # single-valued Dirichlet.
    weak_text = json.dumps(ds)
    multi_valued = False
    for f in (ds.get("fields") or []) if isinstance(ds.get("fields"), list) else []:
        if not isinstance(f, dict):
            continue
        dv = f.get("dirichlet_value")
        vals = list(dv.values()) if isinstance(dv, dict) else \
            (dv if isinstance(dv, list) else ([dv] if dv not in (None, "") else []))
        if len({str(v) for v in vals}) >= 2:
            multi_valued = True
            break
    if re.search(r"\bpen\b", weak_text) and multi_valued:
        consts = ds.get("constants")
        if isinstance(consts, list) and not any(
            isinstance(c, dict) and str(c.get("name", "")).lower() == "pen"
            for c in consts
        ):
            consts.append({
                "name": "pen", "latex": "\\mathrm{pen}",
                "value": 1e15, "unit": "N/m^3",
                "description": "Dirichlet penalty constant",
                "source": "designed",
                "rationale": "project convention: pen = 1e15 for the Dirichlet penalty",
            })
    # shared-source sub-groups: backfill the single source key
    if (ds.get("shape_derivative_volume") or ds.get("shape_derivative_boundary")) \
            and ds.get("shape_derivative_source") not in _SOURCES:
        ds["shape_derivative_source"] = (
            "extracted" if extracted.get("shape_derivative") else "designed"
        )
    if (ds.get("newton_description") or ds.get("newton_weak_form")) \
            and ds.get("newton_source") not in _SOURCES:
        ds["newton_source"] = "extracted" if extracted.get("newton_linearisation") else "designed"
    opt = ds.get("optimization_config")
    if isinstance(opt, dict):
        if opt.get("source") not in _SOURCES:
            opt["source"] = "designed"
        for k in ("boundary_labels", "fixed_labels"):
            if k in opt:
                opt[k] = _norm_labels(opt.get(k))
    return ds


def normalize_extracted_mesh(ex):
    """Normalize a parsed mesh extraction result."""
    if not isinstance(ex, dict):
        return ex
    if ex.get("source") != "extracted" and "category" in ex:
        ex["source"] = "extracted"
    # the pipeline always reads the mesh from "mesh.txt" — a
    # `savemesh(Th, "Mesh1.mesh")` OUTPUT filename must never leak into
    # the extracted mesh_file either
    if ex.get("mesh_file") not in (None, "", "mesh.txt"):
        warnings.warn(
            f"normalize: extracted mesh_file {ex.get('mesh_file')!r} "
            "overridden to the pipeline default 'mesh.txt'"
        )
        ex["mesh_file"] = "mesh.txt"
    ex["labels_found"] = _norm_labels(ex.get("labels_found"))
    for e in ex.get("edges") or []:
        if isinstance(e, dict) and e.get("label") is not None:
            e["label"] = str(e["label"])
    return ex


def normalize_designed_mesh(ds, extracted=None):
    """Normalize a parsed mesh design result."""
    if not isinstance(ds, dict):
        return ds
    extracted = extracted or {}
    # the pipeline always reads/writes the mesh as "mesh.txt" — a
    # `savemesh(Th, "Mesh1.mesh")` OUTPUT filename must never leak in as
    # the input filename
    if ds.get("mesh_file") not in (None, "", "mesh.txt"):
        warnings.warn(
            f"normalize: mesh_file {ds.get('mesh_file')!r} "
            "overridden to the pipeline default 'mesh.txt'"
        )
        ds["mesh_file"] = "mesh.txt"
    ds["labels_used"] = _norm_labels(ds.get("labels_used"))
    # backfill the per-scalar source annotations deterministically
    # (the legacy four-way mesh_code_source enum collapses into the two
    # base values — extracted_repaired -> extracted,
    # designed_from_description -> designed)
    if ds.get("mesh_code_source") == "extracted_repaired":
        ds["mesh_code_source"] = "extracted"
    elif ds.get("mesh_code_source") == "designed_from_description":
        ds["mesh_code_source"] = "designed"
    code_extracted = ds.get("mesh_code_source") == "extracted"
    if ds.get("n_source") not in _SOURCES:
        ds["n_source"] = "extracted" if code_extracted else "designed"
    if ds.get("mesh_file_source") not in _SOURCES:
        ds["mesh_file_source"] = (
            "extracted" if extracted.get("mesh_file") is not None else "designed"
        )
    if ds.get("dimension_source") not in _SOURCES:
        ds["dimension_source"] = "extracted" if code_extracted else "designed"
    if ds.get("labels_used_source") not in _SOURCES:
        same_labels = (
            {str(l) for l in (ds.get("labels_used") or [])}
            == {str(l) for l in (extracted.get("labels_found") or [])}
        )
        ds["labels_used_source"] = "extracted" if same_labels else "designed"
    if ds.get("domain_description_source") not in _SOURCES:
        ds["domain_description_source"] = (
            "extracted" if ds.get("domain_description") == extracted.get("geometry_description")
            else "designed"
        )
    # DETERMINISTIC IMMUTABILITY: an "extracted"-marked edge is VERBATIM —
    # restore the extracted content (role included) where the LLM changed it
    ex_edges = {
        str(e.get("name")): e
        for e in (extracted.get("edges") or [])
        if isinstance(e, dict) and e.get("name") is not None
    }
    for e in ds.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if e.get("label") is not None:
            e["label"] = str(e["label"])
        if e.get("source") not in _SOURCES:
            e["source"] = "extracted" if ds.get("mesh_code_source") == "extracted" else "designed"
        if e.get("source") == "extracted":
            ex_e = ex_edges.get(str(e.get("name")))
            if ex_e is not None:
                for f in ("param", "x", "y", "label", "role"):
                    if ex_e.get(f) is not None:
                        e[f] = str(ex_e[f])
    # AUTO-FILL missing edges from the ACTIVE code — the edges list is
    # mechanically derivable from the borders, so a missing entry is
    # filled deterministically instead of burning repair retries
    active_code = re.sub(r"//[^\n]*", "", ds.get("mesh_code") or "")
    # when the ACTIVE mesh is square(), its implicit labels are 1-4 only —
    # replace labels_used AND the edges with the four canonical square
    # sides (commented-out borders must not leak in).  The detection must
    # not depend on the transform being present — a bare square(n,m) is
    # still the unit square with implicit labels 1-4.
    sm = re.search(
        r"\bmesh\s+Th\s*=\s*square\s*\(\s*[^,)]+?\s*,\s*[^,)]+?\s*"
        r"(?:,\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\])?",
        active_code,
    )
    if sm:
        def _coef(s):
            try:
                f = float(re.search(r"[-+]?\d*\.?\d+", s).group(0))
            except Exception:
                return 1.0
            return abs(f)
        # the transform [a*x+b, c*y+d] is optional (groups 1/2) — a bare
        # square(n, m) is the unit square
        L = _coef(sm.group(1)) if sm.group(1) else 1.0
        H = _coef(sm.group(2)) if sm.group(2) else 1.0
        warnings.warn(
            "normalize: active square() mesh — replaced labels_used/edges "
            "with the four square sides (implicit labels 1-4)"
        )
        ds["labels_used"] = ["1", "2", "3", "4"]
        ds["edges"] = [
            {"name": "S1", "param": f"y=0, 0<=x<={L:g}", "x": "x", "y": "0",
             "label": "1", "role": "bottom side (square default)",
             "source": "designed"},
            {"name": "S2", "param": f"x={L:g}, 0<=y<={H:g}", "x": f"{L:g}", "y": "y",
             "label": "2", "role": "right side (square default)",
             "source": "designed"},
            {"name": "S3", "param": f"y={H:g}, 0<=x<={L:g}", "x": "x", "y": f"{H:g}",
             "label": "3", "role": "top side (square default)",
             "source": "designed"},
            {"name": "S4", "param": f"x=0, 0<=y<={H:g}", "x": "0", "y": "y",
             "label": "4", "role": "left side (square default)",
             "source": "designed"},
        ]
    edges = ds.get("edges") if isinstance(ds.get("edges"), list) else []
    existing = {str(e.get("name")) for e in edges if isinstance(e, dict) and e.get("name")}
    _BORDER_FULL = re.compile(r"\bborder\s+(\w+)\s*\(([^)]*)\)\s*\{(.*?)\}", re.DOTALL)
    # GENERAL rule: auto-fill ONLY the borders that the ACTIVE buildmesh
    # actually references (they are the final mesh's edges).  A stray
    # active border that no buildmesh uses is never appended, and a
    # square() mesh (no buildmesh line) gets no auto-fill at all.
    bm = re.search(r"\bbuildmesh\s*\(", active_code)
    if bm:
        # balanced-paren scan of the buildmesh argument — `[^)]*` would
        # stop at the first inner `)` of a C01(50) term
        i, depth = bm.end(), 1
        start = i
        while i < len(active_code) and depth > 0:
            if active_code[i] == "(":
                depth += 1
            elif active_code[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        arg = active_code[start:i]
        built = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(\s*\d+", arg))
        for m in _BORDER_FULL.finditer(active_code):
            name, args, body = m.group(1), m.group(2), m.group(3)
            if name in existing or name not in built:
                continue
            xm = re.search(r"x\s*=\s*([^;]+);", body)
            ym = re.search(r"y\s*=\s*([^;]+);", body)
            lm = re.search(r"label\s*=\s*([^;]+);", body)
            edges.append({
                "name": name,
                "param": args.strip(),
                "x": xm.group(1).strip() if xm else "",
                "y": ym.group(1).strip() if ym else "",
                "label": str(lm.group(1).strip()) if lm else "",
                "role": f"border {name}",
                "source": "designed",
            })
    if edges:
        ds["edges"] = edges
    # labels_used must be EXACTLY the active mesh's non-10 boundary
    # labels — recompute it from the active code so it can never disagree
    # with the edges again (border meshes: their border labels; square
    # meshes: the replacement above already set 1-4)
    if not sm:
        active_labels = []
        for bm in _BORDER_FULL.finditer(active_code):
            lm = re.search(r"label\s*=\s*(\d+)", bm.group(3))
            if lm and lm.group(1) != "10" and lm.group(1) not in active_labels:
                active_labels.append(lm.group(1))
        if active_labels:
            ds["labels_used"] = sorted(active_labels, key=int)
    return ds


# ──────────────────────────────────────────────────────────────────────
# phi syntax helpers (generated content only — no mesh box needed)
# ──────────────────────────────────────────────────────────────────────


def _sympify_phi(phi):
    """sympy-parse a FreeFEM phi expression (max/min/abs word-replaced).

    Returns the sympy expression, or None on failure."""
    try:
        sp = _get_sympy()
        cleaned = re.sub(
            r"\b(max|min|abs)\b",
            lambda m: {"max": "Max", "min": "Min", "abs": "Abs"}[m.group(1).lower()],
            phi,
        )
        return sp.sympify(cleaned)
    except Exception:
        return None


def _phi_syntax_issues(phi):
    """Deterministic string-level checks on the level-set expression:
    paren balance and sympy parseability.  (Semantic rules — negativity,
    coverage of the fixed labels, solid fraction ~20-80% — are judged by
    the LLM design reviewer.)"""
    checks = []
    if not isinstance(phi, str) or not phi.strip():
        return [_err("phi_defined", "syntax", "phi must be a non-empty string")]
    if phi.count("(") != phi.count(")"):
        checks.append(_err("phi_balanced", "syntax", "phi has unbalanced parentheses"))
    if _sympify_phi(phi) is None:
        checks.append(_err("phi_parseable", "syntax", "phi is not a parseable expression"))
    return checks


def _phi_connectivity_issues(phi, L, H):
    """Deterministic connectivity check on the initial solid region.

    Samples phi < 0 (solid) on a coarse grid over the mesh box and BFS
    the solid cells — a DISCONNECTED solid region (e.g. two separated
    strips) is an ERROR, an all-void design is an ERROR."""
    checks = []
    if not isinstance(phi, str) or not phi.strip():
        return checks
    if L is None or H is None:
        return checks
    expr = _sympify_phi(phi)
    if expr is None:
        return checks
    sp = _get_sympy()
    x, y = sp.symbols("x y")
    n = 64
    solid = []
    for i in range(n):
        for j in range(n):
            try:
                v = complex(expr.subs({x: L * i / (n - 1), y: H * j / (n - 1)}))
            except Exception:
                continue
            if abs(v.imag) < 1e-12 and v.real < 0:
                solid.append((i, j))
    if not solid:
        checks.append(_err("phi_connectivity", "design",
                           "phi >= 0 everywhere on the sampled mesh box — an all-void design"))
        return checks
    # BFS over 4-neighbours: every solid cell must be reachable
    solid_set = set(solid)
    start = solid[0]
    seen = {start}
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (nx, ny) in solid_set and (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny))
    if len(seen) != len(solid_set):
        checks.append(_err("phi_connectivity", "design",
                           f"the initial solid region is DISCONNECTED: {len(solid_set) - len(seen)} "
                           "of the sampled solid cells are unreachable — the design must be one connected piece"))
    return checks


# ──────────────────────────────────────────────────────────────────────
# problem spec — extracted (syntax level, deterministic, generated-only)
# ──────────────────────────────────────────────────────────────────────


def check_extracted_problem(extracted):
    """Deterministic syntax checks on the extracted problem spec.

    Reads ONLY the generated JSON — never the problem input files (the
    fidelity of the extraction is verified by the LLM pass)."""
    checks = []
    if not isinstance(extracted, dict):
        return [_err("json_object", "syntax", "extracted spec is not a JSON object")]

    # ── 1. basic information ──────────────────────────────────────────
    pt = extracted.get("problem_type")
    if pt is None:
        checks.append(_err("problem_type_required", "syntax", "problem_type is required"))
    elif pt not in _PROBLEM_TYPES:
        checks.append(_err(
            "problem_type_enum", "syntax",
            f"problem_type must be one of {sorted(_PROBLEM_TYPES)}, got {pt!r}",
        ))
    if _to_int(extracted.get("dimension")) != 2:
        checks.append(_err(
            "dimension", "syntax",
            f"dimension must be 2, got {extracted.get('dimension')!r}",
        ))
    if (extracted.get("is_nonlinear") is not None
            and _as_bool(extracted.get("is_nonlinear")) is None):
        checks.append(_err("is_nonlinear", "syntax", "is_nonlinear must be a boolean"))

    # source annotations on every extracted scalar / list item / sub-object
    for k in _EXTRACTED_SCALAR_KEYS:
        if k in extracted and extracted.get(f"{k}_source") != "extracted":
            checks.append(_err(
                "source_annotation", "syntax",
                f"extracted scalar {k!r} is missing its '{k}_source': 'extracted' sibling",
            ))
    for k in _EXTRACTED_LIST_KEYS:
        if k in extracted:
            items = extracted[k]
            if not isinstance(items, list):
                checks.append(_err("list_type", "syntax", f"extracted {k!r} must be a list"))
                continue
            for i, it in enumerate(items):
                if not isinstance(it, dict) or it.get("source") != "extracted":
                    checks.append(_err(
                        "source_annotation", "syntax",
                        f"extracted {k}[{i}] is missing \"source\": \"extracted\"",
                    ))
    for k in _EXTRACTED_OBJECT_KEYS:
        obj = extracted.get(k)
        if obj is not None:
            if not isinstance(obj, dict):
                checks.append(_err("object_type", "syntax", f"extracted {k!r} must be an object"))
            elif obj.get("source") != "extracted":
                checks.append(_err(
                    "source_annotation", "syntax",
                    f"extracted sub-object {k!r} is missing \"source\": \"extracted\"",
                ))

    # ── 2. boundaries / interfaces ────────────────────────────────────
    raw_mesh_labels = extracted.get("mesh_labels_used")
    if raw_mesh_labels is not None and not isinstance(raw_mesh_labels, list):
        checks.append(_err(
            "list_type",
            "syntax",
            "mesh_labels_used must be a list of label strings",
        ))
    for i, b in enumerate(extracted.get("boundaries") or []):
        if not isinstance(b, dict):
            continue
        kind = b.get("kind")
        if kind is not None and kind not in _KINDS:
            checks.append(_err(
                "boundary_kind_enum",
                "syntax",
                f"boundaries[{i}] kind must be one of {sorted(_KINDS)}, got {kind!r}",
            ))
        role = b.get("design_role")
        if role is not None and role not in _ROLES:
            checks.append(_err(
                "boundary_role_enum",
                "syntax",
                f"boundaries[{i}] design_role must be one of {sorted(_ROLES)}, got {role!r}",
            ))
    for i, it in enumerate(extracted.get("interfaces") or []):
        if not isinstance(it, dict):
            continue
        phys = it.get("physical")
        if phys is not None and phys not in _INTERFACE_PHYSICALS:
            checks.append(_err(
                "interface_physical_enum",
                "syntax",
                (
                    f"interfaces[{i}] physical must be one of "
                    f"{sorted(_INTERFACE_PHYSICALS)}, got {phys!r}"
                ),
            ))

    # ── 3. fields ─────────────────────────────────────────────────────
    for i, f in enumerate(extracted.get("fields") or []):
        if not isinstance(f, dict):
            continue
        if f.get("type") is not None and f.get("type") not in _FIELD_TYPES:
            checks.append(_err(
                "field_type_enum",
                "syntax",
                f"fields[{i}] type must be one of {sorted(_FIELD_TYPES)}, got {f.get('type')!r}",
            ))
        if f.get("fe_space") is not None and f.get("fe_space") not in _FE_SPACES:
            checks.append(_err(
                "field_fe_enum",
                "syntax",
                (
                    f"fields[{i}] fe_space must be one of "
                    f"{sorted(_FE_SPACES)}, got {f.get('fe_space')!r}"
                ),
            ))

    # ── 4. constants ──────────────────────────────────────────────────
    for i, c in enumerate(extracted.get("constants") or []):
        if not isinstance(c, dict):
            continue
        if c.get("value") is not None and _to_float(c.get("value")) is None:
            checks.append(_err(
                "constant_value_numeric",
                "syntax",
                f"constants[{i}] value must be a number, got {c.get('value')!r}",
            ))

    # ── 10. additional run requirements (optional) ────────────────────
    xr = extracted.get("extra_requirements")
    if xr is not None:
        if not isinstance(xr, list):
            checks.append(_err("list_type", "syntax", "extra_requirements must be a list"))
        else:
            for i, it in enumerate(xr):
                if not isinstance(it, dict) or not it.get("requirement"):
                    checks.append(_err("extra_requirements_item", "syntax",
                                       f"extra_requirements[{i}] must be an object with a non-empty 'requirement'"))

    return checks


# ──────────────────────────────────────────────────────────────────────
# problem spec — designed (syntax level, deterministic, generated-only)
# ──────────────────────────────────────────────────────────────────────


def check_designed_problem(designed, extracted=None):
    """Deterministic syntax checks on the designed problem spec.

    Reads ONLY the generated JSONs (designed + optional extracted for the
    immutability checks) — never the problem input files.  The semantic
    design quality is judged by the LLM design reviewer."""
    checks = []
    if not isinstance(designed, dict):
        return [_err("json_object", "syntax", "designed spec is not a JSON object")]
    extracted = extracted or {}

    # source annotations on every designed scalar / list item / sub-object
    for k in _DESIGNED_SCALAR_KEYS:
        if k in designed and designed.get(f"{k}_source") not in _SOURCES:
            checks.append(_err(
                "source_annotation",
                "syntax",
                f"designed scalar {k!r} is missing a valid '{k}_source'",
            ))
    if "objective_terms" in designed:
        terms = designed.get("objective_terms")
        srcs = designed.get("objective_terms_source")
        if not isinstance(terms, list):
            checks.append(_err("list_type", "syntax", "designed objective_terms must be a list"))
        elif (not isinstance(srcs, list) or len(srcs) != len(terms)
              or any(s not in _SOURCES for s in srcs)):
            checks.append(_err(
                "source_annotation",
                "syntax",
                "objective_terms_source must be a per-element source list",
            ))
    # shared-source sub-groups: shape_derivative_source covers volume+boundary,
    # newton_source covers newton_description/newton_weak_form
    if designed.get("shape_derivative_volume") or designed.get("shape_derivative_boundary"):
        if designed.get("shape_derivative_source") not in _SOURCES:
            checks.append(_err(
                "source_annotation",
                "syntax",
                "shape_derivative_* present but 'shape_derivative_source' is missing",
            ))
    if designed.get("newton_description") or designed.get("newton_weak_form"):
        if designed.get("newton_source") not in _SOURCES:
            checks.append(_err(
                "source_annotation",
                "syntax",
                "newton_* present but 'newton_source' is missing",
            ))
    for k in _DESIGNED_LIST_KEYS:
        if k in designed:
            items = designed[k]
            if not isinstance(items, list):
                checks.append(_err("list_type", "syntax", f"designed {k!r} must be a list"))
                continue
            for i, it in enumerate(items):
                if not isinstance(it, dict):
                    checks.append(_err(
                        "list_item_type",
                        "syntax",
                        f"designed {k}[{i}] must be an object",
                    ))
                    continue
                if it.get("source") not in _SOURCES:
                    checks.append(_err(
                        "source_annotation",
                        "syntax",
                        f"designed {k}[{i}] is missing a valid \"source\"",
                    ))

    # ── 1. basic information ──────────────────────────────────────────
    if (not isinstance(designed.get("problem_type"), str)
            or designed.get("problem_type") not in _PROBLEM_TYPES):
        checks.append(_err(
            "problem_type_enum",
            "syntax",
            f"problem_type must be one of {sorted(_PROBLEM_TYPES)}",
        ))
    if _to_int(designed.get("dimension")) != 2:
        checks.append(_err("dimension", "syntax", "dimension must be 2"))
    if not designed.get("domain_description"):
        checks.append(_err(
            "domain_description",
            "syntax",
            "domain_description (the domain's geometric definition) is required",
        ))

    # ── 2. boundaries ─────────────────────────────────────────────────
    boundary_by_label = {}
    for i, b in enumerate(designed.get("boundaries") or []):
        if not isinstance(b, dict):
            continue
        label, kind, role = b.get("label"), b.get("kind"), b.get("design_role")
        if not label:
            checks.append(_err(
                "boundary_label_required",
                "syntax",
                f"boundaries[{i}] must have a concrete label",
            ))
        # a label-10 entry is ALLOWED here: it describes the material
        # boundary's treatment when the weak form acts on it (e.g. a spring
        # term on the hole).  It still never enters boundary_labels /
        # fixed_labels (checked separately) and carries no design role.
        if kind not in _KINDS:
            checks.append(_err(
                "boundary_kind_enum",
                "syntax",
                f"boundaries[{i}] kind must be one of {sorted(_KINDS)}, got {kind!r}",
            ))
        # the label-10 (material boundary) entry carries NO design role —
        # only non-10 boundaries must have one
        if str(label) != "10" and role not in _ROLES:
            checks.append(_err(
                "boundary_role_enum",
                "syntax",
                f"boundaries[{i}] design_role must be one of {sorted(_ROLES)}, got {role!r}",
            ))
        if label is not None:
            boundary_by_label[str(label)] = b

    # explicit extracted design_role is ground truth — must be preserved
    for i, eb in enumerate(extracted.get("boundaries") or []):
        if not isinstance(eb, dict) or eb.get("design_role") not in _ROLES:
            continue
        key = eb.get("mesh_label")
        db = boundary_by_label.get(str(key)) if key is not None else None
        if db is None:  # fall back to matching by name
            name = eb.get("name")
            db = next((b for b in (designed.get("boundaries") or [])
                       if isinstance(b, dict) and b.get("name") == name), None)
        if db is not None and db.get("design_role") != eb.get("design_role"):
            checks.append(_err("design_role_preserved", "syntax",
                               f"extracted boundary {eb.get('name')!r} design_role {eb.get('design_role')!r} "
                               f"was changed to {db.get('design_role')!r} — extracted roles are immutable"))

    # ── 4/7. constants and state ──────────────────────────────────────
    for i, c in enumerate(designed.get("constants") or []):
        if not isinstance(c, dict):
            continue
        v = c.get("value")
        if v is not None and (_to_float(v) is None or not math.isfinite(_to_float(v))):
            checks.append(_err(
                "constant_value_numeric",
                "syntax",
                f"constants[{i}] value must be a finite number, got {v!r}",
            ))
    if not isinstance(designed.get("state_is_nonlinear"), bool):
        checks.append(_err("state_is_nonlinear", "syntax", "state_is_nonlinear must be a boolean"))
    cu = designed.get("check_units")
    cu_b = _as_bool(cu)
    if cu is not None and cu_b is None:
        checks.append(_err("check_units", "syntax", "check_units must be a boolean"))
    # item 7: check_units belongs to the LAGRANGIAN object — a stray
    # placement (top level / optimization_config) is a schema error, and
    # the lagrangian object must carry it (check_units: true by default)
    if cu is not None:
        checks.append(_err(
            "check_units_placement", "syntax",
            "check_units must sit INSIDE the lagrangian object (item 7) — "
            "move it there",
        ))
    opt_cu = designed.get("optimization_config")
    if isinstance(opt_cu, dict) and "check_units" in opt_cu:
        checks.append(_err(
            "check_units_placement", "syntax",
            "check_units must sit INSIDE the lagrangian object (item 7), "
            "not under optimization_config — move it there",
        ))
    lag_cu = designed.get("Lagrangian") or designed.get("lagrangian")
    if isinstance(lag_cu, dict) and lag_cu.get("check_units") is None:
        checks.append(_err(
            "check_units_placement", "syntax",
            "the lagrangian object is missing check_units — item 7 "
            "requires check_units: true (the default)",
        ))
    chi = designed.get("characteristic_function")
    if isinstance(chi, dict) and chi.get("source") not in _SOURCES:
        checks.append(_err(
            "source_annotation",
            "syntax",
            "characteristic_function is missing a valid \"source\"",
        ))
    if isinstance(chi, dict) and not chi.get("weak_value"):
        checks.append(_err(
            "chi_weak_value",
            "syntax",
            "characteristic_function.weak_value must be given (e.g. \"1/100\" mechanics)",
        ))
    if isinstance(chi, dict) and not chi.get("strong_value"):
        checks.append(_err(
            "chi_strong_value",
            "syntax",
            "characteristic_function.strong_value must be given (the larger value, default 1)",
        ))

    # ── 3. fields ─────────────────────────────────────────────────────
    boundary_labels = {
        str(b.get("label"))
        for b in (designed.get("boundaries") or [])
        if isinstance(b, dict) and b.get("label")
    }
    domains = designed.get("domains") if isinstance(designed.get("domains"), list) else []
    domain_names = [d.get("name") for d in domains if isinstance(d, dict) and d.get("name")]
    for i, f in enumerate(designed.get("fields") or []):
        if not isinstance(f, dict):
            continue
        if f.get("type") not in _FIELD_TYPES:
            checks.append(_err(
                "field_type_enum",
                "syntax",
                f"fields[{i}] type must be one of {sorted(_FIELD_TYPES)}",
            ))
        if f.get("fe_space") not in _FE_SPACES:
            checks.append(_err(
                "field_fe_enum",
                "syntax",
                f"fields[{i}] fe_space must be one of {sorted(_FE_SPACES)}",
            ))
        fdom = f.get("domain")
        if domain_names:
            if fdom not in domain_names:
                checks.append(_err("field_domain", "syntax",
                                   f"fields[{i}] domain must be one of {domain_names} (multi-domain), got {fdom!r}"))
        elif fdom and designed.get("domain_name") and fdom != designed.get("domain_name"):
            checks.append(_err("field_domain", "syntax",
                               f"fields[{i}] domain {fdom!r} does not "
                               f"match domain_name "
                               f"{designed.get('domain_name')!r}"))
        dl = {str(l) for l in (f.get("dirichlet_labels") or [])}
        if dl and boundary_labels and not dl <= boundary_labels:
            checks.append(_err("dirichlet_labels_match", "syntax",
                               f"fields[{i}] dirichlet_labels "
                               f"{sorted(dl - boundary_labels)} are not "
                               f"designed boundary labels"))
    # every test function's fe_space MUST match the physical field it tests
    field_fes = {
        f.get("fe_space") for f in (designed.get("fields") or [])
        if isinstance(f, dict) and f.get("fe_space") in _FE_SPACES
    }
    for i, tf in enumerate(designed.get("test_functions") or []):
        if not isinstance(tf, dict):
            continue
        if tf.get("fe_space") not in _FE_SPACES:
            checks.append(_err(
                "test_fe_enum",
                "syntax",
                f"test_functions[{i}] fe_space must be one of {sorted(_FE_SPACES)}",
            ))
        elif field_fes and tf.get("fe_space") not in field_fes:
            checks.append(_err("test_fe_space_match", "syntax",
                               f"test_functions[{i}] fe_space "
                               f"{tf.get('fe_space')!r} does not match "
                               f"any field's fe_space "
                               f"{sorted(field_fes)}"))

    # ── 8. optimization_config ────────────────────────────────────────
    opt = designed.get("optimization_config")
    if not isinstance(opt, dict):
        return checks + [_err(
            "optimization_config", "syntax",
            "optimization_config is required and must be an object",
        )]
    if opt.get("source") not in _SOURCES:
        checks.append(_err(
            "source_annotation",
            "syntax",
            "optimization_config is missing a valid \"source\"",
        ))

    if not opt.get("mesh_file"):
        checks.append(_err(
            "opt_mesh_file",
            "syntax",
            "optimization_config.mesh_file must be a non-empty string (default \"mesh.txt\")",
        ))
    for k in ("hmin", "hmax", "diffusion", "v0"):
        v = opt.get(k)
        if _to_float(v) is None or not math.isfinite(_to_float(v)) or _to_float(v) <= 0:
            checks.append(_err(
                f"opt_{k}",
                "syntax",
                f"optimization_config.{k} must be a positive finite number, got {v!r}",
            ))
    hmin, hmax = _to_float(opt.get("hmin")), _to_float(opt.get("hmax"))
    if hmin is not None and hmax is not None and hmin >= hmax:
        checks.append(_err("opt_hmin_hmax", "syntax", f"hmin ({hmin}) must be < hmax ({hmax})"))
    iters = _to_int(opt.get("iterations"))
    if iters is None or iters < 1:
        checks.append(_err(
            "opt_iterations",
            "syntax",
            f"iterations must be a positive integer (default 100), got {opt.get('iterations')!r}",
        ))
    swm = _as_bool(opt.get("show_weak_material"))
    if swm is None:
        checks.append(_err(
            "opt_show_weak_material",
            "syntax",
            "show_weak_material must be a boolean (default false)",
        ))
    elif swm:
        checks.append(_err(
            "opt_show_weak_material_false",
            "syntax",
            "show_weak_material must be false",
        ))

    checks += _phi_syntax_issues(opt.get("phi", ""))

    bf_list = opt.get("boundary_labels")
    ff_list = opt.get("fixed_labels")
    lists_ok = isinstance(bf_list, list) and isinstance(ff_list, list)
    if not lists_ok:
        checks.append(_err(
            "opt_label_lists",
            "syntax",
            "boundary_labels and fixed_labels must be lists of label strings",
        ))
    bf = {str(l) for l in bf_list} if isinstance(bf_list, list) else set()
    ff = {str(l) for l in ff_list} if isinstance(ff_list, list) else set()
    if lists_ok:
        for name, labels in (("boundary_labels", bf), ("fixed_labels", ff)):
            if "10" in labels:
                checks.append(_err(
                    "opt_label_10",
                    "syntax",
                    (
                        f"{name} contains label 10 — the material boundary never appears "
                        f"in either list"
                    ),
                ))
            if "999" in labels:
                checks.append(_err(
                    "opt_label_999",
                    "syntax",
                    f"{name} contains label 999 — a code-side reference, not a mesh boundary",
                ))
        # label 0 is the no-effect placeholder (all unlabelled boundaries —
        # usually none): it is exempt from the disjoint/known checks
        if (bf & ff) - {"0"}:
            checks.append(_err(
                "opt_labels_disjoint",
                "syntax",
                f"labels in both lists: {sorted(bf & ff - {'0'})}",
            ))
        mesh_labels = {str(l) for l in (extracted.get("mesh_labels_used") or [])}
        if mesh_labels:
            unknown = (bf | ff) - mesh_labels - {"0"}
            if unknown:
                checks.append(_err(
                    "opt_labels_known",
                    "syntax",
                    f"labels not in the mesh: {sorted(unknown)}",
                ))
        # role-list consistency under the THREE-way boundary treatment:
        # fixed = frozen -> fixedLabels; design = normal-locked ->
        # boundaryLabels, OR FREE (in NEITHER list — the strongest
        # design form); a design boundary must never be frozen
        for i, b in enumerate(designed.get("boundaries") or []):
            if not isinstance(b, dict) or b.get("design_role") not in _ROLES:
                continue
            label = str(b.get("label"))
            if b.get("design_role") == "fixed" and label not in ff:
                checks.append(_err("role_label_consistency", "syntax",
                                   f"boundaries[{i}] role 'fixed' but its label {label!r} is not in fixed_labels"))
            elif b.get("design_role") == "design" and label in ff:
                checks.append(_err("role_label_consistency", "syntax",
                                   f"boundaries[{i}] role 'design' but its label {label!r} is in fixed_labels "
                                   f"(a design boundary is normal-locked via boundaryLabels or FREE — never frozen)"))

    # ── 10. additional run requirements (optional) ────────────────────
    xr = designed.get("extra_requirements")
    if xr is not None:
        if not isinstance(xr, list):
            checks.append(_err("list_type", "syntax", "extra_requirements must be a list"))
        else:
            for i, it in enumerate(xr):
                if not isinstance(it, dict) or not it.get("requirement"):
                    checks.append(_err("extra_requirements_item", "syntax",
                                       f"extra_requirements[{i}] must be an object with a non-empty 'requirement'"))
                elif it.get("source") not in _SOURCES:
                    checks.append(_err("source_annotation", "syntax",
                                       f"extra_requirements[{i}] is missing a valid \"source\""))

    # ── interfaces (multi-domain) ─────────────────────────────────────
    interfaces = designed.get("interfaces") or []
    if isinstance(domains, list) and domains and isinstance(interfaces, list):
        if not interfaces:
            checks.append(_err(
                "interfaces_required",
                "syntax",
                "multi-domain problem (domains non-empty) but interfaces is empty",
            ))
        for i, it in enumerate(interfaces):
            if not isinstance(it, dict):
                continue
            if it.get("physical") not in _INTERFACE_PHYSICALS:
                checks.append(_err(
                    "interface_physical_enum",
                    "syntax",
                    f"interfaces[{i}] physical must be one of {sorted(_INTERFACE_PHYSICALS)}",
                ))
            label = str(it.get("label") or "")
            if not label:
                checks.append(_err(
                    "interface_label",
                    "syntax",
                    f"interfaces[{i}] needs a concrete mesh label",
                ))
            elif label == "10" or label in bf or label in ff:
                checks.append(_err(
                    "interface_label_10",
                    "syntax",
                    (
                        f"interfaces[{i}] label {label!r} must be non-10 and outside "
                        f"boundary/fixed lists"
                    ),
                ))

    return checks


# ──────────────────────────────────────────────────────────────────────
# mesh spec — extracted (syntax level, deterministic, generated-only)
# ──────────────────────────────────────────────────────────────────────


def check_extracted_mesh(extracted):
    """Deterministic syntax checks on the extracted mesh spec.

    Reads ONLY the generated JSON (the fidelity to the sources is
    verified by the LLM pass)."""
    checks = []
    if not isinstance(extracted, dict):
        return [_err("json_object", "syntax", "extracted mesh spec is not a JSON object")]
    if extracted.get("source") != "extracted":
        checks.append(_err(
            "source_annotation",
            "syntax",
            "extracted mesh spec is missing \"source\": \"extracted\"",
        ))
    category = extracted.get("category")
    if category not in _MESH_CATEGORIES:
        checks.append(_err(
            "mesh_category_enum",
            "syntax",
            f"category must be one of {sorted(_MESH_CATEGORIES)}, got {category!r}",
        ))
        return checks
    if category == "absent":
        return checks
    if category == "code" and not extracted.get("mesh_code"):
        checks.append(_err(
            "mesh_code_required",
            "syntax",
            "mesh_code must be non-empty when category is 'code'",
        ))
    if category == "description" and not extracted.get("geometry_description"):
        checks.append(_err("geometry_description_required", "syntax",
                           "geometry_description must be non-empty when category is 'description'"))
    labels = extracted.get("labels_found")
    if not isinstance(labels, list) or any(not isinstance(l, (str, int)) for l in labels):
        checks.append(_err(
            "labels_found_list",
            "syntax",
            "labels_found must be a list of label strings",
        ))
    if _as_bool(extracted.get("has_syntax_errors")) is None:
        checks.append(_err(
            "has_syntax_errors_bool",
            "syntax",
            "has_syntax_errors must be a boolean",
        ))
    return checks


# ──────────────────────────────────────────────────────────────────────
# mesh spec — designed (syntax level, deterministic, generated-only)
# ──────────────────────────────────────────────────────────────────────

_BORDER_RE = re.compile(r"\bborder\s+(\w+)\s*\([^)]*\)\s*\{(.*?)\}", re.DOTALL)
_SQUARE_RE = re.compile(
    r"\bsquare\s*\(\s*[^,]+?\s*,\s*[^,]+?\s*,\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]"
)


_BORDER_FULL = re.compile(r"\bborder\s+(\w+)\s*\(([^)]*)\)\s*\{(.*?)\}", re.DOTALL)


def _mesh_box(code):
    """The domain box of a mesh code: (L, H) or (None, None).

    Parses FreeFEM ``real L/H`` parameterisations (via
    phi_rules.mesh_dims), linear ``square(n, m, [a*x+b, c*y+d])``
    transforms (the unit square scaled to a x c), and generic border
    parameterisations (ALL ``real X = v;`` constants substituted, border
    endpoints evaluated with sympy) so a redesigned border mesh built on
    the wrong constants is caught too."""
    try:
        from Initialization.phi_rules import mesh_dims

        L, H = mesh_dims(code or "")
        if L is not None and H is not None:
            return L, H
        sp = _get_sympy()
        consts = {}
        for line in (code or "").splitlines():
            mm = re.match(r"real\s+(\w+)\s*=\s*([^;]+);", line.strip())
            if mm:
                try:
                    consts[mm.group(1)] = sp.sympify(mm.group(2))
                except Exception:
                    pass
        m = _SQUARE_RE.search(code or "")
        if m:
            try:
                x, y = sp.symbols("x y")
                fx = sp.sympify(m.group(1)).subs(consts)
                fy = sp.sympify(m.group(2)).subs(consts)
                # accept only separable LINEAR transforms — a nonlinear
                # (e.g. polar) mapping has no rectangular box; implicit
                # unit coefficients (``[4 * x, y]``) and symbolic
                # constants (``real L = 2.0; [L * x, L * y]``) both work
                if (fx.diff(x, 2) == 0 and not fx.has(y)
                        and fy.diff(y, 2) == 0 and not fy.has(x)):
                    L2 = abs(float(fx.subs(x, 1)) - float(fx.subs(x, 0)))
                    H2 = abs(float(fy.subs(y, 1)) - float(fy.subs(y, 0)))
                    if L2 and H2:
                        return L2, H2
            except Exception:
                pass
        if re.search(r"\bsquare\s*\(", code or ""):
            # plain square(n, m) without a transform list — the unit square
            return 1.0, 1.0
        if not consts:
            return None, None
        subs = {sp.Symbol(k): v for k, v in consts.items()}
        xs, ys = [], []
        for bm in _BORDER_FULL.finditer(code or ""):
            args, body = bm.group(2), bm.group(3)
            xm = re.search(r"x\s*=\s*([^;]+);", body)
            ym = re.search(r"y\s*=\s*([^;]+);", body)
            tr = re.match(r"\s*t\s*=\s*([^,]+)\s*,\s*([^)]+)", args)
            if not (xm and ym and tr):
                continue
            try:
                t = sp.Symbol("t")
                xf = sp.sympify(xm.group(1)).subs(subs)
                yf = sp.sympify(ym.group(1)).subs(subs)
                for te in (tr.group(1), tr.group(2)):
                    tv = sp.sympify(te).subs(subs)
                    xs.append(complex(xf.subs(t, tv)).real)
                    ys.append(complex(yf.subs(t, tv)).real)
            except Exception:
                continue
        if xs and ys:
            return max(xs) - min(xs), max(ys) - min(ys)
    except Exception:
        pass
    return None, None


def check_designed_mesh(designed, extracted=None, problem_context=None):
    """Deterministic syntax checks on the designed mesh spec.

    Reads ONLY the generated JSONs (the designed spec, the mesh
    extraction for the verbatim rule, and the problem context for the
    cross-file label consistency) — never the problem input files."""
    checks = []
    if not isinstance(designed, dict):
        return [_err("json_object", "syntax", "designed mesh spec is not a JSON object")]
    code = designed.get("mesh_code")
    if not isinstance(code, str) or not code.strip():
        return checks + [_err(
            "mesh_code_required", "syntax",
            "mesh_code must be a non-empty string",
        )]
    if designed.get("mesh_code_source") not in _MESH_SOURCES:
        checks.append(_err(
            "mesh_code_source_enum",
            "syntax",
            f"mesh_code_source must be one of {sorted(_MESH_SOURCES)}",
        ))
    # verbatim immutability: category=code with no syntax errors means the
    # designed mesh_code is a VERBATIM copy of the extracted one — the
    # design phase must not rewrite immutable ground truth while labeling
    # it "extracted"
    if isinstance(extracted, dict):
        ex_code = extracted.get("mesh_code")
        if (ex_code and designed.get("mesh_code_source") == "extracted"
                and code.strip() != str(ex_code).strip()):
            checks.append(_err("mesh_code_verbatim", "syntax",
                               "mesh_code_source is 'extracted' but the code differs from the extracted "
                               "verbatim copy — extracted code is immutable"))
        # a redesigned mesh must keep the ACTIVE source mesh's DOMAIN
        # dimensions (never the coordinates of commented-out borders)
        ex_L, ex_H = _mesh_box(ex_code)
        ds_L, ds_H = _mesh_box(code)
        if (None not in (ex_L, ex_H, ds_L, ds_H)
                and (abs(ex_L - ds_L) > 1e-9 * max(ex_L, 1.0)
                     or abs(ex_H - ds_H) > 1e-9 * max(ex_H, 1.0))):
            checks.append(_err("mesh_domain_preserved", "syntax",
                               f"designed mesh domain [{ds_L:g}x{ds_H:g}] differs from the active "
                               f"source mesh domain [{ex_L:g}x{ex_H:g}] — a redesigned mesh must "
                               "keep the active source dimensions"))
    if _to_int(designed.get("dimension")) != 2:
        checks.append(_err("dimension", "syntax", "dimension must be 2"))
    if not designed.get("domain_description"):
        checks.append(_err("domain_description", "syntax", "domain_description is required"))
    if not designed.get("mesh_file"):
        checks.append(_err(
            "mesh_file",
            "syntax",
            "mesh_file must be a non-empty string (default \"mesh.txt\")",
        ))
    if _to_int(designed.get("n")) is None or _to_int(designed.get("n")) < 1:
        checks.append(_err(
            "mesh_n",
            "syntax",
            f"n (mesh resolution) must be a positive integer, got {designed.get('n')!r}",
        ))
    for src_key in ("n_source", "mesh_file_source", "dimension_source",
                    "labels_used_source", "domain_description_source"):
        if designed.get(src_key) not in _SOURCES:
            checks.append(_err(
                "source_annotation",
                "syntax",
                f"{src_key} is missing a valid source",
            ))
    labels_used = designed.get("labels_used")
    if not isinstance(labels_used, list) or any(not isinstance(l, (str, int)) for l in labels_used):
        checks.append(_err(
            "labels_used_list",
            "syntax",
            "labels_used must be a list of label strings",
        ))
        labels_used = []
    labels_used = [str(l) for l in labels_used]
    if "10" in labels_used:
        checks.append(_err(
            "labels_used_10",
            "syntax",
            "labels_used contains label 10 — the material boundary is not a non-10 label",
        ))

    # the code must produce a mesh named Th
    if not (re.search(r"\bmesh\s+Th\s*=", code) or
            (("buildmesh" in code or "square" in code) and re.search(r"\bTh\b", code))):
        checks.append(_err(
            "mesh_produces_Th",
            "syntax",
            "mesh_code must produce a mesh variable named Th",
        ))

    # ACTIVE borders only — commented-out borders (``// border C04...``)
    # are not part of the mesh and must not drive the checks
    active_code = re.sub(r"//[^\n]*", "", code)
    # every border in the code must carry a label (any assignment form —
    # plain `label=2` or conditional `label=(t>0.5)?2:1`; whitespace is
    # excluded from the RHS so a dangling `label = ;` still fails)
    for m in _BORDER_RE.finditer(active_code):
        body = m.group(2)
        if not re.search(r"label\s*=\s*[^;\n\s]", body):
            checks.append(_err(
                "border_label",
                "syntax",
                f"border {m.group(1)!r} has no label assignment",
            ))

    # structured edges must match the code
    edges = designed.get("edges")
    if not isinstance(edges, list) or not edges:
        checks.append(_err("edges_required", "syntax", "edges must list every border segment"))
        edges = []
    else:
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                checks.append(_err("edges_item", "syntax", f"edges[{i}] must be an object"))
                continue
            for k in ("name", "param", "x", "y", "label", "role"):
                if not e.get(k):
                    checks.append(_err("edges_field", "syntax", f"edges[{i}] is missing {k!r}"))
            if e.get("source") not in _SOURCES:
                checks.append(_err(
                    "source_annotation",
                    "syntax",
                    f"edges[{i}] is missing a valid \"source\"",
                ))
    edge_by_name = {e.get("name"): e for e in edges if isinstance(e, dict) and e.get("name")}
    for m in _BORDER_RE.finditer(active_code):
        name, body = m.group(1), m.group(2)
        lab = re.search(r"label\s*=\s*(\d+)", body)
        e = edge_by_name.get(name)
        if e is None:
            checks.append(_err(
                "edges_cover_borders",
                "syntax",
                f"border {name!r} in mesh_code is missing from edges",
            ))
        elif lab and str(e.get("label")) != lab.group(1):
            checks.append(_err("edges_label_match", "syntax",
                               f"edges entry {name!r} label "
                               f"{e.get('label')!r} does not match the "
                               f"code label {lab.group(1)!r}"))
    edge_labels = {str(e.get("label")) for e in edges if isinstance(e, dict) and e.get("label")}
    missing = set(labels_used) - edge_labels
    if missing:
        checks.append(_err(
            "labels_used_in_edges",
            "syntax",
            f"labels_used entries not in any edge: {sorted(missing)}",
        ))
    unlisted = edge_labels - set(labels_used) - {"10"}
    if unlisted:
        checks.append(_err("edge_labels_in_labels_used", "syntax",
                           f"edge labels not in labels_used (and not 10): {sorted(unlisted)}"))

    # cross-file: the problem's boundary/fixed labels must exist here
    if isinstance(problem_context, dict):
        opt = problem_context.get("optimization_config")
        if isinstance(opt, dict):
            wanted = {
                str(l)
                for l in ((opt.get("boundary_labels") or [])
                          + (opt.get("fixed_labels") or []))
            }
            if wanted - set(labels_used):
                checks.append(_err("problem_labels_in_mesh", "syntax",
                                   f"problem boundary/fixed labels not "
                                   f"in mesh labels_used: "
                                   f"{sorted(wanted - set(labels_used))}"))
        for i, b in enumerate(problem_context.get("boundaries") or []):
            # label 10 (material boundary) is deliberately NOT in
            # labels_used — skip it in the cross-file check
            if (isinstance(b, dict) and b.get("label") and str(b["label"]) != "10"
                    and str(b["label"]) not in labels_used):
                checks.append(_err("problem_labels_in_mesh", "syntax",
                                   f"problem boundaries[{i}] label {b.get('label')!r} not in mesh labels_used"))
    return checks


# ──────────────────────────────────────────────────────────────────────
# the generate -> review -> repair loop lives in
# Initialization.spec_generation.run_spec_generation
