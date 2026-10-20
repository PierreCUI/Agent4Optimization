# -*- coding: utf-8 -*-
"""Design-parameter normalizer — the LAST Generation step, run AFTER the
static check/repair loop and BEFORE execution (Stage 5.5).

USER RULE (2026-10-15): for the tunable parameters, when the SPEC marks
a value as DESIGNED, overwrite the generated code with the canonical
value; EXTRACTED values are never touched:

- material penalty (beta/ell/Afac — the material-cost weight) -> 1
- hmin      -> d / 200      (d = the shorter side of the rectangle
- hmax      -> d / 20        containing the mesh)
- v0        -> 1e-6
- diffusion -> 0.1
- phi       -> a checkerboard, scaled from the two suite baselines
               (diffusion / elasticity — written below as the BASELINES)

Identification of the material penalty: the spec's designed constants
entry whose description speaks of material cost (material cost / volume
cost / Lagrange multiplier — the J material term's weight).
"""
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── checkerboard baselines (the two suite references) ─────────────────
# Each baseline = (base domain LxH, phi formula on that domain).  A
# target domain scales the pattern: x -> x * (L_base / L), y -> y *
# (H_base / H), keeping the same number of cells per domain.
BASELINE_DIFFUSION = {
    "domain": (1.0, 1.0),
    "phi": "-0.1 - sin(pi * 8 * (x + 0.6)) * cos(pi * 8 * y)",
}
BASELINE_ELASTICITY = {
    "domain": (2.0, 1.0),
    "phi": "-0.4 - sin(pi * 3 * (x + 0.5)) * cos(pi * 6 * y)",
}

_MATERIAL_COST_RE = re.compile(
    r"material.{0,30}cost|cost.{0,30}material|volume cost|Lagrange multiplier",
    re.I,
)


def _mesh_box(mesh_txt: str):
    """The rectangle containing the mesh: (L, H), or None."""
    try:
        from Initialization.spec_checks import _mesh_box as _mb
        return _mb(mesh_txt)
    except Exception:
        return None


def _is_elasticity(problem_spec: dict) -> bool:
    """VectorField-driven (elasticity-style) vs scalar (diffusion-style)."""
    ds = (problem_spec or {}).get("designed") or {}
    for f in ds.get("fields") or []:
        if isinstance(f, dict) and str(f.get("type") or "").lower() == "vector":
            return True
    return False


def _checkerboard(problem_spec: dict, mesh_txt: str) -> str:
    base = BASELINE_ELASTICITY if _is_elasticity(problem_spec) \
        else BASELINE_DIFFUSION
    box = _mesh_box(mesh_txt)
    if not box or any(v is None for v in box):
        # cannot scale — fall back to the baseline verbatim
        return base["phi"]
    L, H = box
    Lb, Hb = base["domain"]
    # the suite meshes live in [0,L]x[0,H] — stretch the baseline pattern
    scaled = re.sub(r"\bx\b", f"(x * {Lb}/{L:.6g})", base["phi"])
    scaled = re.sub(r"\by\b", f"(y * {Hb}/{H:.6g})", scaled)
    return scaled


def _material_penalty_name(problem_spec: dict):
    """The designed constant that is the material-cost weight."""
    ds = (problem_spec or {}).get("designed") or {}
    for c in ds.get("constants") or []:
        if not isinstance(c, dict):
            continue
        desc = str(c.get("description") or "")
        if _MATERIAL_COST_RE.search(desc):
            return str(c.get("name") or "")
    return ""


def _sources(problem_spec: dict) -> dict:
    ds = (problem_spec or {}).get("designed") or {}
    opt = ds.get("optimization_config") or {}
    out = {}
    for k in ("hmin", "hmax", "diffusion", "v0", "phi"):
        out[k] = opt.get(f"{k}_source")
    out["material_penalty"] = None
    for c in ds.get("constants") or []:
        if isinstance(c, dict) and _MATERIAL_COST_RE.search(
                str(c.get("description") or "")):
            out["material_penalty"] = c.get("value_source")
            break
    return out


def normalize_code(code: str, problem_spec: dict, mesh_txt: str = "") -> tuple:
    """Overwrite the DESIGNED tunables in *code* with the canonical
    values.  Returns (new_code, changes) where changes is a list of
    human-readable descriptions."""
    if not problem_spec:
        return code, []
    src = _sources(problem_spec)
    changes = []

    def _designed(k):
        return src.get(k) == "designed"

    def _sub_once(text: str, pattern, repl, count=1):
        """Replace once; return (new_text, changed: bool)."""
        new, n = re.subn(pattern, repl, text, count=count)
        return new, (n > 0 and new != text)

    # material penalty -> 1
    if _designed("material_penalty"):
        name = _material_penalty_name(problem_spec)
        if name:
            pat = re.compile(
                r"(" + re.escape(name) + r"\s*=\s*Constant\s*\([^,]*,\s*)"
                r"[^,]*(\s*,)")
            code, changed = _sub_once(code, pat, r"\g<1>1\g<2>")
            if changed:
                changes.append(f"material penalty {name} (designed) -> 1")

    # hmin/hmax/diffusion/v0/phi in setup_optimization
    box = _mesh_box(mesh_txt)
    d = None
    if box and all(v is not None for v in box):
        d = min(box)  # the shorter side of the mesh box
    if _designed("hmin") and d:
        code, changed = _sub_once(code, r"\bhmin\s*=\s*[-+0-9.eE]+",
                                  f"hmin={d / 200.0:.6g}")
        if changed:
            changes.append(f"hmin (designed) -> d/200 = {d / 200.0:.6g}")
    if _designed("hmax") and d:
        code, changed = _sub_once(code, r"\bhmax\s*=\s*[-+0-9.eE]+",
                                  f"hmax={d / 20.0:.6g}")
        if changed:
            changes.append(f"hmax (designed) -> d/20 = {d / 20.0:.6g}")
    if _designed("v0"):
        code, changed = _sub_once(code, r"\bv0\s*=\s*[-+0-9.eE]+", "v0=1e-6")
        if changed:
            changes.append("v0 (designed) -> 1e-6")
    if _designed("diffusion"):
        code, changed = _sub_once(code, r"\bdiffusion\s*=\s*[-+0-9.eE]+",
                                  "diffusion=0.1")
        if changed:
            changes.append("diffusion (designed) -> 0.1")
    if _designed("phi"):
        new_phi = _checkerboard(problem_spec, mesh_txt)
        # the generated code may use single OR double quotes
        code, changed = _sub_once(code, r"\bphi\s*=\s*['\"][^'\"]*['\"]",
                                  f"phi='{new_phi}'")
        if changed:
            changes.append(f"phi (designed) -> checkerboard: {new_phi}")

    return code, changes
