"""Symmetry detection for meshes, initial level sets, and designs.

Shared by the pre-run parameter review (``check_and_fix_params``) and the
reflection loop's quality gate: a symmetric problem must be seeded with a
symmetric initial level set and must produce a symmetric design.
"""

import re
import warnings
from collections import Counter

_sympy = None


def _get_sympy():
    """Lazy sympy handle — the parsing helpers must stay usable without
    the dependency (pure string analysis is not symbolic)."""
    global _sympy
    if _sympy is None:
        import sympy

        _sympy = sympy
    return _sympy


def _parse_mesh(mesh_txt: str):
    """Parse L, H, extra real constants, and border segments from mesh.txt.

    Returns ``(L, H, segments)`` where *segments* is a list of
    ``(label, ((x0, y0), (x1, y1)))`` with the endpoints unordered.
    ``L``/``H`` are floats or None when missing.
    """
    sp = _get_sympy()
    L = H = None
    consts = {}
    segments = []
    for line in (mesh_txt or "").splitlines():
        s = line.strip()
        m = re.match(r"real\s+(\w+)\s*=\s*([0-9.eE+\-*/() ]+);", s)
        if m:
            name, val = m.group(1), m.group(2)
            if name == "L":
                try:
                    L = float(sp.sympify(val))
                except Exception:
                    L = None
            elif name == "H":
                try:
                    H = float(sp.sympify(val))
                except Exception:
                    H = None
            else:
                consts[name] = val
            continue
        m = re.match(
            r"border\s+\w+\s*\(\s*t\s*=\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*\{"
            r"x\s*=\s*([^;]+);\s*y\s*=\s*([^;]+);\s*label\s*=\s*(\d+)\s*;\s*\}",
            s,
        )
        if m and L is not None and H is not None:
            t0, t1, xe, ye, lab = m.groups()
            try:
                subs = {k: sp.sympify(v) for k, v in consts.items()}
                subs.update({"L": L, "H": H})
                xf = sp.sympify(xe).subs(subs)
                yf = sp.sympify(ye).subs(subs)
                t = sp.Symbol("t")
                # the t-interval endpoints are themselves expressions in
                # L, lq, … — substitute the constants there too
                tv0 = sp.sympify(t0).subs(subs)
                tv1 = sp.sympify(t1).subs(subs)
                # round to a shared tolerance so that mirror endpoints
                # computed as (L - 0.30000000000000004) == 0.7 match the
                # directly parsed 0.7 in the Counter comparison
                p0 = (round(float(xf.subs(t, tv0)), 9), round(float(yf.subs(t, tv0)), 9))
                p1 = (round(float(xf.subs(t, tv1)), 9), round(float(yf.subs(t, tv1)), 9))
                if p0 > p1:
                    p0, p1 = p1, p0
                segments.append((int(lab), (p0, p1)))
            except Exception:
                continue
    return L, H, segments


def mesh_dims(mesh_txt: str):
    """Return ``(L, H)`` parsed from *mesh_txt* (either may be None)."""
    L, H, _ = _parse_mesh(mesh_txt)
    return L, H


def detect_symmetry(mesh_txt: str) -> list:
    """Return the mirror symmetry axes of the mesh: [], ['x'], ['y'], or both.

    ``'x'`` means symmetric under ``x -> L - x`` (vertical mirror axis at
    ``x = L/2``); ``'y'`` under ``y -> H - y``.  A problem is symmetric
    when every border segment (geometry AND label) has its mirror image
    among the other segments — the boundary-condition labels must map to
    themselves under the reflection.
    """
    L, H, segments = _parse_mesh(mesh_txt)
    if L is None or H is None or not segments:
        if re.search(r"border\s+\w+\s*\(", mesh_txt or ""):
            # Bordered mesh whose L/H could not be parsed (e.g. an
            # x0/x1/y0/y1 parameterisation) — symmetry detection is
            # silently skipped otherwise; say so once.
            warnings.warn(
                "symmetry: mesh borders found but L/H could not be parsed "
                "(not the 'real L/H' parameterisation?) — symmetry "
                "detection skipped."
            )
        return []
    axes = []
    for axis in ("x", "y"):
        reflected = []
        for lab, (p0, p1) in segments:
            if axis == "x":
                # round to the same tolerance as _parse_mesh so that
                # L - 0.3 == 0.7 exactly in the Counter comparison
                q0 = (round(L - p0[0], 9), round(p0[1], 9))
                q1 = (round(L - p1[0], 9), round(p1[1], 9))
            else:
                q0 = (round(p0[0], 9), round(H - p0[1], 9))
                q1 = (round(p1[0], 9), round(H - p1[1], 9))
            if q0 > q1:
                q0, q1 = q1, q0
            reflected.append((lab, (q0, q1)))
        if Counter(segments) == Counter(reflected):
            axes.append(axis)
    return axes


def is_phi_symmetric(phi: str, L, H, axes) -> bool:
    """True when *phi* (a FreeFem expression string) is symmetric on all axes.

    Non-sympifiable or unprovable expressions count as asymmetric — the
    caller then falls back to a canonical symmetric pattern (the
    conservative direction: a symmetric seed is always acceptable).
    """
    if not phi or not axes or L is None or H is None:
        return False
    sp = _get_sympy()
    try:
        expr = sp.sympify(phi)
    except Exception:
        return False
    x, y = sp.symbols("x y")
    for axis in axes:
        sub = {x: L - x, y: y} if axis == "x" else {x: x, y: H - y}
        try:
            diff = sp.simplify(sp.expand_trig(expr.subs(sub) - expr))
        except Exception:
            return False
        if diff != 0:
            return False
    return True


def canonical_symmetric_phi(L, H, axes) -> str:
    """A canonical symmetric initial level set for the box [0, L] x [0, H].

    ``cos(2*pi*x/L)`` is even about ``x = L/2`` and ``cos(2*pi*y/H)`` even
    about ``y = H/2``, so this pattern is symmetric on every axis.
    """
    return f"-0.4 - cos(2*pi*x/{L})*cos(2*pi*y/{H})"
