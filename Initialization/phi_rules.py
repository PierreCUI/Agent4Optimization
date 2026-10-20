"""Phi (initial level-set) rules and symmetry check rules — the single
source of truth, owned by Initialization.

Initialization designs ``optimization_config.phi`` under PHI_GUIDANCE
(the paper-first rule and the template library); the generate prompt
injects it (the review prompt re-checks its rules inline), so phi is
decided here.  Downstream (Generation's check_static, Reflection's
quality gate) CONSUMES the designed phi and runs the symmetry checks —
it never designs a seed from scratch (at most Reflection replaces phi
with the canonical symmetric fallback as a fix).

Symmetry checking has TWO parts:

1. The INPUT problem — ``detect_symmetry`` analyses the mesh: when the
   problem is NOT symmetric, the check ENDS here (no symmetry
   requirement exists).
2. The RESULT — when the problem IS symmetric, the level set must be
   symmetric about the same axes (``is_phi_symmetric``);
   ``symmetry_issue`` turns a violation into a check error and
   ``canonical_symmetric_phi`` supplies a symmetric fallback.
"""

import re
import warnings
from collections import Counter

_sympy = None
_warned_skip = False  # the L/H-parse skip warning fires once per process


def _get_sympy():
    """Lazy sympy handle — the parsing helpers must stay usable without
    the dependency (pure string analysis is not symbolic)."""
    global _sympy
    if _sympy is None:
        import sympy

        _sympy = sympy
    return _sympy


PHI_RULES = """\
## PHI RULES — choosing the initial level set (phi)

1. **THE PAPER WINS.**  If the problem statement (paper / tex /
   ProblemSpec) prescribes an initial level set or initial design
   (``phi``, "the initial design is ..."), use it EXACTLY —
   scaled to the real domain size when the paper normalises the geometry.
   Never replace a prescribed phi with one of your own.

2. **No prescription -> pick a seed from the PHI TEMPLATE LIBRARY below**
   (the standard initial-design catalogue) and adapt it to this problem
   following the usage guide at the end of the library:

   (a) *Physical validity — the seed must be physically valid:*
       - CONNECTIVITY: the solid must form a connected path from the
         loaded boundary (heat source / load / support) to the sink
         (cooling edge / Dirichlet boundary).  Fragmented seeds (beads,
         islands) cannot conduct and stall the optimiser.
       - CONTACT ON FIXED BOUNDARIES: boundaries listed in
         ``fixedLabels`` do NOT move during the level-set evolution —
         whatever the initial phi puts on them is FROZEN.  Solid material
         must therefore already cover the parts of the load segment and
         of the sink (e.g. a Dirichlet cooled wall) through which heat
         must enter and leave; a seed with void on the sink can never
         conduct to it.
       - SCALE: features must be several times ``hmax``; keep the initial
         solid fraction moderate (roughly 20-80 %) — neither a solid
         block nor a sliver.

   (b) *Choose the template closest to the expected optimum's shape:*
       a solid trunk/branch member (library section 1), a shape with a
       few holes (section 2), the checkerboard (section 3), or a
       combination (section 4).  Do NOT invent constructions outside the
       library unless the geometry forces it.

   (c) *Examples — take the STYLE, not the geometry:* the selected
       examples additionally confirm which library family matches their
       construction style (min/max cutting, signed-distance base,
       oscillatory perturbation) — prefer the library section built in
       that style; never copy an example's phi verbatim.

   (d) *Symmetry:* a problem symmetric about an axis needs a phi
       symmetric about that axis — use cos(2*pi*x/L)-type terms about the
       mirror axis (sin(pi*x)-type terms are anti-symmetric and break it).

   (e) *Robustness:* prefer smooth, low-frequency shapes over
       high-frequency oscillations — fine stripes under-resolve, create
       too-thin features and crash the remesher.
"""

PHI_TEMPLATES = """\
## PHI TEMPLATE LIBRARY — standard initial-design catalogue

All formulas are written for the NORMALISED unit square [0,1] x [0,1]
with phi < 0 = solid, and oriented with the heat source (load) on a
segment of the BOTTOM edge (y = 0) and the sink on the TOP edge (y = 1).
On a real domain [x0,x1] x [y0,y1], substitute x -> (x-x0)/(x1-x0),
y -> (y-y0)/(y1-y0) throughout.  First mirror / rotate / translate the
template so its source-covering part sits on THIS problem's loaded
segment and its sink-covering part on THIS problem's sink segment.

### 1. SOLID shapes without holes (physical experience — start from
### the expected optimum's shape)

- VERTICAL BAND (straight trunk, source to sink):
  phi0 = |x - 0.5| - w/2,  w = 0.2 .. 0.5.
- T-SHAPE (top plate along the sink + stem over the source):
  phi0 = min((1 - th) - y, |x - 0.5| - wm/2),
  th = top-plate thickness 0.1 .. 0.4, wm = stem width 0.3 .. 0.5.
  The top plate must cover the WHOLE sink segment, the stem the WHOLE
  source segment.
- INVERTED T (source-side plate + stem to the sink):
  phi0 = min(y - bh, |x - 0.5| - wm/2),
  bh = source-side plate thickness 0.1 .. 0.4, wm = stem width
  0.3 .. 0.5.
- I-BEAM (top flange + bottom flange + full-height web, no hole):
  phi0 = min(|y - (1 - tf/2)| - tf/2,
             max(|y - bf/2| - bf/2, |x - 0.5| - bl/2),
             |x - 0.5| - ww/2),
  tf = top-flange thickness 0.1 .. 0.2 (full width, covers the sink),
  bf = bottom-flange thickness 0.1 .. 0.2, bl = bottom-flange length
  0.4 .. 0.6 (covers the source), ww = web width 0.1 .. 0.3 (full
  height).  Example: tf=0.2, bf=0.2, bl=0.5, ww=0.2 gives
  phi0 = min(|y - 0.9| - 0.1, max(|y - 0.1| - 0.1, |x - 0.5| - 0.25),
             |x - 0.5| - 0.1).
- CORNER / L-SHAPE (source and sink on ADJACENT edges meeting at a
  corner, e.g. source near the bottom-left corner, sink on the left
  edge):  phi0 = min(x - t, y - t),  t = arm thickness 0.1 .. 0.3.
  Mirror the arms (min((1-x) - t, y - t), min(x - t, (1-y) - t), ...)
  to place the L at the corner that joins the source and sink edges;
  each arm must fully cover its boundary segment.
- FUNNEL (wide source converging to a narrow sink):
  phi0 = |x - 0.5| - (w1/2 + (w2 - w1)/2 * y),
  w1 = width at the source (y = 0), w2 = width at the sink (y = 1),
  w1 >= w2.
- DISC / ELLIPSE (sink and source opposite on a short straight path):
  phi0 = sqrt(((x - xc)/a)^2 + ((y - yc)/b)^2) - 1,
  centred on the source-sink line so the disc touches both the source
  and the sink segments.

### 2. SPARSE HOLES (solid block or shape with a few holes)

- FULL SQUARE + CENTRAL CIRCULAR HOLE:
  phi0 = r - sqrt((x - xc)^2 + (y - yc)^2),  r = 0.1 .. 0.4.
- FULL SQUARE + RECTANGULAR HOLE:
  phi0 = min(a - |x - xc|, b - |y - yc|),  2a, 2b = 0.2 .. 0.6.
- FULL SQUARE + ELLIPTICAL HOLE:
  phi0 = 1 - sqrt(((x - xc)/a)^2 + ((y - yc)/b)^2).
- SEVERAL HOLES: phi0 = max(phi_base, h1, h2, ...), where each hole
  indicator hi is POSITIVE inside its hole (the forms above, centred at
  (xci, yci)); the union of the holes is removed from phi_base.
- HOLE RING ON THE MID-RADIUS (annulus / pipe problems): N holes at
  equal angular steps on the circle of radius d = (R + r)/2 between the
  inner radius r and the outer radius R:
  phi0 = max over k = 0..N-1 of (rc - sqrt((x - d*cos(2*pi*k/N))^2
                                      + (y - d*sin(2*pi*k/N))^2)).
  The hole radius follows the TANGENT-ANGLE rule: the angle between the
  two tangents from the origin to a hole is alpha = 2*asin(rc/d), so
  rc = d*sin(alpha/2), and the angular gap between neighbouring holes
  is 360/N - alpha degrees; keep that gap at least several times hmax
  (smaller rc if needed).  Example — R = 0.5, r = 0.15, d = 0.325,
  N = 12, alpha = 25 deg: rc = 0.325*sin(12.5 deg) = 0.07034, gap
  5 deg = 2*0.325*sin(2.5 deg) = 0.028 (about 5.7*hmin for hmin 0.005).
  Nested ternary max trees: FreeFEM's max takes AT MOST 3 arguments —
  max(max(max(h0,h1,h2), max(h3,h4,h5)), max(max(h6,h7,h8),
  max(h9,h10,h11))) for N = 12.

### 3. CHECKERBOARD (periodic trigonometric seeding)

- phi0 = A - cos(2*pi*m*x)*cos(2*pi*n*y),  m, n = 2 .. 8
  (m x n cells; A in [-1, 1] sets the solid fraction — SMALLER A gives
  MORE solid; tune A so the initial solid fraction is roughly 20-80 %).
- Staggered variant: phi0 = A - sin(2*pi*m*x)*sin(2*pi*n*y).
- Cell size (1/m by 1/n) must be SEVERAL times hmax; more cells than
  the mesh resolves create too-thin features and crash the remesher.
- SYMMETRY: about x = 0.5 only cos terms of the form
  cos(2*pi*m*(x - 0.5)) are symmetric (sin(2*pi*m*x) is
  anti-symmetric); the same holds for y about y = 0.5.  For a symmetric
  problem use the cos form, centred on the mirror axis.
- CAUTION: the checkerboard has NO continuous solid path — it relies on
  the level-set diffusion to connect cells.  Do NOT use it when a fixed
  (frozen) load or sink segment must be covered by solid at iteration 0:
  a checkerboard may leave the frozen segment in void and then block all
  conduction permanently.

### 4. COMBINATIONS

- SHAPE + HOLE: phi0 = max(phi_shape, phi_hole), with the hole
  indicator POSITIVE inside the hole — e.g. the T-shape with a central
  circular hole:
  phi0 = max(min((1 - th) - y, |x - 0.5| - wm/2),
             r - sqrt((x - 0.5)^2 + (y - 0.5)^2)).
  The hole must stay INSIDE the shape (otherwise it opens a gap in the
  trunk) and must NOT interrupt the solid contact on fixed boundaries.
- SHAPE + CHECKERBOARD FINE STRUCTURE:
  phi0 = min(phi_shape, A - cos(2*pi*m*x)*cos(2*pi*n*y))
  seeds branches / fins inside the shape; keep the cell size at least
  several times hmax.

### Usage guide

0. **FreeFEM SYNTAX — min/max take AT MOST 3 ARGUMENTS**: the library's
   ``min(a, b, c)`` forms are already ternary, but any EXTENSION (four
   holes, twelve discs, a shape with four components) must be nested
   into ternary trees — ``max(max(a, b, c), max(d, e, f))``, never a
   flat ``max(a, b, c, d, ...)`` (FreeFEM compile error: "error
   operator (").  Count the parentheses after nesting.
1. ALIGN FIRST: mirror / rotate / translate the template so its
   source-covering part sits on THIS problem's loaded segment and its
   sink-covering part on THIS problem's sink segment; rescale the
   coordinates to the REAL domain.
2. COVER THE FIXED BOUNDARIES: whatever of the template touches the
   load and sink segments is FROZEN — the solid must fully cover both
   segments from iteration 0.
3. SIZE AGAINST hmax: every feature (plate, stem, web, hole wall) must
   be several times hmax; keep the initial solid fraction moderate
   (roughly 20-80 %).
4. SYMMETRY: for a mirror-symmetric problem pick the template variant
   that is symmetric about the mirror axis (centred stems/flanges,
   cos-based checkerboards) — do not offset it.
5. COMBINE when the expected optimum suggests it (section 4); do not
   invent brand-new exotic constructions — the library covers the
   standard seeds.
"""

# Rules followed by the template catalogue — the two blocks are always
# injected together so a prompt never sees one without the other.
PHI_GUIDANCE = PHI_RULES + PHI_TEMPLATES


# ======================================================================
# symmetry detection and checking (two-part)
# ======================================================================


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
        # single AND multi declarations: `real L = 2.0;` and
        # `real L = 2.0, H = 1.0;` (split on commas first)
        if s.startswith("real "):
            body = s[len("real "):].rstrip(";")
            matched = False
            for part in body.split(","):
                m = re.match(r"(\w+)\s*=\s*([0-9.eE+\-*/() ]+?)\s*$", part.strip())
                if not m:
                    continue
                matched = True
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
            if matched:
                continue
        m = re.match(
            r"border\s+\w+\s*\(\s*t\s*=\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*\{"
            r"x\s*=\s*([^;]+);\s*y\s*=\s*([^;]+);\s*label\s*=\s*"
            r"(?:\([^)]*\)\s*\?\s*)?(\d+)\s*(?::\s*\d+\s*)?;\s*\}",
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
            # silently skipped otherwise; say so ONCE per process (the
            # check loop calls this every round).
            global _warned_skip
            if not _warned_skip:
                warnings.warn(
                    "symmetry: mesh borders found but L/H could not be "
                    "parsed (not the 'real L/H' parameterisation?) — "
                    "symmetry detection skipped."
                )
                _warned_skip = True
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


def canonical_symmetric_phi(L, H) -> str:
    """A canonical symmetric initial level set for the box [0, L] x [0, H].

    ``cos(2*pi*x/L)`` is even about ``x = L/2`` and ``cos(2*pi*y/H)`` even
    about ``y = H/2``, so this pattern is symmetric on every axis.
    """
    return f"-0.4 - cos(2*pi*x/{L})*cos(2*pi*y/{H})"


def symmetry_issue(phi: str, mesh_txt: str) -> list:
    """The two-part symmetry check, as a list of check errors.

    Part 1 — the INPUT problem: the mesh is analysed; when the problem
    is NOT symmetric (``detect_symmetry`` returns no axes) the check
    ENDS here and returns nothing — no symmetry requirement exists.
    Part 2 — the RESULT: when the problem IS symmetric, *phi* must be
    symmetric about the same axes; a violation is one error.
    """
    axes = detect_symmetry(mesh_txt)
    if not axes:
        return []
    L, H = mesh_dims(mesh_txt)
    if L is None or H is None or is_phi_symmetric(phi, L, H, axes):
        return []
    return [
        {
            "check": "symmetry",
            "severity": "error",
            "message": (
                f"the mesh is symmetric about {axes} but phi is NOT "
                f"symmetric about these axes — use cos-type terms about "
                f"the mirror axis (cos(2*pi*x/L) about x = L/2, "
                f"cos(2*pi*y/H) about y = H/2)."
            ),
        }
    ]


# ======================================================================
# mesh box — the domain rectangle for phi sampling
# ======================================================================

_SQUARE_RE = re.compile(
    r"\bsquare\s*\(\s*[^,]+?\s*,\s*[^,]+?\s*,\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]"
)


def mesh_box(mesh_txt: str):
    """The domain box ``(x0, x1, y0, y1)`` of *mesh_txt*, or None.

    Parses FreeFEM ``real L/H`` parameterisations (box [0,L] x [0,H]),
    linear ``square(n, m, [a*x+b, c*y+d])`` transforms (the unit square
    mapped onto [fx(0), fx(1)] x [fy(0), fy(1)]), and generic border
    parameterisations (the segment endpoint envelope).  None when the
    domain cannot be determined.
    """
    L, H, segments = _parse_mesh(mesh_txt or "")
    if L is not None and H is not None:
        return 0.0, L, 0.0, H
    m = _SQUARE_RE.search(mesh_txt or "")
    if m:
        sp = _get_sympy()
        try:
            x, y = sp.symbols("x y")
            fx = sp.sympify(m.group(1))
            fy = sp.sympify(m.group(2))
            # accept only separable LINEAR transforms — a nonlinear
            # (e.g. polar) mapping has no rectangular box
            if (fx.diff(x, 2) == 0 and not fx.has(y)
                    and fy.diff(y, 2) == 0 and not fy.has(x)):
                xs = [float(fx.subs(x, v)) for v in (0, 1)]
                ys = [float(fy.subs(y, v)) for v in (0, 1)]
                return min(xs), max(xs), min(ys), max(ys)
        except Exception:
            pass
    if segments:
        xs = [p[0][0] for _, p in segments] + [p[1][0] for _, p in segments]
        ys = [p[0][1] for _, p in segments] + [p[1][1] for _, p in segments]
        return min(xs), max(xs), min(ys), max(ys)
    return None
