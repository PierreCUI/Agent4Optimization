"""Shared phi (initial level-set) guidance for the generation prompts.

Used by the coder (initial code generation) and by check_params (the
pre-run parameter review) so the two cannot drift apart.  The rule is
paper-first: a prescribed initial design is used verbatim; otherwise the
seed is picked from the standard template library below (physics-derived
solid shapes, sparse holes, the trigonometric checkerboard, and their
combinations) and adapted to the problem's physics, with the reference
examples contributing only the construction style.
"""

PHI_RULES = """\
## PHI RULES — choosing the initial level set (phi)

1. **THE PAPER WINS.**  If the problem statement (paper / tex /
   ProblemSpec) prescribes an initial level set or initial design
   (``level_set_initial``, "the initial design is ..."), use it EXACTLY —
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
         solid fraction moderate (roughly 20-70 %) — neither a solid
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
  MORE solid; tune A so the initial solid fraction is roughly 20-70 %).
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
   (roughly 20-70 %).
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
