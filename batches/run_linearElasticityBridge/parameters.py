"""
parameters.py — Input parameter lists for batchrun.py and ablation.py.
(Adapted for run_linearElasticityBridge — bridge deck [0,4]x[0,1] built
from square(4n,n), Dirichlet on supports 2 and 4, surface load on the
deck edge, compliance objective with volume term Afac*X.)

Every input feature is a list.  batchrun.py builds the full Cartesian
product; ablation.py varies one parameter at a time from the defaults
(parsed from the original results/ files).

For the batch phase, hmax / hmin_ratio / mesh_n take 5 values each
(two smaller, original, two larger — all ablation-tested except
hmin_ratio=0.02, the only untested value below the tested minimum);
p_degree keeps all 3 usable degrees.  Every other family holds exactly
ONE value, the tested original.  For future experiments, add values to
a family's list to sweep it again — no code changes needed.

NOTE: this template has NO inner hole — mesh.txt is ``square(4*n, n,
[4*x, y])`` with no ``real a`` line, so there is no hole_a family for
this problem.  The physical constants are f / mu / lam / Afac (not
nu / E / alpha), and the material contrast is '1/1000'.
"""

# ==========================================================================
# Sweepable parameters (modified in main.py via setup_optimization)
# ==========================================================================

# MMG maximum mesh edge length  (primary control: triangle count ≈ 2×area/hmax²)
# Domain is [0,4]×[0,1] (area=4.0) — bridge deck.
# Original value: 0.05
# 5 values: two smaller / original / two larger (all ablation-tested)
HMAX = [
    0.02,
    0.03,
    0.05,
    0.10,
    0.20,
]

# MMG minimum mesh edge length — fraction of hmax  (hmin = ratio × hmax)
# Original value: 0.1  (0.005 / 0.05)
# 5 values: two smaller / original / two larger.
# NOTE: upper end capped at 0.60 — ratio 0.70 crashes FreeFEM
# ("generated 2d mesh is too fine", fem.cpp:1919).  0.02 is the only
# value below the tested minimum (0.05) and is untested.
HMIN_RATIO = [
    0.02,
    0.05,
    0.10,
    0.30,
    0.60,
]

# Shape advection speed (step size for level-set evolution)
# Original value: 1e-4 — fixed for this batch (single value)
V0 = [
    1e-4,
]

# Level-set regularisation (smooths the velocity field)
# Original value: 0.01 — fixed for this batch (single value)
DIFFUSION = [
    0.01,
]

# ==========================================================================
# Modifiable problem parameters
# ==========================================================================

# Initial mesh border resolution (int n from mesh.txt: square(4n,n))
# Original value: 30
# 5 values: two smaller / original / two larger (all ablation-tested)
MESH_N = [
    10,
    20,
    30,
    50,
    100,
]

# Characteristic function contrast ratio  (strong / weak)
# Original value: 1 / (1/1000) = 1000  (weak material '1/1000')
# Fixed for this batch (single value).
# NOTE: the template has a SINGLE characteristic function X — the
# chi_contrast rewrite only targets its quoted weak value.
CHI_CONTRAST = [
    1000,
]

# ==========================================================================
# Finite element degree of the state field  (field and its test function
# always share the same degree — modified together)
# ==========================================================================

# Original value: P2.  Usable domain is P1..P3 (P0 has no gradient) —
# fewer than 10 values, all listed.
P_DEGREE = [
    'P1',
    'P2',
    'P3',
]

# ==========================================================================
# Physical constants (modified in main.py via Constant definitions)
# ==========================================================================

# Applied surface load on the deck edge (boundary label 3)
# Original value: -5 — fixed for this batch (single value)
# NOTE: f=-50 collapses the level set ("nvL>0", msh3.cpp:7127) — the
# swept load would only go up to -20; the fixed original is safe.
F_VALUE = [
    -5,
]

# Shear modulus (Constant latex name: \mu)
# Original value: 8 — fixed for this batch (single value)
MU_VALUE = [
    8,
]

# Lame parameter lambda (Constant latex name: \lambda)
# Original value: 1 — fixed for this batch (single value)
LAM_VALUE = [
    1,
]

# Lagrange multiplier of the volume term (Constant latex name: \ell)
# Original value: 100 — fixed for this batch (single value)
AFAC_VALUE = [
    100,
]
