"""
parameters.py — Input parameter lists for batchrun.py and ablation.py.
(Adapted for run_linearElasticity_Target — vector field, square domain
with inner hole, spring boundary, deformation-dependent target functional.)

Every input feature is a list.  batchrun.py builds the full Cartesian
product; ablation.py varies one parameter at a time from the defaults
(parsed from the original results/ files).

The FULL parameter set is kept here — every family known to the sweep.
For the batch phase, hmax / hmin_ratio / mesh_n take 5 values each
(two smaller, original, two larger — all ablation-tested); p_degree
keeps all 3 usable degrees.  Every other family holds exactly ONE
value, the tested original.  For future experiments, add values to a
family's list to sweep it again — no code changes needed.
"""

# ==========================================================================
# Sweepable parameters (modified in main.py via setup_optimization)
# ==========================================================================

# MMG maximum mesh edge length  (primary control: triangle count ≈ 2×area/hmax²)
# Domain is [0,2]×[0,1] (area=2.0) with an inner square hole.
# Original value: 0.02
# 5 values: two smaller / original / two larger (all ablation-tested)
HMAX = [
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
]

# MMG minimum mesh edge length — fraction of hmax  (hmin = ratio × hmax)
# Original value: 0.25  (0.005 / 0.02)
# 5 values: two smaller / original / two larger (all ablation-tested)
HMIN_RATIO = [
    0.10,
    0.15,
    0.25,
    0.40,
    0.60,
]

# Shape advection speed (step size for level-set evolution)
# Original value: 2e-4 — fixed for this batch (single value)
V0 = [
    2e-4,
]

# Level-set regularisation (smooths the velocity field)
# Original value: 0.01 — fixed for this batch (single value)
DIFFUSION = [
    0.01,
]

# ==========================================================================
# Modifiable problem parameters
# ==========================================================================

# Initial mesh border resolution (int n from mesh.txt)
# Original value: 60
# 5 values: two smaller / original / two larger (all ablation-tested)
MESH_N = [
    30,
    40,
    60,
    75,
    100,
]

# Characteristic function contrast ratio  (strong / weak)
# Original value: 1 / (1/100) = 100  (weak material '1/100')
# Fixed for this batch (single value).
# NOTE: the template has a SECOND characteristic function X2 (1, 0) that
# restricts the volume load to the solid material.  It is NEVER modified
# by the sweep — the chi_contrast rewrite only targets the quoted weak
# value of X, so X2 always keeps its copied default.
CHI_CONTRAST = [
    100,
]

# ==========================================================================
# Inner hole size (mesh.txt: real a).  The hole borders C11-C14 all use a
# as their t-range, so this single value scales the hole.
# Constraints: a > 0 and a <= 0.8 (hole spans [0.2, 0.2+a] in [0,2]×[0,1]).
# Original value: 0.6 — no longer swept (single value)
HOLE_A = [
    0.6,
]

# ==========================================================================
# Finite element degree of the state field  (field and its test function
# always share the same degree — modified together)
# Original value: P2.  Usable domain is P1..P3 (P0 has no gradient) —
# only 3 values, all listed.
P_DEGREE = [
    'P1',
    'P2',
    'P3',
]

# ==========================================================================
# Physical constants (modified in main.py via Constant definitions)
# ==========================================================================

# Vertical load component
# Original value: -10 — fixed for this batch (single value)
F_VALUE = [
    -10,
]

# Poisson's ratio (Lame coefficient lambda diverges as nu -> 0.5)
# Original value: 0.3 — fixed for this batch (single value)
NU_VALUE = [
    0.3,
]

# Young's modulus (objective scales linearly with E)
# Original value: 2000 — fixed for this batch (single value)
E_VALUE = [
    2000,
]

# Spring stiffness of the interface boundary (Constant latex name: c)
# Original value: 1e10 — fixed for this batch (single value)
ALPHA = [
    1e10,
]
