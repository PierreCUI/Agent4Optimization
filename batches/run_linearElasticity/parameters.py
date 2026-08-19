"""
parameters.py — Input parameter lists for batchrun.py and ablation.py.
(Adapted for run_linearElasticity — vector field, cantilever mesh.)

Every input feature is a list.  batchrun.py builds the full Cartesian
product; ablation.py varies one parameter at a time from the defaults
(parsed from the original results/ files).

For the batch phase, hmax / hmin_ratio / mesh_n take 5 values each
(smaller / original / larger around the original — all ablation-tested;
hmax keeps the original as its minimum because smaller values are
infeasible, see the note); p_degree keeps all 3 usable degrees.  Every
other family holds exactly ONE value, the tested original.  For future
experiments, add values to a family's list to sweep it again — no code
changes needed.
"""

# ==========================================================================
# Sweepable parameters (modified in main.py via setup_optimization)
# ==========================================================================

# MMG maximum mesh edge length  (primary control: triangle count ≈ 2×area/hmax²)
# Domain is [0,2]×[0,1] (area=2.0) — twice the diffusion domain.
# Original value: 0.02.  Lower bound 0.02 keeps hmin = hmax×min(ratio)
# >= 0.001 — below that the vector-field remeshing becomes infeasible
# (observed: hmax=0.005 corner failed with rc=8 / rc=-2).
# 5 values: the original IS the feasible minimum, so the batch takes it
# plus four larger values (all ablation-tested).
HMAX = [
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
]

# MMG minimum mesh edge length — fraction of hmax  (hmin = ratio × hmax)
# Original value: 0.50.  Upper bound 0.75 avoids the degenerate hmin = hmax
# corner (ratio 1.0 failed with rc=8 in the ablation).
# 5 values: two smaller / original / two larger (all ablation-tested).
HMIN_RATIO = [
    0.20,
    0.30,
    0.50,
    0.60,
    0.75,
]

# Shape advection speed (step size for level-set evolution)
# Original value: 0.002 (much larger than diffusion's 1e-6)
V0 = [0.002]  # fixed at the original value

# Level-set regularisation (smooths the velocity field)
# Original value: 1./10000 = 1e-4
DIFFUSION = [0.0001]  # fixed at the original value

# ==========================================================================
# Modifiable problem parameters
# ==========================================================================

# Initial mesh border resolution (int n from mesh file)
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
CHI_CONTRAST = [100]  # fixed at the original value

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

# Vertical load component
# Original value: -10
F_VALUE = [-10]  # fixed at the original value

# Poisson's ratio (Lame coefficient lambda diverges as nu -> 0.5)
# Original value: 0.3
NU_VALUE = [0.3]  # fixed at the original value

# Young's modulus (objective scales linearly with E)
# Original value: 200
E_VALUE = [200]  # fixed at the original value

# Lagrange multiplier (weight of the volume term in the objective)
# Original value: 0.25
AFAC = [0.25]  # fixed at the original value
