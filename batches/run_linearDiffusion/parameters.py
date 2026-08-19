"""
parameters.py — Input parameter lists for batchrun.py and ablation.py.

Every input feature is a list.  batchrun.py builds the full Cartesian
product; ablation.py varies one parameter at a time from the defaults
(parsed from the original results/ files).

For the batch phase, hmax / hmin_ratio / mesh_n take 5 values each
(two smaller, original, two larger — all ablation-tested except the
noted untested value); p_degree keeps all 3 usable degrees.  Every
other family holds exactly ONE value, the tested original.  For future
experiments, add values to a family's list to sweep it again — no code
changes needed.
"""

# ==========================================================================
# Sweepable parameters (modified in main.py via setup_optimization)
# ==========================================================================

# MMG maximum mesh edge length  (primary control: triangle count ≈ 2/hmax²)
# Original value: 0.1 / 8 = 0.0125  (domain is [0,2]×[0,0.8])
# 5 values: two smaller / original / two larger (all ablation-tested)
HMAX = [
    0.008,
    0.010,
    0.0125,
    0.015,
    0.020,
]

# MMG minimum mesh edge length — fraction of hmax  (hmin = ratio × hmax)
# Original value: 0.1  (hmin 0.01/8 ÷ hmax 0.1/8)
# 5 values: two smaller / original / two larger.  NOTE: 0.075 is the
# only value below the tested minimum (0.05) and is untested.
HMIN_RATIO = [
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
]

# Shape advection speed (step size for level-set evolution)
# Original value: 1e-6
V0 = [1e-06]  # fixed at the original value

# Level-set regularisation (smooths the velocity field)
# Original value: 2. / 40000. = 5e-5
DIFFUSION = [5e-05]  # fixed at the original value

# ==========================================================================
# Modifiable problem parameters
# ==========================================================================

# Initial mesh border resolution (int n from mesh.txt)
# Original value: 50
# 5 values: two smaller / original / two larger (all ablation-tested)
MESH_N = [
    20,
    30,
    50,
    75,
    100,
]

# Characteristic function contrast ratio  (strong / weak)
# Original value: 10  (weak material '1/10')
CHI_CONTRAST = [10]  # fixed at the original value

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

# Source term strength
# Original value: -10
F_VALUE = [-10]  # fixed at the original value

# Lagrange multiplier (weight of the volume term in the objective)
# Original value: 100
AFAC = [100]  # fixed at the original value

# Diffusion coefficient (multiplies X in the effective conductivity)
# Original value: 1
LAM = [1]  # fixed at the original value
