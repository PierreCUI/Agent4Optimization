"""
parameters.py — Input parameter lists for batchrun.py and ablation.py.
(Adapted for run_thermoElasticGravity — FULLY COUPLED thermo-elastic problem
on [0,4]x[0,1] (square(4n,n)): temperature field T with Dirichlet -30
on labels [1,3], displacement field u held by penalty springs
(penalty) on labels 2/4, thermal strain (alpha), gravity (X*rho*g) and
surface load (q).)

Every input feature is a list.  batchrun.py builds the full Cartesian
product; ablation.py varies one parameter at a time from the defaults
(parsed from the original results/ files).

For the batch phase, hmax / hmin_ratio / mesh_n take 5 values each
(two smaller, original, two larger — all ablation-tested); both degree
lists keep all 3 usable degrees and batchrun only runs the equal-degree
combos (single-degree design).  Every other family holds exactly ONE
value, the tested original.  For future experiments, add values to a
family's list to sweep it again — no code changes needed.

NOTE: this template has NO inner hole — mesh.txt is ``square(4*n, n,
[4*x, y])`` with no ``real a`` line, so there is no hole_a family for
this problem.  Four characteristic functions: X (1, 0) — a pure 0/1
material mask, NEVER modified — E (E0, E0/1000), k (k0, k0/1000) and
alpha_T (alpha0, alpha0/1000), each swept through its own divisor
family.  Unlike thermoElasticB there is only ONE penalty constant
(latex \\beta), named penalty in the template.
"""

# ==========================================================================
# Sweepable parameters (modified in main.py via setup_optimization)
# ==========================================================================

# MMG maximum mesh edge length  (primary control: triangle count ≈ 2×area/hmax²)
# Domain is [0,4]×[0,1] (area=4.0).
# Original value: 0.1
# 5 values: two smaller / original / two larger (all ablation-tested)
HMAX = [
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
]

# MMG minimum mesh edge length — fraction of hmax  (hmin = ratio × hmax)
# Original value: 0.2  (0.02 / 0.1)
# 5 values: two smaller / original / two larger (all ablation-tested;
# upper end stops below the remeshing-crash region)
HMIN_RATIO = [
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
]

# Shape advection speed (step size for level-set evolution)
# Original value: 5.1e-5 — fixed for this batch (single value)
V0 = [
    5.1e-5,
]

# Level-set regularisation (smooths the velocity field)
# Original value: 1./40000. = 2.5e-5 — fixed for this batch (single value)
DIFFUSION = [
    2.5e-5,
]

# ==========================================================================
# Modifiable problem parameters
# ==========================================================================

# Initial mesh border resolution (int n from mesh.txt: square(4n,n))
# Original value: 25
# 5 values: two smaller / original / two larger (all ablation-tested)
MESH_N = [
    10,
    20,
    25,
    40,
    50,
]

# Stiffness contrast ratio of the E characteristic function
# (E0, E0/1000): the sweep rewrites the divisor ``E0 / N``.
# Original value: 1000 — fixed for this batch (single value)
# NOTE: X (1, 0) is a pure 0/1 material mask (volume cost + gravity)
# and is NEVER modified.
CHI_CONTRAST = [
    1000,
]

# Thermal-conductivity contrast ratio of the k characteristic function
# (k0, k0/1000): the sweep rewrites the divisor ``k0 / N``.
# Original value: 1000 — fixed for this batch (single value)
CHI_K = [
    1000,
]

# Thermal-expansion contrast ratio of the alpha_T characteristic
# function (alpha0, alpha0/1000): the sweep rewrites the divisor
# ``alpha0 / N``.
# Original value: 1000 — fixed for this batch (single value)
CHI_ALPHA = [
    1000,
]

# ==========================================================================
# Finite element degrees of the TWO state fields — ablated SEPARATELY
# ==========================================================================

# Displacement field degree (u and its test function, both VectorField)
# Original value: P2.  Usable domain is P1..P3 (P0 has no gradient) —
# fewer than 10 values, all listed.
P_DEGREE_U = [
    'P1',
    'P2',
    'P3',
]

# Temperature field degree (T and its test function, both ScalarField)
# Original value: P2.  Usable domain is P1..P3 (P0 has no gradient) —
# fewer than 10 values, all listed.
P_DEGREE_T = [
    'P1',
    'P2',
    'P3',
]

# ==========================================================================
# Physical constants (modified in main.py via Constant definitions)
# ==========================================================================

# Poisson's ratio (Constant latex name: \nu)
# Original value: 0.23 — fixed for this batch (single value)
# NOTE: nu=0.15 / nu=0.35 crashed FreeFEM during remeshing in the
# sibling thermo problems; 0.23 is safe.
NU_VALUE = [
    0.23,
]

# Young's modulus (Constant latex name: E_0)
# Original value: 32e6 — fixed for this batch (single value)
E0_VALUE = [
    32e6,
]

# Density (Constant latex name: \rho) — enters the gravity term X*rho*g
# Original value: 2500 — fixed for this batch (single value)
RHO_VALUE = [
    2500,
]

# Displacement cost factor in the objective (Constant latex name: \gamma,
# template variable name: fac)
# Original value: 100 — fixed for this batch (single value)
FAC_VALUE = [
    100,
]

# Gravitational acceleration (Constant latex name: g)
# Original value: 9.81 — fixed for this batch (single value)
# NOTE: g=500 crashed Bamg in the sibling thermo problems.
G_VALUE = [
    9.81,
]

# Material cost of the volume term (Constant latex name: \ell)
# Original value: 1 — fixed for this batch (single value)
AFAC_VALUE = [
    1,
]

# Penalty-spring stiffness on labels 2 and 4 (Constant latex name:
# \beta, template variable name: penalty — thermoElasticB had TWO
# penalties betaT/beta, this template has ONE)
# Original value: 1e11 — fixed for this batch (single value)
PENALTY_VALUE = [
    1e11,
]

# Thermal conductivity (Constant latex name: k_0)
# Original value: 1.25 — fixed for this batch (single value)
K0_VALUE = [
    1.25,
]

# Thermal expansion coefficient (Constant latex name: \alpha_T)
# Original value: 3.6e-5 — fixed for this batch (single value)
ALPHA0_VALUE = [
    3.6e-5,
]

# Vertical surface load on the deck edge (Constant latex name: q)
# Original value: 5000 — fixed for this batch (single value)
Q_VALUE = [
    5000,
]
