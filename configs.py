"""Global constants, organised by pipeline module."""

# =========================================================================
# 1.  Shared — used by multiple modules
# =========================================================================

MAX_OUTER_ITERATIONS = 100  # default outer-loop epochs per optimisation problem
MAX_INNER_ITERATIONS = 10  # default inner-loop iterations (e.g. nonlinear search)
MAX_INPUT_TOKENS = 1_000_000  # max tokens per LLM call (example_selector + coder)

# =========================================================================
# 2.  Initialization — problem_parser / mesh_handler
# =========================================================================

MAX_FILE_CHARS = 1_000_000  # max chars per input file sent to LLM

# =========================================================================
# 3.  Generation — example_selector
# =========================================================================

MAX_SELECTOR_ROUNDS = 10  # max reduction rounds before scoring
TOKEN_BUDGET_RATIO = 0.8  # use 80 % of MAX_INPUT_TOKENS as budget

# =========================================================================
# 4.  Execution — python environment (install_python)
# =========================================================================

PYTHON_VERSION = "3.11"
PYTHON_ENV_NAME = "autofreefem"

# =========================================================================
# 5.  Execution — run_autofreefem
# =========================================================================

AUTOFREEFEM_TIMEOUT = 3600  # max seconds for python main.py
AUTOFREEFEM_MAX_RETRIES = 10  # retries when fetching missing .idp files
GITLAB_RAW_URL = "https://gitlab.tugraz.at/autofreefem/autofreefem/-/raw/main/autofreefem/include"

# =========================================================================
# 6.  Execution — run_freefem
# =========================================================================

FREEFEM_TIMEOUT = 3600  # max seconds for FreeFem++ total execution (linear) — -1 disables
FREEFEM_STEP_TIMEOUT = 60  # max seconds between consecutive iteration files (linear) — -1 disables
FREEFEM_NONLINEAR_FACTOR = 5  # multiplier for nonlinear-problem timeouts
EARLY_STAGNATION_EPOCH_FRACTION = 0.2  # check at this fraction of total epochs — -1 disables
EARLY_STAGNATION_MIN_IMPROVEMENT = (
    0.05  # require this fractional drop in best objective — -1 disables
)

# =========================================================================
# 7.  Reflection — reflector / escalation / classifier / fixer
# =========================================================================

MAX_REFLECTION_TIMES = 10  # max reflection-loop iterations
REFLECTION_STALE_THRESHOLD = 5  # consecutive same error → trigger rewrite
REFLECTION_MAX_REWRITES = 5  # max total rewrites before giving up
REFLECTION_EVIDENCE_CHARS = 1_000_000  # max chars per stderr/stdout sent to LLM
OBJECTIVE_STABILIZATION_BAND = 5.0  # acceptable late-stage objective oscillation range
