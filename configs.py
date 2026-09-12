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
TOKEN_BUDGET_RATIO = 0.8  # fraction of MAX_INPUT_TOKENS used as token budget

# =========================================================================
# 4.  Execution — python environment (install_python)
# =========================================================================

PYTHON_VERSION = "3.11"
PYTHON_ENV_NAME = "autofreefem"

# =========================================================================
# 5.  Execution — run_autofreefem
# =========================================================================

AUTOFREEFEM_TIMEOUT = 3600  # max seconds for python main.py
AUTOFREEFEM_MAX_RETRIES = 10  # retries when providing missing .idp files

# =========================================================================
# 6.  Execution — run_freefem
# =========================================================================

FREEFEM_TIMEOUT = 3600  # max seconds for FreeFem++ total execution (linear) — -1 disables
FREEFEM_STEP_TIMEOUT = 60  # max seconds between consecutive iteration files (linear) — -1 disables
FREEFEM_NONLINEAR_FACTOR = 5  # multiplier for nonlinear-problem timeouts
FREEFEM_DONE_GRACE = 5  # grace seconds after the final iteration file before killing optimize

# =========================================================================
# 7.  Reflection — reflector / escalation / classifier / fixer
# =========================================================================

MAX_REFLECTION_TIMES = 20  # max reflection-loop iterations
REFLECTION_STALE_THRESHOLD = 10  # consecutive same error → trigger rewrite
REFLECTION_EVIDENCE_CHARS = 1_000_000  # max chars per stderr/stdout sent to LLM
REFLECTION_PARALLEL_MAX = 8  # max parallel probe fixes tried per reflection round
REFLECTION_PROBE_MIN_EPOCHS = 20  # min epochs per probe run (candidates)
REFLECTION_PROBE_MAX_EPOCHS = 50  # max epochs per probe run (candidates)
MAX_IMAGES_PER_REQUEST = 600  # max images per multimodal LLM call (hard cap)
IMAGE_SAMPLE_HEAD_TAIL = 100  # images always kept from head/tail when capping
IMAGE_SAMPLE_MIDDLE = 400  # evenly-spaced images kept from the middle when capping
