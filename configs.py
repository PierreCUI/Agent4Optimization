"""Global constants, organised by pipeline module."""

# =========================================================================
# 1.  Shared — used by multiple modules
# =========================================================================

MAX_OUTER_ITERATIONS = 100  # default outer-loop epochs per optimisation problem
MAX_INNER_ITERATIONS = 10  # default inner-loop iterations (e.g. nonlinear search)
MAX_INPUT_TOKENS = 1_000_000  # max tokens per LLM call (select_examples + coder)

# =========================================================================
# 2.  Initialization — problem_parser / mesh_handler
# =========================================================================

MAX_FILE_CHARS = 1_000_000  # max chars per input file sent to LLM
MAX_SPEC_FIXES = 10  # repair attempts per case in the generate -> review -> repair loop

# =========================================================================
# 3.  Generation — coder / check_static / select_examples
# =========================================================================

TOKEN_BUDGET_RATIO = 0.8  # fraction of MAX_INPUT_TOKENS used as token budget
EXAMPLE_SCORE_THRESHOLD = 60  # min relevance score (0-100) to keep a candidate example
MAX_STATIC_FIXES = 10  # repair attempts in the check -> review -> repair loop of Stage 5
MAX_GENERATE_RETRIES = 5  # redo attempts for a failed LLM call in Stage 4/5 (generate / review; a failed repair consumes its round instead)

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

# NO overall run budget — only per-step budgets: 60s linear / 5min
# nonlinear (FREEFEM_NONLINEAR_FACTOR), first and last step 10×.
FREEFEM_STEP_TIMEOUT = 60  # max seconds between consecutive iteration files (linear) — -1 disables
FREEFEM_NONLINEAR_FACTOR = 5  # multiplier for nonlinear-problem timeouts
FREEFEM_DONE_GRACE = 5  # grace seconds after the final iteration file before killing optimize

# =========================================================================
# 7.  Reflection — reflector / classifier / fixer
# =========================================================================

MAX_REFLECTION_TIMES = 10  # total repair rounds shared by the five fix types (syntax/formula/design/fe_param/free)
MAX_FIX_CANDIDATES = 8  # max parallel probe candidates per round, shared by formula_fix / design_fix / fe_param_fix
MAX_SNAPSHOT_FRAMES = 50  # max per-snapshot design images per vision call (beyond it, interpolate over the WHOLE run)
FF_CONSOLE_TAIL_CHARS = 8192  # FreeFEM console (stderr/stdout) keeps ONLY its LAST chars — nothing else is truncated
RENDER_MAX_FIELDS = 16  # max VTU fields rendered into the MinTh render summary
RENDER_DPI = 100  # MinTh field-render PNG resolution
RENDER_SNAPSHOT_DPI = 110  # per-snapshot Label PNG resolution
RENDER_FIGSIZE = (6, 5)  # MinTh field-render figure size (inches)
RENDER_SNAPSHOT_FIGSIZE = (4.8, 4.0)  # per-snapshot Label figure size (inches)
REFLECTION_PROBE_EPOCHS = 50  # epochs per probe run — EVERY candidate runs this budget (it is also part of the evidence the synthesis weighs)
MAX_IMAGES_PER_REQUEST = 600  # max images per multimodal LLM call (hard cap)
IMAGE_SAMPLE_HEAD_TAIL = 100  # images always kept from head/tail when capping
IMAGE_SAMPLE_MIDDLE = 400  # evenly-spaced images kept from the middle when capping
