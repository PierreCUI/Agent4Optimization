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

# Sweep (Stage 6): the execution runs in temp/diffv0_beta/<penalty>/
# diff_<D>_v0_<V>/ — concurrent FreeFEM workers for the sweep grid.
SWEEP_WORKERS = 16  # USER RULE: all runs use up to 16 threads
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

MAX_REFLECTION_TIMES = 10  # total repair rounds of the serial reflection loop
MAX_REFLECTION_GROUP = 16  # USER RULE 2026-10-17: max experiments per
                           # reflection repair group (a round's
                           # fixes/designs/param combos create a NEW group,
                           # never modify the previous round's files).
                           # ONLY the FIRST execution's sweep may reach
                           # MAX_SWEEP_CELLS (see the sweep section).
MAX_SWEEP_CELLS = 64       # USER RULE 2026-10-17: the FIRST execution's
                           # parameter sweep runs at most this many cells
                           # (penalty x diffusion x v0); reflection reruns
                           # are capped by MAX_REFLECTION_GROUP instead
REFLECTION_WORKERS = 16    # USER RULE: all runs use up to 16 threads
# The promising/unpromising example lists fed to the repairer share the
# SAME budget convention as the example selector: MAX_INPUT_TOKENS x
# TOKEN_BUDGET_RATIO tokens (~4 chars/token), whole examples only
# (USER 2026-10-18).
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
