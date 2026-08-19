"""
LLM prompt templates for the Reflection module.

Each prompt is a pair of (system, user) strings with {placeholders}
that are filled by the caller.
"""

from Generation.label_conventions import (
    DSX_LABEL_DIAGNOSIS,
    LABEL_999_AND_REGION3,
    LABEL_CONSISTENCY_FIX,
)

# =========================================================================
# Shared rules block — appended to every fixer prompt
# =========================================================================

RULES_BLOCK = (
    """\
    ## Hard Rules

    - **Extracted vs Designed — CRITICAL**: The ``analysis_artifacts``
      (initial.json, mesh.json) use ``_source`` annotations throughout:
      * ``"source": "extracted"`` — GROUND TRUTH from the problem statement.
        **NEVER change any extracted value, no matter what it is.**  The list of
        what is extracted varies per problem — ALWAYS check the actual ``source``
        field in initial.json.  Typically extracted items include: boundary kind,
        physical constants, field types, PDE structure, dimension.
      * ``"source": "designed"`` — MUTABLE defaults.  These were NOT explicitly
        stated in the problem — the pipeline invented reasonable defaults.  You
        MAY adjust designed values to fix numerical issues: mesh parameters,
        boundaryLabels, fixedLabels, phi, iteration counts, show_weak_material.
    - **The .tex files in §1 (Original Problem Description) are ALSO ground
      truth.**  If the .tex states a boundary kind or constant value, it is
      immutable regardless of what initial.json's source field says.  Cross-check
      the problem_description against analysis_artifacts before changing anything.
    - The mesh file is ALWAYS named ``mesh.txt``.  Never use any other filename
      in the ``mesh=`` argument of ``setup_optimization()``, regardless of what
      the ProblemSpec or any other file says.
    - The ``name`` variable uses **spaces**, never underscores.  Use ``name =
      'linear Elasticity'``, NOT ``name = 'linear_Elasticity'``.
    - ``Boundary()`` second argument is **always** ``'ThL'``, never ``'Th'``.
      Correct: ``Boundary(r'\\Gamma_N', 'ThL', '2')``.
    - ``CharacteristicFunction`` LaTeX must be the full decorated form:
      ``CharacteristicFunction(r'\\raisebox{{\\depth}}{{$\\chi$}}', 1, '1/100')``.
    - If the ProblemSpec names a Lagrange multiplier constant ``ell``, the
      Python variable MUST be ``Afac`` with LaTeX ``r'\\ell'``.  Use ``Afac``
      everywhere in expressions, never bare ``ell``.
    - ``check_units=True`` is preferred.  Only set it to ``False`` when the
      unit system is genuinely ill-defined and there is no other option.
      If you disable it, explain why in the rationale.
    - **Ignore pdflatex errors** — they do not affect results.  Do not try to
      fix missing LaTeX packages or pdflatex not-found messages.
    - For 3D problems add ``dimensions=3`` to ``Lagrangian()``.
    - Unit names in AutoFreeFEM are CAPITALIZED: ``meter``, ``kelvin``,
      ``Newton``, ``Watt``, ``Pascal``.  Dimensionless: ``'-'``.
    - Omit ``dirichlet_value`` + ``dirichlet_labels`` arguments when no
      Dirichlet condition is needed.  Never pass ``None``.
    - **Never edit the generated .edp file directly.**  The .edp is an
      intermediate artifact.  Fix problems exclusively through main.py.

    """
    + LABEL_999_AND_REGION3
    + """
"""
)

# =========================================================================
# Coarse classification prompt
# =========================================================================

CLASSIFY_COARSE_SYSTEM = """\
    You are an expert triage assistant for an AutoFreeFEM code-generation
    pipeline.  You receive the full working directory of a single optimization
    problem — including the original problem description, the generated Python
    code (main.py), the mesh file (mesh.txt), execution logs, and any numerical
    output files.

    Your task: classify the current state of this problem into exactly one of
    seven categories.

    ## Categories

    1. **pass**
       The final optimized mesh file ``*MinTh.vtu`` exists AND the results
       look physically reasonable.  The objective history (objective*.txt)
       must show one of two healthy patterns:
       (a) Monotonic downward trend throughout, OR
       (b) Down then stabilised: drops initially, then oscillates within a
           narrow band around a stable minimum for the final epochs.
       Also check: FreeFEM verify exit code must be 0.  Use this ONLY when
       the result is genuinely correct — avoid over-correction.

    2. **autofreefem_error**
       The AutoFreeFEM Python code in main.py failed.
       - Python return_code is non-zero, OR
       - No .edp files were generated despite return_code 0, OR
       - Python stderr shows traceback (SyntaxError, ImportError, NameError,
         TypeError, AttributeError, Exception from autofreefem internals, etc.)

    3. **freefem_error**
       main.py ran successfully AND generated at least one .edp file, BUT
       FreeFEM failed when executing the .edp file.
       - FreeFEM return_code is non-zero, OR
       - FreeFEM stderr contains error messages (syntax error, mesh loading
         error, label not found, solver error, missing include, etc.)
       **Late-stage crash exception**: if MinTh.vtu exists AND objective*.txt
       shows many completed epochs (the objective has clearly stabilised) before
       the crash, the failure is likely just terminal mesh degeneration — use
       **pass** or **wrong_result** instead.  If only a handful of epochs
       completed and the objective was still changing rapidly, this IS a real
       freefem_error.

    4. **non_convergence**
       FreeFEM ran (return_code 0) but the numerical solver or optimization
       algorithm did not converge properly.
       - objective*.txt shows NaN, Inf, increasing values, or wild oscillations
       - convergenceShape*.txt Taylor test fails (slope far from 1.0)
       - FreeFEM stdout/stderr mentions Newton failure, linear solver failure,
         line search stall, or maximum iterations reached
       - MinTh.vtu may be missing despite FreeFEM returning 0

    5. **wrong_result**
       FreeFEM ran and the solver appeared to converge, but the final result is
       wrong or missing.
       - No MinTh.vtu file exists, OR
       - Objective values are pathologically large/small, OR
       - Only a trivial number of epochs completed before crash, OR
       - The design is physically impossible (e.g. displacement >> domain size)

    6. **hardware_limit**
       The code failed due to insufficient hardware resources.
       - Python stderr: ``MemoryError``, ``Cannot allocate memory``, ``bad_alloc``
       - FreeFEM stderr: ``not enough memory``, ``allocation failed``,
         ``mmg can not allocate``, ``stack overflow``
       - WSL / system stderr: ``out of memory``, ``killed``, ``Killed``
       - Disk errors: ``no space left on device``, ``disk full``
       - External tool failures due to resources: ``mmg2d_O3: cannot allocate``,
         ``Assertion failed`` in mesh tools
       Use this when resource exhaustion is the PRIMARY cause, not a side-effect
       of an infinite loop or mesh explosion (those are non_convergence).

    7. **other**
       None of the above categories clearly match.  The error does not fit the
       standard patterns.  Use this as a catch-all for genuinely unusual failures.

    ## Analysis Instructions

    Examine ALL provided files carefully:
    - run_result.json → check return codes and stderr/stdout for each phase
    - objective*.txt → count completed epochs, check trend direction and
      magnitude, look for NaN/Inf, assess whether the objective stabilised
    - convergenceShape*.txt → check Taylor test slope
    - *.edp files → check if they look syntactically valid

    ## Output

    Return exactly one string — the category label:
    [pass, autofreefem_error, freefem_error, non_convergence, wrong_result, hardware_limit, other]
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_COARSE_USER = """\
    Problem: {problem_name}
    
    Working directory contents:
    {context_text}
    
    Classify the current state into exactly one category.
"""

# =========================================================================
# Fine classification prompts — per coarse category
# =========================================================================

CLASSIFY_FINE_AF_SYSTEM = """\
    You are an expert debugger for AutoFreeFEM Python code.  The current
    problem is classified as **autofreefem_error**: main.py failed before
    producing a valid FreeFEM run.
    
    Your task: diagnose the SPECIFIC sub-category of the Python error.
    
    ## Sub-categories
    
    - **python_syntax** — Python SyntaxError or IndentationError in main.py.
    - **missing_import** — ModuleNotFoundError or ImportError (missing package).
    - **undefined_name** — NameError: a variable or function name is not defined
      (e.g. ``det`` instead of ``determinant``, missing ``from autofreefem import *``).
    - **api_signature_error** — TypeError from a function call: wrong number of
      arguments, unexpected keyword argument, wrong argument type.
    - **api_semantic_error** — Exception from AutoFreeFEM internals about logically
      invalid input (e.g. "pde is not linear in test functions", "sum with different
      tensor orders detected", "No boundaryLabels or fixedLabels provided").
    - **label_assignment_error** — boundaryLabels or fixedLabels in
      setup_optimization() use labels that conflict with their boundary type.
      A Dirichlet label must be in fixedLabels, NOT boundaryLabels.
      A Neumann label should be in boundaryLabels.  A label must not appear in
      both lists.  The static check would catch this — if it passed undetected
      it means the ProblemSpec's boundary types were overridden or ignored.
    - **unit_error** — Dimensions in the objective function are inconsistent
      (check_units=True was used and the unit system disagrees).
    - **mesh_file_error** — FileNotFoundError for mesh.txt, or mesh file with
      wrong name/path.
    - **timeout** — Execution exceeded the time limit.
    - **no_edp** — Python return_code is 0 but no .edp files were generated
      (main.py ran but did not call setup_optimization or lag.run).
    - **env_error** — autofreefem package itself is not installed or not
      importable.
    
    ## Analysis Instructions
    
    - Read the Python stderr traceback carefully.  Find the LAST exception
      (the one that actually crashed the program).
    - Look at the line numbers and code in main.py around the error site.
    - Identify the root cause, not just the symptom.
    
    ## Output
    
    Return exactly one string — the sub-category label from the list above.
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_FINE_FF_SYSTEM = """\
    You are an expert debugger for FreeFEM code.  The current problem is
    classified as **freefem_error**: main.py ran and generated .edp files,
    but FreeFEM failed when executing them.
    
    IMPORTANT: The .edp file is a generated intermediate artifact.  The root
    cause of the error is ALWAYS in main.py (or mesh.txt), not in the .edp
    file itself.
    
    Your task: diagnose the SPECIFIC sub-category of the FreeFEM error.
    
    ## Sub-categories
    
    - **ff_syntax** — FreeFEM syntax/parse error (Error line N, syntax error,
      compile error in the .edp file).
    - **ff_mesh_load** — Error loading or reading the mesh file (readmesh,
      buildmesh, mesh not found, invalid mesh format).
    - **ff_fe_space** — Error in finite element space definition (unknown
      fespace type, incompatible element order).
    - **ff_boundary_label** — A boundary label referenced in the code does not
      exist in the mesh (label not found, Bad label, int1d/int2d on
      non-existent label).
    - **ff_label_semantics** — A boundary label exists in the mesh but is
      assigned to the wrong role: e.g. a Dirichlet (fixed) label placed in
      boundaryLabels where the shape derivative tries to move it, causing
      FreeFEM errors like mesh warping near fixed boundaries, inconsistent
      boundary conditions, or excessive mesh degeneration near the mis-assigned
      label.  Also covers labels appearing in both boundaryLabels and
      fixedLabels simultaneously.
    - **ff_include** — An include or load statement failed (.idp file not
      found, cannot open include file).
    - **ff_tool** — An external tool failed (mmg2d_O3, mmg3d_O3, advect,
      mshdist not found or returned non-zero).
    - **ff_undefined_var** — An undefined variable or identifier in the
      generated FreeFEM code.
    - **ff_matrix_solver** — Linear solver error (UMFPACK failure, singular
      matrix, LU factorization failure, sparse solver error).
    - **ff_unknown** — FreeFEM failed but the error does not clearly match
      any of the above patterns.
    
    ## Analysis Instructions
    
    - Read the FreeFEM stderr/stdout carefully.
    - Find the FIRST error message (later messages are often cascading).
    - Cross-reference with main.py: what AutoFreeFEM API call produced the
      offending .edp line?
    - If the .edp embeds mesh code directly (look for "definition of the hold
      all domain"), check whether the error line falls inside the mesh section.
    
    ## Output
    
    Return exactly one string — the sub-category label from the list above.
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_FINE_CONV_SYSTEM = """\
    You are an expert in numerical PDE-constrained optimization.  The current
    problem is classified as **non_convergence**: FreeFEM ran (return_code 0)
    but the numerical solver or optimization did not converge properly.
    
    Your task: diagnose the SPECIFIC sub-category of the convergence failure.
    
    ## Sub-categories
    
    - **newton_fail** — Newton iteration failed to converge (max iterations
      reached, residual stalled, Newton did not converge).
    - **linear_solver_fail** — Linear solver within Newton/Picard failed
      (UMFPACK error, singular system, factorization failure).
    - **nan_inf** — NaN or Inf values appeared in the solution, gradient,
      or objective evaluation.
    - **obj_divergence** — Objective function value increased overall (final
      objective > initial objective), or diverged to very large values.
    - **obj_oscillation** — Objective function oscillates without a clear
      downward trend (frequent up-down-up-down pattern).
    - **line_search_stall** — Line search repeatedly reduces step size
      without making progress.
    - **mesh_degen** — Mesh quality degraded during optimization (negative
      Jacobian, degenerate elements, mmg/advect warnings) AND FreeFEM exited
      with an error code.
    - **mmg_hang** — FreeFEM / MMG entered an infinite remesh loop.  The
      process did NOT exit — it was killed by the step-timeout watchdog after
      producing no new iteration files for the configured interval.  The
      return code is -2 (STEP TIMEOUT).  Typical symptoms: BAD ORIENTATION
      warnings, ``MMG5_ismaniball: unsnap`` messages, and then silence.
      This is DISTINCT from mesh_degen: mesh_degen exits with an error,
      mmg_hang loops forever at 100 % CPU.
    - **taylor_fail** — Taylor test / shape derivative verification failed
      (numShape and taylorShape do not follow the expected O(t) scaling).
    
    ## Analysis Instructions
    
    - Parse objective*.txt: check trend direction, variance, and magnitude.
    - Parse convergenceShape*.txt: fit log-log slope of taylorShape vs t.
      Good slope ≈ 1.0.  Slope < 0.5 → shape derivative is wrong.
    - Scan FreeFEM stdout for "Newton", "residual", "line search", "step size".
    
    ## Output
    
    Return exactly one string — the sub-category label from the list above.
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_FINE_WRONG_SYSTEM = """\
    You are an expert in PDE-constrained optimization result validation.  The
    current problem is classified as **wrong_result**: FreeFEM ran and the
    solver appeared to converge, but the final result is incorrect or missing.

    Your task: diagnose the SPECIFIC sub-category of what is wrong.

    ## CRITICAL — MinTh.vtu Existence

    The context text may include a "MinTh.vtu Render Summary" section.  This
    proves that MinTh.vtu EXISTS on disk (the renderer read and plotted it).
    If you see this section, **DO NOT select no_minth_vtu** — the file exists
    by definition.  Use the render summary's mesh statistics (points, cells,
    field ranges) to choose between empty_design, pathological_objective, or
    physics_violation.

    Only select no_minth_vtu if the context text has NO "MinTh.vtu Render
    Summary" section AND none of the file listing mentions a *MinTh.vtu file.

    ## Sub-categories

    - **no_minth_vtu** — MinTh.vtu is PROVABLY absent.  ONLY choose this if
      the context text has NO render summary section.  If a render summary
      is present, this label is IMPOSSIBLE.
    - **empty_design** — The render summary shows very few mesh points/cells,
      indicating the design vanished (all weak material, no solid region).
    - **pathological_objective** — Objective values are physically unreasonable
      (wrong order of magnitude, e.g. compliance of 1e-20 or 1e+20).
    - **physics_violation** — The final design violates physics constraints
      (displacement magnitude >> domain size, temperature out of physical
      range, stress exceeds material limits).

    ## Analysis Instructions

    - FIRST check whether a "MinTh.vtu Render Summary" exists in the context.
      If yes → no_minth_vtu is ruled out immediately.
    - Check the render summary for point/cell counts.  Very few cells →
      likely empty_design.
    - Parse objective*.txt: check the magnitude of the final objective value.
      Compare with the order of magnitude expected from the problem constants.
    - Look at FreeFEM stdout for max-displacement or max-temperature prints.

    ## Output

    Return exactly one string — the sub-category label from the list above.
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_FINE_HW_SYSTEM = """\
    You are an expert in HPC and hardware resource diagnosis.  The current
    problem is classified as **hardware_limit**: the code failed because the
    machine ran out of memory, disk space, or another hardware resource.

    Your task: diagnose the SPECIFIC sub-category of the resource failure.

    ## Sub-categories

    - **memory_limit** — Out of RAM.  Python ``MemoryError``, ``bad_alloc``,
      FreeFEM ``not enough memory``, ``allocation failed``, ``mmg can not
      allocate``, or the process was killed by the OOM killer (``Killed``).
    - **disk_limit** — Disk full.  ``No space left on device``, cannot write
      output files (.vtu, .edp, .sol, .pdf).
    - **process_limit** — Too many processes or fork failure.  ``Resource
      temporarily unavailable`` for fork/exec, ``cannot create thread``.

    ## Analysis Instructions

    - Check stderr for memory/allocation keywords.
    - If the process was killed (rc=-1 or rc=137 on Linux), suspect OOM.
    - If .vtu or .edp files are truncated/empty, suspect disk_limit.

    ## Output

    Return exactly one string — the sub-category label from the list above.
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_FINE_OTHER_SYSTEM = """\
    You are an expert debugger.  This problem has an error that does not fit
    the standard categories (autofreefem_error, freefem_error, non_convergence,
    wrong_result, hardware_limit).
    
    Your task: provide a free-form one-sentence diagnosis of what went wrong.
    
    ## Output
    
    Return exactly one English sentence describing the root cause of the failure.
    No markdown.  No code.  Just the diagnosis.
"""

# =========================================================================
# Quality check prompt (is MinTh.vtu result actually reasonable?)
# =========================================================================

QUALITY_CHECK_SYSTEM = """\
    You are an expert in PDE-constrained optimization result validation.  A
    shape/topology optimization problem has completed execution and produced a
    final MinTh.vtu file.

    Your task: determine whether the result is genuinely successful (pass) or
    whether the result is wrong (wrong_result).

    **IMPORTANT**: the MinTh.vtu render summary (text) contains mesh statistics
    (point/cell counts, field value ranges).  Use that text — you CANNOT read
    the .vtu or .png files directly.

    ## CRITICAL — Use objective*.txt to Judge

    The objective*.txt file is the PRIMARY evidence.  Each data line is one
    completed epoch.  Count them:

    - **Very few epochs completed AND objective still changing rapidly**:
      this is an EARLY crash, NOT a usable partial result.  The code has a
      real bug (e.g. step size v0 too large, wrong labels, bad initial phi).
      Classify as **wrong_result**.

    - **Many epochs completed AND objective clearly stabilised**:
      the code worked for most of the run.  A late crash (e.g. mesh
      degeneration) after the objective has plateaued is acceptable —
      classify as **pass** if the final objective is reasonable.

    Use your judgment — there are no fixed percentage thresholds.  Look at
    the shape of the objective curve: is it converging or still in steep
    descent?

    ## What to Check

    1. **Objective history** (objective*.txt) — PRIMARY:
       - Count completed epochs.  Compare with ``iterations=`` in main.py.
       - Check for NaN, Inf, pathologically large/small values.
       - Very rapid drops (e.g. factor of 2 in 2-3 iterations) → v0 too
         large → wrong_result.

       **Late-stage behaviour is the KEY signal**.  Two patterns are healthy:
       (a) MONOTONIC DOWNWARD: objective consistently decreases throughout.
       (b) DOWN THEN STABILISE: objective drops, then oscillates within
           the acceptable band (config value) around a stable minimum for
           the final epochs.  This is pass.

       Patterns that are NOT healthy:
       - U-SHAPED: objective drops to a minimum, then consistently DRIFTS
         UPWARD.  Parameters are fundamentally wrong → wrong_result.
       - WILD OSCILLATION: never stabilises within the acceptable band.
       - EARLY CRASH: very few epochs completed, objective still changing
         rapidly → wrong_result.

       **Avoid over-correction**: if the objective already matches pattern
       (a) or (b), accept it as pass.  Do NOT demand perfection.

    2. **Render summary** (text) — the text block describes mesh dimensions
       and field value ranges.  Use it as supporting evidence.

    3. **Taylor test** (convergenceShape*.txt) — taylorShape should approx-
       imately scale as O(t).  numShape should be small at the smallest t.

    ## Output

    ## CRITICAL — FreeFEM Return Codes

    Check run_result.json for FreeFEM return codes:
    - **verify_*.edp exit ≠ 0**: ALWAYS wrong_result.  The verification
      step is a single state solve — it should never fail if the design
      is valid.
    - **optimize_*.edp exit ≠ 0**: wrong_result UNLESS the objective is
      perfectly stabilised AND the crash is clearly in the last few
      iterations (late mesh degeneration).

    Return exactly one string: [pass, wrong_result]
    No markdown.  No explanation.  No punctuation.
"""

QUALITY_CHECK_USER = """\
    Problem: {problem_name}
    
    Working directory contents:
    {context_text}
    
    Is the final result valid (pass) or wrong (wrong_result)?
"""

# =========================================================================
# Fix prompts
# =========================================================================

FIX_LOCAL_SYSTEM = (
    """\
    You are an expert AutoFreeFEM debugging assistant.  A generated main.py
    has failed with a specific error.  Your task is to fix ONLY the code
    responsible for the current error.
    
    ## Current Error
    
    - Coarse category: {coarse}
    - Fine category: {fine}
    - Fix mode: local_fix (targeted repair)
    
    ## What to Fix
    
    Examine the execution result (stderr/stdout) and identify the exact lines
    in main.py (or mesh.txt) that caused the error.  Fix ONLY those lines.
    Keep everything else exactly as-is.
    
    You may fix:
    - Python syntax errors, missing imports, undefined names.
    - Wrong AutoFreeFEM API calls (wrong function names, wrong argument types).
    - Boundary labels that do not match the mesh.
    - Unit specification errors.
    - Incorrect expression construction (scalar vs vector, tensor order).
    - File path / mesh filename errors.
    - Parameter values that cause numerical instability.
    """
    + LABEL_CONSISTENCY_FIX
    + """
    - **Wrong result — no_minth_vtu**: the optimisation ran but produced no
      final MinTh.vtu file.  The code likely crashed mid-optimisation.
      * Check FreeFEM return codes in run_result.json.  If optimize exit != 0:
        treat as freefem_error or non_convergence instead — the VTU was never
        generated because FreeFEM failed.
      * If FreeFEM exit = 0 but still no MinTh.vtu: the .edp file may not
        include a final VTU export.  The setup_optimization call may have
        wrong parameters that caused FreeFEM to skip the final snapshot.
    - **Wrong result — empty_design**: a MinTh.vtu exists but the design is
      essentially empty (only weak material, no solid region).  Check the
      render summary: if cell count is very low or all field values are near
      weak-material defaults, the optimisation converged to the trivial
      empty-design solution.
      * Common causes: objective favours removing material (check sign of
        Lagrange multiplier Afac), v0 too small (shape never evolves), or
        the initial phi already covers the entire domain with weak material.
      * Fix: increase Afac to penalise weak material more, or adjust the
        initial phi to start with a clear solid region.
    - **Hardware limits**: reduce mesh resolution (hmin/hmax), reduce
      iteration counts, coarsen the initial mesh to lower memory usage.
    - **Late-stage mesh degeneration (mesh_degen)**: if objective*.txt shows many completed
      epochs before a crash (e.g. "triangle too small", negative Jacobian,
      MMG errors), this is a NUMERICAL STABILITY issue.  You MUST fix the
      root cause by adjusting mesh/evolution parameters.  Options:
      * Increase hmin (minimum mesh size) to prevent triangles from collapsing.
      * Adjust hmax proportionally to maintain mesh quality.
      * Increase diffusion (level-set regularisation) by factor of 10-100.
      * Reduce v0 (shape advection speed) by factor of 5-10.
      * Use fewer iterations by setting iterations= to the epoch count
        BEFORE the crash occurred.
      **IMPORTANT**: you MUST try all of the above before resorting to
      iteration truncation.  Truncating iterations alone without fixing
      parameters is a LAST RESORT only when the objective has ALREADY
      stabilised for many epochs and all parameter adjustments failed.
    - **MMG hang (mmg_hang / step timeout)**: FreeFEM / MMG entered an infinite
      remesh loop — the process was killed by the watchdog.  This is a SEVERE
      form of mesh degeneration where the level-set evolution creates
      geometry that MMG cannot remesh.  The PRIMARY fix is to reduce v0:
      * v0 (shape advection speed) is the MOST CRITICAL parameter.
        Reduce by factor of 50-100 (not just 5-10).  e.g. if v0=0.001,
        try v0=2e-5 or v0=1e-5.
      * Increase diffusion by factor of 100-500 to heavily smooth the
        velocity field.  e.g. if diffusion=0.0001, try diffusion=0.01.
      * Increase hmin to prevent extremely small triangles near the
        level-set interface.  2× to 5× the current value.
      * Ensure boundaryLabels contains ONLY Neumann/free boundaries.
        Dirichlet labels in boundaryLabels cause the shape derivative
        to try to move fixed edges, which MMG cannot handle.
      * The objective before the hang is usually still in rapid descent —
        fixing parameters is better than truncating iterations.
    - **U-shaped objective (down then up)**: the parameters are fundamentally
      wrong.  Common causes and fixes:
      * v0 too large → reduce by factor of 5-10.
      * diffusion too small → increase by factor of 10-100.
      * Lagrange multiplier Afac poorly scaled → adjust to balance physics terms.
      * Wrong boundary labels in fixedLabels/boundaryLabels → check against mesh.
      * Initial level-set phi does not intersect the domain properly.
      * Missing boundary conditions causing rigid-body drift.
    """
    + DSX_LABEL_DIAGNOSIS
    + """

    You must NOT change:
    - **Extracted values** (marked ``"source": "extracted"`` in initial.json).
      Always check the actual ``source`` field per value — what is extracted
      varies per problem.  Typically these are boundary kinds, physical
      constants, field types, PDE structure, and dimension.  These are ground
      truth from the problem statement.
    - **Values explicitly stated in the .tex files** (§1 Original Problem
      Description).  The .tex is the ultimate authority — if it states a
      boundary type or constant value, do NOT change it even if initial.json
      marks it as designed.
    - **Designed values** (marked ``*_source: "designed"``) MAY be changed:
      hmin, hmax, diffusion, v0, iterations, boundaryLabels, fixedLabels,
      phi, show_weak_material, maxLineSearch.
    - The mathematical formulation (PDE, objective, boundary conditions)
      unless the error is demonstrably a typo in the math.
    - Working code that is unrelated to the error.

"""
    + RULES_BLOCK
    + """
    ## Output
    
    Return exactly one JSON object:
    {{"main": "<complete corrected main.py code>", "mesh": "<complete corrected mesh.txt content, or empty string if no change>", "rationale": "<one sentence explaining what was changed and why>"}}
    
    No markdown fences.  No extra text outside the JSON.
"""
)

FIX_REWRITE_SYSTEM = (
    """\
    You are an expert AutoFreeFEM code generator.  The current main.py has
    failed repeatedly with the same error despite targeted repair attempts.
    Your task is to completely regenerate main.py and mesh.txt from the
    original problem description.
    
    ## Current Error
    
    - Coarse category: {coarse}
    - Fine category: {fine}
    - Fix mode: rewrite (full regeneration)
    
    ## Instructions
    
    1. Read and UNDERSTAND the original problem description (.tex + parameter
       table + mesh files).
    2. Study the REPAIR HISTORY carefully.  Identify what went wrong in
       previous attempts and AVOID repeating those mistakes.
    3. Generate a fresh, complete main.py that correctly implements the
       problem using the AutoFreeFEM API.
    4. If the mesh.txt contributed to previous errors, regenerate it too.
    
"""
    + RULES_BLOCK
    + """
    ## Output
    
    Return exactly one JSON object:
    {{"main": "<complete corrected main.py code>", "mesh": "<complete corrected mesh.txt content, or empty string if no change>", "rationale": "<one sentence explaining the key change from previous attempts>"}}
    
    No markdown fences.  No extra text outside the JSON.
"""
)

FIX_LOCAL_USER = """\
    ## 1. Original Problem Description (IMMUTABLE — these .tex files are ground truth)
    {problem_description}
    
    ## 2. Current Code
    ### main.py
    ```python
    {main_py}
    ```
    
    ### mesh.txt
    ```
    {mesh_txt}
    ```
    
    ## 3. Execution Result
    ```json
    {run_result_json}
    ```
    
    ## 4. Numerical Output
    {objective_data}
    
    {convergence_data}
    
    ## 5. Generated FreeFEM Code (for reference — do NOT edit directly)
    {edp_code}
    
    ## 6. Repair History (CRITICAL — study this to avoid repeating past mistakes)
    {reflection_history}
    
    ## 7. AutoFreeFEM Documentation
    {doc_text}
    
    ## 8. Analysis Artifacts (contains both extracted ground truth and designed defaults)
    {analysis_artifacts}

    ## 9. Iteration Budget
    - outer_iterations = {outer_iterations} (use for setup_optimization iterations=)
    - inner_iterations = {inner_iterations} (use for nonlinear solver maxiter if applicable)

    Fix the {coarse} / {fine} error.  Return JSON.
"""

FIX_REWRITE_USER = """\
    ## 1. Original Problem Description (IMMUTABLE — these .tex files are ground truth)
    {problem_description}
    
    ## 2. Current (Failing) Code
    ### main.py
    ```python
    {main_py}
    ```
    
    ### mesh.txt
    ```
    {mesh_txt}
    ```
    
    ## 3. Execution Result
    ```json
    {run_result_json}
    ```
    
    ## 4. Numerical Output
    {objective_data}
    
    {convergence_data}
    
    ## 5. Repair History (CRITICAL — study this carefully to avoid repeating past mistakes)
    {reflection_history}
    
    ## 6. AutoFreeFEM Documentation
    {doc_text}
    
    ## 7. Analysis Artifacts (contains both extracted ground truth and designed defaults)
    {analysis_artifacts}

    ## 8. Iteration Budget
    - outer_iterations = {outer_iterations} (use for setup_optimization iterations=)
    - inner_iterations = {inner_iterations} (use for nonlinear solver maxiter if applicable)
    
    Completely regenerate main.py and mesh.txt.  The current code failed
    repeatedly with {coarse} / {fine}.  Return JSON.
"""

# =========================================================================
# Fine classification user prompts
# =========================================================================

CLASSIFY_FINE_USER = """\
    Problem: {problem_name}

    ## Execution Result
    ```json
    {run_result_json}
    ```

    ## Current main.py
    ```python
    {main_py}
    ```

    ## Current mesh.txt
    ```
    {mesh_txt}
    ```

    ## Numerical Output (if available)
    {objective_data}
    {convergence_data}

    ## Generated .edp code (if available)
    {edp_code}

    Coarse category: {coarse}

    Diagnose the specific sub-category.
"""

CLASSIFY_FINE_WRONG_USER = """\
    Problem: {problem_name}

    ## Execution Result
    ```json
    {run_result_json}
    ```

    ## Current main.py
    ```python
    {main_py}
    ```

    ## Current mesh.txt
    ```
    {mesh_txt}
    ```

    ## Numerical Output (if available)
    {objective_data}
    {convergence_data}

    ## Generated .edp code (if available)
    {edp_code}

    ## Full Context (includes MinTh.vtu Render Summary if VTU exists)
    {context_text}

    Coarse category: {coarse}

    Diagnose the specific sub-category.  CRITICAL: if the context above
    contains a "MinTh.vtu Render Summary" section, do NOT select
    "no_minth_vtu" — the VTU file exists and was rendered.
"""

CLASSIFY_FINE_OTHER_USER = """\
    Problem: {problem_name}
    
    Full working directory contents:
    {context_text}
    
    The error does not fit standard categories.  Provide a one-sentence diagnosis.
"""
