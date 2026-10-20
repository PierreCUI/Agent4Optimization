"""
LLM prompt templates for the Reflection module.

Each prompt is a pair of (system, user) strings with {placeholders}
that are filled by the caller.
"""

from Initialization.phi_rules import PHI_GUIDANCE

# PHI_GUIDANCE = PHI_RULES + PHI_TEMPLATES — the full phi rule set from
# Initialization (the same rules the Generation stage designs phi with)
# is reused whenever a fix may touch the initial level set.
PHI_TEMPLATES = PHI_GUIDANCE

# label / design-role / dsx fix guidance (absorbed from the former
# Generation.label_conventions — Initialization owns the label
# design; these blocks only guide the Reflection fixer)
# =====================================================================

DESIGN_DIRICHLET_PENALTY = """\
**DESIGN DIRICHLET — implement as a penalty Robin (CRITICAL when a
Dirichlet boundary is a DESIGN boundary)**:
- A Dirichlet boundary that must MOVE (design role — it belongs in
  ``boundaryLabels``) CANNOT be implemented as a ScalarField Dirichlet
  value: moving-Dirichlet shape derivatives are NOT implemented, and a
  Dirichlet label that appears in NEITHER boundaryLabels NOR fixedLabels
  behaves as a free boundary in the gradient identification (the shape
  evolves there with systematically wrong gradients).
- Implement it as a PENALTY ROBIN condition instead:
  ```python
  pen = Constant('pen', 1e15, 'Dirichlet penalty coefficient', Watt / meter ** 2 / kelvin)
  weak = ... + dsx(pen * (u - u_D) * testu, Th, label)
  ```
- Penalty scale: pen = 1e15 W/(m^2 K) — the pipeline-wide standard
  value, ALWAYS use it for design-Dirichlet penalty terms.  The pen
  VALUE is FIXED CONTENT (always exactly 1e15, never a tuning target);
  only the EXISTENCE of the penalty implementation is a design choice.
  With
  k0 ~ 200 W/(m K) and a domain size ~ 1 m, the boundary Biot number
  pen*L/k0 ~ 5e12 holds the boundary temperature within ~1e-10 % of
  u_D.  Do NOT use the FreeFEM tgv scale (1e30) — it would dominate
  the shape derivative and make the optimisation ill-conditioned.
- The label is then a design boundary implemented by the penalty: it
  goes in ``boundaryLabels`` (normal-locked) or NEITHER list (free) per
  the three-way treatment — never in ``fixedLabels`` — and the
  ScalarField gets NO Dirichlet argument for it.
- The prescribed value u_D enters ONLY through the penalty weak-form
  term (plus the temperature-integral weight where the objective needs
  it) — never through the field declaration.
- A Dirichlet boundary that is FIXED (design role fixed) keeps the
  normal ScalarField declaration:
  ``ScalarField('u', 'P2', Th, '0.', [label], kelvin)``.
"""

DSX_LABEL_DIAGNOSIS = """\
**dsx label semantics (CRITICAL for syntax errors around labels)**:
AutoFreeFEM's ``dsx(expr, Th, X)`` third argument is ALWAYS the RAW
MESH LABEL: ``Th = Domain('D','Th')`` then ``dsx(..., Th, 2)``
generates ``int1d(Th, 2)``.  (Boundary objects are never written.)
* **Diagnosis**: if the objective is huge/small/stagnant and the code
  uses integer labels 0, 1, 2, ... in dsx, the labels are likely
  wrong.  Check whether the mesh actually has those labels (read
  mesh.txt) — a label the mesh does not carry makes the boundary
  integral silently evaluate to 0, producing pathological objectives
  (e.g. -1e13, NaN, or near-zero gradients).
* **Fix**: compare the dsx labels to the mesh labels in mesh.txt and
  use labels that actually exist.  For compliance-type problems (no
  target shape), the spring is typically on the fixed (clamped) edge
  labels — the Dirichlet-fixed non-10 boundaries, never on the label-10
  boundary."""

LABEL_999_AND_REGION3 = """\
**IMPORTANT — the label-10 boundary (material boundary)**:
- **The label-10 boundary** is the MATERIAL boundary — the solid/void
  boundary of a hole inside the domain.  It is a **design boundary by
  default** (it moves with the level set) and carries **NO physical
  condition**.  It is NOT a non-10 boundary: **never include it
  in ``boundaryLabels`` or ``fixedLabels``** — both lists are for
  non-10 boundaries only.
- **The label-10 boundary ALWAYS carries mesh label 10** (a FreeFEM
  requirement — the material boundary label must be 10).
  Holes carved by the level-set function ``phi`` are labelled 10
  automatically — no manual handling is needed.  A manually defined hole
  must be assigned ``label=10`` by hand in the mesh.
- **Label 999** is the special dsx reference of the evolving level-set
  boundary (the solid/void boundary tracked by MMG) — NOT a mesh label.
  Use it for spring forces that prevent rigid-body motion:
  ``dsx(..., Th, '999')``.
- **Region 3** is the void/weak-material region.  Meshes with holes
  use ``int[int] re = [0,3]; Th = change(Th, reft=re)`` to mark
  the hole as region 3.
- ``show_weak_material`` is **always** forced to ``False`` — never rely
  on ``True`` for two distinct characteristic functions or boundary
  terms on label 999; express them through the Lagrangian structure
  itself.
- NOTE: the label-10 boundary lives WITHIN a single computational domain.
  It is distinct from the INTERFACES of multi-domain problems — the
  shared boundaries BETWEEN two or more computational domains, which
  are NON-10 boundaries with their own design role and continuity
  conditions (see below)."""

LABEL_CONSISTENCY_FIX = """\
Label assignment fix rules (non-10 boundaries only):
* Non-10 boundaries have two independent aspects: the DESIGN role
  (design = movable, fixed = frozen) and the PHYSICAL condition
  (Dirichlet / Neumann / Robin).  Fix the design role, not the physics.
* **Explicit design_role is ground truth**: if the ProblemSpec's
  ``boundaries`` array gives a boundary an explicit ``design_role``
  ("design" or "fixed"), the code MUST match it — the physical kind does
  NOT override it.  E.g. a Neumann load boundary whose position is
  prescribed by the device layout is legitimately FIXED; do not move it
  to boundaryLabels.
* Without an explicit design_role, the physical-kind convention applies
  (the THREE-way boundary treatment — fixedLabels = FROZEN, the shape
  cannot move there at all; boundaryLabels = NORMAL-LOCKED, the
  boundary keeps its position and may only slide tangentially; a
  boundary in NEITHER list = FREE):
  * Dirichlet boundaries (the field value is prescribed — physics-fixed,
    e.g. clamped edges) → fixedLabels.
  * Load / support edges whose position is prescribed → fixedLabels or
    boundaryLabels (normal-locked), per the problem.
  * Other outer edges → FREE (in neither list).  Two empty lists mean
    every non-10 boundary is FREE.
* A label MUST NOT appear in BOTH lists — it is either movable or fixed,
  never both.  If both contain the same label, remove it from
  boundaryLabels (keep the Dirichlet label in fixedLabels).
* The label-10 boundary (material boundary / holes) is design
  by default, carries NO physical condition, and NEVER appears in either
  list — both lists are for non-10 boundaries only.
* Check the ProblemSpec's ``boundaries`` array in the analysis artifacts:
  each boundary has a ``kind`` (physical condition), an optional
  ``design_role`` (explicit design role — follow it exactly), and a
  ``label`` number.  Map per the three-way treatment above.
* Fix: edit setup_optimization() — move a frozen boundary into
  fixedLabels, a normal-locked one into boundaryLabels, and take a free
  boundary OUT of both lists (empty lists = all free).
* **Design Dirichlet — penalty Robin**: a Dirichlet boundary that the
  problem (or an explicit design_role) makes a DESIGN boundary must be
  implemented as a penalty Robin condition (see the DESIGN DIRICHLET
  rules): replace the ScalarField Dirichlet value with a weak-form term
  ``dsx(pen * (u - u_D) * testu, Th, label)`` with pen = 1e15, and keep
  the label in boundaryLabels."""


# =========================================================================
# Diagnosis rules — the design/fe-parameter judgments and their fixes
# (which situation says a value is too big / too small, and how to fix
# it; symmetry; the three-way boundary lists; other values)
# =========================================================================

PARAM_RULES = """\
    ## Parameter Diagnosis Rules (design_error / fe_parameter_error judgments and their fixes)

    ### a. ell / hmin / hmax / diffusion / v0 — when they are too big or
    ### too small, and how to fix

    * ell (the volume weight Afac):
      - COLLAPSE (the design shed nearly all material — hairlines, beads,
        no load-carrying path): ell is TOO LARGE, material is
        unaffordable.  FIX: DECREASE ell (x3-10 per attempt).
      - BLOCK (the seed's block survived, the design barely moved,
        material stayed cheap): ell is TOO SMALL, material is
        effectively free.  FIX: INCREASE ell (x3-10 per attempt).
      - A design oscillating wildly between dense and empty across
        epochs can also mean ell sits near an unstable balance — nudge
        it ONE direction and let the probe results show which.
    * hmin / hmax (remeshing sizes):
      - Mesh degeneration / mesh reversal / MMG failures mid-run: the
        mesh is TOO FINE for the level-set features — hmin (often with
        hmax) TOO SMALL.  FIX: RAISE hmin (x2-5).
      - The design features look unresolved / blurred / staircase-like
        while the objective stagnates early: the mesh is TOO COARSE —
        hmin/hmax TOO LARGE.  FIX: LOWER hmin and hmax (x2).
      - Very long epochs / near timeouts without a crash: the mesh is
        expensive — RAISE hmin (and hmax) to cut the remeshing cost.
    * diffusion (velocity-field smoothing):
      - The interface motion is noisy, jagged or oscillating epoch to
        epoch: diffusion is TOO SMALL.  FIX: RAISE diffusion (x10-100,
        toward 0.1-1.0).
      - The design barely moves although the physics would reward motion
        (frozen evolution with a healthy ell): diffusion is TOO LARGE
        (over-smoothed velocity) — LOWER it (x5-10).
    * v0 (interface velocity scale):
      - Crashes, NaN/Inf, wild objective oscillation, mesh explosion:
        v0 is TOO LARGE — the interface steps too far per epoch.  FIX:
        LOWER v0 (x5-10).
      - The objective moves only a hair per epoch / the design never
        evolves (frozen evolution): v0 is TOO SMALL.  FIX: RAISE v0
        (x2-5).

    ### b. Symmetry — judgment and repair

    * When the problem is symmetric (the mesh geometry AND its boundary
      labels are mirror-symmetric — the guidance states the axis), the
      final design should LOOK symmetric about that axis.
    * Tolerate SMALL asymmetries: finite-element discretisation and
      remeshing leave ragged edges, slightly different element counts
      and minor detail mismatches — these are NOT failures.
    * Only a CLEARLY lopsided design (structure present on one side and
      absent on the other) is a failure.  Its usual cause is the INITIAL
      phi: sin-based oscillatory terms about the mirror axis are
      anti-symmetric and break the symmetry.
    * FIX: replace phi with a SYMMETRIC family from the PHI TEMPLATE
      LIBRARY (cos-based terms about the mirror axis), e.g. a centred
      shape symmetric about x = L/2 and y = H/2.

    ### c. Other parameter judgments

    * iterations: FIXED content, not a repair target — iterations= must
      match the spec (the syntax check owns it).  A truncated run is
      repaired by the fe-parameter fix (smaller v0, adjusted
      hmin/diffusion), never by raising iterations.
    * Designed physical constants: re-scale ONE designed constant when
      its magnitude is inconsistent with the problem scale (dimension
      mismatch, off by orders of magnitude vs the .tex values' implied
      scale).  Never touch extracted constants.
    * phi (any direction): the initial level set must have a negative
      (solid) region inside the domain and must cover the fixed
      load/sink segments; a phi whose solid region is disconnected from
      the load or the sink can never grow a conducting / load-carrying
      path.
    * When several parameters are suspects, probe DIFFERENT single-axis
      directions in parallel rather than one big combination change —
      the synthesis judges from the objective curves and the design
      images which axis actually helped.

    ### d. boundaryLabels / fixedLabels (the THREE-way boundary treatment)

    * fixedLabels = FROZEN — a hard constraint (hatu=0): the shape
      cannot move there at all.
    * boundaryLabels = NORMAL-LOCKED — a 1e20 penalty forces the
      normal velocity to zero: the boundary keeps its position, only
      tangential sliding is possible.
    * a boundary in NEITHER list = FREE — fully deformable.  Two EMPTY
      lists therefore mean every non-10 boundary is FREE.
    * The lists are mutually exclusive and label 10 never enters
      either.
    * WHEN to change them: the design NEVER evolves although the
      numerical parameters look healthy — a design boundary is wrongly
      frozen (in fixedLabels) or wrongly normal-locked; FIX: move it
      out of fixedLabels (to boundaryLabels for a normal-lock, or to
      NEITHER for full freedom).  Conversely, the outer box deforms
      although the problem keeps it in place; FIX: normal-lock the
      moving edges (boundaryLabels) or freeze them (fixedLabels).
    * HOW: follow the spec's design_role for every boundary — an
      explicit design_role is ground truth; without one the convention
      is Dirichlet (physics-fixed) -> fixedLabels, prescribed-position
      load/support edges -> fixedLabels or boundaryLabels, other outer
      edges -> FREE.  Reassign the LISTS only — never rewrite the
      weak-form dsx labels themselves (formula content is not a
      parameter).
"""

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
        field in initial.json.  Typically extracted items include: the boundary
        PHYSICAL kinds (Dirichlet / Neumann / Robin), physical constants,
        field types, PDE structure, dimension.
      * ``"source": "designed"`` — MUTABLE defaults.  These were NOT explicitly
        stated in the problem — the pipeline invented reasonable defaults.  You
        MAY adjust designed values to fix numerical issues: mesh parameters,
        boundaryLabels, fixedLabels, phi, iteration counts, show_weak_material.
    - **The .tex files in §1 (Original Problem Description) are ALSO ground
      truth.**  If the .tex states a boundary kind or constant value, it is
      immutable regardless of what initial.json's source field says.  Cross-check
      the problem_description against analysis_artifacts before changing anything.
    - **Non-10 boundaries have two INDEPENDENT aspects — do not confuse
      them.**  (1) The DESIGN role: design = movable → ``boundaryLabels``,
      fixed = frozen → ``fixedLabels``.  Convention: Dirichlet (physics-fixed)
      boundaries are normally fixed; Neumann / Robin (load / mixed) boundaries
      are normally design.  **An explicit ``design_role`` in the ProblemSpec
      is ground truth and overrides the convention** — e.g. a Neumann load
      boundary whose position is prescribed by the device layout is
      legitimately FIXED.  (2) The PHYSICAL condition: Dirichlet / Neumann /
      Robin; a zero-load Neumann is a FREE boundary (adiabatic in thermal
      problems, traction-free in mechanics) and is still Neumann.
      The label-10 boundary (material boundary — holes / the solid-void
      boundary of the level set, mesh label 10) is design by default,
      carries NO physical condition, and NEVER appears in either list.
    - **Multi-domain problems and interfaces.**  Most problems use ONE
      computational domain; a problem with two or more materials / regions
      needs one computational domain per material / region.  An INTERFACE
      is the shared boundary of two computational domains (wherever it
      lies — inside or on the outer boundary).  An interface is a NON-10
      boundary with its own design role — design (movable) or fixed
      (frozen), decided per problem (NOT design by default like label
      10) — and its own physical condition, which is one of two
      CONTINUITY conditions between the fields on the two sides: value
      continuity (e.g. ``T1 = T2`` in diffusion) or normal-flux
      continuity (e.g. ``k1 * grad(T1)·n = k2 * grad(T2)·n``).  These
      are NOT Dirichlet/Neumann/Robin conditions (non-10 boundaries
      only).  An interface is NOT the label-10 boundary: label 10 is the
      solid/void boundary of the level set within ONE domain; an
      interface separates TWO computational domains.
    - The mesh file is ALWAYS named ``mesh.txt``.  Never use any other filename
      in the ``mesh=`` argument of ``setup_optimization()``, regardless of what
      the ProblemSpec or any other file says.
    - The ``name`` variable uses **spaces**, never underscores.  Use ``name =
      'linear Elasticity'``, NOT ``name = 'linear_Elasticity'``.
    - **Never declare ``Boundary()`` objects.**  Boundary terms in the
      weak form and boundary conditions use the RAW mesh labels
      directly: ``dsx(expr, Th, <label>)`` with the label numbers from
      mesh.txt.
    - ``CharacteristicFunction`` LaTeX must be the full decorated form:
      ``CharacteristicFunction(r'\\raisebox{{\\depth}}{{$\\chi$}}', 1, '1/100')``.
    - If the ProblemSpec names a Lagrange multiplier constant ``ell``, the
      Python variable MUST be ``Afac`` with LaTeX ``r'\\ell'``.  Use ``Afac``
      everywhere in expressions, never bare ``ell``.
    - The Afac VALUE is a DESIGNED value (the Lagrange multiplier is
      never part of the problem statement) UNLESS the spec's extracted
      constants fix ``ell`` — then it is ground truth and immutable.
      It is re-tuned ONLY by the design repair: a COLLAPSED design
      needs a SMALLER ell, a BLOCKED one a LARGER ell, at most one order
      of magnitude per attempt.  Do NOT change it outside the design
      permission.
    - ``check_units=True`` is preferred.  Only set it to ``False`` when the
      unit system is genuinely ill-defined and there is no other option.
      If you disable it, explain why in the rationale.
    - **Ignore pdflatex errors** — they do not affect results.  Do not try to
      fix missing LaTeX packages or pdflatex not-found messages.
    - Unit names in AutoFreeFEM are case-sensitive: ``meter`` and ``kelvin``
      are lowercase, ``Newton``, ``Watt``, ``Pascal`` are capitalized.
      Dimensionless: ``'-'``.
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
# Coarse classification prompt — 4 error categories + pass
# =========================================================================

CLASSIFY_COARSE_SYSTEM = """\
    You are an expert triage assistant for an AutoFreeFEM code-generation
    pipeline.  You receive the full working directory of a single optimization
    problem — structured into four sections: the problem description (问题),
    the generated Python code (py代码), the generated FreeFEM code
    (freefem代码), and the execution results (结果).

    Your task: classify the current state of this problem into exactly one
    of the categories listed under "Usable error types" in the user
    message (the pipeline has already removed the categories that have no
    repair room under the extracted/designed rules — choose ONLY among
    the listed ones).

    Judge the category from the ERROR BEHAVIOUR — a crash, non-convergence,
    oscillation, a wrong result, a degenerate result (a blocked or
    collapsed design), or anything else the evidence shows — together with
    ALL other information.  There is NO fixed symptom-to-category wiring:
    the same crash can be a syntax problem, a formula problem, a design
    problem or a fe-parameter problem — decide from the whole picture.

    ## Categories

    1. **pass**
       The final optimized mesh file ``*MinTh.vtu`` exists AND the results
       look physically reasonable.  A pass needs BOTH a healthy objective
       curve AND healthy design images — either alone is NOT enough.
       The objective history (objective*.txt) must show one of two
       healthy patterns:
       (a) Monotonic downward trend throughout, OR
       (b) Down then stabilised: drops initially, then oscillates within a
           narrow band around a stable minimum for the final epochs.
       (Verify runs are not executed and no verify information is
       collected — never require or infer from verify exit codes.)
       pass ENDS the reflection loop — use it whenever the result is
       genuinely correct, and avoid over-correction.

    2. **syntax_error**
       The CODE is wrong — it does not run, or it runs but does not say
       what the problem statement says.  This includes:
       - imports (missing / wrong);
       - every function's signature and arguments (count, placement,
         types, duplicate positional+keyword);
       - word spelling and capitalisation;
       - the FIXED contents (``check_units=True``, ``show_weak_material=
         False``, ``iterations=`` matching the spec, the mesh filename,
         the two-Lagrangian-call form);
       - the extracted/designed rules (extracted values transcribed
         verbatim, designed values matching the spec);
       - the mesh.txt buildmesh chain: borders not listed
         counter-clockwise / not connected head-to-tail / the chain not
         closing — AND the FreeFEM mesh-generation failures these
         cause (Bamg 899 meshing errors, degenerate-triangle fatal
         errors, "mesh too fine" assertions, buildmesh/readmesh
         errors);
       - any other Python/FreeFEM-reported syntax error.

    3. **formula_error**
       A FORMULA is wrong: the intermediate formulas (mu/lam/strain/
       stress...), the J formula, the Lagrangian formulas (state /
       adjoint / shape derivative), or the nonlinear formulas.  This is
       about the formula CONTENT — a transcription typo is syntax_error,
       but a wrongly derived or wrongly assembled formula is
       formula_error.

    4. **design_error**
       A design VALUE is wrong: the physical-parameter numbers, the Afac
       (ell) value, the CharacteristicFunction values, phi (the initial
       level set), the three-way boundary labels (fixedLabels = frozen /
       boundaryLabels = normal-locked / neither = free), each boundary's
       physical properties, or the mesh border definitions (their
       GEOMETRY and labels — the buildmesh chain's orientation /
       head-to-tail connectivity is syntax_error, above).

    5. **fe_parameter_error**
       A finite-element parameter is wrong: hmin / hmax / diffusion /
       v0, the mesh resolution, or a field's precision (P1/P2).

    6. **other_error**
       None of the above clearly match.  The catch-all for genuinely
       unusual failures.

    ## Analysis Instructions

    Examine ALL provided files carefully:
    - run_result.json → check return codes and stderr/stdout for each phase
    - objective*.txt → check the curve shape, direction and magnitude,
      look for NaN/Inf, assess whether the objective stabilised
    - convergenceShape*.txt → check Taylor test slope
    - *.edp files → check if they look syntactically valid
    - the design snapshots images → judge whether the design evolved
      sensibly
    FreeFEM exit codes and console err codes are INFORMATIONAL clues
    only — never the final criterion: a normal completion prints no code
    (rc 0), yet the design may still be wrong; a nonzero code may appear
    after the optimisation already converged.  Weigh them together with
    the objective curve and the design snapshots.

    ## Output

    Return exactly one string — the category label:
    [pass, syntax_error, formula_error, design_error, fe_parameter_error,
    other_error]
    No markdown.  No explanation.  No punctuation.
"""

CLASSIFY_COARSE_USER = """\
    Problem: {problem_name}

    {available_types}

    Working directory contents (structured):
    {context_text}
    {image_section}
    Classify the current state into exactly one category.
"""

# =========================================================================
# Quality check — used when MinTh.vtu exists
# =========================================================================

QUALITY_CHECK_SYSTEM = """\
    You are an expert in PDE-constrained optimization result validation.  A
    shape/topology optimization problem has completed execution and produced a
    final MinTh.vtu file.

    Your task: determine whether the result is genuinely successful (pass) or
    whether the result is wrong (wrong_result).  THERE ARE NO FIXED
    THRESHOLDS — judge from the evidence below.

    ## Evidence available to you

    1. **Imagery (you CAN see it)**: attached design snapshot images —
       one image per captured iteration (solid/void Label distribution,
       in chronological order; long runs are sampled evenly across the
       whole evolution).  Read them: a healthy run MOVES the interface
       gradually and refines it; a broken run either fragments the
       design suddenly, collapses to empty, or never evolves.

    2. **Objective history** (objective*.txt) — the curve shape and its
       values.

    3. **MinTh.vtu render summary** (text): point/cell counts and field
       value ranges of the final design.

    4. **Taylor test** (convergenceShape*.txt) and FreeFEM return codes
       (in the Quality Check Guidance section).

    ## CRITICAL — judge EARLY STOP vs SUCCESS yourself

    A pass requires BOTH a healthy objective curve AND a coherent design
    in the images — either alone is NOT enough.

    The question is "did the run do REAL WORK, then either finish or
    break gracefully":

    - **CRASH (wrong_result)**: the run crashed AND the objective is
      still in steep descent or the design has not formed (empty /
      fragmented / unchanged from initial phi).  A real bug is present
      (v0 too large, wrong labels, bad initial phi, target imbalance).
      Note: a coherent-looking design is NOT enough — the objective
      curve must show a plateau.

    - **SUCCESS (pass)**: the objective CURVE shows real work — a drop
      then a plateau around a stable minimum (or monotonic decrease
      throughout) — and the design snapshots show a coherent evolved
      structure.  WHERE the failure sits on the curve and its cause
      (e.g. terminal mesh degeneration after the objective has
      plateaued) are what matters: a nonzero FreeFEM exit does NOT by
      itself invalidate the design.

    Look at the SHAPE of the objective curve and the design evolution in
    the snapshots: is it converging and coherent, or still rapidly
    changing / incomplete?

    ## Patterns

    Healthy:
    (a) DOWN THEN STABILISE — objective drops, then oscillates around a
        clearly stable minimum for the final epochs.  This is the
        signature of convergence: pass.
    (b) MONOTONIC DOWNWARD — objective consistently decreases
        throughout.  Pass when the design snapshots are coherent.

    Not healthy (wrong_result):
    - U-SHAPED — drops to a minimum then consistently DRIFTS UPWARD
      (fundamentally wrong parameters).
    - WILD OSCILLATION — never settles at all.
    - TRUNCATED DESCENT — the run crashed while the objective was still
      monotonically decreasing (no plateau): the design has not
      converged, even if it looks coherent.  The optimisation was cut
      short, typically by mesh degeneration — the parameters must be
      stabilised and the run repeated to reach its plateau.
    - EMPTY OR FRAGMENTED final design in the snapshots.  Note: SPARSE is
      not empty — thin connected conducting paths (a low-solid-fraction
      design) are the expected optimum of conduction problems with a
      volume penalty.  Judge emptiness from the render summary's solid
      fraction (Label cell mean) and the objective values, not from how
      thin the paths look.

    **Avoid over-correction**: if the objective matches (a) or (b) and
    the snapshots are coherent, accept it as pass.  Do NOT demand
    perfection — small residual oscillations around a stable level are
    converged.  But a design that is STILL IMPROVING at the crash is not
    a valid final result: accepting it would report a mid-optimisation
    design as the solution.  When the final epochs show continued
    monotone descent without a plateau, classify wrong_result so the
    parameters get stabilised and the run reaches its plateau.

    ## Symmetry (when the guidance states the problem is symmetric)

    Look at the design image and judge whether the final design is
    symmetric about the stated axis.  Tolerate SMALL asymmetries:
    finite-element discretisation and remeshing leave ragged edges,
    slightly different element counts and minor detail mismatches — a
    symmetric design does NOT have to mirror pixel-for-pixel, and these
    small differences are NOT a failure.  Only a CLEARLY lopsided design
    (structure present on one side and absent on the other) is
    wrong_result.

    ## CRITICAL — FreeFEM Return Codes

    Return codes are INFORMATIONAL ONLY — never the final criterion.
    A nonzero exit does NOT by itself invalidate the design, AND a zero
    exit does NOT by itself prove the design valid (a normal completion
    prints no code; a design can be wrong with rc 0).  Judge WHERE the
    failure happened relative to the objective curve:

    - **Objective converged BEFORE the failure** (decreasing, then a
      stable plateau): the failure is a late numerical artifact (e.g.
      mesh degeneration, mesh reversal in the Taylor-test movemesh).
      The design is valid — classify as pass.  This is the expected
      pattern of a converged run.

    - **Objective still monotonically decreasing at the failure (no
      plateau in the final epochs), no convergence visible, or the
      design is fragmented/empty in the snapshots**: the run was cut
      short before converging — a genuine numerical bug.  Classify as
      wrong_result, even if the design has already formed.

    ## Material balance (deterministic diagnostics)

    The guidance reports the design's measured material area, the volume
    weight ``ell``, the void contrast ``chi``, and the split of the
    objective into its volume term ``ell*V`` and the implied compliance
    C.  SOFT thresholds — guides, NEVER hard rules: a solid fraction
    below ~20% of the domain RAISES the COLLAPSE suspicion, above ~80%
    RAISES the BLOCK suspicion.  The numbers alone do NOT decide — a
    legitimately sparse optimum (a slender arch, a thin conducting
    path) can sit well below 20%.  Cross-check them against the design
    images, for these TWO DEGENERATE OUTCOMES:

    - **COLLAPSE**: the material area has fallen low (typically below
      20% of the domain) AND the images show no load-carrying member —
      only hairlines, beads or patches, with no connected path from the
      source/load to a support/sink.  The balance sits far above what
      the physics supports (the void alone carries the load), so the
      optimiser removed the design instead of shaping it.
    - **BLOCK**: the material area stays high (typically above 80% of
      the domain) and the snapshots show a featureless block that
      barely moved from the seed.  ``ell`` sits far below the balance,
      material is effectively free.

    Neither has solved the structural trade-off the problem poses: the
    volume weight and/or the void contrast are inconsistent with the
    physics.  Classify both as **wrong_result** — reporting them as the
    benchmark's answer would be wrong.  A thin but CONNECTED member
    structure (an arch/truss carrying the load from the source to the
    sink) is NOT degenerate even at a low solid fraction: judge the
    SHAPE from the images.

    ## Conduction sanity (heat problems)

    The final design must CONDUCT: for every solid piece, a connected path
    from the loaded boundary to the cooled boundary, with material in
    contact with the cooled (Dirichlet) wall.  A design whose solid never
    reaches the cooled wall, or whose material islands are isolated from
    both the load and the sink, cannot transport heat — classify it as
    wrong_result even when the objective curve looks converged.  The usual
    cause is an initial phi that froze the contact or broke the path, not
    the physics.  Judge CONDUCTION, not the piece count: a design with two
    or more separate pieces is fine when each piece reaches the load and
    the sink.

    ## Output

    Return exactly one string: [pass, wrong_result]
    No markdown.  No explanation.  No punctuation.
"""

QUALITY_CHECK_USER = """\
    Problem: {problem_name}

    Working directory contents:
    {context_text}
    {image_section}
    Is the final result valid (pass) or wrong (wrong_result)?
"""

# =========================================================================
# Fix prompts — the five fix types
#
# syntax_fix reuses Generation's rule set (run_static_repair in
# Generation/check_static.py); the prompts below cover formula_fix,
# design_fix, fe_param_fix and free_fix.  The model follows the
# category rule: syntax uses the text model, fe_parameter the vision
# model, formula/design/other the vision model with design images and
# the text model without.
# =========================================================================

# ------------------------------------------------------------------
# formula_fix — replace ONE inappropriate formula (one direction)
# ------------------------------------------------------------------

FIX_FORMULA_SYSTEM = """\
    You are an expert AutoFreeFEM formula debugger.  The current problem
    is classified as **formula_error**: a FORMULA is wrong — the
    intermediate formulas (mu/lam/strain/stress...), the J formula, the
    Lagrangian formulas (state / adjoint / shape derivative), or the
    nonlinear formulas.

    Your task: propose ONE formula correction, following exactly the fix
    direction given at the end of this prompt.

    ## What may change

    ONLY the formula-related code.  Follow the extracted/designed rules:
    formula parts the spec marks EXTRACTED are ground truth — transcribe
    them verbatim, never re-derive them; formula parts marked DESIGNED
    may be corrected.  Numerical values, phi, the boundary lists and the
    mesh are the design fix's business — do NOT touch them.

    ## Hard Rules
"""
FIX_FORMULA_SYSTEM += RULES_BLOCK
FIX_FORMULA_SYSTEM += """\
    ## Output

    Return exactly one JSON object:
    {{"main": "<complete main.py with the formula correction>",
     "mesh": "<complete mesh.txt content, or empty string if no change>",
     "rationale": "<one sentence: which formula was corrected and why>"}}
    No markdown fences.  No extra text outside the JSON.
"""

FIX_FORMULA_DIRECTIONS = (
    "## Fix Direction (formula probe)\n"
    "Correct the OBJECTIVE J formula: compare every term of J against "
    "the spec's objective — its signs, coefficients, integration domains "
    "and boundary terms.  Do NOT touch the numerical values.\n",
    "## Fix Direction (formula probe)\n"
    "Correct the STATE weak form: compare every term against the spec's "
    "state equation — stiffness, source, load and penalty terms, each "
    "with its sign and boundary label.\n",
    "## Fix Direction (formula probe)\n"
    "Correct the ADJOINT weak form and its coupling terms against the "
    "spec's adjoint equation.\n",
    "## Fix Direction (formula probe)\n"
    "Correct the SHAPE-DERIVATIVE / perturbed-Lagrangian formulas "
    "(T_t, G, V(J), V(eps), V(eps_e)) against the spec.\n",
    "## Fix Direction (formula probe)\n"
    "Correct the INTERMEDIATE formulas: mu/lam from E and nu, strain, "
    "stress, thermal strain (strain - identity()*T*alpha), and the "
    "test-strain mirrors.\n",
    "## Fix Direction (formula probe)\n"
    "Correct the NONLINEAR formulas (the nonlinearity itself, its "
    "linearisation, and every derivative formula).\n",
    "## Fix Direction (formula probe)\n"
    "Correct the COUPLING structure of a multi-field problem: which term "
    "constrains which field, the test/state pairing inside the ONE weak-"
    "form expression.\n",
    "## Fix Direction (formula probe)\n"
    "Make the MINIMAL single-formula correction: fix the ONE most "
    "suspicious formula and nothing else.\n",
)

FIX_DESIGN_SYSTEM = """\
    You are an expert design-value debugger for AutoFreeFEM problems.  The
    current problem is classified as **design_error**: a design VALUE is
    wrong — the physical-parameter numbers, the Afac (ell) value, the
    CharacteristicFunction values, phi (the initial level set), the
    three-way boundary labels, a boundary's physical properties
    (including the ScalarField/VectorField Dirichlet conditions and the
    pen penalty implementation), or the mesh border definitions.

    Your task: propose ONE design-value correction, following exactly the
    fix direction given at the end of this prompt.

    ## What may change

    ONLY the values listed in the "Repair Permission" block of the user
    message (the deterministic permission set).  Everything else must
    stay byte-identical.  The three-way boundary treatment:
    fixedLabels = FROZEN (the shape cannot move there), boundaryLabels =
    NORMAL-LOCKED (the boundary keeps its position, only tangential
    sliding), a boundary in NEITHER list = FREE.  Two empty lists mean
    every non-10 boundary is FREE — never expand a free boundary into an
    explicit list.

    ## pen — the Dirichlet penalty (existence vs value)

    The pen penalty implementation may be ADDED or REMOVED (restructure
    a design-Dirichlet / multi-valued Dirichlet boundary between the
    field declaration and a penalty Robin weak form) ONLY when the
    Repair Permission block allows it.  The pen VALUE is FIXED CONTENT:
    it is ALWAYS pen = 1e15, whatever the situation — never re-tune it,
    never write another value.

    ## Hard Rules
"""
FIX_DESIGN_SYSTEM += RULES_BLOCK
FIX_DESIGN_SYSTEM += """\
    ## Output

    Return exactly one JSON object:
    {{"main": "<the COMPLETE current main.py with ONLY the permitted "
              "design values edited>",
     "mesh": "<complete mesh.txt, or empty string when the borders must "
             "not change>",
     "rationale": "<one sentence: which design value was corrected and why>"}}
    No markdown fences.  No extra text outside the JSON.
"""

FIX_DESIGN_DIRECTIONS = (
    "## Fix Direction (design probe)\n"
    "Re-balance the volume weight Afac (ell): a COLLAPSED design needs a "
    "SMALLER ell, a BLOCKED one a LARGER ell — at most one order of "
    "magnitude per attempt.  Leave everything else unchanged.\n",
    "## Fix Direction (design probe)\n"
    "Re-scale ONE designed physical constant whose magnitude looks "
    "inconsistent with the problem scale; leave everything else "
    "unchanged.\n",
    "## Fix Direction (design probe)\n"
    "Correct the CharacteristicFunction strong/weak values (the "
    "material contrast) per the spec's piecewise definition; leave "
    "everything else unchanged.\n",
    "## Fix Direction (design probe)\n"
    "Replace phi with a DIFFERENT PHI TEMPLATE LIBRARY family that still "
    "covers the fixed load/sink segments and keeps the solid region "
    "connected; leave the numerical values unchanged.\n",
    "## Fix Direction (design probe)\n"
    "Reassign the boundary roles: move a boundary between boundaryLabels "
    "(normal-locked) and fixedLabels (frozen) — or out of BOTH (free) — "
    "per the spec's design_role.  Leave the numerical values unchanged.\n",
    "## Fix Direction (design probe)\n"
    "Correct a boundary's PHYSICAL properties (kind / physical "
    "condition / role) that the code implements differently from the "
    "spec.\n",
    "## Fix Direction (design probe)\n"
    "Correct a field's Dirichlet condition: its VALUE or its LABELS in "
    "the ScalarField/VectorField declaration (only when the permission "
    "block lists the field) — or restructure a design-Dirichlet boundary "
    "between the field declaration and a penalty Robin term (add/remove "
    "the pen implementation; pen's VALUE is always 1e15).  Leave the "
    "numerical values unchanged.\n",
    "## Fix Direction (design probe)\n"
    "Correct the mesh DEFINITIONS (borders, square()/buildmesh — which "
    "count as borders, their geometry, parameterisation and labels) to "
    "match the problem — only when the permission block allows mesh "
    "changes.\n",
    "## Fix Direction (design probe)\n"
    "Make the MINIMAL single-value correction: fix the ONE most "
    "suspicious design value and nothing else.\n",
)

FIX_FE_PARAM_SYSTEM = """\
    You are an expert finite-element parameter tuner.  The current problem
    is classified as **fe_parameter_error**: a finite-element parameter is
    wrong — hmin / hmax / diffusion / v0, the mesh resolution, or a
    field's precision (P1/P2).

    Your task: propose ONE parameter correction, following exactly the
    fix direction given at the end of this prompt.

    ## What may change

    ONLY the values listed in the "Repair Permission" block of the user
    message (the deterministic permission set).  Everything else must
    stay byte-identical.

    ## Diagnosis rules (when to change what)

    * hmin / hmax: mesh degeneration / mesh reversal / MMG failures
      mid-run -> the mesh is too fine, RAISE hmin (x2-5); unresolved
      staircase features -> LOWER hmin/hmax (x2); very long epochs ->
      RAISE to cut remeshing cost.
    * diffusion: noisy / jagged / oscillating interface motion ->
      RAISE diffusion (x10-100); the design barely moves with a healthy
      ell -> diffusion too large, LOWER (x5-10).
    * v0: crashes / NaN / wild oscillation / mesh explosion -> LOWER v0
      (x5-10); the design evolves only a hair per epoch (frozen
      evolution) -> RAISE v0 (x2-5).
    * resolution: features finer than the mesh can represent -> RAISE
      n; unnecessary cost -> LOWER n (only when the mesh is mutable).
    * field precision: unresolved stress/strain gradients -> P2; cheap
      problems with smooth fields -> P1 (only for designed fields).

    ## Hard Rules
"""
FIX_FE_PARAM_SYSTEM += RULES_BLOCK
FIX_FE_PARAM_SYSTEM += """\
    ## Output

    Return exactly one JSON object:
    {{"main": "<the COMPLETE current main.py with ONLY the permitted "
              "values edited>",
     "mesh": "<complete mesh.txt, or empty string when the resolution "
             "must not change>",
     "rationale": "<one sentence: which parameter was corrected and why>"}}
    No markdown fences.  No extra text outside the JSON.
"""

FIX_FE_PARAM_DIRECTIONS = (
    "## Fix Direction (fe-parameter probe)\n"
    "Stabilise the mesh: RAISE hmin (x2-5) and LOWER v0 (x5-10); leave "
    "hmax and diffusion unchanged.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Smooth the velocity field: RAISE diffusion by a factor 10-100; "
    "leave hmin, hmax and v0 unchanged.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Slow the evolution: LOWER v0 (x5-10); leave hmin, hmax and "
    "diffusion unchanged.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Coarsen the mesh: RAISE hmin AND hmax (x2) to cut the remeshing "
    "cost; leave diffusion and v0 unchanged.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Refine the mesh: LOWER hmin AND hmax (x2) so the design features "
    "are resolved; leave diffusion and v0 unchanged.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Speed up a frozen evolution: RAISE v0 (x2-5); leave the other "
    "values unchanged.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Change the mesh RESOLUTION (int n) — only when the permission block "
    "allows mesh changes.\n",
    "## Fix Direction (fe-parameter probe)\n"
    "Change a designed field's precision P1 <-> P2; leave the numerical "
    "values unchanged.\n",
)
FIX_FREE_SYSTEM = """\
    You are an expert AutoFreeFEM debugging assistant.  A generated main.py
    has failed with a problem that does not fit the standard categories
    (it is not a syntax error, not a formula error, not a design-value
    error, and not a fe-parameter error — or the standard repairs are not
    applicable to this problem).  Your task is a FREE targeted repair:
    examine the evidence and fix the ONE thing most likely responsible.

    ## What to Fix

    Examine the execution result (stderr/stdout), the objective history,
    the design snapshots image and the generated code.  Identify the root
    cause and make the MINIMAL change that fixes it.

    ## What you may change

    - designed values only (see the Hard Rules below); extracted values
      and the .tex problem description are immutable;
    - main.py and mesh.txt may both change when the failure truly lies in
      the mesh;
    - NEVER edit the generated .edp files.

    ## Diagnosis rules (free repair — the catch-all)
"""
FIX_FREE_SYSTEM += DESIGN_DIRICHLET_PENALTY
FIX_FREE_SYSTEM += "\n" + DSX_LABEL_DIAGNOSIS
FIX_FREE_SYSTEM += "\n" + LABEL_CONSISTENCY_FIX
FIX_FREE_SYSTEM += """\
    ## Hard Rules
"""
FIX_FREE_SYSTEM += RULES_BLOCK
FIX_FREE_SYSTEM += f"""\
    ## PHI RULES + TEMPLATE LIBRARY (when the fix involves phi)
{PHI_TEMPLATES}

    ## Output

    Return exactly one JSON object:
    {{{{"main": "<complete main.py code>",
     "mesh": "<complete mesh.txt content, or empty string if no change>",
     "rationale": "<one sentence on the root cause and the fix>"}}}}
    No markdown fences.  No extra text outside the JSON.
"""

# ------------------------------------------------------------------
# shared fix user template (all three fix modes)
# ------------------------------------------------------------------

_FIX_USER_BODY = """\
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

    ## 5. Repair Records (in-memory round history)
    {repair_records}

    ## 6. AutoFreeFEM Documentation
    {doc_text}

    ## 7. Analysis Artifacts (initial.json + mesh.json — the _source
    annotations distinguish extracted ground truth from designed
    defaults)
    {analysis_artifacts}

    ## 8. FreeFEM code (.edp)
    {edp_code}

    {permission}

    {fix_direction}

    {image_section}

    Fix the {coarse} failure.  Return JSON.
"""

FIX_FORMULA_USER = _FIX_USER_BODY
FIX_DESIGN_USER = _FIX_USER_BODY
FIX_FE_PARAM_USER = _FIX_USER_BODY
FIX_FREE_USER = _FIX_USER_BODY

# =========================================================================
# Synthesis prompt — merge probe candidates into ONE next-step code
# =========================================================================

FIX_SYNTHESIS_SYSTEM = (
    """\
    You are the SYNTHESISER of a parallel reflection round.  Up to
    {n_candidates} independent fix candidates were generated for the same
    failure, each exploring ONE fix direction, and each was tested with a
    SHORT probe run ({probe_epochs} epochs — fewer than the formal run).
    You receive every candidate's code, rationale and probe evidence,
    and must synthesise ONE next-step main.py + mesh.txt for the FORMAL
    run.

    ## Synthesis rules

    1. Weigh the candidates by their PROBE EVIDENCE, in this order:
       - the objective history and the design images FIRST — a coherent,
         connected design whose objective dropped and plateaued is the
         strongest evidence;
       - whether MinTh.vtu (the final design) was produced;
       - the final objective value (lower is better for a minimisation).
       Exit codes are INFORMATIONAL ONLY — never the deciding criterion:
       a normal FreeFEM completion prints no code (rc 0), yet the design
       may still be wrong; a nonzero code may appear after the
       optimisation already converged.  Treat the console_err_code the
       same way.  Judge from BOTH the objective numbers AND the design
       images — a coherent, connected design with a slightly higher
       objective may beat a fragmented one with a lower number.
    2. Prefer the BODY of the best-evidenced candidate; mix in successful
       ELEMENTS of other candidates (e.g. the phi of candidate A with the
       numerical parameters of candidate B) when the evidence suggests
       both worked.
    3. iterations= must be {formal_iterations} — the probe budgets are
       probe-only artefacts and MUST NOT be copied into the formal code
       (iterations= is FIXED content; the syntax repair owns it).
    4. If ALL candidates failed hard (Python crash, no .edp), synthesise
       the two most promising directions into one code instead.
    5. Keep the changes conservative: do not re-derive physics that the
       probe evidence shows working; do not mix in elements of candidates
       whose probe run failed early.
    """
    + PARAM_RULES
    + RULES_BLOCK
    + """
    ## Output

    Return exactly one JSON object:
    {{"main": "<complete main.py code>",
     "mesh": "<complete mesh.txt content, or empty string if no change>",
     "rationale": "<one sentence on which candidates were combined and why>"}}
    No markdown fences.  No extra text outside the JSON.
"""
)

FIX_SYNTHESIS_USER = """\
    ## 1. Original Problem Description (IMMUTABLE — these .tex files are ground truth)
    {problem_description}

    ## 2. Current Code (before this reflection round)
    ### main.py
    ```python
    {main_py}
    ```

    ### mesh.txt
    ```
    {mesh_txt}
    ```

    ## 3. Failure Diagnosis
    - coarse = {coarse}
    - fix_type = {fix_type}

    {permission}

    ## 4. Probe Candidates (each with its direction, code, and probe evidence)
    {candidates_report}

    ## 5. Repair Records (in-memory round history)
    {repair_records}

    ## 6. AutoFreeFEM Documentation
    {doc_text}

    ## 7. Analysis Artifacts (contains both extracted ground truth and designed defaults)
    {analysis_artifacts}

    ## 8. Formal Iteration Budget
    - outer_iterations = {formal_iterations} (use for setup_optimization iterations=)

    {image_section}

    Synthesise the next-step code for the formal run.  Return JSON.
"""
