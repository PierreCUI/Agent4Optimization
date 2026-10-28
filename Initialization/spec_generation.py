"""
One-loop spec generation: GENERATE -> REVIEW -> REPAIR.

Each iteration produces BOTH spec JSONs together (initial.json's content
and mesh.json's content in one LLM answer), runs the deterministic syntax
checks (generated content only), then the merged LLM review (extraction
fidelity + design rules + physical plausibility + information omissions —
all judged in ONE review pass) — the review runs EVERY round, and its
errors MERGE with the deterministic issues so a substantive defect is
never hidden behind a persistent syntax stall.  On issues the
loop REPAIRS the previous output (fix only the listed issues, keep
everything else verbatim) instead of regenerating from scratch.  When an
output_dir is given, the two JSONs are written to disk after every
generation and repair round; the loop stops when it passes or after
``configs.MAX_SPEC_FIXES`` repairs.  A no-op repair round (an empty diff
with unchanged issues) does NOT consume budget, and two consecutive
no-ops terminate; a budget-exhausted, review-failed or no-op-terminated
loop returns ``_passed`` False — the caller must abort the case.
"""

import json
import re
import sys
import warnings
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import MAX_FILE_CHARS, MAX_SPEC_FIXES  # noqa: E402
from history import record as record_history, set_label  # noqa: E402
from utils import parser_json  # noqa: E402
from Initialization.phi_rules import PHI_GUIDANCE  # noqa: E402
from Initialization.spec_checks import (  # noqa: E402
    _as_bool,
    _mesh_box,
    _phi_connectivity_issues,
    check_designed_mesh,
    check_designed_problem,
    check_extracted_mesh,
    check_extracted_problem,
    files_block,
    format_errors,
    hard_errors,
    mesh_syntax_problems,
    normalize_designed,
    normalize_designed_mesh,
    normalize_extracted,
    normalize_extracted_mesh,
)

# ======================================================================
# GENERATE — both specs in one answer
# ======================================================================

# The canonical unit syntax — injected into BOTH the generate and the
# review prompts (the review prompt's unit bullet must stay textually
# in sync with this table; see the note there).
_UNIT_RULES = """\
## UNIT RULES — the canonical AutoFreeFEM unit syntax (shared by the
## generator and the reviewer — they must never diverge)
- Base names are CAPITALIZED English: Newton, Watt, Pascal, meter, kelvin.
  'kelvin' is LOWERCASE in every compound — 'Kelvin' is never used.
- Compound units use '/' and '**': Newton/meter**2, Watt/(meter*kelvin),
  Watt*kelvin/meter**3.  '^' is NEVER an exponent operator.
- Singular forms only: meter, never 'metres' or 'm'.
- Dimensionless quantities carry the unit '-'.
- A PRESERVED extracted unit stays EXACTLY as the source writes it (LaTeX
  fractions included) with unit_source "extracted" — the code generator
  converts notation, never the spec.
"""

_GENERATE_SYSTEM = ("""\
You are the problem-analysis stage of a topology-optimization pipeline.
You receive the original problem input files (LaTeX .tex, parameter
tables, mesh file) and must produce TWO specification JSONs in ONE
answer: the problem spec (items 1-8 and 10, stored in initial.json) and
the mesh spec (item 11, stored in mesh.json).  Each spec has an
"extracted" part (VERBATIM transcription of what the sources state —
immutable ground truth) and a "designed" part (the completed spec).

## Output shape (the ONLY accepted shape)

{{
  "initial": {{
    "extracted": {{ ... }},
    "designed": {{ ... }}
  }},
  "mesh": {{
    "extracted": {{ ... }},
    "designed": {{ ... }}
  }}
}}

Return ONLY this JSON.  No markdown, no code fences.

## Annotation requirement (MANDATORY for both specs)
- Every scalar carries a "<key>_source": "extracted"|"designed" sibling;
  every array item carries "source" inside the item; every designed value
  carries a "rationale".  "extracted" means VERBATIM from the sources —
  anything you change, expand or reconcile must be marked "designed".

## EXTRACTED rules (transcribe ONLY what the sources state; nothing
invented, guessed, or completed; omit keys the sources do not state,
except the required classifications)

### initial.extracted — items 1-8 and 10
1. Basic: problem_type ("shape_optimization"|"topology_optimization",
   REQUIRED classification), problem_name, dimension 2, is_nonlinear
   (true/false, REQUIRED classification from the physics).
2. Domain: domain_name, domain_description (only if stated); boundaries
   as LOGICAL boundaries (Gamma names, NOT the C01/C02 segments): name,
   definition, kind (Dirichlet|Neumann|Robin — physical condition only),
   design_role (ONLY if explicit), mesh_label (the boundary's label
   NUMBER — only if stated); domains/interfaces (multi-domain only);
   mesh_labels_used (all labels in the mesh file, including FreeFEM's
   implicit square() labels 1=bottom, 2=right, 3=top, 4=left) — a
   PLAIN list of label strings like ["1", "3"], never annotated
   objects.
3. Fields: name, latex, type (scalar|vector), fe_space (P1|P2), domain,
   unit, dirichlet_value, dirichlet_labels — each only when stated.
   An ADJOINT state defined by the source (e.g. u_tilde in V_delta_u)
   MUST be captured as a field WITH its Dirichlet condition (value 0 and
   the boundary labels).  Test functions: name, fe_space, domain ONLY
   (no latex, no type, no unit, no dirichlet values).
4. Constants: name, latex, value, description, unit — only with an
   explicit numerical value.  ell/Afac is an ORDINARY constant: if the
   source gives its value, transcribe it like any other constant.
5. Characteristic function: latex, strong_value, weak_value — transcribe
   EXACTLY as the source defines chi (e.g. chi = 1 if phi < 0 and
   chi = 0 otherwise means strong_value "1" and weak_value "0" — never
   substitute a physics default for a value the source states).  Also
   transcribe the source's piecewise CONDITION itself as a `condition`
   string (e.g. "chi = 1 if phi < 0 and chi = 1/1000 if phi >= 0") —
   the phi-sign -> strong/weak mapping is RECORDED, never implied.
6. Expressions with explicit formulas (auxiliary formulas and J-related).
   Include EVERY J-related derivative formula the source gives (partial
   derivatives of J, material-time derivatives) and every material
   interpolation (e.g. k(x), E(x), alpha_T(x) as DISTINCT formulas —
   never repeat one formula under several names).
7. Lagrangian: objective; state_equation (description, weak_form_lhs,
   weak_form_rhs) — transcribe EXACTLY: never add terms (in particular
   never insert a penalty term unless the source writes one), and FLIP
   the sign of any term moved across the equals sign; newton /
   adjoint_equation / shape_derivative only if explicitly given.
8. optimization_params: hmin, hmax, diffusion, v0, iterations, phi,
   boundary_labels, fixed_labels, show_weak_material — only stated ones.
10. extra_requirements (optional): items with a requirement string and a
   source annotation, for other run-affecting requirements the sources
   state.

### mesh.extracted — item 11
category ("code"|"description"|"absent"), mesh_code (VERBATIM), edges
(every border segment: name, param, x, y, label, role — role is the
PHYSICAL function from comments/explicit statements only, e.g. "loaded
segment", "clamped edge", "symmetry edge"; NEVER classify design vs fixed
and never derive a role from context; write the border name when nothing
is stated), n (only if stated), mesh_file (only if stated), labels_found,
has_syntax_errors, syntax_error_details, source "extracted".
mesh_file is the PIPELINE'S INPUT mesh file name — always "mesh.txt".
A ``savemesh(Th, "...")`` statement writes an OUTPUT / INTERMEDIATE mesh
file (e.g. "Mesh1.mesh") — NEVER extract that name as mesh_file.
Intermediate output files receive NO checks and no extraction.
COMMENTED-OUT borders (// ...) are part of the file: transcribe them, but
never treat them as the active mesh.

## DESIGNED rules (complete the missing pieces)

### Design priority: the problem's own statements first, then general
physical experience — NO canned defaults except: iterations=100,
show_weak_material=false, check_units=true.

### initial.designed (flat format; every scalar with <key>_source)
1. problem_type / problem_name / dimension / state_is_nonlinear (from
   is_nonlinear) — preserve extracted values.
2. domain_name, domain_description (design if missing).  Boundaries:
   every non-10 boundary needs name, definition, kind, design_role,
   label, source [, rationale].  Reconcile LaTeX names with
   mesh_labels_used by geometric reasoning; preserve extracted kinds and
   explicit design_roles (a load boundary whose position is prescribed
   is FIXED even if Neumann); convention without an explicit role:
   Dirichlet -> fixed, Neumann/Robin -> design, unsure -> fixed.  The
   definitions MUST match the extracted weak form: the boundary the weak
   form loads is LOADED, the Dirichlet one is CLAMPED, an unloaded
   Neumann one is FREE; if the load term is over the WHOLE boundary
   (dD), EVERY external boundary is loaded — never narrow it.  Label 10
   is the material boundary: when the weak form ACTS on it (e.g. a
   spring term on the hole), ADD a boundary entry with label "10"
   describing the treatment (it still NEVER enters boundary_labels /
   fixed_labels and carries no design role); label 0 is a no-effect
   placeholder (all unlabelled boundaries — usually none), never a real
   design/fixed boundary.  Multi-domain: domains + interfaces with
   physical (value_continuity/flux_continuity) and non-10 labels.
   Boundaries referenced BY NAME in the weak form or BCs carry
   explicit names; boundaries used only by their label number need
   no name.
   Convention: Dirichlet boundaries are normally fixed
   (fixed_labels); Neumann / Robin normally design
   (boundary_labels); when unsure, put the label in
   fixed_labels.
3. Fields: fill type/fe_space/unit/domain by physics (P2 vector
   displacement in meter, P1 scalar temperature in kelvin);
   dirichlet_labels must be the boundary labels; a clamped field needs
   its dirichlet labels and value.  Test functions carry ONLY
   name/fe_space/domain — fe_space matching the physical field it
   tests.
4. Constants: preserve extracted; design physically plausible material
   values; units DESIGNED by you follow the canonical unit syntax
   (UNIT RULES below); a PRESERVED
   extracted unit stays EXACTLY as the source writes it (LaTeX
   fractions included) and keeps unit_source "extracted" — the code
   generator converts notation, never the spec; descriptions must match
   the standard meaning of the symbol
   (in elasticity lambda/mu are the LAME parameters, E is Young's
   modulus, nu Poisson's ratio — correct a wrong extracted description,
   marking it designed).  ell/Afac is an ordinary constant: preserve
   the extracted value, or design a reasonable initial value when the
   source gives none (the pipeline calibrates it later).  The
   MATERIAL-PENALTY (beta/ell/Afac) VALUE is NOT validated when its
   source is designed — any value is accepted as-is (no positivity /
   magnitude check; the calibration sweeps decide it later).
5. Characteristic function: strong_value (the larger side) = 1;
   the characteristic function latex carries the FULL decoration
   \\raisebox{{\\depth}}{{$\\chi$}} (material-property fields keep plain
   names);
   weak_value (the smaller side) MUST be given.  PRESERVE the extracted
   weak_value when the source defines one (some sources define chi = 0
   in the void — keep 0); only when the source does not define it, use
   the physics default ("1/100" solid mechanics, "1/10" pure thermal).
   PRESERVE the extracted `condition` (the phi-sign -> strong/weak
   mapping).
6. Expressions: preserve; add standard derived quantities (strain,
   stress, Lame parameters) marked designed.
7. Lagrangian: objective_name/description/terms; state_description /
   weak forms — Dirichlet conditions are carried by the FIELD
   DECLARATION (dirichlet value + labels) as the DEFAULT.  The
   pen penalty terms appear ONLY as a FreeFEM WORKAROUND
   when a field must enforce TWO OR MORE DIFFERENT Dirichlet
   values (FreeFEM cannot express that in one field declaration,
   e.g. u = 0 on one edge and u = 10 on another); in that case
   write the penalty in the state weak form, the adjoint weak
   form AND the shape derivative with the SAME pen, and
   list pen in constants.  pen's VALUE is designed and NOT
   validated (default 1e15 — any designed value is used as-is;
   no range check).  A penalty term for a SINGLE-valued
   Dirichlet is UNNECESSARY (the field declaration already
   enforces it) and must NOT be written.  Neumann loads as
   boundary integrals, Robin as the mixed term; every weak-form term
   keeps the source's sign (flip when moving across the equals sign);
   multi-domain: value/flux continuity at the interfaces; for
   multi-physics problems use a SEPARATE penalty constant per equation
   in that equation's units.  check_units: true (default; false only
   with a rationale).  adjoint designed from the Lagrangian when
   missing.
   Formulas must be mathematically EQUIVALENT to the source — the exact
   notation and wording do not matter (a term may be rewritten, renamed
   or simplified as long as the computation is unchanged).  Write every
   formula as an EXPLICIT expression, never as a bare placeholder or
   reference.  Every BOUNDARY integral must carry its CONCRETE mesh
   label(s): a term written over the whole boundary (dD) must be
   resolved to the labels it actually acts on — restricting a load to
   the non-Dirichlet labels (where the test function vanishes) is
   computationally equivalent and must be stated with those labels.
   When a penalty IS used, distinguish the objective weight (e.g.
   beta inside d/dt J) from the Dirichlet penalty: ONLY the
   multi-valued-Dirichlet penalty terms carry pen (default 1e15 —
   the designed value is not validated) — the
   objective weight keeps its own symbol.
8. optimization_config: mesh_file "mesh.txt"; hmin < hmax resolving this
   problem's features and the interface; diffusion and v0 sane for this
   problem's scale; iterations 100; phi (phi < 0 = solid; negative
   somewhere in the domain; solid connects every loaded/supported
   boundary; fixed boundaries covered by solid at iteration 0; initial
   solid fraction ~20-80%; a phi prescribed by the problem is used
   exactly); WHEN THEIR SOURCE IS DESIGNED, hmin, hmax, diffusion, v0
   and phi are NOT validated — no positivity/ordering/resolution/
   fraction/connectivity checks apply; the designed values are used
   as-is (the reviewer must NOT flag them); the boundary LISTS carry
   the DSL's THREE-way boundary
   treatment: fixedLabels = FROZEN (the shape cannot move there at all);
   boundaryLabels = NORMAL-LOCKED (the boundary keeps its position —
   only tangential sliding is possible); a boundary in NEITHER list is
   FREE (fully deformable).  Disjoint lists, label 10 never in either,
   label 0 allowed as placeholder; the KEYS are always present, the
   VALUES may BOTH be empty — empty-empty means EVERY non-10 boundary
   is FREE.  Convention: Dirichlet (physics-fixed) boundaries and
   load/support edges whose position is prescribed → fixedLabels (or
   boundaryLabels when the problem only pins their position); other
   outer edges → FREE (or boundaryLabels when the outer box must stay
   put);
   show_weak_material false.
10. extra_requirements: preserve extracted ones; add only clearly
    implied ones (designed + rationale); [] when none.

### mesh.designed (item 11)
mesh_code (runnable FreeFEM producing Th), mesh_code_source
(extracted | designed), mesh_code_rationale, edges (every border of
the FINAL mesh
with name, param, x, y, label, role (its function in the optimisation:
design (movable) / fixed (frozen/loaded/clamped) / material boundary),
source [, rationale]), n (+ n_source), mesh_file "mesh.txt" (+
mesh_file_source), labels_used = the non-10 boundary labels (+
labels_used_source), dimension 2 (+ dimension_source),
domain_description (+ domain_description_source).
- Every border carries a label; borders are parameterised
  counter-clockwise; `int n` defined; label 10 only for the material
  boundary (pre-cut holes) — label 10 IS a valid border label value
  for manually-defined holes; interface edges get their own non-10
  labels.  A mesh with pre-cut holes must mark the hole as region 3
  right after the mesh definition: `int[int] re = [0, 3];
  Th = change(Th, reft=re);`
- NO UNDERSCORES in any identifier that enters the FreeFEM code:
  border names, the mesh variable, the problem name, and any other
  name written into mesh_code / main.py must NOT contain underscores
  (FreeFEM identifiers cannot contain them).  Use concatenated
  lowercase or CamelCase instead (bottomleftwall, bottomLeftWall —
  never bottom_left_wall).
- Commented-out borders are NOT part of the final mesh — exclude them
  from the active edges and labels.
- mesh_code contains NO comments (no // lines) — the generated mesh
  carries no comments.
- square() meshes carry the implicit labels 1=bottom, 2=right, 3=top,
  4=left — list those four sides and MAP the problem's boundaries onto
  them.
- PREFER KEEPING THE ACTIVE MESH verbatim (source "extracted") when the
  problem's labels already map onto it.  FALLBACK, ONLY when the
  problem's labels genuinely cannot map: design a border-built mesh on
  the SOURCE'S ACTUAL domain dimensions (from the active mesh line /
  the problem context — NEVER the coordinates of commented-out
  borders), carrying the problem's labels (source "designed",
  rationale).
- PROVIDED mesh.txt (the input files include one): it is IMMUTABLE —
  transcribe it VERBATIM as mesh.extracted.mesh_code AND set
  mesh.designed.mesh_code to the SAME verbatim content with
  mesh_code_source "extracted".  NEVER redesign it, whatever the label
  situation (the pipeline runs ONE syntax check/repair on it separately
  and restores it deterministically).

"""
    + PHI_GUIDANCE
    + _UNIT_RULES
    + """
## Output rules
ONE JSON in the shape above.  English throughout.  Be CONCISE: one short
sentence per rationale — long rationales truncate the output.
""")

_GENERATE_USER = """\
Problem: {problem_name}

Original input files:
{files_block}

Return the complete two-spec JSON (initial + mesh, each with extracted
and designed).
"""

# ======================================================================
# REVIEW — the merged extraction/design/independent review
# ======================================================================

_REVIEW_SYSTEM = """\
You are the reviewer of a topology-optimization pipeline's problem specs.
You receive the ORIGINAL input files and the two generated specs
(initial: items 1-8/10; mesh: item 11).  Review EVERYTHING at once:
transcription fidelity, design rules, information omissions, and
physical plausibility.  Return issues only — the generator repairs them.

## What counts as an issue
- A numeric value that contradicts or does not appear in the sources
  (equivalent notation counts: 3e7 = 3*10^7 = 30000000).
- A name, label, formula or weak form that CONTRADICTS the sources, a
  weak-form term the sources do not contain, or a term whose SIGN
  differs from the source (a term moved across the equals sign flips
  its sign).
- An ADJOINT state defined by the source without its Dirichlet
  conditions captured (e.g. u_tilde = 0 on a boundary must appear in
  the fields or the adjoint description).
- Boundaries stated ONLY in the V-space definitions (e.g.
  u = 0 on Gamma_2, Gamma_4 inside V_u) that are MISSING from the
  extracted boundaries (no name, kind Dirichlet, definition) — the
  V-space definitions ARE source statements and must be captured.
- A source boundary term integrating over the WHOLE boundary dD that
  the design narrows to a subset — and if such a term also acts on the
  material boundary (label 10, e.g. a spring term on the hole), the
  design must state how label 10 is handled.
- Missing J-related derivative formulas the source gives (partial J,
  material-time derivatives, the PERTURBED LAGRANGIAN G and the
  PERTURBATION MAP T_t the source defines) or material
  interpolations repeated under several names instead of distinct
  formulas.
- A phi whose SOLID FRACTION on the designed domain is outside 20-80%
  (estimate it — do not trust the rationale's claim).
- Anything the sources state that the specs OMIT entirely (constants,
  boundary conditions, objectives, load magnitudes, requirements).
- Label violations: label 10 in boundary_labels/fixed_labels; a label in
  both lists; roles contradicting the extracted design_roles; boundary
  definitions contradicting the weak form (e.g. calling a loaded
  boundary free; narrowing a whole-boundary load to one edge).
- Phi violations — judge by THIS priority chain:
  (a) A phi PRESCRIBED by the source must be used EXACTLY — this rule
      WINS over everything below.  When the prescribed phi itself is
      disconnected or leaves a fixed label void at iteration 0 or has
      an extreme solid fraction, do NOT demand a different phi: report
      ONE [warning] that the prescribed initial design conflicts with
      the connectivity / fixed-label-solidity / fraction rules and
      needs human arbitration (never re-report the same conflict in
      later rounds).
  (b) For a DESIGNED (non-prescribed) phi, each is an ERROR: never
      negative in the domain; solid fraction outside 20-80%; the solid
      region DISCONNECTED (e.g. two separated strips) — it must be ONE
      connected piece; fixed labels uncovered (not solid along their
      ENTIRE length) at iteration 0.
  (c) A prescribed phi that the design changed (not used exactly) is
      an ERROR.
- Parameter absurdity: hmin >= hmax, physically impossible material
  constants, diffusion/v0 orders of magnitude wrong for level-set
  advection.
- Lagrangian violations: a MULTI-VALUED Dirichlet (two or more
  DIFFERENT prescribed values) without penalty terms (state,
  adjoint AND shape derivative must all use the SAME pen =
  1e15); an UNNECESSARY penalty for a single-valued Dirichlet;
  multi-physics penalty with wrong units; check_units not true
  without rationale; multi-domain continuity missing.
- A source formula replaced by a PLACEHOLDER or reference instead of an
  explicit expression, or derivative formulas the source gives that the
  designed expressions drop while other entries still reference them.
  (Formulas may be REWRITTEN in an equivalent form or notation —
  judge mathematical equivalence, never textual identity.)
- The Dirichlet penalty (pen = 1e15) misapplied to non-penalty
  terms (e.g. an objective weight beta inside d/dt J must keep
  its own symbol).
- A pen = 1e15 penalty term without justification: pen is a FreeFEM
  WORKAROUND used ONLY when a field enforces two or more
  DIFFERENT Dirichlet values (e.g. 0 on one edge and 10 on
  another); a penalty for a SINGLE-valued Dirichlet is
  UNNECESSARY (the field declaration already enforces it) and
  must be REMOVED from the weak forms and the constants.  When
  a penalty IS justified, pen = 1e15 must be listed in constants
  and appear with the SAME value in the state, adjoint AND shape
  derivative.
- A BOUNDARY INTEGRAL without concrete mesh labels (e.g. written over
  the whole boundary dD instead of the labels it acts on) — the spec
  must state the actual labels; restricting a load to non-Dirichlet
  labels (where the test function vanishes) is computationally
  equivalent and acceptable when stated.
- ell/Afac missing its numerical VALUE — it is an ordinary constant
  (extracted from the source or designed) and the objective cannot be
  evaluated without it.
- A DESIGNED constant whose description contradicts the standard meaning
  of its symbol (in elasticity lambda/mu are the LAME parameters, E is
  Young's modulus, nu Poisson's ratio) — the correction is MANDATORY: a
  designed entry must use a non-conflicting, physically standard symbol
  description and unit, marked designed.  EXTRACTED constants are
  transcribed VERBATIM from the source and are NEVER held to this rule
  (the designed side may note the standard meaning in its rationale).
- A TEST FUNCTION (per the source's V-space definitions, e.g. v or
  delta_u) that is listed under `fields` or carries latex / type /
  unit / dirichlet extras — it must sit in test_functions with ONLY
  name/fe_space/domain; this is an ERROR whether the entry is in
  test_functions or misplaced under fields.
- A test fe_space not matching its physical field.
- Mesh violations: mesh_code not producing Th; borders without labels;
  designed edges not matching the ACTIVE code (commented-out borders are
  NOT part of the final mesh); square() meshes must use their implicit
  labels 1=bottom, 2=right, 3=top, 4=left; a redesigned mesh must keep
  the active source mesh's DOMAIN dimensions; labels_used must be the
  non-10 labels; mesh_file must be "mesh.txt"; edge roles must be
  consistent with the problem's design roles.
- Source annotations: every value carries source; "extracted" values
  must be verbatim traceable.
- A designed unit that is DIMENSIONALLY inconsistent with the
  quantity's physics — e.g. a shear modulus with unit '-', or a
  volume weight ell without a pressure unit (N/m^2) while
  check_units=true makes the objective terms carry units.
- A characteristic function whose piecewise CONDITION (the source's
  phi-sign -> strong/weak mapping) is not recorded — strong/weak
  values alone do not carry the mapping.
- A DESIGNED condition or weak_value that contradicts an EXTRACTED
  weak_value: the extracted weak_value is VERBATIM ground truth (the
  normalizer restores it every round, so a repair can never change it).
  When they conflict, report the DESIGNED CONDITION as the issue — the
  designed side must follow the extracted mapping (e.g. if the source
  says the void is 10 times weaker, the condition must say
  "chi = 1/10 if phi >= 0", and a separate k_void/interpolation may
  model any stricter void behaviour).
- A unit marked DESIGNED but not written in the canonical unit syntax
  (the UNIT RULES the generator follows — must stay in sync with them:
  Newton, Watt, Pascal, meter, LOWERCASE kelvin, '**' exponents,
  singular forms, dimensionless '-').  The fix is EITHER rewriting the
  unit in that syntax OR marking it unit_source "extracted" when it
  reproduces the source's own notation verbatim (extracted units stay
  verbatim — the code generator converts notation).
- The characteristic function latex missing the full
  \\raisebox{{\\depth}}{{$\\chi$}} decoration.
- mesh_code containing comments (// lines) — the generated mesh carries
  no comments.

## NOT issues (never report)
- problem_type / is_nonlinear: the REQUIRED classifications assigned by
  the generator — judge only whether they match the problem's nature.
- problem_name: assigned by the pipeline.
- Boundary names normalized to standard symbols (Gamma_D/Gamma_N) when
  the source describes the boundary without a symbol.
- fe_space/type the source does not state but that do not
  contradict it (the design fills them, marked designed).
- Fields' dirichlet_labels carrying mesh labels in place of the
  extracted LaTeX boundary names (deterministic reconciliation).
- A designed correction of an obviously-mislabeled source constant
  (e.g. lambda/mu written into the Lame combination) — report at most a
  warning.
- An EXTRACTED constant whose description/unit reproduces a source
  mislabel (e.g. the source calls mu "Poisson's ratio" with a
  dimensionless unit) — extracted entries are transcribed verbatim;
  only DESIGNED entries are held to the standard meanings.
- Extracted mesh edges transcribing borders that are commented out in
  the source (the extraction transcribes the file) — only the DESIGNED
  edges must match the active mesh.
- Content problems of a PROVIDED mesh.txt (its geometry, labels,
  borders or resolution) — a provided mesh is IMMUTABLE; the pipeline
  handles its syntax separately.  Only report when the designed spec
  CONTRADICTS it (e.g. designed edges/labels not matching the active
  code).
- Label 0 as a no-effect placeholder.
- Extracted boundary entries without mesh_label/design_role (the design
  phase assigns them).

## Consistency across rounds
- Be consistent with your OWN previous verdicts: never reverse an
  earlier instruction on the same item unless the spec changed in a way
  that makes it newly wrong — and when you do reverse one, say
  explicitly what changed.
- An issue you reported before that now appears fixed is CLOSED:
  verify the fix and drop the issue.
- Do not re-report an unchanged warning twice: report it once, and
  once more after a repair round left it unfixed — then stay silent
  on it.

Return ONE JSON:
{{"verified": true, "issues": []}}
or
{{"verified": false, "issues": [{{"item": "<spec:key>",
"severity": "error"|"warning", "problem": "<one sentence>"}}]}}

No markdown, no code fences.
"""

_REVIEW_USER = """\
Original input files:
{files_block}

Generated specs:
{specs_json}

Previous round (context for consistency — absent on the first round):
{context_text}

Return the review JSON.
"""

# ======================================================================
# REPAIR — fix only the listed issues
# ======================================================================

_REPAIR_SYSTEM = """\
You are repairing a topology-optimization pipeline's two spec JSONs.  You
receive the CURRENT specs and a list of issues found by the checker and
the reviewer.  FIX ONLY the listed issues — keep every other field
EXACTLY as it is (verbatim).  Do not regenerate untouched parts, do not
rewrite wording, do not add new content.  Return the complete corrected
JSON in the SAME shape:

{{
  "initial": {{ "extracted": {{...}}, "designed": {{...}} }},
  "mesh":    {{ "extracted": {{...}}, "designed": {{...}} }}
}}

NEVER repeat a mistake already made in a previous round — the repair
history in the user message lists the earlier errors and what each
earlier repair removed / added / modified.  Learn from it and change
strategy accordingly.

PARTIAL REPAIR: fix the issues you CAN fix; when you are NOT SURE how
to fix an issue correctly, LEAVE it UNCHANGED rather than guessing —
the next round will address it.  Issues the repair history marks
FAILED were not fixed by the previous repair: fix them now with a
DIFFERENT approach.  Issues marked FIXED must STAY fixed (never
reintroduce them).  Issues marked UNVERIFIED are LLM-review findings
that could not be re-checked deterministically: fix them when you
judge them real, otherwise leave them — the next review re-reports
them if they persist.
Issues prefixed [warning] are LOW priority: fix them only when you are
confident, and never break a fixed item for a warning.

When the input files include a mesh.txt, mesh.designed.mesh_code is
FROZEN — never modify it, whatever the listed issues say (the pipeline
restores it deterministically anyway).  Fix everything else normally.

No markdown, no code fences.
"""

_REPAIR_USER = """\
Original input files:
{files_block}

Current specs:
{specs_json}

Issues to fix (level | check | message):
{issues_text}

Repair history:
{history_text}

{regression_text}
{normalize_text}
{noop_text}
Return the corrected two-spec JSON.
"""


# ======================================================================
# the loop
# ======================================================================


def _invoke(agent, system, user, payload):
    agent.set_prompt_list([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return agent.invoke(payload)


def _diff_specs(prev, curr):
    """Compact structural diff between two spec dicts.

    Returns ``(removed, added, modified)`` — lists of dotted paths
    (capped in length for the prompt)."""
    removed, added, modified = [], [], []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                if k not in b:
                    removed.append(f"{path}.{k}" if path else k)
            for k in b:
                if k not in a:
                    added.append(f"{path}.{k}" if path else k)
                else:
                    walk(a[k], b[k], f"{path}.{k}" if path else k)
        elif isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                if i >= len(a):
                    added.append(f"{path}[{i}]")
                elif i >= len(b):
                    removed.append(f"{path}[{i}]")
                else:
                    walk(a[i], b[i], f"{path}[{i}]")
        elif a != b:
            modified.append(path)

    walk(prev, curr, "")
    return removed[:20], added[:20], modified[:20]


def _issue_id(issue):
    """Stable identity of an issue — stable under reviewer rephrasing and
    under value/index changes inside deterministic messages.

    Review issues are keyed on their STRUCTURED ``item`` path (carried
    on the issue dict by ``_review`` — never re-parsed from the
    formatted message, which collapses colon-style items like
    ``initial:optimization_config.phi`` to ``initial``); deterministic
    issues on their check name plus the message with embedded
    values/indices normalised away."""
    check = issue.get("check")
    msg = str(issue.get("message") or "")
    if check == "review":
        item = str(issue.get("item") or "").strip().lower()
        if item:
            return ("review", item)
        # fallback: issue dicts built outside _review (tests) — use the
        # normalised message
        norm = re.sub(r"\([^)]*\)", "(...)", msg)
        norm = re.sub(r"\d+", "N", norm)
        return ("review", re.sub(r"\s+", " ", norm).strip().lower())
    norm = re.sub(r"\([^)]*\)", "(...)", msg)
    norm = re.sub(r"\d+", "N", norm)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    return (check, norm)


def _history_text(history):
    """One compact block per previous round: the errors, what the round
    FIXED, what it FAILED to fix, and the changes."""
    lines = []
    for h in history:
        errs = "; ".join(str(e)[:100] for e in h["errors"][:5])
        fixed = "; ".join(str(x)[:60] for x in h.get("fixed", [])[:4])
        failed = "; ".join(str(x)[:60] for x in h.get("failed", [])[:4])
        unverified = "; ".join(str(x)[:60] for x in h.get("unverified", [])[:4])
        lines.append(
            f"round {h['round']}: errors=[{errs}] fixed=[{fixed}] "
            f"failed=[{failed}] unverified=[{unverified}] "
            f"removed={h['removed'][:8]} "
            f"added={h['added'][:8]} modified={h['modified'][:8]}"
        )
    return "\n".join(lines) if lines else "(no previous rounds)"


def _terminate(round_no, message, issue_list, specs, history, carried_warnings):
    """Terminate the case: record the failure and return the state
    (the caller aborts on ``_passed`` False)."""
    warnings.warn(f"spec_generation: {message}")
    print(f"  [spec loop] TERMINATED: {message}", flush=True)
    history.append({
        "round": round_no,
        "errors": [message[:120]],
        "fixed": [], "failed": [], "unverified": [],
        "removed": [], "added": [], "modified": [],
    })
    specs["_passed"] = False
    specs["_issues"] = issue_list
    specs["_review_warnings"] = carried_warnings
    specs["_history"] = history
    specs["_attempts"] = max(0, round_no - 1)
    return specs


def _apply_review(review, checks, warning_strikes, carried_ids, carried_warnings):
    """Merge a review verdict into the issue list: errors plus
    first-strike warnings.  The 2-strike rule (stable identity,
    ``_issue_id``) makes a repeated warning non-blocking and carries
    it to the exit report instead of looping forever.

    Returns ``(review_issues, blocking_issues)`` — *review_issues* is
    the RAW verdict content (the A3 gate needs the pre-strike view),
    *blocking_issues* feeds the repair loop."""
    review_issues = review["errors"] + review["warnings"]
    fresh_warnings = []
    for w in review_issues:
        if w["severity"] != "warning":
            continue
        key = _issue_id(w)
        warning_strikes[key] = warning_strikes.get(key, 0) + 1
        if warning_strikes[key] >= 2:
            if key not in carried_ids:
                carried_ids.add(key)
                carried_warnings.append(w["message"])
        else:
            fresh_warnings.append(w)
    return review_issues, (
        hard_errors(checks)
        + [w for w in review_issues if w["severity"] == "error"]
        + fresh_warnings
    )


def _parse_two_specs(raw):
    """Parse the {"initial", "mesh"} answer; None on failure."""
    try:
        data = json.loads(parser_json(raw))
    except Exception as e:
        warnings.warn(f"spec_generation JSON parse failed: {e}")
        return None
    if not isinstance(data, dict) or not isinstance(data.get("initial"), dict) \
            or not isinstance(data.get("mesh"), dict):
        return None
    return data


def _normalize_two_specs(data):
    initial = data["initial"]
    mesh = data["mesh"]
    ex = normalize_extracted(initial.get("extracted") or {})
    ds = normalize_designed(initial.get("designed") or {}, ex)
    mex = normalize_extracted_mesh(mesh.get("extracted") or {})
    mds = normalize_designed_mesh(mesh.get("designed") or {}, mex)
    # cross-fix: the problem extraction may omit mesh_labels_used — fill
    # it from the mesh extraction's labels_found (the same universe)
    if not isinstance(ex.get("mesh_labels_used"), list) or not ex.get("mesh_labels_used"):
        labels_found = mex.get("labels_found")
        if isinstance(labels_found, list) and labels_found:
            ex["mesh_labels_used"] = [str(l) for l in labels_found]
            ex["mesh_labels_used_source"] = "extracted"
    # cross-fix: when the active mesh is square(), FreeFEM's implicit
    # labels 1-4 MUST be part of the extracted mesh label universe
    if re.search(r"\bmesh\s+Th\s*=\s*square\s*\(", mex.get("mesh_code") or ""):
        mlu = ex.get("mesh_labels_used")
        if isinstance(mlu, list):
            have = {str(x) for x in mlu}
            for l in ("1", "2", "3", "4"):
                if l not in have:
                    mlu.append(l)
    # deterministic backfill: domain_description is a DESIGNED field —
    # when the LLM omits it, derive it from the active mesh geometry
    # instead of burning repair rounds on the same syntax error (every
    # square()/border mesh with a parseable box qualifies when the
    # problem has no multi-domain list)
    if not ds.get("domain_description") and not (ds.get("domains") or []):
        L, H = _mesh_box(mds.get("mesh_code") or mex.get("mesh_code") or "")
        if L and H:
            ds["domain_description"] = (
                f"the design domain is the rectangle [0, {L:g}] x [0, {H:g}]"
            )
            ds["domain_description_source"] = "designed"
            ds["domain_description_rationale"] = (
                "deterministic backfill from the active mesh geometry "
                "(single rectangular domain)"
            )
    # deterministic decoration: the reviewer requires the characteristic
    # function latex in the FULL decorated form — when a DESIGNED latex
    # is a bare \chi, apply the canonical string instead of burning
    # repair rounds on it (the repairer keeps returning the bare form
    # and the no-op guard terminates the case — verified 2026-10-17).
    # Extracted latex stays verbatim (ground truth).
    cf = ds.get("characteristic_function")
    if isinstance(cf, dict) and cf.get("latex") \
            and cf.get("latex_source") != "extracted":
        bare = (str(cf["latex"])
                .replace("$", "").replace("{", "").replace("}", "")
                .replace(" ", ""))
        if bare == r"\chi":
            cf["latex"] = r"\raisebox{\depth}{$\chi$}"
    # deterministic border reconciliation (USER 2026-10-18): the edge
    # list must contain EXACTLY ONE entry per active mesh_code border,
    # named after it.  The repairer stalls on both observed shapes:
    # duplicate entries with 'border ...' roles (10-round stall,
    # 2026-10-17) and MISSING borders when two borders share a label
    # (2026-10-18).  Rebuild the list border by border: prefer a
    # valid-role entry, then a name match; synthesize a MISSING entry
    # from the border body (param/x/y/label), role 'fixed' when the
    # label sits in the spec's fixed_labels else 'design'.
    edges = mds.get("edges")
    active_code = mds.get("mesh_code") or mex.get("mesh_code") or ""
    # border-mesh reconciliation only — a square() mesh has no border
    # statements and its S1-S4 edges come from normalize_designed_mesh's
    # square branch (clobbering them emptied the list — verified
    # 2026-10-18 on run_heatedPanelNonlinear)
    if isinstance(edges, list) and re.search(r"\bborder\s+", active_code):
        from Initialization.spec_checks import (_BORDER_RE, _ROLES,
                                                _ROLE_ALIASES, _labels_in_text,
                                                _norm_enum)
        fixed_labels = {str(l) for l in
                        ((ds.get("optimization_config") or {})
                         .get("fixed_labels") or [])}
        borders = []
        for bm in _BORDER_RE.finditer(active_code):
            name, body = bm.group(1), bm.group(2)
            param_m = re.search(r"\((\w+)\s*=\s*0\s*,\s*1\)", bm.group(0))
            x_m = re.search(r"\bx\s*=\s*([^;]+)", body)
            y_m = re.search(r"\by\s*=\s*([^;]+)", body)
            borders.append({
                "name": name,
                "labels": [str(l) for l in _labels_in_text(body)],
                "param": param_m.group(1) if param_m else "t",
                "x": x_m.group(1).strip() if x_m else "",
                "y": y_m.group(1).strip() if y_m else "",
            })
        claimed = set()
        keep = []
        for b in borders:
            cands = [e for e in edges if isinstance(e, dict)
                     and id(e) not in claimed and (
                         str(e.get("name") or "") == b["name"]
                         or (b["labels"] and str(e.get("label") or "")
                             in b["labels"]))]
            if not cands:
                role = ("fixed" if b["labels"]
                        and b["labels"][0] in fixed_labels else "design")
                keep.append({
                    "name": b["name"], "name_source": "designed",
                    "param": b["param"], "param_source": "designed",
                    "x": b["x"], "x_source": "designed",
                    "y": b["y"], "y_source": "designed",
                    "label": b["labels"][0] if b["labels"] else "",
                    "label_source": "designed",
                    "role": role, "role_source": "designed",
                    "source": "designed",
                    "rationale": ("deterministic backfill from mesh_code "
                                  f"border {b['name']!r}"),
                })
                continue
            e = max(cands, key=lambda x: (
                int(str(x.get("role") or "") in _ROLES),
                int(str(x.get("name") or "") == b["name"])))
            claimed.add(id(e))
            if e.get("name") != b["name"] \
                    and e.get("name_source") != "extracted":
                e["name"] = b["name"]
            # canonicalize the role: normalize_designed_mesh synthesizes
            # missing entries with the INVALID placeholder role
            # 'border <name>' (spec_checks.py) and re-injects it EVERY
            # round — the reviewer flagged it for days and the repairer
            # could never fix it.  Map to design/fixed (label-aware).
            r = _norm_enum(e.get("role"), _ROLES, _ROLE_ALIASES)
            if r not in _ROLES:
                r = ("fixed" if str(e.get("label") or "") in fixed_labels
                     else "design")
            e["role"] = r
            keep.append(e)
        mds["edges"] = keep
    return {
        "initial": {"extracted": ex, "designed": ds},
        "mesh": {"extracted": mex, "designed": mds},
    }


def _deterministic_checks(specs):
    """Syntax-level checks, generated content only."""
    ex, ds = specs["initial"]["extracted"], specs["initial"]["designed"]
    mex, mds = specs["mesh"]["extracted"], specs["mesh"]["designed"]
    checks = (
        check_extracted_problem(ex)
        + check_designed_problem(ds, ex)
        + check_extracted_mesh(mex)
        + check_designed_mesh(mds, mex, ds)
    )
    # phi connectivity on the mesh box (deterministic — the reviewer
    # sometimes misses a disconnected initial design).  A phi PRESCRIBED
    # by the source wins over the connectivity rule (the reviewer's phi
    # priority chain: prescribed-phi-exactly > connectivity > fixed
    # labels) — the check applies to DESIGNED phis only; a prescribed
    # phi violating the other rules surfaces once in the review as a
    # human-arbitration warning.
    code = (mds.get("mesh_code") or "") or (mex.get("mesh_code") or "")
    L, H = _mesh_box(code)
    ex_params = ex.get("optimization_params")
    ex_phi = ex_params.get("phi") if isinstance(ex_params, dict) else None
    prescribed_phi = bool(
        ex_phi is not None and str(ex_phi).strip() != ""
    )
    opt = ds.get("optimization_config")
    # USER RULE (2026-10-15): a designed phi is NOT validated —
    # connectivity runs only for non-prescribed, non-designed phis.
    if (isinstance(opt, dict) and not prescribed_phi
            and opt.get("phi_source") != "designed"):
        checks += _phi_connectivity_issues(opt.get("phi", ""), L, H)
    return checks


_JSON_REPAIR_SYSTEM = """\
Your previous answer was NOT valid JSON.  Return THE SAME CONTENT —
unchanged in meaning — as exactly ONE valid JSON object.
No markdown, no code fences, no explanations.
"""

_JSON_REPAIR_USER = """\
The previous (invalid) answer:
{raw_text}

Return the same content as one valid JSON object.
"""


def _review(agent, files, specs, prev_issues=None, changes=None,
            review_system=None):
    """The merged LLM review — returns ``{"verified": bool, "errors":
    [issue dicts], "warnings": [issue dicts]}``, or None when the review
    FAILS.  Each issue dict carries its STRUCTURED ``item`` path plus
    the formatted ``message`` (the loop's ``_issue_id`` keys on the
    item, never on the re-parsed text).

    An unparseable/empty/wrong-shape reply ALWAYS goes through ONE
    retry: an unparseable non-empty reply goes back verbatim for a
    JSON-shape repair; a wrong-shape reply (valid JSON missing
    'verified'), an empty reply or a transport failure re-runs the full
    review call; when that ALSO fails the review returns None — the
    caller TERMINATES the case: a failed review must never silently
    pass the spec.  Errors and warnings are split so warnings can flow
    into the repair at low priority instead of being dropped."""
    if prev_issues:
        prev_text = "\n".join(f"- {str(m)[:200]}" for m in prev_issues[:6])
    else:
        prev_text = "(first round — no previous review)"
    chg_text = (
        "\n".join(f"- {p}" for p in (changes or [])[:12])
        if changes else "(no repair changes yet)"
    )
    context_text = (
        f"Issues reported last round:\n{prev_text}\n\n"
        f"Paths changed by the last repair:\n{chg_text}"
    )
    review_payload = {
        "files_block": files,
        "specs_json": json.dumps(specs, indent=2, ensure_ascii=False),
        "context_text": context_text,
    }
    result = None
    first_error = "no reply"
    raw = ""
    shape_bad = False
    try:
        raw = _invoke(agent, review_system or _REVIEW_SYSTEM,
                      _REVIEW_USER, review_payload)
        result = json.loads(parser_json(raw))
        if not (isinstance(result, dict) and "verified" in result):
            result = None
            shape_bad = True
            first_error = "reply is not a review JSON (missing 'verified')"
    except Exception as e:
        result = None
        first_error = str(e)
    if result is None:
        # ONE retry, branched on the failure kind: a wrong-shape reply
        # or an empty/transport failure re-runs the FULL review (the
        # JSON-shape repair is pointless for a reply that is already
        # valid JSON but lacks 'verified')
        try:
            if shape_bad:
                set_label(
                    agent, "Initialization/run_spec_generation (review retry)"
                )
                raw = _invoke(agent, review_system or _REVIEW_SYSTEM,
                      _REVIEW_USER, review_payload)
            elif raw and raw.strip():
                set_label(
                    agent,
                    "Initialization/run_spec_generation (review JSON repair)",
                )
                raw = _invoke(
                    agent, _JSON_REPAIR_SYSTEM, _JSON_REPAIR_USER,
                    {"raw_text": raw},
                )
            else:
                set_label(
                    agent, "Initialization/run_spec_generation (review retry)"
                )
                raw = _invoke(agent, review_system or _REVIEW_SYSTEM,
                      _REVIEW_USER, review_payload)
            result = json.loads(parser_json(raw))
            if not (isinstance(result, dict) and "verified" in result):
                result = None
        except Exception as e2:
            result = None
            first_error = f"{first_error} / {e2}"
    if result is None:
        warnings.warn(
            f"spec review failed after one retry — terminating the case "
            f"(a failed review must never pass silently): {first_error}"
        )
        return None
    errors, warns = [], []
    for it in (result.get("issues") or []):
        if not isinstance(it, dict):
            continue
        sev = str(it.get("severity") or "error").lower()
        item = str(it.get("item") or "?").strip()
        entry = {
            "check": "review",
            "level": "review",
            "severity": sev if sev == "warning" else "error",
            "item": item,
            "message": f"[{sev}] {item}: {it.get('problem', '')}",
        }
        (warns if sev == "warning" else errors).append(entry)
    return {
        "verified": _as_bool(result.get("verified")) is True,
        "errors": errors,
        "warnings": warns,
    }


# ======================================================================
# PROVIDED mesh.txt — immutable input: ONE syntax check + at most ONE
# syntax repair, then frozen into every round.  Content is NEVER
# repaired (the mesh is extracted ground truth).
# ======================================================================

_MESH_SYNTAX_REPAIR_SYSTEM = """\
You repair the SYNTAX of a FreeFEM mesh file — nothing else.  The file
was PROVIDED by the user and its CONTENT (geometry, border definitions,
labels, resolution) is immutable.

Fix ONLY genuine syntax problems: unbalanced delimiters, malformed
lines, missing semicolons, invalid identifier characters.  NEVER change
the geometry, the border parameterisations, the labels, the resolution
``int n``, or the mesh variable name ``Th``.

Return ONLY the complete corrected mesh code — no markdown, no code
fences, no explanation.
"""

_MESH_SYNTAX_REPAIR_USER = """\
Problem: {problem_name}

The provided mesh code:
{mesh_code}

Reported syntax problems:
{problems_text}

Return only the complete corrected mesh code.
"""


def _mesh_syntax_pass(agent, code: str, problem_name=None) -> tuple:
    """The ONE syntax check + at most ONE syntax repair for a PROVIDED
    mesh.txt.

    Returns ``(mesh_code, repaired, problems_text)`` — the (possibly
    syntax-repaired) frozen mesh code, whether the LLM repair was
    applied, and the remaining problems ("" when clean).  Content is
    NEVER touched: the repair fixes syntax only, and it is accepted
    only when it strictly reduces the problem count.
    """
    problems = mesh_syntax_problems(code)
    if not problems:
        return code, False, ""
    problems_text = "\n".join(f"- {p}" for p in problems)
    set_label(agent, "Initialization/mesh syntax repair (provided mesh.txt)")
    try:
        raw = _invoke(
            agent,
            _MESH_SYNTAX_REPAIR_SYSTEM,
            _MESH_SYNTAX_REPAIR_USER,
            {
                "problem_name": problem_name or "unknown",
                "mesh_code": code,
                "problems_text": problems_text,
            },
        )
        fixed = str(raw).strip() if isinstance(raw, str) else ""
    except Exception as e:
        warnings.warn(f"mesh syntax repair call failed: {e}")
        return code, False, problems_text
    if fixed and fixed != code.strip():
        remaining = mesh_syntax_problems(fixed)
        if len(remaining) < len(problems):
            return fixed, True, "\n".join(f"- {p}" for p in remaining)
    return code, False, problems_text


def _freeze_provided_mesh(specs: dict, mesh_code: str, repaired: bool,
                          problems_text: str) -> None:
    """Force the PROVIDED mesh.txt into the spec — immutable.

    *repaired* records whether the ONE syntax repair ran on the content
    (the source stays "extracted" either way — the merged enum has no
    "extracted_repaired").  Both the extracted and the designed sides
    carry the SAME content so the verbatim immutability checks pass and
    no later review/repair round can change the mesh.
    """
    mesh = specs.get("mesh") or {}
    ex = mesh.get("extracted")
    ds = mesh.get("designed")
    if not isinstance(ex, dict) or not isinstance(ds, dict):
        return
    ex["mesh_code"] = mesh_code
    ex["category"] = "code"
    ex["has_syntax_errors"] = bool(problems_text)
    ex["syntax_error_details"] = problems_text
    ds["mesh_code"] = mesh_code
    ds["mesh_code_source"] = "extracted"
    ds["mesh_code_rationale"] = (
        "verbatim copy of the PROVIDED mesh.txt "
        "(immutable — content never repaired)"
        if not repaired
        else "the PROVIDED mesh.txt after the ONE syntax repair "
        "(content untouched)"
    )
    # n derives from the FROZEN code (the normalizers backfill labels
    # and edges but never n — a stale LLM-authored n would fail the
    # mesh_n check)
    nm = re.search(r"\bint\s+n\s*=\s*(\d+)\s*;", mesh_code)
    if nm:
        ex["n"] = nm.group(1)
        ds["n"] = nm.group(1)
        ds["n_source"] = "extracted"


def run_spec_generation(agent, input_files, problem_name=None, max_fixes=None,
                        output_dir=None, history_log=None, seed_specs=None,
                        repair_system=None, history_text_fn=None,
                        generate_system=None, review_system=None):
    """Generate BOTH spec JSONs via the generate -> review -> repair loop.

    Returns ``{"initial": {"extracted","designed"}, "mesh": {"extracted",
    "designed"}}`` — the FIRST state that passes the deterministic
    syntax checks AND the merged LLM review, or the last attempted state
    when the loop exceeds ``configs.MAX_SPEC_FIXES`` repairs, with
    ``_passed`` set to False (callers must treat ``_passed`` as the
    failure signal: the designed parts may still be non-empty in the
    last state).

    A repair history is maintained in memory: the initial generation and
    every repair round record their errors and what the round removed /
    added / modified relative to the previous state; the history is
    given to the repair LLM so it never repeats a mistake.

    When the input files include a ``mesh.txt``, it is PROVIDED and
    IMMUTABLE: one deterministic syntax check runs once, at most ONE
    LLM syntax repair is attempted (syntax only — content is never
    touched), and the resulting code is frozen into every round's
    extracted.mesh_code / designed.mesh_code (source "extracted").
    Without a mesh.txt the mesh is designed by
    the loop as usual.

    When *output_dir* is given, initial.json and mesh.json are written
    after EVERY generation and repair round (not only at the end)."""
    max_fixes = MAX_SPEC_FIXES if max_fixes is None else max_fixes
    files = files_block(input_files, MAX_FILE_CHARS)
    specs = None
    issues = []
    history = []
    # EXPERIMENT SEED: start the loop from a provided (possibly broken)
    # spec state instead of a fresh LLM generation — the first round
    # runs the deterministic checks and the review on the seed as-is,
    # then repairs it like any other candidate.
    seeded_pending = False
    if seed_specs:
        specs = _normalize_two_specs(seed_specs)
        seeded_pending = True
        if history_log is not None:
            record_history(
                history_log,
                "run_spec_generation (experiment seed)",
                label="seeded starting state",
                output_text=json.dumps(specs, indent=2, ensure_ascii=False),
            )

    # PROVIDED mesh.txt — immutable input: ONE syntax check + at most
    # ONE syntax repair, then frozen into every round's output.  No
    # content repair ever touches it.
    provided_mesh = None
    if isinstance(input_files, dict) and "mesh.txt" in input_files:
        provided_mesh = str(input_files["mesh.txt"] or "")
        if not provided_mesh.strip():
            provided_mesh = None
        else:
            try:
                from Generation.check_static import strip_ff_comments

                provided_mesh = strip_ff_comments(provided_mesh)
            except Exception:
                pass
    frozen_mesh = None
    frozen_repaired = False
    frozen_problems = ""
    if provided_mesh:
        frozen_mesh, frozen_repaired, frozen_problems = _mesh_syntax_pass(
            agent, provided_mesh, problem_name
        )
        print(
            f"  [mesh] provided mesh.txt frozen "
            f"({'syntax-repaired' if frozen_repaired else 'verbatim'})"
            + (f" — remaining syntax problems: {frozen_problems}"
               if frozen_problems else ""),
            flush=True,
        )

    def _write_round(specs):
        if output_dir is None:
            return
        out = Path(output_dir)
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / "initial.json").write_text(
                json.dumps(specs["initial"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out / "mesh.json").write_text(
                json.dumps(specs["mesh"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            warnings.warn(f"spec_generation: per-round write failed: {e}")

    # per-run loop state
    warning_strikes = {}   # stable issue id -> count; 2 strikes -> non-blocking
    carried_ids = set()    # ids already carried to the exit report
    carried_warnings = []  # non-blocking warnings, written to the exit report
    normalize_notes = []   # P1: paths the normalizer changed last round
    regressions = []       # P9: issues NEW after the previous repair
    noop_count = 0         # P9: consecutive repair rounds with no changes
    attempts = 0           # repair rounds used (the generate round is attempt 0)

    while attempts <= max_fixes:
        round_no = attempts + 1
        round_label = (
            f"Initialization/run_spec_generation round {round_no}/{max_fixes + 1}"
        )
        prev_issues = list(issues) if issues else []
        if specs is None:
            # fresh generation (or re-generation after an unparseable answer)
            set_label(agent, f"{round_label} (generate)")
            raw = _invoke(
                agent, generate_system or _GENERATE_SYSTEM, _GENERATE_USER,
                {"problem_name": problem_name or "unknown", "files_block": files},
            )
            data = _parse_two_specs(raw)
        elif seeded_pending:
            # experiment seed: the provided state IS this round's
            # candidate — the checks and review run on it directly
            seeded_pending = False
            data = specs
        else:
            issues_text = format_errors(issues) if issues else ""
            regression_text = (
                "NEW issues that appeared AFTER your previous repair "
                "(check whether your changes caused them; if so, undo "
                "the damage):\n"
                + "\n".join(
                    f"- {i['level']} | {i['check']} | {i['message']}"
                    for i in regressions[:8]
                )
            ) if regressions else ""
            normalize_text = (
                "The deterministic normalizer changed these paths from "
                "your previous output (verify that nothing you fixed "
                "was reverted; re-fix it if it was):\n"
                + "\n".join(f"- {p}" for p in normalize_notes[:15])
            ) if normalize_notes else ""
            noop_text = (
                "WARNING: your previous repair made NO changes at all — "
                "the issues below are still open.  Fix them NOW with a "
                "DIFFERENT approach."
            ) if noop_count else ""
            set_label(agent, f"{round_label} (repair)")
            raw = _invoke(
                agent, repair_system or _REPAIR_SYSTEM, _REPAIR_USER,
                {
                    "files_block": files,
                    "specs_json": json.dumps(specs, indent=2, ensure_ascii=False),
                    "issues_text": issues_text,
                    "history_text": (_history_text if history_text_fn is None
                                     else history_text_fn)(history),
                    "regression_text": regression_text,
                    "normalize_text": normalize_text,
                    "noop_text": noop_text,
                },
            )
            data = _parse_two_specs(raw)
        if data is None:
            issues = [{"check": "json_parse", "level": "syntax",
                       "severity": "error", "message": "LLM output failed to parse"}]
            print(
                f"  [spec loop {attempts}/{max_fixes}] unparseable answer — repairing",
                flush=True,
            )
            history.append({
                "round": round_no,
                "errors": ["unparseable answer"],
                "fixed": [], "failed": [], "unverified": [],
                "removed": [], "added": [], "modified": [],
            })
            # an unparseable reply is not a no-op round, and stale
            # regression notes from two rounds ago would mislead the
            # retry's repair prompt
            regressions = []
            noop_count = 0
            attempts += 1
            continue
        prev = specs
        specs = _normalize_two_specs(data)
        if frozen_mesh is not None:
            _freeze_provided_mesh(specs, frozen_mesh, frozen_repaired, frozen_problems)
            # M2: the freeze replaced the mesh code — CLEAR the
            # mesh-derived fields (they described the LLM's discarded
            # mesh) and re-derive them from the FROZEN code so they can
            # never produce unfixable cross-file errors
            mex = specs["mesh"]["extracted"]
            # n is NOT cleared: _freeze_provided_mesh just derived it
            # from the FROZEN code, and the normalizers never backfill
            # n (clearing it here would lose the frozen value and make
            # the mesh_n check unfixable for every provided-mesh case)
            for k in ("labels_found", "edges"):
                mex.pop(k, None)
            specs["mesh"]["extracted"] = normalize_extracted_mesh(mex)
            mds = specs["mesh"]["designed"]
            for k in ("labels_used", "edges"):
                mds.pop(k, None)
            specs["mesh"]["designed"] = normalize_designed_mesh(
                mds, specs["mesh"]["extracted"])
        # P1: the normalizer may (legitimately) overwrite LLM output —
        # surface every path it changed so a silently reverted fix stays
        # visible and the next repair can re-apply it (computed AFTER
        # the provided-mesh freeze so the FROZEN mesh paths never appear
        # as normalizer reverts)
        _rem, _add, _mod = _diff_specs(data, specs)
        normalize_notes = (_rem + _add + _mod)[:20]
        if normalize_notes:
            warnings.warn(f"spec normalize changed paths: {normalize_notes}")
        _write_round(specs)  # every round persists both JSONs
        # full record of the round's OUTPUT (the normalised specs) —
        # the LLM record above already carries the raw response
        if history_log is not None:
            record_history(
                history_log,
                "run_spec_generation (deterministic)",
                label=f"{round_label} (normalised output)",
                input_text=(
                    f"normalised from the {round_label} (generate/repair) response"
                ),
                output_text=json.dumps(specs, indent=2, ensure_ascii=False),
            )

        checks = _deterministic_checks(specs)
        # PARTIAL REPAIR — classify which of the previous round's issues
        # this repair actually fixed: an issue still present in the new
        # deterministic checks is recorded FAILED (skipped, not blocking —
        # the next round addresses it), a disappeared one FIXED.  LLM
        # review issues cannot be re-verified deterministically and are
        # recorded UNVERIFIED (the next review re-reports them if they
        # persist).  The classification lands on the PREVIOUS round's
        # history entry, whose repair just produced this state.  All
        # matching uses the STABLE issue identity (_issue_id) so a
        # rephrased or value-changed message still matches.
        if prev is not None and prev_issues and history:
            det_ids = {_issue_id(c) for c in checks}
            fixed_list, failed_list, unverified = [], [], []
            for i in prev_issues:
                label = str(i.get("message"))[:120]
                if i.get("check") == "review":
                    # M7: a persistent BLOCKING review error is FAILED
                    # (it was not fixed); only warnings stay UNVERIFIED
                    # (they cannot be re-checked deterministically)
                    if i.get("severity") == "warning":
                        unverified.append(label)
                    else:
                        failed_list.append(label)
                elif _issue_id(i) in det_ids:
                    failed_list.append(label)
                else:
                    fixed_list.append(label)
            last = history[-1]
            last["fixed"] = fixed_list
            last["failed"] = failed_list
            last["unverified"] = unverified

        # P3: the reviewer runs EVERY round — deterministic and review
        # issues merge, so a substantive defect is never hidden behind
        # a persistent syntax stall.  P7: the review gets the previous
        # issues and the repair diff for self-consistency.
        set_label(agent, f"{round_label} (review)")
        if prev is not None:
            rmv, add, mod = _diff_specs(prev, specs)
            changes = rmv + add + mod
        else:
            changes = ["(initial generation)"]

        review = _review(
            agent, files, specs,
            prev_issues=[str(i.get("message")) for i in prev_issues],
            changes=changes,
            review_system=review_system,
        )
        if review is None:
            # P6: a failed review TERMINATES the case — never fail open
            message = (
                "the spec review failed after one retry "
                "(unparseable/empty reply) — the case terminates "
                "instead of passing silently"
            )
            return _terminate(round_no, message, [{
                "check": "review_unavailable", "level": "review",
                "severity": "error", "message": message,
            }], specs, history, carried_warnings)
        review_issues, issues = _apply_review(
            review, checks, warning_strikes, carried_ids, carried_warnings)
        # A3: verified:false with NOTHING actionable (no errors AND no
        # warnings in the raw verdict) is a review failure in disguise —
        # re-review ONCE before passing; a second unverified verdict
        # terminates (never pass silently).  A verdict that only carried
        # struck warnings falls through to the pass gate below and is
        # recorded in the exit report instead.
        if not issues and not review_issues and review["verified"] is not True:
            set_label(agent, f"{round_label} (review retry)")
            review = _review(
                agent, files, specs,
                prev_issues=[str(i.get("message")) for i in prev_issues],
                changes=changes,
                review_system=review_system,
            )
            if review is None:
                return _terminate(
                    round_no,
                    "the spec review failed after one retry "
                    "(unparseable/empty reply) — the case terminates "
                    "instead of passing silently",
                    [{"check": "review_unavailable", "level": "review",
                      "severity": "error",
                      "message": "the spec review failed after one retry "
                                 "(unparseable/empty reply)"}],
                    specs, history, carried_warnings,
                )
            review_issues, issues = _apply_review(
                review, checks, warning_strikes, carried_ids, carried_warnings)
            if not issues and not review_issues and review["verified"] is not True:
                return _terminate(
                    round_no,
                    "the reviewer returned verified:false with no issues "
                    "TWICE — the case terminates instead of passing on "
                    "an unverified spec",
                    [{"check": "review_unavailable", "level": "review",
                      "severity": "error",
                      "message": "the reviewer returned verified:false "
                                 "with no issues twice"}],
                    specs, history, carried_warnings,
                )
        # M6: dedupe the DELIVERED list on the exact (check, message)
        # pair — the stable identity is for strikes/cycles, but two
        # genuinely different issues that share an identity (e.g.
        # interfaces[0] vs interfaces[2]) must both reach the repair
        _seen = set()
        _deduped = []
        for i in issues:
            key = (i.get("check"), i.get("message"))
            if key not in _seen:
                _seen.add(key)
                _deduped.append(i)
        issues = _deduped
        # P9: issues NEW since the previous round — flagged in the NEXT
        # repair prompt as possible regressions introduced by the repair
        # (only meaningful after a repair round has actually run)
        prev_ids = {_issue_id(i) for i in prev_issues}
        regressions = [
            i for i in issues if _issue_id(i) not in prev_ids
        ] if (prev is not None and prev_issues) else []
        if not issues:
            # P11 exit: nothing blocking remains.  The last review may
            # still be verified:false on carried warnings — they are
            # written to the exit report instead of looping.
            if carried_warnings and output_dir is not None:
                try:
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "review_warnings.json").write_text(
                        json.dumps(carried_warnings, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except OSError as e:
                    warnings.warn(f"spec_generation: warnings write failed: {e}")
            print(f"  [spec loop] PASSED after {attempts} repair(s)", flush=True)
            specs["_passed"] = True
            specs["_review_warnings"] = carried_warnings
            specs["_history"] = history
            specs["_attempts"] = attempts
            return specs
        print(
            f"  [spec loop {attempts}/{max_fixes}] {len(issues)} issue(s) — repairing",
            flush=True,
        )
        # ascii-safe console output: issue messages may quote LaTeX glyphs
        # that a cp1252 console cannot encode
        for i in issues:
            line = f"    - {i.get('level')} | {i.get('check')} | {i.get('message')}"
            print(line.encode("ascii", errors="replace").decode(), flush=True)

        # ── repair history for the NEXT repair ──
        if prev is not None:
            removed, added, modified = _diff_specs(prev, specs)
        else:
            removed, added, modified = [], [], ["(initial generation)"]
        # P9: no-op detection — an empty round with UNCHANGED issues
        # (stable identities) does NOT consume budget; the retry gets a
        # different wording via the noop note.  Two consecutive no-ops
        # terminate the case.
        no_change = not (removed or added or modified)
        same_issues = {_issue_id(i) for i in issues} == prev_ids
        if prev is not None and no_change and same_issues:
            noop_count += 1
            if noop_count >= 2:
                return _terminate(
                    round_no,
                    "two consecutive no-op repairs — terminating the case",
                    issues, specs, history, carried_warnings,
                )
        else:
            noop_count = 0
            attempts += 1
        history.append({
            "round": round_no,
            "errors": [str(i.get("message"))[:120] for i in issues],
            "removed": removed,
            "added": added,
            "modified": modified,
        })

    warnings.warn("spec_generation: repair limit reached — returning the last state")
    if specs is None:
        return {"initial": {}, "mesh": {}, "_passed": False,
                "_issues": issues or []}
    specs["_passed"] = False
    specs["_issues"] = issues
    specs["_review_warnings"] = carried_warnings
    specs["_history"] = history
    specs["_attempts"] = attempts
    return specs
