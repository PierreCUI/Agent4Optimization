"""
One-loop spec generation: GENERATE -> REVIEW -> REPAIR.

Each iteration produces BOTH spec JSONs together (initial.json's content
and mesh.json's content in one LLM answer), runs the deterministic syntax
checks (generated content only), then ONE combined LLM review (extraction
fidelity + design rules + physical plausibility + information omissions —
all judged in ONE review pass).  On issues the
loop REPAIRS the previous output (fix only the listed issues, keep
everything else verbatim) instead of regenerating from scratch.  When an
output_dir is given, the two JSONs are written to disk after every
generation and repair round; the loop stops when it passes or after
``configs.MAX_SPEC_FIXES`` repairs.
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
    normalize_designed,
    normalize_designed_mesh,
    normalize_extracted,
    normalize_extracted_mesh,
)

# ======================================================================
# GENERATE — both specs in one answer
# ======================================================================

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
   values; units DESIGNED by you are written in AutoFreeFEM syntax:
   CAPITALIZED names (Newton, Watt, Pascal, meter, kelvin), compound
   units as e.g. Newton/meter**2, dimensionless '-'; a PRESERVED
   extracted unit stays EXACTLY as the source writes it (LaTeX
   fractions included) and keeps unit_source "extracted" — the code
   generator converts notation, never the spec; descriptions must match the standard meaning of the symbol
   (in elasticity lambda/mu are the LAME parameters, E is Young's
   modulus, nu Poisson's ratio — correct a wrong extracted description,
   marking it designed).  ell/Afac is an ordinary constant: preserve
   the extracted value, or design a reasonable initial value when the
   source gives none (the pipeline calibrates it later).
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
   pen = 1e15 penalty terms appear ONLY as a FreeFEM WORKAROUND
   when a field must enforce TWO OR MORE DIFFERENT Dirichlet
   values (FreeFEM cannot express that in one field declaration,
   e.g. u = 0 on one edge and u = 10 on another); in that case
   write the penalty in the state weak form, the adjoint weak
   form AND the shape derivative with the SAME pen = 1e15, and
   list pen in constants.  A penalty term for a SINGLE-valued
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
   multi-valued-Dirichlet penalty terms carry pen = 1e15 — the
   objective weight keeps its own symbol.
8. optimization_config: mesh_file "mesh.txt"; hmin < hmax resolving this
   problem's features and the interface; diffusion and v0 sane for this
   problem's scale; iterations 100; phi (phi < 0 = solid; negative
   somewhere in the domain; solid connects every loaded/supported
   boundary; fixed boundaries covered by solid at iteration 0; initial
   solid fraction ~20-80%; a phi prescribed by the problem is used
   exactly); the boundary LISTS carry the DSL's THREE-way boundary
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
- Phi violations (each is an ERROR): never negative in the domain;
  solid fraction outside 20-80%; the solid region DISCONNECTED (e.g.
  two separated strips) — it must be ONE connected piece; fixed labels
  uncovered (not solid along their ENTIRE length) at iteration 0; a
  prescribed phi not used exactly.
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
- A unit marked DESIGNED but not written in AutoFreeFEM syntax
  (CAPITALIZED names — Newton/Watt/Pascal/meter/kelvin — compound
  units like Newton/meter**2, dimensionless '-').  The fix is EITHER
  rewriting the unit in AutoFreeFEM syntax OR marking it unit_source
  "extracted" when it reproduces the source's own notation verbatim
  (extracted units stay verbatim — the code generator converts
  notation).
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

Return ONE JSON:
{{"verified": true, "issues": []}}
or
{{"verified": false, "issues": [{{"item": "<spec:key>", "severity": "error"|"warning", "problem": "<one sentence>"}}]}}

No markdown, no code fences.
"""

_REVIEW_USER = """\
Original input files:
{files_block}

Generated specs:
{specs_json}

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
reintroduce them).

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


def _history_text(history):
    """One compact block per previous round: the errors, what the round
    FIXED, what it FAILED to fix, and the changes."""
    lines = []
    for h in history:
        errs = "; ".join(str(e)[:100] for e in h["errors"][:5])
        fixed = "; ".join(str(x)[:60] for x in h.get("fixed", [])[:4])
        failed = "; ".join(str(x)[:60] for x in h.get("failed", [])[:4])
        lines.append(
            f"round {h['round']}: errors=[{errs}] fixed=[{fixed}] "
            f"failed=[{failed}] removed={h['removed'][:8]} "
            f"added={h['added'][:8]} modified={h['modified'][:8]}"
        )
    return "\n".join(lines) if lines else "(no previous rounds)"


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
    import re as _re
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
    # cross-fix: when the active mesh is square(), FreeFEM's implicit
    # labels 1-4 MUST be part of the extracted mesh label universe
    if _re.search(r"\bmesh\s+Th\s*=\s*square\s*\(", mex.get("mesh_code") or ""):
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
    # sometimes misses a disconnected initial design)
    code = (mds.get("mesh_code") or "") or (mex.get("mesh_code") or "")
    L, H = _mesh_box(code)
    opt = ds.get("optimization_config")
    if isinstance(opt, dict):
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


def _review(agent, files, specs):
    """The merged LLM review — returns error-severity issues only.

    An unparseable reply goes through ONE JSON-repair round (the raw
    text is sent back and the model must return it as valid JSON); only
    when that ALSO fails does the review fail open."""
    result = None
    first_error = "no reply"
    raw = ""
    try:
        raw = _invoke(
            agent, _REVIEW_SYSTEM, _REVIEW_USER,
            {
                "files_block": files,
                "specs_json": json.dumps(specs, indent=2, ensure_ascii=False),
            },
        )
        result = json.loads(parser_json(raw))
    except Exception as e:
        first_error = str(e)
    if result is None and raw:
        # ONE JSON-repair round: the invalid text goes back verbatim —
        # a failed review must never silently pass the spec
        try:
            set_label(agent, "Initialization/run_spec_generation (review JSON repair)")
            raw = _invoke(
                agent, _JSON_REPAIR_SYSTEM, _JSON_REPAIR_USER,
                {"raw_text": raw},
            )
            result = json.loads(parser_json(raw))
        except Exception as e2:
            warnings.warn(
                f"spec review failed after JSON repair (fail-open): "
                f"{first_error} / {e2}"
            )
            return []
    if result is None:
        warnings.warn(f"spec review failed (fail-open): {first_error}")
        return []
    if not isinstance(result, dict) or _as_bool(result.get("verified")) is not False:
        return []
    return [
        f"[{it.get('severity', 'error')}] {it.get('item', '?')}: {it.get('problem', '')}"
        for it in (result.get("issues") or [])
        if isinstance(it, dict) and it.get("severity", "error") == "error"
    ]


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


def _mesh_syntax_problems(code: str) -> list:
    """Deterministic syntax sanity of a provided mesh code.

    Returns a list of problems (empty = OK).  This is the ONE syntax
    check a provided mesh.txt receives — never a content review.
    """
    problems = []
    if not code.strip():
        return ["empty mesh code"]
    if "mesh Th" not in code and not re.search(
        r"\b(?:buildmesh|square|border)\s*\(", code
    ):
        problems.append(
            "no mesh definition found (expected `mesh Th =` with "
            "buildmesh / square / border)"
        )
    for opening, closing in (("(", ")"), ("{", "}"), ("[", "]")):
        if code.count(opening) != code.count(closing):
            problems.append(f"unbalanced {opening}{closing} delimiters")
    return problems


def _mesh_syntax_pass(agent, code: str, problem_name=None) -> tuple:
    """The ONE syntax check + at most ONE syntax repair for a PROVIDED
    mesh.txt.

    Returns ``(mesh_code, repaired, problems_text)`` — the (possibly
    syntax-repaired) frozen mesh code, whether the LLM repair was
    applied, and the remaining problems ("" when clean).  Content is
    NEVER touched: the repair fixes syntax only, and it is accepted
    only when it strictly reduces the problem count.
    """
    problems = _mesh_syntax_problems(code)
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
        remaining = _mesh_syntax_problems(fixed)
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


def run_spec_generation(agent, input_files, problem_name=None, max_fixes=None,
                        output_dir=None, history_log=None):
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

    for attempt in range(1 + max_fixes):
        was_repair = specs is not None
        prev_issues = list(issues) if issues else []
        round_label = f"Initialization/run_spec_generation round {attempt + 1}/{max_fixes + 1}"
        if specs is None:
            # fresh generation (or re-generation after an unparseable answer)
            set_label(agent, f"{round_label} (generate)")
            raw = _invoke(
                agent, _GENERATE_SYSTEM, _GENERATE_USER,
                {"problem_name": problem_name or "unknown", "files_block": files},
            )
        else:
            issues_text = format_errors(issues) if issues else ""
            set_label(agent, f"{round_label} (repair)")
            raw = _invoke(
                agent, _REPAIR_SYSTEM, _REPAIR_USER,
                {
                    "files_block": files,
                    "specs_json": json.dumps(specs, indent=2, ensure_ascii=False),
                    "issues_text": issues_text,
                    "history_text": _history_text(history),
                },
            )
        data = _parse_two_specs(raw)
        if data is None:
            issues = [{"check": "json_parse", "level": "syntax",
                       "severity": "error", "message": "LLM output failed to parse"}]
            print(
                f"  [spec loop {attempt}/{max_fixes}] unparseable answer — repairing",
                flush=True,
            )
            history.append({
                "round": attempt + 1,
                "errors": ["unparseable answer"],
                "fixed": [], "failed": [], "unverified": [],
                "removed": [], "added": [], "modified": [],
            })
            continue
        prev = specs
        specs = _normalize_two_specs(data)
        if frozen_mesh is not None:
            _freeze_provided_mesh(specs, frozen_mesh, frozen_repaired, frozen_problems)
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
        # history entry, whose repair just produced this state.
        if was_repair and prev_issues and history:
            det_keys = {(c.get("check"), c.get("message")) for c in checks}
            fixed_list, failed_list, unverified = [], [], []
            for i in prev_issues:
                label = str(i.get("message"))[:120]
                if i.get("check") == "review":
                    unverified.append(label)
                elif (i.get("check"), i.get("message")) in det_keys:
                    failed_list.append(label)
                else:
                    fixed_list.append(label)
            last = history[-1]
            last["fixed"] = fixed_list
            last["failed"] = failed_list
            last["unverified"] = unverified
        issues = hard_errors(checks)
        if not issues:
            review_errors = _review(agent, files, specs)
            if review_errors:
                issues = [
                    {"check": "review", "level": "review", "severity": "error",
                     "message": m}
                    for m in review_errors
                ]
        if not issues:
            print(f"  [spec loop] PASSED after {attempt} repair(s)", flush=True)
            specs["_passed"] = True
            return specs
        print(
            f"  [spec loop {attempt}/{max_fixes}] {len(issues)} issue(s) — repairing",
            flush=True,
        )
        # ascii-safe console output: issue messages may quote LaTeX glyphs
        # that a cp1252 console cannot encode
        for i in issues:
            line = f"    - {i.get('level')} | {i.get('check')} | {i.get('message')}"
            print(line.encode("ascii", errors="replace").decode(), flush=True)

        # ── repair history for the NEXT repair ────────────────────────
        if prev is not None:
            removed, added, modified = _diff_specs(prev, specs)
        else:
            removed, added, modified = [], [], ["(initial generation)"]
        history.append({
            "round": attempt + 1,
            "errors": [str(i.get("message"))[:120] for i in issues],
            "removed": removed,
            "added": added,
            "modified": modified,
        })

    warnings.warn("spec_generation: repair limit reached — returning the last state")
    if specs is None:
        return {"initial": {}, "mesh": {}, "_passed": False}
    specs["_passed"] = False
    return specs
