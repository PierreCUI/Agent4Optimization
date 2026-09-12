"""
Centralised label conventions — single source of truth for all label-related
rules, semantics, and validation logic.

Imported by coder.py (generation prompt), check_instatic.py (static checks),
and Reflection/prompts.py (fixer guidance).  Every label rule lives HERE;
other files reference this module instead of duplicating rules.

Boundary terminology — boundaries are divided by LABEL: non-10
boundaries vs the label-10 boundary.  (The old internal/external
wording was inaccurate: a boundary can lie inside the domain yet NOT be
label 10, e.g. a multi-domain interface.)

1. Design role (how the shape optimisation treats the boundary):
   design boundary = movable   → boundaryLabels
   fixed  boundary = frozen    → fixedLabels
   (design roles apply to NON-10 boundaries.)

2. Physical condition (what the physics prescribes on the boundary):
   Dirichlet (physics-fixed value), Neumann (load/flux; a zero load is a
   free boundary — adiabatic in thermal problems, traction-free in
   mechanics), Robin (mixed field+flux combination).
   EXCEPTION — design Dirichlet: a Dirichlet boundary that is a DESIGN
   boundary (movable, in boundaryLabels) is implemented as a PENALTY
   ROBIN condition (see DESIGN_DIRICHLET_PENALTY below): moving-
   Dirichlet shape derivatives are not implemented.

3. The LABEL-10 boundary is the MATERIAL boundary (holes / the
   solid-void boundary of the level set inside the domain): design by
   default (it moves with the level set), NO physical condition, mesh
   label ALWAYS 10 (a FreeFEM requirement) — never listed in
   boundaryLabels or fixedLabels.

4. Multi-domain problems: problems with TWO OR MORE computational
   domains (one per material / region).  Their shared boundary is an
   INTERFACE — wherever it lies (inside the domain or on the outer
   boundary).  An interface is a NON-10 boundary: it has its own design
   role (design OR fixed, decided per problem — NOT design by default
   like the label-10 boundary) and its own physical condition, which is
   one of two CONTINUITY conditions (value continuity, e.g. T1 = T2 in
   diffusion; normal-flux continuity, e.g. k1 grad(T1)·n = k2
   grad(T2)·n) — not a Dirichlet/Neumann/Robin condition (those are for
   non-10 boundaries only).
"""

# =========================================================================
# 1.  dsx label semantics — the most frequently misunderstood rule
# =========================================================================

DSX_LABEL_SEMANTICS = """\
**CRITICAL — dsx label semantics**:  The third argument to ``dsx`` is
**NOT a raw mesh label** when the Domain has Boundary objects.  It is an
**index into the Domain's boundary list**:

```python
# WITH Boundaries declared:
bndD = Boundary(r'\\Gamma_D', 'ThL', '2/4')   # labels 2 and 4
Th = Domain('D', 'Th', bndD)                   # bndD = domain.args[2]
dsx(expr, Th, 0)   # '0' → index 0 → bndD → generates int1d(Th, 2,4)

# WITHOUT Boundaries — use RAW mesh labels:
Th = Domain('D', 'Th')
dsx(expr, Th, 2)   # literal label 2
dsx(expr, Th, 4)   # literal label 4
dsx(expr, Th, 3)   # literal label 3

# Label 999 (the evolving level-set boundary) is ALWAYS a raw label —
# use as a string literal regardless of whether Boundaries are declared:
dsx(expr, Th, '999')   # always generates int1d(Th, 999)
```

**Rule**: If you use Boundary objects, ``dsx`` indices (0, 1, 2, ...)
refer to the Boundary's position in Domain().  If you do NOT declare
Boundaries, use the actual mesh label numbers directly.  **Never mix
the two approaches** — copying a Boundary-index label from a
reference example without copying its Boundary declarations will
silently break (label 0 usually does not exist in the mesh)."""

DSX_LABEL_DIAGNOSIS = """\
**dsx label semantics (CRITICAL for label_assignment_error /
ff_label_semantics / physics_violation / pathological_objective)**:
AutoFreeFEM's ``dsx(expr, Th, X)`` has TWO different behaviours depending
on whether the Domain has Boundary objects:
* WITH Boundaries: ``X`` is a 0-based INDEX into Domain's boundary list.
  Example: ``bndD = Boundary(..., '2/4'); Th = Domain('D','Th', bndD)``
  then ``dsx(..., Th, 0)`` → refers to bndD → generates ``int1d(Th, 2,4)``.
* WITHOUT Boundaries: ``X`` is a RAW MESH LABEL.
  Example: ``Th = Domain('D','Th')`` then ``dsx(..., Th, 2)`` →
  generates ``int1d(Th, 2)``.
* **Mixing these two causes SILENT failures**: if code copies a Boundary
  index (e.g. 0) from a reference example but does NOT declare Boundaries,
  the label 0 is treated as a raw mesh label.  Since most meshes have
  NO label 0, ``int1d(Th, 0)(...)`` silently evaluates to 0 in FreeFEM,
  making penalty/spring terms ineffective and producing pathological
  objectives (e.g. -1e13, NaN, or near-zero gradients).
* **Diagnosis**: if the objective is huge/small/stagnant and the code
  uses integer labels 0, 1, 2, ... in dsx without Boundary declarations,
  the labels are likely wrong.  Check whether the mesh actually has those
  labels (read mesh.txt).  If not, either add Boundary declarations or
  change the labels to actual mesh labels.
* **Fix**: compare the dsx labels to the mesh labels in mesh.txt.
  Use labels that actually exist in the mesh, or define Boundary objects
  and use their indices.  For compliance-type problems (no target
  shape), the spring is typically on the fixed (clamped) edge labels —
  the Dirichlet-fixed non-10 boundaries, never on the label-10
  boundary."""


# =========================================================================
# 2.  boundaryLabels vs fixedLabels — the DESIGN role of NON-10 boundaries
# =========================================================================

BOUNDARY_VS_FIXED_LABELS = """\
**IMPORTANT — design role vs physical condition (two INDEPENDENT aspects
of a NON-10 boundary — do not confuse them)**:

1. **Design role** — how the shape optimisation treats the boundary:
   - ``boundaryLabels``: **design boundaries** — non-10 boundaries that
     are FREE TO MOVE.  The shape derivative drives the shape evolution on
     these boundaries.
   - ``fixedLabels``: **fixed boundaries** — non-10 boundaries whose
     position is FROZEN during the optimisation.
   Convention: Dirichlet (physics-fixed) boundaries are normally fixed;
   Neumann / Robin (load / mixed) boundaries are normally design.
   If unsure, place the label in fixedLabels.

2. **Physical condition** — what the physics prescribes ON the boundary
   (recorded as the boundary's ``kind`` in the ProblemSpec):
   - **Dirichlet**: the field value is prescribed (physics-fixed boundary).
   - **Neumann**: a flux / load is prescribed (load boundary).  When the
     load is zero it is a **free boundary** — adiabatic in thermal
     problems, traction-free in solid mechanics — and it is still Neumann.
   - **Robin**: a linear combination of the field and its flux (mixed).

A label MUST NOT appear in BOTH lists — it is either design or fixed,
never both.
Check the ProblemSpec's ``boundaries`` array: each boundary has a
``kind`` (physical condition) and a ``label``.  Dirichlet → fixedLabels.
Neumann / Robin → boundaryLabels.
**EXCEPTION — design Dirichlet**: a Dirichlet boundary whose design
role is DESIGN (movable) is implemented as a penalty Robin condition
(see the DESIGN DIRICHLET rules below) — its label then goes in
boundaryLabels as a Robin design boundary, and the ScalarField gets NO
Dirichlet argument for it."""

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
  value, ALWAYS use it for design-Dirichlet penalty terms.  With
  k0 ~ 200 W/(m K) and a domain size ~ 1 m, the boundary Biot number
  pen*L/k0 ~ 5e12 holds the boundary temperature within ~1e-10 % of
  u_D.  Do NOT use the FreeFEM tgv scale (1e30) — it would dominate
  the shape derivative and make the optimisation ill-conditioned.
- The label is then a ROBIN design boundary: it goes in
  ``boundaryLabels`` (never fixedLabels), and the ScalarField gets NO
  Dirichlet argument for it.
- The prescribed value u_D enters ONLY through the penalty weak-form
  term (plus the temperature-integral weight where the objective needs
  it) — never through the field declaration.
- A Dirichlet boundary that is FIXED (design role fixed) keeps the
  normal ScalarField declaration:
  ``ScalarField('u', 'P2', Th, '0.', [label], kelvin)``.
"""

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
* Without an explicit design_role, the physical-kind convention applies:
  * Dirichlet boundaries (the field value is prescribed — physics-fixed,
    e.g. clamped edges) → fixedLabels.  The shape CANNOT move on these
    boundaries — if they appear in boundaryLabels the shape derivative
    will try to deform a clamped edge.
  * Neumann / Robin boundaries (load / mixed; a zero-load Neumann is a
    free boundary — adiabatic in thermal problems, traction-free in
    mechanics) → boundaryLabels.  The shape CAN evolve on these
    boundaries.
* A label MUST NOT appear in BOTH lists — it is either movable or fixed,
  never both.  If both contain the same label, remove it from
  boundaryLabels (keep the Dirichlet label in fixedLabels).
* The label-10 boundary (material boundary / holes) is design
  by default, carries NO physical condition, and NEVER appears in either
  list — both lists are for non-10 boundaries only.
* Check the ProblemSpec's ``boundaries`` array in the analysis artifacts:
  each boundary has a ``kind`` (physical condition), an optional
  ``design_role`` (explicit design role — follow it exactly), and a
  ``label`` number.  Map Dirichlet → fixedLabels,
  Neumann / Robin → boundaryLabels only when no design_role is given.
* Fix: edit setup_optimization() — move Dirichlet labels out of
  boundaryLabels and into fixedLabels.  Leave Neumann / Robin labels in
  boundaryLabels, EXCEPT those with an explicit design_role='fixed'.
* **Design Dirichlet — penalty Robin**: a Dirichlet boundary that the
  problem (or an explicit design_role) makes a DESIGN boundary must be
  implemented as a penalty Robin condition (see the DESIGN DIRICHLET
  rules): replace the ScalarField Dirichlet value with a weak-form term
  ``dsx(pen * (u - u_D) * testu, Th, label)`` with pen ~ 1e8, and keep
  the label in boundaryLabels."""


# =========================================================================
# 3.  The label-10 boundary (material boundary), and special labels 999 / region 3
# =========================================================================

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
- ``show_weak_material`` is **always** forced to ``False`` by Stage 5.5
  (initial-parameter review) — never rely on ``True`` for two distinct
  characteristic functions or boundary terms on label 999; express them
  through the Lagrangian structure itself.
- NOTE: the label-10 boundary lives WITHIN a single computational domain.
  It is distinct from the INTERFACES of multi-domain problems — the
  shared boundaries BETWEEN two or more computational domains, which
  are NON-10 boundaries with their own design role and continuity
  conditions (see below)."""


# =========================================================================
# 3b.  Multi-domain problems — interfaces between computational domains
# =========================================================================

MULTI_DOMAIN_AND_INTERFACE = """\
**IMPORTANT — multi-domain problems and interfaces**:
- Most problems use ONE computational domain (``Th``).  Some problems —
  e.g. an optimization of TWO materials together — need MULTIPLE
  computational domains, one per material / region (``Th1``, ``Th2``,
  ...).
- An **interface** is the shared boundary of two or more distinct
  computational domains — wherever it lies: inside the domain or on the
  outer boundary.
- **Design role of an interface**: an interface may be a DESIGN boundary
  (movable → ``boundaryLabels``) or a FIXED boundary (frozen →
  ``fixedLabels``), decided per problem.  Unlike the label-10 boundary,
  an interface is NOT design by default.
- **Physical condition of an interface**: exactly one of the two
  CONTINUITY conditions between the fields on the two sides of the
  interface (NOT a Dirichlet / Neumann / Robin condition — those apply
  to NON-10 boundaries only):
  1. **Value continuity**: the field value is continuous across the
     interface, e.g. ``T1 = T2`` in diffusion.
  2. **Normal-flux continuity**: the normal flux is continuous across
     the interface, e.g. ``k1 * grad(T1)·n = k2 * grad(T2)·n`` in
     diffusion.
- Do NOT confuse an interface with the label-10 boundary:
  * the label-10 boundary is the solid/void boundary carved by the
    level set WITHIN one domain — design by default, no physical
    condition;
  * an interface is the shared boundary BETWEEN two computational
    domains — a NON-10 boundary, design or fixed, with a continuity
    condition.
- In the mesh, an interface edge needs its own mesh label (chosen in the
  mesh design, distinct from the other non-10 boundary labels and from
  label 10)."""

# =========================================================================
# 4.  Auto-fix instructions (for static checker)
# =========================================================================

AUTO_FIX_LABEL_INSTRUCTIONS = """\
- If the error is about label mismatches → align with mesh labels.
- If the error is about labels_consistency → fix the DESIGN role of the
  non-10 boundaries: an explicit design_role in the ProblemSpec is
  ground truth and must be matched exactly; WITHOUT one, move Dirichlet
  (physics-fixed) labels from boundaryLabels to fixedLabels and
  Neumann / Robin (load / mixed) labels from fixedLabels to
  boundaryLabels; ensure no label appears in both lists simultaneously.
  A Dirichlet boundary that IS a design boundary must be implemented
  as a penalty Robin condition (DESIGN DIRICHLET rules) — its label
  then legitimately stays in boundaryLabels.
- If the error is about internal_boundary_labels → label 10 is the
  label-10 (material) boundary (design by default, no physical
  condition) — it never belongs in boundaryLabels or fixedLabels;
  remove it from both.
- If the error is about dsx_labels → the dsx(..., Th, X) label X must be
  EITHER a Boundary index (when Domain has Boundary objects declared)
  OR a raw mesh label (when Domain has NO Boundaries).  Check which case
  applies and fix: either add Boundary declarations and use indices,
  or remove Boundaries and use raw mesh labels that exist in the mesh file."""


# =========================================================================
# NOTE: Individual constants above are imported directly by coder.py,
# check_instatic.py, and Reflection/prompts.py.
# =========================================================================
