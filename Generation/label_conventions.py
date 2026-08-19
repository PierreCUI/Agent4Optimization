"""
Centralised label conventions — single source of truth for all label-related
rules, semantics, and validation logic.

Imported by coder.py (generation prompt), check_instatic.py (static checks),
and Reflection/prompts.py (fixer guidance).  Every label rule lives HERE;
other files reference this module instead of duplicating rules.
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

# Label 999 (level-set interface) is ALWAYS a raw label — use as a
# string literal regardless of whether Boundaries are declared:
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
  shape), the spring is typically on the fixed/clamped edge labels."""


# =========================================================================
# 2.  boundaryLabels vs fixedLabels
# =========================================================================

BOUNDARY_VS_FIXED_LABELS = """\
**IMPORTANT — label semantics**:
- ``boundaryLabels``: mesh labels where the level-set shape CAN move
  (typically Neumann / natural / free boundaries).  The shape derivative
  drives the shape evolution on these boundaries.
- ``fixedLabels``: mesh labels where the shape is FROZEN (typically
  Dirichlet boundaries or boundaries where the physics demands a fixed
  position, like clamped edges).
- A label MUST NOT appear in BOTH lists — it is either optimisable or
  fixed, never both.
- Check the ProblemSpec's ``boundaries`` array: each boundary has a
  ``kind`` (Dirichlet/Neumann) and a ``label``.  Dirichlet → fixedLabels.
  Neumann → boundaryLabels.  If unsure, place it in fixedLabels."""

LABEL_CONSISTENCY_FIX = """\
Label assignment fix rules:
* Dirichlet boundaries (where u/field is fixed to a value) → fixedLabels.
  The shape CANNOT move on these boundaries — if they appear in
  boundaryLabels the shape derivative will try to deform a clamped edge.
* Neumann boundaries (free / natural / zero-flux) → boundaryLabels.
  The shape CAN evolve on these boundaries.
* A label MUST NOT appear in BOTH lists — it is either optimisable or
  fixed, never both.  If both contain the same label, remove it from
  boundaryLabels (keep the Dirichlet label in fixedLabels).
* Check the ProblemSpec's ``boundaries`` array in the analysis artifacts:
  each boundary has a ``kind`` (Dirichlet/Neumann) and a ``label`` number.
  Map Dirichlet → fixedLabels, Neumann → boundaryLabels.
* Fix: edit setup_optimization() — move Dirichlet labels out of
  boundaryLabels and into fixedLabels.  Leave Neumann labels in
  boundaryLabels."""


# =========================================================================
# 3.  Special labels — 999 and region 3
# =========================================================================

LABEL_999_AND_REGION3 = """\
**IMPORTANT — MMG / AutoFreeFEM conventions**:
- **Label 999** is the level-set interface (solid/void boundary) tracked
  by MMG.  Use it for interface conditions like spring forces that prevent
  rigid-body motion: ``dsx(..., Th, '999')``.
- **Region 3** is the void/weak-material region.  Meshes with internal
  holes use ``int[int] re = [0,3]; Th = change(Th, reft=re)`` to mark
  the hole as region 3.
- Use ``show_weak_material=True`` when the problem has two distinct
  characteristic functions or interface terms on label 999."""

# =========================================================================
# 4.  Auto-fix instructions (for static checker)
# =========================================================================

AUTO_FIX_LABEL_INSTRUCTIONS = """\
- If the error is about label mismatches → align with mesh labels.
- If the error is about labels_consistency → move Dirichlet labels from
  boundaryLabels to fixedLabels; remove Neumann labels from fixedLabels;
  ensure no label appears in both lists simultaneously.
- If the error is about dsx_labels → the dsx(..., Th, X) label X must be
  EITHER a Boundary index (when Domain has Boundary objects declared)
  OR a raw mesh label (when Domain has NO Boundaries).  Check which case
  applies and fix: either add Boundary declarations and use indices,
  or remove Boundaries and use raw mesh labels that exist in the mesh file."""


# =========================================================================
# NOTE: Individual constants above are imported directly by coder.py,
# check_instatic.py, and Reflection/prompts.py.
# =========================================================================
