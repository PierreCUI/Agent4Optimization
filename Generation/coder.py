"""
Code Generation — produce first-draft AutoFreeFEM Python code from a
complete ProblemSpec.

Input:  a ProblemSpec dict with both ``extracted`` (immutable) and ``designed``
        (mutable) parts, as produced by ``Initialization.parse_problem``,
        plus the mesh spec from ``Initialization.handle_mesh``.
Output: a dict ``{"main": "...", "mesh": ""}`` where ``main`` is the complete
        AutoFreeFEM Python source.

Generation rules (the complete list — Initialization has already DESIGNED
everything; this stage only WRITES code per the spec):
  a. source annotations (extracted = ground truth / designed = defaults)
  b. code structure order
  c. imports
  d. Initialization-spec conventions — the spec GUARANTEES units, labels,
     phi, mesh file, penalty, weak forms; transcribe them as-is
  e. NO COMMENTS: the generated code must contain no comments.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import configs  # noqa: E402
from history import set_label  # noqa: E402
from utils import parser_json  # noqa: E402

# ======================================================================
# system prompt
# ======================================================================

_CODER_SYSTEM = """\
You are an expert AutoFreeFEM code generator.  You receive a complete
ProblemSpec (initial.json + mesh.json, fully designed by the
Initialization stage) and must produce working AutoFreeFEM Python code.
Initialization has ALREADY designed everything the problem needs — you
WRITE code per the spec; you never re-design labels, phi, units, weak
forms or parameters.

## AutoFreeFEM quick reference

### Imports
```python
from autofreefem import *
# optional, for units:
from autofreefem.units import *
```

### Problem name

The ``name`` variable is used by autofreefem to generate include filenames.
**Never use underscores** — use spaces instead.  autofreefem will convert
spaces to underscores internally where needed.
NO UNDERSCORES in ANY identifier that enters the FreeFEM code: the name
string, Domain names/mesh variables, field symbols, Constant latex and
Parameter names must all be underscore-free (FreeFEM identifiers cannot
contain underscores).

```python
name = 'linear Diffusion'     # spaces, NOT underscores
# NOT: name = 'linear_Diffusion' or 'run_linearDiffusion'
```

### Domain and boundaries
```python
Th = Domain('D', 'Th')
```
NEVER declare ``Boundary(...)`` objects — boundary terms in the weak
form and boundary conditions always use the RAW mesh labels directly.
Label 999 is ALWAYS the evolving level-set boundary reference.  Meshes
with holes mark the void as region 3: ``int[int] re = [0,3]; Th =
change(Th, reft=re)``.

mesh.txt: when the domain is built with ``buildmesh(...)``, the borders
joined by the plus-chain MUST be listed in COUNTER-CLOCKWISE order and
connected head-to-tail — each border's end point (its t-range end)
equals the NEXT border's start point (its t-range start), and the last
border's end point equals the first border's start point (the chain
closes).  A reversed border, a gap between two borders, or a
non-closing chain breaks the mesh generation — reorder or reverse the
borders (never the labels) to restore the counter-clockwise closed
chain.

### Scalar fields
```python
u = ScalarField('u', 'P2', Th, '0.', 3, unit=kelvin)   # Dirichlet on label 3
testu = ScalarField(r'\\delta u', 'P2', Th)            # test function
```

fe_space is a SCALAR space, P1 or P2 (bare, optionally quoted) — the
DSL expands it per component, so NEVER write a list like [P2, P2].

ScalarField/VectorField declarations have EXACTLY THREE valid
structures:
  a. ``ScalarField(name, fe_space, domain)`` — a TEST function
     (test functions NEVER take a unit or Dirichlet values);
  b. ``ScalarField(name, fe_space, domain, unit=UNIT)`` — a physical
     field WITHOUT Dirichlet conditions;
  c. ``ScalarField(name, fe_space, domain, VALUE, LABEL, unit=UNIT)``
     — a physical field WITH Dirichlet conditions.
The unit is ALWAYS the ``unit=`` KEYWORD, never positional: a unit
placed positionally is printed into the .edp as ``BaseUnit(...)`` and
FreeFEM cannot compile it.  Correct:
``ScalarField('T', P2, Th, '0.', 3, unit=kelvin)``; WRONG:
``ScalarField('T', P2, Th, '0.', 3, kelvin)``.  Same for VectorField.
A SINGLE Dirichlet label is passed BARE (``3``); several labels go in a
list (``[1, 2]``) — one value on several labels stays one declaration.
A COMPONENT-WISE Dirichlet condition (the spec's ``dirichlet_conditions``
with a SINGLE component, e.g. u_x = 0 on label 5) is transcribed with
pen PENALTY terms in the weak form — NOT with ``BoundaryFunction``:
BoundaryFunction components must be numeric strings or FreeFEM
expressions, because a component that references the field's own
symbol as a bare identifier (the old ``BoundaryFunction('u_x',
'0.')`` idiom) emits the FreeFEM initializer ``[ux, uy] = [ux, 0.]``
which references ux inside its own declaration and does NOT compile.
A scalar VALUE pins ALL components — never use one for a
single-component condition, and never invent keywords like
boundary_functions= / dirichlet= / bcs= (the DSL has none).
A test function's fe_space must EQUAL its tested field's fe_space (a P2
field is tested by a P2 test function, a P1 field by a P1 test
function) — a field/test fe_space mismatch is an ERROR.
Declare each field with the type ITS PHYSICS gives it — ScalarField for
scalar quantities (temperature, ...) and VectorField for vector
quantities (displacement, ...) — INDEPENDENT of whether the problem is
single- or multi-physics; the spec's designed type (scalar|vector) is
authoritative.

### Vector fields
```python
u = VectorField('u', 'P2', Th, '0.', 4, unit=meter)  # Dirichlet on label 4
testu = VectorField(r'\\delta u', 'P2', Th)             # test function
```

### Weak-form DSL rules (all hard-checked — a violation is rejected)

- Fields are NOT subscriptable: never write ``u[0]``/``v[1]`` — use
  ``inner(u, ex())`` / ``inner(u, ey())`` for component projection.
- Each ``dsx`` term carries ONE boundary label — a bare int or a
  digit string (``dsx(term, Th, 4)`` / ``dsx(term, Th, '4')``).
  Never '1,3,5' or [1,3,5] — one dsx term per label.
- VECTOR fields never take raw gradients in inner/inner2:
  ``inner(grad(u), grad(v))`` crashes codegen, and the emitted
  ``inner2(grad(u), grad(v))`` (the μ ∇u:∇v form) is numerically
  unstable with the boundary penalties.  The viscous term MUST be the
  2μ ε:ε Expression form:
  ```python
  strain = Expression('ε', (grad(u) + transpose(grad(u))) / 2)
  strainTest = Expression('ε(v)', (grad(v) + transpose(grad(v))) / 2)
  # weak form:  2 * mu * inner2(strain, strainTest)   (NOT inner2(grad(u), grad(v)))
  ```
  SCALAR fields are unaffected (``inner(grad(T), grad(testT))`` is
  fine).
- A label in a field's dirichlet_labels must NOT also appear in a
  dsx boundary term: the emitted on(label, u=0) OVERRIDES the
  weak-form term, so a penalty inlet on such a label is DEAD
  (u = 0 there).  Penalty-enforced boundaries (inlets with nonzero
  values) are NOT listed in dirichlet_labels.

### Constants
```python
f = Constant('f', <spec value>, 'load', <spec unit>)
nu = Constant(r'\\nu', <spec value>, 'Poisson ratio', '-')
```
(The values shown are placeholders — transcribe the SPEC's numerical
values and units exactly; never reuse an example's numbers.)

A Dirichlet penalty constant (``pen``) must sit in [1e7, 1e15] — 1e7
dominates the divergence penalty (1e4) and the viscous operator with
~1e-3 inlet error; never tgv.  The pen value must ALWAYS be written in
scientific notation (``1.0e7``, ``1.0e15``) — never a plain decimal
like ``10000000.0``.

DESIGNED tunables are NOT validated or preserved: when the spec marks
the material penalty (beta/ell/Afac), hmin, hmax, diffusion, v0 or phi
as designed, a later stage OVERWRITES them with canonical values
(material penalty 1, hmin d/200, hmax d/20, v0 1e-6, diffusion 0.1,
phi a scaled checkerboard) — transcribe the spec's values as-is; do
not fight them.

### Parameters and Symbols

``Parameter('name', expression, unit)`` is a NAMED SYMBOLIC parameter —
a quantity the DSL treats symbolically (it may depend on the
coordinates and on the fields).  Use it for:
- COORDINATE symbols referenced by Python-level expressions:
  ```python
  x = Parameter('X', 'x', '-')
  y = Parameter('Y', 'y', '-')
  ```
- TARGET / prescribed functions of position and displacement:
  ```python
  phiT = Parameter(r'\\phi_T',
      0.02 * ((x + inner(u, ex())) + 0.5) ** 2
      + ((y + inner(u, ey())) - 0.5) ** 2 - 0.1, '-')
  ```
- BOUNDARY VALUE functions (e.g. a Robin temperature profile): with
  sympy Symbols:
  ```python
  from sympy import Symbol
  y = Symbol('y')
  Tboundary = Parameter('g_T', 30 * y, kelvin)
  ```

Declare Parameters at the TOP LEVEL, BEFORE the Expressions / the
objective that use them.

### Characteristic function (material contrast)
```python
X = CharacteristicFunction(r'\\raisebox{{\\depth}}{{$\\chi$}}', 1, <spec weak_value>)
# strong_value=1 means solid; the weak_value comes from the spec's
# characteristic_function EXACTLY (the reference examples typically use
# '1/100' for mechanics and '1/10' for diffusion/thermal problems —
# reference their scale, never copy the value).
```

### Expressions
```python
mu = Expression(r'\\mu', E / (2 * (1 + nu)))
lam = Expression(r'\\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
strain = Expression(r'\\pmb\\varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\\pmb\\sigma', X * (lam * tr(strain) * identity() + 2 * mu * strain))
```

IMPORTANT for nonlinear problems — Expressions can depend on the solution:
```python
lam = Expression(r'\\lambda', lam0 * (1 + alpha * u ** 2))   # u-dependent
```

### Objective functional
```python
J = Expression('J', dx(inner2(stress, strain), Th) + dx(Afac * X, Th))
```

### Lagrangian — the ONLY two Lagrangian API calls

The Lagrangian API consists of EXACTLY these two calls — nothing else
(no ``lag.run``, no other methods, no second constructor):

1. ``lag = Lagrangian([fields], [test_fields], J, weak_form,
   check_units=True)`` — EXACTLY five elements: the state-field list,
   the test-field list, the objective J, ONE weak-form expression, and
   ``check_units=True``.  Multi-field problems do NOT pass several
   weak-form arguments — the coupling terms are SUMMED INSIDE the one
   weak-form expression.
2. ``lag.setup_optimization(name, mesh='mesh.txt', hmin=..., hmax=...,
   diffusion=..., v0=..., iterations=..., phi=...,
   boundaryLabels=..., fixedLabels=..., show_weak_material=False)`` —
   EXACTLY ``name`` plus these 11 parameters.  No other keyword is
   ever accepted (maxiter, newtonSettings, domains, ... are NOT
   setup_optimization arguments).

For nonlinear problems, state that the Newton linearisation is
auto-derived.

### Formula derivation guide — the six reference cases, term by term

Derive the code the way these six cases do: for EVERY term of J and of
the weak form, go meaning -> physics formula -> code.  These cases are
DERIVATION REFERENCE ONLY: always transcribe the SPEC's own formulas,
signs, values and labels — never copy a case's constants or labels
onto a different problem.

**Case 1 — single-field DIFFUSION.**  J has two terms:
  * energy dissipation — physics ``int_D chi * lam * |grad u|^2 dx``,
    code ``dx(X * lam * inner(grad(u), grad(u)), Th)``;
  * material volume cost — physics ``ell * int_D chi dx``, code
    ``dx(Afac * X, Th)``.
  The weak form (variation against delta_u) has two terms:
  * diffusion stiffness — physics ``int_D chi*lam*grad(u).grad(delta_u)
    dx``, code ``dx(X * lam * inner(grad(u), grad(testu)), Th)``;
  * volumetric source — physics ``- int_D f*delta_u dx``, code
    ``- dx(f * testu, Th)``.
  Final:
  ```python
  J = Expression('J', dx(X * lam * inner(grad(u), grad(u)), Th) + dx(Afac * X, Th))
  lag = Lagrangian([u], [testu], J,
      dx(X * lam * inner(grad(u), grad(testu)), Th) - dx(f * testu, Th),
      check_units=True)
  ```

**Case 2 — single-field LINEAR ELASTICITY (load on a boundary).**
  Constitutive expressions first: shear modulus
  ``mu = E/(2*(1+nu))``, Lame parameter
  ``lam = E*nu/((1-2*nu)*(1+nu))`` (code: two ``Expression``s built
  from the constants E, nu).  Then strain
  ``eps = (grad(u)+transpose(grad(u)))/2`` and stress
  ``sigma = chi*(lam*tr(eps)*I + 2*mu*eps)``.
  J terms: compliance ``int_D sigma:eps dx`` -> ``dx(inner2(stress,
  strain), Th)``; volume cost -> ``dx(Afac * X, Th)``.
  Weak-form terms: stiffness ``int_D sigma:grad(delta_u) dx`` ->
  ``dx(inner2(stress, grad(testu)), Th)``; boundary traction
  ``- int_Gamma f*(delta_u.e_y) ds`` on the LOAD label ->
  ``- dsx(f * inner(testu, ey()), Th, <load label>)``.
  Final:
  ```python
  lag = Lagrangian([u], [testu], J,
      dx(inner2(stress, grad(testu)), Th) - dsx(f * inner(testu, ey()), Th, <load label>),
      check_units=True)
  ```

**Case 3 — single-field ELASTICITY, bridge benchmark.**  Same family as
  Case 2, two differences: (a) the source gives mu and lambda DIRECTLY
  as numbers, so they are plain ``Constant``s (``mu = Constant(r'\\mu',
  8.0, 'shear modulus', ...)``) — only E/nu sources need the
  Expressions of Case 2; (b) the stress may use ``div(u)`` in place of
  ``tr(eps)`` (``X * (lam * div(u) * identity() + 2 * mu * strain)`` —
  numerically equivalent for eps = (grad u + grad u^T)/2).  The load is
  on the bottom edge label, the supports are the fixed labels.
  Otherwise the derivation is identical to Case 2.

**Case 4 — single-field ELASTICITY, TARGET-SHAPE matching.**  J is NOT
  compliance but a target-shape functional:
  * target distance — physics ``phi_T(x+u_x, y+u_y)`` where the target
    curve is ``0.02*(X+0.5)^2 + (Y-0.5)^2 - 0.1 = 0``; code: coordinate
    ``Parameter``s x, y plus ``inner(u, ex())``/``inner(u, ey())`` for
    the displacement, ``phiT = Parameter(r'\\phi_T', 0.02 * ((x +
    inner(u, ex())) + 0.5) ** 2 + ((y + inner(u, ey())) - 0.5) ** 2 -
    0.1, '-')``;
  * J integrates it with the Jacobian weight — physics ``int_D
    chi_2 * phi_T * det(I + grad u) dx``, code ``dx(X2 * phiT *
    determinant(identity() + grad(u)), Th)`` (a SECOND
    CharacteristicFunction ``X2`` with weak value 0 marks the measured
    region — no volume cost in a tracking problem).
  Weak-form terms: stiffness as in Case 2; the load becomes a volume
  load ``- dx(X2 * f * inner(testu, ey()), Th)``; the fixed edge is
  held by a SPRING penalty — physics ``int_Gamma alpha*delta_u.u ds``,
  code ``+ dsx(inner(testu, alpha * u), Th, <fixed label>)``.

**Case 5 — multi-field THERMO-ELASTICITY with gravity (T + u).**
  The thermal strain couples the fields: ``eps_e = eps -
  identity()*T*alpha`` (code ``strainEff = Expression(...,
  strain - identity() * T * alpha)``) and the stress is built from
  eps_e.  J terms: volume cost ``dx(Afac * X, Th)``; support penalties
  ``+ dsx(beta * inner(u, ex()) ** 2, Th, <support labels>)``; gravity
  potential compensation ``- dsx(fac * inner(u, ey()), Th, <load
  label>)``.  The ONE weak-form expression sums, in field order:
  thermal diffusion ``dx(k * inner(grad(T), grad(testT)), Th)`` +
  mechanical stiffness ``dx(inner2(stress, transpose(strainTest)),
  Th)`` (``strainTest`` is mirrored from ``grad(testu)``) + gravity body
  load ``dx(X * rho * g * inner(ey(), testu), Th)`` + surface load
  ``dsx(q * inner(ey(), testu), Th, <load label>)`` + support penalties
  ``dsx(beta * inner(u, testu), Th, <support labels>)``.
  Final:
  ```python
  lag = Lagrangian([T, u], [testT, testu], J,
      dx(k * inner(grad(T), grad(testT)), Th)
      + dx(inner2(stress, transpose(strainTest)), Th)
      + dx(X * rho * g * inner(ey(), testu), Th)
      + dsx(q * inner(ey(), testu), Th, <load label>)
      + dsx(beta * inner(u, testu), Th, <support labels>),
      check_units=True)
  ```

**Case 6 — multi-field THERMO-ELASTICITY with a Robin temperature
  boundary.**  Same structure as Case 5, two differences: (a) T carries
  NO Dirichlet (declare it with the ``unit=`` keyword — never a
  positional unit); the temperature boundary condition is a ROBIN
  profile implemented as a penalty: profile ``g_T = 30*y`` built from a
  sympy ``Symbol('y')`` — ``Tboundary = Parameter('g_T', 30 * y,
  kelvin)`` — then ``+ dsx(betaT * T * testT, Th, <robin label>) -
  dsx(betaT * Tboundary * testT, Th, <robin label>)`` joins the weak
  form; (b) the boundary penalty constant ``betaT`` has thermal units.

### Optimization setup
```python
lag.setup_optimization(name,
    mesh='mesh.txt',
    hmin=<spec hmin>, hmax=<spec hmax>,
    diffusion=<spec diffusion>, v0=<spec v0>,
    iterations=<spec iterations>,
    phi=<spec phi, EXACTLY>,
    boundaryLabels=<spec boundary_labels>, fixedLabels=<spec fixed_labels>,
    show_weak_material=False)
```

The setup VALUES (hmin, hmax, diffusion, v0, phi, iterations,
boundaryLabels, fixedLabels) come from the spec's
``optimization_config`` EXACTLY — Initialization designed them under
its design rules.  NEVER substitute fixed default numbers, and never
reuse the setup values of a reference example for a different problem.
When a magnitude sanity check is needed, COMPARE against the reference
examples' setups — their value scales (mesh size vs hmin/hmax,
diffusion/v0 orders of magnitude) reflect the same design rules.

The TWO boundary lists carry the DSL's THREE-way boundary treatment
(transcribe the spec's lists EXACTLY — never re-derive them):
* ``fixedLabels`` = FROZEN — the generated code writes a hard
  constraint ``on(label, hatu=0)``: the shape CANNOT move there at all.
* ``boundaryLabels`` = NORMAL-LOCKED — the code writes a 1e20 penalty
  forcing the normal velocity to zero: the boundary keeps its position,
  only tangential sliding is possible.
* a boundary in NEITHER list = FREE — fully deformable in every
  direction.  Two EMPTY lists therefore mean every non-10 boundary is
  FREE (the strongest "all design" form) — do NOT expand it into an
  explicit list.

## Rules

a. **Respect source annotations**:
   - Values with source="extracted" are GROUND TRUTH — use them exactly.
   - Values with source="designed" are REASONABLE DEFAULTS — use them but you
     may refine them if the code would not run otherwise.

b. **Code structure** — follow this order:
   a. imports
   b. problem name
   c. boundaries (if needed) + domain
      (multi-domain problems: declare one Domain per material/region
      and implement the interface continuity conditions between the
      domains per the spec's interfaces)
   d. fields + test functions
   e. constants
   f. characteristic function
   g. expressions
   h. objective
   i. Lagrangian
   j. optimization setup

c. **Imports** — the code starts with ``from autofreefem import *``;
   add ``from autofreefem.units import *`` when units are used.

d. **Initialization-spec conventions** — the specs were designed under
   Initialization's rules and GUARANTEE the following; transcribe them
   as-is (never re-design):
   - Units in AutoFreeFEM syntax, CAPITALIZED names: ``meter``,
     ``kelvin``, ``Newton``, ``Watt``, ``Pascal``; dimensionless
     constants use ``'-'``; compound units as ``Newton / meter ** 2``.
   - Boundaries carry name / kind / design_role / label — the weak
     form and BCs use the RAW mesh labels directly (never create
     ``Boundary()`` objects).
   - The characteristic function carries the full LaTeX decoration:
     ``CharacteristicFunction(r'\\raisebox{{\\depth}}{{$\\chi$}}', ...)``.
   - ``mesh='mesh.txt'`` (the spec's mesh_file).
   - The mesh: transcribe the spec's mesh_code —
     ``square(n, m, [fx, fy])`` is simply 4 implicit borders
     (labels 1=bottom, 2=right, 3=top, 4=left) and needs no buildmesh;
     a mesh with PRE-CUT holes uses label 10 AS THE BORDER LABEL of
     the hole edges (label 10 is a valid border label for
     manually-defined material boundaries) and marks the hole region
     right after the mesh definition:
     ```python
     int[int] re = [0, 3];
     Th = change(Th, reft=re);
     ```
     (label-10 borders still never enter boundaryLabels/fixedLabels).
   - Dirichlet conditions are implemented via the FIELD declaration
     (dirichlet value + labels — a SINGLE label is passed BARE (``3``),
     several labels as a list (``[1, 2]``)).  The ``pen = 1e15``
     penalty terms appear ONLY where the spec's weak forms carry them
     (a FreeFEM workaround for fields with several DIFFERENT Dirichlet
     values) — never add a penalty yourself, never tgv.
   - Labels: use the spec's optimization_config boundary_labels /
     fixed_labels EXACTLY; label 10 never appears in either list; the
     only raw mesh labels are the spec's labels_used.  The
     boundaryLabels= / fixedLabels= KEYS are always present in
     setup_optimization — their VALUES may be EMPTY lists (empty-empty
     = every non-10 boundary FREE; see the three-way treatment in the
     Optimization setup section — never expand a FREE boundary into an
     explicit list).
   - The weak forms implement the spec's Lagrangian (objective, state,
     adjoint, shape derivative) with the spec's signs and concrete
     boundary labels.
   - phi: use the spec's optimization_config.phi EXACTLY —
     Initialization designed it under its phi rules; never replace or
     rewrite it.
   - iterations / hmin / hmax / diffusion / v0 / show_weak_material:
     use the spec's optimization_config values (iterations exactly;
     show_weak_material False).
   - Fields and test functions: the spec's names, types, fe_space,
     units and Dirichlet values/labels (test functions carry no units
     and no Dirichlet values); the ADJOINT field (u_tilde / \\tilde u)
     is AUTO-DERIVED by the DSL for linear problems — do NOT declare
     it in the code; check_units=True on the Lagrangian.

e. **NO COMMENTS** — the generated code must contain NO comments:
   no '#' lines, no explanatory text inside the code.  Return bare
   code.

## Output

Return exactly one JSON object:
{{"main": "<complete Python source>", "mesh": ""}}
No markdown.  No code fences.  No extra text.
"""

_CODER_USER = """\
AutoFreeFEM documentation:
{doc_text}

Reference examples (most relevant to this problem):
{examples_text}

ProblemSpec (from parse_problem):
{problem_spec_json}

Mesh specification (from handle_mesh):
{mesh_json}

Iteration budget (pipeline defaults):
- outer_iterations = {outer_iterations} (fallback when the spec gives no iterations)
- inner_iterations = {inner_iterations} (use for the nonlinear solver maxiter via newtonSettings if applicable)

Generate the complete AutoFreeFEM Python code.
"""


# ======================================================================
# documentation loader
# ======================================================================


def _load_documentation() -> str:
    """Load the AutoFreeFEM function documentation."""
    doc_path = (
        Path(__file__).resolve().parent.parent
        / "database"
        / "freefem_functions"
        / "list_of_classes.txt"
    )
    if doc_path.exists():
        return doc_path.read_text(encoding="utf-8", errors="replace")
    return ""


# ======================================================================
# main entry point
# ======================================================================


def generate_code(
    agent,
    problem_spec: dict,
    mesh_spec: Optional[dict] = None,
    doc_text: str = "",
    examples: Optional[list] = None,
    outer_iterations: int = 100,
    inner_iterations: int = 10,
    coder_system: str = None,
) -> dict:
    """
    Generate first-draft AutoFreeFEM code from ProblemSpec + mesh data.

    Parameters
    ----------
    agent : CodeAgent
    problem_spec : dict
        Output of ``parse_problem`` — contains ``extracted`` and ``designed``.
    mesh_spec : dict or None
        Output of ``handle_mesh`` — contains mesh code and metadata.
    doc_text : str
        AutoFreeFEM documentation text (loaded once by the caller).
    examples : list or None
        Selected reference examples — ``{"name": ..., "py": ...}`` dicts.
    outer_iterations : int
        Pipeline default for the outer-loop epochs (the spec's
        optimization_config.iterations takes precedence).
    inner_iterations : int
        Target number of inner-loop (nonlinear solver) iterations.

    Returns
    -------
    dict
        ``{"main": "<Python code>", "mesh": ""}`` — or, when every
        attempt failed to parse, ``{"main": "", "mesh": "",
        "_error": ...}``.
    """
    prompt_list = [
        {"role": "system", "content": coder_system or _CODER_SYSTEM},
        {"role": "user", "content": _CODER_USER},
    ]

    spec_json = json.dumps(problem_spec, indent=2, ensure_ascii=False)
    mesh_json = json.dumps(mesh_spec or {}, indent=2, ensure_ascii=False)

    # Build examples text: each example shows name + full .py code
    # Defensive .get: example dicts are LLM-adjacent data — malformed
    # entries must not abort generation.
    examples_text = ""
    for ex in examples or []:
        if not isinstance(ex, dict):
            continue
        examples_text += f"=== EXAMPLE: {ex.get('name', '?')} ===\n{ex.get('py', '')}\n\n"

    agent.set_prompt_list(prompt_list)
    payload = {
        "doc_text": doc_text,
        "examples_text": examples_text,
        "problem_spec_json": spec_json,
        "mesh_json": mesh_json,
        "outer_iterations": str(outer_iterations),
        "inner_iterations": str(inner_iterations),
    }

    # A failed generation is redone up to configs.MAX_GENERATE_RETRIES
    # times before the pipeline gives up on it.
    raw = ""
    last_error = "no usable response"
    for _attempt in range(configs.MAX_GENERATE_RETRIES + 1):
        set_label(
            agent,
            f"Generation/generate_code attempt {_attempt + 1}/"
            f"{configs.MAX_GENERATE_RETRIES + 1}",
        )
        raw = agent.invoke(payload)
        try:
            parsed = json.loads(parser_json(raw))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_error = f"JSON parse failed: {e}"
            warnings.warn(f"generate_code {last_error} — redoing")
            continue
        if not isinstance(parsed, dict):
            last_error = f"non-object JSON response ({type(parsed).__name__})"
            warnings.warn(f"generate_code returned {last_error} — redoing")
            continue
        return parsed
    return {"main": "", "mesh": "", "_error": last_error}


# ======================================================================
# Local test — loads initial.json files from results/ and generates code
# for each demo.  Run the main pipeline (`python main.py`) first to
# populate results/*/initial.json.
# ======================================================================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from API import DeepSeekCodeAgentV4Flash

    agent = DeepSeekCodeAgentV4Flash()
    results_dir = Path("results")

    # Load documentation once
    doc_text = _load_documentation()
    print(f"Loaded doc ({len(doc_text)} chars)")

    # Collect all available initial.json files
    spec_files = sorted(results_dir.glob("*/initial.json"))
    if not spec_files:
        print("No initial.json files found. Run `python main.py` first.")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Coder — Local Test ({len(spec_files)} specs from results/)")
    print(f"{'=' * 60}")

    summary_rows = []

    for spec_file in spec_files:
        case_name = spec_file.parent.name
        print(f"\n  [{case_name}]", end=" ", flush=True)

        with open(spec_file, "r", encoding="utf-8") as f:
            problem_spec = json.load(f)

        # Load mesh data if available
        mesh_file = spec_file.parent / "mesh.json"
        mesh_spec = None
        if mesh_file.exists():
            with open(mesh_file, "r", encoding="utf-8") as f:
                mesh_spec = json.load(f)

        # Load selected examples from examples.txt (shared loader — the
        # same logic select_examples.load_selected_examples provides).
        from Generation.select_examples import load_selected_examples

        examples = load_selected_examples(
            [
                l.strip()
                for l in spec_file.parent.joinpath("examples.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if l.strip()
            ]
            if spec_file.parent.joinpath("examples.txt").exists()
            else []
        )

        result = generate_code(agent, problem_spec, mesh_spec, doc_text=doc_text, examples=examples)

        if "_error" in result:
            print(f"ERROR: {result['_error']}")
            summary_rows.append((case_name, "ERROR", 0, ""))
            continue

        code = result.get("main", "")
        lines = code.count("\n") + 1

        # Save generated code
        out_file = spec_file.parent / "main.py"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Quick validity check
        checks = []
        if "from autofreefem import" not in code:
            checks.append("missing-import")
        if "Lagrangian([" not in code:
            checks.append("missing-lagrangian")
        if "setup_optimization" not in code and "lag.run(" not in code:
            checks.append("missing-setup")
        status = "OK" if not checks else ", ".join(checks)

        print(f"{lines} lines, {len(code)} chars [{status}]")
        summary_rows.append((case_name, status, lines, out_file))

    # ── summary table ──
    print(f"\n{'=' * 80}")
    print("CODER SUMMARY")
    print(f"{'=' * 80}")
    print(f"| {'Problem':<42} | {'Status':<22} | {'Lines':<6} | {'Output':<40} |")
    print(f"|{'-' * 42}|{'-' * 22}|{'-' * 6}|{'-' * 40}|")
    for name, status, lines, out in summary_rows:
        print(f"| {name:<42} | {status:<22} | {lines:<6} | {str(out):<40} |")

    print("\nDone. Generated code saved to results/*/main.py")
