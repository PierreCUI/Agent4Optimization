"""
Code Generation — produce first-draft AutoFreeFEM Python code from a
complete ProblemSpec.

Input:  a ProblemSpec dict with both ``extracted`` (immutable) and ``designed``
        (mutable) parts, as produced by ``Initialization.parse_problem``.
Output: a dict ``{"main": "...", "mesh": ""}`` where ``main`` is the complete
        AutoFreeFEM Python source.
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

from Generation.label_conventions import (
    BOUNDARY_VS_FIXED_LABELS,
    DSX_LABEL_SEMANTICS,
    LABEL_999_AND_REGION3,
)
from utils import parser_json

# ======================================================================
# system prompt
# ======================================================================

_CODER_SYSTEM = (
    """\
    You are an expert AutoFreeFEM code generator.  You receive a complete
    ProblemSpec (a structured JSON describing a shape or topology optimisation
    problem) and must produce working AutoFreeFEM Python code.
    
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

    ```python
    name = 'linear Diffusion'     # spaces, NOT underscores
    # NOT: name = 'linear_Diffusion' or 'run_linearDiffusion'
    ```
    
    ### Domain and boundaries
    ```python
    bndN = Boundary(r'\\Gamma_N', 'ThL', '2')        # Neumann boundary, label 2
    bndD = Boundary(r'\\Gamma_D', 'ThL', '4')        # Dirichlet boundary, label 4
    Th = Domain('D', 'Th', bndN, bndD)               # domain with boundaries
    # If no boundaries need explicit naming:
    Th = Domain('D', 'Th')
    ```

    """
    + DSX_LABEL_SEMANTICS
    + """
    
    ### Scalar fields
    ```python
    u = ScalarField('u', 'P2', Th, '0.', [3], kelvin)   # Dirichlet on label 3
    testu = ScalarField(r'\\delta u', 'P2', Th)          # test function
    ```
    
    ### Vector fields
    ```python
    u = VectorField('u', 'P2', Th, '0.', 4, meter)      # Dirichlet on label 4
    testu = VectorField(r'\\delta u', 'P2', Th)           # test function
    ```
    
    ### Constants
    ```python
    f = Constant('f', -10, 'load', Newton / meter ** 2)
    nu = Constant(r'\\nu', 0.3, 'Poisson ratio', '-')
    ```
    
    ### Characteristic function (material contrast)
    ```python
    X = CharacteristicFunction(r'\\raisebox{{\\depth}}{{$\\chi$}}', 1, '1/100')
    # strong_value=1 means solid, weak_value='1/100' means weak material
    # Use '1/10' for diffusion/thermal problems
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
    
    ### Lagrangian
    ```python
    # state variables, test functions, objective, weak form constraint
    lag = Lagrangian([u], [testu], J,
        dx(inner2(stress, grad(testu)), Th) - dsx(f * inner(testu, ey()), Th, 0),
        check_units=True)
    ```
    
    For 3D problems add dimensions=3:
    ```python
    lag = Lagrangian([u], [testu], J, ..., dimensions=3)
    ```
    
    For nonlinear problems, state the Newton linearisation is auto-derived.
    For multi-field problems list all state and test fields:
    ```python
    lag = Lagrangian([u, p], [testu, testp], J, ...)
    ```
    
    ### Optimization setup

    **IMPORTANT — iteration counts**: use the exact values provided in the
    user prompt for ``iterations`` (outer loop) and any nonlinear solver
    ``maxiter`` (inner loop).  Do NOT make up your own values.

    """
    + BOUNDARY_VS_FIXED_LABELS
    + """

    """
    + LABEL_999_AND_REGION3
    + """

    ```python
    lag.setup_optimization(name,
        mesh='mesh.txt',
        hmin=0.01, hmax=0.02,
        diffusion=1./10000., v0=0.002,
        iterations=<use outer_iterations from prompt>,
        phi='-0.4 - sin(pi*3*(x+0.5))*cos(pi*6*y)',
        boundaryLabels=['1','4'], fixedLabels=['2'],
        show_weak_material=False)
    ```
    
    ## Rules
    
    1. **Respect source annotations**:
       - Values with source="extracted" are GROUND TRUTH — use them exactly.
       - Values with source="designed" are REASONABLE DEFAULTS — use them but you
         may refine them if the code would not run otherwise.
    
    2. **Code structure** — follow this order:
       a. imports
       b. problem name
       c. boundaries (if needed) + domain
       d. fields + test functions (with comments)
       e. constants (with comments)
       f. characteristic function
       g. expressions
       h. objective
       i. Lagrangian
       j. optimization setup
    
    3. **Units**:
       - Use AutoFreeFEM unit system.  Unit names are CAPITALIZED:
         ``meter``, ``kelvin``, ``Newton``, ``Watt`` (NOT watt!), ``Pascal``.
       - If the ProblemSpec gives a unit string like "N/m^2", convert to
         AutoFreeFEM syntax: ``Newton / meter ** 2``.
       - Dimensionless constants use ``'-'``.
       - Common: ``Watt / meter / kelvin`` (NOT watt/m/K).
    
    4. **boundaries vs Boundary**:
       - Only create ``Boundary()`` objects for boundaries that need to be
         referenced by name in the weak form or BCs.
       - If a boundary is only used as a label number (e.g. in fixedLabels),
         you do NOT need a ``Boundary()`` object.
    
    5. **3D problems**:
       - Pass ``dimensions=3`` to ``Lagrangian()``.
       - Use the same field/expression API — it auto-adapts.
    
    6. **Variable naming consistency — CRITICAL**:
       - If the ProblemSpec names a constant "ell" (Lagrange multiplier), the
         Python variable SHOULD be ``Afac`` and the LaTeX symbol should be
         ``r'\\ell'``.  Then use ``Afac`` everywhere in expressions.
       - If the ProblemSpec names it "Afac", keep ``Afac``.
       - In BOTH cases the Constant display name is the LaTeX symbol ``\\ell``.
       - The objective references MUST use the Python variable name, not the
         ProblemSpec key name.  ``dx(Afac * X, Th)`` NOT ``dx(ell * X, Th)``.

    7. **Boundary() second argument**:
       - Use ``'ThL'`` (not ``'Th'``) as the second argument to ``Boundary()``.
         Correct: ``Boundary(r'\\Gamma_N', 'ThL', '2')``.

    8. **CharacteristicFunction LaTeX**:
       - Always include the full LaTeX decoration.
         Correct: ``CharacteristicFunction(r'\\raisebox{{\\depth}}{{$\\chi$}}', 1, '1/100')``.

    9. **Mesh filename — ALWAYS ``mesh.txt``**:  Regardless of what the
       ProblemSpec says (even if it mentions a different filename), the
       ``mesh=`` argument in ``setup_optimization()`` must be ``'mesh.txt'``.

    10. **Edge cases**:
       - If ``dirichlet_value`` is empty or null, omit the Dirichlet BC argument.
       - If there are multiple state fields, list them all in the Lagrangian.
       - The ``show_weak_material=False`` flag is commonly used.
       - 3D problems: add ``dimensions=3`` to ``Lagrangian()``.
    
    ## Output

    Return exactly one JSON object:
    {{"main": "<complete Python source>", "mesh": ""}}
    No markdown.  No code fences.  No extra text.
"""
)

_CODER_USER = """\
    AutoFreeFEM documentation:
    {doc_text}

    Reference examples (most relevant to this problem):
    {examples_text}

    ProblemSpec (from parse_problem):
    {problem_spec_json}

    Mesh specification (from handle_mesh):
    {mesh_json}

    Iteration budget:
    - outer_iterations = {outer_iterations} (use for setup_optimization iterations=)
    - inner_iterations = {inner_iterations} (use for nonlinear solver maxiter if applicable)

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
    examples: Optional[list[dict]] = None,
    outer_iterations: int = 100,
    inner_iterations: int = 10,
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
    outer_iterations : int
        Target number of outer-loop optimisation epochs.
    inner_iterations : int
        Target number of inner-loop (nonlinear solver) iterations.

    Returns
    -------
    dict
        ``{"main": "<Python code>", "mesh": ""}``
    """
    prompt_list = [
        {"role": "system", "content": _CODER_SYSTEM},
        {"role": "user", "content": _CODER_USER},
    ]

    spec_json = json.dumps(problem_spec, indent=2, ensure_ascii=False)
    mesh_json = json.dumps(mesh_spec or {}, indent=2, ensure_ascii=False)

    # Build examples text: each example shows name + full .py code
    examples_text = ""
    for ex in examples or []:
        examples_text += f"=== EXAMPLE: {ex['name']} ===\n{ex['py']}\n\n"

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "doc_text": doc_text,
            "examples_text": examples_text,
            "problem_spec_json": spec_json,
            "mesh_json": mesh_json,
            "outer_iterations": str(outer_iterations),
            "inner_iterations": str(inner_iterations),
        }
    )

    try:
        return json.loads(parser_json(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"generate_code JSON parse failed: {e}")
        return {"main": raw, "mesh": "", "_error": str(e)}


# ======================================================================
# Local test — loads initial.json files from results/ and generates code
# for each demo.  Run `python run_initialization.py` first to populate.
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
        print("No initial.json files found. Run `python run_initialization.py` first.")
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

        # Load selected examples from examples.txt
        examples = []
        examples_file = spec_file.parent / "examples.txt"
        if examples_file.exists():
            selected_names = [
                l.strip()
                for l in examples_file.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            examples_dir = Path("database") / "freefem_examples"
            for ename in selected_names:
                ex_dir = examples_dir / ename
                if not ex_dir.is_dir():
                    continue
                for ef in ex_dir.iterdir():
                    if ef.suffix == ".py":
                        examples.append(
                            {"name": ename, "py": ef.read_text(encoding="utf-8", errors="replace")}
                        )
                        break

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

    print(f"\nDone. Generated code saved to results/*/main.py")
