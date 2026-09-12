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
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Generation.label_conventions import (  # noqa: E402
    BOUNDARY_VS_FIXED_LABELS,
    DESIGN_DIRICHLET_PENALTY,
    DSX_LABEL_SEMANTICS,
    LABEL_999_AND_REGION3,
    MULTI_DOMAIN_AND_INTERFACE,
)
from Generation.phi_conventions import PHI_GUIDANCE  # noqa: E402
from utils import parser_json  # noqa: E402

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
    + DESIGN_DIRICHLET_PENALTY
    + """

    """
    + LABEL_999_AND_REGION3
    + """

    """
    + MULTI_DOMAIN_AND_INTERFACE
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
          (multi-domain problems: declare one Domain per material/region
          and implement the interface continuity conditions between the
          domains per the interface rules above)
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

    11. **Initial level set phi**: obey the PHI RULES below.  In short:
       - if the problem statement prescribes an initial design
         (``level_set_initial``), use it EXACTLY;
       - otherwise pick a seed from the PHI TEMPLATE LIBRARY (solid
         trunk shapes, sparse holes, checkerboard, combinations) and
         adapt it to this problem — a connected load-to-sink path,
         solid contact on the fixed (frozen) load and sink boundaries,
         features several times ``hmax``, symmetric about any mirror
         axis, smooth and low-frequency — with the reference examples
         contributing only the construction STYLE (see the PHI STYLE
         hint in the user prompt).

    A bare half-plane such as ``'y - 0.02'`` is a LAST RESORT — use it
    only when the problem prescribes such an initial design or when no
    other seed fits the geometry.

    12. **Afac (Lagrange multiplier ell) — initial estimate**:
       The ProblemSpec normally carries NO ell constant: ell is the
       BALANCE parameter of the optimisation, not a problem constant.
       The pipeline's reflection loop calibrates it with a geometric
       scan; the coder must supply an initial estimate of the right
       order of magnitude, balancing the two objective terms:
       ``ell0 ≈ Q * L * dT / |D|``, where ``Q`` is the dominant boundary
       weight of the temperature/flux objective term (``q_b`` or ``q``),
       ``L`` the length of that boundary, ``dT`` the characteristic
       temperature difference of the problem, and ``|D|`` the domain
       area.  Round to ONE significant figure (e.g. ``6e5``, ``2e4``).
       Mark it source="designed" and describe it as
       ``'volume weight (initial estimate, calibrated by pipeline)'``.

"""
    + PHI_GUIDANCE
    + """
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

    {phi_style_hint}

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
# phi style analysis — majority vote over the selected examples
# ======================================================================


def _phi_of_example(examples: list[dict]) -> list[str]:
    """Return one phi expression per example (or '' when absent)."""
    phis = []
    for exd in examples or []:
        # Example dicts are LLM-adjacent data — a malformed entry
        # (list/str/non-string py) must not abort generation; skip it.
        if not isinstance(exd, dict):
            continue
        py = exd.get("py", "")
        if not isinstance(py, str):
            continue
        m = re.search(r"""phi\s*=\s*('([^']*)'|"([^"]*)")""", py)
        phis.append(m.group(2) if m and m.group(2) else (m.group(3) if m else ""))
    return phis


# Function CALLS only — a word-boundary prevents "cosine"-like words
# from counting as sin/cos and "admin(" / "analysis(" from counting as
# min/max.
_PHI_PERIODIC_RE = re.compile(r"\b(?:sin|cos)\s*\(")
_PHI_CUTTING_RE = re.compile(r"\b(?:min|max)\s*\(")


def analyze_phi_style(examples: list[dict]) -> str | None:
    """Classify the phi construction styles used by *examples*.

    Returns the MAJORITY style among the examples that carry a phi:
    "periodic" (sin/cos oscillatory forms), "cutting" (min/max
    cutting/cleaning forms such as ``min(min(x, 1-x), min(y, 1-y))``),
    or "simple" (a bare arithmetic expression with neither).  Returns
    None when no example carries a phi.
    """
    styles = []
    for phi in _phi_of_example(examples):
        if not phi:
            continue
        if _PHI_PERIODIC_RE.search(phi):
            styles.append("periodic")
        elif _PHI_CUTTING_RE.search(phi):
            styles.append("cutting")
        else:
            styles.append("simple")
    if not styles:
        return None

    counts = {s: styles.count(s) for s in ("periodic", "cutting", "simple")}
    # strict majority; ties on a non-majority are treated as "no clear style"
    top = max(counts, key=counts.get)
    if counts[top] > len(styles) / 2:
        return top
    return None


def phi_style_hint(examples: list[dict]) -> str:
    """Build the phi-style instruction text for a prompt.

    Tells the LLM which phi construction style the MAJORITY of the
    selected examples uses — periodic (sin/cos) vs cutting (min/max) —
    and to follow it, scaled to the real domain size but NOT copied
    verbatim.  Empty string when there is no clear majority.

    The hint is deliberately subordinate to the PHI RULES: the paper's
    prescribed phi always wins, and the examples only lend their
    construction STYLE to a seed whose geometry is derived from the
    problem's physics.
    """
    style = analyze_phi_style(examples)
    if style is None:
        return ""
    lead = ("PHI STYLE HINT (style only — the PHI RULES come first: a "
            "phi prescribed by the problem statement wins, and the seed "
            "must be picked from the PHI TEMPLATE LIBRARY and adapted to "
            "the problem's physics; this hint tells you which library "
            "family the examples favour): ")
    if style == "periodic":
        return (
            lead + "the selected examples predominantly use a "
            "PERIODIC oscillatory form: a constant offset plus a "
            "sin(...)*cos(...) (or equivalent) oscillation in x/y.  "
            "Use this style for the perturbations of your physics-derived "
            "seed, with frequencies scaled to the REAL domain size of the "
            "mesh (never copy example frequencies verbatim)."
        )
    if style == "cutting":
        return (
            lead + "the selected examples predominantly use a "
            "CUTTING form with min/max: "
            "min or max combinations that cut out a shape (like "
            "min(min(x, 1-x), min(y, 1-y)) or a banded strip).  "
            "Use this min/max cutting style to realise your "
            "physics-derived seed, adapted to the REAL domain size of the "
            "mesh (never copy example geometry verbatim)."
        )
    return (
        lead + "the selected examples predominantly use a SIMPLE "
        "arithmetic form (no sin/cos, no min/max cutting) for phi — "
        "follow the same simple style."
    )


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
    # Defensive .get: example dicts are LLM-adjacent data — malformed
    # entries must not abort generation.
    examples_text = ""
    for ex in examples or []:
        if not isinstance(ex, dict):
            continue
        examples_text += f"=== EXAMPLE: {ex.get('name', '?')} ===\n{ex.get('py', '')}\n\n"

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "doc_text": doc_text,
            "examples_text": examples_text,
            "problem_spec_json": spec_json,
            "mesh_json": mesh_json,
            "outer_iterations": str(outer_iterations),
            "inner_iterations": str(inner_iterations),
            "phi_style_hint": phi_style_hint(examples),
        }
    )

    try:
        parsed = json.loads(parser_json(raw))
        if not isinstance(parsed, dict):
            warnings.warn(f"generate_code returned non-object JSON ({type(parsed).__name__}).")
            return {"main": raw, "mesh": "", "_error": "non-object JSON response"}
        return parsed
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
