import json
import os
import sys
import warnings

# Allow local testing without installing the full package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import parser_json  # noqa: E402

# ======================================================================
# prompts
# ======================================================================

_SAFETY_SYSTEM = """\
    You are a safety filter for a shape-optimization and topology-optimization
    code-generation system.  Your only task is to decide whether a given problem
    description belongs to shape optimization or topology optimization.
    
    Definitions used for this decision:
    
    - **Shape optimization**: optimizing the shape or geometry of a domain to
      minimize an objective functional subject to PDE constraints.  The design
      variable is the boundary of the domain.  Typical ingredients: a level-set
      function, a shape derivative, a Lagrangian, a velocity field, a
      Hamilton-Jacobi equation, boundary perturbation, shape gradient descent.
    
    - **Topology optimization**: optimizing the material distribution inside a
      fixed domain.  Typical ingredients: a density field, SIMP (Solid Isotropic
      Material with Penalization), a characteristic function with weak material
      (e.g. \\chi = 1 or 1/100), compliance minimization, volume constraints.
    
    - **Related PDE-constrained optimization** that uses shape or topology
      derivatives to drive an iterative design update is also considered relevant.
    
    - **NOT relevant**: pure forward PDE solving without any optimization loop,
      pure machine learning, pure data analysis, general programming tasks, or
      any problem that does not involve iteratively changing a domain shape or
      material layout to minimize an objective.
    
    You must classify the problem into exactly one of three categories:
    
    1. "pass" — the problem is clearly about shape optimization or topology
       optimization.  Return the original description unchanged.
    
    2. "reject" — the problem has nothing to do with shape optimization or
       topology optimization.  Return an empty filtered_description.
    
    3. "partial" — the description contains shape/topology optimization content
       mixed with clearly unrelated content.  Remove the unrelated parts and keep
       only the optimization problem statement in filtered_description.
    
    Return exactly one JSON object:
    {{"classification": "<pass|reject|partial>", "filtered_description": "<...>", "reason": "<one sentence>"}}
    Do NOT output markdown.  Do NOT wrap the result in code fences.
    The "reason" field must be a single short sentence in English.
"""

_SAFETY_USER = """\
    Problem description:
    {problem_description}
"""


def safety_check(agent, problem_description):
    """
    Check whether the given problem is related to shape optimization
    or topology optimization.

    Parameters
    ----------
    agent : CodeAgent
    problem_description : str
        Raw problem description (e.g. LaTeX from a .tex file).

    Returns
    -------
    status : str  — "pass", "reject", or "partial".
    filtered_description : str
    message : str  — explanation or warning (empty on "pass").
    """
    prompt_list = [
        {"role": "system", "content": _SAFETY_SYSTEM},
        {"role": "user", "content": _SAFETY_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw_response = agent.invoke({"problem_description": problem_description})

    try:
        result = json.loads(parser_json(raw_response))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(
            f"Safety check JSON parsing failed: {e}. "
            "Falling back to 'pass' to avoid blocking legitimate queries."
        )
        return "pass", problem_description, ""
    if not isinstance(result, dict):
        # Valid JSON that is not an object (array/string/number) — same
        # handling as a parse failure: fail open, never crash the pipeline.
        warnings.warn(
            f"Safety check returned non-object JSON ({type(result).__name__}); "
            "falling back to 'pass'."
        )
        return "pass", problem_description, ""

    classification = result.get("classification")
    if classification is None:
        # A parsable-but-partial response (description present, the
        # classification key omitted) must not hard-reject a legitimate
        # query — same fail-open policy as the JSON parse failures above.
        warnings.warn(
            "Safety check response lacks 'classification'; falling back to 'pass'."
        )
        return "pass", problem_description, ""
    filtered_description = result.get("filtered_description", "")
    reason = result.get("reason", "unknown reason")

    if classification == "reject":
        return "reject", "", f"Query rejected: {reason}"
    elif classification == "partial":
        return "partial", filtered_description, f"Partial match: {reason}"
    elif classification == "pass":
        return "pass", problem_description, ""
    else:
        warnings.warn(f"Unknown safety classification '{classification}'. Falling back to 'pass'.")
        return "pass", problem_description, ""


# ──────────────────────────────────────────────────────────────────────
# Local test — run this file directly to verify safety_check behaviour.
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from API import DeepSeekCodeAgentV4Flash

    agent = DeepSeekCodeAgentV4Flash()

    # ── Test case 1: clearly relevant (shape optimization) ────────────
    test_pass = r"""
        Let $\phi(x)$ be the level-set function. The Lagrangian of the linear
        elasticity shape optimization problem is
        \mathcal L([\mathbf{u}],[\delta\mathbf{u}]) =
        J(\mathbf{u}) - \int_{\Gamma_N} f (\mathbf{e}_y \cdot \delta\mathbf{u}) \,ds_x
        + \int_{D} (\pmb\sigma(\mathbf{u}) : \nabla\delta\mathbf{u}) \,dx.
        The shape derivative DJ(\Omega)(\mathbf{V}) drives the level-set evolution.
    """

    # ── Test case 2: clearly irrelevant ───────────────────────────────
    test_reject = """
        Please write a Python script that scrapes product prices from Amazon
        and stores them in a PostgreSQL database. Use BeautifulSoup for parsing.
    """

    # ── Test case 3: partially relevant ────────────────────────────────
    test_partial = """
        First, help me set up a Docker container for my web application.
        The Dockerfile should use Ubuntu 22.04 as base image.
        Second, I need to solve a topology optimization problem: minimize compliance
        of a cantilever beam under a vertical load using the SIMP method with a
        volume fraction constraint of 0.5.
        Finally, write a unit test for the login endpoint of my REST API.
    """

    print("=" * 60)
    print("Safety Check — Local Test")
    print("=" * 60)

    for label, query in [("PASS", test_pass), ("REJECT", test_reject), ("PARTIAL", test_partial)]:
        print(f"\n-- Test: should be '{label}' --")
        status, desc, msg = safety_check(agent, query)
        print(f"  status:   {status}")
        if status != "pass":
            print(f"  message:  {msg}")
        if status == "partial":
            print(f"  filtered: {desc[:200]}...")
        print(f"  expected: {label.lower()}  {'OK' if status == label.lower() else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Test complete.")
