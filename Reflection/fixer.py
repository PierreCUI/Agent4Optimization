"""
Fix strategies for the Reflection module.

Two modes:
- local_fix: targeted LLM repair of the specific error.
- rewrite: full regeneration of main.py + mesh.txt from original problem.

Both modes return a dict {"main": ..., "mesh": ..., "rationale": ...}.
An optional *fix_direction* instruction (parallel reflection) restricts
the repair to ONE idea — the probe round's diversity comes from each
candidate exploring a different direction.
"""

import json
import warnings

from utils import parser_json


def apply(
    agent,
    problem_description: str,
    main_py: str,
    mesh_txt: str,
    run_result_json: str,
    objective_data: str,
    convergence_data: str,
    edp_code: str,
    reflection_history: str,
    doc_text: str,
    analysis_artifacts: str,
    coarse: str,
    fine: str,
    action: str,
    outer_iterations: int = 100,
    inner_iterations: int = 10,
    images=None,
    image_section: str = "",
    fix_direction: str = "",
) -> dict:
    """
    Apply a fix for the current error.

    Args:
        agent: CodeAgent for LLM calls.
        problem_description: Original problem (.tex + .txt files).
        main_py: Current main.py content.
        mesh_txt: Current mesh.txt content.
        run_result_json: JSON string of run_result.
        objective_data: Content of objective*.txt (or "").
        convergence_data: Content of convergenceShape*.txt (or "").
        edp_code: Content of generated .edp files (or "").
        reflection_history: Full reflection.txt content.
        doc_text: AutoFreeFEM documentation.
        analysis_artifacts: initial.json + mesh.json (for reference).
        coarse: Coarse error category.
        fine: Fine error sub-category.
        action: "local_fix" or "rewrite".
        outer_iterations: Target outer-loop epochs.
        inner_iterations: Target inner-loop iterations.
        fix_direction: Optional probe direction instruction (parallel
            reflection: each candidate explores exactly one direction).

    Returns:
        {"main": str, "mesh": str, "rationale": str}.
        On parse failure, returns {"main": "", "mesh": "", "rationale": "",
        "_error": str}.
    """
    if action == "rewrite":
        from .prompts import FIX_REWRITE_SYSTEM, FIX_REWRITE_USER

        system, user = FIX_REWRITE_SYSTEM, FIX_REWRITE_USER
    else:
        from .prompts import FIX_LOCAL_SYSTEM, FIX_LOCAL_USER

        system, user = FIX_LOCAL_SYSTEM, FIX_LOCAL_USER

    agent.set_prompt_list(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    raw = agent.invoke_multimodal(
        {
            "problem_description": problem_description,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "run_result_json": run_result_json,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "edp_code": edp_code or "(not available)",
            "reflection_history": reflection_history or "(first attempt)",
            "doc_text": doc_text,
            "analysis_artifacts": analysis_artifacts or "(not available)",
            "coarse": coarse,
            "fine": fine,
            "outer_iterations": str(outer_iterations),
            "inner_iterations": str(inner_iterations),
            "fix_direction": fix_direction or "",
            "image_section": image_section,
        },
        images=images,
    )

    return _parse_fix_response(raw)


def _parse_fix_response(raw: str) -> dict:
    """Parse LLM JSON response, with fallback on failure.

    A response carrying NEITHER ``main`` NOR ``mesh`` code (wrong key
    names, schema drift, empty output) is treated as a parse failure —
    the reflector retries instead of permanently giving up.
    """
    try:
        result = json.loads(parser_json(raw))
        if not isinstance(result, dict):
            warnings.warn(f"fixer returned non-object JSON ({type(result).__name__}).")
            return {"main": "", "mesh": "", "rationale": "", "_error": "non-object JSON response"}

        main = result.get("main")
        mesh = result.get("mesh")
        # Type validation first: a non-string flows to write_text and
        # crashes the whole pipeline.
        if main is not None and not isinstance(main, str):
            return {"main": "", "mesh": "", "rationale": "", "_error": "main is not a string"}
        if mesh is not None and not isinstance(mesh, str):
            return {"main": "", "mesh": "", "rationale": "", "_error": "mesh is not a string"}
        has_code = (
            (isinstance(main, str) and main.strip() != "")
            or (isinstance(mesh, str) and mesh.strip() != "")
        )
        if not has_code:
            warnings.warn("fixer response carries neither main nor mesh code.")
            return {"main": "", "mesh": "", "rationale": "", "_error": "no main/mesh in response"}
        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"fixer JSON parse failed: {e}")
        return {"main": "", "mesh": "", "rationale": "", "_error": str(e)}
