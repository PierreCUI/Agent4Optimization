"""
Fix strategies for the Reflection module.

Two modes:
- local_fix: targeted LLM repair of the specific error.
- rewrite: full regeneration of main.py + mesh.txt from original problem.

Both modes return a dict {"main": ..., "mesh": ..., "rationale": ...}.
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

    Returns:
        {"main": str, "mesh": str, "rationale": str}.
        On parse failure, returns {"main": "", "mesh": "", "rationale": "",
        "_error": str}.
    """
    if action == "rewrite":
        return _rewrite(
            agent,
            problem_description,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
            reflection_history,
            doc_text,
            analysis_artifacts,
            coarse,
            fine,
            outer_iterations,
            inner_iterations,
        )
    else:
        return _local_fix(
            agent,
            problem_description,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
            reflection_history,
            doc_text,
            analysis_artifacts,
            coarse,
            fine,
            outer_iterations,
            inner_iterations,
        )


def _local_fix(
    agent,
    problem_description,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
    reflection_history,
    doc_text,
    analysis_artifacts,
    coarse,
    fine,
    outer_iterations,
    inner_iterations,
) -> dict:
    """Targeted repair of the specific (coarse, fine) error."""
    from .prompts import FIX_LOCAL_SYSTEM, FIX_LOCAL_USER

    prompt_list = [
        {"role": "system", "content": FIX_LOCAL_SYSTEM},
        {"role": "user", "content": FIX_LOCAL_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
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
        }
    )

    return _parse_fix_response(raw)


def _rewrite(
    agent,
    problem_description,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
    reflection_history,
    doc_text,
    analysis_artifacts,
    coarse,
    fine,
    outer_iterations,
    inner_iterations,
) -> dict:
    """Full regeneration of main.py + mesh.txt."""
    from .prompts import FIX_REWRITE_SYSTEM, FIX_REWRITE_USER

    prompt_list = [
        {"role": "system", "content": FIX_REWRITE_SYSTEM},
        {"role": "user", "content": FIX_REWRITE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_description": problem_description,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "run_result_json": run_result_json,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "reflection_history": reflection_history or "(first attempt)",
            "doc_text": doc_text,
            "analysis_artifacts": analysis_artifacts or "(not available)",
            "coarse": coarse,
            "fine": fine,
            "outer_iterations": str(outer_iterations),
            "inner_iterations": str(inner_iterations),
        }
    )

    return _parse_fix_response(raw)


def _parse_fix_response(raw: str) -> dict:
    """Parse LLM JSON response, with fallback on failure."""
    try:
        result = json.loads(parser_json(raw))
        if "main" not in result:
            result["main"] = ""
        if "mesh" not in result:
            result["mesh"] = ""
        if "rationale" not in result:
            result["rationale"] = ""
        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"fixer JSON parse failed: {e}")
        return {"main": "", "mesh": "", "rationale": "", "_error": str(e)}
