"""Two-level error classifier."""

import warnings

# ======================================================================
# coarse classification
# ======================================================================


def coarse_classify(agent, problem_name: str, context_text: str) -> str:
    """
    Classify the current problem state into one of 7 coarse categories.

    Args:
        agent: CodeAgent for LLM calls.
        problem_name: Name of the problem.
        context_text: Full working directory contents (text files only).

    Returns:
        One of: pass, autofreefem_error, freefem_error,
                non_convergence, wrong_result, hardware_limit, other.
    """
    from .prompts import CLASSIFY_COARSE_SYSTEM, CLASSIFY_COARSE_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_COARSE_SYSTEM},
        {"role": "user", "content": CLASSIFY_COARSE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke({"problem_name": problem_name, "context_text": context_text})

    label = raw.strip().lower()
    valid = [
        "pass",
        "autofreefem_error",
        "freefem_error",
        "non_convergence",
        "wrong_result",
        "hardware_limit",
        "other",
    ]

    if label not in valid:
        warnings.warn(f"coarse_classify: unknown label '{label}'. Falling back to 'other'.")
        label = "other"

    return label


# ======================================================================
# fine classification
# ======================================================================


def fine_classify(
    agent,
    problem_name: str,
    coarse: str,
    main_py: str,
    mesh_txt: str,
    run_result_json: str,
    objective_data: str,
    convergence_data: str,
    edp_code: str,
    context_text: str = "",
) -> str:
    """
    Diagnose the specific sub-category for a given coarse category.

    Args:
        agent: CodeAgent for LLM calls.
        problem_name: Name of the problem.
        coarse: Coarse category from coarse_classify().
        main_py: Current main.py content.
        mesh_txt: Current mesh.txt content.
        run_result_json: JSON string of run_result.
        objective_data: Content of objective*.txt (or empty).
        convergence_data: Content of convergenceShape*.txt (or empty).
        edp_code: Content of generated .edp files (or empty).
        context_text: Full context (for 'other' category).

    Returns:
        Sub-category label string.
    """
    # Select the right prompt based on coarse category
    if coarse == "autofreefem_error":
        return _fine_classify_af(
            agent,
            problem_name,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
        )
    elif coarse == "freefem_error":
        return _fine_classify_ff(
            agent,
            problem_name,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
        )
    elif coarse == "non_convergence":
        return _fine_classify_conv(
            agent,
            problem_name,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
        )
    elif coarse == "wrong_result":
        return _fine_classify_wrong(
            agent,
            problem_name,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
            context_text=context_text,
        )
    elif coarse == "hardware_limit":
        return _fine_classify_hw(
            agent,
            problem_name,
            main_py,
            mesh_txt,
            run_result_json,
            objective_data,
            convergence_data,
            edp_code,
        )
    elif coarse == "other":
        return _fine_classify_other(agent, problem_name, context_text)
    else:
        return coarse  # coarse is the fine label (only reachable for hw/other fallthrough)


def _fine_classify_af(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
) -> str:
    """Fine classification for autofreefem_error."""
    from .prompts import CLASSIFY_FINE_AF_SYSTEM, CLASSIFY_FINE_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_AF_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_name": problem_name,
            "coarse": "autofreefem_error",
            "run_result_json": run_result_json,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "edp_code": edp_code or "(not available)",
        }
    )

    label = raw.strip().lower()
    valid_af = [
        "python_syntax",
        "missing_import",
        "undefined_name",
        "api_signature_error",
        "api_semantic_error",
        "label_assignment_error",
        "unit_error",
        "mesh_file_error",
        "timeout",
        "no_edp",
        "env_error",
    ]

    if label not in valid_af:
        warnings.warn(
            f"fine_classify AF: unknown label '{label}'. Falling back to 'api_semantic_error'."
        )
        label = "api_semantic_error"

    return label


def _fine_classify_ff(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
) -> str:
    """Fine classification for freefem_error."""
    from .prompts import CLASSIFY_FINE_FF_SYSTEM, CLASSIFY_FINE_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_FF_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_name": problem_name,
            "coarse": "freefem_error",
            "run_result_json": run_result_json,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "edp_code": edp_code or "(not available)",
        }
    )

    label = raw.strip().lower()
    valid_ff = [
        "ff_syntax",
        "ff_mesh_load",
        "ff_fe_space",
        "ff_boundary_label",
        "ff_label_semantics",
        "ff_include",
        "ff_tool",
        "ff_undefined_var",
        "ff_matrix_solver",
        "ff_unknown",
    ]

    if label not in valid_ff:
        warnings.warn(f"fine_classify FF: unknown label '{label}'. Falling back to 'ff_unknown'.")
        label = "ff_unknown"

    return label


def _fine_classify_conv(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
) -> str:
    """Fine classification for non_convergence."""
    from .prompts import CLASSIFY_FINE_CONV_SYSTEM, CLASSIFY_FINE_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_CONV_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_name": problem_name,
            "coarse": "non_convergence",
            "run_result_json": run_result_json,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "edp_code": edp_code or "(not available)",
        }
    )

    label = raw.strip().lower()
    valid_conv = [
        "newton_fail",
        "linear_solver_fail",
        "nan_inf",
        "obj_divergence",
        "obj_oscillation",
        "line_search_stall",
        "mesh_degen",
        "mmg_hang",
        "taylor_fail",
    ]

    if label not in valid_conv:
        warnings.warn(
            f"fine_classify CONV: unknown label '{label}'. Falling back to 'obj_divergence'."
        )
        label = "obj_divergence"

    return label


def _fine_classify_wrong(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
    context_text="",
) -> str:
    """Fine classification for wrong_result."""
    from .prompts import CLASSIFY_FINE_WRONG_SYSTEM, CLASSIFY_FINE_WRONG_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_WRONG_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_WRONG_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_name": problem_name,
            "coarse": "wrong_result",
            "run_result_json": run_result_json,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "edp_code": edp_code or "(not available)",
            "context_text": context_text or "(no context)",
        }
    )

    label = raw.strip().lower()
    valid_wrong = ["no_minth_vtu", "empty_design", "pathological_objective", "physics_violation"]

    # Hard filter: if context_text contains a render summary, MinTh.vtu
    # definitely exists — "no_minth_vtu" is impossible.
    if "minth.vtu render summary" in (context_text or "").lower():
        if label == "no_minth_vtu":
            warnings.warn(
                "fine_classify WRONG: LLM returned 'no_minth_vtu' but MinTh.vtu "
                "render summary is in context.  Falling back to 'empty_design'."
            )
            label = "empty_design"
    if label not in valid_wrong:
        warnings.warn(
            f"fine_classify WRONG: unknown label '{label}'. Falling back to 'empty_design'."
        )
        label = "empty_design"

    return label


def _fine_classify_hw(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
) -> str:
    """Fine classification for hardware_limit."""
    from .prompts import CLASSIFY_FINE_HW_SYSTEM, CLASSIFY_FINE_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_HW_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "problem_name": problem_name,
            "coarse": "hardware_limit",
            "run_result_json": run_result_json,
            "main_py": main_py,
            "mesh_txt": mesh_txt,
            "objective_data": objective_data or "(not available)",
            "convergence_data": convergence_data or "(not available)",
            "edp_code": edp_code or "(not available)",
        }
    )

    label = raw.strip().lower()
    valid_hw = ["memory_limit", "disk_limit", "process_limit"]

    if label not in valid_hw:
        warnings.warn(f"fine_classify HW: unknown label '{label}'. Falling back to 'memory_limit'.")
        label = "memory_limit"

    return label


def _fine_classify_other(agent, problem_name, context_text) -> str:
    """Free-form diagnosis for 'other' category."""
    from .prompts import CLASSIFY_FINE_OTHER_SYSTEM, CLASSIFY_FINE_OTHER_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_OTHER_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_OTHER_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke({"problem_name": problem_name, "context_text": context_text})

    return raw.strip()


# ======================================================================
# quality check — used when MinTh.vtu exists
# ======================================================================


def quality_check(agent, problem_name: str, context_text: str) -> str:
    """
    Check whether a MinTh.vtu result is genuinely successful.

    Args:
        agent: CodeAgent for LLM calls.
        problem_name: Name of the problem.
        context_text: Full working directory contents.

    Returns:
        "pass" or "wrong_result".
    """
    from .prompts import QUALITY_CHECK_SYSTEM, QUALITY_CHECK_USER

    prompt_list = [
        {"role": "system", "content": QUALITY_CHECK_SYSTEM},
        {"role": "user", "content": QUALITY_CHECK_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke({"problem_name": problem_name, "context_text": context_text})

    label = raw.strip().lower()
    if label not in ("pass", "wrong_result"):
        warnings.warn(f"quality_check: unknown label '{label}'. Falling back to 'wrong_result'.")
        label = "wrong_result"

    return label
