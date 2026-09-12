"""Two-level error classifier."""

import re
import warnings

# Shared by the reflector (which appends the header to the context) and
# the classifier (which uses its presence as a hard filter) — one
# definition, owned HERE so the dependency-free classifier stays
# importable without meshio (the renderer imports this module, never
# the other way around).
MINTH_RENDER_SUMMARY_HEADER = "=== MinTh.vtu Render Summary ==="

# ======================================================================
# shared label normalization
# ======================================================================


def _canonical_label(text: str) -> str:
    """Extract the single canonical keyword from an LLM label answer.

    Accepts benign variants ("pass.", "Pass ✓", '"wrong_result"',
    "Wrong Result"); a prose/refusal answer maps to "unknown" so the
    caller's warning path fires instead of silently faking a value.
    """
    t = re.sub(r"[^a-z]+", "", text.strip().lower())
    # Pass verdict checked FIRST so a passing run whose justification
    # mentions a failure word ("pass — not wrong_result") is never
    # inverted into wrong_result by the keyword checks below.
    if t == "pass" or t.startswith("pass"):
        return "pass"
    # A CANONICAL LABEL WORD in the answer wins over any explanatory
    # prose the model appended — "autofreefem_error, not converging" is
    # an autofreefem_error, not a convergence diagnosis.  (Order: autofreefem
    # before freefem — it contains it.  The natural phrasings
    # "noconvergence"/"notconverge" are NOT label words — they live in
    # the heuristic tier below.)
    if "autofreefemerror" in t:
        return "autofreefem_error"
    if "freefemerror" in t:
        return "freefem_error"
    # wrongresult BEFORE nonconvergence: a reply carrying both canonical
    # words ("wrong_result (non convergence)") must take the label the
    # model actually answered, not the prose diagnosis
    if "wrongresult" in t or "invalid" in t:
        return "wrong_result"
    if "nonconvergence" in t:
        return "non_convergence"
    if "hardware" in t:
        return "hardware_limit"
    # Word-bounded on the ORIGINAL text (t has the spaces stripped, so a
    # word boundary there can never fire): "another"/"mother" in prose
    # must not read as the "other" category.
    if re.search(r"\bother\b", text.strip().lower()):
        return "other"
    # No canonical label word — keyword heuristics on the prose.
    # Accept natural phrasings: "no convergence" -> noconvergence,
    # "did/does not converge" -> ...notconverge, "failed to converge" /
    # "convergence failure" -> converg+fail — all before the broad
    # "fail" rule, which would steal them into wrong_result.
    if (
        "noconvergence" in t
        or "notconverge" in t
        or ("converg" in t and "fail" in t)
        or ("converg" in t and "not" in t)
    ):
        return "non_convergence"
    if "fail" in t:
        return "wrong_result"
    return "unknown"


# ======================================================================
# coarse classification
# ======================================================================


def coarse_classify(
    agent, problem_name: str, context_text: str, images=None, image_section: str = ""
) -> str:
    """
    Classify the current problem state into one of 7 coarse categories.

    Args:
        agent: CodeAgent for LLM calls.
        problem_name: Name of the problem.
        context_text: Full working directory contents (text files only).
        images: Optional PNG paths (iteration-label composite) for the
                multimodal reflection loop.
        image_section: Text describing the attached image(s).

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
    raw = agent.invoke_multimodal(
        {"problem_name": problem_name, "context_text": context_text,
         "image_section": image_section},
        images=images,
    )

    label = _canonical_label(raw)
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
        warnings.warn(
            f"coarse_classify: unknown label '{label}' (raw: {raw[:60]!r}). "
            f"Falling back to 'other'."
        )
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
    images=None,
    image_section: str = "",
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
            images=images,
            image_section=image_section,
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
            images=images,
            image_section=image_section,
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
            images=images,
            image_section=image_section,
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
            images=images,
            image_section=image_section,
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
            images=images,
            image_section=image_section,
        )
    elif coarse == "other":
        return _fine_classify_other(
            agent, problem_name, context_text, images=images, image_section=image_section
        )
    else:
        # Defensive fallthrough: no fine label for this coarse category
        # (e.g. "pass", which the reflection loop handles before this
        # point) — reuse the coarse label as-is.
        return coarse


def _fine_payload(
    coarse: str,
    problem_name: str,
    main_py: str,
    mesh_txt: str,
    run_result_json: str,
    objective_data: str,
    convergence_data: str,
    edp_code: str,
    context_text: str = "",
    image_section: str = "",
) -> dict:
    """Build the invoke payload shared by the fine classifiers."""
    payload = {
        "problem_name": problem_name,
        "coarse": coarse,
        "run_result_json": run_result_json,
        "main_py": main_py,
        "mesh_txt": mesh_txt,
        "objective_data": objective_data or "(not available)",
        "convergence_data": convergence_data or "(not available)",
        "edp_code": edp_code or "(not available)",
        "image_section": image_section,
    }
    if context_text:
        payload["context_text"] = context_text
    return payload


def _fine_run(
    agent,
    system: str,
    user: str,
    payload: dict,
    valid: list,
    fallback: str,
    name: str,
    images=None,
) -> str:
    """Invoke one fine-classification prompt and validate the label."""
    agent.set_prompt_list(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    raw = agent.invoke_multimodal(payload, images=images)
    # Same leniency as the coarse path: punctuation / whitespace variants
    # of a label ("ff_syntax.", "newton-fail", "no minth vtu") must not
    # silently re-target the fix onto the per-category default.
    def _norm(s: str) -> str:
        return re.sub(r"[\W]+", "", s.lower())

    label = _norm(raw)
    canonical = next((v for v in valid if _norm(v) == label), None)
    if canonical is None:
        warnings.warn(
            f"fine_classify {name}: unknown label '{raw.strip()[:60]}'. "
            f"Falling back to '{fallback}'."
        )
        label = fallback
    else:
        label = canonical
    return label


def _fine_classify_af(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
    images=None,
    image_section="",
) -> str:
    """Fine classification for autofreefem_error."""
    from .prompts import CLASSIFY_FINE_AF_SYSTEM, CLASSIFY_FINE_USER

    return _fine_run(
        agent,
        CLASSIFY_FINE_AF_SYSTEM,
        CLASSIFY_FINE_USER,
        _fine_payload("autofreefem_error", problem_name, main_py, mesh_txt,
                      run_result_json, objective_data, convergence_data,
                      edp_code, image_section=image_section),
        [
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
        ],
        "api_semantic_error",
        "AF",
        images=images,
    )


def _fine_classify_ff(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
    images=None,
    image_section="",
) -> str:
    """Fine classification for freefem_error."""
    from .prompts import CLASSIFY_FINE_FF_SYSTEM, CLASSIFY_FINE_USER

    return _fine_run(
        agent,
        CLASSIFY_FINE_FF_SYSTEM,
        CLASSIFY_FINE_USER,
        _fine_payload("freefem_error", problem_name, main_py, mesh_txt,
                      run_result_json, objective_data, convergence_data,
                      edp_code, image_section=image_section),
        [
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
        ],
        "ff_unknown",
        "FF",
        images=images,
    )


def _fine_classify_conv(
    agent,
    problem_name,
    main_py,
    mesh_txt,
    run_result_json,
    objective_data,
    convergence_data,
    edp_code,
    images=None,
    image_section="",
) -> str:
    """Fine classification for non_convergence."""
    from .prompts import CLASSIFY_FINE_CONV_SYSTEM, CLASSIFY_FINE_USER

    return _fine_run(
        agent,
        CLASSIFY_FINE_CONV_SYSTEM,
        CLASSIFY_FINE_USER,
        _fine_payload("non_convergence", problem_name, main_py, mesh_txt,
                      run_result_json, objective_data, convergence_data,
                      edp_code, image_section=image_section),
        [
            "newton_fail",
            "linear_solver_fail",
            "nan_inf",
            "obj_divergence",
            "obj_oscillation",
            "line_search_stall",
            "mesh_degen",
            "mmg_hang",
            "taylor_fail",
        ],
        "obj_divergence",
        "CONV",
        images=images,
    )


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
    images=None,
    image_section="",
) -> str:
    """Fine classification for wrong_result."""
    from .prompts import CLASSIFY_FINE_WRONG_SYSTEM, CLASSIFY_FINE_WRONG_USER

    label = _fine_run(
        agent,
        CLASSIFY_FINE_WRONG_SYSTEM,
        CLASSIFY_FINE_WRONG_USER,
        _fine_payload("wrong_result", problem_name, main_py, mesh_txt,
                      run_result_json, objective_data, convergence_data,
                      edp_code, context_text or "(no context)",
                      image_section=image_section),
        ["no_minth_vtu", "empty_design", "pathological_objective",
         "physics_violation", "phi_unsuitable"],
        "empty_design",
        "WRONG",
        images=images,
    )

    # Hard filter: if context_text contains a render summary, MinTh.vtu
    # definitely exists — "no_minth_vtu" is impossible.  The header
    # constant is shared with the reflector, so rewording the header
    # cannot silently disable this filter.
    if MINTH_RENDER_SUMMARY_HEADER.lower() in (context_text or "").lower():
        if label == "no_minth_vtu":
            warnings.warn(
                "fine_classify WRONG: LLM returned 'no_minth_vtu' but MinTh.vtu "
                "render summary is in context.  Falling back to 'empty_design'."
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
    images=None,
    image_section="",
) -> str:
    """Fine classification for hardware_limit."""
    from .prompts import CLASSIFY_FINE_HW_SYSTEM, CLASSIFY_FINE_USER

    return _fine_run(
        agent,
        CLASSIFY_FINE_HW_SYSTEM,
        CLASSIFY_FINE_USER,
        _fine_payload("hardware_limit", problem_name, main_py, mesh_txt,
                      run_result_json, objective_data, convergence_data,
                      edp_code, image_section=image_section),
        ["memory_limit", "disk_limit", "process_limit"],
        "memory_limit",
        "HW",
        images=images,
    )


def _fine_classify_other(agent, problem_name, context_text, images=None,
                         image_section="") -> str:
    """Free-form diagnosis for 'other' category."""
    from .prompts import CLASSIFY_FINE_OTHER_SYSTEM, CLASSIFY_FINE_OTHER_USER

    prompt_list = [
        {"role": "system", "content": CLASSIFY_FINE_OTHER_SYSTEM},
        {"role": "user", "content": CLASSIFY_FINE_OTHER_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke_multimodal(
        {"problem_name": problem_name, "context_text": context_text,
         "image_section": image_section},
        images=images,
    )

    return raw.strip()


# ======================================================================
# quality check — used when MinTh.vtu exists
# ======================================================================


def quality_check(agent, problem_name: str, context_text: str, images=None,
                  image_section: str = "") -> str:
    """
    Check whether a MinTh.vtu result is genuinely successful.

    Args:
        agent: CodeAgent for LLM calls.
        problem_name: Name of the problem.
        context_text: Full working directory contents.
        images: Optional PNG paths (iteration-label composite).
        image_section: Text describing the attached image(s).

    Returns:
        "pass" or "wrong_result".
    """
    from .prompts import QUALITY_CHECK_SYSTEM, QUALITY_CHECK_USER

    prompt_list = [
        {"role": "system", "content": QUALITY_CHECK_SYSTEM},
        {"role": "user", "content": QUALITY_CHECK_USER},
    ]

    agent.set_prompt_list(prompt_list)
    raw = agent.invoke_multimodal(
        {"problem_name": problem_name, "context_text": context_text,
         "image_section": image_section},
        images=images,
    )

    label = _canonical_label(raw)
    if label == "pass":
        return "pass"
    if label == "wrong_result":
        return "wrong_result"
    warnings.warn(
        f"quality_check: unknown label '{label}' (raw: {raw[:60]!r}). "
        f"Falling back to 'wrong_result'."
    )
    return "wrong_result"
