"""Main pipeline — orchestrates all stages for every demo problem."""

import sys
import time
from pathlib import Path

import configs
from API import DeepSeekCodeAgentV4Flash
from Execution.runner import run_problem as run_execution
from Generation import auto_fix, generate_code, static_check
from Generation.select_examples import load_selected_examples, select_examples
from info import Info
from Initialization import collect_input_files, handle_mesh, parse_problem, safety_check
from Initialization.example_initialization import run_example_initialization
from Reflection.reflector import run_reflection

# ======================================================================
# pipeline stages
# ======================================================================


def _stage_safety(agent, info: Info) -> bool:
    """Return True if the problem should be processed further."""
    status, filtered_desc, msg = safety_check(agent, info.problem_description)
    if status == "reject":
        print(f"  [Safety] REJECTED: {msg}")
        return False
    elif status == "partial":
        print(f"  [Safety] PARTIAL: {msg}")
        info.problem_description = filtered_desc
    else:
        print(f"  [Safety] PASS")
    return True


def _stage_copy_inputs(info: Info) -> None:
    """Copy all input files from demos to results so the runner can use them."""
    for fname, content in info.input_files.items():
        info.save_text(fname, content)


def _stage_parse(agent, info: Info) -> None:
    """Run parse_problem + handle_mesh."""
    print(f"  [Parse] problem_parser ...", end=" ", flush=True)
    t0 = time.time()
    info.problem_spec = parse_problem(agent, info.input_files, info.problem_name)
    info.save_json("initial.json", info.problem_spec)
    print(f"({time.time() - t0:.1f}s)", flush=True)

    print(f"  [Parse] mesh_handler ...", end=" ", flush=True)
    t0 = time.time()
    info.mesh_spec = handle_mesh(
        agent,
        info.input_files,
        problem_context=info.problem_spec.get("designed") if info.problem_spec else None,
    )
    info.save_json("mesh.json", info.mesh_spec)
    # Also write mesh.txt for the execution phase
    mesh_code = info.mesh_spec.get("designed", {}).get("mesh_code", "")
    if mesh_code:
        info.save_text("mesh.txt", mesh_code)
    print(f"({time.time() - t0:.1f}s)", flush=True)


def _stage_select_examples(agent, info: Info) -> None:
    """Select relevant reference examples via LLM."""
    print(f"  [Select] example_selector ...", end=" ", flush=True)
    t0 = time.time()
    info.selected_names = select_examples(agent, info.problem_spec, info.mesh_spec)
    if not info.selected_names:
        info.selected_names = _fallback_examples()
        print(f"-> fallback {len(info.selected_names)} examples", end=" ", flush=True)
    info.save_text("examples.txt", "\n".join(info.selected_names))
    print(f"-> {info.selected_names} ({time.time() - t0:.1f}s)", flush=True)


def _fallback_examples() -> list[str]:
    """Return all available example names when the selector returns empty."""
    from pathlib import Path as _Path

    db = _Path("database/freefem_examples")
    if not db.exists():
        return []
    return sorted(d.name for d in db.iterdir() if d.is_dir())


def _stage_generate(agent, info: Info) -> None:
    """Generate AutoFreeFEM code via LLM."""
    print(f"  [Generate] coder ...", end=" ", flush=True)
    t0 = time.time()

    # Load doc
    doc_path = Path("database/freefem_functions/list_of_classes.txt")
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""

    # Load full .py for selected examples
    info.examples = load_selected_examples(info.selected_names)

    result = generate_code(
        agent,
        info.problem_spec,
        info.mesh_spec,
        doc_text=doc_text,
        examples=info.examples,
        outer_iterations=configs.MAX_OUTER_ITERATIONS,
        inner_iterations=configs.MAX_INNER_ITERATIONS,
    )
    info.generated_code = result.get("main", "")
    info.save_text("main.py", info.generated_code)
    lines = info.generated_code.count("\n") + 1
    print(f"-> {lines} lines ({time.time() - t0:.1f}s)", flush=True)


def _stage_static_check(agent, info: Info) -> None:
    """Run static checks and optionally auto-fix."""
    print(f"  [Check] static_check ...", end=" ", flush=True)
    errors = static_check(info.generated_code, info.mesh_spec, info.problem_spec)
    err_n = sum(1 for e in errors if e["severity"] == "error")
    warn_n = sum(1 for e in errors if e["severity"] == "warning")
    if errors:
        print(f"-> {err_n}E+{warn_n}W", flush=True)
        for e in errors:
            print(f"    [{e['severity'].upper()}] {e['check']}: {e['message']}")

        print(f"  [Fix] auto_fix ...", end=" ", flush=True)
        fixed = auto_fix(agent, info.generated_code, errors)
        if "_error" in fixed:
            print(f"FAILED: {fixed['_error']}")
        else:
            info.fixed_code = fixed.get("main", "")
            info.save_text("main.py", info.fixed_code)
            info.generated_code = info.fixed_code
            recheck = static_check(info.generated_code, info.mesh_spec, info.problem_spec)
            print(f"-> {len(recheck)} remaining errors")
    else:
        print(f"OK", flush=True)


def _stage_run(info: Info) -> dict:
    """
    Run the generated code through AutoFreeFEM + FreeFEM.

    Returns the execution result dict (also saved to disk and stored on info).
    """
    main_py = info.results_dir / "main.py"
    if not main_py.exists():
        print(f"  [Run] main.py not found — skipping.")
        return {}

    print(f"  [Run] executing …")
    t0 = time.time()
    result = run_execution(main_py, info.results_dir)
    dt = time.time() - t0

    # Persist for debugging and the reflection loop
    info.run_result = result
    info.save_json("run_result.json", result)

    py_rc = result["autofreefem"]["python"]["return_code"]
    edp_files = result["autofreefem"]["edp_files"]
    print(
        f"  [Run] Python exit={py_rc}, {len(edp_files)} .edp, {len(result['freefem'])} FreeFEM runs ({dt:.0f}s)"
    )
    for ff in result["freefem"]:
        if ff["return_code"] != "0":
            print(f"    FreeFEM {ff['edp_file']}: exit={ff['return_code']}")

    return result


# ======================================================================
# Stage 7 — Reflection
# ======================================================================


def _stage_reflect(agent, info: Info, cfg) -> None:
    """Run the reflection loop: diagnose errors, fix, re-execute."""
    print(f"  [Reflect] starting …")
    t0 = time.time()
    status = run_reflection(agent, info, cfg)
    dt = time.time() - t0
    print(f"  [Reflect] status={status}, attempts={info.reflection_attempts} ({dt:.0f}s)")


# ======================================================================
# main
# ======================================================================


def main(agent, problem_name: str) -> None:
    """Run the full pipeline on a single problem."""
    info = Info(problem_name)
    info.input_files = collect_input_files(Path("demos") / problem_name)
    if not info.input_files:
        print("  No input files found — skipping.")
        return

    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  {problem_name}")
    print(f"{'=' * 60}")

    # Concatenate all input files as the problem description
    info.problem_description = "\n\n".join(
        f"=== {fname} ===\n{content}" for fname, content in info.input_files.items()
    )

    # ── Stage 0: Copy inputs to results ─────────────────────────
    _stage_copy_inputs(info)

    # ── Stage 1: Safety Check ───────────────────────────────────
    if not _stage_safety(agent, info):
        return

    # ── Stage 2: Problem Parsing + Mesh Handling ────────────────
    _stage_parse(agent, info)

    # ── Stage 3: Example Selection ──────────────────────────────
    _stage_select_examples(agent, info)

    # ── Stage 4: Code Generation ────────────────────────────────
    _stage_generate(agent, info)

    # ── Stage 5: Static Check & Auto-Fix ────────────────────────
    _stage_static_check(agent, info)

    # ── Stage 6: Execution ──────────────────────────────────────
    _stage_run(info)

    # ── Stage 7: Reflection ──────────────────────────────────────
    _stage_reflect(agent, info, configs)

    elapsed = time.time() - t_start
    print(f"\n  Pipeline complete.  Reflection: {info.reflection_status}.  Total: {elapsed:.0f}s")


# ======================================================================
if __name__ == "__main__":
    agent = DeepSeekCodeAgentV4Flash()
    # One-time: init database examples before processing any problem
    run_example_initialization(agent)
    demos_dir = Path("demos")
    if not demos_dir.exists():
        print(f"Directory not found: {demos_dir}")
        sys.exit(1)
    for problem in sorted(demos_dir.iterdir()):
        if problem.is_dir():
            main(agent, problem.name)
