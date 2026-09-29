"""Main pipeline — orchestrates all stages for every demo problem."""

import sys
import time
from pathlib import Path

import configs
from API import (
    AgentRouter,
    DeepSeekCodeAgentV4FlashVision,
    DeepSeekCodeAgentV4Pro,
    test_connection,
)
from Execution.runner import run_problem as run_execution
from Generation import generate_code, run_static_repair
from Generation.check_static import strip_ff_comments
from Generation.select_examples import load_selected_examples, select_examples
from info import Info
from Initialization import collect_input_files, run_spec_generation, safety_check
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
        print("  [Safety] PASS")
    return True


def _stage_copy_inputs(info: Info) -> None:
    """Copy all input files from demos to results so the runner can use them."""
    for fname, content in info.input_files.items():
        info.save_text(fname, content)


def _stage_parse(agent, info: Info) -> bool:
    """Run the one-loop spec generation (both JSONs together).

    Returns False when the problem or the mesh spec ended up empty (the
    loop exceeded its repair budget — no spec means code generation has
    nothing to work on, so stop instead of generating on an empty spec).
    """
    print("  [Parse] spec generation (initial + mesh together) ...", end=" ", flush=True)
    t0 = time.time()
    specs = run_spec_generation(
        agent, info.input_files, info.problem_name, output_dir=info.results_dir,
        history_log=info.results_dir / "history.txt",
    )
    info.problem_spec = specs.get("initial") or {}
    info.mesh_spec = specs.get("mesh") or {}
    if not (info.problem_spec or {}).get("designed"):
        print("FAILED: empty ProblemSpec (generation failed after repairs).")
        return False
    if not (info.mesh_spec or {}).get("designed"):
        print("FAILED: empty mesh spec (generation failed after repairs).")
        return False
    info.save_json("initial.json", info.problem_spec)
    info.save_json("mesh.json", info.mesh_spec)
    # Also write mesh.txt for the execution phase (rule: the generated
    # mesh carries no comments — stripped deterministically)
    mesh_code = info.mesh_spec.get("designed", {}).get("mesh_code", "")
    if mesh_code:
        info.save_text("mesh.txt", strip_ff_comments(mesh_code))
    print(f"({time.time() - t0:.1f}s)", flush=True)
    return True


def _stage_select_examples(agent, info: Info) -> None:
    """Select relevant reference examples via LLM."""
    print("  [Select] example_selector ...", end=" ", flush=True)
    t0 = time.time()
    info.selected_names = select_examples(agent, info.problem_spec, info.mesh_spec)
    if not info.selected_names:
        info.selected_names = _fallback_examples()
        print(f"-> fallback {len(info.selected_names)} examples", end=" ", flush=True)
    info.save_text("examples.txt", "\n".join(info.selected_names))
    print(f"-> {info.selected_names} ({time.time() - t0:.1f}s)", flush=True)


def _fallback_examples() -> list[str]:
    """Return all available example names when the selector returns empty."""
    db = Path("database/freefem_examples")
    if not db.exists():
        return []
    return sorted(d.name for d in db.iterdir() if d.is_dir())


def _stage_generate(agent, info: Info) -> bool:
    """Generate AutoFreeFEM code via LLM.

    Returns True on success; False when the LLM response could not be
    parsed (raw text would be garbage as main.py, so the pipeline stops
    instead of running hours of execution+reflection on it).
    """
    print("  [Generate] coder ...", end=" ", flush=True)
    t0 = time.time()

    # Load doc (shared loader — the same logic coder._load_documentation provides)
    from Generation.coder import _load_documentation

    doc_text = _load_documentation()

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
    if not isinstance(result, dict):
        print(f"FAILED: generate_code returned a non-dict result ({type(result).__name__}).")
        return False
    if result.get("_error"):
        print(f"FAILED: {result['_error']}")
        return False
    # The LLM JSON is parsed verbatim — "main" may be null/number/list.
    main_text = result.get("main")
    if not isinstance(main_text, str) or not main_text.strip():
        print(f"FAILED: generate_code returned no usable main code ({type(main_text).__name__}).")
        return False
    info.generated_code = main_text
    info.save_text("main.py", info.generated_code)
    lines = info.generated_code.count("\n") + 1
    print(f"-> {lines} lines ({time.time() - t0:.1f}s)", flush=True)
    return True


def _stage_static_check(agent, info: Info) -> None:
    """Run the merged check -> review -> repair loop (ex-static check +
    ex-parameter review).  The code is generated once; this stage only
    REPAIRS the reported issues (at most configs.MAX_STATIC_FIXES
    repairs), with an in-memory repair history maintained like the
    Initialization spec loop.
    """
    print("  [Check] static repair ...", end=" ", flush=True)
    t0 = time.time()
    mesh_txt = info.mesh_spec.get("designed", {}).get("mesh_code", "") if info.mesh_spec else ""
    result = run_static_repair(
        agent,
        info.generated_code,
        problem_spec=info.problem_spec,
        mesh_spec=info.mesh_spec,
        problem_description=info.problem_description,
        mesh_txt=mesh_txt,
        examples=info.examples,
        history_log=info.results_dir / "history.txt",
    )
    info.generated_code = result.get("main") or info.generated_code
    info.save_text("main.py", info.generated_code)
    history = result.get("history") or []
    if result.get("_passed"):
        print(f"OK ({len(history)} repair(s)) ({time.time() - t0:.1f}s)", flush=True)
    else:
        print(f"FAILED: repair budget exhausted ({len(history)} repair(s)) "
              f"— continuing ({time.time() - t0:.1f}s)", flush=True)
        print("    (the reflection loop traces runtime errors on main.py)")
    for w in result.get("warnings") or []:
        print(f"    [W] {w.get('check')}: {w.get('message')}")


def _stage_run(info: Info) -> None:
    """
    Run the generated code through AutoFreeFEM + FreeFEM.

    run_execution persists run_result.json to the results dir itself —
    the reflection loop reads it back from disk (single writer).
    """
    main_py = info.results_dir / "main.py"
    if not main_py.exists():
        print("  [Run] main.py not found — skipping.")
        return

    print("  [Run] executing …")
    t0 = time.time()
    result = run_execution(
        main_py,
        info.results_dir,
        history_log=info.results_dir / "history.txt",
        history_label="Execution/run_problem (Stage 6 formal run)",
    )
    dt = time.time() - t0

    py_rc = result["autofreefem"]["python"]["return_code"]
    edp_files = result["autofreefem"]["edp_files"]
    print(
        f"  [Run] Python exit={py_rc}, {len(edp_files)} .edp, {len(result['freefem'])} FreeFEM runs ({dt:.0f}s)"
    )
    for ff in result["freefem"]:
        if ff["return_code"] != "0":
            print(f"    FreeFEM {ff['edp_file']}: exit={ff['return_code']}")


# ======================================================================
# Stage 7 — Reflection
# ======================================================================


def _stage_reflect(agent, info: Info, cfg) -> None:
    """Run the reflection loop: diagnose errors, fix, re-execute."""
    print("  [Reflect] starting …")
    t0 = time.time()
    status = run_reflection(agent, info, cfg)
    dt = time.time() - t0
    print(f"  [Reflect] status={status}, attempts={info.reflection_attempts} ({dt:.0f}s)")


# ======================================================================
# main
# ======================================================================


def main(agent, problem_name: str) -> None:
    """Run the full pipeline on a single problem."""
    # UTF-8 stdout: pipeline prints use arrows/bullets that are outside
    # the cp1252 console encoding (e.g. '→' in the reflection loop) —
    # same guard every per-module __main__ block uses.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    info = Info(problem_name)
    info.input_files = collect_input_files(Path("demos") / problem_name)
    if not info.input_files:
        print("  No input files found — skipping.")
        return
    # Every round of the four stages (Initialization -> Generation ->
    # Execution -> Reflection) appends FULL records — which function is
    # working (for a repair, which round), what went in and what came
    # out — to results/<problem>/history.txt.  A record only: never
    # reflection evidence.
    agent.attach_history(info.results_dir / "history.txt")

    # Connection tester FIRST: an unreachable model ends the whole run
    # immediately instead of burning stage after stage on connection
    # errors.
    if not test_connection(agent):
        print("  [main] LLM connection test FAILED — ending immediately.")
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
    if not _stage_parse(agent, info):
        return

    # ── Stage 3: Example Selection ──────────────────────────────
    _stage_select_examples(agent, info)

    # ── Stage 4: Code Generation ────────────────────────────────
    if not _stage_generate(agent, info):
        return

    # ── Stage 5: Static Check & Auto-Fix + Parameter Review ─────
    _stage_static_check(agent, info)

    # ── Stage 6: Execution ──────────────────────────────────────
    _stage_run(info)

    # ── Stage 7: Reflection ──────────────────────────────────────
    _stage_reflect(agent, info, configs)

    elapsed = time.time() - t_start
    print(f"\n  Pipeline complete.  Reflection: {info.reflection_status}.  Total: {elapsed:.0f}s")


# ======================================================================
if __name__ == "__main__":
    # Two-model setup: text-only calls (Stages 1-5, text classification)
    # go to the pro model; calls with images (design snapshots in the
    # reflection loop) go to the vision model.  AgentRouter dispatches.
    demos_dir = Path("demos")
    if not demos_dir.exists():
        print(f"Directory not found: {demos_dir}")
        sys.exit(1)
    text_agent = DeepSeekCodeAgentV4Pro()
    vision_agent = DeepSeekCodeAgentV4FlashVision()
    agent = AgentRouter(text_agent, vision_agent)
    # One-time: init database examples before processing any problem
    run_example_initialization(agent)
    for problem in sorted(demos_dir.iterdir()):
        if not problem.is_dir():
            continue
        # One failing problem (LLM API transport error, fs hiccup) must
        # not abort the whole batch: record the failure and continue.
        try:
            main(agent, problem.name)
        except Exception as e:  # noqa: BLE001 — batch-level isolation
            print(f"\n  [main] FAILED: {problem.name}: {e}", flush=True)
