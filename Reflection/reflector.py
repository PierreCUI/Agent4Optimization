"""
Top-level orchestrator for the Reflection module (Stage 7).

Responsibilities:
1. Collect all files under results/<name>/ using a blacklist strategy.
2. Orchestrate the detect → classify → decide → fix → re-execute loop.
3. Write and read reflection.txt (repair history) to prevent oscillation.
"""

import glob
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .classifier import coarse_classify, fine_classify, quality_check
from .escalation import decide as escalation_decide
from .fixer import apply as fixer_apply
from .renderer import render_and_summarize

# ======================================================================
# file collection — blacklist strategy
# ======================================================================

# File patterns to EXCLUDE (binary, intermediate, or too numerous).
_BLACKLIST_SUFFIXES = {".pdf", ".mesh", ".sol", ".mmg2d", ".mshdist", ".png"}

_BLACKLIST_CONTAINS = {".mesh.gmsh"}

_BLACKLIST_DIRS = {"tmp"}


# Under results/<name>/results/: exclude everything.
# MinTh.vtu existence is checked via _find_minth_vtu (hard rule).
# VTU content goes ONLY to the renderer, never to the LLM.


def _is_blacklisted(rel_path: str) -> bool:
    """Return True if the file should be excluded from context."""
    p = rel_path.lower()

    # Exclude by suffix
    for suffix in _BLACKLIST_SUFFIXES:
        if p.endswith(suffix):
            return True

    # Exclude by substring
    for needle in _BLACKLIST_CONTAINS:
        if needle in p:
            return True

    # Exclude tmp/ entirely
    parts = Path(rel_path).parts
    for part in parts:
        if part in _BLACKLIST_DIRS:
            return True

    # Under results/ subdir — exclude everything (MinTh.vtu is read
    # separately by the renderer; its XML content never goes to the LLM)
    if len(parts) >= 2 and parts[0] == "results":
        return True

    return False


def collect_files(results_dir: Path) -> dict[str, str]:
    """
    Collect all readable text files under results_dir.

    Uses blacklist strategy: reads everything except binary blobs,
    intermediate meshes, PDFs, and iteration snapshots.

    Args:
        results_dir: Path to results/<problem_name>/.

    Returns:
        Dict mapping relative path → file content.
    """
    files = {}
    if not results_dir.exists():
        return files

    for file_path in sorted(results_dir.rglob("*")):
        if not file_path.is_file():
            continue

        rel = str(file_path.relative_to(results_dir))

        if _is_blacklisted(rel):
            continue

        # Read as text; MinTh.vtu XML is only for the renderer text summary
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if not content.strip():
            continue

        files[rel] = content

    return files


# ======================================================================
# context assembly
# ======================================================================


def _assemble_context_text(files: dict[str, str], max_file_chars: int = 50000) -> str:
    """
    Build a single text block from collected files.

    Each file is prefixed with its relative path.
    Content is truncated per-file to max_file_chars.
    """
    parts = []
    for rel, content in sorted(files.items()):
        truncated = (
            content
            if len(content) <= max_file_chars
            else (content[:max_file_chars] + f"\n\n... [truncated, {len(content)} chars total]")
        )
        parts.append(f"=== FILE: {rel} ===\n{truncated}\n")
    return "\n".join(parts)


def _extract_key_files(files: dict[str, str], evidence_chars: int = 1_000_000) -> dict[str, str]:
    """Extract individual key files from the collected files dict."""
    result = {
        "main_py": "",
        "mesh_txt": "",
        "run_result_json": "{}",
        "objective_data": "",
        "convergence_data": "",
        "edp_code": "",
        "problem_description": "",
    }

    for rel, content in files.items():
        filename = Path(rel).name
        path_lower = rel.lower()

        if filename == "main.py":
            result["main_py"] = content
        elif filename == "mesh.txt":
            result["mesh_txt"] = content
        elif filename == "run_result.json":
            result["run_result_json"] = content
        elif filename.startswith("objective") and filename.endswith(".txt"):
            result["objective_data"] += f"=== {rel} ===\n{content[:evidence_chars]}\n\n"
        elif "convergence" in path_lower and filename.endswith(".txt"):
            result["convergence_data"] += f"=== {rel} ===\n{content[:evidence_chars]}\n\n"
        elif filename.endswith(".edp"):
            result["edp_code"] += f"=== {rel} ===\n{content[:evidence_chars]}\n\n"
        elif filename.endswith(".tex"):
            result["problem_description"] += f"=== {rel} ===\n{content}\n\n"
        elif filename.endswith(".txt") and "mesh" not in path_lower:
            result["problem_description"] += f"=== {rel} ===\n{content}\n\n"

    # Also include idp files under include/
    for rel, content in files.items():
        if rel.endswith(".idp"):
            result["edp_code"] += f"=== {rel} ===\n{content[:evidence_chars]}\n\n"

    return result


# ======================================================================
# reflection log
# ======================================================================


def _reflection_log_path(results_dir: Path) -> Path:
    return results_dir / "reflection.txt"


def _init_reflection_log(results_dir: Path, problem_name: str) -> None:
    """Create the reflection log header."""
    log_path = _reflection_log_path(results_dir)
    header = (
        f"{'=' * 70}\nREFLECTION LOG — {problem_name}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 70}\n\n"
    )
    log_path.write_text(header, encoding="utf-8")


def _append_reflection_log(
    results_dir: Path,
    iteration: int,
    coarse: str,
    fine: str,
    action: str,
    rationale: str,
    outcome: str,
) -> None:
    """Append one iteration record to reflection.txt."""
    log_path = _reflection_log_path(results_dir)
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = (
        f"--- Iteration {iteration} [{timestamp}] ---\n"
        f"Coarse:  {coarse}\n"
        f"Fine:    {fine}\n"
        f"Action:  {action}\n"
        f"Fix:     {rationale or '(none)'}\n"
        f"Outcome: {outcome}\n\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def _append_final_status(results_dir: Path, status: str, iterations: int) -> None:
    """Write the final status footer."""
    log_path = _reflection_log_path(results_dir)
    footer = f"{'=' * 70}\nFINAL STATUS: {status} ({iterations} iteration(s))\n{'=' * 70}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(footer)


def _read_reflection_log(results_dir: Path) -> str:
    """Read the full reflection history."""
    log_path = _reflection_log_path(results_dir)
    if log_path.exists():
        return log_path.read_text(encoding="utf-8", errors="replace")
    return ""


# ======================================================================
# MinTh.vtu check + render
# ======================================================================


def _find_minth_vtu(results_dir: Path) -> Optional[Path]:
    """Find a *MinTh.vtu file under results/<name>/results/."""
    pattern = str(results_dir / "results" / "*MinTh.vtu")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def _render_minth(minth_path: Path) -> str:
    """
    Render MinTh.vtu fields to PNGs and return a text summary.

    The PNGs are saved alongside the VTU file.  The returned string
    describes each rendered field and its value range, suitable for
    appending to the LLM context.
    """
    if minth_path is None:
        return ""
    print("    [Reflect] rendering MinTh.vtu → PNG …", end=" ", flush=True)
    summary = render_and_summarize(minth_path)
    png_count = summary.count(".png")
    print(f"{png_count} field(s) rendered.", flush=True)
    return summary


# ======================================================================
# re-execute
# ======================================================================


def _re_execute(results_dir: Path) -> dict:
    """
    Run main.py through AutoFreeFEM + FreeFEM.

    ``run_problem`` persists ``run_result.json`` internally — this
    function does NOT write it a second time.
    """
    from Execution.runner import run_problem

    main_py = results_dir / "main.py"
    if not main_py.exists():
        print("    [Reflect] main.py not found — cannot execute.")
        return {}

    # Clean entire results/ subdir so the next run starts fresh
    import shutil

    results_subdir = results_dir / "results"
    if results_subdir.exists():
        try:
            shutil.rmtree(results_subdir)
        except OSError:
            pass
    results_subdir.mkdir(parents=True, exist_ok=True)

    print("    [Reflect] re-executing …", end=" ", flush=True)
    t0 = time.time()
    result = run_problem(main_py, results_dir)
    dt = time.time() - t0

    py_rc = result["autofreefem"]["python"]["return_code"]
    edp_files = result["autofreefem"]["edp_files"]
    ff_runs = len(result["freefem"])
    print(
        f"Python rc={py_rc}, {len(edp_files)} .edp, {ff_runs} FreeFEM runs ({dt:.0f}s)", flush=True
    )

    return result


# ======================================================================
# hard-pass pre-check (shared by pre-fix and post-fix paths)
# ======================================================================


def _try_hard_pass(key: dict, config) -> tuple[Optional[str], dict]:
    """Return ``(status, diagnostics)``.

    *status* is ``"pass"`` if hard-pass conditions are met, else ``None``.
    *diagnostics* contains ``epochs_completed``, ``ff_info``,
    ``optimize_step_killed``, ``verify_failed`` — usable by callers to
    build quality-check context without re-parsing the same data.

    Pure function — does NOT print or write the reflection log.
    """
    diag = {
        "epochs_completed": 0,
        "ff_info": "",
        "optimize_step_killed": False,
        "verify_failed": False,
    }

    obj_text = key.get("objective_data", "")
    if obj_text:
        diag["epochs_completed"] = max(
            0,
            len([l for l in obj_text.splitlines() if l.strip() and not l.startswith("iteration")]),
        )

    epochs = diag["epochs_completed"]
    if not (
        epochs >= config.MAX_OUTER_ITERATIONS - 2 and epochs >= config.MAX_OUTER_ITERATIONS * 0.95
    ):
        return None, diag

    rr_data = key.get("run_result_json")
    if rr_data and rr_data != "{}":
        try:
            import json as _json

            rr = _json.loads(rr_data)
            for ff in rr.get("freefem", []):
                diag["ff_info"] += (
                    f"FreeFEM {ff.get('edp_file', '?')}: " f"exit={ff.get('return_code', '?')}\n"
                )
                if ff.get("edp_file", "").startswith("optimize") and ff.get("return_code") == "-2":
                    stderr = ff.get("stderr", "")
                    if "STEP TIMEOUT" in stderr and "STARTUP" not in stderr:
                        diag["optimize_step_killed"] = True
            for ff in rr.get("freefem", []):
                if ff.get("edp_file", "").startswith("verify") and ff.get(
                    "return_code", "0"
                ) not in ("0", "-2", "-3"):
                    diag["verify_failed"] = True
        except Exception:
            pass

    if diag["optimize_step_killed"] and not diag["verify_failed"]:
        return "pass", diag
    return None, diag


# ======================================================================
# main orchestrator
# ======================================================================


def run_reflection(agent, info, config) -> str:
    """
    Run the full reflection loop for a single problem.

    Args:
        agent: CodeAgent instance for LLM calls.
        info: Info object with problem_name, results_dir, and reflection_*
              fields.
        config: Module with reflection constants (MAX_REFLECTION_TIMES,
                REFLECTION_STALE_THRESHOLD, etc.).

    Returns:
        Reflection status string: "pass", "give_up_max", "give_up_loop",
        or "give_up" (escalation exhausted).
    """
    results_dir = info.results_dir
    problem_name = info.problem_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize reflection log
    _init_reflection_log(results_dir, problem_name)

    # Load AutoFreeFEM documentation once
    doc_path = Path("database/freefem_functions/list_of_classes.txt")
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""

    for iteration in range(config.MAX_REFLECTION_TIMES):
        info.reflection_attempts = iteration + 1
        print(f"\n  [Reflect] Iteration {iteration + 1}/{config.MAX_REFLECTION_TIMES}", flush=True)

        # --- Step 1: Collect all files ---
        files = collect_files(results_dir)
        if not files:
            print("    [Reflect] No files found — skipping.")
            info.reflection_status = "give_up"
            return "give_up"

        context_text = _assemble_context_text(files)
        key = _extract_key_files(files, evidence_chars=config.REFLECTION_EVIDENCE_CHARS)

        # --- Step 2: Check if we need to (re-)execute ---
        run_result_json = key["run_result_json"]
        if run_result_json == "{}":
            _re_execute(results_dir)
            files = collect_files(results_dir)
            context_text = _assemble_context_text(files)
            key = _extract_key_files(files, evidence_chars=config.REFLECTION_EVIDENCE_CHARS)
            run_result_json = key["run_result_json"]

        # --- Step 3: Coarse classification ---
        # First check MinTh.vtu existence (hard rule)
        minth = _find_minth_vtu(results_dir)

        if minth is not None:
            # Render MinTh.vtu → PNG + text summary for LLM context
            vtu_summary = _render_minth(minth)
            if vtu_summary:
                context_text += f"\n\n=== MinTh.vtu Render Summary ===\n{vtu_summary}\n"

            # ── Hard pre-check: did all expected iterations complete? ──
            hard_pass, diag = _try_hard_pass(key, config)

            if hard_pass == "pass":
                print(
                    f"    [Reflect] PASS — {diag['epochs_completed']}/{config.MAX_OUTER_ITERATIONS}"
                    f" epochs completed, MinTh.vtu exists"
                    f" (optimize step-timeout was post-completion).",
                    flush=True,
                )
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"

            context_text += (
                f"\n=== Quality Check Guidance ===\n"
                f"Expected epochs: {config.MAX_OUTER_ITERATIONS}\n"
                f"Completed epochs (from objective*.txt): {diag['epochs_completed']}\n"
                f"Acceptable late-stage oscillation band: "
                f"{config.OBJECTIVE_STABILIZATION_BAND}\n"
                f"FreeFEM return codes:\n{diag['ff_info']}"
                f"verify exit != 0 = ALWAYS wrong_result.\n"
            )

            # MinTh.vtu exists — but is the result actually good?
            qc = quality_check(agent, problem_name, context_text)
            if qc == "pass":
                print("    [Reflect] PASS — MinTh.vtu exists, result valid.", flush=True)
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"
            # Quality check failed → treat as wrong_result
            coarse = "wrong_result"
            print(f"    [Reflect] Quality check: {qc} — treating as wrong_result.", flush=True)
        else:
            # No MinTh.vtu — LLM classifies
            coarse = coarse_classify(agent, problem_name, context_text)
            print(f"    [Reflect] Coarse: {coarse}", flush=True)

            if coarse == "pass":
                print("    [Reflect] PASS.", flush=True)
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"

        # --- Step 4: Fine classification ---
        reflection_history = _read_reflection_log(results_dir)
        analysis_artifacts = ""
        for rel in sorted(files):
            if rel in ("initial.json", "mesh.json"):
                # Include _source annotations — critical for distinguishing
                # extracted (immutable) vs designed (mutable) values.
                analysis_artifacts += f"=== {rel} ===\n{files[rel][:15000]}\n\n"

        if coarse in (
            "autofreefem_error",
            "freefem_error",
            "non_convergence",
            "wrong_result",
            "hardware_limit",
            "other",
        ):
            fine = fine_classify(
                agent,
                problem_name,
                coarse,
                key["main_py"],
                key["mesh_txt"],
                run_result_json,
                key["objective_data"],
                key["convergence_data"],
                key["edp_code"],
                context_text=context_text,
            )
        else:
            fine = coarse  # pass — no fine classification needed

        print(f"    [Reflect] Fine: {fine}", flush=True)

        # --- Step 5: Escalation decision ---
        action = escalation_decide(
            info.reflection_history,
            coarse,
            fine,
            stale_threshold=config.REFLECTION_STALE_THRESHOLD,
            max_rewrites=config.REFLECTION_MAX_REWRITES,
        )
        print(f"    [Reflect] Action: {action}", flush=True)

        if action == "give_up":
            _append_final_status(results_dir, "give_up", info.reflection_attempts)
            info.reflection_status = "give_up"
            return "give_up"

        # --- Step 6: Apply fix ---
        fix = fixer_apply(
            agent,
            problem_description=key["problem_description"],
            main_py=key["main_py"],
            mesh_txt=key["mesh_txt"],
            run_result_json=run_result_json,
            objective_data=key["objective_data"],
            convergence_data=key["convergence_data"],
            edp_code=key["edp_code"],
            reflection_history=reflection_history,
            doc_text=doc_text,
            analysis_artifacts=analysis_artifacts,
            coarse=coarse,
            fine=fine,
            action=action,
            outer_iterations=config.MAX_OUTER_ITERATIONS,
            inner_iterations=config.MAX_INNER_ITERATIONS,
        )

        if fix.get("_error"):
            print(f"    [Reflect] Fix parse error: {fix['_error']}", flush=True)
            _append_reflection_log(
                results_dir,
                iteration + 1,
                coarse,
                fine,
                action,
                "(parse error)",
                f"fix parse failed: {fix['_error']}",
            )
            info.reflection_history.append(
                {
                    "coarse": coarse,
                    "fine": fine,
                    "action": action,
                    "fix_rationale": "(parse error)",
                    "outcome": f"fix parse failed: {fix['_error']}",
                }
            )
            continue

        rationale = fix.get("rationale", "")
        new_main = fix.get("main", "")
        new_mesh = fix.get("mesh", "")

        if not new_main and not new_mesh:
            print("    [Reflect] Fix produced empty result — giving up.", flush=True)
            _append_reflection_log(
                results_dir,
                iteration + 1,
                coarse,
                fine,
                action,
                rationale,
                "fix produced empty result",
            )
            info.reflection_history.append(
                {
                    "coarse": coarse,
                    "fine": fine,
                    "action": action,
                    "fix_rationale": rationale,
                    "outcome": "fix produced empty result",
                }
            )
            _append_final_status(results_dir, "give_up_loop", info.reflection_attempts)
            info.reflection_status = "give_up_loop"
            return "give_up_loop"

        # --- Step 7: Apply the fix to disk ---
        if new_main:
            out_path = results_dir / "main.py"
            out_path.write_text(new_main, encoding="utf-8")
        if new_mesh:
            out_path = results_dir / "mesh.txt"
            out_path.write_text(new_mesh, encoding="utf-8")

        # --- Step 8: Re-execute ---
        result = _re_execute(results_dir)

        # --- Step 9: Evaluate outcome ---
        minth_after = _find_minth_vtu(results_dir)
        py_rc = result.get("autofreefem", {}).get("python", {}).get("return_code", "?")
        edp_files = result.get("autofreefem", {}).get("edp_files", [])
        ff_records = result.get("freefem", [])
        ff_rcs = [f.get("return_code", "?") for f in ff_records]

        if minth_after is not None:
            outcome = f"MinTh.vtu generated ✓ — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
        elif edp_files and all(rc == "0" for rc in ff_rcs):
            outcome = f"FreeFEM OK but no MinTh.vtu — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
        elif edp_files:
            outcome = (
                f"FreeFEM failed — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
            )
        else:
            outcome = f"Python failed — rc={py_rc}, {len(edp_files)} .edp"

        # --- Step 10: Record history ---
        _append_reflection_log(results_dir, iteration + 1, coarse, fine, action, rationale, outcome)
        info.reflection_history.append(
            {
                "coarse": coarse,
                "fine": fine,
                "action": action,
                "fix_rationale": rationale,
                "outcome": outcome,
            }
        )

        # If MinTh.vtu now exists, do quality check next iteration
        if minth_after is not None:
            files = collect_files(results_dir)
            context_text = _assemble_context_text(files)
            key_after = _extract_key_files(files, evidence_chars=config.REFLECTION_EVIDENCE_CHARS)

            # Render MinTh.vtu → PNG + text summary
            vtu_summary = _render_minth(minth_after)
            if vtu_summary:
                context_text += f"\n\n=== MinTh.vtu Render Summary ===\n{vtu_summary}\n"

            # ── Hard pre-check: all iterations completed? ──
            hp, diag = _try_hard_pass(key_after, config)
            if hp == "pass":
                print(
                    f"    [Reflect] PASS — {diag['epochs_completed']}/{config.MAX_OUTER_ITERATIONS}"
                    f" epochs completed, MinTh.vtu exists"
                    f" (optimize step-timeout was post-completion).",
                    flush=True,
                )
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"

            context_text += (
                f"\n=== Quality Check Guidance ===\n"
                f"Expected epochs: {config.MAX_OUTER_ITERATIONS}\n"
                f"Completed epochs: {diag['epochs_completed']}\n"
                f"Acceptable band: {config.OBJECTIVE_STABILIZATION_BAND}\n"
                f"FreeFEM return codes:\n{diag['ff_info']}"
                f"verify exit != 0 = ALWAYS wrong_result.\n"
            )

            qc = quality_check(agent, problem_name, context_text)
            if qc == "pass":
                print("    [Reflect] PASS — fix succeeded, result valid.", flush=True)
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"

    # --- Loop exhausted ---
    print(f"    [Reflect] Max iterations ({config.MAX_REFLECTION_TIMES}) reached.", flush=True)
    _append_final_status(results_dir, "give_up_max", info.reflection_attempts)
    info.reflection_status = "give_up_max"
    return "give_up_max"
