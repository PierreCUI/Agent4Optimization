"""
Top-level orchestrator for the Reflection module (Stage 7).

Responsibilities:
1. Collect all files under results/<name>/ using a blacklist strategy.
2. Orchestrate the detect → classify → decide → probe → synthesise →
   re-execute loop.  Each round tests up to ``REFLECTION_PARALLEL_MAX``
   candidate fixes IN PARALLEL under ``results/<name>/temp/``, each with
   a short probe run (``REFLECTION_PROBE_MIN_EPOCHS`` ..
   ``REFLECTION_PROBE_MAX_EPOCHS`` epochs); an LLM synthesiser then
   merges the probe evidence into ONE next-step code for the formal run.
3. Write and read reflection.txt (repair history) to prevent oscillation.
4. Remove the temp directory when the process ends (pass / give_up) —
   never mid-process.
"""

import glob
import json
import os
import re
import shutil
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from Generation.symmetry import detect_symmetry, mesh_dims
from utils import code_masked_positions

from .classifier import (
    MINTH_RENDER_SUMMARY_HEADER,
    coarse_classify,
    fine_classify,
    quality_check,
)
from .escalation import decide as escalation_decide
from .fixer import apply as fixer_apply
from .renderer import (
    render_and_summarize,
    render_iteration_labels_composite,
)

# ======================================================================
# file collection — blacklist strategy
# ======================================================================

# File patterns to EXCLUDE (binary, intermediate, or too numerous).
_BLACKLIST_SUFFIXES = {".pdf", ".mesh", ".sol", ".mmg2d", ".mshdist", ".png"}

_BLACKLIST_CONTAINS = {".mesh.gmsh"}

# Pipeline operation log — for HUMAN debugging, not LLM evidence.  It
# holds the pipeline's own LLM inputs/outputs and would bloat (and
# bias) the reflection context if collected.  The reflection log is
# excluded for the same reason: it is the repair history, which
# reaches the fixer through the dedicated reflection_history argument.
_BLACKLIST_NAMES = {"history.txt", "reflection.txt"}

# tmp/ (execution scratch) and temp/ (parallel probe candidates) are
# working directories of the pipeline itself — never reflection evidence.
# Under results/<name>/results/: exclude everything (MinTh.vtu existence
# is checked via _find_minth_vtu — hard rule; VTU content goes ONLY to
# the renderer, never to the LLM).
_BLACKLIST_DIRS = {"tmp", "temp"}


def _is_blacklisted(rel_path: str) -> bool:
    """Return True if the file should be excluded from context."""
    p = rel_path.lower()

    # Exclude by exact name (e.g. the pipeline operation log)
    if Path(rel_path).name in _BLACKLIST_NAMES:
        return True

    # Exclude by suffix
    for suffix in _BLACKLIST_SUFFIXES:
        if p.endswith(suffix):
            return True

    # Exclude by substring
    for needle in _BLACKLIST_CONTAINS:
        if needle in p:
            return True

    # Exclude tmp/ and temp/ entirely (case-insensitive — a hand-made
    # TEMP directory must not leak into the context either)
    parts = Path(rel_path).parts
    for part in parts:
        if part.lower() in _BLACKLIST_DIRS:
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
        # POSIX separators: every consumer matches on '/' — a Windows
        # backslash in the key would silently break string matches.
        rel = rel.replace("\\", "/")

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


def _extract_key_files(files: dict[str, str], evidence_chars: int = None) -> dict[str, str]:
    """Extract individual key files from the collected files dict.

    *evidence_chars* defaults to ``configs.REFLECTION_EVIDENCE_CHARS`` —
    no hard-coded numbers, so a config change cannot drift out of sync.
    """
    if evidence_chars is None:
        from configs import REFLECTION_EVIDENCE_CHARS

        evidence_chars = REFLECTION_EVIDENCE_CHARS
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
            # The full run result embeds the ENTIRE Python + FreeFEM
            # stdout/stderr, which grows unbounded on verbose runs —
            # truncate to the same evidence budget as every other
            # stream before it reaches the LLM.
            if len(content) > evidence_chars:
                result["run_result_json"] = (
                    content[:evidence_chars]
                    + f"\n\n... [truncated, {len(content)} chars total]"
                )
            else:
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
            # The repair history (reflection.txt) is NOT part of the
            # original problem description — it reaches the fixer through
            # the dedicated reflection_history argument instead.
            if filename == "reflection.txt":
                continue
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


def _load_run_result(results_dir: Path) -> dict:
    """Load the full run_result.json from disk (untruncated), or {} when absent."""
    try:
        raw = (results_dir / "run_result.json").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not raw or raw == "{}":
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _format_outcome(result: dict, minth_exists: bool) -> str:
    """Format the post-execution outcome line of a reflection record."""
    py_rc = result.get("autofreefem", {}).get("python", {}).get("return_code", "?")
    edp_files = result.get("autofreefem", {}).get("edp_files", [])
    ff_rcs = [f.get("return_code", "?") for f in result.get("freefem", [])]
    if minth_exists:
        return f"MinTh.vtu generated [ok] — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
    if edp_files and all(rc == "0" for rc in ff_rcs):
        return f"FreeFEM OK but no MinTh.vtu — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
    if edp_files:
        return f"FreeFEM failed — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
    return f"Python failed — rc={py_rc}, {len(edp_files)} .edp"


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
    print("    [Reflect] rendering MinTh.vtu -> PNG ...", end=" ", flush=True)
    try:
        summary = render_and_summarize(minth_path)
    except Exception as e:
        # a render failure (missing meshio/matplotlib, corrupt VTU) must
        # never abort the reflection loop — the text evidence remains
        print(f"render failed: {e}", flush=True)
        return ""
    print("done.", flush=True)
    return summary


def _vision_context(results_dir: Path) -> tuple[list[str], str]:
    """Render the iteration-label composite for the multimodal loop.

    Returns ``(images, image_section)`` — images is a list of PNG paths
    (empty when no iteration snapshots exist), image_section the text
    block describing the attached image for the prompt.
    """
    try:
        snap_path, snap_summary = render_iteration_labels_composite(
            results_dir / "results", results_dir
        )
    except Exception as e:
        print(f"    [Reflect] snapshot composite render failed: {e}", flush=True)
        return [], ""
    if snap_path is None:
        return [], ""
    image_section = (
        "\n\n=== ITERATION SNAPSHOTS (image attached) ===\n"
        + snap_summary
        + "\nYou CAN see the attached composite image — use the design "
          "evolution (solid/void distribution per epoch) to judge the "
          "failure and decide the fix.\n"
    )
    return [snap_path], image_section


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

    # results/<problem>/results is wiped by run_problem (Execution/runner.py)
    # before every FreeFEM pass — no cleanup needed here; duplicating it
    # would only re-do the same work microseconds later.

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
# quality-check diagnostics (shared by pre-fix and post-fix paths)
# ======================================================================


def _quality_diag(results_dir: Path, key: dict) -> dict:
    """Collect the diagnostics that guide the soft quality check.

    *diagnostics* contains ``epochs_completed`` (data rows in
    objective*.txt) and ``ff_info`` (FreeFEM return codes of the
    optimize/verify runs).  The FreeFEM codes are parsed from the FULL
    run_result.json on disk — the LLM-bound copy in *key* is truncated
    and would never parse.  Pure function — does NOT print or write the
    reflection log.
    """
    diag = {"epochs_completed": 0, "ff_info": ""}

    obj_text = key.get("objective_data", "")
    if obj_text:
        for line in obj_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("iteration"):
                continue
            try:
                int(stripped.split(",")[0].strip())
            except ValueError:
                continue
            diag["epochs_completed"] += 1

    rr_data = None
    try:
        rr_data = (results_dir / "run_result.json").read_text(encoding="utf-8", errors="replace")
    except OSError:
        rr_data = key.get("run_result_json")
    if rr_data and rr_data != "{}":
        try:
            rr = json.loads(rr_data)
            for ff in rr.get("freefem", []):
                diag["ff_info"] += (
                    f"FreeFEM {ff.get('edp_file', '?')}: "
                    f"exit={ff.get('return_code', '?')}\n"
                )
        except Exception:
            pass
    return diag


def _expected_epochs(main_py: str, config) -> int:
    """Iterations requested by the problem's own ``setup_optimization``.

    A benchmark may legitimately ask for fewer epochs than the global
    default — reporting ``MAX_OUTER_ITERATIONS`` then would make a
    complete 30/30 run look like a crash at 30/100 and bias the quality
    check toward wrong_result.  Local import mirrors ``_re_execute``.
    """
    from Execution.runner import _extract_iterations

    return _extract_iterations(main_py) or config.MAX_OUTER_ITERATIONS


def _symmetry_guidance(key: dict) -> str:
    """State the problem's symmetry; the design is judged from the image.

    Only the FACT of the problem's symmetry is computed here (mirror
    geometry with matching boundary labels — a property of mesh.txt, not
    of the result).  Whether the optimised design respects it is judged
    by the vision model from the design image: a pixel-level analysis
    cannot distinguish a genuinely asymmetric design from the small
    raggedness that finite-element discretisation and remeshing leave on
    a symmetric design, and the model's tolerance is stated in the text.
    """
    axes = detect_symmetry(key.get("mesh_txt", "") or "")
    if not axes:
        return ""
    L, H = mesh_dims(key.get("mesh_txt", "") or "")
    where = ", ".join(
        (f"x = {L / 2:g}" if a == "x" else f"y = {H / 2:g}") for a in axes
    )
    phi_sug = f"-0.4 - cos(2*pi*x/{L})*cos(2*pi*y/{H})"
    return (
        f"\n=== Symmetry check (judge the design from the image) ===\n"
        f"The problem is symmetric: the mesh geometry and its boundary "
        f"labels are mirror-symmetric about {where} (the heated and the "
        f"cooled segments map onto themselves).  The final design should "
        f"therefore LOOK symmetric in the design image.\n"
        f"Tolerate SMALL asymmetries: finite-element discretisation and "
        f"remeshing leave ragged edges, slightly different numbers of "
        f"elements and minor detail mismatches.  These are expected and "
        f"are NOT a failure.\n"
        f"Only a CLEARLY lopsided design (structure present on one side "
        f"and absent on the other) is wrong_result.  Its fix is a "
        f"symmetric initial phi, e.g. '{phi_sug}' — sin-based oscillatory "
        f"terms about the mirror axis are anti-symmetric and break the "
        f"symmetry.\n"
    )


def _minth_label_png(results_dir: Path) -> Optional[Path]:
    """The rendered Label PNG of the final design (None when absent)."""
    minth = _find_minth_vtu(results_dir)
    if minth is None:
        return None
    png = minth.parent.parent / f"{minth.stem}_cell_Label.png"
    return png if png.exists() else None


def _deterministic_convergence_verdict(diag: dict, key: dict, expected_epochs: int) -> str:
    """Deterministic convergence pre-check for the soft quality gate.

    The LLM judge reads images and curves well but cannot reliably
    discriminate a slow monotone drift from a plateau (a coherent-looking
    design plus a gently descending tail reads as pass).  This check
    hard-fails the obvious case: the run crashed well before the
    requested epochs while the objective was STILL strictly decreasing
    over its final epochs — the design has not converged, and accepting
    it would report a mid-optimisation design as the final result.

    Returns "" (let the LLM decide) or a verdict message that forces
    wrong_result and tells the fixer why.
    """
    if diag["epochs_completed"] >= expected_epochs:
        return ""  # completed run: monotone descent to the end is pattern (b)
    values = []
    for line in (key.get("objective_data") or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("iteration"):
            continue
        parts = stripped.split(",")
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1].strip()))
        except ValueError:
            continue
    if len(values) < 4:
        return ""  # too few epochs for a tail trend — LLM decides
    tail = values[-5:]
    decreasing = all(b < a for a, b in zip(tail, tail[1:]))
    increasing = all(b > a for a, b in zip(tail, tail[1:]))
    if not (decreasing or increasing):
        return ""  # oscillating tail — plateau-like, LLM decides
    if abs(tail[-1]) < 1e-12:
        return ""  # converged to ~0 objective — the drift ratio below
        # would divide by zero; the LLM decides
    rel_drift = abs(tail[0] - tail[-1]) / abs(tail[-1])
    if rel_drift <= 1e-3:
        return ""  # drift is noise-level — LLM decides
    if decreasing:
        detail = (
            f"the objective was STILL strictly decreasing over its final "
            f"{len(tail)} epochs ({tail[0]:.4g} -> {tail[-1]:.4g}, "
            f"{rel_drift * 100:.3f}% over the tail).  The design has not "
            f"converged: the optimisation was cut short (typically by mesh "
            f"degeneration)."
        )
    else:
        detail = (
            f"the objective was strictly INCREASING over its final "
            f"{len(tail)} epochs ({tail[0]:.4g} -> {tail[-1]:.4g}, "
            f"{rel_drift * 100:.3f}% over the tail) — a U-shaped drift "
            f"after a minimum, the signature of fundamentally wrong "
            f"parameters or mesh degeneration feeding the gradient."
        )
    return (
        f"\n=== Deterministic convergence pre-check: FAILED ===\n"
        f"The run crashed after {diag['epochs_completed']} of {expected_epochs} "
        f"requested epochs while {detail}  The fix must stabilise the "
        f"numerical parameters (smaller v0, adjusted hmin/diffusion) and "
        f"re-run so the objective reaches a plateau.\n"
    )


def _quality_guidance(config, diag: dict, expected_epochs: int) -> str:
    """Build the guidance text for the LLM quality check.

    Gives the model the FACTS (expected/completed epochs, FreeFEM exit
    codes) only — judging success vs premature stop is left entirely to
    the LLM, using the objective history, the rendered design snapshots,
    and the return codes.  *expected_epochs* is the problem's own
    ``iterations=`` value.
    """
    return (
        f"\n=== Quality Check Guidance ===\n"
        f"Expected epochs (iterations= in main.py): {expected_epochs}\n"
        f"Completed epochs (from objective*.txt): {diag['epochs_completed']}\n"
        f"FreeFEM return codes:\n{diag['ff_info']}"
        f"A nonzero exit does not by itself invalidate the design: if the "
        f"objective history shows convergence before the failure, and the "
        f"failure is a late numerical artifact (e.g. mesh reversal), the "
        f"design can still be pass.\n"
        f"Imagine how the run proceeded from the completed-epoch count "
        f"and the objective curve below, and decide yourself whether the "
        f"run CRASHED EARLY (a real bug — step too large, wrong labels, "
        f"bad initial phi) or COMPLETED with a usable design (converged "
        f"before the failure).\n"
    )


# ======================================================================
# quality-check block (shared by the pre-fix and post-fix paths)
# ======================================================================


def _quality_check_block(
    agent, problem_name: str, config, results_dir: Path, context_text: str, key: dict,
    vision_images=None, image_section: str = "",
) -> tuple[str, str]:
    """Render evidence and ask the LLM to judge pass vs wrong_result.

    Shared by both call sites (Step 3 MinTh branch and Step 10 after a
    fix) so they cannot diverge again.  *vision_images* may carry the
    snapshots already captured by the Step 1b/2 renders (same results/
    state when nothing ran in between — the pre-fix loop would otherwise
    re-render the composite of an identical state every iteration); when
    None (post-fix step 10, after ``_re_execute`` regenerated results/)
    they are refreshed here.
    """
    minth = _find_minth_vtu(results_dir)
    if minth is not None:
        vtu_summary = _render_minth(minth)
        if vtu_summary:
            context_text += f"\n\n{MINTH_RENDER_SUMMARY_HEADER}\n{vtu_summary}\n"

    diag = _quality_diag(results_dir, key)
    expected = _expected_epochs(key.get("main_py", ""), config)
    context_text += _quality_guidance(config, diag, expected)

    # Deterministic gate before the LLM judge: a crashed run whose
    # objective tail is still strictly descending has not converged —
    # route it to the fixer without an LLM call.
    verdict = _deterministic_convergence_verdict(diag, key, expected)
    if verdict:
        context_text += verdict
        return "wrong_result", context_text

    if vision_images is None:
        # Refresh snapshots AFTER the latest execution — never reuse the
        # composite from an earlier design evolution.
        vision_images, image_section = _vision_context(results_dir)

    # Symmetry (symmetric problems only): the fact comes from the mesh,
    # the design is judged by the vision model from the image above,
    # with the tolerance for FEM-induced raggedness stated in the text.
    sym_text = _symmetry_guidance(key)
    if sym_text:
        context_text += sym_text
        minth_png = _minth_label_png(results_dir)
        if minth_png is not None:
            vision_images = list(vision_images or []) + [str(minth_png)]
            image_section = (image_section or "") + (
                "\n\n=== FINAL DESIGN (image attached) ===\n"
                f"{minth_png.name}: the final solid region (Label = 3 "
                f"cells).  Use it to judge whether the design is symmetric "
                f"about the stated axis, tolerating small "
                f"discretisation-induced asymmetries.\n"
            )

    qc = quality_check(
        agent, problem_name, context_text,
        images=vision_images, image_section=image_section,
    )
    return qc, context_text


# ======================================================================
# parallel probe round — N candidate fixes, each tested with a short run
# ======================================================================

# matplotlib keeps process-global state — the parallel probes serialise
# their design renders through this lock.
_RENDER_LOCK = threading.Lock()


def _rmtree_force(path: Path) -> None:
    """rmtree that also removes read-only directories (copied include/
    dirs keep the read-only bit on Windows)."""

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    if path.exists():
        shutil.rmtree(path, onerror=_onerror)


def _force_iterations(code: str, n: int) -> str:
    """Set iterations=n inside the setup_optimization call of *code*.

    Uses the string-aware call scanner of check_params so a comment
    mention of ``iterations=`` never matches.  When no setup_optimization
    call parses, falls back to the first REAL (non-comment, non-string)
    ``iterations=<digits>`` occurrence.
    """
    from Generation.check_params import _find_setup_optimization, _param_span

    span = _find_setup_optimization(code)
    if span is not None:
        start, end = span
        call_text = code[start:end]
        arg = _param_span(call_text, "iterations")
        if arg is not None:
            new_call = (
                call_text[: arg[0]] + f"iterations={n}" + call_text[arg[1]:]
            )
            return code[:start] + new_call + code[end:]
        from Generation.check_params import _insert_before_closing

        new_call = _insert_before_closing(call_text, f"iterations={n}")
        return code[:start] + new_call + code[end:]
    # Fallback: the first REAL (non-comment, non-string) occurrence —
    # a commented-out example must not swallow the rewrite.
    masked = code_masked_positions(code)
    for m in re.finditer(r"\biterations\s*=\s*\d+", code):
        if m.start() in masked:
            continue
        return code[: m.start()] + f"iterations={n}" + code[m.end():]
    return code


# ======================================================================
# Afac (Lagrange multiplier) calibration — geometric scan + knee pick
#
# The volume weight ``ell`` is a BALANCE parameter of the optimisation,
# not part of the problem statement: the demos carry no value for it and
# the coder only supplies a magnitude estimate.  The reflection loop
# calibrates it once, deterministically: a geometric scan of five
# candidates ell0*10^k (k = -2..2) run as short probes in parallel, the
# solid-fraction curve sampled from each probe's optimal-design mesh, and
# the knee (steepest central slope) chosen as the balance point.
# ======================================================================

# Marker written into reflection.txt when the multiplier has been
# calibrated — the trigger check greps for this string.
_AFAC_CALIBRATED_MARK = "afac calibrated"

# The FreeFEM triangle label that carries the solid region (the label
# used by trunc(Th, region == 3) in the generated optimize script).
_SOLID_MESH_LABEL = 3

_AFAC_VALUE_RE = re.compile(
    r"Afac\s*=\s*Constant\(\s*r?['\"]\\ell['\"]\s*,\s*([-+0-9.eE]+)"
)


def _afac_value(code: str) -> Optional[float]:
    """Extract the numeric value of the Afac constant from main.py."""
    masked = code_masked_positions(code)
    for m in _AFAC_VALUE_RE.finditer(code):
        if m.start() in masked:
            continue
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None


def _afac_replace(code: str, value: float) -> str:
    """Replace the numeric value of the Afac constant (string-masked)."""
    masked = code_masked_positions(code)
    new = f"{value:g}"
    out = []
    pos = 0
    for m in _AFAC_VALUE_RE.finditer(code):
        if m.start() in masked:
            continue
        out.append(code[pos:m.start(1)])
        out.append(new)
        pos = m.end(1)
    out.append(code[pos:])
    return "".join(out)


def _solid_fraction_from_mesh(mesh_path: Path) -> Optional[float]:
    """Solid (label 3) area fraction of a FreeFEM v1 .mesh file.

    The generated optimize script saves the full labelled mesh as
    ``tmp/optimaldesign.mesh`` (best objective) and ``tmp/mesh2D.mesh``
    (every iteration) — the triangle labels distinguish solid (3) from
    weak material.  Vertex numbering in the v1 format is 1-based.
    """
    import numpy as np

    try:
        lines = mesh_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    i = 0

    def _section(name: str) -> Optional[int]:
        nonlocal i
        while i < len(lines):
            if lines[i].strip() == name:
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                try:
                    n = int(lines[i].strip())
                except (ValueError, IndexError):
                    return None
                i += 1
                return n
            i += 1
        return None

    try:
        nv = _section("Vertices")
        if nv is None:
            return None
        verts = np.array(
            [[float(x) for x in lines[i + j].split()[:2]] for j in range(nv)]
        )
        i += nv
        nt = _section("Triangles")
        if nt is None or nt == 0:
            return None
        tris = np.array(
            [[int(x) for x in lines[i + j].split()[:4]] for j in range(nt)]
        )
    except (ValueError, IndexError):
        return None
    if len(tris) == 0:
        return None
    tris[:, :3] -= 1  # FreeFEM v1 vertex numbering is 1-based
    a = verts[tris[:, 1]] - verts[tris[:, 0]]
    b = verts[tris[:, 2]] - verts[tris[:, 0]]
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    total = float(area.sum())
    if total <= 0:
        return None
    return float(area[tris[:, 3] == _SOLID_MESH_LABEL].sum()) / total


def _needs_afac_calibration(results_dir: Path) -> bool:
    """True when the problem statement carries no ell (removed from the
    demos), the problem is NONLINEAR, and the reflection log has no
    calibration record yet.

    The LINEAR benchmark cases keep their fixed Lagrange multipliers —
    only the nonlinear cases calibrate the balance point.
    """
    log = _read_reflection_log(results_dir)
    if _AFAC_CALIBRATED_MARK in log:
        return False
    init = results_dir / "initial.json"
    if not init.exists():
        return False
    try:
        data = json.loads(init.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return False
    constants = (data.get("extracted") or {}).get("constants") or []
    names = {c.get("name") for c in constants if isinstance(c, dict)}
    if "ell" in names or "Afac" in names:
        return False
    # only nonlinear problems are calibrated; a missing flag means
    # linear (the conservative default — never touch the multiplier)
    return bool((data.get("designed") or {}).get("state_is_nonlinear"))


def _afac_calibrate(results_dir: Path, config) -> Optional[float]:
    """Calibrate the Lagrange multiplier Afac and write it back to main.py.

    Geometric scan of five candidates ``ell0 * 10**k`` (k = -2..2), each
    probed with a short run under ``temp/afac_cal/``; the solid fraction
    of every candidate design is read from the FreeFEM output mesh and
    the knee of the resulting curve (steepest central slope in log-ell
    space) becomes the calibrated value.  When the knee lies outside the
    window (all-solid or all-void), the scan extends outward once.

    Returns the chosen value, or None when calibration was skipped.
    """
    from Execution.runner import run_problem

    main_py = results_dir / "main.py"
    try:
        code = main_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        print("    [Afac] main.py unreadable — skip calibration.", flush=True)
        return None
    ell0 = _afac_value(code)
    if ell0 is None:
        print("    [Afac] no Afac constant in main.py — skip calibration.", flush=True)
        return None

    scan_dir = results_dir / "temp" / "afac_cal"
    _rmtree_force(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)

    try:
        mesh_txt = (results_dir / "mesh.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        mesh_txt = ""
    spec_files = [results_dir / n for n in ("initial.json", "mesh.json")]

    probe_epochs = config.REFLECTION_PROBE_MIN_EPOCHS

    def _run_candidate(value: float, tag: str):
        """One short probe at Afac=*value*; returns (solid_fraction, ok).

        ``ok`` is True only when the optimize run exited 0 — a crashed
        run's fraction is a crash-time snapshot that must not steer the
        knee selection.
        """
        cand_dir = scan_dir / tag
        _rmtree_force(cand_dir)
        cand_dir.mkdir(parents=True, exist_ok=True)
        try:
            probe_code = _force_iterations(_afac_replace(code, value), probe_epochs)
            (cand_dir / "main.py").write_text(probe_code, encoding="utf-8")
            (cand_dir / "mesh.txt").write_text(mesh_txt, encoding="utf-8")
            for src in spec_files:
                if src.exists():
                    shutil.copy2(src, cand_dir / src.name)
        except OSError as e:
            print(f"    [Afac] candidate {tag} setup failed: {e}", flush=True)
            return None
        try:
            result = run_problem(cand_dir / "main.py", cand_dir, auto_render=False)
        except Exception as e:
            print(f"    [Afac] candidate {tag} run crashed: {e}", flush=True)
            return None
        ok = False
        for f in result.get("freefem", []):
            if str(f.get("edp_file", "")).startswith("optimize"):
                ok = str(f.get("return_code")) == "0"
                break
        # optimaldesign.mesh (best objective) wins; mesh2D.mesh (last
        # iteration) is the fallback when the short probe never improved.
        for mesh_name in ("optimaldesign.mesh", "mesh2D.mesh"):
            frac = _solid_fraction_from_mesh(cand_dir / "tmp" / mesh_name)
            if frac is not None:
                return frac, ok
        return None

    ks = [-2, -1, 0, 1, 2]
    values: dict = {}  # k -> (solid fraction, ok) — None = failed probe
    workers = max(1, min(len(ks), config.REFLECTION_PARALLEL_MAX))
    for _round in range(3):
        missing = [k for k in ks if k not in values]
        print(
            "    [Afac] scanning "
            + ", ".join(f"{ell0 * 10.0 ** k:g}" for k in missing),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_candidate, ell0 * 10.0 ** k, f"k{k:g}"): k
                for k in missing
            }
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = None
                    print(f"    [Afac] candidate k{k:g} failed: {e}", flush=True)
                values[k] = res
                if res is None:
                    print(f"    [Afac] ell={ell0 * 10.0 ** k:g} -> no design", flush=True)
                else:
                    frac, ok = res
                    print(
                        f"    [Afac] ell={ell0 * 10.0 ** k:g} -> solid fraction "
                        f"{frac:.3f}{'' if ok else ' (CRASHED)'}",
                        flush=True,
                    )
        # knee selection uses SAFE candidates only — a crash-time
        # fraction is not a converged design point
        safe = sorted(k for k in values if values[k] is not None and values[k][1])
        crash = sorted(k for k in values if values[k] is not None and not values[k][1])
        knee, best = None, -1.0
        for j in range(1, len(safe) - 1):
            slope = abs(values[safe[j + 1]][0] - values[safe[j - 1]][0])
            if slope > best:
                best, knee = slope, safe[j]
        lo_s, hi_s = (safe[0], safe[-1]) if safe else (None, None)
        # 1) outward extension when the knee is outside the window
        extend = None
        if len(safe) >= 3:
            if all(values[k][0] > 0.95 for k in safe):
                extend = hi_s + 1  # ell too small — material is free
            elif all(values[k][0] < 0.05 for k in safe):
                extend = lo_s - 1  # ell too large — material unaffordable
            elif knee == lo_s and values[lo_s][0] > 0.9:
                extend = lo_s - 1
            elif knee == hi_s and values[hi_s][0] < 0.1:
                extend = hi_s + 1
        # 2) crash-aware refinement: the knee lies just beyond the
        # largest SAFE candidate when larger ell crashed mid-scan — the
        # true knee is between them; refine log-linearly
        refine = None
        if (
            extend is None
            and safe
            and any(k > hi_s for k in crash)
            and values[hi_s][0] > 0.5
        ):
            refine = (hi_s + 1.0 / 3.0, hi_s + 2.0 / 3.0)
        if extend is not None:
            if _round < 2:
                extra = [extend, extend + (1 if extend > hi_s else -1)]
                ks.extend(sorted(set(k for k in extra if k not in values)))
                print("    [Afac] knee outside window — extending scan.", flush=True)
                continue
        elif refine is not None:
            if _round < 2 and any(r not in values for r in refine):
                ks.extend(sorted(set(refine)))
                print("    [Afac] knee adjacent to crashed candidate — refining scan.", flush=True)
                continue
        break

    safe = sorted(k for k in values if values[k] is not None and values[k][1])
    if not safe:
        print("    [Afac] no safely completed candidate — no calibration.", flush=True)
        return None
    if knee is None or knee not in safe:
        knee = min(safe, key=lambda k: abs(values[k][0] - 0.5))
    chosen = ell0 * 10.0 ** knee
    order = sorted(k for k in values if values[k] is not None)
    scan_summary = ", ".join(
        f"{ell0 * 10.0 ** k:g}:{values[k][0]:.3f}{'' if values[k][1] else '*'}"
        for k in order
    )

    new_code = _afac_replace(code, chosen)
    if new_code != code:
        try:
            main_py.write_text(new_code, encoding="utf-8")
        except OSError as e:
            print(f"    [Afac] main.py write-back failed: {e}", flush=True)
            return None
    else:
        # the estimate already sat at the knee — no rewrite, but the
        # calibration record must still be logged so the trigger check
        # does not fire again on the next round
        print(
            "    [Afac] calibrated value equals the current one — no rewrite.",
            flush=True,
        )

    _append_reflection_log(
        results_dir,
        1,
        "afac_calibrate",
        "-",
        "-",
        f"{_AFAC_CALIBRATED_MARK}: ell {ell0:g} -> {chosen:g} "
        f"(knee of the solid-fraction scan: {scan_summary})",
        f"solid fractions: {scan_summary}",
    )
    print(
        f"    [Afac] calibrated ell: {ell0:g} -> {chosen:g} "
        f"(solid fractions: {scan_summary})",
        flush=True,
    )
    return chosen


def _probe_epochs(idx: int, n_probes: int, config) -> int:
    """Probe budget of candidate *idx* — spread evenly over
    [REFLECTION_PROBE_MIN_EPOCHS, REFLECTION_PROBE_MAX_EPOCHS]."""
    lo = config.REFLECTION_PROBE_MIN_EPOCHS
    hi = config.REFLECTION_PROBE_MAX_EPOCHS
    if n_probes <= 1:
        return lo
    return lo + idx * (hi - lo) // (n_probes - 1)


def _probe_objective_tail(cand_dir: Path, n_tail: int = 5) -> tuple[int, str]:
    """(epoch lines, tail text) of the candidate's objective*.txt."""
    # Most recently written file wins when several objective*.txt exist
    obj_files = sorted(
        cand_dir.glob("objective*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not obj_files:
        return 0, ""
    values = []
    for line in obj_files[0].read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("iteration") or stripped.startswith("#"):
            continue
        parts = stripped.split(",")
        if len(parts) < 2:
            continue
        try:
            float(parts[1].strip())
        except ValueError:
            continue
        values.append(stripped)
    if not values:
        return 0, ""
    return len(values), " | ".join(values[-n_tail:])


def _empty_probe_record(idx: int, error: str) -> dict:
    """A probe record for a worker that never produced one."""
    return {
        "idx": idx,
        "direction": "",
        "rationale": "",
        "main": "",
        "mesh": "",
        "probe_epochs": 0,
        "completed_epochs": 0,
        "final_objective": None,
        "objective_tail": "",
        "py_rc": "?",
        "ff_rc": "?",
        "minth": False,
        "render_summary": "",
        "render_png": None,
        "error": error,
    }


def _run_one_probe(
    idx: int,
    n_probes: int,
    config,
    results_dir: Path,
    temp_dir: Path,
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
    vision_images,
    image_section: str,
) -> dict:
    """Generate ONE candidate fix (one direction) and probe-run it.

    Runs inside a worker thread.  A dedicated agent is built HERE because
    CodeAgent keeps prompt state on the instance and cannot be shared
    across threads.
    """
    from API import (
        AgentRouter,
        DeepSeekCodeAgentV4FlashVision,
        DeepSeekCodeAgentV4Pro,
    )
    from .prompts import FIX_DIRECTIONS

    direction = FIX_DIRECTIONS[idx % len(FIX_DIRECTIONS)]
    probe_epochs = _probe_epochs(idx, n_probes, config)
    record = _empty_probe_record(idx, "")
    record["direction"] = direction
    record["probe_epochs"] = probe_epochs

    cand_dir = temp_dir / f"cand_{idx}"
    _rmtree_force(cand_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)

    try:
        probe_agent = AgentRouter(
            DeepSeekCodeAgentV4Pro(), DeepSeekCodeAgentV4FlashVision()
        )
        fix = fixer_apply(
            probe_agent,
            problem_description=problem_description,
            main_py=main_py,
            mesh_txt=mesh_txt,
            run_result_json=run_result_json,
            objective_data=objective_data,
            convergence_data=convergence_data,
            edp_code=edp_code,
            reflection_history=reflection_history,
            doc_text=doc_text,
            analysis_artifacts=analysis_artifacts,
            coarse=coarse,
            fine=fine,
            action=action,
            outer_iterations=config.MAX_OUTER_ITERATIONS,
            inner_iterations=config.MAX_INNER_ITERATIONS,
            images=vision_images,
            image_section=image_section,
            fix_direction=direction,
        )
    except Exception as e:
        record["error"] = f"fixer call failed: {e}"
        return record

    if fix.get("_error"):
        record["error"] = f"fix parse error: {fix['_error']}"
        return record

    record["rationale"] = fix.get("rationale", "")
    # A fix may legitimately touch ONLY mesh.txt (main absent) — the
    # probe then runs the CURRENT main.py with the candidate mesh.
    record["mesh"] = fix.get("mesh") or ""
    # Keep what the probe ACTUALLY ran (with the probe iteration
    # budget) — the synthesis report and the best-candidate fallback
    # both consume this, and must describe reality.
    record["main"] = _force_iterations(fix.get("main") or main_py, probe_epochs)

    # Prepare the isolated probe directory: candidate code plus the
    # analysis files the runner reads.  (No include/ copy — the runner
    # deletes workdir/include and rebuilds it from the autofreefem
    # package on every run, so a copied tree would be wasted I/O.)
    probe_main = record["main"]
    try:
        (cand_dir / "main.py").write_text(probe_main, encoding="utf-8")
        (cand_dir / "mesh.txt").write_text(record["mesh"] or mesh_txt, encoding="utf-8")
        for name in ("initial.json", "mesh.json"):
            src = results_dir / name
            if src.exists():
                shutil.copy2(src, cand_dir / name)
    except OSError as e:
        record["error"] = f"probe directory setup failed: {e}"
        return record

    # Probe run (short budget — the formal run happens after synthesis)
    from Execution.runner import run_problem

    try:
        result = run_problem(cand_dir / "main.py", cand_dir, auto_render=False)
    except Exception as e:
        record["error"] = f"probe run crashed: {e}"
        return record

    # Evidence collection — guarded as a whole: a filesystem race or a
    # render hiccup must keep the already-gathered fields instead of
    # turning a genuinely completed run into a "worker crashed" record.
    try:
        record["py_rc"] = result.get("autofreefem", {}).get("python", {}).get("return_code", "?")
        ff_rcs = [f.get("return_code", "?") for f in result.get("freefem", [])]
        record["ff_rc"] = ", ".join(str(rc) for rc in ff_rcs) or "-"

        n_epochs, tail = _probe_objective_tail(cand_dir)
        record["completed_epochs"] = n_epochs
        record["objective_tail"] = tail
        if tail:
            try:
                record["final_objective"] = float(tail.split("|")[-1].split(",")[1].strip())
            except (ValueError, IndexError):
                pass

        # Render the probe's final design for the synthesiser.  The
        # lock serialises the matplotlib calls of the parallel probes
        # (pyplot keeps process-global state).
        minth = _find_minth_vtu(cand_dir)
        if minth is not None:
            record["minth"] = True
            with _RENDER_LOCK:
                record["render_summary"] = render_and_summarize(minth)
            png = minth.parent.parent / f"{minth.stem}_cell_Label.png"
            record["render_png"] = str(png) if png.exists() else None
    except Exception as e:
        record["error"] = f"probe evidence collection failed: {e}"

    return record


def _probe_report_text(records: list[dict]) -> str:
    """Assemble the candidate report block for the synthesiser."""
    parts = []
    for r in records:
        # Computed separately — a conditional expression between the
        # adjacent f-string literals would break implicit concatenation
        # (the whole chain parses as ``head if cond else tail``).
        err_line = f"Error (if any): {r['error']}\n" if r.get("error") else ""
        parts.append(
            f"=== Candidate {r['idx']} ===\n"
            f"Direction: {r['direction'] or '(unknown)'}\n"
            f"Fix rationale: {r['rationale'] or '(none)'}\n"
            f"Probe budget: {r['probe_epochs']} epochs\n"
            f"Probe result: Python rc={r['py_rc']}, FreeFEM rc={r['ff_rc']}, "
            f"epochs completed={r['completed_epochs']}/{r['probe_epochs']}, "
            f"final objective={r['final_objective'] if r['final_objective'] is not None else '?'}, "
            f"MinTh.vtu={'yes' if r['minth'] else 'no'}\n"
            f"Objective tail: {r['objective_tail'] or '(none)'}\n"
            f"Render summary: {r['render_summary'] or '(none)'}\n"
            f"{err_line}"
            f"Candidate main.py:\n```python\n{r['main'] or '(none)'}\n```\n"
            f"Candidate mesh.txt:\n```\n{r['mesh'] or '(no change)'}\n```\n\n"
        )
    return "\n".join(parts)


def _best_probe_record(records: list[dict]) -> Optional[dict]:
    """Deterministic fallback choice when the synthesiser fails: the
    best-evidenced candidate (usable code, MinTh + clean FreeFEM first,
    then lowest final objective, then most completed epochs)."""
    def key(r):
        obj = r.get("final_objective") if r.get("final_objective") is not None else 1e300
        if not r.get("minth"):
            return (2, obj, -r.get("completed_epochs", 0))
        # all FreeFEM runs clean (a two-run "0, 0" counts as clean, a
        # single-run "0" too, "0, 3" does not)
        ffs = [str(rc).strip() for rc in str(r.get("ff_rc", "")).split(",")]
        rc_ok = 0 if ffs and all(rc == "0" for rc in ffs) else 1
        return (rc_ok, obj, -r.get("completed_epochs", 0))

    usable = [r for r in records if not r.get("error") and r.get("main")]
    if not usable:
        return None
    return min(usable, key=key)


def _synthesize_next_step(
    agent,
    config,
    temp_dir: Path,
    problem_description: str,
    main_py: str,
    mesh_txt: str,
    reflection_history: str,
    doc_text: str,
    analysis_artifacts: str,
    coarse: str,
    fine: str,
    action: str,
    records: list[dict],
    iteration: int,
) -> dict:
    """Synthesise the formal next-step code from the probe records.

    One LLM call over all candidates: their directions, code, probe
    evidence (return codes, epochs, objective, design renders).  On LLM
    parse failure the best-evidenced candidate is used verbatim — the
    round never dies on a malformed synthesis reply.
    """
    from .prompts import FIX_SYNTHESIS_SYSTEM, FIX_SYNTHESIS_USER

    report = _probe_report_text(records)
    # Keep a human-readable copy of the probe round under temp/
    try:
        (temp_dir / f"probe_report_iter{iteration}.txt").write_text(
            report, encoding="utf-8"
        )
    except OSError:
        pass

    # Pair each image with its OWN candidate — a zip over the filtered
    # list would misalign names when some candidates rendered no PNG.
    images = []
    lines = []
    for r in records:
        if r.get("render_png"):
            images.append(r["render_png"])
            lines.append(
                f"{Path(r['render_png']).name} — Candidate {r['idx']} "
                f"final design (Label=3 cells)"
            )
    image_section = ""
    if images:
        image_section = (
            "\n\n=== PROBE DESIGN IMAGES (attached, in candidate order) ===\n"
            + "\n".join(lines)
            + "\n"
        )

    agent.set_prompt_list(
        [
            {"role": "system", "content": FIX_SYNTHESIS_SYSTEM},
            {"role": "user", "content": FIX_SYNTHESIS_USER},
        ]
    )
    # A transport failure (network, rate limit, API outage) must take
    # the same fallback path as a malformed reply — the round never
    # dies because the synthesiser could not be reached.
    try:
        raw = agent.invoke_multimodal(
            {
                "n_candidates": len(records),
                "probe_min": config.REFLECTION_PROBE_MIN_EPOCHS,
                "probe_max": config.REFLECTION_PROBE_MAX_EPOCHS,
                "formal_iterations": _expected_epochs(main_py, config),
                "problem_description": problem_description,
                "main_py": main_py,
                "mesh_txt": mesh_txt,
                "coarse": coarse,
                "fine": fine,
                "action": action,
                "candidates_report": report,
                "reflection_history": reflection_history or "(first attempt)",
                "doc_text": doc_text,
                "analysis_artifacts": analysis_artifacts or "(not available)",
                "image_section": image_section,
            },
            images=images,
        )
    except Exception as e:
        best = _best_probe_record(records)
        if best is not None:
            return {
                "main": best["main"],
                "mesh": best["mesh"],
                "rationale": (
                    f"synthesis LLM call failed ({e}) — falling back to "
                    f"best probe candidate {best['idx']}: {best['rationale']}"
                ),
            }
        return {
            "main": "",
            "mesh": "",
            "rationale": "",
            "_error": f"synthesis LLM call failed: {e}",
        }

    from .fixer import _parse_fix_response

    parsed = _parse_fix_response(raw)
    if parsed.get("_error"):
        best = _best_probe_record(records)
        if best is not None:
            parsed = {
                "main": best["main"],
                "mesh": best["mesh"],
                "rationale": (
                    f"synthesis parse failed ({parsed['_error']}) — "
                    f"falling back to best probe candidate {best['idx']}: "
                    f"{best['rationale']}"
                ),
            }
    return parsed


# ======================================================================
# main orchestrator
# ======================================================================


def run_reflection(agent, info, config) -> str:
    """
    Run the full reflection loop for a single problem.

    Each round: detect → classify → decide → PARALLEL PROBE (up to
    REFLECTION_PARALLEL_MAX candidate fixes, each tested with a short
    run of REFLECTION_PROBE_MIN_EPOCHS..REFLECTION_PROBE_MAX_EPOCHS
    epochs under results/<name>/temp/) → SYNTHESISE one next-step code
    from the probe evidence → formal re-execute → quality check.

    Args:
        agent: CodeAgent instance for LLM calls.
        info: Info object with problem_name, results_dir, and reflection_*
              fields.
        config: Module with reflection constants (MAX_REFLECTION_TIMES,
                REFLECTION_PARALLEL_MAX, REFLECTION_PROBE_*_EPOCHS,
                REFLECTION_STALE_THRESHOLD, etc.).

    Returns:
        Reflection status string: "pass", "give_up_max",
        or "give_up" (no files found).

    The temp/ directory lives for the WHOLE reflection process and is
    removed when the process exits (pass / give_up / give_up_max) —
    never mid-process.
    """
    results_dir = info.results_dir
    problem_name = info.problem_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parallel-probe scratch space: fresh on each reflection entry,
    # persistent across rounds, removed on exit (finally block).
    temp_dir = results_dir / "temp"
    _rmtree_force(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize reflection log.  The in-memory history is reset with it:
    # the fixer reads the LOG and the escalation decides on the in-memory
    # records, so a re-entered loop (same Info object) must not keep stale
    # in-memory records that no longer exist in the log.
    _init_reflection_log(results_dir, problem_name)
    info.reflection_history = []

    # Load AutoFreeFEM documentation once — anchored on this file, not
    # the CWD (running the pipeline from elsewhere must not silently
    # empty the documentation context).
    doc_path = (
        Path(__file__).resolve().parent.parent
        / "database" / "freefem_functions" / "list_of_classes.txt"
    )
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""

    try:
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

            # --- Step 1a: Afac calibration (first round, only when the
            # problem statement carries no Lagrange multiplier) ---
            if iteration == 0 and _needs_afac_calibration(results_dir):
                chosen = _afac_calibrate(results_dir, config)
                if chosen is not None:
                    # When the calibration actually rewrote main.py, drop
                    # run_result.json so Step 2 re-executes with the
                    # calibrated multiplier; an unchanged value means the
                    # existing run is already valid.
                    try:
                        current = _afac_value(
                            (results_dir / "main.py").read_text(
                                encoding="utf-8", errors="replace"
                            )
                        )
                    except OSError:
                        current = None
                    if current is None or abs(current - chosen) > abs(chosen) * 1e-9:
                        try:
                            (results_dir / "run_result.json").unlink()
                        except OSError:
                            pass
                        files = collect_files(results_dir)
                        context_text = _assemble_context_text(files)
                        key = _extract_key_files(
                            files, evidence_chars=config.REFLECTION_EVIDENCE_CHARS
                        )
                        vision_images, image_section = _vision_context(results_dir)

            # --- Step 1b: iteration-label composite (multimodal evidence) ---
            # Every result*Th<N>.vtu snapshot's Label field is merged, in
            # iteration order, into ONE image the model can see.  The user
            # prompt template carries an {image_section} placeholder; the
            # image itself travels through the multimodal message channel.
            vision_images, image_section = _vision_context(results_dir)

            # --- Step 2: Check if we need to (re-)execute ---
            run_result_json = key["run_result_json"]
            if run_result_json == "{}":
                _re_execute(results_dir)
                files = collect_files(results_dir)
                context_text = _assemble_context_text(files)
                key = _extract_key_files(files, evidence_chars=config.REFLECTION_EVIDENCE_CHARS)
                run_result_json = key["run_result_json"]
                # fresh snapshots after re-execution — refresh the composite
                vision_images, image_section = _vision_context(results_dir)

            # --- Step 3: Coarse classification ---
            # MinTh.vtu existence is the only hard rule; the objective
            # history never short-circuits the classification.
            minth = _find_minth_vtu(results_dir)

            if minth is not None:
                # MinTh.vtu exists — the LLM judges EVERY case (success vs
                # premature stop) from the objective history, the design
                # snapshots, and the return codes.  No epoch-count threshold.
                qc, context_text = _quality_check_block(
                    agent, problem_name, config, results_dir, context_text, key,
                    vision_images=vision_images, image_section=image_section,
                )
                if qc == "pass":
                    print("    [Reflect] PASS — MinTh.vtu exists, result valid.", flush=True)
                    # A clean pass is still a reflection iteration — log it so
                    # reflection.txt always carries a record of the verdict.
                    outcome = _format_outcome(_load_run_result(results_dir), minth_exists=True)
                    _append_reflection_log(
                        results_dir,
                        iteration + 1,
                        "pass",
                        "-",
                        "none",
                        "Quality check: MinTh.vtu exists, design valid — no fix needed.",
                        outcome,
                    )
                    _append_final_status(results_dir, "pass", info.reflection_attempts)
                    info.reflection_status = "pass"
                    return "pass"
                # Quality check failed → treat as wrong_result
                coarse = "wrong_result"
                print(f"    [Reflect] Quality check: {qc} — treating as wrong_result.", flush=True)
            else:
                # No MinTh.vtu — LLM classifies
                coarse = coarse_classify(
                    agent, problem_name, context_text,
                    images=vision_images, image_section=image_section,
                )
                print(f"    [Reflect] Coarse: {coarse}", flush=True)

                if coarse == "pass":
                    print("    [Reflect] PASS.", flush=True)
                    outcome = _format_outcome(_load_run_result(results_dir), minth_exists=False)
                    _append_reflection_log(
                        results_dir,
                        iteration + 1,
                        "pass",
                        "-",
                        "none",
                        "Coarse classification: result valid — no fix needed.",
                        outcome,
                    )
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
                    images=vision_images,
                    image_section=image_section,
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
            )
            print(f"    [Reflect] Action: {action}", flush=True)

            # --- Step 6: Parallel probe round ---
            # Up to REFLECTION_PARALLEL_MAX candidate fixes, each exploring
            # ONE direction, each tested with a short run under temp/.
            n_probes = max(1, int(config.REFLECTION_PARALLEL_MAX))
            print(
                f"    [Reflect] probing {n_probes} candidate fixes in parallel "
                f"({config.REFLECTION_PROBE_MIN_EPOCHS}-"
                f"{config.REFLECTION_PROBE_MAX_EPOCHS} epochs each) …",
                flush=True,
            )
            records = [None] * n_probes
            with ThreadPoolExecutor(max_workers=n_probes) as pool:
                futures = {}
                for i in range(n_probes):
                    fut = pool.submit(
                        _run_one_probe,
                        i,
                        n_probes,
                        config,
                        results_dir,
                        temp_dir,
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
                        vision_images=vision_images,
                        image_section=image_section,
                    )
                    futures[fut] = i
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # noqa: BLE001 — worker isolation
                        r = _empty_probe_record(i, f"probe worker crashed: {e}")
                    records[i] = r
                    if r.get("error"):
                        print(f"    [Probe {i}] {r['error']}", flush=True)
                    else:
                        print(
                            f"    [Probe {i}] py={r['py_rc']}, ff={r['ff_rc']}, "
                            f"epochs={r['completed_epochs']}/{r['probe_epochs']}, "
                            f"obj={r['final_objective']}, "
                            f"MinTh={'yes' if r['minth'] else 'no'}",
                            flush=True,
                        )

            # --- Step 7: Synthesise the next-step code ---
            print("    [Reflect] synthesising next-step code …", flush=True)
            synth = _synthesize_next_step(
                agent,
                config,
                temp_dir,
                problem_description=key["problem_description"],
                main_py=key["main_py"],
                mesh_txt=key["mesh_txt"],
                reflection_history=reflection_history,
                doc_text=doc_text,
                analysis_artifacts=analysis_artifacts,
                coarse=coarse,
                fine=fine,
                action=action,
                records=[r for r in records if r is not None],
                iteration=iteration + 1,
            )

            if synth.get("_error"):
                # No usable candidate either — nothing to write or run.
                print(f"    [Reflect] Synthesis failed, no fallback: {synth['_error']}", flush=True)
                _append_reflection_log(
                    results_dir,
                    iteration + 1,
                    coarse,
                    fine,
                    action,
                    "(synthesis failed)",
                    f"synthesis failed: {synth['_error']}",
                )
                info.reflection_history.append(
                    {
                        "coarse": coarse,
                        "fine": fine,
                        "action": action,
                        "fix_rationale": "(synthesis failed)",
                        "outcome": f"synthesis failed: {synth['_error']}",
                    }
                )
                continue

            rationale = synth.get("rationale", "")
            new_main = synth.get("main", "")
            new_mesh = synth.get("mesh", "")

            # --- Step 8: Apply the synthesised code + formal re-execute ---
            # The probe budgets were probe-only: the formal run always
            # uses the problem's own iteration count.
            formal_epochs = _expected_epochs(key["main_py"], config)
            if new_main:
                out_path = results_dir / "main.py"
                out_path.write_text(
                    _force_iterations(new_main, formal_epochs), encoding="utf-8"
                )
            if new_mesh:
                out_path = results_dir / "mesh.txt"
                out_path.write_text(new_mesh, encoding="utf-8")

            result = _re_execute(results_dir)

            # --- Step 9: Evaluate outcome ---
            minth_after = _find_minth_vtu(results_dir)
            outcome = _format_outcome(result, minth_after is not None)

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

            # Post-fix quality check — only when the re-execution actually ran.
            # `result` is {} when main.py was missing, in which case nothing
            # ran and results/results/ was not wiped: a MinTh.vtu from an
            # EARLIER run may still sit there, and judging it would accept a
            # stale design as the outcome of the fix.
            if minth_after is not None and result:
                files = collect_files(results_dir)
                context_text = _assemble_context_text(files)
                key_after = _extract_key_files(files, evidence_chars=config.REFLECTION_EVIDENCE_CHARS)

                # The refreshed context is not consumed here — the next loop
                # iteration re-collects the files and rebuilds it from scratch.
                qc, _ = _quality_check_block(
                    agent, problem_name, config, results_dir, context_text, key_after
                )
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
    finally:
        # temp/ is removed only when the process ends — pass, give_up or
        # give_up_max — never between rounds.
        _rmtree_force(temp_dir)
        # The iteration-snapshot composite is vision evidence for the
        # reflection loop only — it must not linger in the results folder
        # after the process ends.
        try:
            (results_dir / "iteration_labels_composite.png").unlink(missing_ok=True)
        except OSError:
            pass
