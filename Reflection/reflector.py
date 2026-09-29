"""
Top-level orchestrator for the Reflection module (Stage 7).

Responsibilities:
1. Collect all files under results/<name>/ using a blacklist strategy
   and assemble them into a structured context (问题 / py代码 /
   freefem代码 / 结果).
2. Orchestrate the detect → classify → fix → re-execute loop over the
   five fix types (syntax_fix / formula_fix / design_fix / fe_param_fix
   / free_fix), all sharing ``MAX_REFLECTION_TIMES`` rounds.  The
   repair MODEL follows the category: syntax_error rounds repair with
   the TEXT model, fe_parameter_error with the VISION model, and the
   formula / design / other paths use VISION when design images exist
   and TEXT otherwise.  formula_fix / design_fix / fe_param_fix test
   up to ``MAX_FIX_CANDIDATES``
   candidate fixes IN PARALLEL under ``results/<name>/temp/``, each with
   a short probe run (``REFLECTION_PROBE_EPOCHS`` epochs); an LLM
   synthesiser then merges the probe evidence (objective + design
   images) into ONE next-step code for the formal run.
3. Maintain IN-MEMORY repair records (round 0 = the initial execution's
   console evidence; each round records its target and the added /
   deleted / modified parts) and write reflection.txt (repair history).
4. temp/ survives the process (a record of the attempts); each attempt
   only loses its results/, tex/, tmp/ subdirectories.
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

from history import record as record_history, set_label
from Initialization.phi_rules import detect_symmetry, mesh_dims
from utils import code_masked_positions

from .classifier import (
    MINTH_RENDER_SUMMARY_HEADER,
    available_fix_types,
    available_types_text,
    coarse_classify,
    fix_type_for,
    quality_check,
)
from .fixer import (
    build_design_permission,
    build_fe_param_permission,
    design_enforce,
    design_fix,
    fe_param_enforce,
    fe_param_fix,
    formula_fix,
    free_fix,
    permission_text,
)
from .renderer import (
    render_and_summarize,
    render_snapshot_label_png,
)

# ======================================================================
# file collection — blacklist strategy
# ======================================================================

# File patterns to EXCLUDE (binary, intermediate, or too numerous).
# Per the collection policy: .pdf, results/, tex/, tmp/, *.gmsh, *.mesh,
# history.txt — plus the binary/own-scratch leftovers.
_BLACKLIST_SUFFIXES = {
    ".pdf", ".png", ".sol", ".mmg2d", ".mshdist", ".mesh", ".gmsh",
}

# Pipeline operation log — for HUMAN debugging, not LLM evidence.  It
# holds the pipeline's own LLM inputs/outputs and would bloat (and
# bias) the reflection context if collected.  The reflection log is
# excluded for the same reason: the repair history reaches the fixer
# through the dedicated in-memory repair records.
_BLACKLIST_NAMES = {"history.txt", "reflection.txt"}

# tmp/ (execution scratch), temp/ (parallel probe candidates — the
# attempts are a record, never reflection evidence), tex/ (duplicate
# copies of the problem .tex files — the root-level ones are collected)
# and results/ (iteration snapshots; MinTh.vtu existence is checked via
# _find_minth_vtu — hard rule, its content goes ONLY to the renderer).
_BLACKLIST_DIRS = {"tmp", "temp", "tex", "results"}


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

    # Exclude the blacklisted directories entirely (case-insensitive — a
    # hand-made TEMP directory must not leak into the context either).
    # The include/ subtree (generated .idp/.edp includes) stays
    # collected.
    parts = Path(rel_path).parts
    for part in parts:
        if part.lower() in _BLACKLIST_DIRS:
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
# context assembly — structured sections (问题 / py代码 / freefem代码 / 结果)
# ======================================================================

# Stable section headers (tests and prompts reference these strings).
SECTION_PROBLEM = "== 问题 (Problem) =="
SECTION_PY_CODE = "== py代码 (Python code) =="
SECTION_FF_CODE = "== freefem代码 (FreeFEM code) =="
SECTION_RESULTS = "== 结果 (Results) =="


def _assemble_context_text(files: dict[str, str]) -> str:
    """
    Build the structured context block from collected files.

    Files are routed into four sections:
      * 问题 — the problem description (.tex) and the spec artifacts
        (initial.json, mesh.json, examples.txt);
      * py代码 — the generated Python code (main.py);
      * freefem代码 — the generated FreeFEM code (.edp + include/*.idp);
      * 结果 — everything else (run_result.json, objective/convergence
        logs, other text outputs).
    NO truncation anywhere — the FreeFEM console was already truncated
    at the source (Execution/runner.py).
    """
    sections = {
        SECTION_PROBLEM: [],
        SECTION_PY_CODE: [],
        SECTION_FF_CODE: [],
        SECTION_RESULTS: [],
    }

    def _add(section: str, rel: str, content: str) -> None:
        sections[section].append(f"--- FILE: {rel} ---\n{content}\n")

    for rel in sorted(files):
        content = files[rel]
        filename = Path(rel).name
        if filename.endswith(".tex") or filename in (
            "initial.json", "mesh.json", "examples.txt",
        ):
            _add(SECTION_PROBLEM, rel, content)
        elif filename.endswith(".py"):
            _add(SECTION_PY_CODE, rel, content)
        elif filename.endswith(".edp") or filename.endswith(".idp"):
            _add(SECTION_FF_CODE, rel, content)
        else:
            _add(SECTION_RESULTS, rel, content)

    parts = []
    for header in (SECTION_PROBLEM, SECTION_PY_CODE, SECTION_FF_CODE, SECTION_RESULTS):
        if not sections[header]:
            continue
        parts.append(header)
        parts.extend(sections[header])
    return "\n".join(parts)


def _extract_key_files(files: dict[str, str]) -> dict[str, str]:
    """Extract individual key files from the collected files dict.

    NO truncation anywhere — the FreeFEM console was already truncated
    at the source (Execution/runner.py).
    """
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
            result["objective_data"] += f"=== {rel} ===\n{content}\n\n"
        elif "convergence" in path_lower and filename.endswith(".txt"):
            result["convergence_data"] += f"=== {rel} ===\n{content}\n\n"
        elif filename.endswith(".edp"):
            result["edp_code"] += f"=== {rel} ===\n{content}\n\n"
        elif filename.endswith(".tex"):
            result["problem_description"] += f"=== {rel} ===\n{content}\n\n"
        elif filename.endswith(".txt") and "mesh" not in path_lower:
            # The repair history (reflection.txt) is NOT part of the
            # original problem description — it reaches the fixer through
            # the dedicated in-memory repair records instead.
            if filename == "reflection.txt":
                continue
            result["problem_description"] += f"=== {rel} ===\n{content}\n\n"

    # Also include idp files under include/
    for rel, content in files.items():
        if rel.endswith(".idp"):
            result["edp_code"] += f"=== {rel} ===\n{content}\n\n"

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
    ff_records = result.get("freefem", [])
    ff_rcs_raw = [f.get("return_code", "?") for f in ff_records]
    # A console_err_code on an rc=0 record is the "err code appeared
    # after loop completion" case — surface it so the LLM can weigh it.
    ff_rcs = [
        f"{f.get('return_code', '?')}"
        + (f" (err {f['console_err_code']})" if f.get("console_err_code") is not None else "")
        for f in ff_records
    ]
    if minth_exists:
        return f"MinTh.vtu generated [ok] — Python rc={py_rc}, {len(edp_files)} .edp, FreeFEM rcs={ff_rcs}"
    if edp_files and all(rc == "0" for rc in ff_rcs_raw):
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


def _vision_context(results_dir: Path, config) -> tuple[list[str], str]:
    """Render the design snapshots for the multimodal loop.

    The Label field of every iteration snapshot (result*Th<N>.vtu) is
    rendered into ONE PNG PER SNAPSHOT — the images are passed to the
    model individually, in iteration order.  At most
    ``configs.MAX_SNAPSHOT_FRAMES`` images: beyond that count the cap
    is INTERPOLATED over the WHOLE run (the model sees the entire
    evolution, not just the tail).  Returns ``(images, image_section)``
    — images is the ordered list of PNG paths (empty when no snapshots
    exist), image_section the text block naming them.
    """
    max_frames = getattr(config, "MAX_SNAPSHOT_FRAMES", 50)
    snaps = _iteration_snapshots(results_dir)
    if not snaps:
        return [], ""
    if len(snaps) > max_frames:
        # uniform interpolation over the whole run (no numpy — pure
        # integer rounding, de-duplicated and sorted)
        idxs = sorted({
            round(i * (len(snaps) - 1) / (max_frames - 1))
            for i in range(max_frames)
        })
        snaps = [snaps[i] for i in idxs]
    images = []
    lines = []
    for p in snaps:
        m = re.search(r"Th(\d+)\.vtu$", p.name)
        idx = int(m.group(1)) if m else 0
        out = results_dir / f"snapshot_Th{idx}_label.png"
        try:
            render_snapshot_label_png(p, out)
        except Exception as e:
            lines.append(f"Th{idx}: render failed ({e})")
            continue
        images.append(str(out))
        lines.append(f"{out.name} — iteration {idx} design (Label=3 cells)")
    if not images:
        return [], ""
    image_section = (
        "\n\n=== ITERATION SNAPSHOTS (images attached, in iteration order) ===\n"
        + "\n".join(lines)
        + "\nYou CAN see the attached design snapshots — use the design "
          "evolution (solid/void distribution per epoch) to judge the "
          "failure and decide the fix.\n"
    )
    return images, image_section


# ======================================================================
# re-execute
# ======================================================================


def _re_execute(results_dir: Path, history_label: str = "") -> dict:
    """
    Run main.py through AutoFreeFEM + FreeFEM.

    ``run_problem`` persists ``run_result.json`` internally — this
    function does NOT write it a second time.  The execution is recorded
    in FULL (input code + complete result) into history.txt.
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
    history_log = results_dir / "history.txt"
    result = run_problem(
        main_py,
        results_dir,
        history_log=history_log if history_log.exists() else None,
        history_label=history_label or "Reflection/_re_execute",
    )
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
#
# FACTS only — the pass vs wrong_result judgment and the error-category
# judgment are always the LLM's (no deterministic gates, no fixed
# symptom-to-category wiring; the measured numbers below are evidence).
# ======================================================================


# The weak-material contrast — the 3rd argument of CharacteristicFunction
# (1 = solid, '1/100' = void).  Used by the degeneracy readout below.
_CHI_RE = re.compile(
    r"CharacteristicFunction\(\s*(?:r?['\"][^'\"]*['\"]|\w+)\s*,\s*[^,]+,\s*"
    r"r?['\"]([^'\"]+)['\"]"
)


def _chi_value(code: str) -> Optional[str]:
    """Weak-material value of the CONTRAST CharacteristicFunction.

    Returns the raw string (e.g. '1/100'), or None when no contrast call
    is found.  Comment/string mentions are skipped via the code mask,
    like every other code scan of the pipeline.
    """
    masked = code_masked_positions(code)
    for m in _CHI_RE.finditer(code):
        if m.start() in masked:
            continue
        return m.group(1)
    return None


def _iteration_snapshots(results_dir: Path) -> list[Path]:
    """Numeric ``result*Th<N>.vtu`` snapshots, in iteration order.

    MinTh.vtu is excluded by the numeric-suffix requirement — it is the
    best-objective design, not a step of the evolution.
    """
    snaps = []
    for p in (results_dir / "results").glob("result*Th*.vtu"):
        m = re.search(r"Th(\d+)\.vtu$", p.name)
        if m:
            snaps.append((int(m.group(1)), p))
    snaps.sort(key=lambda t: t[0])
    return [p for _, p in snaps]


def _solid_area_from_vtu(path: Path) -> Optional[float]:
    """Total triangle area of a design snapshot (m^2), or None.

    The iteration meshes are the SOLID region only (the design mesh is
    truncated before every save), so this area IS the material area the
    volume term ``ell*V`` of the objective charges for.  meshio/numpy
    are imported lazily — the diagnostic must never make the reflection
    loop unusable in their absence.
    """
    try:
        import meshio
        import numpy as np
    except Exception:
        return None
    try:
        mesh = meshio.read(str(path))
    except Exception:
        return None
    tris = [cb.data for cb in mesh.cells if cb.type == "triangle"]
    if not tris:
        return None
    tri = np.vstack(tris)
    p = mesh.points
    area = 0.5 * np.abs(
        (p[tri[:, 1], 0] - p[tri[:, 0], 0]) * (p[tri[:, 2], 1] - p[tri[:, 0], 1])
        - (p[tri[:, 2], 0] - p[tri[:, 0], 0]) * (p[tri[:, 1], 1] - p[tri[:, 0], 1])
    )
    return float(area.sum())


def _quality_diag(results_dir: Path, key: dict) -> dict:
    """Collect the diagnostics that guide the soft quality check.

    *diagnostics* contains ``ff_info`` (FreeFEM return code of the
    optimize run — verify runs are not executed) and the
    design-degeneracy signals: the last objective value, the balance
    parameters ``ell`` / ``chi`` read from main.py, and the material
    area of the first and last iteration snapshots.  Epoch counts need
    no dedicated signal — the raw objective data (already in the
    context) carries them as ordinary evidence.  The FreeFEM
    codes are parsed from the FULL run_result.json on disk — the
    LLM-bound copy in *key* keeps only the console tail and would not
    parse.  Pure function — does NOT print or write the reflection log.
    """
    diag = {
        "ff_info": "",
        "objective_last": None,
        "ell": None,
        "chi": None,
        "solid_first": None,
        "solid_last": None,
        "domain_area": None,
        "snapshots": 0,
    }

    obj_values = []
    obj_text = key.get("objective_data", "")
    if obj_text:
        for line in obj_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("iteration"):
                continue
            parts = stripped.split(",")
            if len(parts) < 2:
                continue
            try:
                int(parts[0].strip())
                obj_values.append(float(parts[1].strip()))
            except ValueError:
                continue
    if obj_values:
        diag["objective_last"] = obj_values[-1]

    # ── material balance of the design (degeneracy signals) ──────────
    main_py = key.get("main_py", "") or ""
    diag["ell"] = _afac_value(main_py)
    diag["chi"] = _chi_value(main_py)
    snaps = _iteration_snapshots(results_dir)
    diag["snapshots"] = len(snaps)
    if snaps:
        diag["solid_first"] = _solid_area_from_vtu(snaps[0])
        diag["solid_last"] = _solid_area_from_vtu(snaps[-1])
    try:
        from Initialization.phi_rules import mesh_dims

        L, H = mesh_dims(key.get("mesh_txt", "") or "")
        if L and H:
            diag["domain_area"] = float(L) * float(H)
    except Exception:
        pass

    rr_data = None
    try:
        rr_data = (results_dir / "run_result.json").read_text(encoding="utf-8", errors="replace")
    except OSError:
        rr_data = key.get("run_result_json")
    if rr_data and rr_data != "{}":
        try:
            rr = json.loads(rr_data)
            for ff in rr.get("freefem", []):
                note = (
                    f" (console_err_code={ff['console_err_code']})"
                    if ff.get("console_err_code") is not None
                    else ""
                )
                diag["ff_info"] += (
                    f"FreeFEM {ff.get('edp_file', '?')}: "
                    f"exit={ff.get('return_code', '?')}{note}\n"
                )
        except Exception:
            pass
    return diag


def _expected_epochs(main_py: str, config) -> int:
    """Iterations requested by the problem's own ``setup_optimization``.

    The FORMAL run budget (Step 6 forcing and the synthesis prompt's
    formal_iterations) must reflect the problem's own ``iterations=`` —
    a benchmark may legitimately ask for fewer epochs than the global
    default ``MAX_OUTER_ITERATIONS``.  Local import mirrors
    ``_re_execute``.
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


def _quality_guidance(diag: dict) -> str:
    """Build the guidance text for the LLM quality check.

    Gives the model the FACTS (the FreeFEM exit codes and the design
    diagnostics) only — judging success vs premature stop is left
    entirely to the LLM, from the objective curve, the rendered design
    snapshots, and the return codes.  Epoch counts are just ordinary
    evidence, like any other number.
    """
    return (
        f"\n=== Quality Check Guidance ===\n"
        f"FreeFEM return codes:\n{diag['ff_info']}"
        f"A nonzero exit does not by itself invalidate the design: if the "
        f"objective history shows convergence before the failure, and the "
        f"failure is a late numerical artifact (e.g. mesh reversal), the "
        f"design can still be pass.\n"
        f"Judge the run from the objective curve below and the design "
        f"snapshots — whether it CRASHED (a real bug — step too large, "
        f"wrong labels, bad initial phi) or finished with a usable design "
        f"(converged before the failure).\n"
    ) + _degeneracy_text(diag)


def _degeneracy_text(diag: dict) -> str:
    """Material-balance readout of the finished design (deterministic).

    Reports the measured numbers — ``ell``, ``chi``, the material area of
    the first and last iteration snapshot, and the split of the objective
    into its volume term ``ell*V`` and the implied compliance C — with
    the two degenerate outcomes spelled out.  A design that ends with
    almost no material (the objective dominated by ``ell*V``) or that
    stays a featureless block has not solved the structural trade-off
    the problem poses; the volume weight and/or the void contrast are
    inconsistent with the physics, and such a design must not be
    reported as a valid optimum.  An ENERGY/BALANCE statement, not a
    shape classifier: the actual design is judged from the images.
    """
    ell = diag.get("ell")
    chi = diag.get("chi")
    a0 = diag.get("solid_first")
    a1 = diag.get("solid_last")
    dom = diag.get("domain_area")
    obj = diag.get("objective_last")
    if ell is None and a1 is None:
        return ""

    lines = ["\n=== Design diagnostics (deterministic, from the artefacts) ==="]
    if ell is not None:
        lines.append(f"Volume weight ell (Afac in main.py): {ell:g}")
    if chi is not None:
        lines.append(f"Void contrast chi (CharacteristicFunction weak value): {chi}")
    if diag.get("snapshots"):
        lines.append(f"Iteration snapshots found: {diag['snapshots']}")
    if a0 is not None and a1 is not None:
        frac = f" ({100.0 * a1 / dom:.1f}% of the domain area {dom:g})" if dom else ""
        lines.append(
            f"Material (solid) area: first snapshot {a0:.4g} -> last {a1:.4g}{frac}"
        )
    if obj is not None and ell is not None and a1 is not None:
        vol_term = ell * a1
        lines.append(
            f"Objective (last) {obj:.6g} ~ volume term ell*V = {vol_term:.6g}"
            f" + implied compliance C = {obj - vol_term:.6g}"
        )
    lines.append(
        "Read this as: the material area V is what the volume term pays for, "
        "C is the stiffness it buys.  SOFT thresholds — guides, NEVER hard "
        "rules: below ~20% of the domain RAISES the COLLAPSE suspicion, "
        "above ~80% RAISES the BLOCK suspicion.  The numbers alone do NOT "
        "decide — a legitimately sparse optimum (a thin conducting path or "
        "a slender arch) can sit well below 20%.  Cross-check them against "
        "the design images, for these TWO DEGENERATE OUTCOMES:\n"
        "  * COLLAPSE — V has fallen low (typically below 20% of the "
        "domain) AND the images show no load-carrying member: only "
        "hairlines, beads or patches, with no connected path from the "
        "source/load to a support/sink.  The balance sits far above what "
        "the physics supports (the void alone carries the load), so the "
        "optimiser removed the design instead of shaping it.\n"
        "  * BLOCK — V stays high (typically above 80% of the domain) and "
        "the snapshots show a featureless block that barely moved from the "
        "seed: ell sits far below the balance, material is effectively "
        "free and the optimiser keeps it.\n"
        "Neither solved the structural trade-off — classify wrong_result.  "
        "A thin but CONNECTED member structure is NOT degenerate: judge "
        "conduction/load-path from the images."
    )
    return "\n".join(lines) + "\n"


# ======================================================================
# quality-check block (shared by the pre-fix and post-fix paths)
# ======================================================================


def _quality_check_block(
    agent, problem_name: str, config, results_dir: Path, context_text: str, key: dict,
    vision_images=None, image_section: str = "",
) -> tuple[str, str]:
    """Render evidence and ask the LLM to judge pass vs wrong_result.

    Two call sites: Step 3 (pre-fix, MinTh exists — the verdict is
    EVIDENCE ONLY, the coarse classification that follows is the sole
    authority for pass) and Step 8 (post-fix — a pass verdict ends the
    reflection directly, no extra round).  The function enriches
    *context_text* with the MinTh render summary and the quality
    guidance (facts) in both cases.  When *vision_images* is None
    (Step 8, after ``_re_execute`` regenerated results/) the snapshots
    are refreshed here.
    """
    minth = _find_minth_vtu(results_dir)
    if minth is not None:
        vtu_summary = _render_minth(minth)
        if vtu_summary:
            context_text += f"\n\n{MINTH_RENDER_SUMMARY_HEADER}\n{vtu_summary}\n"

    diag = _quality_diag(results_dir, key)
    # Console visibility: the material balance is the one number a human
    # watching the run needs in order to see a degenerate design.
    if diag.get("solid_last") is not None:
        frac = ""
        if diag.get("domain_area"):
            frac = f" ({100.0 * diag['solid_last'] / diag['domain_area']:.1f}% of domain)"
        ell_txt = f", ell={diag['ell']:g}" if diag.get("ell") is not None else ""
        chi_txt = f", chi={diag['chi']}" if diag.get("chi") else ""
        print(
            f"    [Reflect] design: material area {diag['solid_last']:.4g}{frac}"
            f"{ell_txt}{chi_txt}",
            flush=True,
        )
    # Facts only (material balance, return codes) — the pass vs
    # wrong_result JUDGMENT is always the LLM's (no deterministic gates).
    context_text += _quality_guidance(diag)

    if vision_images is None:
        # Refresh snapshots AFTER the latest execution — never reuse the
        # snapshots from an earlier design evolution.
        vision_images, image_section = _vision_context(results_dir, config)

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

    Uses the string-aware call scanner of check_static so a comment
    mention of ``iterations=`` never matches.  When no setup_optimization
    call parses, falls back to the first REAL (non-comment, non-string)
    ``iterations=<digits>`` occurrence.
    """
    from Generation.check_static import _find_setup_optimization, _param_span

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
        from Generation.check_static import _insert_before_closing

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
# Afac (Lagrange multiplier) value read
#
# The volume weight ``ell`` is a BALANCE parameter of the optimisation,
# not part of the problem statement — Afac is an ordinary DESIGNED
# value and belongs to the design_fix permission set (the
# deterministic permission is built by fixer.build_design_permission;
# the enforcer is fixer.design_enforce).  This helper stays for the
# quality diagnostics (ell readout); write-backs go through the generic
# classifier._constant_replace inside design_enforce.
# ======================================================================

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


def _empty_probe_record(idx: int, error: str) -> dict:
    """A probe record for a worker that never produced one."""
    return {
        "idx": idx,
        "fix_type": "",
        "direction": "",
        "rationale": "",
        "main": "",
        "mesh": "",
        "probe_epochs": 0,
        "completed_epochs": 0,
        "final_objective": None,
        "objective_data": "",
        "run_result_json": "{}",
        "py_rc": "?",
        "ff_rc": "?",
        "minth": False,
        "render_summary": "",
        "images": [],
        "error": error,
    }


def _per_attempt_cleanup(cand_dir: Path) -> None:
    """Remove an attempt's results/, tex/, tmp/ subdirectories.

    The attempt folder itself survives (temp/ is the record of the
    attempts — it is never deleted, neither per-round nor at process
    end); only the execution scratch inside each attempt goes.
    """
    for sub in ("results", "tex", "tmp"):
        _rmtree_force(cand_dir / sub)


def _run_one_probe(
    idx: int,
    iteration: int,
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
    repair_records: str,
    doc_text: str,
    analysis_artifacts: str,
    coarse: str,
    fix_type: str,
    permission: dict,
    use_vision: bool,
    vision_images,
    image_section: str,
) -> dict:
    """Generate ONE candidate fix (one direction) and probe-run it.

    Runs inside a worker thread.  A dedicated agent is built HERE
    because CodeAgent keeps prompt state on the instance and cannot be
    shared across threads.  The agent MODEL follows the caller's
    *use_vision* decision (the category rule): syntax always text,
    fe_parameter always vision, formula/design/other vision with design
    images and text without.  The
    probe's LLM calls and its execution are recorded in FULL into
    history.txt (round = *iteration*).
    """
    from API import (
        DeepSeekCodeAgentV4FlashVision,
        DeepSeekCodeAgentV4Pro,
    )
    from .prompts import (
        FIX_DESIGN_DIRECTIONS,
        FIX_FE_PARAM_DIRECTIONS,
        FIX_FORMULA_DIRECTIONS,
    )

    direction_tables = {
        "formula_fix": FIX_FORMULA_DIRECTIONS,
        "design_fix": FIX_DESIGN_DIRECTIONS,
        "fe_param_fix": FIX_FE_PARAM_DIRECTIONS,
    }
    directions = direction_tables[fix_type]
    direction = directions[idx % len(directions)]
    # Every candidate runs the SAME probe budget (a configs parameter —
    # it is also part of the evidence the synthesis weighs, so a varying
    # budget would confound the comparison).
    probe_epochs = config.REFLECTION_PROBE_EPOCHS
    record = _empty_probe_record(idx, "")
    record["fix_type"] = fix_type
    record["direction"] = direction
    record["probe_epochs"] = probe_epochs

    cand_dir = temp_dir / f"cand_{idx}"
    _rmtree_force(cand_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)

    round_tag = f"Reflection round {iteration}: {fix_type} probe {idx}"

    try:
        probe_agent = (
            DeepSeekCodeAgentV4FlashVision()
            if use_vision
            else DeepSeekCodeAgentV4Pro()
        )
        # the probe's LLM calls go into the SAME history.txt (the header
        # exists — point at it, never rewrite it)
        history_log = results_dir / "history.txt"
        if history_log.exists():
            probe_agent.attach_history(history_log, write_header=False)
        probe_agent.history_label = round_tag
        if fix_type == "design_fix":
            fix = design_fix(
                probe_agent,
                problem_description=problem_description,
                main_py=main_py,
                mesh_txt=mesh_txt,
                run_result_json=run_result_json,
                objective_data=objective_data,
                convergence_data=convergence_data,
                edp_code=edp_code,
                repair_records=repair_records,
                doc_text=doc_text,
                analysis_artifacts=analysis_artifacts,
                coarse=coarse,
                permission=permission,
                images=vision_images if use_vision else None,
                image_section=image_section,
                fix_direction=direction,
            )
        elif fix_type == "fe_param_fix":
            fix = fe_param_fix(
                probe_agent,
                problem_description=problem_description,
                main_py=main_py,
                mesh_txt=mesh_txt,
                run_result_json=run_result_json,
                objective_data=objective_data,
                convergence_data=convergence_data,
                edp_code=edp_code,
                repair_records=repair_records,
                doc_text=doc_text,
                analysis_artifacts=analysis_artifacts,
                coarse=coarse,
                permission=permission,
                images=vision_images if use_vision else None,
                image_section=image_section,
                fix_direction=direction,
            )
        else:
            fix = formula_fix(
                probe_agent,
                problem_description=problem_description,
                main_py=main_py,
                mesh_txt=mesh_txt,
                run_result_json=run_result_json,
                objective_data=objective_data,
                convergence_data=convergence_data,
                edp_code=edp_code,
                repair_records=repair_records,
                doc_text=doc_text,
                analysis_artifacts=analysis_artifacts,
                coarse=coarse,
                images=vision_images if use_vision else None,
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
    try:
        (cand_dir / "main.py").write_text(record["main"], encoding="utf-8")
        (cand_dir / "mesh.txt").write_text(record["mesh"] or mesh_txt, encoding="utf-8")
        for name in ("initial.json", "mesh.json"):
            src = results_dir / name
            if src.exists():
                shutil.copy2(src, cand_dir / name)
    except OSError as e:
        record["error"] = f"probe directory setup failed: {e}"
        return record

    # Probe run (short budget — the formal run happens after synthesis).
    # The attempt's results/, tex/, tmp/ are removed in the finally below
    # as soon as the evidence is gathered — the attempt folder itself
    # stays as a record (temp/ survives the process).
    from Execution.runner import run_problem

    try:
        try:
            history_log = results_dir / "history.txt"
            result = run_problem(
                cand_dir / "main.py",
                cand_dir,
                auto_render=False,
                history_log=history_log if history_log.exists() else None,
                history_label=f"{round_tag} run",
            )
        except Exception as e:
            record["error"] = f"probe run crashed: {e}"
            return record

        # Evidence collection — mirrors the fix-start collection: the SAME
        # blacklist (collect_files), the SAME truncation policy (only the
        # FreeFEM console keeps its last chars — everything else
        # verbatim), and the SAME image rule (configs.MAX_SNAPSHOT_FRAMES
        # with whole-run interpolation).  Guarded as a whole: a
        # filesystem race or a render hiccup must keep the already-
        # gathered fields instead of turning a genuinely completed run
        # into a "worker crashed" record.
        try:
            record["py_rc"] = result.get("autofreefem", {}).get("python", {}).get("return_code", "?")
            ff_rcs = [f.get("return_code", "?") for f in result.get("freefem", [])]
            record["ff_rc"] = ", ".join(str(rc) for rc in ff_rcs) or "-"
            # rc=0 with a console err code is the "code appeared after loop
            # completion" case — keep the code visible WITHOUT touching the
            # raw ff_rc string (parsed for clean-run checks elsewhere).
            errs = [
                f.get("console_err_code")
                for f in result.get("freefem", [])
                if f.get("console_err_code") is not None
            ]
            if errs:
                record["ff_console_err"] = errs

            key_c = _extract_key_files(collect_files(cand_dir))
            record["run_result_json"] = key_c["run_result_json"]
            record["objective_data"] = key_c["objective_data"]
            rows = []
            for line in (key_c["objective_data"] or "").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("iteration", "#", "===")):
                    continue
                parts = stripped.split(",")
                if len(parts) < 2:
                    continue
                try:
                    int(parts[0].strip())
                    float(parts[1].strip())
                except ValueError:
                    continue
                rows.append((parts[0].strip(), parts[1].strip()))
            if rows:
                record["completed_epochs"] = len(rows)
                record["final_objective"] = float(rows[-1][1])

            # Design images per configs (snapshots, whole-run
            # interpolation) — the lock serialises the matplotlib calls of
            # the parallel probes (pyplot keeps process-global state).
            with _RENDER_LOCK:
                record["images"], _ = _vision_context(cand_dir, config)
            minth = _find_minth_vtu(cand_dir)
            if minth is not None:
                record["minth"] = True
                with _RENDER_LOCK:
                    record["render_summary"] = render_and_summarize(minth)
        except Exception as e:
            record["error"] = f"probe evidence collection failed: {e}"
    finally:
        _per_attempt_cleanup(cand_dir)

    return record


def _probe_report_text(records: list[dict]) -> str:
    """Assemble the candidate report block for the synthesiser."""
    parts = []
    for r in records:
        # Computed separately — a conditional expression between the
        # adjacent f-string literals would break implicit concatenation
        # (the whole chain parses as ``head if cond else tail``).
        err_line = f"Error (if any): {r['error']}\n" if r.get("error") else ""
        console_note = (
            f"FreeFEM console_err_code: {r['ff_console_err']}\n"
            if r.get("ff_console_err")
            else ""
        )
        parts.append(
            f"=== Candidate {r['idx']} ===\n"
            f"Fix type: {r.get('fix_type') or '(unknown)'}\n"
            f"Direction: {r['direction'] or '(unknown)'}\n"
            f"Fix rationale: {r['rationale'] or '(none)'}\n"
            f"Probe budget: {r['probe_epochs']} epochs\n"
            f"Probe result: Python rc={r['py_rc']}, FreeFEM rc={r['ff_rc']}, "
            f"{console_note}"
            f"final objective={r['final_objective'] if r['final_objective'] is not None else '?'}, "
            f"MinTh.vtu={'yes' if r['minth'] else 'no'}\n"
            f"Objective history (verbatim):\n{r['objective_data'] or '(none)'}\n"
            f"Execution result (FreeFEM console keeps only its last "
            f"chars):\n{r['run_result_json'] or '(none)'}\n"
            f"Render summary: {r['render_summary'] or '(none)'}\n"
            f"{err_line}"
            f"Candidate main.py:\n```python\n{r['main'] or '(none)'}\n```\n"
            f"Candidate mesh.txt:\n```\n{r['mesh'] or '(no change)'}\n```\n\n"
        )
    return "\n".join(parts)


def _synthesize_next_step(
    agent,
    config,
    temp_dir: Path,
    problem_description: str,
    main_py: str,
    mesh_txt: str,
    repair_records: str,
    doc_text: str,
    analysis_artifacts: str,
    coarse: str,
    fix_type: str,
    permission: dict,
    records: list[dict],
    iteration: int,
    use_vision: bool = True,
) -> dict:
    """Synthesise the formal next-step code from the probe records.

    One LLM call over all candidates: their directions, code, probe
    evidence (return codes, epochs, objective, design renders) — the
    next step is judged from BOTH the objective values and the design
    images (the renders attach only when *use_vision* — a text-model
    round must not receive image blocks).  An unusable reply (malformed
    JSON, missing code keys, unreachable model) returns ``_error`` —
    the caller treats it UNIFORMLY as other_error (no deterministic
    best-candidate fallback).
    The deterministic permission boundary filters the output for the
    design / fe-parameter rounds.
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
    # Text-model rounds (use_vision=False) get NO image blocks — the
    # synthesis then weighs the objective numbers and the code alone.
    # The images follow configs.MAX_SNAPSHOT_FRAMES (whole-run
    # interpolation) — the same rule as the fix-start collection.
    images = []
    lines = []
    if use_vision:
        for r in records:
            for img in r.get("images") or []:
                images.append(img)
                lines.append(
                    f"{Path(img).name} — Candidate {r['idx']} "
                    "design (Label=3 cells)"
                )
    image_section = ""
    if images:
        image_section = (
            "\n\n=== PROBE DESIGN IMAGES (attached, in candidate order) ===\n"
            + "\n".join(lines)
            + "\n"
        )

    permission_block = permission_text(fix_type, permission)

    agent.set_prompt_list(
        [
            {"role": "system", "content": FIX_SYNTHESIS_SYSTEM},
            {"role": "user", "content": FIX_SYNTHESIS_USER},
        ]
    )

    # NO deterministic fallback: an unusable synthesis reply (malformed
    # JSON, missing code keys, or an unreachable model — the connector
    # test already ruled out a dead link) is treated UNIFORMLY as
    # other_error by the caller; no best-probe-candidate substitution.
    try:
        raw = agent.invoke_multimodal(
            {
                "n_candidates": len(records),
                "probe_epochs": config.REFLECTION_PROBE_EPOCHS,
                "formal_iterations": _expected_epochs(main_py, config),
                "problem_description": problem_description,
                "main_py": main_py,
                "mesh_txt": mesh_txt,
                "coarse": coarse,
                "fix_type": fix_type,
                "permission": permission_block,
                "candidates_report": report,
                "repair_records": repair_records or "(no repair records yet)",
                "doc_text": doc_text,
                "analysis_artifacts": analysis_artifacts or "(not available)",
                "image_section": image_section,
            },
            images=images,
        )
    except Exception as e:
        return {
            "main": "",
            "mesh": "",
            "rationale": "",
            "_error": f"synthesis LLM call failed: {e}",
        }

    from .fixer import _parse_fix_response

    parsed = _parse_fix_response(raw)
    if parsed.get("_error"):
        return {
            "main": "",
            "mesh": "",
            "rationale": "",
            "_error": f"synthesis parse failed ({parsed['_error']})",
        }
    # the deterministic permission boundary filters the synthesis output
    # the same way it filtered the candidates (design / fe-parameter
    # rounds only)
    if fix_type == "design_fix":
        parsed["main"], parsed["mesh"], _ = design_enforce(
            main_py, parsed.get("main") or "", parsed.get("mesh") or "", permission,
        )
    elif fix_type == "fe_param_fix":
        parsed["main"], parsed["mesh"], _ = fe_param_enforce(
            main_py, parsed.get("main") or "", parsed.get("mesh") or "", permission,
        )
    return parsed


# ======================================================================
# main orchestrator
# ======================================================================

def _console_evidence(result: dict) -> str:
    """The initial execution's console evidence (the round-0 record).

    Python return code + stderr (verbatim), and each FreeFEM record's
    exit code (with the console err code when present).  The FreeFEM
    console was already truncated at the source (Execution/runner.py) —
    read directly, no re-cutting.
    """
    if not result:
        return "(no run_result.json)"
    parts = []
    py = result.get("autofreefem", {}).get("python", {})
    if py:
        parts.append(f"Python rc={py.get('return_code', '?')}")
        if py.get("stderr"):
            parts.append(f"Python stderr:\n{py['stderr']}")
    for ff in result.get("freefem", []):
        note = (
            f" (console_err_code={ff['console_err_code']})"
            if ff.get("console_err_code") is not None
            else ""
        )
        parts.append(
            f"FreeFEM {ff.get('edp_file', '?')}: exit={ff.get('return_code', '?')}{note}"
        )
        stream = ff.get("stderr")
        if isinstance(stream, str) and stream:
            parts.append(f"stderr:\n{stream}")
    return "\n".join(parts)


def _record_round_summary(results_dir: Path, iteration: int, fix_type: str,
                          coarse: str, outcome: str) -> None:
    """One summary record per reflection round in history.txt."""
    history_log = results_dir / "history.txt"
    if not history_log.exists():
        return
    record_history(
        history_log,
        "Reflection round (summary)",
        label=f"round {iteration}: {fix_type} for {coarse}",
        output_text=outcome,
    )


def _repair_records_text(records: list) -> str:
    """Render the in-memory repair records for prompts.

    Round 0 = the initial execution's console evidence; each later round
    = what problem it aimed to fix (target) and what it output
    (added / deleted / modified parts), plus its outcome.
    """
    if not records:
        return "(no repair records yet)"
    parts = []
    for r in records:
        if r.get("round") == 0:
            parts.append(
                f"--- round 0 (initial execution) ---\n"
                f"console:\n{r.get('console', '')}\n"
                f"outcome: {r.get('outcome', '?')}\n"
            )
            continue
        t = r.get("target") or {}
        if isinstance(t, dict):
            target = (
                f"{t.get('fix_type', '?')} for {t.get('coarse', '?')}"
                f"{' - ' + t['diagnosis'] if t.get('diagnosis') else ''}"
            )
        else:
            target = str(t)
        ch = r.get("changes") or {}
        parts.append(
            f"--- round {r.get('round', '?')} [{target}] ---\n"
            f"removed: {ch.get('removed') or '[]'}\n"
            f"added:   {ch.get('added') or '[]'}\n"
            f"modified: {ch.get('modified') or '[]'}\n"
            f"outcome: {r.get('outcome', '?')}\n"
        )
    return "\n".join(parts)


def _load_specs(results_dir: Path) -> tuple:
    """(initial.json dict, mesh.json dict) from disk — {} on any failure
    (the eligibility gates then fail closed to free_fix)."""
    out = []
    for name in ("initial.json", "mesh.json"):
        try:
            data = json.loads(
                (results_dir / name).read_text(encoding="utf-8", errors="replace")
            )
        except (OSError, json.JSONDecodeError, TypeError):
            data = {}
        out.append(data if isinstance(data, dict) else {})
    return tuple(out)


def _load_examples(results_dir: Path) -> list:
    """Selected reference examples (examples.txt names) for the static
    repair review — [] when absent or unloadable."""
    try:
        names = [
            line.strip()
            for line in (results_dir / "examples.txt").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
    except OSError:
        return []
    if not names:
        return []
    try:
        from Generation.select_examples import load_selected_examples

        return load_selected_examples(names) or []
    except Exception:
        return []


def _run_syntax_fix(fix_agent, config, results_dir: Path, key: dict,
                    history_log=None) -> dict:
    """syntax_fix — the Generation rule set verbatim: deterministic
    static checks -> LLM review -> targeted repair loop (bounded by
    configs.MAX_STATIC_FIXES).  The reflector re-executes the result;
    one reflection round is consumed regardless of the inner repairs.
    Every inner round is recorded in FULL into history.txt."""
    from Generation.check_static import run_static_repair

    spec, mesh_spec = _load_specs(results_dir)
    return run_static_repair(
        fix_agent,
        key["main_py"],
        problem_spec=spec,
        mesh_spec=mesh_spec,
        problem_description=key["problem_description"],
        mesh_txt=key["mesh_txt"],
        examples=_load_examples(results_dir),
        max_fixes=config.MAX_STATIC_FIXES,
        history_log=history_log,
    )


def run_reflection(agent, info, config) -> str:
    """
    Run the full reflection loop for a single problem.

    Each round: collect -> quality-check evidence (when MinTh.vtu exists;
    the PRE-fix verdict is EVIDENCE ONLY) -> classify (5 error
    categories + pass — the classification is the sole authority for
    the category) -> fix (syntax_fix / formula_fix / design_fix /
    fe_param_fix / free_fix, all sharing MAX_REFLECTION_TIMES rounds)
    -> re-execute -> post-fix quality check (pass — or a fix that
    produced NO change — ends the reflection DIRECTLY, no extra round).
    Before the FIRST repair round the usable fix types are pre-computed
    from the extracted/designed rules (the classification prompt shows
    ONLY the usable error types; an unusable type falls back to
    free_fix).  formula_fix / design_fix / fe_param_fix test up to
    MAX_FIX_CANDIDATES candidates in parallel under
    results/<name>/temp/ (short probes of REFLECTION_PROBE_EPOCHS
    epochs); an LLM synthesiser then merges the probe evidence
    (objective + design images) into ONE next-step code for the formal
    run (an unusable synthesis reply is treated uniformly as
    other_error).

    Args:
        agent: CodeAgent instance (an AgentRouter in the pipeline —
               syntax_error rounds repair with its text_agent, the other
               repair paths with its vision_agent).
        info: Info object with problem_name, results_dir, repair_records.
        config: Module with reflection constants (MAX_REFLECTION_TIMES,
                MAX_FIX_CANDIDATES, REFLECTION_PROBE_EPOCHS, etc.).

    Returns:
        Reflection status string: "pass", "give_up_max",
        or "give_up" (no files found).

    temp/ survives the process — the attempts are the record; each
    attempt only loses its results/, tex/, tmp/ subdirectories.
    """
    results_dir = info.results_dir
    problem_name = info.problem_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parallel-probe scratch space — a RECORD of the attempts.  It is
    # never wiped wholesale: not on entry (previous attempts stay), not
    # between rounds, not at process end.  Each candidate wipes its own
    # slot on reuse and sheds only its results/, tex/, tmp/ at the end.
    temp_dir = results_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize reflection log; the in-memory repair records are reset
    # with it.
    _init_reflection_log(results_dir, problem_name)
    info.repair_records = []

    # The repair MODEL depends on the error class: SYNTAX errors are
    # repaired with the TEXT model (static rules + code repair — the
    # vision model degrades code text); FE_PARAMETER errors always use
    # the VISION model; the formula / design / other paths use VISION
    # when design images exist and TEXT otherwise.  The quality check
    # and the coarse classification always see the design images —
    # vision.  main.py passes an AgentRouter(Pro, FlashVision); fall
    # back to the agent itself when a side is absent.
    text_agent = getattr(agent, "text_agent", None) or agent
    vision_agent = getattr(agent, "vision_agent", None) or agent

    # Connection tester FIRST: an unreachable model ends the reflection
    # immediately instead of burning repair rounds on connection errors.
    from API import test_connection

    if not test_connection(agent):
        print("    [Reflect] LLM connection test FAILED — ending immediately.", flush=True)
        _append_final_status(results_dir, "connection_failed", 0)
        info.reflection_status = "connection_failed"
        return "connection_failed"

    # Load AutoFreeFEM documentation once — anchored on this file, not
    # the CWD (running the pipeline from elsewhere must not silently
    # empty the documentation context).
    doc_path = (
        Path(__file__).resolve().parent.parent
        / "database" / "freefem_functions" / "list_of_classes.txt"
    )
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""

    # Pre-computed BEFORE the first repair round (the spec is stable
    # across rounds): which fix types have repair room under the
    # extracted/designed rules.  The classification prompt shows ONLY
    # the usable types; fix_type_for fails closed to free_fix for an
    # unusable one.  mesh_frozen = the mesh code came verbatim from the
    # sources (borders/resolution immutable).
    available = None
    mesh_frozen = False

    try:
        for iteration in range(config.MAX_REFLECTION_TIMES):
            info.reflection_attempts = iteration + 1
            print(f"\n  [Reflect] Iteration {iteration + 1}/{config.MAX_REFLECTION_TIMES}", flush=True)

            # --- Step 1: Collect + structured context ---
            files = collect_files(results_dir)
            if not files:
                print("    [Reflect] No files found — skipping.")
                info.reflection_status = "give_up"
                return "give_up"

            context_text = _assemble_context_text(files)
            key = _extract_key_files(files)

            # --- Step 1b: per-snapshot design images (multimodal evidence) ---
            # Every result*Th<N>.vtu snapshot's Label field is merged, in
            # iteration order, into ONE image the model can see.
            vision_images, image_section = _vision_context(results_dir, config)

            # --- Step 2: (Re)execute when there is no result yet; then
            # capture the round-0 record from the FIRST execution. ---
            run_result_json = key["run_result_json"]
            if run_result_json == "{}":
                _re_execute(
                    results_dir,
                    history_label=f"Reflection round {iteration + 1}: initial re-execute",
                )
                files = collect_files(results_dir)
                context_text = _assemble_context_text(files)
                key = _extract_key_files(files)
                run_result_json = key["run_result_json"]
                # fresh snapshots after re-execution — refresh the images
                vision_images, image_section = _vision_context(results_dir, config)

            if not info.repair_records:
                # Stage 6 always ran before Stage 7 — this IS the initial
                # execution's console evidence (or the Step-2 re-execution
                # when the original result was missing).
                rr = _load_run_result(results_dir)
                info.repair_records.append(
                    {
                        "round": 0,
                        "target": "initial execution",
                        "console": _console_evidence(rr),
                        "outcome": _format_outcome(
                            rr, _find_minth_vtu(results_dir) is not None
                        ),
                    }
                )

            repair_records_text = _repair_records_text(info.repair_records)

            analysis_artifacts = ""
            for rel in sorted(files):
                if rel in ("initial.json", "mesh.json"):
                    # Include _source annotations — critical for distinguishing
                    # extracted (immutable) vs designed (mutable) values.
                    # Verbatim (no truncation).
                    analysis_artifacts += f"=== {rel} ===\n{files[rel]}\n\n"

            spec, mesh_spec = _load_specs(results_dir)
            if available is None:
                mesh_frozen = mesh_spec.get("mesh_code_source") == "extracted"
                available = available_fix_types(
                    spec, key["main_py"], mesh_frozen,
                )
                print(
                    "    [Reflect] usable fix types: "
                    + ", ".join(t for t, (ok, _) in available.items() if ok),
                    flush=True,
                )

            # --- Step 3: Quality gate / coarse classification ---
            # MinTh.vtu existence is the only hard rule; the objective
            # history never short-circuits the classification.
            minth = _find_minth_vtu(results_dir)
            diagnosis = ""

            if minth is not None:
                # MinTh.vtu exists — the quality check enriches the
                # evidence (render summary, material-balance facts,
                # symmetry guidance, final-design image) and gives a
                # verdict.  The verdict is EVIDENCE ONLY: the coarse
                # classification below is the SOLE authority for pass
                # (and for every category — no fixed wiring).
                set_label(
                    vision_agent,
                    f"Reflection round {iteration + 1}/{config.MAX_REFLECTION_TIMES}"
                    ": quality check",
                )
                qc, context_text = _quality_check_block(
                    vision_agent, problem_name, config, results_dir, context_text, key,
                    vision_images=vision_images, image_section=image_section,
                )
                diagnosis = f"quality check: {qc}"
                context_text += (
                    f"\n\n=== Quality check verdict: {qc} (evidence only — "
                    "the classification below is the sole authority for "
                    "pass) ===\n"
                )
                print(f"    [Reflect] Quality check: {qc}", flush=True)

            # --- standard flow: LLM coarse classification — the ONLY
            # category judgment (with MinTh or without, pass or
            # wrong_result).  The prompt shows ONLY the usable error
            # types. ---
            set_label(
                vision_agent,
                f"Reflection round {iteration + 1}/{config.MAX_REFLECTION_TIMES}"
                ": coarse classify",
            )
            coarse = coarse_classify(
                vision_agent, problem_name, context_text,
                images=vision_images, image_section=image_section,
                available_types=available_types_text(available),
            )
            print(f"    [Reflect] Coarse: {coarse}", flush=True)

            if coarse == "pass" and not vision_images:
                # pass judged WITHOUT design images (curve-only evidence):
                # supplement with ONE vision re-judgment — a pass needs a
                # healthy curve AND healthy images.  A failed re-check
                # falls through to the repair flow below.
                set_label(
                    vision_agent,
                    f"Reflection round {iteration + 1}/{config.MAX_REFLECTION_TIMES}"
                    ": pass re-check (vision)",
                )
                recheck = coarse_classify(
                    vision_agent, problem_name, context_text,
                    images=None, image_section="",
                    available_types=available_types_text(available),
                )
                print(f"    [Reflect] pass re-check (vision): {recheck}", flush=True)
                coarse = recheck

            if coarse == "pass":
                print("    [Reflect] PASS.", flush=True)
                outcome = _format_outcome(
                    _load_run_result(results_dir),
                    minth_exists=minth is not None,
                )
                _append_reflection_log(
                    results_dir,
                    iteration + 1,
                    "pass",
                    "-",
                    "none",
                    "Coarse classification: result valid — no fix needed.",
                    outcome,
                )
                info.repair_records.append(
                    {
                        "round": iteration + 1,
                        "target": {
                            "fix_type": "none",
                            "coarse": "pass",
                            "diagnosis": "coarse classification passed",
                        },
                        "changes": {},
                        "outcome": outcome,
                    }
                )
                _record_round_summary(results_dir, iteration + 1, "none", "pass", outcome)
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"

            # --- Step 4: Fix-type selection (the pre-computed
            # eligibility gates — an unusable category falls back to
            # the free repair). ---
            fix_type, gate_note = fix_type_for(coarse, available)
            if gate_note:
                print(f"    [Reflect] {gate_note}", flush=True)
            if not diagnosis:
                diagnosis = gate_note or coarse
            print(f"    [Reflect] Fix type: {fix_type}", flush=True)

            # syntax_error -> TEXT; fe_parameter_error -> VISION (even
            # without design images); formula/design/other -> VISION
            # with design images, TEXT without (the fix_agent below
            # serves Steps 5-7).
            use_vision = (
                False if coarse == "syntax_error"
                else True if coarse == "fe_parameter_error"
                else bool(vision_images)
            )
            fix_agent = vision_agent if use_vision else text_agent

            prev_main = key["main_py"]
            prev_mesh = key["mesh_txt"]
            new_main = ""
            new_mesh = ""
            rationale = ""

            # --- Step 5: Execute the fix strategy ---
            if fix_type == "syntax_fix":
                history_log = results_dir / "history.txt"
                res = _run_syntax_fix(
                    fix_agent, config, results_dir, key,
                    history_log=history_log if history_log.exists() else None,
                )
                for w in res.get("warnings") or []:
                    print(f"    [Syntax] [W] {w.get('check')}: {w.get('message')}", flush=True)
                new_main = res.get("main") or prev_main
                rationale = (
                    "static check -> review -> repair loop"
                    f" ({len(res.get('history') or [])} repair(s),"
                    f" passed={bool(res.get('_passed'))})"
                )
            elif fix_type == "free_fix":
                set_label(
                    fix_agent,
                    f"Reflection round {iteration + 1}/{config.MAX_REFLECTION_TIMES}: free_fix",
                )
                fix = free_fix(
                    fix_agent,
                    problem_description=key["problem_description"],
                    main_py=prev_main,
                    mesh_txt=prev_mesh,
                    run_result_json=run_result_json,
                    objective_data=key["objective_data"],
                    convergence_data=key["convergence_data"],
                    edp_code=key["edp_code"],
                    repair_records=repair_records_text,
                    doc_text=doc_text,
                    analysis_artifacts=analysis_artifacts,
                    coarse=coarse,
                    images=vision_images if use_vision else None,
                    image_section=image_section,
                )
                if fix.get("_error"):
                    print(f"    [Reflect] free_fix failed: {fix['_error']}", flush=True)
                    _append_reflection_log(
                        results_dir, iteration + 1, coarse, "-", fix_type,
                        "(fix failed)", f"free_fix failed: {fix['_error']}",
                    )
                    info.repair_records.append(
                        {
                            "round": iteration + 1,
                            "target": {
                                "fix_type": fix_type,
                                "coarse": coarse,
                                "diagnosis": diagnosis,
                            },
                            "changes": {},
                            "outcome": f"free_fix failed: {fix['_error']}",
                        }
                    )
                    continue
                new_main = fix.get("main") or prev_main
                new_mesh = fix.get("mesh") or ""
                rationale = fix.get("rationale", "")
            else:
                # formula_fix / design_fix / fe_param_fix — parallel
                # probe round under temp/.  The design and fe-parameter
                # probes carry the deterministic permission set.
                permission = {}
                if fix_type == "design_fix":
                    permission = build_design_permission(
                        spec, prev_main, mesh_frozen,
                    )
                elif fix_type == "fe_param_fix":
                    permission = build_fe_param_permission(
                        spec, prev_main, mesh_frozen,
                    )
                n_probes = max(1, int(config.MAX_FIX_CANDIDATES))
                print(
                    f"    [Reflect] probing {n_probes} {fix_type} candidates in "
                    f"parallel ({config.REFLECTION_PROBE_EPOCHS} epochs each) ...",
                    flush=True,
                )
                records = [None] * n_probes
                with ThreadPoolExecutor(max_workers=n_probes) as pool:
                    futures = {}
                    for i in range(n_probes):
                        fut = pool.submit(
                            _run_one_probe,
                            i,
                            iteration + 1,
                            config,
                            results_dir,
                            temp_dir,
                            problem_description=key["problem_description"],
                            main_py=prev_main,
                            mesh_txt=prev_mesh,
                            run_result_json=run_result_json,
                            objective_data=key["objective_data"],
                            convergence_data=key["convergence_data"],
                            edp_code=key["edp_code"],
                            repair_records=repair_records_text,
                            doc_text=doc_text,
                            analysis_artifacts=analysis_artifacts,
                            coarse=coarse,
                            fix_type=fix_type,
                            permission=permission,
                            use_vision=use_vision,
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

                # --- Synthesise the next-step code from the probe
                # evidence — judged from BOTH the objective values and
                # the design images. ---
                print("    [Reflect] synthesising next-step code ...", flush=True)
                set_label(
                    fix_agent,
                    f"Reflection round {iteration + 1}/{config.MAX_REFLECTION_TIMES}"
                    ": synthesis ({fix_type})",
                )
                synth = _synthesize_next_step(
                    fix_agent,
                    config,
                    temp_dir,
                    problem_description=key["problem_description"],
                    main_py=prev_main,
                    mesh_txt=prev_mesh,
                    repair_records=repair_records_text,
                    doc_text=doc_text,
                    analysis_artifacts=analysis_artifacts,
                    coarse=coarse,
                    fix_type=fix_type,
                    permission=permission,
                    records=[r for r in records if r is not None],
                    iteration=iteration + 1,
                    use_vision=use_vision,
                )

                if synth.get("_error"):
                    # An unusable synthesis reply is treated UNIFORMLY as
                    # other_error (no deterministic best-candidate
                    # fallback) — nothing to write or run this round.
                    print(
                        f"    [Reflect] Synthesis unusable: {synth['_error']} "
                        "— treated as other_error",
                        flush=True,
                    )
                    _append_reflection_log(
                        results_dir, iteration + 1, "other_error", "-", fix_type,
                        "(synthesis failed)",
                        f"synthesis failed — treated as other_error: {synth['_error']}",
                    )
                    info.repair_records.append(
                        {
                            "round": iteration + 1,
                            "target": {
                                "fix_type": fix_type,
                                "coarse": "other_error",
                                "diagnosis": (
                                    "synthesis failed — treated as "
                                    f"other_error: {synth['_error']}"
                                ),
                            },
                            "changes": {},
                            "outcome": f"synthesis failed — treated as other_error: {synth['_error']}",
                        }
                    )
                    continue
                new_main = synth.get("main") or prev_main
                new_mesh = synth.get("mesh") or ""
                rationale = synth.get("rationale", "")

            # --- Step 6: Apply + formal re-execute ---
            applied_main = prev_main
            applied_mesh = prev_mesh
            if new_main and new_main != prev_main:
                # Probe budgets are probe-only — the formal run uses the
                # problem's own iteration count (iterations= is FIXED
                # content — the formula/design/fe-parameter repairs may
                # not change it; the syntax repair owns it).
                code_out = _force_iterations(
                    new_main, _expected_epochs(prev_main, config)
                )
                (results_dir / "main.py").write_text(code_out, encoding="utf-8")
                applied_main = code_out
            if new_mesh and new_mesh != prev_mesh:
                (results_dir / "mesh.txt").write_text(new_mesh, encoding="utf-8")
                applied_mesh = new_mesh

            # The fix program had NOTHING to fix (or nothing it could
            # change — the repaired code is byte-identical to the
            # previous one): treat the current state as final and END
            # the reflection instead of looping identical rounds.
            if applied_main == prev_main and applied_mesh == prev_mesh:
                print(
                    "    [Reflect] fix produced NO change — treating as "
                    "PASS and ending.",
                    flush=True,
                )
                outcome_nc = _format_outcome(
                    _load_run_result(results_dir),
                    _find_minth_vtu(results_dir) is not None,
                )
                _append_reflection_log(
                    results_dir, iteration + 1, coarse, "-", fix_type,
                    "(no change — nothing more to fix)",
                    f"fix produced no change — treated as pass: {outcome_nc}",
                )
                info.repair_records.append(
                    {
                        "round": iteration + 1,
                        "target": {
                            "fix_type": fix_type,
                            "coarse": coarse,
                            "diagnosis": diagnosis,
                        },
                        "changes": {},
                        "outcome": "fix produced no change — treated as pass",
                    }
                )
                _append_final_status(results_dir, "pass", info.reflection_attempts)
                info.reflection_status = "pass"
                return "pass"

            result = _re_execute(
                results_dir,
                history_label=f"Reflection round {iteration + 1}: formal re-execute ({fix_type})",
            )

            # --- Step 7: Outcome + repair records + reflection log ---
            minth_after = _find_minth_vtu(results_dir)
            outcome = _format_outcome(result, minth_after is not None)

            from Generation.check_static import _diff_code

            record = {
                "round": iteration + 1,
                "target": {
                    "fix_type": fix_type,
                    "coarse": coarse,
                    "diagnosis": diagnosis,
                },
                "changes": _diff_code(prev_main, applied_main),
                "outcome": outcome,
            }
            if applied_mesh != prev_mesh:
                record["mesh_changes"] = _diff_code(prev_mesh, applied_mesh)
            info.repair_records.append(record)
            _record_round_summary(results_dir, iteration + 1, fix_type, coarse, outcome)

            _append_reflection_log(
                results_dir, iteration + 1, coarse, "-", fix_type, rationale, outcome
            )

            # --- Step 8: Post-fix quality check ---
            # A fix whose re-execution produced a valid result ends the
            # reflection DIRECTLY — no extra round.  Only when the
            # re-execution actually ran: `result` is {} when main.py was
            # missing, in which case a MinTh.vtu from an EARLIER run may
            # still sit there and judging it would accept a stale design
            # as the outcome of the fix.
            if minth_after is not None and result:
                files = collect_files(results_dir)
                context_text = _assemble_context_text(files)
                key_after = _extract_key_files(files)

                qc, _ = _quality_check_block(
                    vision_agent, problem_name, config, results_dir,
                    context_text, key_after,
                )
                if qc == "pass":
                    print("    [Reflect] PASS — fix succeeded, result valid.", flush=True)
                    _append_reflection_log(
                        results_dir, iteration + 1, "pass", "-", "none",
                        "Post-fix quality check: result valid — no fix needed.",
                        outcome,
                    )
                    info.repair_records.append(
                        {
                            "round": iteration + 1,
                            "target": {
                                "fix_type": "none",
                                "coarse": "pass",
                                "diagnosis": "post-fix quality check passed",
                            },
                            "changes": {},
                            "outcome": outcome,
                        }
                    )
                    _record_round_summary(results_dir, iteration + 1, "none", "pass", outcome)
                    _append_final_status(results_dir, "pass", info.reflection_attempts)
                    info.reflection_status = "pass"
                    return "pass"

        # --- Loop exhausted ---
        print(f"    [Reflect] Max iterations ({config.MAX_REFLECTION_TIMES}) reached.", flush=True)
        _append_final_status(results_dir, "give_up_max", info.reflection_attempts)
        info.reflection_status = "give_up_max"
        return "give_up_max"
    finally:
        # temp/ STAYS — the attempts are the record (each attempt already
        # lost its results/, tex/, tmp/ inside the probes).  The
        # per-snapshot design PNGs are vision evidence for the reflection
        # loop only — they must not linger in the results folder after
        # the process ends.
        for p in results_dir.glob("snapshot_Th*_label.png"):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
