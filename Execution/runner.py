"""
Run the full execution pipeline — Python → .edp → FreeFEM.

Phase 1 — ``python main.py`` (on host OS, via conda env).
Phase 2 — ``FreeFem++ <file>.edp`` (on Linux directly, on Windows via WSL).
"""

import glob
import json
import platform
import re
import shutil
import time
from pathlib import Path

import configs as _cfg

from utils import code_masked_positions

from .check_tools import check_tools
from .install_python import setup_environment
from .run_autofreefem import run_autofreefem
from .run_freefem_linux import run_freefem as run_freefem_linux
from .run_freefem_windows import run_freefem as run_freefem_windows


def _list_files(workdir: Path) -> list[str]:
    if not workdir.exists():
        return []
    return sorted(p.name for p in workdir.iterdir() if p.is_file())


def _extract_iterations(code_text: str) -> int:
    """Get the outer iteration count from the generated main.py.

    Scoped to the ``setup_optimization(...)`` arguments (paren-balanced,
    quote-aware) so a comment like ``# iterations = 50`` or a
    ``inner_iterations=`` kwarg elsewhere in the file cannot be matched
    instead.  Returns 0 when no usable value is found (caller falls back
    to the config default).
    """
    # LAST balanced REAL call: mentions in comments/docstrings (the
    # coder prompt's example shape appears in trailing comments, and a
    # balanced span there would hijack the count) are skipped via the
    # code mask; an unbalanced mention is skipped by the paren scan.
    masked = code_masked_positions(code_text)
    last = None
    for m in re.finditer(r"setup_optimization\s*\(", code_text):
        if m.start() in masked:
            continue
        start = m.end()
        depth = 1
        i = start
        quote = None
        while i < len(code_text) and depth > 0:
            ch = code_text[i]
            if quote:
                if ch == quote and code_text[i - 1] != "\\":
                    quote = None
            elif ch in ("'", '"'):
                quote = ch
            elif ch == "#":
                # line comment — skip to end of line: an apostrophe inside
                # a comment must not enter string mode and swallow the
                # real closing parenthesis
                j = code_text.find("\n", i)
                i = len(code_text) if j == -1 else j
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            last = (start, i)
    if last is None:
        return 0
    call_text = code_text[last[0] : last[1] - 1]
    # Word boundary: ``iterations=50`` must not match the tail of
    # ``inner_iterations=5``.
    m2 = re.search(r"(?<![A-Za-z_])iterations\s*=\s*(\d+)", call_text)
    return int(m2.group(1)) if m2 else 0


def _log(title: str) -> None:
    print(f"\n{'-' * 50}\n  {title}\n{'-' * 50}")


def _patch_newton_tolerance(edp_path: Path) -> bool:
    """Patch the generated Newton loop's convergence criterion.

    AutoFreeFEM's stock criterion demands BOTH an absolute residual
    below 1e-6 AND an increment below 1e-5.  For penalty-Dirichlet
    problems (pen = 1e15) the residual plateaus around roundoff of the
    penalty term (~1e0) while the increment already reaches ~1e-13, so
    the loop burns all 40 Newton steps on every shape iteration without
    improving the state.  A vanishing increment means the state no
    longer changes — treat it as convergence.
    """
    try:
        text = edp_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    old = "(rhs.linfty < 1e-6)*(sol.linfty < 1e-5)"
    new = "(rhs.linfty < 1e-6)*(sol.linfty < 1e-5) || (sol.linfty < 1e-8)"
    if old not in text:
        return False
    try:
        edp_path.write_text(text.replace(old, new), encoding="utf-8")
        return True
    except OSError:
        return False


def run_problem(main_py: Path, workdir: Path = None, auto_render: bool = True) -> dict:
    """
    Run *main_py* through AutoFreeFEM then FreeFEM.

    *auto_render* controls the trailing MinTh.vtu -> PNG render.  The
    parallel reflection probes pass False and render their own final
    design under a lock (matplotlib keeps process-global state, so N
    concurrent auto-renders are both redundant and risky).

    Returns::

        {
          "autofreefem": {
            "python": {return_code, stdout, stderr},
            "files_before": [...], "files_after": [...], "edp_files": [...]
          },
          "freefem": [
            {edp_file, return_code, stdout, stderr}, ...
          ]
        }
    """
    if workdir is None:
        workdir = main_py.parent

    system = platform.system()
    print(f"[run] Platform: {system}")

    # ── C++ tool check ─────────────────────────────────────────────────
    check_tools()

    # ── Python environment ─────────────────────────────────────────────
    python_exe = setup_environment()

    # ── Phase 1: AutoFreeFEM ───────────────────────────────────────────
    files_before = _list_files(workdir)
    _log(f"Phase 1 — AutoFreeFEM: {main_py.name}")
    py_result = run_autofreefem(main_py.resolve(), workdir, python_exe)

    files_after = _list_files(workdir)
    new_files = [f for f in files_after if f not in files_before]
    # Find .edp from files_after (not new_files) because run_autofreefem
    # cleans stale .edp then recreates the same filenames — diff misses them.
    edp_files = [f for f in files_after if f.endswith(".edp")]

    print(f"  Python exit: {py_result['return_code']}")
    print(f"  New files:   {new_files}")
    print(f"  .edp files:  {edp_files}")

    # ── Phase 2: FreeFEM ───────────────────────────────────────────────
    freefem_records = []

    # Detect nonlinear: read initial.json if available
    is_nonlinear = False
    try:
        initial_path = workdir / "initial.json"
        if initial_path.exists():
            spec = json.loads(initial_path.read_text(encoding="utf-8"))
            is_nonlinear = spec.get("designed", {}).get("state_is_nonlinear", False)
    except Exception:
        pass

    factor = _cfg.FREEFEM_NONLINEAR_FACTOR if is_nonlinear else 1
    effective_total_timeout = _cfg.FREEFEM_TIMEOUT
    effective_step_timeout = _cfg.FREEFEM_STEP_TIMEOUT
    # Only apply nonlinear factor when timeout is enabled (>= 0)
    if effective_total_timeout >= 0:
        effective_total_timeout *= factor
    if effective_step_timeout >= 0:
        effective_step_timeout *= factor

    # Clean results/<problem>/results before EVERY run — stale iteration
    # files, MinTh.vtu, or PNGs from a previous run must not be picked
    # up as evidence when THIS run never reached FreeFEM (e.g. the
    # AutoFreeFEM Python phase failed before generating any .edp).
    results_sub = workdir / "results"
    if results_sub.exists():
        if results_sub.is_dir():
            try:
                shutil.rmtree(results_sub)
            except OSError:
                pass
        else:
            # results is a FILE (copy mistake / leftover artifact) —
            # clear it too, or the mkdir below raises FileExistsError.
            try:
                results_sub.unlink()
            except OSError:
                pass
    results_sub.mkdir(parents=True, exist_ok=True)

    # The epoch count is LLM-generated (setup_optimization(iterations=N));
    # derive it from the generated code so the progress watchdog matches
    # the actual run, falling back to the config default.
    expected_epochs = _cfg.MAX_OUTER_ITERATIONS
    try:
        mp_text = main_py.read_text(encoding="utf-8", errors="replace")
        found = _extract_iterations(mp_text)
        if found:
            expected_epochs = found
    except Exception:
        pass

    if edp_files and py_result["return_code"] == "0":
        _log(
            f"Phase 2 — FreeFEM ({len(edp_files)} .edp file(s))"
            + (" [nonlinear x%d]" % factor if factor > 1 else "")
        )

        ff_run = run_freefem_windows if system == "Windows" else run_freefem_linux

        for edp_name in edp_files:
            edp_path = workdir / edp_name
            # Newton-loop tolerance patch (penalty-Dirichlet problems
            # otherwise burn all 40 Newton steps per shape iteration) —
            # applied to the generated edp before every FreeFEM launch.
            if _patch_newton_tolerance(edp_path):
                print(f"  [patch] {edp_name}: Newton tolerance patched")
            # Only monitor iteration progress for optimize runs;
            # verify is a single solve that produces no iteration files.
            is_optimize = edp_name.startswith("optimize")
            print(f"  FreeFem++ {edp_name} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                ff = ff_run(
                    edp_path,
                    workdir,
                    monitor_progress=is_optimize,
                    total_timeout=effective_total_timeout,
                    step_timeout=effective_step_timeout,
                    expected_epochs=expected_epochs,
                )
            except Exception as e:
                # An unexpected launcher failure must not leave the old
                # run_result.json on disk after results/ was wiped.
                ff = {
                    "edp_file": edp_name,
                    "return_code": "-1",
                    "stdout": "",
                    "stderr": f"launcher error: {e}",
                }
            dt = time.time() - t0
            print(f"exit={ff['return_code']} ({dt:.1f}s)")
            if ff.get("stderr"):
                for line in ff["stderr"].strip().splitlines()[:5]:
                    print(f"    [stderr] {line}")
            freefem_records.append(
                {
                    "edp_file": edp_name,
                    "return_code": ff["return_code"],
                    "stdout": ff["stdout"],
                    "stderr": ff["stderr"],
                }
            )

    result = {
        "autofreefem": {
            "python": py_result,
            "files_before": files_before,
            "files_after": files_after,
            "edp_files": edp_files,
        },
        "freefem": freefem_records,
    }

    # Persist to disk so callers (main pipeline, test scripts, reflection)
    # always have an up-to-date run_result.json.
    try:
        with open(workdir / "run_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    except OSError:
        pass

    # ── Fallback: if MinTh.vtu was not produced, use the last iteration ──
    # FreeFEM writes *MinTh.vtu whenever the objective improves; a run that
    # crashed before any improvement lacks it, and the latest *Th*.vtu
    # snapshot is then the best available result.
    try:
        results_sub = workdir / "results"
        minth_matches = glob.glob(str(results_sub / "*MinTh.vtu"))
        if not minth_matches:
            th_files = sorted(
                results_sub.glob("result*Th*.vtu"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if th_files:
                latest = th_files[0]
                # Derive MinTh filename: resultFooTh42.vtu → resultFooMinTh.vtu
                base = latest.stem  # e.g. "resultlinearElasticityTargetTh42"
                minth_name = re.sub(r"Th\d+$", "MinTh", base) + ".vtu"
                minth_path = latest.with_name(minth_name)
                if not minth_path.exists():
                    shutil.copy2(latest, minth_path)
    except Exception:
        pass

    # Auto-render MinTh.vtu if it exists, so PNGs stay in sync with the
    # latest FreeFEM output regardless of whether the reflection loop runs.
    if auto_render:
        try:
            minth_matches = glob.glob(str(workdir / "results" / "*MinTh.vtu"))
            if minth_matches:
                from Reflection.renderer import render_and_summarize

                render_and_summarize(Path(minth_matches[0]))
        except Exception:
            pass

    return result
