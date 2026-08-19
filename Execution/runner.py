"""
Run the full execution pipeline — Python → .edp → FreeFEM.

Phase 1 — ``python main.py`` (on host OS, via conda env).
Phase 2 — ``FreeFem++ <file>.edp`` (on Linux directly, on Windows via WSL).
"""

import json
import platform
import time
from pathlib import Path

from .check_tools import check_tools
from .install_python import setup_environment
from .run_autofreefem import run_autofreefem
from .run_freefem_linux import run_freefem as run_freefem_linux
from .run_freefem_windows import run_freefem as run_freefem_windows


def _list_files(workdir: Path) -> list[str]:
    if not workdir.exists():
        return []
    return sorted(p.name for p in workdir.iterdir() if p.is_file())


def _log(title: str) -> None:
    print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")


def run_problem(main_py: Path, workdir: Path = None) -> dict:
    """
    Run *main_py* through AutoFreeFEM then FreeFEM.

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

    import configs as _cfg

    factor = _cfg.FREEFEM_NONLINEAR_FACTOR if is_nonlinear else 1
    effective_total_timeout = _cfg.FREEFEM_TIMEOUT
    effective_step_timeout = _cfg.FREEFEM_STEP_TIMEOUT
    # Only apply nonlinear factor when timeout is enabled (>= 0)
    if effective_total_timeout >= 0:
        effective_total_timeout *= factor
    if effective_step_timeout >= 0:
        effective_step_timeout *= factor

    if edp_files and py_result["return_code"] == "0":
        _log(
            f"Phase 2 — FreeFEM ({len(edp_files)} .edp file(s))"
            + (" [nonlinear x%d]" % factor if factor > 1 else "")
        )

        ff_run = run_freefem_windows if system == "Windows" else run_freefem_linux

        for edp_name in edp_files:
            edp_path = workdir / edp_name
            # Only monitor iteration progress for optimize runs;
            # verify is a single solve that produces no iteration files.
            is_optimize = edp_name.startswith("optimize")
            print(f"  FreeFem++ {edp_name} ...", end=" ", flush=True)
            t0 = time.time()
            ff = ff_run(
                edp_path,
                workdir,
                monitor_progress=is_optimize,
                total_timeout=effective_total_timeout,
                step_timeout=effective_step_timeout,
                expected_epochs=_cfg.MAX_OUTER_ITERATIONS,
            )
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
    # FreeFEM only writes *MinTh.vtu on clean completion.  If it crashed or
    # was killed, the last *Th*.vtu snapshot is the best available result.
    try:
        import glob as _glob

        results_sub = workdir / "results"
        minth_matches = _glob.glob(str(results_sub / "*MinTh.vtu"))
        if not minth_matches:
            th_files = sorted(
                results_sub.glob("result*Th*.vtu"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if th_files:
                latest = th_files[0]
                # Derive MinTh filename: resultFooTh42.vtu → resultFooMinTh.vtu
                import re as _re

                base = latest.stem  # e.g. "resultlinearElasticityTargetTh42"
                minth_name = _re.sub(r"Th\d+$", "MinTh", base) + ".vtu"
                minth_path = latest.with_name(minth_name)
                if not minth_path.exists():
                    import shutil as _shutil

                    _shutil.copy2(latest, minth_path)
    except Exception:
        pass

    # Auto-render MinTh.vtu if it exists, so PNGs stay in sync with the
    # latest FreeFEM output regardless of whether the reflection loop runs.
    try:
        import glob as _glob

        minth_matches = _glob.glob(str(workdir / "results" / "*MinTh.vtu"))
        if minth_matches:
            from Reflection.renderer import render_and_summarize

            render_and_summarize(Path(minth_matches[0]))
    except Exception:
        pass

    return result
