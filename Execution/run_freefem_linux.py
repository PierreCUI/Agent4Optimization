"""Run FreeFEM on Linux — direct execution, with per-iteration progress monitoring."""

import shutil
import subprocess
from pathlib import Path

from configs import FREEFEM_STEP_TIMEOUT
from Execution.runner_shared import monitor_freefem_process, sweep_helper_orphans


def run_freefem(
    edp_file: Path,
    workdir: Path = None,
    monitor_progress: bool = True,
    step_timeout: int = None,
    expected_epochs: int = 0,
) -> dict:
    """Execute ``FreeFem++ <edp_file>`` with optional step-timeout monitoring.

    FreeFem++ console output is logged to ``tmp/freefem_console.log``
    (tail included in the result).  The ONLY run budget is per-step:
    first and last step get 10× *step_timeout*, others 1× — there is no
    overall timeout.

    *step_timeout* overrides the config default (the runner applies the
    nonlinear multiplier).  Set to -1 to disable.

    *expected_epochs* is the target total epochs for the early-stagnation
    watchdog (0 disables the check).
    """
    if step_timeout is None:
        step_timeout = FREEFEM_STEP_TIMEOUT
    step_enabled = step_timeout >= 0

    if workdir is None:
        workdir = edp_file.parent
    if not shutil.which("FreeFem++"):
        return {"return_code": "-1", "stdout": "", "stderr": "FreeFem++ not found."}

    # FreeFem++ output goes to a log file, not a pipe, so its error
    # codes are readable while the process runs.  FreeFem++ does not
    # exit by itself after a successful run; the monitor detects the
    # finished loop (final iteration file) instead.
    log_path = workdir / "tmp" / "freefem_console.log"
    log_path.parent.mkdir(exist_ok=True)

    # ── no progress monitoring — simple subprocess.run ───────────
    # A single-shot run is one "first step" — bound it by the 10× budget.
    if not monitor_progress:
        budget = step_timeout * 10 if step_enabled else None
        with open(log_path, "wb") as log:
            try:
                r = subprocess.run(
                    ["FreeFem++", str(edp_file.name)],
                    cwd=workdir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=budget,
                )
            except subprocess.TimeoutExpired:
                # subprocess.run killed the FreeFem++ parent — its
                # system() children (mmg2d_O3/advect/mshdist) survive and
                # keep writing tmp/ while the next .edp reuses it.
                sweep_helper_orphans(str(workdir.resolve()))
                return {
                    "return_code": "-1",
                    "stdout": "",
                    "stderr": f"TIMEOUT after {budget}s (10× step budget)",
                }
        return {
            "return_code": str(r.returncode),
            "stdout": log_path.read_text(encoding="utf-8", errors="replace")[-8192:],
            "stderr": "",
        }

    # ── progress-monitored execution ────────────────────────────
    results_dir = workdir / "results"

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            ["FreeFem++", str(edp_file.name)],
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        killed = monitor_freefem_process(
            proc,
            results_dir,
            workdir,
            step_enabled=step_enabled,
            step_timeout=step_timeout,
            expected_epochs=expected_epochs,
            # Killing the FreeFem++ parent leaves mmg2d_O3/advect
            # children running and writing tmp/ — sweep them immediately
            # (scoped to this workdir only).
            on_kill=lambda: sweep_helper_orphans(str(workdir.resolve())),
        )
        if killed is not None:
            return killed
        proc.wait()

    return {
        "return_code": str(proc.returncode),
        "stdout": log_path.read_text(encoding="utf-8", errors="replace")[-8192:],
        "stderr": "",
    }
