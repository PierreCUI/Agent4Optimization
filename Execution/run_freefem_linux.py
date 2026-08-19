"""Run FreeFEM on Linux — direct execution, with per-iteration progress monitoring."""

import shutil
import subprocess
from pathlib import Path

from configs import FREEFEM_STEP_TIMEOUT, FREEFEM_TIMEOUT
from Execution.runner_shared import monitor_freefem_process


def run_freefem(
    edp_file: Path,
    workdir: Path = None,
    monitor_progress: bool = True,
    total_timeout: int = None,
    step_timeout: int = None,
    expected_epochs: int = 0,
) -> dict:
    """Execute ``FreeFem++ <edp_file>`` with optional step-timeout monitoring.

    *total_timeout* and *step_timeout* override config defaults (used for
    nonlinear-problem timeout multipliers).  Set to -1 to disable.

    *expected_epochs* is the target total epochs for the early-stagnation
    watchdog (0 disables the check).
    """
    if total_timeout is None:
        total_timeout = FREEFEM_TIMEOUT
    if step_timeout is None:
        step_timeout = FREEFEM_STEP_TIMEOUT
    total_enabled = total_timeout >= 0
    step_enabled = step_timeout >= 0

    if workdir is None:
        workdir = edp_file.parent
    if not shutil.which("FreeFem++"):
        return {"return_code": "-1", "stdout": "", "stderr": "FreeFem++ not found."}

    # ── no progress monitoring — simple subprocess.run ───────────
    if not monitor_progress:
        try:
            r = subprocess.run(
                ["FreeFem++", str(edp_file.name)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=total_timeout if total_enabled else None,
            )
        except subprocess.TimeoutExpired:
            return {
                "return_code": "-1",
                "stdout": "",
                "stderr": f"TIMEOUT after {total_timeout}s",
            }
        return {
            "return_code": str(r.returncode),
            "stdout": r.stdout or "",
            "stderr": r.stderr or "",
        }

    # ── progress-monitored execution ────────────────────────────
    results_dir = workdir / "results"

    proc = subprocess.Popen(
        ["FreeFem++", str(edp_file.name)],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    killed = monitor_freefem_process(
        proc,
        results_dir,
        workdir,
        total_enabled=total_enabled,
        total_timeout=total_timeout,
        step_enabled=step_enabled,
        step_timeout=step_timeout,
        expected_epochs=expected_epochs,
    )
    if killed is not None:
        return killed

    # ── process completed normally ─────────────────────────────────
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    return {
        "return_code": str(proc.returncode),
        "stdout": stdout or "",
        "stderr": stderr or "",
    }
