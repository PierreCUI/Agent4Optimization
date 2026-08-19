"""
Shared FreeFEM runner utilities — imported by both run_freefem_windows.py
and run_freefem_linux.py to avoid duplication.
"""

import subprocess
import time
from pathlib import Path

from configs import EARLY_STAGNATION_EPOCH_FRACTION, EARLY_STAGNATION_MIN_IMPROVEMENT


def check_early_stagnation(workdir: Path, current_epochs: int, expected_epochs: int) -> str | None:
    """Return an error message if early stagnation is detected, else None."""
    if expected_epochs <= 0:
        return None
    if EARLY_STAGNATION_EPOCH_FRACTION < 0 or EARLY_STAGNATION_MIN_IMPROVEMENT < 0:
        return None  # feature disabled via config
    check_at = max(1, int(expected_epochs * EARLY_STAGNATION_EPOCH_FRACTION))
    if current_epochs < check_at:
        return None

    obj_files = sorted(workdir.glob("objective*.txt"))
    if not obj_files:
        return None

    try:
        lines = obj_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    values = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("iteration") or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                values.append(float(parts[1]))
            except ValueError:
                pass

    if len(values) < 2:
        return None

    initial = values[0]
    if initial == 0:
        return None

    best = min(values)
    improvement = (initial - best) / abs(initial)

    if improvement < EARLY_STAGNATION_MIN_IMPROVEMENT:
        return (
            f"EARLY STAGNATION: at epoch {current_epochs}/{expected_epochs} "
            f"(>= {check_at}), objective improved only "
            f"{improvement * 100:.1f}% "
            f"(threshold: {EARLY_STAGNATION_MIN_IMPROVEMENT * 100:.0f}%).  "
            f"Initial={initial:.6g}, best={best:.6g}"
        )
    return None


def count_iteration_files(results_dir: Path) -> int:
    """Count iteration output files in the results sub-directory."""
    if not results_dir.exists():
        return 0
    return len(list(results_dir.glob("result*Th*.vtu")))


def monitor_freefem_process(
    proc: subprocess.Popen,
    results_dir: Path,
    workdir: Path,
    total_enabled: bool,
    total_timeout: int,
    step_enabled: bool,
    step_timeout: int,
    expected_epochs: int = 0,
    on_kill=None,
) -> dict | None:
    """
    Monitor a running FreeFEM process with timeout and stagnation checks.

    Polls the process every 10 seconds.  Kills it and returns a result dict
    when a watchdog condition fires (total timeout, step timeout, or early
    stagnation).  Returns ``None`` when the process completes normally.

    *on_kill* is an optional callback invoked after ``proc.kill()`` (used
    on Windows to clean up orphaned WSL processes).
    """
    t_start = time.time()
    t_last_progress = t_start
    last_count = count_iteration_files(results_dir)

    while proc.poll() is None:
        time.sleep(10)
        elapsed_total = time.time() - t_start

        # ── total timeout ──────────────────────────────────────────
        if total_enabled and elapsed_total > total_timeout:
            proc.kill()
            if on_kill:
                on_kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {
                "return_code": "-1",
                "stdout": "",
                "stderr": f"TOTAL TIMEOUT after {total_timeout}s",
            }

        # ── step timeout — check if new iteration files appeared ──
        current_count = count_iteration_files(results_dir)
        if current_count > last_count:
            last_count = current_count
            t_last_progress = time.time()

            # ── early stagnation check ────────────────────────────
            if expected_epochs > 0 and current_count >= 2:
                msg = check_early_stagnation(
                    workdir,
                    current_count,
                    expected_epochs,
                )
                if msg is not None:
                    proc.kill()
                    if on_kill:
                        on_kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    return {"return_code": "-3", "stdout": "", "stderr": msg}

        # If no iteration files at all after 2× step timeout from start,
        # the process likely has a compile error or early crash.
        no_progress = time.time() - t_last_progress
        if step_enabled and current_count == 0 and no_progress > step_timeout * 2:
            proc.kill()
            if on_kill:
                on_kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {
                "return_code": "-2",
                "stdout": "",
                "stderr": (
                    f"STARTUP TIMEOUT: no iteration files after "
                    f"{no_progress:.0f}s.  "
                    f"Likely FreeFEM compile error or early crash."
                ),
            }

        elif step_enabled and current_count > 0 and no_progress > step_timeout:
            proc.kill()
            if on_kill:
                on_kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {
                "return_code": "-2",
                "stdout": "",
                "stderr": (
                    f"STEP TIMEOUT: no new iteration file for "
                    f"{step_timeout}s "
                    f"(last iteration: {last_count}).  "
                    f"Likely MMG hang or infinite remesh loop."
                ),
            }

    # ── process completed normally ─────────────────────────────────
    return None
