"""
Shared FreeFEM runner utilities — imported by both run_freefem_windows.py
and run_freefem_linux.py to avoid duplication.
"""

import os
import re
import subprocess
import time
from pathlib import Path

from configs import FREEFEM_DONE_GRACE


def count_iteration_files(results_dir: Path) -> int:
    """Count *numeric* iteration snapshot files in the results dir.

    ``result*Th*.vtu`` also matches ``...MinTh.vtu`` (the final design's
    filename ends in the same "Th.vtu" suffix), which would inflate the
    count by one — require an actual iteration number ``Th<N>.vtu``.
    """
    if not results_dir.exists():
        return 0
    n = 0
    for p in results_dir.glob("result*Th*.vtu"):
        if re.search(r"Th(\d+)\.vtu$", p.name):
            n += 1
    return n


_ERR_RE = re.compile(r"err\s*code\s*(\d+)|code\s*=\s*(\d+)", re.IGNORECASE)


_CONSOLE_EVIDENCE_CHARS = 8192


def _console_evidence(workdir: Path) -> str:
    """Last ``_CONSOLE_EVIDENCE_CHARS`` characters of the FreeFem++ console log.

    This is the console evidence recorded for DOWNSTREAM consumers —
    run_result.json is the only channel (tmp/ is blacklisted from the
    reflection context), so it must stay bounded: the log grows
    unbounded on verbose runs and an unbounded tail would drown every
    other signal in the record.
    """
    # Read the last 4×chars bytes: 8192 chars of worst-case 4-byte UTF-8
    # (invalid bytes decode to one replacement char each, which is
    # shorter — so the 8192-char suffix is always inside this window).
    window = _CONSOLE_EVIDENCE_CHARS * 4
    log_path = workdir / "tmp" / "freefem_console.log"
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(size - window if size > window else 0)
            data = f.read(window)
        return data.decode("utf-8", errors="replace")[-_CONSOLE_EVIDENCE_CHARS:]
    except OSError:
        return ""


def _console_text(workdir: Path, tail_bytes: int = None) -> str:
    """Return the FreeFem++ console log content (or its last
    *tail_bytes* bytes), or '' when absent."""
    log_path = workdir / "tmp" / "freefem_console.log"
    if not log_path.exists():
        return ""
    try:
        if tail_bytes is not None:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # seek to the START of the tail — a small file must be
                # read from byte 0, not from the end.
                f.seek(size - tail_bytes if size > tail_bytes else 0)
                data = f.read(tail_bytes)
            return data.decode("utf-8", errors="replace")
        return log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _console_error(workdir: Path) -> int | None:
    """Return the error number printed by FreeFem++ in its console log,
    or None when the run shows no error.

    Scans only the LAST 256 KB: an err code is printed right before the
    run stops, and the log grows unbounded on verbose runs — never
    regex-scan the whole file per poll.
    """
    m = _ERR_RE.search(_console_text(workdir, tail_bytes=256 * 1024))
    if m is None:
        return None
    return int(m.group(1) or m.group(2))


def _final_iteration_reached(results_dir: Path, expected_epochs: int) -> bool:
    """Return True when the last iteration file ``result<name>Th<max-1>.vtu``
    has been written — the design loop has finished."""
    best = 0
    for p in results_dir.glob("result*Th*.vtu"):
        m = re.search(r"Th(\d+)\.vtu$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best >= expected_epochs - 1


def _kill_process(proc: subprocess.Popen, on_kill=None) -> None:
    """Kill *proc* and SYNCHRONOUSLY clean up its helper children.

    The parent (FreeFem++) is NOT the only process: its ``system()``
    children (mmg2d_O3 / advect / mshdist) survive the parent
    kill and keep writing into tmp/ — *on_kill* sweeps them (the sweep
    itself retries until nothing is left) BEFORE this returns, because
    the next .edp starts in the same workdir right away.
    """
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if on_kill:
        on_kill()


# Processes whose cwd is the workdir, i.e. OUR run's helpers — killed
# by /proc/<pid>/cwd comparison so a concurrently running problem in
# the same Linux distro (e.g. the batch harness) is left alone, and so
# OTHER processes and the run's PARENT processes (pipeline python, the
# WSL launcher) are never touched: the pattern names only the four tool
# binaries.
# ``grep -v "^$$"`` excludes the sweeper shell itself: its cmdline
# contains the pattern text, so pgrep matches it and a sweeper whose
# cwd is the workdir would kill -9 its own pid mid-loop.
# The loop retries until a pass finds nothing left to kill — the sweep
# is SYNCHRONOUS: when it returns, no surviving helper is still
# writing tmp/.
_HELPER_SWEEP_SH = (
    "for attempt in 1 2 3; do "
    "hit=0; "
    "for f in $(pgrep -f 'FreeFem\\+\\+|mmg2d_O3|advect|mshdist' | "
    "grep -v \"^$$\"); do "
    "c=$(readlink -f /proc/$f/cwd 2>/dev/null || true); "
    "[ \"$c\" = \"$1\" ] && { kill -9 $f 2>/dev/null || true; hit=1; }; "
    "done; "
    "[ \"$hit\" = \"0\" ] && break; "
    "sleep 2; "
    "done"
)


def sweep_helper_orphans(cwd: str, wsl_mode: bool = False) -> None:
    """Synchronous sweep of orphaned remeshing/FreeFEM helpers in *cwd*.

    Killing only the FreeFem++ parent leaves mmg2d_O3/advect/mshdist
    running; they keep writing tmp/ while the next run reuses it.
    Scoped by /proc/<pid>/cwd, so unrelated runs are never touched;
    the sweep script retries internally until a pass finds nothing
    left to kill.  *wsl_mode* runs the sweep inside WSL with the
    Linux-side *cwd*.
    """
    cmd = (
        ["wsl", "-e", "bash", "-c", _HELPER_SWEEP_SH, "sweep", cwd]
        if wsl_mode
        else ["bash", "-c", _HELPER_SWEEP_SH, "sweep", cwd]
    )
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        # A cold WSL distro boot can exceed the sweep's 30s timeout — the
        # next kill path retries; do not fail the run for it.
        print("  [cleanup] orphan sweep timed out — will retry on next kill.", flush=True)
    except Exception:
        pass  # best-effort cleanup


def monitor_freefem_process(
    proc: subprocess.Popen,
    results_dir: Path,
    workdir: Path,
    step_enabled: bool,
    step_timeout: int,
    expected_epochs: int = 0,
    on_kill=None,
) -> dict | None:
    """
    Monitor a running FreeFEM process with per-step stagnation checks.

    Polls the process every 10 seconds.  There is NO overall run budget —
    only per-step budgets: the first step (no iteration file yet, the
    initial Newton solve can be long) and the last step (final remesh +
    solve) get 10× step_timeout, everything in between gets 1×.  Kills
    the process and returns a result dict when a watchdog condition
    fires (step stagnation, a FreeFEM error code in the console log, or
    a finished optimize loop whose process never exits).  Returns
    ``None`` when the process completes normally.

    *on_kill* is an optional callback invoked after ``proc.kill()`` (used
    on Windows to clean up orphaned WSL processes).
    """
    t_start = time.time()
    t_last_progress = t_start
    t_done = None
    last_count = count_iteration_files(results_dir)

    while proc.poll() is None:
        time.sleep(10)

        # ── step timeout — check if new iteration files appeared ──
        current_count = count_iteration_files(results_dir)
        if current_count > last_count:
            last_count = current_count
            t_last_progress = time.time()

        # ── success / error detection ─────────────────────────────
        # FreeFem++ does not exit by itself after a successful run; it
        # only reports errors through an error code in its console
        # output.  A NORMAL completion prints NO err code at all (not
        # even "0") — so ANY err code means the run did NOT end
        # normally: record it (console_err_code feeds the downstream
        # judgment/repair loop) and fail with it.  End conditions:
        #   1. console err code -> fail now with rc=code (recorded);
        #   2. final iteration file result<name>Th<max-1>.vtu exists ->
        #      kill after a short grace, rc=0 (normal completion);
        #   3. otherwise the per-step budget below bounds the wait
        #      (FREEFEM_STEP_TIMEOUT, multiplied by
        #      FREEFEM_NONLINEAR_FACTOR for nonlinear problems).
        if step_enabled:
            # FreeFem++ reports errors ONLY through a code in its console
            # log — detect it as soon as it appears, independently of
            # whether iteration files are expected (else a real err code
            # is masked by a later rc=-2 watchdog kill and reflection
            # fixes the wrong thing).
            err = _console_error(workdir)
            if err is not None:
                _kill_process(proc, on_kill)
                return {
                    "return_code": str(err),
                    "console_err_code": err,
                    # Keep the console tail so the reflection loop sees the
                    # actual error context, not just the error number.
                    "stdout": _console_evidence(workdir),
                    "stderr": f"FreeFEM error in console output: err code {err}",
                }

            if expected_epochs > 0 and _final_iteration_reached(results_dir, expected_epochs):
                if t_done is None:
                    t_done = time.time()
                if time.time() - t_done > FREEFEM_DONE_GRACE:
                    _kill_process(proc, on_kill)
                    return {
                        "return_code": "0",
                        "stdout": (
                            _console_evidence(workdir)
                            + "\n[runner] design loop completed; process killed "
                            "after grace (FreeFem++ does not exit on success)\n"
                        ),
                        "stderr": "",
                    }

            # ── step stagnation ────────────────────────────────────
            # Per-step no-progress budget: the FIRST step (no iteration
            # file yet — the initial Newton solve can be long) and the
            # LAST step (final remesh + solve) get 10× step_timeout,
            # everything in between gets 1×.  This is the ONLY run
            # budget — there is no overall timeout.
            no_progress = time.time() - t_last_progress
            first = current_count == 0
            last = expected_epochs > 0 and current_count >= expected_epochs - 1
            budget = step_timeout * (10 if (first or last) else 1)
            if no_progress > budget:
                _kill_process(proc, on_kill)
                cause = (
                    "FreeFEM compile error or early crash"
                    if first else "MMG hang or infinite remesh loop"
                )
                return {
                    "return_code": "-2",
                    # Keep the console tail: the reflection loop takes its
                    # evidence from run_result.json, and tmp/ is blacklisted
                    # from its context.
                    "stdout": _console_evidence(workdir),
                    "stderr": (
                        f"{'STARTUP' if first else 'STEP'} TIMEOUT: no new "
                        f"iteration file for {no_progress:.0f}s "
                        f"(budget {budget}s, last iteration: {last_count}).  "
                        f"Likely {cause}."
                    ),
                }

    # ── process completed normally ─────────────────────────────────
    return None
