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


def _console_len(workdir: Path) -> int:
    """Size of the console log in bytes (0 when absent) — a cheap
    per-poll activity probe that never reads the file."""
    try:
        return (workdir / "tmp" / "freefem_console.log").stat().st_size
    except OSError:
        return 0


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
    """Kill *proc* and wait briefly; *on_kill* cleans up orphans.

    The parent (FreeFem++) is NOT the only process: its ``system()``
    children (mmg2d_O3 / mmg3d_O3 / advect / mshdist) survive the parent
    kill and keep writing into tmp/ — *on_kill* must sweep them too,
    because the next .edp starts in the same workdir right away.
    """
    proc.kill()
    if on_kill:
        on_kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# Processes whose cwd is the workdir, i.e. OUR run's helpers — killed
# by /proc/<pid>/cwd comparison so a concurrently running problem in
# the same Linux distro (e.g. the batch harness) is left alone.
# ``grep -v "^$$"`` excludes the sweeper shell itself: its cmdline
# contains the pattern text, so pgrep matches it and a sweeper whose
# cwd is the workdir would kill -9 its own pid mid-loop.
_HELPER_SWEEP_SH = (
    "for f in $(pgrep -f 'FreeFem\\+\\+|mmg2d_O3|mmg3d_O3|advect|mshdist' | "
    "grep -v \"^$$\"); do "
    "c=$(readlink -f /proc/$f/cwd 2>/dev/null || true); "
    "[ \"$c\" = \"$1\" ] && kill -9 $f 2>/dev/null || true; done"
)


def sweep_helper_orphans(cwd: str, wsl_mode: bool = False) -> None:
    """Best-effort sweep of orphaned remeshing/FreeFEM helpers in *cwd*.

    Killing only the FreeFem++ parent leaves mmg2d_O3/advect/mshdist
    running; they keep writing tmp/ while the next run reuses it.
    Scoped by /proc/<pid>/cwd, so unrelated runs are never touched.
    *wsl_mode* runs the sweep inside WSL with the Linux-side *cwd*.
    """
    cmd = (
        ["wsl", "-e", "bash", "-c", _HELPER_SWEEP_SH, "sweep", cwd]
        if wsl_mode
        else ["bash", "-c", _HELPER_SWEEP_SH, "sweep", cwd]
    )
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        # A cold WSL distro boot can exceed the probe timeout — the
        # next kill path retries; do not fail the run for it.
        print("  [cleanup] orphan sweep timed out — will retry on next kill.", flush=True)
    except Exception:
        pass  # best-effort cleanup


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
    when a watchdog condition fires (total timeout, step timeout, early
    stagnation, a FreeFEM error code in the console log, or a finished
    optimize loop whose process never exits).  Returns ``None`` when the
    process completes normally.

    *on_kill* is an optional callback invoked after ``proc.kill()`` (used
    on Windows to clean up orphaned WSL processes).
    """
    t_start = time.time()
    t_last_progress = t_start
    t_last_console = t_start
    t_done = None
    last_count = count_iteration_files(results_dir)
    last_console_len = _console_len(workdir)

    while proc.poll() is None:
        time.sleep(10)
        elapsed_total = time.time() - t_start

        # ── total timeout ──────────────────────────────────────────
        if total_enabled and elapsed_total > total_timeout:
            _kill_process(proc, on_kill)
            return {
                "return_code": "-1",
                # Same evidence as every other watchdog outcome: keep the
                # console tail so the reflection loop sees where the run
                # was when the timeout fired.
                "stdout": _console_text(workdir, tail_bytes=200_000),
                "stderr": f"TOTAL TIMEOUT after {total_timeout}s",
            }

        # ── step timeout — check if new iteration files appeared ──
        current_count = count_iteration_files(results_dir)
        if current_count > last_count:
            last_count = current_count
            t_last_progress = time.time()

        # ── console activity — a growing console means the process is
        #    still working even though no iteration file appeared yet
        #    (e.g. the initial Newton solve is long).  Byte-size probe:
        #    the log grows unbounded and must never be read per poll. ──
        console_len = _console_len(workdir)
        if console_len != last_console_len:
            last_console_len = console_len
            t_last_console = time.time()

        # ── success / error detection ─────────────────────────────
        # FreeFem++ does not exit by itself after a successful run; it
        # only reports errors through an error code in its console
        # output.  Three end conditions:
        #   1. the console log contains an error code -> fail now;
        #   2. the final iteration file result<name>Th<max-1>.vtu
        #      exists -> the loop has finished; kill after a short
        #      grace and report a normal completion (rc=0).  Only a
        #      timeout while waiting for new epochs reports rc=-2;
        #   3. otherwise the step timeout below bounds the wait
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
                    # Keep the console tail so the reflection loop sees the
                    # actual error context, not just the error number.
                    "stdout": _console_text(workdir, tail_bytes=200_000),
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
                            _console_text(workdir, tail_bytes=200_000)
                            + "\n[runner] design loop completed; process killed "
                            "after grace (FreeFem++ does not exit on success)\n"
                        ),
                        "stderr": "",
                    }

        # If no iteration files at all after 2× step timeout from start,
        # the process likely has a compile error or early crash — UNLESS
        # its console is still producing output (a long first Newton solve
        # writes Th0 only after it converges).  The deferral is BOUNDED
        # (10× step_timeout): a process that prints forever but produces
        # no file (infinite while/regeneration loop) is still killed, so
        # a disabled total timeout can never leave the monitor looping.
        no_progress = time.time() - t_last_progress
        console_silence = time.time() - t_last_console
        if step_enabled and current_count == 0 and no_progress > step_timeout * 2:
            if no_progress <= step_timeout * 10 and console_silence <= step_timeout:
                continue
            _kill_process(proc, on_kill)
            return {
                "return_code": "-2",
                # Keep the console tail: the reflection loop takes its
                # evidence from run_result.json, and tmp/ is blacklisted
                # from its context.
                "stdout": _console_text(workdir, tail_bytes=200_000),
                "stderr": (
                    f"STARTUP TIMEOUT: no iteration files after "
                    f"{no_progress:.0f}s and no console output for "
                    f"{console_silence:.0f}s.  "
                    f"Likely FreeFEM compile error or early crash."
                ),
            }

        elif step_enabled and current_count > 0 and no_progress > step_timeout:
            # A slow FINAL iteration (last remesh + solve) can exceed the
            # step window while still making real progress.  With at most
            # one iteration file missing, let the final-iteration grace /
            # total timeout bound the wait instead of mislabeling the run
            # as an MMG hang.
            if expected_epochs > 0 and current_count >= expected_epochs - 1:
                continue
            if no_progress <= step_timeout * 10 and console_silence <= step_timeout:
                continue
            _kill_process(proc, on_kill)
            return {
                "return_code": "-2",
                "stdout": _console_text(workdir, tail_bytes=200_000),
                "stderr": (
                    f"STEP TIMEOUT: no new iteration file for "
                    f"{step_timeout}s and no console output for "
                    f"{console_silence:.0f}s "
                    f"(last iteration: {last_count}).  "
                    f"Likely MMG hang or infinite remesh loop."
                ),
            }

    # ── process completed normally ─────────────────────────────────
    return None
