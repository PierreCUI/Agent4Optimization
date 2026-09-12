"""Run FreeFEM on Windows — delegates to WSL, with per-iteration progress monitoring."""

import json
import shutil
import subprocess
import threading
from pathlib import Path

from configs import FREEFEM_STEP_TIMEOUT, FREEFEM_TIMEOUT
from Execution.runner_shared import monitor_freefem_process, sweep_helper_orphans


def _wsl_path(path: Path) -> str:
    r"""``C:\...\file.edp`` → ``/mnt/c/.../file.edp``."""
    p = str(path.resolve())
    # Windows returns the extended-length form \\?\C:\... for long paths
    # (Path.resolve does this past MAX_PATH, e.g. deep OneDrive trees) —
    # strip the prefix so the drive-letter mapping applies.
    if p.startswith("\\\\?\\"):
        p = p[4:]
    if len(p) > 2 and p[1:3] == ":\\":
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    # UNC paths (\\server\share\...) have no automatic WSL mount — pass
    # the forward-slash form through; the run will fail loudly inside WSL
    # if the share is not mounted rather than silently writing nowhere.
    return p.replace("\\", "/")


def _kill_wsl_orphans(workdir: Path) -> None:
    """Sweep orphaned FreeFEM processes inside WSL, scoped to *workdir*.

    ``proc.kill()`` on a ``wsl.exe`` handle only kills the WSL wrapper,
    not the children inside the Linux VM.  FreeFen++'s own ``system()``
    helpers (mmg2d_O3 / mmg3d_O3 / advect / mshdist) survive even a
    FreeFem++ kill and keep writing tmp/ into the workdir the next
    .edp run immediately reuses — sweep by /proc/<pid>/cwd so only this
    run's pid tree is touched.
    """
    sweep_helper_orphans(_wsl_path(workdir), wsl_mode=True)


def run_freefem(
    edp_file: Path,
    workdir: Path = None,
    monitor_progress: bool = True,
    total_timeout: int = None,
    step_timeout: int = None,
    expected_epochs: int = 0,
) -> dict:
    """Execute ``FreeFem++ <edp_file>`` inside WSL.

    FreeFem++ console output is logged to ``tmp/freefem_console.log``
    (tail included in the result), so the exit code is reported the moment
    FreeFem++ terminates.

    When *monitor_progress* is True (the default), the results/
    sub-directory and console log are polled: a FreeFEM error code in the
    log ends the run immediately, the final iteration file ends the run
    after a short grace (FreeFem++ does not exit by itself after a
    successful run), and otherwise the run is killed with rc=-2 if no new
    iteration file appears for *step_timeout* seconds.  Set to False for
    single-shot runs like verification that do not produce iteration
    files.

    *total_timeout* and *step_timeout* override the config defaults (used
    to apply nonlinear multipliers).
    """
    if total_timeout is None:
        total_timeout = FREEFEM_TIMEOUT
    if step_timeout is None:
        step_timeout = FREEFEM_STEP_TIMEOUT
    total_enabled = total_timeout >= 0
    step_enabled = step_timeout >= 0

    if workdir is None:
        workdir = edp_file.parent
    if shutil.which("wsl") is None:
        return {"return_code": "-1", "stdout": "", "stderr": "WSL not found."}

    edp_wsl = _wsl_path(edp_file)
    wd_wsl = _wsl_path(workdir)

    # FreeFem++ output goes to a log file, not a pipe, so its error
    # codes are readable while the process runs.  FreeFem++ does not
    # exit by itself after a successful run; the monitor detects the
    # finished loop (final iteration file) instead.
    script = (
        f"import subprocess, json, os; os.chdir('{wd_wsl}'); "
        # Make advect (built from ISCDtoolbox/Advection) findable by
        # FreeFEM's system() calls — it installs to ~/bin with a shared
        # lib in ~/lib.  Without this the advection step exits with
        # 'command not found' and the level set never updates (the same
        # injection the batches/*/wsl_monitor.py harness performs).
        f"_b = os.path.expanduser('~/bin'); "
        f"os.environ['PATH'] = _b + ':' + os.environ['PATH'] if os.path.isdir(_b) else os.environ['PATH']; "
        f"_l = os.path.expanduser('~/lib'); "
        f"os.environ['LD_LIBRARY_PATH'] = _l + ':' + os.environ.get('LD_LIBRARY_PATH', '') if os.path.isdir(_l) else os.environ.get('LD_LIBRARY_PATH', ''); "
        f"os.makedirs('tmp', exist_ok=True); "
        f"log = open('tmp/freefem_console.log', 'w'); "
        f"r = subprocess.run(['FreeFem++', '{edp_wsl}'], "
        f"stdout=log, stderr=subprocess.STDOUT); log.close(); "
        f"tail = open('tmp/freefem_console.log', encoding='utf-8', "
        f"errors='replace').read()[-200000:]; "
        f"print('RESULT:' + json.dumps({{'return_code': str(r.returncode), "
        f"'stdout': tail, 'stderr': ''}}))"
    )

    # ── if progress monitoring is disabled, use simple subprocess.run ──
    if not monitor_progress:
        try:
            r = subprocess.run(
                ["wsl", "python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=total_timeout if total_enabled else None,
            )
        except subprocess.TimeoutExpired:
            _kill_wsl_orphans(workdir)
            return {
                "return_code": "-1",
                "stdout": "",
                "stderr": f"TIMEOUT after {total_timeout}s",
            }
        output = (r.stdout or "") + "\n" + (r.stderr or "")
        for line in output.splitlines():
            if line.startswith("RESULT:"):
                try:
                    return json.loads(line[7:])
                except json.JSONDecodeError:
                    pass
        return {"return_code": str(r.returncode), "stdout": output, "stderr": ""}

    # ── progress-monitored execution ────────────────────────────
    results_dir = workdir / "results"

    proc = subprocess.Popen(
        ["wsl", "python3", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # The launcher prints one RESULT: line that may carry a ~200 KB
    # console tail — nobody reads the pipe while the monitor polls, so
    # an undrained stdout=PIPE would block the launcher's final write
    # and its real exit code would never be observable.  Drain both
    # pipes on side threads; a watchdog-kill return path just drops the
    # daemon readers with the process.
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def _drain(stream, sink):
        try:
            for line in stream:
                sink.append(line)
        except Exception:
            pass

    threading.Thread(target=_drain, args=(proc.stdout, stdout_parts), daemon=True).start()
    threading.Thread(target=_drain, args=(proc.stderr, stderr_parts), daemon=True).start()

    killed = monitor_freefem_process(
        proc,
        results_dir,
        workdir,
        total_enabled=total_enabled,
        total_timeout=total_timeout,
        step_enabled=step_enabled,
        step_timeout=step_timeout,
        expected_epochs=expected_epochs,
        on_kill=lambda: _kill_wsl_orphans(workdir),
    )
    if killed is not None:
        return killed

    # ── process completed normally ─────────────────────────────────
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _kill_wsl_orphans(workdir)
        proc.wait()
    stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)

    output = (stdout or "") + "\n" + (stderr or "")

    for line in output.splitlines():
        if line.startswith("RESULT:"):
            try:
                return json.loads(line[7:])
            except json.JSONDecodeError:
                pass

    return {"return_code": str(proc.returncode), "stdout": output, "stderr": ""}
