"""Run FreeFEM on Windows — delegates to WSL, with per-iteration progress monitoring."""

import json
import shutil
import subprocess
from pathlib import Path

from configs import FREEFEM_STEP_TIMEOUT, FREEFEM_TIMEOUT
from Execution.runner_shared import monitor_freefem_process


def _wsl_path(path: Path) -> str:
    r"""``C:\...\file.edp`` → ``/mnt/c/.../file.edp``."""
    p = str(path.resolve())
    if len(p) > 2 and p[1:3] == ":\\":
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")


def _kill_wsl_orphans() -> None:
    """Send SIGTERM to any orphaned FreeFEM processes inside WSL.

    ``proc.kill()`` on a ``wsl.exe`` handle only kills the WSL wrapper,
    not the child processes inside the Linux VM.  This cleans up.
    """
    try:
        subprocess.run(
            ["wsl", "-e", "pkill", "-f", "FreeFem++"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass  # best-effort cleanup


def run_freefem(
    edp_file: Path,
    workdir: Path = None,
    monitor_progress: bool = True,
    total_timeout: int = None,
    step_timeout: int = None,
    expected_epochs: int = 0,
) -> dict:
    """Execute ``FreeFem++ <edp_file>`` inside WSL.

    When *monitor_progress* is True (the default), the results/ sub-directory
    is polled and the process is killed with rc=-2 if no new iteration file
    appears for *step_timeout* seconds.  Set to False for single-shot runs
    like verification that do not produce iteration files.

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

    script = (
        f"import subprocess, json, os; os.chdir('{wd_wsl}'); "
        f"r = subprocess.run(['FreeFem++', '{edp_wsl}'], "
        f"capture_output=True, text=True); "
        f"print('RESULT:' + json.dumps({{'return_code': str(r.returncode), "
        f"'stdout': r.stdout, 'stderr': r.stderr}}))"
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
            _kill_wsl_orphans()
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

    killed = monitor_freefem_process(
        proc,
        results_dir,
        workdir,
        total_enabled=total_enabled,
        total_timeout=total_timeout,
        step_enabled=step_enabled,
        step_timeout=step_timeout,
        expected_epochs=expected_epochs,
        on_kill=_kill_wsl_orphans,
    )
    if killed is not None:
        return killed

    # ── process completed normally ─────────────────────────────────
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _kill_wsl_orphans()
        stdout, stderr = proc.communicate()

    output = (stdout or "") + "\n" + (stderr or "")

    for line in output.splitlines():
        if line.startswith("RESULT:"):
            try:
                return json.loads(line[7:])
            except json.JSONDecodeError:
                pass

    return {"return_code": str(proc.returncode), "stdout": output, "stderr": ""}
