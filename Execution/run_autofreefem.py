"""
Run AutoFreeFEM Python code on the host system.

If autofreefem is missing an ``include/*.idp`` file, this module downloads
it from the upstream GitLab repository before retrying.
"""

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import AUTOFREEFEM_MAX_RETRIES, AUTOFREEFEM_TIMEOUT, GITLAB_RAW_URL


def _fetch_missing_idp(stderr: str, python_exe: str) -> bool:
    m = re.search(r"autofreefem[\\/]include[\\/]([^\s']+\.idp)", stderr)
    if not m:
        return False
    filename = m.group(1)
    url = f"{GITLAB_RAW_URL}/{filename}"
    r = subprocess.run(
        [
            python_exe,
            "-c",
            "import autofreefem,os;print(os.path.join(os.path.dirname(autofreefem.__file__),'include'))",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print(f"  [fetch] cannot locate autofreefem include dir (rc={r.returncode})")
        return False
    dst = Path(r.stdout.strip()) / filename
    print(f"  [fetch] {filename} <- {url}")
    try:
        urllib.request.urlretrieve(url, dst)
        return dst.exists()
    except Exception as e:
        print(f"  [fetch] FAILED: {e}")
        return False


def run_autofreefem(main_py: Path, workdir: Path = None, python_exe: str = None) -> dict:
    if workdir is None:
        workdir = main_py.parent
    if python_exe is None:
        python_exe = sys.executable

    # Clean stale artefacts from previous runs so autofreefem starts fresh
    for pattern in ["*.edp", "*.pdf"]:
        for f in workdir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass
    for sub in ["include", "tmp", "tex"]:
        d = workdir / sub
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError:
                pass
    (workdir / "tmp").mkdir(exist_ok=True)
    results_d = workdir / "results"
    results_d.mkdir(exist_ok=True)
    # Clean stale iteration VTUs so the progress watchdog starts from 0
    for f in results_d.glob("result*Th*.vtu"):
        try:
            f.unlink()
        except OSError:
            pass

    for _ in range(AUTOFREEFEM_MAX_RETRIES):
        try:
            r = subprocess.run(
                [python_exe, str(main_py.resolve())],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=AUTOFREEFEM_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {
                "return_code": "-1",
                "stdout": "",
                "stderr": f"TIMEOUT after {AUTOFREEFEM_TIMEOUT} seconds",
            }
        if r.returncode == 0:
            break
        if not _fetch_missing_idp(r.stderr, python_exe):
            break

    return {"return_code": str(r.returncode), "stdout": r.stdout, "stderr": r.stderr}


if __name__ == "__main__":
    from Execution.install_python import setup_environment

    python_exe = setup_environment()
    results_dir = Path(_ROOT) / "results"
    py_files = sorted(results_dir.glob("*/main.py"))

    if not py_files:
        print("No results/*/main.py found.")
        sys.exit(1)

    print(f"Running {len(py_files)} AutoFreeFEM scripts …\n")

    for py_file in py_files:
        name = py_file.parent.name
        print(f"  [{name}]", end=" ", flush=True)
        r = run_autofreefem(py_file.resolve(), python_exe=python_exe)
        rc = r["return_code"]
        tag = "OK" if rc == "0" else f"FAIL({rc})"
        if rc != "0" and r["stderr"]:
            tag += f" — {r['stderr'].strip().splitlines()[-1][:120]}"
        print(tag)

    print("\nDone.")
