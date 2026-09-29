"""
Run AutoFreeFEM Python code on the host system.

The ``include/*.idp`` files AutoFreeFEM needs are shipped locally in
Execution/ (e.g. ``gradientIdentification.idp``, ``inout.idp``) and are
copied into the autofreefem package's include directory before every
run — nothing is downloaded from the upstream GitLab repository.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import AUTOFREEFEM_MAX_RETRIES, AUTOFREEFEM_TIMEOUT  # noqa: E402

_LOCAL_IDP_DIR = Path(__file__).resolve().parent  # Execution/


def _autofreefem_include_dir(python_exe: str):
    """Return the autofreefem package include directory, or None."""
    r = subprocess.run(
        [
            python_exe,
            "-c",
            "import autofreefem,os;print(os.path.join(os.path.dirname(autofreefem.__file__),'include'))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return Path(r.stdout.strip())


def _install_local_idp(python_exe: str) -> None:
    """Copy the .idp files shipped in Execution/ into the autofreefem
    include dir, overwriting any previously downloaded copies.

    FROZEN FILES: inout.idp and gradientIdentification.idp are reference
    files that must NEVER be modified — they are copied byte-for-byte
    (``shutil.copy2``, no patching of any kind; the only file the
    pipeline ever patches is the generated .edp in runner.py).
    """
    inc = _autofreefem_include_dir(python_exe)
    if inc is None:
        print("  [idp] cannot locate autofreefem include dir — skipping")
        return
    for src in _LOCAL_IDP_DIR.glob("*.idp"):
        shutil.copy2(src, inc / src.name)
        print(f"  [idp] {src.name} -> autofreefem include dir")


def _fetch_missing_idp(stderr: str, python_exe: str) -> bool:
    """Provide a missing .idp from the local Execution/ directory.

    AutoFreeFEM errors name the include file it could not find; the file
    is copied VERBATIM from Execution/ — no upstream download and no
    modification of the shipped .idp files.
    """
    m = re.search(r"autofreefem[\\/]include[\\/]([^\s']+\.idp)", stderr)
    if not m:
        return False
    filename = m.group(1)
    src = _LOCAL_IDP_DIR / filename
    if not src.exists():
        print(f"  [fetch] {filename} not found in Execution/ — giving up")
        return False
    inc = _autofreefem_include_dir(python_exe)
    if inc is None:
        print("  [fetch] cannot locate autofreefem include dir — giving up")
        return False
    dst = inc / filename
    shutil.copy2(src, dst)
    print(f"  [fetch] {filename} <- Execution/")
    return dst.exists()


def run_autofreefem(main_py: Path, workdir: Path = None, python_exe: str = None) -> dict:
    if workdir is None:
        workdir = main_py.parent
    if python_exe is None:
        python_exe = sys.executable

    # Clean stale artefacts from previous runs so autofreefem starts fresh.
    # convergenceShape<problem>.txt / objective<problem>.txt are FreeFEM
    # optimizer outputs from the previous run (<problem> = the problem
    # title in the generated code, e.g. Wheelstrainstiffening — the
    # workdir name minus the run_ prefix).
    for pattern in [
        "*.edp", "*.pdf", "*.png",
        "convergenceShape*.txt", "objective*.txt",
    ]:
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
    # Clean stale iteration VTUs so the progress watchdog starts from 0,
    # plus leftover render PNGs and FreeFEM optimizer logs
    for pat in ["result*Th*.vtu", "*.png", "convergenceShape*.txt", "objective*.txt"]:
        for f in results_d.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass

    # Local .idp overrides — install the include files shipped in
    # Execution/ before running, so the generated .edp scripts find them.
    _install_local_idp(python_exe)

    # Cumulative budget: AUTOFREEFEM_TIMEOUT is the TOTAL budget for the
    # whole retry loop (not per attempt) — pass the REMAINING budget as
    # each attempt's timeout so retries cannot overrun the total.
    t_start = time.time()
    for _ in range(AUTOFREEFEM_MAX_RETRIES):
        remaining = AUTOFREEFEM_TIMEOUT - (time.time() - t_start)
        if remaining <= 0:
            return {
                "return_code": "-1",
                "stdout": "",
                "stderr": f"TIMEOUT: total retry budget ({AUTOFREEFEM_TIMEOUT}s) exhausted",
            }
        try:
            r = subprocess.run(
                [python_exe, str(main_py.resolve())],
                cwd=workdir,
                capture_output=True,
                text=True,
                # Non-UTF-8 locales decode subprocess output with the
                # ANSI codec (cp1252 here): autofreefem / main.py print
                # Greek and arrow characters — never crash on them.
                errors="replace",
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            return {
                "return_code": "-1",
                "stdout": "",
                "stderr": f"TIMEOUT after {AUTOFREEFEM_TIMEOUT} seconds (retry budget)",
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
