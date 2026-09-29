"""
Check external C++ tools — abort fatally if missing, never auto-install.

Required tools (must be installed manually):
  FreeFEM    — https://github.com/FreeFem/FreeFem-sources
  mmg        — https://github.com/MmgTools/mmg
  Advection  — https://github.com/ISCDtoolbox/Advection
  Mshdist    — https://github.com/ISCDtoolbox/Mshdist
"""

import platform
import shutil
import subprocess
import sys

_TOOLS = [
    ("FreeFEM", "FreeFem++"),
    ("mmg", "mmg2d_O3"),
    ("Advection", "advect"),
    ("Mshdist", "mshdist"),
]

_URLS = {
    "FreeFEM": "https://github.com/FreeFem/FreeFem-sources",
    "mmg": "https://github.com/MmgTools/mmg",
    "Advection": "https://github.com/ISCDtoolbox/Advection",
    "Mshdist": "https://github.com/ISCDtoolbox/Mshdist",
}


# Search the whole home directory (bounded depth), not just the PATH:
# advect installs to ~/bin, and locally built tools may live anywhere
# under $HOME — a PATH-only check would report a false MISSING.
_HOME_FIND_SH = (
    "find ~ -maxdepth 4 -type f \\( -name FreeFem++ -o -name mmg2d_O3 "
    "-o -name advect -o -name mshdist \\) 2>/dev/null"
)

# A hung probe (WSL VM booting, slow home dir) must not block every run.
_PROBE_TIMEOUT = 30  # seconds


def _check_native() -> dict[str, bool]:
    results = {name: shutil.which(exe) is not None for name, exe in _TOOLS}
    found = {exe for name, exe in _TOOLS if results[name]}
    try:
        r = subprocess.run(
            ["sh", "-c", _HOME_FIND_SH],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        r = None
    if r is not None:
        for line in r.stdout.splitlines():
            line = line.strip()
            if line:
                found.add(line.split("/")[-1])
    return {name: exe in found for name, exe in _TOOLS}


def _check_wsl() -> dict[str, bool]:
    exe_names = [exe for _, exe in _TOOLS]
    try:
        r = subprocess.run(
            ["wsl", "which"] + exe_names,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {name: False for name, _ in _TOOLS}
    found = {
        line.strip().split("/")[-1]
        for line in r.stdout.splitlines()
        if line.strip() and "/" in line
    }
    # PATH is not the whole story — search the home directory too
    # (the runner mirrors this by injecting ~/bin at execution time).
    try:
        r2 = subprocess.run(
            ["wsl", "sh", "-c", _HOME_FIND_SH],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        r2 = None
    if r2 is not None:
        for line in r2.stdout.splitlines():
            line = line.strip()
            if line:
                found.add(line.split("/")[-1])
    return {name: exe in found for name, exe in _TOOLS}


def check_tools() -> dict[str, bool]:
    """Return ``{tool_name: available}``.  A missing tool is FATAL:
    the pipeline cannot run without it — print a fatal alarm and abort."""
    if platform.system() == "Windows":
        results = _check_wsl()
    else:
        results = _check_native()

    missing = [name for name, ok in results.items() if not ok]
    if missing:
        print("\n" + "!" * 60, file=sys.stderr)
        print("  FATAL — missing external tools:", file=sys.stderr)
        for name in missing:
            base = name.split(" (")[0]
            url = _URLS.get(base, "")
            print(f"    {name}  ->  {url}", file=sys.stderr)
        print("  The pipeline cannot run.  Install these manually, then retry.", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(1)

    return results


if __name__ == "__main__":
    for name, ok in check_tools().items():
        print(f"  {name}: {'OK' if ok else 'MISSING'}")
