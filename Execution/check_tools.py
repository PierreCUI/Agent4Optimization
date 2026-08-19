"""
Check external C++ tools — warn if missing, never auto-install.

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
    ("mmg (2D)", "mmg2d_O3"),
    ("mmg (3D)", "mmg3d_O3"),
    ("Advection", "advect"),
    ("Mshdist", "mshdist"),
]

_URLS = {
    "FreeFEM": "https://github.com/FreeFem/FreeFem-sources",
    "mmg": "https://github.com/MmgTools/mmg",
    "Advection": "https://github.com/ISCDtoolbox/Advection",
    "Mshdist": "https://github.com/ISCDtoolbox/Mshdist",
}


def _check_native() -> dict[str, bool]:
    results = {}
    for name, exe in _TOOLS:
        results[name] = shutil.which(exe) is not None
    return results


def _check_wsl() -> dict[str, bool]:
    exe_names = [exe for _, exe in _TOOLS]
    try:
        r = subprocess.run(["wsl", "which"] + exe_names, capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return {name: False for name, _ in _TOOLS}
    found = {
        line.strip().split("/")[-1]
        for line in r.stdout.splitlines()
        if line.strip() and "/" in line
    }
    return {name: exe in found for name, exe in _TOOLS}


def check_tools() -> dict[str, bool]:
    """Return ``{tool_name: available}``.  Prints warnings to stderr."""
    if platform.system() == "Windows":
        results = _check_wsl()
    else:
        results = _check_native()

    missing = [name for name, ok in results.items() if not ok]
    if missing:
        print("\n" + "!" * 60, file=sys.stderr)
        print("  WARNING — missing external tools:", file=sys.stderr)
        for name in missing:
            base = name.split(" (")[0]
            url = _URLS.get(base, "")
            print(f"    {name}  →  {url}", file=sys.stderr)
        print("  These must be installed manually.", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)

    return results


if __name__ == "__main__":
    for name, ok in check_tools().items():
        print(f"  {name}: {'OK' if ok else 'MISSING'}")
