"""
Install miniconda and create the Python environment for the pipeline.

On both Windows and Linux:
  1. Detect or install miniconda.
  2. Create a conda environment named ``configs.PYTHON_ENV_NAME``
     at ``configs.PYTHON_VERSION``.
  3. ``pip install -r requirements.txt``.
  4. Install ``torch``/``torchvision``/``torchaudio`` matching the machine:
     CUDA wheels for NVIDIA GPUs, ROCm wheels for AMD GPUs on Linux,
     CPU wheels otherwise.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import PYTHON_ENV_NAME, PYTHON_VERSION

_ENV_RECORD = Path.home() / ".autofreefem_pyenv.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REQUIREMENTS = _PROJECT_ROOT / "requirements.txt"


# ── helpers ────────────────────────────────────────────────────────────


def _cmd(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def _find_conda() -> str | None:
    for name in (["conda.exe", "conda"] if platform.system() == "Windows" else ["conda"]):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _record() -> dict:
    if _ENV_RECORD.exists():
        try:
            return json.loads(_ENV_RECORD.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(python_exe: str) -> None:
    _ENV_RECORD.write_text(
        json.dumps({"python": python_exe, "env": PYTHON_ENV_NAME}, indent=2),
        encoding="utf-8",
    )


def _get_python_from_env(conda: str) -> str | None:
    r = _cmd(
        [conda, "run", "-n", PYTHON_ENV_NAME, "python", "-c", "import sys; print(sys.executable)"]
    )
    if r.returncode == 0:
        return r.stdout.strip()
    return None


# ── install miniconda ──────────────────────────────────────────────────


def _install_miniconda() -> str | None:
    """Download and install miniconda, return conda path."""
    if platform.system() == "Windows":
        installer = "Miniconda3-latest-Windows-x86_64.exe"
        url = f"https://repo.anaconda.com/miniconda/{installer}"
        dest = Path.home() / "miniconda3"
        exe = dest / "Scripts" / "conda.exe"
        if exe.exists():
            return str(exe)

        print(f"[install] Downloading {installer} …")
        r = _cmd(["curl", "-L", "-o", str(Path.home() / installer), url])
        if r.returncode != 0:
            print("[install] Download failed.")
            return None

        print("[install] Installing miniconda (silent) …")
        r = _cmd(
            [
                str(Path.home() / installer),
                "/InstallationType=JustMe",
                "/RegisterPython=0",
                "/S",
                f"/D={dest}",
            ]
        )
        if r.returncode != 0:
            print("[install] Miniconda install failed.")
            return None
        return str(exe) if exe.exists() else None

    else:  # Linux
        script = Path.home() / "miniconda_install.sh"
        dest = Path.home() / "miniconda3"
        exe = dest / "bin" / "conda"
        if exe.exists():
            return str(exe)

        print("[install] Downloading Miniconda3-latest-Linux-x86_64.sh …")
        r = _cmd(
            [
                "curl",
                "-L",
                "-o",
                str(script),
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            ]
        )
        if r.returncode != 0:
            print("[install] Download failed.")
            return None

        print("[install] Installing miniconda …")
        r = _cmd(["bash", str(script), "-b", "-p", str(dest)])
        if r.returncode != 0:
            print("[install] Miniconda install failed.")
            return None
        script.unlink()
        return str(exe) if exe.exists() else None


# ── create env + install packages ──────────────────────────────────────


def _accept_conda_tos(conda: str) -> None:
    """Accept conda TOS if needed (conda >= 26.x)."""
    for ch in [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2",
    ]:
        _cmd([conda, "tos", "accept", "--override-channels", "--channel", ch])


def _create_env(conda: str) -> bool:
    r = _cmd([conda, "env", "list"])
    if PYTHON_ENV_NAME in r.stdout:
        print(f"[env] Env '{PYTHON_ENV_NAME}' exists.")
        return True

    _accept_conda_tos(conda)
    print(f"[env] Creating '{PYTHON_ENV_NAME}' (Python {PYTHON_VERSION}) …")
    r = _cmd([conda, "create", "-y", "-n", PYTHON_ENV_NAME, f"python={PYTHON_VERSION}"])
    if r.returncode != 0:
        print(f"[env]  FAILED: {r.stderr[:300]}")
        return False
    return True


def _install_packages(python_exe: str) -> bool:
    if not _REQUIREMENTS.exists():
        print(f"[env] {_REQUIREMENTS} not found.", file=sys.stderr)
        return False

    print(f"[env] pip install -r requirements.txt …")
    r = _cmd([python_exe, "-m", "pip", "install", "-r", str(_REQUIREMENTS)])
    if r.returncode != 0:
        print(f"[env]  FAILED: {r.stderr[:300]}")
        return False

    if not _install_torch(python_exe):
        return False

    print("[env] Packages installed.")
    return True


# ── install PyTorch (GPU-aware) ────────────────────────────────────────

# Wheel index suffixes on download.pytorch.org, newest last: (driver version, index).
_CUDA_INDEXES = [
    ("11.8", "cu118"),
    ("12.1", "cu121"),
    ("12.4", "cu124"),
    ("12.6", "cu126"),
    ("12.8", "cu128"),
    ("12.9", "cu129"),
    ("13.0", "cu130"),
]
_ROCM_INDEXES = [
    ("6.2", "rocm6.2"),
    ("6.3", "rocm6.3"),
    ("6.4", "rocm6.4"),
]
_TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]
_TORCH_INDEX_BASE = "https://download.pytorch.org/whl"


def _detect_cuda_version() -> str | None:
    """CUDA version supported by the NVIDIA driver (e.g. '12.7'), or None."""
    try:
        r = _cmd(["nvidia-smi"], timeout=30)
    except (FileNotFoundError, OSError):
        return None
    if r.returncode != 0:
        return None
    m = re.search(r"CUDA\s+(?:UMD\s+)?Version:\s*(\d+\.\d+)", r.stdout)
    return m.group(1) if m else None


def _detect_rocm_version() -> str | None:
    """ROCm driver version (e.g. '6.3') on Linux AMD systems, or None."""
    if platform.system() == "Windows":
        return None  # PyTorch ships no ROCm wheels for Windows
    try:
        r = _cmd(["rocm-smi", "--showdriverversion"], timeout=30)
    except (FileNotFoundError, OSError):
        return None
    if r.returncode != 0:
        return None
    m = re.search(r"Driver version:\s*(\d+\.\d+)", r.stdout)
    return m.group(1) if m else None


def _pick_index(version: str, indexes: list[tuple[str, str]]) -> str | None:
    """Newest wheel index (e.g. 'cu126') whose CUDA/ROCm version <= driver version."""
    try:
        v = tuple(int(x) for x in version.split(".")[:2])
    except ValueError:
        return None
    for iv_str, index in reversed(indexes):
        iv = tuple(int(x) for x in iv_str.split("."))
        if v >= iv:
            return index
    return None


def _torch_cuda_available(python_exe: str) -> bool | None:
    """True/False if torch is installed and reports its CUDA availability, None if not installed."""
    r = _cmd([python_exe, "-c", "import torch; print(torch.cuda.is_available())"])
    if r.returncode != 0:
        return None
    return r.stdout.strip() == "True"


def _install_torch(python_exe: str) -> bool:
    """Install torch/torchvision/torchaudio for the machine's GPU (CUDA/ROCm), else CPU."""
    index = None
    cuda = _detect_cuda_version()
    if cuda:
        index = _pick_index(cuda, _CUDA_INDEXES)
        if index:
            print(f"[env] NVIDIA GPU detected (driver CUDA {cuda}) -> torch from {index} index …")
    if not index:
        rocm = _detect_rocm_version()
        if rocm:
            index = _pick_index(rocm, _ROCM_INDEXES)
            if index:
                print(f"[env] AMD GPU detected (ROCm {rocm}) -> torch from {index} index …")
    if index:
        state = _torch_cuda_available(python_exe)
        if state:
            print("[env] torch/torchvision/torchaudio (GPU) already installed.")
            return True
        args = [
            python_exe,
            "-m",
            "pip",
            "install",
            *_TORCH_PACKAGES,
            "--index-url",
            f"{_TORCH_INDEX_BASE}/{index}",
        ]
        if state is False:
            args.append("--force-reinstall")  # replace an existing CPU build with the GPU build
        r = _cmd(args)
        if r.returncode == 0:
            print("[env] torch/torchvision/torchaudio (GPU) installed.")
            return True
        print(f"[env]  GPU torch install FAILED ({r.stderr[:300]}) — retrying CPU-only …")

    print("[env] No usable GPU found -> installing CPU-only torch/torchvision/torchaudio …")
    r = _cmd([python_exe, "-m", "pip", "install", *_TORCH_PACKAGES])
    if r.returncode != 0:
        print(f"[env]  FAILED: {r.stderr[:300]}")
        return False
    print("[env] torch/torchvision/torchaudio (CPU) installed.")
    return True


# ── public ─────────────────────────────────────────────────────────────


def setup_environment() -> str:
    """Return the Python executable path, ready to run AutoFreeFEM."""
    rec = _record()
    cached = rec.get("python", "")
    if cached and Path(cached).exists():
        print(f"[env] Using cached: {cached}")
        _install_packages(cached)
        return cached

    conda = _find_conda()
    if not conda:
        conda = _install_miniconda()
    if not conda:
        print("[env] No conda available.", file=sys.stderr)
        sys.exit(1)

    if not _create_env(conda):
        print("[env] Env creation failed.", file=sys.stderr)
        sys.exit(1)

    python_exe = _get_python_from_env(conda)
    if not python_exe:
        print("[env] Could not get python from env.", file=sys.stderr)
        sys.exit(1)

    if not _install_packages(python_exe):
        sys.exit(1)

    _save(python_exe)
    print(f"[env] Python: {python_exe}")
    return python_exe


if __name__ == "__main__":
    py = setup_environment()
    r = _cmd([py, "-c", "import autofreefem,numpy,scipy,meshio; print('all OK')"])
    print(f"  {'OK' if r.returncode == 0 else 'MISSING: ' + r.stderr[:200]}")
    t = _cmd(
        [
            py,
            "-c",
            "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())",
        ]
    )
    print(f"  {t.stdout.strip() if t.returncode == 0 else 'torch check FAILED: ' + t.stderr[:200]}")
