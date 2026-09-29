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
import threading
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import PYTHON_ENV_NAME, PYTHON_VERSION  # noqa: E402

_ENV_RECORD = Path.home() / ".autofreefem_pyenv.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REQUIREMENTS = _PROJECT_ROOT / "requirements.txt"


# ── helpers ────────────────────────────────────────────────────────────


def _fatal(msg: str) -> None:
    """FATAL-level alarm: the Python environment cannot be prepared and
    the pipeline cannot proceed — print a prominent banner and abort."""
    print("\n" + "!" * 60, file=sys.stderr)
    print("  FATAL — Python environment setup failed:", file=sys.stderr)
    print(f"  {msg}", file=sys.stderr)
    print("!" * 60 + "\n", file=sys.stderr)
    raise SystemExit(1)


def _cmd(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, errors="replace", **kw
    )


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


def _large_enough(path) -> bool:
    """Sanity check a downloaded installer: HTTP error pages from ``curl``
    without ``--fail`` would previously be executed as an installer.

    Miniconda installers are ~60-110 MB; anything < 10 MB is a bad
    response (HTML error page / truncated download).
    """
    try:
        return Path(path).stat().st_size > 10_000_000
    except OSError:
        return False


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
        r = _cmd(["curl", "-fSL", "-o", str(Path.home() / installer), url])
        if r.returncode != 0 or not _large_enough(Path.home() / installer):
            print("[install] Download failed (HTTP error or truncated file).")
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
                "-fSL",
                "-o",
                str(script),
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            ]
        )
        if r.returncode != 0 or not _large_enough(script):
            print("[install] Download failed (HTTP error or truncated file).")
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
    # Exact env-name match per line: a substring hit on an unrelated env
    # (autofreefem-dev, myautofreefem) would skip the creation and fail
    # later with "Could not get python from env".
    for line in r.stdout.splitlines():
        tokens = line.strip().lstrip("*").split()
        if tokens and tokens[0] == PYTHON_ENV_NAME:
            print(f"[env] Env '{PYTHON_ENV_NAME}' exists.")
            return True

    _accept_conda_tos(conda)
    print(f"[env] Creating '{PYTHON_ENV_NAME}' (Python {PYTHON_VERSION}) …")
    r = _cmd([conda, "create", "-y", "-n", PYTHON_ENV_NAME, f"python={PYTHON_VERSION}"])
    if r.returncode != 0:
        print(f"[env]  FAILED: {r.stderr[:300]}")
        return False
    return True


def _parse_requirement(line: str) -> tuple[str, str | None]:
    """Split one requirements.txt line into (package_name, version_constraint).

    Handles plain names, version specifiers (``setuptools<81``), extras
    (``pkg[extra]``), and git URLs (``git+https://...#egg=name``).
    """
    line = line.strip()
    if not line:
        return "", None
    # #egg= overrides the name — read it before the comment strip,
    # which would remove the fragment.  Egg names may contain dots
    # (e.g. zope.interface).
    m = re.search(r"#egg=([A-Za-z0-9_.-]+)", line)
    if m:
        return m.group(1), None
    line = line.split("#", 1)[0].strip()
    if not line:
        return "", None
    # environment markers (pkg; python_version < '3.8') — name only
    line = line.split(";", 1)[0].strip()
    if not line:
        return "", None
    if line.startswith(("git+", "http://", "https://")):
        name = line.rstrip("/").split("/")[-1]
        return name.replace(".git", ""), None
    for op in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if op in line:
            name, constraint = line.split(op, 1)
            return name.strip().split("[", 1)[0].strip(), op + constraint.strip()
    return line.split("[", 1)[0].strip(), None


def _norm_name(name: str) -> str:
    """PEP 503 canonical form — pip treats ``-``, ``_`` and ``.`` as equal
    (``langchain_core`` and ``langchain-core`` are the same package)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_key(version: str) -> tuple:
    """Sortable key for a version string — numeric segments compared
    numerically, non-numeric segments as strings.  Tolerates None."""
    parts = []
    for chunk in re.split(r"(\d+)", version or ""):
        if chunk.isdigit():
            parts.append((1, int(chunk)))
        elif chunk:
            parts.append((0, chunk))
    return tuple(parts)


def _split_prerelease(version: str) -> tuple[str, str]:
    """Split ``1.4.5rc1`` into (numeric part, pre-release suffix).

    A pre-release (a/b/rc/dev/pre) sorts BELOW the bare release of the
    same version (``1.0rc1 < 1.0``) — plain numeric comparison would
    order it above.
    """
    m = re.match(r"^([0-9.]+)([a-z][\w.]*)$", version or "", re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return (version or ""), ""


def _numeric_parts(version: str) -> list[int]:
    """Numeric segments of a version string (``1.4.5`` → ``[1, 4, 5]``)."""
    return [int(x) for x in re.findall(r"\d+", version or "")]


def _version_ok(version: str, constraint: str) -> bool:
    """Check a version against a PEP-440-style constraint set.

    Accepts a single operator (``<81``) or a comma-separated set
    (``>=1.0,<2.0`` — every part must hold).  ``~=`` is the compatible
    release: ``>=X.Y[.Z]`` and strictly below the increment of the last
    given segment (``~=1.4`` admits 1.9 but not 2.0).  An unparseable
    constraint is treated as satisfied — never block an install over a
    spec we cannot read.
    """
    for part in constraint.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(==|>=|<=|~=|!=|>|<)\s*([\w.+-]+)", part)
        if not m:
            continue  # unparseable constraint — treat as satisfied
        op, wanted = m.group(1), m.group(2)
        # Pre-release-aware keys: a release (no suffix) sorts ABOVE any
        # pre-release of the same numeric version (1.0 > 1.0rc1), so
        # ordering ops follow PEP 440 for a/b/rc/dev/pre.  The second
        # tuple element is (tier, suffix) with pre-release tier 0 <
        # release tier 1.
        v_num, v_pre = _split_prerelease(version)
        w_num, w_pre = _split_prerelease(wanted)
        v = (_version_key(v_num), (1, "") if not v_pre else (0, v_pre.lower()))
        w = (_version_key(w_num), (1, "") if not w_pre else (0, w_pre.lower()))
        if op == "==" and v != w:
            return False
        if op == "!=" and v == w:
            return False
        if op == ">=" and v < w:
            return False
        if op == "<=" and v > w:
            return False
        if op == ">" and v <= w:
            return False
        if op == "<" and v >= w:
            return False
        if op == "~=":
            # PEP 440 compatible release: ``~=1.4.5`` is >=1.4.5,<1.5.0 —
            # drop the last given segment, increment the new last one.
            w_parts = _numeric_parts(w_num)
            v_parts = _numeric_parts(v_num)
            if not w_parts:
                return False
            if v < w:
                return False
            up = list(w_parts[:-1])
            if up:
                up[-1] += 1
            else:
                up = [w_parts[0] + 1]
            u_full = up + [0] * max(0, len(v_parts) - len(up))
            if v_parts >= u_full:
                return False
    return True


def _packages_installed(python_exe: str) -> bool:
    """Return True when every requirements.txt entry is already installed
    in *python_exe*'s environment, satisfying any version constraint.

    Avoids a pip-install round-trip: autofreefem comes from a remote git
    URL, so an unconditional ``pip install -r`` costs a network timeout
    whenever that host is unreachable — even though the package is
    already installed.
    """
    try:
        r = _cmd([
            python_exe,
            "-c",
            "import json; from importlib import metadata as m; "
            "print(json.dumps({d.metadata.get('Name', ''): d.version "
            "for d in m.distributions() if d.metadata.get('Name')}))",
        ])
        raw = json.loads(r.stdout) if r.returncode == 0 else {}
    except Exception:
        return False
    installed = {_norm_name(k): v for k, v in raw.items()}

    try:
        req_lines = _REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    for line in req_lines:
        name, constraint = _parse_requirement(line)
        if not name:
            continue
        if _norm_name(name) not in installed:
            return False
        if constraint and not _version_ok(installed[_norm_name(name)], constraint):
            return False
    return True


def _install_packages(python_exe: str) -> bool:
    if not _REQUIREMENTS.exists():
        print(f"[env] {_REQUIREMENTS} not found.", file=sys.stderr)
        return False

    if _packages_installed(python_exe):
        print("[env] requirements already satisfied — skipping pip install.")
    else:
        print("[env] pip install -r requirements.txt …")
        r = _cmd([python_exe, "-m", "pip", "install", "-r", str(_REQUIREMENTS)])
        if r.returncode != 0:
            print(f"[env]  FAILED: {r.stderr[:300]}")
            return False

    if not _install_torch(python_exe):
        return False

    print("[env] Packages ready.")
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
        gpu_failed = True
    else:
        gpu_failed = False

    print("[env] No usable GPU found -> CPU-only torch/torchvision/torchaudio …")
    state = _torch_cuda_available(python_exe)
    if state is not None:
        # torch already installed (CPU build — no GPU index applies)
        print("[env] torch/torchvision/torchaudio already installed (CPU).")
        return not gpu_failed  # a GPU was detected but the install failed — propagate
    r = _cmd([python_exe, "-m", "pip", "install", *_TORCH_PACKAGES])
    if r.returncode != 0:
        print(f"[env]  FAILED: {r.stderr[:300]}")
        return False
    print("[env] torch/torchvision/torchaudio (CPU) installed.")
    # Never silently degrade: this machine HAS a GPU but the GPU build
    # could not be installed.  The caller must fail loudly instead of
    # running CPU torch behind a successful setup report.
    if gpu_failed:
        print(
            "[env]  WARNING: GPU torch install FAILED — a CPU-only build was "
            "installed as a fallback.  GPU-accelerated features are UNAVAILABLE; "
            "setup reports failure on purpose.",
            file=sys.stderr,
        )
        return False
    return True


# ── public ─────────────────────────────────────────────────────────────


# The parallel reflection probes call setup_environment() from up to 8
# threads at once — without the lock, concurrent conda create / pip
# install would race and could corrupt the environment.
_ENV_LOCK = threading.Lock()


def _cleanup_problematic_pth(python_exe: str) -> None:
    """Remove the pip_system_certs .pth files from the env's site-packages.

    ``pip_system_certs.pth`` imports pip at interpreter startup.  That
    import runs setuptools' ``DistutilsMetaFinder.spec_for_pip()``, which
    on Python < 3.12 permanently disables setuptools' distutils shim —
    every later ``import setuptools`` then dies with
    ``AssertionError: ... Lib\\distutils\\core.py`` (setuptools
    ``_distutils_hack.ensure_local_distutils``).  The pipeline does not
    need the system-cert injection (pip resolves its own trust), so the
    pth is removed after every environment setup.  Idempotent.
    """
    try:
        r = _cmd(
            [python_exe, "-c", "import site; print(site.getsitepackages()[0])"]
        )
        site_dir = Path(r.stdout.strip())
    except Exception:
        # best effort — a failed probe must not fail the setup
        site_dir = Path(python_exe).resolve().parent / "Lib" / "site-packages"
    for name in ("pip_system_certs.pth", "pip_system_certs.pth.bak"):
        target = site_dir / name
        if target.exists():
            try:
                target.unlink()
                print(f"[env] removed {name} (breaks setuptools' distutils shim)")
            except OSError as e:
                print(f"[env] could not remove {name}: {e}")


def setup_environment() -> str:
    """Return the Python executable path, ready to run AutoFreeFEM."""
    with _ENV_LOCK:
        python_exe = _setup_environment_locked()
    _cleanup_problematic_pth(python_exe)
    return python_exe


def _setup_environment_locked() -> str:
    rec = _record()
    cached = rec.get("python", "")
    if cached and Path(cached).exists():
        print(f"[env] Using cached: {cached}")
        # Same failure semantics as the fresh-install path: a broken
        # package set (failed pip, failed GPU torch) must not silently
        # proceed — fail loudly so it can be repaired.
        if not _install_packages(cached):
            _fatal("Cached environment is BROKEN — re-run install or fix it.")
        return cached

    conda = _find_conda()
    if not conda:
        conda = _install_miniconda()
    if not conda:
        _fatal("No conda available and miniconda could not be installed.")

    if not _create_env(conda):
        _fatal("Conda environment creation failed (see messages above).")

    python_exe = _get_python_from_env(conda)
    if not python_exe:
        _fatal("Could not get python from the conda environment.")

    if not _install_packages(python_exe):
        _fatal("Package installation failed (see messages above).")

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
