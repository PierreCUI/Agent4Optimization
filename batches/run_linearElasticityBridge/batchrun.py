"""
batchrun.py -- Parameter sweep for run_linearElasticityBridge.

Full Cartesian product of all parameters.py lists.  Each config runs in
an isolated worker directory.  8 workers in parallel.
"""

import ast
import csv
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import psutil

from parameters import *  # noqa: F403  (all symbols are float lists)

# -- paths ------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BATCH_DIR = Path(__file__).resolve().parent
_RESULTS_SRC = _PROJECT_ROOT / "results" / "run_linearElasticityBridge"
_BATCH_CSV = _BATCH_DIR / "batch.csv"

_PYTHON_EXE = Path.home() / "miniconda3" / "envs" / "autofreefem" / "python.exe"
if not _PYTHON_EXE.exists():
    for p in Path.home().rglob("miniconda3/envs/autofreefem/python.exe"):
        _PYTHON_EXE = p
        break

# -- parameter sweep grid ---------------------------------------------
# Varying parameters (from parameters.py); fixed ones contribute 1 value
_SWEEP_NAMES = [
    "hmax", "hmin_ratio", "p_degree", "v0", "diffusion",
    "mesh_n", "chi_contrast",
    "f",
    "mu",
    "lam",
    "Afac",
]
_SWEEP_LISTS = [
    HMAX, HMIN_RATIO, P_DEGREE, V0, DIFFUSION,
    MESH_N, CHI_CONTRAST,
    F_VALUE,
    MU_VALUE,
    LAM_VALUE,
    AFAC_VALUE,
]

# Constant definitions in main.py that the sweep modifies
_CONSTANT_PATTERNS = {"f": "f", "mu": r"\mu", "lam": r"\lambda",
                      "Afac": r"\ell"}

AUTOFREEFEM_TIMEOUT = 120
FREEFEM_TOTAL_TIMEOUT = 3600
MEMORY_SAMPLE_INTERVAL = 0.1
# Watchdog: FreeFEM is considered stuck if no new epoch appears for this
# long.  Nonlinear problems solve much slower per epoch, so they get
# NONLINEAR_STEP_MULT x the linear budget (300 * 5 = 1500 s per epoch).
FREEFEM_STEP_TIMEOUT = 300
NONLINEAR_STEP_MULT = 5
# Iterations forced into every worker's main.py.  The generated edp writes
# one objective line per pass of its `for (iter=1; iter<N; iter++)` loop,
# i.e. N-1 lines for N iterations — that count is passed to the WSL monitor
# as expected_epochs.
BATCH_ITERATIONS = 10
WORKERS = 8


def _wsl_path(path: Path) -> str:
    p = str(path.resolve())
    if len(p) > 2 and p[1:3] == ":\\":
        return f"/mnt/{p[0].lower()}/{p[3:].replace(chr(92), '/')}"
    return p.replace("\\", "/")


def _kill_wsl_orphans(workdir_wsl: str):
    """Kill only the WSL processes of one timed-out worker, identified by
    its worker-directory path in the command line.  A blanket
    ``pkill -f FreeFem++`` would kill every config running in parallel."""
    # pkill -f does a regex search: anchor the worker path with a following
    # separator so ``worker_3`` cannot match ``worker_30`` etc.  The path
    # appears as ``.../worker_3/edp`` (FreeFEM) or ``.../worker_3 3600``
    # (the monitor's argv).
    pattern = workdir_wsl + "[/ ]"
    try:
        subprocess.run(["wsl", "-e", "pkill", "-f", pattern],
                       capture_output=True, timeout=10)
    except Exception:
        pass


def _rmtree_force(path: Path) -> None:
    """rmtree that also removes read-only directories.  copytree preserves
    the source's read-only bit (results/.../include is 0o555), and plain
    shutil.rmtree then fails on Windows — silently with ignore_errors."""

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_onerror)


def _f32(v) -> str:
    if v is None or v == "":
        return ""
    return f"{float(v):.6g}"


_STR_COLS = {"p_degree"}


def _write_row(writer, row: dict) -> None:
    out = {}
    for k, v in row.items():
        if k in _STR_COLS:
            out[k] = str(v) if v is not None else ""
        else:
            out[k] = _f32(v)
    writer.writerow(out)


def _fmt(val: float) -> str:
    if val == 0:
        return "0"
    s = f"{val:.10g}"
    if "e" not in s and "." not in s:
        s += "."
    return s


def copy_template_files():
    for p in _BATCH_DIR.iterdir():
        if p.name in ("batchrun.py", "batch.csv", "parameters.py",
                      "results.py", "results.csv",
                      "ablation.py", "ablation.csv", "wsl_monitor.py",
                      "ablation_analyse.py", "ablation_heatmap.png",
                      "ablation_correlation.csv",
                      "batch_analyse.py", "batch_heatmap.png",
                      "batch_correlation.csv"):
            continue
        if p.is_dir() and p.name.startswith("worker_"):
            continue
        if p.is_dir():
            _rmtree_force(p)
        else:
            try:
                p.unlink()
            except OSError:
                pass

    def _ignore(path, names):
        # Skip big mesh files and the template's working subdirectories:
        # workers always copy from _RESULTS_SRC directly, so the batch dir
        # does not need them — otherwise they linger here after the run.
        return [n for n in names if n.endswith(".vtu")
                or n in ("include", "results", "tex", "tmp")]

    shutil.copytree(str(_RESULTS_SRC), str(_BATCH_DIR),
                    dirs_exist_ok=True, ignore=_ignore)


def setup_worker_dir(worker_dir: Path):
    """Copy template files into an isolated worker directory."""
    if worker_dir.exists():
        _rmtree_force(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)

    def _ignore(path, names):
        return [n for n in names if n.endswith(".vtu")]

    shutil.copytree(str(_RESULTS_SRC), str(worker_dir),
                    dirs_exist_ok=True, ignore=_ignore)
    # Remove stale objective/convergence files from the template copy —
    # otherwise the WSL monitor sees fake progress and the step-timeout
    # watchdog kills FreeFEM spuriously.
    for pat in ["objective*.txt", "convergenceShape*.txt"]:
        for f in worker_dir.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass
    # Force a fixed iteration count for every worker.
    mp = worker_dir / "main.py"
    code = mp.read_text(encoding="utf-8")
    code = re.sub(r"iterations\s*=\s*\d+",
                  f"iterations={BATCH_ITERATIONS}", code)
    mp.write_text(code, encoding="utf-8")


def parse_mesh_info(workdir: Path) -> dict:
    mesh_path = workdir / "mesh.txt"
    info = {"mesh_n": 0, "mesh_borders": 0, "mesh_total_len": 0.0,
            "mesh_initial_tri_est": 0}
    if not mesh_path.exists():
        return info
    code = mesh_path.read_text(encoding="utf-8", errors="replace")
    # strip all comments (/* ... */ blocks and // line comments) so that
    # commented-out borders or meshes are never counted
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    m = re.search(r"int\s+n\s*=\s*(\d+)", code)
    if m:
        info["mesh_n"] = int(m.group(1))
    # real parameters (real name = expr;), resolved first so that
    # t-range expressions like bb, 0.45*bb or bb/2 can be evaluated
    params = {pm.group(1): pm.group(2).strip()
              for pm in re.finditer(r"real\s+(\w+)\s*=\s*([^;]+);", code)}
    vals = {}
    _bins = {ast.Add: lambda a, b: a + b,
             ast.Sub: lambda a, b: a - b,
             ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b}

    def eval_expr(expr):
        """Evaluate a numeric FreeFem expression (parameters allowed)."""

        def ev(node):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, (int, float))):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id in vals:
                return vals[node.id]
            if (isinstance(node, ast.UnaryOp)
                    and isinstance(node.op, (ast.USub, ast.UAdd))):
                v = ev(node.operand)
                if v is None:
                    return None
                return -v if isinstance(node.op, ast.USub) else v
            if isinstance(node, ast.BinOp) and type(node.op) in _bins:
                a, b = ev(node.left), ev(node.right)
                if a is None or b is None:
                    return None
                try:
                    return _bins[type(node.op)](a, b)
                except ZeroDivisionError:
                    return None
            return None

        try:
            return ev(ast.parse(expr.strip(), mode="eval").body)
        except Exception:
            return None

    for _ in range(len(params) + 1):
        for name, expr in params.items():
            if name not in vals:
                v = eval_expr(expr)
                if v is not None:
                    vals[name] = v

    def lin_width(expr, var):
        """Width of a linear map in var (value at 1 minus value at 0)."""
        hi = eval_expr(re.sub(rf"\b{re.escape(var)}\b", "1", expr))
        lo = eval_expr(re.sub(rf"\b{re.escape(var)}\b", "0", expr))
        if hi is None or lo is None:
            return None
        return hi - lo

    # only borders referenced by an active buildmesh count (declarations
    # whose usage is commented out are ignored); a square() mesh
    # contributes its four sides instead
    used = set()
    for bm in re.finditer(r"buildmesh\s*\((.*?)\)\s*;", code):
        used.update(re.findall(r"\b(\w+)\s*\([^)]*\)",
                                bm.group(1)))
    n_sq = len(re.findall(r"square\s*\(", code))
    borders = list(re.finditer(r"border\s+(\w+)\s*\([^)]*\)", code))
    info["mesh_borders"] = (sum(1 for m in borders
                                if m.group(1) in used) + 4 * n_sq)
    total_len = 0.0
    for m in borders:
        if m.group(1) not in used:
            continue
        m2 = re.search(r"t\s*=\s*([^,]+)\s*,\s*([^)]+)\)", m.group(0))
        if not m2:
            continue
        a, c = eval_expr(m2.group(1)), eval_expr(m2.group(2))
        if a is None or c is None:
            continue
        total_len += abs(c - a)
    # square(n, m, [fx, fy]) meshes: the boundary is the rectangle
    # [0, a] x [0, b] of the linear map (fx = a*x, fy = b*y), so the
    # perimeter is 2*(a + b); without a map it is the unit square
    for sm in re.finditer(r"square\s*\(([^)]*)\)", code):
        tmap = re.search(r"\[([^\]]*)\]", sm.group(1))
        if tmap:
            fx, fy = (x.strip() for x in tmap.group(1).split(","))
            a, b = lin_width(fx, "x"), lin_width(fy, "y")
            if a is not None and b is not None:
                total_len += 2.0 * (a + b)
        else:
            total_len += 4.0
    info["mesh_total_len"] = round(total_len, 4)
    if info["mesh_n"] and total_len > 0:
        bv = info["mesh_n"] * total_len
        info["mesh_initial_tri_est"] = int(bv * bv / 4)
    return info


def parse_problem_features(workdir: Path) -> dict:
    feat = {"sum_dof_node": 1, "is_nonlinear": 0,
            "num_state_fields": 1, "num_boundary_objects": 0}
    main_py = workdir / "main.py"
    code = main_py.read_text(encoding="utf-8") if main_py.exists() else ""
    ij = workdir / "initial.json"
    dim = 2
    if ij.exists():
        with open(ij, encoding="utf-8") as f:
            ds = json.load(f).get("designed", {})
        if ds.get("state_is_nonlinear"):
            feat["is_nonlinear"] = 1
        if ds.get("dimension") == 3:
            dim = 3
    # sum of the scalar dimensions of every Lagrangian state field
    # (ScalarField = 1, VectorField = problem dimension), e.g. a 2D
    # displacement u and a scalar temperature T give 2 + 1 = 3
    m = re.search(r"Lagrangian\(\s*\[([^\]]+)\]", code)
    fields = ([x.strip() for x in m.group(1).split(",") if x.strip()]
              if m else [])
    if fields:
        feat["num_state_fields"] = len(fields)
        feat["sum_dof_node"] = sum(
            dim if re.search(r"(?<![\w])" + re.escape(f)
                             + r"\s*=\s*VectorField\(", code) else 1
            for f in fields)
    elif re.search(r"VectorField\(", code):
        feat["sum_dof_node"] = dim
    feat["num_boundary_objects"] = len(re.findall(r"=\s*Boundary\(", code))
    return feat


def step_timeout_for(prob_feat: dict) -> int:
    """Per-epoch watchdog timeout for a problem.  Nonlinear problems solve
    much slower per epoch, so they get NONLINEAR_STEP_MULT x the linear
    budget (300 * 5 = 1500 s per epoch)."""
    if prob_feat["is_nonlinear"]:
        return FREEFEM_STEP_TIMEOUT * NONLINEAR_STEP_MULT
    return FREEFEM_STEP_TIMEOUT


def modify_main_py(workdir: Path, config: dict):
    main_py = workdir / "main.py"
    code = main_py.read_text(encoding="utf-8")
    code = re.sub(r"hmin\s*=\s*[^,)]+", f"hmin={_fmt(config['hmin'])}", code)
    code = re.sub(r"hmax\s*=\s*[^,)]+", f"hmax={_fmt(config['hmax'])}", code)
    code = re.sub(r"v0\s*=\s*[^,)]+", f"v0={_fmt(config['v0'])}", code)
    code = re.sub(r"diffusion\s*=\s*[^,)]+",
                  f"diffusion={_fmt(config['diffusion'])}", code)
    # P degree of the state field - the test function keeps the
    # same degree (modified together, both occurrences).
    if config.get('p_degree'):
        code = re.sub(r"((?:ScalarField|VectorField)\([^,]+,\s*)P\d+",
                      rf"\g<1>{config['p_degree']}", code)
    # physical constants
    for key, latex in _CONSTANT_PATTERNS.items():
        if key in config:
            code = re.sub(
                rf"(Constant\(r?'{re.escape(latex)}',\s*)[-\d.eE+]+",
                rf"\g<1>{_fmt(config[key])}", code)
    if 1 <= config['chi_contrast'] <= 1e4:
        weak_val = f"'1/{round(config['chi_contrast'])}'"
        # Only X is rewritten (its weak value is quoted).  The second
        # characteristic function X2 (1, 0) is a solid-material load
        # indicator — unquoted weak value, never matched here.
        code = re.sub(
            r"(CharacteristicFunction\s*\(\s*[^,]+,\s*[\d.eE+\-]+\s*,\s*)'[\d./eE+\-]+'",
            rf"\1{weak_val}", code)
    main_py.write_text(code, encoding="utf-8")


def modify_mesh_txt(workdir: Path, config: dict):
    mesh_path = workdir / "mesh.txt"
    if not mesh_path.exists():
        return
    code = mesh_path.read_text(encoding="utf-8")
    code = re.sub(r"int\s+n\s*=\s*\d+",
                  f"int n = {int(config['mesh_n'])}", code)
    # NOTE: this template's mesh.txt is square(4n, n, [4x, y]) — there is
    # no inner hole and no ``real a`` line, so no hole-size rewrite here.
    mesh_path.write_text(code, encoding="utf-8")


def run_autofreefem(workdir: Path, timeout: int = AUTOFREEFEM_TIMEOUT) -> dict:
    main_py = (workdir / "main.py").resolve()
    for pattern in ["*.edp", "*.pdf"]:
        for f in workdir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass
    for sub in ["tmp", "tex", "results", "include"]:
        d = workdir / sub
        if d.exists():
            _rmtree_force(d)
    (workdir / "tmp").mkdir(exist_ok=True)
    (workdir / "results").mkdir(exist_ok=True)
    orig = _RESULTS_SRC / "include"
    if orig.exists():
        shutil.copytree(orig, workdir / "include", dirs_exist_ok=True)

    # Redirect child output to files: piping would deadlock if AutoFreeFEM
    # fills the pipe buffer while the memory-sampling loop below never
    # drains it.  The logs also survive for debugging.
    log_out = workdir / "tmp" / "af_stdout.log"
    log_err = workdir / "tmp" / "af_stderr.log"

    t0 = time.time()
    max_rss = 0
    timed_out = False
    with open(log_out, "w", encoding="utf-8") as fout, \
            open(log_err, "w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(
            [str(_PYTHON_EXE), str(main_py)],
            cwd=str(workdir),
            stdout=fout, stderr=ferr,
        )
        try:
            psproc = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    total = psproc.memory_info().rss
                    for child in psproc.children(recursive=True):
                        try:
                            total += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    if total > max_rss:
                        max_rss = total
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.time() - t0 > timeout:
                    proc.kill()
                    timed_out = True
                    break
                time.sleep(0.2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        proc.wait()

    if timed_out:
        return {"return_code": "-1", "stdout": "", "stderr": "AutoFreeFEM timeout",
                "wall_time_s": timeout, "max_rss_mb": 0.0}
    stdout = log_out.read_text(encoding="utf-8", errors="replace")
    stderr = log_err.read_text(encoding="utf-8", errors="replace")
    return {
        "return_code": str(proc.returncode),
        "stdout": stdout, "stderr": stderr,
        "wall_time_s": round(time.time() - t0, 2),
        "max_rss_mb": round(max_rss / (1024 * 1024), 2),
    }


def run_freefem_with_monitoring(workdir: Path, step_timeout: int) -> dict:
    if shutil.which("wsl") is None:
        return {"_error": "WSL not found"}
    edp = (workdir / "optimize_linearElasticitybridge.edp").resolve()
    if not edp.exists():
        return {"_error": f"EDP not found: {edp}"}
    print(f"    Launching FreeFEM via WSL (step timeout {step_timeout}s) ...",
          flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            ["wsl", "python3", _wsl_path(_BATCH_DIR / "wsl_monitor.py"),
             _wsl_path(edp), _wsl_path(workdir),
             str(FREEFEM_TOTAL_TIMEOUT), str(MEMORY_SAMPLE_INTERVAL),
             str(step_timeout), str(BATCH_ITERATIONS - 1)],
            capture_output=True, text=True, errors="replace",
            timeout=FREEFEM_TOTAL_TIMEOUT + 120,
        )
    except subprocess.TimeoutExpired:
        _kill_wsl_orphans(_wsl_path(workdir))
        return {"return_code": -1, "wall_time_s": FREEFEM_TOTAL_TIMEOUT,
                "max_rss_kb": 0, "avg_rss_kb": 0.0, "monitor_epochs": 0,
                "num_mem_samples": 0, "stderr": "WSL monitor timed out"}
    dt = time.time() - t0
    output = (r.stdout or "") + "\n" + (r.stderr or "")
    for line in output.splitlines():
        if line.startswith("BATCH_RESULT:"):
            try:
                return json.loads(line[len("BATCH_RESULT:"):])
            except json.JSONDecodeError:
                pass
    return {"return_code": -99, "wall_time_s": dt,
            "max_rss_kb": 0, "avg_rss_kb": 0.0, "monitor_epochs": 0,
            "num_mem_samples": 0, "stderr": f"Parse failed\n{output[:1000]}"}


def parse_objective(workdir: Path) -> tuple:
    obj_file = workdir / "objectivelinearElasticitybridge.txt"
    if not obj_file.exists():
        return 0, None
    lines = []
    with open(obj_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("iteration") or line.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        return 0, None
    parts = lines[-1].split(",")
    last_val = None
    if len(parts) >= 2:
        try:
            last_val = float(parts[1].strip())
        except ValueError:
            pass
    return len(lines), last_val


def run_one_config(run_id: int, config: dict) -> dict:
    worker_dir = _BATCH_DIR / f"worker_{run_id}"

    row = {
        "run_id": float(run_id),
        "mesh_n": float(config["mesh_n"]),
        "mesh_borders": 0.0, "mesh_total_len": 0.0,
        "mesh_initial_tri_est": 0.0,
        "chi_contrast": float(config["chi_contrast"]),
        "p_degree": str(config["p_degree"]),
        "f": float(config["f"]), "mu": float(config["mu"]),
        "lam": float(config["lam"]), "Afac": float(config["Afac"]),
        "sum_dof_node": 0.0, "is_linear": 0.0,
        "num_state_fields": 0.0, "num_boundary_objects": 0.0,
        "hmin": float(config["hmin"]), "hmax": float(config["hmax"]),
        "v0": float(config["v0"]), "diffusion": float(config["diffusion"]),
        "ff_wall_time_s": 0.0, "avg_time_per_epoch_s": 0.0,
        "ff_max_memory_mb": 0.0, "ff_avg_memory_mb": 0.0,
        "epochs_completed": 0.0, "freefem_rc": 0.0,
        "final_objective": 0.0,
    }

    try:
        setup_worker_dir(worker_dir)
        modify_main_py(worker_dir, config)
        modify_mesh_txt(worker_dir, config)
        mesh_info = parse_mesh_info(worker_dir)
        prob_feat = parse_problem_features(worker_dir)
        row["mesh_borders"] = float(mesh_info["mesh_borders"])
        row["mesh_total_len"] = float(mesh_info["mesh_total_len"])
        row["mesh_initial_tri_est"] = float(mesh_info["mesh_initial_tri_est"])
        row["sum_dof_node"] = float(prob_feat["sum_dof_node"])
        row["is_linear"] = float(1 - prob_feat["is_nonlinear"])
        step_timeout = step_timeout_for(prob_feat)
        row["num_state_fields"] = float(prob_feat["num_state_fields"])
        row["num_boundary_objects"] = float(prob_feat["num_boundary_objects"])

        # AutoFreeFEM — same nonlinear multiplier as the per-epoch
        # watchdog: symbolic edp generation is far slower for nonlinear
        # templates (e.g. det(F)/inverse(C) in Neo-Hookean).
        af_timeout = AUTOFREEFEM_TIMEOUT * (
            NONLINEAR_STEP_MULT if prob_feat["is_nonlinear"] else 1)
        af = run_autofreefem(worker_dir, af_timeout)
        if af["return_code"] != "0":
            row["freefem_rc"] = -99.0
            return row

        edp = worker_dir / "optimize_linearElasticitybridge.edp"
        if not edp.exists():
            row["freefem_rc"] = -98.0
            return row

        ff = run_freefem_with_monitoring(worker_dir, step_timeout)
        if "_error" in ff:
            row["freefem_rc"] = -97.0
            return row

        rc = ff["return_code"]
        wall = ff["wall_time_s"]
        max_rss = ff["max_rss_kb"]
        avg_rss = ff["avg_rss_kb"]
        obj_epochs, last_obj = parse_objective(worker_dir)
        epochs = max(ff.get("monitor_epochs", 0), obj_epochs)
        if epochs > 0:
            # The edp writes the objective at the top of each pass of its
            # `for (iter=1; iter<N; iter++)` loop, i.e. line k = objective
            # before pass k+1.  One more epoch (the final state) is actually
            # completed than there are lines.
            epochs += 1
        avg_ep = wall / epochs if epochs > 0 else 0.0
        row.update({
            "ff_wall_time_s": float(wall),
            "avg_time_per_epoch_s": float(avg_ep),
            "ff_max_memory_mb": float(max_rss / 1024),
            "ff_avg_memory_mb": float(avg_rss / 1024),
            "epochs_completed": float(epochs),
            "freefem_rc": float(rc),
            "final_objective": float(last_obj) if last_obj is not None else 0.0,
        })

    except Exception:
        row["freefem_rc"] = -100.0
        print(f"  [#{run_id}] worker exception:", file=sys.stderr, flush=True)
        traceback.print_exc()

    finally:
        # Cleanup worker directory.  Windows can hold the directory
        # handle a moment longer than the just-exited WSL process —
        # one delayed retry clears empty leftover shells.
        if worker_dir.exists():
            _rmtree_force(worker_dir)
        if worker_dir.exists():
            time.sleep(1.0)
            _rmtree_force(worker_dir)

    return row


# ======================================================================
# main
# ======================================================================

def main():
    if not _PYTHON_EXE.exists():
        print(f"ERROR: Python not found at {_PYTHON_EXE}")
        sys.exit(1)

    print("=" * 70)
    print("  batchrun.py -- run_linearElasticityBridge parameter sweep")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python:  {_PYTHON_EXE}")
    print(f"  Batch dir: {_BATCH_DIR}")
    print("=" * 70)

    print("\n[1] Copying template ...")
    copy_template_files()
    mp = _BATCH_DIR / "main.py"
    code = mp.read_text(encoding="utf-8")
    code = re.sub(r"iterations\s*=\s*\d+",
                  f"iterations={BATCH_ITERATIONS}", code)
    mp.write_text(code, encoding="utf-8")

    # Build grid from sweep lists (fixed families contribute 1 value)
    grid = []
    for values in itertools.product(*_SWEEP_LISTS):
        config = dict(zip(_SWEEP_NAMES, values))
        config["hmin"] = config["hmax"] * config["hmin_ratio"]
        grid.append(config)

    lens = [len(lst) for lst in _SWEEP_LISTS]
    total = 1
    for n in lens:
        total *= n
    print(f"\n  Grid: {total:,} configs "
          f"({'x'.join(str(n) for n in lens)}, {WORKERS} parallel)")

    csv_fields = [
        "run_id",
        "mesh_n", "mesh_borders", "mesh_total_len", "mesh_initial_tri_est",
        "chi_contrast", "p_degree",
        "f", "mu", "lam", "Afac",
        "sum_dof_node", "is_linear",
        "num_state_fields", "num_boundary_objects",
        "hmin", "hmax", "v0", "diffusion",
        "ff_wall_time_s", "avg_time_per_epoch_s",
        "ff_max_memory_mb", "ff_avg_memory_mb",
        "epochs_completed", "freefem_rc", "final_objective",
    ]
    csv_file = open(_BATCH_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields,
                            extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()
    results = []

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {}
            for idx, config in enumerate(grid):
                run_id = idx + 1
                fut = pool.submit(run_one_config, run_id, config)
                futures[fut] = run_id

            for fut in as_completed(futures):
                run_id = futures[fut]
                row = fut.result()
                _write_row(writer, row)
                csv_file.flush()
                results.append((run_id, row))
                print(f"  [#{run_id}] rc={row['freefem_rc']}, "
                      f"epochs={int(row['epochs_completed'])}, "
                      f"ff_time={row['ff_wall_time_s']:.1f}s", flush=True)
    finally:
        csv_file.close()
    results.sort(key=lambda x: x[0])

    print(f"\n{'=' * 105}")
    print(f"  Sweep complete -- {len(results)} runs -> {_BATCH_CSV}")
    print(f"{'=' * 105}")
    hdr = (f"{'#':>4} {'hmax':>8} {'ep':>6} "
           f"{'FF_s':>8} {'mxMB':>7} {'avgMB':>7} {'rc':>5}")
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for _, r in results:
        print(f"{int(r['run_id']):>4} {float(r['hmax']):>8.3f} "
              f"{int(r['epochs_completed']):>6} "
              f"{float(r['ff_wall_time_s']):>8.1f} "
              f"{float(r['ff_max_memory_mb']):>7.1f} "
              f"{float(r['ff_avg_memory_mb']):>7.1f} "
              f"{int(r['freefem_rc']):>5}")

    # Final cleanup sweep: a worker dir whose WSL file handle outlived
    # the per-worker retry is removed here, after every FreeFEM process
    # has exited.
    for d in sorted(_BATCH_DIR.glob("worker_*")):
        if d.is_dir():
            _rmtree_force(d)

    # Remove any template working subdirectories copied into the batch dir
    # (e.g. by an older version) so the folder is clean after the run.
    for sub in ["include", "results", "tex", "tmp"]:
        d = _BATCH_DIR / sub
        if d.exists():
            _rmtree_force(d)

    # Automatically run the analysis (correlation table + heatmap).
    print("\n[2] Analysing results ...")
    try:
        import batch_analyse
        batch_analyse.main()
    except Exception:
        print("  Analysis failed:", file=sys.stderr, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
