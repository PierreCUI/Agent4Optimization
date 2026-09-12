"""batchrun.py — Full Cartesian product sweep (edp-reuse design).

Grid: hmin x hmax x diffusion x v0 x mesh_n x p_degree  (5^5 x 3).

The optimize edp is rendered ONCE per p_degree (3 AutoFreeFEM runs on
the template main.py).  Every grid config then only patches the five
numeric values inside the edp text and runs FreeFEM directly — no
symbolic re-rendering per config.  Output: batch.csv.
"""

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

from parameters import CASE, HMIN, HMAX, DIFFUSION, V0, MESH_N, P_DEGREE

# -- paths ------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BATCH_DIR = Path(__file__).resolve().parent
_RESULTS_SRC = _PROJECT_ROOT / "results" / CASE
_BATCH_CSV = _BATCH_DIR / "batch.csv"
_RENDER_DIR = _BATCH_DIR / "render"

_PYTHON_EXE = Path.home() / "miniconda3" / "envs" / "autofreefem" / "python.exe"
if not _PYTHON_EXE.exists():
    for p in Path.home().rglob("miniconda3/envs/autofreefem/python.exe"):
        _PYTHON_EXE = p
        break

AUTOFREEFEM_TIMEOUT = 120
FREEFEM_TOTAL_TIMEOUT = 3600
MEMORY_SAMPLE_INTERVAL = 0.1
FREEFEM_STEP_TIMEOUT = 300
NONLINEAR_STEP_MULT = 5
BATCH_ITERATIONS = int(os.environ.get("BATCH_ITERATIONS", "10"))
WORKERS = 16  # per-case parallel configs; cases run sequentially


# ======================================================================
# helpers
# ======================================================================


def _wsl_path(path: Path) -> str:
    p = str(path.resolve())
    if len(p) > 2 and p[1:3] == ":\\":
        return f"/mnt/{p[0].lower()}/{p[3:].replace(chr(92), '/')}"
    return p.replace("\\", "/")


def _kill_wsl_orphans(workdir_wsl: str):
    pattern = workdir_wsl + "[/ ]"
    try:
        subprocess.run(
            ["wsl", "-e", "bash"],
            input=f"pkill -f '{pattern}'\n".encode(),
            capture_output=True, timeout=10)
    except Exception:
        pass


def _rmtree_force(path: Path) -> None:
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    if path.exists():
        shutil.rmtree(path, onerror=_onerror)


def _f32(v) -> str:
    if v is None or v == "":
        return ""
    return f"{float(v):.10g}"


_STR_COLS = {"p_degree"}


def _write_row(writer, row: dict) -> None:
    out = {}
    for k, v in row.items():
        if k in _STR_COLS:
            out[k] = str(v) if v is not None else ""
        else:
            out[k] = _f32(v)
    writer.writerow(out)


def _fmt(val) -> str:
    if val == 0:
        return "0"
    s = f"{val:.10g}"
    if "e" not in s and "." not in s:
        s += "."
    return s


def _patch_edp(text: str, config: dict) -> str:
    """Patch the five numeric parameters inside the rendered edp."""
    text = re.sub(r"real v0 = [\d.eE+\-]+;",
                  f"real v0 = {_fmt(config['v0'])};", text)
    text = re.sub(r"real diffusion = [\d.eE+\-]+;",
                  f"real diffusion = {_fmt(config['diffusion'])};", text)
    text = re.sub(r"real hmin = [\d.eE+\-]+;",
                  f"real hmin = {_fmt(config['hmin'])};", text)
    text = re.sub(r"real hmax = [\d.eE+\-]+;",
                  f"real hmax = {_fmt(config['hmax'])};", text)
    text = re.sub(r"int n = \d+;",
                  f"int n = {int(config['mesh_n'])};", text)
    return text


# ======================================================================
# render stage — one edp per p_degree
# ======================================================================

def copy_template_files():
    for p in _BATCH_DIR.iterdir():
        if p.name in ("batchrun.py", "batch.csv", "parameters.py",
                      "wsl_monitor.py", "batch_analyse.py", "results.py",
                      "results.csv", "__pycache__", "render"):
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
        return [n for n in names if n.endswith(".vtu")
                or n in ("include", "results", "tex", "tmp")]

    shutil.copytree(str(_RESULTS_SRC), str(_BATCH_DIR),
                    dirs_exist_ok=True, ignore=_ignore)


def render_edps() -> dict:
    """Render the optimize edp once per p_degree; returns {P: ok}."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from Execution.run_autofreefem import run_autofreefem
    from Execution.runner import _patch_newton_tolerance

    _rmtree_force(_RENDER_DIR)
    _RENDER_DIR.mkdir(parents=True, exist_ok=True)

    status = {}
    template_main = _BATCH_DIR / "main.py"
    base_code = template_main.read_text(encoding="utf-8")
    base_code = re.sub(r"iterations\s*=\s*\d+",
                       f"iterations={BATCH_ITERATIONS}", base_code)

    for pdeg in P_DEGREE:
        out_dir = _RENDER_DIR / pdeg
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  rendering {pdeg} ...", flush=True)
        try:
            code = re.sub(
                r"((?:ScalarField|VectorField)\([^,]+,\s*)P\d+",
                rf"\g<1>{pdeg}", base_code)
            # render into a scratch dir under render/
            scratch = _RENDER_DIR / f"_scratch_{pdeg}"
            _rmtree_force(scratch)
            scratch.mkdir(parents=True, exist_ok=True)
            for f in ("main.py", "mesh.txt"):
                src = _BATCH_DIR / f
                if src.exists():
                    shutil.copy2(src, scratch / f)
            (scratch / "main.py").write_text(code, encoding="utf-8")
            r = run_autofreefem(scratch / "main.py", scratch,
                                python_exe=str(_PYTHON_EXE))
            if r["return_code"] != "0":
                print(f"  {pdeg} render FAILED: {r['stderr'][-400:]}",
                      flush=True)
                status[pdeg] = False
                continue
            edps = sorted(scratch.glob("optimize_*.edp"))
            if not edps:
                print(f"  {pdeg} render produced no optimize edp", flush=True)
                status[pdeg] = False
                continue
            # Newton tolerance patch goes into the stored template
            _patch_newton_tolerance(edps[0])
            # store on BOTH sides: Windows copy for debugging, WSL copy
            # as the native-ext4 template the workers clone from
            edp_text = edps[0].read_text(encoding="utf-8", errors="replace")
            (out_dir / "optimize.edp").write_text(edp_text, encoding="utf-8")
            inc = scratch / "include"
            if inc.exists():
                shutil.copytree(inc, out_dir / "include", dirs_exist_ok=True)
            up = _wsl(
                f"mkdir -p {_WSL_RENDER}/{pdeg}",
                f"cp \"{_wsl_path(edps[0])}\" "
                f"{_WSL_RENDER}/{pdeg}/optimize.edp",
                f"cp -r \"{_wsl_path(scratch)}\"/include "
                f"{_WSL_RENDER}/{pdeg}/include")
            if up.returncode != 0:
                print(f"  {pdeg} WSL upload FAILED: {up.stderr[-300:]}",
                      flush=True)
                status[pdeg] = False
                continue
            _rmtree_force(scratch)
            status[pdeg] = True
            print(f"  {pdeg} rendered OK", flush=True)
        except Exception:
            print(f"  {pdeg} render crashed:", flush=True)
            traceback.print_exc()
            status[pdeg] = False
    return status


def is_nonlinear() -> int:
    ij = _RESULTS_SRC / "initial.json"
    if ij.exists():
        with open(ij, encoding="utf-8") as f:
            ds = json.load(f).get("designed", {})
        if ds.get("state_is_nonlinear"):
            return 1
    return 0


def step_timeout_for(nonlinear: int) -> int:
    if nonlinear:
        return FREEFEM_STEP_TIMEOUT * NONLINEAR_STEP_MULT
    return FREEFEM_STEP_TIMEOUT


def run_freefem_with_monitoring_wsl(wsl_worker: str, step_timeout: int) -> dict:
    if shutil.which("wsl") is None:
        return {"_error": "WSL not found"}
    print(f"    Launching FreeFEM via WSL (step timeout {step_timeout}s) ...",
          flush=True)
    t0 = time.time()
    # script via stdin: argv WSL paths would be mangled by MSYS
    script = (
        f"python3 {_WSL_TOOLS}/wsl_monitor.py "
        f"{wsl_worker}/optimize.edp {wsl_worker} "
        f"{FREEFEM_TOTAL_TIMEOUT} {MEMORY_SAMPLE_INTERVAL} "
        f"{step_timeout} {BATCH_ITERATIONS - 1}\n"
    ).encode()
    try:
        r = subprocess.run(["wsl", "-e", "bash"], input=script,
                           capture_output=True,
                           timeout=FREEFEM_TOTAL_TIMEOUT + 120)
    except subprocess.TimeoutExpired:
        _kill_wsl_orphans(wsl_worker)
        return {"return_code": -1, "wall_time_s": FREEFEM_TOTAL_TIMEOUT,
                "max_rss_kb": 0, "avg_rss_kb": 0.0, "monitor_epochs": 0,
                "num_mem_samples": 0, "stderr": "WSL monitor timed out"}
    dt = time.time() - t0
    output = ((r.stdout or b"").decode(errors="replace")) + "\n" + \
        ((r.stderr or b"").decode(errors="replace"))
    for line in output.splitlines():
        if line.startswith("BATCH_RESULT:"):
            try:
                return json.loads(line[len("BATCH_RESULT:"):])
            except json.JSONDecodeError:
                pass
    return {"return_code": -99, "wall_time_s": dt,
            "max_rss_kb": 0, "avg_rss_kb": 0.0, "monitor_epochs": 0,
            "num_mem_samples": 0, "stderr": f"Parse failed\n{output[:1000]}"}


def parse_objective_wsl(wsl_worker: str) -> tuple:
    r = _wsl(f"cat {wsl_worker}/objective*.txt 2>/dev/null | tail -1")
    if r.returncode != 0 or not r.stdout.strip():
        return 0, None
    lines = [ln for ln in r.stdout.splitlines()
             if ln.strip() and not ln.startswith("iteration")]
    if not lines:
        return 0, None
    parts = lines[-1].split(",")
    last_val = None
    if len(parts) >= 2:
        try:
            last_val = float(parts[1].strip())
        except ValueError:
            pass
    # count total data lines
    r2 = _wsl(f"cat {wsl_worker}/objective*.txt 2>/dev/null | grep -v iteration | grep -v '^$' | wc -l")
    try:
        n = int(r2.stdout.strip() or 0)
    except ValueError:
        n = 0
    return n, last_val


# ======================================================================
# WSL-side experiment paths (native ext4 — avoids slow /mnt IO)
# ======================================================================

_WSL_ROOT = "/home/zpcui/Agent4FreeFem"
_WSL_BATCH = f"{_WSL_ROOT}/batches/{CASE}"
_WSL_RENDER = f"{_WSL_BATCH}/render"
_WSL_TOOLS = f"{_WSL_ROOT}/tools"


def _wsl(*args: str) -> subprocess.CompletedProcess:
    """Run commands inside WSL — each arg is one bash line, piped to
    bash through stdin (no shell-quoting / list2cmdline mangling).
    Bytes are passed so Windows never rewrites the line endings."""
    script = ("set -e\n" + "\n".join(args) + "\n").encode()
    r = subprocess.run(["wsl", "-e", "bash"], input=script,
                       capture_output=True, timeout=300)
    return subprocess.CompletedProcess(
        r.args, r.returncode,
        (r.stdout or b"").decode(errors="replace"),
        (r.stderr or b"").decode(errors="replace"))


# ======================================================================
# single-config runner (called by worker threads)
# ======================================================================

def run_one_config(run_id: int, config: dict, nonlinear: int) -> dict:
    wsl_worker = f"{_WSL_BATCH}/worker_{run_id}"

    row = {"run_id": float(run_id)}
    for k in ("hmin", "hmax", "diffusion", "v0", "mesh_n", "p_degree"):
        row[k] = config[k]
    row.update({
        "epochs_completed": 0.0, "freefem_rc": 0.0,
        "ff_wall_time_s": 0.0, "avg_time_per_epoch_s": 0.0,
        "ff_max_memory_mb": 0.0, "ff_avg_memory_mb": 0.0,
        "final_objective": 0.0,
    })

    try:
        # fresh worker on the WSL filesystem (local cp — fast)
        _wsl(f"rm -rf {wsl_worker}",
             f"mkdir -p {wsl_worker}",
             f"cp -r {_WSL_RENDER}/{config['p_degree']}/* {wsl_worker}/",
             f"mkdir -p {wsl_worker}/tmp {wsl_worker}/results")

        # patch the five numeric parameters inside the edp (WSL-side)
        r = _wsl(
            f"python3 {_WSL_TOOLS}/patch_edp.py {wsl_worker}/optimize.edp "
            f"{_fmt(config['v0'])} {_fmt(config['diffusion'])} "
            f"{_fmt(config['hmin'])} {_fmt(config['hmax'])} "
            f"{int(config['mesh_n'])}")
        if r.returncode != 0:
            row["freefem_rc"] = -96.0
            return row

        step_timeout = step_timeout_for(nonlinear)
        ff = run_freefem_with_monitoring_wsl(wsl_worker, step_timeout)
        if "_error" in ff:
            row["freefem_rc"] = -97.0
            return row

        rc = ff["return_code"]
        wall = ff["wall_time_s"]
        max_rss = ff["max_rss_kb"]
        avg_rss = ff["avg_rss_kb"]
        obj_epochs, last_obj = parse_objective_wsl(wsl_worker)
        epochs = max(ff.get("monitor_epochs", 0), obj_epochs)
        if epochs > 0:
            epochs += 1
        avg_ep = wall / epochs if epochs > 0 else 0.0
        row.update({
            "ff_wall_time_s": float(wall),
            "avg_time_per_epoch_s": round(float(avg_ep), 4),
            "ff_max_memory_mb": round(float(max_rss / 1024), 4),
            "ff_avg_memory_mb": round(float(avg_rss / 1024), 4),
            "epochs_completed": float(epochs),
            "freefem_rc": float(rc),
            "final_objective": (round(float(last_obj), 4)
                                 if last_obj is not None else 0.0),
        })

    except Exception:
        row["freefem_rc"] = -100.0
        print(f"  [#{run_id}] worker exception:", file=sys.stderr, flush=True)
        traceback.print_exc()

    finally:
        # WSL-side cleanup (local rm — fast)
        _wsl(f"rm -rf {wsl_worker}")

    return row


# ======================================================================
# main
# ======================================================================

def _prepare_wsl() -> bool:
    """Create the WSL-side directories and upload the two tool scripts."""
    tool_src = _BATCH_DIR / "wsl_monitor.py"
    if not tool_src.exists():
        print("ERROR: wsl_monitor.py missing in batch dir")
        return False
    r = _wsl(f"mkdir -p {_WSL_ROOT} {_WSL_BATCH} {_WSL_RENDER}",
             f"rm -rf {_WSL_TOOLS}",
             f"mkdir -p {_WSL_TOOLS}",
             f"cp \"{_wsl_path(tool_src)}\" {_WSL_TOOLS}/wsl_monitor.py")
    # patch_edp.py travels from the repo temp/ directory
    patch_src = _PROJECT_ROOT / "temp" / "patch_edp.py"
    if patch_src.exists():
        r2 = _wsl(f"cp \"{_wsl_path(patch_src)}\" {_WSL_TOOLS}/patch_edp.py")
        if r2.returncode != 0:
            print(f"ERROR: patch_edp.py upload failed: {r2.stderr[-300:]}")
            return False
    else:
        print("ERROR: temp/patch_edp.py missing")
        return False
    if r.returncode != 0:
        print(f"ERROR: WSL prepare failed: {r.stderr[-300:]}")
        return False
    return True


def main():
    if not _PYTHON_EXE.exists():
        print(f"ERROR: Python not found at {_PYTHON_EXE}")
        sys.exit(1)

    print("\n[0] Preparing WSL-side experiment space ...")
    if not _prepare_wsl():
        sys.exit(1)

    print("\n[1] Copying template ...")
    copy_template_files()

    print("[2] Rendering edps per p_degree ...")
    render_status = render_edps()
    ok_ps = [p for p, ok in render_status.items() if ok]
    if not ok_ps:
        print("ERROR: no p_degree rendered — abort.")
        sys.exit(1)
    print(f"  rendered: {ok_ps}")

    nonlinear = is_nonlinear()

    grid = [(hmin, hmax, diff, v0, n, p)
            for hmin in HMIN for hmax in HMAX
            for diff in DIFFUSION for v0 in V0
            for n in MESH_N for p in ok_ps]
    # smoke-test cap: BATCHRUN_LIMIT=N runs only the first N configs
    _limit = os.environ.get("BATCHRUN_LIMIT")
    if _limit:
        grid = grid[: int(_limit)]
    print("=" * 70)
    print(f"  batchrun.py -- {CASE} Cartesian sweep")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python:  {_PYTHON_EXE}")
    print(f"  Configs: {len(grid)} ({WORKERS} workers, "
          f"nonlinear={nonlinear})")
    print("=" * 70)

    csv_fields = ["run_id", "hmin", "hmax", "diffusion", "v0", "mesh_n",
                  "p_degree", "epochs_completed", "freefem_rc",
                  "ff_wall_time_s", "avg_time_per_epoch_s",
                  "ff_max_memory_mb", "ff_avg_memory_mb",
                  "final_objective"]
    csv_file = open(_BATCH_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields,
                            extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()
    results = []

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {}
            for idx, vals in enumerate(grid):
                run_id = idx + 1
                config = dict(zip(("hmin", "hmax", "diffusion", "v0",
                                   "mesh_n", "p_degree"), vals))
                fut = pool.submit(run_one_config, run_id, config, nonlinear)
                futures[fut] = run_id

            for fut in as_completed(futures):
                run_id = futures[fut]
                row = fut.result()
                _write_row(writer, row)
                csv_file.flush()
                results.append((run_id, row))
                if run_id % 200 == 0 or row["freefem_rc"] not in (0.0,):
                    print(f"  [#{run_id}] rc={row['freefem_rc']}, "
                          f"epochs={int(row['epochs_completed'])}, "
                          f"ff_time={row['ff_wall_time_s']:.1f}s", flush=True)
    finally:
        csv_file.close()
    results.sort(key=lambda x: x[0])

    print(f"\n{'=' * 105}")
    print(f"  Batch complete -- {len(results)} runs -> {_BATCH_CSV}")
    print(f"{'=' * 105}")
    n_ok = sum(1 for _, r in results if r["freefem_rc"] == 0.0)
    print(f"  rc=0 runs: {n_ok}/{len(results)}")

    for d in sorted(_BATCH_DIR.glob("worker_*")):
        if d.is_dir():
            _rmtree_force(d)
    for sub in ["include", "results", "tex", "tmp", "main.py", "mesh.txt",
                "initial.json", "mesh.json"]:
        p = _BATCH_DIR / sub
        if p.exists():
            if p.is_dir():
                _rmtree_force(p)
            else:
                try:
                    p.unlink()
                except OSError:
                    pass

    print("\n[3] Analysing results ...")
    try:
        import batch_analyse
        batch_analyse.main()
    except Exception:
        print("  Analysis failed:", file=sys.stderr, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
