"""Run inside WSL: spawn FreeFEM, monitor memory + epoch progress.

Memory is sampled as the total RSS of FreeFEM *and all descendant
processes* (MMG, advect, ... are spawned by FreeFEM via system() and
would otherwise be missed — the same convention as the AutoFreeFEM
stage's psutil.children(recursive=True) on the Windows side).

FreeFEM's output goes to files (ff_stdout.log / ff_stderr.log), never
pipes — a verbose run would fill a pipe buffer and block.

Outcome handling (every path is bounded; WSL waitpid hangs are worked
around via /proc state checks, never via blocking polls):
  1. normal completion          -> rc = FreeFEM's exit code (usually 0)
  2. exits with an error        -> rc = exit code if readable, else -6
                                   (exited, code unreadable)
  3. hangs: no-CPU stall (e.g. a system() wait hang) -> kill after 120 s
           no-new-epoch stall   -> kill after step_timeout
           total-timeout exceed -> kill after total_timeout
  A stall kill reports rc=0 when all expected epochs were already done,
  rc=-2 otherwise.  A daemon self-destruct timer bounds the monitor
  itself at total_timeout + 60 s.
"""
import json
import os
import subprocess
import sys
import threading
import time

if len(sys.argv) < 7:
    print("usage: wsl_monitor.py <edp> <workdir> <total_timeout> "
          "<sample_interval> <step_timeout> <expected_epochs>",
          file=sys.stderr)
    sys.exit(2)

edp_file = sys.argv[1]
workdir = sys.argv[2]
total_timeout = int(sys.argv[3])
sample_interval = float(sys.argv[4])
step_timeout = int(sys.argv[5])
expected_epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 0

mem_samples = []
mem_lock = threading.Lock()
monitor_done = threading.Event()


def _rss_kb(pid) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def _descendant_pids(pid) -> set:
    """PIDs of all descendant processes (children, grandchildren, ...)."""
    pids = set()
    stack = [pid]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/task/{p}/children") as f:
                kids = [int(x) for x in f.read().split()]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
        for k in kids:
            if k not in pids:
                pids.add(k)
                stack.append(k)
    return pids


def monitor_memory(pid):
    while not monitor_done.is_set():
        try:
            total = _rss_kb(pid)
            for cp in _descendant_pids(pid):
                total += _rss_kb(cp)
            with mem_lock:
                mem_samples.append(total)
        except (FileNotFoundError, ProcessLookupError):
            break
        except Exception:
            pass
        monitor_done.wait(sample_interval)


epoch_times = []
epoch_lock = threading.Lock()
epoch_done = threading.Event()
obj_file = os.path.join(workdir, "objectivelineardiffusion.txt")


def _count_obj_lines() -> int:
    """Number of completed epoch lines in the objective file."""
    try:
        if os.path.exists(obj_file):
            with open(obj_file) as f:
                return len([
                    ln for ln in f
                    if ln.strip()
                    and not ln.startswith("iteration")
                    and not ln.startswith("#")
                ])
    except Exception:
        pass
    return 0


def _is_zombie(pid) -> bool:
    """True if the process has exited (unreaped zombie or /proc entry
    gone).  WSL's waitpid can hang, so liveness must not rely on
    proc.poll() — read the state field from /proc/<pid>/stat instead."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().split(") ", 1)
            return len(rest) > 1 and rest[1][0] == "Z"
    except FileNotFoundError:
        return True  # no /proc entry: the process is gone
    except Exception:
        return False


def _try_poll(proc, timeout: float = 5.0):
    """proc.poll() with a hard timeout — WSL waitpid can block forever.

    Returns the exit code, or None if waitpid hangs."""
    result = []

    def _poll():
        result.append(proc.poll())

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    t.join(timeout)
    return result[0] if result else None


def _cpu_seconds(pid) -> float:
    """Cumulative CPU time of a process, in seconds."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split(") ", 1)[1].split()
        # after ") ": parts[0] is the state, utime=parts[11],
        # stime=parts[12] (100 ticks per second)
        return (int(parts[11]) + int(parts[12])) / 100.0
    except Exception:
        return 0.0


def _read_log(name: str, limit: int = 4000) -> str:
    """Tail of one of FreeFEM's log files."""
    try:
        with open(os.path.join(workdir, name), encoding="utf-8",
                  errors="replace") as f:
            return f.read()[-limit:]
    except OSError:
        return ""


def monitor_epochs():
    last_count = 0
    while not epoch_done.is_set():
        try:
            count = _count_obj_lines()
            if count > last_count:
                with epoch_lock:
                    epoch_times.append((count, time.time()))
                last_count = count
        except Exception:
            pass
        epoch_done.wait(1.0)


os.chdir(workdir)
# Make advect (built from ISCDtoolbox/Advection) findable by FreeFEM's
# system() calls — it installs to ~/bin with a shared lib in ~/lib.
_home_bin = os.path.expanduser("~/bin")
if os.path.isdir(_home_bin) and _home_bin not in os.environ["PATH"]:
    os.environ["PATH"] = _home_bin + ":" + os.environ["PATH"]
_home_lib = os.path.expanduser("~/lib")
if os.path.isdir(_home_lib):
    os.environ["LD_LIBRARY_PATH"] = _home_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

t_start = time.time()

# FreeFEM's output goes to FILES, not pipes: verbose runs (MMG warning
# spam) would fill a pipe buffer and block FreeFEM mid-run.
_ff_out = open(os.path.join(workdir, "ff_stdout.log"), "w")
_ff_err = open(os.path.join(workdir, "ff_stderr.log"), "w")
proc = subprocess.Popen(
    ["FreeFem++", edp_file],
    stdout=_ff_out,
    stderr=_ff_err,
)
threading.Thread(target=monitor_memory, args=(proc.pid,), daemon=True).start()
threading.Thread(target=monitor_epochs, daemon=True).start()

# Last-resort self-destruct: WSL waitpid can hang on defunct children,
# so termination must never rely on process reaping.  Daemon, so the
# monitor itself can exit normally without waiting for the timer.
_timer = threading.Timer(total_timeout + 60, lambda: os._exit(3))
_timer.daemon = True
_timer.start()

deadline = t_start + total_timeout
last_progress = 0
step_deadline = t_start + 2 * step_timeout
watchdog_rc = None
finished = 0
last_epoch_ts = None

zombie_exit = False
last_cpu = _cpu_seconds(proc.pid)
last_active = time.time()
idle_kill = False
while True:
    # Liveness via /proc only — NEVER call proc.poll() in the hot loop:
    # WSL's waitpid can block forever on an exited child.
    if _is_zombie(proc.pid):
        zombie_exit = True
        break
    time.sleep(1)
    cpu = _cpu_seconds(proc.pid)
    if cpu > last_cpu:
        last_cpu = cpu
        last_active = time.time()
    with epoch_lock:
        progress = len(epoch_times)
    if progress > last_progress:
        last_progress = progress
        step_deadline = time.time() + step_timeout
    now = time.time()
    if now > deadline:
        proc.kill()
        watchdog_rc = -1
        break
    if step_timeout >= 0 and now > step_deadline:
        with epoch_lock:
            finished = len(epoch_times)
            last_epoch_ts = epoch_times[-1][1] if epoch_times else None
        # Progress records can lag the file if FreeFEM writes lines in
        # bursts — count the objective lines directly as ground truth.
        finished = max(finished, _count_obj_lines())
        if expected_epochs and finished >= expected_epochs:
            # All optimisation epochs completed; FreeFEM is stuck in the
            # post-completion remeshing/plotting stage.  Count as success
            # with wall time up to the last epoch.
            proc.kill()
            watchdog_rc = 0
        else:
            proc.kill()
            watchdog_rc = -2  # stuck mid-optimisation — honest failure
        break
    # CPU-idle stall: 120 s without any CPU growth means FreeFEM is
    # blocked (e.g. WSL waitpid hang inside its own system() call) —
    # kill it early.  Same completion semantics as the step watchdog.
    # (Initial mesh setup keeps burning CPU, so last_active stays fresh.)
    if now - last_active > 120:
        idle_kill = True
        if expected_epochs and _count_obj_lines() >= expected_epochs:
            proc.kill()
            watchdog_rc = 0
        else:
            proc.kill()
            watchdog_rc = -2
        break

real_rc = None
if watchdog_rc is None and zombie_exit:
    # FreeFEM exited — try to recover its real exit code, but WSL waitpid
    # can hang, so use a timed poll and fall back to the objective file.
    real_rc = _try_poll(proc)
    if real_rc is None:
        if expected_epochs and _count_obj_lines() >= expected_epochs:
            with epoch_lock:
                last_epoch_ts = (epoch_times[-1][1]
                                 if epoch_times else time.time())
            watchdog_rc = 0
        else:
            watchdog_rc = -6  # exited with an error, unreadable rc
        try:
            proc.kill()  # no-op on a zombie, harmless
        except ProcessLookupError:
            pass  # /proc entry already gone
if watchdog_rc is not None:
    # Watchdog path: never block on reaping — WSL waitpid can hang.
    try:
        proc.wait(timeout=10)
    except Exception:
        pass

_ff_out.close()
_ff_err.close()

if watchdog_rc == -1:
    return_code = -1
    wall_time = time.time() - t_start
    stderr_str = _read_log("ff_stderr.log") + "\nTOTAL TIMEOUT"
elif watchdog_rc == -2:
    return_code = -2
    wall_time = time.time() - t_start
    stderr_str = _read_log("ff_stderr.log")
    if idle_kill:
        stderr_str += (
            "\nCPU IDLE STALL: no CPU growth for 120 s "
            "(FreeFEM blocked, e.g. WSL waitpid hang inside its own "
            "system() call)."
        )
    else:
        stderr_str += (
            f"\nSTEP TIMEOUT: no new epoch for {step_timeout}s "
            f"(last epoch: {last_progress}). Likely MMG hang."
        )
elif watchdog_rc == 0:
    return_code = 0
    wall_time = round((last_epoch_ts - t_start) if last_epoch_ts else
                      (time.time() - t_start), 2)
    stderr_str = _read_log("ff_stderr.log")
    stderr_str += (
        f"\nCOMPLETION: {finished} epochs done "
        f"(expected {expected_epochs}); killed FreeFEM in the "
        f"post-completion stage (or after an unreadable exit)."
    )
elif watchdog_rc == -6:
    return_code = -6
    wall_time = time.time() - t_start
    stderr_str = _read_log("ff_stderr.log")
    stderr_str += (
        "\nFREEFEM EXITED WITH AN ERROR; exit code unreadable "
        "(WSL waitpid hang). See stderr log above."
    )
else:
    return_code = real_rc if real_rc is not None else proc.returncode
    wall_time = time.time() - t_start
    stderr_str = _read_log("ff_stderr.log")

epoch_done.set()
monitor_done.set()

with mem_lock:
    rss = list(mem_samples)
with epoch_lock:
    records = list(epoch_times)

result = {
    "return_code": return_code,
    "wall_time_s": round(wall_time, 2),
    "max_rss_kb": max(rss) if rss else 0,
    "avg_rss_kb": round(sum(rss) / len(rss), 1) if rss else 0.0,
    "monitor_epochs": len(records),
    "num_mem_samples": len(rss),
    "stderr": stderr_str[:2000],
}
print("BATCH_RESULT:" + json.dumps(result))
