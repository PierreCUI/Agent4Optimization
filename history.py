"""Shared operation log — results/<problem>/history.txt.

A RECORD ONLY: history.txt is never reflection evidence (it stays
blacklisted from the reflection context).  It records, for every round
of the four stages (Initialization -> Generation -> Execution ->
Reflection): which function is working (for a repair, which round),
what went in and what came out — IN FULL.  The file is allowed to grow
very large by design.

The writer is thread-safe: the reflection probe threads append
concurrently.
"""

import threading
import warnings
from datetime import datetime
from pathlib import Path

_LOCK = threading.Lock()


def init_log(history_path, title: str = "") -> None:
    """Create/truncate the log and write the header.

    One pipeline run = one log: re-running the same problem starts
    fresh (like reflection.txt) instead of accumulating stale records.
    """
    try:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{'=' * 70}\n")
                f.write(f"OPERATION LOG — {title or path.parent.name}\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 70}\n")
    except OSError as e:
        warnings.warn(f"history log init failed: {e}")


def record(history_path, kind: str, label: str = "", input_text: str = "",
           output_text: str = "", images=None) -> None:
    """Append one record (best effort — the log must never break a run).

    *kind* names the working function (e.g. "LLM vision",
    "run_problem"); *label* adds the stage/round context (e.g.
    "Reflection round 3: design_fix probe 2"); *input_text* /
    *output_text* are written IN FULL — the log is a complete record of
    the pipeline's work and may grow large by design.
    """
    if not history_path:
        return
    try:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label_line = f" | {label}" if label else ""
        with _LOCK:
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== {stamp} | {kind}{label_line} ===\n")
                if input_text:
                    f.write("--- input ---\n")
                    f.write(str(input_text).rstrip("\n") + "\n")
                if images:
                    names = [Path(p).name for p in images[:5]]
                    f.write(
                        f"--- images ---\n{len(images)} image(s): {names}"
                        f"{' ...' if len(images) > 5 else ''}\n"
                    )
                if output_text:
                    f.write("--- output ---\n")
                    f.write(str(output_text).rstrip("\n") + "\n")
    except OSError as e:
        warnings.warn(f"history record failed: {e}")


def set_label(agent, label: str) -> None:
    """Tag *agent* so every subsequent LLM record carries *label*.

    The label names the stage and the round (for a repair, WHICH
    round).  A best-effort no-op on objects that carry no history (e.g.
    test stubs).
    """
    try:
        agent.history_label = label
    except Exception:
        pass
