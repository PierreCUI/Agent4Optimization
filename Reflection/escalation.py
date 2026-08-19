"""
Escalation controller for the Reflection module.

Pure hard logic — no LLM calls.  Decides whether to continue with
local_fix, escalate to rewrite, or give up entirely.

Decision is based on:
- Total iteration count vs MAX_REFLECTION_TIMES.
- Consecutive identical (coarse, fine) errors vs STALE_THRESHOLD.
- Total rewrite count vs MAX_REWRITES.
"""


def decide(
    reflection_history: list[dict],
    coarse: str,
    fine: str,
    stale_threshold: int = 5,
    max_rewrites: int = 5,
) -> str:
    """
    Decide the fix strategy for the current error.

    Args:
        reflection_history: List of per-iteration records, each with keys
            {coarse, fine, action, fix_rationale, outcome}.
        coarse: Current coarse category.
        fine: Current fine sub-category.
        stale_threshold: Consecutive same-error count before rewrite
            (REFLECTION_STALE_THRESHOLD).
        max_rewrites: Max total rewrites before giving up
            (REFLECTION_MAX_REWRITES).

    Returns:
        "local_fix" | "rewrite" | "give_up"

    Note: the total-iteration cap is enforced by the caller's loop,
    not by this function.
    """
    # --- count consecutive identical errors (same coarse + fine) ---
    same_count = 0
    same_coarse_count = 0
    for h in reversed(reflection_history):
        if h.get("coarse") == coarse and h.get("fine") == fine:
            same_count += 1
        else:
            break
    for h in reversed(reflection_history):
        if h.get("coarse") == coarse:
            same_coarse_count += 1
        else:
            break

    # Include current error in the counts
    same_count += 1
    same_coarse_count += 1

    # --- stale threshold triggers rewrite ---
    # Also trigger if stuck on the same coarse category with
    # alternating fine labels for too long (oscillation blind spot).
    stale = same_count >= stale_threshold or same_coarse_count >= stale_threshold * 2
    if stale:
        rewrites = sum(1 for h in reflection_history if h.get("action") == "rewrite")
        if rewrites < max_rewrites:
            return "rewrite"
        else:
            return "give_up"

    # --- default: continue with local (conservative) fix ---
    return "local_fix"
