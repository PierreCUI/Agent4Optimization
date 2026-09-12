"""
Escalation controller for the Reflection module.

Pure hard logic — no LLM calls.  Decides whether to continue with a
local_fix or escalate to a rewrite.

Decision is based on:
- Consecutive identical (coarse, fine) errors vs STALE_THRESHOLD.
- Total occurrences of the current coarse category (oscillation between
  different categories also counts).

Giving up is decided ONLY by the caller's loop cap (MAX_REFLECTION_TIMES)
— this function never returns "give_up".
"""


def decide(
    reflection_history: list[dict],
    coarse: str,
    fine: str,
    stale_threshold: int = 5,
) -> str:
    """
    Decide the fix strategy for the current error.

    Args:
        reflection_history: List of per-iteration records, each with keys
            {coarse, fine, action, fix_rationale, outcome}.
        coarse: Current coarse category.
        fine: Current fine sub-category.
        stale_threshold: Same-error count before rewrite
            (REFLECTION_STALE_THRESHOLD).

    Returns:
        "local_fix" | "rewrite"

    Note: the total-iteration cap is enforced by the caller's loop,
    not by this function.
    """
    # --- count identical errors (same coarse + fine) ---
    same_count = 0
    for h in reversed(reflection_history):
        if h.get("coarse") == coarse and h.get("fine") == fine:
            same_count += 1
        else:
            break

    # Total occurrences of the current coarse category across the WHOLE
    # history — catches both consecutive repeats (exactly the old
    # consecutive counter) AND oscillation between DIFFERENT coarse
    # categories (freefem_error ↔ autofreefem_error), where consecutive
    # counters never reach the threshold and the loop burns all
    # MAX_REFLECTION_TIMES on local_fix without ever rewriting.
    #
    # Records whose fix DELIVERED a design (MinTh.vtu generated) are
    # progress, not staleness — a chain of DISTINCT bugs fixed one by one
    # must not escalate to a rewrite just because they share a coarse
    # category.  The SAME fine error recurring after a delivered design is
    # NOT progress: the fix did not hold, and that is exactly the
    # persistence this counter exists to catch.  (Both counters therefore
    # treat a delivered record the same way — it counts when the error is
    # the current one, and never when it is a different one — so the
    # consecutive and the total counter agree on every record class.)
    # "FreeFEM OK but no MinTh.vtu" is never progress either: the
    # optimization ran but the design was never delivered.
    same_coarse_count = 0
    for h in reflection_history:
        if h.get("coarse") != coarse:
            continue
        outcome = str(h.get("outcome", ""))
        if "MinTh.vtu generated" in outcome and h.get("fine") != fine:
            continue
        same_coarse_count += 1

    # Include current error in the counts
    same_count += 1
    same_coarse_count += 1

    # --- stale threshold triggers rewrite ---
    # Giving up is the caller's loop cap (MAX_REFLECTION_TIMES) alone;
    # a stale error always escalates to rewrite.
    drift = max(same_count, same_coarse_count)
    if drift >= stale_threshold:
        return "rewrite"

    # --- default: continue with local (conservative) fix ---
    return "local_fix"
