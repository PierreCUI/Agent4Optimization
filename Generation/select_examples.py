"""
LLM-based example selection for code generation.

Candidates are the FULL initial.json / mesh.json specs of the database
examples — no summaries.  ONE scoring round: each candidate gets a
0-100 relevance score (batched by MAX_INPUT_TOKENS when the full specs
do not fit in one call); every candidate at or above
configs.EXAMPLE_SCORE_THRESHOLD is kept, best first, until the NEXT
example's full .py would exceed the token budget — an example is never
truncated or partially sent.

When a database folder lacks its JSONs, they are generated on the spot
via Initialization's spec loop (a safeguard — Initialization normally
produces them; this only protects against a missing file).
"""

import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import (  # noqa: E402
    EXAMPLE_SCORE_THRESHOLD,
    MAX_INPUT_TOKENS,
    TOKEN_BUDGET_RATIO,
)
from history import set_label  # noqa: E402
from utils import parser_json  # noqa: E402

# ======================================================================
# helpers
# ======================================================================


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _first_py(ex_dir: Path):
    """The first .py file of an example directory, or None."""
    return next((p for p in ex_dir.iterdir() if p.suffix == ".py"), None)


# ======================================================================
# LLM scoring (single round)
# ======================================================================

_SCORER_SYSTEM = """\
    You are an example scorer for a code-generation system.  You are given a
    TARGET problem (its initial.json and mesh.json specs) and a list of
    candidate examples (each with its full initial.json and mesh.json
    specs).

    Your task: assign each candidate a relevance score from 0 to 100, where:
      100 = perfectly relevant (same physics, same dimension, same field types)
      0   = completely irrelevant

    Prioritise: problem_type (shape vs topology optimisation), dimension,
    field types and number of fields, linear vs nonlinear, physics domain
    (elasticity, diffusion, thermal, fluid, ...), mesh complexity (number
    of labels), number of computational domains and interfaces.

    Return exactly one JSON object:
    {{"scores": {{"name1": 95, "name2": 70, ...}}, "reasoning": "one sentence"}}
    Score every candidate — do not omit any.
"""

_SCORER_USER = """\
    Target problem (initial.json + mesh.json):
    {target_json}

    Candidate examples (initial.json + mesh.json each):
    {candidates_json}

    Return the scoring JSON.
"""


def _load_candidate_specs(agent, examples_dir: Path) -> list:
    """Load the FULL initial/mesh specs of every example folder.

    Missing JSONs are generated via Initialization's spec loop (a
    safeguard — Initialization normally produces them) and written back
    so the next run finds them."""
    from Initialization import collect_input_files, run_spec_generation

    candidates = []
    if not examples_dir.is_dir():
        warnings.warn(f"select_examples: examples directory {examples_dir} not found.")
        return candidates
    for folder in sorted(examples_dir.iterdir()):
        if not folder.is_dir():
            continue
        # temp/ is the pipeline's staged-run scratch inside a case
        # folder — never an example candidate (it lacks the spec JSONs
        # and would trigger the on-the-spot safeguard generation).
        if folder.name == "temp":
            continue
        init_p, mesh_p = folder / "initial.json", folder / "mesh.json"
        try:
            if init_p.exists() and mesh_p.exists():
                initial = json.loads(init_p.read_text(encoding="utf-8"))
                mesh = json.loads(mesh_p.read_text(encoding="utf-8"))
            else:
                warnings.warn(
                    f"select_examples: {folder.name} lacks initial.json / "
                    f"mesh.json — generating via Initialization (safeguard)."
                )
                inputs = collect_input_files(folder)
                if not inputs:
                    warnings.warn(
                        f"select_examples: {folder.name} has no input files "
                        f"— skipped."
                    )
                    continue
                specs = run_spec_generation(
                    agent, inputs, problem_name=folder.name, output_dir=folder
                )
                initial = specs.get("initial") or {}
                mesh = specs.get("mesh") or {}
                if (specs.get("_passed") is not True
                        or not initial.get("designed")
                        or not mesh.get("designed")):
                    warnings.warn(
                        f"select_examples: {folder.name} spec generation "
                        f"failed — skipped."
                    )
                    continue
                init_p.write_text(
                    json.dumps(initial, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                mesh_p.write_text(
                    json.dumps(mesh, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as e:
            warnings.warn(f"select_examples: {folder.name} specs unusable: {e}")
            continue
        candidates.append({"name": folder.name, "initial": initial, "mesh": mesh})
    return candidates


def _build_batches(target_json: str, candidates: list) -> list:
    """Split *candidates* into batches that each fit within
    MAX_INPUT_TOKENS (the target JSON rides along in every batch)."""
    target_tokens = _estimate_tokens(target_json)
    batches = []
    current_batch = []
    current_tokens = target_tokens + 200

    for c in candidates:
        c_tokens = _estimate_tokens(json.dumps(c, ensure_ascii=False))
        if current_tokens + c_tokens > MAX_INPUT_TOKENS and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = target_tokens + 200
        current_batch.append(c)
        current_tokens += c_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


def _score_batch(agent, target_json: str, batch: list) -> dict:
    """One scoring call on a batch — returns {name: score}; empty dict
    on any failure (that batch then simply contributes no candidates)."""
    candidates_json = json.dumps(batch, ensure_ascii=False)
    agent.set_prompt_list([
        {"role": "system", "content": _SCORER_SYSTEM},
        {"role": "user", "content": _SCORER_USER},
    ])
    raw = agent.invoke(
        {
            "target_json": target_json,
            "candidates_json": candidates_json,
        }
    )
    try:
        result = json.loads(parser_json(raw))
    except Exception as e:
        warnings.warn(f"select_examples scoring JSON parse failed: {e}")
        return {}
    if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
        warnings.warn(
            f"select_examples scoring returned unusable result "
            f"({type(result).__name__}) — skipping this batch."
        )
        return {}
    scores = result["scores"]
    out = {}
    for name, score in scores.items():
        try:
            out[str(name)] = float(score)
        except (TypeError, ValueError):
            pass
    missing = sorted({c["name"] for c in batch} - set(out))
    if missing:
        warnings.warn(
            f"select_examples: {len(missing)} candidate(s) returned without "
            f"a score ({missing}) — they are dropped."
        )
    return out


# ======================================================================
# main entry point
# ======================================================================


def select_examples(
    agent, problem_spec: dict, mesh_spec: dict = None,
    examples_dir: Path = None, trace: list = None,
) -> list:
    """
    Single-round LLM example selection.

    Every candidate (full initial.json + mesh.json) is scored 0-100 in
    ONE scoring round (batched by MAX_INPUT_TOKENS when needed);
    candidates at or above configs.EXAMPLE_SCORE_THRESHOLD are kept,
    best first, until the NEXT example's full .py would exceed
    TOKEN_BUDGET_RATIO x MAX_INPUT_TOKENS — never truncating an example.

    If *trace* is provided (a list), the selection state is appended as
    a dict ``{"round": N, "action": "...", "count": int, "names": [...]}``.
    """
    if examples_dir is None:
        examples_dir = Path(__file__).resolve().parent.parent / "database" / "freefem_examples"

    target_json = json.dumps(
        {"initial": problem_spec, "mesh": mesh_spec or {}},
        ensure_ascii=False,
    )

    candidates = _load_candidate_specs(agent, examples_dir)
    if not candidates:
        warnings.warn("select_examples: no candidate specs found")
        return []

    batches = _build_batches(target_json, candidates)
    scores = {}
    for bi, batch in enumerate(batches):
        set_label(
            agent,
            f"Generation/select_examples scoring batch {bi + 1}/{len(batches)}",
        )
        scores.update(_score_batch(agent, target_json, batch))

    qualified = sorted(
        (n for n in scores if scores[n] >= EXAMPLE_SCORE_THRESHOLD),
        key=lambda n: scores[n],
        reverse=True,
    )
    if trace is not None:
        trace.append(
            {
                "round": 1,
                "action": (
                    f"Scoring ({len(batches)} batch(es)) — "
                    f"{len(qualified)} above threshold {EXAMPLE_SCORE_THRESHOLD}"
                ),
                "count": len(qualified),
                "names": list(qualified),
            }
        )

    # Pack the best first, by the FULL .py size of each example, until
    # the next one would not fit — never truncate an example.
    token_budget = int(MAX_INPUT_TOKENS * TOKEN_BUDGET_RATIO)
    picked = []
    total = 0
    for name in qualified:
        ex_dir = examples_dir / name
        py_file = _first_py(ex_dir) if ex_dir.is_dir() else None
        c_tokens = (
            _estimate_tokens(py_file.read_text(encoding="utf-8", errors="replace"))
            if py_file
            else 0
        )
        if total + c_tokens > token_budget:
            break
        picked.append(name)
        total += c_tokens

    # Unless the database is EMPTY, never return an empty selection:
    # force-pick the highest-scoring example (even below threshold) —
    # the coder always gets at least one reference.
    if not picked and candidates:
        if scores:
            best = max(scores, key=scores.get)
            warnings.warn(
                f"select_examples: nothing reached threshold "
                f"{EXAMPLE_SCORE_THRESHOLD} — force-selecting the "
                f"highest-scoring example '{best}' ({scores[best]})."
            )
            picked = [best]
        else:
            picked = [candidates[0]["name"]]
            warnings.warn(
                f"select_examples: scoring returned no scores — "
                f"force-selecting '{picked[0]}'."
            )
        if trace is not None:
            trace.append(
                {
                    "round": 1,
                    "action": "Force-selected (empty selection is never "
                              "returned while the database has examples)",
                    "count": len(picked),
                    "names": list(picked),
                }
            )

    if trace is not None:
        trace.append(
            {
                "round": 1,
                "action": (
                    f"Packed {len(picked)} within budget "
                    f"({total} of {token_budget} tokens)"
                ),
                "count": len(picked),
                "names": list(picked),
            }
        )
    return picked


# ======================================================================
# load full example code for selected names
# ======================================================================


def load_selected_examples(selected_names: list, examples_dir: Path = None) -> list:
    """Return full ``{"name": ..., "py": ...}`` dicts for *selected_names*."""
    if examples_dir is None:
        examples_dir = Path(__file__).resolve().parent.parent / "database" / "freefem_examples"

    results = []
    for name in selected_names:
        folder = examples_dir / name
        if not folder.is_dir():
            continue
        py_file = _first_py(folder)
        if py_file:
            results.append(
                {"name": name, "py": py_file.read_text(encoding="utf-8", errors="replace")}
            )
    return results


# ======================================================================
# Local test — run on all demos in results/
# ======================================================================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from API import DeepSeekCodeAgentV4Flash

    agent = DeepSeekCodeAgentV4Flash()
    results_dir = Path("results")

    spec_files = sorted(results_dir.glob("*/initial.json"))
    if not spec_files:
        print("No results/*/initial.json found. Run `python main.py` first.")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"Example Selector — {len(spec_files)} demos")
    print(f"{'=' * 70}")

    all_traces = []

    for spec_file in spec_files:
        name = spec_file.parent.name
        print(f"  [{name}]", flush=True)

        with open(spec_file, "r", encoding="utf-8") as f:
            problem_spec = json.load(f)

        mesh_file = spec_file.parent / "mesh.json"
        mesh_spec = None
        if mesh_file.exists():
            with open(mesh_file, "r", encoding="utf-8") as f:
                mesh_spec = json.load(f)

        trace = []
        selected = select_examples(agent, problem_spec, mesh_spec, trace=trace)
        all_traces.append((name, trace, selected))

        # Save to examples.txt
        out_path = spec_file.parent / "examples.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(selected) + "\n")

        ds = problem_spec.get("designed", {})
        print(
            f"    type={ds.get('problem_type', '?')}, {ds.get('dimension', '?')}D, "
            f"nonlinear={ds.get('state_is_nonlinear', False)}, final={len(selected)} examples"
        )

    # ── detailed trace table ──
    print(f"\n{'=' * 100}")
    print("DETAILED TRACE TABLE")
    print(f"{'=' * 100}")
    for name, trace, selected in all_traces:
        print(f"\n-- {name} --")
        print(f"  {'Round':<8} | {'Action':<56} | {'N':<4} | Names")
        print(f"  {'-' * 8}|{'-' * 56}|{'-' * 4}|{'-' * 40}")
        for t in trace:
            act = t["action"][:54]
            n = t["count"]
            names_str = ", ".join(t["names"][:4])
            if len(t["names"]) > 4:
                names_str += f", ... (+{len(t['names']) - 4})"
            print(f"  {str(t['round']):<8} | {act:<56} | {n:<4} | {names_str}")
        print(f"  {'':>70} | Final: {', '.join(selected)}")

    print("\nDone. Saved to results/*/examples.txt")
