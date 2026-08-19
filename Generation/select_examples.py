"""
LLM-based example selection for code generation.

Reads all pre-computed initial.json / mesh.json from database/freefem_examples,
builds concise summaries, and uses an LLM to pick the most relevant examples
for a given problem.  Batches summaries when they exceed MAX_INPUT_TOKENS.
"""

import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import MAX_INPUT_TOKENS, MAX_SELECTOR_ROUNDS, TOKEN_BUDGET_RATIO
from utils import parser_json

# ======================================================================
# helpers
# ======================================================================


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _build_summary(name: str, initial_path: Path, mesh_path: Path) -> dict:
    """Build a compact summary dict for one example."""
    summary = {"name": name}
    try:
        if initial_path.exists():
            with open(initial_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
            ds = spec.get("designed", {})
            ex = spec.get("extracted", {})
            summary["problem_type"] = ds.get("problem_type", "?")
            summary["dimension"] = ds.get("dimension", "?")
            summary["fields"] = [
                {"name": f.get("name"), "type": f.get("type"), "fe_space": f.get("fe_space")}
                for f in ds.get("fields", [])
            ]
            summary["state_is_nonlinear"] = ds.get("state_is_nonlinear", False)
            summary["num_constants"] = len(ds.get("constants", []))
            summary["has_adjoint"] = bool(
                ex.get("adjoint_equation") or ds.get("adjoint_description")
            )
            summary["has_shape_derivative"] = bool(
                ex.get("shape_derivative") or ds.get("shape_derivative_volume")
            )
    except Exception:
        pass

    try:
        if mesh_path.exists():
            with open(mesh_path, "r", encoding="utf-8") as f:
                mesh = json.load(f)
            mds = mesh.get("designed", {})
            summary["mesh_dim"] = mds.get("dimension")
            summary["mesh_labels"] = mds.get("labels_used", [])
    except Exception:
        pass

    return summary


def load_example_summaries(examples_dir: Path) -> list[dict]:
    """Return a compact summary for every example folder with initial.json."""
    summaries = []
    for folder in sorted(examples_dir.iterdir()):
        if not folder.is_dir():
            continue
        initial = folder / "initial.json"
        mesh = folder / "mesh.json"
        if not initial.exists():
            continue
        summaries.append(_build_summary(folder.name, initial, mesh))
    return summaries


# ======================================================================
# LLM selection
# ======================================================================

_SELECTOR_SYSTEM = """\
    You are an example selector for a code-generation system.  You are given:
    
    1. A **target problem summary** — a JSON object describing the problem for
       which we need to generate AutoFreeFEM code.
    
    2. A list of **candidate example summaries** — each describes a reference
       example that has both a problem description and working AutoFreeFEM code.
    
    Your task: select every example that is RELEVANT to the target problem.
    Prioritise matches on:
    
    - problem_type (shape_optimization vs topology_optimization) — exact match first
    - dimension (2D vs 3D) — exact match
    - field types (scalar vs vector) and number of fields
    - linear vs nonlinear
    - physics domain (elasticity, diffusion, thermal, fluid, etc.)
    - mesh complexity (number of labels / boundaries)
    
    Return exactly one JSON object:
    {{"selected": ["example_name_1", "example_name_2", ...], "reasoning": "one sentence"}}
    Return as many examples as you consider relevant, ordered by relevance
    (most relevant first).  Include only genuinely useful examples — omit those
    that share little with the target problem.
"""

_SELECTOR_USER = """\
    Target problem:
    {target_summary_json}
    
    Candidate examples:
    {candidates_json}
    
    Pick every genuinely relevant example from this batch.  Return the selection JSON.
"""

# ── Round 2: final ranking across shortlisted candidates ──────────────

_RANKER_SYSTEM = """\
    You are an example ranker for a code-generation system.  You are given a
    target problem summary and a list of pre-selected candidate examples (each
    already deemed relevant by a first-pass filter).
    
    Your task: produce a **final ranked list**, ordered by relevance (most
    relevant first).  Drop any candidate that is not actually useful after
    side-by-side comparison.
    
    Return exactly one JSON object:
    {{"selected": ["name1", "name2", ...], "reasoning": "one sentence"}}
    Return as many as are genuinely useful, in descending relevance order.
"""

_RANKER_USER = """\
    Target problem:
    {target_summary_json}
    
    Shortlisted candidates:
    {shortlist_json}
    
    Return the final ranked selection JSON.
"""

# ── Fallback scorer (used when ranking rounds are exhausted) ──────────

_SCORER_SYSTEM = """\
    You are an example relevance scorer.  You are given a target problem summary
    and a list of candidate examples.
    
    Your task: assign each candidate a relevance score from 0 to 100, where:
      100 = perfectly relevant (same physics, same dimension, same field types)
      0   = completely irrelevant
    
    Return exactly one JSON object:
    {{"scores": {{"name1": 95, "name2": 70, ...}}, "reasoning": "one sentence"}}
    Score every candidate — do not omit any.
"""

_SCORER_USER = """\
    Target problem:
    {target_summary_json}
    
    Candidates to score:
    {shortlist_json}
    
    Return the scoring JSON.
"""


# ======================================================================
# main entry point
# ======================================================================


def select_examples(
    agent, problem_spec: dict, mesh_spec: dict = None, examples_dir: Path = None, trace: list = None
) -> list[str]:
    """
    Multi-round LLM-based example selection.

    Round 1 — per-batch filtering.  Then iterative reduction until the
    shortlist fits within 80% of MAX_INPUT_TOKENS (or MAX_SELECTOR_ROUNDS
    is exhausted, in which case score-based trimming kicks in).

    If *trace* is provided (a list), each round's state is appended as a
    dict ``{"round": N, "action": "...", "count": int, "names": [...]}``.
    """
    if examples_dir is None:
        examples_dir = Path(__file__).resolve().parent.parent / "database" / "freefem_examples"

    # Build target summary
    ds = problem_spec.get("designed", {})
    target = {
        "problem_type": ds.get("problem_type", "?"),
        "dimension": ds.get("dimension", "?"),
        "fields": [
            {"name": f.get("name"), "type": f.get("type"), "fe_space": f.get("fe_space")}
            for f in ds.get("fields", [])
        ],
        "state_is_nonlinear": ds.get("state_is_nonlinear", False),
        "num_constants": len(ds.get("constants", [])),
    }
    if mesh_spec:
        mds = mesh_spec.get("designed", {})
        target["mesh_labels"] = mds.get("labels_used", [])

    target_json = json.dumps(target, ensure_ascii=False)

    # Load all candidate summaries
    all_candidates = load_example_summaries(examples_dir)
    if not all_candidates:
        warnings.warn("select_examples: no example summaries found")
        return []

    def _add_trace(action, names):
        if trace is not None:
            trace.append({"action": action, "count": len(names), "names": list(names)})

    # ── Round 1: per-batch filtering ──────────────────────────────────
    batches = _build_batches(target_json, all_candidates)
    shortlist = _run_round1(agent, target_json, batches)
    _add_trace(f"Round 1 ({len(batches)} batch(es))", shortlist)
    if not shortlist:
        return []

    # ── Iterative reduction: keep ranking until under 80% MAX ──────────
    name_to_summary = {c["name"]: c for c in all_candidates}
    token_budget = int(MAX_INPUT_TOKENS * TOKEN_BUDGET_RATIO)
    round_num = 0

    for round_num in range(MAX_SELECTOR_ROUNDS):
        current_tokens = _estimate_tokens(
            json.dumps([name_to_summary.get(n, {"name": n}) for n in shortlist], ensure_ascii=False)
        )
        if current_tokens <= token_budget:
            # Verify with actual .py sizes (summary-JSON may undercount)
            real_tokens = 0
            for n in shortlist:
                ex_dir = examples_dir / n
                if not ex_dir.is_dir():
                    continue
                py_file = next((p for p in ex_dir.iterdir() if p.suffix == ".py"), None)
                if py_file is not None:
                    real_tokens += _estimate_tokens(
                        py_file.read_text(encoding="utf-8", errors="replace")
                    )
            if real_tokens <= token_budget:
                _add_trace(
                    f"Round {round_num + 2} — fits budget ({real_tokens} actual <= {token_budget})",
                    shortlist,
                )
                break
            _add_trace(
                f"Round {round_num + 2} — JSON fits ({current_tokens}) but .py over ({real_tokens}), continuing",
                shortlist,
            )

        shortlist_summaries = [name_to_summary[n] for n in shortlist if n in name_to_summary]
        if not shortlist_summaries:
            break

        prompt_list = [
            {"role": "system", "content": _RANKER_SYSTEM},
            {"role": "user", "content": _RANKER_USER},
        ]
        agent.set_prompt_list(prompt_list)
        raw = agent.invoke(
            {
                "target_summary_json": target_json,
                "shortlist_json": json.dumps(shortlist_summaries, ensure_ascii=False),
            }
        )
        try:
            result = json.loads(parser_json(raw))
            new_shortlist = [n for n in result.get("selected", []) if n in set(shortlist)]
            if not new_shortlist:
                break
            if new_shortlist == shortlist:
                break
            shortlist = new_shortlist
            _add_trace(f"Round {round_num + 2} — re-rank ({len(shortlist)} kept)", shortlist)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            warnings.warn(f"select_examples reduction round failed: {e}")
            break

    if round_num == MAX_SELECTOR_ROUNDS - 1:
        still = _estimate_tokens(
            json.dumps([name_to_summary.get(n, {"name": n}) for n in shortlist], ensure_ascii=False)
        )
        if still > token_budget:
            warnings.warn(
                f"select_examples: still over token budget after {MAX_SELECTOR_ROUNDS} rounds "
                f"({still} > {token_budget}).  Switching to score-based trimming."
            )
            shortlist = _score_and_trim(
                agent, target_json, shortlist, name_to_summary, examples_dir, token_budget
            )
            _add_trace(f"Score round — trimmed to budget", shortlist)

    # ── Final safety trim ──────────────────────────────────────────────
    final = _trim_by_tokens(shortlist, examples_dir)
    if final != shortlist:
        _add_trace("Final hard trim", final)
    return final


# ======================================================================
# internal helpers
# ======================================================================


def _build_batches(target_json: str, candidates: list[dict]) -> list[list[dict]]:
    """Split *candidates* into batches that each fit within MAX_INPUT_TOKENS."""
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


def _run_round1(agent, target_json: str, batches: list[list[dict]]) -> list[str]:
    """Run first-pass selection on each batch; return deduplicated names."""
    all_selected = []
    for batch in batches:
        candidates_json = json.dumps(batch, ensure_ascii=False)
        prompt_list = [
            {"role": "system", "content": _SELECTOR_SYSTEM},
            {"role": "user", "content": _SELECTOR_USER},
        ]
        agent.set_prompt_list(prompt_list)
        raw = agent.invoke(
            {
                "target_summary_json": target_json,
                "candidates_json": candidates_json,
            }
        )
        try:
            result = json.loads(parser_json(raw))
            all_selected.extend(result.get("selected", []))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            warnings.warn(f"select_examples round-1 JSON parse failed: {e}")

    seen = set()
    unique = []
    for name in all_selected:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _score_and_trim(
    agent,
    target_json: str,
    shortlist: list[str],
    name_to_summary: dict,
    examples_dir: Path,
    token_budget: int,
) -> list[str]:
    """
    Let LLM score each candidate 0-100, then keep highest scores within budget.

    Token budget is checked against actual .py file sizes, not summaries.
    """
    shortlist_summaries = [name_to_summary[n] for n in shortlist if n in name_to_summary]
    if not shortlist_summaries:
        return shortlist

    prompt_list = [
        {"role": "system", "content": _SCORER_SYSTEM},
        {"role": "user", "content": _SCORER_USER},
    ]
    agent.set_prompt_list(prompt_list)
    raw = agent.invoke(
        {
            "target_summary_json": target_json,
            "shortlist_json": json.dumps(shortlist_summaries, ensure_ascii=False),
        }
    )

    try:
        result = json.loads(parser_json(raw.strip()))
        scores = result.get("scores", {})
        sorted_names = sorted(
            (n for n in scores if n in shortlist),
            key=lambda n: scores.get(n, 0),
            reverse=True,
        )
        trimmed = []
        total_tokens = 0
        for name in sorted_names:
            py_file = None
            ex_dir = examples_dir / name
            if ex_dir.is_dir():
                for f in ex_dir.iterdir():
                    if f.suffix == ".py":
                        py_file = f
                        break
            c_tokens = (
                _estimate_tokens(py_file.read_text(encoding="utf-8", errors="replace"))
                if py_file
                else 0
            )
            if total_tokens + c_tokens > token_budget:
                break
            trimmed.append(name)
            total_tokens += c_tokens
        return trimmed
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        warnings.warn(f"score_and_trim failed: {e}")
        return shortlist


def _trim_by_tokens(names: list[str], examples_dir: Path) -> list[str]:
    """Keep only the first N examples whose full .py fits in 80% MAX."""
    token_budget = int(MAX_INPUT_TOKENS * TOKEN_BUDGET_RATIO)
    trimmed = []
    total_tokens = 0
    for name in names:
        py_file = None
        ex_dir = examples_dir / name
        if ex_dir.is_dir():
            for f in ex_dir.iterdir():
                if f.suffix == ".py":
                    py_file = f
                    break
        if py_file:
            tokens = _estimate_tokens(py_file.read_text(encoding="utf-8", errors="replace"))
            if total_tokens + tokens > token_budget:
                break
            total_tokens += tokens
        trimmed.append(name)
    return trimmed


# ======================================================================
# load full example code for selected names
# ======================================================================


def load_selected_examples(selected_names: list[str], examples_dir: Path = None) -> list[dict]:
    """Return full ``{"name": ..., "py": ...}`` dicts for *selected_names*."""
    if examples_dir is None:
        examples_dir = Path(__file__).resolve().parent.parent / "database" / "freefem_examples"

    results = []
    for name in selected_names:
        folder = examples_dir / name
        if not folder.is_dir():
            continue
        py_file = None
        for f in folder.iterdir():
            if f.suffix == ".py":
                py_file = f
                break
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
        print("No results/*/initial.json found. Run problem_parser first.")
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
        rounds = len(trace)
        print(
            f"    type={ds.get('problem_type', '?')}, {ds.get('dimension', '?')}D, "
            f"nonlinear={ds.get('state_is_nonlinear', False)}, rounds={rounds}, final={len(selected)} examples"
        )

    # ── detailed trace table ──
    print(f"\n{'=' * 100}")
    print("DETAILED TRACE TABLE")
    print(f"{'=' * 100}")
    for name, trace, selected in all_traces:
        print(f"\n-- {name} --")
        print(f"  {'Round':<34} | {'Action':<42} | {'N':<4} | Names")
        print(f"  {'-' * 34}|{'-' * 42}|{'-' * 4}|{'-' * 40}")
        for i, t in enumerate(trace):
            act = t["action"][:40]
            n = t["count"]
            names_str = ", ".join(t["names"][:4])
            if len(t["names"]) > 4:
                names_str += f", ... (+{len(t['names']) - 4})"
            print(f"  {str(i + 1):<34} | {act:<42} | {n:<4} | {names_str}")
        print(f"  {'':>82} | Final: {', '.join(selected)}")

    print(f"\nDone. Saved to results/*/examples.txt")
