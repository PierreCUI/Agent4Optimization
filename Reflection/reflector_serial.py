# -*- coding: utf-8 -*-
"""Serial reflection loop (USER RULES 2026-10-15).

The loop's input is a GROUP of experiments (the execution sweep's cells
and, after each repair, a NEW group — the previous round's files are
never modified).  Detection runs SERIALLY, one scope at a time:

    syntax_error -> formula_error -> design_error -> DONE

- each detector reports ONLY its own scope (a failing result whose
  cause is outside the scope must PASS that detector);
- there is NO other/fallback category;
- a failing detector repairs and RE-EXECUTES (the execution-reflection
  loop); repairs create up to configs.MAX_REFLECTION_GROUP new
  experiments — multiple fixes / designs / parameter combinations are
  allowed; the parameters are NOT limited to diffusion/v0/material
  penalty (the model may touch other DESIGNED parameters when needed).

ROUND INPUT (USER clarification 2026-10-15):
  1. initial.json + mesh.json
  2. the packaged per-round info (per EXPERIMENT: main.py, mesh.txt,
     objective, python console, FreeFEM console) — written to
     temp/history after every round and read back from there
  3. the composed per-round image (same)
  4. history/record.json: syntax-fix records, formula promising /
     unpromising records, traversed parameter ranges, promising
     parameter combinations.

At the end: all images are copied to results/<case>/figs and the
temp/history folder is deleted.
"""
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import configs

SCOPES = ("syntax_error", "formula_error", "design_error")

_DETECT_SYSTEM = """\
You are the {scope} detector of an optimization-code repair loop.  You
receive a GROUP of experiments (the same problem, different fixes /
designs / parameter combinations) and their execution evidence.

YOUR SCOPE IS {scope} ONLY — {scope_desc}.  Report ONLY issues inside
your scope.  A result that fails for ANY reason outside your scope is a
PASS for you (another detector owns it).  There is no other category.

Return ONE JSON:
{{"passed": true, "issues": []}}
or
{{"passed": false, "issues": [{{"member": "<member id>",
"problem": "<one sentence>", "fix_hint": "<how to repair>"}}]}}
No markdown.  No code fences.
"""

_DETECT_USER = """\
Problem: {problem_name}

Specs (initial.json + mesh.json):
{specs_text}

Round history (packaged per-experiment info, round by round):
{history_text}

Current round's packaged info:
{round_text}

Round records so far (syntax fixes / formula promising-unpromising /
traversed parameter ranges / promising parameter combinations):
{record_text}

Return the detection JSON.
"""

_FIX_SYSTEM = """\
You are the {scope} repairer of an optimization-code repair loop.  You
receive a GROUP of experiments (the same problem, different fixes /
designs / parameter combinations), the detection report for your scope,
and the history so far.

Repair RULES:
- NEVER modify the previous round's files — propose NEW experiments:
  up to {max_group} variants total (multiple repair methods are
  welcome).
- Every variant carries the COMPLETE Python code (main.py).
- START from the base code (original_main_py — it is the code that
  already ran the whole sweep).  Make ONLY the most necessary
  modifications — the smallest change that addresses the issues.
  NO UNRELATED CHANGES: anything not needed for the fix must stay
  exactly as it is in the base code.
- PROMISING / UNPROMISING history (the two lists below): continue from
  PROMISING experiments — their full code is provided, build your new
  variants on them.  NEVER retry an idea marked UNPROMISING (same
  formula shape, same design, same parameter combination) — explore
  something different instead.
- Parameter exploration: for each variant you may give ``param_sweeps``
  — a mapping name -> list of values.  Start with the three standard
  axes (diffusion, v0, and the material penalty constant — whatever
  its name in the code); ONLY when you judge the three cannot fix the
  problem may you add OTHER DESIGNED parameters (hmin, hmax, or any
  designed constant) to the sweeps.  Extracted values are ground
  truth — never sweep them.
- Syntax issues must ALL be repaired.  Design issues may keep several
  promising designs (one variant each).
- The whole group of a parameter sweep counts as ONE experiment for
  acceptance: as soon as one combination works, the sweep passes.

EXECUTION CONTRACT (hard pipeline rules — a variant that breaks them
is DROPPED before it runs and the whole repair round is judged failed):
- main.py ONLY does Phase 1: describe the problem with the AutoFreeFEM
  DSL and let the library emit the optimize*.edp script.  The pipeline
  itself finds and runs FreeFEM afterwards.
- NEVER search for or execute FreeFem++ (no shutil.which('FreeFem++'),
  no subprocess / os.system, no freefem++/freefem++-nw calls) — the
  binary lives in a different environment the Python phase cannot see.
- NEVER write or name .edp files yourself (no tempfile with .edp, no
  .edp filename strings) — the library names them, and the runner only
  executes files named optimize*.edp.
- NEVER add "tool not found -> exit" guards to main.py.

Return ONE JSON:
{{"variants": [{{"main": "<complete Python code>", "note": "...",
"param_sweeps": {{"diffusion": [..], "v0": [..], "<penalty name>": [..]}}}},
 ...]}}
No markdown.  No code fences.
"""

_FIX_USER = """\
Problem: {problem_name}

Specs (initial.json + mesh.json):
{specs_text}

AutoFreeFEM documentation (the SAME document the generator used):
{doc_text}

Reference examples (selected for this problem, full code):
{examples_text}

The base main.py — the code that already ran the whole sweep; edit it
minimally, keep its imports and structure:
{original_main_py}

PROMISING experiments so far (full packaged content — continue from
these):
{promising_text}

UNPROMISING experiments so far (never retry these):
{unpromising_text}

Detection report ({scope}):
{issues_text}

Current group (member id, main.py path, evidence):
{group_text}

History records so far:
{record_text}

Return the variants JSON.
"""

_JUDGE_SYSTEM = """\
You are the candidate-judge of an optimization-code repair loop.  After
a {scope} repair round you receive the rerun results of every new
experiment.  Judge them:

- PROMISING candidates: keep them (they may be repaired further).
- UNPROMISING candidates: record them as such (never retry the same
  idea).
- ACCEPTED parameter combinations: per candidate, every combination
  whose result is acceptable (correct physics, sane objective, no
  crash).  One accepted combination makes that candidate's sweep a
  success.

Return ONE JSON:
{{"candidates": [{{"member": "<member id>", "status": "promising" |
"unpromising", "reason": "...", "accepted_params": [{{...}}]}}],
"any_success": true|false}}
No markdown.  No code fences.
"""

_JUDGE_USER = """\
Problem: {problem_name}

Rerun results (member id, exit codes, objective, evidence):
{results_text}

Return the judge JSON.
"""


# ── group / evidence plumbing ─────────────────────────────────────────

def _member_dirs(results_dir: Path) -> list[Path]:
    """The execution sweep's cells (each leaf holds main.py — the run
    artifacts may or may not exist yet)."""
    root = results_dir / "temp" / "diffv0_beta"
    out = []
    if root.exists():
        for mp in sorted(root.rglob("main.py")):
            out.append(mp.parent)
    return out


def _member_id(results_dir: Path, d: Path) -> str:
    try:
        return str(d.relative_to(results_dir))
    except ValueError:
        return d.name


def _member_evidence(d: Path) -> dict:
    ev = {"main_py": str(d / "main.py"), "exit_codes": [],
          "objective": None, "minth": False, "images": [],
          "python_console": "", "ff_console": "", "warning": ""}
    rr = d / "run_result.json"
    if rr.exists():
        try:
            data = json.loads(rr.read_text(encoding="utf-8"))
            ev["exit_codes"] = [
                rec.get("return_code") for rec in data.get("freefem") or []]
            py = data.get("autofreefem") or {}
            ev["warning"] = str(py.get("warning") or "")
            pyp = py.get("python") or {}
            ev["python_console"] = " ".join([
                str(pyp.get("stdout") or ""), str(pyp.get("stderr") or "")])[-2000:]
            ev["ff_console"] = " ".join(
                str(rec.get("stdout") or "") + " " +
                str(rec.get("stderr") or "")
                for rec in data.get("freefem") or [])[-2000:]
        except Exception:
            pass
    for f in sorted(d.glob("objective*.txt")):
        try:
            lines = f.read_text(encoding="utf-8",
                                errors="replace").splitlines()
            if len(lines) > 1 and "," in lines[-1]:
                ev["objective"] = float(lines[-1].split(",")[1])
                break
        except Exception:
            continue
    if list((d / "results").glob("*MinTh.vtu")):
        ev["minth"] = True
    for p in sorted((d / "results").glob("*.png")):
        ev["images"].append(str(p))
    return ev


def _render_member_image(d: Path, out_png: Path) -> bool:
    """Render the member's MinTh DESIGN into out_png (or return False).

    The design is the Label cell field (solid/void indicator — the
    field every snapshot VTU carries).  Rendering anything else (e.g.
    the mesh silhouette with a constant value, which the previous
    custom reader did) produces gray blobs that show no design.
    Delegate to Reflection/renderer.render_snapshot_label_png — the
    established Label renderer used by the snapshot frames.
    """
    vtus = list((d / "results").glob("*MinTh.vtu"))
    if not vtus:
        return False
    try:
        from .renderer import render_snapshot_label_png
        render_snapshot_label_png(vtus[0], out_png)
        return True
    except Exception:
        return False


def _compose_mosaic(out_png: Path, members: list[dict]) -> Path:
    """One BIG mosaic image: every member's design image in a grid
    (gray placeholder when the member produced none)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import math

    n = len(members)
    cols = min(8, n)
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[a] for a in axes]
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < n and members[i].get("image_png"):
            try:
                img = mpimg.imread(members[i]["image_png"])
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("0.9")
        else:
            ax.set_facecolor("0.9")
        ax.set_title(members[i]["member"][-22:] if i < n else "",
                     fontsize=7)
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    return out_png


def _compose_round_image(history_dir: Path, round_no: int,
                         members: list[dict]) -> Path:
    """The per-round mosaic kept inside the packaged history."""
    out = history_dir / f"round_{round_no:02d}" / f"round_{round_no:02d}.png"
    return _compose_mosaic(out, members)


def _load_record(history_dir: Path) -> dict:
    rp = history_dir / "record.json"
    if rp.exists():
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"syntax_fixes": [], "formula_fixes": [], "designs": [],
            "param_ranges": [], "accepted_params": [], "rounds": []}


def _save_record(history_dir: Path, record: dict) -> None:
    (history_dir / "record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _specs_text(results_dir: Path) -> str:
    parts = []
    for name in ("initial.json", "mesh.json"):
        p = results_dir / name
        if p.exists():
            try:
                parts.append(f"=== {name} ===\n"
                             f"{p.read_text(encoding='utf-8', errors='replace')}")
            except OSError:
                pass
    return "\n\n".join(parts)[:6000]


def _history_text(history_dir: Path, max_rounds: int = 4) -> str:
    """The packaged per-round info (latest rounds first), from history."""
    parts = []
    for rd in sorted(history_dir.glob("round_*/info.json"),
                     reverse=True)[:max_rounds]:
        try:
            parts.append(f"=== {rd.parent.name} ===\n"
                         f"{rd.read_text(encoding='utf-8', errors='replace')}")
        except OSError:
            continue
    return "\n\n".join(parts)[:6000]


def _budget_list(entries: list, budget: int) -> str:
    """Join whole entries until *budget* chars is reached — a WHOLE
    example is kept or dropped, never cut in half (USER 2026-10-18).
    Dropped entries are reported as a count."""
    if budget <= 0:
        return "\n\n".join(entries)
    out = []
    used = 0
    dropped = 0
    for text in entries:
        if used + len(text) > budget:
            dropped += 1
            continue
        out.append(text)
        used += len(text)
    if dropped:
        out.append(f"(... {dropped} more examples omitted by the "
                   "context budget)")
    return "\n\n".join(out)


def _promote_accepted(results_dir: Path, history_dir: Path,
                      members: list, judge: dict) -> int:
    """Copy each ACCEPTED member's design into the final results/ and
    render its screenshot into the history dir — the finalize step
    composes them into figs/accepted.png (USER 2026-10-18: one
    accepted result suffices; several accepted results all appear in
    the accepted mosaic).  Returns the number promoted."""
    by_id = {_member_id(results_dir, d): d for d in members}
    out_dir = results_dir / "results"
    jobs = []
    for jd in judge.get("candidates") or []:
        if not jd.get("accepted_params"):
            continue
        d = by_id.get(str(jd.get("member") or ""))
        if d is None:
            continue
        vtus = list((d / "results").glob("*MinTh.vtu"))
        if not vtus:
            continue
        suffix = Path(str(jd.get("member") or d.name)).name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(vtus[0], out_dir / f"accepted_{suffix}_{vtus[0].name}")
        except OSError:
            continue
        jobs.append((d, vtus[0], history_dir / f"accepted_{suffix}.png"))
    if jobs:
        from .renderer import render_labels_paraview_batch
        failed = set(render_labels_paraview_batch(
            [(str(v), str(p)) for _, v, p in jobs]))
        for i, (d, _, p) in enumerate(jobs):
            if i in failed:
                _render_member_image(d, p)  # matplotlib fallback
    return len(jobs)


def _prospect_texts(history_dir: Path, record: dict) -> tuple:
    """(promising_text, unpromising_text) — the judge's verdicts mapped
    back to the FULL packaged member content (main.py + mesh.txt +
    evidence) kept in history/round_NN/info.json.

    Budget (USER 2026-10-18): the SAME convention the example selector
    uses — configs.MAX_INPUT_TOKENS x configs.TOKEN_BUDGET_RATIO
    tokens, ~4 chars/token, whole examples only (never half an
    example).  The budget is SHARED: promising examples pack first,
    unpromising take the remainder."""
    status_by = {}   # (round, member) -> (status, reason)
    for bucket in ("designs", "formula_fixes"):
        for e in record.get(bucket) or []:
            member = e.get("member")
            if not member or not e.get("status"):
                continue
            status_by[(int(e.get("round") or 0), str(member))] = (
                str(e.get("status")), str(e.get("reason") or "")[:200])
    promising, unpromising = [], []
    for rd in sorted(history_dir.glob("round_*/info.json")):
        try:
            info = json.loads(rd.read_text(encoding="utf-8"))
        except Exception:
            continue
        round_no = int(rd.parent.name.split("_")[1])
        for m in info.get("members") or []:
            key = (round_no, str(m.get("member") or ""))
            if key not in status_by:
                continue
            status, reason = status_by[key]
            text = (
                f"== {m.get('member')} (round {round_no}, {status}) ==\n"
                f"objective={m.get('objective')} "
                f"exit={m.get('exit_codes')}\n"
                f"reason: {reason}\n--- main.py ---\n"
                f"{m.get('main_py') or '(no code)'}\n--- mesh.txt ---\n"
                f"{m.get('mesh_txt') or '(no mesh)'}"
            )
            (promising if status == "promising" else unpromising).append(text)
    budget_chars = int(configs.MAX_INPUT_TOKENS * configs.TOKEN_BUDGET_RATIO) * 4
    promising_text = _budget_list(promising, budget_chars)
    remaining = budget_chars - len(promising_text)
    unpromising_text = _budget_list(unpromising, max(remaining, 0))
    return promising_text, unpromising_text


# ── LLM plumbing (AgentRouter-compatible) ─────────────────────────────

def _invoke(agent, system: str, user_text: str, values: dict, images=None):
    """Pass the TEMPLATES to the agent's writer (langchain fills the
    placeholders from *values* — the {{...}} JSON examples stay escaped
    in the templates)."""
    agent.set_prompt_list([
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ])
    if images:
        return agent.invoke_multimodal(values, images=images)
    return agent.invoke(values)


def _parse_json(raw: str):
    from utils import parser_json
    try:
        return json.loads(parser_json(raw))
    except Exception as e:
        warnings.warn(f"reflector_serial JSON parse failed: {e}")
        return None


# ── group construction (never modifies old files) ─────────────────────

_EXEC_CONTRACT_CALL = re.compile(
    r"\b(subprocess\.\w+|os\.system|os\.popen|shutil\.which)\s*\([^)]*\)")
# only calls that touch the solver toolchain break the contract — other
# subprocess/os usage (data post-processing, plotting, ...) is legal
# (USER 2026-10-17: reject only what is necessary)
_EXEC_CONTRACT_TOOLS = re.compile(r"freefem|mmg|advect|mshdist", re.I)
_EXEC_CONTRACT_EDP = re.compile(r"['\"][^'\"]*\.edp['\"]")


def _sanitize_variant(code: str) -> tuple:
    """Deterministic execution-contract check for repair variants.

    A variant main.py may ONLY describe the problem with the DSL and
    let the library emit optimize*.edp — the pipeline (via WSL) owns
    finding and running FreeFEM.  Variants that search for / execute
    THE SOLVER TOOLCHAIN or write .edp files themselves break that
    contract (observed failure classes: rc=6 'FreeFem++ not found',
    rc=1 WinError 2, rc=0 with a non-optimize .edp that the runner
    ignores).  Everything else is allowed.  Returns (ok, reason).
    """
    if not isinstance(code, str) or not code.strip():
        return False, "empty variant"
    for m in _EXEC_CONTRACT_CALL.finditer(code):
        if _EXEC_CONTRACT_TOOLS.search(m.group(0)):
            return False, (
                f"solver-tool call in variant: {m.group(0)[:80]!r} — "
                "main.py may only describe the problem (the pipeline "
                "runs FreeFEM)")
    m = _EXEC_CONTRACT_EDP.search(code)
    if m:
        return False, (
            f"explicit .edp reference {m.group(0)[:60]!r} — the library "
            "emits optimize*.edp, main.py must not touch .edp files")
    if re.search(r"/\s*(?:dx|dy|dsx|dx2|dy2)\s*\(", code):
        # USER DECISION 2026-10-17: the LaTeX-printer crash on the
        # Pow(dx, -1) rewrite is the library's fault — accepted, and
        # the pattern is hard-blocked unconditionally
        return False, (
            "division by a domain/boundary integral — the LaTeX printer "
            "crashes on the rewritten Pow(dx, -1) form (verified "
            "2026-10-17) and the run dies before emitting any .edp.  "
            "Rewrite as ONE integral (fold a constant denominator "
            "inside the integrand).")
    return True, ""


def _build_group(results_dir: Path, round_no: int, variants: list) -> list[Path]:
    """Materialise the repair variants as NEW experiment dirs under
    temp/group_r<round>/ — cap configs.MAX_REFLECTION_GROUP members."""
    root = results_dir / "temp" / f"group_r{round_no}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    mesh_src = results_dir / "mesh.txt"
    members = []
    idx = 0
    for v in variants:
        main_code = v.get("main") or ""
        if not isinstance(main_code, str) or not main_code.strip():
            continue
        sweeps = v.get("param_sweeps") or {}
        if not isinstance(sweeps, dict) or not sweeps:
            sweeps = {}
        combos = [{}]
        for key, values in sweeps.items():
            if not isinstance(values, list) or not values:
                continue
            combos = [dict(c, **{key: val}) for c in combos for val in values]
        for combo in combos:
            if len(members) >= configs.MAX_REFLECTION_GROUP:
                break
            idx += 1
            d = root / f"m{idx:02d}"
            d.mkdir(parents=True)
            code = main_code
            for key, val in combo.items():
                code = re.sub(
                    r"\b" + re.escape(key) + r"\s*=\s*[-+0-9.eE]+",
                    f"{key}={val:.6g}" if isinstance(val, (int, float))
                    else f"{key}={val}",
                    code, count=1)
                code = re.sub(
                    r"(" + re.escape(key) + r"\s*=\s*Constant\s*\([^,]*,\s*)"
                    r"[^,]*(\s*,)",
                    lambda m, k=key, v=val: m.group(1)
                    + (f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
                    + m.group(2),
                    code, count=1)
            (d / "main.py").write_text(code, encoding="utf-8",
                                       newline="\n")
            if mesh_src.exists():
                shutil.copy2(mesh_src, d / "mesh.txt")
            members.append(d)
        if len(members) >= configs.MAX_REFLECTION_GROUP:
            break
    return members


def _rerun_group(members: list[Path]) -> None:
    """Execution on the new group: run_single per member, up to
    configs.REFLECTION_WORKERS threads."""
    from Execution.runner import run_single
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=configs.REFLECTION_WORKERS) as ex:
        futs = {
            ex.submit(run_single, m / "main.py", m): m for m in members
        }
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                fut.result()
            except (Exception, SystemExit) as e:
                # SystemExit from a member (check_tools FATAL) must not
                # kill the reflection loop
                print(f"  [serial] member {m.name} run failed: {e}",
                      flush=True)


# ── per-round packaging ───────────────────────────────────────────────

def _package_round(history_dir, round_no, members, member_info, scope,
                   record):
    """Write the round's packaged info + images into temp/history/ and
    update record.json (the next round's input)."""
    rd = history_dir / f"round_{round_no:02d}"
    rd.mkdir(parents=True, exist_ok=True)
    info = {"scope": scope, "members": []}
    image_refs = []
    for i, mi in enumerate(member_info):
        png = None
        src = mi.get("image_png")
        if src and Path(src).exists():
            dst = rd / f"m{i:02d}.png"
            try:
                shutil.copy2(src, dst)
                png = str(dst)
            except OSError:
                png = None
        d = Path(mi["dir"]) if mi.get("dir") else None
        main_txt = mesh_txt = ""
        if d and (d / "main.py").exists():
            try:
                main_txt = (d / "main.py").read_text(encoding="utf-8",
                                                     errors="replace")
            except OSError:
                pass
        if d and (d / "mesh.txt").exists():
            try:
                mesh_txt = (d / "mesh.txt").read_text(encoding="utf-8",
                                                      errors="replace")
            except OSError:
                pass
        info["members"].append({
            "member": mi["member"],
            "main_py": main_txt,
            "mesh_txt": mesh_txt,
            "objective": mi.get("objective"),
            "exit_codes": mi.get("exit_codes"),
            "python_console": mi.get("python_console", ""),
            "ff_console": mi.get("ff_console", ""),
            "image": png,
        })
        image_refs.append({"member": mi["member"], "image_png": png})
    (rd / "info.json").write_text(
        json.dumps(info, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    try:
        big = _compose_round_image(history_dir, round_no, image_refs)
    except Exception as e:
        warnings.warn(f"compose round image failed: {e}")
        big = None
    record.setdefault("rounds", []).append({
        "round": round_no,
        "scope": scope,
        "big_image": str(big) if big else None,
    })
    _save_record(history_dir, record)


# ── the serial loop ───────────────────────────────────────────────────

_SCOPE_DESC = {
    "syntax_error": (
        "imports, call signatures/arguments, spelling, fixed contents, "
        "the extracted/designed rules, the mesh buildmesh chain and the "
        "mesh-generation failures it causes, and identifiers entering "
        "the FreeFEM code containing underscores (problem name, Domain "
        "mesh variables, border names)"
    ),
    "formula_error": (
        "every formula: intermediate expressions, the J formula, the "
        "Lagrangian formulas, the nonlinear formulas"
    ),
    "design_error": (
        "design VALUES and finite-element parameters: physical-parameter "
        "numbers, Afac/ell, CharacteristicFunction values, phi, the "
        "three-way boundary labels, boundary physical properties and "
        "Dirichlet conditions, pen implementation, mesh border "
        "definitions, hmin/hmax/diffusion/v0, mesh resolution, field "
        "precision P1/P2, and the parameter-combination exploration"
    ),
}


def run_reflection(agent, info, config) -> str:
    results_dir = info.results_dir
    problem_name = info.problem_name
    history_dir = results_dir / "temp" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    record = _load_record(history_dir)
    specs_text = _specs_text(results_dir)

    # ── the SAME context materials the GENERATOR used (USER 2026-10-17:
    # the repairer was writing variants blind — give it the
    # documentation, the selected reference examples and the working
    # base code, exactly like the coder) ──
    doc_text = ""
    try:
        from Generation.coder import _load_documentation
        doc_text = _load_documentation()
    except Exception:
        pass
    examples_text = ""
    try:
        names = [l.strip() for l in
                 (results_dir / "examples.txt").read_text(
                     encoding="utf-8", errors="replace").splitlines()
                 if l.strip()]
        from Generation.select_examples import load_selected_examples
        for ex in load_selected_examples(names):
            examples_text += f"=== EXAMPLE: {ex['name']} ===\n{ex['py']}\n\n"
    except Exception:
        pass
    base_main = ""
    try:
        base_main = (results_dir / "main.py").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        pass

    members = _member_dirs(results_dir)
    if not members:
        print("  [serial] no execution cells found — nothing to reflect on.")
        info.reflection_status = "give_up"
        return "give_up"

    status = "give_up_max"
    round_no = len(record.get("rounds") or []) + 1

    while round_no <= config.MAX_REFLECTION_TIMES:
        print(f"  [serial] round {round_no} — group of {len(members)}",
              flush=True)
        member_info = []
        # design images: ONE pvbatch call renders the whole group the
        # way the ParaView GUI shows it (Label + Cool to Warm); members
        # pvbatch could not render fall back to the matplotlib renderer
        jobs = []  # (member dir, out png, MinTh vtu or None)
        for d in members:
            # unique per member across the WHOLE run: the member id
            # (relative path) can never collide across penalty levels
            # or rounds — every screenshot survives (USER 2026-10-18)
            member_key = _member_id(results_dir, d).replace(
                "/", "_").replace("\\", "_")
            png = history_dir / f"tmp_r{round_no}_{member_key}.png"
            vtus = list((d / "results").glob("*MinTh.vtu"))
            jobs.append((d, png, vtus[0] if vtus else None))
        from .renderer import render_labels_paraview_batch
        batch_jobs = [(str(v), str(p)) for _, p, v in jobs if v]
        batch_map = [i for i, (_, _, v) in enumerate(jobs) if v]
        failed_batch = set(render_labels_paraview_batch(batch_jobs))
        failed_job_idx = {batch_map[k] for k in failed_batch}
        for i, (d, png, _) in enumerate(jobs):
            if i not in failed_job_idx and png.exists():
                continue
            _render_member_image(d, png)  # matplotlib fallback
        for d, png, _ in jobs:
            ev = _member_evidence(d)
            if png.exists():
                ev["image_png"] = str(png)
            member_info.append({
                "member": _member_id(results_dir, d),
                "dir": str(d),
                **ev,
            })
        round_text = json.dumps(
            [{k: v for k, v in m.items() if k not in ("images", "main_py")}
             for m in member_info],
            indent=1, ensure_ascii=False, default=str)[:5000]
        history_text = _history_text(history_dir)
        record_text = json.dumps(record, indent=1, ensure_ascii=False,
                                 default=str)[:5000]
        promising_text, unpromising_text = _prospect_texts(history_dir,
                                                           record)

        advanced = False
        done_success = False
        for scope in SCOPES:
            # ── detect (scope-restricted) ──
            images = [p for m in member_info for p in m.get("images", [])][:32]
            values = {
                "scope": scope,
                "scope_desc": _SCOPE_DESC[scope],
                "problem_name": problem_name,
                "specs_text": specs_text,
                "history_text": history_text,
                "round_text": round_text,
                "record_text": record_text,
            }
            gate = record.get("execution_gate") or {}
            if (gate.get("round") == round_no - 1 and not gate.get("ran")
                    and scope == gate.get("scope")):
                # HARD GATE: the previous round's repair produced NO
                # experiment that actually ran FreeFEM — detection must
                # not pass; force a repair of the same scope with the
                # dead-run evidence (the LLM detect is skipped)
                if gate.get("total"):
                    problem = (
                        f"EXECUTION GATE: all {gate.get('total')} "
                        "experiments of the last repair died in the "
                        "Python phase — FreeFEM ran zero times. "
                        "Evidence: "
                        + "; ".join(gate.get("evidence") or []))
                else:
                    problem = (
                        "EXECUTION GATE: every variant of the last repair "
                        "violated the execution contract and was dropped "
                        "before running.  Evidence: "
                        + "; ".join(gate.get("evidence") or []))
                report = {
                    "passed": False,
                    "issues": [{
                        "member": "(all)",
                        "problem": problem,
                        "fix_hint": (
                            "variants must follow the execution contract: "
                            "main.py only describes the problem with the "
                            "DSL (the pipeline runs FreeFEM afterwards).  "
                            "Remove any FreeFem++ search/execution, any "
                            ".edp writing, tool-not-found guards and "
                            "non-optimize .edp names."),
                    }],
                }
                print(f"  [serial] {scope}: EXECUTION GATE forces failure "
                      f"(0 FreeFEM runs in round {gate.get('round')})",
                      flush=True)
            else:
                raw = _invoke(agent, _DETECT_SYSTEM, _DETECT_USER, values,
                              images=images or None)
                report = _parse_json(raw) or {"passed": False, "issues": []}
            issues = report.get("issues") or []
            if report.get("passed") and not issues:
                print(f"  [serial] {scope}: PASS", flush=True)
                continue
            print(f"  [serial] {scope}: {len(issues)} issue(s) -> repair",
                  flush=True)

            # ── repair (variants; sweeps allowed) ──
            group_text = "\n".join(
                f"{m['member']}: objective={m.get('objective')} "
                f"exit={m.get('exit_codes')}"
                for m in member_info)[:4000]
            issues_text = "\n".join(
                f"[{i.get('member')}] {i.get('problem')} "
                f"(hint: {i.get('fix_hint', '')})"
                for i in issues)[:3000]
            values2 = {
                "scope": scope,
                "max_group": configs.MAX_REFLECTION_GROUP,
                "problem_name": problem_name,
                "specs_text": specs_text,
                "doc_text": doc_text,
                "examples_text": examples_text,
                "original_main_py": base_main,
                "promising_text": promising_text,
                "unpromising_text": unpromising_text,
                "issues_text": issues_text,
                "group_text": group_text,
                "record_text": record_text,
            }
            raw2 = _invoke(agent, _FIX_SYSTEM, _FIX_USER, values2,
                           images=images or None)
            fix = _parse_json(raw2) or {}
            variants = fix.get("variants") or []
            if not variants:
                print("  [serial] repair produced no variants — "
                      "advancing to the next scope", flush=True)
                continue

            # ── deterministic execution-contract sanitizer ──
            kept = []
            for v in variants:
                code = (v.get("main") if isinstance(v, dict) else v) or ""
                ok, reason = _sanitize_variant(code)
                if ok:
                    kept.append(v)
                    continue
                record.setdefault("variant_rejections", []).append({
                    "round": round_no,
                    "scope": scope,
                    "note": (v.get("note", "") if isinstance(v, dict)
                             else "")[:120],
                    "reason": reason,
                })
                print(f"  [serial] variant REJECTED — {reason}",
                      flush=True)
            variants = kept
            if not variants:
                # every variant broke the execution contract — treat it
                # like a dead rerun: the gate forces the NEXT round to
                # repair this scope with the rejection evidence (a plain
                # 'continue' here would let the loop end with pass)
                record["execution_gate"] = {
                    "round": round_no,
                    "scope": scope,
                    "ran": 0,
                    "total": 0,
                    "evidence": ["all variants rejected: " + "; ".join(
                        r["reason"] for r in record.get(
                            "variant_rejections", [])
                        if r["round"] == round_no)],
                }
                print("  [serial] all variants rejected by the execution "
                      "contract — gate will force a repair next round",
                      flush=True)
                advanced = True
                _package_round(history_dir, round_no, members, member_info,
                               scope, record)
                round_no += 1
                break

            # record the traversed parameter ranges
            for v in variants:
                for key, values in (v.get("param_sweeps") or {}).items():
                    if not isinstance(values, list):
                        continue
                    record["param_ranges"].append({
                        "round": round_no,
                        "param": key,
                        "values": values,
                    })

            # ── build the NEW group + rerun (execution loop) ──
            new_members = _build_group(results_dir, round_no, variants)
            if not new_members:
                continue
            print(f"  [serial] {scope}: rerunning {len(new_members)} "
                  f"new experiment(s) (workers={configs.REFLECTION_WORKERS})",
                  flush=True)
            _rerun_group(new_members)

            # ── EXECUTION GATE: how many members actually ran FreeFEM ──
            ran = 0
            gate_evidence = []
            for d in new_members:
                rr = d / "run_result.json"
                nff = 0
                rc = ""
                tail = ""
                if rr.exists():
                    try:
                        data = json.loads(rr.read_text(encoding="utf-8"))
                        nff = len(data.get("freefem") or [])
                        py = ((data.get("autofreefem") or {})
                              .get("python") or {})
                        rc = str(py.get("return_code") or "?")
                        tail = (((py.get("stdout") or "") + " " +
                                 (py.get("stderr") or ""))[-120:]).strip()
                    except Exception:
                        pass
                if nff:
                    ran += 1
                if len(gate_evidence) < 5:
                    gate_evidence.append(
                        f"{d.name}: ff_records={nff} py_rc={rc} "
                        f"console_tail={tail!r}")
            record["execution_gate"] = {
                "round": round_no,
                "scope": scope,
                "ran": ran,
                "total": len(new_members),
                "evidence": gate_evidence,
            }
            print(f"  [serial] execution gate: {ran}/{len(new_members)} "
                  "members ran FreeFEM", flush=True)

            # ── record the fixes / judge the candidates ──
            if scope == "syntax_error":
                record["syntax_fixes"].append({
                    "round": round_no,
                    "issues": [i.get("problem") for i in issues],
                    "fixes": [v.get("note", "") for v in variants],
                })
            else:
                new_info = []
                for d in new_members:
                    ev = _member_evidence(d)
                    new_info.append({
                        "member": _member_id(results_dir, d),
                        "exit_codes": ev["exit_codes"],
                        "objective": ev["objective"],
                    })
                results_text = json.dumps(new_info, indent=1,
                                          ensure_ascii=False,
                                          default=str)[:4000]
                values3 = {"scope": scope,
                           "problem_name": problem_name,
                           "results_text": results_text}
                raw3 = _invoke(agent, _JUDGE_SYSTEM, _JUDGE_USER, values3)
                judge = _parse_json(raw3) or {}
                if ran == 0:
                    # hard rule: a repair whose experiments never ran
                    # FreeFEM can never be judged "accepted"
                    judge["any_success"] = False
                    record["execution_gate"]["judge_override"] = (
                        "any_success forced False — 0 members ran FreeFEM")
                bucket = ("formula_fixes" if scope == "formula_error"
                          else "designs")
                for jd in judge.get("candidates") or []:
                    record[bucket].append({
                        "round": round_no,
                        "member": jd.get("member"),
                        "status": jd.get("status", "promising"),
                        "reason": jd.get("reason", ""),
                    })
                    for combo in jd.get("accepted_params") or []:
                        record["accepted_params"].append({
                            "round": round_no,
                            "member": jd.get("member"),
                            "combo": combo,
                        })
                if judge.get("any_success"):
                    record[bucket].append({
                        "round": round_no,
                        "note": "any_success — sweep accepted"})
                    # USER 2026-10-18: ONE accepted result is enough —
                    # promote the accepted designs to the final results/
                    # (one screenshot each, no mosaic) and finish
                    n = _promote_accepted(results_dir, history_dir,
                                          new_members, judge)
                    if n:
                        done_success = True
                        print(f"  [serial] JUDGE ACCEPTED {n} design(s) — "
                              "promoting to results/ and finishing",
                              flush=True)

            members = new_members
            advanced = True
            _package_round(history_dir, round_no, members, member_info,
                           scope, record)
            round_no += 1
            break  # one repair per round — restart the serial chain

        if done_success:
            status = "pass"
            print("  [serial] accepted design found — reflection done",
                  flush=True)
            break
        if not advanced:
            status = "pass"
            print("  [serial] all scopes passed — reflection done",
                  flush=True)
            _package_round(history_dir, round_no, members, member_info,
                           "pass", record)
            break
    else:
        status = "give_up_max"

    # ── finalize: MOSAIC images -> results/<case>/figs (no folders),
    # then delete history.  USER 2026-10-18: execution.png = the 64
    # first-sweep cells; round_NN.png = repair round NN's group (16);
    # accepted.png = the finally accepted designs.  Every member's
    # screenshot lives inside its mosaic; the MinTh.vtu files stay in
    # temp/ (kept).
    figs_dir = results_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    # 1) execution.png — the first sweep's cells in order, gray
    #    placeholders for cells that produced no image
    r1_by_key = {}
    for png in history_dir.glob("tmp_r1_*.png"):
        m = re.match(r"tmp_r1_(.*)\.png", png.name)
        if m:
            r1_by_key[m.group(1)] = str(png)
    exec_refs = []
    for d in _member_dirs(results_dir):
        key = _member_id(results_dir, d).replace("/", "_").replace("\\", "_")
        exec_refs.append({"member": key, "image_png": r1_by_key.get(key)})
    if exec_refs:
        _compose_mosaic(figs_dir / "execution.png", exec_refs)
    # 2) round_NN.png — repair round NN's group (temp/group_rN): EVERY
    #    member gets a slot (gray placeholder when it produced no
    #    image) — the group's screenshots are the tmp_r{N+1}_* files
    for g in sorted((results_dir / "temp").glob("group_r*")):
        n = int(g.name.split("_r")[1])
        by_key = {}
        for png in history_dir.glob(f"tmp_r{n + 1}_*.png"):
            m = re.match(rf"tmp_r{n + 1}_(.*)\.png", png.name)
            if m:
                by_key[m.group(1)] = str(png)
        refs = []
        for d in sorted(g.iterdir()):
            key = _member_id(results_dir, d).replace(
                "/", "_").replace("\\", "_")
            refs.append({"member": key, "image_png": by_key.get(key)})
        if refs:
            _compose_mosaic(figs_dir / f"round_{n:02d}.png", refs)
    # 3) accepted.png — the accepted designs (rendered into history by
    #    _promote_accepted)
    acc_refs = [{"member": png.stem, "image_png": str(png)}
                for png in sorted(history_dir.glob("accepted_*.png"))]
    if acc_refs:
        _compose_mosaic(figs_dir / "accepted.png", acc_refs)
    _save_record(history_dir, record)
    info.reflection_attempts = round_no - 1
    info.reflection_status = status
    # Windows: the round dirs may carry the read-only attribute (it
    # blocks rmtree with 'Access is denied') — clear it per entry.
    import gc
    import os
    import stat

    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass

    for _attempt in range(5):
        gc.collect()
        shutil.rmtree(history_dir, onerror=_onerror)
        if not history_dir.exists():
            break
        time.sleep(1)
    return status
