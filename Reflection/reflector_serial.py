# -*- coding: utf-8 -*-
"""Serial reflection loop (USER RULES 2026-10-31, revised).

Reflection works as follows:

  1. SYNTAX REPAIR — when the executed batch produced NO design snapshot
     (MinTh.vtu) at all, the failure is a syntax error.  The TEXT model
     checks the syntax and repairs ONE code, which is written back into
     the SAME batch's cells (the first sweep's 64 cells keep their
     per-cell diffusion/v0/penalty values, re-applied from the cell
     paths) and the SAME batch re-executes.  A syntax repair NEVER
     creates a new batch.

  2. OVERALL JUDGMENT + DESIGN REPAIR — when the batch produced design
     snapshots, the IMAGE model judges every experiment's final results
     against the problem: is at least ONE design correct?  If yes — or
     when the configs.OVERALL_CHECK_ENABLED switch is False (default) —
     Reflection post-processes and finishes.  If no, a design repairer,
     responsible for EVERY kind of problem (formulas, parameters, and
     anything else), proposes a NEW batch of variants (capped by
     configs.MAX_REFLECTION_GROUP), which executes and becomes the next
     round's batch.  A syntax failure inside such a new batch is again
     repaired in place by duty 1.

Post-processing (finalize): mosaics into results/<case>/figs, cleanup.
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

# ======================================================================
# prompts
# ======================================================================

_SYNTAX_DETECT_SYSTEM = """\
You are the syntax detector of an optimization-code repair loop.  You
receive a GROUP of experiments (the same problem, one code per sweep
cell / per variant) and their execution evidence.  The group produced
NO design snapshot at all — the failure is therefore a syntax error:
the program died before writing any design.

Report ONLY syntax-scope issues: imports, call signatures/arguments,
spelling, fixed contents, the extracted/designed rules, the mesh
buildmesh chain and the mesh-generation failures it causes, and
identifiers entering the FreeFEM code containing underscores (problem
name, Domain mesh variables, border names).

Return ONE JSON:
{{"passed": true, "issues": []}}
or
{{"passed": false, "issues": [{{"member": "<member id>",
"problem": "<one sentence>", "fix_hint": "<how to repair>"}}]}}
No markdown.  No code fences.
"""

_SYNTAX_DETECT_USER = """\
Problem: {problem_name}

Specs (initial.json + mesh.json):
{specs_text}

Current group's evidence:
{round_text}

Return the detection JSON.
"""

_SYNTAX_FIX_SYSTEM = """\
You are the syntax repairer of an optimization-code repair loop.  You
receive the detection report and the base code of the current batch.

Repair RULES:
- Return ONE complete main.py that repairs ALL listed syntax issues.
- START from the base code — make ONLY the most necessary
  modifications, the smallest change that addresses the issues.
- NO UNRELATED CHANGES: anything not needed for the fix must stay
  exactly as it is in the base code.
- The repaired code goes back into EVERY cell of the SAME batch (the
  sweep re-runs the same cells) — keep the parameter assignments
  (diffusion, v0, and the material-penalty constant) untouched so the
  per-cell values survive.

EXECUTION CONTRACT (hard pipeline rules — a violation drops the repair):
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
{{"main": "<complete Python code>", "note": "..."}}
No markdown.  No code fences.
"""

_SYNTAX_FIX_USER = """\
Problem: {problem_name}

Specs (initial.json + mesh.json):
{specs_text}

AutoFreeFEM documentation (the SAME document the generator used):
{doc_text}

Reference examples (selected for this problem, full code):
{examples_text}

The base main.py of the current batch — edit it minimally, keep its
imports and structure:
{original_main_py}

Detection report (syntax):
{issues_text}

Current group (member id, main.py path, evidence):
{group_text}

Return the repaired-main JSON.
"""

_OVERALL_JUDGE_SYSTEM = """\
You are the overall judge of an optimization-code repair loop.  You
receive the problem's input description, its specifications, and the
FINAL RESULT of every experiment (rendered design images, objectives,
exit states).

Your task: judge whether AT LEAST ONE experiment produced a CORRECT
final design — a sensible optimized solution of the stated problem:
the physics is right, the material layout matches the loads, supports
and boundary conditions, and the design shows no obvious defect.

Return ONE JSON:
{{"any_correct": true|false, "reason": "<one or two sentences naming the
best experiment, or the common defects when nothing is correct>"}}
No markdown.  No code fences.
"""

_OVERALL_JUDGE_USER = """\
Problem: {problem_name}

Input description:
{description_text}

Specs (initial.json + mesh.json):
{specs_text}

Experiments (member id, objective, exit codes):
{round_text}

Return the judge JSON.
"""

_DESIGN_FIX_SYSTEM = """\
You are the design repairer of an optimization-code repair loop.  The
overall judge found NO correct design among the experiments.  You are
responsible for EVERY kind of problem — formulas, parameters, and
anything else — and must repair the code so that a rerun produces at
least one correct design.

Repair RULES:
- NEVER modify the previous round's files — propose NEW experiments:
  up to {max_group} variants total (multiple repair methods are
  welcome).
- Every variant carries the COMPLETE Python code (main.py).
- START from the base code (original_main_py — the code that already
  ran the whole sweep).  Make ONLY the most necessary modifications.
  Fix formula errors, wrong parameter values, design errors — whatever
  the judge's verdict and the design images show.

EXECUTION CONTRACT (hard pipeline rules — a variant that breaks them
is DROPPED before it runs):
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
{{"variants": [{{"main": "<complete Python code>", "note": "..."}},
 ...]}}
No markdown.  No code fences.
"""

_DESIGN_FIX_USER = """\
Problem: {problem_name}

Input description:
{description_text}

Specs (initial.json + mesh.json):
{specs_text}

AutoFreeFEM documentation (the SAME document the generator used):
{doc_text}

Reference examples (selected for this problem, full code):
{examples_text}

The base main.py — the code that already ran the whole sweep; edit it
minimally, keep its imports and structure:
{original_main_py}

The overall judge's verdict (no correct design — the common defects):
{judge_reason}

Variants rejected by the execution contract in earlier rounds (never
repeat them):
{rejections_text}

Current group (member id, main.py path, evidence):
{group_text}

Return the variants JSON.
"""


# ── group / evidence plumbing ─────────────────────────────────────────

def _member_dirs(results_dir: Path) -> list[Path]:
    """The first sweep's cells (each leaf holds main.py — the run
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
    return {"syntax_fixes": [], "overall_judgments": [], "design_fixes": [],
            "rounds": []}


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


def _description_text(results_dir: Path) -> str:
    for name in ("description.txt",):
        p = results_dir / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8",
                                   errors="replace")[:4000]
            except OSError:
                pass
    return "(no input description recorded)"


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


# ── LLM plumbing (AgentRouter-compatible) ─────────────────────────────

def _invoke(agent, system: str, user_text: str, values: dict, images=None):
    """Pass the TEMPLATES to the agent's writer (langchain fills the
    placeholders from *values* — the {{...}} JSON examples stay escaped
    in the templates).  Text calls (no images) go to the TEXT model;
    image calls go to the IMAGE (vision) model via the router."""
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


# ── group construction / execution ────────────────────────────────────

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
        if len(members) >= configs.MAX_REFLECTION_GROUP:
            break
        idx += 1
        d = root / f"m{idx:02d}"
        d.mkdir(parents=True)
        (d / "main.py").write_text(main_code, encoding="utf-8",
                                   newline="\n")
        if mesh_src.exists():
            shutil.copy2(mesh_src, d / "mesh.txt")
        members.append(d)
    return members


def _rerun_group(members: list[Path]) -> None:
    """Execution on the group: run_single per member, up to
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


def _patch_batch(results_dir: Path, members: list[Path],
                 repaired_code: str) -> None:
    """Write the syntax repair back into the SAME batch (USER RULE
    2026-10-31: a syntax repair never creates a new batch).

    First-sweep cells keep their per-cell material-penalty /
    diffusion / v0 values, re-applied from the cell paths with the
    runner's own sweep substitution; a design-repair group's members
    receive the repaired code as-is."""
    from Execution.runner import _sweep_main, _sweep_meta

    try:
        _, penalties, penalty_name, _ = _sweep_meta(results_dir)
    except Exception:
        penalties, penalty_name = [], ""
    for d in members:
        code = repaired_code
        rel = _member_id(results_dir, d).replace("\\", "/")
        m = re.search(
            r"diffv0_beta/([^/]+)/diff_([0-9.eE+-]+)_v0_([0-9.eE+-]+)$", rel)
        if m and penalty_name:
            lm = re.match(r"[A-Za-z]+_([0-9.eE+-]+)$", m.group(1))
            try:
                penalty = float(lm.group(1)) if lm else penalties[0]
                code = _sweep_main(repaired_code, penalty_name, penalty,
                                   float(m.group(2)), float(m.group(3)))
            except (ValueError, TypeError):
                pass
        try:
            (d / "main.py").write_text(code, encoding="utf-8",
                                       newline="\n")
        except OSError as e:
            warnings.warn(f"reflector_serial patch {d.name} failed: {e}")


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


# ── the loop ──────────────────────────────────────────────────────────

def run_reflection(agent, info, config) -> str:
    results_dir = info.results_dir
    problem_name = info.problem_name
    history_dir = results_dir / "temp" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    record = _load_record(history_dir)
    specs_text = _specs_text(results_dir)
    description_text = _description_text(results_dir)

    # ── the SAME context materials the GENERATOR used ──
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
        images = [p for m in member_info for p in m.get("images", [])][:32]
        group_text = "\n".join(
            f"{m['member']}: objective={m.get('objective')} "
            f"exit={m.get('exit_codes')}"
            for m in member_info)[:4000]

        if any(m.get("minth") for m in member_info):
            # ── OVERALL JUDGMENT (config switch) ──
            # Design snapshots exist.  When configs.OVERALL_CHECK_ENABLED
            # is False (default), skip the judgment and post-process
            # directly.  When it is True, the IMAGE model judges whether
            # at least one design is correct: yes -> post-process; no ->
            # design repair -> a NEW batch executes next round.
            if not getattr(config, "OVERALL_CHECK_ENABLED", False):
                status = "pass"
                print("  [serial] design snapshot(s) produced — overall "
                      "check disabled; post-processing only", flush=True)
                _package_round(history_dir, round_no, members, member_info,
                               "pass", record)
                round_no += 1
                break

            judge_values = {
                "problem_name": problem_name,
                "description_text": description_text,
                "specs_text": specs_text,
                "round_text": round_text,
            }
            raw = _invoke(agent, _OVERALL_JUDGE_SYSTEM, _OVERALL_JUDGE_USER,
                          judge_values, images=images or None)
            judge = _parse_json(raw) or {}
            record.setdefault("overall_judgments", []).append({
                "round": round_no,
                "any_correct": bool(judge.get("any_correct")),
                "reason": str(judge.get("reason", ""))[:300],
            })
            _save_record(history_dir, record)
            if judge.get("any_correct"):
                status = "pass"
                print("  [serial] overall judge: at least one correct "
                      "design — post-processing", flush=True)
                _package_round(history_dir, round_no, members, member_info,
                               "pass", record)
                round_no += 1
                break

            # ── DESIGN REPAIR: new batch (IMAGE model, all problem
            # kinds) ──
            print("  [serial] overall judge: no correct design -> design "
                  "repair (new batch)", flush=True)
            rejections = [r.get("reason", "") for r in
                          record.get("variant_rejections", [])][:5]
            fix_values = {
                "max_group": configs.MAX_REFLECTION_GROUP,
                "problem_name": problem_name,
                "description_text": description_text,
                "specs_text": specs_text,
                "doc_text": doc_text,
                "examples_text": examples_text,
                "original_main_py": base_main,
                "judge_reason": str(judge.get("reason", ""))[:600],
                "rejections_text": ("\n".join(f"- {r}" for r in rejections)
                                    if rejections else "(none so far)"),
                "group_text": group_text,
            }
            raw2 = _invoke(agent, _DESIGN_FIX_SYSTEM, _DESIGN_FIX_USER,
                           fix_values, images=images or None)
            fix = _parse_json(raw2) or {}
            variants = fix.get("variants") or []
            kept = []
            for v in variants:
                code = (v.get("main") if isinstance(v, dict) else v) or ""
                ok, reason = _sanitize_variant(code)
                if ok:
                    kept.append(v)
                    continue
                record.setdefault("variant_rejections", []).append({
                    "round": round_no,
                    "note": (v.get("note", "") if isinstance(v, dict)
                             else "")[:120],
                    "reason": reason,
                })
                print(f"  [serial] variant REJECTED — {reason}",
                      flush=True)
            variants = kept
            if not variants:
                print("  [serial] design repair produced no usable "
                      "variants — giving up", flush=True)
                _package_round(history_dir, round_no, members, member_info,
                               "design_repair", record)
                break
            new_members = _build_group(results_dir, round_no, variants)
            if not new_members:
                print("  [serial] no new experiments materialised — "
                      "giving up", flush=True)
                break
            print(f"  [serial] design repair: rerunning {len(new_members)} "
                  f"new experiment(s) (workers={configs.REFLECTION_WORKERS})",
                  flush=True)
            _rerun_group(new_members)
            record.setdefault("design_fixes", []).append({
                "round": round_no,
                "judge_reason": str(judge.get("reason", ""))[:300],
                "notes": [v.get("note", "") for v in variants],
            })
            _save_record(history_dir, record)
            members = new_members
            _package_round(history_dir, round_no, members, member_info,
                           "design_repair", record)
            round_no += 1
            continue

        # ── SYNTAX REPAIR (TEXT model; the SAME batch, no new batch) ──
        # No MinTh.vtu at all: the failure is a syntax error.  The text
        # model checks the syntax and repairs ONE code; the SAME batch's
        # cells are patched in place (sweep cells keep their per-cell
        # parameters) and re-executed.
        print("  [serial] no design snapshot — syntax repair (text model)",
              flush=True)
        values = {
            "problem_name": problem_name,
            "specs_text": specs_text,
            "round_text": round_text,
        }
        raw = _invoke(agent, _SYNTAX_DETECT_SYSTEM, _SYNTAX_DETECT_USER,
                      values)
        report = _parse_json(raw) or {"passed": False, "issues": []}
        issues = report.get("issues") or []
        if not issues:
            # the detector found nothing — the missing design snapshot
            # is the evidence: the failure is a syntax error regardless
            issues = [{
                "member": "(all)",
                "problem": (
                    "the executed group produced NO design snapshot "
                    "(MinTh.vtu) — classified as a syntax error: the "
                    "program failed before writing any design"),
                "fix_hint": (
                    "find and repair the syntax-level defect that "
                    "prevents the program from completing a run"),
            }]
            print("  [serial] syntax detector passed, but no design "
                  "snapshot exists — forcing a syntax repair", flush=True)
        issues_text = "\n".join(
            f"[{i.get('member')}] {i.get('problem')} "
            f"(hint: {i.get('fix_hint', '')})"
            for i in issues)[:3000]
        # the base code of the CURRENT batch: the first member's code
        # when the batch is a design-repair group (variants), else the
        # original results/main.py
        batch_base = base_main
        try:
            first_main = members[0] / "main.py"
            if first_main.exists():
                batch_base = first_main.read_text(encoding="utf-8",
                                                  errors="replace")
        except (IndexError, OSError):
            pass
        fix_values = {
            "problem_name": problem_name,
            "specs_text": specs_text,
            "doc_text": doc_text,
            "examples_text": examples_text,
            "original_main_py": batch_base,
            "issues_text": issues_text,
            "group_text": group_text,
        }
        raw2 = _invoke(agent, _SYNTAX_FIX_SYSTEM, _SYNTAX_FIX_USER,
                       fix_values)
        fix = _parse_json(raw2) or {}
        repaired = fix.get("main") or ""
        if not isinstance(repaired, str) or not repaired.strip():
            print("  [serial] syntax repair produced no code — giving up",
                  flush=True)
            _package_round(history_dir, round_no, members, member_info,
                           "syntax_repair", record)
            break
        ok, reason = _sanitize_variant(repaired)
        if not ok:
            record.setdefault("variant_rejections", []).append({
                "round": round_no,
                "note": str(fix.get("note", ""))[:120],
                "reason": reason,
            })
            print(f"  [serial] repaired code REJECTED — {reason}; giving up",
                  flush=True)
            _package_round(history_dir, round_no, members, member_info,
                           "syntax_repair", record)
            break
        print(f"  [serial] syntax repair: patching and rerunning the SAME "
              f"{len(members)} cell(s) (workers={configs.REFLECTION_WORKERS})",
              flush=True)
        _patch_batch(results_dir, members, repaired)
        _rerun_group(members)
        record.setdefault("syntax_fixes", []).append({
            "round": round_no,
            "issues": [i.get("problem") for i in issues],
            "note": str(fix.get("note", ""))[:200],
        })
        _save_record(history_dir, record)
        _package_round(history_dir, round_no, members, member_info,
                       "syntax_repair", record)
        round_no += 1

    else:
        status = "give_up_max"

    # ── finalize (POST-PROCESSING): MOSAIC images -> results/<case>/figs
    # (no folders), then delete history.  USER 2026-10-18: execution.png
    # = the 64 first-sweep cells; round_NN.png = repair round NN's group.
    # Every member's screenshot lives inside its mosaic; the MinTh.vtu
    # files stay in temp/ (kept).
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
