"""
Pre-compute ProblemSpec + mesh spec for database examples.

Scans database/freefem_examples/ and runs the one-loop spec generation
(generate -> review -> repair) on any folder missing initial.json or
mesh.json, writing BOTH files together once the loop passes.
"""

import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from API import DeepSeekCodeAgentV4Flash  # noqa: E402
from Initialization import collect_input_files, run_spec_generation  # noqa: E402


def run_example_initialization(agent=None):
    if agent is None:
        agent = DeepSeekCodeAgentV4Flash()
    examples_dir = Path("database") / "freefem_examples"

    if not examples_dir.exists():
        print(f"Directory not found: {examples_dir}")
        return

    folders = sorted([d for d in examples_dir.iterdir() if d.is_dir()])
    print(f"Found {len(folders)} example folders in {examples_dir}\n")

    for i, folder in enumerate(folders):
        name = folder.name
        print(f"[{i + 1}/{len(folders)}] {name}", flush=True)

        # Check what's needed
        need_initial = not (folder / "initial.json").exists()
        need_mesh = not (folder / "mesh.json").exists()

        if not need_initial and not need_mesh:
            print("  -> both exist, skip", flush=True)
            continue

        # Collect non-.py input files
        input_files = collect_input_files(folder)

        # One-loop generation: BOTH specs together, reviewed together,
        # repaired together until the review passes or the fix budget is
        # exhausted.  Database content is initialised WITHOUT reference
        # examples (the default) — these folders ARE the reference set
        # themselves.
        print("  -> running spec generation...", flush=True)
        specs = run_spec_generation(
            agent, input_files, problem_name=name, output_dir=folder,
        )
        initial = specs.get("initial") or {}
        mesh = specs.get("mesh") or {}
        ok_initial = bool(initial.get("designed"))
        ok_mesh = bool(mesh.get("designed"))
        passed = specs.get("_passed") is True and ok_initial and ok_mesh

        if not passed:
            print(
                f"  -> FAILED: designed empty (initial={ok_initial}, mesh={ok_mesh})",
                flush=True,
            )
            # A partial/failed run must not bake in an invalid spec — remove
            # stale artifacts so the next run retries the WHOLE case.
            for stale_name in ("initial.json", "mesh.json"):
                stale = folder / stale_name
                if stale.exists():
                    stale.unlink()
            continue

        # Write BOTH files together (they were generated and reviewed as
        # one unit).
        with open(folder / "initial.json", "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)
        with open(folder / "mesh.json", "w", encoding="utf-8") as f:
            json.dump(mesh, f, indent=2, ensure_ascii=False)

        ds = initial.get("designed", {})
        mds = mesh.get("designed", {})
        print(
            f"  -> OK ({ds.get('problem_type', '?')}, {ds.get('dimension', '?')}D; "
            f"mesh src={mds.get('mesh_code_source', '?')})",
            flush=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    run_example_initialization()
