"""
Pre-compute ProblemSpec + mesh for database examples.

Scans database/freefem_examples/ and runs parse_problem / handle_mesh
on any folder missing initial.json or mesh.json.
"""

import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from API import DeepSeekCodeAgentV4Flash  # noqa: E402
from Initialization import collect_input_files, handle_mesh, parse_problem  # noqa: E402


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

        # Run parse_problem if needed
        if need_initial:
            print("  -> running parse_problem...", end=" ", flush=True)
            result = parse_problem(agent, input_files, problem_name=name)
            if not isinstance(result, dict) or not result or result.get("_error") or not result.get("designed"):
                # `not result` catches the {} path (empty/all-excluded
                # inputs) and `not result.get("designed")` catches a
                # parsed-but-design-failed spec (parse_problem returns
                # {"extracted": ..., "designed": {}} without _error) —
                # an empty spec must never bake in as a reference.
                err = (result or {}).get("_error", "non-dict or empty result") if isinstance(result, dict) else "non-dict result"
                print(f"FAILED: {err}", flush=True)
                # A previous partial run must not bake in an invalid spec
                # — remove stale success artifacts so the next run retries
                # BOTH phases (mesh depends on the parsed spec).
                for stale_name in ("initial.json", "mesh.json"):
                    stale = folder / stale_name
                    if stale.exists():
                        stale.unlink()
                continue
            with open(folder / "initial.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            ds = result.get("designed", {})
            print(f"OK ({ds.get('problem_type', '?')}, {ds.get('dimension', '?')}D)", flush=True)

        # Run handle_mesh if needed
        if need_mesh:
            print("  -> running handle_mesh...", end=" ", flush=True)
            ctx = result.get("designed") if need_initial else None
            if not need_initial and (folder / "initial.json").exists():
                with open(folder / "initial.json", "r", encoding="utf-8") as f:
                    ctx = json.load(f).get("designed")

            mesh_result = handle_mesh(agent, input_files, problem_context=ctx)
            if not isinstance(mesh_result, dict) or not mesh_result or mesh_result.get("_error"):
                # `not mesh_result` catches the {} empty-input path.
                err = (mesh_result or {}).get("_error", "non-dict or empty result") if isinstance(mesh_result, dict) else "non-dict result"
                print(f"FAILED: {err}", flush=True)
                # A failed mesh spec must not become the reference example.
                stale = folder / "mesh.json"
                if stale.exists():
                    stale.unlink()
                continue
            with open(folder / "mesh.json", "w", encoding="utf-8") as f:
                json.dump(mesh_result, f, indent=2, ensure_ascii=False)
            ds = mesh_result.get("designed", {})
            print(
                f"OK (src={ds.get('mesh_code_source', '?')}, {ds.get('dimension', '?')}D)",
                flush=True,
            )

    print("\nDone.")


if __name__ == "__main__":
    run_example_initialization()
