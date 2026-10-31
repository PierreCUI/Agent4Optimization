"""Pipeline state container — holds all data as it flows through the system."""

import json
from pathlib import Path
from typing import Optional


class Info:
    """
    In-memory record for one optimisation problem.

    Each attribute is populated by a different pipeline stage.
    """

    def __init__(self, problem_name: str, results_root: Path = None):
        self.problem_name: str = problem_name

        # ── collected from disk ──────────────────────────────────────
        self.input_files: dict[str, str] = {}  # {filename: content}
        self.problem_description: str = ""  # raw text (possibly filtered)

        # ── initialization ───────────────────────────────────────────
        self.problem_spec: Optional[dict] = None  # parse_problem output
        self.mesh_spec: Optional[dict] = None  # handle_mesh output

        # ── example selection ────────────────────────────────────────
        self.selected_names: list[str] = []  # example folder names
        self.example_specs: list[dict] = []  # [{"name":..., "initial":..., "mesh":...}]
        self.examples: list[dict] = []  # [{"name":..., "py":..., "mesh":...}]

        # ── code generation ──────────────────────────────────────────
        self.generated_code: str = ""  # main.py content

        # ── static check ─────────────────────────────────────────────
        # (the static repair loop writes the fixed code straight into generated_code)

        # ── reflection ─────────────────────────────────────────────
        self.reflection_attempts: int = 0
        self.repair_records: list[dict] = []  # round-0 console + per-round repair records
        self.reflection_status: str = "not_run"  # pass|give_up|give_up_max

        # ── results path ─────────────────────────────────────────────
        self.results_dir: Path = (results_root or Path("results")) / problem_name

    # convenience --------------------------------------------------------

    def save_json(self, filename: str, data) -> None:
        """Save *data* as JSON inside ``results_dir``.

        *filename* may contain a nested relative path (e.g. collected
        input files under ``subdir/file.dat``) — the parent directory is
        created so a subdirectory demo never crashes the pipeline.
        """
        out_path = self.results_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_text(self, filename: str, text: str) -> None:
        """Save *text* inside ``results_dir`` (nested paths supported)."""
        out_path = self.results_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
