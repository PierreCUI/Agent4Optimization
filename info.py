"""Pipeline state container — holds all data as it flows through the system."""

import json
from pathlib import Path
from typing import Optional


class Info:
    """
    In-memory record for one optimisation problem.

    Each attribute is populated by a different pipeline stage.
    """

    def __init__(self, problem_name: str):
        self.problem_name: str = problem_name

        # ── collected from disk ──────────────────────────────────────
        self.input_files: dict[str, str] = {}  # {filename: content}
        self.problem_description: str = ""  # raw text (possibly filtered)

        # ── initialization ───────────────────────────────────────────
        self.problem_spec: Optional[dict] = None  # parse_problem output
        self.mesh_spec: Optional[dict] = None  # handle_mesh output

        # ── example selection ────────────────────────────────────────
        self.selected_names: list[str] = []  # example folder names
        self.examples: list[dict] = []  # [{"name":..., "py":...}]

        # ── code generation ──────────────────────────────────────────
        self.generated_code: str = ""  # main.py content

        # ── static check ─────────────────────────────────────────────
        self.fixed_code: str = ""  # code after auto_fix

        # ── execution ─────────────────────────────────────────────
        self.run_result: Optional[dict] = None  # latest run_problem() output

        # ── reflection ─────────────────────────────────────────────
        self.reflection_attempts: int = 0
        self.reflection_history: list[dict] = []  # per-iteration records
        self.reflection_status: str = "not_run"  # pass|give_up_max|give_up_loop

        # ── results path ─────────────────────────────────────────────
        self.results_dir: Path = Path("results") / problem_name

    # convenience --------------------------------------------------------

    def save_json(self, filename: str, data) -> None:
        """Save *data* as JSON inside ``results_dir``."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_text(self, filename: str, text: str) -> None:
        """Save *text* inside ``results_dir``."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / filename, "w", encoding="utf-8") as f:
            f.write(text)
