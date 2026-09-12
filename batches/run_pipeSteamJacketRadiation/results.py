"""results.py - Convert batch.csv to results.csv (mixed-analysis format).

Keeps only the experiment-identifying + performance columns; cleaning is
rc != 0 and zero-time ONLY (no outlier filtering).
"""
from pathlib import Path

import pandas as pd

EXPERIMENT = "run_pipeSteamJacketRadiation"

_BATCH_DIR = Path(__file__).resolve().parent

_SRC = _BATCH_DIR / "batch.csv"
_OUT = _BATCH_DIR / "results.csv"

KEEP = [
    "run_id", "hmin", "hmax", "diffusion", "v0", "mesh_n",
    "p_degree", "ff_wall_time_s", "avg_time_per_epoch_s",
    "ff_max_memory_mb", "ff_avg_memory_mb",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["freefem_rc"] == 0]
    df = df[df["avg_time_per_epoch_s"] > 0]
    return df


def main():
    df = pd.read_csv(_SRC)
    before = len(df)
    df = clean(df)
    out = df[KEEP].copy()
    out["p_degree"] = out["p_degree"].map(lambda v: float(str(v)[1:]))
    out.insert(1, "experiment", EXPERIMENT)
    out.to_csv(_OUT, index=False)
    print(f"{EXPERIMENT}: {before} -> {len(out)} rows -> {_OUT.name}")


if __name__ == "__main__":
    main()
