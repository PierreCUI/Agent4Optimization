"""
results.py - Convert batch.csv to results.csv (mixed-analysis format).

Keeps only the experiment-identifying + performance columns, converts
p_degree to a float, and applies the SAME 4-layer cleaning as the
analyse scripts (rc!=0 / zero-time / |obj|>2.5*Q3 / sign anomaly), so
rows removed there never appear in results.csv.
"""
from pathlib import Path

import pandas as pd

EXPERIMENT = "run_linearElasticity_Target"

_BATCH_DIR = Path(__file__).resolve().parent
_SRC = _BATCH_DIR / "batch.csv"
_OUT = _BATCH_DIR / "results.csv"

# columns kept for the mixed analysis (plus the added "experiment")
KEEP = [
    "run_id", "mesh_n", "mesh_borders", "mesh_total_len",
    "p_degree", "sum_dof_node", "num_state_fields", "hmin", "hmax",
    "ff_wall_time_s", "avg_time_per_epoch_s",
    "ff_max_memory_mb", "ff_avg_memory_mb",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """The 4-layer cleaning, identical to the analyse scripts."""
    # 1. failed runs
    df = df[df["freefem_rc"] == 0]
    # 2. zero-time rows
    df = df[df["avg_time_per_epoch_s"] > 0]
    # 3. objective outliers
    obj_abs = df["final_objective"].abs()
    if len(df) >= 20 and obj_abs.nunique() >= 2:
        q3 = obj_abs.quantile(0.75)
        if q3 > 0:
            df = df[obj_abs <= 2.5 * q3]
    # 4. sign anomalies
    obj_signed = df["final_objective"]
    pos = (obj_signed > 0).sum()
    neg = (obj_signed < 0).sum()
    minority = min(pos, neg)
    if minority and minority <= max(2, 0.01 * len(df)):
        if pos <= neg:
            df = df[obj_signed <= 0]
        else:
            df = df[obj_signed >= 0]
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
