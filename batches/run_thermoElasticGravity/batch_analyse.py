"""
batch_analyse.py — Analyse batch.csv (full parameter grid).

1. Remove abnormal data (failed runs, zero-time rows).
2. Effect strength via Morris elementary effects: for each feature the
   rows are grouped by parameter level (mean metric per level) and
   mu* = mean(|dy / d(log x)|) between adjacent levels — the average
   metric change per decade of the parameter, in percent of the metric
   mean (per P-order for p_degree, which is linearly spaced).
   Spearman rho is kept as a secondary direction signal.
3. Sequential heatmap (magnitude) saved as PNG + CSV.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend must be set before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_BATCH_DIR = Path(__file__).resolve().parent
_CSV = _BATCH_DIR / "batch.csv"
_OUT_CSV = _BATCH_DIR / "batch_correlation.csv"
_OUT_PNG = _BATCH_DIR / "batch_heatmap.png"

# Input features — only the parameters actually swept in batchrun
# (fixed parameters are not plotted).  The batch uses a single degree
# (unequal u/T degree combos are skipped), recorded as one column.
FEATURES = [
    "hmax", "hmin", "mesh_n", "p_degree",
]

# Output metrics — FreeFEM stage only (per-epoch average time, not total)
METRICS = [
    "avg_time_per_epoch_s", "ff_max_memory_mb", "ff_avg_memory_mb",
]

# Sequential blue ramp for magnitude (reference palette steps, light→dark):
# lightest means "near zero effect", darkest means "strongest effect".
_SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#5598e7",
              "#3987e5", "#1c5cab", "#0d366b"]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV and drop abnormal rows."""
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} rows")

    # 1. Failed runs
    bad_rc = (df["freefem_rc"] != 0).sum()
    df = df[df["freefem_rc"] == 0]
    print(f"Dropped failed runs (rc != 0): {bad_rc}")

    # 2. Zero-time rows (never actually ran)
    zero_time = (df["avg_time_per_epoch_s"] <= 0).sum()
    df = df[df["avg_time_per_epoch_s"] > 0]
    print(f"Dropped zero-time rows: {zero_time}")

    # 3. Outlier protection (nonlinear problems only): rows whose rc is
    # 0 but whose |objective| is an order of magnitude beyond the
    # distribution are failed experiments in disguise (e.g. degenerate
    # solves that still exited cleanly).  abs() because some problems
    # have negative objectives.  DISABLED for linear problems: their
    # objective scales exactly with the parameters (J ~ f^2, J ~ E),
    # so large corner values are legitimate physics, not failures
    # (manual inspection: only the thermo problems have outliers).
    obj_abs = df["final_objective"].abs()
    if len(df) >= 20 and obj_abs.nunique() >= 2:
        q3 = obj_abs.quantile(0.75)
        if q3 > 0:
            keep = obj_abs <= 2.5 * q3
            dropped = (~keep).sum()
            if dropped:
                df = df[keep]
                print(f"Dropped objective outliers (|obj| > 2.5*Q3): "
                      f"{dropped}")
    # 4. Sign anomaly: a handful of opposite-sign objectives among an
    # overwhelmingly single-sign distribution are errors (e.g. one
    # negative value among 164 positive ones -- the alpha0 corner).
    # Genuinely single-sign problems are untouched (no minority).
    obj_signed = df["final_objective"]
    pos = (obj_signed > 0).sum()
    neg = (obj_signed < 0).sum()
    minority = min(pos, neg)
    if minority and minority <= max(2, 0.01 * len(df)):
        if pos <= neg:
            df = df[obj_signed <= 0]
            dropped = pos
        else:
            df = df[obj_signed >= 0]
            dropped = neg
        print(f"Dropped sign anomalies (opposite-sign objectives): "
              f"{dropped}")
    print(f"Kept: {len(df)} rows\n")
    return df


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """Effect strength via Morris mu* FEATURES × METRICS.

    For each feature the rows are grouped by parameter level (mean metric
    per level), then elementary effects are computed between adjacent
    levels: |dy / d(log x)|.  mu* is their mean — the average metric
    change per decade of the parameter — in percent of the metric mean
    (per P-order for p_degree, which is linearly spaced).  Absolute
    values make it work for non-monotone effects.  Spearman rho is
    reported as a secondary direction/consistency signal.
    """
    rows = []
    for f in FEATURES:
        for m in METRICS:
            grouped = df.groupby(f)[m]
            levels = sorted(grouped.groups.keys())
            ys = np.array([grouped.mean()[lv] for lv in levels])
            ref = float(df[m].mean())
            if (len(levels) < 3 or df[m].nunique() < 2):
                # Fewer than 3 distinct levels, or constant metric → no
                # meaningful effect
                rows.append({"feature": f, "metric": m,
                             "n_rows": len(df),
                             "metric_mean": round(ref, 6),
                             "moris_u_star": 0.0,
                             "moris_u_star_pct": 0.0,
                             "effect_rel_pct": 0.0,
                             "spearman_rho": 0.0,
                             "p_value": np.nan})
                continue
            dy = np.abs(np.diff(ys))
            if f == "p_degree":
                # P degree is a string ('P1'..) and linearly spaced —
                # elementary effects per unit degree instead of per decade
                xs = np.array([int(str(lv)[1:]) for lv in levels],
                              dtype=float)
                dx = np.abs(np.diff(xs))
                ees = dy[dx > 0] / dx[dx > 0]
            else:
                xs = np.array(levels, dtype=float)
                # abs() so signed parameters (e.g. f < 0) work in log space
                dlog = np.abs(np.diff(np.log(np.abs(xs))))
                ees = dy[dlog > 0] / dlog[dlog > 0]
            mu_star = float(ees.mean()) if len(ees) else 0.0
            mu_star_pct = (100.0 * mu_star / abs(ref)
                           if ref else np.nan)
            effect_rel = (100.0 * (ys.max() - ys.min()) / abs(ref)
                          if ref else np.nan)
            if f == "p_degree":
                # rank on the numeric degree order, not on the strings
                rho, p = spearmanr(
                    df[f].map(lambda v: int(str(v)[1:])), df[m])
            else:
                rho, p = spearmanr(df[f], df[m])
            rows.append({"feature": f, "metric": m,
                         "n_rows": len(df),
                         "metric_mean": round(ref, 6),
                         "moris_u_star": round(mu_star, 6),
                         "moris_u_star_pct": round(mu_star_pct, 2),
                         "effect_rel_pct": round(effect_rel, 2),
                         "spearman_rho": round(rho, 4),
                         "p_value": round(p, 6)})
    corr = pd.DataFrame(rows)
    corr.to_csv(_OUT_CSV, index=False)
    print(f"Effect table: {_OUT_CSV}")

    # Rank parameters by impact for each metric
    print("\nImpact ranking per metric "
          "(Morris mu*, % of metric mean per decade; "
          "per P-order for p_degree):")
    for m in METRICS:
        ranked = corr[corr["metric"] == m].sort_values(
            "moris_u_star_pct", ascending=False)
        print(f"  {m:<20}: " + "  ".join(
            f"{r['feature']}={r['moris_u_star_pct']:.1f}%"
            for _, r in ranked.iterrows()))
    return corr


def draw_heatmap(corr: pd.DataFrame) -> None:
    """Sequential heatmap of Morris mu* effect strength (% per decade)."""
    matrix = corr.pivot(index="feature", columns="metric",
                        values="moris_u_star_pct").loc[FEATURES, METRICS]

    # Sequential single-hue ramp (magnitude): light = near zero effect.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "effect_strength", _SEQ_STEPS, N=256)

    fig, ax = plt.subplots(figsize=(9, 2 + 0.8 * len(FEATURES)))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    vals = matrix.values.astype(float)
    # sqrt-ish scale so moderate effects stay visible beside large ones
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0,
                                       vmax=np.nanmax(vals))
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

    # Cell labels — values carry the information, never the color alone
    for i in range(len(FEATURES)):
        for j in range(len(METRICS)):
            v = vals[i, j]
            label = "n/a" if np.isnan(v) else f"{v:.1f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#fcfcfb",
                              ec="none", alpha=0.72))

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES, fontsize=10)
    ax.set_title("Morris mu* effect strength "
                 "(% of metric mean per decade; per P-order for p_degree)",
                 fontsize=13, color="#0b0b0b", pad=14)

    # Recessive grid
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FEATURES), 1), minor=True)
    ax.grid(which="minor", color="#e1e0d9", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("mu* (% per decade; per P-order for p_degree)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(_OUT_PNG, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Heatmap: {_OUT_PNG}")


def main():
    if not _CSV.exists():
        print(f"Missing {_CSV} — run batchrun.py first.")
        sys.exit(1)
    df = load_and_clean(_CSV)
    corr = analyse(df)
    draw_heatmap(corr)


if __name__ == "__main__":
    main()
