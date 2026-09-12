"""
batch_analyse.py — Analyse batch.csv (full Cartesian grid).

1. Remove abnormal data: failed runs (rc != 0) and zero-time rows ONLY —
   NO objective-outlier or sign filtering (per user decision).
2. Effect strength via Morris elementary effects, grouped by parameter
   level: mu* = mean(|dy / d(log x)|) between adjacent levels, in
   percent of the metric mean (per P-order for p_degree).
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

FEATURES = ["hmin", "hmax", "diffusion", "v0", "mesh_n", "p_degree"]

METRICS = [
    "avg_time_per_epoch_s", "ff_max_memory_mb", "ff_avg_memory_mb",
]

_SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#5598e7",
              "#3987e5", "#1c5cab", "#0d366b"]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV; drop failed runs and zero-time rows ONLY."""
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} rows")

    bad_rc = (df["freefem_rc"] != 0).sum()
    df = df[df["freefem_rc"] == 0]
    print(f"Dropped failed runs (rc != 0): {bad_rc}")

    zero_time = (df["avg_time_per_epoch_s"] <= 0).sum()
    df = df[df["avg_time_per_epoch_s"] > 0]
    print(f"Dropped zero-time rows: {zero_time}")

    print(f"Kept: {len(df)} rows\n")
    return df


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """Effect strength via Morris mu* FEATURES x METRICS, grouped by
    parameter level."""
    rows = []
    for f in FEATURES:
        for m in METRICS:
            grouped = df.groupby(f)[m]
            levels = sorted(grouped.groups.keys())
            ys = np.array([grouped.mean()[lv] for lv in levels])
            ref = float(df[m].mean())
            if (len(levels) < 3 or df[m].nunique() < 2):
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
                xs = np.array([int(str(lv)[1:]) for lv in levels],
                              dtype=float)
                dx = np.abs(np.diff(xs))
                ees = dy[dx > 0] / dx[dx > 0]
            else:
                xs = np.array(levels, dtype=float)
                dlog = np.abs(np.diff(np.log(np.abs(xs))))
                ees = dy[dlog > 0] / dlog[dlog > 0]
            mu_star = float(ees.mean()) if len(ees) else 0.0
            mu_star_pct = (100.0 * mu_star / abs(ref)
                           if ref else np.nan)
            effect_rel = (100.0 * (ys.max() - ys.min()) / abs(ref)
                          if ref else np.nan)
            if f == "p_degree":
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
    matrix = corr.pivot(index="feature", columns="metric",
                        values="moris_u_star_pct").loc[FEATURES, METRICS]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "effect_strength", _SEQ_STEPS, N=256)

    fig, ax = plt.subplots(figsize=(9, 2 + 0.8 * len(FEATURES)))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    vals = matrix.values.astype(float)
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0,
                                       vmax=np.nanmax(vals))
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

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
