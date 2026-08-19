"""Report writer for the combined batches prediction pipeline.

Called automatically by predict.py after training: builds report.md from
the training configuration and the per-target result statistics.
"""
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT_PT = HERE / "report.md"

# display units per target name (shared with report_multi.py); an empty
# string means the target is dimensionless
UNITS = {"avg_time_per_epoch_s": "s", "ff_max_memory_mb": "MB",
         "ff_avg_memory_mb": "MB"}


def basic_section(cfg):
    """Return the lines for the '## 1. Basic information' section."""
    return [f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Experiment: {cfg['experiment']}",
            f"- Device: {cfg['device']}",
            f"- Seed: {cfg['seed']}"]


def data_section(cfg, feat_sets, parts):
    """Return the lines for the '## 2. Data' section (shared with
    report_multi.py)."""
    n_rows = {k: len(v["X"]) for k, v in parts.items()}
    lines = [f"- Rows: {sum(n_rows.values())} "
             f"(train {n_rows['train']} / validation {n_rows['val']} / "
             f"test {n_rows['test']})",
             "- Split: per-experiment train/validation/test "
             "(70/15/15, seed 42), merged across the 8 experiments and "
             "shuffled",
             "- Feature value sets (ascending):"]
    for name in cfg["features"]:
        vals = feat_sets[name]
        lines.append(f"  - {name}: {', '.join(f'{v:.6g}' for v in vals)} "
                     f"({len(vals)} values)")
    lines += ["- Targets:"]
    lines += [f"  - {t}" for t in cfg["targets"]]
    lines += ["- Preprocessing:",
              f"  - {', '.join(sorted(cfg['log_features']))}: log10 "
              "(grid- / log-spaced values)",
              "  - all targets: log10 (values span orders of magnitude)",
              "  - z-score with train-set statistics only (no leakage)"]
    return lines


def write_report(cfg, feat_sets, parts, epochs, best_epoch, best_val, stats,
                 res=None, out_path=None, stds=None):
    """Save the analysis as a markdown report (4 sections only);
    out_path overrides the default report.md location and stds holds the
    per-target [std_abs, std_pct, std_sq, std_sq_pct] values for the
    final-results table."""
    L = [f"# {cfg['experiment']} prediction report", ""]

    L += ["## 1. Basic information", ""]
    L += basic_section(cfg) + [""]

    L += ["## 2. Data", ""]
    L += data_section(cfg, feat_sets, parts) + [""]

    L += ["## 3. Model", ""]
    L += [f"- Structure: MLP {len(cfg['features'])} -> "
          f"{' -> '.join(map(str, cfg['hidden']))} -> {len(cfg['targets'])} "
          "(ReLU)",
          "- Optimizer:",
          "  - Algorithm: Adam",
          f"  - Learning rate: {cfg['lr']:g}",
          f"  - Weight decay: {cfg['weight_decay']:g}",
          f"- Batch size: {cfg['batch_size']}",
          "- Early stopping:",
          f"  - Patience: {cfg['patience']}",
          f"  - Stopped after: {epochs} epochs",
          f"  - Best validation MSE: {best_val:.4f} (standardized space) "
          f"@ epoch {best_epoch}", ""]

    L += ["## 4. Final results", ""]
    # NOTE: no raw "|" inside cells — markdown table parsers (e.g. MarkText)
    # would split the row on it and break the table.
    L += ["| Target | MAE | STD | MAE + 4 * STD | MAE% | STD% | "
          "MAE% + 4 * STD% |",
          "|--------|-----|-----|---------------|------|------|"
          "----------------|"]
    for j, (name, mse_o, pct_var, max_abs, max_pct, mae_o, mae_pct,
            rmse_o, rmse_pct) in enumerate(stats):
        unit = UNITS.get(name, "")
        u = f" {unit}" if unit else ""
        if stds is not None:
            sa, sp = stds[j][:2]
            cells = (f"{mae_o:.3f}{u} | {sa:.3f}{u} | "
                     f"{mae_o + 4.0 * sa:.3f}{u} | {mae_pct:.3f}% | "
                     f"{sp:.3f}% | {mae_pct + 4.0 * sp:.3f}%")
        else:
            cells = (f"{mae_o:.3f}{u} | - | - | {mae_pct:.3f}% | "
                     f"- | -")
        L.append(f"| {name} | {cells} |")
    if res is not None:
        L += ["", "Run resources:", ""]
        L += ["| Metric | Value |", "|--------|-------|",
              f"| Wall time | {res['wall_time_s']:.2f} s |",
              f"| Max RAM | {res['ram_max_mb']:.1f} MB |",
              f"| Avg RAM | {res['ram_avg_mb']:.1f} MB |"]
        if "vram_max_mb" in res:
            L += [f"| Max VRAM | {res['vram_max_mb']:.1f} MB |",
                  f"| Avg VRAM | {res['vram_avg_mb']:.1f} MB |"]
    L.append("")
    (out_path or REPORT_PT).write_text("\n".join(L) + "\n",
                                       encoding="utf-8")
