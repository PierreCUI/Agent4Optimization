"""Multi-config report writer for the combined batches sweep.

Called automatically by predict_multi.py: builds report_multi.md from the
sweep rows (one per trained config) and their per-target results.
"""
from pathlib import Path

from report import UNITS, basic_section, data_section

HERE = Path(__file__).resolve().parent
REPORT_MULTI_PT = HERE / "report_multi.md"


def write_multi_report(cfg, feat_sets, parts, rows):
    """rows: list of dicts, one per trained config, with keys: id, hidden,
    lr, weight_decay, batch_size, patience, epochs, best_epoch, best_val,
    wall, ram_max, ram_avg, vram_max (or None), vram_avg (or None), stats
    (list of (name, mse, pct_var, max_abs, max_pct, mae, mae_pct, rmse,
    rmse_pct) in cfg['targets'] order) and stds (list of [std_abs,
    std_pct, std_sq, std_sq_pct] per target)."""
    targets = cfg["targets"]
    L = [f"# {cfg['experiment']} multi-config prediction report", ""]

    L += ["## 1. Basic information", ""]
    L += basic_section(cfg) + [""]

    L += ["## 2. Data", ""]
    L += data_section(cfg, feat_sets, parts) + [""]

    # 3. configurations (grid; training outcomes live in section 4)
    L += ["## 3. Configurations", ""]
    L += ["| Config | Structure | LR | Batch size | Weight decay | Patience |",
          "|--------|-----------|----|------------|--------------|----------|"]
    for r in rows:
        L.append(f"| {r['id']} | {' -> '.join(map(str, r['hidden']))} | "
                 f"{r['lr']:g} | {r['batch_size']} | {r['weight_decay']:g} | "
                 f"{r['patience']} |")
    L.append("")

    # 4. final results
    L += ["## 4. Final results", ""]

    L += ["Training outcomes:", ""]
    L += ["| Config | Epochs | Best epoch | Best val MSE | Wall time (s) | "
          "Max RAM (MB) | Avg RAM (MB) | Max VRAM (MB) | Avg VRAM (MB) |",
          "|--------|--------|------------|--------------|---------------|"
          "--------------|--------------|---------------|---------------|"]
    for r in rows:
        vram_max = (f"{r['vram_max']:.1f}"
                    if r.get("vram_max") is not None else "-")
        vram_avg = (f"{r['vram_avg']:.1f}"
                    if r.get("vram_avg") is not None else "-")
        L.append(f"| {r['id']} | {r['epochs']} | {r['best_epoch']} | "
                 f"{r['best_val']:.4f} | {r['wall']:.2f} | "
                 f"{r['ram_max']:.1f} | {r['ram_avg']:.1f} | "
                 f"{vram_max} | {vram_avg} |")
    L.append("")

    # per-target result tables
    for j, t in enumerate(targets):
        unit = UNITS.get(t, "")
        u = f" {unit}" if unit else ""
        L += [f"### {t}", ""]
        L += ["| Config | MAE | STD | MAE + 4 * STD | MAE% | STD% | "
              "MAE% + 4 * STD% |",
              "|--------|-----|-----|---------------|------|------|"
              "----------------|"]
        for r in rows:
            _, _, _, _, _, mae, mapct, _, _ = r["stats"][j]
            std_abs, std_pct = r["stds"][j][:2]
            L.append(f"| {r['id']} | {mae:.3f}{u} | {std_abs:.3f}{u} | "
                     f"{mae + 4.0 * std_abs:.3f}{u} | {mapct:.3f}% | "
                     f"{std_pct:.3f}% | "
                     f"{mapct + 4.0 * std_pct:.3f}% |")
        L.append("")

    # 5. analyse: per-target top-25 configs ranked by the 4-sigma error
    # bound (MAE as % of true value + 4 * Std of % errors, ascending)
    L += ["## 5. Analyse", ""]
    analyse_header = ("| Rank | Config | MAE | STD | MAE + 4 * STD | "
                      "MAE% | STD% | MAE% + 4 * STD% |")
    analyse_sep = ("|------|--------|-----|-----|---------------|"
                   "------|------|----------------|")
    for j, t in enumerate(targets):
        unit = UNITS.get(t, "")
        u = f" {unit}" if unit else ""
        L += [f"### {t}", ""]
        L += [analyse_header, analyse_sep]
        scored = sorted(rows, key=lambda r: r["stats"][j][6]
                        + 4.0 * r["stds"][j][1])
        for k, r in enumerate(scored[:25], 1):
            _, _, _, _, _, mae, mae_pct, _, _ = r["stats"][j]
            std_abs = r["stds"][j][0]
            std_pct = r["stds"][j][1]
            bound = mae_pct + 4.0 * std_pct
            L.append(f"| {k} | {r['id']} | {mae:.3f}{u} | "
                     f"{std_abs:.3f}{u} | {mae + 4.0 * std_abs:.3f}{u} | "
                     f"{mae_pct:.3f}% | {std_pct:.3f}% | {bound:.3f}% |")
        L.append("")
    L.append("")
    REPORT_MULTI_PT.write_text("\n".join(L) + "\n", encoding="utf-8")
