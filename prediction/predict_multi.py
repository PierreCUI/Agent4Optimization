"""Multi-config grid search for the combined batches prediction pipeline
(parallel).

Full grid over the whole hyperparameter space:
25 network structures (1 to 5 hidden layers x 64/128/256/512/1024 units)
x 3 learning rates x 6 batch sizes = 450 configs.
Weight decay (1e-5) and patience (300) stay fixed at the baseline values.
Configs are trained in parallel (WORKERS worker processes; each worker has
its own CUDA context, so results stay deterministic and identical to a
serial run), all evaluated on the same per-experiment split (merged and
shuffled), persisted to grid_results.pkl (so report_multi.md can be
regenerated without re-running), and recorded in report_multi.md.
Progress lines are printed and appended to grid_search.log (created if
missing).

Run with the autofreefem conda env:
    C:/Users/98215/miniconda3/envs/autofreefem/python.exe prediction/predict_multi.py
"""
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import predict as P
from report_multi import write_multi_report

HERE = Path(__file__).resolve().parent
GRID_PKL = HERE / "grid_results.pkl"
GRID_LOG = HERE / "grid_search.log"
MODELS_DIR = HERE / "models"
CSV_DIR = HERE / "csvs"

WORKERS = 8

# 1 to 5 hidden layers for each width (width-major order)
STRUCTURES = [[w] * d
              for w in (64, 128, 256, 512, 1024)
              for d in (1, 2, 3, 4, 5)]
LRS = [1e-4, 1e-3, 1e-2]
BATCH_SIZES = [8, 16, 32, 64, 128, 256]
GRID = [(hidden, lr, bs)
        for hidden in STRUCTURES for lr in LRS for bs in BATCH_SIZES]


def _cid(hidden, lr, bs):
    """Format the config id, e.g. h64x64x64_lr0.01_bs16."""
    return f"h{'x'.join(map(str, hidden))}_lr{lr:g}_bs{bs}"


def run_config(job):
    """Train + evaluate one config.  Runs in a worker process (spawn-safe)."""
    hidden, lr, bs, parts, mu_x, sd_x, mu_y, sd_y, test_true_orig = job
    cfg = {"hidden": hidden, "lr": lr, "batch_size": bs,
           "model_pt": MODELS_DIR / f"{_cid(hidden, lr, bs)}.pt"}
    # every trained model is kept under models/
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sampler = P.ResourceSampler()
    sampler.start()

    model, history, epochs = P.train_model(parts, cfg)
    pred_s = P.predict(model, parts["test"]["X"])
    pred_log = pred_s * sd_y + mu_y
    pred_orig = 10 ** pred_log
    stats = P.target_stats(test_true_orig, pred_orig)
    # per-target spread of the errors (MAE pair from |d|, RMSE pair
    # from the squared errors), shown in the best-config tables
    stds = []
    for j in range(len(P.TARGETS)):
        d = np.abs(pred_orig[:, j] - test_true_orig[:, j])
        rel = d / np.abs(test_true_orig[:, j]) * 100
        stds.append([float(np.std(d)), float(np.std(rel)),
                     float(np.std(d ** 2)), float(np.std(rel ** 2))])

    # per-config test predictions (16 columns: features + experiment +
    # run_id + true/pred per target)
    feat_orig = parts["test"]["X"] * sd_x + mu_x
    for i, name in enumerate(P.FEATURES):
        if name in P.LOG_FEATURES:
            feat_orig[:, i] = 10 ** feat_orig[:, i]
    out = pd.DataFrame(feat_orig, columns=P.FEATURES)
    out.insert(len(P.FEATURES), "experiment",
               parts["test"]["experiment"])
    out.insert(len(P.FEATURES) + 1, "run_id", parts["test"]["id"])
    for j, name in enumerate(P.TARGETS):
        out[f"{name}_true"] = test_true_orig[:, j]
        out[f"{name}_pred"] = pred_orig[:, j]
    out.to_csv(CSV_DIR / f"predictions_{_cid(hidden, lr, bs)}.csv",
               index=False)

    sampler.stop()
    res = sampler.summarize(time.perf_counter() - t0)
    return {"id": _cid(hidden, lr, bs),
            "hidden": hidden, "lr": lr,
            "weight_decay": P.WEIGHT_DECAY,
            "batch_size": bs, "patience": P.PATIENCE,
            "epochs": epochs,
            "best_epoch": int(np.argmin(history["val"])) + 1,
            "best_val": float(min(history["val"])),
            "wall": res["wall_time_s"],
            "ram_max": res["ram_max_mb"],
            "ram_avg": res["ram_avg_mb"],
            "vram_max": res.get("vram_max_mb"),
            "vram_avg": res.get("vram_avg_mb"),
            "stats": stats, "stds": stds}


def ensure_outputs():
    """Create the models/csvs folders and grid_search.log if they do
    not exist yet."""
    MODELS_DIR.mkdir(exist_ok=True)
    CSV_DIR.mkdir(exist_ok=True)
    if not GRID_LOG.exists():
        GRID_LOG.write_text("")


def main():
    """Run all 450 configs in parallel (resuming existing results) and
    write report_multi.md."""
    ensure_outputs()
    # progress goes to stdout and is appended to grid_search.log, so a
    # killed run still leaves a readable progress log behind
    log = GRID_LOG.open("a", encoding="utf-8")

    def report(msg):
        """Print a progress line and append it to the log."""
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    report(f"== batches multi-config grid search "
           f"(WORKERS={WORKERS}) ==")
    t_start = time.perf_counter()
    (parts, mu_x, sd_x, mu_y, sd_y, experiment,
     feat_sets) = P.load_and_split()
    test_true_log = parts["test"]["Y"] * sd_y + mu_y
    test_true_orig = 10 ** test_true_log
    # parts is a small dict of numpy arrays, pickled to every worker
    jobs = [(h, lr, bs, parts, mu_x, sd_x, mu_y, sd_y, test_true_orig)
            for h, lr, bs in GRID]

    # resume: rows already present in grid_results.pkl are reused, only
    # the missing configs are trained
    rows = [None] * len(jobs)
    existing = {}
    if GRID_PKL.exists():
        with open(GRID_PKL, "rb") as f:
            for r in pickle.load(f):
                existing[r["id"]] = r
    todo = [i for i, j in enumerate(jobs)
            if _cid(*j[:3]) not in existing]
    for i, j in enumerate(jobs):
        cid = _cid(*j[:3])
        if cid in existing:
            rows[i] = existing[cid]
    report(f"Resuming: {len(todo)} of {len(jobs)} configs to train "
           f"({len(existing)} already done)")

    total = len(jobs)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_config, jobs[i]): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            rows[i] = fut.result()
            r = rows[i]
            done = sum(1 for x in rows if x is not None)
            report(f"  [{done}/{total}] {r['id']:<30s} "
                   f"epochs={r['epochs']:<5d} wall={r['wall']:.1f}s "
                   f"time_MSE={r['stats'][0][1]:.6g}")

    # persist raw rows so the report can be regenerated without re-running
    with open(GRID_PKL, "wb") as f:
        pickle.dump(rows, f)

    cfg = {"experiment": experiment, "device": P.DEVICE, "seed": P.SEED,
           "features": P.FEATURES, "targets": P.TARGETS,
           "log_features": P.LOG_FEATURES}
    write_multi_report(cfg, feat_sets, parts, rows)
    report(f"\nReport: report_multi.md ({len(rows)} configs) in "
           f"{time.perf_counter() - t_start:.0f}s")
    log.close()


if __name__ == "__main__":
    main()
