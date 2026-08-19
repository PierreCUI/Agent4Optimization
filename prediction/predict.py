"""Winner-selection step of the combined batches prediction pipeline.

Reads grid_results.pkl (written by predict_multi.py), picks the config
with the smallest sum over the three targets of (MAE as % of true value
+ 4 * Std of % errors), copies its model and test predictions into
results/, and writes report.md.  Nothing is trained here — the training,
prediction and metric functions below are shared with predict_multi.py,
which is the only caller that trains.

Run with the autofreefem conda env (has torch + pandas + numpy):
    C:/Users/98215/miniconda3/envs/autofreefem/python.exe prediction/predict.py

Dependency note: pandas / numpy are already in requirements.txt;
torch is installed separately and intentionally NOT added to requirements.txt.
No other packages are used (train/test split is done manually with numpy,
so scikit-learn is not required).
"""
import pickle
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from report import REPORT_PT, write_report

# Configuration
HERE = Path(__file__).resolve().parent
SRC_CSV = HERE / "batches_results.csv"
# grid outputs written by predict_multi.py (same names as there)
GRID_PKL = HERE / "grid_results.pkl"
MODELS_DIR = HERE / "models"
CSV_DIR = HERE / "csvs"
# winner outputs of this script
RESULTS_DIR = HERE / "results"
MODEL_PT = HERE / "model.pt"  # default save target of train_model()

SEED = 42
SPLIT = (0.70, 0.15, 0.15)  # train / validation / test (per experiment)
# defaults for train_model() when no cfg is passed (predict_multi.py
# always passes an explicit cfg; the single-run flow no longer trains)
HIDDEN = [256]  # MLP hidden layers
BATCH_SIZE = 16
MAX_EPOCHS = 5000
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 300  # early stopping on validation loss

# all 8 features (mesh_borders and mesh_total_len now vary across
# experiments, so they are informative; sum_dof_node and
# num_state_fields carry the problem structure)
FEATURES = ["mesh_n", "mesh_borders", "mesh_total_len", "p_degree",
            "sum_dof_node", "num_state_fields", "hmin", "hmax"]
LOG_FEATURES = {"mesh_n", "hmin", "hmax"}  # grid/log-spaced -> log10
# resource targets only (final_objective is not part of the combined
# results.csv)
TARGETS = ["avg_time_per_epoch_s",
           "ff_max_memory_mb", "ff_avg_memory_mb"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Run-resource sampling (RAM + VRAM, recorded in the report)
class ResourceSampler:
    """Sample process RAM and CUDA VRAM in a background daemon thread."""

    def __init__(self, interval=0.5):
        """Set sampling interval (seconds) and prepare empty state."""
        self.interval = interval
        self._stop = threading.Event()
        self._samples = []  # (rss_mb, vram_mb_or_None)
        self._thread = None
        self.peak_rss_mb = None

    def start(self):
        """Start the background sampling thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Append (rss_mb, vram_mb_or_None) samples every interval until
        stop()."""
        proc = psutil.Process()
        while not self._stop.wait(self.interval):
            try:
                rss = proc.memory_info().rss / (1024 ** 2)
            except psutil.Error:
                rss = 0.0
            vram = (torch.cuda.memory_allocated() / (1024 ** 2)
                    if torch.cuda.is_available() else None)
            self._samples.append((rss, vram))

    def stop(self):
        """Stop the thread and record the Windows peak working set."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:  # Windows true peak working set, catches spikes between samples
            self.peak_rss_mb = (psutil.Process().memory_info().peak_wset
                                / (1024 ** 2))
        except Exception:
            self.peak_rss_mb = None

    def summarize(self, wall_time_s):
        """Aggregate samples into wall/RAM/VRAM metrics (dict)."""
        rss = [s[0] for s in self._samples] or [0.0]
        vram = [s[1] for s in self._samples if s[1] is not None]
        res = {"wall_time_s": wall_time_s,
               "ram_max_mb": max(max(rss),
                                 self.peak_rss_mb or 0.0),
               "ram_avg_mb": float(np.mean(rss))}
        if torch.cuda.is_available():
            res["vram_max_mb"] = (torch.cuda.max_memory_allocated()
                                  / (1024 ** 2))
            res["vram_avg_mb"] = float(np.mean(vram)) if vram else 0.0
        return res


# Data: read, transform, shuffle, split
def load_and_split():
    """Split every experiment into train/validation/test (seed fixed),
    merge the splits across experiments and shuffle their order, then
    log10-transform the log-spaced features and all targets and z-score
    with train statistics only (no leakage)."""
    df = pd.read_csv(SRC_CSV)
    X = df[FEATURES].to_numpy(dtype=np.float64)
    Y = df[TARGETS].to_numpy(dtype=np.float64)
    ids = df["run_id"].to_numpy()
    exps = df["experiment"].to_numpy()

    log_cols = [FEATURES.index(f) for f in LOG_FEATURES]
    if (X[:, log_cols] <= 0).any() or (Y <= 0).any():
        raise ValueError("log-transform requires strictly positive "
                         "log-spaced features and targets")

    # per-experiment split with the same ratio, then merged and shuffled
    rng = np.random.default_rng(SEED)
    idx = {"train": [], "val": [], "test": []}
    for exp in np.unique(exps):
        pos = np.where(exps == exp)[0]
        perm = rng.permutation(pos)
        n_tr = int(SPLIT[0] * len(pos))
        n_va = int(SPLIT[1] * len(pos))
        idx["train"].extend(perm[:n_tr])
        idx["val"].extend(perm[n_tr:n_tr + n_va])
        idx["test"].extend(perm[n_tr + n_va:])
    idx = {k: rng.permutation(v) for k, v in idx.items()}

    X = X.copy()
    for i, name in enumerate(FEATURES):
        if name in LOG_FEATURES:
            X[:, i] = np.log10(X[:, i])
    Y = np.log10(Y)  # cost metrics span orders of magnitude

    # Standardize with TRAIN statistics only (no leakage into val/test).
    mu_x, sd_x = X[idx["train"]].mean(0), X[idx["train"]].std(0)
    mu_y, sd_y = Y[idx["train"]].mean(0), Y[idx["train"]].std(0)
    Xs = (X - mu_x) / sd_x
    Ys = (Y - mu_y) / sd_y

    parts = {k: {"X": Xs[i], "Y": Ys[i], "id": ids[i],
                 "experiment": exps[i]}
             for k, i in idx.items()}
    experiment = "batches"
    feat_sets = {name: sorted(np.unique(df[name])) for name in FEATURES}
    return parts, mu_x, sd_x, mu_y, sd_y, experiment, feat_sets


# Model
class MLP(nn.Module):
    """Build an MLP with ReLU hidden layers and a linear output layer."""

    def __init__(self, n_in, n_out, hidden):
        """Build a Sequential net: Linear -> ReLU per hidden layer, then
        a linear output."""
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Run the forward pass through the net."""
        return self.net(x)


def make_loader(X, Y, shuffle, batch_size=None, seed=None):
    """Build a DataLoader over a TensorDataset; shuffling is
    deterministic (fixed generator)."""
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    seed = SEED if seed is None else seed
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=gen)


# Training with early stopping (best validation state is restored)
def train_model(parts, cfg=None):
    """Train the MLP with early stopping.  cfg overrides the module-level
    hyperparameters (hidden, lr, weight_decay, batch_size, patience,
    max_epochs, seed, device, model_pt); model_pt=None skips saving.
    Defaults reproduce the baseline configuration."""
    cfg = dict(cfg or {})
    hidden = cfg.get("hidden", HIDDEN)
    lr = cfg.get("lr", LR)
    wd = cfg.get("weight_decay", WEIGHT_DECAY)
    bs = cfg.get("batch_size", BATCH_SIZE)
    patience = cfg.get("patience", PATIENCE)
    max_epochs = cfg.get("max_epochs", MAX_EPOCHS)
    seed = cfg.get("seed", SEED)
    device = cfg.get("device", DEVICE)
    model_pt = cfg.get("model_pt", MODEL_PT)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True

    model = MLP(len(FEATURES), len(TARGETS), hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    loader = make_loader(parts["train"]["X"], parts["train"]["Y"],
                         True, bs, seed)
    X_val = torch.tensor(parts["val"]["X"], dtype=torch.float32).to(device)
    Y_val = torch.tensor(parts["val"]["Y"], dtype=torch.float32).to(device)

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train": [], "val": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    if model_pt is not None:
        torch.save(model.state_dict(), model_pt)
    return model, history, epoch


def predict(model, X):
    """Run the forward pass, back to numpy (standardized log10 space)."""
    with torch.no_grad():
        dev = next(model.parameters()).device
        x = torch.tensor(X, dtype=torch.float32).to(dev)
        return model(x).cpu().numpy()


def target_stats(true_orig, pred_orig):
    """Compute per-target (name, mse, pct_var, max_abs, max_pct, mae,
    mae_pct, rmse, rmse_pct) in original units.  pct_var is MSE as % of
    the original variance, max_pct is the largest relative offset as %
    of the true value, mae is the mean of |pred - true| and mae_pct the
    mean of the per-point relative errors |pred - true| / true * 100,
    rmse is the root mean squared error and rmse_pct the root mean
    square of the per-point relative errors.  Shared with
    predict_multi.py."""
    stats = []
    for j, name in enumerate(TARGETS):
        t, p = true_orig[:, j], pred_orig[:, j]
        mse_o = float(np.mean((p - t) ** 2))
        rmse_o = float(np.sqrt(mse_o))
        var_o = float(np.var(t))
        d = np.abs(p - t)
        rel = d / np.abs(t) * 100
        k_abs = int(np.argmax(d))
        k_rel = int(np.argmax(rel))
        mae_o = float(np.mean(d))
        stats.append((name, mse_o,
                      100.0 * mse_o / var_o if var_o > 0 else float("nan"),
                      float(d[k_abs]), float(rel[k_rel]),
                      mae_o, float(np.mean(rel)),
                      rmse_o, float(np.sqrt(np.mean(rel ** 2)))))
    return stats


# Main
def winner_bound(r):
    """Sum over the three targets of (MAE as % of true value + 4 * Std
    of % errors) for one grid row."""
    return sum(s[6] + 4.0 * std[1]
               for s, std in zip(r["stats"], r["stds"]))


def main():
    """Pick the grid winner (smallest 4-sigma bound summed over the
    targets), copy its model and test predictions into results/, and
    write report.md — no training happens here."""
    if not GRID_PKL.exists():
        raise SystemExit("grid_results.pkl not found — run "
                         "predict_multi.py first")
    with open(GRID_PKL, "rb") as f:
        rows = [r for r in pickle.load(f) if r is not None]

    winner = min(rows, key=winner_bound)
    print("== batches winner selection ==")
    print(f"Winner: {winner['id']} "
          f"(sum of MAE% + 4 * Std% = {winner_bound(winner):.3f}%)")
    for s, std in zip(winner["stats"], winner["stds"]):
        print(f"  {s[0]:<22s} MAE%={s[6]:.3f}  Std%={std[1]:.3f}  "
              f"MAE% + 4 * Std%={s[6] + 4.0 * std[1]:.3f}%")

    # copy the winner's model and test predictions into results/
    RESULTS_DIR.mkdir(exist_ok=True)
    model_src = MODELS_DIR / f"{winner['id']}.pt"
    csv_src = CSV_DIR / f"predictions_{winner['id']}.csv"
    if not model_src.exists() or not csv_src.exists():
        raise SystemExit(f"winner artifacts missing under "
                         f"{MODELS_DIR.name}/ or {CSV_DIR.name}/ — "
                         "re-run predict_multi.py")
    model_dst = RESULTS_DIR / model_src.name
    csv_dst = RESULTS_DIR / csv_src.name
    shutil.copyfile(model_src, model_dst)
    shutil.copyfile(csv_src, csv_dst)

    # report.md: same writer as before, fed with the winner's grid row
    parts, mu_x, sd_x, mu_y, sd_y, experiment, feat_sets = load_and_split()
    cfg = {"experiment": experiment, "device": DEVICE, "seed": SEED,
           "features": FEATURES, "targets": TARGETS,
           "hidden": winner["hidden"], "log_features": LOG_FEATURES,
           "lr": winner["lr"], "weight_decay": winner["weight_decay"],
           "batch_size": winner["batch_size"],
           "patience": winner["patience"]}
    res = {"wall_time_s": winner["wall"],
           "ram_max_mb": winner["ram_max"],
           "ram_avg_mb": winner["ram_avg"]}
    if winner.get("vram_max") is not None:
        res["vram_max_mb"] = winner["vram_max"]
        res["vram_avg_mb"] = winner["vram_avg"]
    write_report(cfg, feat_sets, parts, winner["epochs"],
                 winner["best_epoch"], winner["best_val"],
                 winner["stats"], res, stds=winner["stds"])

    print("\nCopied winner artifacts:")
    print(f"  {model_dst}")
    print(f"  {csv_dst}")
    print(f"Report: {REPORT_PT}")


if __name__ == "__main__":
    main()
