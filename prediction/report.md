# batches prediction report

## 1. Basic information

- Generated: 2026-08-19 02:27:41
- Experiment: batches
- Device: cuda
- Seed: 42

## 2. Data

- Rows: 2983 (train 2084 / validation 445 / test 454)
- Split: per-experiment train/validation/test (70/15/15, seed 42), merged across the 8 experiments and shuffled
- Feature value sets (ascending):
  - mesh_n: 10, 20, 25, 30, 40, 50, 60, 75, 100 (9 values)
  - mesh_borders: 4, 6, 8 (3 values)
  - mesh_total_len: 4, 6, 6.4, 10 (4 values)
  - p_degree: 1, 2, 3 (3 values)
  - sum_dof_node: 1, 2, 3 (3 values)
  - num_state_fields: 1, 2 (2 values)
  - hmin: 0.0004, 0.0005, 0.0006, 0.000625, 0.00075, 0.0008, 0.0009375, 0.001, 0.001125, 0.0012, 0.00125, 0.0015, 0.0016, 0.001875, 0.002, 0.00225, 0.0025, 0.003, 0.00375, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.0075, 0.008, 0.009, 0.01, 0.0105, 0.012, 0.0125, 0.014, 0.015, 0.016, 0.018, 0.01875, 0.02, 0.021, 0.0225, 0.024, 0.025, 0.028, 0.03, 0.0375, 0.04, 0.045, 0.06, 0.08, 0.09, 0.12 (50 values)
  - hmax: 0.008, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.2, 0.3 (13 values)
- Targets:
  - avg_time_per_epoch_s
  - ff_max_memory_mb
  - ff_avg_memory_mb
- Preprocessing:
  - hmax, hmin, mesh_n: log10 (grid- / log-spaced values)
  - all targets: log10 (values span orders of magnitude)
  - z-score with train-set statistics only (no leakage)

## 3. Model

- Structure: MLP 8 -> 1024 -> 1024 -> 3 (ReLU)
- Optimizer:
  - Algorithm: Adam
  - Learning rate: 0.001
  - Weight decay: 1e-05
- Batch size: 64
- Early stopping:
  - Patience: 300
  - Stopped after: 1672 epochs
  - Best validation MSE: 0.0017 (standardized space) @ epoch 1372

## 4. Final results

| Target | MAE | STD | MAE + 4 * STD | MAE% | STD% | MAE% + 4 * STD% |
|--------|-----|-----|---------------|------|------|----------------|
| avg_time_per_epoch_s | 0.463 s | 0.723 s | 3.356 s | 3.265% | 3.709% | 18.102% |
| ff_max_memory_mb | 13.052 MB | 27.793 MB | 124.224 MB | 2.072% | 2.339% | 11.429% |
| ff_avg_memory_mb | 6.715 MB | 12.032 MB | 54.842 MB | 1.721% | 1.589% | 8.076% |

Run resources:

| Metric | Value |
|--------|-------|
| Wall time | 242.76 s |
| Max RAM | 1290.9 MB |
| Avg RAM | 1286.9 MB |
| Max VRAM | 88.3 MB |
| Avg VRAM | 84.1 MB |

