# Segment × horizon (≥ 2 jaar) sweep — 2026-08-26

Frozen holdout, e8 obs, traj2 segment features, all ≥2-yr origin/end pairs for training, span-weighted (clip 2–5), test = latest ≥2-yr pair per segment; test-set early stopping (same as every other ledger row). Ledger variants `hz-R<R>`. R=1 reproduces k5-H2-multi-w.

| R | ≈ m/segment | n_train | n_test | starved | MAE | tail (n) | naïef | skill | R² | pos_err_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 3 060 | 715 | 0.0% | **1.37** | 3.11 (72) | 1.77 | 0.23 | 0.28 | 1.35 |
| 2 | 50 | 6 029 | 1 397 | 1.7% | **1.22** | 3.07 (132) | 1.64 | 0.25 | 0.38 | 1.27 |
| 5 | 20 | 14 212 | 3 303 | 5.9% | **1.07** | 3.10 (298) | 1.50 | 0.28 | 0.46 | 1.13 |
| 10 | 10 | 26 938 | 6 259 | 9.1% | **1.03** | 3.01 (558) | 1.46 | 0.30 | 0.49 | 1.05 |
| 20 | 5 | 36 312 | 8 578 | 30.9% | **0.85** | 2.89 (612) | 1.20 | 0.29 | 0.49 | 0.85 |
