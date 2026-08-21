# Loop engineering — experiment contract

Branch-local experiment (2026-08-20 →) iterating on three tracks:
**1 cleaning rules · 2 multi-t targets · 3 resolution.** Goal: find what
actually works, graduate the winners to structured implementation on main.

## The frozen ruler

Decided with Alexander on 2026-08-20:

- **Primary loop metric: LGB test tail-MAE** (`v_test > 2 m/yr`) — the
  risicogevallen. A rule/approach must improve it to survive.
- **Hard guardrail: overall test MAE** — the number colleagues know from the
  demo. No variant graduates that regresses it. Stable banks must stay
  accurately predicted.
- **Coverage: soft floor 90%** of CORE. Exclusions beyond that need
  eye-validated evidence the removed cases are artefacts. Repair (drop a
  line/survey) beats removal (drop a region) whenever possible.

## Frozen sets (committed here)

- `frozen_test_regions.csv` — the 1,174 test regions of 20260820-hybrid-masked.
  **No variant ever trains on these.** Drawn once, never redrawn.
- `core_regions.csv` — CORE: frozen-test regions surviving the v0 baseline.
  Coverage and fixed-denominator MAE are reported against CORE.
- `tail_frozen_regions.csv` — CORE regions with v0 `v_test > 2`. Reported as
  the frozen-tail view: because cleaning changes the *target*, the per-variant
  tail is a moving set; this one is not.

## Honesty rules

1. Cleaning changes v_test itself (that is the point — artefact removal is
   measurement correction). Any variant that shrinks `tail_n` must produce a
   before/after grid of the changed regions for eye validation before it is
   marked kept. Tail-MAE dropping via tail shrinkage alone is not a win.
2. Fixed denominators: cross-variant comparisons use CORE ∩ both survivors.
3. Early stopping uses the frozen test set (same as the baseline pipeline) —
   a known optimism source, identical across variants, so deltas are fair.
   Absolute numbers get re-verified with honest validation at graduation.

## Machinery

- `src/loop/rules.py` — composable cleaning rules (sample/line/survey level).
- `src/loop/harness.py` — cached fast pipeline: samples → observations →
  dist_per_year → split (frozen) → features (from caches) → LGB → ledger.
  One variant ≈ 1–2 min. Parity vs the real pipeline verified on v0
  (MAE 3.988 vs 4.014, R² 0.335 vs 0.334, identical exclusions & naive).
- Caches under `data/03_features/loop/` (gitignored), built by
  `scripts/loop_build_caches.py`.
- Per-variant artefacts (test preds, dist_per_year, rule stats) under
  `data/03_features/loop/variants/<name>/`.
- `ledger.csv` (committed) — one row per variant, append-only.

## Ledger column notes

`lgb_tail_mae`/`tail_n` = variant's own tail. `tail_frozen_mae` = on the
frozen tail ids (kept subset). `coverage_core`/`core_mae` = CORE views.
`persist_mae` = v_train passthrough. `vtest_std` tracks target de-noising.
