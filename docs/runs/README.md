# Runs

Every model run and every hybrid adds one line to [ledger.jsonl](ledger.jsonl):
its inputs (path, size, date), settings, code version, counts and metrics, and
the notes given with `--notes`. One file in git, however many experiments; the
outputs themselves stay in the run's folder under `data/` (not in git). A model
run on a hybrid from `python -m src.hybrid` carries that hybrid's record too.

```bash
uv run python -m src.provenance        # the model runs as a table (below)
```

**Reference run: `20261013-gate-sam`** — the current defaults (`python -m src.pipeline`
with no options): Luke's 10 Jul `_nearest` hybrid, 0310 height model, quality gate
only where the hybrid prefers height.

## Ledger

Segment = the segment-horizon model on its test strips (m/yr); skill = against
the naive mean; region = the region-level LightGBM test MAE. All runs use the
test regions of 20260907 (`data/02_context/frozen_split_20260907.parquet`);
runs that admit more regions add new test regions, so their headline numbers
are not comparable — use `scripts/evaluate/compare_runs.py`, which keeps its
report in the challenger's `compare/` folder.

| Run | Created | Segment MAE | Skill | Tail MAE | Region MAE | Regions | Notes |
|---|---|---|---|---|---|---|---|
| 20260907-near-fz | 2026-09-07 | 1.065 | +35% | 3.27 | 2.45 | 5,750 | Reference run behind the September deliverables (05_deliverables) |
| 20261006-base | 2026-10-07 | 1.030 | +36% | 3.01 | 2.47 | 5,751 | Baseline for the pre-hybrid cleaning experiments: same as the reference but on our reprodu |
| 20261006-preclean | 2026-10-07 | 1.013 | +33% | 3.38 | 2.21 | 6,030 | Rules before hybrid |
| 20261007-both | 2026-10-07 | 1.000 | +35% | 3.39 | 2.28 | 5,849 | Row 'both masks' in 20261007_mask_runs.csv. |
| 20261007-jetty | 2026-10-07 | 1.007 | +36% | 3.08 | 2.37 | 5,805 | Row 'jetties' in 20261007_mask_runs.csv. |
| 20261007-side | 2026-10-07 | 1.009 | +37% | 3.21 | 2.35 | 5,796 | Row 'side channels' in 20261007_mask_runs.csv. |
| 20261012-hm0210 | 2026-10-12 | 1.065 | +35% | 3.27 | 2.45 | 5,750 | Re-run of the reference settings on 2026-10-12 to check reproducibility: identical to 2026 |
| 20261012-hm0310 | 2026-10-12 | 1.052 | +34% | 3.41 | 2.45 | 5,738 | Reference settings on Luke's 0310 height model (0210 plus 4,642 side-channel regions; the  |
| 20261012-verify | 2026-10-12 | 1.052 | +34% | 3.41 | 2.45 | 5,738 | Clean-up check: defaults on the new layout must reproduce 20261012-hm0310 |
| 20261012-stage-luke | 2026-10-12 | 1.031 | +35% | 3.17 | 2.44 | 5,738 | Model stage on the hybrid stage's rebuild of Luke's _nearest hybrid; should match 20261012 |
| 20261013-gate-sam | 2026-10-12 | 1.190 | +34% | 3.66 | 2.76 | 6,754 | Quality gate only where the hybrid prefers height: SAM-preferred main-channel regions come |
| 20261013-gate-sam-side | 2026-10-12 | 1.308 | +32% | 3.77 | 3.29 | 8,732 | As 20261013-gate-sam plus SAM-preferred side-channel regions, river category ngeul_<river> |

Records of runs before 2026-10-12 were written afterwards; their settings are
reconstructed (field `reconstructed`).

## What the runs showed

- **20261012-hm0210 / -verify / -stage-luke** — the reference reproduces exactly
  from the current code, on the new data layout, and (up to geometry-library
  noise) from a hybrid rebuilt from the raw files by `python -m src.hybrid`.
- **20261006-\* / 20261007-\*** — cleaning SAM before the hybrid step: small gains
  on the segment MAE, more on the tail for the jetty mask. Comparisons in
  [attachments/](attachments/).
- **20261013-gate-sam (B)** vs 20261012-hm0310 (A) — exempting SAM-preferred
  regions from the height-model quality gate adds ~1,000 modelled regions and
  improves the original test strips (1.052 → 1.027); the new regions move twice
  as much and are predicted with +30 % skill. Now the default.
- **20261013-gate-sam-side (C)** — B plus side channels: +20 % skill there; not the
  default until the predictions are checked in QGIS.
