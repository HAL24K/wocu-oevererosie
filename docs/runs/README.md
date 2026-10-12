# Runs

One JSON file per model run: inputs (path, size, date), settings, code version,
counts and metrics. `src/pipeline/provenance.py` writes it at the end of every
run, into the run folder (`data/04_runs/<experiment>/run.json`) and here. The copy
here is tracked in git, so it outlives the run's data.

Delete a record only together with every number that quotes it.

## Ledger

Segment = the segment-horizon model on test strips (m/yr); skill = against the
naive mean. Region = the region-level LightGBM test MAE (m/yr). All runs share
the frozen test regions of 20260907 (`data/02_context/frozen_split_20260907.parquet`).

| Run | Segment MAE | Skill | Tail MAE | Region MAE | Regions | Data kept |
|---|---|---|---|---|---|---|
| [20260907-near-fz](20260907-near-fz.json) | 1.065 | +35% | 3.27 | 2.45 | 5,750 | yes |
| [20261006-base](20261006-base.json) | 1.030 | +36% | 3.01 | 2.47 | 5,751 | — |
| [20261006-preclean](20261006-preclean.json) | 1.013 | +33% | 3.38 | 2.21 | 6,030 | — |
| [20261007-side](20261007-side.json) | 1.009 | +37% | 3.21 | 2.35 | 5,796 | — |
| [20261007-jetty](20261007-jetty.json) | 1.007 | +36% | 3.08 | 2.37 | 5,805 | — |
| [20261007-both](20261007-both.json) | 1.000 | +35% | 3.39 | 2.28 | 5,849 | — |
| [20261012-hm0210](20261012-hm0210.json) | 1.065 | +35% | 3.27 | 2.45 | 5,750 | — |
| [20261012-hm0310](20261012-hm0310.json) | 1.052 | +34% | 3.41 | 2.45 | 5,738 | yes |
| [20261012-verify](20261012-verify.json) | 1.052 | +34% | 3.41 | 2.45 | 5,738 | — |

- **20260907-near-fz** — behind the September deliverables in `data/05_deliverables/`.
- **20261006-\* / 20261007-\*** — cleaning SAM before the hybrid step, on our own
  reproduction of Luke's hybrid. Comparisons: [20261007_mask_runs.csv](20261007_mask_runs.csv),
  [20261006_compare_prehybrid.txt](20261006_compare_prehybrid.txt); how close the
  reproduction is to Luke's file: [20261006_hybrid_repro_vs_luke.txt](20261006_hybrid_repro_vs_luke.txt).
- **20261012-hm0210** — the reference settings re-run: identical to 20260907-near-fz,
  so the reference is reproducible from the current code.
- **20261012-hm0310** — the same on Luke's 0310 height model, now the default.
- **20261012-verify** — a plain `python -m src.pipeline` after the October clean-up:
  identical to 20261012-hm0310, so the restructure changed no result.

Records before 2026-10-12 were written afterwards; their settings are
reconstructed and say so in `notes`.
