# Pipeline changes — provenance of the August 2026 graduation

What changed in `src/` between the feature branch (`feature/data-exploration-
notebooks`, 2026-08-19) and the merge of `experiment/loop-engineering`
(2026-08-25), and why. The *evidence* for every choice is in
`experiments/loop/` (LEARNINGS.md → TRACK{1,2,3}_REPORT.md → ledger.csv);
this page only records what the product pipeline now does differently.

## The one-line version

`python -m src.pipeline --experiment X` now reads the hybrid **line** delivery
instead of the height-model point cloud, cleans it with explicit rules,
adds trajectory features, validates honestly, and writes a segment-level
forecast next to the region-level product. `--source points` still runs the
pre-August pipeline unchanged.

## Step by step

| step | before (`--source points`) | after (`--source hybrid`, default) | code |
|---|---|---|---|
| 01 observations | height-model points → furthest-3 per (region, date) | hybrid lines → 20 samples/line → **structure mask** (kribben, bruggen, kades, steigers, sluizen, non-channel water; 10 m) → **cleaning rules** → furthest-3 per (region, date) → within-year median → \|v\|>50 guard | `hybrid_prep.py`, `cleaning/rules.py` |
| 02 region split | tail(3) → t1/t2/t3, grouped train/test | unchanged | — |
| 03 features | vegetation, land use, soil, hydrology, bend exposure | unchanged | — |
| 03b trajectory | — | **11 history descriptors** per region from all surveys ≤ origin year t2 (Theil–Sen v, residual spread, acceleration, last gap, recent slope, …); numeric NaN → 0 | `trajectory.py` |
| 04/05 train | LGB early-stops on the **test** set | LGB early-stops on a **15 % slice of train** (`val_frac`); test never touches fitting; extra features stored in the bundle | `train.py` |
| 06–07 rolling prediction, VVR crossing, export | unchanged | unchanged | — |
| 08 segments | — | **segment-horizon artifact**: 60 samples/line, R=5 station bins per region, own furthest-3 series, all ≥2-yr increments as span-weighted rows, grouped honest val, forward forecast → `segment_predictions.parquet` | `segments.py` |

## Cleaning rules (step 01), in application order

All thresholds are `ExperimentConfig` fields.

| rule | drops | default | why |
|---|---|---|---|
| structure mask | samples within `mask_buffer_m` of a structure/water body | 10 m | a groyne flank or bridge pier is not a bank |
| `max_tortuosity_line` | lines with length/chord > `tortuosity_max` | 3.0 | wandering lines trace vegetation, not the waterline |
| `near_bank_line` | lines closer than `nearbank_frac` × reference (min `nearbank_min_ref`) | 0.25 / 30 m | far-bank hits and mid-channel artefacts |
| `maze_survey` | surveys with path/chord > `maze_max_ratio` **and** station IQR > `maze_min_iqr` | 1.8 / 20 m | spatially incoherent surveys |
| `fragment_survey` | surveys covering < `fragment_min_cov` of the region | 0.25 | a fragment gives a biased scalar |
| `min_samples_survey` | (region, date) with < `min_samples_per_obs` samples | 8 | too few samples for a robust furthest-3 |
| `temporal_outlier_survey` | surveys > `temporal_max_dev` from the region's detrended Theil–Sen line; never below `temporal_protect_years` | 15 m / 3 yr | repair over removal: one bad survey should not sink a region |
| far-bank guard | regions with \|v\| > `farbank_v_limit` | 50 m/yr | last resort; 123 regions vs 579 before the rules |

## New inputs

- `data/02_processed/hybrid/hybrid_model_results_20260710.gpkg` — the line delivery (`hybrid_gpkg`).
- `data/02_processed/structures/structures.gpkg` — lean kribben + kunstwerken
  layers derived from the 2026-08-25 RWS GIS deliveries by
  `scripts/prep_structures.py` (`structures_gpkg`, `kunstwerk_categories`).
- `data/02_processed/wfs_context/vegetatielegger.gpkg` — already an input; the
  `Water` class is now also used for the secondary-water mask (`water_mask`).

## New outputs per run

- `03_features/<exp>/samples.parquet`, `observations.parquet` — the kept
  samples and per-(region, date) observations, for QGIS inspection.
- `04_model_outputs/<exp>/segment_predictions.parquet` — one row per
  segment: `location_id, seg, seg_center, last_year, last_dist_m, v_pred,
  pred_dist_m_horizon, n_seg_years`. Not yet consumed by the GeoPackage
  export; input for the planned any-segment-crosses-VVR alert.
- `04_model_outputs/<exp>/segment_metrics.json`.

## What was tried and deliberately *not* brought over

Survey-level increment targets, huber loss, discharge features, gully-polygon
mask, median-referenced temporal repair, year-level multi-origin training.
Each has a ledger row and a paragraph in the track reports.

## Numbers (honest validation, fresh grouped split)

| run | region MAE / tail (v>2) | segment MAE / tail / R² |
|---|---|---|
| 20260820 baseline (points-era conditions, test-ES) | 4.01 / ~8.3 | — |
| 20260822-grad | 2.63 / 5.90 | 1.15 / 3.04 / 0.49 |
| 20260825-structures (full structures mask) | 2.47 / 5.00 | 1.24 / 3.51 / 0.44 |

Tail = regions with observed \|v\| > 2 m/yr (n≈220 region, ≈400 segment).
Region-level and segment-level moved in opposite directions between the last
two runs; that is split-to-split noise, not a signal about the mask.
