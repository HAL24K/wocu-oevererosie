# Notebooks

```
04_model/20260617a/00_master.ipynb   the pipeline, cell by cell
01_scenarios.ipynb                   parameter replay per workflow step
reference/                           6 · kept for a specific reason
archive/                             27 · superseded, kept for traceability
```

## Scenario playground

`01_scenarios.ipynb` — expensive inputs load once, then each section is a
PARAMS cell + run cell + comparison for one workflow step: point-selection
width (step 1), the hybrid within-year rule, split-seed luck (step 2), a
LightGBM grid (step 4), and model choice over the 25-year horizon (step 5).
Committed executed, so the latest results are readable without a kernel.

Was 58 notebooks across 14 directories with no signposting. 24 were deleted in
`3f0…` — near-duplicate model comparisons, notebooks that never ran a cell, and
pre-March exploration. All recoverable from git history.

## The pipeline

```
04_model/20260617a/00_master.ipynb
```

Runs everything end to end by calling `src.pipeline`. Driven by a single `EXPERIMENT`
string; duplicate the folder, change it, re-run. The first markdown cell holds the most
accurate flow diagram in the repository.

Last executed 2026-06-17: 8,094 regions, 2026–2050, 202,350 predicted bank positions,
4/4 acceptance checks. Steps 1–3 reproduce the 20260314 feature parquets byte-for-byte.

## reference/

Not current, but each is here for a reason that outlives it.

| Notebook | Why it is kept |
|---|---|
| `20260311_preprocess_region_split_v2.ipynb` | **Live dependency.** Produced `data/02_processed/erosion/region_features_v2.parquet`, which `feature_engineering.py` still reads for `bend_exposure`. Deleting this loses the provenance of two model features |
| `20260311_explore_point_selection.ipynb` | The only rationale for taking the **3 furthest** OK points rather than the nearest — a deliberate worst-case convention that shapes every velocity in the model |
| `02c_eda_clean_discharge.ipynb` | Produced `data/water_stations_timeseries/cleaned/discharge/`, read by the hydrology features |
| `compare_postprocessed_gpkg_20260310_vs_20260330.ipynb` | The open question of whether to move to the newer delivery. Never executed |
| `01_segmented_bank_lines.ipynb` | Asks whether ~100 m is the right prediction unit; compares 1/5/10/20 sub-segments. **Written but never run** |
| `01_eda_shipping_passages.ipynb` | Why shipping intensity was assessed and *not* adopted — block resolution too coarse for ~12k regions |

## archive/

| Folder | What it is |
|---|---|
| `pipeline-generations/` | The 20260311 → 20260313 → 20260314 sequences that `20260617a` replaced. Directly comparable to the current run, which is why they survive |
| `phase1/` | The WFS/DataHandler era, matching `src/legacy/`. Includes the notebooks that first built the vegetation and land-use context layers |
| `water-stations/` | Discharge and water-level acquisition, cleaning and station selection. The cleaned output is still a pipeline input; only the notebooks are archived |

## Conventions

- Notebooks bootstrap with an `os.chdir` to the repo root plus `sys.path.insert`. Since
  `uv sync` installs the project editable this is no longer needed for imports, but it
  still sets the working directory that relative data paths depend on.
- Kernel: **WOCU erosion (3.12)** — see the root README.
- Outputs are committed deliberately: for the pipeline notebook they are the only record
  of what a run produced. This costs ~80 MB across the repository, almost all of it
  stored figures rather than code.
