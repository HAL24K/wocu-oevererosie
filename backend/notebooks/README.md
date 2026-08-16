# Notebooks

There are 58 notebooks here and only one of them is the pipeline.

## The one that matters

```
04_model/20260617a/00_master.ipynb
```

Runs the whole pipeline end to end by calling `src.pipeline`. Driven by a single
`EXPERIMENT` string; duplicate the folder and change it to start a new run. Its first
markdown cell holds the most accurate flow diagram in the repository.

Last executed 2026-06-17: 8,094 regions, 2026–2050, 202,350 predicted bank positions,
4/4 acceptance checks. Steps 1–3 reproduce the 20260314 feature parquets byte-for-byte.

## Everything else

Historical. Kept because the reasoning behind several decisions exists nowhere else,
but none of it is on the path to a prediction.

| Folder | What it is |
|---|---|
| `04_model/20260314_pipeline_building/` | The six-notebook sequence `20260617a` replaced. Superseded but directly comparable |
| `04_model/20260313_pipeline_building/` | The attempt before that |
| `04_model/20260311_working_results/` | Feature exploration, point-selection study, first iterative prediction |
| `04_model/*.ipynb` (loose) | March model comparisons and one-off analyses. Several near-duplicate names — `20260311_model_comparison` vs `..._OK` vs `20260312_...` — none is current |
| `04_model/archive/` | Pre-March work |
| `01_water_stations/` | Discharge and water-level acquisition and cleaning. Produced `data/water_stations_timeseries/cleaned/`, which the pipeline still reads |
| `02_wfs_pipeline/` | WFS acquisition for the phase-1 stack. Corresponds to `src/legacy/` |
| `03_height_erosion/` | Early temporal exploration of `punten_oever` |
| `05_shipping/` | Shipping-passage EDA (2026-05). Assessed as a feature, not adopted — block resolution too coarse for ~12k regions |
| `06_visualization/` | Sub-region segmentation study (2026-06). **Written but never executed** — an open question about whether ~100 m is the right prediction unit |

## Decisions that live only in here

- **Bank position uses the mean of the 3 *furthest* OK points**, not the nearest — a
  deliberate worst-case reading. Rationale in `04_model/20260311_working_results/20260311_explore_point_selection.ipynb`
  and `scripts/viz_point_selection.py`.
- **`bend_exposure` came from `20260311_working_results/20260311_preprocess_region_split_v2.ipynb`**,
  which is why `feature_engineering.py` still reads it from a parquet instead of
  computing it.
- **The 20260330 delivery was compared against 20260310** in
  `04_model/compare_postprocessed_gpkg_20260310_vs_20260330.ipynb`. The pipeline still
  uses 20260310.

## Conventions

- Notebooks bootstrap with an `os.chdir` to `backend/` plus `sys.path.insert`. Since
  `uv sync` installs the project editable this is no longer needed for imports, but it
  still sets the working directory that relative data paths depend on.
- Kernel: **WOCU erosion (3.12)** — see the root README for registering it.
- Outputs are committed, deliberately: they are the only record of what a run produced.
