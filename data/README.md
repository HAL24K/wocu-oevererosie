# data

Git-ignored and not backed up by git. Older material is archived in Google Drive:
`My Drive/WOCU Oevererosie/archive/` (`repo_data/`, `repo_data_20261012/`).

The numbers say how far along the process a file is. Nothing ever writes into
`01_raw`.

```
01_raw/                 as delivered by someone else, never modified
  sam/                  Lars: SAM bank lines, full AOI (73,810 lines)
  height_model/         Luke: height-model output, 0310 (scope regions, centrelines, bank points)
  luke_postprocessed/   Luke: post-processed 0310 (quality gate, VVR polygons, erosion volumes)
  scope/                scope_fase2, signaleringslijn (VVR), RWS Nevengeulen legger
  soil/ structures/ water_stations/ shipping/

02_context/             context layers we derived; inputs to the model stage
  structures.gpkg       groynes + structures   (scripts/data_prep/prep_structures.py)
  vegetatielegger.gpkg, land_use.gpkg          (phase-1 WFS downloads)
  water_stations_for_modeling.gpkg, discharge/ (notebooks/provenance/)
  region_features_v2.parquet                   bend_exposure (notebooks/provenance/)
  frozen_split_20260907.parquet                the test regions every run is graded on

03_hybrid/              hybrid line sets: the input of the model stage, one folder each
  luke_20260710/        Luke's delivery as received (default and _nearest), his note
  repro_20261006/       our reproduction of his _nearest file with his code
  <date>-<name>/        future hybrids built from 01_raw by our own rules

04_runs/<experiment>/   one model run, and everything made from it: run.json, features/,
                        model/, exports/, figures/, compare/
05_deliverables/        what went out to people, as sent
```

## The two stages

```
01_raw/sam + 01_raw/height_model ──(hybrid stage)──▶ 03_hybrid/<name>/
03_hybrid/<name>/ + 01_raw + 02_context ──(model stage)──▶ 04_runs/<experiment>/
```

`python -m src.hybrid --name <name>` is the hybrid stage, `python -m src.pipeline`
the model stage (`--hybrid <name>` to start from one); see the root README.
With no overrides a model run reproduces the reference run (`docs/runs/README.md`).

A hybrid folder from `src.hybrid` holds `hybrid.gpkg` (`lines`, `lines_all`,
`model_preference`), `nearest.gpkg` (step 1, reusable with `--nearest-from`),
`hybrid.json` and, if SAM was cleaned, `sam_cleaning_stats.csv`.

## Things to know

- **0310 vs 0210 height model.** 0310 is 0210 plus 4,642 side-channel regions
  (`ngeul_*`); the 12,130 shared regions and centrelines are identical. 0210 is in
  the Drive archive.
- **The quality gate** (`estimate_reliability_height_model == OK` in
  `luke_postprocessed/`) applies since 13 Oct only where the hybrid prefers the
  height model. Side channels are all `MISSING_DATA` there; `--side-channels`
  admits the SAM-preferred ones.
- `luke_postprocessed/` is used for the quality gate, VVR polygons, erosion volumes
  and the output GeoPackage's base layers. Luke's newer 0330 version is in Drive
  (`inputs/luke/`); switching is an open question.
