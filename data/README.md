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

04_runs/<experiment>/   one model run: features/, model/, exports/, run.json
05_deliverables/        what went out to people, as sent
```

## The two stages

```
01_raw/sam + 01_raw/height_model ──(hybrid stage)──▶ 03_hybrid/<name>/
03_hybrid/<name>/ + 01_raw + 02_context ──(model stage)──▶ 04_runs/<experiment>/
```

The model stage defaults are in `src/pipeline/config.py`; with no overrides a run
reproduces the reference run (`docs/runs/README.md`). The hybrid stage is
`scripts/hybrid/reproduce_hybrid.py` for now.

## Things to know

- **0310 vs 0210 height model.** 0310 is 0210 plus 4,642 side-channel regions
  (`ngeul_*`); the 12,130 shared regions and centrelines are identical. 0210 is in
  the Drive archive.
- **Side channels are not modelled yet.** Luke's hybrid covers 4,255 of them, but
  the quality gate in `region_split.py` (`estimate_reliability_height_model == OK`)
  drops them all: the height model has `MISSING_DATA` there. The same gate also
  drops ~1,900 main-channel regions where the hybrid prefers SAM.
- `luke_postprocessed/` is used for the quality gate, VVR polygons, erosion volumes
  and the output GeoPackage's base layers. Luke's newer 0330 version is in Drive
  (`inputs/luke/`); switching is an open question.
