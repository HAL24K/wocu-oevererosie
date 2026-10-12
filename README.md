# WOCU River Bank Erosion Prediction

Predicts where Dutch river banks will be in future years, per ~100 m scope region,
along the Rijntakken, IJssel and Maas. Phase 2 of a POC for Rijkswaterstaat.
Output is a GeoPackage opened in QGIS.

**Stack**: Python 3.12 · geopandas · LightGBM · uv.

## Two stages

```
SAM lines (Lars) ─┐
                  ├─ hybrid stage ──▶ hybrid line set ──▶ model stage ──▶ predicted bank positions
height model (Luke)┘   per region: SAM    data/03_hybrid/     clean, features,     data/04_runs/<experiment>/
                       or height lines                       train, predict
```

The model stage can start from any hybrid line set: Luke's delivery, our
reproduction of it, or one we build ourselves with other cleaning rules.

**Model stage** — one command, writes a browsable `report.html` and a run record:

```bash
uv run python -m src.pipeline --experiment 20261012-test --notes "what this run tests"
uv run python -m src.pipeline --experiment 20261012-test --hybrid-gpkg data/03_hybrid/<name>/hybrid.gpkg
```

With no overrides a run reproduces the reference run. Options: `--no-export`
(skip the large GeoPackage), `--resume` (reuse per-step parquets), `--end-year`,
`--raw-gpkg`, `--fresh-split`. All defaults: `src/pipeline/config.py`.

**Hybrid stage** — for now a script (`scripts/hybrid/reproduce_hybrid.py`, Luke's
algorithm on his raw inputs; `prehybrid_clean.py` applies our cleaning rules
first). Moving it into `src/hybrid` as a proper stage is the next step.

## Setup

```bash
uv sync                      # runtime + dev tools (pytest, ruff, jupyter, python-pptx)
uv run pytest                # tests
uv run ruff check src tests && uv run ruff format src tests
```

For notebooks: `uv run python -m ipykernel install --user --name wocu --display-name "WOCU erosion (3.12)"`.

## Where things are

```
src/
  hybrid/      hybrid stage: luke_v2 (port of Luke's hybrid model), prehybrid_clean
  pipeline/    model stage, one module per step (see its __init__.py), config, provenance
  cleaning/    cleaning rules, shared by both stages
  sources/     scope geometry and the bank-observation table
  analysis/    region_inspector, for notebooks and evaluation scripts
scripts/
  hybrid/      build, export and compare hybrid line sets
  evaluate/    worst-predicted regions, galleries, comparisons between runs
  export/      deliverables from a run: predicted bank lines, VVR crossing years
  data_prep/   how 02_context files were made (structures, water stations)
  slides/      figures for the 14/15 Oct sessions (remove afterwards)
qgis/          QGIS project, layer paths relative to ../data
docs/          runs/ (one record per run), PIPELINE_CHANGES.md, loop experiment, analyses
notebooks/     two inspection notebooks + provenance/
data/          git-ignored; layout in data/README.md
```

Scripts run from the repo root: `PYTHONPATH=. uv run python scripts/<group>/<script>.py`.
Their output goes to the run's `exports/` folder; copy to `data/05_deliverables/`
what goes out.

## Provenance

Every run writes `run.json` (inputs with size and date, every setting, git
commit, metrics) into its run folder and a copy into `docs/runs/`, which is
tracked. Add `--notes` to say what a run is for. The ledger is
[docs/runs/README.md](docs/runs/README.md). The state before the October 2026
clean-up is tagged `pre-cleanup-20261012`.

## Conventions

- **CRS** is EPSG:28992 (RD New) everywhere.
- **Region key** is `location_id`, e.g. `ijssel1_l_0010_0020` (river section, bank
  side, chainage). Deliveries call it `position_id` or `scope_region_id`;
  `src.sources.geometry.normalise_location_id` reconciles them.
- **Time** is integer years in the modelling path; the hybrid lines carry real
  dates, collapsed per year in step 1.

## Known issues

- **The quality gate is a height-model flag.** `region_split.py` keeps regions whose
  `estimate_reliability_height_model` is `OK`, so all 4,255 side-channel regions in
  the hybrid and ~1,900 main-channel regions where the hybrid prefers SAM are never
  modelled.
- **Side channels have no river category.** Their `river` value is `ngeul`, which
  the encoding does not know (→ −1), so the parent river is lost.
- **`bend_exposure` is read from a March 2026 parquet**, not computed; regions
  outside it get 0.
- **The train/test split is random across adjacent regions**, so spatial features
  leak between neighbours.
