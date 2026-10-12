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

**Hybrid stage** — raw SAM + height model → `data/03_hybrid/<name>/`:

```bash
# Luke's _nearest hybrid, rebuilt from the raw files (step 1 is the slow part)
uv run python -m src.hybrid --name 20261012-luke-nearest
# our SAM cleaning before the points count, reusing step 1 of the hybrid above
uv run python -m src.hybrid --name 20261013-rules --nearest-from 20261012-luke-nearest \
    --sam-rules --side-channel-m 10 --jetty-m 100 --notes "what this hybrid tests"
```

Steps: (1) Luke's nearest-crossing cut of the SAM lines → `nearest.gpkg`;
(2) optional SAM cleaning: `--sam-rules` (the model stage's rule stack),
`--side-channel-m`, `--jetty-m`; (3) Luke's points count, SAM or height per
region. Regions that lose all SAM pieces in (2) take their height lines
(`--no-height-fallback` drops them). Writes `hybrid.gpkg` and `hybrid.json`.

**Model stage** — a hybrid line set → predictions, a browsable `report.html` and a run record:

```bash
uv run python -m src.pipeline --experiment 20261013-rules --hybrid 20261013-rules
uv run python -m src.pipeline --experiment 20261013-ref --notes "what this run tests"
```

With no `--hybrid` it reads Luke's 10 Jul `_nearest` delivery and reproduces
the reference run (`20261013-gate-sam`, about 4 minutes). Options: `--hybrid-gpkg` (any hybrid file), `--no-export`
(skip the large GeoPackage), `--resume` (reuse per-step parquets), `--end-year`,
`--raw-gpkg`, `--fresh-split`, `--quality-gate all` (the height-model gate on
every region, as before 13 Oct), `--side-channels`. All defaults: `src/pipeline/config.py`.

**Both in one go**: add `--run-model <experiment>` to `python -m src.hybrid`.

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
  hybrid/      hybrid stage: config, run, luke_v2 (port of Luke's model), sam_cleaning
  pipeline/    model stage, one module per step (see its __init__.py), config, provenance
  cleaning/    cleaning rules, shared by both stages
  sources/     scope geometry and the bank-observation table
  analysis/    region_inspector, for notebooks and evaluation scripts
scripts/
  hybrid/      compare a hybrid with Luke's file; export one in his delivery format
  evaluate/    compare_runs (fair run-vs-run), worst-predicted regions, galleries
  export/      deliverables from a run: predicted bank lines, VVR crossing years
  data_prep/   how 02_context files were made (structures, water stations)
  slides/      figures for the 14/15 Oct sessions (remove afterwards)
qgis/          QGIS project, layer paths relative to ../data
docs/          runs/ (one record per run), PIPELINE_CHANGES.md, loop experiment, analyses
notebooks/     two inspection notebooks + provenance/
data/          git-ignored; layout in data/README.md
```

Scripts run from the repo root: `PYTHONPATH=. uv run python scripts/<group>/<script>.py`.

**Every experiment owns its files.** A model run writes only into
`data/04_runs/<experiment>/`, a hybrid only into `data/03_hybrid/<name>/`, and so
do the scripts that work on them (`--run`, `--hybrid`):

```
04_runs/<experiment>/
  run.json     the record (also a line in docs/runs/ledger.jsonl)
  features/    per-step parquets            model/   bundle, predictions, report.html
  exports/     GeoPackages, CSVs, PDFs      figures/ figures and slides
  compare/     comparisons this run makes against another (the challenger owns them)
```

Nothing an experiment produces goes into the repository except its ledger line.
What goes out to people is copied to `data/05_deliverables/`.
`scripts/backup_to_drive.py` copies the light part of every run and hybrid
(records, exports, figures, comparisons, reports) to Drive, `WOCU Oevererosie/runs/`.

## Provenance

Every model run writes `run.json` (inputs with size and date, every setting,
git commit, metrics) and every hybrid `hybrid.json` into its folder, and both add
one line to `docs/runs/ledger.jsonl`, the only experiment file in git. A run on
a hybrid from `src.hybrid` carries that hybrid's record, so every number traces
back to the raw files. Add `--notes` to say what a run or hybrid is for;
`uv run python -m src.provenance` prints the runs as a table. The ledger is
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

- **Side channels are not in the default run.** `--side-channels` admits them
  (run 20261013-gate-sam-side: +20 % skill there, against +34 % elsewhere); check
  their predictions in QGIS before making it the default.
- **1,103 main-channel regions are not in the hybrid**: Luke's model only takes
  regions with at least one SAM line, so height-only regions never enter.
- **Inference-only regions always get `is_nvo = False`** (`region_split.py` computes
  the VVR join for train/test regions only), so their predictions miss that
  feature. The VVR crossing years in the output use their own join and are fine.
- **`bend_exposure` is read from a March 2026 parquet**, not computed; regions
  outside it get 0.
- **The train/test split is random across adjacent regions**, so spatial features
  leak between neighbours.
