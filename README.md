# WOCU River Bank Erosion Prediction

Predicts where Dutch river banks will be in future years, per ~100 m scope region,
across the Rijntakken, IJssel, Maas and their side channels. Phase 2 of a POC for
Rijkswaterstaat.

**Stack**: Python 3.12 · geopandas · LightGBM · uv. Output is a GeoPackage opened in QGIS.

---

## Start here

One command runs the whole pipeline and writes a browsable report:

```bash
uv run python -m src.pipeline --experiment 20260817a
```

Outputs land in `data/03_features/<experiment>/` and `data/04_model_outputs/<experiment>/`,
including `report.html` — figures and tables for every step, viewable in a browser.
Inputs and parameters live in `src/pipeline/config.py` (`ExperimentConfig`); defaults
reproduce the 20260617a reference run. Useful flags: `--no-export` (skip the 185 MB
GeoPackage), `--resume` (reuse per-step parquets), `--end-year`.

The master notebook (`notebooks/04_model/20260617a/00_master.ipynb`) documents the same
flow interactively, and `notebooks/01_scenarios.ipynb` replays individual steps with
different parameters against cached inputs. Everything else in `notebooks/` is history — see
[`notebooks/README.md`](notebooks/README.md).

---

## Setup

The virtualenv must **not** live inside this repository — the folder is synced by
Google Drive, which corrupts venvs (it deletes module files while leaving their
metadata behind) and makes every import crawl.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/wocu-oevererosie"   # add to ~/.zshrc
uv sync
```

Then, for notebooks:

```bash
uv run python -m ipykernel install --user --name wocu --display-name "WOCU erosion (3.12)"
```

and pick **WOCU erosion (3.12)** as the kernel. `src` is installed editable, so
imports work without `PYTHONPATH` fiddling.

| Task | Command |
|---|---|
| Tests | `./run_tests.sh` — or `uv run pytest tests/ -m "not integration"` |
| Lint | `uv run ruff check src tests` |
| Format | `uv run ruff format src tests` |

Integration tests call live WFS services and are excluded from CI. Three of them
currently fail on upstream schema drift; run them with `./run_tests.sh -m integration`.

---

## The pipeline

```
delivery (points or lines)
   │
   ├── src/sources/            read any delivery into bank observations
   │                           (location_id, date, dist_m, source)
   ▼
src/pipeline/
   config.py                   ExperimentConfig — all paths and parameters
   run.py                      orchestrator + CLI; writes report.html per run
   bank_distances.py           01 · reduce a point cloud to one distance per region-year
   region_split.py             02 · pivot to t1/t2/t3, quality filter, train/test split
   feature_engineering.py      03 · vegetation, land use, soil, hydrology, bend exposure
   train.py                    05 · six models, LightGBM primary; saves a bundle
   curvature.py                     bend exposure — not yet wired into 03
   │
   ▼
src/model/
   predictor.py                iterative multi-year prediction
   feature_shifter.py          rolling feature state between prediction steps
   export_utils.py             model bundle save/load
   │
   ▼
src/erosion/
   centerline_utils.py         geometry, VVR crossing years
   export.py                   assemble the output GeoPackage
   plot_utils.py               scope-region figures
```

`scripts/` holds two live utilities — `viz_point_selection.py` (the furthest-points
figure) and `find_nearest_waterweb_stations.py`. Six one-off API and GeoPackage probes
are in `scripts/archive/`.

Shared: `src/constants.py` (category dictionaries, column names), `src/paths.py`.

`src/legacy/` is phase-1 code — the `DataCollector → DataHandler → BaselineErosionModel`
stack. **Nothing in the live pipeline imports it.** It is kept because it holds the only
working WFS clients; see its `__init__.py`.

There is no API. A FastAPI skeleton (`app/`, serving only `/` and `/health`, importing
nothing from `src`) was removed in the flattening commit; recover it from git history if
an API is ever wanted here.

---

## Conventions

- **CRS** is EPSG:28992 (RD New) throughout, in every layer and every output.
- **Region key** is `location_id`, e.g. `ijssel1_l_0010_0020` — cluster, bank side,
  chainage range. Deliveries call it `position_id` or `scope_region_id`;
  `src.sources.geometry.normalise_location_id` is the single place that reconciles them.
- **Data layout** under `data/`: `01_raw` → `02_processed` → `03_features` →
  `04_model_outputs`, with feature and output folders named after the experiment.
- **Time** is integer years in the modelling path. The hybrid delivery carries real
  dates; `BankObservations.to_dist_per_year()` collapses them.
- `data/` is git-ignored — several GB of GeoPackages.

---

## Documentation

| Document | What it is |
|---|---|
| [docs/BACKLOG.md](docs/BACKLOG.md) | **Highest signal.** Numbered, diagnosed open items |
| [docs/DATA_INVENTORY.md](docs/DATA_INVENTORY.md) | Data sources, provenance, status |
| [docs/FEATURE_GAP_ANALYSIS.md](docs/FEATURE_GAP_ANALYSIS.md) | Features we have vs. want, and what blocks the rest |
| [docs/CENTERLINE_IMPLEMENTATION_PLAN.md](docs/CENTERLINE_IMPLEMENTATION_PLAN.md) | Centreline approach design notes |
| [docs/legacy/](docs/legacy/) | Phase-1 architecture and its TODO list — describes `src/legacy/`, not the current pipeline |

---

## Known issues

Read these before trusting a number:

- **The naive mean beats LightGBM** on test MAE (0.784 vs 0.937). A constant currently
  predicts bank retreat as well as the model does.
- **`train.py` early-stops on the test set**, which biases its reported metrics. The
  `0.801` figure in the `20260617a` bundle is optimistic; `0.937` is the honest one.
- **The train/test split is random across adjacent regions**, so spatially derived
  features leak between them.
- **Four hydrology features describe the same window as the target**, and cannot exist
  at prediction time — `FeatureShifter` holds them constant instead.
- **`bend_exposure` is read from a March 2026 parquet** rather than computed, so new
  regions get NaN.
