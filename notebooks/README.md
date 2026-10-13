# Notebooks

The pipeline runs from the command line (`python -m src.pipeline`), not from a
notebook. What is left here:

| Notebook | What it is |
|---|---|
| `02_region_inspector.ipynb` | One figure per scope region: delivered bank lines, distance to centreline, preferred model. For looking at what the hybrid does in a region |
| `03_resolution.ipynb` | Where one scalar per region misleads: regions whose bank halves disagree persistently |

Both use `src.analysis.region_inspector.RegionInspector`, which defaults to the
current reference run.

## provenance/

Records of how a model input or a modelling convention came to be. Kept for
that record, not to be re-run: they import modules that no longer exist (see the
`pre-cleanup-20261012` tag for the code as it was).

| Notebook | Made or decided |
|---|---|
| `region_features_v2.ipynb` | `data/02_context/region_features_v2.parquet` — the `bend_exposure` features |
| `discharge_cleaning.ipynb` | `data/02_context/discharge/` — the hydrology features' input |
| `water_stations_download.ipynb` | the discharge downloads and station set behind `water_stations_for_modeling.gpkg` |
| `point_selection.ipynb` | why a region's distance is the mean of the **3 furthest** points: a deliberate worst-case convention that shapes every velocity |

Everything else that lived here (58 notebooks at the peak) is in git history.
