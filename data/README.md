# data

Git-ignored. Several GB of GeoPackages; the git remote does **not** back this up.

## Layout

```
01_raw/          as delivered, never modified
02_processed/    post-processed deliveries and derived context layers
03_features/     model-ready feature tables, one folder per experiment
04_model_outputs/ trained bundles and prediction GeoPackages, one folder per experiment
archive/         superseded and phase-1 material — see below
water_stations_timeseries/cleaned/  the discharge parquets the pipeline reads
```

Experiment folders under `03_features/` and `04_model_outputs/` share a name.
`20260617a` is the current reference run; `20260314` produced byte-identical features
and is kept for comparison.

## What the pipeline actually reads

Every path below appears in the config cell of
`notebooks/04_model/20260617a/00_master.ipynb`.

| Path | Role |
|---|---|
| `01_raw/erosion/wocu_output_fase2_20260210.gpkg` | Primary input — 4.8 M bank points |
| `02_processed/erosion/wocu_post_processed_fase2_20260310.gpkg` | Quality gate, VVR polygons, erosion volumes |
| `01_raw/scope/scope_fase2.gpkg` | Scope polygons and centrelines |
| `01_raw/scope/20260205_signaleringslijn.gpkg` | VVR threshold lines |
| `01_raw/soil/BRO_DownloadBodemkaart.gpkg` | Soil group feature |
| `02_processed/wfs_context/vegetatielegger.gpkg` | Vegetation feature |
| `02_processed/wfs_context/land_use.gpkg` | Land-use feature |
| `02_processed/water_stations/water_stations_for_modeling.gpkg` | Nearest discharge station |
| `water_stations_timeseries/cleaned/discharge/*.parquet` | High-water metrics, 14 stations |
| `02_processed/erosion/region_features_v2.parquet` | `bend_exposure` — a March artefact the pipeline still depends on |

## Phase 2 hybrid delivery

| Path | Role |
|---|---|
| `02_processed/hybrid/hybrid_model_results_20260710.gpkg` | Current hybrid output — `lines` + `model_preference` |
| `02_processed/hybrid/hybrid_model_results_nearest_20260710.gpkg` | A "nearest" variant, 165k lines vs 91k. Selection rule undocumented |
| `02_processed/hybrid/notitie_hybride_model.docx` | Delivery notes — not yet read into any decision |
| `01_raw/segmentation/oeverlijn_segmentatie_full_aoi.geojson` | Raw SAM segmentation, 73,810 lines, **no `location_id`** — upstream of the hybrid, not directly consumable |
| `01_raw/segmentation/batches/` | The 28 batches the full-AOI file is merged from. Redundant with it |

Not yet processed: nothing hybrid-derived exists in `03_features/` or `04_model_outputs/`.

## QGIS exploration

`02_processed/exploration_targets.gpkg` — 9 named regions worth looking at
(far-bank suspects, clean height-model references, the densest hybrid region,
the strongest March erosion), each with a note saying what to look for. Load it
first, open its attribute table, and right-click → Zoom to Feature.

## 20260330 is newer than what we use

`02_processed/erosion/wocu_post_processed_fase2_20260330.gpkg` postdates the 20260310
file the pipeline reads. It differs — `summary_scope` has 27,067 rows against 16,772
(duplicated per `type_oever`, which `region_split` already handles) and 1,383 VVR
polygons against 1,410. No run has used it. Compared in
`notebooks/04_model/compare_postprocessed_gpkg_20260310_vs_20260330.ipynb`.

## archive/

Moved out of the working tree, not deleted. 3.4 GB.

`superseded/` — replaced by something newer:

| File | Why |
|---|---|
| `20260315_model_results` | Extensionless byte-twin of the `.gpkg` beside it (md5-identical) |
| `wocu_output_fase2_v4.gpkg` | Superseded by `wocu_output_fase2_20260210.gpkg` |
| `wocu_post_processed_fase2_20260212/20260223.gpkg` | Superseded by 20260310 |
| `wocu_lgb_predictions_20260313/20260314.gpkg`, `wocu_erosion_predictions_20260313.gpkg` | Old outputs that sat in `02_processed/` instead of `04_model_outputs/` |
| `wocu_output_fase2_v4_w_*.gpkg`, `all_wfs_data_batch_*.gpkg`, `phase1_*_complete_*.gpkg` | Phase-1 WFS-enriched products, matching `src/legacy/` |
| `hybrid_model_results_20260708.gpkg` | Superseded by the 20260710 delivery |
| `vegetatiemonitor.gpkg` | 371 MB, never referenced in code. Blocked on RWS access |
| `all_water_stations.gpkg` | 367 MB, superseded by the 156 KB `water_stations_for_modeling.gpkg` |
| `station_attribution.gpkg` | Loose output of an old station-matching run |

`superseded/experiments/` — the 20260217, 20260303 and 20260312 model-output
folders in full, plus the prediction GeoPackages from 20260314. That experiment's
`bundle.joblib` and four `model_*.joblib` stay in place: they are small, and their
metrics are the honest baseline the current run is compared against.

`water/` — the old `archive_water/` tree: station caches, historical JSON, and
reliability GeoPackages from the water-station selection work.

Also here from phase 1: `sam/` (the 2025 SAM exports, superseded by the hybrid),
`luke_for_feedback/`, and `feedback.zip`.

## Known duplication outside the repo

`Hybride model/` and `Resultaten/` also exist one level up, in the project folder, with
identical checksums — roughly 480 MB duplicated. The copies here are canonical. The
originals were left in place in case the delivery folder is shared.
