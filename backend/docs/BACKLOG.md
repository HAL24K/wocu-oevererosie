yes # Backlog & TODO

Items collected during development. Roughly ordered by priority within each section.

---

## 🔴 High priority — affects output correctness

### P1. Include `inference_only` (2-timestamp) locations in ML predictions
- 940 locations with 2 timestamps were excluded from `region_features` solely because
  they lack a t3 observation and therefore cannot be split into train/test.
- They have `v_last` (= `v_train`) and `last_dist` (= `dist_t2`) — sufficient for LGB.
- Missing: `erosion_vol_rate_t1` → impute with 0 (the mode; 73% of OK regions are 0).
- All static features (soil, vegetation, hydrology, geometry) are derivable from location.
- **Required changes:**
  1. Preprocessing (`20260311_preprocess_region_split_v2.ipynb`): produce a
     `region_inference_features.parquet` with the same column schema as
     `region_features.parquet`, constructed from `region_inference_only.parquet`.
  2. `02_iterative_prediction.ipynb`: load `region_inference_features.parquet`,
     run it through `predict_iterative_ml` alongside `region_features`.
  3. Union the two prediction DataFrames before building `predicted_bank_positions`.
  4. `is_nvo` for inference-only locations must be derived via spatial join against
     `vvr_rates_of_change` at export time (they have no `is_nvo` in their parquet).
- **142 locations with only 1 timestamp:** treat as MISSING_DATA, no prediction possible.

### P2. Fix `is_nvo` silently dropped on `predicted_bank_positions`
- Python `bool` is stored as a GPKG geometry-type blob → pyogrio drops it silently.
- Fix: cast `bank_export['is_nvo'] = bank_export['is_nvo'].astype(int)` before
  `to_file()` in `src/erosion/export.py`.
- Verify in QGIS: field should appear as integer (0/1) on the point layer.

### P3. Add `signaleringslijn` as a layer in the output GeoPackage
- The signaleringslijn defines the VVR threshold that triggers a crossing alert.
- It is our analytical contribution and should be self-documenting in the GPKG.
- Fix: add one `to_file` call in `export_predictions()` for the signaleringslijn GDF.

---

## 🟡 Medium priority — quality / usability improvements

### P4. Validate single-timestamp locations
- 142 `inference_only` locations have only 1 DTM timestamp.
- Unexpected — most scope regions should have ≥2 surveys.
- Action: check in QGIS which rivers/areas these are concentrated in.
  Possible causes: new scope additions, data delivery gaps, very recent survey only.

### P5. `vvr_rates_of_change` polygon count dropped from 8,148 → 1,410
- OLD gpkg (20260303): 8,148 VVR polygons.
- NEW gpkg (20260310, Luke's latest): 1,410 VVR polygons.
- Only 201/1,410 get a `predicted_vvr_crossing_year`.
- Confirm with Luke whether the 1,410 is intentional (new VVR layer definition)
  or a data delivery issue.

### P6. VVR crossing year: representative point join may miss edge cases
- Current join uses `representative_point().within(scope)` to match VVR → scope.
- VVR polygons that span scope boundaries will get the crossing year of whichever
  scope region their representative point falls in (arbitrary).
- Consider `intersects` + `min(crossing_year)` as a more robust fallback.

### P7. `summary_scope` geometry type shows as `Unknown` in NEW gpkg
- OLD: `MultiPolygon`. NEW: `Unknown`.
- May cause issues in some GIS tools. Investigate in Luke's post-processing pipeline.

### P8. QGIS project file with pre-defined symbology
- Once layer structure is stable, create a `.qgz` project file with:
  - OK regions: colour by NVO status + crossing year warning level
  - NOK regions: grey
  - MISSING_DATA regions: transparent
  - `predicted_bank_positions`: temporal controller on `year` field
  - `signaleringslijn`: purple line
  - `all_lines` (Luke's river bank lines): replace raw points in visualisation

---

## 🟢 Low priority — architecture / housekeeping

### P9. Folder structure rearchitecture
- `region_features.parquet` lives in `02_processed/` → should be `03_features/`
- Output GPKG lives in `02_processed/erosion/` → should be in `04_model_outputs/`
- Address when doing a broader pipeline refactor.

### P10. `DataHandler` consolidation (Module 1)
- Feature preparation logic is spread across multiple preprocessing notebooks.
- Goal: consolidate into `DataHandler` so the pipeline starts from raw GPKG
  with a single `handler.prepare_features()` call.
- Low priority until the feature set stabilises.

### P11. Move `compute_dist_per_year` into `src/erosion/features.py`
- Currently inline in `20260313_erosion_prediction_pipeline.ipynb`.
- Should live in a reusable module.

### P12. `plot_utils.py`: switch from raw bank points to Luke's `all_lines`
- Current visualisation renders individual bank measurement points.
- After the explanation panel (showing how dist is selected), switch to Luke's
  LineString layer (`all_lines`) for consistency with the final product.
- Keep 3 selected points visible only in the "methodology" panel.

---

### P13. Retrain models on new feature set (20260314 experiment)
- The `20260314_pipeline_building` experiment produces updated features:
  - `erosion_vol_rate_t1` now correctly computed from `erosion_vlakken_filtered`
  - `flood_days` now consistently uses station-specific P90 (cleaned timeseries) for both windows
  - HW metrics (n_events, max_rise_rate, drawdown_index) computed from cleaned discharge
- Action: run `05_train_models.ipynb` (to be created) and save new model bundles to
  `04_model_outputs/20260314/`.
- After training, compare RMSE / R² against the old models (stored in `data/20260312/`).

### P14. Recreate performance figures and visualisations
- Once new models are trained, re-run the figures from `20260313_erosion_prediction_pipeline.ipynb`:
  - Feature correlation / importance plots
  - Predicted vs. actual bank position plots for test regions
  - 100-region scope visualisation (signaleringslijn, VVR, predicted banks by year)
- Compare side-by-side with the old model outputs to assess the impact of the improved features.

---

## ✅ Done (reference)

- Iterative prediction with `FeatureShifter` (v_train, dist_t2, spans, hydrology)
- `predictor.py`, `feature_shifter.py`, `model_loader.py`
- `export.py` (fresh-temp + SQLite-ATTACH + trigger drop/recreate pattern)
- `plot_utils.py` extracted from pipeline notebook
- `compute_vvr_crossing_year` — integer years, dynamic year range
- Export to GeoPackage working end-to-end (verified 2026–2050)
- 100-region visual validation in `02_iterative_prediction.ipynb`
