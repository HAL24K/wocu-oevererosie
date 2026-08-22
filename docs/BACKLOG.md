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


---

## 2026-08-17 abstraction audit

Done in the `ExperimentConfig` change set:

- **A1. Config cell → `ExperimentConfig`** (`src/pipeline/config.py`). Paths and
  parameters are an object; defaults reproduce the 20260617a reference.
- **A2. Notebook-only orchestration → `src/pipeline/run.py`.** start_points,
  combined feature table, geometry projection and acceptance checks now live in
  code. `uv run python -m src.pipeline --experiment <name>`.
- **A3. Run report** (`src/pipeline/report.py`): per-run `report.html` with
  figures + tables next to the model outputs — the browsable notebook output,
  without the notebook.
- **A4. `bank_distances` delegated to `HeightModelPointSource`** — one top-N
  implementation instead of two.
- **A5. `location_id` normalisation consolidated** to
  `src.sources.geometry.normalise_location_id` (was implemented three times).

Still open, deliberately:

- **A6. `cluster` vs `river`** — two derivations of near-identical information
  (prefix list in `region_split`, regex in `feature_engineering`). Unifying
  changes an encoded feature, so it belongs with a modelling change.
- **A7. t1/t2/t3 hardcoding** (`region_split._build_split_row` keeps `.tail(3)`).
  Generalising to an observation series is the hybrid-era change.
- **A8. `train.py` evaluation** — test-set early stopping and the random spatial
  split. Modelling decisions, tracked in README → Known issues.
- **A9. `curvature.py` still unwired** — bend_exposure read from the March
  parquet instead of computed.
- **A10. `src/legacy/` deletion** once WFS layers are confirmed frozen.


---

## A7 spec — multi-t ingestion (drafted 2026-08-19, motivated by the span table)

Evidence: on the 20260819-hybrid run, v_test std by test span — 1 yr: 11.41,
2 yr: 7.14, 3 yr: 2.90 (March regime: 2.19). The .tail(3) triple turns annual
observations into 1-year velocity noise; the data is fine, the definition isn't.

Replace the t1/t2/t3 triple in region_split with the full per-region series
{(date_i, dist_i)} that BankObservations already carries:

- **v_hat**    robust slope (Theil–Sen) over all observations up to the holdout
               window — signed velocity (deposition negative), never a 1-year diff
- **target**   slope (or displacement) over a FIXED multi-year holdout window,
               so train and test regions share one noise regime
- **accel**    slope(recent half) − slope(older half): accelerating / decelerating,
               the feature asset managers actually asked for (dynamism index,
               FEATURE_GAP_ANALYSIS "derivable" row)
- **resid_std** residual std around the fit — a free measurement-quality feature
               that will absorb much of the segmentation noise
- **n_obs, span** kept as features, no longer as noise amplifiers

Sub-year observations feed the fit directly (real dates, fractional years) —
no more within_year collapse. Touches region_split + feature_engineering +
FeatureShifter (rolling update of v_hat); predictor and export unchanged.
Grouped-by-region split becomes mandatory the moment sub-region units exist.

---

## 2026-08-19 demo outcomes (internal, height + SAM engineers present)

Received well; the PM will book the asset-manager presentation for **early
October** — that date anchors the roadmap below.

Agreed / assigned:

- **Luke (height model)** investigates regions that HAD a height-model
  measurement but no longer do — this is exactly the `no_pref_was_OK` class
  in `data/02_processed/scope_coverage.gpkg` (326 regions, 34 with NVO).
  Hand him that layer.
- **Outliers + resolution merge into one work item.** Luke's point: raising
  the resolution redefines what an outlier is (a far-bank line is only an
  outlier *relative to its segment*). Design the line/segment-level filter
  and the (region × segment) observation unit together, not sequentially.
- **Outlier filtering: automatic, with manual eye validation.** Auto-flag
  (bimodality + temporal consistency, `farbank_candidates()`), then a
  judgement-by-eye pass over the flagged set — not a review of all 10k
  SAM regions. The inspector notebook is the validation bench.
- **GIS engineers** check whether a GeoPackage of bridge locations exists —
  bridges explain a family of spurious bank lines; filtering those scope
  regions (or masking the bridge footprint) removes them at the source.
- **Resolution ceiling ≈ R=100.** Height model is 0.5–1 m over ~100 m scope
  regions, so segments below ~1 m are below sensor resolution. Practical
  sweet spot much lower (R=10–20); note HybridLineSource/inspector sampling
  (20 pts/line) must densify as R grows or anchors starve.

Commitment made in the room (track it): asked whether the hybrid-run test
error (LGB 4.12 m/jr) can at least be **halved**, answer given was confident
**10×**. Decomposition: halving is near-mechanical — a multi-year / robust-
slope target definition alone collapses target noise (std 11.4 → ~2.9 at
3-yr spans); 10× (≈0.4 m/jr) additionally needs the A7 target + line-level
outlier filtering + resolution to deliver real signal, and honest evaluation
(grouped split, no test-set early stopping) may *raise* measured error even
as the model improves. Frame October numbers accordingly.

## 2026-08-22 loop experiment closed → graduation shipped

Three-track loop experiment (branch `experiment/loop-engineering`,
reports in `experiments/loop/TRACK{1,2,3}_REPORT.md`, every variant in
`experiments/loop/ledger.csv`) closed and graduated into the pipeline
(`--source hybrid`). Winners: e8 cleaning stack (repair-over-removal,
detrended Theil–Sen temporal rule with eligibility guard), traj2
trajectory features, R=5 segment × ≥2-yr horizon model (span-weighted).
Honest validation everywhere: LGB early-stops on a train-carved split,
the test set never touches fitting.

**Graduated run 20260822-grad (honest, fresh grouped split):**
- Region model: LGB test MAE 2.63 m/jr · tail(>2) 5.90 (n=221) — vs the
  20260820 baseline 4.01 / ~8.3 under *easier* (test-ES) conditions.
- Segment-horizon artifact (R=5, H≥2): MAE 1.15 · tail 3.04 · R² 0.49 ·
  median position error **1.08 m** at the ≥2-yr horizon;
  `segment_predictions.parquet` = 48,336 segments / 10,261 regions —
  the input for the any-segment-crosses-VVR alert.
- Full product artifacts: predictions gpkg (182,525 points, 1,129/1,410
  VVR crossing years filled), report.html, segment metrics json.

**October framing (three rulers, tell them apart):**
1-yr region ruler (comparable to March/Aug): 9.13 → 5.90 honest.
≥2-yr horizon region ruler: tail_frozen 2.50 (experiment, k5).
Segment-horizon (the operational quantity): tail 3.04, position error
~1.1 m median — improves automatically as SAM years accumulate.

**Remaining roadmap:**
- Wire `segment_predictions.parquet` → any-segment-crosses-VVR alert in
  the signalering step (Alexander's stated alert semantics).
- Bridges gpkg + IJssel/Maas kribben (GIS engineers) → drop into the
  structures mask slot (kribben cover only Waal + Nederrijn-Lek today).
- Gully-mask refinement: mask only samples inside a new_in_hybrid
  polygon AND far from the main channel (blunt polygon costs coverage;
  contamination measured at 5.9% of normal-region samples).
- H=3 horizon ruler matures with each new SAM year (starved today).
- QGIS labelling queues: protected-but-suspect surveys (temporal rule),
  segment prediction-spread ranking (t3 showcase).
