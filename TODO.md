# WOCU WFS Data Fetching - Task List

## Context
We're fetching WFS data for 233 scope regions (eventually 12,130). The workflow is:
1. Fetch WFS data for all regions (DataCollector per region)
2. Aggregate and deduplicate across regions
3. Save to GeoPackage for QGIS validation
4. Load saved GeoPackage in DataHandler for feature engineering

## Current Status
✅ DataCollector works (fetches for single region with retry logic)
✅ Notebook code successfully fetches, aggregates, deduplicates, saves
✅ WFSDataBundler class created - abstracts multi-region data collection
✅ WFSDataBundler tested with 10 regions - layer naming convention updated to `{service}/{layer}`
✅ DataHandler extended to load from GeoPackage (verified identical to live WFS)
⬜ Need to test bundler at scale (100 → 500 → 12,130 regions)
⬜ Need to add checkpointing before full 12,130-region run

---

## TASK 1: Refactor DataHandler - Add Geopackage Loader ✅ COMPLETED

**Goal:** Let DataHandler load pre-saved WFS data instead of fetching live

**File:** `src/data/data_handler.py`

### Subtasks:

**1.1 Add method: `load_remote_data_from_geopackage()`** ✅
- [x] Read all layers from geopackage using `gpd.list_layers()` (switched from fiona)
- [x] Filter for layers containing `/` (new naming: `{service}/{layer}`)
- [x] Parse layer names: `land_use/BrpGewas` → extract service + layer
- [x] Load each layer as GeoDataFrame
- [x] Store in nested dict: `bundled_wfs_data[service_name][layer_name] = gdf`
- [x] Call `_process_bundled_data_into_features(bundled_wfs_data)`
- [x] Set `self.remote_data_downloaded = True`

**1.2 Add method: `_process_bundled_data_into_features()`** ✅
- [x] Loop through `self.prediction_regions`
- [x] For each region:
  - [x] Buffer the region geometry (use `self.config.prediction_region_buffer`)
  - [x] For each configured layer (not just what's in geopackage):
    - [x] Find layer in bundled data or create empty GeoDataFrame
    - [x] Spatially filter: `layer_gdf[layer_gdf.intersects(buffered_region)]`
    - [x] Get feature config: `self.config.feature_creation_config.get(layer_name)`
    - [x] Call `self._generate_region_features(region_geom, filtered_data, feature_config)`
    - [x] Name features: `{layer_name}_{agg_function}`
  - [x] Flatten nested dict: `UTILS.flatten_dictionary(single_region_features)`
  - [x] Append to `wfs_features` list
- [x] Convert to DataFrame
- [x] Merge with `self.scope_region_features`

**1.3 Test the new methods** ✅
- [x] Created `test_datahandler_geopackage_loading.ipynb`
- [x] Side-by-side comparison: live WFS vs geopackage loading
- [x] Verified identical output (10 regions, 11 columns)
- [x] Confirmed ~200x speedup (16s → 0.08s for 10 regions)
- [x] Handles missing layers (creates default values)

---

## TASK 2: Create Utility Module for WFS Bundling ✅ COMPLETED

**Goal:** Extract notebook boilerplate into reusable module

**File:** `src/data/wfs_bundler.py` (new file)

### Subtasks:

**2.1 Create `WFSDataBundler` class** ✅
- [x] Constructor: `__init__(scope_regions, config)`
- [x] Method: `fetch_all_regions(test_mode, num_test)` - your notebook logic
- [x] Method: `save_to_geopackage(output_path)` - your save logic
- [x] Handle duplicate removal in `_save_single_layer()`
- [x] Add `get_summary_stats()` for diagnostics
- [x] Add `list_geopackage_layers()` for verification

**2.2 Create test notebook** ✅
- [x] Created `notebooks/test_wfs_bundler.ipynb`
- [x] Tests with 10 regions in test mode
- [x] Includes instructions for scaling up

**Usage:**
```python
bundler = WFSDataBundler(scope_regions, config)
bundler.fetch_all_regions(test_mode=True, num_test=10)
bundler.save_to_geopackage(output_path)
```

---

## TASK 3: Optimize for 12,130 Regions (Before Overnight Run)

**Goal:** Make sure overnight run won't crash

### Subtasks:

**3.1 Add progress checkpointing to notebook**
- [ ] Save intermediate results every 1000 regions
- [ ] Store in temporary geopackage: `wocu_wfs_checkpoint.gpkg`
- [ ] On script restart, check for checkpoint and resume

**3.2 Add memory profiling**
- [ ] Track memory usage during 233-region run
- [ ] Estimate memory for 12,130 regions (50x scale)
- [ ] If > 16GB, implement batch processing (1000 regions at a time)

**3.3 Test scaling logic**
- [ ] Run on 500 regions (~21 min)
- [ ] Confirm linear scaling (10 min → 21 min → 7 hours)
- [ ] Check for memory leaks (should be flat, not growing)

---

## TASK 4: Update DataHandler Tests/Documentation

**File:** `src/data/data_handler.py` (docstrings)

### Subtasks:

**4.1 Update class docstring**
- [ ] Mention two workflows: live WFS vs pre-saved geopackage
- [ ] Recommend geopackage workflow for large-scale (>100 regions)

**4.2 Add usage example in docstring**
```python
# Workflow 1: Load from GeoPackage (recommended)
data_handler.load_remote_data_from_geopackage("data/wfs_bundle.gpkg")

# Workflow 2: Live WFS fetching (legacy, slow for >100 regions)
data_handler.create_data_from_remote()
```

---

## DECISION LOG

### ✅ Keep DataCollector Pure
- **Decision:** Do NOT add save functionality to DataCollector
- **Reason:** Single responsibility - it fetches for ONE region only
- **Alternative:** Create separate `WFSDataBundler` utility (Task 2)

### ✅ Separate Fetch and Process
- **Decision:** Fetch → Save → Validate → Process (not Fetch+Process together)
- **Reason:** QGIS validation checkpoint prevents wasted compute

### ✅ Geopackage Layer Naming (Updated 2026-01-27)
- **Format:** `{service}/{layer}` (e.g., `land_use/BrpGewas`, `building_location/bag:pand`)
- **Reason:** Native geopackage folder structure, preserves colons in layer names, simpler parsing
- **Previous format:** `wfs_{service}_{layer_encoded}` - deprecated due to parsing ambiguity

---

## PRIORITY ORDER (Updated)

**COMPLETED:**
- ✅ Task 2: Created WFSDataBundler class and test notebook
- ✅ Task 1: DataHandler geopackage loader implemented and tested (verified with 10 regions)

**NEXT STEPS:**

1. **HIGH:** Test WFSDataBundler at scale (100 → 500 regions)
   - Run `notebooks/test_wfs_bundler.ipynb` with larger datasets
   - Verify output in QGIS
   - Estimate timing for full 12,130 run

2. **HIGH:** Task 3 (Scaling prep) - Add checkpointing before full run
   - Memory profiling at 500 regions
   - Add `fetch_with_checkpoints()` method to bundler
   - Test checkpoint/resume logic

3. **CRITICAL:** Full data collection run (12,130 regions)
   - Run overnight with checkpointing
   - Validate output in QGIS

4. **HIGH:** Update `demo_baseline_model.ipynb` to use geopackage loading
   - Replace `create_data_from_remote()` with `load_remote_data_from_geopackage()`
   - Verify model predictions remain identical

5. **LOW:** Task 4 (Documentation updates)
   - Update DataHandler class docstring
   - Add workflow examples

---

## CURSOR PROMPTS TO USE

**For Task 1.1:**
```
Add a new method to DataHandler called load_remote_data_from_geopackage(gpkg_path).
It should:
1. Use fiona.listlayers() to get all layers in the geopackage
2. Filter for layers starting with "wfs_"
3. Parse layer names like "wfs_pdok_waterdeel" into service="pdok", layer="waterdeel"
4. Load each layer as a GeoDataFrame
5. Store in nested dict: bundled_wfs_data[service][layer] = gdf
6. Call self._process_bundled_data_into_features(bundled_wfs_data)
```

**For Task 1.2:**
```
Add a new method to DataHandler called _process_bundled_data_into_features(bundled_wfs_data).
It should loop through self.prediction_regions and for each region:
1. Buffer the region geometry using self.config.prediction_region_buffer
2. For each WFS service/layer in bundled_wfs_data:
   - Spatially filter to the buffered region
   - Get feature config from self.config.feature_creation_config
   - Call self._generate_region_features()
3. Combine all features for that region into a single dict
4. Append to list, convert to DataFrame, merge with self.scope_region_features
```