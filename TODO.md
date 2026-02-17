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
⬜ Need to test bundler at scale (10 → 100 → 500 → 12,130 regions)
⬜ Need to extend DataHandler to load from GeoPackage (instead of live WFS)

---

## TASK 1: Refactor DataHandler - Add Geopackage Loader

**Goal:** Let DataHandler load pre-saved WFS data instead of fetching live

**File:** `src/data/data_handler.py`

### Subtasks:

**1.1 Add method: `load_remote_data_from_geopackage()`**
- [ ] Read all layers from geopackage using `fiona.listlayers()`
- [ ] Filter for layers starting with `wfs_`
- [ ] Parse layer names: `wfs_{service}_{layer}` → extract service + layer
- [ ] Load each layer as GeoDataFrame
- [ ] Store in nested dict: `bundled_wfs_data[service_name][layer_name] = gdf`
- [ ] Call `_process_bundled_data_into_features(bundled_wfs_data)`
- [ ] Set `self.remote_data_downloaded = True`

**1.2 Add method: `_process_bundled_data_into_features()`**
- [ ] Loop through `self.prediction_regions`
- [ ] For each region:
  - [ ] Buffer the region geometry (use `self.config.prediction_region_buffer`)
  - [ ] For each WFS service/layer in `bundled_wfs_data`:
    - [ ] Spatially filter: `layer_gdf[layer_gdf.intersects(buffered_region)]`
    - [ ] Get feature config: `self.config.feature_creation_config.get(layer_name)`
    - [ ] Call `self._generate_region_features(region_geom, filtered_data, feature_config)`
    - [ ] Name features: `{layer_name}_{agg_function}`
  - [ ] Flatten nested dict: `UTILS.flatten_dictionary(single_region_features)`
  - [ ] Append to `wfs_features` list
- [ ] Convert to DataFrame
- [ ] Merge with `self.scope_region_features`

**1.3 Test the new methods**
- [ ] Load your saved GeoPackage (`wocu_output_fase2_v4_w_features_20260127.gpkg`)
- [ ] Confirm all WFS layers are detected
- [ ] Verify features are calculated correctly for 233 regions
- [ ] Compare output to `create_data_from_remote()` (should match structurally)

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

### ✅ Geopackage Layer Naming
- **Format:** `wfs_{service}_{layer}` (e.g., `wfs_pdok_waterdeel`)
- **Reason:** Unambiguous parsing, avoids name collisions

---

## PRIORITY ORDER (Updated)

**COMPLETED:**
- ✅ Task 2: Created WFSDataBundler class and test notebook

**NEXT STEPS:**

1. **HIGH:** Test WFSDataBundler with 10 → 100 → 500 regions
   - Run `notebooks/test_wfs_bundler.ipynb`
   - Verify output in QGIS
   - Estimate timing for full 12,130 run

2. **HIGH:** Task 3 (Scaling prep) - Add checkpointing before full run
   - Memory profiling at 500 regions
   - Add `fetch_with_checkpoints()` method to bundler
   - Test checkpoint/resume logic

3. **CRITICAL:** Full data collection run (12,130 regions)
   - Run overnight with checkpointing
   - Validate output in QGIS

4. **HIGH:** Task 1.1 + 1.2 (DataHandler geopackage loader) - 2-3 hours
   - Only start AFTER data collection is validated

5. **MEDIUM:** Task 1.3 (Test with your geopackage) - 1 hour

6. **LOW:** Task 4 (Documentation)

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