# DataHandler Current Workflow

**Reference Notebook:** `notebooks/demo_baseline_model.ipynb`

This document describes the **current** DataHandler workflow that we need to preserve when implementing geopackage loading.

---

## 📋 Current Complete Workflow

### **Inputs Required:**

1. **Configuration** (`DataConfiguration`)
   - WFS services to query
   - Feature generation config (what aggregations to compute)
   - Buffer distances, lags, futures, etc.

2. **Prediction Regions** (`GeoDataFrame`)
   - Polygons defining areas to predict for
   - Must have an ID column (e.g., `location_id`, `position_id`)

3. **Erosion Data** (`GeoDataFrame`) - Optional for predictions
   - Historical river bank point locations with timestamps
   - Used for training models

4. **Erosion Border** (`LineString`) - Optional for predictions
   - Reference line for calculating erosion distances
   - Used for training models

5. **Local Geospatial Data** (`dict[str, GeoDataFrame]`) - Optional
   - River centerline
   - Other local reference data

---

## 🔄 Step-by-Step Current Workflow

### **Step 1: Initialize DataHandler**

```python
data_handler = DH.DataHandler(
    config=baseline_configuration,
    prediction_regions=prediction_regions,
    local_data_for_enrichment=local_geospatial_data,  # Optional
    erosion_data=river_bank_locations,                # Optional (for training)
    erosion_border=erosion_border,                    # Optional (for training)
)
```

**What happens:**
- Stores configuration
- Stores prediction regions
- Initializes empty feature storage
- Sets up tracking variables

---

### **Step 2: Fetch WFS Data and Generate Features**

```python
data_handler.create_data_from_remote()
```

**What happens:**
1. **For each prediction region:**
   - Creates a `DataCollector` instance
   - Fetches WFS data (with buffer) - **THIS IS SLOW (~2s per region)**
   - Extracts features using `_generate_region_features()`:
     - `area_fraction` - What % of region is covered?
     - `majority_class` - Most common category
     - `numerical_density` - Count per unit area
     - `count` - Total number of features
     - `centerline_shape` - Geometric shape features
   - Stores features in a list

2. **After all regions:**
   - Combines into DataFrame: `data_handler.scope_region_features`
   - Sets `remote_data_downloaded = True`

**Output:** `data_handler.scope_region_features`
- DataFrame with one row per prediction region
- Columns like:
  ```
  location_id
  geometry
  BrpGewas_area_fraction
  BrpGewas_majority_class_category
  bag:pand_area_fraction
  rws_vegetatielegger:bomen_numerical_density
  ...
  ```

**Timing:**
- 11 regions: ~22 seconds (~2s per region)
- 233 regions: ~8-10 minutes
- 12,130 regions: **~7 hours** 😱

---

### **Step 3: Process Erosion Features** (Optional - for training only)

```python
data_handler.process_erosion_features()
```

**What happens:**
1. Filters erosion data to prediction regions
2. For each (region, timestamp) pair:
   - Calculates distance from river bank to erosion border
   - Computes mean of N closest points
3. Creates time-series structure:
   ```
   location_id | timestamp | distance_to_erosion_border
   -----------+-----------+---------------------------
   waal_949_0 | 2015      | 100.5
   waal_949_0 | 2018      | 95.2
   ```
4. Adds automated features:
   - `timesteps_since_last_measurement`

**Output:** `data_handler.processed_erosion_data`
- MultiIndex DataFrame: `(location_id, timestamp)`
- Contains temporal erosion measurements

---

### **Step 4: Add Remote Features to Erosion Data** (Optional - for training)

```python
data_handler.add_remote_data_to_processed()
```

**What happens:**
1. Merges `scope_region_features` with `processed_erosion_data`
2. Joins on `location_id` (prediction region ID)
3. Each erosion measurement gets the WFS features for its region

**Output:** `data_handler.processed_erosion_data` (updated)
- Now includes WFS features alongside erosion measurements

---

### **Step 5: Generate ML-Ready Features** (Optional - for training)

```python
data_handler.generate_erosion_features()
```

**What happens:**
1. For each column in processed data:
   - Creates lagged versions (past measurements)
   - Creates future versions (targets for prediction)
   - Applies differencing if configured
2. Handles categorical encoding
3. Drops rows with NaN values

**Output:** `data_handler.erosion_features`
- Time-lagged feature matrix ready for ML models
- Format: `[n_samples, n_features]`

---

### **Step 6: Generate PyTorch Dataset** (Optional - for deep learning)

```python
data_handler.generate_pytorch_features()
```

**What happens:**
1. Converts features to PyTorch tensors
2. Applies min-max scaling
3. Reshapes for LSTM/RNN models

**Output:** `data_handler.pytorch_dataset`
- PyTorch-compatible dataset

---

## 🎯 What We Need to Preserve

When implementing `load_remote_data_from_geopackage()`, we must preserve:

### ✅ **Same Output Structure**

`data_handler.scope_region_features` must have:
- Same columns: `location_id`, `geometry`, `{layer}_{aggregation}`, ...
- Same data types
- Same row order (one row per prediction region)

### ✅ **Same Feature Names**

Feature naming convention:
```
{layer_name}_{aggregation_function}
```

Examples:
- `BrpGewas_area_fraction`
- `BrpGewas_majority_class_category`
- `bag:pand_numerical_density`
- `rws_vegetatielegger:bomen_count`

### ✅ **Same Aggregation Functions**

Must support all existing aggregations:
- `area_fraction` - Coverage percentage
- `majority_class` - Most common category  
- `numerical_density` - Features per unit area
- `count` - Total count
- `total_area` - Sum of areas
- `centerline_shape` - Geometric shape features

### ✅ **Same Error Handling**

Gracefully handle:
- Empty geometries (no WFS features in region)
- Missing layers
- CRS mismatches

---

## 🔄 Proposed New Workflow (Task 1)

### **Current (Slow):**
```python
data_handler.create_data_from_remote()  # 7 hours for 12,130 regions
```

### **Proposed (Fast):**
```python
# Option 1: Load from geopackage
data_handler.load_remote_data_from_geopackage("wfs_data.gpkg")  # ~seconds

# Option 2: Still support live fetching (backward compatibility)
data_handler.create_data_from_remote()  # Keep for small datasets
```

**Key difference:**
- **Current:** Fetches WFS data live for each region (slow)
- **Proposed:** Loads pre-fetched data from geopackage (fast)
- **Same output:** Both methods produce identical `scope_region_features`

---

## 📊 Performance Comparison

| Method | 11 regions | 233 regions | 12,130 regions |
|--------|-----------|-------------|----------------|
| **Current** (`create_data_from_remote`) | 22s | 10 min | ~7 hours |
| **Proposed** (`load_from_geopackage`) | <1s | ~5s | ~1-2 min |

**Speedup:** ~200-400x faster! 🚀

---

## 🧪 Testing Requirements

When implementing the new geopackage loader, we must verify:

1. **Output Equivalence:**
   - Run `create_data_from_remote()` → save features
   - Run `load_remote_data_from_geopackage()` → compare features
   - Assert: DataFrames are identical (or near-identical due to floating point)

2. **Feature Name Consistency:**
   - Check all column names match
   - Verify aggregation values are correct

3. **Edge Cases:**
   - Empty regions (no WFS data)
   - Missing layers in geopackage
   - Different CRS in geopackage vs prediction regions

4. **Integration:**
   - Full pipeline still works:
     ```python
     load_remote_data_from_geopackage()
     process_erosion_features()
     add_remote_data_to_processed()
     generate_erosion_features()
     ```

---

## 📁 Files to Modify

1. **`src/data/data_handler.py`**
   - Add `load_remote_data_from_geopackage(gpkg_path)` method
   - Add `_process_bundled_data_into_features(bundled_wfs_data)` helper
   - Keep existing `create_data_from_remote()` for backward compatibility

2. **`notebooks/demo_baseline_model.ipynb`**
   - Update to use geopackage loading
   - Keep old version in comments for comparison

3. **`tests/test_data_handler.py`**
   - Add tests for geopackage loading
   - Test output equivalence

---

## 🔑 Key Implementation Details

### **Spatial Filtering Logic**

The new geopackage loader must replicate this logic from `create_data_from_remote()`:

```python
# For each prediction region
for region in prediction_regions:
    region_buffered = region.buffer(config.prediction_region_buffer)
    
    # For each WFS layer
    for service, layers in bundled_wfs_data.items():
        for layer_name, layer_gdf in layers.items():
            # Spatial filter
            filtered_data = layer_gdf[layer_gdf.intersects(region_buffered)]
            
            # Generate features
            features = _generate_region_features(
                region=region,
                geospatial_data=filtered_data,
                feature_config=config.feature_creation_config[layer_name]
            )
```

### **Feature Generation**

Reuses existing `_generate_region_features()` method - **no changes needed!** ✅

---

## 📝 Summary

**Current workflow works but is slow.**

**Proposed change:**
- Add geopackage loading option
- Keep live fetching for backward compatibility
- Same outputs, same features, same downstream pipeline
- **200-400x faster for large datasets**

**Next step:** Implement Task 1.1 and 1.2 from TODO.md
