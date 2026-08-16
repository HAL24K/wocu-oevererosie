# Data & Notebook Inventory

**Purpose**: Single source of truth for all data files and notebooks  
**Status**: 🚧 In Progress - Collaborative review with QGIS observations  
**Last Updated**: 2026-01-07

---

## 🗺️ Project Context

### **Study Areas** (3 Locations)
1. **Empel** - Maas River, Km 218.0–224.0 (WOCU Maas Empel)
2. **Brakel** - Waal River, Km 951.0–945.0 (WOCU Rijntakken Brakel)
3. **Terwolde** - IJssel River, Km 947.0–953.0 (WOCU IJssel Terwolde)

### **Data Evolution**
- **2025-01-08 → 2025-04-01**: Brakel-only analysis (files 3-5 below)
- **2025-07-15 → 2025-08-15**: Expanded to **all 3 areas** in Phase 1 (files 1-2)
- **Next**: Phase 2 starting this week - Luke Moth (AHN model) + SAM model team will create **hybrid model** to determine best tool for river bank detection (1-2 months)

### **Project Handover Context**
- **Previous**: Ondrej (data scientist, now on parental leave) - did most work at start of 2025
- **Current**: You took over the project this week
- **AHN Model**: Managed by Luke Moth, all files below are iterations of AHN model findings
- **Next Phase**: Field visits to river banks (oevers) planned

### **Key Datasets**

**Local Files** (in `data/`):
- **AHN Data**: Elevation measurements from 3 versions (AHN3, AHN4, AHN5)
- **Shore Points**: `punten_oever` - Points along river banks with erosion measurements
- **Prediction Regions**: `vlakken_scope` - Polygons defining areas for prediction
- **Detected Erosion**: `vlakken_erosie` - Polygons of detected erosion areas
- **Reference Line**: `middenlijn` - River centerline (two sources: own vs. Levering_erosie_data)
- **Protected Shores**: `beschermde_oever` - Stone-protected shore points (only in newest file)

**External WFS Services** (fetched on-demand):
- **Land Use**: BRP Gewaspercelen (crop/agricultural data)
- **Buildings**: BAG (building footprints and usage)
- **Vegetation**: RWS Vegetatielegger (trees, hedges, vegetation classes)

---

## 📂 Main Analysis Files (Comparison Table)

| File | Created | Areas | Layers | Centerline Source | Notes |
|------|---------|-------|--------|-------------------|-------|
| **1. phase1_2025-08-14_v1.gpkg** | 2025-08-15 | ✅ All 3 | punten_oever, vlakken_erosie, vlakken_scope, middenlijn, **beschermde_oever** | Own middenlijn | ⭐ **MOST COMPLETE**: More coverage squares, better defined vlakken_erosie |
| **2. phase1_2025-07-15_v3.gpkg** | 2025-07-15 | ✅ All 3 | punten_oever, vlakken_erosie, vlakken_scope, middenlijn | Own middenlijn | Proof of concept? Less coverage than file 1 |
| **3. luke_inputs_v3.gpkg** | 2025-04-01 | ❌ Brakel only | punten_oever, vlakken_erosie, vlakken_scope, middenlijn | Levering_erosie_data | Used by `demo_baseline_model.ipynb` |
| **4. all_results_20250121_v2.gpkg** | 2025-01-21 | ❌ Brakel only | punten_oever, vlakken_erosie, vlakken_scope | Levering_erosie_data | Older version |
| **5. draft_oever_degradation_data.gpkg** | 2025-01-08 | ❌ Brakel only | punten_oever, vlakken_erosie, vlakken_scope | Levering_erosie_data | Draft version |

### **Your Observations & Answers**:
- ✅ File 1 vs. File 2: File 1 has more squares and better vlakken_erosie definition
- ✅ File 2: Likely proof of concept for all-3-areas expansion
- ✅ Files 3-5: Only Brakel, use external centerline from Levering_erosie_data

### **Decisions Made**:

1. **Canonical file going forward**: ✅ `phase1_2025-08-14_v1.gpkg` (newest, all areas, best quality)

2. **`luke_inputs_v3.gpkg` status**: 
   - Keep for now (used by `demo_baseline_model.ipynb`)
   - Eventually update demo to use `phase1_2025-08-14_v1.gpkg`
   - Used for historical Brakel-only comparison

3. **Protected shores (`beschermde_oever`)**:
   - Only in `phase1_2025-08-14_v1.gpkg`
   - 23,497 stone-protected shore points (all `type_oever = 'steen(bekleding)'`)
   - Need to learn more during field visits (next phase)

4. **Files 4-5 (old Brakel versions)**:
   - ✅ Archive `all_results_20250121_v2.gpkg` (Jan 21)
   - ✅ Archive `draft_oever_degradation_data.gpkg` (Jan 8)

### **Open Questions**:
- ❓ **Centerline accuracy**: Phase1 files have own centerline, files 3-5 use Levering_erosie_data. Which is more accurate? → **Ask Luke Moth**
- ❓ **Versioning confusion**: File 1 is `v1` (Aug), File 2 is `v3` (July). Was there a v2?

---

## 📂 Data Files (Grouped by Purpose)

### **GROUP 1: Current Analysis Data** ⭐ ACTIVE

#### **`phase1_2025-08-14_v1.gpkg`** ✅ MOST COMPLETE
**Created**: 2025-08-15  
**Size**: 16.66 MB  
**Areas**: Empel, Brakel, Terwolde (ALL 3)

**Layers** (6 total):

1. **`punten_oever`** (70,495 points) - Shore points with slope classification
   - Type distribution: MILD_SLOPE, MEDIUM_SLOPE, STEEP_SLOPE, CLIFF
   - Status: OK, UNCERTAIN, OUTLIER
   - 203 unique locations
   - 3 DTM versions: AHN3 (2016), AHN4, AHN5 (2024)

2. **`beschermde_oever`** (23,497 points) - Protected shores **[UNIQUE TO THIS FILE]**
   - All `type_oever = 'steen(bekleding)'` (stone-protected)
   - 132 unique locations
   - 19.2% null geometry (outliers?)

3. **`vlakken_erosie`** (373 polygons) - Detected erosion areas
   - AHN3 → AHN4/AHN5 comparisons
   - 130 unique locations
   - Properties: area, mean_diff_z (elevation change)

4. **`vlakken_scope`** (233 polygons) - Prediction regions
   - 233 unique locations (one per polygon)
   - Timespan: 2016 → 2024 (8 years)
   - **Note**: Seems identical to `summary_layer` in structure

5. **`summary_layer`** (233 polygons) - Aggregated metrics per location
   - Same 233 locations as vlakken_scope
   - Metrics: total_erosion, diff_distance, etc.

6. **`middenlijn`** (626 linestrings) - River centerline
   - Own version (not from Levering_erosie_data)
   - 238 unique location_ids

**Used By**: `height_sam_comparison_viz.ipynb`, `sam_comparison.ipynb`  
**Status**: ✅ **PRIMARY FILE** - Most recent and comprehensive

---

#### **`phase1_2025-07-15_v3.gpkg`** 📦 PROOF OF CONCEPT?
**Created**: 2025-07-15  
**Areas**: Empel, Brakel, Terwolde (ALL 3)  
**Layers**: Similar to above, but NO `beschermde_oever`

**Used By**: None  
**Status**: ❓ **TBD**

**Your Notes**:
- Less coverage than Aug 14 version
- Proof of concept?
- [ ] Keep for comparison or archive?: __________

---

#### **`luke_inputs_v3.gpkg`** ✅ USED BY DEMO (but Brakel-only)
**Created**: 2025-04-01  
**Areas**: Brakel only  
**Layers**: `punten_oever`, `vlakken_scope`, `vlakken_erosie`, `middenlijn`, `samenvatting`

**Used By**: ⭐ `demo_baseline_model.ipynb` (main demo!)  
**Centerline**: Uses `Centreline_River` from `Levering_erosie_data.gpkg`  
**Status**: ✅ **CURRENTLY USED** - But only Brakel

**Your Notes**:
- Only Brakel area
- Used by main demo notebook
- [ ] What's in this that's NOT in phase1 files?: __________
- [ ] Should we migrate demo to use phase1 instead?: __________

---

### **GROUP 2: Historical Versions** 📦 ARCHIVE CANDIDATES

#### **`all_results_20250121_v2.gpkg`** 📦 ARCHIVE
**Created**: 2025-01-21  
**Areas**: Brakel only  
**Used By**: None

**Status**: 📦 **ARCHIVE** - Superseded by luke_inputs_v3 and phase1 files

**Your Decision**: [ ] Archive or Delete?: __________

---

#### **`draft_oever_degradation_data.gpkg`** 📦 ARCHIVE
**Created**: 2025-01-08  
**Areas**: Brakel only  
**Used By**: None

**Status**: 📦 **ARCHIVE** - Draft version, superseded

**Your Decision**: [ ] Archive or Delete?: __________

---

### **GROUP 3: Reference Data** (Stable Context)

#### **`Levering_erosie_data.gpkg`** ✅ KEEP - REFERENCE
**Layers**: 
- `Centreline_River` (13 lines) - River centerlines
- `Uiterwaardegrenzen` (134 polygons) - Floodplain boundaries
- `Kribben_BKN` (1,922 polygons) - Groynes (river structures)
- `Kilometrering` (251 points) - Distance markers
- `Vlak_voor_vrijeruimte_nevengeulen` - Free space for side channels
- `Vlak_voor_vrijeruimte_natuurvriendelijke_oever` - Free space for nature-friendly shores
- And many more...

**Used By**: `demo_baseline_model.ipynb` (for centerline), `explore_raw_data.ipynb`  
**Status**: ✅ **KEEP** - Reference context data

**Your Notes**:
- Provides centerline for files 3-5
- Vrijeruimte layers look similar to `Vlak_Vrijeruimte_ln.gpkg`
- [ ] Observations: __________

---

#### **`Vlak_Vrijeruimte_ln.gpkg`** ❓ DUPLICATE?
**Layers**: `Vrijruimte_NVO_ln` (286 lines)

**Used By**: None  
**Your Notes**: Looks a lot like `Vlak_voor_vrijeruimte_*` layers in `Levering_erosie_data.gpkg`

**Questions for You**:
- [ ] Is this a duplicate of data in Levering_erosie_data.gpkg?: __________
- [ ] If yes, can we delete this and just use Levering_erosie_data?: __________
- [ ] If no, what's different?: __________

---

### **GROUP 4: Erosion Borders** (Critical Thresholds)

#### **`erosion_border_20250129.gpkg`** ✅ CRITICAL
**Created**: 2025-01-29  
**Layers**: `Tekenen_signaallijn_20250129` (signalling line)  
**Coverage**: Part of Brakel area only

**Used By**: ⭐ `demo_baseline_model.ipynb`  
**Status**: ✅ **KEEP** - Defines critical threshold for predictions

**Notes**:
- Manually drawn border for Brakel
- ✅ **Empel and Terwolde need their own signalling lines** (not yet created)
- Plan: Eventually create one comprehensive file with all 3 areas
- This is a key blocker for running predictions on Empel/Terwolde

---

#### **`handdrawn_fake_erosion_border.geojson`** 🗑️ DELETE
**Status**: 🗑️ **DELETE** - Superseded by `erosion_border_20250129.gpkg`

**Your Confirmation**: [ ] Safe to delete?: ✅ Yes (you confirmed useless)

---

### **GROUP 5: SAM (Segment Anything Model) Data**

#### **`sam/sam_processed.gpkg`** ✅ KEEP - SAM PREDICTIONS
**Size**: 34.70 MB  
**Layers**: 87 polygons + 118K points  
**Coverage**: All 3 areas (Empel, Brakel, Terwolde)

**Used By**: `height_sam_comparison_viz.ipynb`  
**Purpose**: Processed SAM predictions for erosion detection

**Status**: ✅ **KEEP** - Training/validation data for Phase 2 hybrid model

**SAM vs. AHN Comparison**:
- **SAM**: Segment Anything Model - image-based erosion detection from aerial photos
- **AHN**: Actueel Hoogtebestand Nederland - elevation-based erosion detection from LIDAR
- **Phase 2 Goal**: Hybrid model combining both approaches (Luke + SAM team, 1-2 months)
- **Raw SAM data**: Complete for all 3 areas (Brakel: 937, Empel: 924, Terwolde: 802 features)

---

#### **`sam/Brakel.geojsonl`** ✅ KEEP - RAW SAM (Brakel)
**Features**: 937  
**Used By**: `sam_comparison.ipynb`  
**Purpose**: Raw SAM annotations for Brakel location

**Status**: ✅ **KEEP** - Raw training data

---

#### **`sam/Empel.geojsonl`** ✅ KEEP - RAW SAM (Empel)
**Size**: 24.90 MB  
**Features**: 924 polygons  
**Used By**: None (yet)

**Purpose**: Raw SAM annotations for Empel location  
**Properties**: observation_id, patch_id, observation_date, water_height_m, rotation_angle_deg, source/mask/soil image filenames

**Status**: ✅ **KEEP** - Raw training data, complete and ready for Phase 2 hybrid model

---

#### **`sam/Terwolde.geojsonl`** ✅ KEEP - RAW SAM (Terwolde)
**Size**: 27.04 MB  
**Features**: 802 polygons  
**Used By**: None (yet)

**Purpose**: Raw SAM annotations for Terwolde location  
**Properties**: observation_id, patch_id, observation_date, water_height_m, rotation_angle_deg, source/mask/soil image filenames

**Status**: ✅ **KEEP** - Raw training data, complete and ready for Phase 2 hybrid model

---

### **GROUP 6: Risk Predictions** (Model Outputs)

#### **`erosion_locations_sam.gpkg`** ✅ KEEP - PREDICTIONS
**Size**: 0.18 MB  
**Layers**: `erosian locations` (93 points)

**Purpose**: Predicted breach locations with years and erosion speed

**Key Fields**:
- `earliest_breach_year`: When erosion will reach critical threshold
- `avg_erosion_speed`: Average erosion rate (m/year)
- `max_erosion_speed`: Maximum erosion rate (m/year)
- `source_polygons`: Which vlak_van_vrije_ruimte_nvo_legger polygons contributed
- `num_points`: Number of points in cluster

**Sample Predictions**:
- ⚠️ **URGENT**: Point 56 → Breach year **2025** (0.85 m/yr avg, 1.05 m/yr max)
- ⚠️ **URGENT**: Point 63 → Breach year **2025** (0.61 m/yr avg, 0.63 m/yr max)
- ⚠️ **URGENT**: Point 67 → Breach year **2025** (3.24 m/yr avg - **VERY FAST**)
- ⚠️ Point 9 → Breach year **2027** (0.93 m/yr avg, 1.19 m/yr max)
- ⚠️ Point 39 → Breach year **2027** (0.58 m/yr avg)

**Used By**: None (yet!)  
**Status**: ✅ **CRITICAL** - Important model output with URGENT 2025 predictions

**Action Items**:
- [ ] Verify 2025 predictions with current field data
- [ ] Check if Point 67 (3.24 m/yr) is an outlier or real crisis
- [ ] Cross-reference with phase1 vlakken_erosie to confirm detections

---

### **GROUP 7: Outdated/Test Files** 🗑️

#### **`VO155184_Scope_Pilot_Bankerosion_20241129.shp`** 🗑️ OUTDATED
**Features**: 4 pilot area polygons  
**Created**: 2024-11-29

**Used By**: `explore_raw_data.ipynb`  
**Your Notes**: Outdated polygons, superseded by other files

**Your Decision**: [ ] Delete?: ✅ Yes (you confirmed outdated)

---

#### **`water_data/*.csv`** ❓ BROKEN
**Status**: Malformed (single column)

**Questions for You**:
- [ ] What should these contain?: __________
- [ ] Worth fixing or just delete?: __________

---

## 🌐 WFS Services (External Data Sources)

The project uses **Web Feature Services (WFS)** to automatically fetch geospatial context data for each prediction region. These services are managed by Dutch government agencies and provide live data.

### **Overview**
- **Purpose**: Enrich erosion data with geospatial context (land use, buildings, vegetation)
- **How it works**: For each prediction region, the `DataCollector` class fetches data within the bounding box
- **Configured in**: `src/config.py` → `KNOWN_WFS_SERVICES`

---

### **1. Land Use Data** 🌾
**Service**: BRP Gewaspercelen (Basisregistratie Gewaspercelen)  
**Provider**: RVO (Rijksdienst voor Ondernemend Nederland)  
**URL**: `https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0`  
**Version**: 1.0.0

**Layers Used**:
- `BrpGewas` - Agricultural crop data per parcel

**Features Extracted**:
- **Majority crop type** (`category`, `gewas` columns)
- **Area fraction**: Percentage of prediction region covered by crops

**Use Case**: Understand if land use (e.g., agriculture vs. natural vegetation) correlates with erosion rates

---

### **2. Building Location Data** 🏠
**Service**: BAG (Basisregistraties Adressen en Gebouwen)  
**Provider**: Kadaster  
**URL**: `https://service.pdok.nl/lv/bag/wfs/v2_0`  
**Version**: 2.0.0

**Layers Used**:
- `bag:pand` - Building footprints and usage

**Features Extracted**:
- **Majority building usage** (`gebruiksdoel` column)
- **Area fraction**: Percentage of prediction region covered by buildings

**Use Case**: Assess if built-up areas have different erosion patterns (e.g., runoff, soil compaction)

---

### **3. Vegetation Data** 🌳
**Service**: RWS Vegetatielegger (Rijkswaterstaat Vegetation Register)  
**Provider**: Rijkswaterstaat  
**URL**: `https://geo.rijkswaterstaat.nl/services/ogc/gdr/rws_vegetatielegger/ows?version=2.0.0`  
**Version**: 2.0.0

**Layers Used**:
- `rws_vegetatielegger:bomen` - Individual trees (points)
- `rws_vegetatielegger:heggen` - Hedges (linestrings) [Currently disabled in feature extraction]
- `rws_vegetatielegger:vegetatieklassen` - Vegetation classes (polygons)

**Features Extracted**:
- **Tree density**: Number of trees per unit area
- **Majority vegetation class** (`vlklasse` column)

**Use Case**: Tree roots stabilize soil, so vegetation density may correlate with lower erosion rates

---

### **How WFS Data is Used**

1. **Data Collection** (`DataCollector` class):
   - For each prediction region in `vlakken_scope`
   - Fetch data from all 3 WFS services within bounding box
   - Handle pagination (max 10,000 features per request)
   - Transform all data to EPSG:28992 (RD New)

2. **Feature Engineering** (`DataHandler` class):
   - Aggregate WFS data per prediction region
   - Calculate features based on `AGGREGATION_COLUMNS` config:
     - **Majority class**: Most common category in region
     - **Area fraction**: % of region covered
     - **Numerical density**: Count per unit area
     - **Centerline shape**: Distance/curvature metrics

3. **ML Model Input**:
   - WFS features are combined with erosion measurements
   - Used to train `BaselineErosionModel` and `ml_predictive_model`
   - Help predict **when** erosion will breach signalling line

---

### **WFS Configuration Details**

**File**: `src/config.py`

```python
KNOWN_WFS_SERVICES = [
    WfsService(
        name="land_use",
        url="https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0",
        relevant_layers=["BrpGewas"],
    ),
    WfsService(
        name="building_location",
        url="https://service.pdok.nl/lv/bag/wfs/v2_0",
        version="2.0.0",
        relevant_layers=["bag:pand"],
    ),
    WfsService(
        name="vegetation",
        url="https://geo.rijkswaterstaat.nl/services/ogc/gdr/rws_vegetatielegger/ows?version=2.0.0",
        version="2.0.0",
        relevant_layers=[
            "rws_vegetatielegger:bomen",
            "rws_vegetatielegger:heggen",
            "rws_vegetatielegger:vegetatieklassen",
        ],
    ),
]
```

---

### **Testing WFS Services**

**Notebook**: `explore_data_collector.ipynb`  
**Purpose**: Demonstrates how to use the `DataCollector` class to fetch WFS data

**To test WFS connectivity**:
```python
from src.data.data_collector import DataCollector
import src.config as CONFIG

# Create collector for a test region
collector = DataCollector(
    source_shape=test_polygon,
    source_epsg_crs=28992,
    wfs_services=CONFIG.KNOWN_WFS_SERVICES
)

# Fetch all data
collector.get_data_from_all_wfs()

# Inspect results
print(collector.relevant_geospatial_data.keys())
```

---

### **WFS Limitations & Notes**

⚠️ **Known Issues**:
1. **Feature limits**: Some WFS services limit max features per request (usually 10,000)
   - Solution: Pagination with `startindex` parameter
2. **High index failures**: Some services refuse requests past index 50,000
   - Mitigation: Code handles this gracefully
3. **CRS assumptions**: Code assumes Dutch RD (EPSG:28992) for most operations
   - Future: Auto-detect CRS from WFS metadata

📝 **Dutch-Centric**:
- All WFS services are from Dutch government agencies
- Data only available for Netherlands
- For international use: Replace with local WFS services

🔄 **Live Data**:
- WFS data is fetched on-demand (not cached)
- Always up-to-date, but requires internet connection
- Consider caching for offline development

---

## 🌍 Soil Composition Data (BRO Bodemkaart)

### **`BRO_DownloadBodemkaart.gpkg`** ✅ SOIL DATA

**Source**: Basisregistratie Ondergrond (BRO) via PDOK  
**Version**: V2025-1 (October 2025)  
**Size**: 146.0 MB  
**Scale**: 1:50,000 (regional, not parcel-level)  
**Coverage**: National (Netherlands), but sparse in urban areas

**Status**: ✅ **DOWNLOADED & VALIDATED** - Ready for integration

---

### **Structure: 2 Geographic Layers + 14 Lookup Tables**

#### **Geographic Layers (With Geometry):**

1. **`soilarea`** (48,025 polygons)
   - Main soil type areas across Netherlands
   - CRS: EPSG:28992 (RD New)
   - Direct columns: `maparea_id`, `soilslope`, `beginlifespan`, `endlifespan`
   - Links to soil types via `soilarea_soilunit` table

2. **`areaofpedologicalinterest`** (6,192 polygons)
   - Special soil characteristics (disturbed areas, terps, etc.)
   - Examples: "Sterk afgegraven terrein" (heavily excavated), "Terp" (dwelling mound)

#### **Lookup/Attribute Tables (No Geometry):**

These decode the soil codes into meaningful properties:

1. **`soil_units`** (307 soil types) - **MOST IMPORTANT**
   - `code`: Soil unit code (e.g., "ABv", "cHn21", "BKh26")
   - `soilclassification`: Full description (Dutch)
   - `mainsoilclassification`: Main category (e.g., "Brikgronden", "Dikke eerdgronden")
   - `url`: Link to online legend

2. **`soilarea_soilunit`** (50,053 rows) - **LINKING TABLE**
   - Links `soilarea` polygons to `soil_units`
   - One polygon can have multiple soil units

3. **`normalsoilprofiles`** (368 profiles)
   - Standard soil profiles with depth information (0-1.2m)
   - Contains `othersoilname` (Dutch description)

4. **`soilhorizon`** (1,568 horizons)
   - **Detailed soil properties per layer:**
   - `organicmattercontent`: % organic matter (affects binding/stability)
   - `loamcontent`: Clay percentage
   - `sandmedian`: Sand grain size (μm)
   - `siltcontent`: Silt percentage
   - `density`: Soil compaction (g/cm³)
   - `acidity`: pH values

5. **Other tables:**
   - `soilhorizon_fractionparticlesize`: Particle size distribution
   - `soillayer`: Geological layers and depositional characteristics
   - `soilcharacteristics_toplayer/bottomlayer`: Special characteristics
   - `normalsoilprofiles_landuse`: Land use classification
   - `soilmap`, `nga_properties`: Metadata

---

### **Data Relationships (For Erosion Modeling)**

```
soilarea (polygons with geometry)
    ↓ maparea_id
soilarea_soilunit (linking table)
    ↓ soilunit_code
soil_units (soil type names)
    → mainsoilclassification (e.g., "Brikgronden", "Veengronden")
    → soilclassification (full Dutch description)
```

**Example:**
- Polygon `V2025-1..soilarea.0000005485` (geometry)
- → Links to `soilunit_code = "ABv"`
- → "ABv" = "Brick soil type with specific properties"

---

### **Relevance for Erosion Prediction**

**Why Soil Type Matters:**
- **Sandy soils** erode faster than clay
- **Peat soils** (veengronden) have different stability characteristics
- **Organic matter content** affects soil binding
- **Particle size** affects erosion susceptibility

**Dutch Soil Types (Examples from data):**
- `cHn21`: Podzol (sandy soils)
- `ABv/ABz`: Brick soils (often clay-rich)
- `hVb/hVs`: Peat soils (veengronden)
- `EZg21`: Thick cultural soils (dikke eerdgronden)

---

### **Integration Strategy**

**Recommended Approach (Simple):**
1. Join `soilarea` → `soilarea_soilunit` → `soil_units`
2. Extract `mainsoilclassification` (10-20 main categories)
3. Use as **majority class** per prediction region
4. Pass as `local_geospatial_data` to `DataCollector`

**Code Example:**
```python
import geopandas as gpd
import pandas as pd
import sqlite3

# Load and join
gpkg_path = "data/BRO_DownloadBodemkaart.gpkg"
conn = sqlite3.connect(gpkg_path)

soilarea = gpd.read_file(gpkg_path, layer="soilarea")
soilarea_units = pd.read_sql_query("SELECT * FROM soilarea_soilunit", conn)
soil_types = pd.read_sql_query("SELECT code, mainsoilclassification FROM soil_units", conn)

# Join
soilarea_enriched = soilarea.merge(
    soilarea_units, on='maparea_id'
).merge(
    soil_types, left_on='soilunit_code', right_on='code'
)

# Use in DataCollector
local_geospatial_data = {
    "bodemkaart": soilarea_enriched
}
```

**Feature Extraction (in config.py):**
```python
AGGREGATION_COLUMNS = {
    "bodemkaart": FGC(
        majority_class={"columns": ["mainsoilclassification"]}
    ),
}
```

---

### **Limitations & Notes**

⚠️ **Resolution**: 1:50,000 scale
- Good for regional patterns along rivers
- Not precise enough for individual parcels
- ~10-20m accuracy

⚠️ **Urban Areas**:
- Sparse data under buildings (not mapped)
- May have gaps near developed riverbanks

⚠️ **Static Data**:
- Version V2025-1 (October 2025)
- Updates infrequent (peat areas updated 2014)
- Won't change over your 8-year study period (2016-2024)
- **This is actually good**: Static soil properties affect erosion rate consistently

⚠️ **Depth**: Only top 1.2 meters
- May not capture deeper substrate
- Sufficient for surface erosion analysis

📝 **Dutch Language**:
- All descriptions in Dutch
- May need translation for international use
- Key terms: "zand" (sand), "klei" (clay), "veen" (peat)

---

### **Exploration Script**

**File**: `scripts/explore_bodemkaart.py`

**Run**: `python scripts/explore_bodemkaart.py`

**Output**: 
- Lists all 16 layers (2 geographic + 14 lookup tables)
- Shows sample data from each table
- Displays column names and row counts
- Helps understand data structure before integration

---

### **Next Steps**

- [ ] Create preprocessing script to join tables
- [ ] Add to `AGGREGATION_COLUMNS` in `src/config.py`
- [ ] Test feature extraction with test regions
- [ ] Validate soil type distribution along river banks
- [ ] Integrate into full data pipeline

---

## 🌊 Water Level & Discharge Data (Waterweb/WADAR)

### **REST API: Waterweb (WADAR System)**

**Source**: Rijkswaterstaat WaterWebservices (New WADAR system)  
**Base URL**: https://ddapi20-waterwebservices.rijkswaterstaat.nl/  
**Status**: ✅ **AVAILABLE & TESTED** - Active data for all study areas

**Data Type**: Time-series water measurements (non-spatial)  
**Temporal Range**: Historical data from **1737 to 2027** (!)  
**Granularity**: **10-minute intervals**  
**Format**: JSON POST API (REST)

---

### **What Does WATHTE Measure?**

**WATHTE** = Water height in **cm relative to NAP** (Normaal Amsterdams Peil)
- **NOT** water depth (surface to river bottom)
- **Absolute water level** relative to Dutch reference datum (≈ sea level)
- Higher values = higher water level = more flooding/erosion pressure

**Example**: 
- Water level at Empel: 29-212 cm NAP (range in last 7 days)
- High water events = values significantly above normal range

---

### **Measurement Stations for Study Areas**

Located using `find_nearest_waterweb_stations.py` - searches 18,974 stations for nearest active ones:

| Study Area | Station Code | Distance | River | Status | Notes |
|------------|--------------|----------|-------|--------|-------|
| **Empel** (Maas, Km 218-224) | `shertogenbosch.empel.maas` | 0.0 km | Maas | ✅ Active | Perfect match! Current data |
| **Brakel** (Waal, Km 951-945) | `sintandries.waal` | 17.7 km upstream | Waal | ⚠️ Semi-active | Last data: March 2025 (320 days ago) |
| **Terwolde** (IJssel, Km 947-953) | `zutphen.ijssel` | 15.4 km downstream | IJssel | ✅ Active | Current data, same river |

**Note on Brakel**: Most Waal stations near Brakel (Km 946-951) inactive since 2013. `sintandries.waal` is the nearest with recent data. Alternative stations closer to Brakel exist but are inactive.

---

### **API Endpoints**

**1. Catalog** - Get available parameters:
```
POST /METADATASERVICES/OphalenCatalogus
```

**2. Latest Observations** - Get most recent measurement:
```
POST /ONLINEWAARNEMINGENSERVICES/OphalenLaatsteWaarnemingen
```

**3. Historical Data** - Get time series:
```
POST /ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen
```

---

### **Parameters Available**

**Primary (Tested):**
- `WATHTE`: Water height (cm relative to NAP) - ✅ Working at all study areas
- `Q`: Discharge (m³/day) - ❌ Exists in catalog but NOT available at any study area stations

**Secondary (Available):**
- `GETETBRKD2`: Calculated tidal extremes (high/low water timing)
- Various chemical parameters (not relevant for erosion)

---

### **Example Request (Historical Water Levels)**

```python
import requests

body = {
    "Locatie": {"Code": "shertogenbosch.empel.maas"},
    "AquoPlusWaarnemingMetadata": {
        "AquoMetadata": {
            "Compartiment": {"Code": "OW"},  # Surface water
            "Grootheid": {"Code": "WATHTE"},  # Water height
            "ProcesType": "meting"  # Actual measurements (not predictions)
        }
    },
    "Periode": {
        "Begindatumtijd": "2024-01-01T00:00:00.000+01:00",
        "Einddatumtijd": "2024-12-31T23:59:59.000+01:00"
    }
}

response = requests.post(
    "https://ddapi20-waterwebservices.rijkswaterstaat.nl/ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen",
    json=body,
    headers={"Content-Type": "application/json"}
)
```

---

### **Data Characteristics (From Testing)**

**Empel Station** (7-day test):
- **Measurements**: 1,079 observations
- **Frequency**: Every 10 minutes (144 per day)
- **Range**: 29-212 cm NAP
- **Mean**: 80.9 cm NAP
- **Quality**: Code "00" (high quality, validated)

**Temporal Coverage**:
- Current data: ✅ Real-time (updated every 10 minutes)
- Historical: Available back to 1737 (though quality varies by period)
- **Recommended range**: 2016-2024 for erosion model

---

### **Feature Extraction for Erosion Model**

**Proposed Features** (per prediction region, per year):

1. **High Water Events**:
   - Count of days above 90th percentile
   - Maximum water level reached
   - Duration of high water events (consecutive days above threshold)

2. **Water Level Statistics**:
   - Annual mean water level
   - Standard deviation (variability)
   - Seasonal patterns (winter vs summer highs)

3. **Event Magnitude**:
   - Peak level vs. normal level
   - Rate of water level change (cm/day)

**Aggregation Strategy**:
- Assign each prediction region to nearest station (< 20 km)
- For regions without nearby stations: interpolate from 2-3 nearest stations
- Calculate annual features from 10-minute data

---

### **Integration Status**

✅ **Completed**:
- API access tested and working
- Active stations identified for all 3 study areas
- Historical data retrieval validated (10-min intervals confirmed)
- Test scripts created (`test_waterweb_api.py`, `find_nearest_waterweb_stations.py`)

⏳ **To Do**:
- [x] Test Q (discharge) parameter availability - **NOT available at study locations**
- [ ] Download full 2016-2024 historical data for all study areas
- [ ] Create feature extraction pipeline
- [ ] Map all 233 scope regions to nearest stations
- [ ] Integrate into DataCollector or separate water data handler

---

### **Test Scripts**

**1. `scripts/test_waterweb_api.py`**
- Tests catalog access
- Gets latest observations for all 3 study areas
- Downloads 7-day historical sample
- Calculates basic statistics
- **Status**: ✅ All tests passing

**2. `scripts/find_nearest_waterweb_stations.py`**
- Loads 18,974 station locations from CSV
- Calculates distance to study area centers
- Tests up to 20 nearest stations for each area
- Finds active stations (data within last 30 days)
- **Output**: Recommended station codes for each area

---

### **Limitations & Considerations**

⚠️ **Station Coverage**:
- Not all river sections have active stations
- Some stations inactive since 2013 (especially on Waal)
- May need to use stations 15-20 km away from study area

⚠️ **Data Completeness**:
- Quality varies by time period (older data less reliable)
- Some gaps in historical record
- Need to filter by quality code (recommend: 00, 10, 20, 25, 30, 40)

⚠️ **Discharge (Q) Not Available**:
- Discharge parameter exists in Waterweb but not at our stations
- Discharge measurements are rare (require specialized ADCP equipment)
- Water level (WATHTE) alone is sufficient for erosion modeling
- High water events correlate strongly with erosion pressure

⚠️ **Non-Spatial Data**:
- Water level is point measurement, not area coverage
- Need to aggregate/interpolate to prediction regions
- Assumes water level consistent along ~20 km river sections

✅ **Advantages**:
- Very high temporal resolution (10 minutes!)
- Long historical record (can go back to 2016 or earlier)
- Quality codes provided
- Real-time updates available
- Free and open API

---

### **Next Steps**

1. ✅ Test Q (discharge) parameter - Not available at study locations (not critical)
2. Download 2016-2024 historical data for erosion modeling period
3. Create water level feature extraction module
4. Map all 233 prediction regions to nearest active stations
5. Integrate high water event features into model training

---

## 📒 Notebooks (Priority Order)

### ⭐⭐⭐ **Critical** - Run These First

#### 1. **`demo_baseline_model.ipynb`** - MAIN DEMO
**Uses**: 
- `luke_inputs_v3.gpkg` (Brakel only - April 2025 data)
- `Levering_erosie_data.gpkg` (centerline)
- `erosion_border_20250129.gpkg` (signalling line)

**Output**: Predictions for erosion breach year

**Status**: ✅ WORKING (but uses old data)

**Why old data?**
- Demo was created by Ondrej at start of 2025
- `phase1_2025-08-14_v1.gpkg` (all 3 areas) didn't exist yet

**Next Steps**:
- [ ] Eventually update to use `phase1_2025-08-14_v1.gpkg` for all 3 areas
- [ ] But blocked: Need signalling lines for Empel and Terwolde first
- [ ] For now: Keep as-is for Brakel-only baseline

---

#### 2. **`data_inventory_complete.ipynb`** - DATA REFERENCE
**Purpose**: Documents all data schemas  
**Status**: ✅ NEW - Use as reference

---

### ⭐⭐ **Important** - Validation

#### 3. **`height_sam_comparison_viz.ipynb`** - SAM vs. AHN
**Uses**: `sam/sam_processed.gpkg`, `phase1_2025-08-14_v1.gpkg`  
**Purpose**: Compare SAM vs. AHN-based erosion detection

---

#### 4. **`sam_comparison.ipynb`** - SAM DEEP DIVE
**Uses**: `sam/Brakel.geojsonl`, `phase1_2025-08-14_v1.gpkg`  
**Purpose**: Detailed comparison for Brakel

---

### 🔧 **Utility** - Optional

#### 5-8. Other notebooks (data collector, testing, exploration)

---

## 🔄 Data Timeline (What Happened When)

```
2025-01-08: draft_oever_degradation_data.gpkg (Brakel only, draft)
              ↓
2025-01-21: all_results_20250121_v2.gpkg (Brakel only, v2)
              ↓
2025-01-29: erosion_border_20250129.gpkg (Signalling line drawn for Brakel)
              ↓
2025-04-01: luke_inputs_v3.gpkg (Brakel only, polished)
              ↓
              ├─→ Used by demo_baseline_model.ipynb
              │
2025-07-15: phase1_2025-07-15_v3.gpkg (ALL 3 AREAS - first expansion!)
              ↓
2025-08-15: phase1_2025-08-14_v1.gpkg (ALL 3 AREAS - most complete)
              │
              ├─→ More coverage squares
              ├─→ Better defined vlakken_erosie
              └─→ Has beschermde_oever layer
```

---

## ✅ Decisions Made & Action Items

### **Completed Decisions**:

1. ✅ **Primary Dataset**: `phase1_2025-08-14_v1.gpkg` is now the canonical file
   - Most recent (Aug 2025), all 3 areas, best quality

2. ✅ **Brakel-Only Files**: 
   - Keep: `luke_inputs_v3.gpkg` (used by demo)
   - Archive: `all_results_20250121_v2.gpkg`, `draft_oever_degradation_data.gpkg`

3. ✅ **Delete Files**:
   - `handdrawn_fake_erosion_border.geojson` (useless, superseded)
   - `VO155184_Scope_Pilot_Bankerosion_20241129.shp` (outdated)

4. ✅ **SAM Data Complete**: All 3 geojsonl files ready for Phase 2 hybrid model

---

### **Open Action Items**:

#### 🚨 **URGENT - 2025 Breach Predictions**
- [ ] Verify predictions from `erosion_locations_sam.gpkg`:
  - Point 56, 63, 67 predicted to breach in **2025**
  - Point 67 has **3.24 m/yr** erosion speed (verify if real or outlier)
- [ ] Cross-reference with phase1 vlakken_erosie detections
- [ ] Plan field visit to verify

#### 🗺️ **Create Signalling Lines**
- [ ] Draw signalling lines for **Empel**
- [ ] Draw signalling lines for **Terwolde**
- [ ] Merge with Brakel line into one comprehensive file
- **Blocker**: Cannot run full predictions on Empel/Terwolde without this

#### 🔍 **Questions for Luke Moth**
- [ ] Which centerline is more accurate: phase1 own version vs. Levering_erosie_data?
- [ ] Why version jump from v3 (July) to v1 (Aug)? Was there a v2?
- [ ] Was July phase1 a proof of concept for all-3-areas expansion?

#### 📦 **File Cleanup**
- [ ] Create `data/archive/` folder
- [ ] Move old files:
  - `all_results_20250121_v2.gpkg`
  - `draft_oever_degradation_data.gpkg`
  - `phase1_2025-07-15_v3.gpkg` (if not needed after Luke confirmation)
- [ ] Delete confirmed useless files
- [ ] Fix or delete `water_data/*.csv` (malformed)

---

## 📋 Summary: What You Have

### **Complete & Ready**:
- ✅ **Analysis Data**: `phase1_2025-08-14_v1.gpkg` (all 3 areas, 16.66 MB)
  - 70,495 shore points, 233 prediction regions, 373 erosion polygons
  - Protected shores data (23,497 points)
- ✅ **Reference Data**: `Levering_erosie_data.gpkg` (river context)
- ✅ **SAM Training Data**: Complete for all 3 areas (2,663 features total)
- ✅ **Risk Predictions**: 93 predicted breach locations (some urgent!)
- ✅ **WFS Services**: 3 live data sources (land use, buildings, vegetation)
- ✅ **Working Pipeline**: `demo_baseline_model.ipynb` runs on Brakel data

### **Missing / Blockers**:
- ❌ **Signalling lines for Empel and Terwolde** (only have Brakel)
- ❓ **Protected shore analysis** (need field visit to understand)
- ❓ **Centerline accuracy** (ask Luke which is canonical)

### **Phase 2 Starting This Week**:
- 🔬 **Hybrid Model**: Luke Moth (AHN) + SAM team → Best tool for river bank detection
- 🚶 **Field Visits**: Visit oevers in person to verify predictions
- ⏱️ **Timeline**: 1-2 months for hybrid model

### **Key Files to Focus On**:
1. **`phase1_2025-08-14_v1.gpkg`** - Your primary dataset
2. **`erosion_locations_sam.gpkg`** - Check those 2025 predictions!
3. **`demo_baseline_model.ipynb`** - Run this to understand the pipeline
4. **`data_inventory_complete.ipynb`** - Your data reference guide

---

## 🚀 Immediate Next Steps

### **This Week**:
1. [ ] Run `demo_baseline_model.ipynb` to see the pipeline in action
2. [ ] Investigate **Point 67** in erosion_locations_sam (3.24 m/yr - urgent!)
3. [ ] Ask Luke about centerline accuracy question
4. [ ] Archive old Brakel-only files (cleanup)

### **Next Week**:
5. [ ] Draw signalling lines for Empel and Terwolde (or ask who should do this)
6. [ ] Plan field visit to verify 2025 breach predictions
7. [ ] Update demo notebook to use `phase1_2025-08-14_v1.gpkg` (after signalling lines exist)

### **Month 1**:
8. [ ] Support Luke + SAM team with hybrid model development
9. [ ] Validate predictions against field observations
10. [ ] Plan API endpoints based on what the hybrid model will output
