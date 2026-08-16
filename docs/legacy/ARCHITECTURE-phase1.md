# River Bank Erosion Prediction System - Architecture

## 🏗️ System Architecture

```
                    ┌─────────────────────────┐
                    │      USER INPUTS        │
                    │   • Scope regions       │
                    │   • Time period         │
                    │   • Configuration       │
                    └───────────┬─────────────┘
                                │
                                ▼
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │                  EXTERNAL DATA SOURCES                │
    │                                                       │
    │   ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
    │   │ WFS Services│  │ Water Level │  │Local Files │  │
    │   │             │  │     API     │  │            │  │
    │   │• Land Use   │  │• Waterweb   │  │• Erosion   │  │
    │   │• Buildings  │  │• Discharge  │  │• Centerline│  │
    │   │• Vegetation │  │• Events     │  │• Soil Data │  │
    │   └─────────────┘  └─────────────┘  └────────────┘  │
    │                                                       │
    └───────────────────────────┬───────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   1. DATA COLLECTOR   │
                    │                       │
                    │  📡 Fetch external    │
                    │     geospatial data   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   2. DATA BUNDLER     │
                    │                       │
                    │  📦 Cache data        │
                    │     for reuse         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   3. DATA HANDLER     │
                    │                       │
                    │  ⚙️  Extract features  │
                    │     • Spatial agg     │
                    │     • Geometry calc   │
                    │     • Time series     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   4. BASELINE MODEL   │
                    │                       │
                    │  🤖 Train & Predict   │
                    │     erosion patterns  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      OUTPUTS          │
                    │                       │
                    │  📊 • Predictions     │
                    │     • Maps            │
                    │     • Risk scores     │
                    └───────────────────────┘
```

---

## 🔄 Component Details

```
┌────────────────────────────────────────────────┐
│         1. DATA COLLECTOR                      │
├────────────────────────────────────────────────┤
│  What: Downloads geospatial data               │
│  From: WFS services (PDOK, RWS)               │
│  How:  Query with buffered bounding boxes      │
│                                                │
│  Example Output:                               │
│    Region_123 →                                │
│      • 500 agricultural parcels                │
│      • 50 buildings                            │
│      • 1000 trees                              │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         2. DATA BUNDLER                        │
├────────────────────────────────────────────────┤
│  What: Caches data locally                     │
│  Why:  Avoid re-downloading (faster + offline) │
│  How:  Save as GeoPackages                     │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         3. DATA HANDLER                        │
├────────────────────────────────────────────────┤
│  What: Converts raw data → ML features         │
│  How:  Spatial aggregation + calculations      │
│                                                │
│  Example Transformation:                       │
│    500 parcels → "60% Grasland, 40% Mais"     │
│    Centerline → "Inner bend" classification    │
│    Bank points → "Distance: 32.6m"            │
│                                                │
│  Output: Feature Matrix                        │
│    (location_id, timestamp) → features         │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         4. BASELINE MODEL                      │
├────────────────────────────────────────────────┤
│  What: Predicts future bank positions          │
│  How:  Linear extrapolation from history       │
│                                                │
│  Example Prediction:                           │
│    Location waal_949_0:                        │
│      Now:      32.6m                           │
│      Year 1:   31.7m                           │
│      Year 5:   29.5m                           │
│      Year 10:  28.0m                           │
│    → Slow erosion (4.6m over 10 years)        │
│                                                │
│    Location waal_949_4:                        │
│      Now:      -29.7m                          │
│      Year 10:  -206.3m                         │
│    → Rapid erosion! (176m over 10 years)      │
└────────────────────────────────────────────────┘
```

---

## 📊 Current Status

```
┌─────────────────────────────────────────────────────┐
│              ✅ WORKING TODAY                        │
├─────────────────────────────────────────────────────┤
│  Components:                                        │
│    ✅ DataCollector - Fetch WFS data                │
│    ✅ DataBundler   - Cache data                    │
│    ✅ DataHandler   - Extract features              │
│    ✅ Model         - Predict erosion               │
│    ✅ Visualization - Interactive maps              │
│                                                     │
│  Features in Model:                                 │
│    ✅ Distance to erosion border                    │
│    ✅ Time since last measurement                   │
│                                                     │
│  Features Integrated (not yet in model):            │
│    ✅ Land use (crops)                              │
│    ✅ Buildings                                     │
│    ✅ Inner/outer bend                              │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│         🚧 NEXT PHASE: Priority Features            │
├─────────────────────────────────────────────────────┤
│  HIGH PRIORITY:                                     │
│    🟡 High water events    (data ready)            │
│    🟡 Shore construction   (need to explore)       │
│    🟡 Soil type           (need to process)        │
│                                                     │
│  MEDIUM PRIORITY:                                   │
│    🔴 Vegetation coverage  (pending RWS access)    │
│    🔴 Shipping intensity   (pending data request)  │
└─────────────────────────────────────────────────────┘
```

---

## 💡 Example: 11 Waal River Sections

```
INPUT                  DATACOLLECTOR           DATAHANDLER              MODEL
─────                  ─────────────           ───────────              ─────

11 scope              Query WFS for            Aggregate:              Predict:
polygons          →   each region          →   • 60% Grasland      →   Year 1-10
(waal_949_0                                     • 3 buildings           positions
 through               Download:                • Inner bend
 waal_950_1)           • 500 parcels            • Distance: 32m
                       • 50 buildings
                       • 1000 trees
                                                                       
                       Cache to disk        →   Create matrix:      →   Output:
                                                33 observations         • Stable sections
                                                × 10 features           • Rapid erosion
                                                                         • Risk zones
```

---

## 🎯 Key Strengths

- **Modular**: Components work independently
- **Scalable**: Handles 12,000+ regions
- **Cacheable**: Fast re-runs (no re-downloading)
- **Stakeholder-driven**: Features from asset manager interviews

---

## 📁 Code Structure

```
backend/src/
  ├── data/
  │   ├── data_collector.py    # Component 1
  │   ├── data_handler.py      # Component 3
  │   └── config.py            # Configuration
  └── model/
      └── baseline_model.py    # Component 4

backend/notebooks/
  └── 04_model/demo_baseline_model.ipynb  # Full demo
```
