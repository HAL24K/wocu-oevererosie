# WOCU River Bank Erosion Prediction

Geospatial ML application for predicting river bank erosion in the Netherlands. Combines research-grade data processing with a production-ready FastAPI backend.

**Stack**: Python 3.12, FastAPI, GeoPandas, PyTorch | React + Leaflet (planned)

---

## Quick Start

```bash
# Setup
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
uv sync --all-extras  # Install all dependencies including notebooks

# Register Jupyter kernel
python -m ipykernel install --user --name=wocu-erosion --display-name="Python 3.12 (wocu-erosion)"

# Run notebooks
jupyter lab  # Select "Python 3.12 (wocu-erosion)" kernel

# Run API
python -m app.main
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs

# Run tests
pytest

# Code quality
ruff check . && ruff format .
```

---

## Repository Structure

```
wocu-oevererosie/
├── backend/
│   ├── src/                      # 🔬 Research/Domain Layer
│   │   ├── data/                 # DataHandler, DataCollector, feature generation
│   │   ├── model/                # Baseline, deep learning models
│   │   ├── config.py             # WFS service configurations
│   │   ├── constants.py          # Project constants
│   │   └── utils.py              # Geospatial utilities
│   │
│   ├── app/                      # 🌐 API/Application Layer
│   │   ├── core/                 # Config, dependencies
│   │   ├── api/routes/           # API endpoints
│   │   ├── services/             # Business logic orchestration
│   │   ├── schemas/              # Pydantic models
│   │   └── main.py               # FastAPI entry point
│   │
│   ├── data/                     # Local data files (git-ignored)
│   ├── notebooks/                # Jupyter exploration
│   ├── tests/                    # Unit + integration tests
│   ├── pyproject.toml            # Dependencies (uv)
│   └── .venv/                    # Virtual environment
│
├── docs/
│   └── DATA_SOURCES.md           # Data inventory and schemas
│
└── README.md                     # This file
```

### Key Directories

**`backend/src/`** - Research/domain code. Pure Python (no web dependencies), used in notebooks and API.

**`backend/app/`** - FastAPI application. HTTP layer that orchestrates `src/` classes. Pattern: Router → Service → Domain.

**`backend/data/`** - GeoPackage (`.gpkg`), GeoJSON files. See `docs/DATA_SOURCES.md` for inventory.

**`backend/notebooks/`** - Jupyter exploration. Clear output before committing.

---

## Architecture

### Core Philosophy

This project transforms research code into a production-ready application while maintaining clean separation between domain logic and API concerns.

**Key Principles**:
- Separate research code (`backend/src/`) from API (`backend/app/`)
- Data-driven development
- Proper CRS handling (RD New ↔ WGS84)
- Start simple, iterate

### Router → Service → Domain Pattern

```
HTTP Request
    ↓
1. Router (app/api/routes/)
   - Validates request, handles HTTP
    ↓
2. Service (app/services/)
   - Orchestrates workflow, transforms data
    ↓
3. Domain (backend/src/)
   - Core logic: DataHandler, DataCollector, Models
    ↓
External APIs (WFS) / File System / Models
```

#### Layer 1: Routers (Controllers)

Handle HTTP requests/responses. Validate input, delegate to services.

```python
# app/api/routes/predictions.py
@router.post("/erosion", response_model=PredictionResponse)
async def predict_erosion(request: PredictionRequest, settings: SettingsDep):
    """Predict river bank erosion for given regions."""
    try:
        service = PredictionService(settings)
        return await service.predict_erosion(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

✅ Validate, delegate, return | ❌ No business logic or file access

#### Layer 2: Services (Business Logic)

Orchestrate workflows, transform data between API and domain layers.

```python
# app/services/prediction_service.py
from src.data.data_handler import DataHandler
from src.model.baseline_model import BaselineErosionModel

class PredictionService:
    async def predict_erosion(self, request: PredictionRequest):
        # 1. Transform WGS84 → RD New
        regions = self._to_geodataframe(request)
        
        # 2. Use domain layer (src/)
        handler = DataHandler(config=..., prediction_regions=regions)
        handler.create_data_from_remote()  # Fetch WFS
        handler.process_erosion_features()
        
        # 3. Load model and predict
        model = BaselineErosionModel.load_model(...)
        predictions = model.predict(...)
        
        # 4. Transform RD New → WGS84
        return self._to_response(predictions)
```

✅ Orchestrate, transform CRS, call `src/` | ❌ No HTTP objects

#### Layer 3: Domain (Research Code)

Core logic in `backend/src/`. Pure Python, no web dependencies. Key classes:

- **DataHandler** (`src/data/data_handler.py`) - Enriches geospatial data, generates ML features
- **DataCollector** (`src/data/data_collector.py`) - Fetches from WFS services (PDOK, RWS)
- **BaselineErosionModel** (`src/model/baseline_model.py`) - Mean erosion speed predictions

The `DataHandler` is the heart of feature engineering. It:
1. Accepts prediction regions (polygons) and configuration
2. Fetches external WFS data (land use, buildings, vegetation)
3. Calculates distances from river bank points to erosion border
4. Generates lagged features for time series modeling
5. Outputs features ready for ML models

✅ Core logic, type hints, extensive logging | ❌ No web framework imports

---

## Geospatial Conventions

### Coordinate Reference Systems (CRS)

**RD New (EPSG:28992)** - Domain layer. Dutch grid, meters, accurate for NL.  
**WGS84 (EPSG:4326)** - API/frontend. Lat/lon degrees, web maps standard.

```python
# API → Domain
gdf = gpd.GeoDataFrame.from_features(geojson, crs=4326).to_crs(28992)

# Domain → API  
gdf_api = gdf.to_crs(4326)
```

✅ Always specify CRS, transform at boundaries | ❌ Never mix CRS

### GeoJSON & File Formats

**API**: Return `FeatureCollection` with properties, [lon, lat] order (WGS84).

**Files**: Prefer GeoPackage (`.gpkg`, multi-layer) > GeoJSON (`.geojson`, single-layer) > Shapefile (legacy).

```python
# Reading
layers = gpd.list_layers("data.gpkg")
gdf = gpd.read_file("data.gpkg", layer="layer_name")
```

### WFS Services

**Configured in `src/config.py`:**
- PDOK BRP (land use/crops) - `https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0`
- PDOK BAG (buildings) - `https://service.pdok.nl/lv/bag/wfs/v2_0`
- RWS Vegetation (trees/vegetation) - `https://geo.rijkswaterstaat.nl/services/ogc/gdr/rws_vegetatielegger/ows`

```python
from src.data.data_collector import DataCollector
import src.config as CONFIG

collector = DataCollector(
    source_shape=polygon, source_epsg_crs=28992,
    buffer_in_metres=100, wfs_services=CONFIG.KNOWN_WFS_SERVICES
)
collector.get_data_from_all_wfs()  # → {service: {layer: GeoDataFrame}}
```

See `docs/DATA_SOURCES.md` for detailed layer schemas and field descriptions.

---

## Development

### Code Quality

**Ruff** for linting/formatting. **Type hints** for all functions. **Extensive logging** with context.

```bash
# Before committing
ruff check . && ruff format .
pytest
```

**Naming conventions**:
- `snake_case` - functions, variables
- `PascalCase` - classes
- `UPPER_SNAKE_CASE` - constants
- `_private` - internal methods

**Error messages**: Specific and actionable.
```python
raise ValueError(f"Unknown fruit: '{fruit}'. Pick one of: {', '.join(KNOWN_FRUITS)}")
```

**Logging**: Contextual and informative.
```python
logger.info(f"Fetching WFS for {len(regions)} regions with {buffer_m}m buffer")
logger.warning(f"Unknown categories {cats} in '{col}'. Mapped to {DEFAULT}")
```

**Docstrings**: Google style for complex functions, brief for simple ones.

### Git Workflow

**Branches**: `feature/`, `fix/`, `docs/`, `refactor/`

**Commits**: `type: description` where type = feat, fix, docs, refactor, test, chore

```bash
git commit -m "feat: add erosion prediction endpoint

- Created PredictionService, added POST /api/v1/predictions/erosion
- Transforms WGS84 ↔ RD New"
```

**PR**: Feature branch → review → merge. Include description, testing notes.

**Review checklist**: Ruff passes, type hints, tests, docs, no hardcoded values, clear errors, CRS handled.

### Testing

**Structure**: `tests/unit/` (domain), `tests/integration/` (API), `tests/assets/` (small test data)

```python
# Unit test example
@pytest.fixture
def sample_regions():
    return gpd.GeoDataFrame({"region_id": ["test_1"], ...}, crs="EPSG:28992")

def test_data_handler_init(sample_regions):
    handler = DataHandler(config=..., prediction_regions=sample_regions)
    assert len(handler.prediction_regions) == 1

# API test example
from fastapi.testclient import TestClient
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
```

**Run**: `pytest`, `pytest --cov=src --cov=app`, `pytest -n auto` (parallel)

✅ Test domain logic, edge cases, CRS transforms | ❌ No large files, don't test libraries

### Package Management

**Tool**: `uv` (fast, Rust-based). Config in `backend/pyproject.toml`.

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate

uv sync              # Install deps from pyproject.toml
uv sync --all-extras # Include dev deps (pytest, jupyterlab, ruff)
uv add package-name  # Add new dependency
uv lock              # Update uv.lock
```

✅ Pin major versions, commit `uv.lock`, use venv | ❌ Don't commit `.venv/`

---

## Good Coding Practices

Follow these principles consistently:

- **DRY** ("Don't Repeat Yourself") - Refactor repeated code into functions
- **No hardcoding** - Use `src/config.py` and `src/constants.py`
- **Descriptive names** - `default_river_depth_near_sea` > `drdns`
- **Automated checks** - Run `ruff` and `pytest` before commits
- **No secrets** - Never commit passwords or API keys
- **Feature branches** - Always branch, test, and get review before merging
- **Verbose errors** - Guide users to solutions:
  - ❌ `Unknown input.`
  - ✅ `Unknown fruit: 'dorian'. Please pick one of: 'apple', 'banana', 'cherry'.`
- **Contextual logging** - Be verbose and reassuring:
  - `Number of data samples: 1000; features: 900. We expect some samples to be dropped due to missing data.`
- **Clear notebook output** - Always clear cell output before committing

---

## Documentation

**Key docs**:
- `README.md` - This file (overview, architecture, practices)
- `docs/DATA_SOURCES.md` - Comprehensive data glossary and schemas
- `/docs` - FastAPI auto-generated API documentation (when server running)

**Code documentation**:
- Module/class/function docstrings
- Inline comments for WHY, not WHAT
- `# TODO:` for future work

**Notebooks**: Clear output before committing, use markdown cells for insights.

---

## Quick Reference

```bash
# Setup
cd backend && source .venv/bin/activate
uv sync --all-extras

# Run API
python -m app.main  # or: uvicorn app.main:app --reload

# Code quality
ruff check . && ruff format .
pytest

# CRS transforms
EPSG:28992 (RD New, meters) ↔ EPSG:4326 (WGS84, lat/lon)
gdf.to_crs(4326)  # Transform to WGS84
```

---

## Project Status

Currently transforming research code into full-stack application:
- ✅ Backend FastAPI bootstrap (health check, config, deps)
- ✅ Python 3.12 environment with `uv`
- ✅ Research code preserved in `backend/src/`
- ✅ Architecture documentation
- 🚧 Data exploration and glossary (`docs/DATA_SOURCES.md`)
- 🚧 API endpoints for predictions and data management
- 📅 Frontend (React + Leaflet)
- 📅 Docker Compose orchestration

See `docs/DATA_SOURCES.md` for current data inventory and WFS service details.
