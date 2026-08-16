# Implementation Plan: Switch to Centerline-Based Erosion Reference

## Overview

Switch the baseline erosion model from using `erosion_border` to `centerline` as the reference line for distance calculations, while maintaining backward compatibility with the legacy approach.

## Motivation

- **Generality**: Centerlines are available for all 12K regions, erosion_border only for demo subset
- **Mathematical equivalence**: Velocity = Δdistance / Δtime works with any reference line
- **Cleaner interpretation**: River banks naturally defined relative to river center

## Implementation Phases

### ✅ Phase 0: Validation (Current Step)
**Notebook**: `notebooks/04_model/validate_centerline_approach.ipynb`

**Goals:**
- Examine centerline geometry structure (LineString vs MultiLineString)
- Test distance calculation methods
- Compare velocity calculations between approaches
- Verify mathematical equivalence

**Run this notebook first to confirm assumptions!**

---

### Phase 1: Update Infrastructure (No Breaking Changes)

#### 1.1 Update Constants
**File**: `src/constants.py`

```python
# Add new constants (keep existing ones!)
DISTANCE_TO_CENTERLINE = "distance_to_centerline"
REFERENCE_LINE_TYPE = "reference_line_type"  # Metadata for tracking which approach used

# Update default columns (optional - can keep erosion_border as default for now)
DEFAULT_UNKNOWN_NUMERICAL_COLUMNS = [
    DISTANCE_TO_EROSION_BORDER,  # Legacy
    # DISTANCE_TO_CENTERLINE,  # Uncomment when ready to switch default
]
```

**Testing**: Run existing tests to ensure no breakage

---

#### 1.2 Add New DataHandler Method
**File**: `src/data/data_handler.py`

**Add new method** (alongside existing erosion_border method):

```python
def calculate_river_bank_distances_to_centerline(
    self,
    bank_points: gpd.GeoDataFrame,
    centerlines: gpd.GeoDataFrame,
    distance_column: str = CONST.DISTANCE_TO_CENTERLINE,
) -> pd.Series:
    """
    Calculate perpendicular distance from bank points to river centerline.
    
    Similar to erosion_border method but uses centerline as reference.
    
    Args:
        bank_points: GeoDataFrame with river bank point geometries
        centerlines: GeoDataFrame with river centerline geometries
        distance_column: Name for the output distance column
        
    Returns:
        Series with distances from each bank point to nearest centerline
        
    Notes:
        - Uses absolute distance (no sign)
        - Positive velocity = erosion (moving away from river)
        - Negative velocity = accretion (moving toward river)
    """
    # TODO: Implement based on validation notebook findings
    # Options:
    #   1. If centerlines is single LineString: use centerlines.geometry.iloc[0]
    #   2. If multiple centerlines: use spatial join to find nearest
    #   3. If centerlines have location_id: match by ID first, then distance
    
    pass  # Implementation after validation
```

**Keep existing method unchanged**:
```python
def calculate_river_bank_distances_to_erosion_border(self, ...):
    """Legacy method - keep as-is for demo notebook"""
    # Existing implementation stays unchanged
    pass
```

---

#### 1.3 Make BaselineModel Flexible
**File**: `src/model/baseline_model.py`

**Modify class initialization**:

```python
class BaselineErosionModel:
    def __init__(self, distance_column: str = CONST.DISTANCE_TO_EROSION_BORDER):
        """
        Physics-based linear extrapolation model for erosion prediction.
        
        Args:
            distance_column: Which distance metric to use for predictions.
                           Options:
                           - CONST.DISTANCE_TO_EROSION_BORDER (legacy, default)
                           - CONST.DISTANCE_TO_CENTERLINE (new approach)
        """
        self.distance_column = distance_column
        self.velocities = None
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Calculate erosion velocities from historical data.
        
        Changes:
        - Replace hardcoded CONST.DISTANCE_TO_EROSION_BORDER 
        - Use self.distance_column instead
        """
        # Line 76-78: Update to use self.distance_column
        erosion_step_size = (
            region_data_simple_index.iloc[i + 1][self.distance_column]
            - region_data_simple_index.iloc[i][self.distance_column]
        )
        # ... rest of logic unchanged
        
    def predict(self, data: pd.DataFrame, prediction_length: int = 1) -> dict:
        """
        Predict future bank positions using linear extrapolation.
        
        Changes:
        - Line 136-139: Update to use self.distance_column
        """
        for prediction_step in range(1, prediction_length + 1):
            predicted_data[
                f"future_{self.distance_column}_{prediction_step}"
            ] = data[self.distance_column] + prediction_step * data.index.get_level_values(
                CONST.PREDICTION_REGION_ID
            ).map(self.velocities)
        # ... rest unchanged
```

**Testing**: 
- Test with `distance_column=CONST.DISTANCE_TO_EROSION_BORDER` (should work as before)
- Test with `distance_column=CONST.DISTANCE_TO_CENTERLINE` (after Phase 2)

---

### Phase 2: Implement Centerline Distance Calculation

#### 2.1 Complete DataHandler Method
**File**: `src/data/data_handler.py`

**Based on validation notebook findings**, implement one of these approaches:

**Option A: Single centerline for entire river**
```python
def calculate_river_bank_distances_to_centerline(
    self, bank_points: gpd.GeoDataFrame, centerlines: gpd.GeoDataFrame
) -> pd.Series:
    # Use first/only centerline as reference
    reference_line = centerlines.geometry.iloc[0]
    distances = bank_points.geometry.distance(reference_line)
    return distances
```

**Option B: Multiple centerlines, find nearest**
```python
def calculate_river_bank_distances_to_centerline(
    self, bank_points: gpd.GeoDataFrame, centerlines: gpd.GeoDataFrame
) -> pd.Series:
    # For each point, find distance to nearest centerline
    distances = bank_points.geometry.apply(
        lambda pt: centerlines.geometry.distance(pt).min()
    )
    return distances
```

**Option C: Match by location_id** (if available)
```python
def calculate_river_bank_distances_to_centerline(
    self, bank_points: gpd.GeoDataFrame, centerlines: gpd.GeoDataFrame
) -> pd.Series:
    # Join on location_id, then calculate distance
    # ... implementation depends on data structure
    pass
```

---

#### 2.2 Update process_erosion_features()
**File**: `src/data/data_handler.py`

```python
def process_erosion_features(
    self,
    reference_line_type: str = "centerline",  # New default
    centerlines: gpd.GeoDataFrame = None,     # New parameter
) -> pd.DataFrame:
    """
    Process erosion features from raw bank point data.
    
    Args:
        reference_line_type: 'centerline' (default) or 'erosion_border' (legacy)
        centerlines: Required if reference_line_type='centerline'
        
    Returns:
        DataFrame with processed erosion features including distance measurements
    """
    if reference_line_type == "centerline":
        if centerlines is None:
            raise ValueError("centerlines parameter required when using centerline approach")
        # Use new centerline method
        erosion_data = self.calculate_river_bank_distances_to_centerline(
            bank_points=self.raw_erosion_data,
            centerlines=centerlines
        )
    elif reference_line_type == "erosion_border":
        if self.erosion_border is None:
            raise ValueError("erosion_border must be provided in __init__ for legacy approach")
        # Use legacy erosion_border method
        erosion_data = self.calculate_river_bank_distances_to_erosion_border(...)
    else:
        raise ValueError(f"Unknown reference_line_type: {reference_line_type}")
    
    # Rest of processing logic unchanged
    # ...
```

---

### Phase 3: Update Notebooks

#### 3.1 Keep Demo Notebook Unchanged
**File**: `notebooks/04_model/demo_baseline_model.ipynb`

**Action**: Add markdown note at top:

```markdown
## ⚠️ Note: Legacy Approach

This notebook uses the **legacy erosion_border approach** for historical compatibility.

The erosion_border is a fixed reference line used in early development.
For the full 12K dataset, see `baseline_model_full_dataset.ipynb` which uses
the more general **centerline approach**.
```

**No code changes needed** - demo continues working as-is!

---

#### 3.2 Update Full Dataset Notebook
**File**: `notebooks/04_model/baseline_model_full_dataset.ipynb`

**Changes needed:**

```python
# In data loading cell - add centerlines
centerlines = gpd.read_file(GPKG_PATH, layer="centrelines")

# In data processing cell - update to use centerline
from src.data.data_handler import DataHandler

# Create data handler (no erosion_border needed!)
data_handler = DataHandler(
    config=config,
    prediction_regions=scope_regions,
    erosion_data=bank_points,
    # erosion_border=None,  # Not needed for centerline approach
)

# Process with centerline
processed_data = data_handler.process_erosion_features(
    reference_line_type="centerline",
    centerlines=centerlines
)

# Initialize model with centerline distance column
baseline_model = BaselineErosionModel(
    distance_column=CONST.DISTANCE_TO_CENTERLINE
)

# Rest of notebook logic stays the same!
```

---

### Phase 4: Testing & Validation

#### 4.1 Unit Tests
**File**: `tests/test_baseline_model.py`

Add new tests:

```python
def test_baseline_model_with_centerline():
    """Test new centerline-based approach"""
    # Create mock data with centerline
    # Test distance calculation
    # Test velocity calculation
    # Test prediction
    pass

def test_baseline_model_with_erosion_border():
    """Test legacy erosion_border approach still works"""
    # Existing test logic
    pass

def test_baseline_model_both_approaches_consistent():
    """Verify both approaches give similar velocity trends"""
    # Run both approaches on same data
    # Compare velocity calculations
    # Should have similar trends (different absolutes OK)
    pass
```

#### 4.2 Integration Test
**File**: `tests/test_data_handler.py`

```python
def test_calculate_distances_to_centerline():
    """Test centerline distance calculation"""
    pass

def test_process_erosion_features_centerline_mode():
    """Test full pipeline with centerline"""
    pass

def test_process_erosion_features_erosion_border_mode():
    """Test full pipeline with erosion_border (legacy)"""
    pass
```

---

### Phase 5: Visualization Updates (Optional)

#### 5.1 Generalize Animation Functions
**File**: Update notebooks with animation code

**Make functions work with either reference line:**

```python
def project_scope_onto_reference_line(
    scope_polygons: gpd.GeoDataFrame,
    reference_line: LineString,  # Can be centerline or erosion_border
    reference_line_name: str = "centerline"
) -> gpd.GeoDataFrame:
    """
    Project scope regions onto reference line (centerline or erosion_border).
    """
    # Same logic, just more generic naming
    pass

def move_projected_scope_perpendicular(
    projected_scope: gpd.GeoDataFrame,
    distance: float,
    reference_line: LineString
) -> gpd.GeoDataFrame:
    """
    Move projected scope perpendicular to reference line by given distance.
    
    Works with curved lines (centerlines) using Shapely's offset_curve.
    """
    # Handle both straight and curved lines
    pass
```

---

## Implementation Order

1. ✅ **Start**: Run validation notebook to confirm assumptions
2. **Phase 1**: Update constants, add flexible model parameter (no breaking changes)
3. **Phase 2**: Implement centerline distance calculation based on validation findings
4. **Phase 3**: Update full dataset notebook to use centerline
5. **Phase 4**: Add tests
6. **Phase 5**: Update visualizations (optional)

## Success Criteria

- [ ] Validation notebook confirms centerline approach is viable
- [ ] Demo notebook still works with erosion_border (backward compatibility)
- [ ] Full dataset notebook works with centerline approach
- [ ] Both approaches produce similar velocity trends on same data
- [ ] All tests pass
- [ ] Code is well-documented

## Notes

- **No breaking changes**: Legacy code continues to work
- **Gradual migration**: Demo uses old approach, full dataset uses new approach
- **Flexibility**: Model can use either distance metric via parameter
- **Extensibility**: Easy to add other reference line types in future (e.g., alert boundaries)
