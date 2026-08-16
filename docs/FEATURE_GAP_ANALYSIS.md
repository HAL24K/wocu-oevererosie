# Feature Gap Analysis

Both CSVs referenced here live alongside this file in `docs/`.

Comparison of **metrics currently in the explore_features notebook** vs **features listed in `feature_requirements_matrix.csv`** and **`data_sources_inventory.csv`** — to identify potentially interesting metrics not yet taken along.

---

## Currently in the notebook

| Feature | Source | Status |
|---------|--------|--------|
| v_train, v_test | region_split | ✓ |
| dist_t2 | region_split | ✓ |
| is_nvo | region_split | ✓ |
| river | parsed from location_id | ✓ |
| vegetation_class | vegetatielegger.gpkg | ✓ (permitted, not actual) |
| land_use | land_use.gpkg (BRP) | ✓ |
| soil_group | BRO Bodemkaart | ✓ |
| peak_p99 | discharge (per station/year) | ✓ |
| days_above_p90 (flood_days) | discharge (per station/year) | ✓ |
| spi, n_events, max_rise_rate, drawdown_index | high-water metrics (per station/year) | ✓ |

---

## Not yet in the notebook — potentially interesting

### High priority (feature_requirements_matrix)

| Feature | Description | Data source | Status / blocker |
|---------|-------------|-------------|------------------|
| **Bend geometry** | Inner vs outer bend, curvature, position in bend. "Outer bend = erosion, inner = deposition." | River centerline (middenlijn) | **Available, "already integrated" in DataHandler** — but not in explore_features. Could add inner/outer bend classification. |
| **Shore construction type** | Protected vs NVO vs unprotected; NVO tranche (1st/2nd phase); construction date | RWS Legger: natuurvriendelijke_oever_vlak_legger, oeverconstructie_verticaal_legger | **Available** — is_nvo is partial; need NVO tranche, construction dates. |
| **Dynamism index** | Bank stability/variability over time (not just linear rate). IJssel: "volume van oevererosie zegt iets over dynamiek." | Erosion volume time series, water level variability | **Derivable** — could compute from erosion_vol_train, erosion_vol_test, or coefficient of variation. |
| **Erosion trend direction** | Accelerating, stable, decelerating, approaching equilibrium | Time-series erosion velocity | **Derivable** — needs sufficient historical points; could use v_train vs v_test slope. |
| **Risk classification** | Combines erosion velocity + proximity to infrastructure + protection status | Multiple | **Output** — for inspection scheduling; depends on other features. |

### Medium priority

| Feature | Description | Data source | Status / blocker |
|---------|-------------|-------------|------------------|
| **Shipping intensity** | Ship traffic volume → wave action. Maas + Rijntakken. | AIS data via RWS (Rik van Neer) | **Pending access** — request in progress. |
| **Actual vegetation coverage** | Vegetation Legger shows permitted; Vegetatiemonitor shows actual. | Vegetatiemonitor (RWS) | **Pending access** — needs RWS permission. |
| **Buildings / built environment** | Building footprints, usage type (gebruiksdoel) | BAG WFS | **In DataCollector** — but not in explore_features. Could add as majority class + area fraction. |

### Lower priority / enhancements

| Feature | Description | Notes |
|---------|-------------|-------|
| **BKN (Beheerkaart Natuur)** | Nature management zones | Complements RWS Legger; available. |
| **Water level** (vs discharge only) | WATHTE parameter from Waterweb | Water levels used at some stations; discharge is primary. |
| **Radius of curvature** | Bend tightness | Enhancement to bend geometry. |
| **Distance to bend apex** | Position within meander | Enhancement to bend geometry. |

---

## Summary: quick wins vs blocked

| Category | Features | Action |
|----------|----------|--------|
| **Quick wins** (data available) | Bend geometry (inner/outer), Buildings (BAG), Shore construction details (RWS Legger) | Add to explore_features; join from existing DataCollector or WFS. |
| **Derivable** | Dynamism index, Erosion trend direction | Compute from existing erosion data; may need more temporal points. |
| **Blocked** | Shipping intensity, Vegetatiemonitor | Await RWS access. |

---

## Overlap between the two CSVs

- **feature_requirements_matrix.csv** — feature-centric: what asset managers want, priority, data source.
- **data_sources_inventory.csv** — data-centric: what raw data exists, status, how to extract.

They describe the same features from different angles. The gap analysis above merges both views.
