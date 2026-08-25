"""Secondary-water mask: mask samples on water bodies that are not a channel.

SAM traces land/water edges faithfully — it just doesn't know which water we
care about. Water polygons (vegetatielegger, klasse 'Water') that intersect
no region centreline are marinas, floodplain pools, ditches: their edges are
genuine waterlines but not riverbanks. Those polygons, buffered, join the
kribben in the structures mask. Clips samples, never lines or regions —
the main-channel part of a survey that also traces a pool now survives.

Runs the e9 variant (e8 rules, kribben+water structures) and reports.
"""

import logging
import warnings

import geopandas as gpd
import pandas as pd
import shapely

from experiments.loop.harness.harness import load_caches, run_variant
from experiments.loop.harness.multi_t import E8_RULES

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

BUFFER_M = 10.0

caches = load_caches()
data_dir = caches.cache_dir.parents[1]

veg = gpd.read_file(
    data_dir / "02_processed/wfs_context/vegetatielegger.gpkg",
    layer="rws_vegetatielegger:vegetatieklassen",
).to_crs(28992)
water = veg[veg["vlklasse"] == "Water"][["geometry"]].explode(index_parts=False)
water["geometry"] = shapely.force_2d(water.geometry.values)

# channel = any water part that touches a region centreline (side channels
# with their own scope regions count as channels for their own banks) or the
# main river centrelines from the delivery.
clines = gpd.read_file(
    data_dir / "01_raw/erosion/wocu_output_fase2_20260210.gpkg", layer="centrelines"
).to_crs(28992)
main = gpd.read_file(
    data_dir / "01_raw/scope/Levering_erosie_data.gpkg", layer="Centreline_River"
).to_crs(28992)
channel_lines = pd.concat([clines[["geometry"]], main[["geometry"]]], ignore_index=True)

hit = gpd.sjoin(
    water.reset_index(drop=True),
    gpd.GeoDataFrame(geometry=channel_lines.geometry.buffer(2.0), crs=28992),
    predicate="intersects",
    how="left",
)
is_channel = hit.groupby(level=0)["index_right"].first().notna()
mask_polys = water.reset_index(drop=True)[~is_channel.values]
print(
    f"water parts: {len(water)} total · {int(is_channel.sum())} channel (kept) · "
    f"{len(mask_polys)} secondary (masked) · "
    f"{mask_polys.area.sum() / 1e6:.1f} km2 masked area"
)

out_gpkg = data_dir / "02_processed/triage/secondary_water_mask.gpkg"
mask_polys.to_file(out_gpkg, layer="secondary_water", driver="GPKG")
print(f"mask geometry → {out_gpkg} (QGIS-inspectable)")

kribs = gpd.read_file(
    data_dir / "01_raw/scope/Levering_erosie_data.gpkg", layer="Kribben_BKN"
).to_crs(28992)
combined = shapely.union_all(
    [
        shapely.union_all(kribs.geometry.buffer(BUFFER_M).values),
        shapely.union_all(mask_polys.geometry.buffer(BUFFER_M).values),
    ]
)
shapely.prepare(combined)

run_variant(
    "e9-water-mask",
    caches,
    E8_RULES,
    structures=combined,
    v_limit=50.0,
    notes="e8 + secondary-water mask (vegetatielegger Water, no-centreline parts)",
)
