"""Deck experiment C — regions whose dist_per_year changes most between
s1-e8-nomask and s3-e8-newstructures (VVR regions only). Writes
docs/presentations/fig/candidates/structures_cleanup_candidates.csv."""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from experiments.loop.harness.harness import load_caches
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
caches = load_caches()
data = caches.cache_dir.parents[1]
vd = caches.cache_dir / "variants"

a = pd.read_parquet(vd / "s1-e8-nomask/dist_per_year.parquet")
c = pd.read_parquet(vd / "s3-e8-newstructures/dist_per_year.parquet")
print("dpy columns:", list(a.columns))
dcol = [x for x in a.columns if x.startswith("dist")][0]
ycol = "year" if "year" in a.columns else [x for x in a.columns if "year" in x][0]


def vel(d: pd.DataFrame) -> pd.DataFrame:
    """Per region: n_years, v over last 2 years (t2→t3), mean |Δ| position."""
    out = {}
    for lid, g in d.sort_values(ycol).groupby(LOCATION_ID):
        y, x = g[ycol].values, g[dcol].values
        if len(y) < 2:
            continue
        span = y[-1] - y[-2]
        out[lid] = {
            "n_years": len(y),
            "v_last": (x[-1] - x[-2]) / span if span else np.nan,
            "dist_last": x[-1],
        }
    return pd.DataFrame(out).T


va, vc = vel(a), vel(c)
both = va.join(vc, lsuffix="_no_mask", rsuffix="_new_mask", how="inner")
# region-year position difference between the two variants
m = a.merge(c, on=[LOCATION_ID, ycol], suffixes=("_a", "_c"))
m["absdiff"] = (m[f"{dcol}_a"] - m[f"{dcol}_c"]).abs()
pos = (
    m.groupby(LOCATION_ID)["absdiff"]
    .agg(["mean", "max"])
    .rename(columns={"mean": "mean_pos_diff_m", "max": "max_pos_diff_m"})
)
both = both.join(pos, how="inner")
both["dv"] = (both["v_last_no_mask"] - both["v_last_new_mask"]).abs()
both["sign_flip"] = np.sign(both["v_last_no_mask"]) != np.sign(both["v_last_new_mask"])
both["is_nvo"] = caches.static["is_nvo"].reindex(both.index).astype(bool)
both["river"] = [i.rsplit("_", 3)[0] for i in both.index]
both = both[both["is_nvo"]]
# lost regions (present without mask, gone with mask) are interesting too but unplottable — skip
both["score"] = both["dv"] + 0.2 * both["mean_pos_diff_m"] + 3 * both["sign_flip"]
top = both.sort_values("score", ascending=False).head(20)

# structures nearby: count new-layer objects within 60 m of the region's samples hull
sg = data / "02_processed/structures/structures.gpkg"
kr = gpd.read_file(sg, layer="kribben")
kw = gpd.read_file(sg, layer="kunstwerken")
kw = kw[kw.categorie.isin(["brug", "kade_damwand", "steiger_afmeer", "sluis_stuw"])]
allst = pd.concat([kr[["geometry"]], kw[["geometry"]]], ignore_index=True)
s = caches.samples[caches.samples[LOCATION_ID].isin(top.index)]
hulls = {
    lid: shapely.convex_hull(shapely.multipoints(g[["x", "y"]].values)).buffer(60)
    for lid, g in s.groupby(LOCATION_ID)
}
top["n_structures_nearby"] = [
    int(allst.intersects(hulls[l]).sum()) if l in hulls else 0 for l in top.index
]

out = (
    top.reset_index()
    .rename(columns={"index": LOCATION_ID, "n_years_no_mask": "n_years"})[
        [
            LOCATION_ID,
            "river",
            "v_last_no_mask",
            "v_last_new_mask",
            "n_years",
            "n_structures_nearby",
            "mean_pos_diff_m",
            "max_pos_diff_m",
            "sign_flip",
        ]
    ]
    .rename(columns={"v_last_no_mask": "v_no_mask", "v_last_new_mask": "v_new_mask"})
)
dst = Path("docs/presentations/fig/candidates")
dst.mkdir(parents=True, exist_ok=True)
out.round(2).to_csv(dst / "structures_cleanup_candidates.csv", index=False)
print(out.round(2).to_string())
lost = set(va.index) - set(vc.index)
print(f"\nregions present without mask but dropped with new mask: {len(lost)}")
