"""Compare a reproduced hybrid gpkg with Luke's delivery, region by region.

Run: ``uv run python scripts/compare_hybrid.py --ours …/hybrid_repro.gpkg --theirs …/hybrid_model_results_nearest_20260710.gpkg``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyogrio

ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True)
ap.add_argument("--theirs", required=True)
ap.add_argument("--examples", type=int, default=8)
args = ap.parse_args()

NUM = [
    "distinct_dates_height", "distinct_dates_segmentation", "nr_points_height", "nr_points_segmentation",
    "p50_dist_height", "p50_dist_segmentation", "p90_dist_height", "p90_dist_segmentation",
    "diff_p50_height", "diff_p50_segmentation", "diff_p90_height", "diff_p90_segmentation",
    "kpi_slope", "points_height [0-10]", "points_segmentation [0-10]",
]

o = pyogrio.read_dataframe(args.ours, layer="model_preference", read_geometry=False).set_index("location_id")
t = pyogrio.read_dataframe(args.theirs, layer="model_preference", read_geometry=False).set_index("location_id")
both = o.index.intersection(t.index)
print(f"regions: ours {len(o)}, theirs {len(t)}, shared {len(both)}, only ours {len(o.index.difference(t.index))}, "
      f"only theirs {len(t.index.difference(o.index))}")
same = o.loc[both, "model_preference"].eq(t.loc[both, "model_preference"])
print(f"preference agrees: {same.sum()} / {len(both)} ({same.mean():.1%})")
print(pd.crosstab(t.loc[both, "model_preference"], o.loc[both, "model_preference"],
                  rownames=["theirs"], colnames=["ours"]).to_string())

print("\nrationale fields — share of shared regions equal (|Δ| < 1e-6 or both NaN):")
for c in NUM:
    if c not in o.columns or c not in t.columns:
        print(f"  {c}: missing ({'ours' if c not in o.columns else 'theirs'})")
        continue
    a, b = o.loc[both, c].astype(float), t.loc[both, c].astype(float)
    eq = (np.isclose(a, b, atol=1e-6, rtol=1e-9)) | (a.isna() & b.isna())
    print(f"  {c:30s} {eq.mean():6.1%}   median |Δ| {np.nanmedian(np.abs(a - b)):.4g}")

lo = pyogrio.read_dataframe(args.ours, layer="lines")
lt = pyogrio.read_dataframe(args.theirs, layer="lines")
lt = lt[lt["location_id"].isin(both)]
lo = lo[lo["location_id"].isin(both)]


def summary(df):
    df = df.assign(length=df.geometry.length, day=pd.to_datetime(df["date"]).dt.date)
    return df.groupby(["location_id", "model"]).agg(n=("length", "size"), length=("length", "sum"),
                                                   dates=("day", lambda d: tuple(sorted(set(d)))))


so, st = summary(lo), summary(lt)
j = st.join(so, lsuffix="_theirs", rsuffix="_ours", how="outer")
j["n_eq"] = j["n_theirs"].eq(j["n_ours"])
j["len_eq"] = np.isclose(j["length_theirs"].fillna(-1), j["length_ours"].fillna(-1), rtol=1e-6, atol=0.01)
j["dates_eq"] = j["dates_theirs"].astype(str).eq(j["dates_ours"].astype(str))
print("\nwinner lines per (region, model):")
for m, g in j.groupby(level="model"):
    print(f"  {m:12s} groups {len(g):6d} · count equal {g.n_eq.mean():6.1%} · total length equal {g.len_eq.mean():6.1%}"
          f" · dates equal {g.dates_eq.mean():6.1%}")
bad = j[~(j.n_eq & j.len_eq & j.dates_eq)].head(args.examples)
if len(bad):
    print("\nexamples of mismatching (region, model):")
    print(bad[["n_theirs", "n_ours", "length_theirs", "length_ours"]].round(1).to_string())
diff = o.loc[both][~same].head(args.examples)
if len(diff):
    print("\nexamples of preference mismatches (ours | theirs):")
    for loc in diff.index:
        print(f"  {loc}: ours {o.at[loc, 'model_preference']} {o.loc[loc, ['points_height [0-10]', 'points_segmentation [0-10]']].tolist()}"
              f" | theirs {t.at[loc, 'model_preference']} {t.loc[loc, ['points_height [0-10]', 'points_segmentation [0-10]']].tolist()}")
