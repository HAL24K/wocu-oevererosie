"""Region data for the interactive figures (method artifact, deck's live explorer).

Four real regions of a run, in local metres: scope polygon, centreline, signal
line and VVR, delivered lines (kept or removed by cleaning), kept samples,
strip forecasts and test rows, facts, and a PDOK aerial photo as background.
Writes ``<run>/figures/methodology_data.json``.

Run: ``PYTHONPATH=. uv run python scripts/slides/methodology_data.py --run 20261013-ref``
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import warnings

import contextily as cx
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
from shapely.geometry import box  # noqa: E402

from src.pipeline.config import ExperimentConfig  # noqa: E402
from src.pipeline.hybrid_prep import load_lines  # noqa: E402
from src.sources.geometry import ScopeGeometry  # noqa: E402

warnings.filterwarnings("ignore")
AERIAL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"
PAD = 60.0

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
cfg = ExperimentConfig.from_run(args.run)
worst = pd.read_csv(cfg.exports_dir / "worst100_sam_regions.csv")
REGIONS = {
    "maas3_l_18300_18310": "Already past its signal line",
    "maas3_r_16860_16870": "Eroding: crosses 2032–2035",
    "maas3_r_16320_16330": "Stable bank",
    worst["location_id"].iloc[0]: "Worst-predicted SAM region",
}

geometry = ScopeGeometry(centreline_gpkg=cfg.raw_gpkg)
lines_all = load_lines(cfg, geometry)  # index = line_idx used by samples
samples = pd.read_parquet(cfg.features_dir / "samples.parquet")
kept_idx = set(samples["line_idx"].unique())
segp = pd.read_parquet(cfg.model_out_dir / "segment_predictions.parquet")
segt = pd.read_parquet(cfg.model_out_dir / "segment_test.parquet")
vvr_x = pyogrio.read_dataframe(
    cfg.exports_dir / f"vvr_kruising_{args.run}.gpkg", read_geometry=False
).set_index("location_id")
wp = pyogrio.read_dataframe(
    cfg.exports_dir / f"worst_predicted_v2_{args.run}.gpkg",
    layer="vlakken",
    read_geometry=False,
).set_index("location_id")
scope = (
    pyogrio.read_dataframe(cfg.raw_gpkg, layer="vlakken_scope")
    .rename(columns={"position_id": "location_id"})
    .set_index("location_id")
)
sig = gpd.read_file(cfg.signalering_gpkg, layer=cfg.signalering_layer).to_crs(28992)
vvr_pl = gpd.read_file(
    cfg.signalering_gpkg, layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl"
).to_crs(28992)


def coords(geom, ox, oy):
    if geom is None or geom.is_empty:
        return []
    out = []
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type == "Polygon":
            g = g.exterior
        xy = np.asarray(g.coords)[:, :2]
        out.append([[round(x - ox, 1), round(y - oy, 1)] for x, y in xy])
    return out


def basemap(bounds):
    x0, y0, x1, y1 = bounds
    fig = plt.figure(figsize=(5, 5 * (y1 - y0) / (x1 - x0)), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    cx.add_basemap(ax, crs="EPSG:28992", source=AERIAL, attribution=False, zoom=18)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", pil_kwargs={"quality": 74})
    plt.close(fig)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


out = {
    "run": args.run,
    "R": cfg.segment_R,
    "basemap": "PDOK luchtfoto (Actueel_orthoHR)",
    "regions": [],
}
for loc, title in REGIONS.items():
    poly = scope.geometry[loc]
    x0, y0, x1, y1 = poly.buffer(PAD).bounds
    area = box(x0, y0, x1, y1)
    ls = lines_all[lines_all["location_id"] == loc]
    s = samples[samples["location_id"] == loc]
    out["regions"].append(
        {
            "id": loc,
            "title": title,
            "size": [round(x1 - x0, 1), round(y1 - y0, 1)],
            "basemap": basemap((x0, y0, x1, y1)),
            "scope": coords(poly, x0, y0),
            "centreline": coords(geometry.centrelines[loc], x0, y0),
            "signal": [
                c
                for g in sig[sig.intersects(area)].geometry
                for c in coords(g.intersection(area), x0, y0)
            ],
            "vvr": [
                c
                for g in vvr_pl[vvr_pl.intersects(area)].geometry
                for c in coords(g.intersection(area), x0, y0)
            ],
            "lines": [
                {
                    "date": str(pd.Timestamp(r.date).date()),
                    "model": r.model,
                    "kept": bool(i in kept_idx),
                    "xy": coords(r.geometry, x0, y0),
                }
                for i, r in ls.iterrows()
            ],
            "samples": s[["station", "dist", "date", "model"]]
            .assign(
                date=pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d"),
                station=s["station"].round(4),
                dist=s["dist"].round(2),
            )
            .values.tolist(),
            "segments": segp[segp["location_id"] == loc][
                ["seg", "last_year", "last_dist_m", "v_pred"]
            ]
            .round(3)
            .values.tolist(),
            "test": segt[segt["location_id"] == loc][["seg", "v_test", "pred", "naive"]]
            .round(3)
            .values.tolist(),
            "facts": {
                "crossing": str(vvr_x.loc[loc, "kruising_klasse"])
                if loc in vvr_x.index
                else None,
                "crossing_year": num(vvr_x.loc[loc, "kruisingsjaar"])
                if loc in vvr_x.index
                else None,
                "v_mean": num(vvr_x.loc[loc, "v_mean_m_per_yr"])
                if loc in vvr_x.index
                else None,
                "preference": str(wp.loc[loc, "model_preference"])
                if loc in wp.index
                else None,
                "split": str(wp.loc[loc, "split"]) if loc in wp.index else None,
                "err_horizon": num(wp.loc[loc, "err_horizon_mean"])
                if loc in wp.index
                else None,
                "lines_delivered": int(len(ls)),
                "lines_kept": int(ls.index.isin(kept_idx).sum()),
            },
        }
    )
    print(loc, title, f"{len(s)} samples")

path = cfg.figures_dir / "methodology_data.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"→ {path} ({path.stat().st_size / 1e6:.2f} MB)")
