"""Worst-predicted regions v2 — QGIS-first, every region with an honest error.

Upgrades scripts/export_worst_predicted.py (v1, top-100 of the run's own test
set) to the browsable design:

  1. Grouped 5-fold CV at region level (stratified by cluster, the same LGB
     recipe as the run incl. honest early stopping) so EVERY region with a
     full t1/t2/t3 carries an out-of-fold 1-yr error (``err_1jr_cv``).
  2. Per-region horizon error: the run's segment model (rebuilt bit-for-bit,
     gated on reproducing its test MAE) scores the latest >=2-yr pair of
     every segment of every region; per region the mean/max |error|.
     CAVEAT: one model scores everything — for regions in its training half
     the attribute is optimistic; ``horizon_in_train=1`` marks them.
  3. gpkg layers: ``vlakken`` (all 5,702 regions, both errors + ranks +
     cleaning-flag counts + hybrid preference), ``voorspelde_lijn`` (the
     R=20 predicted-bank polyline at horizon, one feature per segment with
     its own error where scored; split regions only, to bound size),
     ``gemeten_lijnen`` (measured history of the top-200 by rank_horizon —
     the full delivery is too heavy for one handover file).
  4. PDF primer: cover stats + top-20 by rank_horizon as OSM panels with the
     predicted polyline drawn in orange.

Run: ``uv run python scripts/export_worst_predicted_v2.py``
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.erosion.region_inspector import (
    VVR_PURPLE,
    RegionInspector,
    _line_parts,
    measured_color,
)
from src.pipeline.config import ExperimentConfig
from src.pipeline.segments import (
    REGION_CONTEXT,
    SEG_FEATS,
    SEG_KEY,
    _increment_rows,
    _segment_yearly,
    _traj_for,
)
from src.pipeline.trajectory import TRAJ_FEATS2
from src.sources.geometry import LOCATION_ID, normalise_location_id

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]
GREY = "#444444"
ORANGE = "#E07A3F"
RUN = "20260904-r20"
STAMP = "20260904"
MODES = ["doolhof", "kronkelend", "tijd_uitschieter", "fragment", "verkeerde_oever"]

cfg = ExperimentConfig(experiment=RUN)
OUT_DIR = cfg.data_dir / "02_processed/triage"
bundle = joblib.load(cfg.model_out_dir / "bundle.joblib")
feats = bundle["config"]["FEATS_LGB"]
rf = pd.read_parquet(cfg.features_dir / "region_features.parquet")

# ── 1 · grouped 5-fold CV: honest 1-yr error for every region ────────────────
X = rf[feats].astype(float)
y = rf["v_test"]
cv_pred = pd.Series(index=rf.index, dtype=float)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
for fold, (tr_i, te_i) in enumerate(skf.split(rf, rf["cluster"])):
    tr = rf.iloc[tr_i]
    rng = np.random.default_rng(cfg.seed + fold)
    val_mask = rng.random(len(tr)) < cfg.val_frac
    m = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=cfg.seed,
        verbose=-1,
    )
    m.fit(
        tr[~val_mask][feats].astype(float),
        tr[~val_mask]["v_test"],
        eval_set=[(tr[val_mask][feats].astype(float), tr[val_mask]["v_test"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    cv_pred.iloc[te_i] = m.predict(rf.iloc[te_i][feats].astype(float))
err_cv = (cv_pred - y).abs()
cv_mae = err_cv.mean()
print(f"CV: {len(rf):,} regions · pooled out-of-fold MAE {cv_mae:.4f}")
assert 2.2 <= cv_mae <= 2.9, f"CV MAE {cv_mae:.3f} outside sanity window"

# overlap with v1's test-set top-100
model_run = joblib.load(cfg.model_out_dir / "model_lgb.joblib")
test = rf[rf["split"] == "test"]
run_err = (model_run.predict(test[feats].astype(float)) - test["v_test"]).abs()
v1_top = set(run_err.sort_values(ascending=False).head(100).index)
cv_test_top = set(err_cv.loc[test.index].sort_values(ascending=False).head(100).index)
overlap = len(v1_top & cv_test_top)
print(f"overlap v1 top-100 vs CV top-100 (test subset): {overlap}/100")

# ── 2 · segment model rebuilt + horizon error for ALL regions ────────────────
R, H = cfg.segment_R, cfg.horizon_min_years
kept_line_idx = pd.read_parquet(cfg.features_dir / "samples.parquet")[
    "line_idx"
].unique()
seg_obs = _segment_yearly(cfg, kept_line_idx)
yearly = seg_obs.groupby([*SEG_KEY, "year"])["dist_m"].median().reset_index()

test_ids = set(rf.index[rf["split"] == "test"])
train_ids = set(rf.index[rf["split"] == "train"])
ctx_cols = [c for c in REGION_CONTEXT if c in rf.columns]
ctx = rf[ctx_cols].copy()
if "is_nvo" in ctx:
    ctx["is_nvo"] = ctx["is_nvo"].astype(float)
feat_names = SEG_FEATS + ctx_cols + TRAJ_FEATS2


def decorate(frame):
    frame["seg_center"] = (frame["seg"] + 0.5) / R
    frame["seg_edge"] = np.minimum(frame["seg_center"], 1 - frame["seg_center"])
    frame = frame.join(ctx, on=LOCATION_ID)
    frame = pd.concat([frame, _traj_for(seg_obs, frame)], axis=1)
    frame[feat_names] = frame[feat_names].astype(float).fillna(0.0)
    return frame


frame = decorate(_increment_rows(yearly, H, test_ids, train_ids))
train, stest = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
rng = np.random.default_rng(cfg.seed)
train_locs = train[LOCATION_ID].unique()
val_locs = set(rng.choice(train_locs, int(len(train_locs) * cfg.val_frac), False))
val_mask = train[LOCATION_ID].isin(val_locs)
fit_rows, val_rows = train[~val_mask], train[val_mask]
seg_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=cfg.seed, verbose=-1
)
seg_model.fit(
    fit_rows[feat_names],
    fit_rows["v_test"],
    sample_weight=np.clip(fit_rows["test_span_yr"], 2, 5),
    eval_set=[(val_rows[feat_names], val_rows["v_test"])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
seg_mae = np.abs(seg_model.predict(stest[feat_names]) - stest["v_test"]).mean()
print(f"segment model rebuilt: test MAE {seg_mae:.4f} (run: 0.9743)")
assert abs(seg_mae - 0.9743) < 0.02, "segment model does not reproduce the run"

frame_all = decorate(_increment_rows(yearly, H, set(rf.index), set()))
frame_all["v_pred_seg"] = seg_model.predict(frame_all[feat_names])
frame_all["err_seg"] = (frame_all["v_pred_seg"] - frame_all["v_test"]).abs()
reg_h = frame_all.groupby(LOCATION_ID)["err_seg"].agg(
    err_horizon_mean="mean", err_horizon_max="max", n_seg_scored="size"
)
print(
    f"horizon errors: {len(frame_all):,} segments · {len(reg_h):,} regions scored"
)

# ── 3 · vlakken layer ────────────────────────────────────────────────────────
ri = RegionInspector()
vl = rf[["v_test", "is_nvo", "cluster", "split"]].copy()
vl["v_pred_cv"] = cv_pred.round(2)
vl["err_1jr_cv"] = err_cv.round(2)
vl["rank_1jr"] = err_cv.rank(ascending=False).astype(int)
vl = vl.join(reg_h)
vl["rank_horizon"] = vl["err_horizon_mean"].rank(ascending=False)
vl["horizon_in_train"] = vl.index.isin(train_ids).astype(int)
vl["river"] = vl.index.str.extract(r"^([a-z]+\d*)", expand=False)
flags = gpd.read_file(OUT_DIR / f"sam_review_{STAMP}.gpkg", layer="vlakken")
flags = flags.set_index(LOCATION_ID)[[f"n_{m}" for m in MODES]]
vl = vl.join(flags, how="left")
vl[[f"n_{m}" for m in MODES]] = vl[[f"n_{m}" for m in MODES]].fillna(0).astype(int)
vl["geen_flags"] = (vl[[f"n_{m}" for m in MODES]].sum(axis=1) == 0).astype(int)
pref = normalise_location_id(
    gpd.read_file(cfg.data_dir / "02_processed/hybrid/model_preference_20260710.gpkg")
).set_index(LOCATION_ID)["model_preference"]
vl["model_preference"] = pref.reindex(vl.index).fillna("onbekend")
vl[["err_horizon_mean", "err_horizon_max"]] = vl[
    ["err_horizon_mean", "err_horizon_max"]
].round(2)
scope = ri.geometry.polygons
vlakken = gpd.GeoDataFrame(
    vl.reset_index(names=LOCATION_ID),
    geometry=scope.reindex(vl.index).values,
    crs=28992,
)

# ── voorspelde_lijn: per segment a short line at the predicted position ──────
segp = pd.read_parquet(cfg.model_out_dir / "segment_predictions.parquet")
segp = segp[segp[LOCATION_ID].isin(vl.index)]
serr = frame_all.set_index([LOCATION_ID, "seg"])["err_seg"]
samples = pd.read_parquet(
    cfg.features_dir / "samples.parquet", columns=[LOCATION_ID, "x", "y"]
)
side = samples.groupby(LOCATION_ID)[["x", "y"]].mean()


def anchor(cline, station, dist, sx, sy):
    base = cline.interpolate(station, normalized=True)
    eps = 0.02
    p0 = cline.interpolate(max(station - eps, 0), normalized=True)
    p1 = cline.interpolate(min(station + eps, 1), normalized=True)
    tx, ty = p1.x - p0.x, p1.y - p0.y
    norm = np.hypot(tx, ty) or 1.0
    nx, ny = -ty / norm, tx / norm
    sign = np.sign((sx - base.x) * nx + (sy - base.y) * ny) or 1.0
    return base.x + sign * nx * dist, base.y + sign * ny * dist


from shapely.geometry import LineString  # noqa: E402

pred_rows = []
for loc, g in segp.groupby(LOCATION_ID, sort=False):
    cline = ri.geometry.centrelines.get(loc)
    if cline is None or loc not in side.index:
        continue
    sx, sy = side.loc[loc, "x"], side.loc[loc, "y"]
    for _, r in g.iterrows():
        s0, s1 = r["seg"] / R, (r["seg"] + 1) / R
        pts = [
            anchor(cline, st, r["pred_dist_m_horizon"], sx, sy)
            for st in (s0 + 0.01, (s0 + s1) / 2, s1 - 0.01)
        ]
        e = serr.get((loc, r["seg"]), np.nan)
        pred_rows.append(
            {
                LOCATION_ID: loc,
                "seg": int(r["seg"]),
                "v_pred": round(float(r["v_pred"]), 2),
                "pred_dist_m_horizon": round(float(r["pred_dist_m_horizon"]), 1),
                "last_year": int(r["last_year"]),
                "err_horizon": round(float(e), 2) if pd.notna(e) else None,
                "geometry": LineString(pts),
            }
        )
voorspelde = gpd.GeoDataFrame(pred_rows, crs=28992)
print(f"voorspelde_lijn: {len(voorspelde):,} segment parts")

top200 = vl.sort_values("rank_horizon").head(200).index
gemeten = ri.lines[ri.lines[LOCATION_ID].isin(top200)][
    [LOCATION_ID, "date", "year", "model", "geometry"]
].copy()
gemeten["date"] = pd.to_datetime(gemeten["date"]).dt.strftime("%Y-%m-%d")

gpkg = OUT_DIR / f"worst_predicted_v2_{STAMP}.gpkg"
if gpkg.exists():
    gpkg.unlink()
vlakken.to_file(gpkg, layer="vlakken", driver="GPKG")
voorspelde.to_file(gpkg, layer="voorspelde_lijn", driver="GPKG")
gemeten.to_file(gpkg, layer="gemeten_lijnen", driver="GPKG")
print(f"gpkg: {len(vlakken):,} vlakken · {len(voorspelde):,} pred parts · {len(gemeten):,} lijnen → {gpkg}")

# ── 4 · PDF primer ───────────────────────────────────────────────────────────
vvr = gpd.read_file(
    cfg.data_dir / "01_raw/scope/20260205_signaleringslijn.gpkg",
    layer="Vlak_vrije_ruimte_natuurvriendelijke_oever_pl",
).to_crs(28992)
vvr_sidx = vvr.sindex
HM_GREYS = {2015: "#cfd8dc", 2017: "#cfd8dc", 2021: "#90a4ae", 2025: "#455a64"}


def hm_color(year):
    ks = sorted(HM_GREYS)
    return HM_GREYS[min(ks, key=lambda k: abs(k - year))]


def flag_label(row):
    parts = [f"{m}×{int(row[f'n_{m}'])}" for m in MODES if row[f"n_{m}"] > 0]
    return ", ".join(parts) if parts else "geen"


top100_h = vl.sort_values("rank_horizon").head(100)
n_clean = int(top100_h["geen_flags"].sum())
riv = top100_h["river"].value_counts()
prefc = top100_h["model_preference"].value_counts()

pdf_path = OUT_DIR / f"worst_predicted_v2_{STAMP}.pdf"
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(
        0.06, 0.92,
        "Slechtst voorspelde vlakken v2 · elke regio een eerlijke fout (5-fold CV + horizon)",
        fontsize=15, weight="bold",
    )
    fig.text(
        0.06, 0.87,
        f"1-jaars CV-fout: alle {len(rf):,} vlakken out-of-fold voorspeld · pooled MAE {cv_mae:.2f} m/jr "
        f"(run-testset: 2.47).\n"
        f"Horizonfout (segment × ≥2 jr): {len(reg_h):,} vlakken gescoord; voor {int(vl['horizon_in_train'].eq(1).sum()):,} "
        f"train-vlakken is dit optimistisch (horizon_in_train = 1).\n"
        f"Top-100 op horizonfout: {100 - n_clean} met ≥ 1 cleaning-flag · {n_clean} 'schoon' — daar zit de winst.\n"
        f"Rivieren (top-100): {' · '.join(f'{k} {v}' for k, v in riv.items())}\n"
        f"Hybride voorkeur (top-100): {' · '.join(f'{k} {v}' for k, v in prefc.items())}",
        fontsize=9.5, va="top",
    )
    fig.text(
        0.06, 0.68,
        "De vraag: welke patronen zien jullie in deze vlakken die onze filters missen?",
        fontsize=12, weight="bold",
    )
    ax = fig.add_axes((0.10, 0.10, 0.83, 0.50))
    ax.hist(vl["err_1jr_cv"].dropna(), bins=80, color="#2BB5A6", alpha=0.8, label="1-jaars CV-fout")
    ax.hist(vl["err_horizon_mean"].dropna(), bins=80, color=ORANGE, alpha=0.6, label="horizonfout (gem. per vlak)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_xlabel("|fout| (m/jr) · log-schaal op de y-as")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    pdf.savefig(fig)
    plt.close(fig)

    def panel(ax, loc, row):
        sgeom = scope.get(loc)
        cline = ri.geometry.centrelines.get(loc)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"#{int(row['rank_horizon'])} {loc} · horizon {row['err_horizon_mean']:.1f} "
            f"(max {row['err_horizon_max']:.1f}) · 1 jr {row['err_1jr_cv']:.1f}\n"
            f"flags: {flag_label(row)} · in_train: {int(row['horizon_in_train'])}",
            fontsize=6.2, color=GREY,
        )
        if sgeom is None:
            return
        minx, miny, maxx, maxy = sgeom.bounds
        ax.set_xlim(minx - 40, maxx + 40)
        ax.set_ylim(miny - 40, maxy + 40)
        ax.fill(*sgeom.exterior.xy, fc="#dddddd", ec="#aaaaaa", lw=0.8, alpha=0.4, zorder=1)
        for geom in vvr.iloc[list(vvr_sidx.intersection(sgeom.bounds))][
            lambda d: d.intersects(sgeom)
        ].geometry:
            for poly in getattr(geom, "geoms", [geom]):
                ax.fill(*poly.exterior.xy, fc=VVR_PURPLE, ec=VVR_PURPLE, lw=1.0, alpha=0.18, zorder=2)
        ri._add_basemap(ax)
        if cline is not None:
            ax.plot(*cline.xy, color="black", lw=1.1, zorder=5)
        sub = ri.lines[ri.lines[LOCATION_ID] == loc].sort_values("year")
        for _, lrow in sub.iterrows():
            col = (
                hm_color(int(lrow["year"]))
                if lrow["model"] == "hoogtemodel"
                else measured_color(int(lrow["year"]))
            )
            for part in _line_parts(lrow.geometry):
                ax.plot(*part.xy, color=col, lw=1.1, alpha=0.85, zorder=4)
        pl = voorspelde[voorspelde[LOCATION_ID] == loc].sort_values("seg")
        for geom in pl.geometry:
            ax.plot(*geom.xy, color=ORANGE, lw=2.0, zorder=6)

    order = top100_h.head(20)
    for start in range(0, len(order), 6):
        chunk = order.iloc[start : start + 6]
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
        fig.suptitle(
            "slechtst voorspeld (horizon-ranking) · groen = SAM (donker = recenter) · "
            "blauwgrijs = hoogtemodel · oranje = voorspelde oever (R = 20) · paars = VVR",
            fontsize=9,
        )
        for ax, (loc, row) in zip(axes.flatten(), chunk.iterrows(), strict=False):
            panel(ax, loc, row)
        for ax in axes.flatten()[len(chunk) :]:
            ax.set_axis_off()
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        plt.close(fig)
print(f"pdf: {pdf_path}")
print(
    f"SUMMARY · CV MAE {cv_mae:.3f} · overlap v1 {overlap}/100 · "
    f"top100-horizon geen_flags {n_clean} · pref {prefc.to_dict()} · rivers {riv.to_dict()}"
)
