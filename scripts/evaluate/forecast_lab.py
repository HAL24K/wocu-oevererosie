"""Forecast lab: how should a strip's bank position be projected forward?

The map draws, per 5 m strip, ``position(year) = start + speed * (year - last_year)``.
This lab compares ways to choose ``start`` and ``speed`` on the strips' own measured
histories, without rerunning the pipeline:

- backtests that hide each strip's last measurement(s) and forecast them from the rest
  (``next``: the last year; ``long``: everything after a cut-off at least 3 years before
  the last year), scored as position error, error per year and direction agreement;
- sanity on the real forecast from all data: extreme speeds, speeds against the strip's
  own long-term trend, and jaggedness between neighbouring strips.

Methods are small functions of one strip's history (years, positions, sources) plus
optional smoothing across the strips of a region. The current model (``ml``) is the
saved strip model of the run applied to forecast-style rows built from the history.

Run: ``PYTHONPATH=. uv run python scripts/evaluate/forecast_lab.py --run 20261013-preclean``
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import segments as S
from src.pipeline.config import ExperimentConfig
from src.pipeline.forecast import Hist, jumpfree_valid, smooth, theil_mat  # noqa: F401
from src.pipeline.forecast import fit as _fit
from src.sources.geometry import LOCATION_ID

warnings.filterwarnings("ignore")
KEY = [LOCATION_ID, "seg"]


# ── data ─────────────────────────────────────────────────────────────────────
def strip_obs(cfg: ExperimentConfig) -> pd.DataFrame:
    """Per-date strip positions of the run (step 8's ``_segment_yearly``), cached."""
    path = cfg.features_dir / "segment_obs.parquet"
    if not path.exists():
        kept = pd.read_parquet(
            cfg.features_dir / "samples.parquet", columns=["line_idx"]
        )
        S._segment_yearly(cfg, kept["line_idx"].unique()).to_parquet(path)
    return pd.read_parquet(path)


def yearly(obs: pd.DataFrame) -> pd.DataFrame:
    y = (
        obs.groupby([*KEY, "year"])
        .agg(dist=("dist_m", "median"), model=("model", "first"))
        .reset_index()
        .sort_values([*KEY, "year"])
    )
    y["sam"] = (y["model"] == "segmentation").astype(float)
    return y


# ── per-strip methods, vectorised over all strips ────────────────────────────
# A history becomes matrices (``Hist``: one row per strip, one column per year);
# a method maps them to (start, speed) at each strip's last history year.
def m_none(H):
    return H.d_last, np.zeros(len(H.d_last))


def m_persist(H):
    dt = H.t_last - H.t_prev
    v = np.where(dt > 0, (H.d_last - H.d_prev) / np.where(dt > 0, dt, 1), 0.0)
    return H.d_last, v


def m_theil(H):
    level, v, _, _ = _fit(H, np.isfinite(H.D))
    return level, v


def m_theil_last(H):
    _, v, _, _ = _fit(H, np.isfinite(H.D))
    return H.d_last, v


def make_jumpfree(k: float, start: str = "level"):
    """Drop years more than ``k`` m off the robust line, refit (needs >= 3 years)."""

    def m(H):
        level, v, _, _ = _fit(H, jumpfree_valid(H, k))
        return (level if start == "level" else H.d_last), v

    return m


def make_recent(n_years: int):
    """Theil over the last ``n_years`` of history only."""

    def m(H):
        valid = np.isfinite(H.D) & (H.T >= H.t_last[:, None] - n_years + 0.5)
        valid = np.where((valid.sum(1) >= 2)[:, None], valid, np.isfinite(H.D))
        level, v, _, _ = _fit(H, valid)
        return level, v

    return m


def make_recent_epochs(n: int, alpha: float = 1.0):
    """Slope between the last ``n`` epochs (Theil), scaled by ``alpha``; start = last."""

    def m(H):
        idx = np.arange(H.T.shape[1])[None, :]
        valid = np.isfinite(H.D) & (idx >= (H.cnt - n)[:, None])
        _, v, _, _ = _fit(H, valid)
        return H.d_last, alpha * v

    return m


def make_shrunk(alpha: float, base=None):
    """``base`` method's start with its speed scaled by ``alpha``."""
    base = base or m_theil

    def m(H):
        start, v = base(H)
        return start, alpha * v

    return m


def region_drift(h: pd.DataFrame, alpha: float, k: float = 10.0) -> pd.DataFrame:
    """Lasting drift per region: ``alpha`` x the median over its strips of the
    jump-free Theil slope over each strip's whole history."""
    f = run_method(make_jumpfree(k), h)
    reg = f.groupby(LOCATION_ID)["v"].median().rename("v2") * alpha
    return f[KEY].join(reg, on=LOCATION_ID)


METHODS = {
    "none": m_none,
    "persist": m_persist,
    "theil": m_theil,
    "theil_last": m_theil_last,
    "jumpfree10": make_jumpfree(10),
    "jumpfree5": make_jumpfree(5),
    "jumpfree10_last": make_jumpfree(10, start="last"),
    "recent4": make_recent(4),
    "epochs2": make_recent_epochs(2),
    "epochs2_a50": make_recent_epochs(2, 0.5),
    "epochs3_a50": make_recent_epochs(3, 0.5),
    "theil_a25": make_shrunk(0.25),
    "theil_a50": make_shrunk(0.5),
}


def run_method(fn, hist: pd.DataFrame) -> pd.DataFrame:
    H = Hist(hist)
    start, v = fn(H)
    return H.keys.assign(last_year=H.t_last, start=start, v=v)


# ── the current model, on forecast-style rows from a history ─────────────────
class ML:
    def __init__(self, cfg: ExperimentConfig, obs: pd.DataFrame):
        self.cfg = cfg
        self.model = joblib.load(cfg.model_out_dir / "segment_model.joblib")
        self.feats = list(self.model.feature_name_)
        rf = pd.read_parquet(cfg.features_dir / "region_features.parquet")
        p = cfg.features_dir / "region_inference_features.parquet"
        if p.exists():
            rf = pd.concat([rf, pd.read_parquet(p)])
        if LOCATION_ID in rf.columns:
            rf = rf.set_index(LOCATION_ID)
        rf = rf[~rf.index.duplicated()]
        ctx = rf[[c for c in S.REGION_CONTEXT if c in rf.columns]].copy()
        if "is_nvo" in ctx:
            ctx["is_nvo"] = ctx["is_nvo"].astype(float)
        self.ctx = ctx
        self.obs = obs
        self._cache: dict = {}

    def __call__(self, hist: pd.DataFrame) -> pd.DataFrame:
        key = (len(hist), float(hist["dist"].sum()), float(hist["year"].sum()))
        if key not in self._cache:
            self._cache[key] = self._predict(hist)
        return self._cache[key].copy()

    def _predict(self, hist: pd.DataFrame) -> pd.DataFrame:
        R, H = self.cfg.segment_R, self.cfg.horizon_min_years
        last = hist.groupby(KEY).tail(2)
        rows = []
        for (loc, seg), g in last.groupby(KEY, sort=False):
            ys, ds = g["year"].to_numpy(), g["dist"].to_numpy()
            if len(ys) < 2:
                continue
            rows.append(
                {
                    LOCATION_ID: loc,
                    "seg": seg,
                    "t2": int(ys[-1]),
                    "dist_t2": ds[-1],
                    "train_span_yr": ys[-1] - ys[-2],
                    "test_span_yr": float(H),
                    "v_train": (ds[-1] - ds[-2]) / (ys[-1] - ys[-2]),
                }
            )
        f = pd.DataFrame(rows)
        f["n_seg_years"] = f.merge(
            hist.groupby(KEY).size().rename("n").reset_index(), on=KEY
        )["n"].to_numpy()
        f["seg_center"] = (f["seg"] + 0.5) / R
        f["seg_edge"] = np.minimum(f["seg_center"], 1 - f["seg_center"])
        f = f.join(self.ctx, on=LOCATION_ID)
        o = self.obs.merge(f[[*KEY, "t2"]], on=KEY)
        o = o[o["year"] <= o["t2"]]
        f = pd.concat([f, S._traj_for(o, f)], axis=1)
        X = f.reindex(columns=self.feats).astype(float).fillna(0.0)
        f["v"] = self.model.predict(X)
        return f.rename(columns={"t2": "last_year", "dist_t2": "start"})[
            [*KEY, "last_year", "start", "v"]
        ]


# ── backtests and sanity ─────────────────────────────────────────────────────
def split_history(y: pd.DataFrame, mode: str):
    """(history, target) per strip; ``next`` hides the last year, ``long`` hides all
    years after a cut-off >= 3 years before the last year."""
    y = y.copy()
    y["n"] = y.groupby(KEY)["year"].transform("size")
    y["last"] = y.groupby(KEY)["year"].transform("max")
    y = y[y["n"] >= 3]
    if mode == "next":
        hist = y[y["year"] < y["last"]]
        tgt = y[y["year"] == y["last"]]
    else:
        hist = y[y["year"] <= y["last"] - 3]
        tgt = y[y["year"] == y["last"]]
    ok = hist.groupby(KEY).size() >= 2
    keys = ok[ok].index
    hist = hist.set_index(KEY).loc[keys].reset_index()
    tgt = tgt.set_index(KEY).loc[keys].reset_index()
    return hist, tgt[[*KEY, "year", "dist", "model"]]


def score(fc: pd.DataFrame, tgt: pd.DataFrame, hist: pd.DataFrame) -> dict:
    m = fc.merge(tgt, on=KEY)
    h = m["year"] - m["last_year"]
    cap = m["cap"] if "cap" in m else np.inf
    v2 = m["v2"] if "v2" in m else 0.0
    pred = m["start"] + m["v"] * np.minimum(h, cap) + v2 * h
    err = (pred - m["dist"]).abs()
    last_obs = hist.groupby(KEY).tail(1).set_index(KEY)["dist"]
    actual_move = m["dist"] - m.set_index(KEY).index.map(last_obs).to_numpy()
    moved = actual_move.abs() > 2
    pred_move = pred - m.set_index(KEY).index.map(last_obs).to_numpy()
    sam = (m["model"] == "segmentation").to_numpy()
    return {
        "n": int(len(m)),
        "pos_mae_sam": round(float(err[sam].mean()), 2),
        "pos_mae_height": round(float(err[~sam].mean()), 2),
        "pos_mae_m": round(float(err.mean()), 2),
        "pos_med_m": round(float(err.median()), 2),
        "mae_per_yr": round(float((err / h.clip(lower=1)).mean()), 3),
        "direction_ok": round(
            float((np.sign(pred_move[moved]) == np.sign(actual_move[moved])).mean()), 3
        ),
    }


def sanity(fc: pd.DataFrame, y: pd.DataFrame) -> dict:
    """On the map's forecast (all data, 5 years ahead): mean speed over those years."""
    fc = fc.copy()
    if "cap" in fc:
        fc["v"] = fc["v"] * np.minimum(fc["cap"].fillna(np.inf), 5) / 5
    if "v2" in fc:
        fc["v"] = fc["v"] + fc["v2"]
    tr = run_method(make_jumpfree(10), y).set_index(KEY)["v"].rename("trend")
    f = fc.set_index(KEY).join(tr)
    opp = (
        (np.sign(f["v"]) != np.sign(f["trend"]))
        & (f["v"].abs() > 1)
        & (f["trend"].abs() > 1)
    )
    f = f.reset_index().sort_values(KEY)
    nb = (
        f.groupby(LOCATION_ID)["v"]
        .diff()
        .abs()[f.groupby(LOCATION_ID)["seg"].diff() == 1]
    )
    return {
        "v_median": round(float(f["v"].median()), 3),
        "abs_v_gt3": round(float((f["v"].abs() > 3).mean()), 3),
        "against_trend": round(float(opp.mean()), 3),
        "jagged_med_dv": round(float(nb.median()), 3),
        "jagged_p90_dv": round(float(nb.quantile(0.9)), 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--smooth", nargs="*", type=int, default=[1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save", type=Path, help="write each method's map forecast here")
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_run(args.run)
    obs = strip_obs(cfg)
    y = yearly(obs)
    split = pd.read_parquet(cfg.features_dir / "region_split.parquet")
    test_ids = set(split.index[split["split"] == "test"])
    methods = {k: (lambda h, fn=fn: run_method(fn, h)) for k, fn in METHODS.items()}
    ml = ML(cfg, obs)
    H = cfg.horizon_min_years

    def ml_cap(h):
        return ml(h).assign(cap=H)

    def source_aware(h, height=make_recent_epochs(2, 0.5), sam=ml_cap):
        """The model's short correction where the strip's last year is SAM; a damped
        recent slope where it is the height model (whose jumps persist)."""
        a = sam(h).set_index(KEY)
        b = run_method(height, h).set_index(KEY).assign(cap=np.inf)
        last_sam = h.sort_values("year").groupby(KEY)["sam"].last() == 1
        keep_b = ~last_sam.reindex(a.index).fillna(False)
        a.loc[keep_b, ["start", "v", "cap"]] = b.loc[
            a.index[keep_b], ["start", "v", "cap"]
        ]
        missing = b.index.difference(a.index)
        return pd.concat([a, b.loc[missing]]).reset_index()

    methods["ml"] = ml
    methods["ml_cap"] = ml_cap
    methods["source_aware"] = source_aware
    methods["source_aware_none"] = lambda h: source_aware(h, height=m_none)

    def combo(h, alpha, sam_width=1, start_width=1):
        """Now: model correction (<= H years) for SAM strips, last epoch for height
        strips; plus a lasting regional drift of ``alpha`` x the long-term slope."""
        a = smooth(source_aware(h, height=m_none), sam_width)
        a = smooth(a, start_width, cols=("start",))
        a.loc[~np.isfinite(a["cap"]), "cap"] = 0.0  # height strips: no correction
        d = region_drift(h, alpha)
        return a.merge(d, on=KEY, how="left").fillna({"v2": 0.0})

    for alpha in (0.0, 0.1, 0.25, 0.5, 1.0):
        methods[f"combo_a{int(alpha * 100)}"] = lambda h, a=alpha: combo(h, a)
        methods[f"combo_a{int(alpha * 100)}_s5"] = lambda h, a=alpha: combo(h, a, 5)
        for sw in (3, 5):
            methods[f"combo_a{int(alpha * 100)}_s5_st{sw}"] = (
                lambda h, a=alpha, sw=sw: combo(h, a, 5, sw)
            )
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    rows = []
    for name, fn in methods.items():
        for w in args.smooth:
            r = {"method": name, "smooth": w}
            for mode in () if args.no_backtest else ("next", "long"):
                hist, tgt = split_history(y, mode)
                hist = hist[hist[LOCATION_ID].isin(test_ids)]
                fc = fn(hist)
                fc = smooth(fc, w)
                for k, v in score(fc, tgt, hist).items():
                    r[f"{mode}_{k}"] = v
            fc_all = smooth(fn(y), w)
            r.update(sanity(fc_all, y))
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                fc_all.to_parquet(args.save / f"{name}_s{w}.parquet", index=False)
            rows.append(r)
            print(json.dumps(r), flush=True)
    res = pd.DataFrame(rows)
    out = args.out or cfg.run_dir / "compare" / "forecast_lab.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(res.to_string())
    print(f"→ {out}")
