"""Variant runner + scorecard for the loop-engineering experiment.

One variant = a named list of cleaning rules (plus the legacy region-level
|v| limit) run against the cached sample table, scored against the frozen
holdout, and appended to the ledger. Feature assembly mirrors
``src.pipeline.feature_engineering.build_features`` exactly, but from caches,
so an iteration takes about a minute instead of a feature rebuild.

The contract (see experiments/loop/README.md):
  - the test set is the frozen region list; no variant ever trains on it
  - CORE = frozen test regions that survive the v0 baseline; every variant
    reports coverage of CORE and MAE on CORE ∩ its own survivors
  - primary metric: LGB test tail-MAE (v_test > 2 m/yr); guardrails: overall
    test MAE and CORE coverage (soft floor 90%)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.cleaning.rules import RuleContext, apply_rules
from src.pipeline.config import ExperimentConfig
from src.pipeline.feature_engineering import (
    HW_WINDOW_COLS,
    _hw_window_stats,
    encode_col,
)
from src.pipeline.region_split import _build_split_row
from src.pipeline.train import FEATS_LGB, TAIL_THRESHOLD
from src.sources.geometry import LOCATION_ID
from src.sources.observations import BankObservations

N_POINTS = 3  # furthest samples averaged per survey — the project convention

CAT_KEYS = {
    "river": "river",
    "vegetation_class": "rws_vegetatielegger:vegetatieklassen_majority_class_vlklasse",
    "land_use": "BrpGewas_majority_class_category",
    "soil_group": "soil_group",
}


@dataclass
class Caches:
    samples: pd.DataFrame
    line_metrics: pd.DataFrame  # indexed by line_idx
    static: pd.DataFrame  # indexed by location_id
    hw_metrics: pd.DataFrame  # indexed by (station_id, year)
    station_annual: dict  # station → DataFrame indexed by year
    ev: pd.DataFrame
    frozen_test: set
    exp_dir: Path
    cache_dir: Path

    @property
    def core(self) -> set | None:
        p = self.exp_dir / "core_regions.csv"
        return set(pd.read_csv(p)["location_id"]) if p.exists() else None

    @property
    def tail_frozen(self) -> set | None:
        p = self.exp_dir / "tail_frozen_regions.csv"
        return set(pd.read_csv(p)["location_id"]) if p.exists() else None


def load_caches(data_dir: Path | None = None) -> Caches:
    cfg = ExperimentConfig(
        experiment="loop", **({"data_dir": data_dir} if data_dir else {})
    )
    cache = cfg.data_dir / "03_features/loop"
    exp_dir = cfg.data_dir.parent / "experiments/loop"
    ann_flat = pd.read_parquet(cache / "station_annual.parquet")
    station_annual = {
        code: g.set_index("year")[["days_above_p90"]]
        for code, g in ann_flat.groupby("station_code")
    }
    return Caches(
        samples=pd.read_parquet(cache / "samples.parquet"),
        line_metrics=pd.read_parquet(cache / "line_metrics.parquet").set_index(
            "line_idx"
        ),
        static=pd.read_parquet(cache / "static_features.parquet").set_index(
            LOCATION_ID
        ),
        hw_metrics=pd.read_parquet(cache / "hw_metrics.parquet").set_index(
            ["station_id", "year"]
        ),
        station_annual=station_annual,
        ev=pd.read_parquet(cache / "ev_table.parquet"),
        frozen_test=set(
            pd.read_csv(exp_dir / "frozen_test_regions.csv")["location_id"]
        ),
        exp_dir=exp_dir,
        cache_dir=cache,
    )


# ── observation building ──────────────────────────────────────────────────────


def aggregate_observations(samples: pd.DataFrame) -> pd.DataFrame:
    """Furthest-N pooled samples per survey → one distance per (region, date)."""
    grouped = samples.groupby([LOCATION_ID, "date"], sort=False)
    top = (
        grouped["dist"]
        .nlargest(N_POINTS)
        .groupby(level=[0, 1], sort=False)
        .mean()
        .rename("dist_m")
        .reset_index()
    )
    counts = grouped.agg(
        n_candidates=("dist", "size"), source=("model", "first")
    ).reset_index()
    out = top.merge(counts, on=[LOCATION_ID, "date"], how="left")
    out["n_selected"] = np.minimum(out["n_candidates"], N_POINTS).astype("float64")
    out["n_candidates"] = out["n_candidates"].astype("float64")
    return out.sort_values([LOCATION_ID, "date"]).reset_index(drop=True)


def to_dist_per_year(obs_frame: pd.DataFrame) -> pd.DataFrame:
    return BankObservations(obs_frame).to_dist_per_year(within_year="median")


def farbank_region_filter(
    dpy: pd.DataFrame, v_limit: float
) -> tuple[pd.DataFrame, int]:
    """Legacy region-level exclusion: any yearly |v| above the limit."""
    d = dpy.sort_values([LOCATION_ID, "year"])
    v = d.groupby(LOCATION_ID)["dist_m"].diff() / d.groupby(LOCATION_ID)["year"].diff()
    vmax = v.abs().groupby(d[LOCATION_ID]).max()
    bad = set(vmax[vmax > v_limit].index)
    return dpy[~dpy[LOCATION_ID].isin(bad)].reset_index(drop=True), len(bad)


# ── split + features from caches ─────────────────────────────────────────────


def build_split_frame(dpy: pd.DataFrame, caches: Caches) -> pd.DataFrame:
    """region_split equivalent: tail-3 records, quality gate, frozen split."""
    counts = dpy.groupby(LOCATION_ID)["year"].count()
    eligible = counts[counts >= 3].index
    split = (
        dpy[dpy[LOCATION_ID].isin(eligible)]
        .groupby(LOCATION_ID)
        .apply(_build_split_row, include_groups=False)
    )
    st = caches.static
    split["quality"] = st["quality"].reindex(split.index)
    split = split[split["quality"] == "OK"].copy()
    split["is_nvo"] = st["is_nvo"].reindex(split.index)

    ev = caches.ev.merge(
        split[["t1", "t2", "t3"]], left_on=LOCATION_ID, right_index=True
    )
    for label, (a, b) in {"train": ("t1", "t2"), "test": ("t2", "t3")}.items():
        vol = (
            ev[(ev["_yb"] == ev[a]) & (ev["_ya"] == ev[b])]
            .groupby(LOCATION_ID)["erosion_volume"]
            .sum()
        )
        span = split[f"{label}_span_yr"]
        split[f"erosion_vol_{label}_rate"] = vol.reindex(split.index).fillna(0) / span
    split["erosion_vol_rate_t1"] = split["erosion_vol_train_rate"]

    split["split"] = np.where(split.index.isin(caches.frozen_test), "test", "train")
    return split


def build_features_fast(split: pd.DataFrame, caches: Caches) -> pd.DataFrame:
    """Assemble the exact LGB feature frame from caches."""
    st = caches.static
    feats = split.copy()
    for col in [
        "river",
        "vegetation_class",
        "land_use",
        "soil_group",
        "nearest_station",
        "bend_exposure_n5",
        "bend_exposure_n8",
    ]:
        feats[col] = st[col].reindex(feats.index)

    hw = feats.apply(
        _hw_window_stats,
        axis=1,
        hw_metrics=caches.hw_metrics,
        station_annual=caches.station_annual,
        has_t3=True,
    )
    no_ev = hw[HW_WINDOW_COLS].isna().all(axis=1)
    hw.loc[no_ev, HW_WINDOW_COLS] = 0
    feats[HW_WINDOW_COLS] = hw[HW_WINDOW_COLS]

    for col, key in CAT_KEYS.items():
        feats[f"{col}_enc"] = encode_col(feats[col], key)

    num = feats.select_dtypes("number").columns
    feats[num] = feats[num].fillna(0.0)
    return feats


# ── training + scorecard ──────────────────────────────────────────────────────


def train_score(feats: pd.DataFrame, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Train naive / persistence / LGB; score the frozen test rows."""
    train = feats[feats["split"] == "train"]
    test = feats[feats["split"] == "test"]

    X_tr = train[FEATS_LGB].astype(float)
    X_te = test[FEATS_LGB].astype(float)
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_tr,
        train["v_test"],
        eval_set=[(X_te, test["v_test"])],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds = pd.DataFrame(
        {
            "v_test": test["v_test"],
            "v_train": test["v_train"],
            "pred_naive": train["v_test"].mean(),
            "pred_persist": test["v_train"],
            "pred_lgb": model.predict(X_te),
        },
        index=test.index,
    )

    y = preds["v_test"]
    tail = y > TAIL_THRESHOLD
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "lgb_mae": mean_absolute_error(y, preds["pred_lgb"]),
        "lgb_r2": r2_score(y, preds["pred_lgb"]),
        "lgb_tail_mae": (
            mean_absolute_error(y[tail], preds.loc[tail, "pred_lgb"])
            if tail.any()
            else np.nan
        ),
        "tail_n": int(tail.sum()),
        "naive_mae": mean_absolute_error(y, preds["pred_naive"]),
        "persist_mae": mean_absolute_error(y, preds["pred_persist"]),
        "vtest_std": float(feats["v_test"].std()),
    }
    return metrics, preds


def score_frozen_views(preds: pd.DataFrame, caches: Caches) -> dict:
    """CORE coverage and fixed-denominator views, once CORE exists."""
    out: dict = {}
    core, tailf = caches.core, caches.tail_frozen
    if core:
        kept = preds.index.intersection(core)
        out["coverage_core"] = len(kept) / len(core)
        out["core_mae"] = mean_absolute_error(
            preds.loc[kept, "v_test"], preds.loc[kept, "pred_lgb"]
        )
    if tailf:
        kept_t = preds.index.intersection(tailf)
        out["tail_frozen_n_kept"] = len(kept_t)
        out["tail_frozen_mae"] = (
            mean_absolute_error(
                preds.loc[kept_t, "v_test"], preds.loc[kept_t, "pred_lgb"]
            )
            if len(kept_t)
            else np.nan
        )
    return out


# ── the runner ────────────────────────────────────────────────────────────────


def run_variant(
    name: str,
    caches: Caches,
    rules: list[tuple[str, dict]],
    structures=None,
    v_limit: float | None = 50.0,
    notes: str = "",
    seed: int = 42,
) -> dict:
    """Run one variant end to end and append it to the ledger."""
    ctx = RuleContext(
        line_metrics=caches.line_metrics,
        cl_len=caches.static["cl_len"],
        structures=structures,
    )
    s = apply_rules(caches.samples, ctx, rules)

    obs = aggregate_observations(s)
    dpy = to_dist_per_year(obs)
    n_far = 0
    if v_limit is not None:
        dpy, n_far = farbank_region_filter(dpy, v_limit)

    split = build_split_frame(dpy, caches)
    feats = build_features_fast(split, caches)
    metrics, preds = train_score(feats, seed=seed)
    metrics.update(score_frozen_views(preds, caches))
    metrics["n_regions"] = int(dpy[LOCATION_ID].nunique())
    metrics["regions_excluded_far"] = n_far

    var_dir = caches.cache_dir / "variants" / name
    var_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(var_dir / "test_preds.parquet")
    dpy.to_parquet(var_dir / "dist_per_year.parquet", index=False)
    pd.DataFrame(ctx.stats).to_json(var_dir / "rule_stats.json", orient="records")

    row = {
        "variant": name,
        "when": datetime.now().isoformat(timespec="seconds"),
        "rules": json.dumps([[n, p] for n, p in rules]),
        "v_limit": v_limit,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "notes": notes,
    }
    ledger_path = caches.exp_dir / "ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger.to_csv(ledger_path, index=False)

    print(f"── {name}")
    print(
        f"   regions {metrics['n_regions']:>6} · far-excluded {n_far}"
        f" · test {metrics['n_test']}"
    )
    print(
        f"   LGB mae {metrics['lgb_mae']:.3f} · tail {metrics['lgb_tail_mae']:.3f}"
        f" (n={metrics['tail_n']}) · R² {metrics['lgb_r2']:.3f}"
        f" · naive {metrics['naive_mae']:.3f} · persist {metrics['persist_mae']:.3f}"
    )
    for k in ("coverage_core", "core_mae", "tail_frozen_mae", "tail_frozen_n_kept"):
        if k in metrics:
            print(
                f"   {k}: {metrics[k]:.4f}"
                if isinstance(metrics[k], float)
                else f"   {k}: {metrics[k]}"
            )
    for st in ctx.stats:
        print(
            f"   {st['rule']}: -{st['samples_dropped']} samples,"
            f" -{st['lines_dropped']} lines, -{st['surveys_dropped']} surveys,"
            f" -{st['regions_dropped']} regions"
        )
    return {**metrics, "preds": preds, "rule_stats": ctx.stats, "dpy": dpy}


def freeze_core(v0_preds: pd.DataFrame, caches: Caches) -> None:
    """After v0: freeze CORE (its surviving test regions) and the frozen tail."""
    core = v0_preds.index.to_series()
    core.to_csv(
        caches.exp_dir / "core_regions.csv", index=False, header=["location_id"]
    )
    tail = v0_preds.index[v0_preds["v_test"] > TAIL_THRESHOLD].to_series()
    tail.to_csv(
        caches.exp_dir / "tail_frozen_regions.csv", index=False, header=["location_id"]
    )
    print(f"CORE frozen: {len(core)} regions, tail_frozen: {len(tail)}")
