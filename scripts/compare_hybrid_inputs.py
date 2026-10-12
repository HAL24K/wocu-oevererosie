"""Side-by-side of pipeline runs on different hybrid deliveries / sampling.

For each run: cleaning ladder (from the run log), region model test MAE/tail
(recomputed from the bundle on the run's own test split), segment metrics
(segment_metrics.json), and the funnel counts. Region-level metrics are also
reported on the intersection of test regions so runs on different deliveries
compare like-for-like.

Run: ``uv run python scripts/compare_hybrid_inputs.py 20260904-r20 20260907-raw-len 20260907-near-len``
"""

from __future__ import annotations

import json
import re
import sys

import joblib
import numpy as np
import pandas as pd

from src.pipeline.config import ExperimentConfig

RULE_RE = re.compile(
    r"^\s+(\w+): -(\d+) samples, -(\d+) lines, -(\d+) surveys, -(\d+) regions"
)
LINES_RE = re.compile(r"hybrid lines: (\d+) delivered, (\d+) measurable \((\d+) regions\)")


def ladder(cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    log = cfg.data_dir / "04_model_outputs" / f"run_{cfg.experiment}.log"
    rows, head = [], {}
    if not log.exists():
        return pd.DataFrame(), head
    for line in log.read_text().splitlines():
        m = LINES_RE.search(line)
        if m:
            head = dict(delivered=int(m[1]), measurable=int(m[2]), regions=int(m[3]))
        m = RULE_RE.match(line)
        if m:
            rows.append(
                dict(rule=m[1], samples=int(m[2]), lines=int(m[3]), surveys=int(m[4]), regions=int(m[5]))
            )
    return pd.DataFrame(rows).set_index("rule") if rows else pd.DataFrame(), head


def region_metrics(cfg: ExperimentConfig, restrict=None) -> dict:
    rf = pd.read_parquet(cfg.features_dir / "region_features.parquet")
    bundle = joblib.load(cfg.model_out_dir / "bundle.joblib")
    feats = bundle["config"]["FEATS_LGB"]
    model = joblib.load(cfg.model_out_dir / "model_lgb.joblib")
    test = rf[rf["split"] == "test"]
    if restrict is not None:
        test = test[test.index.isin(restrict)]
    err = np.abs(model.predict(test[feats].astype(float)) - test["v_test"])
    tail = test["v_test"].abs() > 2
    naive = np.abs(rf[rf["split"] == "train"]["v_test"].mean() - test["v_test"])
    return dict(
        n_regions=len(rf),
        n_test=len(test),
        mae=float(err.mean()),
        tail_mae=float(err[tail].mean()) if tail.any() else np.nan,
        n_tail=int(tail.sum()),
        naive_mae=float(naive.mean()),
        p90_err=float(err.quantile(0.9)),
        max_err=float(err.max()),
    )


def main(runs: list[str]) -> None:
    cfgs = {r: ExperimentConfig(experiment=r) for r in runs}
    lads, heads = {}, {}
    for r, cfg in cfgs.items():
        lads[r], heads[r] = ladder(cfg)

    print("== delivery ==")
    print(pd.DataFrame(heads).T.to_string())

    print("\n== cleaning ladder: surveys dropped per rule (applied in sequence) ==")
    surveys = pd.DataFrame({r: lads[r]["surveys"] for r in runs if not lads[r].empty})
    print(surveys.to_string())
    print("\n== cleaning ladder: lines dropped per rule ==")
    print(pd.DataFrame({r: lads[r]["lines"] for r in runs if not lads[r].empty}).to_string())

    print("\n== region model (own test split) ==")
    rm = {r: region_metrics(cfg) for r, cfg in cfgs.items()}
    print(pd.DataFrame(rm).T.round(3).to_string())

    tests = [
        set(pd.read_parquet(cfg.features_dir / "region_features.parquet").query("split == 'test'").index)
        for cfg in cfgs.values()
    ]
    common = set.intersection(*tests)
    print(f"\n== region model on the {len(common)} test regions common to all runs ==")
    rmc = {r: region_metrics(cfg, restrict=common) for r, cfg in cfgs.items()}
    print(pd.DataFrame(rmc).T[["n_test", "mae", "tail_mae", "n_tail", "naive_mae", "p90_err", "max_err"]].round(3).to_string())

    print("\n== segment model (segment_metrics.json) ==")
    sm = {}
    for r, cfg in cfgs.items():
        p = cfg.model_out_dir / "segment_metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            d["skill"] = 1 - d["test_mae"] / d["naive_mae"]
            sm[r] = d
    cols = ["n_train", "n_test", "test_mae", "test_tail_mae", "tail_n", "test_r2", "naive_mae", "skill", "pos_err_median_m", "pos_err_p90_m"]
    print(pd.DataFrame(sm).T[cols].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["20260904-r20", "20260907-raw-len", "20260907-near-len"])
