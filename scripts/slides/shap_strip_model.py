"""SHAP beeswarm of the strip model: which inputs drive its predictions.

Uses LightGBM's own TreeSHAP (``pred_contrib=True``), the same values the shap
package computes, on the test strips the model never saw. Writes to the run's
``figures/``: ``shap_beeswarm.png`` (Dutch labels) and ``shap_importance.csv``
(mean |SHAP| per feature, with meaning and source for the slide's table).

Run: ``PYTHONPATH=. uv run python scripts/slides/shap_strip_model.py --run 20261013-ref [--top 15]``
"""

from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.feature_labels import FEATURES, GROUPS, label  # noqa: E402
from src.pipeline.config import ExperimentConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--top", type=int, default=15, help="features shown in the beeswarm")
args = ap.parse_args()

cfg = ExperimentConfig.from_run(args.run)
model = joblib.load(cfg.model_out_dir / "segment_model.joblib")
X = pd.read_parquet(cfg.model_out_dir / "segment_test_features.parquet")
feats = list(model.booster_.feature_name())
X = X[feats]

contrib = model.booster_.predict(X.to_numpy(), pred_contrib=True)
shap_vals, base = contrib[:, :-1], contrib[:, -1]
# TreeSHAP is additive: the contributions plus the base value give the prediction.
assert np.allclose(shap_vals.sum(1) + base, model.predict(X), atol=1e-6)

imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(shap_vals).mean(0)})
imp["label"] = imp["feature"].map(label)
imp["betekenis"] = imp["feature"].map(lambda f: FEATURES.get(f, ("", "", "", ""))[1])
imp["bron"] = imp["feature"].map(lambda f: FEATURES.get(f, ("", "", "", ""))[2])
imp["groep"] = imp["feature"].map(
    lambda f: GROUPS.get(FEATURES.get(f, ("", "", "", "context"))[3])
)
imp["aandeel"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
cfg.figures_dir.mkdir(parents=True, exist_ok=True)
imp.round(4).to_csv(cfg.figures_dir / "shap_importance.csv", index=False)

# ── beeswarm ──────────────────────────────────────────────────────────────────
top = imp.head(args.top)
fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(top) + 1.4))
rng = np.random.default_rng(0)
cmap = plt.get_cmap("coolwarm")
lim = np.percentile(np.abs(shap_vals), 99.5)
for row, f in enumerate(top["feature"][::-1]):
    j = feats.index(f)
    sv = shap_vals[:, j]
    xv = X[f].to_numpy()
    # colour by the feature's rank, so skewed features still spread over the scale
    rank = pd.Series(xv).rank(pct=True).to_numpy()
    order = rng.permutation(len(sv))
    # vertical jitter grows with local density, as in the shap package
    bins = np.digitize(np.clip(sv, -lim, lim), np.linspace(-lim, lim, 120))
    counts = np.bincount(bins, minlength=122)[bins]
    jitter = rng.uniform(-1, 1, len(sv)) * np.minimum(0.4, 0.04 * np.sqrt(counts))
    categorical = f.endswith("_enc") or f == "is_nvo"  # no low-to-high order
    ax.scatter(
        np.clip(sv[order], -lim, lim),
        row + jitter[order],
        c="#9aa3a6" if categorical else rank[order],
        cmap=None if categorical else cmap,
        s=3,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
ax.axvline(0, color="#999999", lw=0.8)
ax.set_yticks(range(len(top)), top["label"][::-1])
ax.set_xlabel("Invloed op voorspelde snelheid (m/jr)")
ax.set_xlim(-lim, lim)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40, ticks=[0, 1])
cb.ax.set_yticklabels(["laag", "hoog"])
cb.set_label("Waarde van de input (grijs: categorie)", labelpad=-8)
cb.outline.set_visible(False)
fig.tight_layout()
out = cfg.figures_dir / "shap_beeswarm.png"
fig.savefig(out, dpi=200, facecolor="white")
print(imp[["label", "aandeel", "groep"]].head(args.top).to_string(index=False))
print(f"share of the top {args.top}: {top['aandeel'].sum():.0%}; per group:")
print(imp.groupby("groep")["aandeel"].sum().round(3).to_string())
print("→", out)
