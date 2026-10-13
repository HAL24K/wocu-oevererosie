# Track 2 — multi-t: consolidated report (2026-08-20)

13 variants (g1–g6, h1–h5, i1–i2) on the e8-cleaned observations, frozen
holdout, year-target ruler unless stated. Numbers in `ledger.csv`; per-variant
feature importances under `data/03_features/loop/variants/<name>/`.

## Result

**Graduating: `i1-traj2` — trajectory features from the full survey history,
standard training, year target.**

| | e8 (track-1 exit) | i1-traj2 | delta |
|---|---|---|---|
| test MAE | 2.270 | **2.133** | −6% |
| frozen-tail MAE | 3.976 | **3.617** | −9% |
| R² | 0.306 | 0.375 | |
| CORE coverage | 0.936 | 0.936 | unchanged |

Cumulative since the experiment started: overall MAE 3.99 → 2.13 (−47%),
frozen-tail 8.30 → 3.62 (−56%). Against the 20260819-hybrid demo numbers
(risicogevallen MAE 9.13): **2.5× of the 10× committed** — cleaning + multi-t
together, resolution still untouched.

## What the features are

Per region, computed from every cleaned survey at or before the forecast
origin (leakage-safe: year ≤ t2): Theil–Sen velocity, trend-residual spread,
acceleration (second-half minus first-half slope), history span/count/year
count, gap to the last survey, source mix, and the traj2 additions —
**last residual from trend** (mean-reversion), **date-true recent velocity**,
**recent 3-point slope**. `theil_v` and `v_recent` rank directly behind
`dist_t2` in importance; the March hypothesis ("more t-points is the biggest
win") is confirmed, but the win comes through *features*, not through more
target rows.

## Verdicts per lever

| lever | verdict | evidence |
|---|---|---|
| trajectory features (traj) | **keep** | g2: −0.08 MAE, −0.20 tail vs g1 |
| traj2 (mean-reversion/recency) | **keep** | i1: best frozen-tail of the experiment |
| year-level pairwise rows | optional | g3 helped (+22% rows) with traj; with traj2 the gain is within noise (i2) |
| survey-level increment targets | **park — negative result** | h1–h3 all worse: sub-year velocities carry seasonal noise; the model learns span-dependence that does not transfer to year-scale prediction. The richness belongs in features, not targets |
| huber objective | **park** | best overall MAE of the experiment (h5: 2.068) but tail regresses to 4.08 — rejected under the tail-first ruler; remember it if priorities flip |
| date-true target | **park for October** | its own family (g4–g6): relative skill slightly better than year target (0.79 vs 0.82 of naive), absolute errors higher because true spans are shorter. Right for operational reporting later; wrong ruler for comparable October numbers |

## The uneven-t answer (the question that started this track)

The creative solution that survived testing is *not* to force uneven steps
into a consistent training geometry — it is to make the target trivially
consistent (one horizon-normalized increment per region) and hand the
irregular history to the model as trajectory-shape features. Regions with 3
AHN points and regions with 11 SAM surveys then live in one model, each
described by how much history it actually has.

## Caveats

- Early stopping on the frozen test set — unchanged across all variants,
  deltas fair, absolutes optimistic (same as track 1; re-verify at
  graduation).
- Pairwise rows reuse train regions' earlier increments; the region-grouped
  frozen split keeps them leakage-free vs test.

## Hand-off to track 3 (resolution)

Segment-level datasets will multiply row counts per region; the survey-level
negative result here is a warning to keep segment *targets* at year scale
and put the sub-scale richness into features there too. The i1 feature
builder works per (region, origin) and extends to (region, segment, origin)
directly.
