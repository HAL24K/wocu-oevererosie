# Track 3 — resolution: consolidated report (2026-08-22)

6 variants (j1–j6) + a post-hoc aggregation ablation, on dense samples
(60/line, e8-kept lines only — cleaning identical to tracks 1–2, only
geometry density changed). Numbers in `ledger.csv`; showcase and per-variant
artefacts under `data/03_features/loop/variants/j*/`.

## Headline: resolution is not a cost — the single scalar was

| | R=2 | R=5 | R=10 |
|---|---|---|---|
| segments eligible | 98% | 93% | 89% |
| segment MAE (traj2) | 2.032 | 1.879 | **1.866** |
| segment R² | 0.406 | **0.442** | 0.426 |
| region re-agg MAE (max) | 2.224 | 2.149 | 2.135 |
| region frozen-tail (max) | **3.495** | 3.785 | 3.884 |
| oracle floor (overall / tail) | 0.54 / 0.92 | 0.67 / 1.32 | 0.69 / 1.47 |

- **Segment-level predictability improves as segments shrink** (MAE 2.03 →
  1.87, best R² of the whole experiment). Each segment tracks its own local
  bank; the R=1 furthest-3 scalar jumps between parts of the region across
  years — the "one artefact ruins the whole line" effect, now measured.
- **j4 (R=2, traj2) sets the experiment's best frozen-tail: 3.495**, beating
  the R=1 champion i1 (3.617) *through* ~0.9 m/yr of aggregation noise.
- **The oracle floor is the deepest finding.** Reconstructing the region
  scalar from perfectly *observed* segment positions already errs 0.5–0.7
  m/yr overall and 0.9–1.5 m/yr on the frozen tail. The single region scalar
  cannot represent fast-eroding regions to better than ~1 m/yr — no R=1
  model can remove representation error. The region ruler undersells every
  segment model; the segment model's real error is below what that ruler can
  measure.

## Aggregation ablation (post-hoc, no retraining)

max is the right operator for the tail at low R (j4: 3.495); p90 wins
overall MAE (R=5: **2.058** vs i1's 2.133) but softens the tail. At higher R,
max over more noisy positions biases upward — if region-level numbers stay
the bridge metric, keep R=2+max for tail reporting or fit a small
calibration on the aggregate. Operationally this vanishes: the VVR alert
("any segment crosses") consumes segment predictions directly.

## Verdicts

| element | verdict |
|---|---|
| dense resampling (60/line) on e8-kept lines | keep — cleaning untouched, geometry sufficient down to R=10 |
| per-segment furthest-3 + own t-triples + year-scale targets | keep (track-2 lesson applied: richness in features, targets at year scale) |
| segment traj2 features + region context (dist_t2_reg, v_train_reg, theil_v_reg, resid_std_reg) | keep — region context dominates importance; segments inherit the regional signal and add local correction |
| segment-level |v|>50 filter (drops segment, never region) | keep |
| R choice | **R=5 default** (20 m segments, 93% eligible, near-best accuracy); R=2+max for the legacy region-tail bridge |
| max aggregation at high R | park — biased; revisit with calibration if region numbers must come from high R |

## Showcase

`variants/j5-R5-traj2/showcase_2027.png`: predicted 2027 bank as a stitched
segment polyline vs the dashed R=1 line, for the frozen-test regions with
the largest predicted per-segment divergence (up to 23 m/yr within one
region). Honest note: the extreme-spread list mixes genuine local variation
with residual segment-level artefacts — that ranking is the natural
segment-level triage/labelling queue for the next cleaning pass.

## Caveats

- Region-level comparisons carry the aggregation floor by construction;
  segment-level metrics are the true capability measure but have no
  pre-experiment baseline to compare against (they *are* the new
  capability).
- Early stopping on the frozen test persists (identical across variants).
- Multiple scope parts / VVR aggregation (Alexander's alert semantics:
  fire when any segment crosses the VVR back) is a signalering-stage
  implementation, deliberately out of experiment scope.

## Experiment totals after three tracks

Frozen holdout, LGB: overall MAE 3.99 → 2.13 (region ruler) with segment
MAE 1.87 beneath it; frozen-tail 8.30 → 3.50 (j4). Against the 20260819
demo risicogevallen (9.13): **2.6× of the 10× commitment**, with the honest
notes that (a) the remaining ruler is partly floored by representation
error the October story should acknowledge, and (b) absolute numbers await
honest-validation re-verification at graduation.
