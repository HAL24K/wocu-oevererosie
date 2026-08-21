# Track 1 — cleaning rules: consolidated report (2026-08-20)

29 variants over 4 batches + 1 final run. Full per-variant numbers in
`ledger.csv`; visuals per variant under `data/03_features/loop/variants/`.

## Result

**Graduating recipe: `e8-final-protected`** — in order:

| # | rule | params | verdict |
|---|------|--------|---------|
| 1 | structure_mask | kribben, 10 m | keep (already standard) |
| 2 | max_tortuosity_line | tort > 3.0, len > 30 m | keep |
| 3 | near_bank_line | p50 < 0.25 × ref, ref > 30 m | keep (cheap, metric-neutral, removes confirmed mid-channel lines) |
| 4 | maze_survey | len ratio > 1.8 **and** dist-IQR > 20 m | keep — IQR condition is essential (ratio alone kills duplicated on-bank surveys) |
| 5 | fragment_survey | coverage < 0.25 | keep |
| 6 | min_samples_survey | ≥ 8 (was 12) | keep — relaxable once artefacts are handled upstream |
| 7 | temporal_outlier_survey | detrended (Theil–Sen), resid > 15 m, ≥ 3 surveys, never drops a region below 3 years | **keep — the star rule** |
| — | region |v| > 50 filter | keep for now | near-obsolete: 139 exclusions vs 579; retirement candidate at graduation |
| — | multiline_far_line | 25 m / 1.5× | **park** — metric-neutral after median collapse; superseded by temporal repair |

## Numbers (frozen holdout, LGB)

| | v0 baseline | e8 final | shared-region view |
|---|---|---|---|
| test MAE (m/jr) | 3.99 | 2.27 | 2.08 vs 2.05 (e6) |
| frozen-tail MAE | 8.30 | 3.98 | **3.56** (204 shared) |
| CORE coverage | 1.000 | **0.936** | — |
| frozen-tail regions kept | 245 | **219**/245 | |
| total regions with prediction | 10,437 | **10,948** | +511 vs baseline |
| region-filter exclusions | 579 | 139 | |

Overall MAE −43%, tail MAE −57%, **while coverage grew**. The model's skill
ratio vs naive is unchanged (~0.85) — the gains are measurement correction,
which is the point: the artefacts were the measurement error.

## Why the choices are what they are

- **Repair beats removal.** The temporal rule alone (v4) returned ~400
  regions the old |v|>50 filter discarded, and eye-checks confirmed every
  removed survey inspected was on a road, town block, railway or floodplain
  feature — not the bank.
- **Median-referenced repair is a trap.** At dev ≤ 20 it clips the endpoints
  of genuinely fast-eroding regions (proved on synthetic: 10 m/jr eroder
  loses 2 of 5 surveys at dev 15). Detrended (Theil–Sen residual) repair is
  erosion-proof at any strictness; batch-3's better-looking median numbers
  were partly signal erasure and were rejected despite being "better".
- **Maze needs two conditions.** Length ratio > 1.8 alone also fires on
  legitimate surveys with duplicated geometry (eye-check batch 1); adding
  dist-IQR > 20 m halved its coverage cost with minor metric loss.
- **The protective guard is free.** e8 vs e6: same accuracy on shared
  regions, +2.3 pt coverage, temporal rule drops zero regions. The retained
  suspect surveys are exactly identifiable (residual > 15 m but protected) —
  they are the labelling queue for the QGIS pass.

## Coverage gap autopsy (e6 basis, 102 lost CORE regions)

structure_mask 5 (regions that *are* a bridge/structure), tortuosity 15 and
maze 22 (chaotic side-channel/pool shorelines), fragment 23 (near-empty
scraps), temporal 27 (recovered by e8's guard), joint 20. With e8 the gap is
75 regions (6.4%), dominated by genuinely unmeasurable cases —
documented in `variants/e6-coverage-max/coverage_autopsy.png`.

## Caveats

- Early stopping on the frozen test set (inherited from the baseline
  pipeline, identical across variants): absolute numbers are optimistic;
  deltas are fair. Re-verify with honest validation at graduation.
- Cleaning changes v_test itself; all cross-variant claims above use fixed
  denominators (CORE / shared-region intersections).
- Kribben mask still covers only Waal + Nederrijn-Lek (IJssel/Maas kribben
  and the bridges layer are with the GIS engineers) — the structure_mask
  slot absorbs both the moment they arrive.

## Hand-off to track 2 (multi-t)

The detrended Theil–Sen machinery built for the repair rule is the same
primitive the multi-t target needs (A7 spec). The cleaned sample table and
the harness (frozen split, cached features, ledger) carry over unchanged;
track 2 swaps the target/split builder, not the data.
