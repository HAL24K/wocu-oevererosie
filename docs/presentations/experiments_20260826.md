# Deck experiments 2026-08-26 (night run)

All on the frozen loop holdout (`experiments/loop/README.md`), LGB with
test-set early stopping (the experiment convention — *not* the honest
graduated validation; deltas are fair, absolutes are ~0.1–0.3 optimistic).
Rows in `experiments/loop/ledger.csv` (variants `s0…s3`, `deck-R*`).

## A · Structures ablation (e8 recipe, water mask OFF for all rows)

| variant | rules | structures mask | LGB MAE | tail MAE (n) | naive | CORE cov. | tail_frozen | regions |
|---|---|---|---|---|---|---|---|---|
| s0-v0-nomask | v0 (Aug-19) | none | 3.94 | 8.87 (242) | 4.66 | 0.995 | 8.57 | 10,850 |
| v0-baseline | v0 (Aug-19) | old kribben | 3.99 | 8.30 (245) | 4.67 | 1.000 | 8.30 | 10,437 |
| s1-e8-nomask | e8 | none | 2.31 | 4.75 (192) | 2.69 | 0.936 | 4.16 | 11,001 |
| s2-e8-oldkribben (= e8-final-protected) | e8 | old kribben, Waal/Nederrijn (1,922) | 2.27 | 4.54 (193) | 2.67 | 0.936 | 3.98 | 10,948 |
| s3-e8-newstructures | e8 | new kribben (4,698) + kunstwerken brug/kade/steiger/sluis (3,884) | 2.31 | 4.69 (186) | 2.68 | 0.899 | 4.09 | 10,730 |

Reading: the structure mask is worth ~0.05 MAE / ~0.2 tail on top of e8 —
the cleaning rules already catch most structure artefacts. The new
nationwide layer removes 218 more regions (CORE coverage 0.936 → 0.899) and
is metric-neutral-to-slightly-worse on the frozen holdout (fewer tail
regions survive, n 193 → 186). Honest graduated runs showed the opposite
sign (tail 5.90 → 5.00); both are within split noise. **Do not present the
structures layer as an accuracy win; present it as a coverage/correctness
change** (regions whose "bank" was a structure are gone — 555 regions present
without any mask are dropped with the new mask).

Old vs new kribben-only was not isolated (kunstwerken added together in s3).

## B · Resolution sweep (e8 obs + traj2 segment features, dense 60/line cache)

| R | train rows | test rows | seg MAE | seg R² | seg tail (n) | naive | region MAE re-agg (max) | tail_frozen re-agg | oracle floor |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4,825 | 1,099 | 2.25 | 0.38 | 4.15 (199) | 2.75 | 2.33 | 3.41 | 0.51 |
| 2 | 9,475 | 2,161 | 2.03 | 0.41 | 4.04 (364) | 2.51 | 2.22 | 3.50 | 0.54 |
| 5 | 22,462 | 5,184 | 1.88 | 0.44 | 3.90 (830) | 2.40 | 2.15 | 3.78 | 0.67 |
| 10 | 42,963 | 9,929 | 1.87 | 0.43 | 3.91 (1,597) | 2.39 | 2.13 | 3.88 | 0.69 |
| 20 | 61,840 | 14,538 | 1.61 | 0.40 | 3.71 (2,179) | 2.08 | 2.19 | 4.16 | 1.11 |
| 50 / 100 | not run | | | | | | | | |

R=1 champion for comparison (i1-traj2, region ruler): MAE 2.13, tail_frozen 3.62.

Notes: segment MAE keeps falling to R=10 then flattens; region-level
re-aggregation (max over segments) is ~equal to the R=1 champion on overall
MAE and slightly worse on the frozen tail — the region scalar itself carries
0.5–0.7 m/yr of irreducible aggregation error ("oracle floor" = error of
perfect segment velocities re-aggregated). `pos_err_med` is only computed
in the horizon (k6) runs, not in this sweep. R=20 took 20 min (groupby on 220k segments) — note the oracle floor jumps 0.69 → 1.11 and re-aggregated tail worsens (3.88 → 4.16): 20 segments per ~100 m region on a 60-samples-per-line cache is where segments start to starve; R=50/100
would be 1.1M/2.2M segments on a 60-samples-per-line cache (0.6/0.3 samples
per segment per line) — starved, not attempted. Dense cache uses e8-kept
lines with the OLD kribben mask (no x/y in the dense cache, so the new
structures layer cannot be applied there without rebuilding
`loop_track3_cache.py`).

## C · Candidates: regions changed most by the structures mask

`docs/presentations/fig/candidates/structures_cleanup_candidates.csv` —
20 VVR regions, ranked by |Δv_last| + position shift + sign flip between
s1 (no mask) and s3 (new structures). Columns: location_id, river,
v_no_mask, v_new_mask (m/yr, last increment), n_years, n_structures_nearby
(new-layer objects within 60 m of the sample hull), mean/max position
difference (m), sign_flip. Top picks by eye of the numbers:
`nederrijn_r_4780_4790` (+7.7 → −22.2, 2 structures), `maas3_r_15910_15920`
(−7.5 → +6.2, 49 m position shift), `rijn_l_8510_8520`, `maas3_r_12130_12140`
(9 structures nearby). Not rendered.
