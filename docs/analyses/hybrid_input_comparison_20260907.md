# Hybrid input comparison — raw vs `_nearest` delivery (2026-09-07)

Context: the 7 Sep meeting with Luke surfaced that the pipeline had been
reading `hybrid_model_results_20260710.gpkg` (the **raw** segmentation lines)
while Luke's cleaned variant `hybrid_model_results_nearest_20260710.gpkg` —
`get_clean_line`: keep the line parts nearest the river centre, split at
> 5 m projection gaps — sat in the same folder. This note records what
changes when the pipeline runs on the cleaned file. Runs, scripts and
artifacts are listed at the end.

## Deliveries

| | raw | `_nearest` |
|---|---|---|
| lines / measurable | 91,531 / 70,356 | 165,337 / 112,570 |
| median line length | 102 m | 25 m |
| total length | 8,690 km | 6,602 km (−24 %) |
| regions with lines | 15,764 | 15,282 |
| Luke's preference (all regions) | 13,731 seg / 2,033 height | 13,130 seg / 2,152 height |

`_nearest` is not a subset of raw: split pieces now fall *within* region
polygons where the original long line crossed the polygon edge and was
dropped, so some regions gain material while 482 regions lose all lines.
Preference re-run on the cleaned lines barely moves (over our 5,702 modelled
regions: 90 seg→height, 86 height→seg, 183 regions vanish).

## Method

Two confounds had to be removed before comparing:

1. **Sampling.** The pipeline sampled a fixed 20 points per line (60 in the
   segment step). With a 25 m median line and many 2–12 m fragments this
   would give a 3 m fragment the weight of a 100 m bank line. Added
   length-based sampling (`sample_spacing_m` = 5, `segment_sample_spacing_m`
   = 1.5, `min_samples_per_line` = 3 — the same density as 20/60 per 100 m)
   and used it for **both** files.
2. **Split.** Each run draws its own stratified 20 % test set; with different
   region populations only 65 test regions overlapped. Added `--split-from`
   so `near-fz` inherits the raw-len train/test assignment; 1,086 test
   regions are common.

Runs: `20260907-raw-len` (raw, length sampling), `20260907-near-len`
(nearest, own split), `20260907-near-fz` (nearest, frozen split — the
comparison run).

## Cleaning ladder (surveys dropped, rules in sequence)

| rule | raw-len | near-fz |
|---|---|---|
| structure mask | 1,085 | 1,465 |
| kronkelend (`max_tortuosity_line`) | 1,735 | **13** |
| verkeerde oever | 135 | 85 |
| doolhof (`maze_survey`) | 2,294 | **427** |
| fragment | 2,068 | 2,448 |
| te weinig punten | 676 | 876 |
| tijd_uitschieter | 1,137 | 1,234 |

Luke's `_nearest` step *is* our geometry cleaning: kronkelend and doolhof
become almost no-ops. It does nothing for fragments and temporal outliers,
which increase slightly (fragmented surveys, and inland pieces that used to
die with their wandering parent now survive as clean short fragments).

## Model quality

Own test split:

| | region MAE / tail / p90 / max | segment MAE / tail / R² / skill |
|---|---|---|
| raw-len | 2.51 / 5.34 / 5.83 / 68.5 | 1.10 / 3.01 / 0.51 / +35 % |
| near-fz | 2.45 / 5.64 / 5.55 / 46.2 | 1.07 / 3.27 / 0.51 / +35 % |

On the 1,086 common test regions: region MAE **2.27 vs 2.26**, tail 4.98 vs
5.22, p90 5.39 vs 5.17, max **68.5 vs 39.7**.

Verdict: a wash on average, softer at the extremes, slightly heavier tail.
There is no double gain because the geometry cleaning was already being done
by our rules on the raw file. Modelled regions: 5,920 → 5,750 (−170).

Sampling alone (20260904-r20 → raw-len, same delivery): more strips are
scored (n_test 7,934 → 10,626), absolute segment MAE rises 0.97 → 1.10 while
skill improves +33 % → +35 %; absolute MAEs across sampling schemes are not
comparable, skill is.

## Worst-100 (horizon ranking, `export_worst_predicted_v2.py --run …`)

| | raw-len | near-fz |
|---|---|---|
| SAM / height preferred | 87 / 13 | 85 / 15 |
| no cleaning flag at all | 46 | **63** |
| flagged kronkelend / doolhof / temporal / fragment | 13 / 28 / 43 / 12 | 1 / 6 / 28 / 12 |
| horizon error #1 / #10 / #100 (m/yr) | 41.2 / 11.5 / 3.9 | 28.8 / 11.0 / 3.5 |
| median 1-yr CV error of the top-100 | 8.9 | 6.7 |
| all regions: CV error > 10 m/yr | 279 | 287 |
| CV error by preference (mean, all regions) | height 0.97 · SAM 3.03 | height 0.87 · SAM 3.10 |

Top-100 overlap 42/100; 22 of the raw worst regions are absent from the
nearest delivery; several are genuinely fixed (`rijn_r_6050_6060` 18.6 → 4.6,
`ijssel1_l_4520_4530` 11.8 → 0.0). The former #1 `ijssel1_l_7770_7780`
drops to rank 61 under length-based sampling alone and is absent in near-fz
(its 2026 fragments no longer pass the fragment/min-samples rules, so the
region falls below three usable years).

## What this means

- **The preference problem is not a raw-file artifact.** SAM-preferred
  regions still carry ~3× the error of height-preferred ones on the cleaned
  file, and the worst-100 is still 85 % SAM.
- **Our geometry rules and Luke's `_nearest` overlap almost completely.** Keep
  the rules (they are cheap no-ops on the cleaned file and a safety net on
  any raw delivery); expect kronkelend/doolhof counts near zero from now on.
- **63 of the 100 worst regions carry no flag** on the cleaned file: the
  remaining failures are invisible to every geometry rule we have. The next
  rule has to look at *time and position* — a survey (or fragment) sitting
  far from where the region's own history says the bank is. Nothing in our
  stack or in the hybrid preference contest checks that today.
- **Extremes soften, coverage shrinks a little** (−170 modelled regions).
  Length-based sampling is a keeper regardless of delivery (skill +33 → +35 %
  on the raw file).

Recommendation: switch the default input to the `_nearest` delivery (it is
what Luke's pipeline produces going forward), keep the full rule stack, adopt
length-based sampling as default, and build the temporal-consistency rule as
the next cleaning step — validated on the frozen split.

## Artifacts

- Code: `sample_spacing_m` / `segment_sample_spacing_m` / `min_samples_per_line`
  and `split_from` in `src/pipeline/config.py`; variable-count
  `sample_lines` in `hybrid_prep.py`; `--hybrid-gpkg`, `--sample-spacing-m`,
  `--segment-sample-spacing-m`, `--split-from` in `python -m src.pipeline`.
- `scripts/compare_hybrid_inputs.py` (ladder + metrics side by side),
  `scripts/compare_worst100.py`, `scripts/evaluate/export_worst_predicted_v2.py --run`.
- Runs under `data/03_features/` and `data/04_model_outputs/`
  (`20260907-raw-len`, `20260907-near-len`, `20260907-near-fz`; logs
  `run_<exp>.log`, `worst100_<exp>.log`).
- Worst-100 packages: `data/02_processed/triage/worst_predicted_v2_<run>.{gpkg,pdf}`.
