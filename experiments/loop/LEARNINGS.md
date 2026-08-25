# Loop-engineering experiment — what we learned (2026-08-20 → 2026-08-25)

Three tracks, ~50 ledgered variants, every one scored on the same frozen
holdout (`README.md`). Winners graduated into `src/` (commit 169d1dd) and run
with `python -m src.pipeline --experiment X --source hybrid`. Details per track
in `TRACK1_REPORT.md`, `TRACK2_REPORT.md`, `TRACK3_REPORT.md`; every number in
`ledger.csv`.

## The five things that matter

1. **Cleaning beat modelling three to one.** Track 1 (cleaning rules) took
   region MAE 3.99 → 2.27; track 2 (trajectory features, target variants)
   took it 2.27 → 2.13. The model was never the bottleneck — the labels were.
   Every cleaning rule we wrote (tortuosity, near-bank, maze, fragment,
   temporal repair, kribben/kunstwerk/water masks) is a heuristic for a fact
   the SAM and height-model engineers know for certain. **Getting that
   knowledge delivered as flags removes the heuristics and their errors.**
2. **Repair over removal.** Detrended (Theil–Sen) temporal repair recovers
   +511 regions vs the legacy |v|>50 filter (579 → 139 regions dropped).
   Median-referenced repair looked better on paper but clipped genuine
   eroders — rejected under the tail ruler. Coverage stayed 0.936.
3. **The region scalar has a floor.** One number per region carries 0.9–1.5
   m/yr of irreducible representation error on the tail (the oracle test in
   TRACK3). Past that point only *resolution* helps: R=5 segments (≈20 m)
   improve accuracy as they shrink (segment MAE 1.87 → 1.15 at graduation).
4. **Horizons ≥2 years are both the honest ruler and the operational one.**
   1-year increments are dominated by survey noise; forecasting a segment's
   position ≥2 years out from every valid origin gives R² 0.49, skill 0.72
   vs naive, median position error 1.08 m — and it improves automatically
   with every SAM year. H=3 is starved until more years exist.
5. **Negative results worth not repeating**: survey-level increment *targets*
   (sub-year noise; richness belongs in features), huber loss (better mean,
   worse tail), discharge features (corr 0.01), blunt gully-polygon mask
   (5.9 % contamination confirmed, but the mask costs coverage), year-level
   multi-origin training (≤1 qualifying pair per region after year collapse).

## Honest scoreboard (run 20260822-grad, test never touches fitting)

| ruler | before | after |
|---|---|---|
| region, 1-yr, MAE | 4.01 | 2.63 |
| region, 1-yr, tail MAE (v > 2 m/yr) | ~8.3 (demo 9.13) | 5.90 |
| segment (R=5), ≥2-yr horizon, MAE / tail / R² | — | 1.15 / 3.04 / 0.49 |

## What we need from the data owners

- **Height-model engineers**: the `scope_coverage` five-way classification
  at delivery, and a per-line "this is a structure / false bank" flag.
- **SAM engineers**: per-survey quality flags and, above all, more years.
- **GIS**: delivered 2026-08-25 — nationwide kribben + kunstwerken
  (`data/02_processed/structures/structures.gpkg`). Bridges, quays, jetties
  and locks are now masked deterministically instead of guessed.
