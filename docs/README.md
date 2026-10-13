# Documentation

| Where | What it is |
|---|---|
| [runs/](runs/README.md) | `ledger.jsonl`: one line per model run and hybrid (inputs, settings, code version, metrics), and the run table |
| [PIPELINE_CHANGES.md](PIPELINE_CHANGES.md) | What changed in the model stage since August, and why |
| [loop_experiment/](loop_experiment/README.md) | The August experiment that produced the cleaning rules, trajectory features and segment model. `LEARNINGS.md` is the summary; `ledger.csv` its runs |
| [analyses/](analyses/) | Dated analysis notes. They describe the code and data as they were on that date |

The data layout is in [data/README.md](../data/README.md); setup and the stage
overview in the root [README](../README.md). Project status, plans and meeting
notes are in Notion (WOCU hub → "Model as built").

## Decisions recorded elsewhere

- **Shipping intensity** was assessed in March and not adopted: RWS publishes
  passages per river block, far coarser than the ~100 m regions.
- **0310 vs 0330 post-processed file.** Luke's 0330 version (Drive, `inputs/luke/`)
  has 213 more regions marked `OK`. The pipeline still reads 0310; switching is open.
