# Documentation

## Current

| Document | What it is | Read it when |
|---|---|---|
| [BACKLOG.md](BACKLOG.md) | Numbered open items with diagnoses, written during development | Picking up work. Highest signal per line in the repo |
| [DATA_INVENTORY.md](DATA_INVENTORY.md) | Data sources, provenance, delivery status | Working out where a file came from |
| [FEATURE_GAP_ANALYSIS.md](FEATURE_GAP_ANALYSIS.md) | Model features vs. what asset managers asked for; what is available, derivable, or blocked | Considering a new feature |
| [CENTERLINE_IMPLEMENTATION_PLAN.md](CENTERLINE_IMPLEMENTATION_PLAN.md) | Design notes for the centreline approach | Touching geometry |

The root [README](../README.md) covers setup, the pipeline map and known issues.
[backend/notebooks/README.md](../backend/notebooks/README.md) says which notebook is
the pipeline and which 57 are history.

## legacy/

Phase-1 material. Accurate about the `DataCollector → DataHandler → BaselineErosionModel`
stack that now lives in `backend/src/legacy/`, and **wrong about the current pipeline** —
it predates it.

| Document | Note |
|---|---|
| [legacy/ARCHITECTURE-phase1.md](legacy/ARCHITECTURE-phase1.md) | Was `ARCHITECTURE.md` at the repo root, where it was routinely mistaken for current |
| [legacy/TODO-phase1-wfs.md](legacy/TODO-phase1-wfs.md) | Was `TODO.md`. All 13 open items concern scaling WFS fetching to 12,130 regions — an approach abandoned in favour of pre-bundled GeoPackages |
| [legacy/DATAHANDLER_WORKFLOW.md](legacy/DATAHANDLER_WORKFLOW.md) | How `DataHandler` was meant to be driven |
| [legacy/GEOPACKAGE_LOADING.md](legacy/GEOPACKAGE_LOADING.md) | The GeoPackage loader added to `DataHandler` |

## Where the docs are still thin

- No record of why the target is `v_test` (a velocity) rather than a position or a volume.
- No spec for the concept viewer, so "output suitable for a viewer" cannot be checked.
- The hybrid delivery's `notitie_hybride_model.docx` lives in `backend/data/02_processed/hybrid/`
  and has not been read into any decision here.
