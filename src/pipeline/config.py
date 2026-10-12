"""Experiment configuration.

Everything the pipeline needs to run lives here: input paths, parameters, and
the derived output locations. This replaces the config cell of the master
notebook — same defaults, same single ``experiment`` string driving the output
folders — so a run is reproducible from an object rather than from whatever a
notebook cell happened to contain.

Usage::

    from src.pipeline.config import ExperimentConfig
    from src.pipeline.run import run

    cfg = ExperimentConfig(experiment="20260817a")
    result = run(cfg)

Every default reproduces the reference run (``docs/runs/README.md``): Luke's
``_nearest`` hybrid of 10 Jul, length-based sampling and the frozen test split.
Override any input; ``hybrid="<name>"`` takes the hybrid line set that
``python -m src.hybrid --name <name>`` wrote to ``03_hybrid/<name>/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import src.paths as PATHS


@dataclass
class ExperimentConfig:
    """All inputs and parameters for one pipeline run.

    Args:
        experiment: Name of the run, e.g. ``"20261012-hm0310"``. Its outputs go
            to ``04_runs/<experiment>/`` (``features/``, ``model/``, ``run.json``).
        data_dir: Root of the data tree. Defaults to ``src.paths.DATA_DIR``.

    Paths left as ``None`` resolve to the reference-run defaults under
    ``data_dir`` in ``__post_init__``.
    """

    experiment: str
    data_dir: Path = field(default_factory=lambda: PATHS.DATA_DIR)

    # ── inputs (None → default under data_dir) ───────────────────────────────
    raw_gpkg: Path | None = None
    proc_gpkg: Path | None = None
    scope_gpkg: Path | None = None
    veg_gpkg: Path | None = None
    lu_gpkg: Path | None = None
    soil_gpkg: Path | None = None
    stations_gpkg: Path | None = None
    discharge_dir: Path | None = None
    reference_features_v2: Path | None = None
    signalering_gpkg: Path | None = None
    signalering_layer: str = "Vlak_vrije_ruimte_natuurvriendelijke_oever_ln"
    structures_gpkg: Path | None = None
    structures_layer: str = "kribben"
    kunstwerken_layer: str = "kunstwerken"
    # Categories of the kunstwerken layer that produce a false waterline in the
    # height model (scripts/data_prep/prep_structures.py). Culverts and "overig" are kept
    # in the file for QGIS but not masked. Empty tuple → kribben only.
    kunstwerk_categories: tuple[str, ...] = (
        "brug",
        "kade_damwand",
        "steiger_afmeer",
        "sluis_stuw",
    )
    hybrid_gpkg: Path | None = None
    hybrid: str | None = None  # name under 03_hybrid/ → sets hybrid_gpkg

    # ── parameters ────────────────────────────────────────────────────────────
    # Structure mask: at a groyne the water's edge is the structure flank, not
    # the riverbank. Samples within mask_buffer_m of a structure are dropped
    # before the furthest-N selection, and a (region, date) observation needs
    # at least min_samples_per_obs surviving samples to produce a scalar.
    # 10 m is the empirical elbow (padding sweep, August 2026).
    mask_buffer_m: float = 10.0
    min_samples_per_obs: int = 8  # e8: relaxable once artefacts die upstream
    n_points: int = 3  # height-model points per region that fix the bank side
    n_samples: int = 20  # points sampled along each hybrid line
    # Length-based sampling (2026-09-07): one sample per sample_spacing_m of
    # line instead of a fixed count per line, so a 3 m fragment no longer
    # carries the weight of a 100 m bank line. None → fixed n_samples.
    sample_spacing_m: float | None = 5.0
    segment_sample_spacing_m: float | None = 1.5
    min_samples_per_line: int = 3
    # Train/test label per region (default: the reference run's test regions),
    # so runs on different inputs are graded on the same regions. Regions it
    # does not know are split at random. fresh_split=True ignores it.
    split_from: Path | None = None
    fresh_split: bool = False
    # Quality gate on Luke's height-model reliability. "height_preferred"
    # (default since 2026-10-13, run 20261013-gate-sam) applies it only where
    # the hybrid prefers the height model; SAM-preferred regions come in
    # whatever the height model says. "all" = the gate on every region.
    quality_gate: str = "height_preferred"
    side_channels: bool = False  # admit ngeul_* regions (only with height_preferred)
    # CSV with a location_id column: regions left out of training and testing
    # (experiments such as "without the 100 worst SAM regions").
    exclude_regions: Path | None = None
    test_size: float = 0.20
    seed: int = 42
    start_year: int = 2026
    end_year: int = 2050
    model_name: str = "lgb"  # which bundle model drives the prediction

    # ── graduated cleaning stack (TRACK1_REPORT e8 recipe) ──────────────────────
    water_mask: bool = True  # vegetatielegger Water parts without a centreline
    tortuosity_max: float = 3.0
    nearbank_frac: float = 0.25
    nearbank_min_ref: float = 30.0
    maze_max_ratio: float = 1.8
    maze_min_iqr: float = 20.0
    fragment_min_cov: float = 0.25
    temporal_max_dev: float = 15.0
    temporal_min_surveys: int = 3
    temporal_protect_years: int = 3
    farbank_v_limit: float = 50.0

    # ── graduated modelling (TRACK2/TRACK3 winners) ───────────────────────────
    trajectory_features: bool = True  # traj2 history descriptors (hybrid only)
    val_frac: float = 0.15  # honest early stopping: grouped val split, never test
    segment_R: int = (
        20  # segments per region for the horizon artifact (≈ 5 m; chosen 2026-08-26)
    )
    segment_n_samples: int = 60  # dense sampling for segment scalars
    horizon_min_years: int = 2  # forecast horizon of the segment artifact
    build_segments: bool = True  # produce the segment-horizon artifact (hybrid)

    # ── behaviour ─────────────────────────────────────────────────────────────
    export_gpkg: bool = True  # write the (large) output GeoPackage
    resume: bool = False  # reuse existing per-step parquets when present

    _DEFAULTS = {
        "raw_gpkg": "01_raw/height_model/wocu_output_fase2_20260310.gpkg",
        "proc_gpkg": "01_raw/luke_postprocessed/wocu_post_processed_fase2_20260310.gpkg",
        "scope_gpkg": "01_raw/scope/scope_fase2.gpkg",
        "veg_gpkg": "02_context/vegetatielegger.gpkg",
        "lu_gpkg": "02_context/land_use.gpkg",
        "soil_gpkg": "01_raw/soil/BRO_DownloadBodemkaart.gpkg",
        "stations_gpkg": "02_context/water_stations_for_modeling.gpkg",
        "discharge_dir": "02_context/discharge",
        "reference_features_v2": "02_context/region_features_v2.parquet",
        "signalering_gpkg": "01_raw/scope/20260205_signaleringslijn.gpkg",
        "structures_gpkg": "02_context/structures.gpkg",
        "hybrid_gpkg": "03_hybrid/luke_20260710/hybrid_model_results_nearest_20260710.gpkg",
        "split_from": "02_context/frozen_split_20260907.parquet",
    }

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.hybrid is not None:
            if self.hybrid_gpkg is not None:
                raise ValueError("give hybrid or hybrid_gpkg, not both")
            self.hybrid_gpkg = self.data_dir / "03_hybrid" / self.hybrid / "hybrid.gpkg"
        for name, rel in self._DEFAULTS.items():
            if getattr(self, name) is None:
                setattr(self, name, self.data_dir / rel)
            else:
                setattr(self, name, Path(getattr(self, name)))
        if self.exclude_regions is not None:
            self.exclude_regions = Path(self.exclude_regions)

    @classmethod
    def from_run(cls, experiment: str, data_dir: Path | None = None, **overrides):
        """The configuration a finished run used, read from its ``run.json``.

        Scripts that work on a run (exports, galleries, slides) should start
        from this rather than from today's defaults. ``overrides`` that are not
        None win. Input paths that no longer exist fall back to the default.
        """
        import json
        import logging

        data_dir = Path(data_dir or PATHS.DATA_DIR)
        record = json.loads(
            (data_dir / "04_runs" / experiment / "run.json").read_text()
        )
        names = {f.name for f in fields(cls)}
        kwargs: dict = {}
        for name, value in record.get("settings", {}).items():
            if name in names and name not in cls._DEFAULTS and name != "hybrid":
                kwargs[name] = tuple(value) if isinstance(value, list) else value
        for name, entry in record.get("inputs", {}).items():
            if name not in cls._DEFAULTS:
                continue
            path = data_dir / entry["path"]
            if path.exists():
                kwargs[name] = path
            else:
                logging.getLogger(__name__).warning(
                    "%s: recorded %s %s no longer exists; using the default",
                    experiment,
                    name,
                    entry["path"],
                )
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(experiment=experiment, data_dir=data_dir, **kwargs)

    # ── derived locations ─────────────────────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        return self.data_dir / "04_runs" / self.experiment

    @property
    def features_dir(self) -> Path:
        return self.run_dir / "features"

    @property
    def model_out_dir(self) -> Path:
        return self.run_dir / "model"

    @property
    def exports_dir(self) -> Path:
        """GeoPackages, CSVs and PDFs derived from this run (scripts/export, evaluate)."""
        return self.run_dir / "exports"

    @property
    def figures_dir(self) -> Path:
        """Figures and slides made from this run (scripts/slides)."""
        return self.run_dir / "figures"

    @property
    def compare_dir(self) -> Path:
        """Comparisons of this run with another; the challenger owns them."""
        return self.run_dir / "compare"

    @property
    def output_gpkg(self) -> Path:
        return (
            self.model_out_dir
            / f"wocu_{self.model_name}_predictions_{self.experiment}.gpkg"
        )

    @property
    def report_path(self) -> Path:
        return self.model_out_dir / "report.html"

    def missing_inputs(self) -> list[Path]:
        """Input paths that do not exist — check before a long run."""
        return [
            p
            for f in fields(self)
            if f.name in self._DEFAULTS
            for p in [getattr(self, f.name)]
            if not p.exists()
        ]

    def to_table(self) -> dict[str, str]:
        """Flat name → value view, for the run report."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Path):
                try:
                    v = v.relative_to(self.data_dir)
                except ValueError:
                    pass
            out[f.name] = str(v)
        return out
