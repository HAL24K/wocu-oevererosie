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

Every path has a default matching the 20260617a reference run; override any of
them for a different delivery (e.g. point ``raw_gpkg`` at a newer file).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import src.paths as PATHS


@dataclass
class ExperimentConfig:
    """All inputs and parameters for one pipeline run.

    Args:
        experiment: Name of the run, e.g. ``"20260817a"``. Output folders under
            ``03_features/`` and ``04_model_outputs/`` take this name.
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
    # height model (scripts/prep_structures.py). Culverts and "overig" are kept
    # in the file for QGIS but not masked. Empty tuple → kribben only.
    kunstwerk_categories: tuple[str, ...] = (
        "brug",
        "kade_damwand",
        "steiger_afmeer",
        "sluis_stuw",
    )
    hybrid_gpkg: Path | None = None

    # ── source ────────────────────────────────────────────────────────────────
    # "hybrid" (default since 2026-08-25): the hybrid line delivery with the
    # graduated cleaning stack, trajectory features, honest validation and
    # the segment-horizon artifact (docs/PIPELINE_CHANGES.md).
    # "points": the height-model point cloud — the pre-graduation pipeline,
    # kept unchanged for comparison with the 20260617a reference run.
    source: str = "hybrid"

    # ── parameters ────────────────────────────────────────────────────────────
    # Structure mask: at a groyne the water's edge is the structure flank, not
    # the riverbank. Samples within mask_buffer_m of a structure are dropped
    # before the furthest-N selection, and a (region, date) observation needs
    # at least min_samples_per_obs surviving samples to produce a scalar.
    # 10 m is the empirical elbow (scripts/kribben_padding_sweep.py).
    mask_buffer_m: float = 10.0
    min_samples_per_obs: int = 8  # e8: relaxable once artefacts die upstream
    n_points: int = 3  # furthest OK points averaged per (region, year)
    n_samples: int = 20  # points sampled along each hybrid line
    test_size: float = 0.20
    seed: int = 42
    start_year: int = 2026
    end_year: int = 2050
    model_name: str = "lgb"  # which bundle model drives the prediction

    # ── graduated cleaning stack (hybrid source; TRACK1_REPORT e8 recipe) ─────
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
    segment_R: int = 5  # segments per region for the horizon artifact
    segment_n_samples: int = 60  # dense sampling for segment scalars
    horizon_min_years: int = 2  # forecast horizon of the segment artifact
    build_segments: bool = True  # produce the segment-horizon artifact (hybrid)

    # ── behaviour ─────────────────────────────────────────────────────────────
    export_gpkg: bool = True  # write the (large) output GeoPackage
    resume: bool = False  # reuse existing per-step parquets when present

    _DEFAULTS = {
        "raw_gpkg": "01_raw/erosion/wocu_output_fase2_20260210.gpkg",
        "proc_gpkg": "02_processed/erosion/wocu_post_processed_fase2_20260310.gpkg",
        "scope_gpkg": "01_raw/scope/scope_fase2.gpkg",
        "veg_gpkg": "02_processed/wfs_context/vegetatielegger.gpkg",
        "lu_gpkg": "02_processed/wfs_context/land_use.gpkg",
        "soil_gpkg": "01_raw/soil/BRO_DownloadBodemkaart.gpkg",
        "stations_gpkg": "02_processed/water_stations/water_stations_for_modeling.gpkg",
        "discharge_dir": "water_stations_timeseries/cleaned/discharge",
        "reference_features_v2": "02_processed/erosion/region_features_v2.parquet",
        "signalering_gpkg": "01_raw/scope/20260205_signaleringslijn.gpkg",
        "structures_gpkg": "02_processed/structures/structures.gpkg",
        "hybrid_gpkg": "02_processed/hybrid/hybrid_model_results_20260710.gpkg",
    }

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        for name, rel in self._DEFAULTS.items():
            if getattr(self, name) is None:
                setattr(self, name, self.data_dir / rel)
            else:
                setattr(self, name, Path(getattr(self, name)))

    # ── derived locations ─────────────────────────────────────────────────────

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "03_features" / self.experiment

    @property
    def model_out_dir(self) -> Path:
        return self.data_dir / "04_model_outputs" / self.experiment

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
