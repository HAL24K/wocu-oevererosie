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

    # ── parameters ────────────────────────────────────────────────────────────
    n_points: int = 3  # furthest OK points averaged per (region, year)
    test_size: float = 0.20
    seed: int = 42
    start_year: int = 2026
    end_year: int = 2050
    model_name: str = "lgb"  # which bundle model drives the prediction

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
