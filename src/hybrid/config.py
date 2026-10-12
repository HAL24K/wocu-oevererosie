"""Hybrid stage configuration: which raw inputs, which SAM cleaning, where to write.

Usage::

    from src.hybrid.config import HybridConfig
    from src.hybrid.run import build_hybrid

    build_hybrid(HybridConfig(name="20261013-rules", sam_rules=True))

With no cleaning options the stage reproduces Luke's ``_nearest`` hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import src.paths as PATHS


@dataclass
class HybridConfig:
    """All inputs and options for building one hybrid line set.

    Args:
        name: Folder name under ``03_hybrid/``, e.g. ``"20261013-rules"``.
    """

    name: str
    data_dir: Path = field(default_factory=lambda: PATHS.DATA_DIR)

    # ── inputs (None → default under data_dir) ───────────────────────────────
    sam_path: Path | None = None
    height_gpkg: Path | None = None
    legger_gpkg: Path | None = None
    structures_gpkg: Path | None = None

    # ── step 1: Luke's nearest-crossing cut of the raw SAM lines ─────────────
    # Name of an earlier hybrid whose nearest.gpkg to reuse. Step 1 is the slow
    # part and depends only on the SAM file and the scope, so most hybrids can
    # share it. None → run step 1.
    nearest_from: str | None = None

    # ── SAM cleaning between step 1 and the points count (all off = Luke) ────
    sam_rules: bool = False  # the model stage's cleaning rule stack
    side_channel_m: float | None = None  # mask the RWS side-channel register (+ buffer)
    jetty_m: float | None = None  # mask this many metres around mooring jetties
    sample_spacing_m: float = 5.0  # sampling for the rule stack
    # Regions that lose all their SAM pieces to cleaning take their height lines
    # instead of dropping out of the hybrid.
    height_fallback: bool = True

    # ── execution ─────────────────────────────────────────────────────────────
    workers: int = 8
    limit: int | None = None  # first N regions only (smoke test)

    _DEFAULTS = {
        "sam_path": "01_raw/sam/oeverlijn_segmentatie_full_aoi.geojson",
        "height_gpkg": "01_raw/height_model/wocu_output_fase2_20260310.gpkg",
        "legger_gpkg": "01_raw/scope/20260107_Nevengeulen_RWS_Legger.gpkg",
        "structures_gpkg": "02_context/structures.gpkg",
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
    def out_dir(self) -> Path:
        return self.data_dir / "03_hybrid" / self.name

    @property
    def hybrid_gpkg(self) -> Path:
        """Layers ``lines`` (winner), ``lines_all`` (both models), ``model_preference``."""
        return self.out_dir / "hybrid.gpkg"

    @property
    def nearest_gpkg(self) -> Path:
        """Step 1 output: SAM pieces per region, before any of our cleaning."""
        return self.out_dir / "nearest.gpkg"

    @property
    def nearest_source(self) -> Path:
        """Where step 2 reads the nearest pieces from."""
        if self.nearest_from is None:
            return self.nearest_gpkg
        return self.data_dir / "03_hybrid" / self.nearest_from / "nearest.gpkg"

    @property
    def cleans_sam(self) -> bool:
        return (
            self.sam_rules
            or self.side_channel_m is not None
            or self.jetty_m is not None
        )

    def inputs(self) -> dict[str, Path]:
        """Input files this configuration actually reads."""
        out = {"height_gpkg": self.height_gpkg}
        if self.nearest_from is None:
            out["sam_path"] = self.sam_path
        else:
            out["nearest_gpkg"] = self.nearest_source
        if self.side_channel_m is not None:
            out["legger_gpkg"] = self.legger_gpkg
        if self.sam_rules or self.jetty_m is not None:
            out["structures_gpkg"] = self.structures_gpkg
        return out

    def missing_inputs(self) -> list[Path]:
        return [p for p in self.inputs().values() if not p.exists()]

    def settings(self) -> dict:
        """Every option that is not an input path."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._DEFAULTS and f.name not in ("name", "data_dir")
        }
