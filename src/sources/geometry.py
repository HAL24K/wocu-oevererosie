"""Scope-region geometry lookup.

One place that knows how to find the centreline and the scope polygon for a
``location_id``, regardless of which delivery the geometry came from.

This exists because the same two chores were being redone in several places:

  * the region key is called ``position_id`` in ``vlakken_scope`` and
    ``centrelines`` but ``location_id`` everywhere else, and the rename was
    implemented twice (``erosion.centerline_utils.ensure_location_id_column``
    and ``pipeline.region_split._ensure_location_id``);
  * the hybrid delivery covers scope regions that the phase-2 scope file does
    not, so geometry has to be looked up across more than one file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)

#: Canonical name of the scope-region key.
LOCATION_ID = "location_id"

#: Names the key goes by in the delivered GeoPackages.
LOCATION_ID_ALIASES = ("position_id", "scope_region_id")

DEFAULT_CRS = 28992


def normalise_location_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rename whichever alias the delivery used to ``location_id``.

    Returns the frame unchanged if it already has a ``location_id`` column.
    Does not mutate the input.
    """
    if LOCATION_ID in gdf.columns:
        return gdf
    for alias in LOCATION_ID_ALIASES:
        if alias in gdf.columns:
            return gdf.rename(columns={alias: LOCATION_ID})
    return gdf


class ScopeGeometry:
    """Centrelines and scope polygons keyed by ``location_id``.

    Args:
        centreline_gpkg: GeoPackage holding a centreline layer.
        centreline_layer: Layer name within it.
        polygon_gpkg: Optional GeoPackage holding scope polygons. Defaults to
            ``centreline_gpkg`` when omitted.
        polygon_layer: Layer name for the polygons.
        crs: EPSG code everything is reprojected to (RD New by default).

    Notes:
        Loading is lazy — nothing is read until a lookup is made, so building
        this object is cheap.
    """

    def __init__(
        self,
        centreline_gpkg: Path,
        centreline_layer: str = "centrelines",
        polygon_gpkg: Path | None = None,
        polygon_layer: str = "vlakken_scope",
        crs: int = DEFAULT_CRS,
    ) -> None:
        self.centreline_gpkg = Path(centreline_gpkg)
        self.centreline_layer = centreline_layer
        self.polygon_gpkg = Path(polygon_gpkg) if polygon_gpkg else self.centreline_gpkg
        self.polygon_layer = polygon_layer
        self.crs = crs
        self._centrelines: gpd.GeoSeries | None = None
        self._polygons: gpd.GeoSeries | None = None

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self, gpkg: Path, layer: str) -> gpd.GeoSeries:
        gdf = normalise_location_id(gpd.read_file(gpkg, layer=layer))
        if LOCATION_ID not in gdf.columns:
            raise KeyError(
                f"Layer {layer!r} in {gpkg.name} has no {LOCATION_ID} column and none "
                f"of the known aliases {LOCATION_ID_ALIASES}."
            )
        if gdf.crs is not None and gdf.crs.to_epsg() != self.crs:
            gdf = gdf.to_crs(self.crs)
        # A region may appear more than once (the 20260330 delivery repeats
        # scope rows per type_oever); the geometry is the same, so keep the first.
        gdf = gdf.drop_duplicates(subset=[LOCATION_ID])
        return gdf.set_index(LOCATION_ID)["geometry"]

    @property
    def centrelines(self) -> gpd.GeoSeries:
        """Centreline geometry per ``location_id``."""
        if self._centrelines is None:
            self._centrelines = self._load(self.centreline_gpkg, self.centreline_layer)
            logger.info("Loaded %d centrelines", len(self._centrelines))
        return self._centrelines

    @property
    def polygons(self) -> gpd.GeoSeries:
        """Scope polygon geometry per ``location_id``."""
        if self._polygons is None:
            self._polygons = self._load(self.polygon_gpkg, self.polygon_layer)
            logger.info("Loaded %d scope polygons", len(self._polygons))
        return self._polygons

    # ── lookup ────────────────────────────────────────────────────────────────

    def has_centreline(self, location_id: str) -> bool:
        return location_id in self.centrelines.index

    def covered(self, location_ids) -> set[str]:
        """Subset of ``location_ids`` that have a centreline."""
        return set(location_ids) & set(self.centrelines.index)

    def missing(self, location_ids) -> set[str]:
        """Subset of ``location_ids`` with no centreline.

        The hybrid delivery covers ~4,200 regions absent from the phase-2 scope
        file, mostly side channels (``ngeul_*``). Those cannot be turned into a
        centreline distance and are reported rather than silently dropped.
        """
        return set(location_ids) - set(self.centrelines.index)
