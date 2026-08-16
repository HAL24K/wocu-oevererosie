"""WFS Data Bundler - Orchestrate WFS data collection across multiple regions."""

import logging
import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

import src.legacy.data.config as DATA_CONFIG
import src.legacy.data.data_collector as DC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WFSDataBundler:
    """
    Orchestrates WFS data collection across multiple prediction regions.

    This class handles:
    - Fetching WFS data for multiple regions via DataCollector
    - Aggregating data across regions
    - Deduplicating geometries that span multiple regions
    - Saving to GeoPackage with proper layer naming
    - Checkpoint/resume support for long-running operations

    Typical workflow:
        bundler = WFSDataBundler(scope_regions, config)
        bundler.fetch_all_regions(test_mode=True, num_test=10)
        bundler.save_to_geopackage(output_path)
    """

    def __init__(
        self,
        scope_regions: gpd.GeoDataFrame,
        config: DATA_CONFIG.DataConfiguration,
        wfs_timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the WFS data bundler.

        Args:
            scope_regions: GeoDataFrame with prediction regions (polygons)
            config: DataConfiguration with WFS services and buffer settings
            wfs_timeout: Timeout per WFS request in seconds (default: 30)
            max_retries: Number of retry attempts per service (default: 3)
        """
        self.scope_regions = scope_regions
        self.config = config
        self.wfs_timeout = wfs_timeout
        self.max_retries = max_retries

        # Storage for fetched data
        self.wfs_data_collection = {}  # {service: {layer: [gdf1, gdf2, ...]}}
        self.scope_region_ids = []
        self.successful_regions = []
        self.failed_regions = []

        logger.info(f"Initialized WFSDataBundler with {len(scope_regions)} regions")

    def fetch_all_regions(
        self,
        test_mode: bool = False,
        num_test: int = 10,
        show_progress: bool = True,
    ) -> tuple[dict, list, list, list]:
        """
        Fetch WFS data for all (or subset of) scope regions.

        Args:
            test_mode: If True, only process first N regions (default: False)
            num_test: Number of regions to process in test mode (default: 10)
            show_progress: Show progress bar (default: True)

        Returns:
            Tuple of (wfs_data_collection, scope_region_ids, successful_regions, failed_regions)
        """
        # Determine which regions to process
        if test_mode:
            regions_to_process = self.scope_regions.head(num_test)
            num_regions = num_test
        else:
            regions_to_process = self.scope_regions
            num_regions = len(self.scope_regions)

        logger.info(f"Processing {num_regions} scope regions...")
        logger.info(
            f"DataCollector: {self.wfs_timeout}s timeout, {self.max_retries} retries per service"
        )
        if test_mode:
            logger.warning(f"TEST MODE: Processing only first {num_regions} regions")

        # Initialize collections
        self.wfs_data_collection = {}
        self.scope_region_ids = []
        self.successful_regions = []
        self.failed_regions = []

        # Setup progress tracking
        iterator = regions_to_process.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=num_regions, desc="Fetching WFS data")

        # Loop through each scope region
        for idx, row in iterator:
            region_geom = row.geometry
            region_id = row.get("position_id", row.get("location_id", f"region_{idx}"))
            self.scope_region_ids.append(region_id)

            try:
                # Create DataCollector for this region
                data_collector = DC.DataCollector(
                    source_shape=region_geom,
                    source_epsg_crs=self.scope_regions.crs.to_epsg(),
                    buffer_in_metres=self.config.prediction_region_buffer,
                    wfs_services=self.config.known_wfs_services,
                    wfs_timeout=self.wfs_timeout,
                    max_retries=self.max_retries,
                )

                # Fetch data from all WFS services
                data_collector.get_data_from_all_wfs()

                # Extract and store the data
                for (
                    service_name,
                    layers_dict,
                ) in data_collector.relevant_geospatial_data.items():
                    if service_name not in self.wfs_data_collection:
                        self.wfs_data_collection[service_name] = {}

                    for layer_name, gdf in layers_dict.items():
                        if layer_name not in self.wfs_data_collection[service_name]:
                            self.wfs_data_collection[service_name][layer_name] = []

                        # Add this region's data to the collection
                        if gdf is not None and len(gdf) > 0:
                            self.wfs_data_collection[service_name][layer_name].append(
                                gdf.copy()
                            )

                self.successful_regions.append(region_id)

            except Exception as e:
                error_type = type(e).__name__
                logger.error(f"Region {region_id} failed: {error_type}")
                self.failed_regions.append((region_id, str(e)[:100]))
                continue

        # Print summary
        self._print_fetch_summary(num_regions)

        return (
            self.wfs_data_collection,
            self.scope_region_ids,
            self.successful_regions,
            self.failed_regions,
        )

    def _print_fetch_summary(self, num_regions: int):
        """Print summary of data collection results."""
        logger.info("Data collection complete!")
        logger.info(f"  Successful: {len(self.successful_regions)}/{num_regions}")
        logger.info(f"  Failed: {len(self.failed_regions)}/{num_regions}")

        if self.failed_regions:
            logger.warning(f"Failed regions ({len(self.failed_regions)}):")
            for region_id, error in self.failed_regions[:5]:  # Show first 5
                logger.warning(f"  - {region_id}: {error[:50]}")
            if len(self.failed_regions) > 5:
                logger.warning(f"  ... and {len(self.failed_regions) - 5} more")

        logger.info("Summary by service:")
        for service_name, layers_dict in self.wfs_data_collection.items():
            logger.info(f"  {service_name}:")
            for layer_name, gdf_list in layers_dict.items():
                total_features = sum(len(gdf) for gdf in gdf_list)
                logger.info(
                    f"    - {layer_name}: {len(gdf_list)} regions, {total_features} total features"
                )

    def save_to_geopackage(
        self,
        output_path: Path,
        add_region_ids: bool = True,
    ) -> list[str]:
        """
        Save bundled WFS data to GeoPackage with deduplication.

        Args:
            output_path: Path to output GeoPackage file
            add_region_ids: Add scope_region_id column to track origin (default: True)

        Returns:
            List of created layer names
        """
        if not self.wfs_data_collection:
            logger.warning("No WFS data to save - did you call fetch_all_regions()?")
            return []

        logger.info(f"Saving WFS data to {output_path.name}...")
        saved_layers = []

        for service_name, layers_dict in self.wfs_data_collection.items():
            logger.info(f"Processing service: {service_name}")

            for layer_name, gdf_list in layers_dict.items():
                layer_output_name = self._save_single_layer(
                    wfs_data_list=gdf_list,
                    service_name=service_name,
                    layer_name=layer_name,
                    output_path=output_path,
                    add_region_ids=add_region_ids,
                )
                if layer_output_name:
                    saved_layers.append(layer_output_name)

        logger.info(f"DONE! Saved {len(saved_layers)} new layers to {output_path.name}")
        logger.info("New layers:")
        for layer in saved_layers:
            logger.info(f"  - {layer}")

        return saved_layers

    def _save_single_layer(
        self,
        wfs_data_list: list[gpd.GeoDataFrame],
        service_name: str,
        layer_name: str,
        output_path: Path,
        add_region_ids: bool = True,
    ) -> str | None:
        """
        Save a single WFS layer with deduplication.

        Handles duplicates: Large features that span multiple scope regions
        will be fetched multiple times. We deduplicate based on geometry WKT.

        Args:
            wfs_data_list: List of GeoDataFrames (one per scope region)
            service_name: Name of the WFS service
            layer_name: Name of the WFS layer
            output_path: Path to the output GeoPackage
            add_region_ids: Add scope_region_id column

        Returns:
            Output layer name if successful, None otherwise
        """
        if not wfs_data_list:
            logger.warning(f"No data to save for {service_name}:{layer_name}")
            return None

        # Add scope_region_id to each dataframe if requested
        if add_region_ids and len(self.scope_region_ids) >= len(wfs_data_list):
            for gdf, region_id in zip(
                wfs_data_list, self.scope_region_ids[: len(wfs_data_list)], strict=False
            ):
                gdf["scope_region_id"] = region_id

        # Combine all dataframes
        combined = pd.concat(wfs_data_list, ignore_index=True)
        original_count = len(combined)

        # Remove duplicates based on geometry WKT
        combined["_geom_wkt"] = combined.geometry.apply(lambda x: x.wkt)
        combined_dedup = combined.drop_duplicates(subset=["_geom_wkt"], keep="first")
        combined_dedup = combined_dedup.drop(columns=["_geom_wkt"])

        duplicates_removed = original_count - len(combined_dedup)

        # Generate hierarchical layer name: {service}/{layer}
        # This format clearly separates service from layer, making parsing reliable
        clean_service = service_name.replace(" ", "_").lower()
        # Keep colons in layer names (e.g., bag:pand) - they're important identifiers
        output_layer_name = f"{clean_service}/{layer_name}"

        # Save to GeoPackage
        combined_dedup.to_file(output_path, layer=output_layer_name, driver="GPKG")

        if duplicates_removed > 0:
            logger.info(
                f"Saved {len(combined_dedup)} features to layer: {output_layer_name} "
                f"(removed {duplicates_removed} duplicates, {duplicates_removed / original_count * 100:.1f}%)"
            )
        else:
            logger.info(
                f"Saved {len(combined_dedup)} features to layer: {output_layer_name}"
            )

        return output_layer_name

    def list_geopackage_layers(self, gpkg_path: Path) -> dict[str, list[str]]:
        """
        List all layers in a GeoPackage, separated by type.

        Args:
            gpkg_path: Path to GeoPackage file

        Returns:
            Dict with 'original' and 'wfs' keys containing layer names
        """
        conn = sqlite3.connect(gpkg_path)
        cursor = conn.cursor()

        # Query the gpkg_contents table
        cursor.execute(
            "SELECT table_name, data_type FROM gpkg_contents ORDER BY table_name"
        )
        layers = cursor.fetchall()
        conn.close()

        # Separate original layers from WFS layers
        original_layers = [name for name, _ in layers if not name.startswith("wfs_")]
        wfs_layers = [name for name, _ in layers if name.startswith("wfs_")]

        return {
            "original": original_layers,
            "wfs": wfs_layers,
            "all": [name for name, _ in layers],
        }

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics about the bundled data.

        Returns:
            Dict with statistics about services, layers, and features
        """
        if not self.wfs_data_collection:
            return {}

        stats = {
            "num_services": len(self.wfs_data_collection),
            "num_regions_processed": len(self.scope_region_ids),
            "num_successful": len(self.successful_regions),
            "num_failed": len(self.failed_regions),
            "services": {},
        }

        for service_name, layers_dict in self.wfs_data_collection.items():
            service_stats = {"num_layers": len(layers_dict), "layers": {}}

            for layer_name, gdf_list in layers_dict.items():
                total_features = sum(len(gdf) for gdf in gdf_list)
                service_stats["layers"][layer_name] = {
                    "num_regions": len(gdf_list),
                    "total_features": total_features,
                }

            stats["services"][service_name] = service_stats

        return stats
