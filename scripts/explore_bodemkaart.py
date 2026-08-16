#!/usr/bin/env python3
"""Explore BRO Bodemkaart GeoPackage structure (sqlite3 only)."""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sqlite3

def explore_bodemkaart():
    """Analyze the BRO Bodemkaart GeoPackage structure."""
    
    gpkg_path = Path(__file__).parent.parent / "data" / "BRO_DownloadBodemkaart.gpkg"
    
    print("="*80)
    print("BRO BODEMKAART GEOPACKAGE EXPLORATION")
    print("="*80)
    print(f"\nFile: {gpkg_path}")
    print(f"Size: {gpkg_path.stat().st_size / 1024 / 1024:.1f} MB\n")
    
    # Connect with sqlite3 directly (no sqlalchemy needed)
    conn = sqlite3.connect(str(gpkg_path))
    
    # List all layers
    layers_query = """
        SELECT table_name 
        FROM gpkg_contents 
        WHERE data_type IN ('features', 'aspatial', 'attributes')
    """
    layers_df = pd.read_sql_query(layers_query, conn)
    layers = layers_df['table_name'].tolist()
    
    print(f"📦 Total layers: {len(layers)}\n")
    
    # Check which have geometry
    geo_query = "SELECT table_name FROM gpkg_geometry_columns"
    geo_df = pd.read_sql_query(geo_query, conn)
    geo_layers = geo_df['table_name'].tolist()
    table_layers = [l for l in layers if l not in geo_layers]
    
    print(f"🗺️  GEOGRAPHIC LAYERS ({len(geo_layers)}):")
    print("-" * 80)
    for layer in geo_layers:
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer)
            print(f"\n📍 {layer}")
            print(f"   Features: {len(gdf):,}")
            print(f"   CRS: {gdf.crs}")
            print(f"   Geometry type: {gdf.geometry.type.unique()}")
            print(f"   Columns ({len(gdf.columns)}): {', '.join(gdf.columns[:10])}")
            if len(gdf.columns) > 10:
                print(f"              ... and {len(gdf.columns) - 10} more")
            
            # Show first few rows
            print(f"\n   Sample data:")
            sample = gdf.head(3).drop(columns=['geometry'])
            print(sample.to_string(index=False, max_colwidth=30))
        except Exception as e:
            print(f"\n📍 {layer}")
            print(f"   ⚠️  Error: {e}")
    
    print(f"\n\n📊 ATTRIBUTE/LOOKUP TABLES ({len(table_layers)}):")
    print("-" * 80)
    for layer in table_layers:
        try:
            # Use read_sql_query instead of read_sql with connection object
            count_df = pd.read_sql_query(f'SELECT COUNT(*) as cnt FROM "{layer}"', conn)
            df = pd.read_sql_query(f'SELECT * FROM "{layer}" LIMIT 5', conn)
            
            print(f"\n📋 {layer}")
            print(f"   Rows: {count_df['cnt'][0]:,}")
            print(f"   Columns ({len(df.columns)}): {', '.join(df.columns[:8])}")
            if len(df.columns) > 8:
                print(f"            ... and {len(df.columns) - 8} more")
            print(f"\n   Sample (first 5 rows):")
            print(df.to_string(index=False, max_colwidth=40))
        except Exception as e:
            print(f"\n📋 {layer}")
            print(f"   ⚠️  Error: {e}")
    
    conn.close()
    
    print("\n" + "="*80)
    print("\n💡 KEY FINDINGS:")
    print("-" * 80)
    print("""
GEOGRAPHIC LAYERS (2):
  - soilarea: Main soil polygons (48k features)
  - areaofpedologicalinterest: Special areas (6k features)

LOOKUP TABLES (14):
  These contain the actual soil descriptions and properties.
  The geographic layers only have IDs that link to these tables.

NEXT: Examine the lookup tables to understand:
  - Which table contains soil type names (sand/clay/peat)?
  - Which columns to use for erosion modeling?
  - How are the tables linked (foreign keys)?
    """)
    
    print("="*80)

if __name__ == "__main__":
    explore_bodemkaart()