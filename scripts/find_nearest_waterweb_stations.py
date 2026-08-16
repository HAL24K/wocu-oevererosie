#!/usr/bin/env python3
"""
Find nearest active Waterweb stations to study areas.

Strategy:
1. Parse waterweb_locations.csv with coordinates
2. Calculate distance to study area centers
3. Find 10 nearest stations for each area
4. Test each station for recent data availability
"""

import csv
import re
import requests
from datetime import datetime, timedelta
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

# API Configuration
BASE_URL = "https://ddapi20-waterwebservices.rijkswaterstaat.nl"
LATEST_ENDPOINT = f"{BASE_URL}/ONLINEWAARNEMINGENSERVICES/OphalenLaatsteWaarnemingen"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": "dummy-key"
}

# Study area approximate centers (from our current location codes)
STUDY_AREAS = {
    "empel": {
        "name": "Empel (Maas, Km 218-224)",
        "center": (51.740083, 5.332889),  # From shertogenbosch.empel.maas
        "river": "maas"
    },
    "brakel": {
        "name": "Brakel (Waal, Km 951-945)",
        "center": (51.826764, 5.104336),  # From brakel location
        "river": "waal"
    },
    "terwolde": {
        "name": "Terwolde (IJssel, Km 947-953)",
        "center": (52.285151, 6.110531),  # From ijssel.km949p9
        "river": "ijssel"
    }
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in km
    r = 6371
    return c * r


def parse_point_geometry(point_str):
    """Extract lat, lon from 'POINT (lat lon)' string."""
    match = re.match(r'POINT \(([-\d.]+) ([-\d.]+)\)', point_str)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return (lat, lon)
    return None


def load_and_filter_locations(csv_path, study_area_key):
    """Load locations CSV and filter by river name, calculate distances."""
    study_area = STUDY_AREAS[study_area_key]
    center_lat, center_lon = study_area["center"]
    river_name = study_area["river"]
    
    locations = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by river name in location code or name
            location_code = row['LOCATIE_CODE'].lower()
            location_name = row.get('LOCATIE_NAAM', '').lower()
            
            if river_name not in location_code and river_name not in location_name:
                continue
            
            # Parse coordinates
            coords = parse_point_geometry(row['LON_LAT'])
            if not coords:
                continue
            
            lat, lon = coords
            
            # Calculate distance
            distance = haversine_distance(center_lat, center_lon, lat, lon)
            
            locations.append({
                'code': row['LOCATIE_CODE'],
                'name': row.get('LOCATIE_NAAM', ''),
                'coords': coords,
                'distance_km': distance
            })
    
    # Sort by distance
    locations.sort(key=lambda x: x['distance_km'])
    
    return locations


def test_station_activity(location_code, verbose=False):
    """
    Test if a station has recent data (last 30 days).
    Returns: (is_active, latest_timestamp, days_ago)
    """
    body = {
        "LocatieLijst": [{"Code": location_code}],
        "AquoPlusWaarnemingMetadataLijst": [
            {
                "AquoMetadata": {
                    "Compartiment": {"Code": "OW"},
                    "Grootheid": {"Code": "WATHTE"}
                }
            }
        ]
    }
    
    try:
        response = requests.post(LATEST_ENDPOINT, json=body, headers=HEADERS, timeout=10)
        
        if response.status_code == 204:
            return (False, None, None)
        
        response.raise_for_status()
        data = response.json()
        
        if "WaarnemingenLijst" not in data or not data["WaarnemingenLijst"]:
            return (False, None, None)
        
        # Get the most recent measurement from any series
        latest_time = None
        for obs_series in data["WaarnemingenLijst"]:
            if "MetingenLijst" in obs_series and obs_series["MetingenLijst"]:
                timestamp_str = obs_series["MetingenLijst"][0].get("Tijdstip")
                if timestamp_str:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('+01:00', ''))
                    if latest_time is None or timestamp > latest_time:
                        latest_time = timestamp
        
        if latest_time is None:
            return (False, None, None)
        
        # Calculate days ago
        now = datetime.now()
        days_ago = (now - latest_time).days
        
        # Consider active if data within last 30 days
        is_active = days_ago <= 30
        
        return (is_active, latest_time, days_ago)
        
    except Exception as e:
        if verbose:
            print(f"      Error testing {location_code}: {e}")
        return (False, None, None)


def find_active_stations_for_area(study_area_key, max_test=20, max_active=3):
    """
    Find nearest active stations for a study area.
    
    Args:
        study_area_key: Key for STUDY_AREAS dict
        max_test: Maximum number of nearest stations to test
        max_active: Stop after finding this many active stations
    
    Returns:
        List of active stations with details
    """
    print(f"\n{'='*80}")
    print(f"FINDING ACTIVE STATIONS: {STUDY_AREAS[study_area_key]['name']}")
    print(f"{'='*80}")
    
    # Load and filter locations
    csv_path = Path(__file__).parent / "waterweb_locations.csv"
    locations = load_and_filter_locations(csv_path, study_area_key)
    
    print(f"\n📍 Center coordinates: {STUDY_AREAS[study_area_key]['center']}")
    print(f"🔍 Found {len(locations)} locations on {STUDY_AREAS[study_area_key]['river'].upper()} river")
    print(f"🧪 Testing nearest {min(max_test, len(locations))} stations...\n")
    
    active_stations = []
    
    for i, loc in enumerate(locations[:max_test]):
        if len(active_stations) >= max_active:
            print(f"\n✅ Found {max_active} active stations, stopping search")
            break
        
        print(f"   [{i+1:2d}] Testing {loc['code']:40s} ({loc['distance_km']:5.1f} km away)...", end=" ")
        
        is_active, latest_time, days_ago = test_station_activity(loc['code'])
        
        if is_active:
            print(f"✅ ACTIVE ({days_ago} days ago)")
            active_stations.append({
                **loc,
                'latest_time': latest_time,
                'days_ago': days_ago
            })
        elif latest_time:
            print(f"⚠️  OLD DATA ({days_ago} days ago)")
        else:
            print(f"❌ NO DATA")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: Found {len(active_stations)} active stations for {study_area_key.upper()}")
    print(f"{'='*80}")
    
    if active_stations:
        print(f"\n🎯 RECOMMENDED STATIONS:\n")
        for i, station in enumerate(active_stations, 1):
            print(f"   {i}. {station['code']}")
            print(f"      Name: {station['name']}")
            print(f"      Distance: {station['distance_km']:.1f} km")
            print(f"      Latest data: {station['latest_time'].strftime('%Y-%m-%d %H:%M')} ({station['days_ago']} days ago)")
            print()
    else:
        print(f"\n⚠️  No active stations found within {max_test} nearest locations")
        print(f"   Consider:")
        print(f"   - Increasing search radius (max_test parameter)")
        print(f"   - Using nearby active stations from other areas")
        print(f"   - Contacting RWS about station status\n")
    
    return active_stations


def main():
    """Find active stations for all study areas."""
    print("\n" + "#"*80)
    print("# Waterweb Active Station Finder")
    print("# Finding nearest active measurement stations for erosion study areas")
    print("#"*80)
    
    all_results = {}
    
    for area_key in ["empel", "brakel", "terwolde"]:
        active_stations = find_active_stations_for_area(
            area_key, 
            max_test=20,  # Test up to 20 nearest stations
            max_active=3   # Stop after finding 3 active ones
        )
        all_results[area_key] = active_stations
    
    # Final summary
    print("\n" + "#"*80)
    print("# FINAL SUMMARY")
    print("#"*80)
    
    for area_key, stations in all_results.items():
        status = "✅ OK" if stations else "❌ NONE FOUND"
        print(f"\n{STUDY_AREAS[area_key]['name']:40s}: {status}")
        if stations:
            print(f"   Recommended: {stations[0]['code']}")
            print(f"   Distance: {stations[0]['distance_km']:.1f} km, Last data: {stations[0]['days_ago']} days ago")
    
    print("\n" + "#"*80)
    print("\n✅ Next step: Update test_waterweb_api.py with recommended location codes\n")


if __name__ == "__main__":
    main()
