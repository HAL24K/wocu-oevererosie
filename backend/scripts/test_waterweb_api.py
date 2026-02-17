#!/usr/bin/env python3
"""
Test Waterweb (WADAR) API for water level data at study locations.

Tests:
1. Empel (Maas, Km 218-224): shertogenbosch.empel.maas
2. Brakel (Waal, Km 951-945): loevestein.waal.km951.ro
3. Terwolde (IJssel, Km 947-953): ijssel.km949p9.linkeroever

API Documentation: https://waterwebservices.rijkswaterstaat.nl/
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

# API Configuration
BASE_URL = "https://ddapi20-waterwebservices.rijkswaterstaat.nl"
CATALOG_ENDPOINT = f"{BASE_URL}/METADATASERVICES/OphalenCatalogus"
OBSERVATIONS_ENDPOINT = f"{BASE_URL}/ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen"
LATEST_ENDPOINT = f"{BASE_URL}/ONLINEWAARNEMINGENSERVICES/OphalenLaatsteWaarnemingen"

# Study locations with Waterweb location codes
# Note: These are the nearest ACTIVE stations (verified with find_nearest_waterweb_stations.py)
STUDY_LOCATIONS = {
    "empel": {
        "name": "Empel (Maas, Km 218-224)",
        "code": "shertogenbosch.empel.maas",  # 0.0 km away - perfect match!
        "river": "Maas"
    },
    "brakel": {
        "name": "Brakel (Waal, Km 951-945)",
        "code": "sintandries.waal",  # 17.7 km away upstream - most recent data on Waal (March 2025)
        "river": "Waal"
    },
    "terwolde": {
        "name": "Terwolde (IJssel, Km 947-953)",
        "code": "zutphen.ijssel",  # 15.4 km away - nearest active station
        "river": "IJssel"
    }
}

# Headers
HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": "dummy-key-for-future-compatibility"  # Not required yet but helps with debugging
}


def test_catalog_access():
    """Test 1: Can we access the catalog?"""
    print("\n" + "="*80)
    print("TEST 1: Catalog Access")
    print("="*80)
    
    body = {
        "CatalogusFilter": {
            "Compartimenten": True,
            "Grootheden": True,
            "ProcesTypes": True
        }
    }
    
    try:
        response = requests.post(CATALOG_ENDPOINT, json=body, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Catalog access successful!")
        print(f"   Response keys: {list(data.keys())}")
        
        # Show available compartments and quantities
        if "AquoMetadataLijst" in data:
            compartments = {item.get("Compartiment", {}).get("Code") for item in data["AquoMetadataLijst"] if "Compartiment" in item}
            quantities = {item.get("Grootheid", {}).get("Code") for item in data["AquoMetadataLijst"] if "Grootheid" in item}
            print(f"   Found {len(data['AquoMetadataLijst'])} metadata entries")
            print(f"   Sample compartments: {list(compartments)[:5]}")
            print(f"   Sample quantities: {list(quantities)[:5]}")
            
            # Check if water height (WATHTE) is available
            if "WATHTE" in quantities:
                print(f"   ✅ Water height (WATHTE) available in catalog")
            else:
                print(f"   ⚠️  Water height (WATHTE) not found in quantities")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Catalog access failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text[:500]}")
        return False


def test_latest_observations():
    """Test 2: Get latest observations for all study locations"""
    print("\n" + "="*80)
    print("TEST 2: Latest Water Level Observations")
    print("="*80)
    
    # Build request body with all 3 locations
    location_list = [{"Code": loc["code"]} for loc in STUDY_LOCATIONS.values()]
    
    body = {
        "LocatieLijst": location_list,
        "AquoPlusWaarnemingMetadataLijst": [
            {
                "AquoMetadata": {
                    "Compartiment": {"Code": "OW"},  # Surface water
                    "Grootheid": {"Code": "WATHTE"}  # Water height
                }
            }
        ]
    }
    
    try:
        response = requests.post(LATEST_ENDPOINT, json=body, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Latest observations retrieved!")
        
        if "WaarnemingenLijst" in data:
            print(f"\n   Found {len(data['WaarnemingenLijst'])} observation series:")
            
            for obs_series in data["WaarnemingenLijst"]:
                location_code = obs_series.get("Locatie", {}).get("Code", "Unknown")
                
                # Find human-readable name
                location_name = next(
                    (loc["name"] for loc in STUDY_LOCATIONS.values() if loc["code"] == location_code),
                    location_code
                )
                
                print(f"\n   📍 {location_name}")
                print(f"      Location code: {location_code}")
                
                if "MetingenLijst" in obs_series and obs_series["MetingenLijst"]:
                    latest = obs_series["MetingenLijst"][0]
                    timestamp = latest.get("Tijdstip", "Unknown")
                    value = latest.get("Meetwaarde", {}).get("Waarde_Numeriek", "N/A")
                    value_alpha = latest.get("Meetwaarde", {}).get("Meetwaarde_Waarde_Alfanumeriek", "N/A")
                    
                    print(f"      Latest measurement: {timestamp}")
                    print(f"      Water level: {value} (alpha: {value_alpha})")
                    
                    # Check metadata
                    metadata = latest.get("WaarnemingMetadata", {})
                    quality_obj = metadata.get("Kwaliteitswaardecode", {})
                    if isinstance(quality_obj, dict):
                        quality = quality_obj.get("Code", "Unknown")
                    else:
                        quality = quality_obj if quality_obj else "Unknown"
                    print(f"      Quality code: {quality}")
                else:
                    print(f"      ⚠️  No measurements found")
        else:
            print(f"   ⚠️  No WaarnemingenLijst in response")
            print(f"   Response keys: {list(data.keys())}")
        
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 204:
            print(f"⚠️  No content (204): No data available for these locations")
            print(f"   This might mean the location codes are incorrect or no recent measurements exist")
        else:
            print(f"❌ Latest observations failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text[:500]}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Latest observations failed: {e}")
        return False


def test_historical_data_single_location():
    """Test 3: Get historical data for one location (Empel, past 7 days)"""
    print("\n" + "="*80)
    print("TEST 3: Historical Data - Empel (Last 7 Days)")
    print("="*80)
    
    # Date range: last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    location = STUDY_LOCATIONS["empel"]  # Use Empel - has current data
    
    body = {
        "Locatie": {"Code": location["code"]},
        "AquoPlusWaarnemingMetadata": {
            "AquoMetadata": {
                "Compartiment": {"Code": "OW"},
                "Grootheid": {"Code": "WATHTE"},
                "ProcesType": "meting"  # Only actual measurements, not predictions
            }
        },
        "Periode": {
            "Begindatumtijd": start_date.strftime("%Y-%m-%dT00:00:00.000+01:00"),
            "Einddatumtijd": end_date.strftime("%Y-%m-%dT23:59:59.000+01:00")
        }
    }
    
    try:
        response = requests.post(OBSERVATIONS_ENDPOINT, json=body, headers=HEADERS, timeout=60)
        
        # Handle 204 No Content specially
        if response.status_code == 204:
            print(f"⚠️  No content (204): No data available for this location/period")
            print(f"   Location code '{location['code']}' may not have historical measurements")
            print(f"   Try alternative location codes or contact RWS for available locations")
            return False
        
        response.raise_for_status()
        data = response.json()
        print(f"✅ Historical data retrieved for {location['name']}!")
        
        if "WaarnemingenLijst" in data and data["WaarnemingenLijst"]:
            for obs_series in data["WaarnemingenLijst"]:
                if "MetingenLijst" in obs_series:
                    measurements = obs_series["MetingenLijst"]
                    print(f"\n   📊 Found {len(measurements)} measurements")
                    print(f"   Period: {start_date.date()} to {end_date.date()}")
                    
                    # Show first 3 and last 3
                    print(f"\n   First 3 measurements:")
                    for m in measurements[:3]:
                        time = m.get("Tijdstip", "Unknown")
                        value = m.get("Meetwaarde", {}).get("Waarde_Numeriek", "N/A")
                        print(f"      {time}: {value}")
                    
                    if len(measurements) > 6:
                        print(f"   ... ({len(measurements) - 6} more measurements) ...")
                    
                    print(f"\n   Last 3 measurements:")
                    for m in measurements[-3:]:
                        time = m.get("Tijdstip", "Unknown")
                        value = m.get("Meetwaarde", {}).get("Waarde_Numeriek", "N/A")
                        print(f"      {time}: {value}")
                    
                    # Calculate basic stats
                    values = [m.get("Meetwaarde", {}).get("Waarde_Numeriek") for m in measurements]
                    values = [v for v in values if v is not None]
                    if values:
                        print(f"\n   📈 Statistics (last 7 days):")
                        print(f"      Min: {min(values)}")
                        print(f"      Max: {max(values)}")
                        print(f"      Mean: {sum(values) / len(values):.2f}")
                else:
                    print(f"   ⚠️  No MetingenLijst in observation series")
        else:
            print(f"   ⚠️  No observations found for this period")
            print(f"   Response keys: {list(data.keys())}")
        
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 204:
            print(f"⚠️  No content (204): No data available for this location/period")
        else:
            print(f"❌ Historical data request failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text[:500]}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Historical data request failed: {e}")
        return False


def test_discharge_availability():
    """Test 4: Check if discharge (Q) data is available at all stations"""
    print("\n" + "="*80)
    print("TEST 4: Discharge (Q) Data Availability")
    print("="*80)
    
    print("\nTesting discharge parameter at all 3 study locations...")
    print("(This may take a moment...)\n")
    
    discharge_results = {}
    
    for area_key, location in STUDY_LOCATIONS.items():
        print(f"📍 Testing {location['name']}...")
        print(f"   Location: {location['code']}")
        
        # Try to get latest discharge measurement
        body = {
            "LocatieLijst": [{"Code": location['code']}],
            "AquoPlusWaarnemingMetadataLijst": [
                {
                    "AquoMetadata": {
                        "Compartiment": {"Code": "OW"},  # Surface water
                        "Grootheid": {"Code": "Q"}  # Discharge
                    }
                }
            ]
        }
        
        try:
            response = requests.post(LATEST_ENDPOINT, json=body, headers=HEADERS, timeout=30)
            
            if response.status_code == 204:
                print(f"   ❌ No discharge data available")
                discharge_results[area_key] = False
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if "WaarnemingenLijst" in data and data["WaarnemingenLijst"]:
                # Check if we have actual measurements
                has_data = False
                latest_time = None
                latest_value = None
                
                for obs_series in data["WaarnemingenLijst"]:
                    if "MetingenLijst" in obs_series and obs_series["MetingenLijst"]:
                        has_data = True
                        meting = obs_series["MetingenLijst"][0]
                        latest_time = meting.get("Tijdstip", "Unknown")
                        latest_value = meting.get("Meetwaarde", {}).get("Waarde_Numeriek", "N/A")
                        break
                
                if has_data:
                    print(f"   ✅ Discharge data AVAILABLE")
                    print(f"      Latest: {latest_time}")
                    print(f"      Value: {latest_value} m³/s")
                    discharge_results[area_key] = True
                else:
                    print(f"   ❌ No measurements in response")
                    discharge_results[area_key] = False
            else:
                print(f"   ❌ No discharge data available")
                discharge_results[area_key] = False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            discharge_results[area_key] = False
        
        print()
    
    # Summary
    print("="*80)
    print("DISCHARGE (Q) SUMMARY:")
    print("="*80)
    
    available_count = sum(discharge_results.values())
    
    for area_key, has_discharge in discharge_results.items():
        status = "✅ Available" if has_discharge else "❌ Not available"
        location = STUDY_LOCATIONS[area_key]
        print(f"{location['name']:40s}: {status}")
    
    print(f"\n📊 Result: {available_count}/3 stations have discharge data")
    
    if available_count == 0:
        print("\n⚠️  No discharge data available at any station")
        print("   Recommendation: Focus on water level (WATHTE) for erosion modeling")
        print("   Water level alone can indicate high water events and flooding risk")
    elif available_count < 3:
        print("\n⚠️  Discharge data only partially available")
        print("   Recommendation: Use discharge where available, water level for others")
    else:
        print("\n🎉 Discharge data available at all stations!")
    
    return available_count > 0


def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# Waterweb (WADAR) API Test Suite")
    print("# Testing water level & discharge data access for erosion model")
    print("#"*80)
    
    print("\n📋 Study Locations:")
    for key, loc in STUDY_LOCATIONS.items():
        print(f"   {key.upper():12s}: {loc['name']}")
        print(f"                Location code: {loc['code']}")
    
    # Run tests
    results = {
        "catalog": test_catalog_access(),
        "latest": test_latest_observations(),
        "historical": test_historical_data_single_location(),
        "discharge": test_discharge_availability()
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():15s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Waterweb API is accessible and working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    if all_passed:
        print("1. ✅ Update data_sources_inventory_updated.csv with Waterweb details")
        print("2. ✅ Create DataCollector integration for Waterweb API")
        print("3. ✅ Download historical data (2016-2024) for all 233 scope regions")
        print("4. ✅ Extract high water event frequency + magnitude features")
    else:
        print("1. Fix location codes (some may not exist in Waterweb)")
        print("2. Check if alternative location codes are needed")
        print("3. Contact RWS support if locations should exist but don't")
    print("="*80)


if __name__ == "__main__":
    main()
