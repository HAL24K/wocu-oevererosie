#!/usr/bin/env python3
"""
Quick check: Are there ANY discharge (Q) stations in the Waterweb system?
"""

import requests
import json

BASE_URL = "https://ddapi20-waterwebservices.rijkswaterstaat.nl"
CATALOG_ENDPOINT = f"{BASE_URL}/METADATASERVICES/OphalenCatalogus"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": "dummy-key"
}

# Get full catalog with locations
body = {
    "CatalogusFilter": {
        "Compartimenten": True,
        "Grootheden": True,
        "Parameters": True
    }
}

print("Querying Waterweb catalog for discharge (Q) parameter...\n")

try:
    response = requests.post(CATALOG_ENDPOINT, json=body, headers=HEADERS, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Look for Q (discharge) in the catalog
    discharge_entries = []
    
    if "AquoMetadataLijst" in data:
        for item in data["AquoMetadataLijst"]:
            grootheid = item.get("Grootheid", {})
            if grootheid.get("Code") == "Q":
                discharge_entries.append(item)
    
    if discharge_entries:
        print(f"✅ Found {len(discharge_entries)} discharge (Q) entries in catalog!")
        print(f"\nSample discharge entry:")
        print(json.dumps(discharge_entries[0], indent=2))
        
        # Check compartments
        compartments = {entry.get("Compartiment", {}).get("Code") for entry in discharge_entries}
        print(f"\nCompartments with Q: {compartments}")
        
    else:
        print("❌ No discharge (Q) parameter found in catalog")
        print("\nAvailable Grootheden (quantities):")
        if "AquoMetadataLijst" in data:
            grootheden = {item.get("Grootheid", {}).get("Code") for item in data["AquoMetadataLijst"] if "Grootheid" in item}
            for g in sorted(grootheden)[:20]:
                print(f"  - {g}")
            print(f"  ... and {len(grootheden) - 20} more")
    
except Exception as e:
    print(f"❌ Error: {e}")
