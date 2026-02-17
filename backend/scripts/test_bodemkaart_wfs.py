#!/usr/bin/env python3
"""Quick test to verify BRO Bodemkaart WFS connection."""

from owslib.wfs import WebFeatureService

def test_bodemkaart_wfs():
    """Test BRO Bodemkaart WFS and list available layers."""
    
    bodemkaart_url = "https://service.pdok.nl/tno/bro-bodemkaart/wfs/v1_0"
    
    print("🔍 Testing BRO Bodemkaart WFS...\n")
    print("="*70)
    print(f"URL: {bodemkaart_url}\n")
    
    try:
        # Connect to WFS
        print("Connecting...")
        wfs = WebFeatureService(bodemkaart_url, version='2.0.0', timeout=30)
        
        print("✅ CONNECTION SUCCESSFUL!\n")
        print("="*70)
        print(f"\n📦 Available layers: {len(wfs.contents)}\n")
        
        # List all layers with details
        for i, (layer_name, layer) in enumerate(wfs.contents.items(), 1):
            print(f"{i}. {layer_name}")
            
            if hasattr(layer, 'title') and layer.title:
                print(f"   Title: {layer.title}")
            
            if hasattr(layer, 'abstract') and layer.abstract:
                abstract = layer.abstract[:150] + "..." if len(layer.abstract) > 150 else layer.abstract
                print(f"   Description: {abstract}")
            
            if hasattr(layer, 'boundingBoxWGS84'):
                bbox = layer.boundingBoxWGS84
                print(f"   BBox: {bbox}")
            
            print()
        
        print("="*70)
        print("\n🎯 RESULT: WFS is working!")
        print(f"\n✅ Add to your WFS services:")
        print(f"""
SWS.WfsService(
    name="bro_bodemkaart",
    url="{bodemkaart_url}",
    version="2.0.0",
    relevant_layers={list(wfs.contents.keys())},
)
""")
        
        return True
        
    except Exception as e:
        print("❌ CONNECTION FAILED")
        print(f"\nError: {type(e).__name__}")
        print(f"Message: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Try adding '?service=WFS' to URL")
        print("   2. Try version='1.1.0' instead of '2.0.0'")
        print("   3. Check firewall/VPN")
        print("   4. Search PDOK page for exact WFS URL")
        
        return False


if __name__ == "__main__":
    success = test_bodemkaart_wfs()
    exit(0 if success else 1)