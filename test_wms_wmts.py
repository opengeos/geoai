#!/usr/bin/env python
"""
Test script for WMS/WMTS segmentation functionality.
Tests the core components without requiring SAM3 authentication.
"""

import leafmap
from pathlib import Path
import sys

print("=" * 60)
print("Testing WMS/WMTS Segmentation Components")
print("=" * 60)

# Test 1: Download imagery from tile service
print("\n[1/3] Testing tile download from basemap...")
try:
    bbox = [-122.2625, 37.8685, -122.2535, 37.8755]  # UC Berkeley area
    output_path = "/tmp/test_wms_imagery.tif"
    
    leafmap.map_tiles_to_geotiff(
        output=output_path,
        bbox=bbox,
        zoom=18,
        source="Esri.WorldImagery",
        overwrite=True,
    )
    
    if Path(output_path).exists():
        print(f"✓ Successfully downloaded tiles to {output_path}")
        
        # Check file size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
    else:
        print("✗ Failed to download tiles")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error downloading tiles: {e}")
    sys.exit(1)

# Test 2: Verify raster can be read
print("\n[2/3] Testing raster reading...")
try:
    import rasterio
    with rasterio.open(output_path) as src:
        print(f"✓ Raster opened successfully")
        print(f"  Dimensions: {src.width} x {src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bands: {src.count}")
        print(f"  Bounds: {src.bounds}")
except Exception as e:
    print(f"✗ Error reading raster: {e}")
    sys.exit(1)

# Test 3: Verify SAM3 can be imported (but not initialized without auth)
print("\n[3/3] Testing SAM3 import...")
try:
    from samgeo import SamGeo3
    print("✓ SamGeo3 imported successfully")
    print("  Note: Full SAM3 functionality requires HuggingFace authentication")
except Exception as e:
    print(f"✗ Error importing SamGeo3: {e}")
    sys.exit(1)

# Test 4: Test sliding window function definition
print("\n[BONUS] Testing sliding window components...")
try:
    from rasterio.windows import Window
    import numpy as np
    
    # Just verify we can create windows
    test_window = Window(0, 0, 512, 512)
    print(f"✓ Window creation successful: {test_window}")
except Exception as e:
    print(f"✗ Error with window operations: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nThe notebook is ready to use. Note:")
print("- SAM3 requires HuggingFace authentication (see notebook)")
print("- GPU is recommended for SAM3 inference")
print("- Downloaded test file: " + output_path)
