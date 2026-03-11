"""
Data preparation script for impervious surface mapping notebook.

Downloads and prepares:
1. NAIP aerial imagery (RGB) for Washington DC from Planetary Computer
2. DC Impervious Surface 2023 polygons from DC GIS Open Data

Usage:
    python scripts/prepare_dc_impervious_data.py

After running, the files will be saved to docs/examples/:
    - dc_naip.tif (RGB NAIP imagery, clipped to target area)
    - dc_impervious_surfaces.geojson (impervious surface polygons, clipped to same extent)

You can then upload these to HuggingFace or test the notebook locally.
"""

import json
import os
import urllib.request

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

# ── Configuration ──────────────────────────────────────────────────────────
# Target area: a ~2.5 km² region in central DC (National Mall area)
# Covers mix of buildings, roads, parking lots, and green spaces
BBOX = (-77.045, 38.885, -77.025, 38.900)  # (west, south, east, north)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "examples")
RASTER_OUTPUT = os.path.join(OUTPUT_DIR, "dc_naip.tif")
VECTOR_OUTPUT = os.path.join(OUTPUT_DIR, "dc_impervious_surfaces.geojson")

# DC GIS REST API for Impervious Surface 2023 (Layer 73)
DC_GIS_BASE = (
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/"
    "DCGIS_DATA/Environment_Stormwater_Management_WebMercator/MapServer/73/query"
)
PAGE_SIZE = 2000  # max records per request


def download_naip():
    """Download NAIP imagery for DC from Planetary Computer."""
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        print(
            "ERROR: Install planetary-computer and pystac-client:\n"
            "  pip install planetary-computer pystac-client"
        )
        raise

    print("Searching Planetary Computer for DC NAIP imagery...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["naip"],
        bbox=BBOX,
        datetime="2021-01-01/2023-12-31",
        max_items=5,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No NAIP items found for bbox={BBOX}. "
            "Try expanding the bbox or date range."
        )

    # Pick the most recent item
    items.sort(key=lambda x: x.datetime, reverse=True)
    item = items[0]
    print(f"Found NAIP item: {item.id} ({item.datetime.date()})")

    # Download the image asset
    asset = item.assets["image"]
    href = asset.href
    print(f"Downloading from: {href[:100]}...")

    with rasterio.open(href) as src:
        # Clip to our target bbox
        # Reproject bbox to raster CRS if needed
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        clip_geom = [box(minx, miny, maxx, maxy).__geo_interface__]

        out_image, out_transform = mask(src, clip_geom, crop=True)

        # Keep only RGB (first 3 bands) if 4-band NAIP
        if out_image.shape[0] >= 3:
            out_image = out_image[:3]

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=out_image.shape[1],
            width=out_image.shape[2],
            count=3,
            transform=out_transform,
            compress="deflate",
        )

        os.makedirs(os.path.dirname(RASTER_OUTPUT), exist_ok=True)
        with rasterio.open(RASTER_OUTPUT, "w", **profile) as dst:
            dst.write(out_image)

    size_mb = os.path.getsize(RASTER_OUTPUT) / (1024 * 1024)
    print(f"Saved NAIP RGB imagery: {RASTER_OUTPUT} ({size_mb:.1f} MB)")
    print(f"  Shape: {out_image.shape}, CRS: {profile['crs']}")
    return RASTER_OUTPUT


def download_impervious_surfaces():
    """Download DC impervious surface polygons from ArcGIS REST API."""
    print("\nDownloading DC Impervious Surface 2023 polygons...")

    # First, get count of features in bbox
    count_params = (
        f"?where=1%3D1"
        f"&geometry={BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
        f"&geometryType=esriGeometryEnvelope"
        f"&inSR=4326&spatialRel=esriSpatialRelIntersects"
        f"&returnCountOnly=true&f=json"
    )
    count_url = DC_GIS_BASE + count_params
    with urllib.request.urlopen(count_url, timeout=60) as resp:
        count_data = json.loads(resp.read())
    total = count_data.get("count", 0)
    print(f"Total features in bbox: {total}")

    if total == 0:
        raise RuntimeError(
            "No impervious surface features found in the target bbox. "
            "Check the bbox coordinates."
        )

    # Paginate through all features
    all_features = []
    offset = 0
    while offset < total:
        query_params = (
            f"?where=1%3D1"
            f"&geometry={BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
            f"&geometryType=esriGeometryEnvelope"
            f"&inSR=4326&spatialRel=esriSpatialRelIntersects"
            f"&outFields=FEATURECODE,DESCRIPTION"
            f"&returnGeometry=true"
            f"&outSR=4326"
            f"&resultOffset={offset}"
            f"&resultRecordCount={PAGE_SIZE}"
            f"&f=geojson"
        )
        url = DC_GIS_BASE + query_params
        print(f"  Fetching records {offset}-{offset + PAGE_SIZE}...")

        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read())

        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        offset += PAGE_SIZE

    print(f"Downloaded {len(all_features)} features total")

    # Build GeoDataFrame
    geojson = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")

    # Clip to exact bbox
    bbox_geom = box(*BBOX)
    gdf = gdf[gdf.intersects(bbox_geom)].copy()
    gdf["geometry"] = gdf.geometry.intersection(bbox_geom)
    gdf = gdf[~gdf.is_empty].copy()

    # Reproject to match the NAIP raster CRS if raster already exists
    if os.path.exists(RASTER_OUTPUT):
        with rasterio.open(RASTER_OUTPUT) as src:
            raster_crs = src.crs
        gdf = gdf.to_crs(raster_crs)
        print(f"  Reprojected to match raster CRS: {raster_crs}")

    os.makedirs(os.path.dirname(VECTOR_OUTPUT), exist_ok=True)
    gdf.to_file(VECTOR_OUTPUT, driver="GeoJSON")

    size_mb = os.path.getsize(VECTOR_OUTPUT) / (1024 * 1024)
    print(f"Saved impervious surfaces: {VECTOR_OUTPUT} ({size_mb:.1f} MB)")
    print(f"  Features: {len(gdf)}")
    return VECTOR_OUTPUT


if __name__ == "__main__":
    print("=" * 60)
    print("DC Impervious Surface Data Preparation")
    print("=" * 60)
    print(f"Target bbox (WGS84): {BBOX}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print()

    raster_path = download_naip()
    vector_path = download_impervious_surfaces()

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  Raster: {os.path.abspath(raster_path)}")
    print(f"  Vector: {os.path.abspath(vector_path)}")
    print(f"\nNext steps:")
    print(f"  1. Test the notebook: docs/examples/impervious_surface_mapping.ipynb")
    print(
        f"  2. Update download URLs to point to local files (or upload to HuggingFace)"
    )
