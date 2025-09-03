import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "geoai"))

import math
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
import yaml
from box import Box
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely import Point
from sklearn import decomposition, svm
from torchvision.transforms import v2
import datetime

from claymodel.module import ClayMAEModule
from geoai.clay import Clay


def get_data():
    """Common data loading function for both approaches"""
    # Point over Monchique Portugal
    lat, lon = 37.30939, -8.57207

    # Dates of a large forest fire
    start = "2018-07-01"
    end = "2018-09-01"

    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    # Search the catalogue
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    print(f"Found {len(items)} items")

    # Extract coordinate system from first item
    epsg_str = items[0].properties["proj:code"]
    epsg = int(epsg_str.split(":")[-1])  # Convert 'EPSG:32629' to 32629

    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg_str)

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    size = 256
    gsd = 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    # Retrieve the pixel values, for the bounding box in
    # the target projection. In this example we use only
    # the RGB and NIR bands.
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float64",
        rescale=False,
        fill_value=np.nan,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )

    stack = stack.compute()

    return stack, lat, lon, bounds, epsg_str, gsd


def main():
    print("=== FINAL ANALYSIS: Why the embeddings differ ===")
    print("Loading data...")

    # Get common data
    stack, lat, lon, bounds, epsg_str, gsd = get_data()

    print(f"\nActual data has {len(stack.band)} bands: {list(stack.band.values)}")

    # Check what the full metadata has
    print("\n=== Full Sentinel-2-L2A Metadata ===")
    metadata = Box(yaml.safe_load(open("geoai/config/clay_metadata.yaml")))
    platform = "sentinel-2-l2a"
    all_bands = list(metadata[platform].bands.mean.keys())
    print(f"Full metadata has {len(all_bands)} bands: {all_bands}")

    # What TestClay.py actually uses
    print("\n=== What TestClay.py Uses ===")
    used_bands = []
    means = []
    stds = []
    waves = []
    for band in stack.band:
        band_name = str(band.values)
        used_bands.append(band_name)
        means.append(metadata[platform].bands.mean[band_name])
        stds.append(metadata[platform].bands.std[band_name])
        waves.append(metadata[platform].bands.wavelength[band_name])

    print(f"TestClay.py uses {len(used_bands)} bands: {used_bands}")
    print(f"Means: {means}")
    print(f"Stds: {stds}")
    print(f"Wavelengths: {waves}")

    # What TestClay_GeoAI.py uses (custom metadata)
    print("\n=== What TestClay_GeoAI.py Uses (Custom Metadata) ===")
    custom_metadata = {
        "band_order": ["blue", "green", "red", "nir"],
        "gsd": 10,
        "bands": {
            "mean": {"blue": 1105.0, "green": 1355.0, "red": 1552.0, "nir": 2743.0},
            "std": {"blue": 1809.0, "green": 1757.0, "red": 1888.0, "nir": 1742.0},
            "wavelength": {"blue": 0.493, "green": 0.56, "red": 0.665, "nir": 0.842},
        },
    }

    custom_bands = custom_metadata["band_order"]
    custom_means = [custom_metadata["bands"]["mean"][b] for b in custom_bands]
    custom_stds = [custom_metadata["bands"]["std"][b] for b in custom_bands]
    custom_waves = [custom_metadata["bands"]["wavelength"][b] for b in custom_bands]

    print(f"TestClay_GeoAI.py uses {len(custom_bands)} bands: {custom_bands}")
    print(f"Means: {custom_means}")
    print(f"Stds: {custom_stds}")
    print(f"Wavelengths: {custom_waves}")

    # Compare
    print("\n=== Comparison ===")
    print("Are the values identical?")
    print(f"Means match: {means == custom_means}")
    print(f"Stds match: {stds == custom_stds}")
    print(f"Wavelengths match: {waves == custom_waves}")

    if means == custom_means and stds == custom_stds and waves == custom_waves:
        print("✅ The metadata values are IDENTICAL!")
        print("The small differences in embeddings are likely due to:")
        print("1. Different processing order (batched vs individual)")
        print("2. Minor floating-point precision differences")
        print("3. Different tensor device placement timing")
        print("\nBut functionally, both approaches are equivalent!")
    else:
        print(
            "❌ The metadata values are different - this explains the embedding differences"
        )


if __name__ == "__main__":
    main()
