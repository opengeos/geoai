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
    print("=== Testing Wrapper with Built-in Metadata ===")
    print("Loading data...")

    # Get common data
    stack, lat, lon, bounds, epsg_str, gsd = get_data()

    print("Generating embeddings using GeoAI wrapper with sentinel-2-l2a metadata...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Use built-in sentinel-2-l2a metadata instead of custom
    clay_model = Clay(sensor_name="sentinel-2-l2a", device=str(device))

    # Convert WGS84 bounds for Clay model
    # First convert stack bounds back to WGS84
    proj_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
    bounds_gdf = gpd.GeoDataFrame(
        geometry=[Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3])],
        crs=epsg_str,
    ).to_crs("EPSG:4326")

    wgs84_bounds = (
        bounds_gdf.iloc[0].geometry.x,  # min_lon
        bounds_gdf.iloc[0].geometry.y,  # min_lat
        bounds_gdf.iloc[1].geometry.x,  # max_lon
        bounds_gdf.iloc[1].geometry.y,  # max_lat
    )

    print(f"WGS84 bounds: {wgs84_bounds}")

    # Debug: Let's examine what the wrapper is doing step by step for the first image
    datetimes = stack.time.values.astype("datetime64[s]").tolist()

    print("\n=== Debugging First Image ===")
    i = 0
    datetime_obj = datetimes[i]
    image = stack[i].values.transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]

    # Convert numpy datetime64 to Python datetime
    if hasattr(datetime_obj, "astype"):
        timestamp = datetime_obj.astype("datetime64[s]").astype("int")
        date = datetime.datetime.fromtimestamp(timestamp)
    else:
        date = datetime_obj

    print(f"Image shape: {image.shape}")
    print(f"Date: {date}")
    print(f"Bounds: {wgs84_bounds}")
    print(f"GSD: {gsd}")

    # Let's debug what happens in the wrapper's prepare_datacube method
    print("\n=== Wrapper Datacube Preparation ===")

    # Get sensor metadata (same as wrapper does)
    band_order = clay_model.metadata.band_order
    gsd_used = gsd if gsd is not None else clay_model.metadata.gsd

    print(f"Band order: {band_order}")
    print(f"GSD used: {gsd_used}")

    # Extract normalization parameters (same as wrapper does)
    means = [clay_model.metadata.bands.mean[band] for band in band_order]
    stds = [clay_model.metadata.bands.std[band] for band in band_order]
    wavelengths = [clay_model.metadata.bands.wavelength[band] for band in band_order]

    print(f"Means: {means}")
    print(f"Stds: {stds}")
    print(f"Wavelengths: {wavelengths}")

    # Now let's call the wrapper and see what it produces
    embedding_wrapper = clay_model.generate(
        image=image, bounds=wgs84_bounds, date=date, gsd=gsd, only_cls_token=True
    )

    print(f"Wrapper embedding shape: {embedding_wrapper.shape}")
    print(f"Wrapper embedding first 5 values: {embedding_wrapper[0, :5]}")

    # Now let's try to replicate the same process manually
    print("\n=== Manual Replication ===")

    # Convert to tensor and transpose to [C, H, W]
    if isinstance(image, torch.Tensor):
        pixels = image.float()
        if (
            pixels.dim() == 3 and pixels.shape[-1] != pixels.shape[0]
        ):  # [H, W, C] format
            pixels = pixels.permute(2, 0, 1)
    else:
        pixels = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

    print(f"Pixels shape after permute: {pixels.shape}")

    # Normalize
    transform = v2.Compose([v2.Normalize(mean=means, std=stds)])
    pixels = transform(pixels).unsqueeze(0)  # Add batch dimension

    print(f"Pixels shape after normalize and unsqueeze: {pixels.shape}")

    # Prepare temporal encoding
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(bounds):
        lon = bounds[0] + (bounds[2] - bounds[0]) / 2
        lat = bounds[1] + (bounds[3] - bounds[1]) / 2
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    week_norm, hour_norm = normalize_timestamp(date)
    time_tensor = torch.tensor(
        week_norm
        + hour_norm,  # Clay expects 4 elements: [week_sin, week_cos, hour_sin, hour_cos]
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    print(f"Time tensor: {time_tensor}")

    # Prepare spatial encoding
    lat_norm, lon_norm = normalize_latlon(wgs84_bounds)
    latlon_tensor = torch.tensor(
        lat_norm
        + lon_norm,  # Clay expects 4 elements: [sin_lat, cos_lat, sin_lon, cos_lon]
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    print(f"LatLon tensor: {latlon_tensor}")

    # Create datacube
    datacube = {
        "pixels": pixels.to(device),
        "time": time_tensor,
        "latlon": latlon_tensor,
        "gsd": torch.tensor(gsd_used, device=device),
        "waves": torch.tensor(wavelengths, device=device),
    }

    print(f"Datacube keys: {datacube.keys()}")
    for key, value in datacube.items():
        if isinstance(value, torch.Tensor):
            print(
                f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}"
            )

    # Run through model
    with torch.no_grad():
        encoded_patches, _, _, _ = clay_model.module.model.encoder(datacube)
        embedding_manual = encoded_patches[:, 0, :]

    print(f"Manual embedding shape: {embedding_manual.shape}")
    print(f"Manual embedding first 5 values: {embedding_manual[0, :5]}")

    # Compare
    diff = torch.abs(embedding_wrapper - embedding_manual.cpu())
    max_diff = torch.max(diff)
    print(f"Max difference between wrapper and manual: {max_diff}")

    if max_diff < 1e-10:
        print("✅ EXACT MATCH!")
    elif max_diff < 1e-6:
        print("✅ Very close match (within 1e-6)")
    else:
        print("❌ Significant difference")


if __name__ == "__main__":
    main()
