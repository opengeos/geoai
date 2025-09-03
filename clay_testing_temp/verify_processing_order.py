"""
Final verification: Test if processing order (batched vs individual) causes the differences
"""

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
from rasterio.enums import Resampling
from shapely import Point
from torchvision.transforms import v2
import datetime

from claymodel.module import ClayMAEModule


def get_sample_data():
    """Get just the first 3 images for quick testing"""
    lat, lon = 37.30939, -8.57207
    start, end = "2018-07-01", "2018-09-01"
    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())
            if len(items) >= 3:  # Only get first 3
                break

    epsg_str = items[0].properties["proj:code"]
    epsg = int(epsg_str.split(":")[-1])

    poidf = gpd.GeoDataFrame(
        pd.DataFrame(), crs="EPSG:4326", geometry=[Point(lon, lat)]
    ).to_crs(epsg_str)
    coords = poidf.iloc[0].geometry.coords[0]

    size, gsd = 256, 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

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
    ).compute()

    return stack, lat, lon


def setup_model():
    """Setup the Clay model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "~/.cache/clay/clay-v1.5.ckpt"
    torch.set_default_device(device)

    model = (
        ClayMAEModule.load_from_checkpoint(
            ckpt,
            model_size="large",
            metadata_path="geoai/config/clay_metadata.yaml",
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
        .eval()
        .to(device)
    )

    return model, device


def get_metadata_for_bands(stack):
    """Get metadata values for the specific bands in our stack"""
    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("geoai/config/clay_metadata.yaml")))

    means, stds, waves = [], [], []
    for band in stack.band:
        band_name = str(band.values)
        means.append(metadata[platform].bands.mean[band_name])
        stds.append(metadata[platform].bands.std[band_name])
        waves.append(metadata[platform].bands.wavelength[band_name])

    return means, stds, waves, platform


def method_batched(stack, lat, lon, model, device):
    """Original TestClay.py approach - process all images in one batch"""
    means, stds, waves, platform = get_metadata_for_bands(stack)
    transform = v2.Compose([v2.Normalize(mean=means, std=stds)])

    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat, lon = lat * np.pi / 180, lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dt) for dt in datetimes]
    week_norm = [t[0] for t in times]
    hour_norm = [t[1] for t in times]

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [ll[0] for ll in latlons]
    lon_norm = [ll[1] for ll in latlons]

    pixels = transform(torch.from_numpy(stack.data.astype(np.float32)))

    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=device
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.gsd.values, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    with torch.no_grad():
        unmsk_patch, _, _, _ = model.model.encoder(datacube)
        embeddings = unmsk_patch[:, 0, :].cpu().numpy()

    return embeddings


def method_individual(stack, lat, lon, model, device):
    """Process each image individually (like wrapper approach)"""
    means, stds, waves, platform = get_metadata_for_bands(stack)
    transform = v2.Compose([v2.Normalize(mean=means, std=stds)])

    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat, lon = lat * np.pi / 180, lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    embeddings_list = []

    for i, datetime_obj in enumerate(datetimes):
        # Process single image
        pixels_single = transform(
            torch.from_numpy(stack[i].data.astype(np.float32))
        ).unsqueeze(0)

        # Process single timestamp
        week_norm, hour_norm = normalize_timestamp(datetime_obj)
        time_tensor = torch.tensor(
            week_norm + hour_norm, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Process single location
        lat_norm, lon_norm = normalize_latlon(lat, lon)
        latlon_tensor = torch.tensor(
            lat_norm + lon_norm, dtype=torch.float32, device=device
        ).unsqueeze(0)

        datacube = {
            "platform": platform,
            "time": time_tensor,
            "latlon": latlon_tensor,
            "pixels": pixels_single.to(device),
            "gsd": torch.tensor(stack.gsd.values, device=device),
            "waves": torch.tensor(waves, device=device),
        }

        with torch.no_grad():
            unmsk_patch, _, _, _ = model.model.encoder(datacube)
            embedding = unmsk_patch[:, 0, :].cpu().numpy()
            embeddings_list.append(embedding)

    return np.vstack(embeddings_list)


def main():
    print("=== Testing Processing Order Impact ===")

    print("Loading sample data (3 images)...")
    stack, lat, lon = get_sample_data()
    print(f"Stack shape: {stack.shape}")

    print("Setting up model...")
    model, device = setup_model()

    print("\nMethod 1: Batched processing (like TestClay.py)...")
    embeddings_batched = method_batched(stack, lat, lon, model, device)
    print(f"Batched embeddings shape: {embeddings_batched.shape}")

    print("\nMethod 2: Individual processing (like TestClay_GeoAI.py)...")
    embeddings_individual = method_individual(stack, lat, lon, model, device)
    print(f"Individual embeddings shape: {embeddings_individual.shape}")

    print("\n=== Comparison ===")
    max_diff = np.max(np.abs(embeddings_batched - embeddings_individual))
    mse = np.mean((embeddings_batched - embeddings_individual) ** 2)

    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean squared error: {mse}")

    # Check tolerance levels
    for tol in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        identical = np.allclose(embeddings_batched, embeddings_individual, atol=tol)
        print(f"Identical at tolerance {tol}: {identical}")
        if identical:
            break

    print(f"\n=== CONCLUSION ===")
    if max_diff < 1e-8:
        print("✅ The processing order causes minimal differences (< 1e-8)")
        print("This confirms both approaches are functionally equivalent!")
    else:
        print("❌ Processing order causes significant differences")
        print("Further investigation needed...")


if __name__ == "__main__":
    main()
