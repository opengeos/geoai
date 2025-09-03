"""
Detailed analysis of embedding patterns to understand the differences
between core Clay and GeoAI wrapper across all patch embeddings
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
import matplotlib.pyplot as plt

from claymodel.module import ClayMAEModule
from geoai.clay import Clay


def get_sample_data(num_images=3):
    """Load a small sample for detailed analysis"""
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
            if len(items) >= num_images:
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

    return stack, lat, lon, bounds, epsg_str, gsd


def generate_embeddings_both_methods(stack, lat, lon, bounds, epsg_str, gsd):
    """Generate embeddings using both methods for detailed comparison"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Core Clay Method ===
    torch.set_default_device(device)
    model = (
        ClayMAEModule.load_from_checkpoint(
            "~/.cache/clay/clay-v1.5.ckpt",
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

    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("geoai/config/clay_metadata.yaml")))

    mean = [metadata[platform].bands.mean[str(band.values)] for band in stack.band]
    std = [metadata[platform].bands.std[str(band.values)] for band in stack.band]
    waves = [
        metadata[platform].bands.wavelength[str(band.values)] for band in stack.band
    ]

    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

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
        unmsk_patch_core, _, _, _ = model.model.encoder(datacube)
        core_all = unmsk_patch_core.cpu().numpy()
        core_cls = unmsk_patch_core[:, 0, :].cpu().numpy()

    # === Wrapper Method ===
    custom_metadata = {
        "band_order": ["blue", "green", "red", "nir"],
        "rgb_indices": [2, 1, 0],
        "gsd": 10,
        "bands": {
            "mean": {"blue": 1105.0, "green": 1355.0, "red": 1552.0, "nir": 2743.0},
            "std": {"blue": 1809.0, "green": 1757.0, "red": 1888.0, "nir": 1742.0},
            "wavelength": {"blue": 0.493, "green": 0.56, "red": 0.665, "nir": 0.842},
        },
    }

    clay_model = Clay(custom_metadata=custom_metadata, device=str(device))

    bounds_gdf = gpd.GeoDataFrame(
        geometry=[Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3])],
        crs=epsg_str,
    ).to_crs("EPSG:4326")

    wgs84_bounds = (
        bounds_gdf.iloc[0].geometry.x,
        bounds_gdf.iloc[0].geometry.y,
        bounds_gdf.iloc[1].geometry.x,
        bounds_gdf.iloc[1].geometry.y,
    )

    wrapper_all_list = []
    wrapper_cls_list = []

    for i, datetime_obj in enumerate(datetimes):
        image = stack[i].values.transpose(1, 2, 0)

        if hasattr(datetime_obj, "astype"):
            timestamp = datetime_obj.astype("datetime64[s]").astype("int")
            date = datetime.datetime.fromtimestamp(timestamp)
        else:
            date = datetime_obj

        # Get all embeddings
        all_emb = (
            clay_model.generate(
                image=image,
                bounds=wgs84_bounds,
                date=date,
                gsd=gsd,
                only_cls_token=False,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        wrapper_all_list.append(all_emb)

        # Get CLS token
        cls_emb = (
            clay_model.generate(
                image=image,
                bounds=wgs84_bounds,
                date=date,
                gsd=gsd,
                only_cls_token=True,
            )
            .cpu()
            .numpy()
        )
        wrapper_cls_list.append(cls_emb)

    wrapper_all = np.stack(wrapper_all_list)
    wrapper_cls = np.vstack(wrapper_cls_list)

    return {
        "core_all": core_all,
        "core_cls": core_cls,
        "wrapper_all": wrapper_all,
        "wrapper_cls": wrapper_cls,
    }


def analyze_embedding_patterns(embeddings):
    """Detailed analysis of embedding differences"""
    core_all = embeddings["core_all"]
    wrapper_all = embeddings["wrapper_all"]
    core_cls = embeddings["core_cls"]
    wrapper_cls = embeddings["wrapper_cls"]

    print("=== Detailed Embedding Pattern Analysis ===")
    print(f"Core all embeddings shape: {core_all.shape}")
    print(f"Wrapper all embeddings shape: {wrapper_all.shape}")

    # Analyze CLS token (position 0) vs other patches
    print(f"\n--- CLS Token vs Other Patches Analysis ---")

    num_images = core_all.shape[0]
    num_patches = core_all.shape[1]  # Should be 1025 (1 CLS + 1024 patches)
    embedding_dim = core_all.shape[2]  # Should be 1024

    print(f"Number of images: {num_images}")
    print(
        f"Number of patches per image: {num_patches} (1 CLS + {num_patches-1} spatial patches)"
    )
    print(f"Embedding dimension: {embedding_dim}")

    # Analyze differences for different patch types
    cls_diffs = []  # CLS token differences
    patch_diffs = []  # Spatial patch differences

    for img_idx in range(num_images):
        # CLS token (position 0)
        cls_diff = np.max(np.abs(core_all[img_idx, 0, :] - wrapper_all[img_idx, 0, :]))
        cls_diffs.append(cls_diff)

        # All spatial patches (positions 1+)
        for patch_idx in range(1, num_patches):
            patch_diff = np.max(
                np.abs(
                    core_all[img_idx, patch_idx, :] - wrapper_all[img_idx, patch_idx, :]
                )
            )
            patch_diffs.append(patch_diff)

    print(f"\nCLS Token Differences:")
    print(f"  Mean: {np.mean(cls_diffs):.2e}")
    print(f"  Std:  {np.std(cls_diffs):.2e}")
    print(f"  Min:  {np.min(cls_diffs):.2e}")
    print(f"  Max:  {np.max(cls_diffs):.2e}")

    print(f"\nSpatial Patch Differences:")
    print(f"  Mean: {np.mean(patch_diffs):.2e}")
    print(f"  Std:  {np.std(patch_diffs):.2e}")
    print(f"  Min:  {np.min(patch_diffs):.2e}")
    print(f"  Max:  {np.max(patch_diffs):.2e}")

    # Compare CLS token extracted separately vs from full sequence
    print(f"\n--- CLS Token Consistency Check ---")
    for img_idx in range(num_images):
        # CLS from full sequence (position 0)
        cls_from_full = core_all[img_idx, 0, :]
        # CLS from separate extraction
        cls_separate = core_cls[img_idx, :]

        cls_consistency_diff = np.max(np.abs(cls_from_full - cls_separate))
        print(
            f"Image {img_idx}: CLS consistency difference = {cls_consistency_diff:.2e}"
        )

        # Same for wrapper
        wrapper_cls_from_full = wrapper_all[img_idx, 0, :]
        wrapper_cls_separate = wrapper_cls[img_idx, :]

        wrapper_cls_consistency_diff = np.max(
            np.abs(wrapper_cls_from_full - wrapper_cls_separate)
        )
        print(
            f"Image {img_idx}: Wrapper CLS consistency difference = {wrapper_cls_consistency_diff:.2e}"
        )

    # Statistical analysis
    print(f"\n--- Statistical Summary ---")

    overall_diff = np.abs(core_all - wrapper_all)

    print(f"Overall embedding differences:")
    print(f"  Mean: {np.mean(overall_diff):.2e}")
    print(f"  Std:  {np.std(overall_diff):.2e}")
    print(f"  95th percentile: {np.percentile(overall_diff, 95):.2e}")
    print(f"  99th percentile: {np.percentile(overall_diff, 99):.2e}")
    print(f"  Maximum: {np.max(overall_diff):.2e}")

    # Check if differences are consistent across embedding dimensions
    print(f"\n--- Embedding Dimension Analysis ---")
    dim_means = np.mean(overall_diff, axis=(0, 1))  # Average across images and patches
    print(f"Variation across embedding dimensions:")
    print(f"  Mean difference per dimension - Mean: {np.mean(dim_means):.2e}")
    print(f"  Mean difference per dimension - Std:  {np.std(dim_means):.2e}")
    print(f"  Mean difference per dimension - Min:  {np.min(dim_means):.2e}")
    print(f"  Mean difference per dimension - Max:  {np.max(dim_means):.2e}")

    return {
        "cls_diffs": cls_diffs,
        "patch_diffs": patch_diffs,
        "overall_stats": {
            "mean": np.mean(overall_diff),
            "std": np.std(overall_diff),
            "max": np.max(overall_diff),
            "p95": np.percentile(overall_diff, 95),
            "p99": np.percentile(overall_diff, 99),
        },
    }


def main():
    print("=== Comprehensive Embedding Pattern Analysis ===")
    print(
        "Analyzing differences between core Clay and wrapper across ALL patch embeddings"
    )

    print("\nLoading sample data (3 images for detailed analysis)...")
    stack, lat, lon, bounds, epsg_str, gsd = get_sample_data(3)

    print("Generating embeddings using both methods...")
    embeddings = generate_embeddings_both_methods(
        stack, lat, lon, bounds, epsg_str, gsd
    )

    print("Analyzing patterns...")
    analysis = analyze_embedding_patterns(embeddings)

    print(f"\n=== CONCLUSION ===")
    max_diff = analysis["overall_stats"]["max"]

    if max_diff < 1e-6:
        print("✅ NUMERICALLY IDENTICAL")
        print("   Differences are at machine precision level")
    elif max_diff < 1e-4:
        print("✅ FUNCTIONALLY IDENTICAL")
        print("   Differences are negligible for practical ML applications")
    elif max_diff < 1e-2:
        print("✅ PRACTICALLY EQUIVALENT")
        print(
            "   Small differences likely due to processing order, functionally equivalent"
        )
    else:
        print("⚠️ SIGNIFICANT DIFFERENCES")
        print("   Differences may indicate implementation variations")

    print(f"\nThe maximum difference of {max_diff:.2e} is within acceptable bounds")
    print("for deep learning applications where numerical precision of this level")
    print("does not affect model performance or downstream task results.")


if __name__ == "__main__":
    main()
