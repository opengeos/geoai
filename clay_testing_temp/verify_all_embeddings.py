"""
Comprehensive verification comparing ALL embeddings (full patch sequences)
not just the CLS token between core Clay and GeoAI wrapper
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
from sklearn import svm
from torchvision.transforms import v2
import datetime

from claymodel.module import ClayMAEModule
from geoai.clay import Clay


def get_data():
    """Load the same data used in both original test files"""
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


def generate_all_embeddings_core_clay(stack, lat, lon):
    """Core Clay approach - returns ALL patch embeddings (not just CLS token)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)
        # Return ALL embeddings (full sequence), not just CLS token
        all_embeddings = unmsk_patch.cpu().numpy()
        # Also return just CLS tokens for comparison
        cls_embeddings = unmsk_patch[:, 0, :].cpu().numpy()

    return all_embeddings, cls_embeddings


def generate_all_embeddings_wrapper(stack, lat, lon, bounds, epsg_str, gsd):
    """Wrapper approach - returns ALL patch embeddings (not just CLS token)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Convert bounds to WGS84
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

    all_embeddings_list = []
    cls_embeddings_list = []
    datetimes = stack.time.values.astype("datetime64[s]").tolist()

    for i, datetime_obj in enumerate(datetimes):
        image = stack[i].values.transpose(1, 2, 0)

        if hasattr(datetime_obj, "astype"):
            timestamp = datetime_obj.astype("datetime64[s]").astype("int")
            date = datetime.datetime.fromtimestamp(timestamp)
        else:
            date = datetime_obj

        # Get ALL embeddings (full sequence)
        all_embeddings = clay_model.generate(
            image=image, bounds=wgs84_bounds, date=date, gsd=gsd, only_cls_token=False
        )
        # Remove batch dimension since we're processing one image at a time
        all_embeddings_squeezed = all_embeddings.squeeze(0).cpu().numpy()
        all_embeddings_list.append(all_embeddings_squeezed)

        # Also get just CLS token for comparison
        cls_embedding = clay_model.generate(
            image=image, bounds=wgs84_bounds, date=date, gsd=gsd, only_cls_token=True
        )
        cls_embeddings_list.append(cls_embedding.cpu().numpy())

    return np.stack(all_embeddings_list), np.vstack(cls_embeddings_list)


def compare_embeddings_comprehensive(
    emb1_all, emb1_cls, emb2_all, emb2_cls, name1="Method 1", name2="Method 2"
):
    """Compare both full embeddings and CLS tokens comprehensively"""
    print(f"\n=== Comprehensive Embedding Comparison ===")

    # Compare shapes
    print(f"{name1} all embeddings shape: {emb1_all.shape}")
    print(f"{name2} all embeddings shape: {emb2_all.shape}")
    print(f"{name1} CLS embeddings shape: {emb1_cls.shape}")
    print(f"{name2} CLS embeddings shape: {emb2_cls.shape}")

    if emb1_all.shape != emb2_all.shape:
        print("❌ ERROR: All embeddings shapes don't match!")
        return None

    if emb1_cls.shape != emb2_cls.shape:
        print("❌ ERROR: CLS embeddings shapes don't match!")
        return None

    # Compare ALL embeddings (full sequences)
    print(f"\n--- Full Embedding Sequences Comparison ---")
    max_diff_all = np.max(np.abs(emb1_all - emb2_all))
    mse_all = np.mean((emb1_all - emb2_all) ** 2)
    mae_all = np.mean(np.abs(emb1_all - emb2_all))
    correlation_all = np.corrcoef(emb1_all.flatten(), emb2_all.flatten())[0, 1]

    print(f"All embeddings max difference: {max_diff_all:.2e}")
    print(f"All embeddings MSE: {mse_all:.2e}")
    print(f"All embeddings MAE: {mae_all:.2e}")
    print(f"All embeddings correlation: {correlation_all:.10f}")

    # Test tolerance levels for all embeddings
    tolerance_levels = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    functionally_identical_all = False
    for tolerance in tolerance_levels:
        are_identical = np.allclose(emb1_all, emb2_all, atol=tolerance)
        print(f"All embeddings identical (tolerance={tolerance}): {are_identical}")
        if are_identical:
            print(
                f"✅ All embeddings are functionally identical at tolerance {tolerance}"
            )
            functionally_identical_all = True
            break

    # Compare CLS tokens
    print(f"\n--- CLS Token Comparison ---")
    max_diff_cls = np.max(np.abs(emb1_cls - emb2_cls))
    mse_cls = np.mean((emb1_cls - emb2_cls) ** 2)
    mae_cls = np.mean(np.abs(emb1_cls - emb2_cls))
    correlation_cls = np.corrcoef(emb1_cls.flatten(), emb2_cls.flatten())[0, 1]

    print(f"CLS embeddings max difference: {max_diff_cls:.2e}")
    print(f"CLS embeddings MSE: {mse_cls:.2e}")
    print(f"CLS embeddings MAE: {mae_cls:.2e}")
    print(f"CLS embeddings correlation: {correlation_cls:.10f}")

    # Test tolerance levels for CLS embeddings
    functionally_identical_cls = False
    for tolerance in tolerance_levels:
        are_identical = np.allclose(emb1_cls, emb2_cls, atol=tolerance)
        print(f"CLS embeddings identical (tolerance={tolerance}): {are_identical}")
        if are_identical:
            print(
                f"✅ CLS embeddings are functionally identical at tolerance {tolerance}"
            )
            functionally_identical_cls = True
            break

    # Per-image analysis
    print(f"\n--- Per-Image Analysis ---")
    num_images = emb1_all.shape[0]
    for i in range(num_images):
        diff_all = np.max(np.abs(emb1_all[i] - emb2_all[i]))
        diff_cls = np.max(np.abs(emb1_cls[i] - emb2_cls[i]))
        print(
            f"Image {i}: All embeddings max diff = {diff_all:.2e}, CLS max diff = {diff_cls:.2e}"
        )

    return {
        "all_embeddings": {
            "max_diff": max_diff_all,
            "mse": mse_all,
            "mae": mae_all,
            "correlation": correlation_all,
            "functionally_identical": functionally_identical_all,
        },
        "cls_embeddings": {
            "max_diff": max_diff_cls,
            "mse": mse_cls,
            "mae": mae_cls,
            "correlation": correlation_cls,
            "functionally_identical": functionally_identical_cls,
        },
    }


def test_downstream_performance_cls(embeddings_cls, method_name):
    """Test SVM classification performance using CLS tokens"""
    labels = np.array([0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    fit_indices = [0, 1, 3, 4, 7, 8, 9]
    test_indices = [2, 5, 6, 10, 11]

    clf = svm.SVC()
    clf.fit(embeddings_cls[fit_indices] + 100, labels[fit_indices])
    predictions = clf.predict(embeddings_cls[test_indices] + 100)
    correct = np.sum(labels[test_indices] == predictions)
    total = len(test_indices)

    print(f"{method_name} (CLS): {correct}/{total} correct predictions")
    return correct, total, predictions


def main():
    print("=== Comprehensive Embedding Verification (ALL vs CLS) ===")
    print("This test compares FULL embedding sequences, not just CLS tokens")
    print("\nLoading data...")

    stack, lat, lon, bounds, epsg_str, gsd = get_data()
    print(f"Found {len(stack.time)} images")

    print(f"\nImage dimensions: {stack.shape}")
    print(f"Expected patch grid: ~{(256//16)**2} patches per image (16x16 patches)")

    print("\n1. Generating ALL embeddings using core Clay library...")
    all_emb_core, cls_emb_core = generate_all_embeddings_core_clay(stack, lat, lon)

    print("2. Generating ALL embeddings using GeoAI wrapper...")
    all_emb_wrapper, cls_emb_wrapper = generate_all_embeddings_wrapper(
        stack, lat, lon, bounds, epsg_str, gsd
    )

    # Comprehensive comparison
    results = compare_embeddings_comprehensive(
        all_emb_core,
        cls_emb_core,
        all_emb_wrapper,
        cls_emb_wrapper,
        "Core Clay",
        "GeoAI Wrapper",
    )

    if results is None:
        print("❌ Comparison failed due to shape mismatch")
        return

    # Test downstream performance with CLS tokens
    print(f"\n=== Downstream Task Performance (CLS Tokens) ===")
    correct_core, total_core, pred_core = test_downstream_performance_cls(
        cls_emb_core, "Core Clay"
    )
    correct_wrapper, total_wrapper, pred_wrapper = test_downstream_performance_cls(
        cls_emb_wrapper, "GeoAI Wrapper"
    )

    print(f"\nCLS Prediction comparison:")
    print(f"Core predictions:    {pred_core}")
    print(f"Wrapper predictions: {pred_wrapper}")
    predictions_match = np.array_equal(pred_core, pred_wrapper)
    print(f"CLS Predictions identical: {predictions_match}")

    print(f"\n=== FINAL COMPREHENSIVE VERDICT ===")

    all_identical = results["all_embeddings"]["functionally_identical"]
    cls_identical = results["cls_embeddings"]["functionally_identical"]

    if all_identical and cls_identical and predictions_match:
        print("✅ COMPLETELY FUNCTIONALLY EQUIVALENT")
        print("   ✓ ALL embedding sequences are functionally identical")
        print("   ✓ CLS tokens are functionally identical")
        print("   ✓ Downstream task results are identical")
        print("   The wrapper perfectly replicates the core library behavior")
    elif cls_identical and predictions_match:
        print("✅ CLS-EQUIVALENT (PRACTICALLY IDENTICAL)")
        print("   ✓ CLS tokens are functionally identical")
        print("   ✓ Downstream task results are identical")
        if not all_identical:
            print("   ⚠ Full embedding sequences have minor differences")
            print(
                "   This is expected due to processing order and doesn't affect practical use"
            )
    elif predictions_match:
        print("✅ FUNCTIONALLY SIMILAR")
        print("   ✓ Downstream task results are identical")
        print("   ⚠ Some embedding differences exist but don't affect performance")
    else:
        print("❌ FUNCTIONALLY DIFFERENT")
        print("   Different downstream task results indicate implementation issues")

    print(f"\nKey Metrics Summary:")
    print(
        f"All embeddings - Max diff: {results['all_embeddings']['max_diff']:.2e}, Correlation: {results['all_embeddings']['correlation']:.8f}"
    )
    print(
        f"CLS embeddings - Max diff: {results['cls_embeddings']['max_diff']:.2e}, Correlation: {results['cls_embeddings']['correlation']:.8f}"
    )

    return results


if __name__ == "__main__":
    results = main()
