"""
Test Clay library and GeoAI wrapper with identical random data.
Generate random tensors, metadata, and compare all embedding tokens.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "geoai"))

import math
import numpy as np
import torch
import yaml
import datetime
from box import Box
from torchvision.transforms import v2

from claymodel.module import ClayMAEModule
from geoai.clay import Clay


def generate_random_metadata(num_bands=4, seed=42):
    """Generate random but valid metadata"""
    np.random.seed(seed)

    # Generate random band names
    band_names = [f"band_{i+1}" for i in range(num_bands)]

    # Generate realistic random values
    means = np.random.uniform(1000, 3000, num_bands).tolist()
    stds = np.random.uniform(1500, 2000, num_bands).tolist()
    wavelengths = np.random.uniform(0.4, 2.5, num_bands).tolist()

    metadata = {
        "band_order": band_names,
        "rgb_indices": [0, 1, 2] if num_bands >= 3 else [0, 0, 0],
        "gsd": 10.0,
        "bands": {
            "mean": {band: mean for band, mean in zip(band_names, means)},
            "std": {band: std for band, std in zip(band_names, stds)},
            "wavelength": {band: wave for band, wave in zip(band_names, wavelengths)},
        },
    }

    return metadata


def generate_random_inputs(height=256, width=256, num_bands=4, num_images=3, seed=42):
    """Generate random input tensors and metadata"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random image data [num_images, height, width, channels]
    images = np.random.uniform(0, 10000, (num_images, height, width, num_bands)).astype(
        np.float32
    )

    # Generate random bounds (WGS84 coordinates)
    center_lon = np.random.uniform(-180, 180)
    center_lat = np.random.uniform(-85, 85)

    # Small bounding box around center
    bounds = (
        center_lon - 0.01,  # min_lon
        center_lat - 0.01,  # min_lat
        center_lon + 0.01,  # max_lon
        center_lat + 0.01,  # max_lat
    )

    # Generate random dates
    base_date = datetime.datetime(2020, 1, 1)
    dates = [
        base_date + datetime.timedelta(days=int(np.random.uniform(0, 365)))
        for _ in range(num_images)
    ]

    # Generate random GSD
    gsd = 10.0

    return images, bounds, dates, gsd


def test_core_clay_batched(images, metadata_dict, bounds, dates, gsd):
    """Test core Clay library with batched processing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Load model
    model = (
        ClayMAEModule.load_from_checkpoint(
            "~/.cache/clay/clay-v1.5.ckpt",
            model_size="large",
            metadata_path="random_metadata.yaml",  # We'll create this
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
        .eval()
        .to(device)
    )

    # Extract metadata values
    band_order = metadata_dict["band_order"]
    means = [metadata_dict["bands"]["mean"][band] for band in band_order]
    stds = [metadata_dict["bands"]["std"][band] for band in band_order]
    waves = [metadata_dict["bands"]["wavelength"][band] for band in band_order]

    transform = v2.Compose([v2.Normalize(mean=means, std=stds)])

    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(bounds):
        lon = bounds[0] + (bounds[2] - bounds[0]) / 2
        lat = bounds[1] + (bounds[3] - bounds[1]) / 2
        lat, lon = lat * np.pi / 180, lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    # Process all images in batch
    times = [normalize_timestamp(date) for date in dates]
    week_norm = [t[0] for t in times]
    hour_norm = [t[1] for t in times]

    latlons = [normalize_latlon(bounds)] * len(dates)
    lat_norm = [ll[0] for ll in latlons]
    lon_norm = [ll[1] for ll in latlons]

    # Convert images to tensor [batch, channels, height, width]
    pixels = torch.from_numpy(images.transpose(0, 3, 1, 2))
    pixels = transform(pixels)

    datacube = {
        "platform": "random-sensor",
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=device
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor([gsd], device=device),  # Single GSD value as array
        "waves": torch.tensor(waves, device=device),
    }

    with torch.no_grad():
        unmsk_patch, _, _, _ = model.model.encoder(datacube)
        all_embeddings = unmsk_patch.cpu().numpy()
        cls_embeddings = unmsk_patch[:, 0, :].cpu().numpy()

    return all_embeddings, cls_embeddings


def test_geoai_wrapper(images, metadata_dict, bounds, dates, gsd):
    """Test GeoAI wrapper with individual processing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Clay wrapper with custom metadata
    clay_model = Clay(custom_metadata=metadata_dict, device=str(device))

    all_embeddings_list = []
    cls_embeddings_list = []

    for i, date in enumerate(dates):
        image = images[i]  # [H, W, C]

        # Get all embeddings
        all_embedding = clay_model.generate(
            image=image, bounds=bounds, date=date, gsd=gsd, only_cls_token=False
        )
        all_embeddings_list.append(all_embedding.squeeze(0).cpu().numpy())

        # Get CLS token only
        cls_embedding = clay_model.generate(
            image=image, bounds=bounds, date=date, gsd=gsd, only_cls_token=True
        )
        cls_embeddings_list.append(cls_embedding.cpu().numpy())

    return np.stack(all_embeddings_list), np.vstack(cls_embeddings_list)


def save_metadata_to_file(metadata_dict, filename="random_metadata.yaml"):
    """Save metadata to YAML file in Clay format"""
    # Create Clay-compatible metadata structure
    clay_metadata = {"random-sensor": metadata_dict}

    with open(filename, "w") as f:
        yaml.dump(clay_metadata, f, default_flow_style=False)

    return filename


def compare_embeddings(emb1, emb2, name="embeddings"):
    """Compare embeddings with multiple tolerance levels"""
    print(f"\n=== {name.upper()} COMPARISON ===")
    print(f"Shape 1: {emb1.shape}")
    print(f"Shape 2: {emb2.shape}")

    if emb1.shape != emb2.shape:
        print("❌ ERROR: Shapes do not match!")
        return False

    # Calculate metrics
    max_diff = np.max(np.abs(emb1 - emb2))
    mse = np.mean((emb1 - emb2) ** 2)
    mae = np.mean(np.abs(emb1 - emb2))

    print(f"Max difference: {max_diff:.2e}")
    print(f"MSE: {mse:.2e}")
    print(f"MAE: {mae:.2e}")

    # Test tolerance levels
    tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    results = {}

    print("\nAllclose tests:")
    for tol in tolerances:
        is_close = np.allclose(emb1, emb2, atol=tol)
        results[tol] = is_close
        status = "✅" if is_close else "❌"
        print(f"  {tol}: {status}")
        if is_close:
            break

    # Test with torch as well
    print("\nTorch allclose tests:")
    tensor1 = torch.from_numpy(emb1)
    tensor2 = torch.from_numpy(emb2)

    for tol in tolerances:
        is_close = torch.allclose(tensor1, tensor2, atol=tol)
        status = "✅" if is_close else "❌"
        print(f"  {tol}: {status}")
        if is_close:
            break

    return results


def main():
    print("=== RANDOM DATA EMBEDDING COMPARISON TEST ===")
    print("Testing Clay library vs GeoAI wrapper with identical random data\n")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("1. Generating random metadata...")
    metadata = generate_random_metadata(num_bands=4, seed=42)
    print(f"   Bands: {metadata['band_order']}")
    print(f"   Means: {[f'{v:.1f}' for v in metadata['bands']['mean'].values()]}")
    print(f"   Stds: {[f'{v:.1f}' for v in metadata['bands']['std'].values()]}")
    print(
        f"   Wavelengths: {[f'{v:.3f}' for v in metadata['bands']['wavelength'].values()]}"
    )

    print("\n2. Saving metadata to file...")
    metadata_file = save_metadata_to_file(metadata)
    print(f"   Saved to: {metadata_file}")

    print("\n3. Generating random input data...")
    images, bounds, dates, gsd = generate_random_inputs(
        height=256, width=256, num_bands=4, num_images=3, seed=42
    )
    print(f"   Images shape: {images.shape}")
    print(f"   Bounds: {bounds}")
    print(f"   Dates: {[d.strftime('%Y-%m-%d %H:%M:%S') for d in dates]}")
    print(f"   GSD: {gsd}")

    print("\n4. Testing core Clay library (batched processing)...")
    all_core, cls_core = test_core_clay_batched(images, metadata, bounds, dates, gsd)
    print(f"   Core all embeddings shape: {all_core.shape}")
    print(f"   Core CLS embeddings shape: {cls_core.shape}")

    print("\n5. Testing GeoAI wrapper (individual processing)...")
    all_wrapper, cls_wrapper = test_geoai_wrapper(images, metadata, bounds, dates, gsd)
    print(f"   Wrapper all embeddings shape: {all_wrapper.shape}")
    print(f"   Wrapper CLS embeddings shape: {cls_wrapper.shape}")

    print("\n6. Comparing results...")

    # Compare CLS tokens
    cls_results = compare_embeddings(cls_core, cls_wrapper, "CLS tokens")

    # Compare all embeddings
    all_results = compare_embeddings(all_core, all_wrapper, "All embeddings")

    print("\n=== FINAL RESULTS SUMMARY ===")

    # Find best tolerance for each
    def get_best_tolerance(results):
        for tol, passed in results.items():
            if passed:
                return tol
        return None

    cls_best_tol = get_best_tolerance(cls_results)
    all_best_tol = get_best_tolerance(all_results)

    print(f"CLS tokens: {'✅ PASSED' if cls_best_tol else '❌ FAILED'}")
    if cls_best_tol:
        print(f"  Best tolerance: {cls_best_tol}")

    print(f"All embeddings: {'✅ PASSED' if all_best_tol else '❌ FAILED'}")
    if all_best_tol:
        print(f"  Best tolerance: {all_best_tol}")

    if cls_best_tol and cls_best_tol <= 1e-4:
        print("\n🎉 SUCCESS: Both methods produce nearly identical results!")
        print(
            "   Small differences are likely due to processing order (batched vs individual)"
        )
    elif cls_best_tol and cls_best_tol <= 1e-3:
        print("\n✅ SUCCESS: Both methods are functionally equivalent!")
        print("   Differences are within acceptable range for ML applications")
    else:
        print("\n❌ CONCERN: Methods may have implementation differences")
        print("   Further investigation recommended")

    # Cleanup
    try:
        os.remove(metadata_file)
        print(f"\nCleaned up temporary file: {metadata_file}")
    except:
        pass


if __name__ == "__main__":
    main()
