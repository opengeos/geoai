import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "geoai"))

import yaml
from box import Box
from geoai.clay import Clay

# Check what metadata the core Clay uses
print("=== Core Clay Metadata ===")
metadata = Box(yaml.safe_load(open("geoai/config/clay_metadata.yaml")))
platform = "sentinel-2-l2a"
bands = ["blue", "green", "red", "nir"]

print("Core Clay metadata for sentinel-2-l2a:")
for band in bands:
    mean_val = metadata[platform].bands.mean[band]
    std_val = metadata[platform].bands.std[band]
    wave_val = metadata[platform].bands.wavelength[band]
    print(f"  {band}: mean={mean_val}, std={std_val}, wavelength={wave_val}")

print("\n=== Wrapper Custom Metadata ===")
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

print("Wrapper custom metadata:")
for band in bands:
    mean_val = custom_metadata["bands"]["mean"][band]
    std_val = custom_metadata["bands"]["std"][band]
    wave_val = custom_metadata["bands"]["wavelength"][band]
    print(f"  {band}: mean={mean_val}, std={std_val}, wavelength={wave_val}")

print("\n=== Differences ===")
for band in bands:
    core_mean = metadata[platform].bands.mean[band]
    core_std = metadata[platform].bands.std[band]
    core_wave = metadata[platform].bands.wavelength[band]

    custom_mean = custom_metadata["bands"]["mean"][band]
    custom_std = custom_metadata["bands"]["std"][band]
    custom_wave = custom_metadata["bands"]["wavelength"][band]

    print(f"{band}:")
    print(
        f"  mean: core={core_mean}, custom={custom_mean}, diff={abs(core_mean - custom_mean)}"
    )
    print(
        f"  std: core={core_std}, custom={custom_std}, diff={abs(core_std - custom_std)}"
    )
    print(
        f"  wavelength: core={core_wave}, custom={custom_wave}, diff={abs(core_wave - custom_wave)}"
    )
