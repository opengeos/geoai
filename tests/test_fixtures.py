"""Test fixtures and utilities for creating test data on-the-fly."""

import json
import os

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import box


def ensure_test_data_exists():
    """Ensure test data directory and files exist, creating them if necessary."""
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # List of required test files
    required_files = [
        "test_raster_rgb.tif",
        "test_raster_single.tif",
        "test_raster_multi.tif",
        "test_polygons.geojson",
        "test_config.json",
    ]

    # Check if any files are missing
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    # Create missing files
    if missing_files:
        _create_missing_files(data_dir, missing_files)

    # Ensure chips directory exists
    chips_dir = os.path.join(data_dir, "chips")
    if not os.path.exists(chips_dir) or len(os.listdir(chips_dir)) == 0:
        _create_dummy_image_chips(chips_dir)


def _create_dummy_raster(output_path, width=100, height=100, bands=3, dtype=np.uint8):
    """Create a dummy raster file for testing."""
    # Create random data
    data = np.random.randint(0, 256, (bands, height, width), dtype=dtype)

    # Define transform (arbitrary geographic bounds)
    transform = from_bounds(-122.5, 37.7, -122.3, 37.9, width, height)

    # Write raster
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def _create_dummy_vector(output_path, num_features=10):
    """Create a dummy vector file for testing."""
    geometries = []

    # Create random polygons
    for i in range(num_features):
        x = -122.5 + np.random.random() * 0.2
        y = 37.7 + np.random.random() * 0.2
        size = 0.01 + np.random.random() * 0.02

        polygon = box(x, y, x + size, y + size)
        geometries.append(
            {
                "geometry": polygon,
                "properties": {
                    "id": i,
                    "class": f"class_{i % 3}",
                    "area": polygon.area,
                },
            }
        )

    gdf = gpd.GeoDataFrame(geometries, crs="EPSG:4326")
    gdf.to_file(output_path)


def _create_dummy_image_chips(output_dir, num_chips=5, chip_size=256):
    """Create dummy image chips for training."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chips):
        # Create RGB image
        img_data = np.random.randint(0, 256, (chip_size, chip_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_data)
        img.save(os.path.join(output_dir, f"chip_{i:03d}.tif"))

        # Create corresponding mask
        mask_data = np.random.randint(0, 3, (chip_size, chip_size), dtype=np.uint8)
        mask = Image.fromarray(mask_data)
        mask.save(os.path.join(output_dir, f"mask_{i:03d}.png"))


def _create_config_file(output_path):
    """Create a test configuration JSON file."""
    config = {
        "model_config": {"num_classes": 3, "batch_size": 2, "learning_rate": 0.001},
        "data_config": {"image_size": 256, "bands": 3},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _create_missing_files(data_dir, missing_files):
    """Create specific missing files."""
    for filename in missing_files:
        filepath = os.path.join(data_dir, filename)

        if filename == "test_raster_rgb.tif":
            _create_dummy_raster(filepath, bands=3)
        elif filename == "test_raster_single.tif":
            _create_dummy_raster(filepath, bands=1)
        elif filename == "test_raster_multi.tif":
            _create_dummy_raster(filepath, bands=4)
        elif filename == "test_polygons.geojson":
            _create_dummy_vector(filepath)
        elif filename == "test_config.json":
            _create_config_file(filepath)


def get_test_data_paths():
    """Get paths to test data files, ensuring they exist first."""
    ensure_test_data_exists()

    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")

    return {
        "data_dir": data_dir,
        "test_raster_rgb": os.path.join(data_dir, "test_raster_rgb.tif"),
        "test_raster_single": os.path.join(data_dir, "test_raster_single.tif"),
        "test_raster_multi": os.path.join(data_dir, "test_raster_multi.tif"),
        "test_polygons": os.path.join(data_dir, "test_polygons.geojson"),
        "test_config": os.path.join(data_dir, "test_config.json"),
        "chips_dir": os.path.join(data_dir, "chips"),
    }
