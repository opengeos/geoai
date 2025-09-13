#!/usr/bin/env python3
"""Script to create dummy datasets for testing."""

import json
import os

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import box


def create_dummy_raster(output_path, width=100, height=100, bands=3, dtype=np.uint8):
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


def create_dummy_vector(output_path, num_features=10):
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


def create_dummy_image_chips(output_dir, num_chips=5, chip_size=256):
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


def create_test_datasets():
    """Create all test datasets."""
    data_dir = "tests/data"
    os.makedirs(data_dir, exist_ok=True)

    # Create dummy raster files
    create_dummy_raster(os.path.join(data_dir, "test_raster_rgb.tif"), bands=3)
    create_dummy_raster(os.path.join(data_dir, "test_raster_single.tif"), bands=1)
    create_dummy_raster(os.path.join(data_dir, "test_raster_multi.tif"), bands=4)

    # Create dummy vector files
    create_dummy_vector(os.path.join(data_dir, "test_polygons.geojson"))

    # Create image chips directory
    create_dummy_image_chips(os.path.join(data_dir, "chips"))

    # Create a simple JSON config file
    config = {
        "model_config": {"num_classes": 3, "batch_size": 2, "learning_rate": 0.001},
        "data_config": {"image_size": 256, "bands": 3},
    }

    with open(os.path.join(data_dir, "test_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Created test datasets in {data_dir}")


if __name__ == "__main__":
    create_test_datasets()
