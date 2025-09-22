#!/usr/bin/env python

"""Tests for `geoai` package."""

import os
import unittest

import rasterio

import geoai
from geoai import classify, extract, utils

from .test_fixtures import get_test_data_paths


class TestGeoai(unittest.TestCase):
    """Tests for `geoai` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]
        self.test_raster_single = self.test_paths["test_raster_single"]
        self.test_polygons = self.test_paths["test_polygons"]

    def test_package_import(self):
        """Test that the package imports correctly."""
        self.assertIsNotNone(geoai)
        self.assertTrue(hasattr(geoai, "Map"))
        self.assertTrue(hasattr(geoai, "LeafMap"))

    def test_map_creation(self):
        """Test Map class instantiation."""
        # Test basic map creation
        m = geoai.LeafMap()
        self.assertIsInstance(m, geoai.LeafMap)

    def test_maplibre_creation(self):
        """Test MapLibre class instantiation."""
        # Test basic MapLibre creation
        m = geoai.Map()
        self.assertIsInstance(m, geoai.Map)

    def test_test_data_exists(self):
        """Test that test data files exist."""
        self.assertTrue(
            os.path.exists(self.test_raster_rgb),
            f"Test raster {self.test_raster_rgb} not found",
        )
        self.assertTrue(
            os.path.exists(self.test_polygons),
            f"Test polygons {self.test_polygons} not found",
        )


class TestUtils(unittest.TestCase):
    """Tests for utils module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]

    def test_device_detection(self):
        """Test device detection function."""
        import torch

        device = utils.get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(str(device.type), ["cpu", "cuda", "mps"])

    def test_raster_file_operations(self):
        """Test basic raster file operations."""
        if os.path.exists(self.test_raster_rgb):
            # Test reading raster info
            with rasterio.open(self.test_raster_rgb) as src:
                self.assertEqual(src.count, 3)  # RGB bands
                self.assertGreater(src.width, 0)
                self.assertGreater(src.height, 0)


class TestClassify(unittest.TestCase):
    """Tests for classify module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]

    def test_classify_image_parameters(self):
        """Test classify_image function parameters."""
        # Test that function exists and has expected signature
        self.assertTrue(hasattr(classify, "classify_image"))

        # Test with invalid inputs (should handle gracefully)
        with self.assertRaises((FileNotFoundError, ValueError, TypeError)):
            classify.classify_image("nonexistent_file.tif", "nonexistent_model.pth")


class TestExtract(unittest.TestCase):
    """Tests for extract module."""

    def test_extract_module_imports(self):
        """Test that extract module imports correctly."""
        self.assertTrue(hasattr(extract, "__name__"))

    def test_extract_functions_exist(self):
        """Test that key extract functions exist."""
        # Test that common extract functions are available
        extract_functions = [attr for attr in dir(extract) if not attr.startswith("_")]
        self.assertGreater(
            len(extract_functions), 0, "No public functions found in extract module"
        )


if __name__ == "__main__":
    unittest.main()
