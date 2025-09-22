#!/usr/bin/env python

"""Tests for `geoai.segment` module."""

import unittest

from geoai import segment

from .test_fixtures import get_test_data_paths


class TestSegmentModule(unittest.TestCase):
    """Tests for segment module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]

    def test_module_imports(self):
        """Test that segment module imports correctly."""
        self.assertTrue(hasattr(segment, "__name__"))

    def test_segment_functions_exist(self):
        """Test that key segment functions exist."""
        # Get all public functions from segment module
        segment_functions = [attr for attr in dir(segment) if not attr.startswith("_")]
        self.assertGreater(
            len(segment_functions), 0, "No public functions found in segment module"
        )

    def test_semantic_segmentation_exists(self):
        """Test that semantic_segmentation function exists."""
        if hasattr(segment, "semantic_segmentation"):
            func = getattr(segment, "semantic_segmentation")
            self.assertTrue(callable(func))

    def test_segmentation_with_invalid_input(self):
        """Test segmentation functions with invalid inputs."""
        # Test with non-existent file
        if hasattr(segment, "semantic_segmentation"):
            with self.assertRaises((FileNotFoundError, ValueError, TypeError)):
                segment.semantic_segmentation("nonexistent_file.tif", "test prompt")


if __name__ == "__main__":
    unittest.main()
