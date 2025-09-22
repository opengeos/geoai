#!/usr/bin/env python

"""Tests for `geoai.utils` module."""

import os
import unittest
from unittest.mock import patch

from geoai import utils

from .test_fixtures import get_test_data_paths


class TestUtilsFunctions(unittest.TestCase):
    """Tests for utils module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]
        self.test_raster_single = self.test_paths["test_raster_single"]

    def test_get_device(self):
        """Test device detection function."""
        import torch

        device = utils.get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(str(device.type), ["cpu", "cuda", "mps"])

    def test_get_device_with_cuda_available(self):
        """Test device detection when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = utils.get_device()
            self.assertEqual(str(device.type), "cuda")

    def test_get_device_with_mps_available(self):
        """Test device detection when MPS is available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                device = utils.get_device()
                self.assertEqual(str(device.type), "mps")

    def test_view_raster(self):
        """Test view_raster function."""
        if hasattr(utils, "view_raster") and os.path.exists(self.test_raster_rgb):
            # Test that function doesn't crash with valid input
            try:
                utils.view_raster(self.test_raster_rgb, show_colorbar=False)
                # Should return successfully without error
            except Exception:
                self.fail("view_raster failed with valid input")

    def test_utility_functions_exist(self):
        """Test that key utility functions exist."""
        expected_functions = ["get_device"]

        for func_name in expected_functions:
            self.assertTrue(
                hasattr(utils, func_name),
                f"Function {func_name} not found in utils module",
            )

    def test_import_checks(self):
        """Test that required imports are available."""
        # Test that utils module imports without error
        try:
            import geoai.utils  # noqa: F401
        except ImportError as e:
            self.fail(f"Failed to import geoai.utils: {e}")


if __name__ == "__main__":
    unittest.main()
