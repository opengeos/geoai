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

    def test_get_device_cpu_fallback(self):
        """Test device detection falls back to CPU when no GPU available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                device = utils.get_device()
                self.assertEqual(str(device.type), "cpu")

    def test_empty_cache_no_error(self):
        """Test that empty_cache runs without error."""
        try:
            utils.empty_cache()
        except Exception as e:
            self.fail(f"empty_cache raised {type(e).__name__}: {e}")

    def test_install_package_signature(self):
        """Test that install_package has expected parameter."""
        import inspect

        sig = inspect.signature(utils.install_package)
        self.assertIn("package", sig.parameters)

    @patch("subprocess.Popen")
    def test_install_package_calls_pip(self, mock_popen):
        """Test that install_package invokes pip via subprocess."""
        mock_process = mock_popen.return_value
        mock_process.stdout.readline.side_effect = [b"", b""]
        mock_process.poll.return_value = 0
        mock_process.wait.return_value = 0
        utils.install_package("fake-package")
        self.assertTrue(mock_popen.called)

    def test_key_functions_in_all(self):
        """Test that key functions are listed in __all__."""
        expected = [
            "get_raster_info",
            "get_vector_info",
            "calc_iou",
            "download_file",
            "temp_file_path",
        ]
        for name in expected:
            self.assertIn(name, utils.__all__, f"{name} not found in utils.__all__")


if __name__ == "__main__":
    unittest.main()
