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

    def test_focal_loss_exists(self):
        """Test that FocalLoss class exists."""
        self.assertTrue(hasattr(utils, "FocalLoss"))

    def test_focal_loss_init(self):
        """Test FocalLoss initialization."""
        import torch
        
        # Test basic initialization
        loss_fn = utils.FocalLoss()
        self.assertIsInstance(loss_fn, torch.nn.Module)
        self.assertEqual(loss_fn.alpha, 1.0)
        self.assertEqual(loss_fn.gamma, 2.0)
        self.assertEqual(loss_fn.ignore_index, -100)
        
        # Test custom parameters
        loss_fn = utils.FocalLoss(alpha=0.5, gamma=3.0, ignore_index=0)
        self.assertEqual(loss_fn.alpha, 0.5)
        self.assertEqual(loss_fn.gamma, 3.0)
        self.assertEqual(loss_fn.ignore_index, 0)
        
        # Test with ignore_index=False
        loss_fn = utils.FocalLoss(ignore_index=False)
        self.assertEqual(loss_fn.ignore_index, False)

    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass."""
        import torch
        
        loss_fn = utils.FocalLoss()
        
        # Create sample inputs
        batch_size, num_classes, height, width = 2, 3, 4, 4
        inputs = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Test forward pass
        loss = loss_fn(inputs, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.ndim == 0)  # Scalar loss
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative

    def test_get_loss_function_exists(self):
        """Test that get_loss_function exists."""
        self.assertTrue(hasattr(utils, "get_loss_function"))

    def test_get_loss_function_crossentropy(self):
        """Test get_loss_function with CrossEntropy."""
        import torch
        
        loss_fn = utils.get_loss_function("crossentropy", num_classes=3)
        self.assertIsInstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_get_loss_function_focal(self):
        """Test get_loss_function with Focal loss."""
        import torch
        
        loss_fn = utils.get_loss_function("focal", num_classes=3, focal_alpha=0.5, focal_gamma=2.5)
        self.assertIsInstance(loss_fn, utils.FocalLoss)
        self.assertEqual(loss_fn.alpha, 0.5)
        self.assertEqual(loss_fn.gamma, 2.5)

    def test_get_loss_function_with_class_weights(self):
        """Test get_loss_function with class weights."""
        import torch
        
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = utils.get_loss_function(
            "crossentropy", 
            num_classes=3, 
            use_class_weights=True,
            class_weights=weights
        )
        self.assertIsInstance(loss_fn, torch.nn.CrossEntropyLoss)
        self.assertIsNotNone(loss_fn.weight)

    def test_get_loss_function_invalid(self):
        """Test get_loss_function with invalid loss name."""
        with self.assertRaises(ValueError):
            utils.get_loss_function("invalid_loss", num_classes=3)

    def test_compute_class_weights_exists(self):
        """Test that compute_class_weights function exists."""
        self.assertTrue(hasattr(utils, "compute_class_weights"))


if __name__ == "__main__":
    unittest.main()
