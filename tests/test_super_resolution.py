#!/usr/bin/env python

"""Tests for `geoai.super_resolution` module."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
import torch

from geoai import SuperResolutionModel, create_super_resolution_model

from .test_fixtures import get_test_data_paths


class TestSuperResolution(unittest.TestCase):
    """Tests for super-resolution module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]

        # Create a simple model for testing
        self.sr_model = SuperResolutionModel(upscale_factor=2, model_type="srcnn")

    def test_module_imports(self):
        """Test that super_resolution module imports correctly."""
        try:
            from geoai import super_resolution
            self.assertTrue(hasattr(super_resolution, "SuperResolutionModel"))
        except ImportError as e:
            self.fail(f"Failed to import super_resolution module: {e}")

    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test ESRGAN model
        model_esrgan = SuperResolutionModel(model_type="esrgan", upscale_factor=4)
        self.assertEqual(model_esrgan.model_type, "esrgan")
        self.assertEqual(model_esrgan.upscale_factor, 4)

        # Test SRCNN model
        model_srcnn = SuperResolutionModel(model_type="srcnn", upscale_factor=2)
        self.assertEqual(model_srcnn.model_type, "srcnn")

        # Test invalid model type
        with self.assertRaises(ValueError):
            SuperResolutionModel(model_type="invalid")

    def test_device_setup(self):
        """Test device setup."""
        # Test CPU device
        model_cpu = SuperResolutionModel(device="cpu")
        self.assertEqual(str(model_cpu.device), "cpu")

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = SuperResolutionModel(device="cuda")
            self.assertEqual(str(model_cuda.device), "cuda")

    def test_model_save_load(self):
        """Test saving and loading model weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")

            # Save model
            self.sr_model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Load model
            new_model = SuperResolutionModel(upscale_factor=2, model_type="srcnn")
            new_model.load_model(model_path)

            # Check that weights are loaded (basic check)
            self.assertIsNotNone(new_model.model)

    def test_enhance_image_basic(self):
        """Test basic image enhancement functionality."""
        if not os.path.exists(self.test_raster_rgb):
            self.skipTest("Test raster file not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "enhanced.tif")

            # Enhance image
            result = self.sr_model.enhance_image(self.test_raster_rgb, output_path)

            # Check output
            self.assertEqual(result, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify dimensions
            with rasterio.open(self.test_raster_rgb) as src:
                original_width, original_height = src.width, src.height

            with rasterio.open(output_path) as dst:
                enhanced_width, enhanced_height = dst.width, dst.height

            # Should be 2x larger
            self.assertEqual(enhanced_width, original_width * 2)
            self.assertEqual(enhanced_height, original_height * 2)

    def test_enhance_image_return_array(self):
        """Test image enhancement returning array instead of saving."""
        if not os.path.exists(self.test_raster_rgb):
            self.skipTest("Test raster file not found")

        # Enhance image without saving
        result = self.sr_model.enhance_image(self.test_raster_rgb)

        # Should return numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 3)  # Should be (channels, height, width)

    def test_metadata_preservation(self):
        """Test that geospatial metadata is preserved."""
        if not os.path.exists(self.test_raster_rgb):
            self.skipTest("Test raster file not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "enhanced.tif")

            # Get original metadata
            with rasterio.open(self.test_raster_rgb) as src:
                original_crs = src.crs
                original_bounds = src.bounds

            # Enhance image
            self.sr_model.enhance_image(self.test_raster_rgb, output_path)

            # Check metadata preservation
            with rasterio.open(output_path) as dst:
                enhanced_crs = dst.crs
                enhanced_bounds = dst.bounds

            self.assertEqual(enhanced_crs, original_crs)
            # Bounds should be scaled appropriately
            self.assertAlmostEqual(enhanced_bounds.left, original_bounds.left, places=1)
            self.assertAlmostEqual(enhanced_bounds.top, original_bounds.top, places=1)

    def test_different_upscale_factors(self):
        """Test different upscaling factors."""
        if not os.path.exists(self.test_raster_rgb):
            self.skipTest("Test raster file not found")

        for factor in [2, 4]:
            with self.subTest(factor=factor):
                model = SuperResolutionModel(upscale_factor=factor, model_type="srcnn")

                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = os.path.join(temp_dir, f"enhanced_{factor}x.tif")
                    model.enhance_image(self.test_raster_rgb, output_path)

                    with rasterio.open(self.test_raster_rgb) as src:
                        original_width, original_height = src.width, src.height

                    with rasterio.open(output_path) as dst:
                        enhanced_width, enhanced_height = dst.width, dst.height

                    self.assertEqual(enhanced_width, original_width * factor)
                    self.assertEqual(enhanced_height, original_height * factor)

    def test_tiled_processing(self):
        """Test tiled processing for large images."""
        # Create a larger test image
        with tempfile.TemporaryDirectory() as temp_dir:
            large_image_path = os.path.join(temp_dir, "large_test.tif")

            # Create a 1024x1024 test image
            data = np.random.randint(0, 256, (3, 1024, 1024), dtype=np.uint8)
            transform = rasterio.transform.from_bounds(-122.5, 37.7, -122.3, 37.9, 1024, 1024)

            with rasterio.open(
                large_image_path,
                "w",
                driver="GTiff",
                height=1024,
                width=1024,
                count=3,
                dtype=np.uint8,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)

            # Test with small tile size to force tiling
            output_path = os.path.join(temp_dir, "enhanced_large.tif")
            self.sr_model.enhance_image(large_image_path, output_path, tile_size=256)

            # Verify output dimensions
            with rasterio.open(output_path) as dst:
                self.assertEqual(dst.width, 2048)  # 2x upscale
                self.assertEqual(dst.height, 2048)

    def test_create_super_resolution_model(self):
        """Test the convenience function for creating models."""
        model = create_super_resolution_model(model_type="srcnn", upscale_factor=2)
        self.assertIsInstance(model, SuperResolutionModel)
        self.assertEqual(model.model_type, "srcnn")
        self.assertEqual(model.upscale_factor, 2)

    def test_model_evaluation(self):
        """Test model evaluation with metrics."""
        # Create dummy test data
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_data")
            os.makedirs(test_dir)

            # Create a simple test image
            data = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
            transform = rasterio.transform.from_bounds(-122.5, 37.7, -122.3, 37.9, 64, 64)

            test_image_path = os.path.join(test_dir, "test.tif")
            with rasterio.open(
                test_image_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=3,
                dtype=np.uint8,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)

            # Test evaluation (this will use the untrained model)
            try:
                results = self.sr_model.evaluate(test_dir, metrics=['psnr'])
                self.assertIn('psnr', results)
                self.assertIsInstance(results['psnr'], (int, float))
            except ImportError:
                # Skip if scikit-image is not available
                self.skipTest("scikit-image not available for evaluation")

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with non-existent file
        with self.assertRaises((FileNotFoundError, rasterio.errors.RasterioIOError)):
            self.sr_model.enhance_image("nonexistent.tif")

        # Test with invalid model path
        with self.assertRaises(FileNotFoundError):
            self.sr_model.load_model("nonexistent.pth")

    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_fallback(self, mock_cuda):
        """Test CPU fallback when CUDA is not available."""
        model = SuperResolutionModel(device=None)  # Should default to CPU
        self.assertEqual(str(model.device), "cpu")


if __name__ == "__main__":
    unittest.main()