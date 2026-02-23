#!/usr/bin/env python

"""Tests for `geoai.timm_regress` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestTimmRegressImport(unittest.TestCase):
    """Tests for timm_regress module import behavior."""

    def test_module_imports(self):
        """Test that the timm_regress module can be imported."""
        import geoai.timm_regress

        self.assertTrue(hasattr(geoai.timm_regress, "PixelRegressionModel"))

    def test_pixel_regression_model_exists(self):
        """Test that PixelRegressionModel class exists."""
        from geoai.timm_regress import PixelRegressionModel

        self.assertTrue(callable(PixelRegressionModel))

    def test_pixel_regression_dataset_exists(self):
        """Test that PixelRegressionDataset class exists."""
        from geoai.timm_regress import PixelRegressionDataset

        self.assertTrue(callable(PixelRegressionDataset))

    def test_train_pixel_regressor_exists(self):
        """Test that train_pixel_regressor function exists."""
        from geoai.timm_regress import train_pixel_regressor

        self.assertTrue(callable(train_pixel_regressor))

    def test_predict_raster_exists(self):
        """Test that predict_raster function exists."""
        from geoai.timm_regress import predict_raster

        self.assertTrue(callable(predict_raster))

    def test_evaluate_regression_exists(self):
        """Test that evaluate_regression function exists."""
        from geoai.timm_regress import evaluate_regression

        self.assertTrue(callable(evaluate_regression))

    def test_create_regression_tiles_exists(self):
        """Test that create_regression_tiles function exists."""
        from geoai.timm_regress import create_regression_tiles

        self.assertTrue(callable(create_regression_tiles))

    def test_plot_regression_comparison_exists(self):
        """Test that plot_regression_comparison function exists."""
        from geoai.timm_regress import plot_regression_comparison

        self.assertTrue(callable(plot_regression_comparison))


class TestTimmRegressBackwardCompat(unittest.TestCase):
    """Tests for backward compatibility aliases."""

    def test_timm_regressor_alias(self):
        """Test that TimmRegressor alias exists and points to PixelRegressionModel."""
        from geoai.timm_regress import TimmRegressor, PixelRegressionModel

        self.assertIs(TimmRegressor, PixelRegressionModel)

    def test_regression_dataset_alias(self):
        """Test that RegressionDataset alias exists and points to PixelRegressionDataset."""
        from geoai.timm_regress import RegressionDataset, PixelRegressionDataset

        self.assertIs(RegressionDataset, PixelRegressionDataset)

    def test_train_timm_regressor_alias(self):
        """Test that train_timm_regressor alias exists and points to train_pixel_regressor."""
        from geoai.timm_regress import train_timm_regressor, train_pixel_regressor

        self.assertIs(train_timm_regressor, train_pixel_regressor)

    def test_create_regression_patches_alias(self):
        """Test that create_regression_patches alias exists and points to create_regression_tiles."""
        from geoai.timm_regress import (
            create_regression_patches,
            create_regression_tiles,
        )

        self.assertIs(create_regression_patches, create_regression_tiles)


class TestPixelRegressionModelSignatures(unittest.TestCase):
    """Tests for PixelRegressionModel class signatures."""

    def test_init_params(self):
        """Test PixelRegressionModel.__init__ has expected parameters."""
        from geoai.timm_regress import PixelRegressionModel

        sig = inspect.signature(PixelRegressionModel.__init__)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("in_channels", sig.parameters)
        self.assertIn("encoder_weights", sig.parameters)
        self.assertIn("learning_rate", sig.parameters)
        self.assertIn("weight_decay", sig.parameters)
        self.assertIn("freeze_encoder", sig.parameters)
        self.assertIn("loss_fn", sig.parameters)
        self.assertIn("loss_type", sig.parameters)

    def test_init_default_values(self):
        """Test PixelRegressionModel.__init__ default parameter values."""
        from geoai.timm_regress import PixelRegressionModel

        sig = inspect.signature(PixelRegressionModel.__init__)
        self.assertEqual(sig.parameters["encoder_name"].default, "resnet50")
        self.assertEqual(sig.parameters["architecture"].default, "unet")
        self.assertEqual(sig.parameters["in_channels"].default, 3)
        self.assertEqual(sig.parameters["encoder_weights"].default, "imagenet")
        self.assertEqual(sig.parameters["learning_rate"].default, 1e-4)
        self.assertEqual(sig.parameters["loss_type"].default, "mse")
        self.assertEqual(sig.parameters["freeze_encoder"].default, False)


class TestTimmRegressFunctionSignatures(unittest.TestCase):
    """Tests for timm_regress function signatures."""

    def test_train_pixel_regressor_params(self):
        """Test train_pixel_regressor has expected parameters."""
        from geoai.timm_regress import train_pixel_regressor

        sig = inspect.signature(train_pixel_regressor)
        self.assertIn("train_image_paths", sig.parameters)
        self.assertIn("train_target_paths", sig.parameters)
        self.assertIn("val_image_paths", sig.parameters)
        self.assertIn("val_target_paths", sig.parameters)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("in_channels", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("num_epochs", sig.parameters)
        self.assertIn("learning_rate", sig.parameters)
        self.assertIn("loss_type", sig.parameters)
        self.assertIn("patience", sig.parameters)
        self.assertIn("verbose", sig.parameters)

    def test_predict_raster_params(self):
        """Test predict_raster has expected parameters."""
        from geoai.timm_regress import predict_raster

        sig = inspect.signature(predict_raster)
        self.assertIn("model", sig.parameters)
        self.assertIn("input_raster", sig.parameters)
        self.assertIn("output_raster", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("input_bands", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("output_nodata", sig.parameters)
        self.assertIn("clip_range", sig.parameters)

    def test_evaluate_regression_params(self):
        """Test evaluate_regression has expected parameters."""
        from geoai.timm_regress import evaluate_regression

        sig = inspect.signature(evaluate_regression)
        self.assertIn("y_true", sig.parameters)
        self.assertIn("y_pred", sig.parameters)
        self.assertIn("mask", sig.parameters)
        self.assertIn("print_results", sig.parameters)

    def test_create_regression_tiles_params(self):
        """Test create_regression_tiles has expected parameters."""
        from geoai.timm_regress import create_regression_tiles

        sig = inspect.signature(create_regression_tiles)
        self.assertIn("input_raster", sig.parameters)
        self.assertIn("target_raster", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("stride", sig.parameters)
        self.assertIn("input_bands", sig.parameters)
        self.assertIn("target_band", sig.parameters)
        self.assertIn("min_valid_ratio", sig.parameters)


class TestEvaluateRegression(unittest.TestCase):
    """Tests for evaluate_regression function behavior."""

    def test_evaluate_regression_returns_metrics(self):
        """Test that evaluate_regression returns expected metric keys."""
        from geoai.timm_regress import evaluate_regression

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = evaluate_regression(y_true, y_pred, print_results=False)

        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

    def test_evaluate_regression_perfect_prediction(self):
        """Test evaluate_regression with perfect predictions."""
        from geoai.timm_regress import evaluate_regression

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = evaluate_regression(y_true, y_pred, print_results=False)

        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_evaluate_regression_with_mask(self):
        """Test evaluate_regression with a valid pixel mask."""
        from geoai.timm_regress import evaluate_regression

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, True, True, False, False])

        metrics = evaluate_regression(y_true, y_pred, mask=mask, print_results=False)

        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)


class TestTimmRegressAvailabilityFlags(unittest.TestCase):
    """Tests for optional dependency availability flags."""

    def test_timm_available_flag_exists(self):
        """Test that TIMM_AVAILABLE flag is defined."""
        from geoai.timm_regress import TIMM_AVAILABLE

        self.assertIsInstance(TIMM_AVAILABLE, bool)

    def test_smp_available_flag_exists(self):
        """Test that SMP_AVAILABLE flag is defined."""
        from geoai.timm_regress import SMP_AVAILABLE

        self.assertIsInstance(SMP_AVAILABLE, bool)

    def test_lightning_available_flag_exists(self):
        """Test that LIGHTNING_AVAILABLE flag is defined."""
        from geoai.timm_regress import LIGHTNING_AVAILABLE

        self.assertIsInstance(LIGHTNING_AVAILABLE, bool)


class TestPixelRegressionDatasetInit(unittest.TestCase):
    """Tests for PixelRegressionDataset initialization."""

    def test_dataset_init_params(self):
        """Test PixelRegressionDataset.__init__ has expected parameters."""
        from geoai.timm_regress import PixelRegressionDataset

        sig = inspect.signature(PixelRegressionDataset.__init__)
        self.assertIn("image_paths", sig.parameters)
        self.assertIn("target_paths", sig.parameters)
        self.assertIn("input_bands", sig.parameters)
        self.assertIn("target_band", sig.parameters)
        self.assertIn("transform", sig.parameters)
        self.assertIn("normalize_input", sig.parameters)

    def test_dataset_mismatched_lengths(self):
        """Test that PixelRegressionDataset raises ValueError for mismatched lengths."""
        from geoai.timm_regress import PixelRegressionDataset

        with self.assertRaises(ValueError):
            PixelRegressionDataset(
                image_paths=["a.tif", "b.tif"],
                target_paths=["a.tif"],
            )


if __name__ == "__main__":
    unittest.main()
