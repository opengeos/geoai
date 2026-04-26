#!/usr/bin/env python

"""Tests for `geoai.canopy` module."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestCanopyImport(unittest.TestCase):
    """Tests for canopy module import behavior."""

    def test_module_imports(self):
        import geoai.canopy

        self.assertTrue(hasattr(geoai.canopy, "CanopyHeightEstimation"))

    def test_canopy_height_estimation_class_exists(self):
        from geoai.canopy import CanopyHeightEstimation

        self.assertTrue(callable(CanopyHeightEstimation))

    def test_canopy_height_estimation_function_exists(self):
        from geoai.canopy import canopy_height_estimation

        self.assertTrue(callable(canopy_height_estimation))

    def test_list_canopy_models_exists(self):
        from geoai.canopy import list_canopy_models

        result = list_canopy_models()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_model_variants_exists(self):
        from geoai.canopy import MODEL_VARIANTS

        self.assertIsInstance(MODEL_VARIANTS, dict)
        self.assertIn("compressed_SSLhuge", MODEL_VARIANTS)

    def test_default_cache_dir_exists(self):
        from geoai.canopy import DEFAULT_CACHE_DIR

        self.assertIsInstance(DEFAULT_CACHE_DIR, str)
        self.assertIn("canopy", DEFAULT_CACHE_DIR)


class TestCanopyAllExports(unittest.TestCase):
    def test_all_exports_defined(self):
        import geoai.canopy

        self.assertTrue(hasattr(geoai.canopy, "__all__"))

    def test_all_exports_contain_expected_names(self):
        from geoai.canopy import __all__

        expected = [
            "CanopyHeightEstimation",
            "canopy_height_estimation",
            "list_canopy_models",
            "MODEL_VARIANTS",
            "DEFAULT_CACHE_DIR",
        ]
        for name in expected:
            self.assertIn(name, __all__)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestCanopySignatures(unittest.TestCase):
    def test_canopy_height_estimation_init_params(self):
        from geoai.canopy import CanopyHeightEstimation

        sig = inspect.signature(CanopyHeightEstimation.__init__)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("checkpoint_path", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("cache_dir", sig.parameters)

    def test_predict_method_params(self):
        from geoai.canopy import CanopyHeightEstimation

        sig = inspect.signature(CanopyHeightEstimation.predict)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("scale_factor", sig.parameters)

    def test_visualize_method_params(self):
        from geoai.canopy import CanopyHeightEstimation

        sig = inspect.signature(CanopyHeightEstimation.visualize)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("height_map", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("cmap", sig.parameters)

    def test_convenience_function_params(self):
        from geoai.canopy import canopy_height_estimation

        sig = inspect.signature(canopy_height_estimation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)


# ---------------------------------------------------------------------------
# Init / config tests
# ---------------------------------------------------------------------------


class TestCanopyInit(unittest.TestCase):
    def test_init_raises_for_unknown_model(self):
        from geoai.canopy import CanopyHeightEstimation

        with self.assertRaises(ValueError) as ctx:
            CanopyHeightEstimation(model_name="nonexistent_model")
        self.assertIn("nonexistent_model", str(ctx.exception))

    @patch("geoai.canopy._load_model")
    @patch("geoai.canopy._download_checkpoint")
    def test_init_with_mocked_dependencies(self, mock_download, mock_load):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_load.return_value = mock_model
        mock_download.return_value = "/fake/checkpoint.pth"

        from geoai.canopy import CanopyHeightEstimation

        estimator = CanopyHeightEstimation(device="cpu")
        self.assertIsNotNone(estimator.model)
        self.assertEqual(estimator.model_name, "compressed_SSLhuge")
        self.assertEqual(estimator.device, "cpu")
        mock_download.assert_called_once()
        mock_load.assert_called_once()

    @patch("geoai.canopy._load_model")
    @patch("geoai.canopy._download_checkpoint")
    def test_init_custom_checkpoint(self, mock_download, mock_load):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_load.return_value = mock_model

        from geoai.canopy import CanopyHeightEstimation

        estimator = CanopyHeightEstimation(
            checkpoint_path="/custom/checkpoint.pth", device="cpu"
        )
        self.assertEqual(estimator.checkpoint_path, "/custom/checkpoint.pth")
        mock_download.assert_not_called()

    def test_list_canopy_models_returns_descriptions(self):
        from geoai.canopy import MODEL_VARIANTS, list_canopy_models

        models = list_canopy_models()
        for name, desc in models.items():
            self.assertIn(name, MODEL_VARIANTS)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)

    def test_model_variants_has_required_keys(self):
        from geoai.canopy import MODEL_VARIANTS

        for name, info in MODEL_VARIANTS.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(info, dict)
            self.assertIn("url", info)
            self.assertIn("description", info)


# ---------------------------------------------------------------------------
# _make_weight_map tests
# ---------------------------------------------------------------------------


class TestMakeWeightMap(unittest.TestCase):
    """Tests for the raised-cosine weight map used in tile blending."""

    def _make_map(self, tile_size, overlap):
        from geoai.canopy import CanopyHeightEstimation

        return CanopyHeightEstimation._make_weight_map(tile_size, overlap)

    def test_no_overlap_returns_ones(self):
        w = self._make_map(256, 0)
        np.testing.assert_array_equal(w, np.ones((256, 256), dtype=np.float32))

    def test_negative_overlap_returns_ones(self):
        w = self._make_map(256, -10)
        np.testing.assert_array_equal(w, np.ones((256, 256), dtype=np.float32))

    def test_shape_and_dtype(self):
        w = self._make_map(128, 32)
        self.assertEqual(w.shape, (128, 128))
        self.assertEqual(w.dtype, np.float32)

    def test_center_is_one(self):
        w = self._make_map(256, 64)
        center = 256 // 2
        self.assertAlmostEqual(w[center, center], 1.0, places=5)

    def test_corners_are_small(self):
        w = self._make_map(256, 64)
        self.assertLess(w[0, 0], 0.01)

    def test_edges_taper(self):
        w = self._make_map(128, 32)
        # First pixel should be less than middle pixel in overlap zone
        self.assertLess(w[0, 64], w[16, 64])

    def test_symmetry(self):
        w = self._make_map(128, 32)
        np.testing.assert_allclose(w, w[::-1, :], atol=1e-6)
        np.testing.assert_allclose(w, w[:, ::-1], atol=1e-6)

    def test_values_in_zero_one_range(self):
        w = self._make_map(256, 128)
        self.assertGreaterEqual(w.min(), 0.0)
        self.assertLessEqual(w.max(), 1.0)


# ---------------------------------------------------------------------------
# _normalize_image tests
# ---------------------------------------------------------------------------


class TestNormalizeImage(unittest.TestCase):
    """Tests for image normalization in CanopyHeightEstimation."""

    @patch("geoai.canopy._load_model")
    @patch("geoai.canopy._download_checkpoint")
    def setUp(self, mock_download, mock_load):
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_load.return_value = mock_model
        mock_download.return_value = "/fake/checkpoint.pth"

        from geoai.canopy import CanopyHeightEstimation

        self.estimator = CanopyHeightEstimation(device="cpu")

    def test_normalize_applies_mean_std(self):
        import torch

        img = torch.rand(1, 3, 64, 64)
        normalized = self.estimator._normalize_image(img)
        self.assertEqual(normalized.shape, img.shape)
        # Should not be identical after normalization
        self.assertFalse(torch.allclose(img, normalized))

    def test_normalize_constant_image(self):
        import torch

        img = torch.full((1, 3, 64, 64), 0.5)
        normalized = self.estimator._normalize_image(img)
        # Should produce valid output (not NaN or Inf)
        self.assertFalse(torch.isnan(normalized).any())
        self.assertFalse(torch.isinf(normalized).any())

    def test_normalize_zeros(self):
        import torch

        img = torch.zeros(1, 3, 64, 64)
        normalized = self.estimator._normalize_image(img)
        self.assertFalse(torch.isnan(normalized).any())


# ---------------------------------------------------------------------------
# _download_checkpoint tests
# ---------------------------------------------------------------------------


class TestDownloadCheckpoint(unittest.TestCase):
    def test_raises_for_invalid_model(self):
        from geoai.canopy import _download_checkpoint

        with self.assertRaises(ValueError):
            _download_checkpoint("invalid_model_name")

    def test_skips_download_if_exists(self):
        from geoai.canopy import _download_checkpoint

        with tempfile.TemporaryDirectory() as td:
            # Pre-create the expected checkpoint file
            from geoai.canopy import MODEL_VARIANTS

            info = MODEL_VARIANTS["compressed_SSLhuge"]
            ckpt_path = os.path.join(td, info["filename"])
            with open(ckpt_path, "w") as f:
                f.write("fake")

            path = _download_checkpoint("compressed_SSLhuge", td)
            self.assertEqual(path, ckpt_path)


if __name__ == "__main__":
    unittest.main()
