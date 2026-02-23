#!/usr/bin/env python

"""Tests for `geoai.canopy` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestCanopyImport(unittest.TestCase):
    """Tests for canopy module import behavior."""

    def test_module_imports(self):
        """Test that the canopy module can be imported."""
        import geoai.canopy

        self.assertTrue(hasattr(geoai.canopy, "CanopyHeightEstimation"))

    def test_canopy_height_estimation_class_exists(self):
        """Test that CanopyHeightEstimation class exists."""
        from geoai.canopy import CanopyHeightEstimation

        self.assertTrue(callable(CanopyHeightEstimation))

    def test_canopy_height_estimation_function_exists(self):
        """Test that canopy_height_estimation convenience function exists."""
        from geoai.canopy import canopy_height_estimation

        self.assertTrue(callable(canopy_height_estimation))

    def test_list_canopy_models_exists(self):
        """Test that list_canopy_models function exists and returns a dict."""
        from geoai.canopy import list_canopy_models

        self.assertTrue(callable(list_canopy_models))
        result = list_canopy_models()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_model_variants_exists(self):
        """Test that MODEL_VARIANTS dict is available."""
        from geoai.canopy import MODEL_VARIANTS

        self.assertIsInstance(MODEL_VARIANTS, dict)
        self.assertIn("compressed_SSLhuge", MODEL_VARIANTS)

    def test_default_cache_dir_exists(self):
        """Test that DEFAULT_CACHE_DIR string is available."""
        from geoai.canopy import DEFAULT_CACHE_DIR

        self.assertIsInstance(DEFAULT_CACHE_DIR, str)
        self.assertIn("canopy", DEFAULT_CACHE_DIR)


class TestCanopyAllExports(unittest.TestCase):
    """Tests for canopy module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined in the canopy module."""
        import geoai.canopy

        self.assertTrue(hasattr(geoai.canopy, "__all__"))

    def test_all_exports_contain_expected_names(self):
        """Test that __all__ contains the expected public API names."""
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


class TestCanopySignatures(unittest.TestCase):
    """Tests for canopy class and function signatures."""

    def test_canopy_height_estimation_init_params(self):
        """Test CanopyHeightEstimation.__init__ has expected parameters."""
        from geoai.canopy import CanopyHeightEstimation

        sig = inspect.signature(CanopyHeightEstimation.__init__)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("checkpoint_path", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("cache_dir", sig.parameters)

    def test_predict_method_params(self):
        """Test predict method has expected parameters."""
        from geoai.canopy import CanopyHeightEstimation

        sig = inspect.signature(CanopyHeightEstimation.predict)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("scale_factor", sig.parameters)

    def test_visualize_method_params(self):
        """Test visualize method has expected parameters."""
        from geoai.canopy import CanopyHeightEstimation

        sig = inspect.signature(CanopyHeightEstimation.visualize)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("height_map", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("cmap", sig.parameters)

    def test_convenience_function_params(self):
        """Test canopy_height_estimation function has expected parameters."""
        from geoai.canopy import canopy_height_estimation

        sig = inspect.signature(canopy_height_estimation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)


class TestCanopyInit(unittest.TestCase):
    """Tests for CanopyHeightEstimation initialization."""

    def test_init_raises_for_unknown_model(self):
        """Test that init raises ValueError for an unknown model variant."""
        from geoai.canopy import CanopyHeightEstimation

        with self.assertRaises(ValueError):
            CanopyHeightEstimation(model_name="nonexistent_model")

    @patch("geoai.canopy._load_model")
    @patch("geoai.canopy._download_checkpoint")
    def test_init_with_mocked_dependencies(self, mock_download, mock_load):
        """Test that init succeeds with mocked model loading."""
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

    def test_list_canopy_models_returns_descriptions(self):
        """Test that list_canopy_models returns model descriptions."""
        from geoai.canopy import MODEL_VARIANTS, list_canopy_models

        models = list_canopy_models()
        for name, desc in models.items():
            self.assertIn(name, MODEL_VARIANTS)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


if __name__ == "__main__":
    unittest.main()
