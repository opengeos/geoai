#!/usr/bin/env python

"""Tests for `geoai.prithvi` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestPrithviImport(unittest.TestCase):
    """Tests for prithvi module import behavior."""

    def test_module_imports(self):
        """Test that the prithvi module can be imported."""
        import geoai.prithvi

        self.assertTrue(hasattr(geoai.prithvi, "PrithviProcessor"))

    def test_prithvi_processor_class_exists(self):
        """Test that PrithviProcessor class exists."""
        from geoai.prithvi import PrithviProcessor

        self.assertTrue(callable(PrithviProcessor))

    def test_prithvi_inference_exists(self):
        """Test that prithvi_inference convenience function exists."""
        from geoai.prithvi import prithvi_inference

        self.assertTrue(callable(prithvi_inference))

    def test_load_prithvi_model_exists(self):
        """Test that load_prithvi_model convenience function exists."""
        from geoai.prithvi import load_prithvi_model

        self.assertTrue(callable(load_prithvi_model))

    def test_get_available_prithvi_models_exists_and_returns_list(self):
        """Test that get_available_prithvi_models exists and returns a list."""
        from geoai.prithvi import get_available_prithvi_models

        self.assertTrue(callable(get_available_prithvi_models))
        result = get_available_prithvi_models()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_available_models_constant(self):
        """Test that AVAILABLE_MODELS list is accessible and non-empty."""
        from geoai.prithvi import AVAILABLE_MODELS

        self.assertIsInstance(AVAILABLE_MODELS, list)
        self.assertIn("Prithvi-EO-2.0-300M-TL", AVAILABLE_MODELS)


class TestPrithviAllExports(unittest.TestCase):
    """Tests for prithvi module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined in the prithvi module."""
        import geoai.prithvi

        self.assertTrue(hasattr(geoai.prithvi, "__all__"))

    def test_all_exports_contain_expected_names(self):
        """Test that __all__ contains the expected public API names."""
        from geoai.prithvi import __all__

        expected = [
            "PrithviProcessor",
            "PrithviViT",
            "PrithviMAE",
            "MAEDecoder",
            "PatchEmbed",
            "TemporalEncoder",
            "LocationEncoder",
            "get_available_prithvi_models",
            "load_prithvi_model",
            "prithvi_inference",
            "AVAILABLE_MODELS",
        ]
        for name in expected:
            self.assertIn(name, __all__)


class TestPrithviSignatures(unittest.TestCase):
    """Tests for prithvi function and method signatures."""

    def test_prithvi_processor_init_params(self):
        """Test PrithviProcessor.__init__ has expected parameters."""
        from geoai.prithvi import PrithviProcessor

        sig = inspect.signature(PrithviProcessor.__init__)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("config_path", sig.parameters)
        self.assertIn("checkpoint_path", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("cache_dir", sig.parameters)

    def test_load_prithvi_model_params(self):
        """Test load_prithvi_model function has expected parameters."""
        from geoai.prithvi import load_prithvi_model

        sig = inspect.signature(load_prithvi_model)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("cache_dir", sig.parameters)

    def test_prithvi_inference_params(self):
        """Test prithvi_inference function has expected parameters."""
        from geoai.prithvi import prithvi_inference

        sig = inspect.signature(prithvi_inference)
        self.assertIn("file_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("mask_ratio", sig.parameters)
        self.assertIn("device", sig.parameters)

    def test_process_files_params(self):
        """Test PrithviProcessor.process_files has expected parameters."""
        from geoai.prithvi import PrithviProcessor

        sig = inspect.signature(PrithviProcessor.process_files)
        self.assertIn("file_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("mask_ratio", sig.parameters)
        self.assertIn("indices", sig.parameters)

    def test_run_inference_params(self):
        """Test PrithviProcessor.run_inference has expected parameters."""
        from geoai.prithvi import PrithviProcessor

        sig = inspect.signature(PrithviProcessor.run_inference)
        self.assertIn("input_data", sig.parameters)
        self.assertIn("temporal_coords", sig.parameters)
        self.assertIn("location_coords", sig.parameters)
        self.assertIn("mask_ratio", sig.parameters)


class TestPrithviInit(unittest.TestCase):
    """Tests for PrithviProcessor initialization."""

    @patch("geoai.prithvi.PrithviProcessor._load_model")
    @patch("geoai.prithvi.PrithviProcessor.download_model")
    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_init_with_mocked_dependencies(
        self, mock_json_load, mock_open, mock_download, mock_load_model
    ):
        """Test that init succeeds with mocked model loading."""
        mock_download.return_value = ("/fake/config.json", "/fake/checkpoint.pt")
        mock_json_load.return_value = {
            "pretrained_cfg": {
                "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
                "img_size": 224,
                "patch_size": [1, 16, 16],
                "mask_ratio": 0.75,
                "num_frames": 4,
                "coords_encoding": ["time", "location"],
            }
        }
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        from geoai.prithvi import PrithviProcessor

        processor = PrithviProcessor()
        self.assertIsNotNone(processor.model)
        self.assertEqual(processor.model_name, "Prithvi-EO-2.0-300M-TL")
        mock_download.assert_called_once()

    def test_get_available_prithvi_models_returns_copy(self):
        """Test that get_available_prithvi_models returns a copy, not the original."""
        from geoai.prithvi import AVAILABLE_MODELS, get_available_prithvi_models

        result = get_available_prithvi_models()
        self.assertEqual(result, AVAILABLE_MODELS)
        # Mutating the result should not affect the original
        result.append("fake_model")
        self.assertNotIn("fake_model", AVAILABLE_MODELS)


if __name__ == "__main__":
    unittest.main()
