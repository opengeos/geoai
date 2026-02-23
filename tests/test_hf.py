#!/usr/bin/env python

"""Tests for `geoai.hf` module (Hugging Face utilities)."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from geoai import hf


class TestGetModelConfig(unittest.TestCase):
    """Tests for the get_model_config function."""

    @patch("geoai.hf.AutoConfig.from_pretrained")
    def test_returns_config(self, mock_from_pretrained):
        """Test that get_model_config calls AutoConfig.from_pretrained."""
        mock_config = MagicMock()
        mock_from_pretrained.return_value = mock_config
        result = hf.get_model_config("test-model/id")
        mock_from_pretrained.assert_called_once_with("test-model/id")
        self.assertEqual(result, mock_config)

    def test_function_signature(self):
        """Test that get_model_config accepts model_id parameter."""
        sig = inspect.signature(hf.get_model_config)
        self.assertIn("model_id", sig.parameters)


class TestGetModelInputChannels(unittest.TestCase):
    """Tests for the get_model_input_channels function."""

    @patch("geoai.hf.AutoConfig.from_pretrained")
    def test_returns_channels_from_backbone_config(self, mock_from_pretrained):
        """Test extraction of num_channels from backbone_config."""
        mock_config = MagicMock()
        mock_config.backbone_config.num_channels = 4
        mock_from_pretrained.return_value = mock_config

        result = hf.get_model_input_channels("test-model")
        self.assertEqual(result, 4)

    @patch("geoai.hf.AutoModelForMaskedImageModeling.from_pretrained")
    @patch("geoai.hf.AutoConfig.from_pretrained")
    def test_defaults_to_3_channels(self, mock_config, mock_model):
        """Test that function defaults to 3 when channels can't be determined."""
        # Config without backbone_config
        config = MagicMock(spec=[])
        mock_config.return_value = config
        mock_model.side_effect = Exception("Cannot load model")

        result = hf.get_model_input_channels("unknown-model")
        self.assertEqual(result, 3)


class TestImageSegmentation(unittest.TestCase):
    """Tests for the image_segmentation function."""

    def test_function_exists(self):
        """Test that image_segmentation is available."""
        self.assertTrue(hasattr(hf, "image_segmentation"))
        self.assertTrue(callable(hf.image_segmentation))

    def test_function_signature(self):
        """Test that image_segmentation has expected parameters."""
        sig = inspect.signature(hf.image_segmentation)
        self.assertIn("tif_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("labels_to_extract", sig.parameters)


class TestMaskGeneration(unittest.TestCase):
    """Tests for the mask_generation function."""

    def test_function_exists(self):
        """Test that mask_generation is available."""
        self.assertTrue(hasattr(hf, "mask_generation"))
        self.assertTrue(callable(hf.mask_generation))


class TestModuleExports(unittest.TestCase):
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected function names."""
        expected = [
            "get_model_config",
            "get_model_input_channels",
            "image_segmentation",
            "mask_generation",
        ]
        for name in expected:
            self.assertIn(name, hf.__all__)


if __name__ == "__main__":
    unittest.main()
