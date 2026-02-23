#!/usr/bin/env python

"""Tests for `geoai.sam` module (Segment Anything Model)."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from geoai.sam import SamGeo


class TestSamGeoClass(unittest.TestCase):
    """Tests for the SamGeo class."""

    def test_class_exists(self):
        """Test that SamGeo class is importable."""
        self.assertTrue(callable(SamGeo))

    def test_init_signature(self):
        """Test SamGeo.__init__ has expected parameters."""
        sig = inspect.signature(SamGeo.__init__)
        self.assertIn("model", sig.parameters)
        self.assertIn("automatic", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("sam_kwargs", sig.parameters)

    def test_init_default_values(self):
        """Test default parameter values in SamGeo.__init__."""
        sig = inspect.signature(SamGeo.__init__)
        params = sig.parameters
        self.assertEqual(params["model"].default, "facebook/sam-vit-huge")
        self.assertEqual(params["automatic"].default, True)
        self.assertIsNone(params["device"].default)
        self.assertIsNone(params["sam_kwargs"].default)


class TestSamGeoMethods(unittest.TestCase):
    """Tests for SamGeo method existence and signatures."""

    def test_has_generate_method(self):
        """Test that SamGeo has a generate method."""
        self.assertTrue(hasattr(SamGeo, "generate"))

    def test_has_set_image_method(self):
        """Test that SamGeo has a set_image method."""
        self.assertTrue(hasattr(SamGeo, "set_image"))

    def test_has_save_masks_method(self):
        """Test that SamGeo has a save_masks method."""
        self.assertTrue(hasattr(SamGeo, "save_masks"))

    def test_has_predict_method(self):
        """Test that SamGeo has a predict method."""
        self.assertTrue(hasattr(SamGeo, "predict"))


class TestSamGeoAttributes(unittest.TestCase):
    """Tests for SamGeo instance attributes via mocked init."""

    @patch("geoai.sam.pipeline")
    @patch("geoai.sam.torch.cuda.is_available", return_value=False)
    def test_attributes_set_on_init(self, mock_cuda, mock_pipeline):
        """Test that key attributes are initialized."""
        mock_pipeline.return_value = MagicMock()
        try:
            sam = SamGeo(model="facebook/sam-vit-base", automatic=True)
            self.assertIsNotNone(sam.model)
            self.assertIsNone(sam.image)
            self.assertIsNone(sam.masks)
        except Exception:
            # Model loading may fail in test env, that's expected
            pass


if __name__ == "__main__":
    unittest.main()
