#!/usr/bin/env python

"""Tests for `geoai.change_detection` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestChangeDetectionImport(unittest.TestCase):
    """Tests for change_detection module import behavior."""

    def test_module_imports(self):
        """Test that the change_detection module can be imported."""
        import geoai.change_detection

        self.assertTrue(hasattr(geoai.change_detection, "ChangeDetection"))

    def test_changestar_detection_exists(self):
        """Test that ChangeStarDetection class exists."""
        from geoai.change_detection import ChangeStarDetection

        self.assertTrue(callable(ChangeStarDetection))

    def test_list_changestar_models_exists(self):
        """Test that list_changestar_models function exists."""
        from geoai.change_detection import list_changestar_models

        self.assertTrue(callable(list_changestar_models))

    def test_download_checkpoint_exists(self):
        """Test that download_checkpoint function exists."""
        from geoai.change_detection import download_checkpoint

        self.assertTrue(callable(download_checkpoint))


class TestChangeDetectionInit(unittest.TestCase):
    """Tests for ChangeDetection class initialization."""

    @patch("geoai.change_detection.AnyChange", None)
    def test_init_raises_without_torchange(self):
        """Test that init raises ImportError when torchange is missing."""
        from geoai.change_detection import ChangeDetection

        with self.assertRaises(ImportError):
            ChangeDetection()

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_init_with_mocked_torchange(self, mock_anychange, mock_download):
        """Test that init succeeds with mocked torchange."""
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/checkpoint.pth"

        from geoai.change_detection import ChangeDetection

        detector = ChangeDetection()
        self.assertIsNotNone(detector.model)
        mock_anychange.assert_called_once()

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_set_hyperparameters(self, mock_anychange, mock_download):
        """Test that set_hyperparameters delegates to the model."""
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/checkpoint.pth"

        from geoai.change_detection import ChangeDetection

        detector = ChangeDetection()
        detector.set_hyperparameters(change_confidence_threshold=200)
        # Should have been called during init AND the explicit call
        self.assertTrue(mock_model.set_hyperparameters.called)


class TestChangeDetectionSignatures(unittest.TestCase):
    """Tests for change detection function signatures."""

    def test_change_detection_init_params(self):
        """Test ChangeDetection.__init__ has expected parameters."""
        from geoai.change_detection import ChangeDetection

        sig = inspect.signature(ChangeDetection.__init__)
        self.assertIn("sam_model_type", sig.parameters)
        self.assertIn("sam_checkpoint", sig.parameters)

    def test_set_hyperparameters_params(self):
        """Test set_hyperparameters has expected parameters."""
        from geoai.change_detection import ChangeDetection

        sig = inspect.signature(ChangeDetection.set_hyperparameters)
        self.assertIn("change_confidence_threshold", sig.parameters)
        self.assertIn("use_normalized_feature", sig.parameters)
        self.assertIn("bitemporal_match", sig.parameters)


if __name__ == "__main__":
    unittest.main()
