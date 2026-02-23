#!/usr/bin/env python

"""Tests for `geoai.auto` module (Auto classes for geospatial inference)."""

import inspect
import unittest

from geoai import auto


class TestAutoGeoImageProcessor(unittest.TestCase):
    """Tests for the AutoGeoImageProcessor class."""

    def test_class_exists(self):
        """Test that AutoGeoImageProcessor is available."""
        self.assertTrue(hasattr(auto, "AutoGeoImageProcessor"))

    def test_is_class(self):
        """Test that AutoGeoImageProcessor is a class."""
        self.assertTrue(inspect.isclass(auto.AutoGeoImageProcessor))


class TestAutoGeoModel(unittest.TestCase):
    """Tests for the AutoGeoModel class."""

    def test_class_exists(self):
        """Test that AutoGeoModel is available."""
        self.assertTrue(hasattr(auto, "AutoGeoModel"))

    def test_is_class(self):
        """Test that AutoGeoModel is a class."""
        self.assertTrue(inspect.isclass(auto.AutoGeoModel))


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module-level convenience functions."""

    def test_semantic_segmentation_exists(self):
        """Test that semantic_segmentation function exists and is callable."""
        self.assertTrue(hasattr(auto, "semantic_segmentation"))
        self.assertTrue(callable(auto.semantic_segmentation))

    def test_depth_estimation_exists(self):
        """Test that depth_estimation function exists and is callable."""
        self.assertTrue(hasattr(auto, "depth_estimation"))
        self.assertTrue(callable(auto.depth_estimation))

    def test_image_classification_exists(self):
        """Test that image_classification function exists and is callable."""
        self.assertTrue(hasattr(auto, "image_classification"))
        self.assertTrue(callable(auto.image_classification))

    def test_object_detection_exists(self):
        """Test that object_detection function exists and is callable."""
        self.assertTrue(hasattr(auto, "object_detection"))
        self.assertTrue(callable(auto.object_detection))

    def test_get_hf_tasks_exists(self):
        """Test that get_hf_tasks function exists and is callable."""
        self.assertTrue(hasattr(auto, "get_hf_tasks"))
        self.assertTrue(callable(auto.get_hf_tasks))

    def test_get_hf_model_config_exists(self):
        """Test that get_hf_model_config function exists."""
        self.assertTrue(hasattr(auto, "get_hf_model_config"))
        self.assertTrue(callable(auto.get_hf_model_config))


class TestModuleExports(unittest.TestCase):
    """Tests for module __all__ exports."""

    def test_all_contains_expected_names(self):
        """Test that __all__ contains key exported names."""
        expected = [
            "AutoGeoImageProcessor",
            "AutoGeoModel",
            "semantic_segmentation",
            "depth_estimation",
            "image_classification",
            "object_detection",
            "get_hf_tasks",
        ]
        for name in expected:
            self.assertIn(name, auto.__all__)

    def test_semantic_segmentation_signature(self):
        """Test semantic_segmentation has expected parameters."""
        sig = inspect.signature(auto.semantic_segmentation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)

    def test_depth_estimation_signature(self):
        """Test depth_estimation has expected parameters."""
        sig = inspect.signature(auto.depth_estimation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)


if __name__ == "__main__":
    unittest.main()
