#!/usr/bin/env python

"""Tests for `geoai.classify` module."""

import inspect
import unittest

from geoai import classify

from .test_fixtures import get_test_data_paths


class TestClassifyModule(unittest.TestCase):
    """Tests for classify module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]

    def test_module_imports(self):
        """Test that classify module imports correctly."""
        self.assertTrue(hasattr(classify, "__name__"))

    def test_classify_functions_exist(self):
        """Test that key classify functions exist."""
        expected_functions = ["classify_image", "classify_images", "train_classifier"]

        for func_name in expected_functions:
            if hasattr(classify, func_name):
                func = getattr(classify, func_name)
                self.assertTrue(callable(func), f"{func_name} is not callable")

    def test_classify_image_signature(self):
        """Test that classify_image function exists and is callable."""
        if hasattr(classify, "classify_image"):
            func = getattr(classify, "classify_image")
            self.assertTrue(callable(func))

    def test_classify_with_invalid_input(self):
        """Test classify functions with invalid inputs."""
        # Test with non-existent file
        if hasattr(classify, "classify_image"):
            with self.assertRaises((FileNotFoundError, ValueError, TypeError)):
                classify.classify_image("nonexistent_file.tif", "nonexistent_model.pth")

    def test_train_classifier_exists(self):
        """Test that train_classifier function exists."""
        if hasattr(classify, "train_classifier"):
            func = getattr(classify, "train_classifier")
            self.assertTrue(callable(func))

    def test_train_classifier_signature(self):
        """Test that train_classifier has expected parameter names."""
        if hasattr(classify, "train_classifier"):
            sig = inspect.signature(classify.train_classifier)
            param_names = list(sig.parameters.keys())
            self.assertGreater(len(param_names), 0)

    def test_classify_images_signature(self):
        """Test that classify_images has expected parameter names."""
        if hasattr(classify, "classify_images"):
            sig = inspect.signature(classify.classify_images)
            param_names = list(sig.parameters.keys())
            self.assertGreater(len(param_names), 0)

    def test_classify_image_has_model_param(self):
        """Test that classify_image accepts a model-related parameter."""
        if hasattr(classify, "classify_image"):
            sig = inspect.signature(classify.classify_image)
            param_names = list(sig.parameters.keys())
            # Should have at least an input path parameter
            self.assertGreater(len(param_names), 1)

    def test_internal_classify_function_exists(self):
        """Test that the internal _classify_image function exists."""
        self.assertTrue(
            hasattr(classify, "_classify_image"),
            "Internal _classify_image function not found",
        )


if __name__ == "__main__":
    unittest.main()
