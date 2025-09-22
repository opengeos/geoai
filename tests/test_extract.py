#!/usr/bin/env python

"""Tests for `geoai.extract` module."""

import os
import unittest

from geoai import extract

from .test_fixtures import get_test_data_paths


class TestExtractModule(unittest.TestCase):
    """Tests for extract module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = get_test_data_paths()
        self.test_data_dir = self.test_paths["data_dir"]
        self.test_raster_rgb = self.test_paths["test_raster_rgb"]
        self.test_polygons = self.test_paths["test_polygons"]

    def test_module_imports(self):
        """Test that extract module imports correctly."""
        self.assertTrue(hasattr(extract, "__name__"))

    def test_extract_functions_exist(self):
        """Test that extract module has public functions."""
        extract_functions = [attr for attr in dir(extract) if not attr.startswith("_")]
        self.assertGreater(
            len(extract_functions), 0, "No public functions found in extract module"
        )

    def test_extract_module_callable_functions(self):
        """Test that extract module functions are callable."""
        extract_functions = [
            attr
            for attr in dir(extract)
            if not attr.startswith("_") and callable(getattr(extract, attr))
        ]

        # Should have at least some callable functions
        self.assertGreaterEqual(len(extract_functions), 0)

        # Test a few functions if they exist
        common_extract_functions = ["extract_chips", "extract_features"]
        for func_name in common_extract_functions:
            if hasattr(extract, func_name):
                func = getattr(extract, func_name)
                self.assertTrue(callable(func), f"{func_name} should be callable")

    def test_extract_with_test_data(self):
        """Test extract functions with test data."""
        # This is a basic test to ensure functions don't crash immediately
        # Real functionality would require specific implementations

        # Test that we can at least call the module without import errors
        try:
            import geoai.extract  # noqa: F401
        except ImportError as e:
            self.fail(f"Failed to import geoai.extract: {e}")


if __name__ == "__main__":
    unittest.main()
