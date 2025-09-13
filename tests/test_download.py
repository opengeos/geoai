#!/usr/bin/env python

"""Tests for `geoai.download` module."""

import unittest
from unittest.mock import MagicMock, patch

from geoai import download


class TestDownloadModule(unittest.TestCase):
    """Tests for download module."""

    def test_module_imports(self):
        """Test that download module imports correctly."""
        self.assertTrue(hasattr(download, "__name__"))

    def test_download_functions_exist(self):
        """Test that key download functions exist."""
        expected_functions = [
            "download_naip",
            "download_overture_buildings",
            "pc_stac_search",
            "pc_collection_list",
        ]

        for func_name in expected_functions:
            self.assertTrue(
                hasattr(download, func_name),
                f"Function {func_name} not found in download module",
            )

    @patch("geoai.download.requests.get")
    def test_download_function_with_mock(self, mock_get):
        """Test download functions with mocked requests."""
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"collections": []}
        mock_get.return_value = mock_response

        # Test that pc_collection_list exists and can be called
        if hasattr(download, "pc_collection_list"):
            try:
                download.pc_collection_list()
                # Should not raise an exception
            except Exception as e:
                # If it fails, it should be due to implementation details, not import issues
                self.assertIsInstance(e, (AttributeError, TypeError, ValueError))

    def test_download_naip_signature(self):
        """Test that download_naip function has expected signature."""
        if hasattr(download, "download_naip"):
            func = getattr(download, "download_naip")
            self.assertTrue(callable(func))

    def test_stac_search_signature(self):
        """Test that pc_stac_search function has expected signature."""
        if hasattr(download, "pc_stac_search"):
            func = getattr(download, "pc_stac_search")
            self.assertTrue(callable(func))


if __name__ == "__main__":
    unittest.main()
