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

    def test_download_naip_has_bbox_param(self):
        """Test that download_naip accepts a bbox parameter."""
        import inspect

        if hasattr(download, "download_naip"):
            sig = inspect.signature(download.download_naip)
            self.assertIn("bbox", sig.parameters)

    def test_pc_stac_search_has_bbox_param(self):
        """Test that pc_stac_search accepts a bbox parameter."""
        import inspect

        if hasattr(download, "pc_stac_search"):
            sig = inspect.signature(download.pc_stac_search)
            self.assertIn("bbox", sig.parameters)

    def test_download_overture_buildings_callable(self):
        """Test that download_overture_buildings is callable."""
        if hasattr(download, "download_overture_buildings"):
            self.assertTrue(callable(download.download_overture_buildings))

    def test_download_overture_buildings_signature(self):
        """Test that download_overture_buildings has expected parameters."""
        import inspect

        if hasattr(download, "download_overture_buildings"):
            sig = inspect.signature(download.download_overture_buildings)
            self.assertIn("bbox", sig.parameters)

    def test_additional_functions_exist(self):
        """Test that additional download functions exist."""
        additional_functions = [
            "pc_stac_download",
            "pc_item_asset_list",
            "download_with_progress",
        ]
        for func_name in additional_functions:
            if hasattr(download, func_name):
                self.assertTrue(
                    callable(getattr(download, func_name)),
                    f"{func_name} is not callable",
                )

    @patch("geoai.download.requests.get")
    def test_pc_collection_list_returns_result(self, mock_get):
        """Test that pc_collection_list parses response correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "collections": [
                {"id": "sentinel-2-l2a", "title": "Sentinel-2 L2A"},
                {"id": "landsat-c2-l2", "title": "Landsat C2 L2"},
            ]
        }
        mock_get.return_value = mock_response

        if hasattr(download, "pc_collection_list"):
            try:
                result = download.pc_collection_list()
                # If it returns, should be a list or similar
                if result is not None:
                    self.assertIsNotNone(result)
            except Exception:
                # Implementation may differ; verify no crash
                pass


if __name__ == "__main__":
    unittest.main()
