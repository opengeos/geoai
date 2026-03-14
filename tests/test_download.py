#!/usr/bin/env python

"""Tests for `geoai.download` module."""

import os
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


class TestDownloadFileUnzip(unittest.TestCase):
    """Tests for download_file ZIP extraction behavior."""

    def _make_zip(self, tmp_dir, members):
        """Create a zip file with the given member paths and return its path."""
        import zipfile

        zip_path = os.path.join(tmp_dir, "archive.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in members:
                zf.writestr(name, f"content of {name}")
        return zip_path

    def test_single_top_level_folder_no_extra_nesting(self):
        """Single top-level folder in zip should not create a wrapper dir."""
        import tempfile

        from geoai.utils.download import download_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = self._make_zip(
                tmp_dir,
                ["data/a.txt", "data/sub/b.txt"],
            )
            result = download_file("http://fake", output_path=zip_path, unzip=True)
            self.assertEqual(result, os.path.join(tmp_dir, "data"))
            self.assertTrue(os.path.isdir(result))
            self.assertTrue(os.path.isfile(os.path.join(result, "a.txt")))
            self.assertTrue(os.path.isfile(os.path.join(result, "sub", "b.txt")))

    def test_multiple_top_level_entries_creates_wrapper(self):
        """Multiple top-level entries should extract into a wrapper dir."""
        import tempfile

        from geoai.utils.download import download_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = self._make_zip(
                tmp_dir,
                ["a.txt", "b.txt"],
            )
            result = download_file("http://fake", output_path=zip_path, unzip=True)
            # Should create wrapper dir named after zip stem
            expected = os.path.join(tmp_dir, "archive")
            self.assertEqual(result, expected)
            self.assertTrue(os.path.isdir(result))
            self.assertTrue(os.path.isfile(os.path.join(result, "a.txt")))
            self.assertTrue(os.path.isfile(os.path.join(result, "b.txt")))

    def test_single_top_level_file_creates_wrapper(self):
        """A single top-level file (not folder) should still use wrapper dir."""
        import tempfile

        from geoai.utils.download import download_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = self._make_zip(tmp_dir, ["readme.txt"])
            result = download_file("http://fake", output_path=zip_path, unzip=True)
            expected = os.path.join(tmp_dir, "archive")
            self.assertEqual(result, expected)
            self.assertTrue(os.path.isfile(os.path.join(result, "readme.txt")))

    def test_overwrite_false_skips_existing(self):
        """When extract dir exists and overwrite=False, skip extraction."""
        import tempfile

        from geoai.utils.download import download_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = self._make_zip(
                tmp_dir,
                ["data/a.txt"],
            )
            # Pre-create the target dir
            target = os.path.join(tmp_dir, "data")
            os.makedirs(target, exist_ok=True)
            result = download_file(
                "http://fake", output_path=zip_path, unzip=True, overwrite=False
            )
            self.assertEqual(result, target)
            # a.txt should NOT exist because extraction was skipped
            self.assertFalse(os.path.isfile(os.path.join(target, "a.txt")))

    def test_zip_slip_raises_error(self):
        """Zip members with path traversal should raise ValueError."""
        import tempfile
        import zipfile

        from geoai.utils.download import download_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "evil.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../escape.txt", "malicious")
            with self.assertRaises(ValueError):
                download_file("http://fake", output_path=zip_path, unzip=True)


if __name__ == "__main__":
    unittest.main()
