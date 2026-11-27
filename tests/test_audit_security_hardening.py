#!/usr/bin/env python

"""Regression tests for download, archive extraction, and path validation hardening."""

import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from geoai.download import (
    download_with_progress,
    get_overture_latest_release,
    pc_stac_download,
)
from geoai.ftw import download_ftw


class TestFtwZipValidation(unittest.TestCase):
    """Tests for archive member validation and temp cleanup in download_ftw."""

    def _mock_download_zip(self, members):
        def fake_download(url, output_path=None, overwrite=False, unzip=False):
            with zipfile.ZipFile(output_path, "w") as zf:
                for name, content in members:
                    zf.writestr(name, content)
            return output_path

        return fake_download

    def test_rejects_parent_traversal_member_and_cleans_tmp_dir(self):
        """Zip member with parent traversal raises ValueError and removes temp dir."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "geoai.utils.download.download_file",
                side_effect=self._mock_download_zip([("../outside.txt", "data")]),
            ):
                with self.assertRaises(ValueError):
                    download_ftw(countries=["luxembourg"], output_dir=tmp)

            self.assertFalse(os.path.exists(os.path.join(tmp, "_tmp_luxembourg")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "luxembourg")))
            self.assertFalse(
                os.path.exists(os.path.join(os.path.dirname(tmp), "outside.txt"))
            )

    def test_rejects_absolute_member_and_cleans_tmp_dir(self):
        """Zip member with an absolute path raises ValueError and removes temp dir."""
        with tempfile.TemporaryDirectory() as tmp:
            abs_target = os.path.join(tmp, "abs_outside.txt")
            with patch(
                "geoai.utils.download.download_file",
                side_effect=self._mock_download_zip([(abs_target, "data")]),
            ):
                with self.assertRaises(ValueError):
                    download_ftw(countries=["luxembourg"], output_dir=tmp)

            self.assertFalse(os.path.exists(os.path.join(tmp, "_tmp_luxembourg")))
            self.assertFalse(os.path.exists(abs_target))

    def test_cleans_tmp_dir_on_extraction_error(self):
        """Unexpected errors during extraction remove the temporary directory."""
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "geoai.utils.download.download_file",
                    side_effect=self._mock_download_zip([("luxembourg/a.txt", "ok")]),
                ),
                patch.object(
                    zipfile.ZipFile,
                    "extractall",
                    side_effect=RuntimeError("disk full"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    download_ftw(countries=["luxembourg"], output_dir=tmp)

            self.assertFalse(os.path.exists(os.path.join(tmp, "_tmp_luxembourg")))

    def test_valid_archive_extracts_cleanly(self):
        """Valid FTW archive extracts to country directory without leaving temp dir."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "geoai.utils.download.download_file",
                side_effect=self._mock_download_zip([("luxembourg/chip.txt", "ok")]),
            ):
                download_ftw(countries=["luxembourg"], output_dir=tmp)

            self.assertFalse(os.path.exists(os.path.join(tmp, "_tmp_luxembourg")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "luxembourg", "chip.txt")))


class TestDownloadHardening(unittest.TestCase):
    """Tests for streaming size limits, request timeouts, and STAC filename checks."""

    def test_download_with_progress_enforces_max_size_without_content_length(self):
        """Streaming bytes exceeding max_size raise ValueError when Content-Length is absent."""
        response = MagicMock()
        response.headers = {}
        response.iter_content.return_value = [b"a" * 6, b"b" * 6]

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "file.bin")
            with patch(
                "geoai.download.requests.get", return_value=response
            ) as mock_get:
                with self.assertRaises(ValueError):
                    download_with_progress(
                        "https://example.com/file.bin", out, max_size=10
                    )

            self.assertEqual(os.listdir(tmp), [])
            response.close.assert_called_once()
            self.assertIn("timeout", mock_get.call_args.kwargs)

    def test_download_with_progress_enforces_max_size_with_smaller_content_length(self):
        """Streaming bytes exceeding max_size raise ValueError even if Content-Length is smaller."""
        response = MagicMock()
        response.headers = {"content-length": "5"}
        response.iter_content.return_value = [b"12345", b"678901"]

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "file.bin")
            with patch("geoai.download.requests.get", return_value=response):
                with self.assertRaises(ValueError):
                    download_with_progress(
                        "https://example.com/file.bin", out, max_size=10
                    )

            self.assertEqual(os.listdir(tmp), [])
            response.close.assert_called_once()

    def test_download_with_progress_passes_timeout(self):
        """Explicit timeout argument is forwarded to requests.get."""
        response = MagicMock()
        response.headers = {"content-length": "4"}
        response.iter_content.return_value = [b"abcd"]

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "file.bin")
            with patch(
                "geoai.download.requests.get", return_value=response
            ) as mock_get:
                download_with_progress(
                    "https://example.com/file.bin", out, timeout=(10, 45)
                )

            mock_get.assert_called_once_with(
                "https://example.com/file.bin", stream=True, timeout=(10, 45)
            )
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"abcd")

    def test_get_overture_latest_release_passes_timeout(self):
        """get_overture_latest_release passes an explicit timeout to requests.get."""
        response = MagicMock()
        response.json.return_value = {"latest": "2025-02-19.0"}

        with patch("geoai.download.requests.get", return_value=response) as mock_get:
            release = get_overture_latest_release()

        self.assertEqual(release, "2025-02-19.0")
        self.assertEqual(mock_get.call_args.kwargs.get("timeout"), 60)

    def _make_stac_item(
        self, item_id, asset_key="image", href="https://example.com/a.tif"
    ):
        import datetime

        import pystac

        item = pystac.Item(
            id=item_id,
            geometry=None,
            bbox=None,
            datetime=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
            properties={},
        )
        item.add_asset(
            asset_key,
            pystac.Asset(href=href, media_type="image/tiff"),
        )
        return item

    def test_pc_stac_download_rejects_item_id_traversal(self):
        """pc_stac_download rejects STAC item IDs containing path traversal."""
        item = self._make_stac_item("../escaped")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("geoai.download.pc.sign", side_effect=lambda x: x),
                patch("geoai.download.requests.get") as mock_get,
            ):
                with self.assertRaises(ValueError):
                    pc_stac_download(item, output_dir=tmp)

            mock_get.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])

    def test_pc_stac_download_rejects_asset_key_traversal(self):
        """pc_stac_download rejects STAC asset keys containing path separators."""
        item = self._make_stac_item("valid_item", asset_key="../escaped")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("geoai.download.pc.sign", side_effect=lambda x: x),
                patch("geoai.download.requests.get") as mock_get,
            ):
                with self.assertRaises(ValueError):
                    pc_stac_download(item, output_dir=tmp)

            mock_get.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])

    def test_pc_stac_download_valid_asset_uses_timeout(self):
        """pc_stac_download downloads valid assets with an explicit request timeout."""
        item = self._make_stac_item("scene_01", asset_key="B04")
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"content-length": "4"}
        response.iter_content.return_value = [b"data"]

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("geoai.download.pc.sign", side_effect=lambda x: x),
                patch("geoai.download.requests.get", return_value=response) as mock_get,
            ):
                result = pc_stac_download(item, output_dir=tmp)

            self.assertIn("scene_01", result)
            expected_path = os.path.join(tmp, "scene_01_B04.tif")
            self.assertEqual(result["scene_01"]["B04"], expected_path)
            self.assertTrue(os.path.isfile(expected_path))
            self.assertEqual(mock_get.call_args.kwargs.get("timeout"), (30, 300))


if __name__ == "__main__":
    unittest.main()
