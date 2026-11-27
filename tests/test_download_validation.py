#!/usr/bin/env python

"""Tests for URL validation and download safety in geoai.download."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from geoai.download import _validate_url


class TestValidateUrl(unittest.TestCase):
    """Tests for the _validate_url helper."""

    def test_https_url_passes(self):
        _validate_url("https://example.com/data.tif")

    def test_http_url_passes(self):
        _validate_url("http://example.com/data.tif")

    def test_ftp_url_rejected(self):
        with self.assertRaises(ValueError):
            _validate_url("ftp://example.com/data.tif")

    def test_file_url_rejected(self):
        with self.assertRaises(ValueError):
            _validate_url("file:///etc/passwd")

    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError):
            _validate_url("")

    def test_no_host_rejected(self):
        with self.assertRaises(ValueError):
            _validate_url("https://")

    def test_data_uri_rejected(self):
        with self.assertRaises(ValueError):
            _validate_url("data:text/plain;base64,SGVsbG8=")

    def test_url_with_path_and_query(self):
        _validate_url("https://example.com/path/to/file.tif?token=abc&format=cog")

    def test_url_with_port(self):
        _validate_url("https://example.com:8080/data.tif")


class TestDownloadWithProgressAtomic(unittest.TestCase):
    """A failed download must not leave a truncated file at the final path."""

    def _failing_response(self):
        response = MagicMock()
        response.headers = {"content-length": "10"}

        def chunks(_size):
            yield b"abc"
            raise ConnectionError("connection dropped")

        response.iter_content.side_effect = chunks
        return response

    def test_failed_download_leaves_no_file(self):
        from geoai.download import download_with_progress

        response = self._failing_response()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "img.tif")
            with patch("geoai.download.requests.get", return_value=response):
                with self.assertRaises(ConnectionError):
                    download_with_progress("https://example.com/img.tif", out)
            self.assertEqual(os.listdir(tmp), [])
            response.close.assert_called_once()

    def test_successful_download_writes_final_file(self):
        from geoai.download import download_with_progress

        response = MagicMock()
        response.headers = {"content-length": "6"}
        response.iter_content.return_value = [b"abc", b"def"]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "img.tif")
            with patch("geoai.download.requests.get", return_value=response):
                download_with_progress("https://example.com/img.tif", out)
            self.assertEqual(os.listdir(tmp), ["img.tif"])
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"abcdef")
            response.close.assert_called_once()

    def test_truncated_download_without_exception_raises_and_cleans_up(self):
        from geoai.download import download_with_progress

        response = MagicMock()
        response.headers = {"content-length": "10"}
        response.iter_content.return_value = [b"abc"]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "img.tif")
            with patch("geoai.download.requests.get", return_value=response):
                with self.assertRaises(OSError):
                    download_with_progress("https://example.com/img.tif", out)
            self.assertEqual(os.listdir(tmp), [])
            response.close.assert_called_once()

    def test_existing_part_file_is_not_clobbered(self):
        from geoai.download import download_with_progress

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "img.tif")
            existing_part = out + ".part"
            with open(existing_part, "wb") as f:
                f.write(b"keep-me")
            with patch(
                "geoai.download.requests.get", return_value=self._failing_response()
            ):
                with self.assertRaises(ConnectionError):
                    download_with_progress("https://example.com/img.tif", out)
            self.assertEqual(os.listdir(tmp), ["img.tif.part"])
            with open(existing_part, "rb") as f:
                self.assertEqual(f.read(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
