#!/usr/bin/env python

"""Tests for URL validation and download safety in geoai.download."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
