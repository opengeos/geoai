#!/usr/bin/env python

"""Tests for `geoai.cli` module."""

import unittest

from click.testing import CliRunner

from geoai.cli import main

from .test_fixtures import get_test_data_paths


class TestCLI(unittest.TestCase):
    """Tests for CLI commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.test_paths = get_test_data_paths()

    def test_main_help(self):
        """Test that main --help runs successfully."""
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("GeoAI", result.output)

    def test_main_version(self):
        """Test that main --version runs successfully."""
        result = self.runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)

    def test_info_help(self):
        """Test that info --help runs successfully."""
        result = self.runner.invoke(main, ["info", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("FILEPATH", result.output)

    def test_info_raster(self):
        """Test info command with a raster file."""
        result = self.runner.invoke(main, ["info", self.test_paths["test_raster_rgb"]])
        self.assertEqual(result.exit_code, 0)

    def test_info_vector(self):
        """Test info command with a vector file."""
        result = self.runner.invoke(main, ["info", self.test_paths["test_polygons"]])
        self.assertEqual(result.exit_code, 0)

    def test_info_nonexistent_file(self):
        """Test info command with a nonexistent file."""
        result = self.runner.invoke(main, ["info", "/nonexistent/file.tif"])
        self.assertNotEqual(result.exit_code, 0)

    def test_download_help(self):
        """Test that download --help runs successfully."""
        result = self.runner.invoke(main, ["download", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--bbox", result.output)

    def test_download_invalid_bbox(self):
        """Test download command with invalid bbox."""
        result = self.runner.invoke(
            main, ["download", "naip", "--bbox", "invalid", "--output", "/tmp/out.tif"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)


if __name__ == "__main__":
    unittest.main()
