#!/usr/bin/env python

"""Tests for `geoai.water` module."""

import inspect
import unittest
from unittest.mock import MagicMock, mock_open, patch

from geoai.water import BAND_ORDER_PRESETS


class TestBandOrderPresets(unittest.TestCase):
    """Tests for the BAND_ORDER_PRESETS constant."""

    def test_contains_expected_sensors(self):
        """Test that presets contain naip, sentinel2, and landsat keys."""
        for key in ["naip", "sentinel2", "landsat"]:
            self.assertIn(key, BAND_ORDER_PRESETS)

    def test_values_are_int_lists(self):
        """Test that each preset value is a list of integers."""
        for sensor, bands in BAND_ORDER_PRESETS.items():
            self.assertIsInstance(bands, list, f"{sensor} should be a list")
            for band in bands:
                self.assertIsInstance(band, int, f"Band in {sensor} should be int")

    def test_naip_has_4_bands(self):
        """Test that NAIP preset has 4 band indices."""
        self.assertEqual(len(BAND_ORDER_PRESETS["naip"]), 4)

    def test_sentinel2_has_4_bands(self):
        """Test that Sentinel-2 preset has 4 band indices."""
        self.assertEqual(len(BAND_ORDER_PRESETS["sentinel2"]), 4)

    def test_landsat_has_4_bands(self):
        """Test that Landsat preset has 4 band indices."""
        self.assertEqual(len(BAND_ORDER_PRESETS["landsat"]), 4)

    def test_band_indices_are_positive(self):
        """Test that all band indices are positive (1-based)."""
        for sensor, bands in BAND_ORDER_PRESETS.items():
            for band in bands:
                self.assertGreater(band, 0, f"Band index in {sensor} must be positive")


class TestSegmentWaterSignature(unittest.TestCase):
    """Tests for the segment_water function signature."""

    def test_function_exists(self):
        """Test that segment_water is importable."""
        from geoai.water import segment_water

        self.assertTrue(callable(segment_water))

    def test_function_signature(self):
        """Test that segment_water has expected parameters."""
        from geoai.water import segment_water

        sig = inspect.signature(segment_water)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("band_order", sig.parameters)


class TestExtractWaterBand(unittest.TestCase):
    """Tests for the _extract_water_band function."""

    def test_function_exists(self):
        """Test that _extract_water_band function is importable."""
        from geoai.water import _extract_water_band

        self.assertTrue(callable(_extract_water_band))

    def test_function_signature(self):
        """Test that _extract_water_band has src_path and dst_path params."""
        from geoai.water import _extract_water_band

        sig = inspect.signature(_extract_water_band)
        self.assertIn("src_path", sig.parameters)
        self.assertIn("dst_path", sig.parameters)


if __name__ == "__main__":
    unittest.main()
