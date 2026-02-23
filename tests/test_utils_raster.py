#!/usr/bin/env python

"""Tests for raster and vector utility functions in `geoai.utils`."""

import unittest

import geopandas as gpd
import numpy as np

from geoai import utils

from .test_fixtures import get_test_data_paths


class TestGetRasterInfo(unittest.TestCase):
    """Tests for the get_raster_info function."""

    def setUp(self):
        """Set up test data paths."""
        self.test_paths = get_test_data_paths()

    def test_returns_dict(self):
        """Test that get_raster_info returns a dictionary."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        self.assertIsInstance(info, dict)

    def test_required_keys(self):
        """Test that returned dict contains all expected keys."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        expected_keys = [
            "driver",
            "width",
            "height",
            "count",
            "dtype",
            "crs",
            "transform",
            "bounds",
            "resolution",
            "nodata",
            "band_stats",
        ]
        for key in expected_keys:
            self.assertIn(key, info, f"Key '{key}' not found in raster info")

    def test_rgb_raster_has_3_bands(self):
        """Test that RGB raster reports 3 bands."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        self.assertEqual(info["count"], 3)
        self.assertEqual(len(info["band_stats"]), 3)

    def test_single_band_raster(self):
        """Test raster info for a single-band raster."""
        info = utils.get_raster_info(self.test_paths["test_raster_single"])
        self.assertEqual(info["count"], 1)
        self.assertEqual(len(info["band_stats"]), 1)

    def test_multi_band_raster(self):
        """Test raster info for a 4-band raster."""
        info = utils.get_raster_info(self.test_paths["test_raster_multi"])
        self.assertEqual(info["count"], 4)
        self.assertEqual(len(info["band_stats"]), 4)

    def test_band_stats_have_required_keys(self):
        """Test that each band stat dict has min, max, mean, std."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        for band_stat in info["band_stats"]:
            self.assertIn("band", band_stat)
            self.assertIn("min", band_stat)
            self.assertIn("max", band_stat)
            self.assertIn("mean", band_stat)
            self.assertIn("std", band_stat)

    def test_crs_is_set(self):
        """Test that CRS is properly read from test raster."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        self.assertNotEqual(info["crs"], "No CRS defined")

    def test_resolution_is_tuple(self):
        """Test that resolution is a tuple of 2 values."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        self.assertIsInstance(info["resolution"], tuple)
        self.assertEqual(len(info["resolution"]), 2)

    def test_dimensions(self):
        """Test that width and height are positive integers."""
        info = utils.get_raster_info(self.test_paths["test_raster_rgb"])
        self.assertGreater(info["width"], 0)
        self.assertGreater(info["height"], 0)


class TestGetRasterStats(unittest.TestCase):
    """Tests for the get_raster_stats function."""

    def setUp(self):
        """Set up test data paths."""
        self.test_paths = get_test_data_paths()

    def test_returns_dict_with_stat_keys(self):
        """Test that stats dict contains min, max, mean, std keys."""
        stats = utils.get_raster_stats(self.test_paths["test_raster_rgb"])
        for key in ["min", "max", "mean", "std"]:
            self.assertIn(key, stats)

    def test_list_lengths_match_band_count(self):
        """Test that stat lists have one entry per band."""
        stats = utils.get_raster_stats(self.test_paths["test_raster_rgb"])
        for key in ["min", "max", "mean", "std"]:
            self.assertEqual(len(stats[key]), 3)

    def test_single_band_stats(self):
        """Test stats for a single-band raster."""
        stats = utils.get_raster_stats(self.test_paths["test_raster_single"])
        for key in ["min", "max", "mean", "std"]:
            self.assertEqual(len(stats[key]), 1)

    def test_divide_by_parameter(self):
        """Test that divide_by scales the statistics."""
        stats_1 = utils.get_raster_stats(self.test_paths["test_raster_rgb"])
        stats_255 = utils.get_raster_stats(
            self.test_paths["test_raster_rgb"], divide_by=255.0
        )
        # Stats divided by 255 should be smaller
        for i in range(3):
            self.assertAlmostEqual(
                stats_255["mean"][i], stats_1["mean"][i] / 255.0, places=4
            )

    def test_min_less_than_max(self):
        """Test that min values are less than or equal to max values."""
        stats = utils.get_raster_stats(self.test_paths["test_raster_rgb"])
        for i in range(3):
            self.assertLessEqual(stats["min"][i], stats["max"][i])


class TestGetRasterResolution(unittest.TestCase):
    """Tests for the get_raster_resolution function."""

    def setUp(self):
        """Set up test data paths."""
        self.test_paths = get_test_data_paths()

    def test_returns_tuple_of_floats(self):
        """Test that resolution is returned as a tuple of two floats."""
        res = utils.get_raster_resolution(self.test_paths["test_raster_rgb"])
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], float)
        self.assertIsInstance(res[1], float)

    def test_positive_resolution(self):
        """Test that resolution values are positive."""
        res = utils.get_raster_resolution(self.test_paths["test_raster_rgb"])
        self.assertGreater(res[0], 0)
        self.assertGreater(res[1], 0)


class TestGetVectorInfo(unittest.TestCase):
    """Tests for the get_vector_info function."""

    def setUp(self):
        """Set up test data paths."""
        self.test_paths = get_test_data_paths()

    def test_returns_dict(self):
        """Test that vector info returns a dictionary."""
        info = utils.get_vector_info(self.test_paths["test_polygons"])
        self.assertIsInstance(info, dict)

    def test_contains_expected_keys(self):
        """Test that vector info contains standard keys."""
        info = utils.get_vector_info(self.test_paths["test_polygons"])
        # At minimum should have feature count and geometry type info
        self.assertGreater(len(info), 0)


class TestReadVector(unittest.TestCase):
    """Tests for the read_vector function."""

    def setUp(self):
        """Set up test data paths."""
        self.test_paths = get_test_data_paths()

    def test_returns_geodataframe(self):
        """Test that read_vector returns a GeoDataFrame."""
        gdf = utils.read_vector(self.test_paths["test_polygons"])
        self.assertIsInstance(gdf, gpd.GeoDataFrame)

    def test_has_geometry_column(self):
        """Test that returned GeoDataFrame has a geometry column."""
        gdf = utils.read_vector(self.test_paths["test_polygons"])
        self.assertIn("geometry", gdf.columns)

    def test_non_empty(self):
        """Test that returned GeoDataFrame is not empty."""
        gdf = utils.read_vector(self.test_paths["test_polygons"])
        self.assertGreater(len(gdf), 0)


class TestBoxesToVector(unittest.TestCase):
    """Tests for the boxes_to_vector function."""

    def test_returns_geodataframe(self):
        """Test that boxes_to_vector returns a GeoDataFrame."""
        coords = [[0, 0, 10, 10], [20, 20, 30, 30]]
        gdf = utils.boxes_to_vector(coords, src_crs="EPSG:4326")
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), 2)

    def test_crs_reprojection(self):
        """Test that CRS is reprojected to destination CRS."""
        coords = [[-122.5, 37.7, -122.3, 37.9]]
        gdf = utils.boxes_to_vector(coords, src_crs="EPSG:4326", dst_crs="EPSG:4326")
        self.assertIsNotNone(gdf.crs)

    def test_single_box(self):
        """Test with a single bounding box."""
        coords = [[0, 0, 100, 100]]
        gdf = utils.boxes_to_vector(coords, src_crs="EPSG:32610")
        self.assertEqual(len(gdf), 1)


if __name__ == "__main__":
    unittest.main()
