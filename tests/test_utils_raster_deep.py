#!/usr/bin/env python

"""Deep tests for `geoai.utils.raster` module — tests actual computation, not just signatures."""

import os
import shutil
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from geoai.utils.raster import (
    clip_raster_by_bbox,
    get_raster_info,
    get_raster_resolution,
    get_raster_stats,
    raster_to_vector,
    read_raster,
    vector_to_raster,
)


def _create_test_raster(path, width=50, height=50, bands=3, dtype=np.uint8, nodata=0):
    """Create a test raster with known pixel values for deterministic assertions."""
    data = np.zeros((bands, height, width), dtype=dtype)
    for b in range(bands):
        data[b] = np.arange(width, dtype=dtype) + b * 10
    transform = from_bounds(-122.5, 37.7, -122.3, 37.9, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return data


class TestGetRasterInfo(unittest.TestCase):
    """Tests that get_raster_info returns accurate metadata."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.raster_path = os.path.join(self._tmpdir, "info_test.tif")
        self.data = _create_test_raster(self.raster_path, width=60, height=40, bands=4)

    def test_dimensions(self):
        info = get_raster_info(self.raster_path)
        self.assertEqual(info["width"], 60)
        self.assertEqual(info["height"], 40)

    def test_band_count(self):
        info = get_raster_info(self.raster_path)
        self.assertEqual(info["count"], 4)

    def test_crs(self):
        info = get_raster_info(self.raster_path)
        self.assertIn("4326", info["crs"])

    def test_bounds(self):
        info = get_raster_info(self.raster_path)
        self.assertAlmostEqual(info["bounds"].left, -122.5, places=1)
        self.assertAlmostEqual(info["bounds"].bottom, 37.7, places=1)

    def test_band_stats_values(self):
        info = get_raster_info(self.raster_path)
        stats = info["band_stats"]
        self.assertEqual(len(stats), 4)
        # Band 1: values 0..59 (width=60), nodata=0 masks zeros → min=1, max=59
        self.assertEqual(stats[0]["min"], 1)
        self.assertEqual(stats[0]["max"], 59)

    def test_nodata(self):
        info = get_raster_info(self.raster_path)
        self.assertEqual(info["nodata"], 0)

    def test_driver(self):
        info = get_raster_info(self.raster_path)
        self.assertEqual(info["driver"], "GTiff")


class TestGetRasterStats(unittest.TestCase):
    """Tests for get_raster_stats with known data."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.raster_path = os.path.join(self._tmpdir, "stats_test.tif")
        _create_test_raster(self.raster_path, width=50, height=50, bands=2)

    def test_returns_all_stat_keys(self):
        stats = get_raster_stats(self.raster_path)
        for key in ("min", "max", "mean", "std"):
            self.assertIn(key, stats)
            self.assertEqual(len(stats[key]), 2)

    def test_divide_by(self):
        stats_raw = get_raster_stats(self.raster_path, divide_by=1.0)
        stats_div = get_raster_stats(self.raster_path, divide_by=2.0)
        self.assertAlmostEqual(
            stats_div["mean"][0], stats_raw["mean"][0] / 2.0, places=2
        )

    def test_min_max_range(self):
        stats = get_raster_stats(self.raster_path)
        for i in range(2):
            self.assertLessEqual(stats["min"][i], stats["mean"][i])
            self.assertLessEqual(stats["mean"][i], stats["max"][i])


class TestGetRasterResolution(unittest.TestCase):
    """Tests for get_raster_resolution."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.raster_path = os.path.join(self._tmpdir, "res_test.tif")
        _create_test_raster(self.raster_path, width=100, height=50, bands=1)

    def test_resolution_values(self):
        res = get_raster_resolution(self.raster_path)
        self.assertEqual(len(res), 2)
        # Width span: 0.2 degrees over 100 pixels = 0.002
        self.assertAlmostEqual(res[0], 0.002, places=4)
        # Height span: 0.2 degrees over 50 pixels = 0.004
        self.assertAlmostEqual(res[1], 0.004, places=4)


class TestClipRasterByBbox(unittest.TestCase):
    """Tests for clip_raster_by_bbox with actual raster data."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.input_path = os.path.join(self._tmpdir, "clip_input.tif")
        self.output_path = os.path.join(self._tmpdir, "clip_output.tif")
        _create_test_raster(self.input_path, width=100, height=100, bands=3)

    def test_clip_reduces_dimensions(self):
        # Clip to the left half of the raster
        bbox = [-122.5, 37.7, -122.4, 37.9]
        clip_raster_by_bbox(self.input_path, self.output_path, bbox)
        self.assertTrue(os.path.exists(self.output_path))
        with rasterio.open(self.output_path) as src:
            self.assertLess(src.width, 100)
            self.assertEqual(src.count, 3)

    def test_clip_preserves_crs(self):
        bbox = [-122.5, 37.7, -122.4, 37.9]
        clip_raster_by_bbox(self.input_path, self.output_path, bbox)
        with rasterio.open(self.output_path) as src:
            self.assertEqual(src.crs.to_epsg(), 4326)

    def test_clip_band_selection(self):
        bbox = [-122.5, 37.7, -122.4, 37.9]
        clip_raster_by_bbox(self.input_path, self.output_path, bbox, bands=[1, 2])
        with rasterio.open(self.output_path) as src:
            self.assertEqual(src.count, 2)


class TestReadRaster(unittest.TestCase):
    """Tests for read_raster function."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.raster_path = os.path.join(self._tmpdir, "read_test.tif")
        self.data = _create_test_raster(self.raster_path, width=30, height=20, bands=3)

    def test_returns_xarray(self):
        import xarray as xr

        result = read_raster(self.raster_path)
        self.assertIsInstance(result, xr.DataArray)

    def test_single_band_selection(self):
        result = read_raster(self.raster_path, band=1)
        self.assertIsNotNone(result)

    def test_multi_band_selection(self):
        result = read_raster(self.raster_path, band=[1, 3])
        self.assertIsNotNone(result)


class TestRasterVectorConversion(unittest.TestCase):
    """Tests for raster_to_vector and vector_to_raster round-trip."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir)
        self.raster_path = os.path.join(self._tmpdir, "convert_input.tif")
        # Create a simple binary raster with a clear shape
        data = np.zeros((1, 50, 50), dtype=np.uint8)
        data[0, 10:40, 10:40] = 1  # A square block of 1s
        transform = from_bounds(-122.5, 37.7, -122.3, 37.9, 50, 50)
        with rasterio.open(
            self.raster_path,
            "w",
            driver="GTiff",
            height=50,
            width=50,
            count=1,
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

    def test_raster_to_vector_produces_geodataframe(self):
        output_path = os.path.join(self._tmpdir, "output.geojson")
        result = raster_to_vector(self.raster_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        gdf = gpd.read_file(output_path)
        self.assertGreater(len(gdf), 0)

    def test_vector_to_raster_produces_geotiff(self):
        vector_path = os.path.join(self._tmpdir, "polys.geojson")
        # Create a simple vector
        gdf = gpd.GeoDataFrame(
            {"class": [1]},
            geometry=[box(-122.45, 37.75, -122.35, 37.85)],
            crs="EPSG:4326",
        )
        gdf.to_file(vector_path)
        output_raster = os.path.join(self._tmpdir, "vec_to_raster.tif")
        vector_to_raster(
            vector_path,
            output_raster,
            reference_raster=self.raster_path,
        )
        self.assertTrue(os.path.exists(output_raster))
        with rasterio.open(output_raster) as src:
            self.assertEqual(src.count, 1)
            self.assertEqual(src.width, 50)


if __name__ == "__main__":
    unittest.main()
