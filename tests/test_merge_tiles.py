#!/usr/bin/env python

"""Tests for `_merge_tiles` in `geoai.embeddings` module."""

import unittest

import numpy as np
from rasterio.crs import CRS


from geoai.embeddings import _merge_tiles


class TestMergeTilesSingleTile(unittest.TestCase):
    """Single-tile fast path should return data unchanged."""

    def test_returns_original_data(self):
        data = np.random.randint(0, 100, (3, 32, 32), dtype=np.int8)
        bounds = (500000.0, 4000000.0, 500320.0, 4000320.0)
        crs = CRS.from_epsg(32610)
        tiles = [{"data": data, "bounds": bounds, "crs": crs}]

        mosaic, out_bounds = _merge_tiles(tiles)
        np.testing.assert_array_equal(mosaic, data)
        self.assertEqual(out_bounds, bounds)

    def test_single_tile_preserves_shape(self):
        data = np.ones((5, 10, 20), dtype=np.int8)
        bounds = (0.0, 0.0, 200.0, 100.0)
        crs = CRS.from_epsg(32610)
        tiles = [{"data": data, "bounds": bounds, "crs": crs}]

        mosaic, _ = _merge_tiles(tiles)
        self.assertEqual(mosaic.shape, (5, 10, 20))


class TestMergeTilesAdjacent(unittest.TestCase):
    """Two side-by-side tiles should merge into a wider mosaic."""

    def _make_adjacent_tiles(self, fill_left=1, fill_right=2):
        """Create two adjacent tiles (left and right) at 10m resolution."""
        crs = CRS.from_epsg(32610)
        h, w, bands = 32, 32, 3
        res = 10.0

        left_data = np.full((bands, h, w), fill_left, dtype=np.int8)
        left_bounds = (500000.0, 4000000.0, 500000.0 + w * res, 4000000.0 + h * res)

        right_data = np.full((bands, h, w), fill_right, dtype=np.int8)
        right_bounds = (
            500000.0 + w * res,
            4000000.0,
            500000.0 + 2 * w * res,
            4000000.0 + h * res,
        )

        return [
            {"data": left_data, "bounds": left_bounds, "crs": crs},
            {"data": right_data, "bounds": right_bounds, "crs": crs},
        ]

    def test_mosaic_width_doubles(self):
        tiles = self._make_adjacent_tiles()
        mosaic, _ = _merge_tiles(tiles)
        # Two 32-wide tiles side by side → 64 wide
        self.assertEqual(mosaic.shape[2], 64)
        self.assertEqual(mosaic.shape[1], 32)
        self.assertEqual(mosaic.shape[0], 3)

    def test_mosaic_contains_both_tiles(self):
        tiles = self._make_adjacent_tiles(fill_left=10, fill_right=20)
        mosaic, _ = _merge_tiles(tiles)
        # Left half should be 10, right half should be 20
        self.assertTrue(np.all(mosaic[:, :, :32] == 10))
        self.assertTrue(np.all(mosaic[:, :, 32:] == 20))

    def test_mosaic_bounds_span_both_tiles(self):
        tiles = self._make_adjacent_tiles()
        _, bounds = _merge_tiles(tiles)
        west, south, east, north = bounds
        self.assertAlmostEqual(west, 500000.0, places=1)
        self.assertAlmostEqual(east, 500640.0, places=1)
        self.assertAlmostEqual(south, 4000000.0, places=1)
        self.assertAlmostEqual(north, 4000320.0, places=1)


class TestMergeTilesVertical(unittest.TestCase):
    """Two vertically stacked tiles should merge into a taller mosaic."""

    def test_mosaic_height_doubles(self):
        crs = CRS.from_epsg(32610)
        h, w, bands = 32, 32, 2
        res = 10.0

        bottom_data = np.full((bands, h, w), 5, dtype=np.int8)
        bottom_bounds = (500000.0, 4000000.0, 500000.0 + w * res, 4000000.0 + h * res)

        top_data = np.full((bands, h, w), 15, dtype=np.int8)
        top_bounds = (
            500000.0,
            4000000.0 + h * res,
            500000.0 + w * res,
            4000000.0 + 2 * h * res,
        )

        tiles = [
            {"data": bottom_data, "bounds": bottom_bounds, "crs": crs},
            {"data": top_data, "bounds": top_bounds, "crs": crs},
        ]
        mosaic, bounds = _merge_tiles(tiles)
        self.assertEqual(mosaic.shape[1], 64)  # height doubled
        self.assertEqual(mosaic.shape[2], 32)  # width same


class TestMergeTilesGrid(unittest.TestCase):
    """2x2 grid of tiles should merge into a single mosaic."""

    def test_four_tile_grid(self):
        crs = CRS.from_epsg(32610)
        h, w, bands = 16, 16, 1
        res = 10.0
        base_x, base_y = 500000.0, 4000000.0

        tiles = []
        for row in range(2):
            for col in range(2):
                fill_val = row * 2 + col + 1  # 1, 2, 3, 4
                data = np.full((bands, h, w), fill_val, dtype=np.int8)
                x0 = base_x + col * w * res
                y0 = base_y + row * h * res
                bounds = (x0, y0, x0 + w * res, y0 + h * res)
                tiles.append({"data": data, "bounds": bounds, "crs": crs})

        mosaic, bounds = _merge_tiles(tiles)
        self.assertEqual(mosaic.shape, (1, 32, 32))

        west, south, east, north = bounds
        self.assertAlmostEqual(west, base_x, places=1)
        self.assertAlmostEqual(east, base_x + 2 * w * res, places=1)


class TestMergeTilesOverlap(unittest.TestCase):
    """Overlapping tiles should merge without error (last-write-wins)."""

    def test_overlapping_tiles_do_not_error(self):
        crs = CRS.from_epsg(32610)
        h, w, bands = 32, 32, 2
        res = 10.0

        # 50% horizontal overlap
        tile1_data = np.full((bands, h, w), 10, dtype=np.int8)
        tile1_bounds = (500000.0, 4000000.0, 500000.0 + w * res, 4000000.0 + h * res)

        tile2_data = np.full((bands, h, w), 20, dtype=np.int8)
        overlap_shift = w * res / 2  # 50% overlap
        tile2_bounds = (
            500000.0 + overlap_shift,
            4000000.0,
            500000.0 + overlap_shift + w * res,
            4000000.0 + h * res,
        )

        tiles = [
            {"data": tile1_data, "bounds": tile1_bounds, "crs": crs},
            {"data": tile2_data, "bounds": tile2_bounds, "crs": crs},
        ]
        mosaic, bounds = _merge_tiles(tiles)
        # Width should be ~1.5x a single tile (due to 50% overlap)
        self.assertGreater(mosaic.shape[2], w)
        self.assertLess(mosaic.shape[2], 2 * w)


class TestMergeTilesDtype(unittest.TestCase):
    """Merged mosaic should preserve the input dtype."""

    def test_int8_preserved(self):
        crs = CRS.from_epsg(32610)
        tiles = self._make_two_tiles(np.int8)
        mosaic, _ = _merge_tiles(tiles)
        self.assertEqual(mosaic.dtype, np.int8)

    def test_float32_preserved(self):
        crs = CRS.from_epsg(32610)
        tiles = self._make_two_tiles(np.float32)
        mosaic, _ = _merge_tiles(tiles)
        self.assertEqual(mosaic.dtype, np.float32)

    def _make_two_tiles(self, dtype):
        crs = CRS.from_epsg(32610)
        h, w, bands = 16, 16, 2
        res = 10.0
        data1 = np.ones((bands, h, w), dtype=dtype)
        data2 = np.ones((bands, h, w), dtype=dtype) * 2
        bounds1 = (500000.0, 4000000.0, 500000.0 + w * res, 4000000.0 + h * res)
        bounds2 = (
            500000.0 + w * res,
            4000000.0,
            500000.0 + 2 * w * res,
            4000000.0 + h * res,
        )
        return [
            {"data": data1, "bounds": bounds1, "crs": crs},
            {"data": data2, "bounds": bounds2, "crs": crs},
        ]


class TestMergeTilesMultiband(unittest.TestCase):
    """Merging should work correctly with many bands (like 64-band embeddings)."""

    def test_64_band_merge(self):
        crs = CRS.from_epsg(32610)
        h, w, bands = 8, 8, 64
        res = 10.0

        data1 = np.random.randint(-127, 127, (bands, h, w), dtype=np.int8)
        data2 = np.random.randint(-127, 127, (bands, h, w), dtype=np.int8)
        bounds1 = (500000.0, 4000000.0, 500000.0 + w * res, 4000000.0 + h * res)
        bounds2 = (
            500000.0 + w * res,
            4000000.0,
            500000.0 + 2 * w * res,
            4000000.0 + h * res,
        )

        tiles = [
            {"data": data1, "bounds": bounds1, "crs": crs},
            {"data": data2, "bounds": bounds2, "crs": crs},
        ]
        mosaic, _ = _merge_tiles(tiles)
        self.assertEqual(mosaic.shape[0], 64)
        self.assertEqual(mosaic.shape[2], 16)  # 2 * 8 wide

        # Verify data integrity: left half matches data1
        np.testing.assert_array_equal(mosaic[:, :, :8], data1)
        # Right half matches data2
        np.testing.assert_array_equal(mosaic[:, :, 8:], data2)


class TestMergeTilesThreeTiles(unittest.TestCase):
    """Three tiles in a row should merge correctly."""

    def test_three_horizontal_tiles(self):
        crs = CRS.from_epsg(32610)
        h, w, bands = 16, 16, 3
        res = 10.0
        base_x = 500000.0
        base_y = 4000000.0

        tiles = []
        for i in range(3):
            data = np.full((bands, h, w), (i + 1) * 10, dtype=np.int8)
            x0 = base_x + i * w * res
            bounds = (x0, base_y, x0 + w * res, base_y + h * res)
            tiles.append({"data": data, "bounds": bounds, "crs": crs})

        mosaic, bounds = _merge_tiles(tiles)
        self.assertEqual(mosaic.shape[2], 48)  # 3 * 16

        # Verify each third
        self.assertTrue(np.all(mosaic[:, :, :16] == 10))
        self.assertTrue(np.all(mosaic[:, :, 16:32] == 20))
        self.assertTrue(np.all(mosaic[:, :, 32:] == 30))


class TestMergeTilesMixedCRS(unittest.TestCase):
    """Test that mixed CRS raises an error."""

    def test_mixed_crs_raises(self):
        tile_a = {
            "data": np.ones((1, 4, 4), dtype=np.float32),
            "bounds": (0.0, 0.0, 1.0, 1.0),
            "crs": CRS.from_epsg(4326),
        }
        tile_b = {
            "data": np.ones((1, 4, 4), dtype=np.float32),
            "bounds": (1.0, 0.0, 2.0, 1.0),
            "crs": CRS.from_epsg(32617),
        }
        with self.assertRaises(ValueError, msg="same CRS"):
            _merge_tiles([tile_a, tile_b])


if __name__ == "__main__":
    unittest.main()
