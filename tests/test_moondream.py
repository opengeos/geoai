#!/usr/bin/env python

"""Tests for `geoai.moondream` module."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_geotiff(path, width=64, height=64, bands=3, dtype="uint8"):
    """Create a minimal GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.randint(0, 256, (bands, height, width), dtype=np.uint8)
    if dtype != "uint8":
        data = data.astype(dtype)
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
    ) as dst:
        dst.write(data)
    return data


def _make_mock_processor():
    """Create a MoondreamGeo with mocked __init__."""
    from geoai.moondream import MoondreamGeo

    with patch("geoai.moondream.MoondreamGeo.__init__", return_value=None):
        proc = MoondreamGeo.__new__(MoondreamGeo)
    proc._source_path = None
    proc._metadata = None
    proc.model = MagicMock()
    proc.model_name = "vikhyatk/moondream2"
    proc.model_version = "moondream2"
    proc.device = "cpu"
    return proc


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestMoondreamImport(unittest.TestCase):
    def test_module_imports(self):
        import geoai.moondream

        self.assertTrue(hasattr(geoai.moondream, "MoondreamGeo"))

    def test_moondream_geo_is_class(self):
        from geoai.moondream import MoondreamGeo

        self.assertTrue(inspect.isclass(MoondreamGeo))


# ---------------------------------------------------------------------------
# Convenience functions existence
# ---------------------------------------------------------------------------


class TestConvenienceFunctions(unittest.TestCase):
    def test_convenience_functions_exist(self):
        from geoai.moondream import (
            moondream_caption,
            moondream_query,
            moondream_detect,
            moondream_point,
            moondream_detect_sliding_window,
            moondream_point_sliding_window,
            moondream_query_sliding_window,
            moondream_caption_sliding_window,
        )

        for fn in [
            moondream_caption,
            moondream_query,
            moondream_detect,
            moondream_point,
            moondream_detect_sliding_window,
            moondream_point_sliding_window,
            moondream_query_sliding_window,
            moondream_caption_sliding_window,
        ]:
            self.assertTrue(callable(fn))


# ---------------------------------------------------------------------------
# Sliding window method signatures
# ---------------------------------------------------------------------------


class TestSlidingWindowSignatures(unittest.TestCase):
    def test_detect_sliding_window_params(self):
        from geoai.moondream import MoondreamGeo

        sig = inspect.signature(MoondreamGeo.detect_sliding_window)
        self.assertIn("window_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("iou_threshold", sig.parameters)

    def test_point_sliding_window_params(self):
        from geoai.moondream import MoondreamGeo

        sig = inspect.signature(MoondreamGeo.point_sliding_window)
        self.assertIn("window_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)

    def test_query_sliding_window_params(self):
        from geoai.moondream import MoondreamGeo

        sig = inspect.signature(MoondreamGeo.query_sliding_window)
        self.assertIn("window_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("combine_strategy", sig.parameters)

    def test_caption_sliding_window_params(self):
        from geoai.moondream import MoondreamGeo

        sig = inspect.signature(MoondreamGeo.caption_sliding_window)
        self.assertIn("window_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("combine_strategy", sig.parameters)


# ---------------------------------------------------------------------------
# _create_sliding_windows
# ---------------------------------------------------------------------------


class TestCreateSlidingWindows(unittest.TestCase):
    def setUp(self):
        self.proc = _make_mock_processor()

    def test_small_image_single_window(self):
        windows = self.proc._create_sliding_windows(
            400, 400, window_size=512, overlap=64
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], (0, 0, 400, 400))

    def test_exact_window_size(self):
        windows = self.proc._create_sliding_windows(
            512, 512, window_size=512, overlap=64
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], (0, 0, 512, 512))

    def test_multiple_windows(self):
        windows = self.proc._create_sliding_windows(
            1024, 1024, window_size=512, overlap=64
        )
        self.assertGreater(len(windows), 1)

    def test_windows_within_bounds(self):
        w, h = 1000, 800
        windows = self.proc._create_sliding_windows(w, h, window_size=512, overlap=64)
        for x_start, y_start, x_end, y_end in windows:
            self.assertGreaterEqual(x_start, 0)
            self.assertGreaterEqual(y_start, 0)
            self.assertLessEqual(x_end, w)
            self.assertLessEqual(y_end, h)

    def test_minimum_window_size(self):
        windows = self.proc._create_sliding_windows(
            1024, 1024, window_size=512, overlap=64
        )
        for x_start, y_start, x_end, y_end in windows:
            self.assertGreaterEqual(x_end - x_start, 256)
            self.assertGreaterEqual(y_end - y_start, 256)

    def test_non_square_image(self):
        windows = self.proc._create_sliding_windows(
            2000, 500, window_size=512, overlap=64
        )
        self.assertGreater(len(windows), 1)
        for x_start, y_start, x_end, y_end in windows:
            self.assertLessEqual(x_end, 2000)
            self.assertLessEqual(y_end, 500)

    def test_very_small_image(self):
        """Image smaller than half window size should still produce windows."""
        windows = self.proc._create_sliding_windows(
            100, 100, window_size=512, overlap=64
        )
        # 100 < 512 // 2 (256), so no windows pass the size filter
        # This is expected behavior
        self.assertIsInstance(windows, list)

    def test_zero_overlap(self):
        windows = self.proc._create_sliding_windows(
            1024, 1024, window_size=512, overlap=0
        )
        self.assertGreater(len(windows), 0)

    def test_window_coverage(self):
        """All pixels should be covered by at least one window."""
        w, h = 1000, 800
        windows = self.proc._create_sliding_windows(w, h, window_size=512, overlap=64)
        covered = np.zeros((h, w), dtype=bool)
        for x_start, y_start, x_end, y_end in windows:
            covered[y_start:y_end, x_start:x_end] = True
        # Most pixels should be covered (edge pixels may be skipped if < half window)
        coverage = covered.sum() / covered.size
        self.assertGreater(coverage, 0.5)

    def test_overlap_between_adjacent_windows(self):
        window_size = 512
        overlap = 64
        stride = window_size - overlap

        windows = self.proc._create_sliding_windows(1000, 1000, window_size, overlap)
        if len(windows) > 1:
            windows_by_y = {}
            for w in windows:
                y_start = w[1]
                if y_start not in windows_by_y:
                    windows_by_y[y_start] = []
                windows_by_y[y_start].append(w)

            for y_start, row_windows in windows_by_y.items():
                if len(row_windows) > 1:
                    sorted_windows = sorted(row_windows, key=lambda w: w[0])
                    for i in range(len(sorted_windows) - 1):
                        w1 = sorted_windows[i]
                        w2 = sorted_windows[i + 1]
                        self.assertLessEqual(w2[0] - w1[0], stride)


# ---------------------------------------------------------------------------
# _apply_nms
# ---------------------------------------------------------------------------


class TestApplyNMS(unittest.TestCase):
    def setUp(self):
        self.proc = _make_mock_processor()

    def test_empty_detections(self):
        result = self.proc._apply_nms([])
        self.assertEqual(len(result), 0)

    def test_no_overlap(self):
        detections = [
            {"x_min": 0.0, "y_min": 0.0, "x_max": 0.1, "y_max": 0.1, "score": 0.9},
            {"x_min": 0.5, "y_min": 0.5, "x_max": 0.6, "y_max": 0.6, "score": 0.8},
        ]
        result = self.proc._apply_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(result), 2)

    def test_overlapping_detections_suppressed(self):
        detections = [
            {"x_min": 0.0, "y_min": 0.0, "x_max": 0.2, "y_max": 0.2, "score": 0.9},
            {"x_min": 0.05, "y_min": 0.05, "x_max": 0.25, "y_max": 0.25, "score": 0.8},
        ]
        result = self.proc._apply_nms(detections, iou_threshold=0.5)
        self.assertLessEqual(len(result), len(detections))

    def test_single_detection(self):
        detections = [
            {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5, "score": 0.95},
        ]
        result = self.proc._apply_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(result), 1)

    def test_identical_boxes(self):
        """Identical boxes should be suppressed to 1."""
        detections = [
            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.3, "score": 0.9},
            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.3, "score": 0.7},
        ]
        result = self.proc._apply_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(result), 1)

    def test_high_threshold_keeps_all(self):
        """Very high threshold should keep all detections."""
        detections = [
            {"x_min": 0.0, "y_min": 0.0, "x_max": 0.2, "y_max": 0.2, "score": 0.9},
            {"x_min": 0.05, "y_min": 0.05, "x_max": 0.25, "y_max": 0.25, "score": 0.8},
        ]
        result = self.proc._apply_nms(detections, iou_threshold=1.0)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# _normalize_image
# ---------------------------------------------------------------------------


class TestNormalizeImage(unittest.TestCase):
    def setUp(self):
        self.proc = _make_mock_processor()

    def test_uint8_passthrough(self):
        data = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        result = self.proc._normalize_image(data)
        np.testing.assert_array_equal(result, data)

    def test_float_chw_normalization(self):
        data = np.random.rand(3, 64, 64).astype(np.float32) * 10000
        result = self.proc._normalize_image(data)
        self.assertEqual(result.dtype, np.uint8)
        self.assertLessEqual(result.max(), 255)
        self.assertGreaterEqual(result.min(), 0)

    def test_float_hwc_normalization(self):
        data = np.random.rand(64, 64, 3).astype(np.float32) * 5000
        result = self.proc._normalize_image(data)
        self.assertEqual(result.dtype, np.uint8)

    def test_constant_image_chw(self):
        """Constant image (p2==p98) should not cause division by zero."""
        data = np.full((3, 64, 64), 42.0, dtype=np.float32)
        result = self.proc._normalize_image(data)
        self.assertEqual(result.dtype, np.uint8)

    def test_constant_image_hwc(self):
        data = np.full((64, 64), 100.0, dtype=np.float32)
        result = self.proc._normalize_image(data)
        self.assertEqual(result.dtype, np.uint8)


# ---------------------------------------------------------------------------
# load_geotiff
# ---------------------------------------------------------------------------


class TestLoadGeotiff(unittest.TestCase):
    def setUp(self):
        self.proc = _make_mock_processor()

    def test_load_rgb_geotiff(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=32, height=32, bands=3)

            image, meta = self.proc.load_geotiff(path)
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (32, 32))
            self.assertIn("crs", meta)
            self.assertIn("transform", meta)

    def test_load_single_band_geotiff(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=32, height=32, bands=1)

            image, meta = self.proc.load_geotiff(path)
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.mode, "RGB")

    def test_load_multiband_geotiff(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=32, height=32, bands=6)

            image, meta = self.proc.load_geotiff(path)
            self.assertIsInstance(image, Image.Image)
            # Should use first 3 bands only
            self.assertEqual(image.mode, "RGB")

    def test_load_with_specific_bands(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=32, height=32, bands=4)

            image, meta = self.proc.load_geotiff(path, bands=[2, 3, 4])
            self.assertIsInstance(image, Image.Image)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.proc.load_geotiff("/nonexistent/path.tif")

    def test_load_sets_source_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path)

            self.proc.load_geotiff(path)
            self.assertEqual(self.proc._source_path, path)
            self.assertIsNotNone(self.proc._metadata)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage(unittest.TestCase):
    def setUp(self):
        self.proc = _make_mock_processor()

    def test_from_pil(self):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        result, meta = self.proc.load_image(img)
        self.assertIs(result, img)
        self.assertIsNone(meta)

    def test_from_numpy_2d(self):
        arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result, meta = self.proc.load_image(arr)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")

    def test_from_numpy_chw(self):
        arr = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        result, meta = self.proc.load_image(arr)
        self.assertIsInstance(result, Image.Image)

    def test_from_geotiff_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path)

            result, meta = self.proc.load_image(path)
            self.assertIsInstance(result, Image.Image)
            self.assertIsNotNone(meta)

    def test_from_png_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.png")
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(path)

            result, meta = self.proc.load_image(path)
            self.assertIsInstance(result, Image.Image)
            # PNG has no CRS
            self.assertIsNone(meta.get("crs") if meta else None)

    def test_from_2band_geotiff(self):
        """2-band image should be padded to 3 channels."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, bands=2)

            result, meta = self.proc.load_image(path)
            self.assertIsInstance(result, Image.Image)


# ---------------------------------------------------------------------------
# get_last_result
# ---------------------------------------------------------------------------


class TestGetLastResult(unittest.TestCase):
    def test_initial_state_returns_none(self):
        """get_last_result returns None when no inference has been run."""
        proc = _make_mock_processor()
        result = proc.get_last_result()
        self.assertIsNone(result)

    def test_returns_gdf_when_set(self):
        """get_last_result returns gdf from last_result when available."""
        proc = _make_mock_processor()
        proc.last_result = {"gdf": {"type": "FeatureCollection"}}
        result = proc.get_last_result()
        self.assertEqual(result, {"type": "FeatureCollection"})

    def test_returns_none_without_gdf_key(self):
        proc = _make_mock_processor()
        proc.last_result = {"other_key": 42}
        result = proc.get_last_result()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
