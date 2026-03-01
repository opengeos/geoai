#!/usr/bin/env python

"""Tests for radiometric normalization in `geoai.landcover_utils` module."""

import inspect
import os
import tempfile
import unittest
import warnings

import numpy as np

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from geoai.landcover_utils import (
    _compute_distances,
    _compute_sample,
    _linear_reg,
    _lirrn,
    _load_raster,
    _sample_selection,
    _save_raster,
    normalize_radiometric,
)

# ---------------------------------------------------------------------------
# Helper to build synthetic rasters on disk
# ---------------------------------------------------------------------------


def _create_test_raster(path, height=50, width=50, bands=3, seed=42):
    """Write a small synthetic GeoTIFF for testing."""
    rng = np.random.default_rng(seed)
    data = (rng.random((bands, height, width)) * 255).astype(np.float64)
    profile = {
        "driver": "GTiff",
        "dtype": "float64",
        "width": width,
        "height": height,
        "count": bands,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_bounds(
            -122.5, 37.7, -122.3, 37.9, width, height
        ),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return data, profile


# ===================================================================
# Import & discoverability
# ===================================================================


class TestNormalizeRadiometricImport(unittest.TestCase):
    """Ensure the public function is importable and discoverable."""

    def test_importable_from_module(self):
        from geoai.landcover_utils import normalize_radiometric

        self.assertTrue(callable(normalize_radiometric))

    def test_importable_from_package(self):
        from geoai import normalize_radiometric

        self.assertTrue(callable(normalize_radiometric))

    def test_in_module_all(self):
        import geoai.landcover_utils

        self.assertIn("normalize_radiometric", geoai.landcover_utils.__all__)

    def test_in_lazy_symbol_map(self):
        import geoai

        self.assertIn("normalize_radiometric", geoai._LAZY_SYMBOL_MAP)


# ===================================================================
# Signature
# ===================================================================


class TestNormalizeRadiometricSignature(unittest.TestCase):
    """Verify function signature matches the documented API."""

    def setUp(self):
        self.sig = inspect.signature(normalize_radiometric)

    def test_has_expected_parameters(self):
        expected = {
            "subject_image",
            "reference_image",
            "output_path",
            "method",
            "p_n",
            "num_quantisation_classes",
            "num_sampling_rounds",
            "subsample_ratio",
            "random_state",
        }
        self.assertEqual(set(self.sig.parameters.keys()), expected)

    def test_default_method(self):
        self.assertEqual(self.sig.parameters["method"].default, "lirrn")

    def test_default_p_n(self):
        self.assertEqual(self.sig.parameters["p_n"].default, 500)

    def test_default_num_sampling_rounds(self):
        self.assertEqual(self.sig.parameters["num_sampling_rounds"].default, 3)

    def test_default_output_path(self):
        self.assertIsNone(self.sig.parameters["output_path"].default)

    def test_default_random_state(self):
        self.assertIsNone(self.sig.parameters["random_state"].default)


# ===================================================================
# Input validation
# ===================================================================


class TestNormalizeRadiometricValidation(unittest.TestCase):
    """Verify input validation and error messages."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.sub = rng.random((10, 10, 3)) * 255
        self.ref = rng.random((10, 10, 3)) * 255

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, method="histogram")

    def test_p_n_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, p_n=0)

    def test_p_n_negative_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, p_n=-5)

    def test_num_sampling_rounds_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, num_sampling_rounds=0)

    def test_subsample_ratio_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, subsample_ratio=0.0)

    def test_subsample_ratio_above_one_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, subsample_ratio=1.5)

    def test_band_count_mismatch_raises(self):
        ref_4band = np.random.default_rng(0).random((10, 10, 4)) * 255
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, ref_4band)

    def test_2d_subject_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(np.random.default_rng(0).random((10, 10)), self.ref)

    def test_2d_reference_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, np.random.default_rng(0).random((10, 10)))

    def test_1d_input_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(np.array([1, 2, 3]), np.array([4, 5, 6]))

    def test_file_not_found_subject(self):
        with self.assertRaises(FileNotFoundError):
            normalize_radiometric("/nonexistent/path.tif", self.ref)

    def test_file_not_found_reference(self):
        with self.assertRaises(FileNotFoundError):
            normalize_radiometric(self.sub, "/nonexistent/path.tif")

    def test_output_path_with_array_raises(self):
        with self.assertRaises(ValueError):
            normalize_radiometric(self.sub, self.ref, output_path="/tmp/out.tif")


# ===================================================================
# Core array functionality
# ===================================================================


class TestNormalizeRadiometricArrays(unittest.TestCase):
    """Core functional tests with numpy arrays (no file I/O)."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.subject = (rng.random((50, 50, 3)) * 200 + 30).astype(np.float64)
        self.reference = (rng.random((50, 50, 3)) * 180 + 40).astype(np.float64)

    def test_returns_tuple(self):
        result = normalize_radiometric(self.subject, self.reference, random_state=42)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_returns_ndarray_and_dict(self):
        norm_img, metrics = normalize_radiometric(
            self.subject, self.reference, random_state=42
        )
        self.assertIsInstance(norm_img, np.ndarray)
        self.assertIsInstance(metrics, dict)

    def test_metrics_keys(self):
        _, metrics = normalize_radiometric(
            self.subject, self.reference, random_state=42
        )
        self.assertIn("rmse", metrics)
        self.assertIn("r_adj", metrics)

    def test_metrics_shapes(self):
        _, metrics = normalize_radiometric(
            self.subject, self.reference, random_state=42
        )
        self.assertEqual(len(metrics["rmse"]), 3)
        self.assertEqual(len(metrics["r_adj"]), 3)

    def test_output_shape_matches_subject(self):
        norm_img, _ = normalize_radiometric(
            self.subject, self.reference, random_state=42
        )
        self.assertEqual(norm_img.shape, self.subject.shape)

    def test_output_dtype_is_float64(self):
        norm_img, _ = normalize_radiometric(
            self.subject, self.reference, random_state=42
        )
        self.assertEqual(norm_img.dtype, np.float64)

    def test_different_spatial_dimensions(self):
        ref_diff = (np.random.default_rng(42).random((30, 40, 3)) * 180 + 40).astype(
            np.float64
        )
        norm_img, _ = normalize_radiometric(self.subject, ref_diff, random_state=42)
        self.assertEqual(norm_img.shape, self.subject.shape)

    def test_single_band_image(self):
        sub_1 = self.subject[:, :, :1]
        ref_1 = self.reference[:, :, :1]
        norm_img, _ = normalize_radiometric(sub_1, ref_1, random_state=42)
        self.assertEqual(norm_img.shape, sub_1.shape)

    def test_reproducible_with_seed(self):
        r1, _ = normalize_radiometric(self.subject, self.reference, random_state=42)
        r2, _ = normalize_radiometric(self.subject, self.reference, random_state=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_run_without_error(self):
        r1, _ = normalize_radiometric(self.subject, self.reference, random_state=42)
        r2, _ = normalize_radiometric(self.subject, self.reference, random_state=99)
        self.assertEqual(r1.shape, r2.shape)

    def test_identical_images_approximately_unchanged(self):
        norm_img, _ = normalize_radiometric(
            self.subject, self.subject.copy(), random_state=42
        )
        np.testing.assert_allclose(norm_img, self.subject, rtol=0.15)

    def test_integer_input_converted(self):
        sub_int = self.subject.astype(np.uint8)
        ref_int = self.reference.astype(np.uint8)
        norm_img, _ = normalize_radiometric(sub_int, ref_int, random_state=42)
        self.assertEqual(norm_img.dtype, np.float64)

    def test_custom_p_n(self):
        norm_img, _ = normalize_radiometric(
            self.subject, self.reference, p_n=50, random_state=42
        )
        self.assertEqual(norm_img.shape, self.subject.shape)

    def test_custom_num_sampling_rounds(self):
        norm_img, _ = normalize_radiometric(
            self.subject,
            self.reference,
            num_sampling_rounds=1,
            random_state=42,
        )
        self.assertEqual(norm_img.shape, self.subject.shape)

    def test_custom_num_quantisation_classes(self):
        norm_img, _ = normalize_radiometric(
            self.subject,
            self.reference,
            num_quantisation_classes=2,
            random_state=42,
        )
        self.assertEqual(norm_img.shape, self.subject.shape)

    def test_generator_as_random_state(self):
        gen = np.random.default_rng(42)
        norm_img, _ = normalize_radiometric(
            self.subject, self.reference, random_state=gen
        )
        self.assertEqual(norm_img.shape, self.subject.shape)


# ===================================================================
# Edge cases
# ===================================================================


class TestNormalizeRadiometricEdgeCases(unittest.TestCase):
    """Edge-case handling (constant bands, tiny images, NaN, etc.)."""

    def test_all_zero_band(self):
        sub = np.zeros((20, 20, 2), dtype=np.float64)
        sub[:, :, 1] = np.random.default_rng(42).random((20, 20)) * 100
        ref = np.random.default_rng(42).random((20, 20, 2)).astype(np.float64) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm_img, _ = normalize_radiometric(sub, ref, random_state=42)
        self.assertEqual(norm_img.shape, sub.shape)

    def test_constant_value_band(self):
        sub = np.full((20, 20, 2), 128.0, dtype=np.float64)
        ref = np.random.default_rng(42).random((20, 20, 2)).astype(np.float64) * 255
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm_img, _ = normalize_radiometric(sub, ref, random_state=42)
        self.assertEqual(norm_img.shape, sub.shape)

    def test_very_small_image(self):
        sub = np.random.default_rng(42).random((3, 3, 2)).astype(np.float64) * 255
        ref = np.random.default_rng(99).random((3, 3, 2)).astype(np.float64) * 255
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm_img, _ = normalize_radiometric(sub, ref, p_n=500, random_state=42)
        self.assertEqual(norm_img.shape, sub.shape)

    def test_nan_in_input_warns(self):
        sub = np.random.default_rng(42).random((20, 20, 2)).astype(np.float64) * 255
        sub[5, 5, 0] = np.nan
        ref = np.random.default_rng(99).random((20, 20, 2)).astype(np.float64) * 255
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            norm_img, _ = normalize_radiometric(sub, ref, random_state=42)
        self.assertEqual(norm_img.shape, sub.shape)
        self.assertFalse(np.any(np.isnan(norm_img)))
        nan_warnings = [x for x in w if "NaN" in str(x.message)]
        self.assertGreater(len(nan_warnings), 0)

    def test_inf_in_input_warns(self):
        sub = np.random.default_rng(42).random((20, 20, 2)).astype(np.float64) * 255
        sub[5, 5, 0] = np.inf
        ref = np.random.default_rng(99).random((20, 20, 2)).astype(np.float64) * 255
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            norm_img, _ = normalize_radiometric(sub, ref, random_state=42)
        self.assertFalse(np.any(np.isinf(norm_img)))
        inf_warnings = [x for x in w if "infinite" in str(x.message)]
        self.assertGreater(len(inf_warnings), 0)

    def test_large_band_count(self):
        rng = np.random.default_rng(42)
        sub = (rng.random((20, 20, 8)) * 200 + 30).astype(np.float64)
        ref = (rng.random((20, 20, 8)) * 180 + 40).astype(np.float64)
        norm_img, metrics = normalize_radiometric(sub, ref, random_state=42)
        self.assertEqual(norm_img.shape, (20, 20, 8))
        self.assertEqual(len(metrics["rmse"]), 8)
        self.assertEqual(len(metrics["r_adj"]), 8)


# ===================================================================
# File I/O
# ===================================================================


@unittest.skipUnless(HAS_RASTERIO, "rasterio not available")
class TestNormalizeRadiometricFileIO(unittest.TestCase):
    """File-based input/output tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sub_path = os.path.join(self.tmpdir, "subject.tif")
        self.ref_path = os.path.join(self.tmpdir, "reference.tif")
        self.out_path = os.path.join(self.tmpdir, "normalized.tif")
        _create_test_raster(self.sub_path, seed=42)
        _create_test_raster(self.ref_path, seed=99)

    def tearDown(self):
        for f in [self.sub_path, self.ref_path, self.out_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def test_file_path_returns_array(self):
        norm_img, metrics = normalize_radiometric(
            self.sub_path, self.ref_path, random_state=42
        )
        self.assertIsInstance(norm_img, np.ndarray)
        self.assertEqual(norm_img.shape, (50, 50, 3))

    def test_output_path_saves_file(self):
        norm_img, _ = normalize_radiometric(
            self.sub_path,
            self.ref_path,
            output_path=self.out_path,
            random_state=42,
        )
        self.assertTrue(os.path.exists(self.out_path))

    def test_output_file_is_valid_geotiff(self):
        normalize_radiometric(
            self.sub_path,
            self.ref_path,
            output_path=self.out_path,
            random_state=42,
        )
        with rasterio.open(self.out_path) as src:
            self.assertEqual(src.count, 3)
            self.assertEqual(src.height, 50)
            self.assertEqual(src.width, 50)
            self.assertIsNotNone(src.crs)

    def test_mixed_input_file_and_array(self):
        ref_arr = np.random.default_rng(99).random((50, 50, 3)).astype(np.float64) * 255
        norm_img, _ = normalize_radiometric(self.sub_path, ref_arr, random_state=42)
        self.assertIsInstance(norm_img, np.ndarray)
        self.assertEqual(norm_img.shape, (50, 50, 3))


# ===================================================================
# Private helper functions
# ===================================================================


class TestHelperComputeDistances(unittest.TestCase):
    """Tests for _compute_distances edge cases."""

    def test_fewer_samples_than_pn(self):
        a1 = np.array([1.0, 2.0, 3.0])
        b1 = np.array([4.0, 5.0, 6.0])
        indices = np.array([0, 1])
        sub, ref = _compute_distances(10, a1, b1, indices)
        self.assertGreater(len(sub), 0)
        self.assertGreater(len(ref), 0)

    def test_exact_pn(self):
        a1 = np.array([1.0, 2.0, 3.0])
        b1 = np.array([4.0, 5.0, 6.0])
        indices = np.array([0, 1, 2])
        sub, ref = _compute_distances(3, a1, b1, indices)
        self.assertEqual(len(sub), 3)
        self.assertEqual(len(ref), 3)

    def test_indices_out_of_bounds_handled(self):
        a1 = np.array([1.0, 2.0])
        b1 = np.array([3.0, 4.0])
        indices = np.array([0, 1, 5, 10])
        sub, ref = _compute_distances(2, a1, b1, indices)
        self.assertGreater(len(sub), 0)


class TestHelperSampleSelection(unittest.TestCase):
    """Tests for _sample_selection edge cases."""

    def test_returns_paired_arrays(self):
        rng = np.random.default_rng(42)
        a = rng.random(100) * 255
        b = rng.random(100) * 255
        indices = rng.integers(0, 20, size=5)
        sub_s, ref_s = _sample_selection(20, a, b, indices)
        self.assertEqual(len(sub_s), len(ref_s))

    def test_empty_inputs(self):
        a = np.zeros(50)
        b = np.zeros(50)
        indices = np.array([0, 1, 2])
        sub_s, ref_s = _sample_selection(10, a, b, indices)
        self.assertEqual(len(sub_s), len(ref_s))


class TestHelperLinearReg(unittest.TestCase):
    """Tests for _linear_reg."""

    def test_perfect_correlation(self):
        sub = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ref = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        band = np.array([[1.0, 2.0], [3.0, 4.0]])
        result, r_adj, rmse = _linear_reg(sub, ref, band)
        np.testing.assert_allclose(result, [[2.0, 4.0], [6.0, 8.0]], rtol=1e-5)
        self.assertAlmostEqual(r_adj, 1.0, places=3)
        self.assertAlmostEqual(rmse, 0.0, places=5)

    def test_insufficient_samples_raises(self):
        with self.assertRaises(ValueError):
            _linear_reg(np.array([1.0]), np.array([2.0]), np.array([[1.0, 2.0]]))

    def test_zero_samples_filtered(self):
        sub = np.array([0.0, 0.0, 3.0, 4.0, 5.0])
        ref = np.array([0.0, 0.0, 6.0, 8.0, 10.0])
        band = np.array([[3.0, 4.0], [5.0, 6.0]])
        result, r_adj, rmse = _linear_reg(sub, ref, band)
        self.assertEqual(result.shape, band.shape)


@unittest.skipUnless(HAS_RASTERIO, "rasterio not available")
class TestHelperLoadSaveRaster(unittest.TestCase):
    """Tests for _load_raster and _save_raster."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.raster_path = os.path.join(self.tmpdir, "test.tif")
        _create_test_raster(self.raster_path, height=20, width=20, bands=3)

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_load_returns_hwb(self):
        img, profile = _load_raster(self.raster_path)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape, (20, 20, 3))
        self.assertEqual(img.dtype, np.float64)

    def test_load_returns_profile(self):
        _, profile = _load_raster(self.raster_path)
        self.assertIn("crs", profile)
        self.assertIn("transform", profile)

    def test_load_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _load_raster("/nonexistent/test.tif")

    def test_save_roundtrip(self):
        img, profile = _load_raster(self.raster_path)
        out_path = os.path.join(self.tmpdir, "roundtrip.tif")
        _save_raster(out_path, img, profile)

        img2, _ = _load_raster(out_path)
        np.testing.assert_allclose(img, img2, rtol=1e-10)

    def test_save_2d_raises(self):
        _, profile = _load_raster(self.raster_path)
        with self.assertRaises(ValueError):
            _save_raster(
                os.path.join(self.tmpdir, "bad.tif"),
                np.zeros((20, 20)),
                profile,
            )


if __name__ == "__main__":
    unittest.main()
