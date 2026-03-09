#!/usr/bin/env python

"""Tests for ``geoai.inference`` module."""

import inspect
import unittest
from unittest.mock import MagicMock

import numpy as np


class TestInferenceImport(unittest.TestCase):
    """Tests for inference module import behaviour."""

    def test_module_imports(self):
        """Test that the inference module can be imported."""
        import geoai.inference

        self.assertTrue(hasattr(geoai.inference, "predict_geotiff"))

    def test_predict_geotiff_exists(self):
        """Test that predict_geotiff function exists and is callable."""
        from geoai.inference import predict_geotiff

        self.assertTrue(callable(predict_geotiff))

    def test_create_weight_mask_exists(self):
        """Test that create_weight_mask function exists and is callable."""
        from geoai.inference import create_weight_mask

        self.assertTrue(callable(create_weight_mask))

    def test_blend_mode_exists(self):
        """Test that BlendMode enum exists with expected members."""
        from geoai.inference import BlendMode

        self.assertTrue(hasattr(BlendMode, "NONE"))
        self.assertTrue(hasattr(BlendMode, "LINEAR"))
        self.assertTrue(hasattr(BlendMode, "COSINE"))
        self.assertTrue(hasattr(BlendMode, "SPLINE"))

    def test_d4_forward_exists(self):
        """Test that d4_forward function exists and is callable."""
        from geoai.inference import d4_forward

        self.assertTrue(callable(d4_forward))

    def test_d4_inverse_exists(self):
        """Test that d4_inverse function exists and is callable."""
        from geoai.inference import d4_inverse

        self.assertTrue(callable(d4_inverse))

    def test_d4_tta_forward_exists(self):
        """Test that d4_tta_forward function exists and is callable."""
        from geoai.inference import d4_tta_forward

        self.assertTrue(callable(d4_tta_forward))


class TestInferenceExports(unittest.TestCase):
    """Tests for inference module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined in the inference module."""
        import geoai.inference

        self.assertTrue(hasattr(geoai.inference, "__all__"))

    def test_all_exports_contain_expected_names(self):
        """Test that __all__ contains the expected public API names."""
        from geoai.inference import __all__

        expected = [
            "BlendMode",
            "create_weight_mask",
            "predict_geotiff",
            "d4_forward",
            "d4_inverse",
            "d4_tta_forward",
        ]
        for name in expected:
            self.assertIn(name, __all__)


class TestPredictGeotiffSignature(unittest.TestCase):
    """Tests for predict_geotiff function signature."""

    def test_predict_geotiff_params(self):
        """Test predict_geotiff has expected parameters."""
        from geoai.inference import predict_geotiff

        sig = inspect.signature(predict_geotiff)
        expected_params = [
            "model",
            "input_raster",
            "output_raster",
            "tile_size",
            "overlap",
            "batch_size",
            "input_bands",
            "num_classes",
            "output_dtype",
            "output_nodata",
            "blend_mode",
            "blend_power",
            "tta",
            "preprocess_fn",
            "postprocess_fn",
            "device",
            "compress",
            "verbose",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)

    def test_predict_geotiff_defaults(self):
        """Test predict_geotiff default parameter values."""
        from geoai.inference import predict_geotiff

        sig = inspect.signature(predict_geotiff)
        self.assertEqual(sig.parameters["tile_size"].default, 256)
        self.assertEqual(sig.parameters["overlap"].default, 64)
        self.assertEqual(sig.parameters["batch_size"].default, 4)
        self.assertEqual(sig.parameters["num_classes"].default, 1)
        self.assertEqual(sig.parameters["tta"].default, False)
        self.assertEqual(sig.parameters["blend_mode"].default, "spline")
        self.assertIsNone(sig.parameters["device"].default)
        self.assertIsNone(sig.parameters["preprocess_fn"].default)
        self.assertIsNone(sig.parameters["postprocess_fn"].default)
        self.assertEqual(sig.parameters["verbose"].default, True)


class TestCreateWeightMask(unittest.TestCase):
    """Tests for create_weight_mask function behaviour."""

    def test_none_mode_returns_ones(self):
        """Test that 'none' mode returns an all-ones mask."""
        from geoai.inference import create_weight_mask

        mask = create_weight_mask(64, 16, mode="none")
        np.testing.assert_array_equal(mask, np.ones((64, 64), dtype=np.float32))

    def test_linear_mode_shape(self):
        """Test that 'linear' mode returns correct shape and dtype."""
        from geoai.inference import create_weight_mask

        mask = create_weight_mask(128, 32, mode="linear")
        self.assertEqual(mask.shape, (128, 128))
        self.assertEqual(mask.dtype, np.float32)

    def test_cosine_mode_shape(self):
        """Test that 'cosine' mode returns correct shape and dtype."""
        from geoai.inference import create_weight_mask

        mask = create_weight_mask(128, 32, mode="cosine")
        self.assertEqual(mask.shape, (128, 128))
        self.assertEqual(mask.dtype, np.float32)

    def test_spline_mode_shape(self):
        """Test that 'spline' mode returns correct shape and dtype."""
        from geoai.inference import create_weight_mask

        mask = create_weight_mask(128, 32, mode="spline")
        self.assertEqual(mask.shape, (128, 128))
        self.assertEqual(mask.dtype, np.float32)

    def test_center_has_max_weight(self):
        """Test that center pixel has higher weight than corner for all modes."""
        from geoai.inference import create_weight_mask

        for mode in ["linear", "cosine", "spline"]:
            mask = create_weight_mask(64, 16, mode=mode)
            center = mask[32, 32]
            corner = mask[0, 0]
            self.assertGreater(
                center,
                corner,
                f"Center should have higher weight than corner for mode={mode}",
            )

    def test_symmetry(self):
        """Test that weight mask is symmetric along both axes."""
        from geoai.inference import create_weight_mask

        for mode in ["linear", "cosine", "spline"]:
            mask = create_weight_mask(64, 16, mode=mode)
            np.testing.assert_array_almost_equal(
                mask, mask[::-1, :], err_msg=f"Not vertically symmetric for {mode}"
            )
            np.testing.assert_array_almost_equal(
                mask, mask[:, ::-1], err_msg=f"Not horizontally symmetric for {mode}"
            )

    def test_zero_overlap_returns_ones(self):
        """Test that zero overlap returns ones for all modes."""
        from geoai.inference import create_weight_mask

        for mode in ["none", "linear", "cosine", "spline"]:
            mask = create_weight_mask(64, 0, mode=mode)
            np.testing.assert_array_equal(
                mask,
                np.ones((64, 64), dtype=np.float32),
                err_msg=f"Zero overlap should return ones for mode={mode}",
            )

    def test_invalid_overlap_too_large_raises(self):
        """Test that overlap >= tile_size raises ValueError."""
        from geoai.inference import create_weight_mask

        with self.assertRaises(ValueError):
            create_weight_mask(64, 64, mode="spline")

    def test_invalid_overlap_negative_raises(self):
        """Test that negative overlap raises ValueError."""
        from geoai.inference import create_weight_mask

        with self.assertRaises(ValueError):
            create_weight_mask(64, -1, mode="spline")

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        from geoai.inference import create_weight_mask

        with self.assertRaises(ValueError):
            create_weight_mask(64, 16, mode="invalid")

    def test_blend_mode_enum_accepted(self):
        """Test that BlendMode enum values are accepted."""
        from geoai.inference import BlendMode, create_weight_mask

        mask = create_weight_mask(64, 16, mode=BlendMode.SPLINE)
        self.assertEqual(mask.shape, (64, 64))

    def test_values_in_valid_range(self):
        """Test that all weight values are non-negative."""
        from geoai.inference import create_weight_mask

        for mode in ["linear", "cosine", "spline"]:
            mask = create_weight_mask(128, 32, mode=mode)
            self.assertTrue(np.all(mask >= 0), f"Negative values for mode={mode}")

    def test_large_overlap_no_corruption(self):
        """Test that overlap > tile_size/2 does not corrupt the ramp."""
        from geoai.inference import create_weight_mask

        for mode in ["linear", "cosine"]:
            mask = create_weight_mask(64, 40, mode=mode)
            row = mask[32, :]
            # Values must stay in [0, 1]
            self.assertTrue(np.all(row >= 0.0), f"{mode}: negative weights")
            self.assertTrue(np.all(row <= 1.0), f"{mode}: weights exceed 1.0")
            # Must be symmetric
            np.testing.assert_array_almost_equal(
                row, row[::-1], err_msg=f"{mode}: not symmetric with large overlap"
            )
            # Must be monotone increasing then decreasing (unimodal)
            half = len(row) // 2
            self.assertTrue(
                np.all(np.diff(row[:half]) >= -1e-6),
                f"{mode}: not monotone increasing with large overlap",
            )
            self.assertTrue(
                np.all(np.diff(row[half:]) <= 1e-6),
                f"{mode}: not monotone decreasing with large overlap",
            )


class TestSplineWindowFixes(unittest.TestCase):
    """Tests for spline window edge cases and fixes."""

    def test_spline_taper_reaches_one(self):
        """Test that the spline taper smoothly reaches 1.0 at the boundary."""
        from geoai.inference import _spline_window_1d

        for overlap in [4, 8, 16, 64]:
            w = _spline_window_1d(128, overlap)
            # The last taper value (at index overlap-1) should be exactly 1.0
            self.assertAlmostEqual(
                w[overlap - 1],
                1.0,
                places=10,
                msg=f"Taper should reach 1.0 at boundary for overlap={overlap}",
            )
            # Centre should be 1.0
            self.assertAlmostEqual(w[64], 1.0)

    def test_spline_taper_continuity(self):
        """Test no discontinuity between taper end and centre region."""
        from geoai.inference import _spline_window_1d

        w = _spline_window_1d(128, 32)
        # The jump between the last taper pixel and the first centre pixel
        # should be zero (both are 1.0).
        self.assertAlmostEqual(
            abs(w[31] - w[32]),
            0.0,
            places=10,
            msg="No discontinuity at taper-centre boundary",
        )

    def test_spline_large_overlap_raises(self):
        """Test that spline mode raises for overlap > tile_size // 2."""
        from geoai.inference import create_weight_mask

        with self.assertRaises(ValueError, msg="Should reject overlap > tile_size//2"):
            create_weight_mask(64, 40, mode="spline")

    def test_spline_half_overlap_ok(self):
        """Test that spline mode works at exactly tile_size // 2."""
        from geoai.inference import create_weight_mask

        mask = create_weight_mask(64, 32, mode="spline")
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(np.all(mask >= 0))
        self.assertTrue(np.all(mask <= 1.0 + 1e-7))

    def test_spline_monotone_increasing_left_half(self):
        """Test that spline 1D window is monotone non-decreasing in left half."""
        from geoai.inference import _spline_window_1d

        w = _spline_window_1d(128, 32)
        half = len(w) // 2
        self.assertTrue(
            np.all(np.diff(w[:half]) >= -1e-10),
            "Spline window should be monotone non-decreasing in left half",
        )


class TestNodataValidation(unittest.TestCase):
    """Tests for output_nodata / output_dtype compatibility."""

    def test_nodata_incompatible_with_uint8_raises(self):
        """Test that nodata=-9999 with uint8 raises ValueError."""
        from geoai.inference import predict_geotiff

        with self.assertRaises(ValueError, msg="Should reject nodata=-9999 for uint8"):
            predict_geotiff(
                model=MagicMock(),
                input_raster=__file__,
                output_raster="/tmp/out.tif",
                output_dtype="uint8",
                output_nodata=-9999.0,
            )

    def test_nodata_compatible_with_uint8_ok(self):
        """Test that nodata=255 with uint8 passes validation (fails later on missing raster)."""
        from geoai.inference import predict_geotiff

        # Should NOT raise ValueError for nodata, but will fail at rasterio open
        # since __file__ is not a GeoTIFF. We just check that the nodata
        # validation itself passes.
        with self.assertRaises(Exception) as ctx:
            predict_geotiff(
                model=MagicMock(),
                input_raster=__file__,
                output_raster="/tmp/out.tif",
                output_dtype="uint8",
                output_nodata=255,
            )
        # The error should NOT be about nodata validation
        self.assertNotIn("output_nodata", str(ctx.exception))

    def test_nodata_float32_default_ok(self):
        """Test that default nodata=-9999 with float32 passes validation."""
        from geoai.inference import predict_geotiff

        # Should fail at rasterio, not at nodata validation
        with self.assertRaises(Exception) as ctx:
            predict_geotiff(
                model=MagicMock(),
                input_raster=__file__,
                output_raster="/tmp/out.tif",
                output_dtype="float32",
                output_nodata=-9999.0,
            )
        self.assertNotIn("output_nodata", str(ctx.exception))


class TestPredictGeotiffValidation(unittest.TestCase):
    """Tests for predict_geotiff input validation."""

    def test_file_not_found_raises(self):
        """Test that missing input file raises FileNotFoundError."""
        from geoai.inference import predict_geotiff

        with self.assertRaises(FileNotFoundError):
            predict_geotiff(
                model=MagicMock(),
                input_raster="/nonexistent/path/to/file.tif",
                output_raster="/tmp/out.tif",
            )

    def test_overlap_equal_tile_size_raises(self):
        """Test that overlap == tile_size raises ValueError."""
        from geoai.inference import predict_geotiff

        with self.assertRaises(ValueError):
            predict_geotiff(
                model=MagicMock(),
                input_raster=__file__,
                output_raster="/tmp/out.tif",
                tile_size=256,
                overlap=256,
            )

    def test_negative_overlap_raises(self):
        """Test that negative overlap raises ValueError."""
        from geoai.inference import predict_geotiff

        with self.assertRaises(ValueError):
            predict_geotiff(
                model=MagicMock(),
                input_raster=__file__,
                output_raster="/tmp/out.tif",
                tile_size=256,
                overlap=-1,
            )


class TestD4Transforms(unittest.TestCase):
    """Tests for D4 dihedral group transform functions."""

    def test_d4_forward_returns_8_tensors(self):
        """Test that d4_forward returns exactly 8 transformed tensors."""
        import torch
        from geoai.inference import d4_forward

        t = torch.randn(1, 3, 32, 32)
        result = d4_forward(t)
        self.assertEqual(len(result), 8)

    def test_d4_forward_preserves_shape(self):
        """Test that all D4 transforms preserve tensor shape."""
        import torch
        from geoai.inference import d4_forward

        t = torch.randn(2, 3, 32, 32)
        result = d4_forward(t)
        for i, tensor in enumerate(result):
            self.assertEqual(
                tensor.shape,
                t.shape,
                f"Transform {i} changed shape from {t.shape} to {tensor.shape}",
            )

    def test_d4_roundtrip(self):
        """Test that d4_inverse undoes d4_forward for all 8 transforms."""
        import torch
        from geoai.inference import d4_forward, d4_inverse

        t = torch.randn(1, 3, 32, 32)
        fwd = d4_forward(t)
        inv = d4_inverse(fwd)
        for i, tensor in enumerate(inv):
            torch.testing.assert_close(tensor, t, msg=f"Transform {i} roundtrip failed")

    def test_d4_tta_forward_shape(self):
        """Test that d4_tta_forward returns correct output shape."""
        import torch
        from geoai.inference import d4_tta_forward

        model = MagicMock()
        model.return_value = torch.randn(1, 2, 32, 32)
        t = torch.randn(1, 3, 32, 32)
        result = d4_tta_forward(model, t)
        self.assertEqual(result.shape, (1, 2, 32, 32))

    def test_d4_tta_forward_calls_model_8_times(self):
        """Test that d4_tta_forward calls the model exactly 8 times."""
        import torch
        from geoai.inference import d4_tta_forward

        model = MagicMock()
        model.return_value = torch.randn(1, 2, 32, 32)
        t = torch.randn(1, 3, 32, 32)
        d4_tta_forward(model, t)
        self.assertEqual(model.call_count, 8)

    def test_d4_identity_is_first(self):
        """Test that the first D4 transform is identity."""
        import torch
        from geoai.inference import d4_forward

        t = torch.randn(1, 3, 16, 16)
        result = d4_forward(t)
        torch.testing.assert_close(result[0], t)


class TestNormalization(unittest.TestCase):
    """Tests for weight normalization with multi-class output."""

    def test_multiclass_normalization(self):
        """Test that np.where broadcasting works for num_classes > 1."""
        from geoai.inference import create_weight_mask

        num_classes = 3
        height, width = 64, 64
        tile_size = 32
        overlap = 8

        weight_mask = create_weight_mask(tile_size, overlap, mode="spline")

        output_sum = np.random.rand(num_classes, height, width).astype(np.float64)
        weight_sum = np.random.rand(1, height, width).astype(np.float64) + 0.1

        valid = weight_sum > 0
        # This is the exact code from predict_geotiff — must not raise
        output_array = np.where(
            valid,
            output_sum / (weight_sum + 1e-8),
            -9999.0,
        ).astype(np.float32)

        self.assertEqual(output_array.shape, (num_classes, height, width))
        self.assertTrue(np.all(np.isfinite(output_array)))

    def test_zero_weight_gets_nodata(self):
        """Test that pixels with zero weight receive nodata value."""
        nodata = -9999.0
        output_sum = np.zeros((1, 4, 4), dtype=np.float64)
        weight_sum = np.zeros((1, 4, 4), dtype=np.float64)

        valid = weight_sum > 0
        result = np.where(valid, output_sum / (weight_sum + 1e-8), nodata)

        np.testing.assert_array_equal(result, np.full((1, 4, 4), nodata))


class TestLazyImport(unittest.TestCase):
    """Tests for lazy import registration in geoai.__init__."""

    def test_predict_geotiff_importable_from_geoai(self):
        """Test that predict_geotiff is importable from top-level geoai."""
        from geoai import predict_geotiff

        self.assertTrue(callable(predict_geotiff))

    def test_blend_mode_importable_from_geoai(self):
        """Test that BlendMode is importable from top-level geoai."""
        from geoai import BlendMode

        self.assertTrue(hasattr(BlendMode, "SPLINE"))

    def test_create_weight_mask_importable_from_geoai(self):
        """Test that create_weight_mask is importable from top-level geoai."""
        from geoai import create_weight_mask

        self.assertTrue(callable(create_weight_mask))


if __name__ == "__main__":
    unittest.main()
