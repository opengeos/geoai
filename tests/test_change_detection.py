#!/usr/bin/env python

"""Tests for `geoai.change_detection` module."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAnyChange:
    """Mimics torchange's AnyChange hyperparameter storage.

    ``set_hyperparameters`` assigns every argument unconditionally, exactly
    like the upstream implementation, and ``bitemporal_match`` is stored as
    ``use_bitemporal_match``.
    """

    def __init__(self, *args, **kwargs):
        pass

    def make_mask_generator(self, **kwargs):
        pass

    def set_hyperparameters(
        self,
        change_confidence_threshold=155,
        auto_threshold=False,
        use_normalized_feature=True,
        area_thresh=0.8,
        match_hist=False,
        object_sim_thresh=60,
        bitemporal_match=True,
    ):
        self.area_thresh = area_thresh
        self.match_hist = match_hist
        self.change_confidence_threshold = change_confidence_threshold
        self.auto_threshold = auto_threshold
        self.use_normalized_feature = use_normalized_feature
        self.object_sim_thresh = object_sim_thresh
        self.use_bitemporal_match = bitemporal_match


def _create_test_geotiff(path, width=64, height=64, bands=3, dtype="uint8"):
    """Create a minimal GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.randint(0, 256, (bands, height, width), dtype=np.uint8)
    if dtype != "uint8":
        data = data.astype(dtype)
    transform = from_bounds(-77.05, 38.88, -77.04, 38.89, width, height)

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


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestChangeDetectionImport(unittest.TestCase):
    def test_module_imports(self):
        import geoai.change_detection

        self.assertTrue(hasattr(geoai.change_detection, "ChangeDetection"))

    def test_changestar_detection_exists(self):
        from geoai.change_detection import ChangeStarDetection

        self.assertTrue(callable(ChangeStarDetection))

    def test_list_changestar_models_exists(self):
        from geoai.change_detection import list_changestar_models

        self.assertTrue(callable(list_changestar_models))

    def test_download_checkpoint_exists(self):
        from geoai.change_detection import download_checkpoint

        self.assertTrue(callable(download_checkpoint))


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestChangeDetectionInit(unittest.TestCase):
    @patch("geoai.change_detection.AnyChange", None)
    def test_init_raises_without_torchange(self):
        from geoai.change_detection import ChangeDetection

        with self.assertRaises(ImportError) as ctx:
            ChangeDetection()
        self.assertIn("torchange", str(ctx.exception))

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_init_with_mocked_torchange(self, mock_anychange, mock_download):
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/checkpoint.pth"

        from geoai.change_detection import ChangeDetection

        detector = ChangeDetection()
        self.assertIsNotNone(detector.model)
        mock_anychange.assert_called_once()
        mock_model.make_mask_generator.assert_called_once()
        mock_model.set_hyperparameters.assert_called_once()

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_init_with_custom_checkpoint(self, mock_anychange, mock_download):
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model

        from geoai.change_detection import ChangeDetection

        detector = ChangeDetection(sam_checkpoint="/custom/ckpt.pth")
        self.assertEqual(detector.sam_checkpoint, "/custom/ckpt.pth")
        mock_download.assert_not_called()

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_init_model_types(self, mock_anychange, mock_download):
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        for model_type in ["vit_h", "vit_l", "vit_b"]:
            detector = ChangeDetection(sam_model_type=model_type)
            self.assertEqual(detector.sam_model_type, model_type)


# ---------------------------------------------------------------------------
# set_hyperparameters / set_mask_generator_params
# ---------------------------------------------------------------------------


class TestSetHyperparameters(unittest.TestCase):
    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def setUp(self, mock_anychange, mock_download):
        self.mock_model = MagicMock()
        mock_anychange.return_value = self.mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        self.detector = ChangeDetection()

    def test_set_hyperparameters_delegates(self):
        self.detector.set_hyperparameters(change_confidence_threshold=200)
        # Called during init + explicit call
        self.assertGreaterEqual(self.mock_model.set_hyperparameters.call_count, 2)

    def test_set_hyperparameters_passes_kwargs(self):
        self.detector.set_hyperparameters(
            change_confidence_threshold=180,
            auto_threshold=True,
            bitemporal_match=False,
        )
        call_kwargs = self.mock_model.set_hyperparameters.call_args[1]
        self.assertEqual(call_kwargs["change_confidence_threshold"], 180)
        self.assertTrue(call_kwargs["auto_threshold"])
        self.assertFalse(call_kwargs["bitemporal_match"])

    def test_set_hyperparameters_noop_without_model(self):
        self.detector.model = None
        # Should not raise
        self.detector.set_hyperparameters()

    def test_set_mask_generator_params_delegates(self):
        self.detector.set_mask_generator_params(
            points_per_side=16, stability_score_thresh=0.9
        )
        self.assertGreaterEqual(self.mock_model.make_mask_generator.call_count, 2)

    def test_set_mask_generator_params_noop_without_model(self):
        self.detector.model = None
        self.detector.set_mask_generator_params()


class TestSetHyperparametersPreservesUnspecified(unittest.TestCase):
    """Regression tests for issue #844: set_hyperparameters() must not reset
    parameters the caller did not pass."""

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange", _FakeAnyChange)
    def setUp(self, mock_download):
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        self.detector = ChangeDetection()

    def test_init_defaults_applied(self):
        self.assertEqual(self.detector.model.change_confidence_threshold, 145)
        self.assertTrue(self.detector.model.use_normalized_feature)
        self.assertTrue(self.detector.model.use_bitemporal_match)

    def test_updating_one_param_preserves_others(self):
        self.detector.set_hyperparameters(area_thresh=0.9)
        self.assertEqual(self.detector.model.area_thresh, 0.9)
        # Threshold must stay at the init value (145), not reset to
        # torchange's default (155).
        self.assertEqual(self.detector.model.change_confidence_threshold, 145)
        self.assertEqual(self.detector.model.object_sim_thresh, 60)
        self.assertTrue(self.detector.model.use_bitemporal_match)

    def test_no_args_is_a_noop(self):
        self.detector.set_hyperparameters(change_confidence_threshold=170)
        self.detector.set_hyperparameters()
        self.assertEqual(self.detector.model.change_confidence_threshold, 170)

    def test_noop_without_model_even_with_args(self):
        self.detector.model = None
        # Should not raise
        self.detector.set_hyperparameters(area_thresh=0.9)

    def test_bitemporal_match_maps_to_model_attribute(self):
        self.detector.set_hyperparameters(bitemporal_match=False)
        self.assertFalse(self.detector.model.use_bitemporal_match)
        self.detector.set_hyperparameters(match_hist=True)
        self.assertFalse(self.detector.model.use_bitemporal_match)

    def test_false_and_zero_values_are_forwarded(self):
        self.detector.set_hyperparameters(
            use_normalized_feature=False, object_sim_thresh=0
        )
        self.assertFalse(self.detector.model.use_normalized_feature)
        self.assertEqual(self.detector.model.object_sim_thresh, 0)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestChangeDetectionSignatures(unittest.TestCase):
    def test_change_detection_init_params(self):
        from geoai.change_detection import ChangeDetection

        sig = inspect.signature(ChangeDetection.__init__)
        self.assertIn("sam_model_type", sig.parameters)
        self.assertIn("sam_checkpoint", sig.parameters)

    def test_set_hyperparameters_params(self):
        from geoai.change_detection import ChangeDetection

        sig = inspect.signature(ChangeDetection.set_hyperparameters)
        expected = [
            "change_confidence_threshold",
            "auto_threshold",
            "use_normalized_feature",
            "area_thresh",
            "match_hist",
            "object_sim_thresh",
            "bitemporal_match",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_set_mask_generator_params_signature(self):
        from geoai.change_detection import ChangeDetection

        sig = inspect.signature(ChangeDetection.set_mask_generator_params)
        expected = [
            "points_per_side",
            "points_per_batch",
            "pred_iou_thresh",
            "stability_score_thresh",
            "box_nms_thresh",
            "min_mask_region_area",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_detect_changes_params(self):
        from geoai.change_detection import ChangeDetection

        sig = inspect.signature(ChangeDetection.detect_changes)
        expected = [
            "image1_path",
            "image2_path",
            "output_path",
            "target_size",
            "return_results",
            "export_probability",
            "return_detailed_results",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters)


# ---------------------------------------------------------------------------
# _read_and_align_images
# ---------------------------------------------------------------------------


class TestReadAndAlignImages(unittest.TestCase):
    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def setUp(self, mock_anychange, mock_download):
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        self.detector = ChangeDetection()

    def test_read_and_align_overlapping_images(self):
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1, width=64, height=64, bands=3)
            _create_test_geotiff(path2, width=64, height=64, bands=3)

            img1, img2, transform, crs, orig_shape = (
                self.detector._read_and_align_images(path1, path2, target_size=32)
            )
            self.assertEqual(img1.shape, (32, 32, 3))
            self.assertEqual(img2.shape, (32, 32, 3))
            self.assertIsNotNone(transform)
            self.assertIsNotNone(crs)

    def test_read_and_align_no_resize_needed(self):
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1, width=32, height=32, bands=3)
            _create_test_geotiff(path2, width=32, height=32, bands=3)

            img1, img2, _, _, _ = self.detector._read_and_align_images(
                path1, path2, target_size=32
            )
            # Shape should be roughly target_size (may differ slightly due to alignment)
            self.assertEqual(img1.ndim, 3)

    def test_read_and_align_non_overlapping_raises(self):
        """Images with non-overlapping bounds should raise ValueError."""
        import rasterio
        from rasterio.transform import from_bounds

        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")

            # First image at one location
            data = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
            t1 = from_bounds(0, 0, 1, 1, 32, 32)
            with rasterio.open(
                path1,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=t1,
            ) as dst:
                dst.write(data)

            # Second image at distant location
            t2 = from_bounds(100, 100, 101, 101, 32, 32)
            with rasterio.open(
                path2,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=t2,
            ) as dst:
                dst.write(data)

            with self.assertRaises(ValueError) as ctx:
                self.detector._read_and_align_images(path1, path2)
            self.assertIn("overlap", str(ctx.exception).lower())

    def test_read_and_align_multiband_truncation(self):
        """Images with >3 bands should be truncated to RGB."""
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1, width=32, height=32, bands=6)
            _create_test_geotiff(path2, width=32, height=32, bands=6)

            img1, img2, _, _, _ = self.detector._read_and_align_images(
                path1, path2, target_size=32
            )
            self.assertEqual(img1.shape[2], 3)
            self.assertEqual(img2.shape[2], 3)

    def test_read_and_align_float_images(self):
        """Float dtype images should be normalized to uint8."""
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1, width=32, height=32, bands=3, dtype="float32")
            _create_test_geotiff(path2, width=32, height=32, bands=3, dtype="float32")

            img1, img2, _, _, _ = self.detector._read_and_align_images(
                path1, path2, target_size=32
            )
            self.assertEqual(img1.dtype, np.uint8)
            self.assertEqual(img2.dtype, np.uint8)


# ---------------------------------------------------------------------------
# detect_changes
# ---------------------------------------------------------------------------


class TestDetectChanges(unittest.TestCase):
    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def setUp(self, mock_anychange, mock_download):
        self.mock_model = MagicMock()
        mock_anychange.return_value = self.mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        self.detector = ChangeDetection()

    def test_detect_changes_returns_tuple(self):
        """With return_results=True, should return (masks, img1, img2)."""
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1)
            _create_test_geotiff(path2)

            mock_masks = MagicMock()
            self.mock_model.forward.return_value = (mock_masks, None, None)

            result = self.detector.detect_changes(path1, path2, return_results=True)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)

    def test_detect_changes_returns_none(self):
        """With return_results=False and no detail, should return None."""
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1)
            _create_test_geotiff(path2)

            mock_masks = MagicMock()
            self.mock_model.forward.return_value = (mock_masks, None, None)

            result = self.detector.detect_changes(
                path1, path2, return_results=False, return_detailed_results=False
            )
            self.assertIsNone(result)

    def test_detect_changes_export_probability_requires_path(self):
        """export_probability=True without path should raise ValueError."""
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1)
            _create_test_geotiff(path2)

            mock_masks = MagicMock()
            self.mock_model.forward.return_value = (mock_masks, None, None)

            with self.assertRaises(ValueError):
                self.detector.detect_changes(path1, path2, export_probability=True)

    def test_detect_changes_export_instances_requires_path(self):
        """export_instance_masks=True without path should raise ValueError."""
        with tempfile.TemporaryDirectory() as td:
            path1 = os.path.join(td, "img1.tif")
            path2 = os.path.join(td, "img2.tif")
            _create_test_geotiff(path1)
            _create_test_geotiff(path2)

            mock_masks = MagicMock()
            self.mock_model.forward.return_value = (mock_masks, None, None)

            with self.assertRaises(ValueError):
                self.detector.detect_changes(path1, path2, export_instance_masks=True)


# ---------------------------------------------------------------------------
# Comprehensive report edge cases
# ---------------------------------------------------------------------------


class TestComprehensiveReport(unittest.TestCase):
    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def setUp(self, mock_anychange, mock_download):
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        self.detector = ChangeDetection()

    def test_report_with_empty_results(self):
        """Should handle None/empty results gracefully."""
        # Should not raise
        self.detector.create_comprehensive_report(None)
        self.detector.create_comprehensive_report({})
        self.detector.create_comprehensive_report({"other_key": 42})


# ---------------------------------------------------------------------------
# Change confidence normalization (issue #840)
# ---------------------------------------------------------------------------


class TestNormalizeChangeConfidence(unittest.TestCase):
    """Raw change confidence is a negative cosine similarity in [-1, 1]."""

    def test_threshold_maps_to_half(self):
        import math

        from geoai.change_detection import _normalize_change_confidence

        for threshold in [60.0, 145.0, 155.0]:
            threshold_cos = math.cos(math.radians(threshold))
            self.assertAlmostEqual(
                _normalize_change_confidence(threshold_cos, threshold), 0.5
            )

    def test_endpoints(self):
        from geoai.change_detection import _normalize_change_confidence

        self.assertAlmostEqual(_normalize_change_confidence(-1.0, 145.0), 0.0)
        self.assertAlmostEqual(_normalize_change_confidence(1.0, 145.0), 1.0)

    def test_monotonic_over_full_range(self):
        from geoai.change_detection import _normalize_change_confidence

        values = [
            _normalize_change_confidence(c, 145.0) for c in np.linspace(-1, 1, 41)
        ]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))

    def test_kept_masks_score_above_half(self):
        """Masks surviving torchange's filter (conf > cos(threshold)) map above 0.5."""
        from geoai.change_detection import _normalize_change_confidence

        # cos(145 degrees) is about -0.819; kept masks exceed it
        for conf in [-0.8, -0.5, 0.0, 0.5, 0.9]:
            self.assertGreater(_normalize_change_confidence(conf, 145.0), 0.5)

    def test_out_of_range_input_clamped(self):
        from geoai.change_detection import _normalize_change_confidence

        self.assertAlmostEqual(_normalize_change_confidence(-2.0, 145.0), 0.0)
        self.assertAlmostEqual(_normalize_change_confidence(2.0, 145.0), 1.0)

    def test_degenerate_threshold_angles(self):
        """Threshold angles at 0/180 degrees must not divide by zero."""
        from geoai.change_detection import _normalize_change_confidence

        for threshold in [0.0, 180.0]:
            for conf in [-1.0, 0.0, 1.0]:
                value = _normalize_change_confidence(conf, threshold)
                self.assertTrue(0.0 <= value <= 1.0)


class TestGetChangeConfidenceThreshold(unittest.TestCase):
    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_reads_threshold_from_model(self, mock_anychange, mock_download):
        mock_model = MagicMock()
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        detector = ChangeDetection()
        detector.model.change_confidence_threshold = 155
        self.assertEqual(detector._get_change_confidence_threshold(), 155.0)

    @patch("geoai.change_detection.download_checkpoint")
    @patch("geoai.change_detection.AnyChange")
    def test_falls_back_to_default(self, mock_anychange, mock_download):
        mock_model = MagicMock(spec=[])  # no change_confidence_threshold attribute
        mock_anychange.return_value = mock_model
        mock_download.return_value = "/fake/ckpt.pth"

        from geoai.change_detection import ChangeDetection

        detector = ChangeDetection.__new__(ChangeDetection)
        detector.model = mock_model
        self.assertEqual(detector._get_change_confidence_threshold(), 145.0)


if __name__ == "__main__":
    unittest.main()
