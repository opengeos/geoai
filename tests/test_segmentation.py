#!/usr/bin/env python

"""Tests for `geoai.segmentation` module."""

import inspect
import unittest

import numpy as np

try:
    import geoai.segmentation

    HAS_SEGMENTATION = True
except ImportError:
    HAS_SEGMENTATION = False


@unittest.skipUnless(HAS_SEGMENTATION, "geoai.segmentation dependencies not available")
class TestSegmentationImport(unittest.TestCase):
    """Tests for segmentation module import behavior."""

    def test_module_imports(self):
        """Test that the segmentation module can be imported."""
        import geoai.segmentation

        self.assertTrue(hasattr(geoai.segmentation, "CustomDataset"))
        self.assertTrue(hasattr(geoai.segmentation, "get_transform"))

    def test_custom_dataset_exists(self):
        """Test that CustomDataset class exists and is callable."""
        from geoai.segmentation import CustomDataset

        self.assertTrue(callable(CustomDataset))

    def test_get_transform_exists(self):
        """Test that get_transform function exists and is callable."""
        from geoai.segmentation import get_transform

        self.assertTrue(callable(get_transform))

    def test_prepare_datasets_exists(self):
        """Test that prepare_datasets function exists and is callable."""
        from geoai.segmentation import prepare_datasets

        self.assertTrue(callable(prepare_datasets))

    def test_train_model_exists(self):
        """Test that train_model function exists and is callable."""
        from geoai.segmentation import train_model

        self.assertTrue(callable(train_model))


@unittest.skipUnless(HAS_SEGMENTATION, "geoai.segmentation dependencies not available")
class TestSegmentationSignatures(unittest.TestCase):
    """Tests for segmentation function signatures."""

    def test_custom_dataset_init_params(self):
        """Test CustomDataset.__init__ has expected parameters."""
        from geoai.segmentation import CustomDataset

        sig = inspect.signature(CustomDataset.__init__)
        self.assertIn("images_dir", sig.parameters)
        self.assertIn("masks_dir", sig.parameters)
        self.assertIn("transform", sig.parameters)
        self.assertIn("target_size", sig.parameters)
        self.assertIn("num_classes", sig.parameters)

    def test_prepare_datasets_params(self):
        """Test prepare_datasets has expected parameters."""
        from geoai.segmentation import prepare_datasets

        sig = inspect.signature(prepare_datasets)
        self.assertIn("images_dir", sig.parameters)
        self.assertIn("masks_dir", sig.parameters)
        self.assertIn("transform", sig.parameters)
        self.assertIn("test_size", sig.parameters)
        self.assertIn("random_state", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertEqual(sig.parameters["num_classes"].default, 2)

    def test_train_model_params(self):
        """Test train_model has expected parameters."""
        from geoai.segmentation import train_model

        sig = inspect.signature(train_model)
        self.assertIn("train_dataset", sig.parameters)
        self.assertIn("val_dataset", sig.parameters)
        self.assertIn("pretrained_model", sig.parameters)
        self.assertIn("model_save_path", sig.parameters)
        self.assertIn("num_epochs", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("learning_rate", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertEqual(sig.parameters["num_classes"].default, 2)

    def test_segment_image_params(self):
        """Test segment_image has expected parameters."""
        from geoai.segmentation import segment_image

        sig = inspect.signature(segment_image)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("target_size", sig.parameters)
        self.assertIn("device", sig.parameters)

    def test_visualize_predictions_params(self):
        """Test visualize_predictions has expected parameters."""
        from geoai.segmentation import visualize_predictions

        sig = inspect.signature(visualize_predictions)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("segmented_mask", sig.parameters)
        self.assertIn("target_size", sig.parameters)
        self.assertIn("reference_image_path", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertEqual(sig.parameters["num_classes"].default, 2)


@unittest.skipUnless(HAS_SEGMENTATION, "geoai.segmentation dependencies not available")
class TestSegmentationGetTransform(unittest.TestCase):
    """Tests for get_transform function behavior."""

    def test_get_transform_returns_compose(self):
        """Test that get_transform returns an albumentations Compose object."""
        from geoai.segmentation import get_transform

        result = get_transform()
        import albumentations as A

        self.assertIsInstance(result, A.Compose)


class TestNormalizeMask(unittest.TestCase):
    """Tests for _normalize_mask helper function.

    These tests exercise pure-numpy logic and do not require torch or
    other heavy dependencies.
    """

    def _get_normalize_mask(self):
        """Import ``_normalize_mask`` or skip the test if unavailable."""
        try:
            from geoai.segmentation import _normalize_mask

            return _normalize_mask
        except ImportError:
            self.skipTest("geoai.segmentation._normalize_mask not available")

    def test_binary_mask_255_to_01(self):
        """Binary mask with 0/255 values maps to 0/1."""
        _normalize_mask = self._get_normalize_mask()

        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = _normalize_mask(mask, num_classes=2)
        np.testing.assert_array_equal(result, [[0, 1], [1, 0]])
        self.assertEqual(result.dtype, np.int64)

    def test_binary_mask_already_01(self):
        """Binary mask already [0, 1] is unchanged."""
        _normalize_mask = self._get_normalize_mask()
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = _normalize_mask(mask, num_classes=2)
        np.testing.assert_array_equal(result, [[0, 1], [1, 0]])

    def test_binary_mask_mixed_nonzero(self):
        """Binary: any nonzero value becomes 1."""
        _normalize_mask = self._get_normalize_mask()
        mask = np.array([[0, 50], [127, 200]], dtype=np.uint8)
        result = _normalize_mask(mask, num_classes=2)
        np.testing.assert_array_equal(result, [[0, 1], [1, 1]])

    def test_multiclass_within_range(self):
        """Multi-class mask within range is unchanged."""
        _normalize_mask = self._get_normalize_mask()
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = _normalize_mask(mask, num_classes=4)
        np.testing.assert_array_equal(result, [[0, 1], [2, 3]])

    def test_multiclass_clips_out_of_range(self):
        """Multi-class mask with out-of-range values is clipped."""
        _normalize_mask = self._get_normalize_mask()
        mask = np.array([[0, 5], [10, 255]], dtype=np.uint8)
        result = _normalize_mask(mask, num_classes=4)
        np.testing.assert_array_equal(result, [[0, 3], [3, 3]])

    def test_multiclass_preserves_zero(self):
        """Multi-class: background class 0 is preserved."""
        _normalize_mask = self._get_normalize_mask()
        mask = np.zeros((4, 4), dtype=np.uint8)
        result = _normalize_mask(mask, num_classes=5)
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.int64))

    def test_output_dtype_is_int64(self):
        """Output dtype is always int64 regardless of num_classes."""
        _normalize_mask = self._get_normalize_mask()
        for num_classes in [2, 5, 10]:
            mask = np.array([[0, 1]], dtype=np.uint8)
            result = _normalize_mask(mask, num_classes)
            self.assertEqual(result.dtype, np.int64)

    def test_num_classes_below_2_raises(self):
        """num_classes < 2 raises ValueError."""
        _normalize_mask = self._get_normalize_mask()
        mask = np.array([[0, 1]], dtype=np.uint8)
        for bad_val in [0, 1, -1]:
            with self.assertRaises(ValueError):
                _normalize_mask(mask, num_classes=bad_val)


class TestVisualizationScaling(unittest.TestCase):
    """Tests for multi-class mask visualization scaling logic.

    These tests exercise pure-numpy scaling and do not require torch.
    """

    def test_binary_mask_scales_to_255(self):
        """Binary mask [0, 1] scales to [0, 255]."""
        mask = np.array([[0, 1], [1, 0]], dtype=np.int64)
        result = (mask * 255).astype(np.uint8)
        self.assertEqual(result.max(), 255)
        self.assertEqual(result.min(), 0)

    def test_multiclass_mask_no_overflow(self):
        """Multi-class mask scales without uint8 overflow."""
        mask = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        num_classes = 5
        max_val = max(num_classes - 1, 1)
        scaled = (mask.astype(np.float64) / max_val * 255).astype(np.uint8)
        self.assertEqual(scaled[0, 0], 0)
        self.assertEqual(scaled[0, 4], 255)
        self.assertTrue((scaled >= 0).all())
        self.assertTrue((scaled <= 255).all())


if __name__ == "__main__":
    unittest.main()
