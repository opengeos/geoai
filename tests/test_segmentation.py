#!/usr/bin/env python

"""Tests for `geoai.segmentation` module."""

import inspect
import unittest

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


@unittest.skipUnless(HAS_SEGMENTATION, "geoai.segmentation dependencies not available")
class TestSegmentationGetTransform(unittest.TestCase):
    """Tests for get_transform function behavior."""

    def test_get_transform_returns_compose(self):
        """Test that get_transform returns an albumentations Compose object."""
        from geoai.segmentation import get_transform

        result = get_transform()
        import albumentations as A

        self.assertIsInstance(result, A.Compose)


if __name__ == "__main__":
    unittest.main()
