#!/usr/bin/env python

"""Tests for `geoai.timm_segment` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestTimmSegmentImport(unittest.TestCase):
    """Tests for timm_segment module import behavior."""

    def test_module_imports(self):
        """Test that the timm_segment module can be imported."""
        import geoai.timm_segment

        self.assertTrue(hasattr(geoai.timm_segment, "TimmSegmentationModel"))

    def test_timm_segmentation_model_exists(self):
        """Test that TimmSegmentationModel class exists."""
        from geoai.timm_segment import TimmSegmentationModel

        self.assertTrue(callable(TimmSegmentationModel))

    def test_segmentation_dataset_exists(self):
        """Test that SegmentationDataset class exists."""
        from geoai.timm_segment import SegmentationDataset

        self.assertTrue(callable(SegmentationDataset))

    def test_train_timm_segmentation_exists(self):
        """Test that train_timm_segmentation function exists."""
        from geoai.timm_segment import train_timm_segmentation

        self.assertTrue(callable(train_timm_segmentation))

    def test_predict_segmentation_exists(self):
        """Test that predict_segmentation function exists."""
        from geoai.timm_segment import predict_segmentation

        self.assertTrue(callable(predict_segmentation))

    def test_train_timm_segmentation_model_exists(self):
        """Test that train_timm_segmentation_model function exists."""
        from geoai.timm_segment import train_timm_segmentation_model

        self.assertTrue(callable(train_timm_segmentation_model))

    def test_timm_semantic_segmentation_exists(self):
        """Test that timm_semantic_segmentation function exists."""
        from geoai.timm_segment import timm_semantic_segmentation

        self.assertTrue(callable(timm_semantic_segmentation))

    def test_push_timm_model_to_hub_exists(self):
        """Test that push_timm_model_to_hub function exists."""
        from geoai.timm_segment import push_timm_model_to_hub

        self.assertTrue(callable(push_timm_model_to_hub))

    def test_timm_segmentation_from_hub_exists(self):
        """Test that timm_segmentation_from_hub function exists."""
        from geoai.timm_segment import timm_segmentation_from_hub

        self.assertTrue(callable(timm_segmentation_from_hub))


class TestTimmSegmentInitParams(unittest.TestCase):
    """Tests for TimmSegmentationModel class initialization parameters."""

    def test_init_params(self):
        """Test TimmSegmentationModel.__init__ has expected parameters."""
        from geoai.timm_segment import TimmSegmentationModel

        sig = inspect.signature(TimmSegmentationModel.__init__)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("in_channels", sig.parameters)
        self.assertIn("encoder_weights", sig.parameters)
        self.assertIn("learning_rate", sig.parameters)
        self.assertIn("weight_decay", sig.parameters)
        self.assertIn("freeze_encoder", sig.parameters)
        self.assertIn("loss_fn", sig.parameters)
        self.assertIn("class_weights", sig.parameters)
        self.assertIn("use_timm_model", sig.parameters)
        self.assertIn("timm_model_name", sig.parameters)

    def test_init_default_values(self):
        """Test TimmSegmentationModel.__init__ default parameter values."""
        from geoai.timm_segment import TimmSegmentationModel

        sig = inspect.signature(TimmSegmentationModel.__init__)
        self.assertEqual(sig.parameters["encoder_name"].default, "resnet50")
        self.assertEqual(sig.parameters["architecture"].default, "unet")
        self.assertEqual(sig.parameters["num_classes"].default, 2)
        self.assertEqual(sig.parameters["in_channels"].default, 3)
        self.assertEqual(sig.parameters["encoder_weights"].default, "imagenet")
        self.assertEqual(sig.parameters["learning_rate"].default, 1e-3)
        self.assertEqual(sig.parameters["freeze_encoder"].default, False)
        self.assertEqual(sig.parameters["use_timm_model"].default, False)


class TestTimmSegmentSignatures(unittest.TestCase):
    """Tests for timm_segment function signatures."""

    def test_train_timm_segmentation_params(self):
        """Test train_timm_segmentation has expected parameters."""
        from geoai.timm_segment import train_timm_segmentation

        sig = inspect.signature(train_timm_segmentation)
        self.assertIn("train_dataset", sig.parameters)
        self.assertIn("val_dataset", sig.parameters)
        self.assertIn("test_dataset", sig.parameters)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("in_channels", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("num_epochs", sig.parameters)
        self.assertIn("learning_rate", sig.parameters)
        self.assertIn("patience", sig.parameters)
        self.assertIn("use_timm_model", sig.parameters)
        self.assertIn("timm_model_name", sig.parameters)

    def test_predict_segmentation_params(self):
        """Test predict_segmentation has expected parameters."""
        from geoai.timm_segment import predict_segmentation

        sig = inspect.signature(predict_segmentation)
        self.assertIn("model", sig.parameters)
        self.assertIn("image_paths", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("num_workers", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("return_probabilities", sig.parameters)

    def test_timm_semantic_segmentation_params(self):
        """Test timm_semantic_segmentation has expected parameters."""
        from geoai.timm_segment import timm_semantic_segmentation

        sig = inspect.signature(timm_semantic_segmentation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("num_channels", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("window_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)

    def test_push_timm_model_to_hub_params(self):
        """Test push_timm_model_to_hub has expected parameters."""
        from geoai.timm_segment import push_timm_model_to_hub

        sig = inspect.signature(push_timm_model_to_hub)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("repo_id", sig.parameters)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("num_channels", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("commit_message", sig.parameters)
        self.assertIn("private", sig.parameters)
        self.assertIn("token", sig.parameters)

    def test_timm_segmentation_from_hub_params(self):
        """Test timm_segmentation_from_hub has expected parameters."""
        from geoai.timm_segment import timm_segmentation_from_hub

        sig = inspect.signature(timm_segmentation_from_hub)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("repo_id", sig.parameters)
        self.assertIn("window_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("token", sig.parameters)

    def test_train_timm_segmentation_model_params(self):
        """Test train_timm_segmentation_model has expected parameters."""
        from geoai.timm_segment import train_timm_segmentation_model

        sig = inspect.signature(train_timm_segmentation_model)
        self.assertIn("images_dir", sig.parameters)
        self.assertIn("labels_dir", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("input_format", sig.parameters)
        self.assertIn("encoder_name", sig.parameters)
        self.assertIn("architecture", sig.parameters)
        self.assertIn("num_channels", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("val_split", sig.parameters)


class TestTimmSegmentAvailabilityFlags(unittest.TestCase):
    """Tests for optional dependency availability flags."""

    def test_timm_available_flag_exists(self):
        """Test that TIMM_AVAILABLE flag is defined."""
        from geoai.timm_segment import TIMM_AVAILABLE

        self.assertIsInstance(TIMM_AVAILABLE, bool)

    def test_smp_available_flag_exists(self):
        """Test that SMP_AVAILABLE flag is defined."""
        from geoai.timm_segment import SMP_AVAILABLE

        self.assertIsInstance(SMP_AVAILABLE, bool)

    def test_lightning_available_flag_exists(self):
        """Test that LIGHTNING_AVAILABLE flag is defined."""
        from geoai.timm_segment import LIGHTNING_AVAILABLE

        self.assertIsInstance(LIGHTNING_AVAILABLE, bool)


class TestSegmentationDatasetInit(unittest.TestCase):
    """Tests for SegmentationDataset initialization."""

    def test_segmentation_dataset_init_params(self):
        """Test SegmentationDataset.__init__ has expected parameters."""
        from geoai.timm_segment import SegmentationDataset

        sig = inspect.signature(SegmentationDataset.__init__)
        self.assertIn("image_paths", sig.parameters)
        self.assertIn("mask_paths", sig.parameters)
        self.assertIn("transform", sig.parameters)
        self.assertIn("num_channels", sig.parameters)

    def test_segmentation_dataset_mismatched_lengths(self):
        """Test that SegmentationDataset raises ValueError for mismatched lengths."""
        from geoai.timm_segment import SegmentationDataset

        with self.assertRaises(ValueError):
            SegmentationDataset(
                image_paths=["a.tif", "b.tif"],
                mask_paths=["a.tif"],
            )


if __name__ == "__main__":
    unittest.main()
