#!/usr/bin/env python

"""Tests for ignore_index handling in `geoai.train` semantic segmentation."""

import unittest

import numpy as np
import torch

from geoai.train import (
    SemanticSegmentationDataset,
    f1_score,
    iou_coefficient,
    precision_score,
    recall_score,
)


class TestSemanticSegmentationDatasetIgnoreIndex(unittest.TestCase):
    """Tests for preserving ignored pixels in semantic segmentation labels."""

    def test_multiclass_normalization_preserves_negative_ignore_index(self):
        """Multi-class label clipping should not turn -100 into class 0."""
        dataset = SemanticSegmentationDataset(
            [],
            [],
            num_channels=3,
            num_classes=4,
            ignore_index=-100,
        )
        mask = np.array([[0, 1, -100], [5, 2, -100]], dtype=np.int16)

        result = dataset._normalize_label_mask(mask)

        np.testing.assert_array_equal(
            result,
            np.array([[0, 1, -100], [3, 2, -100]], dtype=np.int64),
        )

    def test_binary_normalization_preserves_positive_ignore_index(self):
        """Binary non-zero normalization should not turn 255 into foreground."""
        dataset = SemanticSegmentationDataset(
            [],
            [],
            num_channels=3,
            num_classes=2,
            ignore_index=255,
        )
        mask = np.array([[0, 255], [128, 1]], dtype=np.uint8)

        result = dataset._normalize_label_mask(mask)

        np.testing.assert_array_equal(
            result,
            np.array([[0, 255], [1, 1]], dtype=np.int64),
        )

    def test_resize_preserves_ignore_index(self):
        """Nearest-neighbor resizing should keep ignored pixels ignored."""
        dataset = SemanticSegmentationDataset(
            [],
            [],
            num_channels=3,
            target_size=(4, 4),
            resize_mode="resize",
            num_classes=3,
            ignore_index=-100,
        )
        image = torch.zeros((3, 2, 2), dtype=torch.float32)
        mask = torch.tensor([[0, -100], [2, 5]], dtype=torch.long)

        _, resized = dataset._resize_image_and_mask(image, mask)

        self.assertIn(-100, resized.tolist()[0])
        valid = resized != -100
        self.assertTrue(torch.all(resized[valid] >= 0))
        self.assertTrue(torch.all(resized[valid] <= 2))


class TestSemanticMetricsIgnoreIndex(unittest.TestCase):
    """Tests for excluding ignored pixels from semantic metrics."""

    def test_metrics_exclude_ignore_index_pixels(self):
        """Wrong predictions on ignored pixels should not reduce metrics."""
        pred = torch.tensor([[0, 0], [1, 1]])
        target = torch.tensor([[0, -100], [1, 1]])

        self.assertAlmostEqual(
            iou_coefficient(pred, target, num_classes=2, ignore_index=-100),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            f1_score(pred, target, num_classes=2, ignore_index=-100),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            precision_score(pred, target, num_classes=2, ignore_index=-100),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            recall_score(pred, target, num_classes=2, ignore_index=-100),
            1.0,
            places=6,
        )

    def test_metrics_return_zero_when_all_pixels_ignored(self):
        """All-ignored targets should produce a defined zero score for every metric."""
        pred = torch.tensor([[0, 1], [1, 0]])
        target = torch.full((2, 2), -100, dtype=torch.long)

        self.assertEqual(
            iou_coefficient(pred, target, num_classes=2, ignore_index=-100),
            0.0,
        )
        self.assertEqual(
            f1_score(pred, target, num_classes=2, ignore_index=-100),
            0.0,
        )
        self.assertEqual(
            precision_score(pred, target, num_classes=2, ignore_index=-100),
            0.0,
        )
        self.assertEqual(
            recall_score(pred, target, num_classes=2, ignore_index=-100),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
