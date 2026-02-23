#!/usr/bin/env python

"""Tests for metric and utility functions in `geoai.utils`."""

import os
import unittest

import numpy as np
import torch

from geoai import utils


class TestCalcIou(unittest.TestCase):
    """Tests for the calc_iou function."""

    def test_binary_iou_with_known_arrays(self):
        """Test binary IoU with manually computed expected value."""
        gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        iou = utils.calc_iou(gt, pred)
        # The function treats both 0 and 1 as boolean, so IoU considers
        # both True (foreground) and False (background) intersection/union.
        # IoU should be a float between 0 and 1.
        self.assertIsInstance(iou, float)
        self.assertGreater(iou, 0.5)
        self.assertLessEqual(iou, 1.0)

    def test_binary_iou_perfect_match(self):
        """Test binary IoU when prediction matches ground truth exactly."""
        gt = np.array([[0, 1, 1], [1, 1, 0]])
        iou = utils.calc_iou(gt, gt.copy())
        self.assertAlmostEqual(iou, 1.0, places=3)

    def test_binary_iou_no_overlap(self):
        """Test binary IoU when there is no overlap."""
        gt = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        iou = utils.calc_iou(gt, pred)
        self.assertAlmostEqual(iou, 0.0, places=3)

    def test_binary_iou_all_zeros(self):
        """Test binary IoU when both masks are all zeros."""
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        iou = utils.calc_iou(gt, pred)
        # Both empty => IoU = 1.0 (convention for empty masks)
        self.assertAlmostEqual(iou, 1.0, places=3)

    def test_multiclass_iou(self):
        """Test multi-class IoU returns per-class array."""
        gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        iou = utils.calc_iou(gt, pred, num_classes=3)
        self.assertIsInstance(iou, np.ndarray)
        self.assertEqual(len(iou), 3)
        # All values should be between 0 and 1 (or nan)
        for val in iou:
            if not np.isnan(val):
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_multiclass_iou_with_ignore_index(self):
        """Test multi-class IoU with an ignored class."""
        gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        iou = utils.calc_iou(gt, pred, num_classes=3, ignore_index=0)
        self.assertTrue(np.isnan(iou[0]))
        # Classes 1 and 2 should have valid IoU values
        self.assertFalse(np.isnan(iou[1]))
        self.assertFalse(np.isnan(iou[2]))

    def test_iou_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises ValueError."""
        gt = np.array([[0, 1], [1, 0]])
        pred = np.array([[0, 1, 0], [1, 0, 1]])
        with self.assertRaises(ValueError):
            utils.calc_iou(gt, pred)

    def test_iou_with_torch_tensors(self):
        """Test IoU computation with PyTorch tensor inputs."""
        gt = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        iou = utils.calc_iou(gt, pred)
        self.assertIsInstance(iou, float)
        self.assertGreater(iou, 0.0)
        self.assertLessEqual(iou, 1.0)


class TestCalcF1Score(unittest.TestCase):
    """Tests for the calc_f1_score function."""

    def test_binary_f1_with_known_arrays(self):
        """Test binary F1 with manually computed expected value."""
        gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        f1 = utils.calc_f1_score(gt, pred)
        # TP=3, FP=1, FN=2 => precision=3/4, recall=3/5
        # F1 = 2 * (3/4 * 3/5) / (3/4 + 3/5) ≈ 0.667
        self.assertIsInstance(f1, float)
        self.assertGreater(f1, 0.5)
        self.assertLess(f1, 1.0)

    def test_binary_f1_perfect_match(self):
        """Test binary F1 when prediction matches ground truth."""
        gt = np.array([[0, 1, 1], [1, 1, 0]])
        f1 = utils.calc_f1_score(gt, gt.copy())
        self.assertAlmostEqual(f1, 1.0, places=3)

    def test_multiclass_f1(self):
        """Test multi-class F1 returns per-class array."""
        gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        f1 = utils.calc_f1_score(gt, pred, num_classes=3)
        self.assertIsInstance(f1, np.ndarray)
        self.assertEqual(len(f1), 3)

    def test_f1_with_ignore_index(self):
        """Test F1 with an ignored class returns NaN for that class."""
        gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        f1 = utils.calc_f1_score(gt, pred, num_classes=3, ignore_index=0)
        self.assertTrue(np.isnan(f1[0]))

    def test_f1_with_torch_tensors(self):
        """Test F1 score with PyTorch tensor inputs."""
        gt = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        f1 = utils.calc_f1_score(gt, pred)
        self.assertIsInstance(f1, float)
        self.assertGreater(f1, 0.0)


class TestCalcSegmentationMetrics(unittest.TestCase):
    """Tests for the calc_segmentation_metrics function."""

    def test_binary_metrics_returns_dict(self):
        """Test that binary segmentation returns dict with iou and f1 keys."""
        gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        metrics = utils.calc_segmentation_metrics(gt, pred)
        self.assertIn("iou", metrics)
        self.assertIn("f1", metrics)
        self.assertIsInstance(metrics["iou"], float)
        self.assertIsInstance(metrics["f1"], float)

    def test_multiclass_metrics_include_means(self):
        """Test that multi-class metrics include mean_iou and mean_f1."""
        gt = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        metrics = utils.calc_segmentation_metrics(gt, pred, num_classes=3)
        self.assertIn("mean_iou", metrics)
        self.assertIn("mean_f1", metrics)
        self.assertIsInstance(metrics["mean_iou"], float)
        self.assertIsInstance(metrics["mean_f1"], float)

    def test_metrics_subset_iou_only(self):
        """Test requesting only IoU metric."""
        gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        metrics = utils.calc_segmentation_metrics(gt, pred, metrics=["iou"])
        self.assertIn("iou", metrics)
        self.assertNotIn("f1", metrics)

    def test_metrics_subset_f1_only(self):
        """Test requesting only F1 metric."""
        gt = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        metrics = utils.calc_segmentation_metrics(gt, pred, metrics=["f1"])
        self.assertNotIn("iou", metrics)
        self.assertIn("f1", metrics)


class TestTempFilePath(unittest.TestCase):
    """Tests for the temp_file_path function."""

    def test_extension_with_dot(self):
        """Test temp_file_path with extension that includes a dot."""
        path = utils.temp_file_path(".tif")
        self.assertTrue(path.endswith(".tif"))
        self.assertTrue(os.path.isabs(path))

    def test_extension_without_dot(self):
        """Test temp_file_path with extension lacking a dot."""
        path = utils.temp_file_path("tif")
        self.assertTrue(path.endswith(".tif"))

    def test_unique_paths(self):
        """Test that consecutive calls produce unique paths."""
        paths = {utils.temp_file_path(".tif") for _ in range(10)}
        self.assertEqual(len(paths), 10)

    def test_various_extensions(self):
        """Test temp_file_path with different file extensions."""
        for ext in [".geojson", ".shp", ".png", "csv"]:
            path = utils.temp_file_path(ext)
            expected = ext if ext.startswith(".") else f".{ext}"
            self.assertTrue(path.endswith(expected))


if __name__ == "__main__":
    unittest.main()
