#!/usr/bin/env python

"""Tests for metric functions in `geoai.train` module."""

import unittest

import torch

from geoai.train import f1_score, iou_coefficient, precision_score, recall_score


class TestF1Score(unittest.TestCase):
    """Tests for the f1_score function."""

    def test_binary_hw_format(self):
        """Test F1 with binary [H, W] format tensors."""
        pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        score = f1_score(pred, target)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_multiclass_chw_format(self):
        """Test F1 with multi-class [C, H, W] logit format."""
        # 3 classes, 2x2 image
        pred = torch.randn(3, 2, 2)
        target = torch.tensor([[0, 1], [2, 0]])
        score = f1_score(pred, target, num_classes=3)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_perfect_prediction(self):
        """Test F1 score with perfect prediction gives ~1.0."""
        target = torch.tensor([[0, 1, 1], [1, 0, 0]])
        score = f1_score(target, target)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_invalid_dimensions_raises(self):
        """Test that invalid tensor dimensions raise ValueError."""
        pred = torch.randn(2, 3, 4, 4)  # 4D tensor
        target = torch.tensor([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            f1_score(pred, target)


class TestIouCoefficient(unittest.TestCase):
    """Tests for the iou_coefficient function."""

    def test_binary_iou(self):
        """Test IoU with binary [H, W] tensors."""
        pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        score = iou_coefficient(pred, target)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_perfect_match(self):
        """Test IoU with identical prediction and target gives ~1.0."""
        target = torch.tensor([[0, 1, 2], [1, 0, 2]])
        score = iou_coefficient(target, target, num_classes=3)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_multiclass_iou(self):
        """Test IoU with multi-class [C, H, W] logit input."""
        pred = torch.randn(3, 4, 4)
        target = torch.randint(0, 3, (4, 4))
        score = iou_coefficient(pred, target, num_classes=3)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_zero_overlap(self):
        """Test IoU when prediction and target have no overlap in positive class."""
        pred = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        score = iou_coefficient(pred, target)
        # Both classes exist but are swapped; IoU should be 0 for each class
        self.assertAlmostEqual(score, 0.0, places=3)


class TestPrecisionScore(unittest.TestCase):
    """Tests for the precision_score function."""

    def test_binary_precision(self):
        """Test precision with binary tensors."""
        pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        score = precision_score(pred, target)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_perfect_precision(self):
        """Test precision with perfect prediction gives ~1.0."""
        target = torch.tensor([[0, 1, 1], [1, 0, 0]])
        score = precision_score(target, target)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_multiclass_precision(self):
        """Test precision with multi-class logit input."""
        pred = torch.randn(3, 4, 4)
        target = torch.randint(0, 3, (4, 4))
        score = precision_score(pred, target, num_classes=3)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)


class TestRecallScore(unittest.TestCase):
    """Tests for the recall_score function."""

    def test_binary_recall(self):
        """Test recall with binary tensors."""
        pred = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        target = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        score = recall_score(pred, target)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_perfect_recall(self):
        """Test recall with perfect prediction gives ~1.0."""
        target = torch.tensor([[0, 1, 1], [1, 0, 0]])
        score = recall_score(target, target)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_multiclass_recall(self):
        """Test recall with multi-class logit input."""
        pred = torch.randn(3, 4, 4)
        target = torch.randint(0, 3, (4, 4))
        score = recall_score(pred, target, num_classes=3)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_invalid_dimensions_raises(self):
        """Test that 4D tensor raises ValueError."""
        pred = torch.randn(1, 3, 4, 4)
        target = torch.randint(0, 3, (4, 4))
        with self.assertRaises(ValueError):
            recall_score(pred, target)


if __name__ == "__main__":
    unittest.main()
