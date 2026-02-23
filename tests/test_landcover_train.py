#!/usr/bin/env python

"""Tests for loss functions and IoU in `geoai.landcover_train` module."""

import unittest

import torch

from geoai.landcover_train import FocalLoss, LandcoverCrossEntropyLoss, landcover_iou


class TestFocalLoss(unittest.TestCase):
    """Tests for the FocalLoss class."""

    def setUp(self):
        """Set up small test tensors."""
        self.num_classes = 3
        self.batch_size = 2
        self.h, self.w = 4, 4
        # Logits: (N, C, H, W)
        self.inputs = torch.randn(self.batch_size, self.num_classes, self.h, self.w)
        # Targets: (N, H, W)
        self.targets = torch.randint(
            0, self.num_classes, (self.batch_size, self.h, self.w)
        )

    def test_forward_returns_scalar(self):
        """Test that forward pass returns a scalar loss with mean reduction."""
        loss_fn = FocalLoss()
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)

    def test_gamma_zero_approaches_ce(self):
        """Test that gamma=0 makes focal loss equivalent to CE loss."""
        focal = FocalLoss(alpha=1.0, gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()
        focal_val = focal(self.inputs, self.targets)
        ce_val = ce(self.inputs, self.targets)
        self.assertAlmostEqual(focal_val.item(), ce_val.item(), places=4)

    def test_reduction_sum(self):
        """Test sum reduction returns a larger value than mean."""
        loss_mean = FocalLoss(reduction="mean")(self.inputs, self.targets)
        loss_sum = FocalLoss(reduction="sum")(self.inputs, self.targets)
        # Sum should be >= mean for positive losses
        self.assertGreaterEqual(loss_sum.item(), loss_mean.item())

    def test_reduction_none(self):
        """Test none reduction returns per-element losses."""
        loss_fn = FocalLoss(reduction="none")
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.shape, (self.batch_size, self.h, self.w))

    def test_with_class_weights(self):
        """Test focal loss with per-class weights."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = FocalLoss(weight=weights)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_ignore_index(self):
        """Test focal loss ignores specified class index."""
        loss_fn = FocalLoss(ignore_index=0)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)


class TestLandcoverCrossEntropyLoss(unittest.TestCase):
    """Tests for the LandcoverCrossEntropyLoss class."""

    def setUp(self):
        """Set up test tensors."""
        self.num_classes = 3
        self.inputs = torch.randn(2, self.num_classes, 4, 4)
        self.targets = torch.randint(0, self.num_classes, (2, 4, 4))

    def test_forward_returns_scalar(self):
        """Test basic forward pass returns scalar loss with int ignore_index."""
        loss_fn = LandcoverCrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_ignore_index_int(self):
        """Test that integer ignore_index is preserved."""
        loss_fn = LandcoverCrossEntropyLoss(ignore_index=0)
        self.assertEqual(loss_fn.ignore_index, 0)

    def test_ignore_index_negative(self):
        """Test that negative ignore_index is preserved."""
        loss_fn = LandcoverCrossEntropyLoss(ignore_index=-100)
        self.assertEqual(loss_fn.ignore_index, -100)

    def test_with_class_weights(self):
        """Test loss with per-class weights and int ignore_index."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = LandcoverCrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)


class TestLandcoverIou(unittest.TestCase):
    """Tests for the landcover_iou function."""

    def setUp(self):
        """Set up small test tensors."""
        self.num_classes = 3
        # Predictions: (N, H, W) with class indices
        self.pred = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
        self.target = torch.tensor([[[0, 1, 2], [0, 1, 2]]])

    def test_mean_mode_perfect(self):
        """Test mean mode with perfect prediction returns ~1.0."""
        iou = landcover_iou(self.pred, self.target, self.num_classes, mode="mean")
        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 1.0, places=3)

    def test_mean_mode_imperfect(self):
        """Test mean mode with imperfect prediction returns value in [0, 1]."""
        pred = torch.tensor([[[1, 1, 2], [0, 0, 2]]])
        iou = landcover_iou(pred, self.target, self.num_classes, mode="mean")
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_perclass_frequency_returns_tuple(self):
        """Test perclass_frequency mode returns (weighted_iou, ious, counts)."""
        result = landcover_iou(
            self.pred, self.target, self.num_classes, mode="perclass_frequency"
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        weighted_iou, ious, counts = result
        self.assertIsInstance(weighted_iou, float)
        self.assertIsInstance(ious, list)
        self.assertIsInstance(counts, list)

    def test_boundary_weighted_requires_weight_map(self):
        """Test that boundary_weighted mode raises error without weight map."""
        with self.assertRaises(ValueError):
            landcover_iou(
                self.pred, self.target, self.num_classes, mode="boundary_weighted"
            )

    def test_boundary_weighted_with_weight_map(self):
        """Test boundary_weighted mode with a provided weight map."""
        weight_map = torch.ones(1, 2, 3)
        iou = landcover_iou(
            self.pred,
            self.target,
            self.num_classes,
            mode="boundary_weighted",
            boundary_weight_map=weight_map,
        )
        self.assertIsInstance(iou, float)

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched pred/target shapes raise ValueError."""
        pred = torch.tensor([[[0, 1], [2, 0]]])
        target = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
        with self.assertRaises(ValueError):
            landcover_iou(pred, target, self.num_classes, mode="mean")

    def test_logit_input_auto_argmax(self):
        """Test that 4D logit input is auto-argmaxed to class predictions."""
        logits = torch.randn(1, self.num_classes, 2, 3)
        iou = landcover_iou(logits, self.target, self.num_classes, mode="mean")
        self.assertIsInstance(iou, float)
        self.assertGreaterEqual(iou, 0.0)

    def test_ignore_index(self):
        """Test that ignore_index excludes the specified class."""
        iou_all = landcover_iou(self.pred, self.target, self.num_classes, mode="mean")
        iou_ignore = landcover_iou(
            self.pred, self.target, self.num_classes, ignore_index=0, mode="mean"
        )
        # Both should be valid floats
        self.assertIsInstance(iou_all, float)
        self.assertIsInstance(iou_ignore, float)


if __name__ == "__main__":
    unittest.main()
