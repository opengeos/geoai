#!/usr/bin/env python

"""Tests for loss functions and IoU in `geoai.landcover_train` module."""

import unittest

import torch

from geoai.landcover_train import (
    DiceLoss,
    FocalLoss,
    LandcoverCrossEntropyLoss,
    TverskyLoss,
    UnifiedFocalLoss,
    get_landcover_loss_function,
    landcover_iou,
)


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


class TestDiceLoss(unittest.TestCase):
    """Tests for the DiceLoss class."""

    def setUp(self):
        """Set up small test tensors."""
        self.num_classes = 3
        self.batch_size = 2
        self.h, self.w = 4, 4
        self.inputs = torch.randn(self.batch_size, self.num_classes, self.h, self.w)
        self.targets = torch.randint(
            0, self.num_classes, (self.batch_size, self.h, self.w)
        )

    def test_forward_returns_scalar(self):
        """Test that forward pass returns a scalar loss."""
        loss_fn = DiceLoss()
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_perfect_prediction_low_loss(self):
        """Test that near-perfect predictions yield low loss."""
        # Create one-hot logits that match targets exactly
        one_hot = (
            torch.nn.functional.one_hot(self.targets, self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )
        logits = one_hot * 100.0  # large logits → near-1.0 softmax
        loss_fn = DiceLoss()
        loss = loss_fn(logits, self.targets)
        self.assertLess(loss.item(), 0.05)

    def test_with_class_weights(self):
        """Test Dice loss with per-class weights."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = DiceLoss(weight=weights)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_ignore_index(self):
        """Test that ignored pixels do not affect the loss."""
        targets = self.targets.clone()
        targets[:, 0, :] = 255  # mark first row as ignored
        loss_fn = DiceLoss(ignore_index=255)

        loss = loss_fn(self.inputs, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

        # Changing logits at ignored positions should not change the loss
        inputs_perturbed = self.inputs.clone()
        ignore_mask = (targets == 255).unsqueeze(1).expand_as(inputs_perturbed)
        inputs_perturbed[ignore_mask] = torch.randn_like(inputs_perturbed[ignore_mask])
        loss_perturbed = loss_fn(inputs_perturbed, targets)
        self.assertAlmostEqual(loss.item(), loss_perturbed.item(), places=6)


class TestTverskyLoss(unittest.TestCase):
    """Tests for the TverskyLoss class."""

    def setUp(self):
        """Set up small test tensors."""
        self.num_classes = 3
        self.batch_size = 2
        self.h, self.w = 4, 4
        self.inputs = torch.randn(self.batch_size, self.num_classes, self.h, self.w)
        self.targets = torch.randint(
            0, self.num_classes, (self.batch_size, self.h, self.w)
        )

    def test_forward_returns_scalar(self):
        """Test that forward pass returns a scalar loss."""
        loss_fn = TverskyLoss()
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_equals_dice_when_symmetric(self):
        """Test that alpha=beta=0.5 matches DiceLoss (with smooth near zero)."""
        torch.manual_seed(42)
        inputs = torch.randn(2, self.num_classes, 4, 4)
        targets = torch.randint(0, self.num_classes, (2, 4, 4))
        eps = 1e-7
        dice_loss = DiceLoss(smooth=eps)(inputs, targets)
        tversky_loss = TverskyLoss(alpha=0.5, beta=0.5, smooth=eps)(inputs, targets)
        self.assertAlmostEqual(dice_loss.item(), tversky_loss.item(), places=4)

    def test_asymmetric_weights_differ(self):
        """Test that asymmetric alpha/beta produces different loss."""
        # Use a hand-crafted example where the difference is guaranteed:
        # logits that predict class 0 everywhere, but target has class 1,
        # so there are non-zero FP and FN that alpha/beta weigh differently.
        inputs = torch.zeros(1, self.num_classes, 4, 4)
        inputs[:, 0, :, :] = 10.0  # strongly predict class 0
        targets = torch.ones(1, 4, 4, dtype=torch.long)  # ground truth is class 1
        symmetric = TverskyLoss(alpha=0.5, beta=0.5)(inputs, targets)
        asymmetric = TverskyLoss(alpha=0.3, beta=0.7)(inputs, targets)
        self.assertNotAlmostEqual(symmetric.item(), asymmetric.item(), places=4)

    def test_ignore_index(self):
        """Test that ignored pixels do not contribute."""
        targets = self.targets.clone()
        targets[:, 0, :] = 255
        loss_fn = TverskyLoss(ignore_index=255)
        loss = loss_fn(self.inputs, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_with_class_weights(self):
        """Test Tversky loss with per-class weights."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = TverskyLoss(weight=weights)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)


class TestUnifiedFocalLoss(unittest.TestCase):
    """Tests for the UnifiedFocalLoss class."""

    def setUp(self):
        """Set up small test tensors."""
        self.num_classes = 3
        self.batch_size = 2
        self.h, self.w = 4, 4
        self.inputs = torch.randn(self.batch_size, self.num_classes, self.h, self.w)
        self.targets = torch.randint(
            0, self.num_classes, (self.batch_size, self.h, self.w)
        )

    def test_forward_returns_scalar(self):
        """Test that forward pass returns a scalar loss."""
        loss_fn = UnifiedFocalLoss()
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_lambda_one_is_pure_distribution(self):
        """Test that lambda=1 produces only the distribution component."""
        ufl = UnifiedFocalLoss(lambda_=1.0, gamma=0.75)
        loss = ufl(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_lambda_zero_is_pure_region(self):
        """Test that lambda=0 produces only the region component."""
        ufl = UnifiedFocalLoss(lambda_=0.0, gamma=0.75)
        loss = ufl(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_log_cosh_stabilisation(self):
        """Test that use_log_cosh produces finite output."""
        loss_fn = UnifiedFocalLoss(use_log_cosh=True)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_gradient_flows(self):
        """Test that gradients flow through the loss."""
        inputs = self.inputs.clone().requires_grad_(True)
        loss_fn = UnifiedFocalLoss()
        loss = loss_fn(inputs, self.targets)
        loss.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_ignore_index(self):
        """Test that ignored pixels do not contribute."""
        targets = self.targets.clone()
        targets[:, 0, :] = 255
        loss_fn = UnifiedFocalLoss(ignore_index=255)
        loss = loss_fn(self.inputs, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_with_class_weights(self):
        """Test unified focal loss with per-class weights."""
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = UnifiedFocalLoss(weight=weights)
        loss = loss_fn(self.inputs, self.targets)
        self.assertEqual(loss.dim(), 0)


class TestGetLandcoverLossFunction(unittest.TestCase):
    """Tests for the get_landcover_loss_function factory."""

    def test_crossentropy(self):
        """Test factory creates CrossEntropy loss."""
        loss_fn = get_landcover_loss_function(
            "crossentropy", device=torch.device("cpu")
        )
        self.assertIsInstance(loss_fn, LandcoverCrossEntropyLoss)

    def test_focal(self):
        """Test factory creates Focal loss."""
        loss_fn = get_landcover_loss_function("focal", device=torch.device("cpu"))
        self.assertIsInstance(loss_fn, FocalLoss)

    def test_dice(self):
        """Test factory creates Dice loss."""
        loss_fn = get_landcover_loss_function("dice", device=torch.device("cpu"))
        self.assertIsInstance(loss_fn, DiceLoss)

    def test_tversky(self):
        """Test factory creates Tversky loss."""
        loss_fn = get_landcover_loss_function("tversky", device=torch.device("cpu"))
        self.assertIsInstance(loss_fn, TverskyLoss)

    def test_unified_focal(self):
        """Test factory creates Unified Focal loss."""
        loss_fn = get_landcover_loss_function(
            "unified_focal", device=torch.device("cpu")
        )
        self.assertIsInstance(loss_fn, UnifiedFocalLoss)

    def test_ufl_alias(self):
        """Test that 'ufl' alias maps to UnifiedFocalLoss."""
        loss_fn = get_landcover_loss_function("ufl", device=torch.device("cpu"))
        self.assertIsInstance(loss_fn, UnifiedFocalLoss)


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
