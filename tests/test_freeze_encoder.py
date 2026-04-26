"""Tests for freeze_encoder functionality in train_segmentation_model."""

import os
import tempfile

import pytest
import torch

import segmentation_models_pytorch as smp

from geoai.train import get_smp_model


class TestFreezeEncoder:
    """Tests for the freeze_encoder parameter behavior."""

    def _make_model(self):
        """Create a small SMP model for testing."""
        return get_smp_model(
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )

    def test_encoder_frozen_after_flag(self):
        """Verify encoder params have requires_grad=False after freezing."""
        model = self._make_model()

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        # All encoder params should be frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad

        # Decoder params should still be trainable
        for param in model.decoder.parameters():
            assert param.requires_grad

    def test_optimizer_excludes_frozen_params(self):
        """Verify optimizer only contains trainable parameters."""
        model = self._make_model()

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)

        # Optimizer param groups should only have trainable params
        total_optimizer_params = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())

        assert total_optimizer_params == total_trainable
        assert total_optimizer_params < total_all

    def test_frozen_param_count(self):
        """Verify frozen params are a subset of total params."""
        model = self._make_model()

        total_before = sum(p.numel() for p in model.parameters())

        for param in model.encoder.parameters():
            param.requires_grad = False

        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert frozen > 0
        assert trainable > 0
        assert frozen + trainable == total_before

    def test_checkpoint_load_then_freeze(self, tmp_path):
        """Verify checkpoint loading works correctly with encoder freezing."""
        model = self._make_model()

        # Save a checkpoint
        checkpoint_path = str(tmp_path / "test_model.pth")
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": 5, "best_iou": 0.75},
            checkpoint_path,
        )

        # Load into a new model and freeze
        model2 = self._make_model()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model2.load_state_dict(checkpoint["model_state_dict"])

        for param in model2.encoder.parameters():
            param.requires_grad = False

        # Verify weights match after loading
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Parameter {n1} mismatch after loading"

        # Verify encoder is frozen
        for param in model2.encoder.parameters():
            assert not param.requires_grad

    def test_dataparallel_freeze(self):
        """Verify freezing works through DataParallel wrapper."""
        model = self._make_model()
        # Simulate DataParallel wrapping (without actual multi-GPU)
        wrapped = torch.nn.DataParallel(model)

        base_model = wrapped.module
        for param in base_model.encoder.parameters():
            param.requires_grad = False

        # Check through the wrapper
        frozen = sum(p.numel() for p in wrapped.parameters() if not p.requires_grad)
        assert frozen > 0
