#!/usr/bin/env python

"""Tests for `geoai.dinov3_finetune` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

# Detect whether Lightning is available so we can skip tests that
# instantiate DINOv3Segmenter (which requires Lightning at runtime).
try:
    import lightning.pytorch  # noqa: F401

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False


# ------------------------------------------------------------------ #
# Import tests
# ------------------------------------------------------------------ #


class TestDinov3FinetuneImport(unittest.TestCase):
    """Tests for dinov3_finetune module import behaviour."""

    def test_module_imports(self):
        """Test that the dinov3_finetune module can be imported."""
        import geoai.dinov3_finetune

        self.assertTrue(hasattr(geoai.dinov3_finetune, "DINOv3Segmenter"))

    def test_dinov3_segmenter_exists(self):
        """Test that DINOv3Segmenter class exists."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        self.assertTrue(callable(DINOv3Segmenter))

    def test_dpt_segmentation_head_exists(self):
        """Test that DPTSegmentationHead class exists."""
        from geoai.dinov3_finetune import DPTSegmentationHead

        self.assertTrue(callable(DPTSegmentationHead))

    def test_lora_linear_exists(self):
        """Test that LoRALinear class exists."""
        from geoai.dinov3_finetune import LoRALinear

        self.assertTrue(callable(LoRALinear))

    def test_dinov3_segmentation_dataset_exists(self):
        """Test that DINOv3SegmentationDataset class exists."""
        from geoai.dinov3_finetune import DINOv3SegmentationDataset

        self.assertTrue(callable(DINOv3SegmentationDataset))

    def test_train_dinov3_segmentation_exists(self):
        """Test that train_dinov3_segmentation function exists."""
        from geoai.dinov3_finetune import train_dinov3_segmentation

        self.assertTrue(callable(train_dinov3_segmentation))

    def test_dinov3_segment_geotiff_exists(self):
        """Test that dinov3_segment_geotiff function exists."""
        from geoai.dinov3_finetune import dinov3_segment_geotiff

        self.assertTrue(callable(dinov3_segment_geotiff))

    def test_lightning_available_flag(self):
        """Test that LIGHTNING_AVAILABLE flag is defined."""
        from geoai.dinov3_finetune import LIGHTNING_AVAILABLE

        self.assertIsInstance(LIGHTNING_AVAILABLE, bool)


# ------------------------------------------------------------------ #
# Lazy-import / __init__.py tests
# ------------------------------------------------------------------ #


class TestDinov3FinetuneExports(unittest.TestCase):
    """Verify that lazy exports from ``geoai.__init__`` resolve correctly."""

    def test_dinov3_segmenter_in_init(self):
        """Test DINOv3Segmenter is accessible from geoai namespace."""
        import geoai

        self.assertIn("DINOv3Segmenter", dir(geoai))

    def test_train_function_in_init(self):
        """Test train_dinov3_segmentation is accessible from geoai namespace."""
        import geoai

        self.assertIn("train_dinov3_segmentation", dir(geoai))

    def test_segment_geotiff_in_init(self):
        """Test dinov3_segment_geotiff is accessible from geoai namespace."""
        import geoai

        self.assertIn("dinov3_segment_geotiff", dir(geoai))

    def test_dataset_in_init(self):
        """Test DINOv3SegmentationDataset is accessible from geoai namespace."""
        import geoai

        self.assertIn("DINOv3SegmentationDataset", dir(geoai))


# ------------------------------------------------------------------ #
# Signature tests
# ------------------------------------------------------------------ #


class TestDinov3FinetuneSignatures(unittest.TestCase):
    """Tests for function and method signatures."""

    def test_dinov3_segmenter_init_params(self):
        """Test DINOv3Segmenter.__init__ has expected parameters."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        sig = inspect.signature(DINOv3Segmenter.__init__)
        expected = [
            "model_name",
            "weights_path",
            "num_classes",
            "decoder_features",
            "learning_rate",
            "weight_decay",
            "freeze_backbone",
            "use_lora",
            "lora_rank",
            "lora_alpha",
            "loss_fn",
            "class_weights",
            "ignore_index",
        ]
        for param in expected:
            self.assertIn(param, sig.parameters)

    def test_dinov3_segmenter_init_defaults(self):
        """Test DINOv3Segmenter.__init__ default values."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        sig = inspect.signature(DINOv3Segmenter.__init__)
        self.assertEqual(sig.parameters["model_name"].default, "dinov3_vitl16")
        self.assertEqual(sig.parameters["num_classes"].default, 2)
        self.assertEqual(sig.parameters["decoder_features"].default, 256)
        self.assertEqual(sig.parameters["learning_rate"].default, 1e-4)
        self.assertEqual(sig.parameters["freeze_backbone"].default, True)
        self.assertEqual(sig.parameters["use_lora"].default, False)
        self.assertEqual(sig.parameters["lora_rank"].default, 4)
        self.assertEqual(sig.parameters["ignore_index"].default, 255)

    def test_train_dinov3_segmentation_params(self):
        """Test train_dinov3_segmentation has expected parameters."""
        from geoai.dinov3_finetune import train_dinov3_segmentation

        sig = inspect.signature(train_dinov3_segmentation)
        expected = [
            "train_dataset",
            "val_dataset",
            "test_dataset",
            "model_name",
            "weights_path",
            "num_classes",
            "decoder_features",
            "output_dir",
            "batch_size",
            "num_epochs",
            "learning_rate",
            "weight_decay",
            "num_workers",
            "freeze_backbone",
            "use_lora",
            "lora_rank",
            "lora_alpha",
            "class_weights",
            "ignore_index",
            "accelerator",
            "devices",
            "monitor_metric",
            "mode",
            "patience",
            "save_top_k",
            "checkpoint_path",
        ]
        for param in expected:
            self.assertIn(param, sig.parameters)

    def test_dinov3_segment_geotiff_params(self):
        """Test dinov3_segment_geotiff has expected parameters."""
        from geoai.dinov3_finetune import dinov3_segment_geotiff

        sig = inspect.signature(dinov3_segment_geotiff)
        expected = [
            "input_path",
            "output_path",
            "checkpoint_path",
            "model_name",
            "weights_path",
            "num_classes",
            "decoder_features",
            "window_size",
            "overlap",
            "batch_size",
            "device",
            "quiet",
        ]
        for param in expected:
            self.assertIn(param, sig.parameters)

    def test_dpt_segmentation_head_init_params(self):
        """Test DPTSegmentationHead.__init__ has expected parameters."""
        from geoai.dinov3_finetune import DPTSegmentationHead

        sig = inspect.signature(DPTSegmentationHead.__init__)
        self.assertIn("embed_dim", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("features", sig.parameters)

    def test_lora_linear_init_params(self):
        """Test LoRALinear.__init__ has expected parameters."""
        from geoai.dinov3_finetune import LoRALinear

        sig = inspect.signature(LoRALinear.__init__)
        self.assertIn("original", sig.parameters)
        self.assertIn("rank", sig.parameters)
        self.assertIn("alpha", sig.parameters)

    def test_dataset_init_params(self):
        """Test DINOv3SegmentationDataset.__init__ has expected parameters."""
        from geoai.dinov3_finetune import DINOv3SegmentationDataset

        sig = inspect.signature(DINOv3SegmentationDataset.__init__)
        expected = [
            "image_paths",
            "mask_paths",
            "patch_size",
            "target_size",
            "num_channels",
            "transform",
        ]
        for param in expected:
            self.assertIn(param, sig.parameters)


# ------------------------------------------------------------------ #
# LoRALinear unit tests
# ------------------------------------------------------------------ #


class TestLoRALinear(unittest.TestCase):
    """Tests for LoRALinear layer."""

    def test_output_shape(self):
        """LoRA output shape must match original linear output."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4)

        x = torch.randn(2, 64)
        out = lora(x)
        self.assertEqual(out.shape, (2, 32))

    def test_original_frozen(self):
        """Original linear parameters should be frozen."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4)

        for p in lora.original.parameters():
            self.assertFalse(p.requires_grad)

    def test_lora_params_trainable(self):
        """LoRA A and B matrices should require grad."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4)

        self.assertTrue(lora.lora_A.requires_grad)
        self.assertTrue(lora.lora_B.requires_grad)

    def test_lora_b_initialized_to_zero(self):
        """LoRA B matrix must start at zero so initial output is unchanged."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4)

        self.assertTrue(torch.all(lora.lora_B == 0))

    def test_alpha_defaults_to_rank(self):
        """When alpha is not specified it should default to the rank value."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=8)

        self.assertEqual(lora.alpha, 8.0)

    def test_custom_alpha(self):
        """Custom alpha should be stored correctly."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=16.0)

        self.assertEqual(lora.alpha, 16.0)

    def test_initial_output_matches_original(self):
        """With B=0 the LoRA output should match the original linear exactly."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4)

        x = torch.randn(3, 64)
        with torch.no_grad():
            expected = original(x)
            actual = lora(x)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_in_out_features_exposed(self):
        """LoRALinear must expose in_features/out_features for Lightning summary."""
        from geoai.dinov3_finetune import LoRALinear

        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4)

        self.assertEqual(lora.in_features, 64)
        self.assertEqual(lora.out_features, 32)


# ------------------------------------------------------------------ #
# DPTSegmentationHead unit tests
# ------------------------------------------------------------------ #


class TestDPTSegmentationHead(unittest.TestCase):
    """Tests for DPTSegmentationHead decoder."""

    def test_output_shape(self):
        """Logits shape should be (B, num_classes, H, W)."""
        from geoai.dinov3_finetune import DPTSegmentationHead

        embed_dim = 1024
        num_classes = 5
        head = DPTSegmentationHead(embed_dim, num_classes, features=128)

        B, H_p, W_p = 2, 16, 16
        features = [torch.randn(B, embed_dim, H_p, W_p) for _ in range(4)]
        target_size = (256, 256)

        logits = head(features, target_size)
        self.assertEqual(logits.shape, (B, num_classes, 256, 256))

    def test_binary_segmentation(self):
        """Binary segmentation (num_classes=2) should work."""
        from geoai.dinov3_finetune import DPTSegmentationHead

        head = DPTSegmentationHead(384, 2, features=64)
        features = [torch.randn(1, 384, 8, 8) for _ in range(4)]
        logits = head(features, (128, 128))
        self.assertEqual(logits.shape, (1, 2, 128, 128))

    def test_different_feature_spatial_sizes(self):
        """Features at different spatial sizes should still fuse correctly."""
        from geoai.dinov3_finetune import DPTSegmentationHead

        head = DPTSegmentationHead(256, 3, features=64)
        features = [torch.randn(1, 256, 14, 14) for _ in range(4)]
        logits = head(features, (224, 224))
        self.assertEqual(logits.shape, (1, 3, 224, 224))


# ------------------------------------------------------------------ #
# Extraction layer mapping tests
# ------------------------------------------------------------------ #


class TestExtractionLayers(unittest.TestCase):
    """Tests for ``_get_extraction_layers`` helper."""

    def test_known_depths(self):
        """Known ViT depths should return the correct layer indices."""
        from geoai.dinov3_finetune import _get_extraction_layers

        self.assertEqual(_get_extraction_layers(12), [2, 5, 8, 11])
        self.assertEqual(_get_extraction_layers(24), [5, 11, 17, 23])
        self.assertEqual(_get_extraction_layers(32), [7, 15, 23, 31])
        self.assertEqual(_get_extraction_layers(40), [9, 19, 29, 39])

    def test_unsupported_depth_raises(self):
        """Unsupported depths should raise a ValueError."""
        from geoai.dinov3_finetune import _get_extraction_layers

        with self.assertRaises(ValueError):
            _get_extraction_layers(16)


# ------------------------------------------------------------------ #
# Dataset tests
# ------------------------------------------------------------------ #


class TestDINOv3SegmentationDataset(unittest.TestCase):
    """Tests for DINOv3SegmentationDataset."""

    def test_mismatched_lengths_raises(self):
        """Mismatched image/mask list lengths should raise ValueError."""
        from geoai.dinov3_finetune import DINOv3SegmentationDataset

        with self.assertRaises(ValueError):
            DINOv3SegmentationDataset(
                image_paths=["a.tif", "b.tif"],
                mask_paths=["a.tif"],
            )

    def test_empty_paths_raises(self):
        """Empty image list should raise ValueError."""
        from geoai.dinov3_finetune import DINOv3SegmentationDataset

        with self.assertRaises(ValueError):
            DINOv3SegmentationDataset(image_paths=[], mask_paths=[])

    def test_len(self):
        """__len__ should match the number of paths."""
        from geoai.dinov3_finetune import DINOv3SegmentationDataset

        ds = DINOv3SegmentationDataset.__new__(DINOv3SegmentationDataset)
        ds.image_paths = ["a.tif", "b.tif", "c.tif"]
        ds.mask_paths = ["a.tif", "b.tif", "c.tif"]
        self.assertEqual(len(ds), 3)

    def test_init_stores_parameters(self):
        """Constructor should store all parameters correctly."""
        from geoai.dinov3_finetune import DINOv3SegmentationDataset

        ds = DINOv3SegmentationDataset(
            image_paths=["a.tif"],
            mask_paths=["a.tif"],
            patch_size=16,
            target_size=256,
            num_channels=3,
        )
        self.assertEqual(ds.patch_size, 16)
        self.assertEqual(ds.target_size, 256)
        self.assertEqual(ds.num_channels, 3)
        self.assertIsNone(ds.transform)


# ------------------------------------------------------------------ #
# DINOv3Segmenter tests (mocked backbone) -- require Lightning
# ------------------------------------------------------------------ #


@unittest.skipUnless(HAS_LIGHTNING, "PyTorch Lightning not installed")
class TestDINOv3Segmenter(unittest.TestCase):
    """Tests for DINOv3Segmenter with mocked backbone loading."""

    def _make_mock_backbone(self, embed_dim=1024, patch_size=16, num_blocks=24):
        """Create a mock backbone that mimics a DINOv3 ViT."""
        backbone = MagicMock()
        backbone.patch_size = patch_size
        backbone.embed_dim = embed_dim
        backbone.blocks = [MagicMock() for _ in range(num_blocks)]

        def mock_intermediate(x, n=4, reshape=True, norm=True):
            B = x.shape[0]
            H_p = x.shape[2] // patch_size
            W_p = x.shape[3] // patch_size
            num_features = len(n) if isinstance(n, (list, tuple)) else n
            return [torch.randn(B, embed_dim, H_p, W_p) for _ in range(num_features)]

        backbone.get_intermediate_layers = mock_intermediate
        backbone.parameters.return_value = iter([torch.nn.Parameter(torch.randn(2, 2))])
        backbone.named_modules.return_value = []
        return backbone

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_init_with_mocked_backbone(self, mock_load):
        """DINOv3Segmenter should initialise correctly with a mocked backbone."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(num_classes=5)
        self.assertEqual(model.hparams.num_classes, 5)
        self.assertEqual(model.patch_size, 16)
        self.assertEqual(model.embed_dim, 1024)
        mock_load.assert_called_once()

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_num_classes_less_than_2_raises(self, mock_load):
        """num_classes < 2 should raise ValueError."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        with self.assertRaises(ValueError):
            DINOv3Segmenter(num_classes=1)

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_forward_shape(self, mock_load):
        """forward() should produce logits of shape (B, C, H, W)."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(num_classes=3)
        model.eval()

        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (2, 3, 256, 256))

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_backbone_frozen_by_default(self, mock_load):
        """With freeze_backbone=True, backbone params should not require grad."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        bb = self._make_mock_backbone()
        real_param = torch.nn.Parameter(torch.randn(2, 2))
        bb.parameters.return_value = iter([real_param])
        mock_load.return_value = bb

        DINOv3Segmenter(freeze_backbone=True)
        self.assertFalse(real_param.requires_grad)

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_decoder_trainable(self, mock_load):
        """Decoder parameters should always be trainable."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(freeze_backbone=True)
        decoder_params = list(model.decoder.parameters())
        self.assertTrue(all(p.requires_grad for p in decoder_params))

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_default_loss_is_cross_entropy(self, mock_load):
        """Default loss should be CrossEntropyLoss with ignore_index=255."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter()
        self.assertIsInstance(model.loss_fn, nn.CrossEntropyLoss)
        self.assertEqual(model.loss_fn.ignore_index, 255)

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_custom_ignore_index(self, mock_load):
        """Custom ignore_index should be propagated to the loss."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(ignore_index=0)
        self.assertEqual(model.loss_fn.ignore_index, 0)

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_class_weights_in_loss(self, mock_load):
        """Providing class_weights should create a weighted loss."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        weights = torch.tensor([1.0, 2.0])
        model = DINOv3Segmenter(num_classes=2, class_weights=weights)
        self.assertTrue(torch.equal(model.loss_fn.weight, weights))

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_miou_computation(self, mock_load):
        """_compute_miou should return a scalar in [0, 1]."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(num_classes=3, ignore_index=255)
        pred = torch.tensor([[0, 1, 2, 0]])
        target = torch.tensor([[0, 1, 2, 1]])
        iou = model._compute_miou(pred, target)
        self.assertGreater(iou.item(), 0.0)
        self.assertLessEqual(iou.item(), 1.0)

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_miou_ignores_index(self, mock_load):
        """Pixels with ignore_index should be excluded from mIoU."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(num_classes=2, ignore_index=255)
        pred = torch.tensor([[0, 0, 0, 1]])
        target = torch.tensor([[0, 0, 255, 1]])
        iou = model._compute_miou(pred, target)
        self.assertAlmostEqual(iou.item(), 1.0, places=4)

    @patch("geoai.dinov3_finetune.DINOv3Segmenter._load_backbone")
    def test_miou_all_ignored_returns_zero(self, mock_load):
        """If all pixels are ignored, mIoU should return 0."""
        from geoai.dinov3_finetune import DINOv3Segmenter

        mock_load.return_value = self._make_mock_backbone()

        model = DINOv3Segmenter(num_classes=2, ignore_index=255)
        pred = torch.tensor([[0, 1, 0, 1]])
        target = torch.tensor([[255, 255, 255, 255]])
        iou = model._compute_miou(pred, target)
        self.assertEqual(iou.item(), 0.0)


# ------------------------------------------------------------------ #
# train_dinov3_segmentation guard tests
# ------------------------------------------------------------------ #


class TestTrainFunction(unittest.TestCase):
    """Tests for train_dinov3_segmentation function guards."""

    def test_callable(self):
        """train_dinov3_segmentation should be callable."""
        from geoai.dinov3_finetune import train_dinov3_segmentation

        self.assertTrue(callable(train_dinov3_segmentation))

    @unittest.skipIf(HAS_LIGHTNING, "Only tests the guard when Lightning is absent")
    def test_raises_without_lightning(self):
        """Should raise ImportError when Lightning is not installed."""
        from geoai.dinov3_finetune import train_dinov3_segmentation

        with self.assertRaises(ImportError):
            train_dinov3_segmentation(train_dataset=MagicMock())


# ------------------------------------------------------------------ #
# dinov3_segment_geotiff signature tests
# ------------------------------------------------------------------ #


class TestSegmentGeotiffFunction(unittest.TestCase):
    """Tests for dinov3_segment_geotiff function."""

    def test_callable(self):
        """dinov3_segment_geotiff should be callable."""
        from geoai.dinov3_finetune import dinov3_segment_geotiff

        self.assertTrue(callable(dinov3_segment_geotiff))

    def test_default_values(self):
        """Check default parameter values."""
        from geoai.dinov3_finetune import dinov3_segment_geotiff

        sig = inspect.signature(dinov3_segment_geotiff)
        self.assertEqual(sig.parameters["model_name"].default, "dinov3_vitl16")
        self.assertEqual(sig.parameters["num_classes"].default, 2)
        self.assertEqual(sig.parameters["window_size"].default, 512)
        self.assertEqual(sig.parameters["overlap"].default, 256)
        self.assertEqual(sig.parameters["batch_size"].default, 4)
        self.assertIsNone(sig.parameters["device"].default)
        self.assertFalse(sig.parameters["quiet"].default)

    def test_overlap_ge_window_size_raises(self):
        """overlap >= window_size should raise ValueError."""
        from geoai.dinov3_finetune import dinov3_segment_geotiff

        with self.assertRaises(ValueError):
            dinov3_segment_geotiff(
                input_path="dummy.tif",
                output_path="out.tif",
                checkpoint_path="model.ckpt",
                window_size=256,
                overlap=256,
            )

    def test_overlap_gt_window_size_raises(self):
        """overlap > window_size should also raise ValueError."""
        from geoai.dinov3_finetune import dinov3_segment_geotiff

        with self.assertRaises(ValueError):
            dinov3_segment_geotiff(
                input_path="dummy.tif",
                output_path="out.tif",
                checkpoint_path="model.ckpt",
                window_size=256,
                overlap=512,
            )


if __name__ == "__main__":
    unittest.main()
