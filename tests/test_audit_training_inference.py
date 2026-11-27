#!/usr/bin/env python
"""Regression tests for Batch 2 training and inference correctness fixes."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import torch.nn as nn


def _write_geotiff(path, data, crs="EPSG:32617", origin=(500000.0, 4000000.0)):
    """Write a CHW or HW numpy array as a GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(origin[0], origin[1], 1.0, 1.0),
    ) as dst:
        dst.write(data)
    return path


class TestLandcoverTrainFixes(unittest.TestCase):
    """Group 1: geoai.landcover_train correctness fixes."""

    def test_ignore_index_false_not_treated_as_class_zero(self):
        from geoai.landcover_train import (
            LandcoverCrossEntropyLoss,
            compute_class_weights,
            get_landcover_loss_function,
            landcover_iou,
        )

        ce = LandcoverCrossEntropyLoss(ignore_index=False)
        self.assertEqual(ce.ignore_index, -100)

        factory_loss = get_landcover_loss_function(
            "crossentropy", ignore_index=False, device=torch.device("cpu")
        )
        self.assertEqual(factory_loss.ignore_index, -100)

        # Target has class 0 (completely mispredicted as 1) and class 1 (perfect).
        # If ignore_index=False were treated as int 0, class 0 would be skipped
        # and IoU would wrongly be 1.0.
        target = torch.tensor([[[0, 0], [1, 1]]])
        pred = torch.tensor([[[1, 1], [1, 1]]])

        mean_iou = landcover_iou(
            pred, target, num_classes=2, ignore_index=False, mode="mean"
        )
        self.assertLess(mean_iou, 0.5)

        _, per_class_ious, class_counts = landcover_iou(
            pred,
            target,
            num_classes=2,
            ignore_index=False,
            mode="perclass_frequency",
        )
        self.assertEqual(len(per_class_ious), 2)
        self.assertEqual(class_counts[0], 2)

        b_iou = landcover_iou(
            pred,
            target,
            num_classes=2,
            ignore_index=False,
            mode="boundary_weighted",
            boundary_weight_map=torch.ones_like(target, dtype=torch.float32),
        )
        self.assertLess(b_iou, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            lbl = np.array([[0, 0], [1, 1]], dtype=np.uint8)
            _write_geotiff(os.path.join(tmpdir, "lbl.tif"), lbl)
            weights = compute_class_weights(tmpdir, num_classes=2, ignore_index=False)
            self.assertGreater(weights[0].item(), 0.0)
            self.assertAlmostEqual(weights[0].item(), weights[1].item(), places=5)

    def test_unified_focal_loss_absent_class_finite_gradients(self):
        from geoai.landcover_train import UnifiedFocalLoss

        # Class 2 is absent from targets and has near-zero predicted probability,
        # so tversky[2] == 1.0 and (1 - tversky[2]) == 0.0.
        logits = torch.zeros(1, 3, 4, 4, dtype=torch.float32)
        logits[:, 0, :, :] = 10.0
        logits[:, 1, :, :] = 10.0
        logits[:, 2, :, :] = -100.0
        logits.requires_grad_(True)

        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[:, :, 2:] = 1

        loss = UnifiedFocalLoss(gamma=0.75)(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focal_loss_and_ufl_unweighted_pt_and_valid_mean(self):
        from geoai.landcover_train import FocalLoss, UnifiedFocalLoss

        logits = torch.tensor(
            [[[[2.0, 0.5], [1.0, -1.0]], [[0.5, 2.0], [-1.0, 1.0]]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[[0, 1], [255, 255]]], dtype=torch.long)
        weights = torch.tensor([1.0, 4.0], dtype=torch.float32)
        gamma = 2.0
        alpha = 0.5

        # Analytical focal loss over valid pixels only
        ce_unweighted = torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=255, reduction="none"
        )
        ce_weighted = torch.nn.functional.cross_entropy(
            logits, targets, weight=weights, ignore_index=255, reduction="none"
        )
        p_t = torch.exp(-ce_unweighted)
        valid = targets != 255
        expected_focal = (alpha * ((1.0 - p_t) ** gamma) * ce_weighted)[valid].mean()

        fl = FocalLoss(
            alpha=alpha, gamma=gamma, ignore_index=255, reduction="mean", weight=weights
        )
        actual_focal = fl(logits, targets)
        self.assertAlmostEqual(actual_focal.item(), expected_focal.item(), places=5)

        # UnifiedFocalLoss with lambda_=1.0 (pure distribution component)
        expected_ufl = (((1.0 - p_t) ** gamma) * ce_weighted)[valid].mean()
        ufl = UnifiedFocalLoss(
            lambda_=1.0, gamma=gamma, ignore_index=255, weight=weights
        )
        actual_ufl = ufl(logits, targets)
        self.assertAlmostEqual(actual_ufl.item(), expected_ufl.item(), places=5)

    def test_evaluate_sparse_iou_indexing_and_foreground_means(self):
        from geoai.landcover_train import evaluate_sparse_iou

        class PredictClassOne(nn.Module):
            def forward(self, x):
                n, _, h, w = x.shape
                out = torch.zeros((n, 3, h, w), dtype=torch.float32, device=x.device)
                out[:, 1, :, :] = 10.0
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            img = np.zeros((3, 4, 4), dtype=np.uint8)
            # Row 0: background (0), rows 1-2: class 1 (8 px), row 3: class 2 (4 px)
            lbl = np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                ],
                dtype=np.uint8,
            )
            _write_geotiff(os.path.join(img_dir, "tile_0.tif"), img)
            _write_geotiff(os.path.join(lbl_dir, "tile_0.tif"), lbl)

            res = evaluate_sparse_iou(
                model=PredictClassOne(),
                images_dir=img_dir,
                labels_dir=lbl_dir,
                num_classes=3,
                num_channels=3,
                background_class=0,
                ignore_index=False,
                device=torch.device("cpu"),
                verbose=False,
            )

            # Class 1: tp=8, fn=0, fp_sparse=4 -> recall=1.0, precision=8/12=2/3
            # Class 2: tp=0, fn=4, fp_sparse=0 -> recall=0.0, precision=0.0
            self.assertAlmostEqual(res["per_class_recall"][1], 1.0, places=5)
            self.assertAlmostEqual(res["per_class_recall"][2], 0.0, places=5)
            self.assertAlmostEqual(res["per_class_precision"][1], 2.0 / 3.0, places=5)
            self.assertAlmostEqual(res["per_class_precision"][2], 0.0, places=5)
            self.assertAlmostEqual(res["mean_recall"], 0.5, places=5)
            self.assertAlmostEqual(res["mean_precision"], 1.0 / 3.0, places=5)

    def test_train_segmentation_landcover_loads_raw_state_dict(self):
        from geoai.landcover_train import train_segmentation_landcover

        class TinySegNet(nn.Module):
            def __init__(self, in_channels=3, classes=2, **kwargs):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, classes, kernel_size=1)

            def forward(self, x):
                return self.conv(x)

        created_models = []

        def _fake_unet(**kwargs):
            m = TinySegNet(**kwargs)
            created_models.append(m)
            return m

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            for i in range(4):
                _write_geotiff(
                    os.path.join(img_dir, f"tile_{i}.tif"),
                    np.full((3, 8, 8), 100, dtype=np.uint8),
                )
                _write_geotiff(
                    os.path.join(lbl_dir, f"tile_{i}.tif"),
                    np.ones((8, 8), dtype=np.uint8),
                )

            # Save a raw state_dict (OrderedDict, no "model_state_dict" wrapper)
            ref = TinySegNet(in_channels=3, classes=2)
            with torch.no_grad():
                ref.conv.weight.fill_(7.25)
                ref.conv.bias.fill_(-1.5)
            ckpt_path = os.path.join(tmpdir, "raw_weights.pth")
            torch.save(ref.state_dict(), ckpt_path)

            with (
                patch("segmentation_models_pytorch.Unet", side_effect=_fake_unet),
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.cuda.device_count", return_value=0),
            ):
                train_segmentation_landcover(
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    output_dir=out_dir,
                    architecture="unet",
                    encoder_weights=None,
                    num_channels=3,
                    num_classes=2,
                    batch_size=2,
                    num_epochs=0,
                    checkpoint_path=ckpt_path,
                    num_workers=0,
                    verbose=False,
                )

            self.assertEqual(len(created_models), 1)
            loaded = created_models[0]
            self.assertTrue(
                torch.allclose(
                    loaded.conv.weight.detach().cpu(),
                    torch.full_like(loaded.conv.weight, 7.25).cpu(),
                )
            )
            self.assertTrue(
                torch.allclose(
                    loaded.conv.bias.detach().cpu(),
                    torch.full_like(loaded.conv.bias, -1.5).cpu(),
                )
            )


class TestTrainModuleFixes(unittest.TestCase):
    """Group 2: geoai.train correctness fixes."""

    def test_evaluate_freezes_bn_and_dropout_after_train_mode(self):
        from geoai.train import evaluate

        class DummyDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3)
                self.drop = nn.Dropout(0.5)
                self.observed_modes = None

            def forward(self, images, targets=None):
                if targets is not None:
                    self.observed_modes = (
                        self.training,
                        self.bn.training,
                        self.drop.training,
                    )
                    return {"loss_box": torch.tensor(0.25)}
                return [
                    {
                        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                        "labels": torch.tensor([1]),
                        "scores": torch.tensor([0.9]),
                    }
                    for _ in images
                ]

        model = DummyDetector()
        loader = [
            (
                [torch.zeros(3, 8, 8)],
                [
                    {
                        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                        "labels": torch.tensor([1]),
                    }
                ],
            )
        ]

        metrics = evaluate(model, loader, torch.device("cpu"), use_mask_iou=False)
        self.assertIsNotNone(model.observed_modes)
        self.assertEqual(model.observed_modes, (True, False, False))
        self.assertAlmostEqual(metrics["loss"], 0.25, places=5)

    def test_inference_on_geotiff_zero_overlap_borders_and_empty_windows(self):
        from geoai.train import inference_on_geotiff

        class AlwaysOneMaskModel(nn.Module):
            def forward(self, images):
                out = []
                for img in images:
                    _, h, w = img.shape
                    out.append(
                        {
                            "scores": torch.tensor([0.95]),
                            "masks": torch.ones((1, 1, h, w), dtype=torch.float32),
                        }
                    )
                return out

        class FirstTileOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_idx = 0

            def forward(self, images):
                out = []
                for img in images:
                    _, h, w = img.shape
                    if self.call_idx == 0:
                        out.append(
                            {
                                "scores": torch.tensor([0.95]),
                                "masks": torch.ones((1, 1, h, w), dtype=torch.float32),
                            }
                        )
                    else:
                        out.append(
                            {
                                "scores": torch.tensor([]),
                                "masks": torch.zeros((0, 1, h, w), dtype=torch.float32),
                            }
                        )
                    self.call_idx += 1
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) overlap == 0 must not divide by 0 / produce NaN
            src_a = _write_geotiff(
                os.path.join(tmpdir, "a.tif"),
                np.full((3, 16, 16), 120, dtype=np.uint8),
            )
            out_a = os.path.join(tmpdir, "out_a.tif")
            inference_on_geotiff(
                AlwaysOneMaskModel(),
                src_a,
                out_a,
                window_size=8,
                overlap=0,
                confidence_threshold=0.5,
                batch_size=2,
                num_channels=3,
                device=torch.device("cpu"),
            )
            with rasterio.open(out_a) as ds:
                mask_a = ds.read(1)
            np.testing.assert_array_equal(mask_a, np.ones((16, 16), dtype=np.uint8))

            # 2) Outer raster boundaries must keep full weight when overlap > 0
            src_b = _write_geotiff(
                os.path.join(tmpdir, "b.tif"),
                np.full((3, 20, 20), 120, dtype=np.uint8),
            )
            out_b = os.path.join(tmpdir, "out_b.tif")
            inference_on_geotiff(
                AlwaysOneMaskModel(),
                src_b,
                out_b,
                window_size=12,
                overlap=4,
                confidence_threshold=0.5,
                batch_size=2,
                num_channels=3,
                device=torch.device("cpu"),
            )
            with rasterio.open(out_b) as ds:
                mask_b = ds.read(1)
            np.testing.assert_array_equal(mask_b, np.ones((20, 20), dtype=np.uint8))

            # 3) Empty-detection window must still contribute to count_accumulator
            src_c = _write_geotiff(
                os.path.join(tmpdir, "c.tif"),
                np.full((3, 12, 20), 120, dtype=np.uint8),
            )
            out_c = os.path.join(tmpdir, "out_c.tif")
            inference_on_geotiff(
                FirstTileOnlyModel(),
                src_c,
                out_c,
                window_size=12,
                overlap=8,
                confidence_threshold=0.5,
                batch_size=1,
                num_channels=3,
                device=torch.device("cpu"),
            )
            with rasterio.open(out_c) as ds:
                mask_c = ds.read(1)
            # In the overlap [8..11], column 10 is closer to tile 0's edge (dist=1, w=0.25)
            # than to tile 1's edge (dist=2, w=0.5). Weighted average = 0.25 / 0.75 = 1/3 < 0.5.
            self.assertEqual(int(mask_c[6, 10]), 0)

    def test_evaluate_coco_metrics_class_id_zero_no_negative_wrap(self):
        from geoai.train import evaluate_coco_metrics

        class ReplayModel(nn.Module):
            def forward(self, images):
                return [
                    {
                        "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
                        "labels": torch.tensor([0]),
                        "scores": torch.tensor([0.95]),
                    }
                ]

        gt = {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
            "labels": torch.tensor([0]),
        }
        loader = [([torch.zeros(3, 8, 8)], [gt])]
        results = evaluate_coco_metrics(
            ReplayModel(),
            loader,
            torch.device("cpu"),
            iou_thresholds=[0.5],
            class_names=["building", "road"],
            verbose=False,
        )
        self.assertIn("AP@0.5/class_0", results)
        self.assertNotIn("AP@0.5/road", results)


class TestSegmentFixes(unittest.TestCase):
    """Group 3: geoai.segment NMS and tiling fixes."""

    def test_grounded_sam_apply_nms_uses_xywh(self):
        from geoai.segment import BoundingBox, DetectionResult, GroundedSAM

        sam = GroundedSAM.__new__(GroundedSAM)
        sam.threshold = 0.3

        # Two disjoint 20x20 boxes far from origin (5px horizontal gap)
        det1 = DetectionResult(
            score=0.95, label="building", box=BoundingBox(100, 100, 120, 120)
        )
        det2 = DetectionResult(
            score=0.85, label="building", box=BoundingBox(125, 100, 145, 120)
        )
        kept = sam._apply_nms([det1, det2], iou_threshold=0.5)
        self.assertEqual(len(kept), 2)

    def test_grounded_sam_and_clip_tiling_no_strip_gap_and_validation(self):
        from geoai.segment import (
            BoundingBox,
            CLIPSegmentation,
            DetectionResult,
            GroundedSAM,
        )

        # tile_size=100, overlap=20 -> effective_tile_size=60; 70 % 60 = 10 < 20
        with tempfile.TemporaryDirectory() as tmpdir:
            ramp = np.linspace(10, 240, 70 * 70, dtype=np.uint8).reshape(70, 70)
            src_path = _write_geotiff(
                os.path.join(tmpdir, "in.tif"),
                np.stack([ramp, ramp, ramp], axis=0),
            )

            # GroundedSAM: overlap validation
            bad_sam = GroundedSAM.__new__(GroundedSAM)
            bad_sam.tile_size = 64
            bad_sam.overlap = 32
            with self.assertRaises(ValueError):
                bad_sam.segment_image(
                    src_path, os.path.join(tmpdir, "bad_sam.tif"), text_prompts="house"
                )

            # GroundedSAM: no strip gap on 70x70 with tile_size=100, overlap=20
            sam = GroundedSAM.__new__(GroundedSAM)
            sam.tile_size = 100
            sam.overlap = 20
            sam.threshold = 0.3
            sam.device = "cpu"

            def _fake_detect(image, labels, **kwargs):
                w, h = image.size
                return [
                    DetectionResult(
                        score=0.9,
                        label="house",
                        box=BoundingBox(0, 0, w, h),
                    )
                ]

            def _fake_segment(image, detections, polygon_refinement=False):
                w, h = image.size
                for det in detections:
                    det.mask = np.ones((h, w), dtype=np.uint8)
                return detections

            sam._detect = _fake_detect
            sam._segment = _fake_segment
            sam_out = os.path.join(tmpdir, "sam_out.tif")
            sam.segment_image(
                src_path,
                sam_out,
                text_prompts="house",
                polygon_refinement=False,
            )
            with rasterio.open(sam_out) as ds:
                combined = ds.read(1)
            np.testing.assert_array_equal(combined, np.ones((70, 70), dtype=np.uint8))

            # CLIPSegmentation: overlap validation
            bad_clip = CLIPSegmentation.__new__(CLIPSegmentation)
            bad_clip.tile_size = 64
            bad_clip.overlap = 32
            bad_clip.device = "cpu"
            with self.assertRaises(ValueError):
                bad_clip.segment_image(
                    src_path, os.path.join(tmpdir, "bad_clip.tif"), text_prompt="water"
                )

            # CLIPSegmentation: no strip gap on 70x70 with tile_size=100, overlap=20
            class _Batch(dict):
                def to(self, device):
                    return self

            clip = CLIPSegmentation.__new__(CLIPSegmentation)
            clip.tile_size = 100
            clip.overlap = 20
            clip.device = "cpu"
            clip.processor = MagicMock(
                return_value=_Batch(pixel_values=torch.zeros((1, 3, 16, 16)))
            )
            fake_out = MagicMock()
            fake_out.logits = torch.full((1, 16, 16), 10.0)
            clip.model = MagicMock(return_value=fake_out)

            clip_out = os.path.join(tmpdir, "clip_out.tif")
            clip.segment_image(
                src_path,
                clip_out,
                text_prompt="water",
                threshold=0.5,
                smoothing_sigma=0.0,
            )
            with rasterio.open(clip_out) as ds:
                seg = ds.read(1)
            np.testing.assert_array_equal(seg, np.ones((70, 70), dtype=np.float32))


class TestOnnxFixes(unittest.TestCase):
    """Group 4: geoai.onnx _prepare_input normalization fix."""

    def test_prepare_input_uint16_no_truncation_and_no_mutation(self):
        from geoai.onnx import ONNXGeoModel

        model = ONNXGeoModel.__new__(ONNXGeoModel)
        model._model_height = 32
        model._model_width = 32

        grad = np.linspace(100, 10000, 32 * 32, dtype=np.uint16).reshape(32, 32)
        data = np.stack([grad, grad, grad], axis=0)
        snapshot = data.copy()

        tensor = model._prepare_input(data)
        np.testing.assert_array_equal(data, snapshot)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertGreater(len(np.unique(tensor)), 50)


class TestBlendingFixes(unittest.TestCase):
    """Group 5: geoai.inference and geoai.canopy weight map fixes."""

    def test_inference_blend_weight_overlap_one_no_zero_seam(self):
        from geoai.inference import BlendMode, _blend_weight_1d

        for mode in (BlendMode.LINEAR, BlendMode.COSINE, BlendMode.SPLINE):
            w = _blend_weight_1d(32, overlap=1, mode=mode, power=2)
            self.assertTrue(np.all(w > 0.0), f"Zero weight seam for mode={mode}")
            np.testing.assert_array_equal(w, np.ones(32, dtype=np.float32))

    def test_canopy_make_weight_map_overlap_one_and_large_overlap(self):
        from geoai.canopy import CanopyHeightEstimation

        w1 = CanopyHeightEstimation._make_weight_map(32, 1)
        np.testing.assert_array_equal(w1, np.ones((32, 32), dtype=np.float32))

        w_large = CanopyHeightEstimation._make_weight_map(64, 40)
        np.testing.assert_allclose(w_large, w_large[::-1, :], atol=1e-6)
        np.testing.assert_allclose(w_large, w_large[:, ::-1], atol=1e-6)
        row = w_large[32, :]
        half = len(row) // 2
        self.assertTrue(np.all(np.diff(row[:half]) >= -1e-6))
        self.assertTrue(np.all(np.diff(row[half:]) <= 1e-6))


class TestExportTilesAugmentationFix(unittest.TestCase):
    """Group 6: geoai.utils.training export_geotiff_tiles augmentation ID fix."""

    def test_export_geotiff_tiles_augmentation_no_overwrite(self):
        import albumentations as A
        from geoai.utils.training import export_geotiff_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.zeros((3, 64, 64), dtype=np.uint8)
            img[:, :32, :32] = 10
            img[:, :32, 32:] = 50
            img[:, 32:, :32] = 100
            img[:, 32:, 32:] = 150

            mask = np.ones((64, 64), dtype=np.uint8)
            in_tif = _write_geotiff(os.path.join(tmpdir, "img.tif"), img)
            in_msk = _write_geotiff(os.path.join(tmpdir, "msk.tif"), mask)
            out_dir = os.path.join(tmpdir, "tiles")

            export_geotiff_tiles(
                in_raster=in_tif,
                out_folder=out_dir,
                in_class_data=in_msk,
                tile_size=32,
                stride=32,
                apply_augmentation=True,
                augmentation_count=2,
                augmentation_transforms=A.Compose([A.HorizontalFlip(p=1.0)]),
                quiet=True,
            )

            img_files = sorted(os.listdir(os.path.join(out_dir, "images")))
            lbl_files = sorted(os.listdir(os.path.join(out_dir, "labels")))
            # 4 original tiles * (1 original + 2 augmented) = 12 files
            self.assertEqual(len(img_files), 12)
            self.assertEqual(len(lbl_files), 12)

            expected_means = {0: 10, 3: 50, 6: 100, 9: 150}
            for tid, val in expected_means.items():
                path = os.path.join(out_dir, "images", f"tile_{tid:06d}.tif")
                with rasterio.open(path) as ds:
                    arr = ds.read()
                self.assertEqual(int(arr[0, 0, 0]), val)


class TestSlidingWindowFixes(unittest.TestCase):
    """Group 7: geoai.moondream and geoai.vllm_geo sliding window fixes."""

    def _assert_sliding_window_behavior(self, fn):
        # Invalid overlap / window_size
        with self.assertRaises(ValueError):
            fn(1000, 1000, window_size=512, overlap=512)
        with self.assertRaises(ValueError):
            fn(1000, 1000, window_size=512, overlap=-1)
        with self.assertRaises(ValueError):
            fn(1000, 1000, window_size=0, overlap=0)

        # Full coverage when right/bottom remainder < window_size // 2
        w, h = 1000, 600
        windows = fn(w, h, window_size=512, overlap=64)
        self.assertEqual(len(windows), len(set(windows)))
        covered = np.zeros((h, w), dtype=bool)
        for x0, y0, x1, y1 in windows:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, w)
            self.assertLessEqual(y1, h)
            covered[y0:y1, x0:x1] = True
        self.assertTrue(covered.all())

        # Narrow strip image (width < window_size // 2) must not drop all windows
        sw, sh = 200, 1200
        strip_windows = fn(sw, sh, window_size=512, overlap=64)
        self.assertGreater(len(strip_windows), 0)
        strip_cov = np.zeros((sh, sw), dtype=bool)
        for x0, y0, x1, y1 in strip_windows:
            strip_cov[y0:y1, x0:x1] = True
        self.assertTrue(strip_cov.all())

    def test_moondream_create_sliding_windows(self):
        from geoai.moondream import MoondreamGeo

        proc = MoondreamGeo.__new__(MoondreamGeo)
        self._assert_sliding_window_behavior(proc._create_sliding_windows)

    def test_vllm_geo_create_sliding_windows(self):
        from geoai.vllm_geo import VLLMGeo

        proc = VLLMGeo(model_id="test", base_url="http://localhost:8000/v1")
        self._assert_sliding_window_behavior(proc._create_sliding_windows)


class TestStacToolsExtraBracesFix(unittest.TestCase):
    """Group 8: geoai.agents.stac_tools extra closing braces fix."""

    def test_search_items_removes_single_extra_closing_brace(self):
        import importlib
        import sys
        import types
        import geoai

        def _fake_tool(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn

        fake_strands = types.ModuleType("strands")
        fake_strands.tool = _fake_tool

        fake_agents_pkg = types.ModuleType("geoai.agents")
        fake_agents_pkg.__path__ = [
            os.path.join(os.path.dirname(geoai.__file__), "agents")
        ]

        overrides = {
            "strands": fake_strands,
            "geoai.agents": fake_agents_pkg,
        }
        with patch.dict(sys.modules, overrides):
            sys.modules.pop("geoai.agents.stac_tools", None)
            stac_tools_mod = importlib.import_module("geoai.agents.stac_tools")
            tools = stac_tools_mod.STACTools()
            with patch.object(
                stac_tools_mod, "pc_stac_search", return_value=[]
            ) as mock_search:
                tools.search_items(
                    collection="sentinel-2-l2a",
                    query='{"eo:cloud_cover": {"lt": 10}}}',
                )
                mock_search.assert_called_once()
                self.assertEqual(
                    mock_search.call_args.kwargs["query"],
                    {"eo:cloud_cover": {"lt": 10}},
                )
            sys.modules.pop("geoai.agents.stac_tools", None)


if __name__ == "__main__":
    unittest.main()
