import contextlib

#!/usr/bin/env python
"""Regression tests for Issue #1013 Batch A (core ML modules)."""

import builtins
import inspect
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from rasterio.transform import from_origin
from torch.utils.data import DataLoader, TensorDataset


def _write_geotiff(
    path,
    data,
    nodata=None,
    crs="EPSG:32617",
    origin=(500000.0, 4000000.0),
):
    """Write a CHW or HW numpy array as a GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, height, width = data.shape
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(origin[0], origin[1], 1.0, 1.0),
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


class TestTimmRegressFixes(unittest.TestCase):
    """Regression tests for geoai.timm_regress fixes."""

    def test_pooled_regression_metrics_exact_and_constant_safe(self):
        pl = pytest.importorskip("lightning.pytorch")
        pytest.importorskip("segmentation_models_pytorch")
        from geoai.timm_regress import PixelRegressionModel, _PooledRegressionMetric

        class FixedRegressor(PixelRegressionModel):
            def forward(self, x):
                # Channel 0 carries the fixed prediction
                return x[:, 0] + 0.0 * next(self.model.parameters()).sum()

        # Batch 0: constant target 1.0, pred 1.5 (old per-batch R2 exploded to -5e7)
        # Batch 1: constant target 3.0, pred 2.5
        targets = torch.stack([torch.full((4, 4), 1.0), torch.full((4, 4), 3.0)], dim=0)
        preds = torch.stack([torch.full((4, 4), 1.5), torch.full((4, 4), 2.5)], dim=0)
        x = preds.unsqueeze(1).repeat(1, 3, 1, 1)

        # Pooled ground-truth values: rmse=0.5, mae=0.5, r2=1 - 0.25/1.0 = 0.75
        for batch_size in (1, 2):
            model = FixedRegressor(
                encoder_name="resnet18", encoder_weights=None, in_channels=3
            )
            # Metric state must not leak into checkpoints
            self.assertFalse(
                any(
                    k.startswith(("train_", "val_", "test_"))
                    for k in model.state_dict()
                )
            )

            loader = DataLoader(TensorDataset(x, targets), batch_size=batch_size)
            trainer = pl.Trainer(
                accelerator="cpu",
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
            )
            val_out = trainer.validate(model, loader, verbose=False)[0]
            self.assertAlmostEqual(val_out["val_rmse"], 0.5, places=5)
            self.assertAlmostEqual(val_out["val_mae"], 0.5, places=5)
            self.assertAlmostEqual(val_out["val_r2"], 0.75, places=5)

            test_out = trainer.test(model, loader, verbose=False)[0]
            self.assertAlmostEqual(test_out["test_rmse"], 0.5, places=5)
            self.assertAlmostEqual(test_out["test_mae"], 0.5, places=5)
            self.assertAlmostEqual(test_out["test_r2"], 0.75, places=5)

        # Constant target across entire epoch must not explode to -1e8
        m_imperfect = _PooledRegressionMetric("r2")
        m_imperfect.update(torch.full((4,), 6.0), torch.full((4,), 5.0))
        self.assertAlmostEqual(m_imperfect.compute().item(), 0.0, places=6)

        m_perfect = _PooledRegressionMetric("r2")
        m_perfect.update(torch.full((4,), 5.0), torch.full((4,), 5.0))
        self.assertAlmostEqual(m_perfect.compute().item(), 1.0, places=6)

        # Tiny-scale targets (~1e-7) must not be misclassified as constant
        m_tiny = _PooledRegressionMetric("r2")
        t_tiny = torch.tensor([1e-7, 2e-7, 3e-7], dtype=torch.float32)
        p_tiny = torch.tensor([1.1e-7, 2.0e-7, 2.9e-7], dtype=torch.float32)
        m_tiny.update(p_tiny, t_tiny)
        self.assertAlmostEqual(m_tiny.compute().item(), 0.99, places=4)

        # Large-mean elevation targets (1000m +/- 1m) across multiple batches
        # must avoid catastrophic cancellation in both float64 and float32 (MPS).
        def _sim_mps_apply(t):
            if t.dtype == torch.float64:
                raise TypeError(
                    "Cannot convert a MPS Tensor to float64 dtype as the MPS "
                    "framework doesn't support float64."
                )
            return t

        for use_mps_sim in (False, True):
            m_elev = _PooledRegressionMetric("r2")
            if use_mps_sim:
                m_elev._apply(_sim_mps_apply)
                self.assertEqual(m_elev.sum_sq_err.dtype, torch.float32)
                self.assertEqual(m_elev.mean_y.dtype, torch.float32)
                self.assertEqual(m_elev.m2_y.dtype, torch.float32)
                # reset() must keep float32 defaults on MPS
                m_elev.reset()
                self.assertEqual(m_elev.sum_sq_err.dtype, torch.float32)
            b1_t = torch.tensor([999.0, 1000.0, 1001.0], dtype=torch.float32)
            b2_t = torch.tensor([999.0, 1000.0, 1001.0], dtype=torch.float32)
            b1_p = b1_t + 0.5
            b2_p = b2_t - 0.5
            m_elev.update(b1_p, b1_t)
            m_elev.update(b2_p, b2_t)
            # ss_tot = 4.0, ss_res = 6 * 0.25 = 1.5 -> R2 = 1 - 1.5/4.0 = 0.625
            self.assertAlmostEqual(m_elev.compute().item(), 0.625, places=4)

        # Constant large-mean target in float32 (MPS) returns 1.0 / 0.0
        m_const_mps = _PooledRegressionMetric("r2")
        m_const_mps._apply(_sim_mps_apply)
        m_const_mps.update(
            torch.full((16,), 1000.3, dtype=torch.float32),
            torch.full((16,), 1000.3, dtype=torch.float32),
        )
        self.assertAlmostEqual(m_const_mps.compute().item(), 1.0, places=5)
        m_const_mps.reset()
        m_const_mps.update(
            torch.full((16,), 1001.3, dtype=torch.float32),
            torch.full((16,), 1000.3, dtype=torch.float32),
        )
        self.assertAlmostEqual(m_const_mps.compute().item(), 0.0, places=5)

    def test_create_regression_tiles_preserves_target_nodata_on_clip(self):
        from geoai.timm_regress import create_regression_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            in_arr = np.full((3, 16, 16), 0.5, dtype=np.float32)
            tgt_arr = np.ones((16, 16), dtype=np.float32)
            # 12.5% NoData (-9999), 87.5% valid (1.0) -> passes min_valid_ratio=0.8
            tgt_arr[:2, :] = -9999.0
            in_tif = _write_geotiff(os.path.join(tmpdir, "in.tif"), in_arr)
            tgt_tif = _write_geotiff(
                os.path.join(tmpdir, "tgt.tif"), tgt_arr, nodata=-9999.0
            )
            imgs, tgts = create_regression_tiles(
                in_tif,
                tgt_tif,
                os.path.join(tmpdir, "out"),
                tile_size=16,
                min_valid_ratio=0.8,
                target_min=0.0,
                target_max=2.0,
            )
            self.assertEqual(len(imgs), 1)
            self.assertEqual(len(tgts), 1)
            with rasterio.open(tgts[0]) as ds:
                self.assertEqual(ds.nodata, -9999.0)
                saved = ds.read(1)
            np.testing.assert_array_equal(saved[:2, :], -9999.0)
            np.testing.assert_array_equal(saved[2:, :], 1.0)

    def test_dataset_and_predict_raster_share_finite_input_nodata_cleanup(self):
        from geoai.timm_regress import PixelRegressionDataset, predict_raster

        with tempfile.TemporaryDirectory() as tmpdir:
            in_arr = np.full((3, 16, 16), 0.5, dtype=np.float32)
            # Finite NoData sentinel 65535 must NOT force /10000 scaling on valid 0.5 pixels
            in_arr[:, 0, 0] = 65535.0
            tgt_arr = np.ones((16, 16), dtype=np.float32)
            in_tif = _write_geotiff(
                os.path.join(tmpdir, "in.tif"), in_arr, nodata=65535.0
            )
            tgt_tif = _write_geotiff(os.path.join(tmpdir, "tgt.tif"), tgt_arr)

            ds = PixelRegressionDataset([in_tif], [tgt_tif], normalize_input=True)
            train_img, _ = ds[0]

            captured = []

            class CaptureModel(nn.Module):
                def forward(self, x):
                    captured.append(x[0].detach().cpu().clone())
                    return torch.zeros((x.shape[0], x.shape[2], x.shape[3]))

            predict_raster(
                CaptureModel(),
                in_tif,
                os.path.join(tmpdir, "pred.tif"),
                tile_size=16,
                overlap=0,
                device="cpu",
                use_model_preprocessing=False,
            )
            pred_img = captured[0]

            # Both training and prediction see 0.5 on valid pixels and 0.0 on NoData pixels
            self.assertAlmostEqual(float(train_img[0, 5, 5]), 0.5, places=5)
            self.assertAlmostEqual(float(train_img[0, 0, 0]), 0.0, places=5)
            torch.testing.assert_close(train_img, pred_img)

    def test_predict_raster_overlap_validation_dedup_streaming_nodata(self):
        from geoai.timm_regress import _create_weight_mask, predict_raster

        # _create_weight_mask handles overlap <= 1 and overlap > tile_size // 2
        w1 = _create_weight_mask(16, 1)
        np.testing.assert_array_equal(w1, np.ones((16, 16), dtype=np.float32))

        w_large = _create_weight_mask(
            16, 12, top=True, bottom=True, left=True, right=True
        )
        np.testing.assert_allclose(w_large, w_large[::-1, :], atol=1e-6)
        np.testing.assert_allclose(w_large, w_large[:, ::-1], atol=1e-6)

        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tile_count = 0

            def forward(self, x):
                self.tile_count += x.shape[0]
                return torch.full((x.shape[0], x.shape[2], x.shape[3]), 4.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.full((3, 20, 20), 0.5, dtype=np.float32)
            data[:, 0, 0] = np.nan
            data[:, 1, 1] = -9999.0
            src = _write_geotiff(os.path.join(tmpdir, "in.tif"), data, nodata=-9999.0)
            dst = os.path.join(tmpdir, "out.tif")

            # Invalid overlap must raise ValueError
            for bad_overlap in (-1, 16, 20):
                with self.assertRaises(ValueError):
                    predict_raster(
                        CountingModel(),
                        src,
                        dst,
                        tile_size=16,
                        overlap=bad_overlap,
                        device="cpu",
                        use_model_preprocessing=False,
                    )

            # Track full-raster reads (window=None) vs windowed reads
            orig_read = rasterio.io.DatasetReader.read
            full_reads = []

            def spy_read(self_ds, *args, **kwargs):
                if kwargs.get("window") is None:
                    full_reads.append(args)
                return orig_read(self_ds, *args, **kwargs)

            model = CountingModel()
            with patch.object(rasterio.io.DatasetReader, "read", spy_read):
                predict_raster(
                    model,
                    src,
                    dst,
                    tile_size=16,
                    overlap=8,
                    batch_size=4,
                    device="cpu",
                    output_nodata=-1234.0,
                    use_model_preprocessing=False,
                )

            # Must not read full raster into memory
            self.assertEqual(len(full_reads), 0)
            # 20x20 with tile_size=16, stride=8 has only 2x2=4 unique tiles (not 9)
            self.assertEqual(model.tile_count, 4)

            with rasterio.open(dst) as ds:
                out = ds.read(1)
            self.assertEqual(out[0, 0], -1234.0)
            self.assertEqual(out[1, 1], -1234.0)
            self.assertAlmostEqual(float(out[10, 10]), 4.2, places=4)

    def test_plot_training_history_numeric_version_sorting(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from geoai.timm_regress import plot_training_history

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_root = os.path.join(tmpdir, "models", "lightning_logs")
            for ver, val_r2 in [(2, 0.2), (10, 0.95)]:
                vdir = os.path.join(logs_root, f"version_{ver}")
                os.makedirs(vdir)
                with open(os.path.join(vdir, "metrics.csv"), "w") as f:
                    f.write("epoch,train_loss_epoch,val_loss,train_r2,val_r2\n")
                    f.write(f"0,0.5,0.4,0.1,{val_r2}\n")

            with patch.object(plt, "show"):
                fig, df = plot_training_history(tmpdir, metrics=["r2"])
            plt.close(fig)
            # Must pick version_10 (val_r2=0.95), not lexicographically larger version_2
            self.assertAlmostEqual(float(df["val_r2"].iloc[-1]), 0.95, places=5)

    def test_visualize_prediction_handles_zero_nodata(self):
        import matplotlib

        matplotlib.use("Agg")
        from geoai.timm_regress import visualize_prediction

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mostly 0 (nodata=0) pixels plus valid pixels in [100, 200]
            rgb = np.zeros((3, 10, 10), dtype=np.float32)
            rgb[:, :2, :2] = 100.0
            rgb[:, 2:4, :2] = 200.0
            in_tif = _write_geotiff(os.path.join(tmpdir, "in.tif"), rgb, nodata=0.0)
            pred_tif = _write_geotiff(
                os.path.join(tmpdir, "pred.tif"),
                np.ones((10, 10), dtype=np.float32),
                nodata=-9999.0,
            )

            seen_mins = []
            orig_percentile = np.percentile

            def spy_percentile(a, q, *args, **kwargs):
                arr = np.asarray(a)
                if arr.size > 0 and q == [2, 98]:
                    seen_mins.append(float(arr.min()))
                return orig_percentile(a, q, *args, **kwargs)

            with patch("numpy.percentile", side_effect=spy_percentile):
                visualize_prediction(in_tif, pred_tif)

            self.assertEqual(len(seen_mins), 3)
            # Nodata=0 pixels must have been excluded from RGB stretch
            for m in seen_mins:
                self.assertGreaterEqual(m, 100.0)

    def test_create_regression_tiles_nodata_small_raster_and_zero_valid(self):
        from geoai.timm_regress import create_regression_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) Respects nodata=0 on input/target and avoids ZeroDivisionError
            in_arr = np.zeros((3, 16, 16), dtype=np.float32)
            tgt_arr = np.zeros((16, 16), dtype=np.float32)
            in_tif = _write_geotiff(os.path.join(tmpdir, "in0.tif"), in_arr, nodata=0.0)
            tgt_tif = _write_geotiff(
                os.path.join(tmpdir, "tgt0.tif"), tgt_arr, nodata=0.0
            )
            imgs, tgts = create_regression_tiles(
                in_tif,
                tgt_tif,
                os.path.join(tmpdir, "out0"),
                tile_size=16,
                min_valid_ratio=0.0,
                target_min=-1.0,
                target_max=1.0,
            )
            self.assertEqual(imgs, [])
            self.assertEqual(tgts, [])

            # 2) Small raster (< tile_size) clamps expected tile count log to >= 1
            small_in = _write_geotiff(
                os.path.join(tmpdir, "small_in.tif"),
                np.ones((3, 8, 8), dtype=np.float32),
            )
            small_tgt = _write_geotiff(
                os.path.join(tmpdir, "small_tgt.tif"),
                np.ones((8, 8), dtype=np.float32),
            )
            with patch("geoai.timm_regress.logger") as mock_logger:
                create_regression_tiles(
                    small_in,
                    small_tgt,
                    os.path.join(tmpdir, "out_small"),
                    tile_size=32,
                    stride=16,
                )
            expected_calls = [
                c.args
                for c in mock_logger.info.call_args_list
                if c.args and "Expected tiles" in str(c.args[0])
            ]
            self.assertEqual(len(expected_calls), 1)
            _, ny, nx, total = expected_calls[0]
            self.assertGreaterEqual(ny, 1)
            self.assertGreaterEqual(nx, 1)
            self.assertGreaterEqual(total, 1)

    def test_lightning_modules_import_without_lightning(self):
        """timm_regress, timm_segment, and timm_train import cleanly without lightning."""
        code = """
import builtins, sys
real_import = builtins.__import__
def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "lightning" or name.startswith("lightning.") or name == "torchmetrics" or name.startswith("torchmetrics."):
        raise ImportError(f"blocked {name}")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = blocked_import
import geoai.timm_regress as tr
import geoai.timm_segment as ts
import geoai.timm_train as tt
assert tr.LIGHTNING_AVAILABLE is False
assert ts.LIGHTNING_AVAILABLE is False
assert tt.LIGHTNING_AVAILABLE is False
print("OK")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("OK", proc.stdout)


class TestTimmSegmentFixes(unittest.TestCase):
    """Regression tests for geoai.timm_segment fixes."""

    def test_timm_semantic_segmentation_batches_borders_nodata_and_safe_load(self):
        from geoai.timm_segment import timm_semantic_segmentation

        class BatchSpyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, kernel_size=1)
                self.batch_sizes = []

            def forward(self, x):
                self.batch_sizes.append(int(x.shape[0]))
                n, _, h, w = x.shape
                out = torch.zeros((n, 2, h, w), dtype=torch.float32, device=x.device)
                out[:, 1] = 5.0
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            ref = BatchSpyModel()
            ckpt = os.path.join(tmpdir, "model.pth")
            torch.save(ref.state_dict(), ckpt)

            # Float32 input with incompatible uint8 nodata (-9999.0)
            src = _write_geotiff(
                os.path.join(tmpdir, "in.tif"),
                np.full((3, 24, 24), 0.5, dtype=np.float32),
                nodata=-9999.0,
            )
            dst = os.path.join(tmpdir, "out.tif")

            spy_model = BatchSpyModel()
            load_calls = []
            orig_load = torch.load

            def spy_load(*args, **kwargs):
                load_calls.append(kwargs)
                return orig_load(*args, **kwargs)

            with (
                patch(
                    "segmentation_models_pytorch.create_model", return_value=spy_model
                ),
                patch("geoai.timm_segment.torch.load", side_effect=spy_load),
            ):
                timm_semantic_segmentation(
                    src,
                    dst,
                    ckpt,
                    encoder_name="resnet18",
                    architecture="unet",
                    num_channels=3,
                    num_classes=2,
                    window_size=16,
                    overlap=8,
                    batch_size=4,
                    device="cpu",
                    quiet=True,
                )

            # Must batch all 4 windows in a single forward call
            self.assertEqual(spy_model.batch_sizes, [4])
            # Must pass weights_only=True on torch.load
            self.assertTrue(len(load_calls) >= 1)
            self.assertTrue(load_calls[0].get("weights_only"))

            # Output must exist, be uint8, have all 1s, and cleared incompatible nodata
            with rasterio.open(dst) as ds:
                self.assertEqual(ds.dtypes[0], "uint8")
                self.assertIsNone(ds.nodata)
                np.testing.assert_array_equal(
                    ds.read(1), np.ones((24, 24), dtype=np.uint8)
                )

            # Also test overlap=1 works cleanly without dividing by 0.5 or crashing
            dst_ov1 = os.path.join(tmpdir, "out_ov1.tif")
            with patch(
                "segmentation_models_pytorch.create_model", return_value=BatchSpyModel()
            ):
                timm_semantic_segmentation(
                    src,
                    dst_ov1,
                    ckpt,
                    encoder_name="resnet18",
                    architecture="unet",
                    num_channels=3,
                    num_classes=2,
                    window_size=16,
                    overlap=1,
                    batch_size=2,
                    device="cpu",
                    quiet=True,
                )
            with rasterio.open(dst_ov1) as ds:
                np.testing.assert_array_equal(
                    ds.read(1), np.ones((24, 24), dtype=np.uint8)
                )


class TestTimmTrainFixes(unittest.TestCase):
    """Regression tests for geoai.timm_train fixes."""

    def test_freeze_backbone_freezes_vit_mlp_fc_layers_and_fallback(self):
        from geoai.timm_train import TimmClassifier

        class FakeViT(nn.Module):
            def __init__(self, return_head=True):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "mlp": nn.ModuleDict(
                                    {
                                        "fc1": nn.Linear(8, 16),
                                        "fc2": nn.Linear(16, 8),
                                    }
                                )
                            }
                        )
                    ]
                )
                self.head = nn.Linear(8, 3)
                self._return_head = return_head

            def get_classifier(self):
                return self.head if self._return_head else None

        # Primary path: via get_classifier()
        clf = TimmClassifier.__new__(TimmClassifier)
        nn.Module.__init__(clf)
        clf.model = FakeViT(return_head=True)
        clf._freeze_backbone()

        for name, param in clf.model.named_parameters():
            if name.startswith("head."):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Fallback path: when get_classifier() returns None
        clf_fb = TimmClassifier.__new__(TimmClassifier)
        nn.Module.__init__(clf_fb)
        clf_fb.model = FakeViT(return_head=False)
        clf_fb._freeze_backbone()

        for name, param in clf_fb.model.named_parameters():
            if name.startswith("head."):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

    def test_freeze_backbone_unfreezes_tuple_classifier_heads(self):
        from geoai.timm_train import TimmClassifier

        class FakeDistilledDeiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "mlp": nn.ModuleDict(
                                    {
                                        "fc1": nn.Linear(8, 16),
                                        "fc2": nn.Linear(16, 8),
                                    }
                                )
                            }
                        )
                    ]
                )
                self.head = nn.Linear(8, 3)
                self.head_dist = nn.Linear(8, 3)

            def get_classifier(self):
                return (self.head, self.head_dist)

        clf = TimmClassifier.__new__(TimmClassifier)
        nn.Module.__init__(clf)
        clf.model = FakeDistilledDeiT()
        clf._freeze_backbone()

        for name, param in clf.model.named_parameters():
            if name.startswith(("head.", "head_dist.")):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")


class TestTrainFixes(unittest.TestCase):
    """Regression tests for geoai.train fixes."""

    def test_multiclass_detection_validation_padding_compact_masks_and_nodata(self):
        from geoai.train import multiclass_detection_inference_on_geotiff

        class ShapeCheckingDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_shapes = []

            def forward(self, images):
                out = []
                for img in images:
                    self.seen_shapes.append(tuple(img.shape))
                    _, h, w = img.shape
                    mask = torch.zeros((1, 1, h, w), dtype=torch.float32)
                    mask[:, :, 2:6, 2:6] = 0.9
                    out.append(
                        {
                            "scores": torch.tensor([0.95]),
                            "boxes": torch.tensor([[2.0, 2.0, 6.0, 6.0]]),
                            "labels": torch.tensor([2]),
                            "masks": mask,
                        }
                    )
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            # Image smaller than window_size (10x12 < 16x16), with negative nodata
            src = _write_geotiff(
                os.path.join(tmpdir, "small.tif"),
                np.full((3, 10, 12), 100, dtype=np.int16),
                nodata=-9999,
            )
            dst = os.path.join(tmpdir, "out.tif")

            # Invalid overlap raises ValueError
            for bad_overlap in (-1, 16, 20):
                with self.assertRaises(ValueError):
                    multiclass_detection_inference_on_geotiff(
                        ShapeCheckingDetector(),
                        src,
                        dst,
                        window_size=16,
                        overlap=bad_overlap,
                        device=torch.device("cpu"),
                    )

            detector = ShapeCheckingDetector()
            out_path, _, dets = multiclass_detection_inference_on_geotiff(
                detector,
                src,
                dst,
                window_size=16,
                overlap=4,
                batch_size=2,
                num_channels=3,
                device=torch.device("cpu"),
            )
            # Edge tile must be padded to (3, 16, 16)
            self.assertEqual(detector.seen_shapes, [(3, 16, 16)])
            self.assertEqual(len(dets), 1)
            # Detection retains compact mask + mask_offset instead of full-raster copy
            self.assertIn("mask_offset", dets[0])
            self.assertEqual(dets[0]["mask_offset"], (0, 0, 10, 12))
            self.assertEqual(dets[0]["mask"].shape, (10, 12))

            # Output GeoTIFF must succeed with incompatible nodata cleared
            with rasterio.open(out_path) as ds:
                self.assertEqual(ds.dtypes[0], "uint32")
                self.assertIsNone(ds.nodata)
                cls_band = ds.read(1)
                inst_band = ds.read(2)
            self.assertEqual(int(cls_band[3, 3]), 2)
            self.assertEqual(int(inst_band[3, 3]), 1)

    def test_semantic_inference_weights_nodata_and_nested_parent_dir(self):
        from geoai.train import (
            _compute_window_weight,
            semantic_inference_on_geotiff,
            semantic_inference_on_image,
        )

        # Outer raster border keeps full weight 1.0; interior edge tapers
        w = _compute_window_weight(
            h=16, w=16, y_pos=0, x_pos=0, height=32, width=32, overlap=8
        )
        np.testing.assert_array_equal(w[:12, :12], 1.0)
        self.assertLess(float(w[-1, -1]), 1.0)

        # overlap <= 1 gives uniform 1.0 weights
        np.testing.assert_array_equal(
            _compute_window_weight(16, 16, 4, 4, 32, 32, overlap=1),
            np.ones((16, 16), dtype=np.float32),
        )

        class AlwaysClassOne(nn.Module):
            def forward(self, x):
                n, _, h, w = x.shape
                out = torch.zeros((n, 3, h, w), dtype=torch.float32)
                out[:, 1] = 5.0
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            # GeoTIFF with incompatible nodata=-9999.0 -> cleared to None
            src_neg = _write_geotiff(
                os.path.join(tmpdir, "neg.tif"),
                np.full((3, 16, 16), 0.5, dtype=np.float32),
                nodata=-9999.0,
            )
            dst_neg = os.path.join(tmpdir, "out_neg.tif")
            semantic_inference_on_geotiff(
                AlwaysClassOne(),
                src_neg,
                dst_neg,
                window_size=16,
                overlap=1,
                num_channels=3,
                num_classes=3,
                device=torch.device("cpu"),
                quiet=True,
            )
            with rasterio.open(dst_neg) as ds:
                self.assertIsNone(ds.nodata)
                np.testing.assert_array_equal(
                    ds.read(1), np.ones((16, 16), dtype=np.uint8)
                )

            # GeoTIFF with valid uint8 nodata=0 -> preserved
            src_zero = _write_geotiff(
                os.path.join(tmpdir, "zero.tif"),
                np.full((3, 16, 16), 100, dtype=np.uint8),
                nodata=0,
            )
            dst_zero = os.path.join(tmpdir, "out_zero.tif")
            semantic_inference_on_geotiff(
                AlwaysClassOne(),
                src_zero,
                dst_zero,
                window_size=16,
                overlap=0,
                num_channels=3,
                num_classes=3,
                device=torch.device("cpu"),
                quiet=True,
            )
            with rasterio.open(dst_zero) as ds:
                self.assertEqual(ds.nodata, 0.0)

            # semantic_inference_on_image creates nested parent dir when binary_output=False
            img_path = os.path.join(tmpdir, "in.png")
            Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8)).save(img_path)
            nested_out = os.path.join(tmpdir, "nested", "sub", "mask.png")
            semantic_inference_on_image(
                AlwaysClassOne(),
                img_path,
                nested_out,
                window_size=16,
                overlap=1,
                num_channels=3,
                num_classes=3,
                binary_output=False,
                device=torch.device("cpu"),
                quiet=True,
            )
            self.assertTrue(os.path.exists(nested_out))

    def test_evaluate_records_zero_iou_on_missed_gt_or_false_positives(self):
        from geoai.train import evaluate

        class SplitDetector(nn.Module):
            def __init__(self, preds):
                super().__init__()
                self.preds = preds

            def forward(self, images, targets=None):
                if targets is not None:
                    return {"loss": torch.tensor(0.1)}
                return self.preds

        # Box mode: img 0 perfect (1.0), img 1 missed GT (0.0), img 2 FP on empty GT (0.0), img 3 both empty (skip)
        box_preds = [
            {"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]])},
            {"boxes": torch.zeros((0, 4))},
            {"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]])},
            {"boxes": torch.zeros((0, 4))},
        ]
        box_targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]])},
            {"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]])},
            {"boxes": torch.zeros((0, 4))},
            {"boxes": torch.zeros((0, 4))},
        ]
        imgs = [torch.zeros(3, 8, 8) for _ in range(4)]
        box_metrics = evaluate(
            SplitDetector(box_preds),
            [(imgs, box_targets)],
            torch.device("cpu"),
            use_mask_iou=False,
        )
        self.assertAlmostEqual(box_metrics["IoU"], 1.0 / 3.0, places=5)

        # Mask mode: same 4 cases
        one_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
        one_tgt = torch.ones((1, 8, 8), dtype=torch.float32)
        empty_mask = torch.zeros((0, 1, 8, 8), dtype=torch.float32)
        empty_tgt = torch.zeros((0, 8, 8), dtype=torch.float32)
        mask_preds = [
            {"masks": one_mask},
            {"masks": empty_mask},
            {"masks": one_mask},
            {"masks": empty_mask},
        ]
        mask_targets = [
            {"masks": one_tgt},
            {"masks": one_tgt},
            {"masks": empty_tgt},
            {"masks": empty_tgt},
        ]
        mask_metrics = evaluate(
            SplitDetector(mask_preds),
            [(imgs, mask_targets)],
            torch.device("cpu"),
            use_mask_iou=True,
        )
        self.assertAlmostEqual(mask_metrics["IoU"], 1.0 / 3.0, places=5)

    def test_train_maskrcnn_and_instance_segmentation_wiring(self):
        import geoai.train as train_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            for idx in range(2):
                _write_geotiff(
                    os.path.join(img_dir, f"chip_{idx}.tif"),
                    np.zeros((4, 16, 16), dtype=np.uint8),
                )
                _write_geotiff(
                    os.path.join(lbl_dir, f"chip_{idx}.tif"),
                    np.zeros((16, 16), dtype=np.uint8),
                )

            captured_datasets = []
            eval_kwargs = []
            orig_ds = train_mod.ObjectDetectionDataset

            def spy_ds(*args, **kwargs):
                ds = orig_ds(*args, **kwargs)
                captured_datasets.append(kwargs)
                return ds

            def spy_eval(*args, **kwargs):
                eval_kwargs.append(kwargs)
                # Return IoU=0.0 so best_model.pth is NOT created
                return {"loss": 0.5, "IoU": 0.0}

            with (
                patch.object(train_mod, "ObjectDetectionDataset", side_effect=spy_ds),
                patch.object(train_mod, "train_one_epoch", return_value=0.4),
                patch.object(train_mod, "evaluate", side_effect=spy_eval),
            ):
                train_mod.train_MaskRCNN_model(
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    output_dir=out_dir,
                    num_channels=4,
                    model=nn.Linear(1, 1),
                    model_name="fasterrcnn_resnet50_fpn_v2",
                    batch_size=1,
                    num_epochs=1,
                    pretrained=False,
                    device=torch.device("cpu"),
                    verbose=False,
                )

            # num_channels must be forwarded to both train and val datasets
            self.assertEqual(len(captured_datasets), 2)
            self.assertEqual(captured_datasets[0]["num_channels"], 4)
            self.assertEqual(captured_datasets[1]["num_channels"], 4)
            # Final evaluate must pass use_mask_iou=False for Faster R-CNN and not crash without best_model.pth
            self.assertEqual(len(eval_kwargs), 2)
            self.assertIs(eval_kwargs[-1].get("use_mask_iou"), False)

            # train_instance_segmentation_model must forward num_classes
            with (
                patch.object(
                    train_mod,
                    "get_instance_segmentation_model",
                    return_value=nn.Linear(1, 1),
                ),
                patch.object(train_mod, "train_MaskRCNN_model") as mock_train,
            ):
                train_mod.train_instance_segmentation_model(
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    output_dir=out_dir,
                    num_classes=5,
                    num_channels=4,
                )
            mock_train.assert_called_once()
            self.assertEqual(mock_train.call_args.kwargs["num_classes"], 5)

    def test_lightly_train_model_default_method_and_lr_forwarding(self):
        import geoai.train as train_mod

        sig = inspect.signature(train_mod.lightly_train_model)
        self.assertEqual(sig.parameters["method"].default, "simclr")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(data_dir)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                os.path.join(data_dir, "a.png")
            )

            fake_lt = types.SimpleNamespace(train=MagicMock())
            with (
                patch.object(train_mod, "LIGHTLY_TRAIN_AVAILABLE", True),
                patch.object(train_mod, "lightly_train", fake_lt, create=True),
            ):
                # Default call must NOT raise ValueError and must forward learning_rate
                train_mod.lightly_train_model(
                    data_dir=data_dir,
                    output_dir=out_dir,
                    learning_rate=3e-4,
                )
                kwargs_1 = fake_lt.train.call_args.kwargs
                self.assertEqual(kwargs_1["method"], "simclr")
                self.assertEqual(kwargs_1["optim_args"]["lr"], 3e-4)

                # Explicit optim_args["lr"] must be preserved without mutating caller dict
                user_optim = {"lr": 1e-3, "weight_decay": 0.01}
                train_mod.lightly_train_model(
                    data_dir=data_dir,
                    output_dir=out_dir,
                    learning_rate=3e-4,
                    optim_args=user_optim,
                )
                kwargs_2 = fake_lt.train.call_args.kwargs
                self.assertEqual(kwargs_2["optim_args"]["lr"], 1e-3)
                self.assertEqual(kwargs_2["optim_args"]["weight_decay"], 0.01)

    def test_semantic_random_rotation90_nonsquare_and_binary_negative_clamp(self):
        from geoai.train import SemanticRandomRotation90, SemanticSegmentationDataset

        rot = SemanticRandomRotation90(prob=1.0)
        img = torch.randn(3, 8, 16)
        msk = torch.zeros(8, 16, dtype=torch.long)
        for _ in range(10):
            r_img, r_msk = rot(img, msk)
            self.assertEqual(tuple(r_img.shape), (3, 8, 16))
            self.assertEqual(tuple(r_msk.shape), (8, 16))

        ds = SemanticSegmentationDataset(
            [], [], num_channels=3, num_classes=2, ignore_index=-100
        )
        raw = np.array([[-5, -1, 0], [1, -100, 0]], dtype=np.int16)
        norm = ds._normalize_label_mask(raw)
        np.testing.assert_array_equal(
            norm, np.array([[0, 0, 0], [1, -100, 0]], dtype=np.int64)
        )

    def test_safe_torch_load_prefers_weights_only_and_requires_opt_in(self):
        import pickle
        import geoai._torch_io as torch_io
        import geoai.landcover_train as lc_mod
        import geoai.timm_segment as seg_mod
        import geoai.train as train_mod

        # All modules share the single consolidated helper
        self.assertIs(train_mod._safe_torch_load, torch_io._safe_torch_load)
        self.assertIs(lc_mod._safe_torch_load, torch_io._safe_torch_load)
        self.assertIs(seg_mod._safe_torch_load, torch_io._safe_torch_load)
        self.assertIs(
            train_mod._clear_incompatible_nodata, torch_io._clear_incompatible_nodata
        )
        self.assertIs(
            seg_mod._clear_incompatible_nodata, torch_io._clear_incompatible_nodata
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "w.pth")
            torch.save({"a": torch.tensor([1.0])}, path)
            calls = []
            orig_load = torch.load

            def spy_load(*args, **kwargs):
                calls.append(kwargs)
                return orig_load(*args, **kwargs)

            with patch("geoai._torch_io.torch.load", side_effect=spy_load):
                loaded = train_mod._safe_torch_load(path, map_location="cpu")

            self.assertTrue(calls[0].get("weights_only"))
            self.assertEqual(loaded["a"].item(), 1.0)

            # Restricted load failure must NOT silently retry weights_only=False
            unsafe_calls = []

            def failing_weights_only(*args, **kwargs):
                unsafe_calls.append(kwargs)
                if kwargs.get("weights_only"):
                    raise pickle.UnpicklingError("Weights only load failed")
                return {"legacy": True}

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("geoai._torch_io.torch.load", side_effect=failing_weights_only),
            ):
                os.environ.pop("GEOAI_ALLOW_UNSAFE_TORCH_LOAD", None)
                with self.assertRaises(pickle.UnpicklingError) as ctx:
                    torch_io._safe_torch_load(path, map_location="cpu")
                self.assertIn("allow_unsafe=True", str(ctx.exception))
                self.assertIn("GEOAI_ALLOW_UNSAFE_TORCH_LOAD=1", str(ctx.exception))
                # Only the safe attempt ran
                self.assertEqual(len(unsafe_calls), 1)
                self.assertTrue(unsafe_calls[0].get("weights_only"))

                # Explicit allow_unsafe=True permits fallback
                res = torch_io._safe_torch_load(
                    path, map_location="cpu", allow_unsafe=True
                )
                self.assertEqual(res, {"legacy": True})

            # Environment variable opt-in permits fallback
            with (
                patch.dict(os.environ, {"GEOAI_ALLOW_UNSAFE_TORCH_LOAD": "1"}),
                patch("geoai._torch_io.torch.load", side_effect=failing_weights_only),
            ):
                res = torch_io._safe_torch_load(path, map_location="cpu")
                self.assertEqual(res, {"legacy": True})

            # Unrelated RuntimeError (e.g. corrupt file) must never trigger unsafe retry
            corrupt_calls = []

            def corrupt_load(*args, **kwargs):
                corrupt_calls.append(kwargs)
                raise RuntimeError("PytorchStreamReader failed reading zip archive")

            with patch("geoai._torch_io.torch.load", side_effect=corrupt_load):
                with self.assertRaises(RuntimeError):
                    torch_io._safe_torch_load(
                        path, map_location="cpu", allow_unsafe=True
                    )
                self.assertEqual(len(corrupt_calls), 1)


class TestSegmentationFixes(unittest.TestCase):
    """Regression tests for geoai.segmentation fixes."""

    def test_normalize_mask_binary_clamps_negatives(self):
        from geoai.segmentation import _normalize_mask

        mask = np.array([[-5, -1], [0, 1]], dtype=np.int16)
        result = _normalize_mask(mask, num_classes=2)
        np.testing.assert_array_equal(
            result, np.array([[0, 0], [0, 1]], dtype=np.int64)
        )

    def test_target_size_hw_and_val_has_no_random_augmentations(self):
        import albumentations as A
        from geoai.segmentation import CustomDataset, get_transform, prepare_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            msk_dir = os.path.join(tmpdir, "masks")
            os.makedirs(img_dir)
            os.makedirs(msk_dir)

            # Create asymmetric images so flips/rotations would visibly change values
            for idx in range(5):
                arr = np.zeros((40, 80, 3), dtype=np.uint8)
                arr[:10, :20, :] = 255
                Image.fromarray(arr).save(os.path.join(img_dir, f"img_{idx}.png"))
                msk = np.zeros((40, 80), dtype=np.uint8)
                msk[:10, :20] = 255
                Image.fromarray(msk).save(os.path.join(msk_dir, f"img_{idx}.png"))

            # Non-square target_size=(height=32, width=64)
            ds = CustomDataset(img_dir, msk_dir, transform=None, target_size=(32, 64))
            sample = ds[0]
            self.assertEqual(sample["pixel_values"].shape, (32, 64, 3))
            self.assertEqual(tuple(sample["labels"].shape), (32, 64))

            # get_transform accepts target_size=(32, 64)
            tf = get_transform(target_size=(32, 64))
            train_ds, val_ds = prepare_datasets(
                img_dir, msk_dir, transform=tf, test_size=0.4, random_state=42
            )
            self.assertIsNot(train_ds.dataset, val_ds.dataset)

            # Validation transform must not contain random flip/rotate augmentations
            val_types = tuple(type(t) for t in val_ds.dataset.transform.transforms)
            self.assertNotIn(A.HorizontalFlip, val_types)
            self.assertNotIn(A.VerticalFlip, val_types)
            self.assertNotIn(A.RandomRotate90, val_types)

            # Validation output has exact (C, H, W) = (3, 32, 64) and is deterministic
            first = val_ds[0]
            self.assertEqual(tuple(first["pixel_values"].shape), (3, 32, 64))
            self.assertEqual(tuple(first["labels"].shape), (32, 64))
            for _ in range(5):
                again = val_ds[0]
                self.assertTrue(
                    torch.equal(first["pixel_values"], again["pixel_values"])
                )
                self.assertTrue(torch.equal(first["labels"], again["labels"]))


class TestLandcoverTrainFixes(unittest.TestCase):
    """Regression tests for geoai.landcover_train fixes."""

    def test_evaluate_sparse_iou_batched_streaming_matches_landcover_iou(self):
        from geoai.landcover_train import evaluate_sparse_iou, landcover_iou

        class BatchRecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, x):
                self.batch_sizes.append(int(x.shape[0]))
                n, _, h, w = x.shape
                out = torch.zeros((n, 3, h, w), dtype=torch.float32, device=x.device)
                out[:, 1, :2, :] = 10.0
                out[:, 2, 2:, :] = 10.0
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            all_targets = []
            for idx in range(5):
                img = np.full((3, 4, 4), 100, dtype=np.uint8)
                lbl = np.array(
                    [
                        [0, 0, 1, 1],
                        [1, 1, 2, 2],
                        [2, 2, 0, 0],
                        [1, 2, 2, 0],
                    ],
                    dtype=np.uint8,
                )
                if idx % 2 == 1:
                    lbl = np.rot90(lbl).copy()
                _write_geotiff(os.path.join(img_dir, f"tile_{idx}.tif"), img)
                _write_geotiff(os.path.join(lbl_dir, f"tile_{idx}.tif"), lbl)
                all_targets.append(torch.from_numpy(lbl.astype(np.int64)))

            model = BatchRecordingModel()
            with patch(
                "geoai.landcover_train.torch.cat",
                side_effect=AssertionError(
                    "evaluate_sparse_iou must stream, not torch.cat"
                ),
            ):
                res = evaluate_sparse_iou(
                    model=model,
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    num_classes=3,
                    num_channels=3,
                    batch_size=2,
                    background_class=0,
                    ignore_index=False,
                    device=torch.device("cpu"),
                    verbose=False,
                )

            # 5 tiles with batch_size=2 -> batches [2, 2, 1]
            self.assertEqual(model.batch_sizes, [2, 2, 1])

            # Compare against reference landcover_iou over stacked tensors
            stacked_targets = torch.stack(all_targets, dim=0)
            ref_preds = torch.zeros_like(stacked_targets)
            ref_preds[:, :2, :] = 1
            ref_preds[:, 2:, :] = 2
            ref_mean, ref_ious, ref_rec, ref_prec = landcover_iou(
                ref_preds,
                stacked_targets,
                num_classes=3,
                ignore_index=False,
                mode="sparse_labels",
                background_class=0,
            )
            self.assertAlmostEqual(res["mean_sparse_iou"], ref_mean, places=5)
            self.assertAlmostEqual(res["per_class_iou"][1], ref_ious[1], places=5)
            self.assertAlmostEqual(res["per_class_iou"][2], ref_ious[2], places=5)
            self.assertAlmostEqual(res["per_class_recall"][1], ref_rec[1], places=5)
            self.assertAlmostEqual(res["per_class_precision"][2], ref_prec[2], places=5)

    def test_evaluate_sparse_iou_rgb_png_labels_ignore_index_and_shape_mismatch(self):
        from geoai.landcover_train import evaluate_sparse_iou, landcover_iou

        class FixedThreeClassModel(nn.Module):
            def __init__(self, out_h=None):
                super().__init__()
                self.out_h = out_h

            def forward(self, x):
                n, _, h, w = x.shape
                oh = self.out_h if self.out_h is not None else h
                out = torch.zeros((n, 4, oh, w), dtype=torch.float32, device=x.device)
                out[:, 1, : min(2, oh), :] = 10.0
                if oh > 2:
                    out[:, 2, 2:, :] = 10.0
                return out

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)

            # 3-band RGB PNG labels (class IDs in band 0); class 3 has zero union
            lbl_2d = np.array(
                [
                    [0, 1, 1, 2],
                    [1, 1, 2, 0],
                    [2, 2, 1, 0],
                    [0, 2, 2, 1],
                ],
                dtype=np.uint8,
            )
            rgb_lbl = np.stack([lbl_2d, lbl_2d, lbl_2d], axis=-1)
            rgb_img = np.full((4, 4, 3), 128, dtype=np.uint8)
            Image.fromarray(rgb_img).save(os.path.join(img_dir, "tile_0.png"))
            Image.fromarray(rgb_lbl).save(os.path.join(lbl_dir, "tile_0.png"))

            # Parity check with both ignore_index=False and ignore_index=2
            for ign in (False, 2):
                res = evaluate_sparse_iou(
                    model=FixedThreeClassModel(),
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    num_classes=4,
                    num_channels=3,
                    batch_size=2,
                    background_class=0,
                    ignore_index=ign,
                    device=torch.device("cpu"),
                    verbose=False,
                )
                ref_targets = torch.from_numpy(lbl_2d.astype(np.int64)).unsqueeze(0)
                ref_preds = torch.zeros_like(ref_targets)
                ref_preds[:, :2, :] = 1
                ref_preds[:, 2:, :] = 2
                ref_mean, ref_ious, ref_rec, ref_prec = landcover_iou(
                    ref_preds,
                    ref_targets,
                    num_classes=4,
                    ignore_index=ign,
                    mode="sparse_labels",
                    background_class=0,
                )
                self.assertAlmostEqual(res["mean_sparse_iou"], ref_mean, places=6)
                for cls, val in res["per_class_iou"].items():
                    self.assertAlmostEqual(val, ref_ious[cls], places=6)
                    self.assertAlmostEqual(
                        res["per_class_recall"][cls], ref_rec[cls], places=6
                    )
                    self.assertAlmostEqual(
                        res["per_class_precision"][cls], ref_prec[cls], places=6
                    )

            # Shape mismatch (1, 1, 4) vs (1, 4, 4) must raise ValueError, not broadcast
            with self.assertRaises(ValueError):
                evaluate_sparse_iou(
                    model=FixedThreeClassModel(out_h=1),
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    num_classes=4,
                    num_channels=3,
                    batch_size=1,
                    background_class=0,
                    ignore_index=False,
                    device=torch.device("cpu"),
                    verbose=False,
                )

    def test_train_segmentation_landcover_multispectral_resize_cpu_gating_and_safe_load(
        self,
    ):
        from geoai.landcover_train import train_segmentation_landcover

        class TinySegNet(nn.Module):
            def __init__(self, in_channels=3, classes=2, **kwargs):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, classes, kernel_size=1)

            def forward(self, x):
                return self.conv(x)

        for num_channels in (1, 6):
            with tempfile.TemporaryDirectory() as tmpdir:
                img_dir = os.path.join(tmpdir, "images")
                lbl_dir = os.path.join(tmpdir, "labels")
                out_dir = os.path.join(tmpdir, "out")
                os.makedirs(img_dir)
                os.makedirs(lbl_dir)

                for idx in range(4):
                    _write_geotiff(
                        os.path.join(img_dir, f"tile_{idx}.tif"),
                        np.full((num_channels, 8, 10), 100, dtype=np.uint8),
                    )
                    _write_geotiff(
                        os.path.join(lbl_dir, f"tile_{idx}.tif"),
                        np.ones((8, 10), dtype=np.uint8),
                    )

                ref = TinySegNet(in_channels=num_channels, classes=2)
                ckpt = os.path.join(tmpdir, "weights.pth")
                torch.save(ref.state_dict(), ckpt)

                load_calls = []
                orig_load = torch.load

                def spy_load(*args, **kwargs):
                    load_calls.append(kwargs)
                    return orig_load(*args, **kwargs)

                # Simulate multi-GPU host while explicitly requesting CPU device.
                # Keep torch.accelerator.is_available() False so PyTorch's internal
                # Adam.step() graph-capture health check does not call _cuda_init()
                # on CPU-only CI runners.
                extra_ctx = (
                    patch("torch.accelerator.is_available", return_value=False)
                    if hasattr(torch, "accelerator")
                    else contextlib.nullcontext()
                )
                with (
                    patch("segmentation_models_pytorch.Unet", side_effect=TinySegNet),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.device_count", return_value=2),
                    extra_ctx,
                    patch("geoai.landcover_train.torch.load", side_effect=spy_load),
                ):
                    model = train_segmentation_landcover(
                        images_dir=img_dir,
                        labels_dir=lbl_dir,
                        output_dir=out_dir,
                        architecture="unet",
                        encoder_weights=None,
                        num_channels=num_channels,
                        num_classes=2,
                        batch_size=2,
                        num_epochs=1,
                        target_size=(12, 16),
                        checkpoint_path=ckpt,
                        device=torch.device("cpu"),
                        num_workers=0,
                        verbose=False,
                    )

                # Must NOT wrap CPU model in DataParallel
                self.assertNotIsInstance(model, nn.DataParallel)
                # torch.load must use weights_only=True
                self.assertTrue(len(load_calls) >= 1)
                self.assertTrue(all(c.get("weights_only") for c in load_calls))


if __name__ == "__main__":
    unittest.main()
