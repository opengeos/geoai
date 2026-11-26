"""Tests for per-channel normalization in semantic segmentation training and inference."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

import geoai
import geoai.train as training


def _write_multiband_geotiff(path: Path, data: np.ndarray, nodata=None) -> Path:
    """Write a (C, H, W) array to a GeoTIFF."""
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(data.dtype),
        "crs": "EPSG:32614",
        "transform": from_origin(430000.0, 5230000.0, 1.0, 1.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def _make_float32_lidar_dataset(tmp_path: Path, n_tiles: int = 4, size: int = 64):
    """Create 6-band float32 GeoTIFFs (4 NAIP-scale + 2 LiDAR-scale) and binary masks."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    rng = np.random.default_rng(42)
    center = size // 2

    for i in range(n_tiles):
        h, w = size, size
        img = np.zeros((6, h, w), dtype=np.float32)
        # Bands 0..3: NAIP-like 0..255
        img[:4] = rng.uniform(20.0, 220.0, size=(4, h, w)).astype(np.float32)
        # Band 4: sink depth 0..3 m (deeper in center pothole)
        yy, xx = np.indices((h, w))
        pothole = (yy - center) ** 2 + (xx - center) ** 2 <= (size // 4) ** 2
        img[4] = np.where(pothole, 1.8, 0.0).astype(np.float32)
        # Band 5: TWI 4..20
        img[5] = np.where(pothole, 16.0, 6.0).astype(np.float32)
        if i == 0:
            img[5, 0, 0] = np.nan  # ensure NaN NoData is sanitized cleanly
        _write_multiband_geotiff(img_dir / f"tile_{i}.tif", img)

        mask = pothole.astype(np.uint8)[None, :, :]
        _write_multiband_geotiff(lbl_dir / f"tile_{i}.tif", mask)

    return img_dir, lbl_dir


def test_compute_channel_stats_and_lazy_export(tmp_path: Path):
    img_dir, _ = _make_float32_lidar_dataset(tmp_path)
    assert geoai.compute_channel_stats is training.compute_channel_stats

    stats = geoai.compute_channel_stats(str(img_dir), num_channels=6)
    assert len(stats["means"]) == 6
    assert len(stats["stds"]) == 6
    assert len(stats["mins"]) == 6
    assert len(stats["maxs"]) == 6
    assert all(np.isfinite(stats["means"]))
    assert all(s > 0 for s in stats["stds"])
    # Band 4 (depth) range should be [0.0, 1.8]
    assert stats["mins"][4] == pytest.approx(0.0, abs=1e-5)
    assert stats["maxs"][4] == pytest.approx(1.8, abs=1e-5)
    # Band 5 (TWI) should ignore NaN pixel
    assert stats["mins"][5] == pytest.approx(6.0, abs=1e-5)
    assert stats["maxs"][5] == pytest.approx(16.0, abs=1e-5)


def test_dataset_per_channel_minmax_and_zscore_normalization(tmp_path: Path):
    img_dir, lbl_dir = _make_float32_lidar_dataset(tmp_path, n_tiles=1)
    imgs = [str(img_dir / "tile_0.tif")]
    lbls = [str(lbl_dir / "tile_0.tif")]

    # Default legacy / 255.0
    ds_default = training.SemanticSegmentationDataset(imgs, lbls, num_channels=6)
    x_def, _ = ds_default[0]
    assert x_def.shape == (6, 64, 64)
    assert torch.isfinite(x_def).all()
    assert x_def[4, 32, 32].item() == pytest.approx(1.8 / 255.0, rel=1e-5)

    # Min-max per channel: NAIP [0,255], depth [0,3], TWI [0,20]
    ds_mm = training.SemanticSegmentationDataset(
        imgs,
        lbls,
        num_channels=6,
        channel_mins=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        channel_maxs=[255.0, 255.0, 255.0, 255.0, 3.0, 20.0],
    )
    x_mm, _ = ds_mm[0]
    assert torch.isfinite(x_mm).all()
    assert x_mm[4, 32, 32].item() == pytest.approx(1.8 / 3.0, rel=1e-5)
    assert x_mm[5, 32, 32].item() == pytest.approx(16.0 / 20.0, rel=1e-5)
    # NaN pixel sanitized to 0 before normalization
    assert x_mm[5, 0, 0].item() == pytest.approx(0.0, abs=1e-5)

    # Z-score per channel
    means = [120.0, 120.0, 120.0, 120.0, 0.5, 10.0]
    stds = [50.0, 50.0, 50.0, 50.0, 0.5, 5.0]
    ds_z = training.SemanticSegmentationDataset(
        imgs, lbls, num_channels=6, channel_means=means, channel_stds=stds
    )
    x_z, _ = ds_z[0]
    assert torch.isfinite(x_z).all()
    assert x_z[4, 32, 32].item() == pytest.approx((1.8 - 0.5) / 0.5, rel=1e-5)
    assert x_z[5, 32, 32].item() == pytest.approx((16.0 - 10.0) / 5.0, rel=1e-5)


def test_channel_normalization_validation_errors(tmp_path: Path):
    img_dir, lbl_dir = _make_float32_lidar_dataset(tmp_path, n_tiles=1)
    imgs = [str(img_dir / "tile_0.tif")]
    lbls = [str(lbl_dir / "tile_0.tif")]

    # Mixing zscore and minmax rejected
    with pytest.raises(ValueError, match="not both"):
        training.SemanticSegmentationDataset(
            imgs,
            lbls,
            num_channels=6,
            channel_means=[0.0] * 6,
            channel_maxs=[1.0] * 6,
        )

    # Wrong length rejected
    with pytest.raises(ValueError, match="length"):
        training.SemanticSegmentationDataset(
            imgs, lbls, num_channels=6, channel_stds=[1.0, 1.0]
        )

    # Non-positive std rejected
    with pytest.raises(ValueError, match="> 0"):
        training.SemanticSegmentationDataset(
            imgs, lbls, num_channels=6, channel_stds=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        )

    # max <= min rejected
    with pytest.raises(ValueError, match="greater than"):
        training.SemanticSegmentationDataset(
            imgs,
            lbls,
            num_channels=6,
            channel_mins=[0.0] * 6,
            channel_maxs=[255.0, 255.0, 255.0, 255.0, 0.0, 20.0],
        )


def test_train_and_inference_roundtrip_with_normalization_sidecar(tmp_path: Path):
    """Training writes normalization.json and semantic_segmentation auto-loads it."""
    img_dir, lbl_dir = _make_float32_lidar_dataset(tmp_path, n_tiles=4)
    out_dir = tmp_path / "model_out"
    channel_mins = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    channel_maxs = [255.0, 255.0, 255.0, 255.0, 3.0, 20.0]

    training.train_segmentation_model(
        images_dir=str(img_dir),
        labels_dir=str(lbl_dir),
        output_dir=str(out_dir),
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        num_channels=6,
        num_classes=2,
        batch_size=2,
        num_epochs=1,
        val_split=0.25,
        num_workers=0,
        device=torch.device("cpu"),
        verbose=False,
        channel_mins=channel_mins,
        channel_maxs=channel_maxs,
    )

    norm_file = out_dir / "normalization.json"
    assert norm_file.is_file()
    saved = json.loads(norm_file.read_text(encoding="utf-8"))
    assert saved["mode"] == "minmax"
    assert saved["mins"] == channel_mins
    assert saved["maxs"] == channel_maxs

    # Inference without explicitly repeating channel_mins/channel_maxs auto-loads sidecar
    pred_auto = tmp_path / "pred_auto.tif"
    prob_auto = tmp_path / "prob_auto.tif"
    training.semantic_segmentation(
        input_path=str(img_dir / "tile_0.tif"),
        output_path=str(pred_auto),
        probability_path=str(prob_auto),
        model_path=str(out_dir / "best_model.pth"),
        architecture="unet",
        encoder_name="resnet18",
        num_channels=6,
        num_classes=2,
        window_size=64,
        overlap=0,
        batch_size=1,
        device=torch.device("cpu"),
        quiet=True,
    )

    # Inference with explicit matching mins/maxs must yield identical probabilities
    pred_explicit = tmp_path / "pred_explicit.tif"
    prob_explicit = tmp_path / "prob_explicit.tif"
    training.semantic_segmentation(
        input_path=str(img_dir / "tile_0.tif"),
        output_path=str(pred_explicit),
        probability_path=str(prob_explicit),
        model_path=str(out_dir / "best_model.pth"),
        architecture="unet",
        encoder_name="resnet18",
        num_channels=6,
        num_classes=2,
        window_size=64,
        overlap=0,
        batch_size=1,
        device=torch.device("cpu"),
        quiet=True,
        channel_mins=channel_mins,
        channel_maxs=channel_maxs,
    )

    with rasterio.open(prob_auto) as s1, rasterio.open(prob_explicit) as s2:
        np.testing.assert_allclose(s1.read(), s2.read(), rtol=1e-6, atol=1e-6)


def test_compute_channel_stats_retains_inferred_channel_count(tmp_path: Path):
    img_a = np.full((3, 16, 16), 10.0, dtype=np.float32)
    img_b = np.full((5, 16, 16), 20.0, dtype=np.float32)
    p_a = _write_multiband_geotiff(tmp_path / "a.tif", img_a)
    p_b = _write_multiband_geotiff(tmp_path / "b.tif", img_b)

    stats = geoai.compute_channel_stats([p_a, p_b], num_channels=None)
    assert len(stats["means"]) == 3
    assert stats["means"] == pytest.approx([15.0, 15.0, 15.0])


def test_zscore_default_transform_does_not_clamp_negative_values():
    tf = training.get_semantic_transform(train=True, include_photometric=False)
    img = torch.tensor([[[-2.5, 3.5], [0.0, -1.0]]], dtype=torch.float32)
    mask = torch.zeros((2, 2), dtype=torch.long)
    out_img, _ = tf(img, mask)
    assert out_img.min().item() == pytest.approx(-2.5)
    assert out_img.max().item() == pytest.approx(3.5)


def test_sidecar_preserved_on_failed_run_and_cleared_on_legacy_retrain(tmp_path: Path):
    img_dir, lbl_dir = _make_float32_lidar_dataset(tmp_path, n_tiles=4)
    out_dir = tmp_path / "model_out"
    out_dir.mkdir()
    norm_file = out_dir / "normalization.json"
    original_cfg = {"mode": "minmax", "mins": [0.0] * 6, "maxs": [1.0] * 6}
    norm_file.write_text(json.dumps(original_cfg), encoding="utf-8")

    # Failed run (invalid architecture) must leave existing normalization.json intact
    with pytest.raises((ValueError, KeyError)):
        training.train_segmentation_model(
            images_dir=str(img_dir),
            labels_dir=str(lbl_dir),
            output_dir=str(out_dir),
            architecture="nonexistent_arch_xyz",
            encoder_name="resnet18",
            encoder_weights=None,
            num_channels=6,
            num_classes=2,
            batch_size=2,
            num_epochs=1,
            num_workers=0,
            device=torch.device("cpu"),
            verbose=False,
            channel_mins=[0.0] * 6,
            channel_maxs=[255.0] * 6,
        )
    assert json.loads(norm_file.read_text(encoding="utf-8")) == original_cfg

    # Successful legacy (/255.0) retrain into same directory clears stale normalization.json
    training.train_segmentation_model(
        images_dir=str(img_dir),
        labels_dir=str(lbl_dir),
        output_dir=str(out_dir),
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        num_channels=6,
        num_classes=2,
        batch_size=2,
        num_epochs=1,
        val_split=0.25,
        save_best_only=True,
        plot_curves=False,
        num_workers=0,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert not norm_file.exists()


def test_resolve_inference_normalization_rejects_mixed_legacy_and_sidecar(
    tmp_path: Path,
):
    dir_legacy = tmp_path / "legacy"
    dir_norm = tmp_path / "norm"
    dir_legacy.mkdir()
    dir_norm.mkdir()
    ckpt_a = dir_legacy / "best_model.pth"
    ckpt_b = dir_norm / "best_model.pth"
    ckpt_a.write_bytes(b"")
    ckpt_b.write_bytes(b"")
    (dir_norm / "normalization.json").write_text(
        json.dumps({"mode": "zscore", "means": [10.0, 20.0], "stds": [2.0, 4.0]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting normalization.json"):
        training._resolve_inference_normalization(
            model_path=[str(ckpt_a), str(ckpt_b)], num_channels=2
        )
