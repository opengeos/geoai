"""Tests that timm_semantic_segmentation predicts every pixel, whatever the raster size."""

import os

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

smp = pytest.importorskip("segmentation_models_pytorch")

from geoai.timm_segment import timm_semantic_segmentation  # noqa: E402


@pytest.fixture(scope="module")
def class_one_model(tmp_path_factory):
    """State dict of a small U-Net whose head predicts class 1 everywhere."""
    model = smp.create_model(
        arch="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    head = model.segmentation_head[0]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([0.0, 5.0]))
    path = tmp_path_factory.mktemp("model") / "model.pth"
    torch.save(model.state_dict(), path)
    return str(path)


def _write_geotiff(path, height, width):
    data = (np.random.default_rng(0).random((3, height, width)) * 255).astype(np.uint8)
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500000, 4000000, 10, 10),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


# (height, width): with the default overlap of 256, the first four have at least one
# side no longer than the overlap; the last two did not lose any pixels before the fix.
SHAPES = [(200, 200), (256, 256), (256, 1000), (1000, 256), (300, 300), (1000, 1000)]


@pytest.mark.parametrize("height, width", SHAPES)
def test_every_pixel_is_predicted(tmp_path, class_one_model, height, width):
    src = os.path.join(tmp_path, "in.tif")
    dst = os.path.join(tmp_path, "out.tif")
    prob = os.path.join(tmp_path, "prob.tif")
    _write_geotiff(src, height, width)
    timm_semantic_segmentation(
        src,
        dst,
        class_one_model,
        encoder_name="resnet18",
        architecture="unet",
        num_channels=3,
        num_classes=2,
        window_size=512,
        overlap=256,
        device="cpu",
        quiet=True,
        probability_path=prob,
    )
    with rasterio.open(dst) as f:
        mask = f.read(1)
    with rasterio.open(prob) as f:
        probs = f.read()
    assert mask.shape == (height, width)
    assert (mask == 1).all(), f"{int((mask != 1).sum())} pixels were not predicted"
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-4)


@pytest.mark.parametrize("overlap", [512, 600, -1])
def test_invalid_overlap_raises(tmp_path, overlap):
    src = os.path.join(tmp_path, "in.tif")
    dst = os.path.join(tmp_path, "out.tif")
    _write_geotiff(src, 300, 300)
    with pytest.raises(ValueError, match="overlap"):
        timm_semantic_segmentation(
            src,
            dst,
            os.path.join(tmp_path, "unused.pth"),
            window_size=512,
            overlap=overlap,
            device="cpu",
            quiet=True,
        )
    assert not os.path.exists(dst)
