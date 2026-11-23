"""Tests that sliding-window detection and instance-segmentation inference run every window."""

import os

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from geoai.train import (
    _sliding_window_starts,
    inference_on_geotiff,
    instance_segmentation_inference_on_geotiff,
)


class FullTileInstance(torch.nn.Module):
    """Mask R-CNN-like model: one instance covering each whole input tile.

    Counts the windows it sees.
    """

    def __init__(self):
        super().__init__()
        self.windows = 0

    def forward(self, images):
        self.windows += len(images)
        outputs = []
        for image in images:
            h, w = image.shape[-2:]
            outputs.append(
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[0.0, 0.0, float(w), float(h)]]),
                    "masks": torch.ones(1, 1, h, w),
                }
            )
        return outputs


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


def _expected_windows(height, width):
    return len(_sliding_window_starts(height, 512, 256)) * len(
        _sliding_window_starts(width, 512, 256)
    )


# (height, width): smaller than the window, one side smaller, a short last batch, full size
SHAPES = [(400, 400), (1300, 400), (400, 1300), (2000, 300), (1500, 1500)]


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("height, width", SHAPES)
def test_inference_on_geotiff_runs_every_window(tmp_path, height, width, batch_size):
    src = os.path.join(tmp_path, "in.tif")
    dst = os.path.join(tmp_path, "out.tif")
    _write_geotiff(src, height, width)
    model = FullTileInstance()
    inference_on_geotiff(
        model,
        src,
        dst,
        window_size=512,
        overlap=256,
        batch_size=batch_size,
        num_channels=3,
        device=torch.device("cpu"),
    )
    with rasterio.open(dst) as f:
        out = f.read(1)
    assert out.shape == (height, width)
    assert model.windows == _expected_windows(height, width)
    # The blend weight is 0 on each tile's outermost pixel, so the raster's own
    # 1-pixel border stays empty; every other pixel must be predicted.
    interior = out[1:-1, 1:-1]
    assert (
        interior == 1
    ).all(), f"{int((interior != 1).sum())} pixels were not predicted"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("height, width", SHAPES)
def test_instance_segmentation_runs_every_window(tmp_path, height, width, batch_size):
    src = os.path.join(tmp_path, "in.tif")
    dst = os.path.join(tmp_path, "out.tif")
    _write_geotiff(src, height, width)
    model = FullTileInstance()
    # The dummy instances of neighboring windows overlap, so NMS would drop some of
    # them. nms_threshold=1.0 keeps them all, and the raster then shows exactly which
    # pixels a window covered.
    instance_segmentation_inference_on_geotiff(
        model,
        src,
        dst,
        window_size=512,
        overlap=256,
        nms_threshold=1.0,
        batch_size=batch_size,
        num_channels=3,
        device=torch.device("cpu"),
    )
    with rasterio.open(dst) as f:
        out = f.read(1)
    assert out.shape == (height, width)
    assert model.windows == _expected_windows(height, width)
    assert (out > 0).all(), f"{int((out == 0).sum())} pixels were not predicted"
