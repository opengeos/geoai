"""Tests that sliding-window semantic inference predicts every pixel of the input."""

import os

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin

from geoai.train import (
    _sliding_window_starts,
    semantic_inference_on_geotiff,
    semantic_inference_on_image,
)


class AlwaysClassOne(torch.nn.Module):
    """Predicts class 1 everywhere and counts the windows it sees."""

    def __init__(self):
        super().__init__()
        self.windows = 0

    def forward(self, x):
        self.windows += x.shape[0]
        logits = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)
        logits[:, 1] = 5.0
        return logits


class TestSlidingWindowStarts:
    @pytest.mark.parametrize("length", [1, 100, 256, 400, 512])
    def test_single_window_when_image_fits(self, length):
        assert _sliding_window_starts(length, 512, 256) == [0]

    @pytest.mark.parametrize("length", [513, 1000, 1300, 1500, 2048, 4097])
    def test_covers_every_pixel_without_repeats(self, length):
        starts = _sliding_window_starts(length, 512, 256)
        assert starts == sorted(set(starts))
        assert starts[0] == 0
        assert starts[-1] == length - 512
        assert all(b - a <= 256 for a, b in zip(starts, starts[1:]))

    def test_no_overlap(self):
        assert _sliding_window_starts(1024, 256, 0) == [0, 256, 512, 768]

    @pytest.mark.parametrize("overlap", [512, 600])
    def test_overlap_must_be_smaller_than_window(self, overlap):
        with pytest.raises(ValueError, match="overlap"):
            _sliding_window_starts(1000, 512, overlap)

    @pytest.mark.parametrize("overlap", [-1, -128])
    def test_negative_overlap_is_rejected(self, overlap):
        # A negative overlap steps further than the window (stride 640 for -128) and
        # would leave rows [512, 640) and [1152, 1280) of a 2048-pixel axis uncovered.
        with pytest.raises(ValueError, match="overlap"):
            _sliding_window_starts(2048, 512, overlap)


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


# (height, width): smaller than the window, one side smaller, a short last batch, full size
SHAPES = [(400, 400), (1300, 400), (400, 1300), (2000, 300), (1500, 1500), (2048, 2048)]


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("height, width", SHAPES)
def test_geotiff_inference_predicts_every_pixel(tmp_path, height, width, batch_size):
    src = os.path.join(tmp_path, "in.tif")
    dst = os.path.join(tmp_path, "out.tif")
    _write_geotiff(src, height, width)
    model = AlwaysClassOne()
    semantic_inference_on_geotiff(
        model,
        src,
        dst,
        window_size=512,
        overlap=256,
        batch_size=batch_size,
        num_channels=3,
        num_classes=2,
        device=torch.device("cpu"),
        quiet=True,
    )
    with rasterio.open(dst) as f:
        out = f.read(1)
    assert out.shape == (height, width)
    assert (out == 1).all(), f"{int((out != 1).sum())} pixels were not predicted"
    expected = len(_sliding_window_starts(height, 512, 256)) * len(
        _sliding_window_starts(width, 512, 256)
    )
    assert model.windows == expected


@pytest.mark.parametrize("height, width", [(400, 400), (1300, 400), (1500, 1500)])
def test_image_inference_predicts_every_pixel(tmp_path, height, width):
    src = os.path.join(tmp_path, "in.png")
    dst = os.path.join(tmp_path, "out.png")
    rgb = (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(src)
    model = AlwaysClassOne()
    semantic_inference_on_image(
        model,
        src,
        dst,
        window_size=512,
        overlap=256,
        batch_size=4,
        num_channels=3,
        num_classes=2,
        device=torch.device("cpu"),
        quiet=True,
    )
    out = np.array(Image.open(dst))
    assert out.shape[:2] == (height, width)
    assert (out > 0).all(), f"{int((out == 0).sum())} pixels were not predicted"


def test_negative_overlap_raises_instead_of_leaving_gaps(tmp_path):
    src = os.path.join(tmp_path, "in.tif")
    dst = os.path.join(tmp_path, "out.tif")
    _write_geotiff(src, 2048, 600)
    with pytest.raises(ValueError, match="overlap"):
        semantic_inference_on_geotiff(
            AlwaysClassOne(),
            src,
            dst,
            window_size=512,
            overlap=-128,
            num_channels=3,
            num_classes=2,
            device=torch.device("cpu"),
            quiet=True,
        )
    assert not os.path.exists(dst)
