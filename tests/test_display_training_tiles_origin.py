"""Tests for orientation-aware ``imshow`` origin in ``display_training_tiles``.

Issue #703: south-up rasters (positive y-scale in the affine transform) were
being rendered upside-down because ``imshow`` defaulted to ``origin='upper'``.
``display_training_tiles`` now derives ``origin`` from ``src.transform.e`` in
the ``show_axes=True`` branch. These tests pin that behavior for both image
and mask paths in both orientations.
"""

import matplotlib

matplotlib.use("Agg")  # Headless backend so plt.show is a no-op in CI.

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from geoai.utils.visualization import display_training_tiles


def _write_tile(path, data, transform):
    """Write a small single-tile GeoTIFF used as a fixture.

    Args:
        path: Destination path for the GeoTIFF.
        data: Array of shape (bands, H, W).
        transform: Rasterio affine transform.
    """
    bands, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(data)


def _build_tile_pair(output_dir, transform):
    """Create one image tile and matching mask tile under ``output_dir``."""
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = np.random.randint(0, 255, size=(3, 8, 8), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(1, 8, 8), dtype=np.uint8)

    _write_tile(images_dir / "tile.tif", image, transform)
    _write_tile(masks_dir / "tile.tif", mask, transform)


# Affine ordering: (a, b, c, d, e, f) with ``e`` as the y-scale.
# Non-zero translations keep GDAL from warning the matrix is identity-like.
NORTH_UP = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 208.0)  # e < 0
SOUTH_UP = Affine(1.0, 0.0, 100.0, 0.0, 1.0, 200.0)  # e > 0


@pytest.mark.parametrize(
    "transform,expected_origin",
    [(NORTH_UP, "upper"), (SOUTH_UP, "lower")],
)
def test_display_training_tiles_origin_matches_transform(
    tmp_path, transform, expected_origin
):
    """Both image and mask imshow calls pick origin from the affine y-scale."""
    _build_tile_pair(tmp_path, transform)

    fig, axes = display_training_tiles(str(tmp_path), num_tiles=1, show_axes=True)
    try:
        image_axes_image = axes[0, 0].get_images()[0]
        mask_axes_image = axes[1, 0].get_images()[0]

        assert image_axes_image.origin == expected_origin
        assert mask_axes_image.origin == expected_origin
    finally:
        plt.close(fig)
