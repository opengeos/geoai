#!/usr/bin/env python

"""Tests for CLIPSegmentation in geoai.segment module."""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Helper: create a synthetic raster file
# ---------------------------------------------------------------------------


def _create_test_raster(
    path,
    height=64,
    width=64,
    bands=3,
    dtype="uint8",
    crs="EPSG:4326",
):
    """Write a small synthetic raster to *path*."""
    transform = from_bounds(0, 0, 1, 1, width, height)
    if dtype == "uint8":
        data = np.random.randint(1, 255, (bands, height, width), dtype=np.uint8)
    else:
        data = np.random.rand(bands, height, width).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Helper: create a mock CLIPSegmentation without loading real models
# ---------------------------------------------------------------------------


def _mock_clip_seg(**kwargs):
    """Return a CLIPSegmentation instance with mocked model and processor."""
    from geoai.segment import CLIPSegmentation

    with patch.object(CLIPSegmentation, "__init__", lambda self, **kw: None):
        seg = CLIPSegmentation()

    seg.device = kwargs.get("device", "cpu")
    seg.tile_size = kwargs.get("tile_size", 512)
    seg.overlap = kwargs.get("overlap", 32)
    seg.processor = MagicMock()
    seg.model = MagicMock()
    return seg


# ===================================================================
# Module-level checks
# ===================================================================


class TestCLIPSegmentationModule(unittest.TestCase):
    """Verify CLIPSegmentation is accessible and exported."""

    def test_class_in_all(self):
        from geoai.segment import __all__

        self.assertIn("CLIPSegmentation", __all__)

    def test_class_is_callable(self):
        from geoai.segment import CLIPSegmentation

        self.assertTrue(callable(CLIPSegmentation))

    def test_logger_exists(self):
        """Module should have a logger configured."""
        from geoai import segment

        self.assertTrue(hasattr(segment, "logger"))
        self.assertIsInstance(segment.logger, logging.Logger)


# ===================================================================
# Device / init tests
# ===================================================================


class TestCLIPSegmentationInit(unittest.TestCase):
    """Tests for __init__ device handling."""

    def test_uses_get_device(self):
        """__init__ should call get_device when device is None."""
        from geoai.segment import CLIPSegmentation
        import inspect

        source = inspect.getsource(CLIPSegmentation.__init__)
        self.assertIn("get_device", source)
        self.assertNotIn("torch.cuda.is_available", source)

    def test_docstring_mentions_mps(self):
        """Docstring should mention MPS device option."""
        from geoai.segment import CLIPSegmentation

        doc = CLIPSegmentation.__init__.__doc__
        self.assertIn("mps", doc.lower())

    def test_init_logger_not_print(self):
        """__init__ should use logger, not print."""
        from geoai.segment import CLIPSegmentation
        import inspect

        source = inspect.getsource(CLIPSegmentation.__init__)
        self.assertNotIn("print(", source)
        self.assertIn("logger.info", source)


# ===================================================================
# Normalization tests (via segment_image)
# ===================================================================


class TestCLIPSegNormalization(unittest.TestCase):
    """Tests for tile normalization inside segment_image."""

    def _run_segment(self, raster_path, seg=None):
        """Run segment_image with mocked model, return output path."""
        import torch

        if seg is None:
            seg = _mock_clip_seg(tile_size=64, overlap=0)

        # Mock processor to return tensor-like dict
        def fake_processor(**kwargs):
            result = MagicMock()
            result.to = MagicMock(return_value=result)
            return result

        seg.processor = MagicMock(side_effect=fake_processor)

        # Mock model forward pass
        fake_logits = torch.zeros(1, 16, 16)
        fake_output = MagicMock()
        fake_output.logits = fake_logits.unsqueeze(
            0
        )  # shape: [1, 1, 16, 16] -> [1, 16, 16]
        seg.model = MagicMock(return_value=fake_output)
        seg.model.__call__ = MagicMock(return_value=fake_output)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            output_path = f.name
        try:
            result = seg.segment_image(
                input_path=raster_path,
                output_path=output_path,
                text_prompt="water",
            )
            self.assertEqual(result, output_path)
            self.assertTrue(os.path.isfile(output_path))
            return output_path
        finally:
            if os.path.isfile(output_path):
                os.unlink(output_path)

    def test_uint8_three_band(self):
        """uint8 3-band rasters should pass through without re-normalization."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path, bands=3, dtype="uint8")
            self._run_segment(raster_path)
        finally:
            os.unlink(raster_path)

    def test_float_three_band(self):
        """float32 3-band rasters should be normalized per-channel."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path, bands=3, dtype="float32")
            self._run_segment(raster_path)
        finally:
            os.unlink(raster_path)

    def test_single_band(self):
        """Single-band rasters should be replicated to 3 channels."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path, bands=1, dtype="uint8")
            self._run_segment(raster_path)
        finally:
            os.unlink(raster_path)

    def test_two_band(self):
        """2-band rasters should be handled (replicate first band)."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path, bands=2, dtype="uint8")
            self._run_segment(raster_path)
        finally:
            os.unlink(raster_path)

    def test_four_band_uses_first_three(self):
        """>3 band rasters should use first 3 bands."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path, bands=4, dtype="uint8")
            self._run_segment(raster_path)
        finally:
            os.unlink(raster_path)


# ===================================================================
# segment_image output validation
# ===================================================================


class TestCLIPSegOutput(unittest.TestCase):
    """Test segment_image output format and content."""

    def test_output_has_two_bands(self):
        """Output GeoTIFF should have 2 bands: binary mask + probabilities."""
        import torch

        seg = _mock_clip_seg(tile_size=64, overlap=0)

        def fake_processor(**kwargs):
            result = MagicMock()
            result.to = MagicMock(return_value=result)
            return result

        seg.processor = MagicMock(side_effect=fake_processor)

        fake_logits = torch.zeros(1, 16, 16)
        fake_output = MagicMock()
        fake_output.logits = fake_logits.unsqueeze(0)
        seg.model = MagicMock(return_value=fake_output)
        seg.model.__call__ = MagicMock(return_value=fake_output)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            output_path = f.name
        try:
            _create_test_raster(raster_path, height=32, width=32, bands=3)
            seg.segment_image(raster_path, output_path, "water")
            with rasterio.open(output_path) as src:
                self.assertEqual(src.count, 2)
                self.assertEqual(src.dtypes[0], "float32")
        finally:
            os.unlink(raster_path)
            if os.path.isfile(output_path):
                os.unlink(output_path)


# ===================================================================
# segment_image_batch
# ===================================================================


class TestCLIPSegBatch(unittest.TestCase):
    """Tests for segment_image_batch."""

    def test_creates_output_dir(self):
        """segment_image_batch should create the output directory."""
        import torch

        seg = _mock_clip_seg(tile_size=64, overlap=0)

        def fake_processor(**kwargs):
            result = MagicMock()
            result.to = MagicMock(return_value=result)
            return result

        seg.processor = MagicMock(side_effect=fake_processor)

        fake_logits = torch.zeros(1, 16, 16)
        fake_output = MagicMock()
        fake_output.logits = fake_logits.unsqueeze(0)
        seg.model = MagicMock(return_value=fake_output)
        seg.model.__call__ = MagicMock(return_value=fake_output)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        output_dir = tempfile.mkdtemp()
        nested_dir = os.path.join(output_dir, "subdir")

        try:
            _create_test_raster(raster_path, height=32, width=32, bands=3)
            results = seg.segment_image_batch([raster_path], nested_dir, "water")
            self.assertTrue(os.path.isdir(nested_dir))
            self.assertEqual(len(results), 1)
        finally:
            os.unlink(raster_path)
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)


# ===================================================================
# Logging (no print)
# ===================================================================


class TestCLIPSegLogging(unittest.TestCase):
    """Verify CLIPSegmentation uses logger, not print."""

    def test_segment_image_no_print(self):
        """segment_image should use logger, not print."""
        from geoai.segment import CLIPSegmentation
        import inspect

        source = inspect.getsource(CLIPSegmentation.segment_image)
        self.assertNotIn("print(", source)

    def test_segment_image_batch_no_print(self):
        """segment_image_batch should not use print."""
        from geoai.segment import CLIPSegmentation
        import inspect

        source = inspect.getsource(CLIPSegmentation.segment_image_batch)
        self.assertNotIn("print(", source)


if __name__ == "__main__":
    unittest.main()
