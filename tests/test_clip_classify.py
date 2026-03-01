#!/usr/bin/env python

"""Tests for `geoai.clip_classify` module."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from geoai.clip_classify import (
    CLIPVectorClassifier,
    _to_rgb_uint8,
    clip_classify_vector,
)

# ---------------------------------------------------------------------------
# Helper to build synthetic rasters on disk
# ---------------------------------------------------------------------------


def _create_test_raster(path, height=50, width=50, bands=3, seed=42):
    """Write a small synthetic GeoTIFF for testing."""
    rng = np.random.default_rng(seed)
    data = (rng.random((bands, height, width)) * 255).astype(np.uint8)
    transform = from_bounds(0.0, 0.0, 1.0, 1.0, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": bands,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _mock_classifier():
    """Create a CLIPVectorClassifier with mocked model/processor."""
    with patch.object(CLIPVectorClassifier, "__init__", lambda self, **kw: None):
        clf = CLIPVectorClassifier.__new__(CLIPVectorClassifier)
        clf.model_name = "mock-model"
        clf.device = "cpu"
        clf.processor = MagicMock()
        clf.model = MagicMock()
        clf.model.logit_scale = MagicMock()
        clf.model.logit_scale.exp.return_value = 1.0
    return clf


# ===================================================================
# Import and export tests
# ===================================================================


class TestClipClassifyImport(unittest.TestCase):
    """Tests for clip_classify module import behavior."""

    def test_module_imports(self):
        """Test that the clip_classify module can be imported."""
        import geoai.clip_classify

        self.assertTrue(hasattr(geoai.clip_classify, "CLIPVectorClassifier"))
        self.assertTrue(hasattr(geoai.clip_classify, "clip_classify_vector"))

    def test_classifier_class_is_callable(self):
        """Test that CLIPVectorClassifier class is callable."""
        self.assertTrue(callable(CLIPVectorClassifier))

    def test_convenience_function_is_callable(self):
        """Test that clip_classify_vector function is callable."""
        self.assertTrue(callable(clip_classify_vector))


class TestClipClassifyExports(unittest.TestCase):
    """Tests for __all__ exports."""

    def test_all_is_defined(self):
        """Test that __all__ is defined."""
        import geoai.clip_classify

        self.assertTrue(hasattr(geoai.clip_classify, "__all__"))

    def test_all_contains_expected_names(self):
        """Test that __all__ contains the expected public API."""
        from geoai.clip_classify import __all__

        self.assertIn("CLIPVectorClassifier", __all__)
        self.assertIn("clip_classify_vector", __all__)

    def test_all_names_are_importable(self):
        """Test that every name in __all__ is importable."""
        import geoai.clip_classify

        for name in geoai.clip_classify.__all__:
            self.assertTrue(
                hasattr(geoai.clip_classify, name),
                f"{name} listed in __all__ but not found",
            )


# ===================================================================
# Signature tests
# ===================================================================


class TestClipClassifySignatures(unittest.TestCase):
    """Tests for class and function signatures."""

    def test_init_params(self):
        """Test CLIPVectorClassifier.__init__ has expected parameters."""
        sig = inspect.signature(CLIPVectorClassifier.__init__)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("device", sig.parameters)

    def test_init_defaults(self):
        """Test CLIPVectorClassifier.__init__ default values."""
        sig = inspect.signature(CLIPVectorClassifier.__init__)
        self.assertEqual(
            sig.parameters["model_name"].default,
            "openai/clip-vit-base-patch32",
        )
        self.assertIsNone(sig.parameters["device"].default)

    def test_classify_params(self):
        """Test classify method has expected parameters."""
        sig = inspect.signature(CLIPVectorClassifier.classify)
        expected = [
            "vector_data",
            "raster_path",
            "labels",
            "label_prefix",
            "top_k",
            "batch_size",
            "min_chip_size",
            "output_path",
            "quiet",
        ]
        for name in expected:
            self.assertIn(name, sig.parameters)

    def test_classify_defaults(self):
        """Test classify method default values."""
        sig = inspect.signature(CLIPVectorClassifier.classify)
        self.assertEqual(
            sig.parameters["label_prefix"].default, "a satellite image of "
        )
        self.assertEqual(sig.parameters["top_k"].default, 1)
        self.assertEqual(sig.parameters["batch_size"].default, 16)
        self.assertEqual(sig.parameters["min_chip_size"].default, 10)
        self.assertIsNone(sig.parameters["output_path"].default)
        self.assertFalse(sig.parameters["quiet"].default)

    def test_convenience_function_params(self):
        """Test clip_classify_vector function has expected parameters."""
        sig = inspect.signature(clip_classify_vector)
        expected = [
            "vector_data",
            "raster_path",
            "labels",
            "model_name",
            "device",
            "label_prefix",
            "top_k",
            "batch_size",
            "min_chip_size",
            "output_path",
            "quiet",
        ]
        for name in expected:
            self.assertIn(name, sig.parameters)

    def test_convenience_function_defaults(self):
        """Test clip_classify_vector default values."""
        sig = inspect.signature(clip_classify_vector)
        self.assertEqual(
            sig.parameters["model_name"].default,
            "openai/clip-vit-base-patch32",
        )
        self.assertEqual(sig.parameters["top_k"].default, 1)
        self.assertEqual(sig.parameters["batch_size"].default, 16)


# ===================================================================
# Validation tests (mocked model)
# ===================================================================


class TestClipClassifyValidation(unittest.TestCase):
    """Tests for input validation without actual model loading."""

    def test_empty_labels_raises(self):
        """Test that classify raises ValueError for empty labels."""
        clf = _mock_classifier()
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        with self.assertRaises(ValueError):
            clf.classify(vector_data=gdf, raster_path="fake.tif", labels=[])

    def test_nonexistent_raster_raises(self):
        """Test that classify raises FileNotFoundError for missing raster."""
        clf = _mock_classifier()
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        with self.assertRaises(FileNotFoundError):
            clf.classify(
                vector_data=gdf,
                raster_path="/nonexistent/path.tif",
                labels=["urban"],
            )

    def test_nonexistent_vector_file_raises(self):
        """Test that classify raises FileNotFoundError for missing vector."""
        clf = _mock_classifier()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path)
            with self.assertRaises(FileNotFoundError):
                clf.classify(
                    vector_data="/nonexistent/buildings.geojson",
                    raster_path=raster_path,
                    labels=["urban"],
                )
        finally:
            os.unlink(raster_path)

    def test_invalid_vector_type_raises(self):
        """Test that classify raises TypeError for invalid vector_data type."""
        clf = _mock_classifier()
        with self.assertRaises(TypeError):
            clf._load_vector(12345)

    def test_empty_gdf_returns_empty(self):
        """Test that an empty GeoDataFrame returns empty with new columns."""
        clf = _mock_classifier()
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path)
            result = clf.classify(
                vector_data=gdf,
                raster_path=raster_path,
                labels=["urban", "forest"],
            )
            self.assertIn("clip_label", result.columns)
            self.assertIn("clip_confidence", result.columns)
            self.assertEqual(len(result), 0)
        finally:
            os.unlink(raster_path)


# ===================================================================
# Lazy import registration
# ===================================================================


class TestLazyImportRegistration(unittest.TestCase):
    """Tests that clip_classify symbols are registered in geoai.__init__."""

    def test_symbols_in_lazy_map(self):
        """Test symbols are in _LAZY_SYMBOL_MAP."""
        import geoai

        self.assertIn("CLIPVectorClassifier", geoai._LAZY_SYMBOL_MAP)
        self.assertIn("clip_classify_vector", geoai._LAZY_SYMBOL_MAP)

    def test_submodule_in_lazy_submodules(self):
        """Test clip_classify is in _LAZY_SUBMODULES."""
        import geoai

        self.assertIn("clip_classify", geoai._LAZY_SUBMODULES)

    def test_symbols_in_dir(self):
        """Test symbols appear in dir(geoai)."""
        import geoai

        geoai_dir = dir(geoai)
        self.assertIn("CLIPVectorClassifier", geoai_dir)
        self.assertIn("clip_classify_vector", geoai_dir)


# ===================================================================
# _to_rgb_uint8 helper tests
# ===================================================================


class TestToRgbUint8(unittest.TestCase):
    """Tests for the _to_rgb_uint8 helper function."""

    def test_three_band_input(self):
        """Test 3-band raster data converts correctly."""
        data = np.array([[[100, 200]], [[50, 150]], [[0, 255]]], dtype=np.uint8)
        result = _to_rgb_uint8(data)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 2, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_multiband_uses_first_three(self):
        """Test >3 band raster uses only first 3 bands."""
        data = np.random.default_rng(42).integers(0, 255, (5, 10, 10), dtype=np.uint8)
        result = _to_rgb_uint8(data)
        self.assertEqual(result.shape, (10, 10, 3))

    def test_single_band_replication(self):
        """Test single-band raster is replicated to 3 channels."""
        data = np.array([[[100, 200], [50, 150]]], dtype=np.uint8)
        result = _to_rgb_uint8(data)
        self.assertEqual(result.shape, (2, 2, 3))
        # All channels should be identical
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_all_zeros_returns_none(self):
        """Test all-zero data returns None."""
        data = np.zeros((3, 10, 10), dtype=np.uint8)
        result = _to_rgb_uint8(data)
        self.assertIsNone(result)

    def test_empty_array_returns_none(self):
        """Test empty array returns None."""
        data = np.zeros((3, 0, 0), dtype=np.uint8)
        result = _to_rgb_uint8(data)
        self.assertIsNone(result)

    def test_float_raster_normalized(self):
        """Test float data is normalized to 0-255 uint8."""
        data = np.array([[[0.0, 1.0]], [[0.5, 0.5]], [[0.2, 0.8]]])
        result = _to_rgb_uint8(data)
        self.assertEqual(result.dtype, np.uint8)
        self.assertLessEqual(result.max(), 255)
        self.assertGreaterEqual(result.min(), 0)


# ===================================================================
# Chip extraction tests (with real rasters, mocked model)
# ===================================================================


class TestExtractChip(unittest.TestCase):
    """Tests for _extract_chip with synthetic rasters."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.raster_path = os.path.join(self.tmpdir, "test.tif")
        _create_test_raster(self.raster_path, height=100, width=100, bands=3)
        self.clf = _mock_classifier()

    def tearDown(self):
        os.unlink(self.raster_path)
        os.rmdir(self.tmpdir)

    def test_valid_chip_extraction(self):
        """Test that a valid polygon extracts a chip."""
        with rasterio.open(self.raster_path) as src:
            geom = box(0.1, 0.1, 0.5, 0.5)
            chip = self.clf._extract_chip(src, geom, min_chip_size=1)
        self.assertIsNotNone(chip)
        from PIL import Image

        self.assertIsInstance(chip, Image.Image)

    def test_out_of_bounds_returns_none(self):
        """Test that a polygon outside raster returns None."""
        with rasterio.open(self.raster_path) as src:
            geom = box(10.0, 10.0, 20.0, 20.0)
            chip = self.clf._extract_chip(src, geom, min_chip_size=1)
        self.assertIsNone(chip)

    def test_tiny_chip_returns_none(self):
        """Test that a very small polygon returns None when min_chip_size is large."""
        with rasterio.open(self.raster_path) as src:
            # Very small polygon: ~1 pixel
            geom = box(0.001, 0.001, 0.009, 0.009)
            chip = self.clf._extract_chip(src, geom, min_chip_size=50)
        self.assertIsNone(chip)

    def test_null_geometry_returns_none(self):
        """Test that None geometry returns None."""
        with rasterio.open(self.raster_path) as src:
            chip = self.clf._extract_chip(src, None, min_chip_size=1)
        self.assertIsNone(chip)

    def test_empty_geometry_returns_none(self):
        """Test that an empty geometry returns None."""
        from shapely.geometry import Polygon

        empty_geom = Polygon()
        with rasterio.open(self.raster_path) as src:
            chip = self.clf._extract_chip(src, empty_geom, min_chip_size=1)
        self.assertIsNone(chip)

    def test_narrow_chip_returns_none(self):
        """Test that a chip small in one dimension is rejected (or, not and)."""
        with rasterio.open(self.raster_path) as src:
            # Very narrow horizontally (~1 pixel wide) but tall
            geom = box(0.001, 0.1, 0.009, 0.9)
            chip = self.clf._extract_chip(src, geom, min_chip_size=5)
        self.assertIsNone(chip)

    def test_single_band_raster(self):
        """Test chip extraction from a single-band raster."""
        single_path = os.path.join(self.tmpdir, "single.tif")
        _create_test_raster(single_path, height=50, width=50, bands=1)
        try:
            with rasterio.open(single_path) as src:
                geom = box(0.1, 0.1, 0.9, 0.9)
                chip = self.clf._extract_chip(src, geom, min_chip_size=1)
            self.assertIsNotNone(chip)
            # Should be RGB (3 channels via replication)
            arr = np.array(chip)
            self.assertEqual(arr.shape[2], 3)
        finally:
            os.unlink(single_path)


# ===================================================================
# Batch classification end-to-end tests (mocked model)
# ===================================================================


class TestClipClassifyBatchPath(unittest.TestCase):
    """Tests that exercise the batch classification code path."""

    def test_batch_classification_top_k_and_flush(self):
        """End-to-end test hitting _process_batch, top_k>1, and batch flushing."""
        import torch

        clf = _mock_classifier()

        # Set up mock model to return realistic tensor outputs
        embed_dim = 8

        def fake_get_image_features(**kwargs):
            batch_size = kwargs["pixel_values"].shape[0]
            return torch.randn(batch_size, embed_dim)

        def fake_get_text_features(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            return torch.randn(batch_size, embed_dim)

        clf.model.get_image_features = MagicMock(side_effect=fake_get_image_features)
        clf.model.get_text_features = MagicMock(side_effect=fake_get_text_features)
        clf.model.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

        # Mock processor to return tensors with the right batch dimension
        def fake_processor(**kwargs):
            if "images" in kwargs:
                n = len(kwargs["images"])
                return {"pixel_values": torch.randn(n, 3, 224, 224)}
            if "text" in kwargs:
                n = len(kwargs["text"])
                return {
                    "input_ids": torch.randint(0, 1000, (n, 10)),
                    "attention_mask": torch.ones(n, 10, dtype=torch.long),
                }
            return {}

        clf.processor = MagicMock(side_effect=fake_processor)

        # Create a small test raster on disk
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            raster_path = f.name
        try:
            _create_test_raster(raster_path, height=32, width=32, bands=3)

            # Create multiple polygons within the raster bounds [0, 1] x [0, 1]
            geoms = [
                box(0.1, 0.1, 0.3, 0.3),
                box(0.4, 0.4, 0.6, 0.6),
                box(0.7, 0.7, 0.9, 0.9),
            ]
            gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")

            labels = ["urban", "forest", "water"]

            # Use top_k > 1 and a small batch_size to force batching and flushing.
            # min_chip_size=1 because the 32x32 raster yields ~6px chips.
            result = clf.classify(
                vector_data=gdf,
                raster_path=raster_path,
                labels=labels,
                top_k=2,
                batch_size=1,
                min_chip_size=1,
                quiet=True,
            )

            # We expect one result per input geometry.
            self.assertEqual(len(result), len(gdf))

            # Verify main label/confidence columns are present and populated.
            self.assertIn("clip_label", result.columns)
            self.assertIn("clip_confidence", result.columns)
            self.assertFalse(result["clip_label"].isna().any())
            self.assertFalse(result["clip_confidence"].isna().any())

            # Verify top-k columns are present and contain per-geometry sequences.
            self.assertIn("clip_top_k_labels", result.columns)
            self.assertIn("clip_top_k_scores", result.columns)
            for labels_list, scores_list in zip(
                result["clip_top_k_labels"], result["clip_top_k_scores"]
            ):
                self.assertIsNotNone(labels_list)
                self.assertIsNotNone(scores_list)
                self.assertEqual(len(labels_list), 2)
                self.assertEqual(len(scores_list), 2)
        finally:
            os.unlink(raster_path)


if __name__ == "__main__":
    unittest.main()
