#!/usr/bin/env python

"""Tests for `geoai.auto` module (Auto classes for geospatial inference)."""

import inspect
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from geoai import auto
from geoai.auto import AutoGeoImageProcessor, AutoGeoModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_geotiff(
    path, width=64, height=64, bands=3, dtype="uint8", crs="EPSG:4326"
):
    """Create a minimal GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.randint(0, 256, (bands, height, width), dtype=np.uint8)
    if dtype != "uint8":
        data = data.astype(dtype)
    transform = from_bounds(-122.5, 37.7, -122.3, 37.9, width, height)

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
    return data


def _create_test_png(path, width=64, height=64):
    """Create a minimal PNG for testing."""
    img = Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))
    img.save(path)
    return img


# ---------------------------------------------------------------------------
# AutoGeoImageProcessor tests
# ---------------------------------------------------------------------------


class TestAutoGeoImageProcessor(unittest.TestCase):
    """Tests for the AutoGeoImageProcessor class."""

    def test_class_exists(self):
        self.assertTrue(hasattr(auto, "AutoGeoImageProcessor"))

    def test_is_class(self):
        self.assertTrue(inspect.isclass(AutoGeoImageProcessor))

    # -- __init__ -----------------------------------------------------------

    @patch("geoai.auto.get_device", return_value="cpu")
    def test_init_defaults_device_to_auto(self, mock_dev):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc)
        self.assertEqual(obj.device, "cpu")
        self.assertIs(obj.processor, proc)
        self.assertIsNone(obj.processor_name)

    def test_init_explicit_device(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, processor_name="test-proc", device="cuda:1")
        self.assertEqual(obj.device, "cuda:1")
        self.assertEqual(obj.processor_name, "test-proc")

    # -- load_geotiff -------------------------------------------------------

    def test_load_geotiff_from_path(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=32, height=32, bands=3)

            data, meta = obj.load_geotiff(path)
            self.assertEqual(data.shape, (3, 32, 32))
            self.assertIn("crs", meta)
            self.assertIn("transform", meta)
            self.assertIn("profile", meta)
            self.assertEqual(meta["width"], 32)
            self.assertEqual(meta["height"], 32)

    def test_load_geotiff_with_bands(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=32, height=32, bands=4)

            data, meta = obj.load_geotiff(path, bands=[1, 3])
            self.assertEqual(data.shape[0], 2)
            self.assertEqual(meta["count"], 2)

    def test_load_geotiff_with_window(self):
        from rasterio.windows import Window

        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path, width=64, height=64, bands=3)

            window = Window(10, 10, 20, 20)
            data, meta = obj.load_geotiff(path, window=window)
            self.assertEqual(data.shape, (3, 20, 20))
            self.assertEqual(meta["width"], 20)
            self.assertEqual(meta["height"], 20)

    def test_load_geotiff_from_dataset_reader(self):
        import rasterio

        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path)

            with rasterio.open(path) as src:
                data, meta = obj.load_geotiff(src)
                self.assertEqual(data.ndim, 3)

    # -- load_image ---------------------------------------------------------

    def test_load_image_from_geotiff(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path)

            data, meta = obj.load_image(path)
            self.assertEqual(data.ndim, 3)
            self.assertIsNotNone(meta)

    def test_load_image_from_png(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.png")
            _create_test_png(path)

            data, meta = obj.load_image(path)
            self.assertEqual(data.shape[0], 3)  # CHW
            self.assertIsNone(meta)

    def test_load_image_from_numpy_hwc(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        data, meta = obj.load_image(arr)
        self.assertEqual(data.shape, (3, 64, 64))
        self.assertIsNone(meta)

    def test_load_image_from_numpy_2d(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        data, meta = obj.load_image(arr)
        self.assertEqual(data.shape, (1, 64, 64))
        self.assertIsNone(meta)

    def test_load_image_from_pil(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        data, meta = obj.load_image(img)
        self.assertEqual(data.shape[0], 3)
        self.assertIsNone(meta)

    def test_load_image_unsupported_type(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")
        with self.assertRaises(TypeError):
            obj.load_image(42)

    # -- prepare_for_model --------------------------------------------------

    def test_prepare_for_model_3band(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        data = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        result = obj.prepare_for_model(data)
        self.assertIn("pixel_values", result)
        mock_proc.assert_called_once()

    def test_prepare_for_model_1band(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        data = np.random.randint(0, 255, (1, 64, 64), dtype=np.uint8)
        result = obj.prepare_for_model(data)
        self.assertIn("pixel_values", result)

    def test_prepare_for_model_4band_truncates(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        data = np.random.randint(0, 255, (4, 64, 64), dtype=np.uint8)
        result = obj.prepare_for_model(data)
        self.assertIn("pixel_values", result)

    def test_prepare_for_model_2d_input(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = obj.prepare_for_model(data)
        self.assertIn("pixel_values", result)

    def test_prepare_percentile_clip_uniform_band(self):
        """When p2 == p98 (constant band), output should be zero."""
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        data = np.full((3, 64, 64), 42, dtype=np.float32)
        obj.prepare_for_model(data, normalize=True, percentile_clip=True)
        # Should not raise

    def test_prepare_no_normalize(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        data = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        result = obj.prepare_for_model(data, normalize=False)
        self.assertIn("pixel_values", result)

    # -- save_geotiff -------------------------------------------------------

    def test_save_geotiff_2d(self):
        import rasterio

        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")

        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "src.tif")
            _create_test_geotiff(src_path, width=32, height=32)

            _, meta = obj.load_geotiff(src_path)
            output_data = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
            out_path = os.path.join(td, "out.tif")
            result = obj.save_geotiff(output_data, out_path, meta)

            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))

            with rasterio.open(out_path) as dst:
                self.assertEqual(dst.count, 1)
                self.assertEqual(dst.width, 32)
                self.assertEqual(dst.height, 32)

    def test_save_geotiff_3d(self):
        import rasterio

        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")

        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "src.tif")
            _create_test_geotiff(src_path, width=32, height=32)

            _, meta = obj.load_geotiff(src_path)
            output_data = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
            out_path = os.path.join(td, "out.tif")
            obj.save_geotiff(output_data, out_path, meta)

            with rasterio.open(out_path) as dst:
                self.assertEqual(dst.count, 3)

    def test_save_geotiff_creates_parent_dirs(self):
        proc = MagicMock()
        obj = AutoGeoImageProcessor(proc, device="cpu")

        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "src.tif")
            _create_test_geotiff(src_path, width=32, height=32)

            _, meta = obj.load_geotiff(src_path)
            output_data = np.zeros((32, 32), dtype=np.uint8)
            out_path = os.path.join(td, "sub", "dir", "out.tif")
            obj.save_geotiff(output_data, out_path, meta)
            self.assertTrue(os.path.exists(out_path))

    # -- __call__ -----------------------------------------------------------

    def test_call_with_single_pil_image(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = obj(img)
        self.assertIn("pixel_values", result)

    def test_call_with_list_of_arrays(self):
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        arrs = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]
        result = obj(arrs)
        mock_proc.assert_called_once()

    def test_call_with_float_array(self):
        """Float arrays should be percentile-normalized to uint8."""
        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": "dummy"}
        obj = AutoGeoImageProcessor(mock_proc, device="cpu")

        arr = np.random.rand(64, 64, 3).astype(np.float32) * 10000
        result = obj(arr)
        self.assertIn("pixel_values", result)


# ---------------------------------------------------------------------------
# AutoGeoModel tests
# ---------------------------------------------------------------------------


class TestAutoGeoModel(unittest.TestCase):
    """Tests for the AutoGeoModel class."""

    def test_class_exists(self):
        self.assertTrue(hasattr(auto, "AutoGeoModel"))

    def test_is_class(self):
        self.assertTrue(inspect.isclass(AutoGeoModel))

    def test_task_model_mapping(self):
        expected_tasks = [
            "segmentation",
            "semantic-segmentation",
            "depth-estimation",
            "mask-generation",
            "object-detection",
            "zero-shot-object-detection",
            "classification",
            "image-classification",
        ]
        for task in expected_tasks:
            self.assertIn(task, AutoGeoModel.TASK_MODEL_MAPPING)

    @patch("geoai.auto.get_device", return_value="cpu")
    def test_init_sets_attributes(self, _):
        import torch

        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, task="segmentation", tile_size=512, overlap=64)
        self.assertEqual(obj.task, "segmentation")
        self.assertEqual(obj.tile_size, 512)
        self.assertEqual(obj.overlap, 64)
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    # -- _process_outputs ---------------------------------------------------

    def test_process_outputs_segmentation_logits(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        # Simulate segmentation output (batch, classes, H, W)
        logits = torch.randn(1, 5, 64, 64)
        outputs = types.SimpleNamespace(logits=logits)

        result = obj._process_outputs(outputs, (3, 64, 64), threshold=0.5)
        self.assertIn("mask", result)
        self.assertEqual(result["mask"].shape, (64, 64))
        self.assertEqual(result["mask"].dtype, np.uint8)

    def test_process_outputs_classification_logits(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        logits = torch.randn(1, 10)
        outputs = types.SimpleNamespace(logits=logits)

        result = obj._process_outputs(outputs, (3, 64, 64))
        self.assertIn("class", result)
        self.assertIn("probabilities", result)
        self.assertEqual(result["probabilities"].shape, (10,))

    def test_process_outputs_depth(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        outputs = types.SimpleNamespace(predicted_depth=torch.randn(1, 64, 64))

        result = obj._process_outputs(outputs, (3, 64, 64))
        self.assertIn("depth", result)
        self.assertEqual(result["depth"].shape, (64, 64))

    def test_process_outputs_with_probabilities(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        logits = torch.randn(1, 5, 64, 64)
        outputs = types.SimpleNamespace(logits=logits)

        result = obj._process_outputs(outputs, (3, 64, 64), return_probabilities=True)
        self.assertIn("probabilities", result)
        self.assertEqual(result["probabilities"].shape[0], 5)

    # -- mask_to_vector -----------------------------------------------------

    def test_mask_to_vector_no_metadata(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        mask = np.ones((64, 64), dtype=np.uint8)
        result = obj.mask_to_vector(mask, None)
        self.assertIsNone(result)

    def test_mask_to_vector_no_crs(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        mask = np.ones((64, 64), dtype=np.uint8)
        meta = {"crs": None, "transform": None}
        result = obj.mask_to_vector(mask, meta)
        self.assertIsNone(result)

    def test_mask_to_vector_no_transform(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        mask = np.ones((64, 64), dtype=np.uint8)
        meta = {"crs": "EPSG:4326", "transform": None}
        result = obj.mask_to_vector(mask, meta)
        self.assertIsNone(result)

    def test_mask_to_vector_empty_mask(self):
        import torch
        from rasterio.transform import from_bounds

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        mask = np.zeros((64, 64), dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        meta = {"crs": "EPSG:4326", "transform": transform}
        result = obj.mask_to_vector(mask, meta)
        self.assertIsNone(result)

    def test_mask_to_vector_with_valid_mask(self):
        import torch
        from rasterio.transform import from_bounds

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        # Large contiguous block -> at least 1 polygon
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        meta = {"crs": "EPSG:4326", "transform": transform}
        result = obj.mask_to_vector(mask, meta, min_object_area=0)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_mask_to_vector_float_mask(self):
        import torch
        from rasterio.transform import from_bounds

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 0.9
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        meta = {"crs": "EPSG:4326", "transform": transform}
        result = obj.mask_to_vector(mask, meta, threshold=0.5, min_object_area=0)
        self.assertIsNotNone(result)

    def test_mask_to_vector_max_area_filter(self):
        import torch
        from rasterio.transform import from_bounds

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        meta = {"crs": "EPSG:4326", "transform": transform}
        result = obj.mask_to_vector(mask, meta, min_object_area=0, max_object_area=1)
        # Large polygon should be filtered out by small max_object_area
        self.assertIsNone(result)

    # -- _detections_to_geodataframe ----------------------------------------

    def test_detections_to_geodataframe_empty(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        result = obj._detections_to_geodataframe({"boxes": np.array([])}, (100, 100))
        self.assertIsNone(result)

    def test_detections_to_geodataframe_valid(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        detections = {
            "boxes": np.array([[10, 10, 50, 50], [60, 60, 90, 90]]),
            "scores": np.array([0.9, 0.8]),
            "labels": ["building", "tree"],
        }
        gdf = obj._detections_to_geodataframe(detections, (100, 100))
        self.assertIsNotNone(gdf)
        self.assertEqual(len(gdf), 2)
        self.assertIn("score", gdf.columns)
        self.assertIn("label", gdf.columns)

    def test_detections_to_geodataframe_no_boxes_key(self):
        import torch

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        result = obj._detections_to_geodataframe({}, (100, 100))
        self.assertIsNone(result)

    # -- save_vector --------------------------------------------------------

    def test_save_vector(self):
        import geopandas as gpd
        import torch
        from shapely.geometry import box as shp_box

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        gdf = gpd.GeoDataFrame(
            {"geometry": [shp_box(0, 0, 1, 1)], "value": [1]}, crs="EPSG:4326"
        )
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.geojson")
            result = obj.save_vector(gdf, out)
            self.assertEqual(result, out)
            self.assertTrue(os.path.exists(out))

    def test_save_vector_auto_driver(self):
        import geopandas as gpd
        import torch
        from shapely.geometry import box as shp_box

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        obj = AutoGeoModel(mock_model, device="cpu")

        gdf = gpd.GeoDataFrame({"geometry": [shp_box(0, 0, 1, 1)]}, crs="EPSG:4326")
        with tempfile.TemporaryDirectory() as td:
            for ext in [".geojson", ".gpkg"]:
                out = os.path.join(td, f"out{ext}")
                obj.save_vector(gdf, out)
                self.assertTrue(os.path.exists(out))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions(unittest.TestCase):
    def test_semantic_segmentation_exists(self):
        self.assertTrue(callable(auto.semantic_segmentation))

    def test_depth_estimation_exists(self):
        self.assertTrue(callable(auto.depth_estimation))

    def test_image_classification_exists(self):
        self.assertTrue(callable(auto.image_classification))

    def test_object_detection_exists(self):
        self.assertTrue(callable(auto.object_detection))

    def test_get_hf_tasks_exists(self):
        self.assertTrue(callable(auto.get_hf_tasks))

    def test_get_hf_model_config_exists(self):
        self.assertTrue(callable(auto.get_hf_model_config))


class TestConvenienceFunctionSignatures(unittest.TestCase):
    def test_semantic_segmentation_signature(self):
        sig = inspect.signature(auto.semantic_segmentation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)

    def test_depth_estimation_signature(self):
        sig = inspect.signature(auto.depth_estimation)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)

    def test_image_classification_signature(self):
        sig = inspect.signature(auto.image_classification)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)

    def test_object_detection_signature(self):
        sig = inspect.signature(auto.object_detection)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("text", sig.parameters)
        self.assertIn("labels", sig.parameters)
        self.assertIn("box_threshold", sig.parameters)
        self.assertIn("text_threshold", sig.parameters)


class TestModuleExports(unittest.TestCase):
    def test_all_contains_expected_names(self):
        expected = [
            "AutoGeoImageProcessor",
            "AutoGeoModel",
            "semantic_segmentation",
            "depth_estimation",
            "image_classification",
            "object_detection",
            "get_hf_tasks",
            "get_hf_model_config",
            "show_image",
            "show_detections",
            "show_segmentation",
            "show_depth",
        ]
        for name in expected:
            self.assertIn(name, auto.__all__)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------


class TestLoadImageForDisplay(unittest.TestCase):
    def test_from_pil(self):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        result, meta = auto._load_image_for_display(img)
        self.assertEqual(result.shape, (32, 32, 3))
        self.assertIsNone(meta)

    def test_from_numpy_chw(self):
        arr = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        result, meta = auto._load_image_for_display(arr)
        self.assertEqual(result.shape[2], 3)
        self.assertIsNone(meta)

    def test_from_numpy_2d(self):
        arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result, meta = auto._load_image_for_display(arr)
        self.assertEqual(result.shape, (32, 32, 3))

    def test_from_geotiff(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.tif")
            _create_test_geotiff(path)
            result, meta = auto._load_image_for_display(path)
            self.assertEqual(result.ndim, 3)
            self.assertIsNotNone(meta)

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            auto._load_image_for_display(42)

    def test_float_normalization(self):
        arr = np.random.rand(32, 32, 3).astype(np.float32) * 10000
        result, _ = auto._load_image_for_display(arr)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
