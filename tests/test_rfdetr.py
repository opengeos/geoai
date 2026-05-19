#!/usr/bin/env python

"""Tests for `geoai.rfdetr` module."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pytest


class TestRFDETRImport(unittest.TestCase):
    """Tests for rfdetr module import behavior."""

    def test_module_imports(self):
        """Test that the rfdetr module can be imported."""
        import geoai.rfdetr

        self.assertTrue(hasattr(geoai.rfdetr, "RFDETR_MODELS"))
        self.assertTrue(hasattr(geoai.rfdetr, "rfdetr_detect"))

    def test_rfdetr_available_flag_exists(self):
        """Test that RFDETR_AVAILABLE flag is defined."""
        from geoai.rfdetr import RFDETR_AVAILABLE

        self.assertIsInstance(RFDETR_AVAILABLE, bool)

    def test_check_rfdetr_available_exists(self):
        """Test that check_rfdetr_available function exists."""
        from geoai.rfdetr import check_rfdetr_available

        self.assertTrue(callable(check_rfdetr_available))

    def test_rfdetr_detect_function_exists(self):
        """Test that rfdetr_detect convenience function exists."""
        from geoai.rfdetr import rfdetr_detect

        self.assertTrue(callable(rfdetr_detect))

    def test_rfdetr_segment_function_exists(self):
        """Test that rfdetr_segment convenience function exists."""
        from geoai.rfdetr import rfdetr_segment

        self.assertTrue(callable(rfdetr_segment))

    def test_rfdetr_detect_batch_function_exists(self):
        """Test that rfdetr_detect_batch convenience function exists."""
        from geoai.rfdetr import rfdetr_detect_batch

        self.assertTrue(callable(rfdetr_detect_batch))

    def test_rfdetr_train_function_exists(self):
        """Test that rfdetr_train function exists."""
        from geoai.rfdetr import rfdetr_train

        self.assertTrue(callable(rfdetr_train))

    def test_push_rfdetr_to_hub_function_exists(self):
        """Test that push_rfdetr_to_hub function exists."""
        from geoai.rfdetr import push_rfdetr_to_hub

        self.assertTrue(callable(push_rfdetr_to_hub))

    def test_rfdetr_detect_from_hub_function_exists(self):
        """Test that rfdetr_detect_from_hub function exists."""
        from geoai.rfdetr import rfdetr_detect_from_hub

        self.assertTrue(callable(rfdetr_detect_from_hub))


class TestRFDETRAllExports(unittest.TestCase):
    """Tests for rfdetr module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined in the rfdetr module."""
        import geoai.rfdetr

        self.assertTrue(hasattr(geoai.rfdetr, "__all__"))

    def test_all_exports_contain_expected_names(self):
        """Test that __all__ contains the expected public API names."""
        from geoai.rfdetr import __all__

        expected = [
            "RFDETR_MODELS",
            "check_rfdetr_available",
            "list_rfdetr_models",
            "rfdetr_detect",
            "rfdetr_segment",
            "rfdetr_detect_batch",
            "rfdetr_train",
            "push_rfdetr_to_hub",
            "rfdetr_detect_from_hub",
        ]
        for name in expected:
            self.assertIn(name, __all__)


class TestRFDETRModels(unittest.TestCase):
    """Tests for RFDETR_MODELS registry."""

    def test_rfdetr_models_is_dict(self):
        """Test that RFDETR_MODELS is a dictionary."""
        from geoai.rfdetr import RFDETR_MODELS

        self.assertIsInstance(RFDETR_MODELS, dict)

    def test_all_expected_models_present(self):
        """Test that all expected model variants are registered."""
        from geoai.rfdetr import RFDETR_MODELS

        expected_models = [
            "base",
            "nano",
            "small",
            "medium",
            "large",
            "seg-nano",
            "seg-small",
            "seg-medium",
            "seg-large",
            "seg-xlarge",
            "seg-2xlarge",
        ]
        for model_id in expected_models:
            self.assertIn(model_id, RFDETR_MODELS)

    def test_model_entries_have_required_keys(self):
        """Test that every model entry has class_name, resolution, description."""
        from geoai.rfdetr import RFDETR_MODELS

        required_keys = {"class_name", "resolution", "description"}
        for model_id, info in RFDETR_MODELS.items():
            for key in required_keys:
                self.assertIn(
                    key,
                    info,
                    f"Model '{model_id}' missing key '{key}'",
                )

    def test_model_resolutions_are_positive_ints(self):
        """Test that all model resolutions are positive integers."""
        from geoai.rfdetr import RFDETR_MODELS

        for model_id, info in RFDETR_MODELS.items():
            self.assertIsInstance(info["resolution"], int)
            self.assertGreater(
                info["resolution"],
                0,
                f"Model '{model_id}' has non-positive resolution",
            )

    def test_detection_models_count(self):
        """Test that there are 5 detection model variants."""
        from geoai.rfdetr import RFDETR_MODELS

        det_count = sum(1 for name in RFDETR_MODELS if not name.startswith("seg-"))
        self.assertEqual(det_count, 5)

    def test_segmentation_models_count(self):
        """Test that there are 6 segmentation model variants."""
        from geoai.rfdetr import RFDETR_MODELS

        seg_count = sum(1 for name in RFDETR_MODELS if name.startswith("seg-"))
        self.assertEqual(seg_count, 6)

    def test_segmentation_model_resolutions_match_current_rfdetr(self):
        """Test RF-DETR-Seg native resolutions."""
        from geoai.rfdetr import RFDETR_MODELS

        expected = {
            "seg-nano": 312,
            "seg-small": 384,
            "seg-medium": 432,
            "seg-large": 504,
            "seg-xlarge": 624,
            "seg-2xlarge": 768,
        }
        for model_id, resolution in expected.items():
            self.assertEqual(RFDETR_MODELS[model_id]["resolution"], resolution)

    def test_list_rfdetr_models_returns_descriptions(self):
        """Test that list_rfdetr_models returns model descriptions."""
        from geoai.rfdetr import RFDETR_MODELS, list_rfdetr_models

        models = list_rfdetr_models()
        self.assertEqual(len(models), len(RFDETR_MODELS))
        for name, desc in models.items():
            self.assertIn(name, RFDETR_MODELS)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


class TestRFDETRSignatures(unittest.TestCase):
    """Tests for function signatures."""

    def test_rfdetr_detect_params(self):
        """Test rfdetr_detect has expected parameters."""
        from geoai.rfdetr import rfdetr_detect

        sig = inspect.signature(rfdetr_detect)
        expected_params = [
            "input_path",
            "output_path",
            "model_variant",
            "pretrain_weights",
            "confidence_threshold",
            "nms_threshold",
            "window_size",
            "overlap",
            "batch_size",
            "class_names",
            "use_mask_geometry",
            "simplify_tolerance",
            "device",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)

    def test_rfdetr_segment_params(self):
        """Test rfdetr_segment has expected parameters."""
        from geoai.rfdetr import rfdetr_segment

        sig = inspect.signature(rfdetr_segment)
        expected_params = [
            "input_path",
            "output_path",
            "model_variant",
            "pretrain_weights",
            "confidence_threshold",
            "nms_threshold",
            "window_size",
            "overlap",
            "batch_size",
            "class_names",
            "simplify_tolerance",
            "device",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)

    def test_rfdetr_detect_batch_params(self):
        """Test rfdetr_detect_batch has expected parameters."""
        from geoai.rfdetr import rfdetr_detect_batch

        sig = inspect.signature(rfdetr_detect_batch)
        expected_params = [
            "input_paths",
            "output_dir",
            "model_variant",
            "confidence_threshold",
            "nms_threshold",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)

    def test_rfdetr_train_params(self):
        """Test rfdetr_train has expected parameters."""
        from geoai.rfdetr import rfdetr_train

        sig = inspect.signature(rfdetr_train)
        expected_params = [
            "dataset_dir",
            "model_variant",
            "epochs",
            "batch_size",
            "output_dir",
            "pretrain_weights",
            "device",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)

    def test_push_rfdetr_to_hub_params(self):
        """Test push_rfdetr_to_hub has expected parameters."""
        from geoai.rfdetr import push_rfdetr_to_hub

        sig = inspect.signature(push_rfdetr_to_hub)
        expected_params = [
            "model_path",
            "repo_id",
            "model_variant",
            "num_classes",
            "class_names",
            "private",
            "token",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)

    def test_rfdetr_detect_from_hub_params(self):
        """Test rfdetr_detect_from_hub has expected parameters."""
        from geoai.rfdetr import rfdetr_detect_from_hub

        sig = inspect.signature(rfdetr_detect_from_hub)
        expected_params = [
            "input_path",
            "repo_id",
            "filename",
            "output_path",
            "confidence_threshold",
        ]
        for param in expected_params:
            self.assertIn(param, sig.parameters)


class TestRFDETRValidation(unittest.TestCase):
    """Tests for input validation."""

    def test_invalid_model_variant_raises(self):
        """Test that an invalid model variant raises ValueError."""
        from geoai.rfdetr import _get_rfdetr_model_class

        with self.assertRaises((ValueError, ImportError)):
            _get_rfdetr_model_class("nonexistent_variant")

    def test_rfdetr_segment_rejects_detection_variant(self):
        """Test rfdetr_segment requires a segmentation model variant."""
        from geoai.rfdetr import rfdetr_segment

        with self.assertRaises(ValueError):
            rfdetr_segment("dummy.tif", model_variant="base")


class TestRFDETRLazyImport(unittest.TestCase):
    """Tests for lazy import from the top-level geoai package."""

    def test_lazy_import_rfdetr_detect(self):
        """Test rfdetr_detect can be imported from geoai via lazy loading."""
        from geoai import rfdetr_detect

        self.assertTrue(callable(rfdetr_detect))

    def test_lazy_import_rfdetr_segment(self):
        """Test rfdetr_segment can be imported from geoai via lazy loading."""
        from geoai import rfdetr_segment

        self.assertTrue(callable(rfdetr_segment))

    def test_lazy_import_list_rfdetr_models(self):
        """Test list_rfdetr_models can be imported from geoai."""
        from geoai import list_rfdetr_models

        result = list_rfdetr_models()
        self.assertIsInstance(result, dict)
        self.assertIn("base", result)

    def test_lazy_import_rfdetr_models(self):
        """Test RFDETR_MODELS can be imported from geoai."""
        from geoai import RFDETR_MODELS

        self.assertIsInstance(RFDETR_MODELS, dict)
        self.assertIn("base", RFDETR_MODELS)

    def test_lazy_import_convenience_functions(self):
        """Test convenience functions can be imported from geoai."""
        from geoai import (
            rfdetr_detect,
            rfdetr_detect_batch,
            rfdetr_segment,
            rfdetr_train,
        )

        self.assertTrue(callable(rfdetr_detect))
        self.assertTrue(callable(rfdetr_detect_batch))
        self.assertTrue(callable(rfdetr_segment))
        self.assertTrue(callable(rfdetr_train))

    def test_lazy_import_hub_functions(self):
        """Test Hub functions can be imported from geoai."""
        from geoai import push_rfdetr_to_hub, rfdetr_detect_from_hub

        self.assertTrue(callable(push_rfdetr_to_hub))
        self.assertTrue(callable(rfdetr_detect_from_hub))


class _FakeRFDETRDetections:
    """Small Supervision-style detections object for RF-DETR tests."""

    def __init__(self):
        self.xyxy = np.array([[8.0, 8.0, 28.0, 28.0]], dtype=float)
        self.confidence = np.array([0.92], dtype=float)
        self.class_id = np.array([0], dtype=int)
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:28, 8:18] = True
        mask[18:28, 18:28] = True
        self.mask = mask[np.newaxis, ...]


class _FakeRFDETRModel:
    """Small RF-DETR model double returning deterministic masks."""

    class_names = ["building"]

    def predict(self, images, threshold=0.5):
        return [_FakeRFDETRDetections() for _ in images]


def test_rfdetr_segment_vectorizes_masks(tmp_path):
    """RF-DETR-Seg masks are returned as georeferenced mask polygons."""
    pytest.importorskip("geopandas")
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from rasterio.transform import from_origin

    from geoai.rfdetr import rfdetr_segment

    image_path = tmp_path / "image.tif"
    transform = from_origin(0, 64, 1, 1)
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))

    with (
        patch("geoai.rfdetr.RFDETR_AVAILABLE", True),
        patch("geoai.rfdetr._create_rfdetr_model", return_value=_FakeRFDETRModel()),
    ):
        gdf = rfdetr_segment(
            str(image_path),
            model_variant="seg-medium",
            window_size=64,
            overlap=0,
            batch_size=1,
        )

    assert len(gdf) == 1
    assert gdf.iloc[0]["class_name"] == "building"
    assert gdf.iloc[0]["area_pixels"] == 300
    assert gdf.geometry.iloc[0].area == 300
    assert gdf.geometry.iloc[0].area < 400


if __name__ == "__main__":
    unittest.main()
