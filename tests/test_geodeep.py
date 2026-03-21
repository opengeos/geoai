#!/usr/bin/env python

"""Tests for `geoai.geodeep` module."""

import inspect
import unittest
from unittest.mock import patch


class TestGeoDeepImport(unittest.TestCase):
    """Tests for geodeep module import behavior."""

    def test_module_imports(self):
        """Test that the geodeep module can be imported."""
        import geoai.geodeep

        self.assertTrue(hasattr(geoai.geodeep, "GeoDeep"))
        self.assertTrue(hasattr(geoai.geodeep, "GEODEEP_MODELS"))

    def test_geodeep_class_exists(self):
        """Test that GeoDeep class exists and is callable."""
        from geoai.geodeep import GeoDeep

        self.assertTrue(callable(GeoDeep))

    def test_geodeep_detect_function_exists(self):
        """Test that geodeep_detect convenience function exists."""
        from geoai.geodeep import geodeep_detect

        self.assertTrue(callable(geodeep_detect))

    def test_geodeep_segment_function_exists(self):
        """Test that geodeep_segment convenience function exists."""
        from geoai.geodeep import geodeep_segment

        self.assertTrue(callable(geodeep_segment))

    def test_geodeep_detect_batch_function_exists(self):
        """Test that geodeep_detect_batch convenience function exists."""
        from geoai.geodeep import geodeep_detect_batch

        self.assertTrue(callable(geodeep_detect_batch))

    def test_geodeep_segment_batch_function_exists(self):
        """Test that geodeep_segment_batch convenience function exists."""
        from geoai.geodeep import geodeep_segment_batch

        self.assertTrue(callable(geodeep_segment_batch))

    def test_check_geodeep_available_exists(self):
        """Test that check_geodeep_available function exists."""
        from geoai.geodeep import check_geodeep_available

        self.assertTrue(callable(check_geodeep_available))

    def test_list_geodeep_models_exists(self):
        """Test that list_geodeep_models function exists and returns a dict."""
        from geoai.geodeep import list_geodeep_models

        self.assertTrue(callable(list_geodeep_models))
        result = list_geodeep_models()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_geodeep_available_flag_exists(self):
        """Test that GEODEEP_AVAILABLE flag is defined."""
        import geoai.geodeep

        self.assertTrue(hasattr(geoai.geodeep, "GEODEEP_AVAILABLE"))

    def test_geodeep_available_flag_is_bool(self):
        """Test that GEODEEP_AVAILABLE flag is a boolean."""
        from geoai.geodeep import GEODEEP_AVAILABLE

        self.assertIsInstance(GEODEEP_AVAILABLE, bool)


class TestGeoDeepAllExports(unittest.TestCase):
    """Tests for geodeep module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined in the geodeep module."""
        import geoai.geodeep

        self.assertTrue(hasattr(geoai.geodeep, "__all__"))

    def test_all_exports_contain_expected_names(self):
        """Test that __all__ contains the expected public API names."""
        from geoai.geodeep import __all__

        expected = [
            "GeoDeep",
            "geodeep_detect",
            "geodeep_segment",
            "geodeep_detect_batch",
            "geodeep_segment_batch",
            "list_geodeep_models",
            "check_geodeep_available",
            "GEODEEP_MODELS",
        ]
        for name in expected:
            self.assertIn(name, __all__)


class TestGeoDeepModels(unittest.TestCase):
    """Tests for GEODEEP_MODELS registry."""

    def test_geodeep_models_is_dict(self):
        """Test that GEODEEP_MODELS is a dictionary."""
        from geoai.geodeep import GEODEEP_MODELS

        self.assertIsInstance(GEODEEP_MODELS, dict)

    def test_all_expected_models_present(self):
        """Test that all 9 built-in models are registered."""
        from geoai.geodeep import GEODEEP_MODELS

        expected_models = [
            "cars",
            "trees",
            "trees_yolov9",
            "birds",
            "planes",
            "aerovision",
            "utilities",
            "buildings",
            "roads",
        ]
        for model_id in expected_models:
            self.assertIn(model_id, GEODEEP_MODELS)

    def test_model_entries_have_required_keys(self):
        """Test that every model entry has type, description, resolution, classes."""
        from geoai.geodeep import GEODEEP_MODELS

        required_keys = {"type", "description", "resolution", "classes"}
        for model_id, info in GEODEEP_MODELS.items():
            for key in required_keys:
                self.assertIn(
                    key,
                    info,
                    f"Model '{model_id}' missing key '{key}'",
                )

    def test_model_types_are_valid(self):
        """Test that all model types are either 'detection' or 'segmentation'."""
        from geoai.geodeep import GEODEEP_MODELS

        valid_types = {"detection", "segmentation"}
        for model_id, info in GEODEEP_MODELS.items():
            self.assertIn(
                info["type"],
                valid_types,
                f"Model '{model_id}' has invalid type '{info['type']}'",
            )

    def test_detection_models_count(self):
        """Test that there are 7 detection models."""
        from geoai.geodeep import GEODEEP_MODELS

        det_count = sum(
            1 for info in GEODEEP_MODELS.values() if info["type"] == "detection"
        )
        self.assertEqual(det_count, 7)

    def test_segmentation_models_count(self):
        """Test that there are 2 segmentation models."""
        from geoai.geodeep import GEODEEP_MODELS

        seg_count = sum(
            1 for info in GEODEEP_MODELS.values() if info["type"] == "segmentation"
        )
        self.assertEqual(seg_count, 2)

    def test_list_geodeep_models_returns_descriptions(self):
        """Test that list_geodeep_models returns model descriptions."""
        from geoai.geodeep import GEODEEP_MODELS, list_geodeep_models

        models = list_geodeep_models()
        self.assertEqual(len(models), len(GEODEEP_MODELS))
        for name, desc in models.items():
            self.assertIn(name, GEODEEP_MODELS)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


class TestGeoDeepSignatures(unittest.TestCase):
    """Tests for GeoDeep class and function signatures."""

    def test_geodeep_init_params(self):
        """Test GeoDeep.__init__ has expected parameters."""
        from geoai.geodeep import GeoDeep

        sig = inspect.signature(GeoDeep.__init__)
        self.assertIn("model_id", sig.parameters)
        self.assertIn("conf_threshold", sig.parameters)
        self.assertIn("classes", sig.parameters)
        self.assertIn("resolution", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("max_threads", sig.parameters)

    def test_detect_method_params(self):
        """Test detect method has expected parameters."""
        from geoai.geodeep import GeoDeep

        sig = inspect.signature(GeoDeep.detect)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("conf_threshold", sig.parameters)
        self.assertIn("classes", sig.parameters)
        self.assertIn("resolution", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("verbose", sig.parameters)

    def test_segment_method_params(self):
        """Test segment method has expected parameters."""
        from geoai.geodeep import GeoDeep

        sig = inspect.signature(GeoDeep.segment)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("output_raster_path", sig.parameters)
        self.assertIn("output_vector_path", sig.parameters)
        self.assertIn("resolution", sig.parameters)
        self.assertIn("verbose", sig.parameters)

    def test_detect_batch_method_params(self):
        """Test detect_batch method has expected parameters."""
        from geoai.geodeep import GeoDeep

        sig = inspect.signature(GeoDeep.detect_batch)
        self.assertIn("image_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("conf_threshold", sig.parameters)
        self.assertIn("classes", sig.parameters)
        self.assertIn("verbose", sig.parameters)

    def test_segment_batch_method_params(self):
        """Test segment_batch method has expected parameters."""
        from geoai.geodeep import GeoDeep

        sig = inspect.signature(GeoDeep.segment_batch)
        self.assertIn("image_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("output_format", sig.parameters)
        self.assertIn("verbose", sig.parameters)

    def test_geodeep_detect_function_params(self):
        """Test geodeep_detect convenience function has expected parameters."""
        from geoai.geodeep import geodeep_detect

        sig = inspect.signature(geodeep_detect)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("model_id", sig.parameters)
        self.assertIn("conf_threshold", sig.parameters)
        self.assertIn("classes", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("max_threads", sig.parameters)

    def test_geodeep_segment_function_params(self):
        """Test geodeep_segment convenience function has expected parameters."""
        from geoai.geodeep import geodeep_segment

        sig = inspect.signature(geodeep_segment)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("model_id", sig.parameters)
        self.assertIn("output_raster_path", sig.parameters)
        self.assertIn("output_vector_path", sig.parameters)
        self.assertIn("max_threads", sig.parameters)


def _make_geodeep(model_id="cars", **attrs):
    """Create a GeoDeep instance via __new__ without calling __init__."""
    from geoai.geodeep import GEODEEP_MODELS, GeoDeep

    gd = GeoDeep.__new__(GeoDeep)
    gd.model_id = model_id
    gd.conf_threshold = attrs.get("conf_threshold")
    gd.classes = attrs.get("classes")
    gd.resolution = attrs.get("resolution")
    gd.max_threads = attrs.get("max_threads")
    gd._device = attrs.get("device", "cpu")
    gd._model_info = GEODEEP_MODELS.get(model_id)
    return gd


class TestGeoDeepProperties(unittest.TestCase):
    """Tests for GeoDeep class properties and OOP behavior."""

    def test_model_type_detection(self):
        """Test model_type property returns 'detection' for detection models."""
        gd = _make_geodeep("cars")
        self.assertEqual(gd.model_type, "detection")

    def test_model_type_segmentation(self):
        """Test model_type property returns 'segmentation' for seg models."""
        gd = _make_geodeep("buildings")
        self.assertEqual(gd.model_type, "segmentation")

    def test_model_type_custom_returns_none(self):
        """Test model_type returns None for custom ONNX models."""
        gd = _make_geodeep("/path/to/custom.onnx")
        self.assertIsNone(gd.model_type)

    def test_model_info_property(self):
        """Test model_info returns model metadata dict."""
        gd = _make_geodeep("aerovision")
        info = gd.model_info
        self.assertIsInstance(info, dict)
        self.assertEqual(info["type"], "detection")
        self.assertIn("small-vehicle", info["classes"])

    def test_available_classes_property(self):
        """Test available_classes returns class list."""
        gd = _make_geodeep("utilities")
        classes = gd.available_classes
        self.assertIsInstance(classes, list)
        self.assertIn("Gas", classes)
        self.assertIn("Water", classes)

    def test_available_classes_returns_copy(self):
        """Test available_classes returns a copy, not the original list."""
        from geoai.geodeep import GEODEEP_MODELS

        gd = _make_geodeep("cars")
        classes = gd.available_classes
        classes.append("truck")
        self.assertNotIn("truck", GEODEEP_MODELS["cars"]["classes"])

    def test_device_property(self):
        """Test device property returns the configured device."""
        gd = _make_geodeep("cars", device="cpu")
        self.assertEqual(gd.device, "cpu")

    def test_device_property_cuda(self):
        """Test device property returns 'cuda' when configured."""
        gd = _make_geodeep("cars", device="cuda")
        self.assertEqual(gd.device, "cuda")

    def test_repr(self):
        """Test __repr__ returns expected format."""
        gd = _make_geodeep("cars", conf_threshold=0.5)
        r = repr(gd)
        self.assertIn("model_id='cars'", r)
        self.assertIn("conf_threshold=0.5", r)
        self.assertIn("device='cpu'", r)

    def test_str(self):
        """Test __str__ returns user-friendly format."""
        gd = _make_geodeep("cars")
        result = str(gd)
        self.assertIn("GeoDeep[cars]", result)
        self.assertIn("detection", result)
        self.assertIn("cpu", result)

    def test_str_custom_model(self):
        """Test __str__ for custom model shows 'custom' type."""
        gd = _make_geodeep("my_model.onnx")
        result = str(gd)
        self.assertIn("custom", result)


class TestGeoDeepBuildRunKwargs(unittest.TestCase):
    """Tests for _build_run_kwargs parameter merging."""

    def test_per_call_overrides_instance(self):
        """Test that per-call values override instance defaults."""
        gd = _make_geodeep(
            "cars",
            conf_threshold=0.3,
            classes=["car"],
            resolution=10.0,
            max_threads=4,
        )
        kwargs = gd._build_run_kwargs(
            verbose=True,
            conf_threshold=0.8,
            classes=["truck"],
            resolution=20.0,
        )
        self.assertEqual(kwargs["conf_threshold"], 0.8)
        self.assertEqual(kwargs["classes"], ["truck"])
        self.assertEqual(kwargs["resolution"], 20.0)
        self.assertEqual(kwargs["max_threads"], 4)

    def test_falls_back_to_instance_defaults(self):
        """Test that None per-call values fall back to instance settings."""
        gd = _make_geodeep(
            "cars",
            conf_threshold=0.5,
            classes=["bird"],
            resolution=5.0,
        )
        kwargs = gd._build_run_kwargs(verbose=True)
        self.assertEqual(kwargs["conf_threshold"], 0.5)
        self.assertEqual(kwargs["classes"], ["bird"])
        self.assertEqual(kwargs["resolution"], 5.0)
        self.assertNotIn("max_threads", kwargs)

    def test_zero_conf_threshold_not_treated_as_falsy(self):
        """Test that conf_threshold=0.0 is respected, not treated as None."""
        gd = _make_geodeep("cars", conf_threshold=0.5)
        kwargs = gd._build_run_kwargs(verbose=True, conf_threshold=0.0)
        self.assertEqual(kwargs["conf_threshold"], 0.0)

    def test_empty_classes_not_treated_as_falsy(self):
        """Test that classes=[] is respected, not treated as None."""
        gd = _make_geodeep("cars", classes=["car", "truck"])
        kwargs = gd._build_run_kwargs(verbose=True, classes=[])
        self.assertEqual(kwargs["classes"], [])

    def test_verbose_false_adds_progress_callback(self):
        """Test that verbose=False adds a no-op progress_callback."""
        gd = _make_geodeep("cars")
        kwargs = gd._build_run_kwargs(verbose=False)
        self.assertIn("progress_callback", kwargs)
        self.assertTrue(callable(kwargs["progress_callback"]))


class TestGeoDeepValidation(unittest.TestCase):
    """Tests for input validation."""

    def test_validate_image_path_raises_on_missing_file(self):
        """Test _validate_image_path raises FileNotFoundError."""
        from geoai.geodeep import GeoDeep

        with self.assertRaises(FileNotFoundError):
            GeoDeep._validate_image_path("/nonexistent/path/image.tif")


class TestGeoDeepLazyImport(unittest.TestCase):
    """Tests for lazy import from the top-level geoai package."""

    def test_lazy_import_geodeep_class(self):
        """Test GeoDeep can be imported from geoai via lazy loading."""
        from geoai import GeoDeep

        self.assertTrue(callable(GeoDeep))

    def test_lazy_import_list_geodeep_models(self):
        """Test list_geodeep_models can be imported from geoai."""
        from geoai import list_geodeep_models

        result = list_geodeep_models()
        self.assertIsInstance(result, dict)
        self.assertIn("cars", result)

    def test_lazy_import_geodeep_models(self):
        """Test GEODEEP_MODELS can be imported from geoai."""
        from geoai import GEODEEP_MODELS

        self.assertIsInstance(GEODEEP_MODELS, dict)
        self.assertIn("buildings", GEODEEP_MODELS)

    def test_lazy_import_convenience_functions(self):
        """Test convenience functions can be imported from geoai."""
        from geoai import (
            geodeep_detect,
            geodeep_detect_batch,
            geodeep_segment,
            geodeep_segment_batch,
        )

        self.assertTrue(callable(geodeep_detect))
        self.assertTrue(callable(geodeep_segment))
        self.assertTrue(callable(geodeep_detect_batch))
        self.assertTrue(callable(geodeep_segment_batch))


class TestGeoDeepGPU(unittest.TestCase):
    """Tests for GPU / device support."""

    def test_get_onnx_device_returns_string(self):
        """Test _get_onnx_device returns 'cpu' or 'cuda'."""
        from geoai.geodeep import _get_onnx_device

        device = _get_onnx_device()
        self.assertIn(device, ("cpu", "cuda"))

    def test_device_auto_resolves(self):
        """Test device='auto' resolves to a concrete device string."""
        gd = _make_geodeep("cars")
        self.assertIn(gd.device, ("cpu", "cuda"))
        self.assertNotEqual(gd.device, "auto")

    def test_device_cpu_explicit(self):
        """Test device='cpu' is respected."""
        gd = _make_geodeep("cars", device="cpu")
        self.assertEqual(gd.device, "cpu")

    def test_device_cuda_explicit(self):
        """Test device='cuda' is stored correctly."""
        gd = _make_geodeep("cars", device="cuda")
        self.assertEqual(gd.device, "cuda")

    def test_device_in_repr(self):
        """Test device appears in __repr__."""
        gd = _make_geodeep("cars", device="cpu")
        self.assertIn("device='cpu'", repr(gd))

    def test_device_in_str(self):
        """Test device appears in __str__."""
        gd = _make_geodeep("buildings", device="cuda")
        result = str(gd)
        self.assertIn("cuda", result)

    @patch("geoai.geodeep._get_onnx_device", return_value="cuda")
    @patch("geoai.geodeep.GEODEEP_AVAILABLE", True)
    def test_auto_selects_cuda_when_available(self, mock_device):
        """Test that device='auto' picks CUDA when provider is available."""
        from geoai.geodeep import GeoDeep

        gd = GeoDeep(model_id="cars", device="auto")
        self.assertEqual(gd.device, "cuda")

    @patch("geoai.geodeep._get_onnx_device", return_value="cpu")
    @patch("geoai.geodeep.GEODEEP_AVAILABLE", True)
    def test_auto_falls_back_to_cpu(self, mock_device):
        """Test that device='auto' falls back to CPU."""
        from geoai.geodeep import GeoDeep

        gd = GeoDeep(model_id="cars", device="auto")
        self.assertEqual(gd.device, "cpu")


if __name__ == "__main__":
    unittest.main()
