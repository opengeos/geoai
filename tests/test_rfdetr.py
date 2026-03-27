#!/usr/bin/env python

"""Tests for `geoai.rfdetr` module."""

import inspect
import unittest


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


class TestRFDETRLazyImport(unittest.TestCase):
    """Tests for lazy import from the top-level geoai package."""

    def test_lazy_import_rfdetr_detect(self):
        """Test rfdetr_detect can be imported from geoai via lazy loading."""
        from geoai import rfdetr_detect

        self.assertTrue(callable(rfdetr_detect))

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
            rfdetr_train,
        )

        self.assertTrue(callable(rfdetr_detect))
        self.assertTrue(callable(rfdetr_detect_batch))
        self.assertTrue(callable(rfdetr_train))

    def test_lazy_import_hub_functions(self):
        """Test Hub functions can be imported from geoai."""
        from geoai import push_rfdetr_to_hub, rfdetr_detect_from_hub

        self.assertTrue(callable(push_rfdetr_to_hub))
        self.assertTrue(callable(rfdetr_detect_from_hub))


if __name__ == "__main__":
    unittest.main()
