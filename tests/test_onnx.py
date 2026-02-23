#!/usr/bin/env python

"""Tests for `geoai.onnx` module (ONNX Runtime support)."""

import inspect
import sys
import unittest
from unittest.mock import MagicMock, patch

from geoai import onnx


class TestCheckOnnxDeps(unittest.TestCase):
    """Tests for the _check_onnx_deps function."""

    def test_success_when_both_installed(self):
        """Test that no error is raised when onnx and onnxruntime are available."""
        with patch.dict(sys.modules, {"onnx": MagicMock(), "onnxruntime": MagicMock()}):
            try:
                onnx._check_onnx_deps()
            except ImportError:
                self.fail("_check_onnx_deps raised ImportError unexpectedly")

    def test_raises_when_onnx_missing(self):
        """Test ImportError when onnx package is not installed."""
        with patch.dict(sys.modules, {"onnx": None}):
            with self.assertRaises(ImportError) as ctx:
                onnx._check_onnx_deps()
            self.assertIn("onnx", str(ctx.exception))

    def test_raises_when_onnxruntime_missing(self):
        """Test ImportError when onnxruntime is not installed."""
        with patch.dict(sys.modules, {"onnx": MagicMock(), "onnxruntime": None}):
            with self.assertRaises(ImportError) as ctx:
                onnx._check_onnx_deps()
            self.assertIn("onnxruntime", str(ctx.exception))


class TestCheckTorchDeps(unittest.TestCase):
    """Tests for the _check_torch_deps function."""

    def test_success_when_torch_installed(self):
        """Test no error when torch is available."""
        # torch is already installed in test environment
        try:
            onnx._check_torch_deps()
        except ImportError:
            self.fail("_check_torch_deps raised ImportError unexpectedly")


class TestONNXGeoModelClass(unittest.TestCase):
    """Tests for the ONNXGeoModel class."""

    def test_class_exists(self):
        """Test that ONNXGeoModel is available."""
        self.assertTrue(hasattr(onnx, "ONNXGeoModel"))

    def test_init_signature(self):
        """Test ONNXGeoModel.__init__ has expected parameters."""
        sig = inspect.signature(onnx.ONNXGeoModel.__init__)
        params = list(sig.parameters.keys())
        self.assertIn("self", params)

    def test_predict_method_exists(self):
        """Test that ONNXGeoModel has a predict method."""
        self.assertTrue(hasattr(onnx.ONNXGeoModel, "predict"))


class TestExportToOnnx(unittest.TestCase):
    """Tests for the export_to_onnx function."""

    def test_function_exists(self):
        """Test that export_to_onnx is available."""
        self.assertTrue(hasattr(onnx, "export_to_onnx"))
        self.assertTrue(callable(onnx.export_to_onnx))

    def test_function_signature(self):
        """Test that export_to_onnx has expected parameters."""
        sig = inspect.signature(onnx.export_to_onnx)
        self.assertIn("model_name_or_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)


class TestOnnxModuleFunctions(unittest.TestCase):
    """Tests for module-level ONNX functions."""

    def test_onnx_semantic_segmentation_exists(self):
        """Test that onnx_semantic_segmentation is available."""
        self.assertTrue(hasattr(onnx, "onnx_semantic_segmentation"))
        self.assertTrue(callable(onnx.onnx_semantic_segmentation))

    def test_onnx_image_classification_exists(self):
        """Test that onnx_image_classification is available."""
        self.assertTrue(hasattr(onnx, "onnx_image_classification"))
        self.assertTrue(callable(onnx.onnx_image_classification))


if __name__ == "__main__":
    unittest.main()
