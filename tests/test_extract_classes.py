#!/usr/bin/env python

"""Tests for `geoai.extract` module classes and exports."""

import inspect
import os
import unittest
from unittest import mock


class TestExtractImport(unittest.TestCase):
    """Tests for extract module import behavior."""

    def test_module_imports(self):
        """Test that the extract module can be imported."""
        import geoai.extract

        self.assertTrue(hasattr(geoai.extract, "CustomDataset"))
        self.assertTrue(hasattr(geoai.extract, "ObjectDetector"))

    def test_custom_dataset_exists(self):
        """Test that CustomDataset class exists and is callable."""
        from geoai.extract import CustomDataset

        self.assertTrue(callable(CustomDataset))

    def test_object_detector_exists(self):
        """Test that ObjectDetector class exists and is callable."""
        from geoai.extract import ObjectDetector

        self.assertTrue(callable(ObjectDetector))

    def test_building_footprint_extractor_exists(self):
        """Test that BuildingFootprintExtractor class exists and is callable."""
        from geoai.extract import BuildingFootprintExtractor

        self.assertTrue(callable(BuildingFootprintExtractor))

    def test_car_detector_exists(self):
        """Test that CarDetector class exists and is callable."""
        from geoai.extract import CarDetector

        self.assertTrue(callable(CarDetector))

    def test_ship_detector_exists(self):
        """Test that ShipDetector class exists and is callable."""
        from geoai.extract import ShipDetector

        self.assertTrue(callable(ShipDetector))

    def test_solar_panel_detector_exists(self):
        """Test that SolarPanelDetector class exists and is callable."""
        from geoai.extract import SolarPanelDetector

        self.assertTrue(callable(SolarPanelDetector))

    def test_parking_spot_detector_exists(self):
        """Test that ParkingSpotDetector class exists and is callable."""
        from geoai.extract import ParkingSpotDetector

        self.assertTrue(callable(ParkingSpotDetector))

    def test_parking_splot_detector_alias(self):
        """Test that the deprecated ParkingSplotDetector alias still works."""
        from geoai.extract import ParkingSplotDetector, ParkingSpotDetector

        self.assertTrue(issubclass(ParkingSplotDetector, ParkingSpotDetector))

    def test_parking_splot_detector_alias_warns(self):
        """Test that instantiating the deprecated alias emits a DeprecationWarning."""
        from geoai.extract import ObjectDetector, ParkingSplotDetector

        with mock.patch.object(ObjectDetector, "__init__", return_value=None):
            with self.assertWarns(DeprecationWarning) as ctx:
                ParkingSplotDetector()

        self.assertIn("ParkingSpotDetector", str(ctx.warning))

    def test_agriculture_field_delineator_exists(self):
        """Test that AgricultureFieldDelineator class exists and is callable."""
        from geoai.extract import AgricultureFieldDelineator

        self.assertTrue(callable(AgricultureFieldDelineator))


class TestExtractAllExports(unittest.TestCase):
    """Tests for extract module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined in the extract module."""
        import geoai.extract

        self.assertTrue(hasattr(geoai.extract, "__all__"))

    def test_all_exports_contents(self):
        """Test that __all__ contains all expected class names."""
        from geoai.extract import __all__

        expected = [
            "CustomDataset",
            "ObjectDetector",
            "BuildingFootprintExtractor",
            "CarDetector",
            "ShipDetector",
            "SolarPanelDetector",
            "ParkingSpotDetector",
            "ParkingSplotDetector",
            "AgricultureFieldDelineator",
        ]
        for name in expected:
            self.assertIn(name, __all__)

    def test_all_exports_are_importable(self):
        """Test that every name in __all__ can be imported."""
        import geoai.extract

        for name in geoai.extract.__all__:
            self.assertTrue(
                hasattr(geoai.extract, name),
                f"{name} listed in __all__ but not found in module",
            )


class TestExtractDetectorInheritance(unittest.TestCase):
    """Tests for detector class inheritance."""

    def test_detectors_inherit_from_object_detector(self):
        """Test that all detector subclasses inherit from ObjectDetector."""
        from geoai.extract import (
            AgricultureFieldDelineator,
            BuildingFootprintExtractor,
            CarDetector,
            ObjectDetector,
            ParkingSpotDetector,
            ShipDetector,
            SolarPanelDetector,
        )

        subclasses = [
            BuildingFootprintExtractor,
            CarDetector,
            ShipDetector,
            SolarPanelDetector,
            ParkingSpotDetector,
            AgricultureFieldDelineator,
        ]
        for cls in subclasses:
            self.assertTrue(
                issubclass(cls, ObjectDetector),
                f"{cls.__name__} should inherit from ObjectDetector",
            )


class TestExtractSignatures(unittest.TestCase):
    """Tests for extract class signatures."""

    def test_custom_dataset_init_params(self):
        """Test CustomDataset.__init__ has expected parameters."""
        from geoai.extract import CustomDataset

        sig = inspect.signature(CustomDataset.__init__)
        self.assertIn("raster_path", sig.parameters)
        self.assertIn("chip_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)
        self.assertIn("transforms", sig.parameters)
        self.assertIn("band_indexes", sig.parameters)
        self.assertIn("verbose", sig.parameters)

    def test_object_detector_init_params(self):
        """Test ObjectDetector.__init__ has expected parameters."""
        from geoai.extract import ObjectDetector

        sig = inspect.signature(ObjectDetector.__init__)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("repo_id", sig.parameters)
        self.assertIn("model", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("device", sig.parameters)

    def test_agriculture_field_delineator_init_params(self):
        """Test AgricultureFieldDelineator.__init__ has expected parameters."""
        from geoai.extract import AgricultureFieldDelineator

        sig = inspect.signature(AgricultureFieldDelineator.__init__)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("band_selection", sig.parameters)
        self.assertIn("use_ndvi", sig.parameters)

    def test_parking_spot_detector_init_params(self):
        """Test ParkingSpotDetector.__init__ has expected parameters."""
        from geoai.extract import ParkingSpotDetector

        sig = inspect.signature(ParkingSpotDetector.__init__)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("device", sig.parameters)


class TestSafetensorsWeights(unittest.TestCase):
    """Tests for loading ``.safetensors`` checkpoints."""

    def setUp(self):
        """Create a temporary .safetensors checkpoint for a small linear model."""
        import tempfile

        import torch
        from safetensors.torch import save_file

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = os.path.join(self.tmpdir.name, "weights.safetensors")
        self.state_dict = {
            "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "bias": torch.zeros(2, dtype=torch.float32),
        }
        save_file(self.state_dict, self.path, metadata={"num_classes": "2"})

    def test_load_safetensors_returns_state_dict(self):
        """Test that _load_safetensors reads tensors back unchanged."""
        import torch

        from geoai.extract import ObjectDetector

        stub = mock.Mock(device=torch.device("cpu"))
        loaded = ObjectDetector._load_safetensors(stub, self.path)

        self.assertEqual(set(loaded), set(self.state_dict))
        for key, value in self.state_dict.items():
            self.assertTrue(torch.equal(value, loaded[key]))

    def test_load_weights_dispatches_on_safetensors_extension(self):
        """Test that load_weights loads .safetensors files into the model."""
        import torch

        from geoai.extract import ObjectDetector

        model = torch.nn.Linear(2, 2)
        stub = mock.Mock(device=torch.device("cpu"), model=model)
        stub._load_safetensors = ObjectDetector._load_safetensors.__get__(stub)

        ObjectDetector.load_weights(stub, self.path)

        self.assertTrue(torch.equal(model.weight.data, self.state_dict["weight"]))
        self.assertTrue(torch.equal(model.bias.data, self.state_dict["bias"]))

    def test_load_weights_missing_file_raises(self):
        """Test that load_weights raises FileNotFoundError for a missing file."""
        import torch

        from geoai.extract import ObjectDetector

        stub = mock.Mock(device=torch.device("cpu"))
        with self.assertRaises(FileNotFoundError):
            ObjectDetector.load_weights(
                stub, os.path.join(self.tmpdir.name, "nope.safetensors")
            )


if __name__ == "__main__":
    unittest.main()
