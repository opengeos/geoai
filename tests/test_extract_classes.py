#!/usr/bin/env python

"""Tests for `geoai.extract` module classes and exports."""

import inspect
import unittest


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

    def test_parking_splot_detector_exists(self):
        """Test that ParkingSplotDetector class exists and is callable."""
        from geoai.extract import ParkingSplotDetector

        self.assertTrue(callable(ParkingSplotDetector))

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
            ParkingSplotDetector,
            ShipDetector,
            SolarPanelDetector,
        )

        subclasses = [
            BuildingFootprintExtractor,
            CarDetector,
            ShipDetector,
            SolarPanelDetector,
            ParkingSplotDetector,
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

    def test_parking_splot_detector_init_params(self):
        """Test ParkingSplotDetector.__init__ has expected parameters."""
        from geoai.extract import ParkingSplotDetector

        sig = inspect.signature(ParkingSplotDetector.__init__)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("num_classes", sig.parameters)
        self.assertIn("device", sig.parameters)


if __name__ == "__main__":
    unittest.main()
