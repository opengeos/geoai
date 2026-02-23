#!/usr/bin/env python

"""Tests for `geoai.map_widgets` module."""

import inspect
import unittest


class TestMapWidgetsImport(unittest.TestCase):
    """Tests for map_widgets module import behavior."""

    def test_module_imports(self):
        """Test that the map_widgets module can be imported."""
        import geoai.map_widgets

        self.assertTrue(hasattr(geoai.map_widgets, "random_string"))
        self.assertTrue(hasattr(geoai.map_widgets, "DINOv3GUI"))

    def test_random_string_exists(self):
        """Test that random_string function exists and is callable."""
        from geoai.map_widgets import random_string

        self.assertTrue(callable(random_string))

    def test_dinov3gui_exists(self):
        """Test that DINOv3GUI class exists and is callable."""
        from geoai.map_widgets import DINOv3GUI

        self.assertTrue(callable(DINOv3GUI))

    def test_moondream_gui_exists(self):
        """Test that moondream_gui function exists and is callable."""
        from geoai.map_widgets import moondream_gui

        self.assertTrue(callable(moondream_gui))


class TestMapWidgetsRandomString(unittest.TestCase):
    """Tests for random_string function behavior."""

    def test_random_string_default_length(self):
        """Test that random_string returns a string of default length 6."""
        from geoai.map_widgets import random_string

        result = random_string()
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 6)

    def test_random_string_custom_length(self):
        """Test that random_string returns a string of the specified length."""
        from geoai.map_widgets import random_string

        for length in [1, 5, 10, 20]:
            result = random_string(length)
            self.assertIsInstance(result, str)
            self.assertEqual(len(result), length)

    def test_random_string_contains_only_lowercase(self):
        """Test that random_string only contains lowercase ASCII letters."""
        from geoai.map_widgets import random_string

        result = random_string(100)
        self.assertTrue(result.isalpha())
        self.assertTrue(result.islower())


class TestMapWidgetsSignatures(unittest.TestCase):
    """Tests for map_widgets function and class signatures."""

    def test_random_string_params(self):
        """Test random_string has expected parameters."""
        from geoai.map_widgets import random_string

        sig = inspect.signature(random_string)
        self.assertIn("string_length", sig.parameters)
        default = sig.parameters["string_length"].default
        self.assertEqual(default, 6)

    def test_dinov3gui_init_params(self):
        """Test DINOv3GUI.__init__ has expected parameters."""
        from geoai.map_widgets import DINOv3GUI

        sig = inspect.signature(DINOv3GUI.__init__)
        self.assertIn("raster", sig.parameters)
        self.assertIn("processor", sig.parameters)
        self.assertIn("features", sig.parameters)
        self.assertIn("host_map", sig.parameters)
        self.assertIn("position", sig.parameters)
        self.assertIn("colormap_options", sig.parameters)
        self.assertIn("raster_args", sig.parameters)

    def test_moondream_gui_params(self):
        """Test moondream_gui has expected parameters."""
        from geoai.map_widgets import moondream_gui

        sig = inspect.signature(moondream_gui)
        self.assertIn("moondream", sig.parameters)
        self.assertIn("basemap", sig.parameters)
        self.assertIn("out_dir", sig.parameters)
        self.assertIn("opacity", sig.parameters)


if __name__ == "__main__":
    unittest.main()
