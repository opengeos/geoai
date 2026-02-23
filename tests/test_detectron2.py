#!/usr/bin/env python

"""Tests for `geoai.detectron2` module."""

import inspect
import unittest


class TestDetectron2Import(unittest.TestCase):
    """Tests for detectron2 module import behavior."""

    def test_module_imports(self):
        """Test that the detectron2 module can be imported."""
        import geoai.detectron2

        self.assertTrue(hasattr(geoai.detectron2, "check_detectron2"))
        self.assertTrue(hasattr(geoai.detectron2, "load_detectron2_model"))

    def test_check_detectron2_exists(self):
        """Test that check_detectron2 function exists and is callable."""
        from geoai.detectron2 import check_detectron2

        self.assertTrue(callable(check_detectron2))

    def test_load_detectron2_model_exists(self):
        """Test that load_detectron2_model function exists and is callable."""
        from geoai.detectron2 import load_detectron2_model

        self.assertTrue(callable(load_detectron2_model))

    def test_detectron2_segment_exists(self):
        """Test that detectron2_segment function exists and is callable."""
        from geoai.detectron2 import detectron2_segment

        self.assertTrue(callable(detectron2_segment))

    def test_create_instance_mask_exists(self):
        """Test that create_instance_mask function exists and is callable."""
        from geoai.detectron2 import create_instance_mask

        self.assertTrue(callable(create_instance_mask))

    def test_create_probability_mask_exists(self):
        """Test that create_probability_mask function exists and is callable."""
        from geoai.detectron2 import create_probability_mask

        self.assertTrue(callable(create_probability_mask))


class TestDetectron2Signatures(unittest.TestCase):
    """Tests for detectron2 function signatures."""

    def test_load_detectron2_model_params(self):
        """Test load_detectron2_model has expected parameters."""
        from geoai.detectron2 import load_detectron2_model

        sig = inspect.signature(load_detectron2_model)
        self.assertIn("model_config", sig.parameters)
        self.assertIn("model_weights", sig.parameters)
        self.assertIn("score_threshold", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("num_classes", sig.parameters)

    def test_detectron2_segment_params(self):
        """Test detectron2_segment has expected parameters."""
        from geoai.detectron2 import detectron2_segment

        sig = inspect.signature(detectron2_segment)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("model_config", sig.parameters)
        self.assertIn("score_threshold", sig.parameters)
        self.assertIn("save_masks", sig.parameters)
        self.assertIn("save_probability", sig.parameters)

    def test_save_geotiff_mask_params(self):
        """Test save_geotiff_mask has expected parameters."""
        from geoai.detectron2 import save_geotiff_mask

        sig = inspect.signature(save_geotiff_mask)
        self.assertIn("mask", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("transform", sig.parameters)
        self.assertIn("crs", sig.parameters)
        self.assertIn("dtype", sig.parameters)

    def test_batch_detectron2_segment_params(self):
        """Test batch_detectron2_segment has expected parameters."""
        from geoai.detectron2 import batch_detectron2_segment

        sig = inspect.signature(batch_detectron2_segment)
        self.assertIn("image_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("model_config", sig.parameters)
        self.assertIn("score_threshold", sig.parameters)

    def test_get_class_id_name_mapping_params(self):
        """Test get_class_id_name_mapping has expected parameters."""
        from geoai.detectron2 import get_class_id_name_mapping

        sig = inspect.signature(get_class_id_name_mapping)
        self.assertIn("config_path", sig.parameters)
        self.assertIn("lazy", sig.parameters)


class TestDetectron2HasFlag(unittest.TestCase):
    """Tests for detectron2 availability flag."""

    def test_has_detectron2_flag_exists(self):
        """Test that HAS_DETECTRON2 flag is defined."""
        import geoai.detectron2

        self.assertTrue(hasattr(geoai.detectron2, "HAS_DETECTRON2"))

    def test_has_detectron2_flag_is_bool(self):
        """Test that HAS_DETECTRON2 flag is a boolean."""
        from geoai.detectron2 import HAS_DETECTRON2

        self.assertIsInstance(HAS_DETECTRON2, bool)


if __name__ == "__main__":
    unittest.main()
