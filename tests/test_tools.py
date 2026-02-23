#!/usr/bin/env python

"""Tests for `geoai.tools` subpackage."""

import inspect
import unittest


class TestToolsImport(unittest.TestCase):
    """Tests for tools subpackage import behavior."""

    def test_tools_package_imports(self):
        """Test that the tools package can be imported."""
        import geoai.tools

        self.assertTrue(hasattr(geoai.tools, "__all__"))

    def test_all_is_list(self):
        """Test that __all__ is a list."""
        import geoai.tools

        self.assertIsInstance(geoai.tools.__all__, list)


class TestCloudMaskModule(unittest.TestCase):
    """Tests for the cloudmask module."""

    def test_module_imports(self):
        """Test that cloudmask module can be imported."""
        from geoai.tools import cloudmask

        self.assertTrue(hasattr(cloudmask, "CLEAR"))
        self.assertTrue(hasattr(cloudmask, "THICK_CLOUD"))
        self.assertTrue(hasattr(cloudmask, "THIN_CLOUD"))
        self.assertTrue(hasattr(cloudmask, "CLOUD_SHADOW"))

    def test_cloud_mask_constants(self):
        """Test cloud mask constant values."""
        from geoai.tools.cloudmask import (
            CLEAR,
            CLOUD_SHADOW,
            THICK_CLOUD,
            THIN_CLOUD,
        )

        self.assertEqual(CLEAR, 0)
        self.assertEqual(THICK_CLOUD, 1)
        self.assertEqual(THIN_CLOUD, 2)
        self.assertEqual(CLOUD_SHADOW, 3)

    def test_check_omnicloudmask_available_exists(self):
        """Test that check_omnicloudmask_available function exists."""
        from geoai.tools.cloudmask import check_omnicloudmask_available

        self.assertTrue(callable(check_omnicloudmask_available))

    def test_predict_cloud_mask_exists(self):
        """Test that predict_cloud_mask function exists and is callable."""
        from geoai.tools.cloudmask import predict_cloud_mask

        self.assertTrue(callable(predict_cloud_mask))

    def test_predict_cloud_mask_signature(self):
        """Test predict_cloud_mask has expected parameters."""
        from geoai.tools.cloudmask import predict_cloud_mask

        sig = inspect.signature(predict_cloud_mask)
        self.assertIn("image", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("inference_device", sig.parameters)


class TestMultiCleanModule(unittest.TestCase):
    """Tests for the multiclean module."""

    def test_module_imports(self):
        """Test that multiclean module can be imported."""
        from geoai.tools import multiclean

        self.assertTrue(hasattr(multiclean, "check_multiclean_available"))

    def test_check_multiclean_available_exists(self):
        """Test that check_multiclean_available function exists."""
        from geoai.tools.multiclean import check_multiclean_available

        self.assertTrue(callable(check_multiclean_available))

    def test_clean_segmentation_mask_exists(self):
        """Test that clean_segmentation_mask function exists."""
        from geoai.tools.multiclean import clean_segmentation_mask

        self.assertTrue(callable(clean_segmentation_mask))

    def test_clean_segmentation_mask_signature(self):
        """Test clean_segmentation_mask has expected parameters."""
        from geoai.tools.multiclean import clean_segmentation_mask

        sig = inspect.signature(clean_segmentation_mask)
        self.assertIn("mask", sig.parameters)
        self.assertIn("smooth_edge_size", sig.parameters)
        self.assertIn("min_island_size", sig.parameters)


if __name__ == "__main__":
    unittest.main()
