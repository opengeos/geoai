#!/usr/bin/env python

"""Tests for `geoai.dinov3` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestDinov3Import(unittest.TestCase):
    """Tests for dinov3 module import behavior."""

    def test_module_imports(self):
        """Test that the dinov3 module can be imported."""
        import geoai.dinov3

        self.assertTrue(hasattr(geoai.dinov3, "DINOv3GeoProcessor"))

    def test_dinov3_geo_processor_class_exists(self):
        """Test that DINOv3GeoProcessor class exists."""
        from geoai.dinov3 import DINOv3GeoProcessor

        self.assertTrue(callable(DINOv3GeoProcessor))

    def test_create_similarity_map_exists(self):
        """Test that create_similarity_map function exists."""
        from geoai.dinov3 import create_similarity_map

        self.assertTrue(callable(create_similarity_map))

    def test_analyze_image_patches_exists(self):
        """Test that analyze_image_patches function exists."""
        from geoai.dinov3 import analyze_image_patches

        self.assertTrue(callable(analyze_image_patches))

    def test_visualize_similarity_results_exists(self):
        """Test that visualize_similarity_results function exists."""
        from geoai.dinov3 import visualize_similarity_results

        self.assertTrue(callable(visualize_similarity_results))


class TestDinov3Signatures(unittest.TestCase):
    """Tests for dinov3 function and method signatures."""

    def test_dinov3_geo_processor_init_params(self):
        """Test DINOv3GeoProcessor.__init__ has expected parameters."""
        from geoai.dinov3 import DINOv3GeoProcessor

        sig = inspect.signature(DINOv3GeoProcessor.__init__)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("weights_path", sig.parameters)
        self.assertIn("device", sig.parameters)

    def test_compute_similarity_params(self):
        """Test compute_similarity method has expected parameters."""
        from geoai.dinov3 import DINOv3GeoProcessor

        sig = inspect.signature(DINOv3GeoProcessor.compute_similarity)
        self.assertIn("source", sig.parameters)
        self.assertIn("features", sig.parameters)
        self.assertIn("query_coords", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("target_size", sig.parameters)
        self.assertIn("coord_crs", sig.parameters)
        self.assertIn("use_interpolation", sig.parameters)

    def test_create_similarity_map_params(self):
        """Test create_similarity_map function has expected parameters."""
        from geoai.dinov3 import create_similarity_map

        sig = inspect.signature(create_similarity_map)
        self.assertIn("input_image", sig.parameters)
        self.assertIn("query_coords", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("weights_path", sig.parameters)
        self.assertIn("target_size", sig.parameters)

    def test_analyze_image_patches_params(self):
        """Test analyze_image_patches function has expected parameters."""
        from geoai.dinov3 import analyze_image_patches

        sig = inspect.signature(analyze_image_patches)
        self.assertIn("input_image", sig.parameters)
        self.assertIn("query_points", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("weights_path", sig.parameters)

    def test_visualize_similarity_params(self):
        """Test visualize_similarity method has expected parameters."""
        from geoai.dinov3 import DINOv3GeoProcessor

        sig = inspect.signature(DINOv3GeoProcessor.visualize_similarity)
        self.assertIn("source", sig.parameters)
        self.assertIn("similarity_data", sig.parameters)
        self.assertIn("query_coords", sig.parameters)
        self.assertIn("colormap", sig.parameters)
        self.assertIn("overlay", sig.parameters)


class TestDinov3Init(unittest.TestCase):
    """Tests for DINOv3GeoProcessor initialization."""

    @patch("geoai.dinov3.DINOv3GeoProcessor._load_model")
    def test_init_with_mocked_model(self, mock_load_model):
        """Test that init succeeds with mocked model loading."""
        mock_model = MagicMock()
        mock_model.patch_size = 16
        mock_model.embed_dim = 1024
        mock_load_model.return_value = mock_model

        from geoai.dinov3 import DINOv3GeoProcessor

        processor = DINOv3GeoProcessor()
        self.assertIsNotNone(processor.model)
        self.assertEqual(processor.model_name, "dinov3_vitl16")
        self.assertEqual(processor.patch_size, 16)
        self.assertEqual(processor.embed_dim, 1024)
        mock_load_model.assert_called_once()

    @patch("geoai.dinov3.DINOv3GeoProcessor._load_model")
    def test_init_custom_model_name(self, mock_load_model):
        """Test init with a custom model name."""
        mock_model = MagicMock()
        mock_model.patch_size = 16
        mock_model.embed_dim = 384
        mock_load_model.return_value = mock_model

        from geoai.dinov3 import DINOv3GeoProcessor

        processor = DINOv3GeoProcessor(model_name="dinov3_vits16")
        self.assertEqual(processor.model_name, "dinov3_vits16")

    @patch("geoai.dinov3.DINOv3GeoProcessor._load_model")
    def test_init_sets_transform(self, mock_load_model):
        """Test that init sets up image transforms."""
        mock_model = MagicMock()
        mock_model.patch_size = 16
        mock_model.embed_dim = 1024
        mock_load_model.return_value = mock_model

        from geoai.dinov3 import DINOv3GeoProcessor

        processor = DINOv3GeoProcessor()
        self.assertIsNotNone(processor.transform)


if __name__ == "__main__":
    unittest.main()
