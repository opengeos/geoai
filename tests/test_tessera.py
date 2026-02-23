#!/usr/bin/env python

"""Tests for `geoai.tessera` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestTesseraImport(unittest.TestCase):
    """Tests for tessera module import behavior."""

    def test_module_imports(self):
        """Test that the tessera module can be imported."""
        import geoai.tessera

        self.assertTrue(hasattr(geoai.tessera, "tessera_download"))

    def test_tessera_download_exists(self):
        """Test that tessera_download function exists."""
        from geoai.tessera import tessera_download

        self.assertTrue(callable(tessera_download))

    def test_tessera_fetch_embeddings_exists(self):
        """Test that tessera_fetch_embeddings function exists."""
        from geoai.tessera import tessera_fetch_embeddings

        self.assertTrue(callable(tessera_fetch_embeddings))

    def test_tessera_coverage_exists(self):
        """Test that tessera_coverage function exists."""
        from geoai.tessera import tessera_coverage

        self.assertTrue(callable(tessera_coverage))

    def test_tessera_visualize_rgb_exists(self):
        """Test that tessera_visualize_rgb function exists."""
        from geoai.tessera import tessera_visualize_rgb

        self.assertTrue(callable(tessera_visualize_rgb))

    def test_tessera_tile_count_exists(self):
        """Test that tessera_tile_count function exists."""
        from geoai.tessera import tessera_tile_count

        self.assertTrue(callable(tessera_tile_count))

    def test_tessera_available_years_exists(self):
        """Test that tessera_available_years function exists."""
        from geoai.tessera import tessera_available_years

        self.assertTrue(callable(tessera_available_years))

    def test_tessera_sample_points_exists(self):
        """Test that tessera_sample_points function exists."""
        from geoai.tessera import tessera_sample_points

        self.assertTrue(callable(tessera_sample_points))


class TestTesseraConstants(unittest.TestCase):
    """Tests for tessera module constants."""

    def test_embedding_dim_constant(self):
        """Test that TESSERA_EMBEDDING_DIM constant is defined correctly."""
        from geoai.tessera import TESSERA_EMBEDDING_DIM

        self.assertEqual(TESSERA_EMBEDDING_DIM, 128)


class TestTesseraCheckGeotessera(unittest.TestCase):
    """Tests for _check_geotessera helper function."""

    def test_check_geotessera_exists(self):
        """Test that _check_geotessera helper function exists."""
        from geoai.tessera import _check_geotessera

        self.assertTrue(callable(_check_geotessera))

    @patch.dict("sys.modules", {"geotessera": None})
    def test_check_geotessera_raises_without_package(self):
        """Test that _check_geotessera raises ImportError when geotessera is missing."""
        from geoai.tessera import _check_geotessera

        with self.assertRaises(ImportError) as ctx:
            _check_geotessera()
        self.assertIn("geotessera", str(ctx.exception))


class TestTesseraSignatures(unittest.TestCase):
    """Tests for tessera function signatures."""

    def test_tessera_download_params(self):
        """Test tessera_download has expected parameters."""
        from geoai.tessera import tessera_download

        sig = inspect.signature(tessera_download)
        self.assertIn("bbox", sig.parameters)
        self.assertIn("lon", sig.parameters)
        self.assertIn("lat", sig.parameters)
        self.assertIn("year", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("output_format", sig.parameters)
        self.assertIn("bands", sig.parameters)
        self.assertIn("compress", sig.parameters)
        self.assertIn("region_file", sig.parameters)
        self.assertIn("dataset_version", sig.parameters)

    def test_tessera_fetch_embeddings_params(self):
        """Test tessera_fetch_embeddings has expected parameters."""
        from geoai.tessera import tessera_fetch_embeddings

        sig = inspect.signature(tessera_fetch_embeddings)
        self.assertIn("bbox", sig.parameters)
        self.assertIn("year", sig.parameters)
        self.assertIn("bands", sig.parameters)
        self.assertIn("dataset_version", sig.parameters)

    def test_tessera_coverage_params(self):
        """Test tessera_coverage has expected parameters."""
        from geoai.tessera import tessera_coverage

        sig = inspect.signature(tessera_coverage)
        self.assertIn("year", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("region_bbox", sig.parameters)
        self.assertIn("region_file", sig.parameters)
        self.assertIn("tile_color", sig.parameters)
        self.assertIn("tile_alpha", sig.parameters)
        self.assertIn("width_pixels", sig.parameters)
        self.assertIn("show_countries", sig.parameters)

    def test_tessera_visualize_rgb_params(self):
        """Test tessera_visualize_rgb has expected parameters."""
        from geoai.tessera import tessera_visualize_rgb

        sig = inspect.signature(tessera_visualize_rgb)
        self.assertIn("geotiff_dir", sig.parameters)
        self.assertIn("bands", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("normalize", sig.parameters)
        self.assertIn("figsize", sig.parameters)
        self.assertIn("title", sig.parameters)

    def test_tessera_tile_count_params(self):
        """Test tessera_tile_count has expected parameters."""
        from geoai.tessera import tessera_tile_count

        sig = inspect.signature(tessera_tile_count)
        self.assertIn("bbox", sig.parameters)
        self.assertIn("year", sig.parameters)
        self.assertIn("dataset_version", sig.parameters)

    def test_tessera_available_years_params(self):
        """Test tessera_available_years has expected parameters."""
        from geoai.tessera import tessera_available_years

        sig = inspect.signature(tessera_available_years)
        self.assertIn("dataset_version", sig.parameters)

    def test_tessera_sample_points_params(self):
        """Test tessera_sample_points has expected parameters."""
        from geoai.tessera import tessera_sample_points

        sig = inspect.signature(tessera_sample_points)
        self.assertIn("points", sig.parameters)
        self.assertIn("year", sig.parameters)
        self.assertIn("embeddings_dir", sig.parameters)
        self.assertIn("auto_download", sig.parameters)
        self.assertIn("dataset_version", sig.parameters)


class TestTesseraDefaults(unittest.TestCase):
    """Tests for tessera function default values."""

    def test_tessera_download_defaults(self):
        """Test tessera_download has correct default values."""
        from geoai.tessera import tessera_download

        sig = inspect.signature(tessera_download)
        self.assertEqual(sig.parameters["year"].default, 2024)
        self.assertEqual(sig.parameters["output_dir"].default, "./tessera_output")
        self.assertEqual(sig.parameters["output_format"].default, "tiff")
        self.assertEqual(sig.parameters["compress"].default, "lzw")
        self.assertEqual(sig.parameters["dataset_version"].default, "v1")

    def test_tessera_download_raises_without_location(self):
        """Test tessera_download raises ValueError without bbox, lon/lat, or region_file."""
        from geoai.tessera import tessera_download

        mock_geotessera = MagicMock()
        with patch.dict("sys.modules", {"geotessera": mock_geotessera}):
            with self.assertRaises(Exception):
                # Should raise because no bbox, lon/lat, or region_file provided
                tessera_download()


if __name__ == "__main__":
    unittest.main()
