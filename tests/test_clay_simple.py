"""
Simple test suite for clay.py module - focused on core functionality.

Tests basic model operations with reduced test scope for faster execution.
"""

import os
import tempfile
import datetime
import numpy as np
import torch
import unittest

from geoai.clay import Clay, load_metadata, validate_metadata


class TestClayBasic(unittest.TestCase):
    """Test basic Clay functionality."""

    @classmethod
    def setUpClass(cls):
        """Create a Clay model instance for testing."""
        # Use default checkpoint to avoid repeated downloads
        cls.clay_model = Clay(model_size="large", sensor_name="sentinel-2-l2a")

    def test_model_initialization(self):
        """Test Clay model initializes correctly."""
        self.assertTrue(hasattr(self.clay_model, "module"))
        self.assertIsNotNone(self.clay_model.module)
        self.assertIsNotNone(self.clay_model.device)
        self.assertEqual(self.clay_model.sensor_name, "sentinel-2-l2a")
        self.assertEqual(self.clay_model.model_size, "large")

    def test_embedding_generation_cls_token(self):
        """Test embedding generation returns correct shape for CLS token."""
        # Small image for faster testing
        image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)
        bounds = (-74.5, 40.5, -74.0, 41.0)  # NYC bounds
        date = datetime.datetime(2023, 6, 15, 12, 0, 0)

        embeddings = self.clay_model.generate(
            image=image, bounds=bounds, date=date, only_cls_token=True
        )

        # Check shape: should be [batch=1, features]
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.dim(), 2)  # [batch, features]
        self.assertEqual(embeddings.shape[0], 1)  # batch size
        self.assertGreater(embeddings.shape[1], 0)  # feature dimension
        self.assertFalse(torch.isnan(embeddings).any())

    def test_embedding_generation_torch_input(self):
        """Test embedding generation with torch tensor input."""
        image = torch.randint(0, 10000, size=(256, 256, 10), dtype=torch.float32)

        embeddings = self.clay_model.generate(image=image, only_cls_token=True)

        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.dim(), 2)
        self.assertEqual(embeddings.shape[0], 1)
        self.assertFalse(torch.isnan(embeddings).any())

    def test_save_load_embeddings_npz(self):
        """Test saving and loading embeddings in NPZ format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate embeddings
            image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)
            bounds = (-74.5, 40.5, -74.0, 41.0)
            date = datetime.datetime(2023, 6, 15, 12, 0, 0)

            embeddings = self.clay_model.generate(
                image, bounds, date, only_cls_token=True
            )
            embeddings_np = embeddings.cpu().numpy()

            output_path = os.path.join(temp_dir, "test_embeddings.npz")

            # Save embeddings
            self.clay_model.save_embeddings(
                embeddings=embeddings_np,
                bounds=bounds,
                date=date,
                image_shape=(256, 256),
                output_path=output_path,
                format="npz",
            )

            # Check file exists
            self.assertTrue(os.path.exists(output_path))

            # Load and verify
            data = np.load(output_path)

            np.testing.assert_array_equal(data["embeddings"], embeddings_np)
            self.assertEqual(str(data["sensor_type"]), "sentinel-2-l2a")
            self.assertEqual(str(data["date"]), date.isoformat())
            self.assertEqual(tuple(data["image_shape"]), (256, 256))


class TestClayMetadata(unittest.TestCase):
    """Test Clay metadata functionality."""

    def test_load_metadata_sentinel2(self):
        """Test loading Sentinel-2 metadata."""
        config_path, metadata = load_metadata(sensor_name="sentinel-2-l2a")

        self.assertIn("clay_metadata.yaml", config_path)
        self.assertIn("band_order", metadata)
        self.assertIn("gsd", metadata)
        self.assertIn("bands", metadata)
        self.assertEqual(len(metadata.band_order), 10)
        self.assertEqual(metadata.gsd, 10)

    def test_load_metadata_custom(self):
        """Test loading custom metadata."""
        custom_metadata = {
            "band_order": ["red", "green", "blue"],
            "gsd": 5.0,
            "bands": {
                "mean": {"red": 100, "green": 120, "blue": 80},
                "std": {"red": 50, "green": 60, "blue": 40},
                "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48},
            },
        }

        config_path, metadata = load_metadata(custom_metadata=custom_metadata)

        self.assertIn("clay_metadata.yaml", config_path)
        self.assertEqual(metadata.band_order, ["red", "green", "blue"])
        self.assertEqual(metadata.gsd, 5.0)
        self.assertEqual(metadata.bands.mean.red, 100)

    def test_validate_metadata_valid(self):
        """Test metadata validation with valid metadata."""
        valid_metadata = {
            "band_order": ["red", "green", "blue"],
            "gsd": 10.0,
            "bands": {
                "mean": {"red": 100, "green": 120, "blue": 80},
                "std": {"red": 50, "green": 60, "blue": 40},
                "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48},
            },
        }

        # Should not raise any exception
        validate_metadata(valid_metadata)

    def test_validate_metadata_invalid(self):
        """Test metadata validation with invalid metadata."""
        # Missing band_order
        invalid_metadata = {
            "gsd": 10.0,
            "bands": {
                "mean": {"red": 100},
                "std": {"red": 50},
                "wavelength": {"red": 0.65},
            },
        }

        with self.assertRaisesRegex(ValueError, "Missing required key: band_order"):
            validate_metadata(invalid_metadata)

        # Missing band statistics
        invalid_metadata2 = {
            "band_order": ["red", "green"],
            "gsd": 10.0,
            "bands": {
                "mean": {"red": 100},  # Missing green
                "std": {"red": 50, "green": 60},
                "wavelength": {"red": 0.65, "green": 0.56},
            },
        }

        with self.assertRaisesRegex(ValueError, "Missing mean value for band: green"):
            validate_metadata(invalid_metadata2)


class TestClayCustomSensor(unittest.TestCase):
    """Test Clay with custom sensor."""

    def test_custom_sensor_integration(self):
        """Test Clay with custom 4-band sensor."""
        custom_metadata = {
            "band_order": ["red", "green", "blue", "nir"],
            "gsd": 2.0,
            "bands": {
                "mean": {"red": 120, "green": 140, "blue": 100, "nir": 160},
                "std": {"red": 60, "green": 70, "blue": 50, "nir": 80},
                "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir": 0.84},
            },
        }

        clay = Clay(model_size="large", custom_metadata=custom_metadata)

        # Test with 4-band image
        image = np.random.randint(0, 1000, size=(256, 256, 4), dtype=np.uint16)

        embeddings = clay.generate(image=image, only_cls_token=True)

        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 1)
        self.assertFalse(torch.isnan(embeddings).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
