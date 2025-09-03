"""
Test suite for clay.py module.

Tests model download, embedding generation, save/load functionality,
and metadata validation using the actual Clay model.
"""

import os
import tempfile
import datetime
import numpy as np
import torch
import unittest

from geoai.clay import Clay, load_embeddings, load_metadata, validate_metadata


class TestClayModelDownload(unittest.TestCase):
    """Test Clay model download functionality."""

    def test_model_download_and_initialization(self):
        """Test that model downloads and initializes correctly when checkpoint doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_clay.ckpt")

            # Initialize Clay with non-existent checkpoint - should trigger download
            # Note: v1.5 checkpoint is large size, so we use large to match
            clay = Clay(
                checkpoint_path=checkpoint_path,
                model_size="large",  # Use large to match v1.5 checkpoint
                sensor_name="sentinel-2-l2a",
            )

            # Verify the model was loaded
            self.assertTrue(hasattr(clay, "module"))
            self.assertIsNotNone(clay.module)
            self.assertIsNotNone(clay.device)

            # Verify checkpoint was downloaded
            self.assertTrue(os.path.exists(clay.checkpoint_path))


class TestClayEmbeddingGeneration(unittest.TestCase):
    """Test Clay embedding generation with random data."""

    @classmethod
    def setUpClass(cls):
        """Create a Clay model instance for testing."""
        cls.clay_model = Clay(
            model_size="large",  # Use large to match v1.5 checkpoint
            sensor_name="sentinel-2-l2a",
        )

    def test_embedding_shape_full_sequence(self):
        """Test embedding generation returns correct shape for full sequence."""
        # Create random test data matching Sentinel-2 L2A format (10 bands) - smaller for faster testing
        image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)
        bounds = (-74.5, 40.5, -74.0, 41.0)  # NYC bounds
        date = datetime.datetime(2023, 6, 15, 12, 0, 0)

        embeddings = self.clay_model.generate(
            image=image, bounds=bounds, date=date, only_cls_token=False
        )

        # Check embeddings are generated
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.dim(), 3)  # [batch, tokens, features]
        self.assertEqual(embeddings.shape[0], 1)  # batch size
        self.assertGreater(embeddings.shape[2], 0)  # feature dimension
        self.assertFalse(torch.isnan(embeddings).any())

    def test_embedding_shape_cls_token_only(self):
        """Test embedding generation returns correct shape for CLS token only."""
        image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)
        bounds = (-74.5, 40.5, -74.0, 41.0)
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

    def test_embedding_with_torch_tensor_input(self):
        """Test embedding generation with torch tensor input."""
        # Create torch tensor input
        image = torch.randint(0, 10000, size=(256, 256, 10), dtype=torch.float32)

        embeddings = self.clay_model.generate(image=image, only_cls_token=True)

        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.dim(), 2)
        self.assertEqual(embeddings.shape[0], 1)
        self.assertFalse(torch.isnan(embeddings).any())

    def test_embedding_without_bounds_date(self):
        """Test embedding generation without bounds and date."""
        image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)

        embeddings = self.clay_model.generate(image=image, only_cls_token=True)

        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.dim(), 2)
        self.assertEqual(embeddings.shape[0], 1)
        self.assertFalse(torch.isnan(embeddings).any())

    def test_embedding_consistency(self):
        """Test that same input produces same embeddings."""
        image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)
        bounds = (-74.5, 40.5, -74.0, 41.0)
        date = datetime.datetime(2023, 6, 15, 12, 0, 0)

        # Generate embeddings twice
        embeddings1 = self.clay_model.generate(
            image=image, bounds=bounds, date=date, only_cls_token=True
        )
        embeddings2 = self.clay_model.generate(
            image=image, bounds=bounds, date=date, only_cls_token=True
        )

        # Should be identical
        torch.testing.assert_close(embeddings1, embeddings2)


class TestClayEmbeddingSaveLoad(unittest.TestCase):
    """Test Clay embedding save and load functionality."""

    def setUp(self):
        """Create a Clay model instance for testing."""
        self.clay_model = Clay(model_size="large", sensor_name="sentinel-2-l2a")

    def test_save_load_embeddings_npz(self):
        """Test saving and loading embeddings in NPZ format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate real embeddings
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
                image_shape=(512, 512),
                output_path=output_path,
                format="npz",
            )

            # Check file exists
            self.assertTrue(os.path.exists(output_path))

            # Load and verify - note that our save_embeddings doesn't include tile_coords and num_tiles
            # so we need to load directly from NPZ instead of using load_embeddings function
            data = np.load(output_path)

            np.testing.assert_array_equal(data["embeddings"], embeddings_np)
            self.assertEqual(str(data["sensor_type"]), "sentinel-2-l2a")
            self.assertEqual(str(data["date"]), date.isoformat())
            self.assertEqual(tuple(data["image_shape"]), (512, 512))
            np.testing.assert_array_equal(data["bounds"], np.array(bounds))

    def test_save_load_embeddings_pt(self):
        """Test saving and loading embeddings in PT format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate real embeddings
            image = np.random.randint(0, 10000, size=(256, 256, 10), dtype=np.uint16)
            bounds = (-74.5, 40.5, -74.0, 41.0)
            date = datetime.datetime(2023, 6, 15, 12, 0, 0)

            embeddings = self.clay_model.generate(
                image, bounds, date, only_cls_token=True
            )
            embeddings_np = embeddings.cpu().numpy()

            output_path = os.path.join(temp_dir, "test_embeddings.pt")

            # Save embeddings
            self.clay_model.save_embeddings(
                embeddings=embeddings_np,
                bounds=bounds,
                date=date,
                image_shape=(512, 512),
                output_path=output_path,
                format="pt",
            )

            # Check file exists
            self.assertTrue(os.path.exists(output_path))

            # Load and verify
            loaded_data = torch.load(output_path, map_location="cpu")

            torch.testing.assert_close(
                loaded_data["embeddings"], torch.from_numpy(embeddings_np)
            )
            self.assertEqual(loaded_data["sensor_type"], "sentinel-2-l2a")
            self.assertEqual(loaded_data["date"], date.isoformat())
            self.assertEqual(loaded_data["image_shape"], (512, 512))


class TestClayLoadMetadata(unittest.TestCase):
    """Test Clay load_metadata functionality."""

    def test_load_metadata_with_sensor_name(self):
        """Test loading metadata with valid sensor name."""
        config_path, metadata = load_metadata(sensor_name="sentinel-2-l2a")

        self.assertIn("clay_metadata.yaml", config_path)
        self.assertIn("band_order", metadata)
        self.assertIn("gsd", metadata)
        self.assertIn("bands", metadata)
        self.assertEqual(len(metadata.band_order), 10)
        self.assertEqual(metadata.gsd, 10)

    def test_load_metadata_with_landsat_sensor(self):
        """Test loading metadata with Landsat sensor."""
        config_path, metadata = load_metadata(sensor_name="landsat-c2l2-sr")

        self.assertEqual(len(metadata.band_order), 6)
        self.assertEqual(metadata.gsd, 30)
        self.assertIn("red", metadata.bands.mean)

    def test_load_metadata_with_naip_sensor(self):
        """Test loading metadata with NAIP sensor."""
        config_path, metadata = load_metadata(sensor_name="naip")

        self.assertEqual(len(metadata.band_order), 4)
        self.assertEqual(metadata.gsd, 1.0)
        self.assertIn("nir", metadata.bands.mean)

    def test_load_metadata_with_invalid_sensor(self):
        """Test loading metadata with invalid sensor name."""
        with self.assertRaisesRegex(ValueError, "Unknown sensor"):
            load_metadata(sensor_name="invalid-sensor")

    def test_load_metadata_with_custom_metadata(self):
        """Test loading metadata with custom metadata dictionary."""
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

    def test_load_metadata_no_parameters(self):
        """Test loading metadata with no parameters provided."""
        with self.assertRaisesRegex(
            ValueError, "Must provide either sensor_name or custom_metadata"
        ):
            load_metadata()


class TestClayValidateMetadata(unittest.TestCase):
    """Test Clay validate_metadata functionality."""

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

    def test_validate_metadata_missing_band_order(self):
        """Test metadata validation with missing band_order."""
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

    def test_validate_metadata_missing_gsd(self):
        """Test metadata validation with missing gsd."""
        invalid_metadata = {
            "band_order": ["red"],
            "bands": {
                "mean": {"red": 100},
                "std": {"red": 50},
                "wavelength": {"red": 0.65},
            },
        }

        with self.assertRaisesRegex(ValueError, "Missing required key: gsd"):
            validate_metadata(invalid_metadata)

    def test_validate_metadata_missing_bands(self):
        """Test metadata validation with missing bands."""
        invalid_metadata = {"band_order": ["red"], "gsd": 10.0}

        with self.assertRaisesRegex(ValueError, "Missing required key: bands"):
            validate_metadata(invalid_metadata)

    def test_validate_metadata_empty_band_order(self):
        """Test metadata validation with empty band_order."""
        invalid_metadata = {
            "band_order": [],
            "gsd": 10.0,
            "bands": {"mean": {}, "std": {}, "wavelength": {}},
        }

        with self.assertRaisesRegex(ValueError, "band_order must be a non-empty list"):
            validate_metadata(invalid_metadata)

    def test_validate_metadata_missing_band_stats(self):
        """Test metadata validation with missing band statistics."""
        invalid_metadata = {
            "band_order": ["red", "green"],
            "gsd": 10.0,
            "bands": {
                "mean": {"red": 100},  # Missing green
                "std": {"red": 50, "green": 60},
                "wavelength": {"red": 0.65, "green": 0.56},
            },
        }

        with self.assertRaisesRegex(ValueError, "Missing mean value for band: green"):
            validate_metadata(invalid_metadata)

    def test_validate_metadata_extra_band_stats(self):
        """Test metadata validation with extra band statistics."""
        invalid_metadata = {
            "band_order": ["red"],
            "gsd": 10.0,
            "bands": {
                "mean": {"red": 100, "green": 120},  # Extra green
                "std": {"red": 50, "green": 60},
                "wavelength": {"red": 0.65, "green": 0.56},
            },
        }

        with self.assertRaisesRegex(
            ValueError, "bands.mean has 2 entries but expected 1"
        ):
            validate_metadata(invalid_metadata)

    def test_validate_metadata_invalid_bands_structure(self):
        """Test metadata validation with invalid bands structure."""
        invalid_metadata = {
            "band_order": ["red"],
            "gsd": 10.0,
            "bands": {
                "mean": "not_a_dict",
                "std": {"red": 50},
                "wavelength": {"red": 0.65},
            },
        }

        with self.assertRaisesRegex(ValueError, "bands.mean must be a dictionary"):
            validate_metadata(invalid_metadata)

    def test_validate_metadata_missing_bands_key(self):
        """Test metadata validation with missing bands key."""
        invalid_metadata = {
            "band_order": ["red"],
            "gsd": 10.0,
            "bands": {
                "mean": {"red": 100},
                "std": {"red": 50},
                # Missing wavelength
            },
        }

        with self.assertRaisesRegex(
            ValueError, "Missing required bands key: wavelength"
        ):
            validate_metadata(invalid_metadata)


class TestClayCustomSensorIntegration(unittest.TestCase):
    """Test Clay with custom sensor metadata."""

    def test_clay_with_custom_sensor(self):
        """Test Clay initialization and embedding generation with custom sensor."""
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


class TestLoadEmbeddingsInvalidFormat(unittest.TestCase):
    """Test invalid format handling for load_embeddings."""

    def test_load_embeddings_invalid_format(self):
        """Test loading embeddings with invalid file format."""
        with self.assertRaisesRegex(ValueError, "Unsupported file format"):
            load_embeddings("test.txt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
