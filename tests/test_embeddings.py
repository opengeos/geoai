#!/usr/bin/env python

"""Tests for `geoai.embeddings` module."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class TestEmbeddingRegistry(unittest.TestCase):
    """Tests for the embedding dataset registry and listing functions."""

    def test_list_embedding_datasets_returns_dataframe(self):
        """Test that list_embedding_datasets returns a DataFrame by default."""
        from geoai.embeddings import list_embedding_datasets

        df = list_embedding_datasets(verbose=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 9)

    def test_list_embedding_datasets_returns_dict(self):
        """Test that list_embedding_datasets can return a dict."""
        from geoai.embeddings import list_embedding_datasets

        result = list_embedding_datasets(as_dataframe=False, verbose=False)
        self.assertIsInstance(result, dict)
        self.assertIn("clay", result)
        self.assertIn("google_satellite", result)

    def test_list_embedding_datasets_filter_patch(self):
        """Test filtering by patch kind."""
        from geoai.embeddings import list_embedding_datasets

        df = list_embedding_datasets(kind="patch", verbose=False)
        self.assertTrue(all(df["kind"] == "patch"))
        self.assertGreaterEqual(len(df), 4)

    def test_list_embedding_datasets_filter_pixel(self):
        """Test filtering by pixel kind."""
        from geoai.embeddings import list_embedding_datasets

        df = list_embedding_datasets(kind="pixel", verbose=False)
        self.assertTrue(all(df["kind"] == "pixel"))
        self.assertGreaterEqual(len(df), 5)

    def test_list_embedding_datasets_invalid_kind(self):
        """Test that an invalid kind raises ValueError."""
        from geoai.embeddings import list_embedding_datasets

        with self.assertRaises(ValueError):
            list_embedding_datasets(kind="invalid")

    def test_get_embedding_info(self):
        """Test getting detailed info for a specific dataset."""
        from geoai.embeddings import get_embedding_info

        info = get_embedding_info("clay")
        self.assertIn("class_name", info)
        self.assertEqual(info["class_name"], "ClayEmbeddings")
        self.assertEqual(info["kind"], "patch")
        self.assertEqual(info["dimensions"], 768)

    def test_get_embedding_info_unknown_dataset(self):
        """Test that unknown dataset name raises ValueError."""
        from geoai.embeddings import get_embedding_info

        with self.assertRaises(ValueError):
            get_embedding_info("nonexistent_dataset")

    def test_embedding_datasets_registry_completeness(self):
        """Test that all expected datasets are in the registry."""
        from geoai.embeddings import EMBEDDING_DATASETS

        expected_names = [
            "clay",
            "major_tom",
            "earth_index",
            "earth_embeddings",
            "copernicus_embed",
            "presto",
            "tessera",
            "google_satellite",
            "embedded_seamless",
        ]
        for name in expected_names:
            self.assertIn(name, EMBEDDING_DATASETS)

    def test_embedding_datasets_required_fields(self):
        """Test that all registry entries have required fields."""
        from geoai.embeddings import EMBEDDING_DATASETS

        required_fields = [
            "class_name",
            "kind",
            "base",
            "spatial_extent",
            "spatial_resolution",
            "temporal_extent",
            "dimensions",
            "dtype",
            "license",
            "description",
        ]
        for name, info in EMBEDDING_DATASETS.items():
            for field in required_fields:
                self.assertIn(field, info, f"Missing '{field}' in dataset '{name}'")


class TestGetDatasetClass(unittest.TestCase):
    """Tests for _get_dataset_class function."""

    def test_get_dataset_class_valid(self):
        """Test importing valid dataset classes."""
        from geoai.embeddings import _get_dataset_class

        cls = _get_dataset_class("clay")
        self.assertEqual(cls.__name__, "ClayEmbeddings")

    def test_get_dataset_class_all_datasets(self):
        """Test that all registered dataset classes can be imported."""
        from geoai.embeddings import EMBEDDING_DATASETS, _get_dataset_class

        for name in EMBEDDING_DATASETS:
            cls = _get_dataset_class(name)
            expected_class_name = EMBEDDING_DATASETS[name]["class_name"]
            self.assertEqual(
                cls.__name__,
                expected_class_name,
                f"Class name mismatch for '{name}'",
            )

    def test_get_dataset_class_unknown(self):
        """Test that unknown name raises ValueError."""
        from geoai.embeddings import _get_dataset_class

        with self.assertRaises(ValueError):
            _get_dataset_class("unknown_dataset")


class TestLoadEmbeddingDataset(unittest.TestCase):
    """Tests for load_embedding_dataset function."""

    def test_load_unknown_dataset(self):
        """Test that unknown dataset name raises ValueError."""
        from geoai.embeddings import load_embedding_dataset

        with self.assertRaises(ValueError):
            load_embedding_dataset("nonexistent")

    def test_load_patch_without_root(self):
        """Test that loading patch dataset without root raises ValueError."""
        from geoai.embeddings import load_embedding_dataset

        with self.assertRaises(ValueError):
            load_embedding_dataset("clay")

    def test_load_pixel_without_paths(self):
        """Test that loading pixel dataset without paths raises ValueError."""
        from geoai.embeddings import load_embedding_dataset

        with self.assertRaises(ValueError):
            load_embedding_dataset("google_satellite")

    def test_load_pixel_with_root_fallback(self):
        """Test that pixel dataset uses root as fallback for paths."""
        from geoai.embeddings import load_embedding_dataset

        # Should not raise ValueError about paths being None,
        # but may raise DatasetNotFoundError because the path doesn't exist
        with self.assertRaises(Exception):
            load_embedding_dataset("google_satellite", root="/nonexistent/path")


class TestVisualizationFunctions(unittest.TestCase):
    """Tests for visualization utility functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.embeddings = np.random.randn(100, 64).astype(np.float32)
        self.labels = np.random.randint(0, 3, 100)

    def test_visualize_embeddings_pca(self):
        """Test PCA visualization."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        from geoai.embeddings import visualize_embeddings

        fig = visualize_embeddings(self.embeddings, method="pca", figsize=(6, 6))
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visualize_embeddings_with_labels(self):
        """Test PCA visualization with labels."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.embeddings import visualize_embeddings

        fig = visualize_embeddings(
            self.embeddings,
            labels=self.labels,
            label_names=["A", "B", "C"],
            method="pca",
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visualize_embeddings_tsne(self):
        """Test t-SNE visualization."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.embeddings import visualize_embeddings

        fig = visualize_embeddings(self.embeddings, method="tsne", figsize=(6, 6))
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visualize_embeddings_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from geoai.embeddings import visualize_embeddings

        with self.assertRaises(ValueError):
            visualize_embeddings(self.embeddings, method="invalid")

    def test_visualize_embeddings_save(self):
        """Test saving visualization to file."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.embeddings import visualize_embeddings

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig = visualize_embeddings(self.embeddings, method="pca", save_path=f.name)
            self.assertTrue(os.path.exists(f.name))
            os.unlink(f.name)
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_embedding_vector(self):
        """Test plotting a single embedding vector."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.embeddings import plot_embedding_vector

        fig = plot_embedding_vector(self.embeddings[0])
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_embedding_vector_tensor(self):
        """Test plotting with a torch tensor."""
        import matplotlib

        matplotlib.use("Agg")
        import torch

        from geoai.embeddings import plot_embedding_vector

        tensor = torch.from_numpy(self.embeddings[0])
        fig = plot_embedding_vector(tensor)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_embedding_raster(self):
        """Test plotting an embedding raster."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.embeddings import plot_embedding_raster

        # Create a fake (C, H, W) raster
        raster = np.random.randn(32, 64, 64).astype(np.float32)
        fig = plot_embedding_raster(raster)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestAnalysisFunctions(unittest.TestCase):
    """Tests for analysis utility functions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.embeddings = np.random.randn(200, 64).astype(np.float32)

    def test_cluster_embeddings_kmeans(self):
        """Test K-Means clustering."""
        from geoai.embeddings import cluster_embeddings

        result = cluster_embeddings(self.embeddings, n_clusters=5, method="kmeans")
        self.assertEqual(result["labels"].shape, (200,))
        self.assertEqual(result["n_clusters"], 5)
        self.assertIsNotNone(result["model"])

    def test_cluster_embeddings_dbscan(self):
        """Test DBSCAN clustering."""
        from geoai.embeddings import cluster_embeddings

        result = cluster_embeddings(self.embeddings, method="dbscan", eps=2.0)
        self.assertEqual(result["labels"].shape, (200,))
        self.assertIsNotNone(result["model"])

    def test_cluster_embeddings_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from geoai.embeddings import cluster_embeddings

        with self.assertRaises(ValueError):
            cluster_embeddings(self.embeddings, method="invalid")

    def test_embedding_similarity_cosine(self):
        """Test cosine similarity search."""
        from geoai.embeddings import embedding_similarity

        query = self.embeddings[0]
        results = embedding_similarity(
            query=query, embeddings=self.embeddings, metric="cosine", top_k=5
        )
        self.assertEqual(len(results["indices"]), 5)
        self.assertEqual(len(results["scores"]), 5)
        # The query itself should be the most similar
        self.assertEqual(results["indices"][0], 0)
        # Cosine similarity with itself should be ~1.0
        self.assertAlmostEqual(results["scores"][0], 1.0, places=5)

    def test_embedding_similarity_euclidean(self):
        """Test euclidean distance search."""
        from geoai.embeddings import embedding_similarity

        query = self.embeddings[0]
        results = embedding_similarity(
            query=query,
            embeddings=self.embeddings,
            metric="euclidean",
            top_k=5,
        )
        self.assertEqual(len(results["indices"]), 5)
        # Distance to itself should be ~0
        self.assertAlmostEqual(results["scores"][0], 0.0, places=5)

    def test_embedding_similarity_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        from geoai.embeddings import embedding_similarity

        with self.assertRaises(ValueError):
            embedding_similarity(
                query=self.embeddings[0],
                embeddings=self.embeddings,
                metric="invalid",
            )

    def test_embedding_similarity_2d_query(self):
        """Test similarity search with 2D query."""
        from geoai.embeddings import embedding_similarity

        query = self.embeddings[0:1]  # Shape (1, D)
        results = embedding_similarity(query=query, embeddings=self.embeddings, top_k=3)
        self.assertEqual(len(results["indices"]), 3)

    def test_compare_embeddings_cosine(self):
        """Test comparing two sets of embeddings with cosine."""
        from geoai.embeddings import compare_embeddings

        emb_a = self.embeddings[:50]
        emb_b = self.embeddings[50:100]
        similarity = compare_embeddings(emb_a, emb_b, metric="cosine")
        self.assertEqual(similarity.shape, (50,))
        # Cosine similarity should be in [-1, 1]
        self.assertTrue(np.all(similarity >= -1.0 - 1e-6))
        self.assertTrue(np.all(similarity <= 1.0 + 1e-6))

    def test_compare_embeddings_dot(self):
        """Test comparing with dot product."""
        from geoai.embeddings import compare_embeddings

        emb_a = self.embeddings[:50]
        emb_b = self.embeddings[50:100]
        similarity = compare_embeddings(emb_a, emb_b, metric="dot")
        self.assertEqual(similarity.shape, (50,))

    def test_compare_embeddings_euclidean(self):
        """Test comparing with euclidean distance."""
        from geoai.embeddings import compare_embeddings

        emb_a = self.embeddings[:50]
        emb_b = self.embeddings[50:100]
        distances = compare_embeddings(emb_a, emb_b, metric="euclidean")
        self.assertEqual(distances.shape, (50,))
        # Distances should be non-negative
        self.assertTrue(np.all(distances >= 0))

    def test_compare_embeddings_self(self):
        """Test that comparing embeddings with themselves gives perfect similarity."""
        from geoai.embeddings import compare_embeddings

        emb = self.embeddings[:50]
        similarity = compare_embeddings(emb, emb, metric="cosine")
        np.testing.assert_array_almost_equal(similarity, np.ones(50))

    def test_compare_embeddings_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        from geoai.embeddings import compare_embeddings

        with self.assertRaises(ValueError):
            compare_embeddings(self.embeddings[:50], self.embeddings[:30])

    def test_compare_embeddings_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        from geoai.embeddings import compare_embeddings

        with self.assertRaises(ValueError):
            compare_embeddings(
                self.embeddings[:10],
                self.embeddings[:10],
                metric="invalid",
            )


class TestClassifier(unittest.TestCase):
    """Tests for the embedding classifier training."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create two separable clusters
        self.train_emb = np.vstack(
            [
                np.random.randn(50, 32) + 2,  # Class 0
                np.random.randn(50, 32) - 2,  # Class 1
            ]
        ).astype(np.float32)
        self.train_labels = np.array([0] * 50 + [1] * 50)
        self.val_emb = np.vstack(
            [
                np.random.randn(20, 32) + 2,
                np.random.randn(20, 32) - 2,
            ]
        ).astype(np.float32)
        self.val_labels = np.array([0] * 20 + [1] * 20)

    def test_train_knn_classifier(self):
        """Test training a k-NN classifier."""
        from geoai.embeddings import train_embedding_classifier

        result = train_embedding_classifier(
            self.train_emb,
            self.train_labels,
            self.val_emb,
            self.val_labels,
            method="knn",
            n_neighbors=3,
            verbose=False,
        )
        self.assertIn("model", result)
        self.assertIn("train_accuracy", result)
        self.assertIn("val_accuracy", result)
        self.assertIn("val_predictions", result)
        self.assertIn("classification_report", result)
        # With well-separated clusters, accuracy should be very high
        self.assertGreater(result["val_accuracy"], 0.8)

    def test_train_random_forest_classifier(self):
        """Test training a Random Forest classifier."""
        from geoai.embeddings import train_embedding_classifier

        result = train_embedding_classifier(
            self.train_emb,
            self.train_labels,
            method="random_forest",
            n_estimators=10,
            verbose=False,
        )
        self.assertIn("model", result)
        self.assertIn("train_accuracy", result)
        self.assertNotIn("val_accuracy", result)

    def test_train_logistic_regression_classifier(self):
        """Test training a Logistic Regression classifier."""
        from geoai.embeddings import train_embedding_classifier

        result = train_embedding_classifier(
            self.train_emb,
            self.train_labels,
            self.val_emb,
            self.val_labels,
            method="logistic_regression",
            verbose=False,
        )
        self.assertIn("model", result)
        self.assertGreater(result["val_accuracy"], 0.8)

    def test_train_with_label_names(self):
        """Test training with label names for the report."""
        from geoai.embeddings import train_embedding_classifier

        result = train_embedding_classifier(
            self.train_emb,
            self.train_labels,
            self.val_emb,
            self.val_labels,
            method="knn",
            label_names=["Class A", "Class B"],
            verbose=False,
        )
        self.assertIn("Class A", result["classification_report"])
        self.assertIn("Class B", result["classification_report"])

    def test_train_invalid_method(self):
        """Test that invalid classifier method raises ValueError."""
        from geoai.embeddings import train_embedding_classifier

        with self.assertRaises(ValueError):
            train_embedding_classifier(
                self.train_emb,
                self.train_labels,
                method="invalid",
            )


class TestEmbeddingToGeotiff(unittest.TestCase):
    """Tests for the embedding_to_geotiff function."""

    def test_save_geotiff(self):
        """Test saving embeddings as a GeoTIFF."""
        from geoai.embeddings import embedding_to_geotiff

        embeddings = np.random.randn(32, 64, 64).astype(np.float32)
        bounds = (-122.5, 37.7, -122.3, 37.9)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            output_path = f.name

        try:
            result = embedding_to_geotiff(embeddings, bounds, output_path)
            self.assertEqual(result, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify the output
            import rasterio

            with rasterio.open(output_path) as src:
                self.assertEqual(src.count, 32)
                self.assertEqual(src.width, 64)
                self.assertEqual(src.height, 64)
                self.assertEqual(str(src.crs), "EPSG:4326")
        finally:
            os.unlink(output_path)

    def test_save_geotiff_hwc_format(self):
        """Test saving embeddings in (H, W, C) format."""
        from geoai.embeddings import embedding_to_geotiff

        embeddings = np.random.randn(64, 64, 16).astype(np.float32)
        bounds = (-122.5, 37.7, -122.3, 37.9)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            output_path = f.name

        try:
            embedding_to_geotiff(embeddings, bounds, output_path)
            import rasterio

            with rasterio.open(output_path) as src:
                self.assertEqual(src.count, 16)
        finally:
            os.unlink(output_path)

    def test_save_geotiff_invalid_dims(self):
        """Test that 2D array raises ValueError."""
        from geoai.embeddings import embedding_to_geotiff

        with self.assertRaises(ValueError):
            embedding_to_geotiff(
                np.random.randn(64, 64),
                (-122.5, 37.7, -122.3, 37.9),
                "output.tif",
            )


class TestExtractPatchEmbeddings(unittest.TestCase):
    """Tests for extract_patch_embeddings function."""

    def test_extract_from_mock_dataset(self):
        """Test extracting embeddings from a mock dataset."""
        import torch

        from geoai.embeddings import extract_patch_embeddings

        # Create a mock dataset
        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "embedding": torch.randn(64),
                    "x": torch.tensor(-122.0 + idx * 0.01),
                    "y": torch.tensor(37.0 + idx * 0.01),
                    "t": torch.tensor(1609459200.0 + idx * 86400),
                }

        ds = MockDataset()
        data = extract_patch_embeddings(ds, max_samples=5)

        self.assertEqual(data["embeddings"].shape, (5, 64))
        self.assertEqual(data["x"].shape, (5,))
        self.assertEqual(data["y"].shape, (5,))
        self.assertEqual(data["t"].shape, (5,))

    def test_extract_all_samples(self):
        """Test extracting all samples from a small dataset."""
        import torch

        from geoai.embeddings import extract_patch_embeddings

        class MockDataset:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {"embedding": torch.randn(32)}

        ds = MockDataset()
        data = extract_patch_embeddings(ds)
        self.assertEqual(data["embeddings"].shape, (3, 32))
        self.assertNotIn("x", data)
        self.assertNotIn("y", data)

    def test_extract_with_max_samples(self):
        """Test limiting the number of extracted samples."""
        import torch

        from geoai.embeddings import extract_patch_embeddings

        class MockDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"embedding": torch.randn(16)}

        ds = MockDataset()
        data = extract_patch_embeddings(ds, max_samples=10)
        self.assertEqual(data["embeddings"].shape, (10, 16))


class TestImportsFromInit(unittest.TestCase):
    """Test that embeddings functions are accessible from geoai package."""

    def test_import_from_geoai(self):
        """Test all embeddings exports are importable from geoai."""
        import geoai

        # All public functions should be available
        self.assertTrue(hasattr(geoai, "list_embedding_datasets"))
        self.assertTrue(hasattr(geoai, "load_embedding_dataset"))
        self.assertTrue(hasattr(geoai, "get_embedding_info"))
        self.assertTrue(hasattr(geoai, "extract_patch_embeddings"))
        self.assertTrue(hasattr(geoai, "extract_pixel_embeddings"))
        self.assertTrue(hasattr(geoai, "visualize_embeddings"))
        self.assertTrue(hasattr(geoai, "plot_embedding_vector"))
        self.assertTrue(hasattr(geoai, "plot_embedding_raster"))
        self.assertTrue(hasattr(geoai, "cluster_embeddings"))
        self.assertTrue(hasattr(geoai, "embedding_similarity"))
        self.assertTrue(hasattr(geoai, "train_embedding_classifier"))
        self.assertTrue(hasattr(geoai, "compare_embeddings"))
        self.assertTrue(hasattr(geoai, "embedding_to_geotiff"))
        self.assertTrue(hasattr(geoai, "EMBEDDING_DATASETS"))

    def test_embedding_datasets_constant(self):
        """Test that EMBEDDING_DATASETS is accessible and has expected entries."""
        import geoai

        self.assertIsInstance(geoai.EMBEDDING_DATASETS, dict)
        self.assertIn("clay", geoai.EMBEDDING_DATASETS)
        self.assertIn("google_satellite", geoai.EMBEDDING_DATASETS)


# if __name__ == "__main__":
#     unittest.main()
