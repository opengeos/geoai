#!/usr/bin/env python

"""Tests for `geoai.pointcloud` module."""

import inspect
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import pytest

laspy = pytest.importorskip("laspy", reason="laspy not installed")

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestPointcloudImport(unittest.TestCase):
    """Tests for pointcloud module import behaviour."""

    @patch.dict(
        "sys.modules", {"open3d.ml.torch": MagicMock(), "open3d.ml": MagicMock()}
    )
    @patch.dict("sys.modules", {"laspy": MagicMock()})
    def test_module_imports(self):
        import importlib

        mod = importlib.import_module("geoai.pointcloud")
        self.assertTrue(hasattr(mod, "PointCloudClassifier"))

    def test_constants_importable(self):
        # These should be importable without heavy deps via lazy imports
        from geoai.pointcloud import ASPRS_CLASSES, SUPPORTED_MODELS, DEFAULT_CACHE_DIR

        self.assertIsInstance(ASPRS_CLASSES, dict)
        self.assertIsInstance(SUPPORTED_MODELS, dict)
        self.assertIsInstance(DEFAULT_CACHE_DIR, str)

    def test_asprs_classes_content(self):
        from geoai.pointcloud import ASPRS_CLASSES

        self.assertIn(2, ASPRS_CLASSES)
        self.assertEqual(ASPRS_CLASSES[2], "Ground")
        self.assertIn(6, ASPRS_CLASSES)
        self.assertEqual(ASPRS_CLASSES[6], "Building")
        self.assertIn(9, ASPRS_CLASSES)
        self.assertEqual(ASPRS_CLASSES[9], "Water")

    def test_supported_models_content(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        self.assertIn("RandLANet_DALES", SUPPORTED_MODELS)
        self.assertIn("RandLANet_3DEP", SUPPORTED_MODELS)
        self.assertIn("RandLANet_SemanticKITTI", SUPPORTED_MODELS)
        self.assertIn("RandLANet_Toronto3D", SUPPORTED_MODELS)
        self.assertIn("RandLANet_S3DIS", SUPPORTED_MODELS)

    def test_supported_models_required_keys(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        for name, info in SUPPORTED_MODELS.items():
            self.assertIn("url", info, f"Missing 'url' in {name}")
            self.assertIn("description", info, f"Missing 'description' in {name}")
            self.assertIn("num_classes", info, f"Missing 'num_classes' in {name}")
            self.assertIn("dataset", info, f"Missing 'dataset' in {name}")
            self.assertIn("class_names", info, f"Missing 'class_names' in {name}")
            self.assertIn("config", info, f"Missing 'config' in {name}")
            self.assertEqual(
                len(info["class_names"]),
                info["num_classes"],
                f"class_names length mismatch in {name}",
            )


class TestPointcloudAllExports(unittest.TestCase):
    def test_all_exports_defined(self):
        from geoai.pointcloud import __all__

        expected = [
            "PointCloudClassifier",
            "classify_point_cloud",
            "list_pointcloud_models",
            "ASPRS_CLASSES",
            "SUPPORTED_MODELS",
            "DEFAULT_CACHE_DIR",
        ]
        for name in expected:
            self.assertIn(name, __all__)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestPointCloudClassifierSignatures(unittest.TestCase):
    def test_init_params(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.__init__)
        expected = [
            "model_name",
            "checkpoint_path",
            "device",
            "cache_dir",
            "num_classes",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_init_defaults(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.__init__)
        self.assertEqual(
            sig.parameters["model_name"].default,
            "RandLANet_DALES",
        )

    def test_classify_params(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.classify)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)

    def test_classify_batch_params(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.classify_batch)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)

    def test_train_params(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.train)
        expected = [
            "train_dir",
            "val_dir",
            "epochs",
            "learning_rate",
            "batch_size",
            "save_dir",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_visualize_params(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.visualize)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("backend", sig.parameters)
        self.assertIn("cmap", sig.parameters)

    def test_summary_params(self):
        from geoai.pointcloud import PointCloudClassifier

        sig = inspect.signature(PointCloudClassifier.summary)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("class_map", sig.parameters)


class TestConvenienceFunctionSignatures(unittest.TestCase):
    def test_classify_point_cloud_params(self):
        from geoai.pointcloud import classify_point_cloud

        sig = inspect.signature(classify_point_cloud)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("cache_dir", sig.parameters)

    def test_list_pointcloud_models_returns_dict(self):
        from geoai.pointcloud import list_pointcloud_models

        result = list_pointcloud_models()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        for name, desc in result.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)


# ---------------------------------------------------------------------------
# Init / validation tests
# ---------------------------------------------------------------------------


class TestPointCloudClassifierInit(unittest.TestCase):
    def test_raises_for_unknown_model(self):
        from geoai.pointcloud import PointCloudClassifier

        with self.assertRaises(ValueError) as ctx:
            PointCloudClassifier(model_name="nonexistent_model")
        self.assertIn("nonexistent_model", str(ctx.exception))


class TestDownloadCheckpoint(unittest.TestCase):
    def test_raises_for_invalid_model(self):
        from geoai.pointcloud import _download_checkpoint

        with self.assertRaises(ValueError):
            _download_checkpoint("invalid_model_name")

    def test_skips_download_if_exists(self):
        from geoai.pointcloud import SUPPORTED_MODELS, _download_checkpoint

        with tempfile.TemporaryDirectory() as td:
            info = SUPPORTED_MODELS["RandLANet_Toronto3D"]
            filename = os.path.basename(info["url"])
            ckpt_path = os.path.join(td, filename)
            with open(ckpt_path, "w") as f:
                f.write("fake")

            path = _download_checkpoint("RandLANet_Toronto3D", td)
            self.assertEqual(path, ckpt_path)


# ---------------------------------------------------------------------------
# LAS I/O tests (using laspy directly)
# ---------------------------------------------------------------------------


class TestReadPointCloud(unittest.TestCase):
    """Tests for _read_point_cloud helper."""

    def _make_test_las(self, n_points=100):
        """Create a temporary LAS file for testing."""
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n_points)
        las.y = np.random.uniform(0, 100, n_points)
        las.z = np.random.uniform(0, 50, n_points)
        las.intensity = np.random.randint(0, 65535, n_points).astype(np.uint16)
        las.classification = np.random.randint(0, 6, n_points).astype(np.uint8)
        return las

    def test_read_returns_correct_shapes(self):
        from geoai.pointcloud import _read_point_cloud

        las = self._make_test_las(200)
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            xyz, features, las_obj = _read_point_cloud(f.name, in_channels=6)
        os.unlink(f.name)

        self.assertEqual(xyz.shape, (200, 3))
        self.assertEqual(xyz.dtype, np.float64)
        self.assertEqual(features.shape, (200, 3))

    def test_read_in_channels_3_returns_no_extra_features(self):
        from geoai.pointcloud import _read_point_cloud

        las = self._make_test_las(50)
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            xyz, features, _ = _read_point_cloud(f.name, in_channels=3)
        os.unlink(f.name)

        self.assertEqual(xyz.shape, (50, 3))
        self.assertEqual(features.shape[1], 0)

    def test_read_preserves_coordinates(self):
        from geoai.pointcloud import _read_point_cloud

        las = self._make_test_las(50)
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            xyz, _, _ = _read_point_cloud(f.name)
        os.unlink(f.name)

        np.testing.assert_allclose(xyz[:, 0], las.x, atol=0.01)
        np.testing.assert_allclose(xyz[:, 1], las.y, atol=0.01)
        np.testing.assert_allclose(xyz[:, 2], las.z, atol=0.01)


class TestWritePointCloud(unittest.TestCase):
    """Tests for _write_point_cloud helper."""

    def test_write_updates_classification(self):
        from geoai.pointcloud import _write_point_cloud

        n = 100
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n)
        las.y = np.random.uniform(0, 100, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.zeros(n, dtype=np.uint16)
        las.classification = np.zeros(n, dtype=np.uint8)

        new_labels = np.random.randint(0, 6, n).astype(np.int32)

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            _write_point_cloud(las, new_labels, f.name)
            result = laspy.read(f.name)
        os.unlink(f.name)

        np.testing.assert_array_equal(
            np.asarray(result.classification), new_labels.astype(np.uint8)
        )

    def test_write_preserves_coordinates(self):
        from geoai.pointcloud import _write_point_cloud

        n = 50
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n)
        las.y = np.random.uniform(0, 100, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.zeros(n, dtype=np.uint16)
        las.classification = np.zeros(n, dtype=np.uint8)

        labels = np.ones(n, dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            _write_point_cloud(las, labels, f.name)
            result = laspy.read(f.name)
        os.unlink(f.name)

        np.testing.assert_allclose(result.x, las.x, atol=0.01)
        np.testing.assert_allclose(result.y, las.y, atol=0.01)
        np.testing.assert_allclose(result.z, las.z, atol=0.01)


# ---------------------------------------------------------------------------
# Summary tests (no model required)
# ---------------------------------------------------------------------------


class TestSummary(unittest.TestCase):
    """Tests for PointCloudClassifier.summary (no model needed)."""

    def _make_classified_las(self, n=500):
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(100, 200, n)
        las.y = np.random.uniform(300, 400, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.zeros(n, dtype=np.uint16)
        # Mix of classes: 50% ground (2), 30% building (6), 20% vegetation (5)
        classes = np.zeros(n, dtype=np.uint8)
        classes[: n // 2] = 2
        classes[n // 2 : n // 2 + 3 * n // 10] = 6
        classes[n // 2 + 3 * n // 10 :] = 5
        las.classification = classes
        return las

    @patch("geoai.pointcloud.ml3d")
    @patch("geoai.pointcloud._download_checkpoint")
    def test_summary_counts(self, mock_download, mock_ml3d):
        from geoai.pointcloud import PointCloudClassifier, SUPPORTED_MODELS

        mock_download.return_value = "/fake/ckpt.pth"
        mock_model = MagicMock()
        mock_ml3d.models.RandLANet.return_value = mock_model
        mock_pipeline = MagicMock()
        mock_ml3d.pipelines.SemanticSegmentation.return_value = mock_pipeline

        clf = PointCloudClassifier.__new__(PointCloudClassifier)
        clf.model_name = "RandLANet_Toronto3D"
        clf.num_classes = 8
        clf.device = "cpu"
        clf.class_names = SUPPORTED_MODELS["RandLANet_Toronto3D"]["class_names"]

        las = self._make_classified_las(500)
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            stats = clf.summary(f.name)
        os.unlink(f.name)

        self.assertEqual(stats["total_points"], 500)
        self.assertIn("class_counts", stats)
        self.assertIn("class_percentages", stats)
        self.assertIn("bounds", stats)
        self.assertEqual(len(stats["bounds"]), 6)

    @patch("geoai.pointcloud.ml3d")
    @patch("geoai.pointcloud._download_checkpoint")
    def test_summary_file_not_found(self, mock_download, mock_ml3d):
        from geoai.pointcloud import PointCloudClassifier

        clf = PointCloudClassifier.__new__(PointCloudClassifier)
        with self.assertRaises(FileNotFoundError):
            clf.summary("/nonexistent/path.las")


# ---------------------------------------------------------------------------
# _remap_labels tests
# ---------------------------------------------------------------------------


class TestRemapLabels(unittest.TestCase):
    """Tests for the _remap_labels helper."""

    def test_basic_remapping(self):
        from geoai.pointcloud import _remap_labels

        labels = np.array([2, 6, 9, 3, 5, 1, 0], dtype=np.int32)
        asprs_to_model = {0: 0, 1: 0, 2: 1, 3: 2, 5: 2, 6: 3, 9: 4}
        result = _remap_labels(labels, asprs_to_model)
        expected = np.array([1, 3, 4, 2, 2, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_unmapped_codes_get_default(self):
        from geoai.pointcloud import _remap_labels

        labels = np.array([2, 99, 255], dtype=np.int32)
        asprs_to_model = {2: 1}
        result = _remap_labels(labels, asprs_to_model, default=0)
        expected = np.array([1, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_empty_labels(self):
        from geoai.pointcloud import _remap_labels

        labels = np.array([], dtype=np.int32)
        result = _remap_labels(labels, {2: 1})
        self.assertEqual(len(result), 0)

    def test_3dep_full_mapping(self):
        from geoai.pointcloud import SUPPORTED_MODELS, _remap_labels

        asprs_to_model = SUPPORTED_MODELS["RandLANet_3DEP"]["asprs_to_model"]
        # All ASPRS codes that 3DEP maps
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18], dtype=np.int32)
        result = _remap_labels(labels, asprs_to_model)
        expected = np.array([0, 0, 1, 2, 2, 2, 3, 6, 4, 5, 6], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 3DEP model config tests
# ---------------------------------------------------------------------------


class TestRandLANet3DEPConfig(unittest.TestCase):
    """Tests specific to the RandLANet_3DEP model config."""

    def test_asprs_to_model_mapping_exists(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        info = SUPPORTED_MODELS["RandLANet_3DEP"]
        self.assertIn("asprs_to_model", info)
        self.assertIsInstance(info["asprs_to_model"], dict)

    def test_model_to_asprs_mapping_exists(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        info = SUPPORTED_MODELS["RandLANet_3DEP"]
        self.assertIn("model_to_asprs", info)
        self.assertIsInstance(info["model_to_asprs"], dict)

    def test_all_model_indices_covered_in_reverse_map(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        info = SUPPORTED_MODELS["RandLANet_3DEP"]
        num_classes = info["num_classes"]
        model_to_asprs = info["model_to_asprs"]
        for i in range(num_classes):
            self.assertIn(i, model_to_asprs, f"Model index {i} not in model_to_asprs")

    def test_class_names_match_num_classes(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        info = SUPPORTED_MODELS["RandLANet_3DEP"]
        self.assertEqual(len(info["class_names"]), info["num_classes"])

    def test_asprs_to_model_values_in_range(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        info = SUPPORTED_MODELS["RandLANet_3DEP"]
        num_classes = info["num_classes"]
        for asprs_code, model_idx in info["asprs_to_model"].items():
            self.assertGreaterEqual(model_idx, 0)
            self.assertLess(model_idx, num_classes)


# ---------------------------------------------------------------------------
# _LASDatasetSplit tests
# ---------------------------------------------------------------------------


class TestLASDatasetSplit(unittest.TestCase):
    def test_len(self):
        from geoai.pointcloud import _LASDatasetSplit

        split = _LASDatasetSplit(["a.las", "b.las", "c.las"], num_classes=8)
        self.assertEqual(len(split), 3)

    def test_get_attr(self):
        from geoai.pointcloud import _LASDatasetSplit

        split = _LASDatasetSplit(["/data/tile_001.las"], num_classes=8)
        attr = split.get_attr(0)
        self.assertEqual(attr["name"], "tile_001")
        self.assertEqual(attr["path"], "/data/tile_001.las")

    def test_get_data_reads_las(self):
        from geoai.pointcloud import _LASDatasetSplit

        n = 50
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n)
        las.y = np.random.uniform(0, 100, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.random.randint(0, 65535, n).astype(np.uint16)
        las.classification = np.ones(n, dtype=np.uint8) * 2

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            split = _LASDatasetSplit([f.name], num_classes=8)
            data = split.get_data(0)
        os.unlink(f.name)

        self.assertIn("point", data)
        self.assertIn("feat", data)
        self.assertIn("label", data)
        self.assertEqual(data["point"].shape, (n, 3))
        self.assertTrue(np.all(data["label"] == 2))

    def test_get_data_with_asprs_remap(self):
        from geoai.pointcloud import _LASDatasetSplit

        n = 60
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n)
        las.y = np.random.uniform(0, 100, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.zeros(n, dtype=np.uint16)
        # 30 ground (ASPRS 2), 30 building (ASPRS 6)
        classes = np.zeros(n, dtype=np.uint8)
        classes[:30] = 2
        classes[30:] = 6
        las.classification = classes

        asprs_map = {2: 1, 6: 3}

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            split = _LASDatasetSplit([f.name], num_classes=7, asprs_to_model=asprs_map)
            data = split.get_data(0)
        os.unlink(f.name)

        # Ground (ASPRS 2) -> model index 1
        self.assertTrue(np.all(data["label"][:30] == 1))
        # Building (ASPRS 6) -> model index 3
        self.assertTrue(np.all(data["label"][30:] == 3))


# ---------------------------------------------------------------------------
# _LASDataset tests
# ---------------------------------------------------------------------------


class TestLASDataset(unittest.TestCase):
    """Tests for the _LASDataset Open3D-ML compatible wrapper."""

    def test_get_split_training(self):
        from geoai.pointcloud import _LASDataset

        ds = _LASDataset(
            train_files=["a.las", "b.las"],
            val_files=["c.las"],
            num_classes=9,
        )
        split = ds.get_split("training")
        self.assertEqual(len(split), 2)

    def test_get_split_validation(self):
        from geoai.pointcloud import _LASDataset

        ds = _LASDataset(
            train_files=["a.las"],
            val_files=["b.las", "c.las"],
            num_classes=9,
        )
        split = ds.get_split("validation")
        self.assertEqual(len(split), 2)

    def test_get_split_unknown_raises(self):
        from geoai.pointcloud import _LASDataset

        ds = _LASDataset(train_files=[], val_files=[], num_classes=9)
        with self.assertRaises(ValueError):
            ds.get_split("unknown")

    def test_get_split_test_raises(self):
        from geoai.pointcloud import _LASDataset

        ds = _LASDataset(train_files=[], val_files=[], num_classes=9)
        with self.assertRaises(NotImplementedError):
            ds.get_split("test")

    def test_get_label_to_names_roundtrip(self):
        from geoai.pointcloud import _LASDataset

        labels = {0: "ground", 1: "vegetation", 2: "building"}
        ds = _LASDataset(
            train_files=[],
            val_files=[],
            num_classes=3,
            label_to_names=labels,
        )
        self.assertEqual(ds.get_label_to_names(), labels)

    def test_cfg_num_points(self):
        from geoai.pointcloud import _LASDataset

        ds = _LASDataset(train_files=[], val_files=[], num_classes=9, num_points=45056)
        self.assertEqual(ds.cfg.num_points, 45056)


# ---------------------------------------------------------------------------
# Classify method tests (mocked pipeline)
# ---------------------------------------------------------------------------


class TestClassifyMocked(unittest.TestCase):
    """Test classify() with a mocked Open3D-ML pipeline."""

    def _make_mock_classifier(self, num_classes=8):
        from geoai.pointcloud import PointCloudClassifier, SUPPORTED_MODELS

        clf = PointCloudClassifier.__new__(PointCloudClassifier)
        clf.model_name = "RandLANet_Toronto3D"
        clf.num_classes = num_classes
        clf.in_channels = SUPPORTED_MODELS["RandLANet_Toronto3D"].get("in_channels", 3)
        clf.class_names = SUPPORTED_MODELS["RandLANet_Toronto3D"]["class_names"]
        clf.device = "cpu"
        clf.checkpoint_path = "/fake/ckpt.pth"
        clf._config = SUPPORTED_MODELS["RandLANet_Toronto3D"]["config"].copy()
        clf._model = MagicMock()
        clf._pipeline = MagicMock()
        return clf

    def test_classify_returns_predictions(self):
        clf = self._make_mock_classifier()
        n = 100

        # Mock pipeline.run_inference
        clf._pipeline.run_inference.return_value = {
            "predict_labels": np.random.randint(0, 8, n),
            "predict_scores": np.random.rand(n, 8).astype(np.float32),
        }

        # Create test LAS file
        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n)
        las.y = np.random.uniform(0, 100, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.zeros(n, dtype=np.uint16)
        las.classification = np.zeros(n, dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            labels, probs = clf.classify(f.name)
        os.unlink(f.name)

        self.assertEqual(labels.shape, (n,))
        self.assertEqual(probs.shape, (n, 8))
        self.assertEqual(labels.dtype, np.int32)

    def test_classify_writes_output(self):
        clf = self._make_mock_classifier()
        n = 50

        clf._pipeline.run_inference.return_value = {
            "predict_labels": np.ones(n, dtype=np.int32) * 3,
        }

        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.random.uniform(0, 100, n)
        las.y = np.random.uniform(0, 100, n)
        las.z = np.random.uniform(0, 50, n)
        las.intensity = np.zeros(n, dtype=np.uint16)
        las.classification = np.zeros(n, dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as fin:
            las.write(fin.name)
            with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as fout:
                clf.classify(fin.name, output_path=fout.name)
                result = laspy.read(fout.name)
            os.unlink(fout.name)
        os.unlink(fin.name)

        np.testing.assert_array_equal(
            np.asarray(result.classification),
            np.ones(n, dtype=np.uint8) * 3,
        )

    def test_classify_file_not_found(self):
        clf = self._make_mock_classifier()
        with self.assertRaises(FileNotFoundError):
            clf.classify("/nonexistent/input.las")


# ---------------------------------------------------------------------------
# Train method tests (mocked)
# ---------------------------------------------------------------------------


class TestTrainMocked(unittest.TestCase):
    """Test train() input validation."""

    def _make_mock_classifier(self):
        from geoai.pointcloud import PointCloudClassifier, SUPPORTED_MODELS

        clf = PointCloudClassifier.__new__(PointCloudClassifier)
        clf.model_name = "RandLANet_Toronto3D"
        clf.num_classes = 8
        clf.device = "cpu"
        clf._model = MagicMock()
        clf._pipeline = MagicMock()
        return clf

    def test_train_dir_not_found(self):
        clf = self._make_mock_classifier()
        with self.assertRaises(FileNotFoundError):
            clf.train("/nonexistent/train_dir")

    def test_train_no_las_files(self):
        clf = self._make_mock_classifier()
        with tempfile.TemporaryDirectory() as td:
            # Empty directory — no LAS files
            with self.assertRaises(ValueError):
                clf.train(td)

    def test_train_val_dir_not_found(self):
        clf = self._make_mock_classifier()
        with tempfile.TemporaryDirectory() as td:
            # Create a dummy LAS file so train_dir validation passes
            header = laspy.LasHeader(point_format=0, version="1.2")
            las = laspy.LasData(header)
            las.x = np.array([0.0])
            las.y = np.array([0.0])
            las.z = np.array([0.0])
            las.intensity = np.array([0], dtype=np.uint16)
            las.classification = np.array([0], dtype=np.uint8)
            las.write(os.path.join(td, "train.las"))

            with self.assertRaises(FileNotFoundError):
                clf.train(td, val_dir="/nonexistent/val_dir")


# ---------------------------------------------------------------------------
# Visualize tests (mocked leafmap)
# ---------------------------------------------------------------------------


class TestVisualizeMocked(unittest.TestCase):
    def _make_mock_classifier(self):
        from geoai.pointcloud import PointCloudClassifier

        clf = PointCloudClassifier.__new__(PointCloudClassifier)
        clf.device = "cpu"
        return clf

    def test_visualize_calls_leafmap(self):
        clf = self._make_mock_classifier()

        header = laspy.LasHeader(point_format=0, version="1.2")
        las = laspy.LasData(header)
        las.x = np.array([0.0, 1.0])
        las.y = np.array([0.0, 1.0])
        las.z = np.array([0.0, 1.0])
        las.intensity = np.array([0, 0], dtype=np.uint16)
        las.classification = np.array([0, 0], dtype=np.uint8)

        mock_leafmap = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            las.write(f.name)
            with patch.dict("sys.modules", {"leafmap": mock_leafmap}):
                clf.visualize(f.name, backend="pyvista")
        os.unlink(f.name)

        mock_leafmap.view_lidar.assert_called_once()

    def test_visualize_file_not_found(self):
        clf = self._make_mock_classifier()
        with self.assertRaises(FileNotFoundError):
            clf.visualize("/nonexistent/file.las")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    def test_asprs_classes_all_integers(self):
        from geoai.pointcloud import ASPRS_CLASSES

        for key in ASPRS_CLASSES:
            self.assertIsInstance(key, int)
            self.assertGreaterEqual(key, 0)
            self.assertLessEqual(key, 255)

    def test_default_cache_dir_is_string(self):
        from geoai.pointcloud import DEFAULT_CACHE_DIR

        self.assertIsInstance(DEFAULT_CACHE_DIR, str)
        self.assertIn("pointcloud", DEFAULT_CACHE_DIR)

    def test_supported_models_urls_are_https(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        for name, info in SUPPORTED_MODELS.items():
            self.assertTrue(
                info["url"].startswith("https://"),
                f"URL for {name} is not HTTPS",
            )

    def test_supported_models_configs_have_required_keys(self):
        from geoai.pointcloud import SUPPORTED_MODELS

        required = {"name", "num_neighbors", "num_layers", "num_classes"}
        for name, info in SUPPORTED_MODELS.items():
            for key in required:
                self.assertIn(
                    key,
                    info["config"],
                    f"Missing config key '{key}' in {name}",
                )


if __name__ == "__main__":
    unittest.main()
