#!/usr/bin/env python

"""Tests for `geoai.recognize` module."""

import inspect
import os
import tempfile
import unittest

import numpy as np
import torch

from geoai import recognize


class TestRecognizeModule(unittest.TestCase):
    """Tests for recognize module."""

    def test_module_imports(self):
        """Test that recognize module imports correctly."""
        self.assertTrue(hasattr(recognize, "__name__"))

    def test_functions_exist(self):
        """Test that key recognize functions exist and are callable."""
        expected_functions = [
            "load_image_dataset",
            "train_image_classifier",
            "predict_images",
            "evaluate_classifier",
            "plot_training_history",
            "plot_confusion_matrix",
            "plot_predictions",
        ]
        for func_name in expected_functions:
            self.assertTrue(
                hasattr(recognize, func_name),
                f"{func_name} not found in recognize module",
            )
            func = getattr(recognize, func_name)
            self.assertTrue(callable(func), f"{func_name} is not callable")

    def test_image_dataset_class_exists(self):
        """Test that ImageDataset class exists."""
        self.assertTrue(hasattr(recognize, "ImageDataset"))
        self.assertTrue(issubclass(recognize.ImageDataset, torch.utils.data.Dataset))

    def test_train_image_classifier_signature(self):
        """Test that train_image_classifier has expected parameters."""
        sig = inspect.signature(recognize.train_image_classifier)
        param_names = list(sig.parameters.keys())
        expected_params = [
            "data_dir",
            "model_name",
            "num_epochs",
            "batch_size",
            "learning_rate",
            "image_size",
            "in_channels",
            "pretrained",
            "output_dir",
        ]
        for param in expected_params:
            self.assertIn(param, param_names, f"Missing parameter: {param}")

    def test_predict_images_signature(self):
        """Test that predict_images has expected parameters."""
        sig = inspect.signature(recognize.predict_images)
        param_names = list(sig.parameters.keys())
        expected_params = [
            "model",
            "image_paths",
            "class_names",
            "image_size",
            "in_channels",
        ]
        for param in expected_params:
            self.assertIn(param, param_names, f"Missing parameter: {param}")

    def test_evaluate_classifier_signature(self):
        """Test that evaluate_classifier has expected parameters."""
        sig = inspect.signature(recognize.evaluate_classifier)
        param_names = list(sig.parameters.keys())
        expected_params = ["model", "dataset", "class_names"]
        for param in expected_params:
            self.assertIn(param, param_names, f"Missing parameter: {param}")

    def test_load_image_dataset_signature(self):
        """Test that load_image_dataset has expected parameters."""
        sig = inspect.signature(recognize.load_image_dataset)
        param_names = list(sig.parameters.keys())
        self.assertIn("data_dir", param_names)
        self.assertIn("extensions", param_names)

    def test_load_image_dataset_with_temp_dir(self):
        """Test load_image_dataset with a temporary ImageFolder structure."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create class directories with dummy images
            for class_name in ["cats", "dogs"]:
                class_dir = os.path.join(tmp_dir, class_name)
                os.makedirs(class_dir)
                for i in range(3):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                    )
                    img.save(os.path.join(class_dir, f"img_{i}.jpg"))

            result = recognize.load_image_dataset(tmp_dir)

            self.assertEqual(len(result["image_paths"]), 6)
            self.assertEqual(len(result["labels"]), 6)
            self.assertEqual(result["class_names"], ["cats", "dogs"])
            self.assertEqual(result["class_to_idx"], {"cats": 0, "dogs": 1})

    def test_load_image_dataset_nonexistent_dir(self):
        """Test load_image_dataset raises error for nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            recognize.load_image_dataset("/nonexistent/path/to/dataset")

    def test_load_image_dataset_empty_dir(self):
        """Test load_image_dataset raises error for directory with no classes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                recognize.load_image_dataset(tmp_dir)

    def test_image_dataset_creation(self):
        """Test ImageDataset creation and basic operations."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy images
            paths = []
            for i in range(5):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                path = os.path.join(tmp_dir, f"img_{i}.jpg")
                img.save(path)
                paths.append(path)

            labels = [0, 1, 0, 1, 0]
            dataset = recognize.ImageDataset(
                paths, labels, image_size=32, in_channels=3
            )

            self.assertEqual(len(dataset), 5)

            # Check __getitem__
            image, label = dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(label, torch.Tensor)
            self.assertEqual(image.shape, (3, 32, 32))
            self.assertEqual(label.item(), 0)

    def test_image_dataset_mismatched_lengths(self):
        """Test ImageDataset raises error when paths and labels have different lengths."""
        with self.assertRaises(ValueError):
            recognize.ImageDataset(
                image_paths=["a.jpg", "b.jpg"],
                labels=[0],
            )

    def test_image_dataset_channel_adjustment(self):
        """Test ImageDataset pads or truncates channels correctly."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            path = os.path.join(tmp_dir, "img.jpg")
            img.save(path)

            # Request 5 channels from a 3-channel image (should pad)
            dataset = recognize.ImageDataset([path], [0], image_size=32, in_channels=5)
            image, _ = dataset[0]
            self.assertEqual(image.shape[0], 5)

            # Request 1 channel from a 3-channel image (should truncate)
            dataset = recognize.ImageDataset([path], [0], image_size=32, in_channels=1)
            image, _ = dataset[0]
            self.assertEqual(image.shape[0], 1)

    def test_train_image_classifier_invalid_dir(self):
        """Test train_image_classifier raises error for nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            recognize.train_image_classifier(
                data_dir="/nonexistent/path",
                num_epochs=1,
            )

    def test_plot_confusion_matrix(self):
        """Test plot_confusion_matrix produces a figure."""
        import matplotlib

        matplotlib.use("Agg")

        cm = np.array([[10, 2], [3, 15]])
        class_names = ["ClassA", "ClassB"]
        fig = recognize.plot_confusion_matrix(cm, class_names)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_confusion_matrix_normalized(self):
        """Test plot_confusion_matrix with normalization."""
        import matplotlib

        matplotlib.use("Agg")

        cm = np.array([[10, 2], [3, 15]])
        class_names = ["ClassA", "ClassB"]
        fig = recognize.plot_confusion_matrix(cm, class_names, normalize=True)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_predictions(self):
        """Test plot_predictions produces a figure."""
        import matplotlib

        matplotlib.use("Agg")

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                path = os.path.join(tmp_dir, f"img_{i}.jpg")
                img.save(path)
                paths.append(path)

            predictions = np.array([0, 1, 0, 1])
            true_labels = [0, 0, 1, 1]
            class_names = ["ClassA", "ClassB"]

            fig = recognize.plot_predictions(
                paths, predictions, true_labels, class_names, num_images=4
            )
            self.assertIsNotNone(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_training_history_missing_dir(self):
        """Test plot_training_history raises error for missing metrics file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                recognize.plot_training_history(tmp_dir)


if __name__ == "__main__":
    unittest.main()
