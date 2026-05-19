#!/usr/bin/env python

"""Tests for ``geoai.utils.sampling``."""

import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset


class TinyDataset(Dataset):
    """Small dataset used by dataloader tests."""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"image": torch.tensor([idx]), "mask": torch.tensor([idx])}


class FakeGeoDataset(Dataset):
    """Minimal TorchGeo-like GeoDataset."""

    def __and__(self, other):
        return ("intersection", self, other)


class FakeRasterDataset(FakeGeoDataset):
    """Minimal TorchGeo-like RasterDataset."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRandomGeoSampler:
    """Minimal random sampler."""

    def __init__(self, dataset, size, length=None, roi=None, units=None):
        self.kwargs = {
            "dataset": dataset,
            "size": size,
            "length": length,
            "roi": roi,
            "units": units,
        }

    def __iter__(self):
        return iter([0, 1])

    def __len__(self):
        return 2


class FakeGridGeoSampler:
    """Minimal grid sampler."""

    def __init__(self, dataset, size, stride=None, roi=None, units=None):
        self.kwargs = {
            "dataset": dataset,
            "size": size,
            "stride": stride,
            "roi": roi,
            "units": units,
        }

    def __iter__(self):
        return iter([0, 1])

    def __len__(self):
        return 2


def fake_stack_samples(samples):
    """Simple collate function for fake TorchGeo batches."""
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
    }


def fake_torchgeo_modules():
    """Return fake TorchGeo modules for lazy import tests."""
    torchgeo = types.ModuleType("torchgeo")
    datasets = types.ModuleType("torchgeo.datasets")
    samplers = types.ModuleType("torchgeo.samplers")

    datasets.GeoDataset = FakeGeoDataset
    datasets.RasterDataset = FakeRasterDataset
    datasets.stack_samples = fake_stack_samples
    samplers.RandomGeoSampler = FakeRandomGeoSampler
    samplers.GridGeoSampler = FakeGridGeoSampler

    return {
        "torchgeo": torchgeo,
        "torchgeo.datasets": datasets,
        "torchgeo.samplers": samplers,
    }


class TestSamplingExports(unittest.TestCase):
    """Export tests."""

    def test_utils_exports_sampling_helpers(self):
        import geoai.utils as utils

        expected = [
            "create_raster_dataset",
            "create_segmentation_dataset",
            "create_geo_sampler",
            "create_geo_dataloader",
            "create_geo_dataloaders",
            "create_torchgeo_segmentation_dataloaders",
            "geo_sample_to_tuple",
            "predict_torchgeo_segmentation_batch",
            "plot_torchgeo_segmentation_predictions",
            "train_torchgeo_segmentation_model",
        ]
        for name in expected:
            self.assertIn(name, utils.__all__)
            self.assertTrue(hasattr(utils, name))

    def test_geoai_top_level_exports_sampling_helpers(self):
        import geoai

        self.assertIn("create_geo_sampler", dir(geoai))
        self.assertIn("geo_sample_to_tuple", dir(geoai))
        self.assertIn("train_torchgeo_segmentation_model", dir(geoai))


class TestSamplingTorchGeoImports(unittest.TestCase):
    """TorchGeo import behavior tests."""

    def test_missing_torchgeo_raises_clear_error_when_called(self):
        from geoai.utils import sampling

        missing = {
            "torchgeo": None,
            "torchgeo.datasets": None,
            "torchgeo.samplers": None,
        }
        with patch.dict(sys.modules, missing):
            with self.assertRaisesRegex(ImportError, "TorchGeo is required"):
                sampling.create_geo_sampler(TinyDataset(), size=16)

    def test_create_raster_dataset_uses_lazy_torchgeo_classes(self):
        from geoai.utils import sampling

        with patch.dict(sys.modules, fake_torchgeo_modules()):
            ds = sampling.create_raster_dataset(
                "images", is_image=False, filename_glob="*.tif", cache=False
            )

        self.assertIsInstance(ds, FakeRasterDataset)
        self.assertEqual(ds.kwargs["paths"], "images")
        self.assertFalse(ds.kwargs["cache"])
        self.assertFalse(ds.is_image)
        self.assertEqual(ds.filename_glob, "*.tif")

    def test_create_segmentation_dataset_intersects_inputs(self):
        from geoai.utils import sampling

        with patch.dict(sys.modules, fake_torchgeo_modules()):
            result = sampling.create_segmentation_dataset("images", "masks")

        self.assertEqual(result[0], "intersection")
        self.assertIsInstance(result[1], FakeRasterDataset)
        self.assertIsInstance(result[2], FakeRasterDataset)
        self.assertTrue(result[1].is_image)
        self.assertFalse(result[2].is_image)

    def test_create_geo_sampler_random_and_grid(self):
        from geoai.utils import sampling

        dataset = TinyDataset()
        with patch.dict(sys.modules, fake_torchgeo_modules()):
            random_sampler = sampling.create_geo_sampler(
                dataset, sampler="random", size=32, length=5, toi="ignored"
            )
            grid_sampler = sampling.create_geo_sampler(
                dataset, sampler="grid", size=32, generator="ignored"
            )

        self.assertIsInstance(random_sampler, FakeRandomGeoSampler)
        self.assertEqual(random_sampler.kwargs["length"], 5)
        self.assertIsInstance(grid_sampler, FakeGridGeoSampler)
        self.assertEqual(grid_sampler.kwargs["stride"], 32)

    def test_create_geo_dataloader_uses_stack_samples_by_default(self):
        from geoai.utils import sampling

        with patch.dict(sys.modules, fake_torchgeo_modules()):
            loader = sampling.create_geo_dataloader(
                TinyDataset(), sampler_type="grid", size=16, batch_size=2
            )
            batch = next(iter(loader))

        self.assertEqual(batch["image"].shape, (2, 1))
        self.assertEqual(batch["mask"].shape, (2, 1))

    def test_create_geo_dataloaders_returns_expected_keys(self):
        from geoai.utils import sampling

        with patch.dict(sys.modules, fake_torchgeo_modules()):
            loaders = sampling.create_geo_dataloaders(
                TinyDataset(), val_dataset=TinyDataset(), size=16
            )

        self.assertEqual(set(loaders), {"train", "val", "test"})
        self.assertIsNotNone(loaders["train"])
        self.assertIsNotNone(loaders["val"])
        self.assertIsNone(loaders["test"])

    def test_create_torchgeo_segmentation_dataloaders_returns_dataset_and_loaders(self):
        from geoai.utils import sampling

        with patch.dict(sys.modules, fake_torchgeo_modules()):
            loaders = sampling.create_torchgeo_segmentation_dataloaders(
                "images", "masks", chip_size=16, train_length=2, val_length=1
            )

        self.assertEqual(set(loaders), {"dataset", "train", "val", "grid"})
        self.assertEqual(loaders["dataset"][0], "intersection")
        self.assertIsNotNone(loaders["train"])
        self.assertIsNotNone(loaders["val"])
        self.assertIsNotNone(loaders["grid"])


class TestGeoSampleToTuple(unittest.TestCase):
    """Batch conversion tests."""

    def test_tuple_batch_is_supported(self):
        from geoai.utils.sampling import geo_sample_to_tuple

        image = torch.ones(2, 3, 4, 4)
        mask = torch.ones(2, 4, 4)
        x, y = geo_sample_to_tuple((image, mask), normalize=True)

        self.assertTrue(torch.equal(x, image.float()))
        self.assertEqual(y.dtype, torch.long)

    def test_dict_batch_squeezes_mask_and_normalizes_image(self):
        from geoai.utils.sampling import geo_sample_to_tuple

        batch = {
            "image": torch.full((2, 4, 3, 3), 255, dtype=torch.uint8),
            "mask": torch.ones(2, 1, 3, 3, dtype=torch.uint8),
        }
        x, y = geo_sample_to_tuple(batch, num_channels=3, normalize=True)

        self.assertEqual(x.shape, (2, 3, 3, 3))
        self.assertLessEqual(float(x.max()), 1.0)
        self.assertEqual(y.shape, (2, 3, 3))
        self.assertEqual(y.dtype, torch.long)

    def test_channel_padding(self):
        from geoai.utils.sampling import geo_sample_to_tuple

        batch = {
            "image": torch.ones(1, 1, 2, 2),
            "mask": torch.zeros(1, 2, 2),
        }
        x, _ = geo_sample_to_tuple(batch, num_channels=3)

        self.assertEqual(x.shape, (1, 3, 2, 2))
        self.assertTrue(torch.all(x[:, 1:] == 0))

    def test_batch_size_one_mask_is_not_squeezed(self):
        from geoai.utils.sampling import geo_sample_to_tuple

        batch = {
            "image": torch.ones(1, 3, 2, 2),
            "mask": torch.zeros(1, 2, 2),
        }
        _, y = geo_sample_to_tuple(batch)

        self.assertEqual(y.shape, (1, 2, 2))

    def test_missing_image_key_raises(self):
        from geoai.utils.sampling import geo_sample_to_tuple

        with self.assertRaises(KeyError):
            geo_sample_to_tuple({"mask": torch.zeros(1, 2, 2)})


class TestTorchGeoSegmentationTraining(unittest.TestCase):
    """End-to-end tests for high-level TorchGeo segmentation helpers."""

    def test_train_and_predict_with_local_rasters(self):
        from geoai.utils.sampling import (
            predict_torchgeo_segmentation_batch,
            train_torchgeo_segmentation_model,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_torchgeo_segmentation_model(
                image_path="tests/data/test_raster_rgb.tif",
                mask_path="tests/data/test_raster_single.tif",
                output_dir=tmpdir,
                chip_size=16,
                train_length=2,
                val_length=1,
                batch_size=1,
                num_epochs=1,
                device="cpu",
                verbose=False,
            )

            self.assertIn("model", result)
            self.assertEqual(len(result["history"]), 1)
            self.assertIsNotNone(result["best_model_path"])
            self.assertIsNotNone(result["history_path"])

            pred = predict_torchgeo_segmentation_batch(
                result["model"],
                result["val_loader"],
                device="cpu",
                num_channels=3,
            )
            self.assertEqual(pred["images"].shape[1:], (3, 16, 16))
            self.assertEqual(pred["masks"].shape[1:], (16, 16))
            self.assertEqual(pred["predictions"].shape[1:], (16, 16))


if __name__ == "__main__":
    unittest.main()
