#!/usr/bin/env python

"""Tests for `geoai.esrgan` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestEsrganImport(unittest.TestCase):
    """Tests for esrgan module import behavior."""

    def test_module_imports(self):
        """Test that the esrgan module can be imported."""
        import geoai.esrgan

        self.assertTrue(hasattr(geoai.esrgan, "ESRGAN"))
        self.assertTrue(hasattr(geoai.esrgan, "ESRGANGenerator"))
        self.assertTrue(hasattr(geoai.esrgan, "Discriminator"))
        self.assertTrue(hasattr(geoai.esrgan, "NormalizeToVGG"))


class TestEsrganSignatures(unittest.TestCase):
    """Signature-focused tests to keep things lightweight."""

    def test_esrgan_init_params(self):
        """Test ESRGAN.__init__ has expected parameters."""
        from geoai.esrgan import ESRGAN

        sig = inspect.signature(ESRGAN.__init__)
        self.assertIn("scale", sig.parameters)
        self.assertIn("band", sig.parameters)
        self.assertIn("manual_seed", sig.parameters)
        self.assertIn("data_path", sig.parameters)
        self.assertIn("model_path", sig.parameters)

    def test_train_esrgan_params(self):
        """Test train_esrgan has expected parameters."""
        from geoai.esrgan import ESRGAN

        sig = inspect.signature(ESRGAN.train_esrgan)
        self.assertIn("low_res", sig.parameters)
        self.assertIn("high_res", sig.parameters)
        self.assertIn("load_tensors", sig.parameters)
        self.assertIn("lr_generator", sig.parameters)
        self.assertIn("lr_discriminator", sig.parameters)
        self.assertIn("lambda_pixel", sig.parameters)
        self.assertIn("lambda_vgg", sig.parameters)
        self.assertIn("lambda_discriminator", sig.parameters)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("num_epochs", sig.parameters)
        self.assertIn("warmup_epochs", sig.parameters)


class TestEsrganForwardPasses(unittest.TestCase):
    """Forward-pass smoke tests for core torch modules."""

    def test_generator_forward_shape(self):
        """Generator should return a tensor with scaled spatial dimensions."""
        import torch
        from geoai.esrgan import ESRGANGenerator

        model = ESRGANGenerator(in_nc=1, out_nc=1, nf=16, nb=2, gc=8, scale=4)
        x = torch.randn(2, 1, 16, 16)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))

    def test_discriminator_forward_shape(self):
        """Discriminator should return a (N, 1) score tensor."""
        import torch
        from geoai.esrgan import Discriminator

        model = Discriminator()
        x = torch.randn(4, 1, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4, 1))


class TestNormalizeToVGG(unittest.TestCase):
    """Tests for NormalizeToVGG."""

    def test_normalize_buffers_exist(self):
        """NormalizeToVGG should register mean/std buffers."""
        from geoai.esrgan import NormalizeToVGG

        norm = NormalizeToVGG()
        self.assertTrue(hasattr(norm, "mean"))
        self.assertTrue(hasattr(norm, "std"))

    def test_normalize_forward_broadcast(self):
        """Normalize should broadcast (1,3,1,1) stats across a batch."""
        import torch
        from geoai.esrgan import NormalizeToVGG

        norm = NormalizeToVGG()
        x = torch.zeros(2, 3, 8, 8)
        y = norm(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8, 8))


class TestVGGPerceptualLoss(unittest.TestCase):
    """Tests for VGGPerceptualLoss with mocked VGG to avoid downloads."""

    @patch("geoai.esrgan.models.vgg19")
    def test_init_uses_vgg_features(self, mock_vgg19):
        """VGGPerceptualLoss should slice vgg19().features without requiring weights."""
        import torch
        import torch.nn as nn
        from geoai.esrgan import VGGPerceptualLoss

        mock_model = MagicMock()
        mock_model.features = nn.Sequential(*[nn.Identity() for _ in range(20)])
        mock_vgg19.return_value = mock_model

        loss_fn = VGGPerceptualLoss(layer_index=9)
        self.assertTrue(hasattr(loss_fn, "slice"))

        x = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        out = loss_fn(x, y)
        self.assertTrue(torch.is_tensor(out))


class TestEsrganDataPreprocessHelpers(unittest.TestCase):
    """Tests for preprocessing helper methods."""

    def test_match_nulls_same_shape(self):
        """_match_nulls should zero-out arr1 where arr2 is <=0 or nan when shapes match."""
        from geoai.esrgan import ESRGANDataPreprocess

        arr1 = np.ones((4, 4), dtype=np.float32)
        arr2 = np.ones((4, 4), dtype=np.float32)
        arr2[0, 0] = 0
        arr2[1, 1] = np.nan

        out1, out2 = ESRGANDataPreprocess._match_nulls(arr1.copy(), arr2.copy())
        self.assertEqual(out1[0, 0], 0)
        self.assertEqual(out1[1, 1], 0)
        self.assertEqual(out2[0, 0], 0)
        self.assertEqual(out2[1, 1], 0)

    def test_match_nulls_resample_shape(self):
        """_match_nulls should handle shape mismatch by resampling arr2 to arr1's shape."""
        from geoai.esrgan import ESRGANDataPreprocess

        arr1 = np.ones((8, 8), dtype=np.float32)
        arr2 = np.ones((16, 16), dtype=np.float32)
        arr2[0:4, 0:4] = 0

        out1, out2 = ESRGANDataPreprocess._match_nulls(arr1.copy(), arr2.copy())
        self.assertEqual(out1.shape, (8, 8))
        self.assertEqual(out2.shape, (16, 16))
        self.assertTrue(np.any(out1 == 0))


class TestPredictGeotiffSuperres(unittest.TestCase):
    def test_predict_geotiff_superres_writes_scaled_raster(self):
        try:
            import tempfile

            import rasterio
            from rasterio.transform import Affine
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            from geoai.inference import predict_geotiff_superres
        except ImportError as e:
            self.skipTest(f"Missing dependency for superres inference test: {e}")

        class DummyUpsample(nn.Module):
            def __init__(self, scale: int):
                super().__init__()
                self.scale = scale

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.interpolate(x, scale_factor=self.scale, mode="nearest")

        scale = 4
        h, w = 10, 12
        transform = Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
        crs = "EPSG:4326"

        data = np.arange(h * w, dtype=np.float32).reshape(h, w)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = f"{tmpdir}/in.tif"
            out_path = f"{tmpdir}/out.tif"

            profile = {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": 1,
                "dtype": "float32",
                "crs": crs,
                "transform": transform,
                "nodata": None,
            }

            with rasterio.open(in_path, "w", **profile) as dst:
                dst.write(data, 1)

            model = DummyUpsample(scale=scale)

            predict_geotiff_superres(
                model=model,
                input_raster=in_path,
                output_raster=out_path,
                scale=scale,
                tile_size=6,
                overlap=2,
                batch_size=2,
                input_bands=[1],
                num_classes=1,
                output_dtype="float32",
                output_nodata=-9999.0,
                blend_mode="spline",
                tta=False,
                preprocess_fn=lambda x: x.astype(np.float32),
                device="cpu",
                verbose=False,
            )

            with rasterio.open(out_path) as src:
                self.assertEqual(src.width, w * scale)
                self.assertEqual(src.height, h * scale)
                self.assertEqual(src.count, 1)
                self.assertEqual(src.crs.to_string(), crs)

                # Pixel sizes should be divided by scale
                self.assertAlmostEqual(src.transform.a, transform.a / scale)
                self.assertAlmostEqual(src.transform.e, transform.e / scale)
                self.assertAlmostEqual(src.transform.c, transform.c)
                self.assertAlmostEqual(src.transform.f, transform.f)

                out = src.read(1)
                self.assertEqual(out.shape, (h * scale, w * scale))
                self.assertNotEqual(float(out[0, 0]), -9999.0)

                # With nearest-neighbor upsampling, the top-left scale x scale block matches input[0,0]
                self.assertAlmostEqual(float(out[0, 0]), float(data[0, 0]))
                self.assertAlmostEqual(float(out[scale - 1, scale - 1]), float(data[0, 0]))


if __name__ == "__main__":
    unittest.main()
