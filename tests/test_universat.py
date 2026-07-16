import unittest, numpy as np, torch, torch.nn as nn
from unittest.mock import MagicMock, patch, ANY
import geoai
from geoai.foundation_models import FOUNDATION_MODELS
from geoai.universat import (
    UniverSatProcessor,
    load_universat_model,
    universat_inference,
    get_tile_embedding,
    get_pca_rgb,
)


class TestUniverSatBasics(unittest.TestCase):
    def test_exports(self):
        for sym in [
            "UniverSatProcessor",
            "load_universat_model",
            "universat_inference",
            "get_tile_embedding",
            "get_pca_rgb",
            "universat_train",
        ]:
            self.assertTrue(hasattr(geoai, sym))
        self.assertTrue(
            all(
                callable(x)
                for x in (
                    UniverSatProcessor,
                    load_universat_model,
                    universat_inference,
                    geoai.universat_train,
                )
            )
        )

    def test_registry(self):
        self.assertIn("universat", FOUNDATION_MODELS)
        e = FOUNDATION_MODELS["universat"]
        for k, v in {
            "name": "UniverSat",
            "abbreviation": "UniverSat",
            "category": "vision",
            "modality": "multimodal",
            "terratorch_supported": False,
            "huggingface_id": "g-astruc/UniverSat",
        }.items():
            self.assertEqual(e[k], v)


class TestUniverSatProcessor(unittest.TestCase):
    @patch("geoai.universat.load_universat_model")
    def setUp(self, mock_load_model):
        self.mock_model = MagicMock(spec=nn.Module, n_registers=4)
        mock_load_model.return_value = self.mock_model
        self.patcher = patch.dict(
            "sys.modules",
            {
                "modality_registry": MagicMock(
                    INPUT_RES={"s2": 10.0, "spot": 1.0},
                    SUBPATCHES={"s2": 1, "spot": 10},
                    WAVELENGTHS={
                        "s2": [0.49, 0.56, 0.665],
                        "spot": [0.665, 0.56, 0.49],
                    },
                )
            },
        )
        self.patcher.start()
        self.processor = UniverSatProcessor(device="cpu")

    def tearDown(self):
        self.patcher.stop()

    def test_preprocess_image(self):
        t1 = self.processor.preprocess_image(
            np.random.randint(0, 256, (3, 32, 32)).astype(np.uint8), "spot", scale=255.0
        )
        self.assertEqual((t1.ndim, t1.shape, t1.dtype), (3, (3, 32, 32), torch.float32))
        self.assertLessEqual(t1.max().item(), 1.0)
        t2 = self.processor.preprocess_image(
            np.random.randint(0, 5000, (3, 32, 32)).astype(np.uint16),
            "s2",
            scale=10000.0,
        )
        self.assertEqual(
            (t2.ndim, t2.shape, t2.dtype), (4, (1, 3, 32, 32), torch.float32)
        )
        self.assertLessEqual(t2.max().item(), 0.5)
        t3 = self.processor.preprocess_image(
            np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32),
            "spot",
            scale=10.0,
        )
        self.assertEqual((t3[0, 0, 0].item(), t3[0, 1, 1].item()), (1.0, 4.0))

    def test_format_batch(self):
        s1, s2 = {
            "spot": np.random.randint(0, 256, (3, 16, 16)).astype(np.uint8),
            "s2": np.random.randint(0, 5000, (3, 16, 16)).astype(np.uint16),
            "s2_dates": [10],
        }, {
            "spot": np.random.randint(0, 256, (3, 16, 16)).astype(np.uint8),
            "s2": np.random.randint(0, 5000, (3, 16, 16)).astype(np.uint16),
        }
        b = self.processor.format_batch([s1, s2])
        self.assertTrue(all(k in b for k in ("spot", "s2", "s2_dates")))
        self.assertEqual(
            (b["spot"].shape, b["s2"].shape, b["s2_dates"].shape),
            ((2, 3, 16, 16), (2, 1, 3, 16, 16), (2, 1)),
        )

    @patch("rasterio.open")
    def test_read_geotiff(self, mock_open):
        mock_open.return_value.__enter__.return_value = MagicMock(
            read=lambda: np.zeros((3, 10, 10)), meta={"driver": "GTiff", "width": 10}
        )
        img, meta = self.processor.read_geotiff("fake.tif")
        self.assertEqual((img.shape, meta["width"]), ((3, 10, 10), 10))

    def test_encode_raster(self):
        self.processor.model.encode = MagicMock(
            return_value=(torch.zeros(2, 9, 768), {})
        )
        t, _ = self.processor.encode_raster(
            {"spot": np.zeros((3, 16, 16), dtype=np.uint8)},
            patch_size=20.0,
            output_grid=3,
        )
        self.assertEqual(t.shape, (2, 9, 768))
        self.processor.model.encode.assert_called_once_with(
            ANY, patch_size=20.0, output_grid=3
        )

    def test_encode_multimodal(self):
        self.processor.model.encode = MagicMock(
            return_value=(torch.zeros(2, 9, 768), {})
        )
        self.processor.encode_raster(
            {
                "spot": np.zeros((3, 360, 360), dtype=np.uint8),
                "s2": np.zeros((3, 36, 36), dtype=np.uint16),
            },
            patch_size=40.0,
            output_grid=9,
        )
        self.assertTrue(
            "spot" in self.processor.model.encode.call_args[0][0]
            and "s2" in self.processor.model.encode.call_args[0][0]
        )

    def test_encode_unseen_sensor(self):
        self.processor.model.encode = MagicMock(
            return_value=(torch.zeros(2, 9, 768), {})
        )
        self.processor.encode_raster(
            {"mycam": np.zeros((4, 144, 144), dtype=np.uint8)},
            wavelengths={"mycam": [0.49, 0.56, 0.665, 0.842]},
            input_res={"mycam": 2.5},
            subpatches={"mycam": 1},
        )
        self.assertIn("mycam", self.processor.model.encode.call_args[0][0])
        self.assertEqual(
            {
                k: self.processor.model.encode.call_args[1].get(k)
                for k in ["wavelengths", "input_res", "subpatches"]
            },
            {
                "wavelengths": {"mycam": [0.49, 0.56, 0.665, 0.842]},
                "input_res": {"mycam": 2.5},
                "subpatches": {"mycam": 1},
            },
        )

    def test_different_output_grids(self):
        for grid in [9, 36, 180]:
            self.processor.model.encode = MagicMock(
                return_value=(torch.zeros(2, grid**2, 768), {})
            )
            t, _ = self.processor.encode_raster(
                {"spot": np.zeros((3, 16, 16), dtype=np.uint8)},
                patch_size=40.0,
                output_grid=grid,
            )
            self.assertEqual(
                self.processor.model.encode.call_args[1]["output_grid"], grid
            )
            self.assertEqual(t.shape, (2, grid**2, 768))


class TestUniverSatHelpers(unittest.TestCase):
    def test_get_tile_embedding(self):
        self.assertEqual(
            get_tile_embedding(
                torch.arange(12.0, dtype=torch.float32).reshape(2, 2, 3)
            ).shape,
            (2, 3),
        )
        self.assertEqual(
            get_tile_embedding(
                torch.arange(6.0, dtype=torch.float32).reshape(2, 3)
            ).shape,
            (3,),
        )
        with self.assertRaises(ValueError):
            get_tile_embedding(torch.zeros(2, 2, 2, 2))

    def test_get_pca_rgb(self):
        for shape, res in [
            ((16, 128), (4, 4, 3)),
            ((4, 4, 128), (4, 4, 3)),
            ((2, 16, 128), (2, 4, 4, 3)),
            ((2, 4, 4, 128), (2, 4, 4, 3)),
        ]:
            self.assertEqual(get_pca_rgb(torch.randn(*shape)).shape, res)


if __name__ == "__main__":
    unittest.main()
