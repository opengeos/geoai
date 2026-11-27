#!/usr/bin/env python
"""Offline regression tests for core runtime crashes."""

import importlib
import inspect
import json
import os
import sys
import tempfile
import types
import typing
import unittest
import warnings
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import rasterio
import torch
from click.testing import CliRunner
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import Polygon


class TestTimmLoggerShadowing(unittest.TestCase):
    """Regression tests for CSVLogger shadowing module-level logger."""

    def test_timm_segment_train_does_not_shadow_module_logger(self):
        """train_segmentation_model must not shadow module logger with CSVLogger."""
        import geoai.timm_segment as ts

        dummy_ds = torch.utils.data.TensorDataset(
            torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long)
        )

        class FakeCSVLogger:
            """Minimal stand-in without .info() like lightning CSVLogger."""

            def __init__(self, save_dir, name="lightning_logs"):
                self.save_dir = save_dir
                self.name = name

        fake_trainer = MagicMock()
        fake_trainer.logged_metrics = {}
        fake_ckpt = MagicMock()
        fake_ckpt.best_model_path = "/tmp/best.ckpt"

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(ts, "LIGHTNING_AVAILABLE", True),
                patch.object(ts, "TimmSegmentationModel", return_value=MagicMock()),
                patch.object(ts, "ModelCheckpoint", return_value=fake_ckpt),
                patch.object(ts, "EarlyStopping", return_value=MagicMock()),
                patch.object(ts, "CSVLogger", FakeCSVLogger),
                patch.object(
                    ts.pl, "Trainer", return_value=fake_trainer
                ) as mock_trainer_cls,
            ):
                model = ts.train_timm_segmentation(
                    train_dataset=dummy_ds,
                    val_dataset=dummy_ds,
                    test_dataset=dummy_ds,
                    output_dir=tmpdir,
                    num_epochs=1,
                    num_workers=0,
                    accelerator="cpu",
                    devices=1,
                )

            self.assertIsNotNone(model)
            trainer_kwargs = mock_trainer_cls.call_args.kwargs
            self.assertIsInstance(trainer_kwargs["logger"], FakeCSVLogger)
            fake_trainer.fit.assert_called_once()
            fake_trainer.test.assert_called_once()

    def test_timm_train_does_not_shadow_module_logger(self):
        """train_timm_classifier must not shadow module logger with CSVLogger."""
        import geoai.timm_train as tt

        dummy_ds = torch.utils.data.TensorDataset(
            torch.zeros(2, 3, 8, 8), torch.zeros(2, dtype=torch.long)
        )

        class FakeCSVLogger:
            """Minimal stand-in without .info() like lightning CSVLogger."""

            def __init__(self, save_dir, name="lightning_logs"):
                self.save_dir = save_dir
                self.name = name

        fake_trainer = MagicMock()
        fake_ckpt = MagicMock()
        fake_ckpt.best_model_path = "/tmp/best.ckpt"

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(tt, "LIGHTNING_AVAILABLE", True),
                patch.object(tt, "TimmClassifier", return_value=MagicMock()),
                patch.object(tt, "ModelCheckpoint", return_value=fake_ckpt),
                patch.object(tt, "EarlyStopping", return_value=MagicMock()),
                patch.object(tt, "CSVLogger", FakeCSVLogger),
                patch.object(
                    tt.pl, "Trainer", return_value=fake_trainer
                ) as mock_trainer_cls,
            ):
                model = tt.train_timm_classifier(
                    train_dataset=dummy_ds,
                    val_dataset=dummy_ds,
                    test_dataset=dummy_ds,
                    output_dir=tmpdir,
                    num_epochs=1,
                    num_workers=0,
                    accelerator="cpu",
                    devices=1,
                )

            self.assertIsNotNone(model)
            trainer_kwargs = mock_trainer_cls.call_args.kwargs
            self.assertIsInstance(trainer_kwargs["logger"], FakeCSVLogger)
            fake_trainer.fit.assert_called_once()
            fake_trainer.test.assert_called_once()


class TestEsrganFixes(unittest.TestCase):
    """Regression tests for geoai.esrgan annotation and checkpoint resume fixes."""

    def test_preprocess_single_inputs_type_hints_resolve(self):
        """resample_method must use Literal, not non-subscriptable LiteralString."""
        from geoai.esrgan import ESRGANDataPreprocess

        hints = typing.get_type_hints(
            ESRGANDataPreprocess.preprocess_single_inputs,
            localns={"gdal": types.SimpleNamespace(Dataset=object)},
        )
        self.assertIn("resample_method", hints)
        self.assertEqual(typing.get_origin(hints["resample_method"]), typing.Literal)
        self.assertIn("bilinear", typing.get_args(hints["resample_method"]))

    def test_train_esrgan_resume_loads_checkpoints_with_map_location(self):
        """Resuming train_esrgan with last_epoch >= 0 must load checkpoints cleanly."""
        import torch.nn as nn
        from geoai.esrgan import Discriminator, ESRGAN, ESRGANGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ESRGAN(scale=4, data_path=tmpdir, model_path=tmpdir)
            os.makedirs(trainer.model_path, exist_ok=True)
            gen = ESRGANGenerator(in_nc=1, out_nc=1, nf=4, nb=1, gc=2, scale=4)
            dis = Discriminator()
            torch.save(
                gen.state_dict(),
                os.path.join(trainer.model_path, "generator_checkpoint.pt"),
            )
            torch.save(
                dis.state_dict(),
                os.path.join(trainer.model_path, "discriminator_checkpoint.pt"),
            )

            low_res = [torch.zeros(1, 8, 8) for _ in range(4)]
            high_res = [torch.zeros(1, 32, 32) for _ in range(4)]

            fake_vgg = MagicMock()
            fake_vgg.to.return_value = fake_vgg
            real_load = torch.load
            load_calls = []

            def spy_load(*args, **kwargs):
                load_calls.append((args, kwargs))
                return real_load(*args, **kwargs)

            with (
                patch("geoai.esrgan.get_device", return_value=torch.device("cpu")),
                patch("geoai.esrgan.VGGPerceptualLoss", return_value=fake_vgg),
                patch("geoai.esrgan.torch.load", side_effect=spy_load),
            ):
                # last_epoch=0, num_epochs=1 loads checkpoints and skips loop cleanly
                trainer.train_esrgan(
                    low_res=low_res,
                    high_res=high_res,
                    load_tensors=False,
                    batch_size=2,
                    num_epochs=1,
                    last_epoch=0,
                    nfilters=4,
                    nblocks=1,
                    ngroups=2,
                )

            self.assertEqual(len(load_calls), 2)
            for _args, kwargs in load_calls:
                self.assertEqual(kwargs.get("map_location"), torch.device("cpu"))
                self.assertTrue(kwargs.get("weights_only"))


class TestGeoAgentsTyping(unittest.TestCase):
    """Regression test for missing List import in geoai.agents.geo_agents."""

    def test_catalog_agent_search_datasets_type_hints_resolve(self):
        """List must be imported in geoai.agents.geo_agents for search_datasets."""

        class _StubClass:
            pass

        stub_modules = {}
        for mod_name in [
            "boto3",
            "botocore",
            "botocore.config",
            "ipyevents",
            "ipywidgets",
            "IPython",
            "IPython.display",
            "leafmap",
            "leafmap.maplibregl",
            "strands",
            "strands.models",
            "strands.models.anthropic",
            "strands.models.ollama",
            "strands.models.openai",
            "strands.models.gemini",
            "geoai.agents.catalog_tools",
            "geoai.agents.data_tools",
            "geoai.agents.map_tools",
            "geoai.agents.stac_tools",
        ]:
            if mod_name not in sys.modules:
                stub = types.ModuleType(mod_name)
                for attr in [
                    "Config",
                    "Event",
                    "display",
                    "Agent",
                    "tool",
                    "BedrockModel",
                    "AnthropicModel",
                    "OllamaModel",
                    "OpenAIModel",
                    "GeminiModel",
                    "CatalogTools",
                    "DataTools",
                    "MapSession",
                    "MapTools",
                    "STACTools",
                ]:
                    setattr(stub, attr, _StubClass)
                stub_modules[mod_name] = stub

        sys.modules.pop("geoai.agents.geo_agents", None)
        try:
            with patch.dict(sys.modules, stub_modules):
                mod = importlib.import_module("geoai.agents.geo_agents")
                hints = typing.get_type_hints(mod.CatalogAgent.search_datasets)
                self.assertEqual(
                    hints["return"], typing.List[typing.Dict[str, typing.Any]]
                )
        finally:
            sys.modules.pop("geoai.agents.geo_agents", None)


class TestVectorFixes(unittest.TestCase):
    """Regression tests for geoai.utils.vector."""

    def test_add_geometric_properties_convex_hull_area_only(self):
        """Requesting only convex_hull_area must not raise KeyError or add area_convex."""
        from geoai.utils.vector import add_geometric_properties

        poly = Polygon([(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32610")

        for unit, expected in [("m2", 100.0), ("ha", 0.01), ("km2", 0.0001)]:
            res = add_geometric_properties(
                gdf, properties=["convex_hull_area"], area_unit=unit
            )
            self.assertIn(f"convex_hull_area_{unit}", res.columns)
            self.assertNotIn(f"area_convex_{unit}", res.columns)
            self.assertAlmostEqual(
                float(res[f"convex_hull_area_{unit}"].iloc[0]), expected, places=6
            )

    def test_add_geometric_properties_both_convex_columns(self):
        """Requesting both area_convex and convex_hull_area must populate both columns."""
        from geoai.utils.vector import add_geometric_properties

        poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32610")

        res = add_geometric_properties(
            gdf, properties=["area_convex", "convex_hull_area"], area_unit="ha"
        )
        self.assertIn("area_convex_ha", res.columns)
        self.assertIn("convex_hull_area_ha", res.columns)
        self.assertAlmostEqual(float(res["area_convex_ha"].iloc[0]), 1.0, places=6)
        self.assertAlmostEqual(float(res["convex_hull_area_ha"].iloc[0]), 1.0, places=6)

    def test_geojson_to_coords_transforms_crs_and_scopes_warnings(self):
        """geojson_to_coords must transform coordinates and not mutate global warnings."""
        from geoai.utils.vector import geojson_to_coords

        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-122.5, 37.7],
                                [-122.3, 37.7],
                                [-122.3, 37.9],
                                [-122.5, 37.9],
                                [-122.5, 37.7],
                            ]
                        ],
                    },
                }
            ],
        }

        filters_before = list(warnings.filters)
        coords = geojson_to_coords(fc, src_crs="epsg:4326", dst_crs="epsg:3857")
        filters_after = list(warnings.filters)

        self.assertEqual(filters_before, filters_after)
        self.assertEqual(len(coords), 1)
        x, y = coords[0]
        # Centroid (-122.4, 37.8) in EPSG:3857 is approx (-13625505.67, 4551210.92)
        self.assertAlmostEqual(x, -13625505.67, delta=10.0)
        self.assertAlmostEqual(y, 4551210.92, delta=10.0)


class TestGeometryFixes(unittest.TestCase):
    """Regression test for region_groups default property typo."""

    def test_region_groups_default_properties(self):
        """region_groups must succeed with default properties (equivalent_diameter_area)."""
        from geoai.utils.geometry import region_groups

        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[2:8, 2:8] = 1
        arr[12:18, 12:18] = 1

        labeled, df = region_groups(arr, min_size=1)
        self.assertEqual(labeled.shape, (20, 20))
        self.assertEqual(len(df), 2)
        self.assertIn("equivalent_diameter_area", df.columns)
        self.assertNotIn("diameter_areagth", df.columns)


class TestVisualizationFixes(unittest.TestCase):
    """Regression test for UnboundLocalError on rasterio in create_overview_image."""

    def test_create_overview_image_with_raster_class_data(self):
        """create_overview_image must read raster in_class_data without UnboundLocalError."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.utils.visualization import create_overview_image

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.tif")
            mask_path = os.path.join(tmpdir, "mask.tif")
            out_path = os.path.join(tmpdir, "overview.png")
            transform = from_origin(0.0, 32.0, 1.0, 1.0)

            with rasterio.open(
                img_path,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=3,
                dtype="uint8",
                crs="EPSG:32610",
                transform=transform,
            ) as dst:
                dst.write(np.full((3, 32, 32), 200, dtype=np.uint8))

            mask_arr = np.zeros((32, 32), dtype=np.uint8)
            mask_arr[:16, :16] = 1
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=1,
                dtype="uint8",
                crs="EPSG:32610",
                transform=transform,
            ) as dst:
                dst.write(mask_arr, 1)

            tile_coords = [
                {
                    "x": 0,
                    "y": 0,
                    "bounds": (0.0, 16.0, 16.0, 32.0),
                    "has_features": True,
                }
            ]

            real_open = rasterio.open
            opened_paths = []

            def spy_open(path, *args, **kwargs):
                opened_paths.append(str(path))
                return real_open(path, *args, **kwargs)

            with real_open(img_path) as src:
                with patch(
                    "geoai.utils.visualization.rasterio.open", side_effect=spy_open
                ):
                    result_path = create_overview_image(
                        src=src,
                        tile_coordinates=tile_coords,
                        output_path=out_path,
                        tile_size=16,
                        stride=16,
                        in_class_data=mask_path,
                    )

            self.assertEqual(result_path, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn(mask_path, opened_paths)

    def test_create_overview_image_without_class_data(self):
        """create_overview_image must also work when in_class_data is None."""
        import matplotlib

        matplotlib.use("Agg")
        from geoai.utils.visualization import create_overview_image

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.tif")
            out_path = os.path.join(tmpdir, "overview_raw.png")
            transform = from_origin(0.0, 32.0, 1.0, 1.0)

            with rasterio.open(
                img_path,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=3,
                dtype="uint8",
                crs="EPSG:32610",
                transform=transform,
            ) as dst:
                arr = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
                dst.write(np.stack([arr, arr, arr], axis=0))

            with rasterio.open(img_path) as src:
                result_path = create_overview_image(
                    src=src,
                    tile_coordinates=[],
                    output_path=out_path,
                    tile_size=16,
                    stride=16,
                    in_class_data=None,
                )

            self.assertEqual(result_path, out_path)
            self.assertTrue(os.path.exists(out_path))


class TestSamFixes(unittest.TestCase):
    """Regression tests for geoai.sam generate, save_masks, and predict."""

    def _bare_sam(self):
        from geoai.sam import SamGeo

        sam = object.__new__(SamGeo)
        sam.source = None
        sam.image = None
        sam.masks = None
        sam.device = "cpu"
        sam.embeddings = torch.zeros(1)
        return sam

    def test_generate_with_ndarray_passes_pil_image(self):
        """generate(np.ndarray) must pass a PIL Image to mask_generator, not None."""
        sam = self._bare_sam()
        captured = {}

        def fake_generator(inp, **kwargs):
            captured["input"] = inp
            mask = np.zeros((16, 16), dtype=bool)
            mask[2:6, 2:6] = True
            return {"masks": [mask], "scores": [0.95]}

        sam.mask_generator = fake_generator
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        sam.generate(arr)

        self.assertIsNone(sam.source)
        self.assertIsInstance(captured["input"], Image.Image)
        self.assertEqual(captured["input"].size, (16, 16))
        self.assertEqual(sam.objects.shape, (16, 16))

    def test_save_masks_empty_unique(self):
        """save_masks(unique=True) must handle empty masks without IndexError."""
        sam = self._bare_sam()
        sam.image = np.zeros((16, 16, 3), dtype=np.uint8)
        sam.masks = []

        sam.save_masks(unique=True)
        self.assertEqual(sam.objects.shape, (16, 16))
        self.assertEqual(int(sam.objects.sum()), 0)

    def test_save_masks_erosion_kernel_imports_cv2(self):
        """save_masks(unique=False, erosion_kernel=...) must import cv2 lazily."""
        sam = self._bare_sam()
        sam.image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[3:13, 3:13] = True
        sam.masks = [{"segmentation": mask, "area": int(mask.sum())}]

        sam.save_masks(unique=False, erosion_kernel=np.ones((3, 3), np.uint8))
        self.assertEqual(sam.objects.shape, (16, 16))
        self.assertGreater(int(sam.objects.sum()), 0)
        # Border pixel eroded away, interior pixel retained
        self.assertEqual(int(sam.objects[3, 3]), 0)
        self.assertEqual(int(sam.objects[8, 8]), 255)

    def test_predict_boxes_geojson_file_and_dict(self):
        """predict(boxes=str|dict) must lazily import geopandas without NameError."""
        sam = self._bare_sam()
        sam.image = np.zeros((16, 16, 3), dtype=np.uint8)
        fake_outputs = MagicMock()
        fake_outputs.pred_masks = torch.zeros(1, 1, 1, 16, 16)
        fake_outputs.iou_scores = torch.ones(1, 1, 1)
        sam.predictor = MagicMock(return_value=fake_outputs)

        fake_inputs = MagicMock()
        fake_inputs.to.return_value = {
            "input_boxes": torch.zeros(1, 1, 4),
            "original_sizes": torch.tensor([[16, 16]]),
            "reshaped_input_sizes": torch.tensor([[16, 16]]),
        }
        sam.processor = MagicMock(return_value=fake_inputs)
        sam.processor.image_processor.post_process_masks.return_value = [
            torch.zeros(1, 1, 16, 16)
        ]

        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]],
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            geojson_path = os.path.join(tmpdir, "boxes.geojson")
            with open(geojson_path, "w", encoding="utf-8") as f:
                json.dump(fc, f)

            sam.predict(boxes=geojson_path)
            self.assertIsNotNone(sam.boxes)
            self.assertEqual(sam.boxes.shape, (1, 4))

            sam.predict(boxes=fc)
            self.assertIsNotNone(sam.boxes)
            self.assertEqual(sam.boxes.shape, (1, 4))


class TestCLIDownloadNaip(unittest.TestCase):
    """Regression test for geoai CLI download naip output_dir kwarg."""

    def test_cli_download_naip_passes_output_dir(self):
        """CLI download naip must pass output_dir matching download_naip signature."""
        from geoai.cli import main
        from geoai.download import download_naip as real_download_naip

        sig = inspect.signature(real_download_naip)
        called = {}

        def strict_stub(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            called.update(bound.arguments)
            return ["/tmp/tile.tif"]

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("geoai.download.download_naip", side_effect=strict_stub):
                result = runner.invoke(
                    main,
                    [
                        "download",
                        "naip",
                        "--bbox",
                        "-122.5,37.7,-122.4,37.8",
                        "--output",
                        tmpdir,
                    ],
                )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(called["output_dir"], tmpdir)
        self.assertEqual(called["bbox"], [-122.5, 37.7, -122.4, 37.8])


class TestPipelineFixes(unittest.TestCase):
    """Regression tests for SemanticSegmentationStep serialization and checkpoint paths."""

    def test_semantic_segmentation_step_json_roundtrip(self):
        """SemanticSegmentationStep must round-trip through to_json and load_pipeline."""
        from geoai.pipeline import Pipeline, SemanticSegmentationStep, load_pipeline

        step = SemanticSegmentationStep(
            name="seg",
            model_path="/tmp/model.pth",
            num_classes=4,
            device="cpu",
        )

        pipe = Pipeline(steps=[step], name="seg_roundtrip")
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "pipe.json")
            pipe.to_json(cfg_path)
            loaded = load_pipeline(cfg_path)

        self.assertEqual(len(loaded.steps), 1)
        loaded_step = loaded.steps[0]
        self.assertIsInstance(loaded_step, SemanticSegmentationStep)
        self.assertEqual(loaded_step.device, "cpu")
        self.assertEqual(loaded_step.device_str, "cpu")
        self.assertEqual(loaded_step.num_classes, 4)

    def test_pipeline_checkpoint_rejects_path_traversal_name(self):
        """_init_checkpoint must reject pipeline names that could escape checkpoint_dir."""
        from geoai.pipeline import Pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            for bad_name in ["../escape", "sub/dir", "a\\b", "..", "foo/../bar"]:
                pipe = Pipeline(steps=[], checkpoint_dir=tmpdir, name=bad_name)
                with self.assertRaises(ValueError, msg=f"Accepted {bad_name!r}"):
                    pipe._init_checkpoint()


if __name__ == "__main__":
    unittest.main()
