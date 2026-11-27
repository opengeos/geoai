"""Regression tests for Issue #1013 Batch C (models, pipeline, CLI, agents)."""

import ast
import datetime
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import requests
from click.testing import CliRunner
from PIL import Image
from rasterio.transform import Affine, from_bounds


def _make_geotiff(
    path: str,
    width: int = 32,
    height: int = 32,
    bands: int = 3,
    dtype: str = "uint8",
    crs: str = "EPSG:4326",
    bounds=(-122.5, 37.7, -122.3, 37.9),
    data: np.ndarray = None,
    photometric: str = None,
):
    """Create a small GeoTIFF for offline tests."""
    if data is None:
        if dtype == "uint16":
            data = np.linspace(
                500, 3500, bands * height * width, dtype=np.uint16
            ).reshape(bands, height, width)
        elif dtype == "float32":
            data = np.linspace(
                0.1, 0.9, bands * height * width, dtype=np.float32
            ).reshape(bands, height, width)
        else:
            data = np.full((bands, height, width), 120, dtype=np.uint8)
    transform = from_bounds(*bounds, width, height)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    if photometric is not None:
        profile["photometric"] = photometric
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return data, transform


def _png_bytes(width: int = 16, height: int = 16) -> bytes:
    """Return raw PNG bytes for mocked HTTP image downloads."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _import_agent_module_with_stubs(module_name: str):
    """Import a geoai.agents.* module even when optional strands/boto3 are absent."""
    if "strands" in sys.modules and not getattr(
        sys.modules["strands"], "_is_stub", False
    ):
        return importlib.import_module(module_name)

    fake_modules = {}

    def _passthrough_tool(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator

    strands_mod = types.ModuleType("strands")
    strands_mod._is_stub = True
    strands_mod.tool = _passthrough_tool

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name")
            self.model = kwargs.get("model")
            self.tools_list = kwargs.get("tools", [])
            self.system_prompt = kwargs.get("system_prompt")
            self.callback_handler = kwargs.get("callback_handler")

    strands_mod.Agent = _FakeAgent

    class _BaseFakeModel:
        def __init__(self, model_id="default", client_args=None, **kwargs):
            self.config = {"model_id": model_id}
            self.client_args = client_args if client_args is not None else {}
            self.extra_kwargs = kwargs

    class _FakeOllamaModel(_BaseFakeModel):
        def __init__(
            self,
            host="http://localhost:11434",
            model_id="llama3.1",
            client_args=None,
            **kwargs,
        ):
            super().__init__(model_id=model_id, client_args=client_args, **kwargs)
            self.host = host

    class _FakeOpenAIModel(_BaseFakeModel):
        pass

    class _FakeAnthropicModel(_BaseFakeModel):
        pass

    class _FakeGeminiModel(_BaseFakeModel):
        pass

    class _FakeBedrockModel(_BaseFakeModel):
        def __init__(self, model_id="bedrock", region_name=None, **kwargs):
            super().__init__(model_id=model_id, **kwargs)
            self.region_name = region_name

    models_mod = types.ModuleType("strands.models")
    models_mod.BedrockModel = _FakeBedrockModel
    anthropic_mod = types.ModuleType("strands.models.anthropic")
    anthropic_mod.AnthropicModel = _FakeAnthropicModel
    ollama_mod = types.ModuleType("strands.models.ollama")
    ollama_mod.OllamaModel = _FakeOllamaModel
    openai_mod = types.ModuleType("strands.models.openai")
    openai_mod.OpenAIModel = _FakeOpenAIModel
    gemini_mod = types.ModuleType("strands.models.gemini")
    gemini_mod.GeminiModel = _FakeGeminiModel

    fake_modules.update(
        {
            "strands": strands_mod,
            "strands.models": models_mod,
            "strands.models.anthropic": anthropic_mod,
            "strands.models.ollama": ollama_mod,
            "strands.models.openai": openai_mod,
            "strands.models.gemini": gemini_mod,
        }
    )

    if importlib.util.find_spec("boto3") is None:
        boto3_mod = types.ModuleType("boto3")
        boto3_mod.Session = MagicMock
        fake_modules["boto3"] = boto3_mod
    if importlib.util.find_spec("botocore") is None:
        botocore_mod = types.ModuleType("botocore")
        botocore_cfg = types.ModuleType("botocore.config")
        botocore_cfg.Config = MagicMock
        fake_modules["botocore"] = botocore_mod
        fake_modules["botocore.config"] = botocore_cfg
    if importlib.util.find_spec("ipyevents") is None:
        ipyevents_mod = types.ModuleType("ipyevents")
        ipyevents_mod.Event = MagicMock
        fake_modules["ipyevents"] = ipyevents_mod

    return fake_modules


# ---------------------------------------------------------------------------
# Group 1: geoai/geoai.py lazy imports
# ---------------------------------------------------------------------------


class TestGroup1GeoaiLazyImports(unittest.TestCase):
    """Importing geoai.geoai / Map / LeafMap must not pull in torch/timm/transformers."""

    def test_import_geoai_and_map_does_not_load_ml_frameworks(self):
        repo_root = str(Path(__file__).resolve().parents[1])
        code = (
            "import sys\n"
            f"sys.path.insert(0, {repo_root!r})\n"
            "import geoai\n"
            "import geoai.geoai\n"
            "_ = geoai.Map\n"
            "_ = geoai.LeafMap\n"
            "loaded = [m for m in ('torch', 'timm', 'transformers') if m in sys.modules]\n"
            "assert not loaded, f'Heavy ML modules loaded unexpectedly: {loaded}'\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Subprocess failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}",
        )


# ---------------------------------------------------------------------------
# Group 2: geoai/onnx.py and geoai/auto.py
# ---------------------------------------------------------------------------


class TestGroup2OnnxAndAuto(unittest.TestCase):
    """Tests for tiled inference, box_threshold, georeferencing, and HTTP hygiene."""

    def _make_stub_onnx_model(
        self, task="semantic-segmentation", tile_size=16, overlap=4
    ):
        from geoai.onnx import ONNXGeoModel

        model = ONNXGeoModel.__new__(ONNXGeoModel)
        model.task = task
        model.tile_size = tile_size
        model.overlap = overlap
        model.id2label = {}
        model.input_name = "input"
        model.output_names = ["output"]
        model._model_height = None
        model._model_width = None
        model.session = MagicMock()
        return model

    def _make_stub_auto_model(
        self, task="semantic-segmentation", tile_size=16, overlap=4
    ):
        from geoai.auto import AutoGeoModel

        model = AutoGeoModel.__new__(AutoGeoModel)
        model.task = task
        model.tile_size = tile_size
        model.overlap = overlap
        model.device = "cpu"
        model.processor = None
        model.model_name = "stub-model"
        model.model = MagicMock()
        return model

    def test_onnx_predict_tiled_validates_overlap(self):
        data = np.zeros((3, 32, 32), dtype=np.uint8)
        for bad_overlap in (-1, 8, 10):
            model = self._make_stub_onnx_model(tile_size=16, overlap=bad_overlap)
            with self.assertRaises(ValueError):
                model._predict_tiled(data, None)

    def test_auto_predict_tiled_validates_overlap(self):
        data = np.zeros((3, 32, 32), dtype=np.uint8)
        for bad_overlap in (-1, 8, 12):
            model = self._make_stub_auto_model(tile_size=16, overlap=bad_overlap)
            with self.assertRaises(ValueError):
                model._predict_tiled("dummy", data, None)

    def test_onnx_predict_tiled_multiclass_accumulates_probabilities(self):
        model = self._make_stub_onnx_model(
            task="semantic-segmentation", tile_size=16, overlap=2
        )

        def fake_single(
            tile, metadata, threshold=0.5, return_probabilities=False, box_threshold=0.3
        ):
            _, h, w = tile.shape
            probs = np.zeros((3, h, w), dtype=np.float32)
            probs[0] = 0.05
            probs[1] = 0.10
            probs[2] = 0.85  # class 2 should win everywhere
            return {
                "mask": np.full((h, w), 2, dtype=np.uint8),
                "probabilities": probs if return_probabilities else None,
            }

        model._predict_single = fake_single
        data = np.zeros((3, 28, 28), dtype=np.uint8)
        res = model._predict_tiled(data, None, return_probabilities=True)

        self.assertEqual(res["mask"].shape, (28, 28))
        self.assertTrue(np.all(res["mask"] == 2))
        self.assertEqual(res["probabilities"].shape, (3, 28, 28))
        np.testing.assert_allclose(res["probabilities"][2], 0.85, atol=1e-5)

    def test_onnx_predict_tiled_discrete_multiclass_preserves_ids(self):
        model = self._make_stub_onnx_model(
            task="semantic-segmentation", tile_size=16, overlap=2
        )

        def fake_single(
            tile, metadata, threshold=0.5, return_probabilities=False, box_threshold=0.3
        ):
            _, h, w = tile.shape
            return {"mask": np.full((h, w), 3, dtype=np.uint8)}

        model._predict_single = fake_single
        data = np.zeros((3, 24, 24), dtype=np.uint8)
        res = model._predict_tiled(data, None)
        self.assertTrue(np.all(res["mask"] == 3))

    def test_onnx_predict_tiled_binary_still_works(self):
        model = self._make_stub_onnx_model(
            task="semantic-segmentation", tile_size=16, overlap=2
        )

        def fake_single(
            tile, metadata, threshold=0.5, return_probabilities=False, box_threshold=0.3
        ):
            _, h, w = tile.shape
            probs = np.full((h, w), 0.7, dtype=np.float32)
            return {
                "mask": (probs > threshold).astype(np.uint8),
                "probabilities": probs,
            }

        model._predict_single = fake_single
        data = np.zeros((3, 24, 24), dtype=np.uint8)
        res_low = model._predict_tiled(data, None, threshold=0.5)
        res_high = model._predict_tiled(data, None, threshold=0.8)
        self.assertTrue(np.all(res_low["mask"] == 1))
        self.assertTrue(np.all(res_high["mask"] == 0))

    def test_onnx_predict_tiled_depth_returns_continuous_values(self):
        model = self._make_stub_onnx_model(
            task="depth-estimation", tile_size=16, overlap=2
        )

        def fake_single(
            tile, metadata, threshold=0.5, return_probabilities=False, box_threshold=0.3
        ):
            _, h, w = tile.shape
            depth = np.full((h, w), 14.5, dtype=np.float32)
            return {"depth": depth, "output": depth}

        model._predict_single = fake_single
        data = np.zeros((3, 24, 24), dtype=np.uint8)
        res = model._predict_tiled(data, None)
        self.assertIn("depth", res)
        self.assertIn("output", res)
        self.assertNotIn("mask", res)
        np.testing.assert_allclose(res["depth"], 14.5, atol=1e-5)

    def test_auto_predict_tiled_multiclass_and_depth(self):
        # Multi-class
        seg_model = self._make_stub_auto_model(
            task="semantic-segmentation", tile_size=16, overlap=2
        )

        def fake_seg_single(tile, metadata, threshold=0.5, return_probabilities=False):
            _, h, w = tile.shape
            probs = np.zeros((4, h, w), dtype=np.float32)
            probs[3] = 0.9
            probs[0] = 0.1
            return {
                "mask": np.full((h, w), 3, dtype=np.uint8),
                "probabilities": probs if return_probabilities else None,
            }

        seg_model._predict_single = fake_seg_single
        data = np.zeros((3, 24, 24), dtype=np.uint8)
        seg_res = seg_model._predict_tiled(
            "dummy", data, None, return_probabilities=True
        )
        self.assertTrue(np.all(seg_res["mask"] == 3))
        self.assertEqual(seg_res["probabilities"].shape, (4, 24, 24))

        # Depth
        depth_model = self._make_stub_auto_model(
            task="depth-estimation", tile_size=16, overlap=2
        )

        def fake_depth_single(
            tile, metadata, threshold=0.5, return_probabilities=False
        ):
            _, h, w = tile.shape
            d = np.full((h, w), 7.25, dtype=np.float32)
            return {"depth": d, "output": d}

        depth_model._predict_single = fake_depth_single
        depth_res = depth_model._predict_tiled("dummy", data, None)
        self.assertIn("depth", depth_res)
        self.assertNotIn("mask", depth_res)
        np.testing.assert_allclose(depth_res["depth"], 7.25, atol=1e-5)

    def test_onnx_box_threshold_forwarded_to_process_outputs(self):
        model = self._make_stub_onnx_model(
            task="object-detection", tile_size=64, overlap=0
        )
        # sigmoid(0.85) ~= 0.70, sigmoid(2.2) ~= 0.90
        logits = np.array([[[0.85, -2.0], [2.2, -2.0]]], dtype=np.float32)
        boxes = np.array([[[0, 0, 10, 10], [20, 20, 40, 40]]], dtype=np.float32)
        model.session.run.return_value = [logits, boxes]

        img = np.zeros((3, 32, 32), dtype=np.uint8)
        res = model.predict(img, threshold=0.5, box_threshold=0.8)
        # Only the second box (score ~0.90 > 0.8) should survive; first (~0.70) filtered out.
        self.assertEqual(len(res["boxes"]), 1)
        np.testing.assert_array_equal(res["boxes"][0], [20, 20, 40, 40])

    def test_onnx_url_source_uses_with_timeout_and_raise_for_status(self):
        model = self._make_stub_onnx_model(
            task="depth-estimation", tile_size=64, overlap=0
        )
        model._predict_single = MagicMock(
            return_value={"depth": np.zeros((16, 16), dtype=np.float32)}
        )

        resp = MagicMock()
        resp.raw = io.BytesIO(_png_bytes(16, 16))
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with patch("requests.get", return_value=resp) as mock_get:
            model.predict("https://example.com/image.png")

        mock_get.assert_called_once_with(
            "https://example.com/image.png", stream=True, timeout=60
        )
        resp.raise_for_status.assert_called_once()
        resp.__exit__.assert_called_once()

    def test_auto_predict_and_detection_url_use_with_timeout_and_raise_for_status(self):
        model = self._make_stub_auto_model(
            task="depth-estimation", tile_size=64, overlap=0
        )
        model._predict_single = MagicMock(
            return_value={"depth": np.zeros((16, 16), dtype=np.float32)}
        )

        resp = MagicMock()
        resp.raw = io.BytesIO(_png_bytes(16, 16))
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with patch("geoai.auto.requests.get", return_value=resp) as mock_get:
            model.predict("https://example.com/image.png")

        mock_get.assert_called_once_with(
            "https://example.com/image.png", stream=True, timeout=60
        )
        resp.raise_for_status.assert_called_once()
        resp.__exit__.assert_called_once()

    def test_auto_predict_detection_georeferences_boxes(self):
        import geopandas as gpd

        model = self._make_stub_auto_model(task="object-detection")
        fake_proc = MagicMock()
        fake_inputs = MagicMock()
        fake_inputs.to.return_value = fake_inputs
        fake_inputs.items.return_value = []
        fake_inputs.keys.return_value = []
        fake_proc.return_value = fake_inputs
        model.processor = MagicMock(processor=fake_proc)
        model.model.return_value = MagicMock()
        model._process_detection_outputs = MagicMock(
            return_value={
                "boxes": np.array([[10.0, 5.0, 20.0, 15.0]]),
                "scores": np.array([0.95]),
                "labels": ["building"],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            tif_path = os.path.join(td, "scene.tif")
            out_vec = os.path.join(td, "dets.geojson")
            bounds = (500000.0, 4000000.0, 500320.0, 4000320.0)
            _, transform = _make_geotiff(
                tif_path, width=32, height=32, crs="EPSG:32617", bounds=bounds
            )

            res = model._predict_detection(tif_path, output_vector_path=out_vec)
            self.assertIn("geodataframe", res)
            gdf = res["geodataframe"]
            self.assertEqual(gdf.crs.to_epsg(), 32617)

            gx1, gy1 = transform * (10.0, 5.0)
            gx2, gy2 = transform * (20.0, 15.0)
            geom_bounds = gdf.geometry.iloc[0].bounds
            np.testing.assert_allclose(
                geom_bounds,
                (min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)),
                atol=1e-6,
            )
            reloaded = gpd.read_file(out_vec)
            self.assertEqual(reloaded.crs.to_epsg(), 32617)


# ---------------------------------------------------------------------------
# Group 3: geoai/tools/sr.py
# ---------------------------------------------------------------------------


class TestGroup3SuperResolution(unittest.TestCase):
    """Tests for geoai.tools.sr streaming, transform scaling, and config cache."""

    def test_save_geotiff_scales_rotated_affine_properly(self):
        pytest.importorskip("omegaconf")
        from geoai.tools.sr import save_geotiff

        old_transform = Affine(10.0, 2.0, 500000.0, 1.0, -10.0, 4000000.0)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": 8,
            "height": 8,
            "count": 4,
            "crs": "EPSG:32617",
            "transform": old_transform,
        }
        data = np.ones((4, 32, 32), dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "sr.tif")
            save_geotiff(data, profile, out_path, scale=4)
            with rasterio.open(out_path) as src:
                expected = old_transform * Affine.scale(0.25, 0.25)
                self.assertEqual(src.transform, expected)
                # Specifically verify off-diagonal shear/rotation terms b and d scaled!
                self.assertAlmostEqual(src.transform.b, 0.5)
                self.assertAlmostEqual(src.transform.d, 0.25)

    def test_cached_config_downloads_with_timeout_then_reuses_cache(self):
        pytest.importorskip("omegaconf")
        from geoai.tools.sr import _get_cached_config

        resp = MagicMock()
        resp.text = "ckpt_version: opensr_v1.ckpt\nscale: 4\n"
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as td:
            with (
                patch("geoai.tools.sr.torch.hub.get_dir", return_value=td),
                patch("geoai.tools.sr.requests.get", return_value=resp) as mock_get,
            ):
                cfg1 = _get_cached_config("https://example.com/config_10m.yaml")
                cfg2 = _get_cached_config("https://example.com/config_10m.yaml")

            self.assertEqual(cfg1.ckpt_version, "opensr_v1.ckpt")
            self.assertEqual(cfg2.ckpt_version, "opensr_v1.ckpt")
            mock_get.assert_called_once_with(
                "https://example.com/config_10m.yaml", timeout=60
            )
            resp.raise_for_status.assert_called_once()

    def test_streaming_patched_inference_does_not_buffer_patches(self):
        pytest.importorskip("omegaconf")
        import torch
        from geoai.tools import sr

        class DummySRModel:
            def __init__(self):
                self.seen_shapes = []

            def forward(self, patch, sampling_steps=1):
                self.seen_shapes.append(tuple(patch.shape))
                return patch.repeat_interleave(2, -2).repeat_interleave(2, -1)

            def uncertainty_map(self, patch, n_variations=4, sampling_steps=1):
                h, w = patch.shape[-2] * 2, patch.shape[-1] * 2
                return torch.full((1, 1, h, w), 0.25)

        lr = torch.rand(1, 4, 48, 48)
        model = DummySRModel()

        with patch.object(
            sr,
            "_stitch_patches",
            side_effect=AssertionError("Should stream, not buffer!"),
        ):
            out = sr._process_patched(
                model,
                lr,
                patch_size=32,
                overlap=8,
                scale=2,
                sampling_steps=1,
                device="cpu",
            )
            unc = sr._process_patched_uncertainty(
                model,
                lr,
                patch_size=32,
                overlap=8,
                scale=2,
                n_variations=4,
                sampling_steps=1,
                device="cpu",
            )

        self.assertEqual(out.shape, (4, 96, 96))
        self.assertEqual(unc.shape, (96, 96))
        self.assertGreater(len(model.seen_shapes), 1)
        for shape in model.seen_shapes:
            self.assertLessEqual(shape[-2], 32)
            self.assertLessEqual(shape[-1], 32)


# ---------------------------------------------------------------------------
# Group 4: geoai/prithvi.py
# ---------------------------------------------------------------------------


class TestGroup4Prithvi(unittest.TestCase):
    """Tests for PrithviProcessor input validation, padding, frames, mask_ratio, metadata."""

    def _make_processor(self, num_frames=3, img_size=16, mask_ratio=0.75):
        from geoai.prithvi import PrithviProcessor

        proc = PrithviProcessor.__new__(PrithviProcessor)
        proc.num_frames = num_frames
        proc.img_size = img_size
        proc.mask_ratio = mask_ratio
        proc.bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]
        proc.mean = [0.0] * 6
        proc.std = [1.0] * 6
        proc.device = "cpu"
        proc.model = MagicMock()
        return proc

    def test_load_images_empty_raises_value_error(self):
        proc = self._make_processor()
        with self.assertRaises(ValueError):
            proc.load_images([])

    def test_run_inference_respects_zero_mask_ratio(self):
        import torch

        proc = self._make_processor(num_frames=1, img_size=16, mask_ratio=0.75)
        captured = {}

        def fake_model(x, temporal_coords, location_coords, mr):
            captured["mask_ratio"] = mr
            # Return dummy (loss, pred, mask)
            pred = torch.zeros((1, 1, 6 * 16 * 16))
            mask = torch.zeros((1, 1))
            return None, pred, mask

        proc.model = MagicMock(side_effect=fake_model)
        proc.model.unpatchify = MagicMock(
            side_effect=lambda t: torch.zeros((1, 6, 1, 16, 16))
        )
        inp = torch.zeros((1, 6, 1, 16, 16))
        proc.run_inference(inp, mask_ratio=0.0)
        self.assertEqual(captured["mask_ratio"], 0.0)

    def test_process_images_and_files_padding_frames_and_save_metadata(self):
        import torch

        proc = self._make_processor(num_frames=3, img_size=16, mask_ratio=0.5)
        calls = []

        def fake_run_inference(x, tc, lc, mr):
            calls.append(tuple(x.shape))
            return x.clone(), torch.ones_like(x)

        proc.run_inference = fake_run_inference

        with tempfile.TemporaryDirectory() as td:
            # 8-band float32 GeoTIFF with exact multiple of img_size (16x16),
            # passing only 1 file when num_frames=3.
            tif_path = os.path.join(td, "frame0.tif")
            data = np.ones((8, 16, 16), dtype=np.float32) * 100.0
            _make_geotiff(
                tif_path, width=16, height=16, bands=8, dtype="float32", data=data
            )

            inp, rec, msk = proc.process_images([tif_path], indices=[0, 1, 2, 3, 4, 5])
            self.assertEqual(rec.shape, (1, 6, 3, 16, 16))
            self.assertEqual(len(calls), 1)  # No redundant padding windows!

            calls.clear()
            out_dir = os.path.join(td, "out")
            proc.process_files([tif_path], out_dir, indices=[0, 1, 2, 3, 4, 5])
            # Exact 16x16 must NOT pad to 32x32 (which would produce 4 windows instead of 1)
            self.assertEqual(len(calls), 1)
            rec_file = os.path.join(out_dir, "reconstructed_t0.tif")
            self.assertTrue(os.path.exists(rec_file))
            with rasterio.open(rec_file) as src:
                self.assertEqual(src.dtypes[0], "int16")
                self.assertEqual(src.count, 6)


# ---------------------------------------------------------------------------
# Group 5: geoai/change_detection.py
# ---------------------------------------------------------------------------


class TestGroup5ChangeDetection(unittest.TestCase):
    """Tests for CRS verification and tiled deduplication in ChangeStarDetection."""

    def test_read_and_align_raises_on_crs_mismatch(self):
        from geoai.change_detection import ChangeStarDetection

        det = ChangeStarDetection.__new__(ChangeStarDetection)
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "t1.tif")
            p2 = os.path.join(td, "t2.tif")
            _make_geotiff(p1, crs="EPSG:4326", bounds=(-77.05, 38.88, -77.04, 38.89))
            _make_geotiff(
                p2, crs="EPSG:32618", bounds=(320000, 4300000, 321000, 4301000)
            )
            with self.assertRaisesRegex(ValueError, "CRS mismatch"):
                det._read_and_align_images(p1, p2)

    def test_predict_tiled_validates_overlap_and_deduplicates_boundary_tiles(self):
        from geoai.change_detection import ChangeStarDetection

        det = ChangeStarDetection.__new__(ChangeStarDetection)
        img = np.ones((32, 32, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            det._predict_tiled(img, img, tile_size=32, overlap=32)
        with self.assertRaises(ValueError):
            det._predict_tiled(img, img, tile_size=32, overlap=-1)

        call_count = {"n": 0}

        def fake_tile(t1, t2):
            call_count["n"] += 1
            h, w = t1.shape[:2]
            return {
                "change_prob": np.full((h, w), 0.6, dtype=np.float32),
                "t1_semantic_prob": np.full((h, w), 0.2, dtype=np.float32),
                "t2_semantic_prob": np.full((h, w), 0.8, dtype=np.float32),
                "num_semantic_classes": 1,
            }

        det._predict_tile = fake_tile
        # 32x32 image with tile_size=32, overlap=8: stride=24 would produce
        # y=0->(0,32) and y=24->(0,32) without deduplication (4 calls instead of 1).
        det._predict_tiled(img, img, tile_size=32, overlap=8)
        self.assertEqual(call_count["n"], 1)


# ---------------------------------------------------------------------------
# Group 6: geoai/rfdetr.py and geoai/dinov3_finetune.py
# ---------------------------------------------------------------------------


class TestGroup6RfdetrAndDinov3Finetune(unittest.TestCase):
    """Tests for RF-DETR non-uint8 normalization and DINOv3 segmentation fixes."""

    def test_rfdetr_normalize_tile_to_uint8(self):
        from geoai.rfdetr import _normalize_tile_to_uint8

        # uint16 values > 255 must be stretched, not wrapped modulo 256
        u16 = np.linspace(500, 3500, 16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
        out_u16 = _normalize_tile_to_uint8(u16)
        self.assertEqual(out_u16.dtype, np.uint8)
        self.assertEqual(int(out_u16.min()), 0)
        self.assertEqual(int(out_u16.max()), 255)
        # Monotonicity along flattened channel 0 proves no wraparound
        self.assertGreater(int(out_u16[-1, -1, 0]), int(out_u16[0, 0, 0]))

        # float32 in [0, 1] must scale to [0, 255] instead of truncating to 0
        f32 = np.full((8, 8, 3), 0.5, dtype=np.float32)
        out_f32 = _normalize_tile_to_uint8(f32)
        self.assertEqual(out_f32.dtype, np.uint8)
        self.assertTrue(np.all((out_f32 >= 127) & (out_f32 <= 128)))

    def test_dinov3_segment_geotiff_1band_uint16_photometric_and_nested_dir(self):
        import torch
        from geoai import dinov3_finetune

        captured = {}

        class DummySegmenter(torch.nn.Module):
            patch_size = 16

            def forward(self, x):
                captured["shape"] = tuple(x.shape)
                captured["max"] = float(x.max())
                captured["min"] = float(x.min())
                b, _, h, w = x.shape
                logits = torch.zeros((b, 2, h, w), dtype=torch.float32)
                logits[:, 1] = 1.0
                return logits

        with tempfile.TemporaryDirectory() as td:
            in_tif = os.path.join(td, "single_u16.tif")
            u16_data = np.full((1, 32, 32), 32768, dtype=np.uint16)
            _make_geotiff(
                in_tif,
                width=32,
                height=32,
                bands=1,
                dtype="uint16",
                data=u16_data,
                photometric="minisblack",
            )
            ckpt_path = os.path.join(td, "dummy.pth")
            torch.save({}, ckpt_path)
            out_tif = os.path.join(td, "nested", "subdir", "mask.tif")

            with patch.object(
                dinov3_finetune, "DINOv3Segmenter", return_value=DummySegmenter()
            ):
                dinov3_finetune.dinov3_segment_geotiff(
                    input_path=in_tif,
                    output_path=out_tif,
                    checkpoint_path=ckpt_path,
                    window_size=32,
                    overlap=0,
                    batch_size=1,
                    device="cpu",
                    quiet=True,
                )

            self.assertTrue(os.path.exists(out_tif))
            # 1-band input repeated to 3 channels
            self.assertEqual(captured["shape"][1], 3)
            # uint16 32768 / 65535 ~= 0.5 (not 32768 / 255 ~= 128!)
            self.assertAlmostEqual(captured["max"], 32768.0 / 65535.0, places=3)
            with rasterio.open(out_tif) as src:
                self.assertEqual(src.count, 1)
                self.assertEqual(src.dtypes[0], "uint8")


# ---------------------------------------------------------------------------
# Group 7: weights_only=True, atomic downloads, HTTP hygiene
# ---------------------------------------------------------------------------


class TestGroup7SecurityAndHttpHygiene(unittest.TestCase):
    """Tests for weights_only=True, canopy atomic download, STAC download, vllm session."""

    def test_all_torch_load_calls_specify_weights_only_true(self):
        repo_root = Path(__file__).resolve().parents[1]
        target_files = [
            repo_root / "geoai/dinov3.py",
            repo_root / "geoai/dinov3_finetune.py",
            repo_root / "geoai/object_detect.py",
            repo_root / "geoai/recognize.py",
        ]
        for path in target_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found = 0
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                ):
                    found += 1
                    kw = {k.arg: k.value for k in node.keywords}
                    self.assertIn(
                        "weights_only",
                        kw,
                        f"{path.name}:{node.lineno} missing weights_only=True",
                    )
                    self.assertIsInstance(kw["weights_only"], ast.Constant)
                    self.assertTrue(
                        kw["weights_only"].value,
                        f"{path.name}:{node.lineno} weights_only is not True",
                    )
            self.assertGreater(found, 0, f"Expected torch.load calls in {path.name}")

    def test_canopy_download_checkpoint_atomic_and_cleans_up_on_keyboard_interrupt(
        self,
    ):
        from geoai.canopy import MODEL_VARIANTS, _download_checkpoint

        def interrupted_chunks(chunk_size=8192):
            yield b"partial"
            raise KeyboardInterrupt("aborted")

        resp_fail = MagicMock()
        resp_fail.iter_content.side_effect = interrupted_chunks
        resp_fail.__enter__.return_value = resp_fail
        resp_fail.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as td:
            with patch("requests.get", return_value=resp_fail) as mock_get:
                with self.assertRaises(KeyboardInterrupt):
                    _download_checkpoint("compressed_SSLhuge", cache_dir=td)
            mock_get.assert_called_once_with(
                MODEL_VARIANTS["compressed_SSLhuge"]["url"],
                stream=True,
                timeout=(30, 300),
            )
            self.assertEqual(os.listdir(td), [])

            resp_ok = MagicMock()
            resp_ok.iter_content.return_value = [b"abc", b"def"]
            resp_ok.__enter__.return_value = resp_ok
            resp_ok.__exit__.return_value = False
            with patch("requests.get", return_value=resp_ok):
                ckpt = _download_checkpoint("compressed_SSLhuge", cache_dir=td)
            self.assertTrue(os.path.isfile(ckpt))
            with open(ckpt, "rb") as f:
                self.assertEqual(f.read(), b"abcdef")

    def test_canopy_predict_reads_only_first_3_bands(self):
        from geoai.canopy import CanopyHeightEstimation

        est = CanopyHeightEstimation.__new__(CanopyHeightEstimation)
        est.device = "cpu"
        est.model = lambda batch: batch[:, :1]
        est._normalize_image = lambda batch: batch

        read_args = []
        orig_open = rasterio.open

        class _SpyReader:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

            def read(self, *args, **kwargs):
                read_args.append((args, kwargs))
                return self._inner.read(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        with tempfile.TemporaryDirectory() as td:
            tif = os.path.join(td, "sixband.tif")
            _make_geotiff(tif, width=32, height=32, bands=6, dtype="uint8")
            with patch(
                "geoai.canopy.rasterio.open",
                side_effect=lambda p, *a, **kw: _SpyReader(orig_open(p, *a, **kw)),
            ):
                est.predict(tif, tile_size=32, overlap=0, batch_size=1)

        self.assertEqual(len(read_args), 1)
        self.assertEqual(read_args[0][0], ([1, 2, 3],))

    def test_pc_stac_search_supports_ids_and_uses_items(self):
        from geoai.download import pc_stac_search

        fake_search = MagicMock()
        fake_search.items.return_value = iter([MagicMock(id="item-1")])
        fake_search.get_items.side_effect = AssertionError(
            "Deprecated get_items called!"
        )
        fake_catalog = MagicMock()
        fake_catalog.search.return_value = fake_search

        with patch("geoai.download.Client.open", return_value=fake_catalog):
            items = pc_stac_search(
                collection="sentinel-2-l2a", ids=["item-1"], quiet=True
            )

        self.assertEqual(len(items), 1)
        self.assertEqual(fake_catalog.search.call_args.kwargs["ids"], ["item-1"])
        fake_search.items.assert_called_once()

    def test_pc_stac_download_signs_item_once_not_per_asset(self):
        import pystac
        from geoai.download import pc_stac_download

        item = pystac.Item(
            id="scene1",
            geometry=None,
            bbox=None,
            datetime=datetime.datetime(2024, 1, 1),
            properties={},
        )
        for band in ("B02", "B03", "B04"):
            item.add_asset(
                band,
                pystac.Asset(
                    href=f"https://example.com/{band}.tif",
                    media_type="image/tiff",
                ),
            )

        resp = MagicMock()
        resp.headers = {"content-length": "4"}
        resp.iter_content.return_value = [b"data"]
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as td:
            with (
                patch("geoai.download.pc.sign", side_effect=lambda x: x) as mock_sign,
                patch("geoai.download.requests.get", return_value=resp),
            ):
                results = pc_stac_download(item, output_dir=td, max_workers=2)

        self.assertEqual(mock_sign.call_count, 1)
        self.assertEqual(set(results["scene1"].keys()), {"B02", "B03", "B04"})

    def test_download_pc_stac_item_closes_rioxarray_handles(self):
        from geoai.download import download_pc_stac_item

        fake_asset = MagicMock(href="https://example.com/B02.tif")
        fake_item = MagicMock(id="item1", assets={"B02": fake_asset})
        fake_search = MagicMock()
        fake_search.items.return_value = iter([fake_item])
        fake_catalog = MagicMock()
        fake_catalog.search.return_value = fake_search

        fake_da = MagicMock()
        fake_da.__enter__.return_value = fake_da
        fake_da.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as td:
            with (
                patch("geoai.download.Client.open", return_value=fake_catalog),
                patch("geoai.download.rxr.open_rasterio", return_value=fake_da),
            ):
                download_pc_stac_item(
                    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/item1",
                    bands=["B02"],
                    output_dir=td,
                    show_progress=False,
                )

        fake_da.__exit__.assert_called_once()

    def test_view_pc_items_does_not_mutate_assets_across_items(self):
        import pystac
        from geoai.download import view_pc_items

        item1 = pystac.Item(
            id="item1",
            geometry=None,
            bbox=None,
            datetime=datetime.datetime(2024, 1, 1),
            properties={},
            collection="col-a",
        )
        item1.add_asset("visual", pystac.Asset(href="https://example.com/v.tif"))

        item2 = pystac.Item(
            id="item2",
            geometry=None,
            bbox=None,
            datetime=datetime.datetime(2024, 1, 2),
            properties={},
            collection="col-b",
        )
        item2.add_asset("data", pystac.Asset(href="https://example.com/d.tif"))

        fake_map = MagicMock()
        fake_foliumap = types.ModuleType("leafmap.foliumap")
        fake_foliumap.Map = MagicMock(return_value=fake_map)

        with patch.dict(sys.modules, {"leafmap.foliumap": fake_foliumap}):
            view_pc_items(items=[item1, item2], assets=None, backend="folium")

        self.assertEqual(fake_map.add_stac_layer.call_count, 2)
        first_call = fake_map.add_stac_layer.call_args_list[0].kwargs
        second_call = fake_map.add_stac_layer.call_args_list[1].kwargs
        self.assertEqual(first_call["assets"], ["visual"])
        self.assertEqual(second_call["assets"], ["data"])

    def test_utils_download_strips_query_string_and_closes_response(self):
        from geoai.utils.download import download_file

        resp = MagicMock()
        resp.headers = {"content-length": "5"}
        resp.iter_content.return_value = [b"hello"]

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                with patch("geoai.utils.download.requests.get", return_value=resp):
                    out = download_file(
                        "https://example.com/files/sample.tif?sig=abc&exp=999",
                        unzip=False,
                    )
                self.assertEqual(out, "sample.tif")
                self.assertTrue(os.path.isfile(os.path.join(td, "sample.tif")))
                resp.close.assert_called_once()
            finally:
                os.chdir(old_cwd)

    def test_vllm_geo_reuses_requests_session(self):
        from geoai.vllm_geo import VLLMGeo

        vlm = VLLMGeo(model_id="test-model", base_url="http://localhost:8000/v1")
        self.assertIsInstance(vlm._session, requests.Session)

        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(vlm._session, "post", return_value=fake_resp) as mock_post:
            r1 = vlm.query("first")
            r2 = vlm.query("second")

        self.assertEqual(r1["answer"], "ok")
        self.assertEqual(r2["answer"], "ok")
        self.assertEqual(mock_post.call_count, 2)


# ---------------------------------------------------------------------------
# Group 8: geoai/agents/
# ---------------------------------------------------------------------------


class TestGroup8Agents(unittest.TestCase):
    """Tests for DataTools sandboxing, STACTools ids lookup, and geo_agents helpers."""

    def test_data_tools_workspace_dir_blocks_traversal_and_symlinks(self):
        stubs = _import_agent_module_with_stubs("geoai.agents.data_tools")
        ctx = (
            patch.dict(sys.modules, stubs)
            if isinstance(stubs, dict)
            else patch.dict(sys.modules, {})
        )
        with ctx:
            sys.modules.pop("geoai.agents.data_tools", None)
            data_tools_mod = importlib.import_module("geoai.agents.data_tools")
            DataTools = data_tools_mod.DataTools

        with tempfile.TemporaryDirectory() as outer:
            workspace = os.path.join(outer, "workspace")
            os.makedirs(workspace)
            inside = os.path.join(workspace, "ok.tif")
            outside = os.path.join(outer, "secret.tif")
            _make_geotiff(inside)
            _make_geotiff(outside)

            symlink_escape = os.path.join(workspace, "escape.tif")
            os.symlink(outside, symlink_escape)

            tools = DataTools(workspace_dir=workspace)

            # Inside workspace succeeds
            info = json.loads(tools.inspect_raster(inside))
            self.assertEqual(info["width"], 32)
            listed = json.loads(tools.list_files(workspace))
            self.assertIn(
                os.path.realpath(inside), [os.path.realpath(p) for p in listed]
            )
            # Symlink pointing outside is excluded from listing
            self.assertNotIn(symlink_escape, listed)

            # Traversal, absolute outside path, and symlink escape are rejected
            for bad_path in (
                outside,
                os.path.join(workspace, "..", "secret.tif"),
                symlink_escape,
            ):
                with self.assertRaises(ValueError):
                    tools.inspect_raster(bad_path)
                with self.assertRaises(ValueError):
                    tools.inspect_vector(bad_path)
                with self.assertRaises(ValueError):
                    tools.clip_raster(inside, bad_path, -122.5, 37.7, -122.3, 37.9)
                with self.assertRaises(ValueError):
                    tools.raster_to_vector(inside, bad_path)

            with self.assertRaises(ValueError):
                tools.list_files(outer)

            # Default DataTools() stays backward-compatible (no restriction)
            unrestricted = DataTools()
            info_out = json.loads(unrestricted.inspect_raster(outside))
            self.assertEqual(info_out["width"], 32)

    def test_stac_tools_get_item_info_uses_ids_parameter(self):
        stubs = _import_agent_module_with_stubs("geoai.agents.stac_tools")
        ctx = (
            patch.dict(sys.modules, stubs)
            if isinstance(stubs, dict)
            else patch.dict(sys.modules, {})
        )
        with ctx:
            sys.modules.pop("geoai.agents.stac_tools", None)
            stac_mod = importlib.import_module("geoai.agents.stac_tools")
            tools = stac_mod.STACTools()

            fake_item = MagicMock(
                id="S2A_123",
                collection_id="sentinel-2-l2a",
                datetime="2024-01-01",
                bbox=[-1, -1, 1, 1],
                assets={},
            )
            with patch.object(
                stac_mod, "pc_stac_search", return_value=[fake_item]
            ) as mock_search:
                out = json.loads(tools.get_item_info("S2A_123", "sentinel-2-l2a"))

            self.assertEqual(out["id"], "S2A_123")
            mock_search.assert_called_once()
            self.assertEqual(mock_search.call_args.kwargs["ids"], ["S2A_123"])
            self.assertNotIn("query", mock_search.call_args.kwargs)

    def test_geo_agents_resolve_model_no_client_args_mutation_and_show_ui(self):
        stubs = _import_agent_module_with_stubs("geoai.agents.geo_agents")
        ctx = (
            patch.dict(sys.modules, stubs)
            if isinstance(stubs, dict)
            else patch.dict(sys.modules, {})
        )
        with ctx:
            for mod_name in (
                "geoai.agents.geo_agents",
                "geoai.agents.catalog_tools",
                "geoai.agents.map_tools",
                "geoai.agents.stac_tools",
            ):
                sys.modules.pop(mod_name, None)
            ga = importlib.import_module("geoai.agents.geo_agents")

            # 1. create_*_model must not mutate caller client_args dict
            caller_args = {"timeout": 30}
            ga.create_openai_model(api_key="sk-test", client_args=caller_args)
            self.assertEqual(caller_args, {"timeout": 30})

            caller_args2 = {"timeout": 30}
            ga.create_anthropic_model(api_key="ant-test", client_args=caller_args2)
            self.assertEqual(caller_args2, {"timeout": 30})

            caller_args3 = {"timeout": 30}
            ga.create_gemini_model(api_key="gem-test", client_args=caller_args3)
            self.assertEqual(caller_args3, {"timeout": 30})

            caller_args4 = {"timeout": 30}
            ga.create_minimax_model(api_key="mm-test", client_args=caller_args4)
            self.assertEqual(caller_args4, {"timeout": 30})

            caller_args5 = {"timeout": 30}
            ga.create_vllm_model(api_key="EMPTY", client_args=caller_args5)
            self.assertEqual(caller_args5, {"timeout": 30})

            # 2. _resolve_model handles all 5 model classes + strings
            for inst, expected_cls in (
                (ga.create_ollama_model(model_id="llama3.1"), ga.OllamaModel),
                (
                    ga.create_openai_model(model_id="gpt-4o", api_key="k"),
                    ga.OpenAIModel,
                ),
                (
                    ga.create_anthropic_model(model_id="claude", api_key="k"),
                    ga.AnthropicModel,
                ),
                (
                    ga.create_gemini_model(model_id="gemini", api_key="k"),
                    ga.GeminiModel,
                ),
                (
                    ga.create_bedrock_model(
                        model_id="bedrock-id", region_name="us-west-2"
                    ),
                    ga.BedrockModel,
                ),
            ):
                factory = ga._resolve_model(inst)
                built = factory()
                self.assertIsInstance(built, expected_cls)

            # Also check GeminiModel & BedrockModel work in STACAgent and CatalogAgent
            gem = ga.create_gemini_model(model_id="gemini", api_key="k")
            bed = ga.create_bedrock_model(model_id="bedrock-id")
            stac_agent = ga.STACAgent(model=gem, map_instance=MagicMock())
            self.assertIsInstance(stac_agent.model, ga.GeminiModel)
            cat_agent = ga.CatalogAgent(model=bed, catalog_df=MagicMock())
            self.assertIsInstance(cat_agent.model, ga.BedrockModel)

        # 3. Verify old_callback is assigned before try: in both show_ui() methods
        src_path = Path(__file__).resolve().parents[1] / "geoai/agents/geo_agents.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        on_send_fns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_on_send"
        ]
        self.assertEqual(len(on_send_fns), 2)
        for fn in on_send_fns:
            try_idx = next(i for i, st in enumerate(fn.body) if isinstance(st, ast.Try))
            assign_idx = next(
                (
                    i
                    for i, st in enumerate(fn.body)
                    if isinstance(st, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "old_callback"
                        for t in st.targets
                    )
                ),
                None,
            )
            self.assertIsNotNone(
                assign_idx, "old_callback must be assigned at _on_send scope"
            )
            self.assertLess(
                assign_idx,
                try_idx,
                "old_callback must be assigned BEFORE try: so finally never hits UnboundLocalError",
            )


# ---------------------------------------------------------------------------
# Group 9: geoai/pipeline.py and geoai/cli.py
# ---------------------------------------------------------------------------


class TestGroup9PipelineAndCLI(unittest.TestCase):
    """Tests for pipeline atomicity/teardown/steps and CLI overture/stac/--json."""

    def test_checkpoint_manager_save_is_atomic(self):
        from geoai.pipeline import CheckpointManager

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "state.json")
            mgr = CheckpointManager(ckpt_path, config_hash="h1")
            mgr.mark_completed("a.tif", ["s1"])
            mgr.save()

            # Simulate crash mid-dump; existing valid checkpoint must stay intact
            mgr.mark_completed("b.tif", ["s1"])
            with patch("geoai.pipeline.json.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    mgr.save()

            # No leftover temp files, original file still valid JSON with only a.tif
            self.assertEqual(os.listdir(td), ["state.json"])
            with open(ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(list(data["entries"].keys()), ["a.tif"])

    def test_pipeline_partial_setup_tears_down_already_setup_steps(self):
        from geoai.pipeline import FunctionStep, Pipeline

        events = []
        s1 = FunctionStep(
            "s1",
            fn=lambda item: item,
            setup_fn=lambda: events.append("s1.setup"),
            teardown_fn=lambda: events.append("s1.teardown"),
        )
        s2 = FunctionStep(
            "s2",
            fn=lambda item: item,
            setup_fn=lambda: (_ for _ in ()).throw(RuntimeError("s2 boom")),
            teardown_fn=lambda: events.append("s2.teardown"),
        )
        s3 = FunctionStep(
            "s3",
            fn=lambda item: item,
            setup_fn=lambda: events.append("s3.setup"),
            teardown_fn=lambda: events.append("s3.teardown"),
        )

        pipe = Pipeline(steps=[s1, s2, s3], quiet=True)
        with self.assertRaisesRegex(RuntimeError, "s2 boom"):
            pipe.run(items=[{"input_path": "x.tif"}])

        self.assertEqual(events, ["s1.setup", "s1.teardown"])

    def test_pipeline_run_parallel_checkpoints_periodically(self):
        from geoai.pipeline import CheckpointManager, FunctionStep, Pipeline

        with tempfile.TemporaryDirectory() as td:
            pipe = Pipeline(
                steps=[FunctionStep("id", lambda item: item)],
                max_workers=2,
                checkpoint_dir=td,
                quiet=True,
            )
            items = [{"input_path": f"img_{i}.tif"} for i in range(12)]
            with patch.object(
                CheckpointManager, "save", wraps=CheckpointManager.save
            ) as spy_save:
                # Patch instance method via autospec-style wrapper
                orig_save = CheckpointManager.save
                save_calls = []

                def tracked_save(self_ckpt):
                    save_calls.append(len(self_ckpt._entries))
                    return orig_save(self_ckpt)

                with patch.object(CheckpointManager, "save", tracked_save):
                    pipe.run(items=items)

            # Should have saved at 10 items (periodic) AND at end (12 items)
            self.assertIn(10, save_calls)
            self.assertGreaterEqual(len(save_calls), 2)

    def test_pipeline_builtin_onnx_canopy_clean_steps(self):
        from geoai.pipeline import (
            CanopyHeightStep,
            CleanMaskStep,
            ONNXInferenceStep,
            Pipeline,
            _STEP_REGISTRY,
            _step_from_dict,
            _step_to_dict,
        )

        for name in ("ONNXInferenceStep", "CanopyHeightStep", "CleanMaskStep"):
            self.assertIn(name, _STEP_REGISTRY)

        # Round-trip serialization
        for step in (
            ONNXInferenceStep(model_path="m.onnx", task="semantic-segmentation"),
            CanopyHeightStep(model_name="compressed_SSLhuge", device="cpu"),
            CleanMaskStep(min_island_size=50),
        ):
            d = _step_to_dict(step)
            rebuilt = _step_from_dict(d)
            self.assertIsInstance(rebuilt, type(step))

        # ONNXInferenceStep loads model ONCE in setup() across multiple items
        fake_onnx = MagicMock()
        with tempfile.TemporaryDirectory() as td:
            with patch("geoai.onnx.ONNXGeoModel", return_value=fake_onnx) as mock_cls:
                step = ONNXInferenceStep(model_path="model.onnx")
                pipe = Pipeline(steps=[step], quiet=True)
                res = pipe.run(
                    items=[
                        {"input_path": os.path.join(td, "a.tif"), "output_dir": td},
                        {"input_path": os.path.join(td, "b.tif"), "output_dir": td},
                    ]
                )
            self.assertEqual(len(res.completed), 2)
            mock_cls.assert_called_once()
            self.assertEqual(fake_onnx.predict.call_count, 2)

        # CanopyHeightStep loads model ONCE in setup() across multiple items
        fake_canopy = MagicMock()
        with tempfile.TemporaryDirectory() as td:
            with patch(
                "geoai.canopy.CanopyHeightEstimation", return_value=fake_canopy
            ) as mock_cls:
                step = CanopyHeightStep(device="cpu")
                pipe = Pipeline(steps=[step], quiet=True)
                res = pipe.run(
                    items=[
                        {"input_path": os.path.join(td, "a.tif"), "output_dir": td},
                        {"input_path": os.path.join(td, "b.tif"), "output_dir": td},
                    ]
                )
            self.assertEqual(len(res.completed), 2)
            mock_cls.assert_called_once()
            self.assertEqual(fake_canopy.predict.call_count, 2)

        # CleanMaskStep invokes clean_raster
        with tempfile.TemporaryDirectory() as td:
            with patch("geoai.tools.multiclean.clean_raster") as mock_clean:
                step = CleanMaskStep(min_island_size=25)
                out = step.process(
                    {
                        "output_path": os.path.join(td, "a_mask.tif"),
                        "output_dir": td,
                    }
                )
            mock_clean.assert_called_once()
            self.assertTrue(out["cleaned_path"].endswith("a_mask_cleaned.tif"))

    def test_cli_info_and_pipeline_run_json_and_download_sources(self):
        from geoai.cli import main

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            # 1. geoai info --json on raster
            tif = os.path.join(td, "r.tif")
            _make_geotiff(tif)
            res = runner.invoke(main, ["info", tif, "--json"])
            self.assertEqual(res.exit_code, 0, res.output)
            parsed = json.loads(res.output)
            self.assertEqual(parsed["width"], 32)
            self.assertEqual(parsed["height"], 32)

            # 2. geoai pipeline run --json
            cfg_path = os.path.join(td, "pipe.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"name": "demo", "steps": [{"type": "GlobStep", "name": "glob"}]}, f
                )
            res_pipe = runner.invoke(
                main, ["pipeline", "run", cfg_path, "-i", td, "--json"]
            )
            self.assertEqual(res_pipe.exit_code, 0, res_pipe.output)
            pipe_json = json.loads(res_pipe.output)
            self.assertEqual(pipe_json["completed"], 1)
            self.assertEqual(pipe_json["failed"], 0)

            # 3. geoai download overture
            out_ov = os.path.join(td, "buildings.geojson")
            with patch("geoai.download.download_overture_buildings") as mock_ov:
                res_ov = runner.invoke(
                    main,
                    [
                        "download",
                        "overture",
                        "--bbox",
                        "-122.5,37.7,-122.4,37.8",
                        "-o",
                        out_ov,
                        "--overture-type",
                        "building",
                    ],
                )
            self.assertEqual(res_ov.exit_code, 0, res_ov.output)
            mock_ov.assert_called_once_with(
                bbox=(-122.5, 37.7, -122.4, 37.8),
                output=out_ov,
                overture_type="building",
            )

            # 4. geoai download stac
            out_stac = os.path.join(td, "stac_out")
            fake_items = [MagicMock(id="s2_item")]
            with (
                patch(
                    "geoai.download.pc_stac_search", return_value=fake_items
                ) as mock_search,
                patch(
                    "geoai.download.pc_stac_download",
                    return_value={"s2_item": {"B04": "f.tif"}},
                ) as mock_dl,
            ):
                res_stac = runner.invoke(
                    main,
                    [
                        "download",
                        "stac",
                        "--bbox",
                        "-122.5,37.7,-122.4,37.8",
                        "-o",
                        out_stac,
                        "--collection",
                        "sentinel-2-l2a",
                        "--assets",
                        "B04,B03",
                        "--max-items",
                        "2",
                    ],
                )
            self.assertEqual(res_stac.exit_code, 0, res_stac.output)
            mock_search.assert_called_once_with(
                collection="sentinel-2-l2a",
                bbox=[-122.5, 37.7, -122.4, 37.8],
                time_range=None,
                max_items=2,
            )
            mock_dl.assert_called_once_with(
                items=fake_items,
                output_dir=out_stac,
                assets=["B04", "B03"],
            )


if __name__ == "__main__":
    unittest.main()
