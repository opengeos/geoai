#!/usr/bin/env python

"""Tests for `geoai.vllm_geo` module."""

import inspect
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_geotiff(path, width=64, height=64, bands=3, dtype="uint8"):
    """Create a minimal GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.randint(0, 256, (bands, height, width), dtype=np.uint8)
    if dtype != "uint8":
        data = data.astype(dtype)
    transform = from_bounds(-122.5, 37.7, -122.3, 37.9, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return data


def _make_processor():
    """Create a VLLMGeo in server mode (no network calls at init)."""
    from geoai.vllm_geo import VLLMGeo

    return VLLMGeo(model_id="test-model", base_url="http://localhost:8000/v1")


def _mock_server_response(content):
    """Build a mock requests response with the given message content."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestVLLMGeoImport(unittest.TestCase):
    def test_module_imports(self):
        import geoai.vllm_geo

        self.assertTrue(hasattr(geoai.vllm_geo, "VLLMGeo"))

    def test_vllm_geo_is_class(self):
        from geoai.vllm_geo import VLLMGeo

        self.assertTrue(inspect.isclass(VLLMGeo))

    def test_convenience_functions_exist(self):
        from geoai import vllm_geo

        for name in ("vllm_caption", "vllm_query", "vllm_detect"):
            self.assertTrue(callable(getattr(vllm_geo, name)))

    def test_check_vllm_available_returns_bool(self):
        from geoai.vllm_geo import check_vllm_available

        self.assertIsInstance(check_vllm_available(), bool)

    def test_lazy_top_level_exports(self):
        import geoai

        self.assertTrue(inspect.isclass(geoai.VLLMGeo))
        self.assertTrue(callable(geoai.check_vllm_available))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestVLLMGeoInit(unittest.TestCase):
    def test_server_mode_defaults(self):
        proc = _make_processor()
        self.assertFalse(proc.offline)
        self.assertEqual(proc.base_url, "http://localhost:8000/v1")
        self.assertIsNone(proc._llm)

    def test_base_url_trailing_slash_stripped(self):
        from geoai.vllm_geo import VLLMGeo

        proc = VLLMGeo(base_url="http://localhost:8000/v1/")
        self.assertEqual(proc.base_url, "http://localhost:8000/v1")

    def test_offline_without_vllm_raises(self):
        from geoai.vllm_geo import VLLMGeo

        with patch("geoai.vllm_geo.check_vllm_available", return_value=False):
            with self.assertRaises(ImportError):
                VLLMGeo(offline=True)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


class TestImageLoading(unittest.TestCase):
    def test_load_pil_image(self):
        proc = _make_processor()
        img = Image.new("RGB", (32, 32))
        image, metadata = proc.load_image(img)
        self.assertIs(image, img)
        self.assertIsNone(metadata)

    def test_load_numpy_hwc(self):
        proc = _make_processor()
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        image, metadata = proc.load_image(arr)
        self.assertEqual(image.size, (32, 32))
        self.assertIsNone(metadata)

    def test_load_numpy_grayscale(self):
        proc = _make_processor()
        arr = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        image, _ = proc.load_image(arr)
        self.assertEqual(image.mode, "RGB")

    def test_load_geotiff(self):
        proc = _make_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")
            _create_test_geotiff(path)
            image, metadata = proc.load_image(path)
            self.assertEqual(image.size, (64, 64))
            self.assertIsNotNone(metadata["crs"])
            self.assertIsNotNone(metadata["transform"])

    def test_load_missing_file_raises(self):
        proc = _make_processor()
        with self.assertRaises(FileNotFoundError):
            proc.load_geotiff("/nonexistent/file.tif")

    def test_load_unsupported_type_raises(self):
        proc = _make_processor()
        with self.assertRaises(TypeError):
            proc.load_image(12345)

    def test_normalize_uint16(self):
        proc = _make_processor()
        arr = np.random.randint(0, 65535, (3, 16, 16)).astype(np.uint16)
        result = proc._normalize_image(arr)
        self.assertEqual(result.dtype, np.uint8)

    def test_encode_image_base64(self):
        proc = _make_processor()
        img = Image.new("RGB", (8, 8))
        uri = proc._encode_image_base64(img)
        self.assertTrue(uri.startswith("data:image/png;base64,"))


# ---------------------------------------------------------------------------
# Server-mode inference (mocked)
# ---------------------------------------------------------------------------


class TestServerInference(unittest.TestCase):
    @patch("geoai.vllm_geo.requests.post")
    def test_caption(self, mock_post):
        mock_post.return_value = _mock_server_response("A satellite image.")
        proc = _make_processor()
        result = proc.caption(Image.new("RGB", (32, 32)))
        self.assertEqual(result["caption"], "A satellite image.")
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "test-model")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")

    @patch("geoai.vllm_geo.requests.post")
    def test_query_with_image(self, mock_post):
        mock_post.return_value = _mock_server_response("Yes, there is water.")
        proc = _make_processor()
        result = proc.query("Is there water?", Image.new("RGB", (32, 32)))
        self.assertEqual(result["answer"], "Yes, there is water.")

    @patch("geoai.vllm_geo.requests.post")
    def test_query_text_only(self, mock_post):
        mock_post.return_value = _mock_server_response("Hello.")
        proc = _make_processor()
        result = proc.query("Say hello")
        self.assertEqual(result["answer"], "Hello.")
        content = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIsInstance(content, str)

    @patch("geoai.vllm_geo.requests.post")
    def test_server_error_raises_runtime_error(self, mock_post):
        import requests as requests_lib

        mock_post.side_effect = requests_lib.ConnectionError("refused")
        proc = _make_processor()
        with self.assertRaises(RuntimeError):
            proc.query("test")

    @patch("geoai.vllm_geo.requests.post")
    def test_malformed_response_raises_runtime_error(self, mock_post):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"unexpected": True}
        mock_post.return_value = response
        proc = _make_processor()
        with self.assertRaises(RuntimeError):
            proc.query("test")


# ---------------------------------------------------------------------------
# Detection parsing and georeferencing
# ---------------------------------------------------------------------------


class TestDetection(unittest.TestCase):
    def test_parse_valid_json(self):
        proc = _make_processor()
        text = json.dumps([{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}])
        detections = proc._parse_detections(text)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["x_min"], 0.1)

    def test_parse_json_with_surrounding_text(self):
        proc = _make_processor()
        text = (
            'Here are the boxes: [{"x_min": 0.1, "y_min": 0.2, '
            '"x_max": 0.3, "y_max": 0.4}] as requested.'
        )
        self.assertEqual(len(proc._parse_detections(text)), 1)

    def test_parse_empty_array(self):
        proc = _make_processor()
        self.assertEqual(proc._parse_detections("[]"), [])

    def test_parse_no_json(self):
        proc = _make_processor()
        self.assertEqual(proc._parse_detections("no objects found"), [])

    def test_parse_invalid_json(self):
        proc = _make_processor()
        self.assertEqual(proc._parse_detections("[{broken"), [])

    def test_parse_drops_invalid_entries(self):
        proc = _make_processor()
        text = json.dumps(
            [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                {"x_min": 0.5},  # missing keys
                {"x_min": 0.9, "y_min": 0.9, "x_max": 0.1, "y_max": 0.1},  # degenerate
            ]
        )
        self.assertEqual(len(proc._parse_detections(text)), 1)

    def test_parse_clamps_out_of_range(self):
        proc = _make_processor()
        text = json.dumps([{"x_min": -0.5, "y_min": 0.0, "x_max": 1.5, "y_max": 0.9}])
        det = proc._parse_detections(text)[0]
        self.assertEqual(det["x_min"], 0.0)
        self.assertEqual(det["x_max"], 1.0)

    @patch("geoai.vllm_geo.requests.post")
    def test_detect_geotiff_returns_gdf(self, mock_post):
        mock_post.return_value = _mock_server_response(
            '[{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]'
        )
        proc = _make_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")
            _create_test_geotiff(path)
            result = proc.detect(path, "building")
            self.assertEqual(len(result["objects"]), 1)
            self.assertIn("gdf", result)
            self.assertEqual(len(result["gdf"]), 1)

    @patch("geoai.vllm_geo.requests.post")
    def test_detect_saves_geojson(self, mock_post):
        mock_post.return_value = _mock_server_response(
            '[{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]'
        )
        proc = _make_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")
            out = os.path.join(tmpdir, "out.geojson")
            _create_test_geotiff(path)
            proc.detect(path, "building", output_path=out)
            self.assertTrue(os.path.exists(out))

    def test_georef_empty_detections(self):
        proc = _make_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")
            _create_test_geotiff(path)
            _, metadata = proc.load_image(path)
            result = proc._georef_detections({"objects": []}, metadata)
            self.assertEqual(len(result["gdf"]), 0)


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------


class TestSlidingWindows(unittest.TestCase):
    def test_create_windows_small_image(self):
        proc = _make_processor()
        windows = proc._create_sliding_windows(512, 512, window_size=512)
        self.assertEqual(windows, [(0, 0, 512, 512)])

    def test_create_windows_large_image(self):
        proc = _make_processor()
        windows = proc._create_sliding_windows(1024, 1024, window_size=512, overlap=64)
        self.assertGreater(len(windows), 1)
        for x0, y0, x1, y1 in windows:
            self.assertLessEqual(x1, 1024)
            self.assertLessEqual(y1, 1024)

    def test_nms_removes_duplicates(self):
        proc = _make_processor()
        detections = [
            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.3},
            {"x_min": 0.11, "y_min": 0.11, "x_max": 0.31, "y_max": 0.31},
            {"x_min": 0.7, "y_min": 0.7, "x_max": 0.9, "y_max": 0.9},
        ]
        kept = proc._apply_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)

    def test_nms_empty(self):
        proc = _make_processor()
        self.assertEqual(proc._apply_nms([]), [])

    @patch("geoai.vllm_geo.requests.post")
    def test_query_sliding_window_small_image_single_call(self, mock_post):
        mock_post.return_value = _mock_server_response("answer")
        proc = _make_processor()
        result = proc.query_sliding_window(
            "What is this?", Image.new("RGB", (64, 64)), window_size=512
        )
        self.assertEqual(result["answer"], "answer")
        self.assertEqual(mock_post.call_count, 1)

    @patch("geoai.vllm_geo.requests.post")
    def test_caption_sliding_window_tiles(self, mock_post):
        mock_post.return_value = _mock_server_response("tile caption")
        proc = _make_processor()
        result = proc.caption_sliding_window(
            Image.new("RGB", (1024, 1024)),
            window_size=512,
            overlap=64,
            show_progress=False,
        )
        self.assertGreater(len(result["tile_captions"]), 1)
        self.assertIn("tile caption", result["caption"])

    @patch("geoai.vllm_geo.requests.post")
    def test_detect_sliding_window_merges(self, mock_post):
        mock_post.return_value = _mock_server_response(
            '[{"x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.6}]'
        )
        proc = _make_processor()
        result = proc.detect_sliding_window(
            Image.new("RGB", (1024, 1024)),
            "car",
            window_size=512,
            overlap=64,
            show_progress=False,
        )
        self.assertIn("objects", result)
        self.assertGreater(len(result["objects"]), 0)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions(unittest.TestCase):
    @patch("geoai.vllm_geo.requests.post")
    def test_vllm_caption(self, mock_post):
        from geoai.vllm_geo import vllm_caption

        mock_post.return_value = _mock_server_response("A field.")
        self.assertEqual(vllm_caption(Image.new("RGB", (32, 32))), "A field.")

    @patch("geoai.vllm_geo.requests.post")
    def test_vllm_query(self, mock_post):
        from geoai.vllm_geo import vllm_query

        mock_post.return_value = _mock_server_response("42")
        self.assertEqual(vllm_query("How many?", Image.new("RGB", (32, 32))), "42")

    @patch("geoai.vllm_geo.requests.post")
    def test_vllm_detect(self, mock_post):
        from geoai.vllm_geo import vllm_detect

        mock_post.return_value = _mock_server_response("[]")
        result = vllm_detect(Image.new("RGB", (32, 32)), "car")
        self.assertEqual(result["objects"], [])


# ---------------------------------------------------------------------------
# Agent model factory (Layer 1)
# ---------------------------------------------------------------------------


class TestCreateVllmModel(unittest.TestCase):
    def setUp(self):
        try:
            import strands  # noqa: F401
        except ImportError:
            self.skipTest("strands-agents not installed")

    def test_create_vllm_model_defaults(self):
        from geoai.agents.geo_agents import create_vllm_model

        model = create_vllm_model()
        self.assertEqual(model.config["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(model.client_args["base_url"], "http://localhost:8000/v1")
        self.assertEqual(model.client_args["api_key"], "EMPTY")

    def test_create_vllm_model_custom(self):
        from geoai.agents.geo_agents import create_vllm_model

        model = create_vllm_model(
            base_url="http://gpu-server:8000/v1",
            model_id="Qwen/Qwen2-VL-7B-Instruct",
            api_key="secret",
        )
        self.assertEqual(model.config["model_id"], "Qwen/Qwen2-VL-7B-Instruct")
        self.assertEqual(model.client_args["base_url"], "http://gpu-server:8000/v1")
        self.assertEqual(model.client_args["api_key"], "secret")

    def test_client_args_take_precedence(self):
        from geoai.agents.geo_agents import create_vllm_model

        model = create_vllm_model(
            api_key="ignored", client_args={"api_key": "explicit"}
        )
        self.assertEqual(model.client_args["api_key"], "explicit")


if __name__ == "__main__":
    unittest.main()
