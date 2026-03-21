#!/usr/bin/env python

"""Tests for `geoai.caption` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch


class TestCaptionImport(unittest.TestCase):
    """Tests for caption module import behavior."""

    def test_module_imports(self):
        """Test that the caption module can be imported."""
        import geoai.caption

        self.assertTrue(hasattr(geoai.caption, "ImageCaptioner"))

    def test_image_captioner_class_exists(self):
        """Test that ImageCaptioner class exists."""
        from geoai.caption import ImageCaptioner

        self.assertTrue(callable(ImageCaptioner))

    def test_blip_analyze_image_exists(self):
        """Test that blip_analyze_image convenience function exists."""
        from geoai.caption import blip_analyze_image

        self.assertTrue(callable(blip_analyze_image))

    def test_extract_features_from_caption_exists(self):
        """Test that extract_features_from_caption function exists."""
        from geoai.caption import extract_features_from_caption

        self.assertTrue(callable(extract_features_from_caption))

    def test_load_image_exists(self):
        """Test that load_image helper function exists."""
        from geoai.caption import load_image

        self.assertTrue(callable(load_image))

    def test_ensure_spacy_model_exists(self):
        """Test that ensure_spacy_model function exists."""
        from geoai.caption import ensure_spacy_model

        self.assertTrue(callable(ensure_spacy_model))

    def test_constants_defined(self):
        """Test that module constants are defined."""
        from geoai.caption import (
            AERIAL_FEATURES_URL,
            DEFAULT_BLIP_MODEL,
            DEFAULT_SPACY_MODEL,
        )

        self.assertIsInstance(AERIAL_FEATURES_URL, str)
        self.assertIsInstance(DEFAULT_BLIP_MODEL, str)
        self.assertIsInstance(DEFAULT_SPACY_MODEL, str)

    def test_no_http_call_on_import(self):
        """Test that importing the module does not trigger HTTP requests."""
        import importlib

        with patch("geoai.caption.requests.get") as mock_get:
            importlib.reload(__import__("geoai.caption"))
            mock_get.assert_not_called()


class TestCaptionSignatures(unittest.TestCase):
    """Tests for caption class and function signatures."""

    def test_image_captioner_init_params(self):
        """Test ImageCaptioner.__init__ has expected parameters."""
        from geoai.caption import ImageCaptioner

        sig = inspect.signature(ImageCaptioner.__init__)
        self.assertIn("blip_model_name", sig.parameters)
        self.assertIn("spacy_model_name", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("auto_download", sig.parameters)

    def test_generate_caption_params(self):
        """Test generate_caption method has expected parameters."""
        from geoai.caption import ImageCaptioner

        sig = inspect.signature(ImageCaptioner.generate_caption)
        self.assertIn("image_source", sig.parameters)

    def test_extract_features_params(self):
        """Test extract_features method has expected parameters."""
        from geoai.caption import ImageCaptioner

        sig = inspect.signature(ImageCaptioner.extract_features)
        self.assertIn("caption", sig.parameters)
        self.assertIn("include_features", sig.parameters)
        self.assertIn("exclude_features", sig.parameters)

    def test_analyze_params(self):
        """Test analyze method has expected parameters."""
        from geoai.caption import ImageCaptioner

        sig = inspect.signature(ImageCaptioner.analyze)
        self.assertIn("image_source", sig.parameters)
        self.assertIn("include_features", sig.parameters)
        self.assertIn("exclude_features", sig.parameters)

    def test_blip_analyze_image_params(self):
        """Test blip_analyze_image function has expected parameters."""
        from geoai.caption import blip_analyze_image

        sig = inspect.signature(blip_analyze_image)
        self.assertIn("image_source", sig.parameters)
        self.assertIn("include_features", sig.parameters)
        self.assertIn("exclude_features", sig.parameters)
        self.assertIn("blip_model_name", sig.parameters)
        self.assertIn("spacy_model_name", sig.parameters)

    def test_load_image_params(self):
        """Test load_image function has expected parameters."""
        from geoai.caption import load_image

        sig = inspect.signature(load_image)
        self.assertIn("source", sig.parameters)


class TestEnsureSpacyModel(unittest.TestCase):
    """Tests for ensure_spacy_model function."""

    def test_existing_model_no_download(self):
        """Test that no download occurs when the model is already installed."""
        from geoai.caption import ensure_spacy_model

        with patch("geoai.caption.download") as mock_download:
            with patch("geoai.caption.importlib.import_module"):
                ensure_spacy_model("en_core_web_sm")
            mock_download.assert_not_called()

    def test_missing_model_auto_download_false_raises(self):
        """Test that RuntimeError is raised when model is missing and auto_download=False."""
        from geoai.caption import ensure_spacy_model

        with patch("geoai.caption.importlib.import_module", side_effect=ImportError):
            with self.assertRaises(RuntimeError):
                ensure_spacy_model("nonexistent_model", auto_download=False)

    def test_missing_model_auto_download_true(self):
        """Test that download is attempted when model is missing and auto_download=True."""
        from geoai.caption import ensure_spacy_model

        with patch("geoai.caption.download") as mock_download:
            with patch(
                "geoai.caption.importlib.import_module", side_effect=ImportError
            ):
                ensure_spacy_model("fake_model", auto_download=True)
            mock_download.assert_called_once_with("fake_model")


class TestLoadAerialFeatureVocab(unittest.TestCase):
    """Tests for load_aerial_feature_vocab function."""

    def setUp(self):
        """Reset the vocab cache before each test."""
        import geoai.caption

        geoai.caption._aerial_vocab_cache = None

    def test_returns_sorted_list(self):
        """Test that the function returns a sorted list of strings."""
        mock_json = {
            "structures": ["building", "bridge"],
            "vehicles": {"land": ["car", "truck"], "water": ["boat"]},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json

        from geoai.caption import load_aerial_feature_vocab

        with patch("geoai.caption.requests.get", return_value=mock_resp):
            result = load_aerial_feature_vocab()

        self.assertEqual(result, ["boat", "bridge", "building", "car", "truck"])

    def test_caches_result(self):
        """Test that repeated calls use the cache."""
        mock_json = {"items": ["tree"]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json

        from geoai.caption import load_aerial_feature_vocab

        with patch("geoai.caption.requests.get", return_value=mock_resp) as mock_get:
            load_aerial_feature_vocab()
            load_aerial_feature_vocab()
            mock_get.assert_called_once()

    def test_uses_timeout(self):
        """Test that requests.get is called with a timeout."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": ["tree"]}

        from geoai.caption import load_aerial_feature_vocab

        with patch("geoai.caption.requests.get", return_value=mock_resp) as mock_get:
            load_aerial_feature_vocab()
            call_kwargs = mock_get.call_args[1]
            self.assertIn("timeout", call_kwargs)


class TestLoadImage(unittest.TestCase):
    """Tests for load_image function."""

    def test_pil_image_passthrough(self):
        """Test that PIL images are returned as-is (converted to RGB)."""
        from PIL import Image

        from geoai.caption import load_image

        img = Image.new("RGBA", (10, 10), color="red")
        result = load_image(img)
        self.assertEqual(result.mode, "RGB")

    def test_unsupported_type_raises(self):
        """Test that unsupported types raise TypeError."""
        from geoai.caption import load_image

        with self.assertRaises(TypeError):
            load_image(12345)

    def test_url_uses_timeout(self):
        """Test that URL loading uses a timeout."""
        from PIL import Image

        from geoai.caption import load_image

        fake_img = Image.new("RGB", (10, 10))
        import io

        buf = io.BytesIO()
        fake_img.save(buf, format="PNG")
        buf.seek(0)

        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()

        with patch("geoai.caption.requests.get", return_value=mock_resp) as mock_get:
            load_image("https://example.com/test.png")
            call_kwargs = mock_get.call_args[1]
            self.assertIn("timeout", call_kwargs)


def _spacy_available():
    """Check if en_core_web_sm is installed."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False


@unittest.skipUnless(_spacy_available(), "en_core_web_sm not installed")
class TestExtractFeatures(unittest.TestCase):
    """Tests for feature extraction logic with a mocked captioner."""

    @classmethod
    def setUpClass(cls):
        """Create a captioner with mocked BLIP model."""
        import spacy

        from geoai.caption import ImageCaptioner

        with (patch.object(ImageCaptioner, "__init__", lambda self, **kw: None),):
            cls.captioner = ImageCaptioner()
            cls.captioner.nlp = spacy.load("en_core_web_sm")

    def test_no_include_returns_all_nouns(self):
        """Test that None include_features returns all non-excluded nouns."""
        result = self.captioner.extract_features(
            "a building with a parking lot and trees"
        )
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertTrue(all(isinstance(f, str) for f in result))

    def test_custom_include_filters(self):
        """Test that custom include_features filters to only those terms."""
        result = self.captioner.extract_features(
            "a building with cars and trees",
            include_features=["building", "car"],
        )
        for feat in result:
            self.assertIn(feat, ["building", "car"])

    def test_exclude_features(self):
        """Test that exclude_features removes specified nouns."""
        result = self.captioner.extract_features(
            "a building with a parking lot",
            exclude_features=["building"],
        )
        self.assertNotIn("building", result)

    def test_returns_sorted(self):
        """Test that results are sorted."""
        result = self.captioner.extract_features("trees and buildings and roads")
        self.assertEqual(result, sorted(result))

    def test_default_vocab_with_mock(self):
        """Test include_features='default' uses aerial vocabulary."""
        mock_maps = (
            {"building": "building", "car": "car"},
            {"building": "building", "car": "car"},
            {},
        )
        with patch("geoai.caption._get_aerial_feature_maps", return_value=mock_maps):
            result = self.captioner.extract_features(
                "a building with several cars",
                include_features="default",
            )
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
