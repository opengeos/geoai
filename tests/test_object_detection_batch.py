"""Tests for object_detection_batch input parsing and filename generation."""

import os
import glob
from unittest.mock import patch, MagicMock

import pytest

from geoai.train import object_detection_batch


@pytest.fixture
def mock_model():
    """Create a mock model that passes through load_state_dict and eval."""
    model = MagicMock()
    model.to.return_value = model
    return model


@pytest.fixture
def mock_deps(mock_model, tmp_path):
    """Patch torch.load and inference_on_geotiff to isolate input parsing."""
    real_exists = os.path.exists

    def _exists(path):
        if path == "dummy":
            return True
        return real_exists(path)

    with (
        patch("geoai.train.torch.load", return_value={}),
        patch("geoai.train.inference_on_geotiff") as mock_infer,
        patch("geoai.train.get_device", return_value="cpu"),
        patch("geoai.train.os.path.exists", side_effect=_exists),
    ):
        yield mock_infer, tmp_path


class TestInputParsing:
    """Test that object_detection_batch resolves files correctly."""

    def test_list_input(self, mock_deps, mock_model):
        """A list of paths should be used as-is."""
        mock_infer, tmp_path = mock_deps
        input_files = ["/data/a.tif", "/data/b.tiff"]

        object_detection_batch(
            input_paths=input_files,
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        assert mock_infer.call_count == 2
        called_paths = [
            call.kwargs["geotiff_path"] for call in mock_infer.call_args_list
        ]
        assert called_paths == input_files

    def test_tiff_single_file(self, mock_deps, mock_model):
        """A single .tiff path should be treated as a file, not a directory."""
        mock_infer, tmp_path = mock_deps

        object_detection_batch(
            input_paths="/data/image.tiff",
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        assert mock_infer.call_count == 1
        assert mock_infer.call_args.kwargs["geotiff_path"] == "/data/image.tiff"

    def test_tif_single_file(self, mock_deps, mock_model):
        """A single .tif path should be treated as a file."""
        mock_infer, tmp_path = mock_deps

        object_detection_batch(
            input_paths="/data/image.tif",
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        assert mock_infer.call_count == 1
        assert mock_infer.call_args.kwargs["geotiff_path"] == "/data/image.tif"

    def test_case_insensitive_extension(self, mock_deps, mock_model):
        """Uppercase extensions like .TIF should be treated as files."""
        mock_infer, tmp_path = mock_deps

        object_detection_batch(
            input_paths="/data/image.TIF",
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        assert mock_infer.call_count == 1
        assert mock_infer.call_args.kwargs["geotiff_path"] == "/data/image.TIF"

    def test_directory_input(self, mock_deps, mock_model):
        """A directory path should glob for .tif and .tiff files."""
        mock_infer, tmp_path = mock_deps

        with patch("geoai.train.glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern: {
                os.path.join("/data", "*.tif"): ["/data/a.tif", "/data/c.tif"],
                os.path.join("/data", "*.tiff"): ["/data/b.tiff"],
            }.get(pattern, [])

            object_detection_batch(
                input_paths="/data",
                output_dir=str(tmp_path),
                model_path="dummy",
                model=mock_model,
            )

        assert mock_infer.call_count == 3
        called_paths = [
            call.kwargs["geotiff_path"] for call in mock_infer.call_args_list
        ]
        assert called_paths == ["/data/a.tif", "/data/b.tiff", "/data/c.tif"]


class TestFilenameGeneration:
    """Test that default output filenames are generated correctly."""

    def test_tif_output_name(self, mock_deps, mock_model):
        """A .tif input should produce <name>_mask.tif output."""
        mock_infer, tmp_path = mock_deps

        object_detection_batch(
            input_paths="/data/image.tif",
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        expected = os.path.join(str(tmp_path), "image_mask.tif")
        assert mock_infer.call_args.kwargs["output_path"] == expected

    def test_tiff_output_name(self, mock_deps, mock_model):
        """A .tiff input should produce <name>_mask.tif, not <name>_mask.tiff."""
        mock_infer, tmp_path = mock_deps

        object_detection_batch(
            input_paths="/data/image.tiff",
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        expected = os.path.join(str(tmp_path), "image_mask.tif")
        assert mock_infer.call_args.kwargs["output_path"] == expected

    def test_uppercase_tif_output_name(self, mock_deps, mock_model):
        """A .TIF input should produce <name>_mask.tif output."""
        mock_infer, tmp_path = mock_deps

        object_detection_batch(
            input_paths="/data/image.TIF",
            output_dir=str(tmp_path),
            model_path="dummy",
            model=mock_model,
        )

        expected = os.path.join(str(tmp_path), "image_mask.tif")
        assert mock_infer.call_args.kwargs["output_path"] == expected
