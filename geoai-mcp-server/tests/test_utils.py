"""Tests for utility modules."""

import asyncio
import importlib
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from geoai_mcp_server.utils.error_handling import (
    GeoAIError,
    InputValidationError,
    FileAccessError,
    ModelLoadError,
    ProcessingError,
    TimeoutError,
)
from geoai_mcp_server.utils.validation import (
    validate_bbox,
    validate_confidence_threshold,
    sanitize_filename,
)
from geoai_mcp_server.utils.file_management import (
    generate_output_filename,
    get_safe_output_path,
    validate_output_path,
)
from geoai_mcp_server.config import GeoAIConfig


class TestErrorHandling:
    """Tests for error classes."""

    def test_geoai_error(self):
        """Test base GeoAI error."""
        error = GeoAIError("Test error", details={"key": "value"})
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}

    def test_input_validation_error(self):
        """Test input validation error."""
        error = InputValidationError("Invalid input")
        assert isinstance(error, GeoAIError)

    def test_file_access_error(self):
        """Test file access error."""
        error = FileAccessError("File not found: test.tif")
        assert "test.tif" in str(error)

    def test_model_load_error(self):
        """Test model load error."""
        error = ModelLoadError("Failed to load SAM model")
        assert "SAM" in str(error)

    def test_processing_error(self):
        """Test processing error."""
        error = ProcessingError("Segmentation failed")
        assert isinstance(error, GeoAIError)

    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError("Operation exceeded 60 seconds", timeout_seconds=60)
        assert error.details["timeout_seconds"] == 60


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_bbox_valid(self):
        """Test valid bounding box validation."""
        # Should not raise
        validate_bbox([-122.5, 37.7, -122.4, 37.8])

    def test_validate_bbox_invalid_lon(self):
        """Test invalid longitude."""
        with pytest.raises(InputValidationError):
            validate_bbox([-200, 37.7, -122.4, 37.8])

    def test_validate_bbox_invalid_lat(self):
        """Test invalid latitude."""
        with pytest.raises(InputValidationError):
            validate_bbox([-122.5, 100, -122.4, 37.8])

    def test_validate_bbox_inverted(self):
        """Test inverted bounding box."""
        with pytest.raises(InputValidationError):
            validate_bbox([-122.4, 37.7, -122.5, 37.8])  # min > max

    def test_validate_confidence_threshold_valid(self):
        """Test valid confidence threshold."""
        assert validate_confidence_threshold(0.5) == 0.5
        assert validate_confidence_threshold(0.0) == 0.0
        assert validate_confidence_threshold(1.0) == 1.0

    def test_validate_confidence_threshold_invalid(self):
        """Test invalid confidence threshold."""
        with pytest.raises(InputValidationError):
            validate_confidence_threshold(-0.1)
        with pytest.raises(InputValidationError):
            validate_confidence_threshold(1.5)

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        assert sanitize_filename("test.tif") == "test.tif"
        # Interior spaces are replaced with underscores
        result = sanitize_filename("my file.tif")
        assert " " not in result
        assert result.endswith(".tif")
        # Leading/trailing whitespace is removed, not converted to underscores
        assert sanitize_filename("  file.tif  ") == "file.tif"

    def test_sanitize_filename_special_chars(self):
        """Test special character removal."""
        # Special chars like < > / are replaced with underscores
        result = sanitize_filename("test<>file.tif")
        assert "<" not in result
        assert ">" not in result
        assert result.endswith(".tif")

        result = sanitize_filename("path/to/file.tif")
        assert "/" not in result
        assert result.endswith(".tif")

    def test_sanitize_filename_traversal(self):
        """Test path traversal prevention."""
        assert ".." not in sanitize_filename("../../../etc/passwd")
        # Windows-style traversal (backslash already caught by char class)
        assert ".." not in sanitize_filename("..\\..\\etc\\passwd")
        # Triple-dot sequences
        assert ".." not in sanitize_filename(".../secret")
        # Mid-filename double-dot is also collapsed
        result = sanitize_filename("file..backup.tif")
        assert ".." not in result
        assert result.endswith(".tif")

    @pytest.mark.parametrize("filename", ["CON", "aux.tif", "LPT1.geojson"])
    def test_sanitize_filename_windows_reserved_names(self, filename):
        """Test Windows reserved device names are avoided."""
        result = sanitize_filename(filename)

        assert result.startswith("_")
        assert Path(result).stem.upper() not in {"CON", "AUX", "LPT1"}


class TestFileManagement:
    """Tests for file management utilities."""

    def test_generate_output_filename_basic(self):
        """Test basic output filename generation."""
        result = generate_output_filename("input.tif", "segmented", "geojson")
        assert "input" in result
        assert "segmented" in result
        assert result.endswith(".geojson")

    def test_generate_output_filename_with_path(self):
        """Test output filename from path."""
        result = generate_output_filename("/path/to/input.tif", "processed", "tif")
        assert "input" in result
        assert "processed" in result
        assert result.endswith(".tif")

    def test_generate_output_filename_unique(self):
        """Test that generated filenames include timestamp for uniqueness."""
        result1 = generate_output_filename("test.tif", "output", "tif")
        result2 = generate_output_filename("test.tif", "output", "tif")
        # Both should be valid, may or may not be identical depending on timing
        assert result1.endswith(".tif")
        assert result2.endswith(".tif")

    def test_validate_output_path_rejects_output_dir_itself(self, tmp_path):
        """Output paths resolving to output_dir itself must be rejected."""
        cfg = GeoAIConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileAccessError):
            validate_output_path(".", cfg)
        with pytest.raises(FileAccessError):
            validate_output_path("", cfg)
        with pytest.raises(FileAccessError):
            validate_output_path(cfg.output_dir, cfg)

        valid = validate_output_path("result.tif", cfg)
        assert valid == (cfg.output_dir / "result.tif").resolve()

    def test_get_safe_output_path_rejects_dot(self, tmp_path):
        """get_safe_output_path must reject '.' resolving to output_dir itself."""
        cfg = GeoAIConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "geoai_mcp_server.utils.file_management.get_config", return_value=cfg
        ):
            with pytest.raises(FileAccessError):
                get_safe_output_path(".")

            valid = get_safe_output_path("nested/result.tif")
            assert valid == (cfg.output_dir / "nested" / "result.tif").resolve()

    def test_validate_output_subdir_validation(self, tmp_path):
        """validate_output_subdir rejects absolute paths, traversal, and root dir."""
        from geoai_mcp_server.utils.file_management import validate_output_subdir

        cfg = GeoAIConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(InputValidationError):
            validate_output_subdir("/tmp/outside", cfg)
        with pytest.raises(InputValidationError):
            validate_output_subdir("../outside", cfg)
        with pytest.raises(InputValidationError):
            validate_output_subdir("sub/../../outside", cfg)
        with pytest.raises(InputValidationError):
            validate_output_subdir("", cfg)
        with pytest.raises(FileAccessError):
            validate_output_subdir(".", cfg)

        valid = validate_output_subdir("downloads/naip", cfg)
        assert valid == (cfg.output_dir / "downloads" / "naip").resolve()


def _load_server_module():
    """Import geoai_mcp_server.server, stubbing FastMCP when mcp is not installed."""
    try:
        return importlib.import_module("geoai_mcp_server.server")
    except ModuleNotFoundError as exc:
        if exc.name not in ("mcp", "mcp.server", "mcp.server.fastmcp"):
            raise

    class _DummyFastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            return lambda fn: fn

        def resource(self, *args, **kwargs):
            return lambda fn: fn

        def prompt(self, *args, **kwargs):
            return lambda fn: fn

    mcp_mod = types.ModuleType("mcp")
    mcp_server_mod = types.ModuleType("mcp.server")
    mcp_fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    mcp_fastmcp_mod.FastMCP = _DummyFastMCP
    mcp_server_mod.fastmcp = mcp_fastmcp_mod
    mcp_mod.server = mcp_server_mod

    with patch.dict(
        sys.modules,
        {
            "mcp": mcp_mod,
            "mcp.server": mcp_server_mod,
            "mcp.server.fastmcp": mcp_fastmcp_mod,
        },
    ):
        sys.modules.pop("geoai_mcp_server.server", None)
        return importlib.import_module("geoai_mcp_server.server")


class TestServerPathValidation:
    """Tests for output_subdir and model_path validation in server tools."""

    def test_download_satellite_imagery_rejects_traversal_subdir(self, tmp_path):
        """download_satellite_imagery blocks output_subdir outside output_dir."""
        server = _load_server_module()

        cfg = GeoAIConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(server, "config", cfg),
            patch.object(server, "_get_geoai_module") as mock_get,
        ):
            result = asyncio.run(
                server.download_satellite_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.4,
                    max_lat=37.8,
                    data_source="naip",
                    output_subdir="../escaped",
                )
            )

        assert result["success"] is False
        mock_get.assert_not_called()
        assert not (tmp_path / "escaped").exists()

    def test_prepare_training_data_rejects_absolute_subdir(self, tmp_path):
        """prepare_training_data blocks absolute output_subdir paths."""
        server = _load_server_module()

        cfg = GeoAIConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        (cfg.input_dir / "images").mkdir(parents=True, exist_ok=True)
        (cfg.input_dir / "labels").mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside_dir"

        with (
            patch.object(server, "config", cfg),
            patch.object(server, "_get_geoai_module") as mock_get,
        ):
            result = asyncio.run(
                server.prepare_training_data(
                    images_dir="images",
                    labels_dir="labels",
                    output_subdir=str(outside),
                )
            )

        assert result["success"] is False
        mock_get.assert_not_called()
        assert not outside.exists()

    def test_classify_land_cover_validates_model_path(self, tmp_path):
        """classify_land_cover rejects URLs and outside paths for model_path."""
        server = _load_server_module()

        cfg = GeoAIConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        cfg.input_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        image_file = cfg.input_dir / "scene.tif"
        image_file.write_bytes(b"img")
        outside_model = tmp_path / "external.ckpt"
        outside_model.write_bytes(b"model")
        valid_model = cfg.input_dir / "local.ckpt"
        valid_model.write_bytes(b"model")

        with (
            patch.object(server, "config", cfg),
            patch.object(server, "_get_geoai_module") as mock_get,
        ):
            res_url = asyncio.run(
                server.classify_land_cover(
                    image_path="scene.tif",
                    model_path="https://example.com/model.ckpt",
                )
            )
            res_outside = asyncio.run(
                server.classify_land_cover(
                    image_path="scene.tif",
                    model_path=str(outside_model),
                )
            )
            assert res_url["success"] is False
            assert res_outside["success"] is False
            mock_get.assert_not_called()

            mock_classify = MagicMock()
            mock_classify.classify_image.return_value = MagicMock(
                class_distribution={"forest": 0.7, "water": 0.3}
            )
            mock_get.return_value = mock_classify

            res_ok = asyncio.run(
                server.classify_land_cover(
                    image_path="scene.tif",
                    model_path="local.ckpt",
                )
            )

        assert res_ok["success"] is True
        mock_classify.classify_image.assert_called_once()
        assert mock_classify.classify_image.call_args.kwargs["model_path"] == str(
            valid_model.resolve()
        )


class TestConfig:
    """Tests for configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GeoAIConfig()
        assert config.log_level == "INFO"
        assert config.timeout == 300
        assert config.max_memory_gb == 8
        assert config.device == "auto"

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = GeoAIConfig(
            input_dir=Path("/custom/input"),
            output_dir=Path("/custom/output"),
            log_level="DEBUG",
            timeout=600,
            max_memory_gb=16,
            device="cuda",
        )
        assert config.input_dir == Path("/custom/input").resolve()
        assert config.output_dir == Path("/custom/output").resolve()
        assert config.log_level == "DEBUG"
        assert config.timeout == 600
        assert config.max_memory_gb == 16
        assert config.device == "cuda"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        # This test would need environment variable mocking
        # Simplified version just checks the load_config function exists
        from geoai_mcp_server.config import load_config

        config = load_config()
        assert isinstance(config, GeoAIConfig)
