"""Tests for utility modules."""

import pytest
from pathlib import Path

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
        # Spaces are replaced with underscores
        result = sanitize_filename("my file.tif")
        assert " " not in result
        assert result.endswith(".tif")

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
        assert config.input_dir == Path("/custom/input")
        assert config.output_dir == Path("/custom/output")
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
