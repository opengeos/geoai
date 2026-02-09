"""Tests for schema validation."""

import pytest
from pydantic import ValidationError

from geoai_mcp_server.schemas import (
    BoundingBox,
    SegmentObjectsInput,
    AutoSegmentInput,
    DetectFeaturesInput,
    ChangeDetectionInput,
    DownloadImageryInput,
    CanopyHeightInput,
    VLMAnalysisInput,
    CleanResultsInput,
    OutputFormat,
    DataSource,
    SegmentationModel,
    DetectionTarget,
    FoundationModel,
)


class TestBoundingBox:
    """Tests for BoundingBox schema."""

    def test_valid_bbox(self):
        """Test valid bounding box creation."""
        bbox = BoundingBox(
            min_lon=-122.5,
            min_lat=37.7,
            max_lon=-122.4,
            max_lat=37.8,
        )
        assert bbox.min_lon == -122.5
        assert bbox.min_lat == 37.7
        assert bbox.max_lon == -122.4
        assert bbox.max_lat == 37.8

    def test_to_tuple(self):
        """Test tuple conversion."""
        bbox = BoundingBox(
            min_lon=-122.5,
            min_lat=37.7,
            max_lon=-122.4,
            max_lat=37.8,
        )
        assert bbox.to_tuple() == (-122.5, 37.7, -122.4, 37.8)

    def test_invalid_longitude(self):
        """Test invalid longitude validation."""
        with pytest.raises(ValidationError):
            BoundingBox(
                min_lon=-200,  # Invalid
                min_lat=37.7,
                max_lon=-122.4,
                max_lat=37.8,
            )

    def test_invalid_latitude(self):
        """Test invalid latitude validation."""
        with pytest.raises(ValidationError):
            BoundingBox(
                min_lon=-122.5,
                min_lat=100,  # Invalid
                max_lon=-122.4,
                max_lat=37.8,
            )


class TestSegmentObjectsInput:
    """Tests for SegmentObjectsInput schema."""

    def test_valid_input(self):
        """Test valid segmentation input."""
        input_data = SegmentObjectsInput(
            image_path="test.tif",
            prompts=["building", "road"],
        )
        assert input_data.image_path == "test.tif"
        assert input_data.prompts == ["building", "road"]
        assert input_data.output_format == OutputFormat.GEOJSON
        assert input_data.model == SegmentationModel.AUTO
        assert input_data.confidence_threshold == 0.3

    def test_empty_prompts_fails(self):
        """Test that empty prompts list fails validation."""
        with pytest.raises(ValidationError):
            SegmentObjectsInput(
                image_path="test.tif",
                prompts=[],  # Empty list
            )

    def test_confidence_threshold_bounds(self):
        """Test confidence threshold validation."""
        # Valid at boundaries
        input_data = SegmentObjectsInput(
            image_path="test.tif",
            prompts=["building"],
            confidence_threshold=0.0,
        )
        assert input_data.confidence_threshold == 0.0

        input_data = SegmentObjectsInput(
            image_path="test.tif",
            prompts=["building"],
            confidence_threshold=1.0,
        )
        assert input_data.confidence_threshold == 1.0

        # Invalid outside boundaries
        with pytest.raises(ValidationError):
            SegmentObjectsInput(
                image_path="test.tif",
                prompts=["building"],
                confidence_threshold=1.5,
            )

    def test_tile_size_bounds(self):
        """Test tile size validation."""
        # Valid
        input_data = SegmentObjectsInput(
            image_path="test.tif",
            prompts=["building"],
            tile_size=512,
        )
        assert input_data.tile_size == 512

        # Too small
        with pytest.raises(ValidationError):
            SegmentObjectsInput(
                image_path="test.tif",
                prompts=["building"],
                tile_size=100,  # Below 256
            )


class TestAutoSegmentInput:
    """Tests for AutoSegmentInput schema."""

    def test_defaults(self):
        """Test default values."""
        input_data = AutoSegmentInput(image_path="test.tif")
        assert input_data.output_format == OutputFormat.GEOTIFF
        assert input_data.min_object_size == 100
        assert input_data.max_object_size is None
        assert input_data.clean_results is True


class TestDetectFeaturesInput:
    """Tests for DetectFeaturesInput schema."""

    def test_valid_input(self):
        """Test valid detection input."""
        input_data = DetectFeaturesInput(
            image_path="test.tif",
            feature_types=[DetectionTarget.BUILDINGS, DetectionTarget.VEHICLES],
        )
        assert len(input_data.feature_types) == 2

    def test_custom_prompts(self):
        """Test custom prompts for detection."""
        input_data = DetectFeaturesInput(
            image_path="test.tif",
            feature_types=[DetectionTarget.CUSTOM],
            custom_prompts=["swimming pool", "tennis court"],
        )
        assert input_data.custom_prompts == ["swimming pool", "tennis court"]


class TestChangeDetectionInput:
    """Tests for ChangeDetectionInput schema."""

    def test_valid_input(self):
        """Test valid change detection input."""
        input_data = ChangeDetectionInput(
            image1_path="area_2020.tif",
            image2_path="area_2023.tif",
        )
        assert input_data.change_threshold == 0.5
        assert input_data.include_statistics is True


class TestDownloadImageryInput:
    """Tests for DownloadImageryInput schema."""

    def test_valid_input(self):
        """Test valid download input."""
        input_data = DownloadImageryInput(
            bbox=BoundingBox(
                min_lon=-122.5,
                min_lat=37.7,
                max_lon=-122.4,
                max_lat=37.8,
            ),
            data_source=DataSource.NAIP,
        )
        assert input_data.max_cloud_cover == 20
        assert input_data.max_items == 10


class TestVLMAnalysisInput:
    """Tests for VLMAnalysisInput schema."""

    def test_caption_task(self):
        """Test caption task input."""
        input_data = VLMAnalysisInput(
            image_path="test.jpg",
            task="caption",
        )
        assert input_data.query is None

    def test_query_task(self):
        """Test query task input."""
        input_data = VLMAnalysisInput(
            image_path="test.jpg",
            task="query",
            query="How many buildings are visible?",
        )
        assert input_data.query == "How many buildings are visible?"


class TestCleanResultsInput:
    """Tests for CleanResultsInput schema."""

    def test_defaults(self):
        """Test default values."""
        input_data = CleanResultsInput(input_path="results.geojson")
        assert input_data.operation == "all"
        assert input_data.min_size == 100
        assert input_data.regularize_buildings is False
