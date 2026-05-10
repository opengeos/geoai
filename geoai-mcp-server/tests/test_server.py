"""Tests for GeoAI MCP Server tools."""

import pytest
from unittest.mock import MagicMock, patch


class TestSegmentationTools:
    """Tests for segmentation-related tools."""

    @pytest.mark.asyncio
    async def test_segment_objects_with_prompts_missing_file(
        self, temp_input_dir, temp_output_dir
    ):
        """Test segmentation with missing input file."""
        from geoai_mcp_server.server import segment_objects_with_prompts

        # Patch config
        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await segment_objects_with_prompts(
                image_path="nonexistent.tif",
                prompts=["building"],
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_auto_segment_image_missing_file(
        self, temp_input_dir, temp_output_dir
    ):
        """Test auto segmentation with missing input file."""
        from geoai_mcp_server.server import auto_segment_image

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await auto_segment_image(
                image_path="nonexistent.tif",
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_segment_objects_grounded_sam_success(
        self, sample_geotiff, temp_output_dir
    ):
        """GroundedSAM path: output_files lists all result files; num_objects reflects detected count."""
        from geoai_mcp_server.server import segment_objects_with_prompts

        raster_path = temp_output_dir / "sample_segmented.tif"
        polygons_path = temp_output_dir / "sample_segmented_polygons.geojson"

        fake_result = {
            "segmentation": str(raster_path),
            "polygons": str(polygons_path),
        }

        mock_segmenter = MagicMock()
        mock_segmenter.segment_image.return_value = fake_result
        mock_segment = MagicMock()
        mock_segment.GroundedSAM.return_value = mock_segmenter

        mock_gdf = MagicMock()
        mock_gdf.__len__ = MagicMock(return_value=3)

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                return_value=mock_segment,
            ),
            patch("geopandas.read_file", return_value=mock_gdf),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = await segment_objects_with_prompts(
                image_path=sample_geotiff.name,
                prompts=["building"],
                model="grounded_sam",
                output_format="geojson",
            )

        assert result["success"] is True
        assert str(raster_path) in result["output_files"]
        assert str(polygons_path) in result["output_files"]
        assert result["num_objects"] == 3

    @pytest.mark.asyncio
    async def test_segment_objects_clipseg_uses_clipsegmentation(
        self, sample_geotiff, temp_output_dir
    ):
        """model='clipseg' routes to CLIPSegmentation, not GroundedSAM."""
        from geoai_mcp_server.server import segment_objects_with_prompts

        raster_path = temp_output_dir / "sample_segmented.tif"

        mock_segmenter = MagicMock()
        mock_segmenter.segment_image.return_value = str(raster_path)
        mock_segment = MagicMock()
        mock_segment.CLIPSegmentation.return_value = mock_segmenter

        mock_utils = MagicMock()

        def get_module(name):
            return mock_segment if name == "segment" else mock_utils

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", side_effect=get_module),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = await segment_objects_with_prompts(
                image_path=sample_geotiff.name,
                prompts=["building"],
                model="clipseg",
                output_format="geotiff",
            )

        assert result["success"] is True
        mock_segment.CLIPSegmentation.assert_called_once()
        mock_segment.GroundedSAM.assert_not_called()
        # CLIPSeg writes to raster_path (.tif), not to the user-facing output_path
        # (.geotiff). output_files must point to the file that was actually written.
        assert result["output_files"] == [str(raster_path)]


class TestDetectionTools:
    """Tests for detection-related tools."""

    @pytest.mark.asyncio
    async def test_detect_features_missing_file(self, temp_input_dir, temp_output_dir):
        """Test detection with missing input file."""
        from geoai_mcp_server.server import detect_and_classify_features

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await detect_and_classify_features(
                image_path="nonexistent.tif",
                feature_types=["buildings"],
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()


class TestChangeDetectionTools:
    """Tests for change detection tools."""

    @pytest.mark.asyncio
    async def test_detect_changes_missing_files(self, temp_input_dir, temp_output_dir):
        """Test change detection with missing files."""
        from geoai_mcp_server.server import detect_temporal_changes

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await detect_temporal_changes(
                image1_path="area_2020.tif",
                image2_path="area_2023.tif",
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()


class TestDownloadTools:
    """Tests for data download tools."""

    @pytest.mark.asyncio
    async def test_download_imagery_invalid_bbox(self, temp_output_dir):
        """Test download with valid bounding box structure."""
        from geoai_mcp_server.server import download_satellite_imagery

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.output_dir = temp_output_dir

            # Mock the geoai module
            with patch("geoai_mcp_server.server._get_geoai_module") as mock_get:
                mock_download = MagicMock()
                mock_download.download_naip.return_value = []
                mock_get.return_value = mock_download

                result = await download_satellite_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.4,
                    max_lat=37.8,
                    data_source="naip",
                )

                # Should succeed even with no files downloaded
                assert result["success"] is True
                assert "downloaded" in result["message"].lower()


class TestFoundationModelTools:
    """Tests for foundation model tools."""

    @pytest.mark.asyncio
    async def test_extract_features_missing_file(self, temp_input_dir, temp_output_dir):
        """Test feature extraction with missing file."""
        from geoai_mcp_server.server import extract_features_with_foundation_model

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await extract_features_with_foundation_model(
                image_path="nonexistent.tif",
                model="dinov3",
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_estimate_canopy_height_missing_file(
        self, temp_input_dir, temp_output_dir
    ):
        """Test canopy height with missing file."""
        from geoai_mcp_server.server import estimate_canopy_height

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await estimate_canopy_height(
                image_path="nonexistent.tif",
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_vlm_analysis_missing_query(
        self, temp_input_dir, temp_output_dir, sample_geotiff
    ):
        """Test VLM analysis with missing required query."""
        from geoai_mcp_server.server import analyze_with_vision_language_model

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            # Query task without query should fail
            result = await analyze_with_vision_language_model(
                image_path="nonexistent.tif",
                task="query",
                # Missing query parameter
            )

            assert result["success"] is False


class TestUtilityTools:
    """Tests for utility tools."""

    @pytest.mark.asyncio
    async def test_list_files_empty_directory(self, temp_input_dir, temp_output_dir):
        """Test listing files in empty directory."""
        from geoai_mcp_server.server import list_available_files

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await list_available_files(
                directory="input",
                pattern="*.tif",
            )

            assert result["success"] is True
            assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_list_files_with_files(self, temp_input_dir, temp_output_dir):
        """Test listing files with actual files."""
        from geoai_mcp_server.server import list_available_files

        # Create some test files
        (temp_input_dir / "test1.tif").touch()
        (temp_input_dir / "test2.tif").touch()
        (temp_input_dir / "test.jpg").touch()

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await list_available_files(
                directory="input",
                pattern="*.tif",
            )

            assert result["success"] is True
            assert result["total_count"] == 2

    @pytest.mark.asyncio
    async def test_list_files_invalid_directory(self, temp_input_dir, temp_output_dir):
        """Test listing files with invalid directory name."""
        from geoai_mcp_server.server import list_available_files

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await list_available_files(
                directory="invalid",  # Should fail
            )

            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_clean_results_missing_file(self, temp_input_dir, temp_output_dir):
        """Test cleaning results with missing file."""
        from geoai_mcp_server.server import clean_segmentation_results

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = await clean_segmentation_results(
                input_path="nonexistent.geojson",
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()


class TestToolIntegration:
    """Integration tests for tool workflows."""

    @pytest.mark.asyncio
    async def test_segmentation_workflow(
        self, sample_geotiff, temp_output_dir, mock_geoai_modules
    ):
        """Test a complete segmentation workflow."""
        from geoai_mcp_server.server import (
            segment_objects_with_prompts,
            list_available_files,
        )

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            # First, list available files
            list_result = await list_available_files(
                directory="input",
                pattern="*.tif",
            )

            if list_result["success"]:
                # Then attempt segmentation (will use mocked modules)
                seg_result = await segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                )

                # Result depends on mock behavior
                assert "message" in seg_result
