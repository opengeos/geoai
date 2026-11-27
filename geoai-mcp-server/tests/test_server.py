"""Tests for GeoAI MCP Server tools."""

import asyncio
import importlib
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSegmentationTools:
    """Tests for segmentation-related tools."""

    def test_segment_objects_with_prompts_missing_file(
        self, temp_input_dir, temp_output_dir
    ):
        """Test segmentation with missing input file."""
        from geoai_mcp_server.server import segment_objects_with_prompts

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                segment_objects_with_prompts(
                    image_path="nonexistent.tif",
                    prompts=["building"],
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    def test_auto_segment_image_missing_file(self, temp_input_dir, temp_output_dir):
        """Test auto segmentation with missing input file."""
        from geoai_mcp_server.server import auto_segment_image

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                auto_segment_image(
                    image_path="nonexistent.tif",
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    def test_segment_objects_grounded_sam_success(
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

            result = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="grounded_sam",
                    output_format="geojson",
                )
            )

        assert result["success"] is True
        assert str(raster_path) in result["output_files"]
        assert str(polygons_path) in result["output_files"]
        assert result["num_objects"] == 3

    def test_segment_objects_clipseg_uses_clipsegmentation(
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

            result = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="clipseg",
                    output_format="geotiff",
                )
            )

        assert result["success"] is True
        mock_segment.CLIPSegmentation.assert_called_once()
        mock_segment.GroundedSAM.assert_not_called()
        assert result["output_files"] == [str(raster_path)]

    def test_auto_segment_cleans_up_temp_files(self, sample_geotiff, temp_output_dir):
        """auto_segment_image cleans up intermediate temp GeoTIFFs after vectorization."""
        from geoai_mcp_server.server import auto_segment_image

        mock_segmenter = MagicMock()
        mock_segmenter.masks = [{"area": 200}, {"area": 300}]
        mock_sam = MagicMock()
        mock_sam.SamGeo.return_value = mock_segmenter

        mock_utils = MagicMock()
        mock_utils.raster_to_vector.return_value = [1, 2]

        def get_module(name):
            return mock_sam if name == "sam" else mock_utils

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", side_effect=get_module),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                auto_segment_image(
                    image_path=sample_geotiff.name,
                    output_format="geojson",
                    min_object_size=150,
                    clean_results=True,
                )
            )

        assert result["success"] is True
        assert result["output_files"][0].endswith(".geojson")
        mock_sam.SamGeo.assert_called_once_with(
            model="facebook/sam-vit-huge", automatic=True
        )
        mock_utils.clean_instance_mask.assert_called_once()
        mock_utils.raster_to_vector.assert_called_once()
        # No leftover temporary .tif files in output_dir
        assert list(temp_output_dir.glob("auto_seg_*.tif")) == []


class TestDetectionTools:
    """Tests for detection-related tools."""

    def test_detect_features_missing_file(self, temp_input_dir, temp_output_dir):
        """Test detection with missing input file."""
        from geoai_mcp_server.server import detect_and_classify_features

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                detect_and_classify_features(
                    image_path="nonexistent.tif",
                    feature_types=["buildings"],
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    def test_detect_features_does_not_swallow_errors(
        self, sample_geotiff, temp_output_dir
    ):
        """Detection errors must return success=False rather than fake success."""
        from geoai_mcp_server.server import detect_and_classify_features

        mock_segment = MagicMock()
        mock_segment.GroundedSAM.side_effect = RuntimeError("CUDA out of memory")

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module", return_value=mock_segment
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                detect_and_classify_features(
                    image_path=sample_geotiff.name,
                    feature_types=["buildings"],
                )
            )

        assert result["success"] is False
        assert "cuda out of memory" in result["message"].lower()

    def test_classify_land_cover_requires_model_path(
        self, sample_geotiff, temp_output_dir
    ):
        """classify_land_cover must reject None model_path with validation error."""
        from geoai_mcp_server.server import classify_land_cover

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                classify_land_cover(
                    image_path=sample_geotiff.name,
                    model_path=None,
                )
            )

        assert result["success"] is False
        assert "model_path" in result["message"].lower()


class TestChangeDetectionTools:
    """Tests for change detection tools."""

    def test_detect_changes_missing_files(self, temp_input_dir, temp_output_dir):
        """Test change detection with missing files."""
        from geoai_mcp_server.server import detect_temporal_changes

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                detect_temporal_changes(
                    image1_path="area_2020.tif",
                    image2_path="area_2023.tif",
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()


class TestDownloadTools:
    """Tests for data download tools."""

    def test_download_imagery_naip(self, temp_output_dir):
        """Test download with valid bounding box for NAIP."""
        from geoai_mcp_server.server import download_satellite_imagery

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.output_dir = temp_output_dir

            with patch("geoai_mcp_server.server._get_geoai_module") as mock_get:
                mock_download = MagicMock()
                mock_download.download_naip.return_value = []
                mock_get.return_value = mock_download

                result = asyncio.run(
                    download_satellite_imagery(
                        min_lon=-122.5,
                        min_lat=37.7,
                        max_lon=-122.4,
                        max_lat=37.8,
                        data_source="naip",
                    )
                )

                assert result["success"] is True
                assert "downloaded" in result["message"].lower()

    def test_download_imagery_rejects_unknown_source(self, temp_output_dir):
        """Unsupported data_source must fail validation."""
        from geoai_mcp_server.server import download_satellite_imagery

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                download_satellite_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.4,
                    max_lat=37.8,
                    data_source="unknown_provider",
                )
            )

        assert result["success"] is False
        assert "data_source" in result["message"].lower()

    def test_download_imagery_sentinel2_uses_stac(self, temp_output_dir):
        """Sentinel-2 routes through pc_stac_search and pc_stac_download."""
        from geoai_mcp_server.server import download_satellite_imagery

        fake_file = temp_output_dir / "item1_B04.tif"
        fake_file.write_bytes(b"0" * 1024)
        fake_item = MagicMock(id="item1")

        mock_download = MagicMock()
        mock_download.pc_stac_search.return_value = [fake_item]
        mock_download.pc_stac_download.return_value = {"item1": {"B04": str(fake_file)}}

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module", return_value=mock_download
            ),
        ):
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                download_satellite_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.4,
                    max_lat=37.8,
                    data_source="sentinel2",
                    date_range="2024-01-01/2024-06-01",
                    max_cloud_cover=15,
                    max_items=2,
                )
            )

        assert result["success"] is True
        assert result["num_files"] == 1
        assert result["output_files"] == [str(fake_file)]
        mock_download.pc_stac_search.assert_called_once()
        mock_download.pc_stac_download.assert_called_once()


class TestFoundationModelTools:
    """Tests for foundation model tools."""

    def test_extract_features_missing_file(self, temp_input_dir, temp_output_dir):
        """Test feature extraction with missing file."""
        from geoai_mcp_server.server import extract_features_with_foundation_model

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                extract_features_with_foundation_model(
                    image_path="nonexistent.tif",
                    model="dinov3",
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    def test_extract_features_rejects_partial_reference_coords(
        self, sample_geotiff, temp_output_dir
    ):
        """Providing reference_x without reference_y must fail validation."""
        from geoai_mcp_server.server import extract_features_with_foundation_model

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name,
                    model="dinov3",
                    output_type="similarity_map",
                    reference_x=50,
                    reference_y=None,
                )
            )

        assert result["success"] is False
        assert "reference_x" in result["message"].lower()

    def test_estimate_canopy_height_missing_file(self, temp_input_dir, temp_output_dir):
        """Test canopy height with missing file."""
        from geoai_mcp_server.server import estimate_canopy_height

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                estimate_canopy_height(
                    image_path="nonexistent.tif",
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    def test_vlm_analysis_missing_query(
        self, temp_input_dir, temp_output_dir, sample_geotiff
    ):
        """Test VLM analysis with existing image but missing required query."""
        from geoai_mcp_server.server import analyze_with_vision_language_model

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="query",
                )
            )

            assert result["success"] is False
            assert "query" in result["message"].lower()

    def test_vlm_analysis_rejects_invalid_task(self, sample_geotiff, temp_output_dir):
        """Invalid VLM task name must fail validation."""
        from geoai_mcp_server.server import analyze_with_vision_language_model

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="unsupported_task",
                )
            )

        assert result["success"] is False
        assert "invalid vlm task" in result["message"].lower()


class TestUtilityTools:
    """Tests for utility and read-only inspect/search tools."""

    def test_list_files_empty_directory(self, temp_input_dir, temp_output_dir):
        """Test listing files in empty directory."""
        from geoai_mcp_server.server import list_available_files

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                list_available_files(
                    directory="input",
                    pattern="*.tif",
                )
            )

            assert result["success"] is True
            assert result["total_count"] == 0

    def test_list_files_with_files(self, temp_input_dir, temp_output_dir):
        """Test listing files with actual files."""
        from geoai_mcp_server.server import list_available_files

        (temp_input_dir / "test1.tif").touch()
        (temp_input_dir / "test2.tif").touch()
        (temp_input_dir / "test.jpg").touch()

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                list_available_files(
                    directory="input",
                    pattern="*.tif",
                )
            )

            assert result["success"] is True
            assert result["total_count"] == 2

    def test_list_files_invalid_directory(self, temp_input_dir, temp_output_dir):
        """Test listing files with invalid directory name."""
        from geoai_mcp_server.server import list_available_files

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                list_available_files(
                    directory="invalid",
                )
            )

            assert result["success"] is False

    def test_clean_results_missing_file(self, temp_input_dir, temp_output_dir):
        """Test cleaning results with missing file."""
        from geoai_mcp_server.server import clean_segmentation_results

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                clean_segmentation_results(
                    input_path="nonexistent.geojson",
                )
            )

            assert result["success"] is False
            assert "not found" in result["message"].lower()

    def test_clean_results_from_output_dir_and_geographic_crs(
        self, sample_geojson, temp_output_dir
    ):
        """clean_segmentation_results supports source='output' and metric CRS projection."""
        import shutil
        from geoai_mcp_server.server import clean_segmentation_results

        out_copy = temp_output_dir / "seg_out.geojson"
        shutil.copy(sample_geojson, out_copy)

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geojson.parent
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                clean_segmentation_results(
                    input_path="seg_out.geojson",
                    source="output",
                    operation="remove_noise",
                    min_size=100,
                )
            )

        assert result["success"] is True
        # Polygons in sample_geojson are ~0.01 deg (~1km x 1km), so both survive
        # a 100 sq meter filter when properly projected from EPSG:4326 to UTM.
        assert result["original_count"] == 2
        assert result["final_count"] == 2
        assert Path(result["output_files"][0]).exists()

    def test_clean_results_fails_on_corrupt_vector(
        self, temp_input_dir, temp_output_dir
    ):
        """Unreadable vector file must return success=False rather than fake success."""
        from geoai_mcp_server.server import clean_segmentation_results

        bad = temp_input_dir / "corrupt.geojson"
        bad.write_text("not valid geojson {{{", encoding="utf-8")

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = temp_input_dir
            mock_config.output_dir = temp_output_dir

            result = asyncio.run(
                clean_segmentation_results(input_path="corrupt.geojson")
            )

        assert result["success"] is False

    def test_inspect_raster_and_vector_and_stac_search(
        self, sample_geotiff, sample_geojson, temp_output_dir
    ):
        """Read-only inspect_raster, inspect_vector, and search_stac_imagery work end-to-end."""
        from geoai_mcp_server.server import (
            inspect_raster,
            inspect_vector,
            search_stac_imagery,
        )

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r_res = asyncio.run(
                inspect_raster(image_path=sample_geotiff.name, include_stats=True)
            )
            assert r_res["success"] is True
            assert r_res["metadata"]["width"] == 100
            assert r_res["metadata"]["height"] == 100
            assert r_res["metadata"]["bands"] == 3
            assert len(r_res["band_stats"]) == 3

            v_res = asyncio.run(inspect_vector(vector_path=sample_geojson.name))
            assert v_res["success"] is True
            assert v_res["metadata"]["feature_count"] == 2
            assert "Polygon" in v_res["metadata"]["geometry_types"]

        fake_item = MagicMock()
        fake_item.id = "S2A_test_scene"
        fake_item.datetime = "2024-05-01T18:00:00Z"
        fake_item.bbox = [-122.5, 37.7, -122.4, 37.8]
        fake_item.properties = {"eo:cloud_cover": 4.2}
        fake_item.assets = {"visual": MagicMock(), "B04": MagicMock()}

        mock_download = MagicMock()
        mock_download.pc_stac_search.return_value = [fake_item]

        with patch(
            "geoai_mcp_server.server._get_geoai_module", return_value=mock_download
        ):
            stac_res = asyncio.run(
                search_stac_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.4,
                    max_lat=37.8,
                    collection="sentinel-2-l2a",
                    max_items=5,
                )
            )

        assert stac_res["success"] is True
        assert stac_res["total_items"] == 1
        assert stac_res["items"][0]["id"] == "S2A_test_scene"
        assert stac_res["items"][0]["cloud_cover"] == 4.2


class TestRuntimeSafety:
    """Tests for stdout redirection, timeouts, and side-effect-free imports."""

    def test_run_blocking_redirects_stdout_to_stderr(self, capsys):
        """Synchronous print() calls inside _run_blocking must not pollute stdout."""
        from geoai_mcp_server.server import _run_blocking

        def noisy():
            print("NOISY_LIBRARY_PRINT")
            return 42

        val = asyncio.run(_run_blocking(noisy))
        captured = capsys.readouterr()
        assert val == 42
        assert "NOISY_LIBRARY_PRINT" not in captured.out
        assert "NOISY_LIBRARY_PRINT" in captured.err

    def test_run_blocking_enforces_timeout(self):
        """_run_blocking raises OperationTimeoutError when config.timeout is exceeded."""
        from geoai_mcp_server.server import _run_blocking
        from geoai_mcp_server.utils.error_handling import OperationTimeoutError

        def slow():
            time.sleep(0.5)

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.timeout = 0.05
            with pytest.raises(OperationTimeoutError):
                asyncio.run(_run_blocking(slow))

    def test_import_server_has_no_filesystem_side_effects(self, tmp_path):
        """Importing geoai_mcp_server.server must not create ./input or ./output dirs."""
        src_dir = str(Path(__file__).parent.parent / "src")
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import geoai_mcp_server, geoai_mcp_server.server; "
                "assert geoai_mcp_server.mcp is geoai_mcp_server.server.mcp",
            ],
            cwd=str(tmp_path),
            env={
                "PATH": "",
                "PYTHONPATH": src_dir,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert not (tmp_path / "input").exists()
        assert not (tmp_path / "output").exists()


class TestToolIntegration:
    """Integration tests for tool workflows."""

    def test_segmentation_workflow(self, sample_geotiff, temp_output_dir):
        """Test listing input files and running GroundedSAM segmentation."""
        from geoai_mcp_server.server import (
            segment_objects_with_prompts,
            list_available_files,
        )

        out_tif = temp_output_dir / "out.tif"
        out_poly = temp_output_dir / "out_polygons.geojson"
        mock_segmenter = MagicMock()
        mock_segmenter.segment_image.return_value = {
            "segmentation": str(out_tif),
            "polygons": str(out_poly),
        }
        mock_segment = MagicMock()
        mock_segment.GroundedSAM.return_value = mock_segmenter

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module", return_value=mock_segment
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            list_result = asyncio.run(
                list_available_files(
                    directory="input",
                    pattern="*.tif",
                )
            )
            assert list_result["success"] is True
            assert list_result["total_count"] == 1
            assert list_result["files"][0]["name"] == sample_geotiff.name

            seg_result = asyncio.run(
                segment_objects_with_prompts(
                    image_path=list_result["files"][0]["name"],
                    prompts=["building"],
                    model="grounded_sam",
                )
            )
            assert seg_result["success"] is True
            assert str(out_poly) in seg_result["output_files"]
            mock_segmenter.segment_image.assert_called_once()
