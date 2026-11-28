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
        """auto_segment_image cleans up intermediate temp GeoTIFFs and staging dir after vectorization."""
        from geoai_mcp_server.server import auto_segment_image

        seen_raw_paths: list[Path] = []
        seen_clean_paths: list[Path] = []
        existed_during_vectorize: list[bool] = []

        class _FakeSam:
            def __init__(self, **kwargs):
                self.masks = [{"area": 200}, {"area": 300}]

            def generate(self, source, output, min_size=0, max_size=None):
                p = Path(output)
                p.write_bytes(b"raw_mask")
                seen_raw_paths.append(p)

        def _fake_clean(input_path, output_path, min_area=0):
            assert Path(input_path).is_file()
            p = Path(output_path)
            p.write_bytes(b"clean_mask")
            seen_clean_paths.append(p)

        def _fake_r2v(raster_path, output_path, min_area=0):
            existed_during_vectorize.append(
                Path(raster_path).is_file()
                and seen_raw_paths[0].is_file()
                and seen_clean_paths[0].is_file()
            )
            Path(output_path).write_text(
                '{"type": "FeatureCollection", "features": []}'
            )
            return [1, 2]

        mock_sam = MagicMock()
        mock_sam.SamGeo.side_effect = lambda **kw: _FakeSam(**kw)
        mock_utils = MagicMock()
        mock_utils.clean_instance_mask.side_effect = _fake_clean
        mock_utils.raster_to_vector.side_effect = _fake_r2v

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
        assert existed_during_vectorize == [True]
        assert not seen_raw_paths[0].exists()
        assert not seen_clean_paths[0].exists()
        assert not seen_raw_paths[0].parent.exists()
        assert list(temp_output_dir.glob(".auto_seg_tmp_*")) == []


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
        """Sentinel-2 routes through pc_stac_search and staged pc_stac_download publication."""
        from geoai_mcp_server.server import download_satellite_imagery

        payload = b"0" * 1024
        seen_stage_dirs: list[Path] = []

        def _fake_stac_dl(items, output_dir):
            stage_p = Path(output_dir)
            seen_stage_dirs.append(stage_p)
            staged_file = stage_p / "item1_B04.tif"
            staged_file.write_bytes(payload)
            return {"item1": {"B04": str(staged_file)}}

        fake_item = MagicMock(id="item1")
        fake_item.assets = {"B04": "href"}

        mock_download = MagicMock()
        mock_download.pc_stac_search.return_value = [fake_item]
        mock_download.pc_stac_download.side_effect = _fake_stac_dl

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

        expected_final = temp_output_dir / "item1_B04.tif"
        assert result["success"] is True
        assert result["num_files"] == 1
        assert result["output_files"] == [str(expected_final)]
        assert expected_final.is_file()
        assert expected_final.read_bytes() == payload
        assert seen_stage_dirs and not seen_stage_dirs[0].exists()
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
        """Test listing input files and running GroundedSAM segmentation with published polygon."""
        import geopandas as gpd
        from shapely.geometry import box
        from geoai_mcp_server.server import (
            segment_objects_with_prompts,
            list_available_files,
        )

        def _fake_segment_image(
            input_path,
            output_path,
            text_prompts,
            export_polygons=True,
            export_boxes=False,
        ):
            Path(output_path).write_bytes(b"fake_tif")
            poly_path = output_path.replace(".tif", "_polygons.geojson")
            gdf = gpd.GeoDataFrame(
                {"label": ["building"]},
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:4326",
            )
            gdf.to_file(poly_path, driver="GeoJSON")
            return {"segmentation": output_path, "polygons": poly_path}

        mock_segmenter = MagicMock()
        mock_segmenter.segment_image.side_effect = _fake_segment_image
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
            geojson_outputs = [
                p for p in seg_result["output_files"] if p.endswith(".geojson")
            ]
            assert len(geojson_outputs) == 1
            published_poly = Path(geojson_outputs[0])
            assert published_poly.is_file()
            assert published_poly.resolve().is_relative_to(temp_output_dir.resolve())
            assert len(gpd.read_file(published_poly)) == 1
            mock_segmenter.segment_image.assert_called_once()


class TestCodeRabbitRegressions:
    """Regression tests covering all PR #1018 review fixes."""

    def test_stac_search_allows_point_line_antimeridian_and_inclusive_cloud_cover(self):
        """search_stac_imagery accepts point/line/antimeridian bboxes and uses inclusive lte cloud filter."""
        from geoai_mcp_server.server import (
            download_satellite_imagery,
            search_stac_imagery,
        )

        mock_download = MagicMock()
        mock_download.pc_stac_search.return_value = []

        with patch(
            "geoai_mcp_server.server._get_geoai_module", return_value=mock_download
        ):
            res_pt = asyncio.run(
                search_stac_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.5,
                    max_lat=37.7,
                    max_cloud_cover=0,
                )
            )
            res_none = asyncio.run(
                search_stac_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.4,
                    max_lat=37.7,
                    max_cloud_cover=None,
                )
            )
            res_antimeridian = asyncio.run(
                search_stac_imagery(
                    min_lon=160.6,
                    min_lat=-55.95,
                    max_lon=-170.0,
                    max_lat=-25.89,
                )
            )
            # Download providers still require nonzero-area bounding boxes and
            # must reject before touching the provider.
            res_dl = asyncio.run(
                download_satellite_imagery(
                    min_lon=-122.5,
                    min_lat=37.7,
                    max_lon=-122.5,
                    max_lat=37.7,
                    data_source="naip",
                )
            )

        assert res_pt["success"] is True
        assert res_none["success"] is True
        assert res_antimeridian["success"] is True
        assert mock_download.pc_stac_search.call_args_list[0].kwargs["query"] == {
            "eo:cloud_cover": {"lte": 0}
        }
        assert mock_download.pc_stac_search.call_args_list[1].kwargs["query"] is None
        assert mock_download.pc_stac_search.call_args_list[2].kwargs["bbox"] == [
            160.6,
            -55.95,
            -170.0,
            -25.89,
        ]
        assert res_dl["success"] is False
        assert "min_lon" in res_dl["message"]
        mock_download.download_naip.assert_not_called()

    def test_concurrent_run_blocking_does_not_leak_stdout(self, capsys):
        """Concurrent _run_blocking calls never leak diagnostics to stdout or corrupt sys.stdout."""
        from geoai_mcp_server.server import _run_blocking

        orig_stdout = sys.stdout

        def worker(tag: str, delay: float):
            print(f"WORKER_{tag}_START")
            time.sleep(delay)
            print(f"WORKER_{tag}_END")
            return tag

        async def _run_both():
            return await asyncio.gather(
                _run_blocking(worker, "A", 0.02),
                _run_blocking(worker, "B", 0.06),
            )

        res = asyncio.run(_run_both())
        captured = capsys.readouterr()
        assert res == ["A", "B"]
        assert sys.stdout is orig_stdout
        assert "WORKER_" not in captured.out
        assert "WORKER_A_START" in captured.err
        assert "WORKER_B_END" in captured.err

    def test_auto_segment_counts_remaining_geotiff_instances_and_separates_units(
        self, sample_geotiff, temp_output_dir
    ):
        """GeoTIFF counts remaining raster instances; vectorization does not pass pixel min_object_size as map min_area."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from geoai_mcp_server.server import auto_segment_image

        class _FakeSam:
            def __init__(self, **kwargs):
                self.masks = [{"area": 20}, {"area": 200}]

            def generate(self, source, output, min_size=0, max_size=None):
                arr = np.zeros((1, 20, 20), dtype=np.uint8)
                arr[0, 2:16, 2:16] = 1  # only 1 instance remains after filtering
                with rasterio.open(
                    output,
                    "w",
                    driver="GTiff",
                    height=20,
                    width=20,
                    count=1,
                    dtype="uint8",
                    crs="EPSG:32610",
                    transform=from_bounds(0, 0, 10, 10, 20, 20),
                ) as dst:
                    dst.write(arr)

        mock_sam = MagicMock()
        mock_sam.SamGeo.side_effect = lambda **kw: _FakeSam(**kw)
        mock_utils = MagicMock()
        mock_utils.raster_to_vector.return_value = [1]

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: mock_sam if name == "sam" else mock_utils,
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_tif = asyncio.run(
                auto_segment_image(
                    image_path=sample_geotiff.name,
                    output_format="geotiff",
                    min_object_size=100,
                    clean_results=False,
                )
            )
            res_vec = asyncio.run(
                auto_segment_image(
                    image_path=sample_geotiff.name,
                    output_format="geojson",
                    min_object_size=100,
                    clean_results=False,
                )
            )

        assert res_tif["success"] is True
        assert res_tif["num_objects"] == 1
        assert res_vec["success"] is True
        assert mock_utils.raster_to_vector.call_args.kwargs.get("min_area") == 0

    def test_grounded_sam_honors_geopackage_and_reports_mean_score(
        self, sample_geotiff, temp_output_dir
    ):
        """GroundedSAM converts staged GeoJSON to requested GPKG and reports mean detection score."""
        import geopandas as gpd
        from shapely.geometry import box
        from geoai_mcp_server.server import (
            detect_and_classify_features,
            segment_objects_with_prompts,
        )

        class _FakeGSAM:
            def __init__(self, **kwargs):
                pass

            def segment_image(
                self,
                input_path,
                output_path,
                text_prompts,
                export_boxes=False,
                export_polygons=True,
            ):
                Path(output_path).write_bytes(b"fake_tif")
                out = {"segmentation": output_path}
                gdf = gpd.GeoDataFrame(
                    {
                        "label": ["building", "building"],
                        "score": [0.8, 0.9],
                    },
                    geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
                    crs="EPSG:4326",
                )
                if export_boxes:
                    bp = output_path.replace(".tif", "_boxes.geojson")
                    gdf.to_file(bp, driver="GeoJSON")
                    out["boxes"] = bp
                if export_polygons:
                    pp = output_path.replace(".tif", "_polygons.geojson")
                    gdf.to_file(pp, driver="GeoJSON")
                    out["polygons"] = pp
                return out

        mock_seg = MagicMock()
        mock_seg.GroundedSAM.side_effect = lambda **kw: _FakeGSAM(**kw)

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_seg),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_det = asyncio.run(
                detect_and_classify_features(
                    image_path=sample_geotiff.name,
                    feature_types=["buildings"],
                    confidence_threshold=0.5,
                    output_format="geopackage",
                    output_filename="det_out.gpkg",
                )
            )
            res_seg = asyncio.run(
                segment_objects_with_prompts(
                    image_path=sample_geotiff.name,
                    prompts=["building"],
                    model="grounded_sam",
                    output_format="geopackage",
                    output_filename="seg_out.gpkg",
                )
            )

        assert res_det["success"] is True
        assert res_det["average_confidence"] == pytest.approx(0.85)
        assert any(p.endswith(".gpkg") for p in res_det["output_files"])
        assert not any(p.endswith(".geojson") for p in res_det["output_files"])
        assert (temp_output_dir / "det_out.gpkg").is_file()

        assert res_seg["success"] is True
        assert res_seg["num_objects"] == 2
        assert any(p.endswith(".gpkg") for p in res_seg["output_files"])
        assert not any(p.endswith(".geojson") for p in res_seg["output_files"])
        assert (temp_output_dir / "seg_out.gpkg").is_file()

    def test_classify_and_canopy_reject_vector_formats_and_classify_excludes_nodata(
        self, sample_geotiff, temp_output_dir
    ):
        """classify_land_cover and estimate_canopy_height reject vector formats; classify excludes nodata."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from geoai_mcp_server.server import (
            classify_land_cover,
            estimate_canopy_height,
        )

        model_file = sample_geotiff.parent / "cls_model.ckpt"
        model_file.write_bytes(b"ckpt")

        def _fake_classify(input_path, model_path, output_path):
            arr = np.array([[[0, 0], [0, 2]]], dtype=np.uint8)
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                nodata=0,
                crs="EPSG:4326",
                transform=from_bounds(0, 0, 1, 1, 2, 2),
            ) as dst:
                dst.write(arr)
            return output_path

        mock_cls = MagicMock()
        mock_cls.classify_image.side_effect = _fake_classify

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_cls),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_bad_cls = asyncio.run(
                classify_land_cover(
                    image_path=sample_geotiff.name,
                    model_path="cls_model.ckpt",
                    output_format="geojson",
                )
            )
            res_bad_can = asyncio.run(
                estimate_canopy_height(
                    image_path=sample_geotiff.name,
                    output_format="geojson",
                )
            )
            res_ok_cls = asyncio.run(
                classify_land_cover(
                    image_path=sample_geotiff.name,
                    model_path="cls_model.ckpt",
                    output_format="geotiff",
                )
            )

        assert res_bad_cls["success"] is False
        assert res_bad_can["success"] is False
        assert res_ok_cls["success"] is True
        assert res_ok_cls["dominant_class"] == "2"
        assert res_ok_cls["class_distribution"] == {"2": 100.0}

    def test_change_detection_zero_mask_does_not_advertise_stale_vector(
        self, sample_geotiff, temp_output_dir
    ):
        """All-zero change mask returns output_files=[] even if a prior vector exists at output_path."""
        import numpy as np
        from geoai_mcp_server.server import detect_temporal_changes

        stale = temp_output_dir / "chg.geojson"
        stale.write_text('{"type": "FeatureCollection", "features": []}')

        mock_cd = MagicMock()
        mock_cd.changestar_detect.return_value = {
            "change_map": np.zeros((4, 4), dtype=np.uint8)
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_cd),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res = asyncio.run(
                detect_temporal_changes(
                    image1_path=sample_geotiff.name,
                    image2_path=sample_geotiff.name,
                    output_format="geojson",
                    output_filename="chg.geojson",
                )
            )

        assert res["success"] is True
        assert res["change_percentage"] == 0.0
        assert res["output_files"] == []

    def test_prepare_training_data_partial_and_current_export_count(
        self, sample_geotiff, temp_output_dir
    ):
        """Partial batch export reports partial/errors/failed_files; single export uses total_tiles not pre-existing files."""
        from geoai_mcp_server.server import prepare_training_data

        img_dir = sample_geotiff.parent / "train_imgs"
        lbl_dir = sample_geotiff.parent / "train_lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Pre-populate stale tiles in output_dir/images
        pre_dir = temp_output_dir / "images"
        pre_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (pre_dir / f"old_{i}.tif").write_bytes(b"old")

        mock_utils = MagicMock()
        mock_utils.export_geotiff_tiles_batch.return_value = {
            "total_tiles": 4,
            "errors": 1,
            "failed_files": [{"image": "bad.tif", "error": "corrupt"}],
        }
        mock_utils.export_geotiff_tiles.return_value = {"total_tiles": 2, "errors": 0}

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_utils),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_partial = asyncio.run(
                prepare_training_data(images_dir="train_imgs", labels_dir="train_lbls")
            )
            res_single = asyncio.run(
                prepare_training_data(
                    images_dir=sample_geotiff.name, labels_dir=sample_geotiff.name
                )
            )

        assert res_partial["success"] is False
        assert res_partial["partial"] is True
        assert res_partial["errors"] == 1
        assert len(res_partial["failed_files"]) == 1
        assert res_partial["num_tiles"] == 4

        assert res_single["success"] is True
        assert res_single["partial"] is False
        assert res_single["num_tiles"] == 2

    def test_prithvi_rejects_similarity_map_and_normalizes_npy_filename(
        self, sample_geotiff, temp_output_dir
    ):
        """Prithvi rejects similarity_map and np.save output_filename is normalized with .npy."""
        import numpy as np
        from geoai_mcp_server.server import extract_features_with_foundation_model

        class _FakeDino:
            def load_image(self, path):
                return np.zeros((3, 8, 8), dtype=np.uint8), {}

            def preprocess_image_for_dinov3(self, data):
                return data

            def extract_features(self, img):
                return np.zeros((4, 8), dtype=np.float32), 2, 2

        mock_dino = MagicMock()
        mock_dino.DINOv3GeoProcessor.return_value = _FakeDino()

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_dino),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_bad = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name,
                    model="prithvi",
                    output_type="similarity_map",
                    reference_x=10,
                    reference_y=20,
                )
            )
            res_npy = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name,
                    model="dinov3",
                    output_filename="features",
                )
            )

        assert res_bad["success"] is False
        assert "prithvi" in res_bad["message"].lower()
        assert res_npy["success"] is True
        assert res_npy["output_files"] == [str(temp_output_dir / "features.npy")]
        assert (temp_output_dir / "features.npy").is_file()

    def test_vlm_does_not_return_stale_geojson(self, sample_geotiff, temp_output_dir):
        """VLM detect/point does not return a pre-existing GeoJSON when no file is written."""
        from geoai_mcp_server.server import analyze_with_vision_language_model

        stale = temp_output_dir / "vlm_out.geojson"
        stale.write_text('{"type": "FeatureCollection", "features": []}')

        mock_md = MagicMock()
        mock_md.moondream_detect.return_value = {"objects": [{"x_min": 0.1}]}
        mock_md.moondream_point.return_value = {"points": [{"x": 0.5, "y": 0.5}]}

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_md),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_det = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="detect",
                    detect_target="building",
                    output_filename="vlm_out.geojson",
                )
            )
            res_pt = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="point",
                    detect_target="building",
                    output_filename="vlm_out.geojson",
                )
            )

        assert res_det["success"] is True
        assert res_det["output_files"] == []
        assert res_pt["success"] is True
        assert res_pt["output_files"] == []

    def test_clean_skips_regularize_when_empty_and_inspect_uses_band_mask(
        self, sample_geojson, temp_input_dir, temp_output_dir
    ):
        """clean_segmentation_results skips regularize on empty result; inspect_raster honors band mask."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from geoai_mcp_server.server import (
            clean_segmentation_results,
            inspect_raster,
            list_available_files,
        )

        # Create raster with no scalar nodata but a validity mask excluding 100
        masked_tif = temp_input_dir / "masked.tif"
        with rasterio.open(
            masked_tif,
            "w",
            driver="GTiff",
            height=1,
            width=2,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 2, 1, 2, 1),
        ) as dst:
            dst.write(np.array([[[1, 100]]], dtype=np.uint8))
            dst.write_mask(np.array([[255, 0]], dtype=np.uint8))

        # Create extra files to test list_available_files truncation flag
        for i in range(5):
            (temp_input_dir / f"extra_{i}.tif").write_bytes(b"x")

        mock_utils = MagicMock()
        mock_utils.regularize.side_effect = AssertionError(
            "regularize must not be called on empty GeoDataFrame"
        )

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_utils),
        ):
            mock_config.input_dir = sample_geojson.parent
            mock_config.output_dir = temp_output_dir

            res_clean = asyncio.run(
                clean_segmentation_results(
                    input_path=sample_geojson.name,
                    operation="all",
                    min_size=10**12,
                    regularize_buildings=True,
                )
            )
            res_inspect = asyncio.run(
                inspect_raster(image_path="masked.tif", include_stats=True)
            )
            res_list = asyncio.run(
                list_available_files(directory="input", pattern="*.tif", max_files=3)
            )

        assert res_clean["success"] is True
        assert res_clean["final_count"] == 0
        mock_utils.regularize.assert_not_called()

        assert res_inspect["success"] is True
        assert res_inspect["band_stats"][0]["valid_pixels"] == 1
        assert res_inspect["band_stats"][0]["mean"] == 1.0

        assert res_list["success"] is True
        assert res_list["total_count"] == 3
        assert res_list["truncated"] is True

    def test_stac_download_timeout_leaves_no_partial_file(self, temp_output_dir):
        """Timed-out STAC download stages outside output_dir and never publishes after timeout."""
        import threading
        from geoai_mcp_server.server import (
            download_satellite_imagery,
            list_available_files,
        )

        staged_written = threading.Event()
        release_worker = threading.Event()
        worker_exited = threading.Event()
        staged_paths: list[Path] = []
        listing_while_streaming: list[dict] = []

        def _slow_download(items, output_dir):
            try:
                stage_path = Path(output_dir)
                part = stage_path / "scene_B04.tif"
                part.write_bytes(b"partial")
                staged_paths.append(stage_path)
                staged_written.set()
                release_worker.wait(timeout=5.0)
                return {"scene": {"B04": str(part)}}
            finally:
                worker_exited.set()

        mock_dl = MagicMock()
        fake_item = MagicMock(id="scene")
        fake_item.assets = {"B04": "href"}
        mock_dl.pc_stac_search.return_value = [fake_item]
        mock_dl.pc_stac_download.side_effect = _slow_download

        real_wait_for = asyncio.wait_for
        wait_calls = {"n": 0}

        async def _gated_wait_for(aw, timeout):
            wait_calls["n"] += 1
            # Call 1 is pc_stac_search; call 2 is _stage_stac_download.
            # Start the short deadline only after the fake downloader confirms
            # its staged write so slow CI startup cannot expire beforehand.
            if wait_calls["n"] == 2:
                task = asyncio.ensure_future(aw)
                assert await asyncio.to_thread(staged_written.wait, 5.0)
                listing_while_streaming.append(
                    await list_available_files(directory="output", pattern="*/*.tif")
                )
                return await real_wait_for(task, timeout=0.01)
            return await real_wait_for(aw, timeout=timeout)

        async def _drive_timeout():
            res = await download_satellite_imagery(
                min_lon=-122.5,
                min_lat=37.7,
                max_lon=-122.4,
                max_lat=37.8,
                data_source="sentinel2",
                max_cloud_cover=0,
            )
            release_worker.set()
            assert await asyncio.to_thread(worker_exited.wait, 5.0)
            return res

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_dl),
            patch(
                "geoai_mcp_server.server.asyncio.wait_for", side_effect=_gated_wait_for
            ),
        ):
            mock_config.output_dir = temp_output_dir
            mock_config.timeout = 30.0

            res = asyncio.run(_drive_timeout())

        assert res["success"] is False
        assert "timed out" in res["message"].lower()
        assert mock_dl.pc_stac_search.call_args.kwargs["query"] == {
            "eo:cloud_cover": {"lte": 0}
        }
        assert staged_paths and not staged_paths[0].resolve().is_relative_to(
            temp_output_dir.resolve()
        )
        assert listing_while_streaming[0]["total_count"] == 0
        assert not (temp_output_dir / "scene_B04.tif").exists()
        deadline = time.time() + 2.0
        while staged_paths[0].exists() and time.time() < deadline:
            time.sleep(0.01)
        assert not staged_paths[0].exists()

    def test_prepare_training_data_rejects_batch_raster_label_and_replaces_retry_unit(
        self, sample_geotiff, temp_output_dir
    ):
        """Batch mode rejects single raster label; retrying output_subdir replaces failed dataset as a unit."""
        from geoai_mcp_server.server import prepare_training_data

        img_dir = sample_geotiff.parent / "batch_imgs"
        lbl_dir = sample_geotiff.parent / "batch_lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()

        call_num = {"n": 0}

        def _fake_batch(
            images_folder,
            output_folder,
            tile_size,
            stride,
            quiet=True,
            masks_folder=None,
            masks_file=None,
        ):
            call_num["n"] += 1
            imgs_out = Path(output_folder) / "images"
            imgs_out.mkdir(parents=True, exist_ok=True)
            if call_num["n"] == 1:
                # Simulate A failing and B writing B_000000.tif
                (imgs_out / "B_000000.tif").write_bytes(b"tileB_old")
                return {
                    "total_tiles": 1,
                    "errors": 1,
                    "failed_files": [{"image": "A.tif", "error": "bad"}],
                }
            # Simulate clean retry writing A_000000.tif and B_000001.tif
            (imgs_out / "A_000000.tif").write_bytes(b"tileA")
            (imgs_out / "B_000001.tif").write_bytes(b"tileB_new")
            return {"total_tiles": 2, "errors": 0, "failed_files": []}

        mock_utils = MagicMock()
        mock_utils.export_geotiff_tiles_batch.side_effect = _fake_batch

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_utils),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            # Single GeoTIFF mask with an image directory must be rejected
            res_bad_mask = asyncio.run(
                prepare_training_data(
                    images_dir="batch_imgs",
                    labels_dir=sample_geotiff.name,
                )
            )
            res_first = asyncio.run(
                prepare_training_data(
                    images_dir="batch_imgs",
                    labels_dir="batch_lbls",
                    output_subdir="ds_unit",
                )
            )
            res_retry = asyncio.run(
                prepare_training_data(
                    images_dir="batch_imgs",
                    labels_dir="batch_lbls",
                    output_subdir="ds_unit",
                )
            )

        assert res_bad_mask["success"] is False
        assert "vector" in res_bad_mask["message"].lower()
        assert res_first["success"] is False
        assert res_first["partial"] is True
        assert res_retry["success"] is True
        assert res_retry["partial"] is False
        final_tiles = sorted(
            p.name for p in (temp_output_dir / "ds_unit" / "images").iterdir()
        )
        assert final_tiles == ["A_000000.tif", "B_000001.tif"]

    def test_vlm_publishes_all_shapefile_sidecars_and_inspect_honors_explicit_mask(
        self, sample_geotiff, temp_input_dir, temp_output_dir
    ):
        """VLM shapefile output publishes all sidecars; inspect_raster lets explicit mask override scalar nodata."""
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from shapely.geometry import box
        from geoai_mcp_server.server import (
            analyze_with_vision_language_model,
            inspect_raster,
        )

        # Explicit mask marks both pixels valid even though scalar nodata=0
        override_tif = temp_input_dir / "mask_override.tif"
        with rasterio.open(
            override_tif,
            "w",
            driver="GTiff",
            height=1,
            width=2,
            count=1,
            dtype="uint8",
            nodata=0,
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 2, 1, 2, 1),
        ) as dst:
            dst.write(np.array([[[0, 10]]], dtype=np.uint8))
            dst.write_mask(np.array([[255, 255]], dtype=np.uint8))

        def _write_shp(source, object_type, output_path):
            gdf = gpd.GeoDataFrame(
                {"label": [object_type]},
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:4326",
            )
            gdf.to_file(output_path, driver="ESRI Shapefile")
            return {"objects": [{"x_min": 0.1}]}

        mock_md = MagicMock()
        mock_md.moondream_detect.side_effect = _write_shp

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_md),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            res_shp = asyncio.run(
                analyze_with_vision_language_model(
                    image_path=sample_geotiff.name,
                    task="detect",
                    detect_target="building",
                    output_filename="objects.shp",
                )
            )
            res_inspect = asyncio.run(
                inspect_raster(image_path="mask_override.tif", include_stats=True)
            )

        assert res_shp["success"] is True
        assert res_shp["output_files"] == [str(temp_output_dir / "objects.shp")]
        assert (temp_output_dir / "objects.shp").is_file()
        assert (temp_output_dir / "objects.shx").is_file()
        assert (temp_output_dir / "objects.dbf").is_file()
        reloaded = gpd.read_file(temp_output_dir / "objects.shp")
        assert len(reloaded) == 1

        assert res_inspect["success"] is True
        assert res_inspect["band_stats"][0]["valid_pixels"] == 2
        assert res_inspect["band_stats"][0]["mean"] == 5.0

    def test_round3_fixes_worker_timeout_naip_stac_partial_npy_sim_prithvi(
        self, sample_geotiff, temp_output_dir
    ):
        """Verify round-3 fixes: worker TimeoutError, NAIP year filter, failed STAC download,
        partial training preserving usable dataset, .NPY suffix, similarity map isolation,
        and Prithvi coordinate forwarding.
        """
        import numpy as np
        import torch
        from types import SimpleNamespace
        from geoai_mcp_server.server import (
            _run_blocking,
            download_satellite_imagery,
            extract_features_with_foundation_model,
            prepare_training_data,
        )

        # 1. Worker-raised built-in TimeoutError propagates unchanged (not OperationTimeoutError)
        def _upstream_timeout():
            raise TimeoutError("upstream socket timed out")

        with patch("geoai_mcp_server.server.config") as mock_config:
            mock_config.timeout = 30.0
            with pytest.raises(TimeoutError, match="upstream socket timed out"):
                asyncio.run(_run_blocking(_upstream_timeout))

        # 2. NAIP date_range maps whole year to year=2018 and rejects partial-year range
        mock_dl = MagicMock()
        mock_dl.download_naip.return_value = []
        mock_dl.pc_stac_search.side_effect = [
            [],  # empty search -> success=True
            [
                SimpleNamespace(id="s1", assets={"B04": "href"})
            ],  # failed download -> success=False
        ]
        mock_dl.pc_stac_download.return_value = {}

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_dl),
        ):
            mock_config.output_dir = temp_output_dir

            res_naip_ok = asyncio.run(
                download_satellite_imagery(
                    -122.5,
                    37.7,
                    -122.4,
                    37.8,
                    data_source="naip",
                    date_range="2018-01-01/2018-12-31",
                )
            )
            res_naip_bad = asyncio.run(
                download_satellite_imagery(
                    -122.5,
                    37.7,
                    -122.4,
                    37.8,
                    data_source="naip",
                    date_range="2018-03-01/2018-06-01",
                )
            )
            res_stac_empty = asyncio.run(
                download_satellite_imagery(
                    -122.5, 37.7, -122.4, 37.8, data_source="sentinel2"
                )
            )
            res_stac_fail = asyncio.run(
                download_satellite_imagery(
                    -122.5, 37.7, -122.4, 37.8, data_source="sentinel2"
                )
            )

        assert res_naip_ok["success"] is True
        assert mock_dl.download_naip.call_args.kwargs["year"] == 2018
        assert res_naip_bad["success"] is False
        assert res_stac_empty["success"] is True and res_stac_empty["num_files"] == 0
        assert res_stac_fail["success"] is False and res_stac_fail["num_files"] == 0

        # 3. Partial training export does not overwrite an existing usable dataset
        img_dir = sample_geotiff.parent / "r3_imgs"
        lbl_dir = sample_geotiff.parent / "r3_lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()

        export_round = {"n": 0}

        def _batch_usable_then_partial(
            images_folder,
            output_folder,
            tile_size,
            stride,
            quiet=True,
            masks_folder=None,
        ):
            export_round["n"] += 1
            out_imgs = Path(output_folder) / "images"
            out_imgs.mkdir(parents=True, exist_ok=True)
            if export_round["n"] == 1:
                (out_imgs / "good_0.tif").write_bytes(b"good")
                return {"total_tiles": 1, "errors": 0, "failed_files": []}
            (out_imgs / "partial_0.tif").write_bytes(b"partial")
            return {
                "total_tiles": 1,
                "errors": 1,
                "failed_files": [{"image": "bad.tif", "error": "err"}],
            }

        mock_utils = MagicMock()
        mock_utils.export_geotiff_tiles_batch.side_effect = _batch_usable_then_partial

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch("geoai_mcp_server.server._get_geoai_module", return_value=mock_utils),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r_good = asyncio.run(
                prepare_training_data(
                    images_dir="r3_imgs", labels_dir="r3_lbls", output_subdir="ds_keep"
                )
            )
            r_part = asyncio.run(
                prepare_training_data(
                    images_dir="r3_imgs", labels_dir="r3_lbls", output_subdir="ds_keep"
                )
            )

        assert r_good["success"] is True
        assert r_part["success"] is False
        assert r_part["partial"] is True
        assert (temp_output_dir / "ds_keep" / "images" / "good_0.tif").is_file()
        assert not (temp_output_dir / "ds_keep" / "images" / "partial_0.tif").exists()

        # 4. Uppercase .NPY suffix, DINOv3 similarity map isolation, and Prithvi coords
        stale_sim_dir = temp_output_dir / "sim_reuse"
        stale_sim_dir.mkdir(parents=True, exist_ok=True)
        (stale_sim_dir / "stale_old.png").write_bytes(b"old")

        def _fake_sim_map(input_image, query_coords, output_dir):
            (Path(output_dir) / "fresh_sim.png").write_bytes(b"new")
            return {"similarity_map": np.zeros((4, 4))}

        seen_encoder_coords = {}

        class _FakePrithviEnc:
            def __call__(
                self, x, temporal_coords=None, location_coords=None, mask_ratio=0.0
            ):
                seen_encoder_coords["temporal"] = temporal_coords
                seen_encoder_coords["location"] = location_coords
                return (
                    torch.zeros((1, 5, 8)),
                    torch.zeros((1, 4)),
                    torch.arange(4, dtype=torch.long).unsqueeze(0),
                )

        class _FakePrithvi:
            def __init__(self):
                self.img_size = 16
                self.device = torch.device("cpu")
                self.coords_encoding = ["time", "location"]
                self.model = SimpleNamespace(encoder=_FakePrithviEnc())

            def load_images(self, file_paths):
                return (
                    np.zeros((1, 6, 1, 16, 16), dtype=np.float32),
                    [{}],
                    [[2024.0, 120.0]],
                    [37.7, -122.5],
                )

        modules = {
            "dinov3": SimpleNamespace(create_similarity_map=_fake_sim_map),
            "prithvi": SimpleNamespace(PrithviProcessor=_FakePrithvi),
        }

        with (
            patch("geoai_mcp_server.server.config") as mock_config,
            patch(
                "geoai_mcp_server.server._get_geoai_module",
                side_effect=lambda name: modules[name],
            ),
        ):
            mock_config.input_dir = sample_geotiff.parent
            mock_config.output_dir = temp_output_dir

            r_sim = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name,
                    model="dinov3",
                    output_type="similarity_map",
                    reference_x=1,
                    reference_y=2,
                    output_filename="sim_reuse.npy",
                )
            )
            r_pr = asyncio.run(
                extract_features_with_foundation_model(
                    image_path=sample_geotiff.name,
                    model="prithvi",
                    output_filename="features.NPY",
                )
            )

        assert r_sim["success"] is True
        assert r_sim["output_files"] == [str(stale_sim_dir / "fresh_sim.png")]
        assert r_pr["success"] is True
        assert r_pr["output_files"] == [str(temp_output_dir / "features.NPY.npy")]
        assert (temp_output_dir / "features.NPY.npy").is_file()
        assert seen_encoder_coords["temporal"] is not None
        assert tuple(seen_encoder_coords["temporal"].shape) == (1, 1, 2)
        assert seen_encoder_coords["location"] is not None
        assert tuple(seen_encoder_coords["location"].shape) == (1, 2)
