"""Unit tests for geoai CLI core modules.

Tests core logic with synthetic data where possible, real GeoTIFF files
for raster/vector operations. No model inference tests here (those are E2E).
"""

import json
import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from cli_anything.geoai.core import (
    change as change_mod,
    classify as classify_mod,
    data as data_mod,
    detect as detect_mod,
    project as project_mod,
    raster as raster_mod,
    segment as segment_mod,
    session as session_mod,
    vector as vector_mod,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)

LANDSAT_2022 = os.path.join(REPO_ROOT, "knoxville_landsat_2022.tif")
LANDSAT_2023 = os.path.join(REPO_ROOT, "knoxville_landsat_2023.tif")
NDVI_2022 = os.path.join(REPO_ROOT, "knoxville_ndvi_2022.tif")


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="geoai_test_") as d:
        yield d


@pytest.fixture
def sample_raster(tmp_dir):
    """Create a small synthetic GeoTIFF for testing."""
    path = os.path.join(tmp_dir, "sample.tif")
    data = np.random.rand(3, 64, 64).astype(np.float32)
    transform = from_bounds(-84.0, 35.9, -83.9, 36.0, 64, 64)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def sample_vector(tmp_dir):
    """Create a sample GeoJSON file for testing."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"value": [1, 2, 3]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)],
        crs="EPSG:4326",
    )
    path = os.path.join(tmp_dir, "sample.geojson")
    gdf.to_file(path, driver="GeoJSON")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# core/project.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProject:
    """Tests for project management."""

    def test_create_project_defaults(self):
        """Create project with default values."""
        proj = project_mod.create_project()
        assert proj["name"] == "untitled"
        assert proj["crs"] == "EPSG:4326"
        assert proj["files"] == []
        assert proj["results"] == []
        assert "created" in proj

    def test_create_project_custom(self):
        """Create project with custom parameters."""
        proj = project_mod.create_project(
            name="landsat_analysis", crs="EPSG:32617"
        )
        assert proj["name"] == "landsat_analysis"
        assert proj["crs"] == "EPSG:32617"

    def test_open_project_valid(self, tmp_dir):
        """Open a valid JSON project file."""
        path = os.path.join(tmp_dir, "proj.json")
        proj = project_mod.create_project(name="test")
        project_mod.save_project(proj, path)

        loaded = project_mod.open_project(path)
        assert loaded["name"] == "test"

    def test_open_project_missing_file(self):
        """FileNotFoundError for missing project file."""
        with pytest.raises(FileNotFoundError):
            project_mod.open_project("/nonexistent/project.json")

    def test_open_project_invalid_json(self, tmp_dir):
        """ValueError for invalid JSON."""
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("not json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            project_mod.open_project(path)

    def test_save_project(self, tmp_dir):
        """Save and reload project round-trip."""
        path = os.path.join(tmp_dir, "proj.json")
        proj = project_mod.create_project(name="roundtrip")
        saved_path = project_mod.save_project(proj, path)

        assert os.path.isfile(saved_path)
        with open(saved_path) as f:
            data = json.load(f)
        assert data["name"] == "roundtrip"

    def test_get_project_info(self):
        """Verify project info summary."""
        proj = project_mod.create_project(name="info_test")
        proj["files"] = [
            {"type": "raster"},
            {"type": "raster"},
            {"type": "vector"},
        ]
        info = project_mod.get_project_info(proj)
        assert info["file_count"] == 3
        assert info["raster_count"] == 2
        assert info["vector_count"] == 1

    def test_add_file_raster(self, tmp_dir, sample_raster):
        """Add a raster file entry to project."""
        proj = project_mod.create_project()
        entry = project_mod.add_file(proj, sample_raster, file_type="raster")
        assert entry["type"] == "raster"
        assert entry["id"] == 0
        assert len(proj["files"]) == 1

    def test_add_file_auto_detect(self, sample_raster):
        """Auto-detect file type from extension."""
        proj = project_mod.create_project()
        entry = project_mod.add_file(proj, sample_raster)
        assert entry["type"] == "raster"

    def test_remove_file(self, sample_raster):
        """Remove a file by index."""
        proj = project_mod.create_project()
        project_mod.add_file(proj, sample_raster)
        assert len(proj["files"]) == 1

        removed = project_mod.remove_file(proj, 0)
        assert removed["type"] == "raster"
        assert len(proj["files"]) == 0

        with pytest.raises(IndexError):
            project_mod.remove_file(proj, 0)


# ═══════════════════════════════════════════════════════════════════════════
# core/session.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSession:
    """Tests for session state management."""

    def test_session_empty(self):
        """New session has no project."""
        sess = session_mod.Session()
        assert not sess.has_project()

    def test_session_set_project(self):
        """Load project into session."""
        sess = session_mod.Session()
        proj = project_mod.create_project(name="test")
        sess.set_project(proj, "/tmp/test.json")
        assert sess.has_project()
        assert sess.project["name"] == "test"

    def test_session_get_project_no_project(self):
        """RuntimeError when no project is loaded."""
        sess = session_mod.Session()
        with pytest.raises(RuntimeError, match="No project loaded"):
            sess.get_project()

    def test_session_snapshot_and_undo(self):
        """Snapshot, mutate, undo restores original state."""
        sess = session_mod.Session()
        proj = project_mod.create_project(name="original")
        sess.set_project(proj)

        sess.snapshot("rename")
        sess.project["name"] = "modified"
        assert sess.project["name"] == "modified"

        desc = sess.undo()
        assert desc == "rename"
        assert sess.project["name"] == "original"

    def test_session_redo(self):
        """Undo then redo restores the modification."""
        sess = session_mod.Session()
        proj = project_mod.create_project(name="original")
        sess.set_project(proj)

        sess.snapshot("rename")
        sess.project["name"] = "modified"

        sess.undo()
        assert sess.project["name"] == "original"

        sess.redo()
        assert sess.project["name"] == "modified"

    def test_session_undo_empty(self):
        """RuntimeError when nothing to undo."""
        sess = session_mod.Session()
        proj = project_mod.create_project()
        sess.set_project(proj)
        with pytest.raises(RuntimeError, match="Nothing to undo"):
            sess.undo()

    def test_session_redo_empty(self):
        """RuntimeError when nothing to redo."""
        sess = session_mod.Session()
        proj = project_mod.create_project()
        sess.set_project(proj)
        with pytest.raises(RuntimeError, match="Nothing to redo"):
            sess.redo()

    def test_session_history(self):
        """Verify history entries after multiple snapshots."""
        sess = session_mod.Session()
        proj = project_mod.create_project()
        sess.set_project(proj)

        sess.snapshot("step 1")
        sess.snapshot("step 2")
        sess.snapshot("step 3")

        hist = sess.history()
        assert len(hist) == 3
        assert hist[0]["description"] == "step 1"
        assert hist[2]["description"] == "step 3"

    def test_session_max_undo_limit(self):
        """Undo stack respects MAX_UNDO."""
        sess = session_mod.Session()
        sess.MAX_UNDO = 5
        proj = project_mod.create_project()
        sess.set_project(proj)

        for i in range(10):
            sess.snapshot(f"step {i}")

        assert len(sess._undo_stack) == 5
        assert sess._undo_stack[0]["description"] == "step 5"

    def test_session_save_and_load(self, tmp_dir):
        """Save session to disk, verify file exists and loads."""
        sess = session_mod.Session()
        proj = project_mod.create_project(name="saveable")
        path = os.path.join(tmp_dir, "session.json")
        sess.set_project(proj, path)

        saved_path = sess.save_session()
        assert os.path.isfile(saved_path)

        loaded = project_mod.open_project(saved_path)
        assert loaded["name"] == "saveable"


# ═══════════════════════════════════════════════════════════════════════════
# core/raster.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRaster:
    """Tests for raster operations."""

    def test_get_raster_info(self, sample_raster):
        """Verify info dict keys from a real GeoTIFF."""
        info = raster_mod.get_raster_info(sample_raster)
        assert isinstance(info, dict)
        # Should have standard raster metadata
        assert "crs" in info or "driver" in info

    def test_get_raster_info_missing_file(self):
        """FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            raster_mod.get_raster_info("/nonexistent/file.tif")

    def test_get_raster_stats(self, sample_raster):
        """Verify statistics for a synthetic raster."""
        stats = raster_mod.get_raster_stats(sample_raster, band=1)
        assert stats["band"] == 1
        assert stats["min"] is not None
        assert stats["max"] is not None
        assert stats["mean"] is not None
        assert 0.0 <= stats["min"] <= stats["max"] <= 1.0  # random [0,1) data
        assert stats["valid_pixels"] > 0

    def test_get_raster_stats_bad_band(self, sample_raster):
        """ValueError for out-of-range band."""
        with pytest.raises(ValueError, match="Band .* out of range"):
            raster_mod.get_raster_stats(sample_raster, band=10)

    def test_tile_raster(self, sample_raster, tmp_dir):
        """Verify tiles are created from a raster."""
        output_dir = os.path.join(tmp_dir, "tiles")
        result = raster_mod.tile_raster(
            sample_raster, output_dir, tile_size=32, overlap=0
        )
        assert result["tile_count"] > 0
        assert os.path.isdir(output_dir)

    def test_vectorize_raster(self, tmp_dir):
        """Vectorize a classified raster."""
        # Create a simple classified raster
        raster_path = os.path.join(tmp_dir, "classified.tif")
        data = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype=np.uint8)
        transform = from_bounds(0, 0, 3, 3, 3, 3)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=3,
            width=3,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        output = os.path.join(tmp_dir, "polygons.geojson")
        result = raster_mod.vectorize_raster(raster_path, output)
        assert os.path.isfile(output)
        assert result["feature_count"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# core/vector.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVector:
    """Tests for vector operations."""

    def test_get_vector_info(self, sample_vector):
        """Verify info dict from a GeoJSON file."""
        info = vector_mod.get_vector_info(sample_vector)
        assert isinstance(info, dict)
        assert info.get("feature_count", 0) == 3

    def test_get_vector_info_missing_file(self):
        """FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            vector_mod.get_vector_info("/nonexistent/file.geojson")

    def test_rasterize_vector(self, sample_vector, sample_raster, tmp_dir):
        """Rasterize a vector using a template raster."""
        output = os.path.join(tmp_dir, "rasterized.tif")
        result = vector_mod.rasterize_vector(
            sample_vector, sample_raster, output, attribute="value"
        )
        assert os.path.isfile(output)
        assert result["width"] > 0
        assert result["height"] > 0

    def test_rasterize_vector_missing_file(self, sample_raster, tmp_dir):
        """FileNotFoundError for missing vector file."""
        with pytest.raises(FileNotFoundError):
            vector_mod.rasterize_vector(
                "/nonexistent.geojson",
                sample_raster,
                os.path.join(tmp_dir, "out.tif"),
            )


# ═══════════════════════════════════════════════════════════════════════════
# core/data.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestData:
    """Tests for data operations."""

    def test_parse_bbox_valid(self):
        """Parse a valid bbox string."""
        bbox = data_mod.parse_bbox("-84.0,35.9,-83.9,36.0")
        assert bbox == (-84.0, 35.9, -83.9, 36.0)

    def test_parse_bbox_invalid(self):
        """ValueError for invalid bbox formats."""
        with pytest.raises(ValueError, match="Invalid bbox"):
            data_mod.parse_bbox("not,a,bbox")
        with pytest.raises(ValueError, match="Invalid bbox"):
            data_mod.parse_bbox("1.0,2.0")

    def test_list_sources(self):
        """Verify source list structure."""
        sources = data_mod.list_sources()
        assert len(sources) > 0
        assert all("name" in s for s in sources)
        assert all("description" in s for s in sources)

    def test_list_sources_contains_naip(self):
        """Verify NAIP is in the source list."""
        sources = data_mod.list_sources()
        names = [s["name"] for s in sources]
        assert "naip" in names


# ═══════════════════════════════════════════════════════════════════════════
# core/segment.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSegment:
    """Tests for segmentation operations (no model inference)."""

    def test_list_sam_models(self):
        """Verify SAM model list structure."""
        models = segment_mod.list_sam_models()
        assert len(models) > 0
        assert all("model_id" in m for m in models)
        assert any("sam-vit-huge" in m["model_id"] for m in models)

    def test_list_architectures(self):
        """Verify architecture list."""
        archs = segment_mod.list_architectures()
        assert len(archs) > 0
        names = [a["name"] for a in archs]
        assert "unet" in names
        assert "deeplabv3" in names

    def test_run_sam_missing_file(self):
        """FileNotFoundError for missing raster."""
        with pytest.raises(FileNotFoundError):
            segment_mod.run_sam(
                "/nonexistent/image.tif", "/tmp/out.tif"
            )


# ═══════════════════════════════════════════════════════════════════════════
# core/detect.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDetect:
    """Tests for detection operations (no model inference)."""

    def test_list_models(self):
        """Verify detection model list."""
        models = detect_mod.list_models()
        assert len(models) > 0
        names = [m["name"] for m in models]
        assert "maskrcnn_resnet50_fpn" in names

    def test_list_input_formats(self):
        """Verify input format list."""
        formats = detect_mod.list_input_formats()
        assert len(formats) > 0
        names = [f["name"] for f in formats]
        assert "coco" in names
        assert "yolo" in names

    def test_run_detection_missing_file(self):
        """FileNotFoundError for missing raster or model."""
        with pytest.raises(FileNotFoundError):
            detect_mod.run_detection(
                "/nonexistent/image.tif",
                "/nonexistent/model.pth",
                num_classes=2,
            )


# ═══════════════════════════════════════════════════════════════════════════
# core/change.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestChange:
    """Tests for change detection operations (no model inference)."""

    def test_list_methods(self):
        """Verify change detection method list."""
        methods = change_mod.list_methods()
        assert len(methods) > 0
        names = [m["name"] for m in methods]
        assert "changestar" in names
        assert "anychange" in names

    def test_detect_changes_missing_file(self):
        """FileNotFoundError for missing images."""
        with pytest.raises(FileNotFoundError):
            change_mod.detect_changes(
                "/nonexistent/img1.tif",
                "/nonexistent/img2.tif",
                "/tmp/change.tif",
            )
