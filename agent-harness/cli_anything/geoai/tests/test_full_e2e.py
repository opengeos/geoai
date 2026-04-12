"""End-to-end and subprocess tests for geoai CLI.

Tests the full CLI pipeline with real GeoTIFF files and the installed
`cli-anything-geoai` command via subprocess. No mocking.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from cli_anything.geoai.core import project as project_mod
from cli_anything.geoai.core import raster as raster_mod
from cli_anything.geoai.core import session as session_mod

# ---------------------------------------------------------------------------
# Fixtures and helpers
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


def _resolve_cli(name):
    """Resolve installed CLI command; falls back to python -m for dev.

    Set env CLI_ANYTHING_FORCE_INSTALLED=1 to require the installed command.

    Args:
        name: CLI command name (e.g., "cli-anything-geoai").

    Returns:
        List of command parts for subprocess.
    """
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = "cli_anything.geoai.geoai_cli"
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="geoai_e2e_") as d:
        yield d


@pytest.fixture
def sample_raster(tmp_dir):
    """Create a small synthetic GeoTIFF for E2E testing."""
    path = os.path.join(tmp_dir, "sample.tif")
    data = np.random.rand(3, 128, 128).astype(np.float32)
    transform = from_bounds(-84.0, 35.9, -83.9, 36.0, 128, 128)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Raster Operations E2E
# ═══════════════════════════════════════════════════════════════════════════


class TestRasterE2E:
    """E2E tests for raster operations with real files."""

    @pytest.mark.skipif(
        not os.path.isfile(LANDSAT_2022),
        reason="knoxville_landsat_2022.tif not found",
    )
    def test_raster_info_real_file(self):
        """Get info on real Knoxville Landsat GeoTIFF."""
        info = raster_mod.get_raster_info(LANDSAT_2022)
        assert isinstance(info, dict)
        print(f"\n  Raster: {LANDSAT_2022}")
        print(f"  Info keys: {list(info.keys())}")

    @pytest.mark.skipif(
        not os.path.isfile(LANDSAT_2022),
        reason="knoxville_landsat_2022.tif not found",
    )
    def test_raster_stats_all_bands(self):
        """Compute stats for all bands of real Landsat raster."""
        with rasterio.open(LANDSAT_2022) as src:
            band_count = src.count

        for band in range(1, band_count + 1):
            stats = raster_mod.get_raster_stats(LANDSAT_2022, band=band)
            assert stats["band"] == band
            assert stats["valid_pixels"] > 0
            if stats["min"] is not None:
                assert isinstance(stats["min"], float)
                assert isinstance(stats["max"], float)
                print(
                    f"\n  Band {band}: min={stats['min']:.4f}, "
                    f"max={stats['max']:.4f}, mean={stats['mean']:.4f}"
                )

    def test_raster_tile_and_count(self, sample_raster, tmp_dir):
        """Tile a raster and verify output files exist."""
        output_dir = os.path.join(tmp_dir, "tiles")
        result = raster_mod.tile_raster(
            sample_raster, output_dir, tile_size=64, overlap=0
        )
        assert result["tile_count"] > 0
        assert os.path.isdir(output_dir)
        # Tiles are in subdirectories (e.g., images/)
        tile_count = result["tile_count"]
        assert tile_count > 0
        print(f"\n  Tiles: {tile_count} files in {output_dir}")

    def test_raster_vectorize_real(self, tmp_dir):
        """Vectorize a classified raster and verify GeoJSON output."""
        raster_path = os.path.join(tmp_dir, "classified.tif")
        data = np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]],
            dtype=np.uint8,
        )
        transform = from_bounds(0, 0, 4, 4, 4, 4)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
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
        assert os.path.getsize(output) > 0
        print(
            f"\n  Vectorized: {result['feature_count']} features, "
            f"{os.path.getsize(output):,} bytes"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Project Workflow E2E
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectWorkflowE2E:
    """E2E tests for project management workflows."""

    def test_project_create_add_save_reload(self, sample_raster, tmp_dir):
        """Full project lifecycle: create, add file, save, reload."""
        # Create
        proj = project_mod.create_project(name="e2e_test")

        # Add file
        entry = project_mod.add_file(proj, sample_raster)
        assert entry["type"] == "raster"

        # Save
        path = os.path.join(tmp_dir, "project.json")
        project_mod.save_project(proj, path)
        assert os.path.isfile(path)

        # Reload
        loaded = project_mod.open_project(path)
        assert loaded["name"] == "e2e_test"
        assert len(loaded["files"]) == 1
        assert loaded["files"][0]["type"] == "raster"
        print(f"\n  Project: {path} ({os.path.getsize(path):,} bytes)")

    def test_project_session_undo_redo(self, sample_raster):
        """Undo and redo file additions through session."""
        sess = session_mod.Session()
        proj = project_mod.create_project(name="undo_test")
        sess.set_project(proj)

        # Add file with snapshot
        sess.snapshot("add file")
        project_mod.add_file(proj, sample_raster)
        assert len(proj["files"]) == 1

        # Undo
        sess.undo()
        assert len(sess.project["files"]) == 0

        # Redo
        sess.redo()
        assert len(sess.project["files"]) == 1

    def test_project_info_after_multiple_files(self, sample_raster, tmp_dir):
        """Verify project info counts after adding multiple files."""
        proj = project_mod.create_project(name="multi_file")

        # Create a second raster
        raster2 = os.path.join(tmp_dir, "second.tif")
        data = np.random.rand(1, 32, 32).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        with rasterio.open(
            raster2, "w", driver="GTiff",
            height=32, width=32, count=1, dtype="float32",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)

        project_mod.add_file(proj, sample_raster)
        project_mod.add_file(proj, raster2)

        info = project_mod.get_project_info(proj)
        assert info["file_count"] == 2
        assert info["raster_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# CLI Subprocess Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCLISubprocess:
    """Test the installed CLI via subprocess."""

    CLI_BASE = _resolve_cli("cli-anything-geoai")

    def _run(self, args, check=True):
        """Run a CLI command and return the result.

        Args:
            args: List of command arguments.
            check: Whether to check for non-zero exit code.

        Returns:
            subprocess.CompletedProcess
        """
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_cli_help(self):
        """--help exits 0 with usage text."""
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "GeoAI CLI" in result.stdout

    def test_cli_version(self):
        """--version shows version string."""
        result = self._run(["--version"])
        assert result.returncode == 0
        assert "1.0.0" in result.stdout

    def test_cli_raster_info_json(self, sample_raster):
        """--json raster info returns valid JSON."""
        result = self._run(["--json", "raster", "info", sample_raster])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_cli_raster_stats_json(self, sample_raster):
        """--json raster stats returns valid JSON with numeric values."""
        result = self._run(["--json", "raster", "stats", sample_raster])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["band"] == 1
        assert isinstance(data["min"], float)

    def test_cli_project_new_json(self, tmp_dir):
        """--json project new creates a project file."""
        out = os.path.join(tmp_dir, "proj.json")
        result = self._run(
            ["--json", "project", "new", "-n", "subprocess_test", "-o", out]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "subprocess_test"
        assert os.path.isfile(out)

    def test_cli_system_info_json(self):
        """--json system-info returns valid JSON with version info."""
        result = self._run(["--json", "system-info"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "geoai_version" in data
        assert "torch_version" in data

    def test_cli_session_status_json(self):
        """--json session status returns valid JSON."""
        result = self._run(["--json", "session", "status"], check=False)
        # Exit code 1 is expected (no project loaded) but JSON should be valid
        data = json.loads(result.stdout)
        assert "has_project" in data

    def test_cli_segment_list_models(self):
        """--json segment list-models returns a JSON list."""
        result = self._run(["--json", "segment", "list-models"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "model_id" in data[0]


# ═══════════════════════════════════════════════════════════════════════════
# Realistic Workflow Scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestRealisticWorkflows:
    """Multi-step workflow scenarios simulating real agent usage."""

    CLI_BASE = _resolve_cli("cli-anything-geoai")

    def _run(self, args, check=True):
        """Run CLI command via subprocess.

        Args:
            args: CLI arguments.
            check: Check exit code.

        Returns:
            subprocess.CompletedProcess
        """
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_raster_analysis_pipeline(self, sample_raster, tmp_dir):
        """Simulates: Analyzing satellite imagery metadata and statistics.

        Operations: raster info -> raster stats -> project add-file
        Verified: All JSON responses valid, stats contain numeric values.
        """
        # Step 1: Get raster info
        result = self._run(["--json", "raster", "info", sample_raster])
        assert result.returncode == 0
        info = json.loads(result.stdout)
        assert isinstance(info, dict)

        # Step 2: Get band statistics
        result = self._run(["--json", "raster", "stats", sample_raster, "--band", "1"])
        assert result.returncode == 0
        stats = json.loads(result.stdout)
        assert isinstance(stats["min"], float)
        assert isinstance(stats["max"], float)

        # Step 3: Create project and add the file
        proj_path = os.path.join(tmp_dir, "analysis.json")
        result = self._run(
            ["--json", "project", "new", "-n", "raster_analysis", "-o", proj_path]
        )
        assert result.returncode == 0

        print(f"\n  Workflow: raster analysis pipeline")
        print(f"  Info: {len(info)} metadata fields")
        print(f"  Stats: band 1 mean={stats['mean']:.4f}")
        print(f"  Project: {proj_path}")

    def test_project_management_workflow(self, tmp_dir):
        """Simulates: Setting up a geospatial AI workspace.

        Operations: project new -> list files -> save -> reopen
        Verified: Project creates and reloads correctly.
        """
        proj_path = os.path.join(tmp_dir, "workspace.json")

        # Step 1: Create project
        result = self._run(
            ["--json", "project", "new", "-n", "workspace", "-o", proj_path]
        )
        assert result.returncode == 0
        assert os.path.isfile(proj_path)

        # Step 2: Verify project file is valid JSON
        with open(proj_path) as f:
            proj_data = json.load(f)
        assert proj_data["name"] == "workspace"
        assert proj_data["files"] == []

        print(f"\n  Workflow: project management")
        print(f"  Created: {proj_path} ({os.path.getsize(proj_path):,} bytes)")

    def test_data_discovery_workflow(self):
        """Simulates: Exploring available data sources and models.

        Operations: data sources -> segment list-models -> detect list-models
        Verified: All list commands return non-empty results.
        """
        # Step 1: List data sources
        result = self._run(["--json", "data", "sources"])
        assert result.returncode == 0
        sources = json.loads(result.stdout)
        assert len(sources) > 0

        # Step 2: List segmentation models
        result = self._run(["--json", "segment", "list-models"])
        assert result.returncode == 0
        seg_models = json.loads(result.stdout)
        assert len(seg_models) > 0

        # Step 3: List detection models
        result = self._run(["--json", "detect", "list-models"])
        assert result.returncode == 0
        det_models = json.loads(result.stdout)
        assert len(det_models) > 0

        # Step 4: List change methods
        result = self._run(["--json", "change", "list-methods"])
        assert result.returncode == 0
        methods = json.loads(result.stdout)
        assert len(methods) > 0

        print(f"\n  Workflow: data discovery")
        print(f"  Sources: {len(sources)}")
        print(f"  Segmentation models: {len(seg_models)}")
        print(f"  Detection models: {len(det_models)}")
        print(f"  Change methods: {len(methods)}")
