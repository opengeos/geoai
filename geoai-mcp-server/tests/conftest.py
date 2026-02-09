"""Pytest configuration and fixtures for GeoAI MCP Server tests."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add the src directory to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_input_dir() -> Generator[Path, None, None]:
    """Create a temporary input directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_input_dir: Path, temp_output_dir: Path):
    """Create a sample configuration for testing."""
    from geoai_mcp_server.config import GeoAIConfig

    return GeoAIConfig(
        input_dir=temp_input_dir,
        output_dir=temp_output_dir,
        temp_dir=temp_output_dir / "temp",
        log_level="DEBUG",
        max_file_size_mb=100,
        max_processing_time_seconds=60,
    )


@pytest.fixture
def sample_geotiff(temp_input_dir: Path) -> Path:
    """Create a minimal sample GeoTIFF file for testing."""
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Create a simple 100x100 RGB image
        data = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)

        filepath = temp_input_dir / "sample.tif"

        transform = from_bounds(-122.5, 37.7, -122.4, 37.8, 100, 100)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        return filepath
    except ImportError:
        pytest.skip("rasterio not installed")


@pytest.fixture
def sample_geojson(temp_input_dir: Path) -> Path:
    """Create a sample GeoJSON file for testing."""
    import json

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1, "class": "building"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.45, 37.75],
                            [-122.44, 37.75],
                            [-122.44, 37.76],
                            [-122.45, 37.76],
                            [-122.45, 37.75],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": 2, "class": "building"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.43, 37.74],
                            [-122.42, 37.74],
                            [-122.42, 37.75],
                            [-122.43, 37.75],
                            [-122.43, 37.74],
                        ]
                    ],
                },
            },
        ],
    }

    filepath = temp_input_dir / "buildings.geojson"
    with open(filepath, "w") as f:
        json.dump(geojson, f)

    return filepath


@pytest.fixture
def mock_geoai_modules(monkeypatch):
    """Mock GeoAI modules for testing without actual dependencies."""
    from unittest.mock import MagicMock

    # Create mock modules
    mock_sam = MagicMock()
    mock_segment = MagicMock()
    mock_utils = MagicMock()
    mock_download = MagicMock()
    mock_change_detection = MagicMock()
    mock_dinov3 = MagicMock()
    mock_canopy = MagicMock()
    mock_moondream = MagicMock()
    mock_classify = MagicMock()
    mock_train = MagicMock()

    # Set up mock return values
    mock_sam.SamGeo.return_value.generate.return_value = []
    mock_segment.segment_with_text.return_value = []
    mock_download.download_naip.return_value = []

    # Patch the import mechanism
    modules = {
        "geoai.sam": mock_sam,
        "geoai.segment": mock_segment,
        "geoai.utils": mock_utils,
        "geoai.download": mock_download,
        "geoai.change_detection": mock_change_detection,
        "geoai.dinov3": mock_dinov3,
        "geoai.canopy": mock_canopy,
        "geoai.moondream": mock_moondream,
        "geoai.classify": mock_classify,
        "geoai.train": mock_train,
    }

    for mod_name, mock_mod in modules.items():
        monkeypatch.setitem(sys.modules, mod_name, mock_mod)

    return modules
