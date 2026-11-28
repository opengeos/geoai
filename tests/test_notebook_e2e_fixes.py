"""Regression tests for bugs found during end-to-end example notebook testing."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from unittest import mock

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_add_geometric_properties_handles_degenerate_collinear_polygon() -> None:
    """add_geometric_properties must not crash when minimum_rotated_rectangle is a LineString."""
    from geoai.utils.vector import add_geometric_properties

    # Collinear vertices form a degenerate Polygon whose minimum_rotated_rectangle is a LineString
    degenerate = Polygon([(0, 0), (5, 0), (10, 0), (0, 0)])
    normal = Polygon([(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)])
    gdf = gpd.GeoDataFrame(geometry=[degenerate, normal], crs="EPSG:32610")

    result = add_geometric_properties(gdf)

    assert len(result) == 2
    assert (
        np.isnan(result.loc[0, "major_length_m"])
        or result.loc[0, "major_length_m"] is None
    )
    assert result.loc[1, "major_length_m"] == pytest.approx(10.0, rel=1e-3)
    assert result.loc[1, "minor_length_m"] == pytest.approx(5.0, rel=1e-3)


def test_stack_bands_rasterio_fallback_without_gdal_cli(tmp_path: pathlib.Path) -> None:
    """stack_bands falls back cleanly to rasterio when gdalbuildvrt/gdal_translate are missing."""
    from geoai.utils.raster import stack_bands

    rgb_path = tmp_path / "rgb.tif"
    hag_path = tmp_path / "hag.tif"
    out_path = tmp_path / "stacked.tif"

    transform = from_bounds(0, 0, 100, 100, 10, 10)
    rgb = np.stack(
        [
            np.full((10, 10), 10, dtype=np.uint8),
            np.full((10, 10), 20, dtype=np.uint8),
            np.full((10, 10), 30, dtype=np.uint8),
        ]
    )
    hag = np.full((1, 10, 10), 42.5, dtype=np.float32)
    hag[0, 0, 0] = np.nan

    with rasterio.open(
        rgb_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=3,
        dtype="uint8",
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(rgb)

    with rasterio.open(
        hag_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(hag)

    with (
        mock.patch(
            "geoai.utils.raster.subprocess.run",
            side_effect=FileNotFoundError(
                2, "No such file or directory", "gdalbuildvrt"
            ),
        ),
        mock.patch("geoai.utils.raster.shutil.which", return_value=None),
    ):
        result = stack_bands(
            input_files=[str(rgb_path), str(hag_path)],
            output_file=str(out_path),
            dtype="Byte",
            overwrite=True,
        )

    assert result == str(out_path)
    assert out_path.exists()
    with rasterio.open(out_path) as src:
        assert src.count == 4
        assert src.dtypes == ("uint8", "uint8", "uint8", "uint8")
        assert src.width == 10
        assert src.height == 10
        assert src.read(1)[5, 5] == 10
        assert src.read(2)[5, 5] == 20
        assert src.read(3)[5, 5] == 30
        assert src.read(4)[5, 5] == 42
        assert src.read(4)[0, 0] == 0


def test_ensure_spacy_model_restores_path_and_handles_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_spacy_model prepends sys.executable dir, passes --python for uv, and catches SystemExit."""
    pytest.importorskip("spacy")
    from geoai import caption

    seen_calls = []

    def fake_import(name: str):
        raise ImportError(name)

    def fake_download(*args):
        seen_calls.append((args, os.environ.get("PATH", "")))
        raise SystemExit(2)

    monkeypatch.setattr(caption.importlib, "import_module", fake_import)
    monkeypatch.setattr(caption, "download", fake_download)
    monkeypatch.setattr(
        caption.shutil,
        "which",
        lambda cmd: "/usr/bin/uv" if cmd == "uv" else None,
    )
    orig_path = "/usr/bin"
    monkeypatch.setenv("PATH", orig_path)

    with pytest.raises(RuntimeError, match="Failed to download spaCy model"):
        caption.ensure_spacy_model("en_core_web_sm", auto_download=True)

    assert os.environ["PATH"] == orig_path
    assert len(seen_calls) == 1
    args, path_during = seen_calls[0]
    assert args == ("en_core_web_sm", False, False, None, "--python", sys.executable)
    assert os.path.dirname(sys.executable) in path_during.split(os.pathsep)


def test_example_notebooks_python_cells_parse_cleanly() -> None:
    """All code cells in example notebooks (excluding IPython magics/shell) must parse as valid Python."""
    examples_dir = REPO_ROOT / "docs" / "examples"
    syntax_errors = []
    for nb_path in sorted(examples_dir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            lines = "".join(cell.get("source", [])).splitlines()
            if lines and lines[0].lstrip().startswith("%%"):
                continue
            # Filter out IPython line magics and shell escapes (including backslash continuations)
            cleaned = []
            in_shell_continuation = False
            for line in lines:
                stripped = line.lstrip()
                if in_shell_continuation:
                    cleaned.append("pass" if not line.startswith((" ", "\t")) else "")
                    in_shell_continuation = line.rstrip().endswith("\\")
                    continue
                if stripped.startswith(("!", "%", "?")) or line.rstrip().endswith("?"):
                    indent = line[: len(line) - len(stripped)]
                    cleaned.append(f"{indent}pass")
                    in_shell_continuation = line.rstrip().endswith("\\")
                else:
                    cleaned.append(line)
            code = "\n".join(cleaned)
            try:
                compile(code, f"{nb_path.name}:cell[{idx}]", "exec")
            except SyntaxError as exc:
                syntax_errors.append(f"{nb_path.name}:cell[{idx}]: {exc}")

    assert not syntax_errors, "Syntax errors in example notebooks:\n" + "\n".join(
        syntax_errors
    )


def test_requirements_includes_termcolor_for_dinov3() -> None:
    """requirements.txt includes termcolor required by facebookresearch/dinov3 hub loader."""
    req_lines = [
        line.strip().lower()
        for line in (REPO_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any(
        line == "termcolor" or line.startswith("termcolor") for line in req_lines
    )
