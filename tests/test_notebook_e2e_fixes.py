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


def test_stack_bands_rasterio_fallback_preserves_resolution_nearest_nodata_and_no_crs(
    tmp_path: pathlib.Path,
) -> None:
    """stack_bands fallback preserves exact resolution, nearest resampling, nodata, and CRS-less grids."""
    from geoai.utils.raster import stack_bands

    # 1. Exact resolution + nearest resampling (categorical values 0 and 100) + nodata
    cat_path = tmp_path / "cat.tif"
    f32_path = tmp_path / "f32.tif"
    out_path = tmp_path / "stacked_cat.tif"

    transform = from_bounds(0, 0, 100, 100, 10, 10)
    cat = np.zeros((1, 10, 10), dtype=np.float32)
    cat[0, :, 5:] = 100.0
    cat[0, 0, 0] = -9999.0

    with rasterio.open(
        cat_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(cat)

    with rasterio.open(
        f32_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(cat)

    with (
        mock.patch(
            "geoai.utils.raster.subprocess.run",
            side_effect=FileNotFoundError(
                2, "No such file or directory", "gdalbuildvrt"
            ),
        ),
        mock.patch("geoai.utils.raster.shutil.which", return_value=None),
    ):
        stack_bands(
            input_files=[str(cat_path), str(f32_path)],
            output_file=str(out_path),
            resolution=3.0,
            overwrite=True,
        )

    with rasterio.open(out_path) as src:
        assert src.res == pytest.approx((3.0, 3.0))
        assert src.nodata == -9999.0
        vals = set(np.unique(src.read(1)).tolist())
        assert vals.issubset({-9999.0, 0.0, 100.0})

    # 2. CRS-less aligned inputs do not raise CRSError
    nocrs_a = tmp_path / "nocrs_a.tif"
    nocrs_b = tmp_path / "nocrs_b.tif"
    nocrs_out = tmp_path / "nocrs_stacked.tif"
    for p, val in ((nocrs_a, 7), (nocrs_b, 9)):
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="uint8",
            transform=from_bounds(0, 0, 8, 8, 8, 8),
        ) as dst:
            dst.write(np.full((1, 8, 8), val, dtype=np.uint8))

    with (
        mock.patch(
            "geoai.utils.raster.subprocess.run",
            side_effect=FileNotFoundError(
                2, "No such file or directory", "gdalbuildvrt"
            ),
        ),
        mock.patch("geoai.utils.raster.shutil.which", return_value=None),
    ):
        stack_bands(
            input_files=[str(nocrs_a), str(nocrs_b)],
            output_file=str(nocrs_out),
            overwrite=True,
        )

    with rasterio.open(nocrs_out) as src:
        assert src.count == 2
        assert src.crs is None
        assert int(src.read(1)[0, 0]) == 7
        assert int(src.read(2)[0, 0]) == 9


def test_ensure_spacy_model_matches_spacy_installer_and_restores_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_spacy_model only forwards --python for uv, preserves absent PATH, and wraps SystemExit."""
    pytest.importorskip("spacy")
    import importlib

    from geoai import caption

    spacy_dl_mod = importlib.import_module("spacy.cli.download")

    def fake_import(name: str):
        if name == "en_core_web_sm":
            raise ImportError(name)
        return spacy_dl_mod

    # Case 1: spaCy selects uv -> --python sys.executable is forwarded, absent PATH stays absent
    seen_uv = []

    def fake_download_uv(model, direct=False, sdist=False, custom_url=None, *pip_args):
        seen_uv.append(
            ((model, direct, sdist, custom_url, pip_args), os.environ.get("PATH"))
        )
        raise SystemExit(2)

    monkeypatch.setattr(caption.importlib, "import_module", fake_import)
    monkeypatch.setattr(caption, "download", fake_download_uv)
    monkeypatch.setattr(
        spacy_dl_mod, "_get_pip_install_cmd", lambda: ["uv", "pip", "install"]
    )
    monkeypatch.delenv("PATH", raising=False)

    with pytest.raises(RuntimeError, match="Failed to download spaCy model"):
        caption.ensure_spacy_model("en_core_web_sm", auto_download=True)

    assert "PATH" not in os.environ
    assert len(seen_uv) == 1
    call_args, path_during = seen_uv[0]
    assert call_args == (
        "en_core_web_sm",
        False,
        False,
        None,
        ("--python", sys.executable),
    )
    assert os.path.dirname(sys.executable) in (path_during or "").split(os.pathsep)

    # Case 2: spaCy selects pip -> --python must NOT be forwarded
    seen_pip = []

    def fake_download_pip(model, direct=False, sdist=False, custom_url=None, *pip_args):
        seen_pip.append((model, direct, sdist, custom_url, pip_args))

    monkeypatch.setattr(caption, "download", fake_download_pip)
    monkeypatch.setattr(
        spacy_dl_mod,
        "_get_pip_install_cmd",
        lambda: [sys.executable, "-m", "pip", "install"],
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    caption.ensure_spacy_model("en_core_web_sm", auto_download=True)

    assert os.environ["PATH"] == "/usr/bin"
    assert seen_pip == [("en_core_web_sm", False, False, None, ())]


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
