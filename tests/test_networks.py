"""Tests for road network simplification helpers."""

from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from geoai.networks import simplify_road_network


def test_simplify_road_network_calls_neatnet_neatify(monkeypatch):
    """simplify_road_network should delegate GeoDataFrame inputs to neatnet.neatify."""
    roads = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
        crs="EPSG:3857",
    )
    simplified = roads.copy()
    simplified["_status"] = ["original", "original"]
    buildings = gpd.GeoSeries(
        [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])], crs=roads.crs
    )
    calls = {}

    def fake_neatify(gdf, **kwargs):
        calls["gdf"] = gdf
        calls["kwargs"] = kwargs
        return simplified

    monkeypatch.setitem(
        __import__("sys").modules, "neatnet", SimpleNamespace(neatify=fake_neatify)
    )

    result = simplify_road_network(
        roads,
        exclusion_mask=buildings,
        artifact_threshold=8,
    )

    assert result is simplified
    assert calls["gdf"] is roads
    assert calls["kwargs"]["exclusion_mask"] is buildings
    assert calls["kwargs"]["artifact_threshold"] == 8


def test_simplify_road_network_writes_output(monkeypatch, tmp_path):
    """simplify_road_network should write outputs when an output path is supplied."""
    roads = gpd.GeoDataFrame(
        {"name": ["A"]}, geometry=[LineString([(0, 0), (1, 0)])], crs="EPSG:3857"
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "neatnet",
        SimpleNamespace(neatify=lambda gdf, **kwargs: roads.copy()),
    )
    output = tmp_path / "simplified.gpkg"

    result = simplify_road_network(roads, output=output)

    assert output.exists()
    assert isinstance(result, gpd.GeoDataFrame)


def test_simplify_road_network_requires_neatnet(monkeypatch):
    """A clear installation hint should be raised when neatnet is unavailable."""
    monkeypatch.delitem(__import__("sys").modules, "neatnet", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "neatnet":
            raise ImportError("missing neatnet")
        return original_import(name, *args, **kwargs)

    original_import = __import__("builtins").__import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="pip install geoai-py\[networks\]"):
        simplify_road_network(gpd.GeoDataFrame(geometry=[]))
