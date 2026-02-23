#!/usr/bin/env python

"""Tests for Pydantic models in `geoai.agents` module."""

import importlib.util
import os
import sys
import unittest

from pydantic import ValidationError

# Load model modules directly from file to avoid the agents/__init__.py chain
# which requires strands-agents and opentelemetry dependencies.
_agents_dir = os.path.join(os.path.dirname(__file__), os.pardir, "geoai", "agents")

_catalog_spec = importlib.util.spec_from_file_location(
    "catalog_models", os.path.join(_agents_dir, "catalog_models.py")
)
_catalog_models = importlib.util.module_from_spec(_catalog_spec)
_catalog_spec.loader.exec_module(_catalog_models)

_stac_spec = importlib.util.spec_from_file_location(
    "stac_models", os.path.join(_agents_dir, "stac_models.py")
)
_stac_models = importlib.util.module_from_spec(_stac_spec)
_stac_spec.loader.exec_module(_stac_models)

CatalogDatasetInfo = _catalog_models.CatalogDatasetInfo
CatalogSearchResult = _catalog_models.CatalogSearchResult
CatalogLocationInfo = _catalog_models.LocationInfo

STACCollectionInfo = _stac_models.STACCollectionInfo
STACAssetInfo = _stac_models.STACAssetInfo
STACItemInfo = _stac_models.STACItemInfo
STACSearchResult = _stac_models.STACSearchResult
STACLocationInfo = _stac_models.LocationInfo


class TestCatalogDatasetInfo(unittest.TestCase):
    """Tests for the CatalogDatasetInfo Pydantic model."""

    def test_required_fields(self):
        """Test construction with only required fields."""
        info = CatalogDatasetInfo(id="landsat-8", title="Landsat 8")
        self.assertEqual(info.id, "landsat-8")
        self.assertEqual(info.title, "Landsat 8")

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        info = CatalogDatasetInfo(id="test", title="Test")
        self.assertIsNone(info.type)
        self.assertIsNone(info.provider)
        self.assertIsNone(info.description)
        self.assertIsNone(info.keywords)
        self.assertIsNone(info.url)
        self.assertIsNone(info.license)

    def test_all_fields(self):
        """Test construction with all fields populated."""
        info = CatalogDatasetInfo(
            id="sentinel-2",
            title="Sentinel-2 L2A",
            type="image_collection",
            provider="ESA",
            description="Sentinel-2 data",
            keywords="satellite,optical",
            start_date="2015-06-23",
            end_date="2026-01-01",
            bbox="-180,-90,180,90",
            license="open",
            url="https://example.com",
            catalog="https://catalog.example.com",
            deprecated="false",
        )
        self.assertEqual(info.provider, "ESA")
        self.assertEqual(info.type, "image_collection")

    def test_serialization(self):
        """Test model serialization to dict."""
        info = CatalogDatasetInfo(id="test", title="Test Dataset")
        data = info.model_dump()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["id"], "test")
        self.assertEqual(data["title"], "Test Dataset")

    def test_missing_required_field_raises(self):
        """Test that missing required fields raise ValidationError."""
        with self.assertRaises(ValidationError):
            CatalogDatasetInfo(title="No ID")


class TestCatalogSearchResult(unittest.TestCase):
    """Tests for the CatalogSearchResult Pydantic model."""

    def test_empty_datasets(self):
        """Test construction with empty datasets list."""
        result = CatalogSearchResult(query="landsat", dataset_count=0, datasets=[])
        self.assertEqual(result.dataset_count, 0)
        self.assertEqual(len(result.datasets), 0)

    def test_with_datasets(self):
        """Test construction with populated datasets."""
        datasets = [
            CatalogDatasetInfo(id="d1", title="Dataset 1"),
            CatalogDatasetInfo(id="d2", title="Dataset 2"),
        ]
        result = CatalogSearchResult(query="test", dataset_count=2, datasets=datasets)
        self.assertEqual(len(result.datasets), 2)
        self.assertEqual(result.datasets[0].id, "d1")

    def test_filters_default_none(self):
        """Test that filters defaults to None."""
        result = CatalogSearchResult(query="q", dataset_count=0)
        self.assertIsNone(result.filters)


class TestCatalogLocationInfo(unittest.TestCase):
    """Tests for the catalog LocationInfo Pydantic model."""

    def test_construction(self):
        """Test basic construction with required fields."""
        loc = CatalogLocationInfo(
            name="San Francisco",
            bbox=[-122.5, 37.7, -122.3, 37.9],
            center=[-122.4, 37.8],
        )
        self.assertEqual(loc.name, "San Francisco")
        self.assertEqual(len(loc.bbox), 4)
        self.assertEqual(len(loc.center), 2)

    def test_bbox_as_float_list(self):
        """Test that bbox values are stored as floats."""
        loc = CatalogLocationInfo(
            name="Test", bbox=[0.0, 0.0, 1.0, 1.0], center=[0.5, 0.5]
        )
        for val in loc.bbox:
            self.assertIsInstance(val, float)


class TestSTACCollectionInfo(unittest.TestCase):
    """Tests for the STACCollectionInfo Pydantic model."""

    def test_required_fields(self):
        """Test construction with required fields."""
        info = STACCollectionInfo(id="sentinel-2-l2a", title="Sentinel-2 Level 2A")
        self.assertEqual(info.id, "sentinel-2-l2a")
        self.assertEqual(info.title, "Sentinel-2 Level 2A")

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        info = STACCollectionInfo(id="test", title="Test")
        self.assertIsNone(info.description)
        self.assertIsNone(info.license)
        self.assertIsNone(info.temporal_extent)
        self.assertIsNone(info.spatial_extent)
        self.assertIsNone(info.providers)
        self.assertIsNone(info.keywords)


class TestSTACAssetInfo(unittest.TestCase):
    """Tests for the STACAssetInfo Pydantic model."""

    def test_construction(self):
        """Test basic construction."""
        asset = STACAssetInfo(key="B04", title="Red Band")
        self.assertEqual(asset.key, "B04")
        self.assertEqual(asset.title, "Red Band")

    def test_optional_title(self):
        """Test that title is optional."""
        asset = STACAssetInfo(key="visual")
        self.assertIsNone(asset.title)


class TestSTACItemInfo(unittest.TestCase):
    """Tests for the STACItemInfo Pydantic model."""

    def test_required_fields(self):
        """Test construction with required fields."""
        item = STACItemInfo(id="S2A_MSIL2A_20230101", collection="sentinel-2-l2a")
        self.assertEqual(item.id, "S2A_MSIL2A_20230101")
        self.assertEqual(item.collection, "sentinel-2-l2a")

    def test_with_assets(self):
        """Test construction with asset list."""
        assets = [
            STACAssetInfo(key="B04", title="Red"),
            STACAssetInfo(key="B08", title="NIR"),
        ]
        item = STACItemInfo(id="item1", collection="col1", assets=assets)
        self.assertEqual(len(item.assets), 2)

    def test_empty_assets_default(self):
        """Test that assets defaults to empty list."""
        item = STACItemInfo(id="item1", collection="col1")
        self.assertEqual(item.assets, [])

    def test_bbox_field(self):
        """Test bbox as list of floats."""
        item = STACItemInfo(
            id="item1",
            collection="col1",
            bbox=[-122.5, 37.7, -122.3, 37.9],
        )
        self.assertEqual(len(item.bbox), 4)


class TestSTACSearchResult(unittest.TestCase):
    """Tests for the STACSearchResult Pydantic model."""

    def test_empty_items(self):
        """Test construction with empty items."""
        result = STACSearchResult(query="sentinel", item_count=0)
        self.assertEqual(result.item_count, 0)
        self.assertEqual(len(result.items), 0)

    def test_with_items(self):
        """Test construction with populated items."""
        items = [STACItemInfo(id="i1", collection="c1")]
        result = STACSearchResult(query="q", item_count=1, items=items)
        self.assertEqual(len(result.items), 1)

    def test_optional_fields(self):
        """Test that optional fields default to None."""
        result = STACSearchResult(query="q", item_count=0)
        self.assertIsNone(result.collection)
        self.assertIsNone(result.bbox)
        self.assertIsNone(result.time_range)


class TestSTACLocationInfo(unittest.TestCase):
    """Tests for the STAC LocationInfo Pydantic model."""

    def test_construction(self):
        """Test basic construction."""
        loc = STACLocationInfo(
            name="New York",
            bbox=[-74.3, 40.5, -73.7, 40.9],
            center=[-74.0, 40.7],
        )
        self.assertEqual(loc.name, "New York")

    def test_missing_required_raises(self):
        """Test that missing required fields raise ValidationError."""
        with self.assertRaises(ValidationError):
            STACLocationInfo(name="Test")


if __name__ == "__main__":
    unittest.main()
