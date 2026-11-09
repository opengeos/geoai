import unittest
from unittest.mock import MagicMock
import sys

# Mock torch dependencies if not installed
for m in ['torch', 'torchgeo', 'torchvision', 'torchvision.transforms']:
    if m not in sys.modules:
        sys.modules[m] = MagicMock()

import geopandas as gpd
from shapely.geometry import box
import tempfile
import os

from geoai.utils.training import _load_class_data_vector


class TestSortedClassIds(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_class_data_vector_sorts_classes_deterministically(self):
        """Verify _load_class_data_vector creates sorted class IDs irrespective of row order (Issue #956)."""
        # Order 1: water, building, road
        gdf1 = gpd.GeoDataFrame(
            {
                "class": ["water", "building", "road", "water"],
                "geometry": [box(0, 0, 10, 10), box(10, 10, 20, 20), box(20, 20, 30, 30), box(30, 30, 40, 40)],
            },
            crs="EPSG:4326",
        )
        vec_path1 = os.path.join(self.temp_dir.name, "vec1.geojson")
        gdf1.to_file(vec_path1, driver="GeoJSON")

        _, class_to_id1, coco_cats1 = _load_class_data_vector(
            vector_path=vec_path1,
            src_crs="EPSG:4326",
            class_value_field="class",
            quiet=True,
            metadata_format="COCO",
        )

        # Order 2: road, water, building (reversed/different order)
        gdf2 = gpd.GeoDataFrame(
            {
                "class": ["road", "water", "building", "road"],
                "geometry": [box(0, 0, 10, 10), box(10, 10, 20, 20), box(20, 20, 30, 30), box(30, 30, 40, 40)],
            },
            crs="EPSG:4326",
        )
        vec_path2 = os.path.join(self.temp_dir.name, "vec2.geojson")
        gdf2.to_file(vec_path2, driver="GeoJSON")

        _, class_to_id2, coco_cats2 = _load_class_data_vector(
            vector_path=vec_path2,
            src_crs="EPSG:4326",
            class_value_field="class",
            quiet=True,
            metadata_format="COCO",
        )

        expected_mapping = {"building": 1, "road": 2, "water": 3}
        self.assertEqual(class_to_id1, expected_mapping)
        self.assertEqual(class_to_id2, expected_mapping)
        self.assertEqual(class_to_id1, class_to_id2)

        # Verify COCO categories IDs also match sorted order
        coco_ids1 = {c["name"]: c["id"] for c in coco_cats1}
        coco_ids2 = {c["name"]: c["id"] for c in coco_cats2}
        self.assertEqual(coco_ids1, expected_mapping)
        self.assertEqual(coco_ids2, expected_mapping)


if __name__ == "__main__":
    unittest.main()
