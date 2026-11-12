import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

try:
    from geoai.utils.training import (
        _load_class_data_vector,
        _process_image_mask_pair,
        export_training_data,
    )

    _TRAINING_AVAILABLE = True
except ImportError:
    _load_class_data_vector = None
    _process_image_mask_pair = None
    export_training_data = None
    _TRAINING_AVAILABLE = False

_MOCKED_IMPORT_PREAMBLE = """
import sys
from unittest.mock import MagicMock
for mod_name in ("torch", "torchgeo", "torchvision", "torchvision.transforms"):
    sys.modules[mod_name] = MagicMock()
"""


def _run_in_isolated_subprocess(body: str) -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = (
        f"import sys\nsys.path.insert(0, {repo_root!r})\n"
        + _MOCKED_IMPORT_PREAMBLE
        + textwrap.dedent(body)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Isolated subprocess check failed (exit={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _write_test_raster(path: str, width: int = 64, height: int = 64) -> None:
    data = np.full((3, height, width), 120, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(data)


def _write_vector(path: str, classes, geometries, field: str = "class") -> str:
    gdf = gpd.GeoDataFrame(
        {field: list(classes), "geometry": list(geometries)},
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


class TestSortedClassIds(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_class_data_vector_sorts_classes_deterministically(self):
        """Sort class IDs consistently regardless of vector row order (#956)."""
        geoms = [
            box(0, 0, 10, 10),
            box(10, 10, 20, 20),
            box(20, 20, 30, 30),
            box(30, 30, 40, 40),
        ]
        vec1 = _write_vector(
            os.path.join(self.temp_dir.name, "vec1.geojson"),
            ["water", "building", "road", "water"],
            geoms,
        )
        vec2 = _write_vector(
            os.path.join(self.temp_dir.name, "vec2.geojson"),
            ["road", "water", "building", "road"],
            geoms,
        )

        if _TRAINING_AVAILABLE:
            _, c2id1, cats1 = _load_class_data_vector(
                vector_path=vec1,
                src_crs="EPSG:4326",
                class_value_field="class",
                quiet=True,
                metadata_format="COCO",
            )
            _, c2id2, cats2 = _load_class_data_vector(
                vector_path=vec2,
                src_crs="EPSG:4326",
                class_value_field="class",
                quiet=True,
                metadata_format="COCO",
            )
        else:
            out_json = os.path.join(self.temp_dir.name, "load_out.json")
            _run_in_isolated_subprocess(f"""
                import json
                from geoai.utils.training import _load_class_data_vector
                kw = {{
                    "src_crs": "EPSG:4326",
                    "class_value_field": "class",
                    "quiet": True,
                    "metadata_format": "COCO",
                }}
                _, c2id1, cats1 = _load_class_data_vector(vector_path={vec1!r}, **kw)
                _, c2id2, cats2 = _load_class_data_vector(vector_path={vec2!r}, **kw)
                payload = {{
                    "c2id1": c2id1,
                    "c2id2": c2id2,
                    "cats1": cats1,
                    "cats2": cats2,
                }}
                with open({out_json!r}, "w") as f:
                    json.dump(payload, f)
            """)
            with open(out_json) as f:
                res = json.load(f)
            c2id1, c2id2, cats1, cats2 = (
                res["c2id1"],
                res["c2id2"],
                res["cats1"],
                res["cats2"],
            )

        expected_mapping = {"building": 1, "road": 2, "water": 3}
        self.assertEqual(c2id1, expected_mapping)
        self.assertEqual(c2id2, expected_mapping)
        self.assertEqual({c["name"]: c["id"] for c in cats1}, expected_mapping)
        self.assertEqual({c["name"]: c["id"] for c in cats2}, expected_mapping)

    def test_load_class_data_vector_mixed_types_fallback(self):
        """Fallback to str sort key when mixed int and str values preserve types."""
        if _TRAINING_AVAILABLE:
            mixed_gdf = gpd.GeoDataFrame(
                {
                    "class": pd.Series(["zebra", 1, "a", 10], dtype=object),
                    "geometry": [
                        box(0, 0, 1, 1),
                        box(1, 1, 2, 2),
                        box(2, 2, 3, 3),
                        box(3, 3, 4, 4),
                    ],
                },
                crs="EPSG:4326",
            )
            with patch("geopandas.read_file", return_value=mixed_gdf):
                _, class_to_id, _ = _load_class_data_vector(
                    vector_path="in_memory_mixed.geojson",
                    src_crs="EPSG:4326",
                    class_value_field="class",
                    quiet=True,
                    metadata_format="COCO",
                )
            self.assertIn(1, class_to_id)
            self.assertIn("a", class_to_id)
            self.assertEqual(list(class_to_id.keys()), [1, 10, "a", "zebra"])
            self.assertEqual(list(class_to_id.values()), [1, 2, 3, 4])
        else:
            out_json = os.path.join(self.temp_dir.name, "mixed_out.json")
            _run_in_isolated_subprocess(f"""
                import json
                from unittest.mock import patch
                import geopandas as gpd
                import pandas as pd
                from shapely.geometry import box
                from geoai.utils.training import _load_class_data_vector

                mixed_gdf = gpd.GeoDataFrame(
                    {{
                        "class": pd.Series(["zebra", 1, "a", 10], dtype=object),
                        "geometry": [
                            box(0, 0, 1, 1),
                            box(1, 1, 2, 2),
                            box(2, 2, 3, 3),
                            box(3, 3, 4, 4),
                        ],
                    }},
                    crs="EPSG:4326",
                )
                with patch("geopandas.read_file", return_value=mixed_gdf):
                    _, class_to_id, _ = _load_class_data_vector(
                        vector_path="in_memory_mixed.geojson",
                        src_crs="EPSG:4326",
                        class_value_field="class",
                        quiet=True,
                        metadata_format="COCO",
                    )
                with open({out_json!r}, "w") as f:
                    json.dump(
                        {{
                            "types": [type(k).__name__ for k in class_to_id.keys()],
                            "keys": [str(k) for k in class_to_id.keys()],
                            "values": list(class_to_id.values()),
                        }},
                        f,
                    )
            """)
            with open(out_json) as f:
                res = json.load(f)
            self.assertEqual(res["types"], ["int", "int", "str", "str"])
            self.assertEqual(res["keys"], ["1", "10", "a", "zebra"])
            self.assertEqual(res["values"], [1, 2, 3, 4])

    def test_process_image_mask_pair_without_batch_mapping_is_order_independent(self):
        """Ensure _process_image_mask_pair without class_to_id is order-independent."""
        img_path = os.path.join(self.temp_dir.name, "img.tif")
        _write_test_raster(img_path, width=64, height=64)

        geom_water = box(4, 32, 24, 56)
        geom_building = box(36, 8, 56, 28)

        vec1 = _write_vector(
            os.path.join(self.temp_dir.name, "pair1.geojson"),
            ["water", "building"],
            [geom_water, geom_building],
        )
        vec2 = _write_vector(
            os.path.join(self.temp_dir.name, "pair2.geojson"),
            ["building", "water"],
            [geom_building, geom_water],
        )

        out1_img = os.path.join(self.temp_dir.name, "p1_imgs")
        out1_msk = os.path.join(self.temp_dir.name, "p1_msks")
        out2_img = os.path.join(self.temp_dir.name, "p2_imgs")
        out2_msk = os.path.join(self.temp_dir.name, "p2_msks")
        for d in (out1_img, out1_msk, out2_img, out2_msk):
            os.makedirs(d, exist_ok=True)

        if _TRAINING_AVAILABLE:
            _process_image_mask_pair(
                img_path,
                vec1,
                "tile",
                out1_img,
                out1_msk,
                0,
                tile_size=64,
                stride=64,
                class_value_field="class",
                quiet=True,
                metadata_format="YOLO",
                class_to_id=None,
            )
            _process_image_mask_pair(
                img_path,
                vec2,
                "tile",
                out2_img,
                out2_msk,
                0,
                tile_size=64,
                stride=64,
                class_value_field="class",
                quiet=True,
                metadata_format="YOLO",
                class_to_id=None,
            )
        else:
            _run_in_isolated_subprocess(f"""
                from geoai.utils.training import _process_image_mask_pair
                _process_image_mask_pair(
                    {img_path!r}, {vec1!r}, "tile", {out1_img!r}, {out1_msk!r}, 0,
                    tile_size=64, stride=64, class_value_field="class",
                    quiet=True, metadata_format="YOLO", class_to_id=None,
                )
                _process_image_mask_pair(
                    {img_path!r}, {vec2!r}, "tile", {out2_img!r}, {out2_msk!r}, 0,
                    tile_size=64, stride=64, class_value_field="class",
                    quiet=True, metadata_format="YOLO", class_to_id=None,
                )
            """)

        with (
            rasterio.open(os.path.join(out1_msk, "tile_000000.tif")) as s1,
            rasterio.open(os.path.join(out2_msk, "tile_000000.tif")) as s2,
        ):
            m1 = s1.read(1)
            m2 = s2.read(1)
        np.testing.assert_array_equal(m1, m2)
        self.assertEqual(sorted(int(v) for v in np.unique(m1) if v != 0), [1, 2])

    def test_export_training_data_numeric_classes_consistent_masks_and_coco(self):
        """Map numeric classes consistently across masks and COCO categories."""
        img_path = os.path.join(self.temp_dir.name, "export_img.tif")
        _write_test_raster(img_path, width=64, height=64)

        geom5 = box(4, 32, 24, 56)
        geom2 = box(36, 8, 56, 28)

        vec1 = _write_vector(
            os.path.join(self.temp_dir.name, "exp1.geojson"),
            [5, 2],
            [geom5, geom2],
        )
        vec2 = _write_vector(
            os.path.join(self.temp_dir.name, "exp2.geojson"),
            [2, 5],
            [geom2, geom5],
        )

        out1 = os.path.join(self.temp_dir.name, "out1")
        out2 = os.path.join(self.temp_dir.name, "out2")

        if _TRAINING_AVAILABLE:
            export_training_data(
                img_path,
                out1,
                vec1,
                tile_size_x=64,
                tile_size_y=64,
                stride_x=64,
                stride_y=64,
                metadata_format="COCO",
                class_value_field="class",
                quiet=True,
            )
            export_training_data(
                img_path,
                out2,
                vec2,
                tile_size_x=64,
                tile_size_y=64,
                stride_x=64,
                stride_y=64,
                metadata_format="COCO",
                class_value_field="class",
                quiet=True,
            )
        else:
            _run_in_isolated_subprocess(f"""
                from geoai.utils.training import export_training_data
                export_training_data(
                    {img_path!r}, {out1!r}, {vec1!r},
                    tile_size_x=64, tile_size_y=64, stride_x=64, stride_y=64,
                    metadata_format="COCO", class_value_field="class", quiet=True,
                )
                export_training_data(
                    {img_path!r}, {out2!r}, {vec2!r},
                    tile_size_x=64, tile_size_y=64, stride_x=64, stride_y=64,
                    metadata_format="COCO", class_value_field="class", quiet=True,
                )
            """)

        with (
            rasterio.open(os.path.join(out1, "labels", "tile_000000.tif")) as s1,
            rasterio.open(os.path.join(out2, "labels", "tile_000000.tif")) as s2,
        ):
            m1 = s1.read(1)
            m2 = s2.read(1)
        np.testing.assert_array_equal(m1, m2)
        self.assertEqual(sorted(int(v) for v in np.unique(m1) if v != 0), [1, 2])

        with (
            open(os.path.join(out1, "annotations", "instances.json")) as f1,
            open(os.path.join(out2, "annotations", "instances.json")) as f2,
        ):
            coco1 = json.load(f1)
            coco2 = json.load(f2)

        expected_cats = {"2": 1, "5": 2}
        self.assertEqual(
            {c["name"]: c["id"] for c in coco1["categories"]}, expected_cats
        )
        self.assertEqual(
            {c["name"]: c["id"] for c in coco2["categories"]}, expected_cats
        )
        self.assertEqual(
            sorted(a["category_id"] for a in coco1["annotations"]),
            [1, 2],
        )
        self.assertEqual(
            sorted(a["category_id"] for a in coco2["annotations"]),
            [1, 2],
        )

    def test_subprocess_isolated_mock_import_does_not_pollute_parent(self):
        """Run mocked torch imports in a child process without mutating parent."""
        _run_in_isolated_subprocess("""
            from geoai.utils.training import _load_class_data_vector
            assert callable(_load_class_data_vector)
        """)
        if not _TRAINING_AVAILABLE:
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("geoai.utils.training", sys.modules)
