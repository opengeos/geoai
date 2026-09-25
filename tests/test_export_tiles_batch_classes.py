#!/usr/bin/env python

"""Tests for batch-wide class ID consistency in `export_geotiff_tiles_batch`.

Regression coverage for issue #920: class IDs used to be derived from each
image/mask pair in isolation, so a mask missing one of the classes shifted the
pixel values and annotation category IDs of every remaining class.
"""

import json
import os
import tempfile
import unittest
import unittest.mock

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geoai.utils.training import (
    _collect_batch_class_mapping,
    export_geotiff_tiles_batch,
)

TILE = 64
CRS = "EPSG:32617"


def _write_image(folder, name, x0, y0):
    """Write a small 3-band RGB GeoTIFF.

    Args:
        folder: Directory to write the image into.
        name: Base filename without extension.
        x0: Left coordinate of the raster.
        y0: Top coordinate of the raster.

    Returns:
        str: Path to the written GeoTIFF.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=TILE,
        width=TILE,
        count=3,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(x0, y0, 1, 1),
    ) as dst:
        dst.write(np.full((3, TILE, TILE), 128, dtype=np.uint8))
    return path


def _class_boxes(x0, y0, count):
    """Build one small non-overlapping box per class.

    Args:
        x0: Left coordinate of the parent raster.
        y0: Top coordinate of the parent raster.
        count: Number of boxes to create.

    Returns:
        list: Shapely polygons inside the raster footprint.
    """
    return [
        box(x0 + 4 + i * 16, y0 - 20, x0 + 4 + i * 16 + 10, y0 - 4)
        for i in range(count)
    ]


def _write_vector(folder, name, x0, y0, classes, field="name"):
    """Write a GeoJSON mask with one labelled box per class.

    Args:
        folder: Directory to write the vector into.
        name: Base filename without extension.
        x0: Left coordinate of the parent raster.
        y0: Top coordinate of the parent raster.
        classes: Sequence of class values.
        field: Name of the class value field.

    Returns:
        str: Path to the written GeoJSON file.
    """
    os.makedirs(folder, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {field: list(classes), "geometry": _class_boxes(x0, y0, len(classes))},
        crs=CRS,
    )
    path = os.path.join(folder, f"{name}.geojson")
    gdf.to_file(path, driver="GeoJSON")
    return path


def _write_raster_mask(folder, name, x0, y0, class_values):
    """Write a single-band raster mask containing the given class values.

    Args:
        folder: Directory to write the mask into.
        name: Base filename without extension.
        x0: Left coordinate of the raster.
        y0: Top coordinate of the raster.
        class_values: Sequence of pixel values to burn in.

    Returns:
        str: Path to the written GeoTIFF.
    """
    os.makedirs(folder, exist_ok=True)
    data = np.zeros((TILE, TILE), dtype=np.uint8)
    for i, value in enumerate(class_values):
        col = 4 + i * 16
        data[4:20, col : col + 10] = value
    path = os.path.join(folder, f"{name}.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=TILE,
        width=TILE,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(x0, y0, 1, 1),
    ) as dst:
        dst.write(data, 1)
    return path


def _mask_values(output_folder):
    """Collect the unique non-zero pixel values of every exported mask tile.

    Args:
        output_folder: Batch export output folder.

    Returns:
        dict: Mapping of mask filename to a sorted list of class IDs.
    """
    masks_dir = os.path.join(output_folder, "masks")
    values = {}
    for filename in sorted(os.listdir(masks_dir)):
        with rasterio.open(os.path.join(masks_dir, filename)) as src:
            unique = np.unique(src.read(1))
        values[filename] = [int(v) for v in unique if v != 0]
    return values


class TestCollectBatchClassMapping(unittest.TestCase):
    """Unit tests for the batch-wide class scanner."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_union_of_vector_classes_is_sorted(self):
        a = _write_vector(self.root, "a", 0, 100, ["car", "truck", "bus"])
        b = _write_vector(self.root, "b", 200, 100, ["truck", "bus"])

        mapping, unclassified = _collect_batch_class_mapping([a, b], "name", quiet=True)

        self.assertEqual(mapping, {"bus": 1, "car": 2, "truck": 3})
        self.assertEqual(unclassified, 1)

    def test_mapping_is_order_independent(self):
        a = _write_vector(self.root, "a", 0, 100, ["car", "truck", "bus"])
        b = _write_vector(self.root, "b", 200, 100, ["truck", "bus"])

        self.assertEqual(
            _collect_batch_class_mapping([a, b], "name", quiet=True),
            _collect_batch_class_mapping([b, a], "name", quiet=True),
        )

    def test_rare_raster_class_is_not_missed(self):
        # A class covering a handful of pixels in a large raster is invisible to
        # a decimated read but must still get an ID (issue #920 follow-up).
        path = os.path.join(self.root, "rare.tif")
        data = np.zeros((2048, 2048), dtype=np.uint8)
        data[:1024, :] = 1
        data[2040:2043, 2040:2043] = 2
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2048,
            width=2048,
            count=1,
            dtype="uint8",
            crs=CRS,
            transform=from_origin(0, 2048, 1, 1),
        ) as dst:
            dst.write(data, 1)

        mapping, _ = _collect_batch_class_mapping([path], quiet=True)

        self.assertEqual(mapping, {1: 1, 2: 2})

    def test_masks_without_class_field_get_their_own_id(self):
        # ID 1 belongs to a real class, so field-less features must not silently
        # be merged into it.
        labelled = _write_vector(self.root, "a", 0, 100, ["car", "bus"])
        unlabelled = _write_vector(self.root, "b", 200, 100, ["x"], field="other")

        mapping, unclassified = _collect_batch_class_mapping(
            [labelled, unlabelled], "name", quiet=True
        )

        self.assertEqual(mapping["bus"], 1)
        self.assertEqual(mapping["car"], 2)
        self.assertEqual(unclassified, 3)
        self.assertEqual(mapping["unclassified"], 3)

    def test_null_class_values_get_the_unclassified_id(self):
        # The column exists but some rows are null, so those features must not
        # land on whichever real class sorts to ID 1.
        gdf = gpd.GeoDataFrame(
            {"name": ["car", None, "bus"], "geometry": _class_boxes(0, 100, 3)},
            crs=CRS,
        )
        path = os.path.join(self.root, "nulls.geojson")
        gdf.to_file(path, driver="GeoJSON")

        mapping, unclassified = _collect_batch_class_mapping([path], "name", quiet=True)

        self.assertEqual(mapping["bus"], 1)
        self.assertEqual(mapping["car"], 2)
        self.assertEqual(unclassified, 3)
        self.assertEqual(mapping["unclassified"], 3)

    def test_unclassified_name_avoids_collision(self):
        path = _write_vector(self.root, "a", 0, 100, ["unclassified", "car"])
        unlabelled = _write_vector(self.root, "b", 200, 100, ["x"], field="other")

        mapping, unclassified = _collect_batch_class_mapping(
            [path, unlabelled], "name", quiet=True
        )

        self.assertEqual(mapping["unclassified"], 2)
        self.assertEqual(mapping["unclassified_"], unclassified)

    def test_raster_masks_use_union_of_pixel_values(self):
        a = _write_raster_mask(self.root, "a", 0, 100, [7, 8, 9])
        b = _write_raster_mask(self.root, "b", 200, 100, [8, 9])

        mapping, _ = _collect_batch_class_mapping([a, b], quiet=True)

        self.assertEqual(mapping, {7: 1, 8: 2, 9: 3})

    def test_missing_class_field_falls_back_to_single_class(self):
        path = _write_vector(self.root, "a", 0, 100, ["car"], field="label")

        mapping, unclassified = _collect_batch_class_mapping([path], "name", quiet=True)

        self.assertEqual(mapping, {1: 1})
        self.assertEqual(unclassified, 1)

    def test_missing_field_everywhere_warns(self):
        # A typo in class_value_field should not fail silently.
        path = _write_vector(self.root, "a", 0, 100, ["car"], field="label")

        with self.assertLogs("geoai", level="WARNING") as captured:
            mapping, unclassified = _collect_batch_class_mapping(
                [path], "name", quiet=False
            )

        self.assertEqual(mapping, {1: 1})
        self.assertEqual(unclassified, 1)
        self.assertTrue(any("not found in 1 mask file" in m for m in captured.output))

    def test_preloaded_geodataframe_is_reused(self):
        path = _write_vector(self.root, "a", 0, 100, ["car", "bus"])
        gdf = gpd.read_file(path)

        with unittest.mock.patch(
            "geopandas.read_file", side_effect=AssertionError("should not re-read")
        ):
            mapping, _ = _collect_batch_class_mapping(
                [path], "name", quiet=True, preloaded={path: gdf}
            )

        self.assertEqual(mapping, {"bus": 1, "car": 2})

    def test_too_many_classes_for_uint8_warns(self):
        # Mask tiles are uint8, so more than 255 class IDs cannot be stored.
        classes = [f"c{i:04d}" for i in range(300)]
        gdf = gpd.GeoDataFrame(
            {"name": classes, "geometry": _class_boxes(0, 100, len(classes))},
            crs=CRS,
        )
        path = os.path.join(self.root, "many.geojson")
        gdf.to_file(path, driver="GeoJSON")

        with self.assertWarns(UserWarning) as captured:
            mapping, _ = _collect_batch_class_mapping([path], "name", quiet=True)

        self.assertEqual(len(mapping), 300)
        self.assertIn("uint8", str(captured.warning))

    def test_unreadable_masks_are_skipped(self):
        good = _write_vector(self.root, "good", 0, 100, ["car"])
        missing = os.path.join(self.root, "missing.geojson")

        mapping, _ = _collect_batch_class_mapping(
            [good, missing, None], "name", quiet=True
        )

        self.assertEqual(mapping, {"car": 1})

    def test_mixed_raster_and_vector_masks_are_still_mapped(self):
        # Raster masks contribute int class values and vector masks contribute
        # strings, which are not sortable against each other.
        vector = _write_vector(self.root, "a", 0, 100, ["car", "bus"])
        raster = _write_raster_mask(self.root, "b", 200, 100, [7, 8])

        mapping, _ = _collect_batch_class_mapping([vector, raster], "name", quiet=True)

        self.assertEqual(set(mapping.keys()), {"car", "bus", 7, 8})
        self.assertEqual(sorted(mapping.values()), [1, 2, 3, 4])


class TestExportGeotiffTilesBatchClassIds(unittest.TestCase):
    """End-to-end class consistency checks across the batch export modes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.images = os.path.join(self.root, "images")
        self.output = os.path.join(self.root, "tiles")

    def _read_coco(self):
        with open(os.path.join(self.output, "annotations", "instances.json")) as fh:
            return json.load(fh)

    def test_vector_folder_masks_share_class_ids(self):
        vectors = os.path.join(self.root, "vectors")
        _write_image(self.images, "a", 0, 1000)
        _write_vector(vectors, "a", 0, 1000, ["car", "truck", "bus"])
        _write_image(self.images, "b", 1000, 1000)
        _write_vector(vectors, "b", 1000, 1000, ["truck", "bus"])

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=vectors,
            output_folder=self.output,
            match_by_name=True,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        coco = self._read_coco()
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        self.assertEqual(categories, {1: "bus", 2: "car", 3: "truck"})

        values = _mask_values(self.output)
        self.assertEqual(len(values), 2)
        by_base = {name.split("_")[0]: ids for name, ids in values.items()}
        # "car" (ID 2) only exists in mask A, but bus/truck keep IDs 1 and 3.
        self.assertEqual(by_base["a"], [1, 2, 3])
        self.assertEqual(by_base["b"], [1, 3])

    def test_coco_annotations_use_the_batch_categories(self):
        vectors = os.path.join(self.root, "vectors")
        _write_image(self.images, "a", 0, 1000)
        _write_vector(vectors, "a", 0, 1000, ["car", "truck", "bus"])
        _write_image(self.images, "b", 1000, 1000)
        _write_vector(vectors, "b", 1000, 1000, ["truck", "bus"])

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=vectors,
            output_folder=self.output,
            match_by_name=True,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        coco = self._read_coco()
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        file_names = {img["id"]: img["file_name"] for img in coco["images"]}
        per_tile = {}
        for ann in coco["annotations"]:
            per_tile.setdefault(file_names[ann["image_id"]], set()).add(
                categories[ann["category_id"]]
            )

        named = {name.split("_")[0]: sorted(v) for name, v in per_tile.items()}
        self.assertEqual(named["a"], ["bus", "car", "truck"])
        self.assertEqual(named["b"], ["bus", "truck"])

    def test_raster_folder_masks_share_class_ids(self):
        masks = os.path.join(self.root, "masks")
        _write_image(self.images, "a", 0, 1000)
        _write_raster_mask(masks, "a", 0, 1000, [7, 8, 9])
        _write_image(self.images, "b", 1000, 1000)
        _write_raster_mask(masks, "b", 1000, 1000, [8, 9])

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=masks,
            output_folder=self.output,
            match_by_name=True,
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        by_base = {
            name.split("_")[0]: ids for name, ids in _mask_values(self.output).items()
        }
        self.assertEqual(by_base["a"], [1, 2, 3])
        self.assertEqual(by_base["b"], [2, 3])

    def test_single_vector_file_shares_class_ids(self):
        _write_image(self.images, "a", 0, 1000)
        _write_image(self.images, "b", 1000, 1000)
        gdf = gpd.GeoDataFrame(
            {
                "name": ["car", "truck", "bus", "truck", "bus"],
                "geometry": _class_boxes(0, 1000, 3) + _class_boxes(1000, 1000, 2),
            },
            crs=CRS,
        )
        masks_file = os.path.join(self.root, "all.geojson")
        gdf.to_file(masks_file, driver="GeoJSON")

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_file=masks_file,
            output_folder=self.output,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        by_base = {
            name.split("_")[0]: ids for name, ids in _mask_values(self.output).items()
        }
        self.assertEqual(by_base["a"], [1, 2, 3])
        self.assertEqual(by_base["b"], [1, 3])

    def test_yolo_class_indices_match_classes_file(self):
        vectors = os.path.join(self.root, "vectors")
        _write_image(self.images, "a", 0, 1000)
        _write_vector(vectors, "a", 0, 1000, ["car", "truck", "bus"])
        _write_image(self.images, "b", 1000, 1000)
        _write_vector(vectors, "b", 1000, 1000, ["truck", "bus"])

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=vectors,
            output_folder=self.output,
            match_by_name=True,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="YOLO",
            quiet=True,
        )

        labels_dir = os.path.join(self.output, "labels")
        with open(os.path.join(labels_dir, "classes.txt")) as fh:
            classes = [line.strip() for line in fh if line.strip()]
        self.assertEqual(classes, ["bus", "car", "truck"])

        found = {}
        for filename in sorted(os.listdir(labels_dir)):
            if filename == "classes.txt" or not filename.endswith(".txt"):
                continue
            with open(os.path.join(labels_dir, filename)) as fh:
                indices = {int(line.split()[0]) for line in fh if line.strip()}
            found[filename.split("_")[0]] = sorted(classes[i] for i in indices)

        self.assertEqual(found["a"], ["bus", "car", "truck"])
        self.assertEqual(found["b"], ["bus", "truck"])

    def test_unlabelled_mask_does_not_borrow_a_real_class_id(self):
        vectors = os.path.join(self.root, "vectors")
        _write_image(self.images, "a", 0, 1000)
        _write_vector(vectors, "a", 0, 1000, ["car", "bus"])
        _write_image(self.images, "b", 1000, 1000)
        _write_vector(vectors, "b", 1000, 1000, ["x"], field="other")

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=vectors,
            output_folder=self.output,
            match_by_name=True,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        categories = {c["id"]: c["name"] for c in self._read_coco()["categories"]}
        self.assertEqual(categories, {1: "bus", 2: "car", 3: "unclassified"})

        by_base = {
            name.split("_")[0]: ids for name, ids in _mask_values(self.output).items()
        }
        self.assertEqual(by_base["a"], [1, 2])
        # The unlabelled mask gets its own ID instead of colliding with "bus".
        self.assertEqual(by_base["b"], [3])

    def test_null_class_values_do_not_borrow_a_real_class_id(self):
        vectors = os.path.join(self.root, "vectors")
        _write_image(self.images, "a", 0, 1000)
        os.makedirs(vectors, exist_ok=True)
        gdf = gpd.GeoDataFrame(
            {"name": ["car", None, "bus"], "geometry": _class_boxes(0, 1000, 3)},
            crs=CRS,
        )
        gdf.to_file(os.path.join(vectors, "a.geojson"), driver="GeoJSON")

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=vectors,
            output_folder=self.output,
            match_by_name=True,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        categories = {c["id"]: c["name"] for c in self._read_coco()["categories"]}
        self.assertEqual(categories, {1: "bus", 2: "car", 3: "unclassified"})
        # bus, car and the null feature each keep a distinct ID.
        self.assertEqual(list(_mask_values(self.output).values())[0], [1, 2, 3])

    def test_mixed_raster_and_vector_masks_share_one_id_space(self):
        # A masks folder may legally hold both kinds. The raster remap compares
        # its integer pixels against every mapping key, including the string
        # keys contributed by the vector masks, so check they stay disjoint.
        masks = os.path.join(self.root, "masks")
        _write_image(self.images, "a", 0, 1000)
        _write_vector(masks, "a", 0, 1000, ["car", "bus"])
        _write_image(self.images, "b", 1000, 1000)
        _write_raster_mask(masks, "b", 1000, 1000, [7, 8])

        export_geotiff_tiles_batch(
            images_folder=self.images,
            masks_folder=masks,
            output_folder=self.output,
            match_by_name=True,
            class_value_field="name",
            tile_size=TILE,
            stride=TILE,
            skip_empty_tiles=True,
            metadata_format="COCO",
            quiet=True,
        )

        categories = {c["id"]: c["name"] for c in self._read_coco()["categories"]}
        self.assertEqual(categories, {1: "7", 2: "8", 3: "bus", 4: "car"})

        by_base = {
            name.split("_")[0]: ids for name, ids in _mask_values(self.output).items()
        }
        self.assertEqual(by_base["b"], [1, 2])
        self.assertEqual(by_base["a"], [3, 4])

    def test_images_only_mode_is_unaffected(self):
        _write_image(self.images, "a", 0, 1000)

        stats = export_geotiff_tiles_batch(
            images_folder=self.images,
            output_folder=self.output,
            tile_size=TILE,
            stride=TILE,
            quiet=True,
        )

        self.assertEqual(stats["total_tiles"], 1)
        self.assertFalse(os.path.exists(os.path.join(self.output, "masks")))


if __name__ == "__main__":
    unittest.main()
