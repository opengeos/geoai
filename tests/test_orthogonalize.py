"""Synthetic geometry and GeoTIFF regressions for the independent fitter."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

# Exercise the real module without loading unrelated model/visualization utilities.
spec = importlib.util.spec_from_file_location(
    "orthogonalization", Path(__file__).parents[1] / "geoai/utils/_orthogonalize.py"
)
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)


class GeometryTests(unittest.TestCase):
    def fit(self, polygon, **overrides):
        options = dict(
            epsilon=0,
            min_area=0,
            min_segments=4,
            area_tolerance=0.7,
            detect_triangles=True,
        )
        options.update(overrides)
        return fitter._fit_polygon(polygon, **options)

    def assert_perpendicular(self, ring):
        points = list(ring.coords)
        vectors = np.diff(points, axis=0)
        for a, b in zip(vectors, np.roll(vectors, -1, axis=0)):
            # Collinear intermediate vertices are allowed.
            scale = np.linalg.norm(a) * np.linalg.norm(b)
            self.assertGreater(scale, 0)
            cosine = abs(float(a @ b)) / scale
            self.assertTrue(cosine < 1e-7 or abs(cosine - 1) < 1e-7)

    def test_skewed_outline_is_adjusted_to_perpendicular_edges(self):
        polygon = Polygon([(0, 0), (10, 0.5), (10.5, 6), (-0.2, 6.5)])
        fitted = self.fit(polygon)
        self.assertFalse(fitted.equals(polygon))
        self.assertTrue(fitted.is_valid)
        self.assert_perpendicular(fitted.exterior)
        self.assertTrue(0.7 <= fitted.area / polygon.area <= 1 / 0.7)

    def test_concave_outline_is_not_replaced_with_its_bounding_rectangle(self):
        polygon = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])
        self.assertTrue(self.fit(polygon).equals(polygon))
        self.assertLess(self.fit(polygon).area, polygon.minimum_rotated_rectangle.area)

    def test_holes_use_the_same_axes_and_remain_valid(self):
        polygon = Polygon(
            [(0, 0), (12, 0.4), (12.4, 12), (-0.2, 12)],
            [[(4, 4), (8, 4.2), (8.1, 8), (4, 8)]],
        )
        fitted = self.fit(rotate(polygon, 27, origin=(0, 0)))
        self.assertTrue(fitted.is_valid)
        self.assertEqual(len(fitted.interiors), 1)
        self.assert_perpendicular(fitted.exterior)
        self.assert_perpendicular(fitted.interiors[0])

    def test_large_coordinates_are_fitted_in_a_local_frame(self):
        polygon = Polygon([(0, 0), (10, 0.5), (10.5, 6), (-0.2, 6.5)])
        fitted = self.fit(polygon)
        shifted = self.fit(translate(polygon, 1e9, 1e9))
        restored = translate(shifted, -1e9, -1e9)
        self.assertLess(fitted.hausdorff_distance(restored), 1e-6)

    def test_too_large_area_change_retains_original(self):
        polygon = Polygon([(0, 0), (10, 0.5), (10.5, 6), (-0.2, 6.5)])
        self.assertTrue(self.fit(polygon, area_tolerance=1).equals(polygon))

    def test_triangle_small_outline_and_segment_floor_are_preserved(self):
        triangle = Polygon([(0, 0), (8, 0), (2, 8)])
        self.assertTrue(self.fit(triangle).equals(triangle))
        polygon = Polygon([(0, 0), (10, 0.5), (10.5, 6), (-0.2, 6.5)])
        self.assertTrue(self.fit(polygon, min_area=100).equals(polygon))
        self.assertTrue(self.fit(polygon, min_segments=5).equals(polygon))

    def test_invalid_neighbor_is_retained_without_overlay_failure(self):
        invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
        original = box(3, 0, 5, 2)
        result = fitter._avoid_overlaps([original, invalid], [box(1, 0, 5, 2), invalid])
        self.assertTrue(result[0].equals(original))
        self.assertIs(result[1], invalid)

    def test_overlap_rejection_checks_original_and_proposed_neighbors(self):
        originals = [box(0, 0, 2, 2), box(3, 0, 5, 2)]
        candidates = [box(0, 0, 3.5, 2), box(3.5, 0, 5.5, 2)]
        result = fitter._avoid_overlaps(originals, candidates)
        self.assertTrue(result[0].equals(originals[0]))
        self.assertEqual(result[0].intersection(result[1]).area, 0)
        # Fits can also conflict in a gap, without touching the other original.
        candidates = [box(0, 0, 2.6, 2), box(2.4, 0, 5, 2)]
        result = fitter._avoid_overlaps(originals, candidates)
        self.assertTrue(all(a.equals(b) for a, b in zip(originals, result)))


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def raster(self, data, nodata=None, crs="EPSG:3857"):
        path = self.root / "mask.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=from_origin(100, 200, 2, 2),
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)
        return path

    def test_labels_holes_nodata_and_crs_are_preserved(self):
        data = np.zeros((12, 16), dtype="uint8")
        data[1:10, 1:10] = 7
        data[4:6, 4:6] = 0
        data[2:4, 12:14] = 3
        data[10, 15] = 255
        result = fitter.orthogonalize(self.raster(data, nodata=255), epsilon=0)
        self.assertEqual(sorted(result.value), [3, 7])
        self.assertEqual(result.crs.to_epsg(), 3857)
        building = result[result.value == 7].geometry.iloc[0]
        self.assertEqual(len(building.interiors), 1)
        self.assertTrue(result.geometry.is_valid.all())
        self.assertEqual(building.area, (81 - 4) * 4)

    def test_skewed_raster_building_is_adjusted_end_to_end(self):
        from rasterio.features import rasterize, shapes
        from shapely.geometry import shape

        data = rasterize(
            [(Polygon([(10, 10), (44, 18), (40, 40), (8, 34)]), 1)],
            out_shape=(60, 60),
            dtype="uint8",
        )
        path = self.raster(data)
        result = fitter.orthogonalize(path, epsilon=2, min_area=0)
        with rasterio.open(path) as source:
            original = shape(
                next(shapes(data, mask=data > 0, transform=source.transform))[0]
            )
        adjusted = result.geometry.iloc[0]
        self.assertFalse(adjusted.equals(original))
        self.assertTrue(adjusted.is_valid)
        GeometryTests().assert_perpendicular(adjusted.exterior)
        self.assertTrue(0.7 <= adjusted.area / original.area <= 1 / 0.7)

    def test_empty_foreground_has_geometry_column_and_crs(self):
        data = np.zeros((3, 3), dtype="uint8")
        result = fitter.orthogonalize(self.raster(data))
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), ["value", "geometry"])
        self.assertEqual(result.crs.to_epsg(), 3857)

    def test_negative_and_positive_nodata_are_not_regions(self):
        data = np.array([[-3, 0, 2], [99, 0, 2]], dtype="int16")
        result = fitter.orthogonalize(self.raster(data, nodata=99), min_area=100)
        self.assertEqual(result.value.tolist(), [2])
        self.assertEqual(result.geometry.iloc[0].area, 8)

    def test_geojson_output_and_fractional_labels(self):
        data = np.array([[0, 1.5], [0, 1.5]], dtype="float32")
        output = self.root / "result.geojson"
        result = fitter.orthogonalize(self.raster(data), output_path=output)
        restored = gpd.read_file(output)
        self.assertEqual(result.value.tolist(), [1.5])
        self.assertEqual(restored.value.tolist(), [1.5])
        self.assertTrue(restored.geometry.iloc[0].equals(result.geometry.iloc[0]))

    def test_invalid_parameters_fail_before_opening_input(self):
        for option in (
            {"epsilon": -1},
            {"epsilon": float("nan")},
            {"min_area": float("inf")},
            {"min_area": -1},
            {"area_tolerance": 0},
            {"area_tolerance": 1.1},
            {"min_segments": 3},
            {"min_segments": True},
            {"detect_triangles": "yes"},
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                fitter.orthogonalize("does-not-exist.tif", **option)
