#!/usr/bin/env python
"""Regression tests for #1013 Batch B (extract, raster, geometry, training, landcover, device/models/viz)."""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Point, Polygon, box


def _write_geotiff(
    path,
    data,
    crs="EPSG:32617",
    origin=(500000.0, 4000000.0),
    pixel_size=1.0,
    nodata=None,
):
    """Write a 2D (H, W) or 3D (B, H, W) numpy array as a GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    bands, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": data.dtype,
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], pixel_size, pixel_size),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


class TestExtractFixes(unittest.TestCase):
    """Group 1: geoai/extract.py correctness, performance, and security fixes."""

    def test_agriculture_field_delineator_initializes_maskrcnn(self):
        from torchvision.models.detection import MaskRCNN

        from geoai.extract import AgricultureFieldDelineator

        delineator = AgricultureFieldDelineator.__new__(AgricultureFieldDelineator)
        delineator.custom_band_selection = [3, 2, 1]
        delineator.use_ndvi = True
        delineator.device = torch.device("cpu")

        model = delineator.initialize_sentinel2_model(model=None)
        self.assertIsInstance(model, MaskRCNN)
        self.assertEqual(model.backbone.body.conv1.in_channels, 4)

    def test_agriculture_band_stats_and_preprocess_indexing(self):
        from geoai.extract import AgricultureFieldDelineator

        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(7)
            data = np.zeros((10, 32, 32), dtype=np.float32)
            for b in range(10):
                data[b] = (b + 1) * 100.0 + rng.normal(0.0, 5.0, size=(32, 32))
            tif_path = _write_geotiff(os.path.join(tmpdir, "s2.tif"), data)

            delineator = AgricultureFieldDelineator.__new__(AgricultureFieldDelineator)
            delineator.band_selection = [0, 3, 7]
            delineator.use_ndvi = True

            # 0-based band_selection including band 0 must read src.read(band + 1)
            stats = delineator.update_band_stats(tif_path, band_selection=[0, 3, 7])
            self.assertEqual(len(stats["means"]), 3)
            self.assertAlmostEqual(stats["means"][0], 100.0, delta=5.0)
            self.assertAlmostEqual(stats["means"][1], 400.0, delta=5.0)
            self.assertAlmostEqual(stats["means"][2], 800.0, delta=5.0)

            tensor = delineator.preprocess_sentinel_bands(
                data, band_selection=[0, 3, 7], use_ndvi=True
            )
            self.assertEqual(tensor.shape, (4, 32, 32))
            # Each selected band should be normalized around 0 (mean ~ 0, std ~ 1)
            for idx in range(3):
                self.assertAlmostEqual(float(tensor[idx].mean()), 0.0, delta=0.5)
                self.assertAlmostEqual(float(tensor[idx].std()), 1.0, delta=0.5)

            # Non-monotonic / reversed 12-band selection must map each band to
            # its own statistic, not positional order.
            data12 = np.zeros((12, 8, 8), dtype=np.float32)
            for b in range(12):
                data12[b] = float(b * 10 + 5)
            rev_bands = list(range(11, -1, -1))
            delineator.sentinel_band_stats = {
                "bands": list(range(12)),
                "means": [float(b * 10 + 5) for b in range(12)],
                "stds": [1.0] * 12,
            }
            t_rev = delineator.preprocess_sentinel_bands(
                data12, band_selection=rev_bands, use_ndvi=False
            )
            np.testing.assert_allclose(t_rev.numpy(), 0.0, atol=1e-5)

            # Even without "bands" metadata, matching list lengths must not
            # switch to positional order unless explicitly opted in.
            delineator.sentinel_band_stats = {
                "means": [float(b * 10 + 5) for b in range(12)],
                "stds": [1.0] * 12,
            }
            t_rev_legacy = delineator.preprocess_sentinel_bands(
                data12, band_selection=rev_bands, use_ndvi=False
            )
            np.testing.assert_allclose(t_rev_legacy.numpy(), 0.0, atol=1e-5)

            # Explicit positional contract uses selection order.
            delineator.sentinel_band_stats = {
                "means": [float(b * 10 + 5) for b in rev_bands],
                "stds": [1.0] * 12,
                "positional": True,
            }
            t_rev_pos = delineator.preprocess_sentinel_bands(
                data12, band_selection=rev_bands, use_ndvi=False
            )
            np.testing.assert_allclose(t_rev_pos.numpy(), 0.0, atol=1e-5)

    def test_process_sentinel_raster_wires_sentinel2_dataset(self):
        from geoai.extract import AgricultureFieldDelineator

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.full((10, 64, 64), 500.0, dtype=np.float32)
            data[7] = 900.0
            data[3] = 300.0
            tif_path = _write_geotiff(os.path.join(tmpdir, "s2.tif"), data)

            seen_channels = []

            def fake_model(images):
                seen_channels.append(images.shape[1])
                b, _, h, w = images.shape
                mask = torch.zeros((1, 1, h, w), dtype=torch.float32)
                mask[0, 0, 10:50, 10:50] = 1.0
                return [
                    {
                        "masks": mask,
                        "scores": torch.tensor([0.95]),
                        "labels": torch.tensor([1]),
                    }
                    for _ in range(b)
                ]

            delineator = AgricultureFieldDelineator.__new__(AgricultureFieldDelineator)
            delineator.device = torch.device("cpu")
            delineator.model = fake_model
            delineator.band_selection = [3, 2, 1]
            delineator.use_ndvi = True
            delineator.confidence_threshold = 0.5
            delineator.overlap = 0.0
            delineator.chip_size = (64, 64)
            delineator.nms_iou_threshold = 0.5
            delineator.mask_threshold = 0.5
            delineator.min_object_area = 10
            delineator.max_object_area = None
            delineator.simplify_tolerance = 1.0

            gdf = delineator.process_sentinel_raster(
                tif_path, filter_edges=False, chip_size=(64, 64), overlap=0.0
            )
            self.assertIsNotNone(gdf)
            self.assertEqual(len(gdf), 1)
            # 3 RGB bands + 1 NDVI channel = 4 channels from Sentinel2Dataset
            self.assertEqual(seen_channels, [4])

    def test_process_raster_preserves_labels_and_reuses_transform(self):
        from geoai.extract import ObjectDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.full((3, 64, 64), 120, dtype=np.uint8)
            tif_path = _write_geotiff(os.path.join(tmpdir, "rgb.tif"), data)

            def fake_model(images):
                _, _, h, w = images.shape
                masks = torch.zeros((2, 1, h, w), dtype=torch.float32)
                masks[0, 0, 5:25, 5:25] = 1.0
                masks[1, 0, 35:55, 35:55] = 1.0
                return [
                    {
                        "masks": masks,
                        "scores": torch.tensor([0.95, 0.85]),
                        "labels": torch.tensor([2, 5]),
                    }
                ]

            detector = ObjectDetector.__new__(ObjectDetector)
            detector.device = torch.device("cpu")
            detector.model = fake_model
            detector.confidence_threshold = 0.5
            detector.overlap = 0.0
            detector.chip_size = (64, 64)
            detector.nms_iou_threshold = 0.5
            detector.mask_threshold = 0.5
            detector.min_object_area = 10
            detector.max_object_area = None
            detector.simplify_tolerance = 1.0

            real_open = rasterio.open
            open_calls = []

            def spy_open(path, *args, **kwargs):
                open_calls.append(str(path))
                return real_open(path, *args, **kwargs)

            with patch("geoai.extract.rasterio.open", side_effect=spy_open):
                gdf = detector.process_raster(
                    tif_path,
                    filter_edges=False,
                    chip_size=(64, 64),
                    overlap=0.0,
                )

            self.assertIsNotNone(gdf)
            self.assertEqual(set(gdf["class"].tolist()), {2, 5})
            # Only 2 opens expected: 1 in CustomDataset.__init__, 1 in __getitem__
            # (none inside the per-mask contour transform loop).
            self.assertEqual(len(open_calls), 2)

    def test_custom_dataset_sliding_windows_and_lazy_stats(self):
        from geoai.extract import CustomDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.full((3, 1000, 1024), 100, dtype=np.uint8)
            tif_path = _write_geotiff(os.path.join(tmpdir, "grid.tif"), data)

            with patch("geoai.extract.get_raster_stats") as mock_stats:
                ds = CustomDataset(tif_path, chip_size=(512, 512), overlap=0.5)
                mock_stats.assert_not_called()

            self.assertEqual(ds.row_starts, [0, 256, 488])
            self.assertEqual(ds.col_starts, [0, 256, 512])
            self.assertEqual(ds.row_starts, sorted(set(ds.row_starts)))
            self.assertEqual(ds.col_starts, sorted(set(ds.col_starts)))

            # Accessing raster_stats lazily computes stats once
            stats = ds.raster_stats
            self.assertIn("mean", stats)

            # Invalid overlaps raise ValueError
            for bad_overlap in (-0.1, 1.0, 1.5):
                with self.assertRaises(ValueError):
                    CustomDataset(tif_path, chip_size=(512, 512), overlap=bad_overlap)

    def test_regularize_objects_preserves_utm_precision(self):
        from geoai.extract import ObjectDetector

        # Large UTM coordinates where float32 ULP is ~0.5m
        cx, cy = 500000.123, 4500000.456
        poly = shapely_rotate(
            box(cx - 5.0, cy - 2.5, cx + 5.0, cy + 2.5), 25.0, origin="centroid"
        )
        gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32617")

        result = ObjectDetector.regularize_objects(None, gdf)
        out_centroid = result.geometry.iloc[0].centroid
        self.assertAlmostEqual(out_centroid.x, cx, delta=1e-3)
        self.assertAlmostEqual(out_centroid.y, cy, delta=1e-3)

    def test_simplify_tolerance_pixel_epsilon_preserves_large_contours(self):
        from geoai.extract import ObjectDetector

        # Circle of radius 40 in a 128x128 raster produces >50 contour vertices
        yy, xx = np.ogrid[:128, :128]
        circle = (((yy - 64) ** 2 + (xx - 64) ** 2) <= 40**2).astype(np.uint8)

        detector = ObjectDetector.__new__(ObjectDetector)
        detector.simplify_tolerance = 1.0
        detector.mask_threshold = 0.5
        detector.min_object_area = 50
        detector.max_object_area = None
        detector.nms_iou_threshold = 0.5

        polys = detector.mask_to_polygons(circle, simplify_tolerance=1.0)
        self.assertEqual(len(polys), 1)
        self.assertGreaterEqual(len(polys[0]), 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = _write_geotiff(
                os.path.join(tmpdir, "circle_mask.tif"), (circle * 255).astype(np.uint8)
            )
            gdf = detector.masks_to_vector(
                mask_path,
                simplify_tolerance=1.0,
                min_object_area=50,
                regularize=False,
            )
            self.assertIsNotNone(gdf)
            self.assertEqual(len(gdf), 1)
            self.assertGreater(gdf.geometry.iloc[0].area, 4000.0)

    def test_masks_to_vector_vectorize_masks_and_sindex_nms(self):
        import cv2

        from geoai.extract import CarDetector, ObjectDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            # 64 separated 8x8 objects in a 512x512 mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            conf = np.zeros((512, 512), dtype=np.uint8)
            for r in range(8):
                for c in range(8):
                    y0, x0 = r * 64 + 10, c * 64 + 10
                    mask[y0 : y0 + 8, x0 : x0 + 8] = 255
                    conf[y0 : y0 + 8, x0 : x0 + 8] = 230

            single_path = _write_geotiff(os.path.join(tmpdir, "objs.tif"), mask)
            two_band_path = _write_geotiff(
                os.path.join(tmpdir, "cars.tif"), np.stack([mask, conf], axis=0)
            )

            detector = ObjectDetector.__new__(ObjectDetector)
            detector.simplify_tolerance = 1.0
            detector.mask_threshold = 0.5
            detector.min_object_area = 10
            detector.max_object_area = None
            detector.nms_iou_threshold = 0.5

            real_find_contours = cv2.findContours
            contour_shapes = []

            def spy_find_contours(img, *args, **kwargs):
                contour_shapes.append(img.shape)
                return real_find_contours(img, *args, **kwargs)

            with patch.object(cv2, "findContours", side_effect=spy_find_contours):
                gdf = detector.masks_to_vector(
                    single_path, min_object_area=10, regularize=False
                )
            self.assertEqual(len(gdf), 64)
            # Every component must be sliced to its tight bounding box, not full 512x512
            self.assertTrue(all(s == (8, 8) for s in contour_shapes))

            car_contour_shapes = []

            def spy_car_contours(img, *args, **kwargs):
                car_contour_shapes.append(img.shape)
                return real_find_contours(img, *args, **kwargs)

            car_det = CarDetector.__new__(CarDetector)
            with patch.object(cv2, "findContours", side_effect=spy_car_contours):
                gdf_cars_1 = car_det.vectorize_masks(
                    two_band_path,
                    confidence_threshold=0.5,
                    min_object_area=10,
                    n_workers=1,
                )
            gdf_cars_2 = car_det.vectorize_masks(
                two_band_path,
                confidence_threshold=0.5,
                min_object_area=10,
                n_workers=2,
            )
            self.assertEqual(len(gdf_cars_1), 64)
            self.assertEqual(len(gdf_cars_2), 64)
            self.assertTrue(all(s == (8, 8) for s in car_contour_shapes))

            # Spatial-index NMS check with overlapping + disjoint polygons
            geoms = [box(i * 20, 0, i * 20 + 10, 10) for i in range(100)]
            confs = [0.9] * 100
            # Add lower-confidence duplicates overlapping first 10 polygons
            geoms += [box(i * 20 + 1, 0, i * 20 + 11, 10) for i in range(10)]
            confs += [0.4] * 10
            nms_gdf = gpd.GeoDataFrame(
                {"geometry": geoms, "confidence": confs}, crs="EPSG:32617"
            )
            filtered = detector.filter_overlapping_polygons(
                nms_gdf, nms_iou_threshold=0.5
            )
            self.assertEqual(len(filtered), 100)
            self.assertTrue((filtered["confidence"] >= 0.9).all())

            # Dense nested polygons: suppressed candidates must be skipped
            # without repeating sindex.query on every suppressed polygon.
            nested_geoms = [
                box(0, 0, 100.0 - 0.01 * k, 100.0 - 0.01 * k) for k in range(200)
            ]
            nested_confs = [1.0 - 0.001 * k for k in range(200)]
            nested_gdf = gpd.GeoDataFrame(
                {"geometry": nested_geoms, "confidence": nested_confs},
                crs="EPSG:32617",
            )
            query_calls = [0]
            orig_sindex_prop = gpd.GeoDataFrame.sindex

            class CountingSindex:
                def __init__(self, inner):
                    self._inner = inner

                def query(self, *args, **kwargs):
                    query_calls[0] += 1
                    return self._inner.query(*args, **kwargs)

            with patch.object(
                gpd.GeoDataFrame,
                "sindex",
                new=property(
                    lambda self_gdf: CountingSindex(orig_sindex_prop.fget(self_gdf))
                ),
            ):
                nested_filtered = detector.filter_overlapping_polygons(
                    nested_gdf, nms_iou_threshold=0.5
                )
            self.assertEqual(len(nested_filtered), 1)
            self.assertEqual(query_calls[0], 1)

    def test_load_weights_uses_weights_only_true(self):
        from geoai.extract import ObjectDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(2, 2)
            ckpt = os.path.join(tmpdir, "model.pth")
            torch.save(model.state_dict(), ckpt)

            detector = ObjectDetector.__new__(ObjectDetector)
            detector.device = torch.device("cpu")
            detector.model = torch.nn.Linear(2, 2)

            real_load = torch.load
            seen_kwargs = []

            def spy_load(*args, **kwargs):
                seen_kwargs.append(kwargs)
                return real_load(*args, **kwargs)

            with patch("geoai.extract.torch.load", side_effect=spy_load):
                detector.load_weights(ckpt)

            self.assertEqual(len(seen_kwargs), 1)
            self.assertTrue(seen_kwargs[0].get("weights_only"))

            # Restricted-load rejection must NOT auto-retry without opt-in
            reject_calls = []

            def rejecting_load(*args, **kwargs):
                reject_calls.append(dict(kwargs))
                if kwargs.get("weights_only") is True:
                    raise pickle.UnpicklingError("legacy global")
                return model.state_dict()

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("geoai.extract.torch.load", side_effect=rejecting_load),
            ):
                os.environ.pop("GEOAI_ALLOW_UNSAFE_TORCH_LOAD", None)
                with self.assertRaises(RuntimeError) as ctx:
                    detector.load_weights(ckpt)
            self.assertIn("allow_unsafe=True", str(ctx.exception))
            self.assertIn("GEOAI_ALLOW_UNSAFE_TORCH_LOAD=1", str(ctx.exception))
            self.assertEqual(len(reject_calls), 1)

            # Explicit allow_unsafe=True opts in to fallback
            reject_calls.clear()
            with (
                patch("geoai.extract.torch.load", side_effect=rejecting_load),
                self.assertLogs("geoai.extract", level="WARNING"),
            ):
                detector.load_weights(ckpt, allow_unsafe=True)
            self.assertEqual(len(reject_calls), 2)
            self.assertIs(reject_calls[0].get("weights_only"), True)
            self.assertIs(reject_calls[1].get("weights_only"), False)

            # Explicit GEOAI_ALLOW_UNSAFE_TORCH_LOAD=1 opts in to fallback
            reject_calls.clear()
            with (
                patch.dict(os.environ, {"GEOAI_ALLOW_UNSAFE_TORCH_LOAD": "1"}),
                patch("geoai.extract.torch.load", side_effect=rejecting_load),
                self.assertLogs("geoai.extract", level="WARNING"),
            ):
                detector.load_weights(ckpt)
            self.assertEqual(len(reject_calls), 2)
            self.assertIs(reject_calls[0].get("weights_only"), True)
            self.assertIs(reject_calls[1].get("weights_only"), False)


class TestRasterUtilsFixes(unittest.TestCase):
    """Group 2: geoai/utils/raster.py fixes."""

    def test_raster_to_vector_unique_attribute_value_monotonic(self):
        from geoai.utils.raster import raster_to_vector

        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((64, 64), dtype=np.uint8)
            arr[2:12, 2:12] = 1
            arr[2:12, 20:30] = 1
            arr[30:40, 2:12] = 2
            arr[30:40, 20:30] = 2
            tif = _write_geotiff(os.path.join(tmpdir, "classes.tif"), arr)

            gdf_all = raster_to_vector(
                tif, min_area=1, unique_attribute_value=True, attribute_name="id"
            )
            self.assertEqual(sorted(gdf_all["id"].tolist()), [1, 2, 3, 4])

            gdf_sel = raster_to_vector(
                tif,
                min_area=1,
                class_values=[1, 2],
                unique_attribute_value=True,
                attribute_name="id",
            )
            self.assertEqual(sorted(gdf_sel["id"].tolist()), [1, 2, 3, 4])

    def test_clip_raster_by_bbox_fractional_geo_window(self):
        from geoai.utils.raster import clip_raster_by_bbox

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.arange(3 * 100 * 100, dtype=np.uint16).reshape(3, 100, 100)
            in_tif = _write_geotiff(
                os.path.join(tmpdir, "in.tif"),
                data,
                origin=(1000.0, 2000.0),
                pixel_size=2.0,
            )
            out_tif = os.path.join(tmpdir, "out.tif")

            # Fractional offsets (1000 + 2*10.4 = 1020.8, 1000 + 2*60.9 = 1121.8)
            # and partially extending beyond north edge (2010 > 2000).
            bbox = (1020.8, 1900.3, 1121.8, 2010.0)
            clip_raster_by_bbox(in_tif, out_tif, bbox, bbox_type="geo")

            with rasterio.open(out_tif) as dst:
                arr = dst.read()
                self.assertEqual(arr.shape[1], dst.height)
                self.assertEqual(arr.shape[2], dst.width)
                # Pixel scale must remain exactly 2.0 (un-stretched)
                self.assertAlmostEqual(dst.transform.a, 2.0, places=6)
                self.assertAlmostEqual(dst.transform.e, -2.0, places=6)

            # Bbox extending past west edge (-0.75..0.625 in pixel units)
            # overlaps col 0 and must succeed rather than rounding away overlap.
            unit_tif = _write_geotiff(
                os.path.join(tmpdir, "unit.tif"),
                np.arange(100, dtype=np.uint16).reshape(1, 10, 10),
                origin=(0.0, 10.0),
                pixel_size=1.0,
            )
            out_edge = os.path.join(tmpdir, "out_edge.tif")
            clip_raster_by_bbox(
                unit_tif, out_edge, (-0.75, 8.0, 0.625, 10.0), bbox_type="geo"
            )
            with rasterio.open(out_edge) as dst:
                self.assertEqual(dst.width, 1)
                self.assertEqual(dst.height, 2)
                self.assertAlmostEqual(dst.transform.c, 0.0, places=6)
                self.assertAlmostEqual(dst.transform.f, 10.0, places=6)

            # Strictly non-overlapping bbox still raises ValueError
            with self.assertRaises(ValueError):
                clip_raster_by_bbox(
                    unit_tif,
                    os.path.join(tmpdir, "out_empty.tif"),
                    (-2.0, 8.0, -0.5, 10.0),
                    bbox_type="geo",
                )

    def test_clean_instance_mask_and_masks_to_vector_tight_slices(self):
        from geoai.utils.raster import clean_instance_mask, masks_to_vector

        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((256, 256), dtype=np.uint16)
            # 20 valid instances + 1 small instance + 1 hole + 1 disconnected fragment
            idx = 1
            for r in range(4):
                for c in range(5):
                    y0, x0 = r * 60 + 10, c * 48 + 10
                    mask[y0 : y0 + 24, x0 : x0 + 24] = idx
                    idx += 1
            # Put a 4px hole in instance 1
            mask[20:22, 20:22] = 0
            # Put a small 4px instance (id=99)
            mask[240:242, 240:242] = 99
            # Put a disconnected 4px fragment of instance 1
            mask[250:252, 10:12] = 1

            in_tif = _write_geotiff(os.path.join(tmpdir, "inst.tif"), mask, nodata=0)
            out_tif = os.path.join(tmpdir, "inst_clean.tif")

            t0 = time.perf_counter()
            clean_instance_mask(
                in_tif,
                out_tif,
                min_area=20,
                fill_holes=True,
                max_hole_area=50,
                smooth=True,
                smooth_sigma=1.0,
            )
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 5.0)

            with rasterio.open(out_tif) as src:
                cleaned = src.read(1)
            self.assertFalse((cleaned == 99).any())
            self.assertEqual(int(cleaned[20, 20]), 1)
            self.assertEqual(int(cleaned[250, 10]), 0)
            self.assertEqual(len(np.unique(cleaned)) - 1, 20)

            # Float32 instance mask must use membership path & compact remap
            # instead of indexing keep_lut with float32 values.
            fmask = np.zeros((32, 32), dtype=np.float32)
            fmask[4:20, 4:20] = 1.0  # 256 px
            fmask[10:12, 10:12] = 0.0  # 4 px hole inside 1.0
            fmask[28, 28] = 2.0  # 1 px small instance
            fin_tif = _write_geotiff(os.path.join(tmpdir, "finst.tif"), fmask)
            fout_nosmooth = os.path.join(tmpdir, "finst_nosmooth.tif")
            clean_instance_mask(
                fin_tif,
                fout_nosmooth,
                min_area=10,
                fill_holes=False,
                smooth=False,
            )
            with rasterio.open(fout_nosmooth) as src:
                fcleaned = src.read(1)
            self.assertFalse((fcleaned == 2.0).any())
            self.assertTrue((fcleaned == 1.0).any())
            self.assertEqual(float(fcleaned[10, 10]), 0.0)

            fout_smooth = os.path.join(tmpdir, "finst_smooth.tif")
            clean_instance_mask(
                fin_tif,
                fout_smooth,
                min_area=10,
                fill_holes=True,
                max_hole_area=20,
                smooth=True,
                smooth_sigma=1.0,
            )
            with rasterio.open(fout_smooth) as src:
                fcleaned2 = src.read(1)
            self.assertFalse((fcleaned2 == 2.0).any())
            self.assertEqual(float(fcleaned2[10, 10]), 1.0)

            # masks_to_vector in utils.raster with >50-vertex circle and simplify_tolerance=1.0
            yy, xx = np.ogrid[:128, :128]
            circle = (((yy - 64) ** 2 + (xx - 64) ** 2) <= 40**2).astype(np.uint8) * 255
            circ_tif = _write_geotiff(os.path.join(tmpdir, "circ.tif"), circle)
            gdf = masks_to_vector(circ_tif, simplify_tolerance=1.0, min_object_area=50)
            self.assertEqual(len(gdf), 1)
            self.assertGreater(gdf.geometry.iloc[0].area, 4000.0)

    def test_vector_to_raster_clears_incompatible_nodata(self):
        from geoai.utils.raster import vector_to_raster

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = _write_geotiff(
                os.path.join(tmpdir, "ref_f32.tif"),
                np.ones((32, 32), dtype=np.float32),
                origin=(0.0, 32.0),
                nodata=-9999.0,
            )
            gdf = gpd.GeoDataFrame(
                {"class": [1]},
                geometry=[box(5.0, 5.0, 25.0, 25.0)],
                crs="EPSG:32617",
            )
            out_tif = os.path.join(tmpdir, "out_u8.tif")
            vector_to_raster(
                gdf,
                output_path=out_tif,
                reference_raster=ref_path,
                dtype=np.uint8,
            )
            with rasterio.open(out_tif) as src:
                self.assertEqual(src.dtypes[0], "uint8")
                self.assertIsNone(src.nodata)
                self.assertGreater(int(src.read(1).sum()), 0)

            # Float64 reference with out-of-range float32 nodata (1e100) is cleared
            ref_f64_oob = _write_geotiff(
                os.path.join(tmpdir, "ref_f64_oob.tif"),
                np.ones((32, 32), dtype=np.float64),
                origin=(0.0, 32.0),
                nodata=1e100,
            )
            out_f32_oob = os.path.join(tmpdir, "out_f32_oob.tif")
            vector_to_raster(
                gdf,
                output_path=out_f32_oob,
                reference_raster=ref_f64_oob,
                dtype=np.float32,
            )
            with rasterio.open(out_f32_oob) as src:
                self.assertEqual(src.dtypes[0], "float32")
                self.assertIsNone(src.nodata)

            # Float64 reference with NaN nodata is preserved for float32
            ref_f64_nan = _write_geotiff(
                os.path.join(tmpdir, "ref_f64_nan.tif"),
                np.ones((32, 32), dtype=np.float64),
                origin=(0.0, 32.0),
                nodata=float("nan"),
            )
            out_f32_nan = os.path.join(tmpdir, "out_f32_nan.tif")
            vector_to_raster(
                gdf,
                output_path=out_f32_nan,
                reference_raster=ref_f64_nan,
                dtype=np.float32,
            )
            with rasterio.open(out_f32_nan) as src:
                self.assertTrue(np.isnan(src.nodata))

            # Explicit caller nodata is left untouched
            out_f32_explicit = os.path.join(tmpdir, "out_f32_explicit.tif")
            vector_to_raster(
                gdf,
                output_path=out_f32_explicit,
                reference_raster=ref_f64_oob,
                dtype=np.float32,
                nodata=-1.0,
            )
            with rasterio.open(out_f32_explicit) as src:
                self.assertEqual(src.nodata, -1.0)

    def test_calc_stats_empty_guard_and_context_manager(self):
        from geoai.utils.raster import calc_stats

        empty_ds = MagicMock()
        empty_ds.index.bounds = (0, 0, 1, 1)
        empty_ds.index.intersection.return_value = []
        with self.assertRaises(ValueError):
            calc_stats(empty_ds)

        # Empty rtree index reporting get_size() == 0 raises ValueError before query
        empty_sized = MagicMock()
        empty_sized.index.get_size.return_value = 0
        with self.assertRaises(ValueError):
            calc_stats(empty_sized)
        empty_sized.index.intersection.assert_not_called()

        # Empty rtree index with inverted bounds raises ValueError before intersection
        inverted_ds = MagicMock()
        inverted_ds.index.bounds = (1e308, 1e308, -1e308, -1e308)
        with self.assertRaises(ValueError):
            calc_stats(inverted_ds)
        inverted_ds.index.intersection.assert_not_called()

        # Simulated RTreeError from empty index is converted to ValueError
        class RTreeError(Exception):
            pass

        rtree_err_ds = MagicMock()
        rtree_err_ds.index.bounds = (0, 0, 1, 1)
        rtree_err_ds.index.intersection.side_effect = RTreeError("empty index")
        with self.assertRaises(ValueError):
            calc_stats(rtree_err_ds)

        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.full((2, 16, 16), 100, dtype=np.uint8)
            arr[1] = 200
            tif = _write_geotiff(os.path.join(tmpdir, "s.tif"), arr)
            item = MagicMock(object=tif)
            ds = MagicMock()
            ds.index.bounds = (0, 0, 1, 1)
            ds.index.intersection.return_value = [item]

            opened_handles = []
            real_open = rasterio.open

            def spy_open(*args, **kwargs):
                h = real_open(*args, **kwargs)
                opened_handles.append(h)
                return h

            with patch("geoai.utils.raster.rasterio.open", side_effect=spy_open):
                mean, std = calc_stats(ds, divide_by=10.0)

            np.testing.assert_allclose(mean, [10.0, 20.0])
            np.testing.assert_allclose(std, [0.0, 0.0])
            self.assertTrue(all(h.closed for h in opened_handles))

    def test_mosaic_geotiffs_tempdir_under_output_dir(self):
        from geoai.utils.raster import mosaic_geotiffs

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "inputs")
            out_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(in_dir)
            _write_geotiff(
                os.path.join(in_dir, "tile1.tif"), np.ones((16, 16), dtype=np.uint8)
            )
            out_tif = os.path.join(out_dir, "mosaic.tif")

            fake_band = MagicMock()
            fake_band.GetNoDataValue.return_value = None
            fake_ds = MagicMock()
            fake_ds.GetMetadata.return_value = {}
            fake_ds.GetRasterBand.return_value = fake_band

            fake_gdal = MagicMock()
            fake_gdal.Open.return_value = fake_ds
            fake_gdal.GA_Update = 1

            fake_osgeo = MagicMock()
            fake_osgeo.gdal = fake_gdal

            seen_dirs = []
            real_tempdir = tempfile.TemporaryDirectory

            def spy_tempdir(*args, **kwargs):
                seen_dirs.append(kwargs.get("dir"))
                return real_tempdir(*args, **kwargs)

            with (
                patch.dict(sys.modules, {"osgeo": fake_osgeo, "osgeo.gdal": fake_gdal}),
                patch(
                    "geoai.utils.raster.tempfile.TemporaryDirectory",
                    side_effect=spy_tempdir,
                ),
            ):
                ok = mosaic_geotiffs(in_dir, out_tif)

            self.assertTrue(ok)
            self.assertEqual(seen_dirs, [os.path.abspath(out_dir)])

    def test_stack_bands_uses_tempdir_for_vrt(self):
        from geoai.utils.raster import stack_bands

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = _write_geotiff(
                os.path.join(tmpdir, "b1.tif"), np.ones((16, 16), dtype=np.uint8)
            )
            f2 = _write_geotiff(
                os.path.join(tmpdir, "b2.tif"), np.ones((16, 16), dtype=np.uint8)
            )
            out_tif = os.path.join(tmpdir, "stacked.tif")
            captured_cmds = []

            def fake_run(cmd, check=False, **kwargs):
                captured_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0)

            with patch("geoai.utils.raster.subprocess.run", side_effect=fake_run):
                stack_bands([f1, f2], out_tif)

            self.assertEqual(len(captured_cmds), 2)
            vrt_arg = captured_cmds[0][2]
            self.assertNotEqual(vrt_arg, "stack.vrt")
            self.assertTrue(os.path.isabs(vrt_arg))
            self.assertFalse(os.path.exists(vrt_arg))

            # Explicit temp_vrt must NOT be deleted if _run_stack fails
            existing_vrt = os.path.join(tmpdir, "caller.vrt")
            with open(existing_vrt, "w") as fh:
                fh.write("keep me")

            with patch(
                "geoai.utils.raster.subprocess.run",
                side_effect=FileNotFoundError("gdalbuildvrt missing"),
            ):
                with self.assertRaises(FileNotFoundError):
                    stack_bands([f1, f2], out_tif, temp_vrt=existing_vrt)
            self.assertTrue(os.path.exists(existing_vrt))

            # Explicit temp_vrt IS removed after _run_stack succeeds
            with patch("geoai.utils.raster.subprocess.run", side_effect=fake_run):
                stack_bands([f1, f2], out_tif, temp_vrt=existing_vrt)
            self.assertFalse(os.path.exists(existing_vrt))

    def test_masks_to_vector_nms_skips_suppressed_nested_polygons(self):
        import cv2

        from geoai.utils.raster import masks_to_vector

        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((128, 128), dtype=np.uint8)
            mask[10:110, 10:110] = 255
            tif = _write_geotiff(os.path.join(tmpdir, "nested.tif"), mask)

            # Return 50 nearly identical nested square contours to trigger dense NMS
            nested_contours = [
                np.array(
                    [
                        [[10, 10]],
                        [[109 - (k % 2), 10]],
                        [[109 - (k % 2), 109 - (k % 2)]],
                        [[10, 109 - (k % 2)]],
                    ],
                    dtype=np.int32,
                )
                for k in range(50)
            ]

            query_calls = [0]
            orig_sindex_prop = gpd.GeoDataFrame.sindex

            class CountingSindex:
                def __init__(self, inner):
                    self._inner = inner

                def query(self, *args, **kwargs):
                    query_calls[0] += 1
                    return self._inner.query(*args, **kwargs)

            with (
                patch.object(cv2, "findContours", return_value=(nested_contours, None)),
                patch.object(
                    gpd.GeoDataFrame,
                    "sindex",
                    new=property(
                        lambda self_gdf: CountingSindex(orig_sindex_prop.fget(self_gdf))
                    ),
                ),
            ):
                gdf = masks_to_vector(
                    tif,
                    simplify_tolerance=0.0,
                    min_object_area=10,
                    nms_iou_threshold=0.5,
                )

            self.assertEqual(len(gdf), 1)
            self.assertEqual(query_calls[0], 1)


class TestGeometryFixes(unittest.TestCase):
    """Group 3: geoai/utils/geometry.py region_groups fixes."""

    def test_region_groups_no_mutation_elongation_and_vector_merge(self):
        import rioxarray as rxr

        from geoai.utils.geometry import region_groups

        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((40, 40), dtype=np.uint8)
            arr[2:12, 2:12] = 1  # 100 px
            arr[20:35, 20:30] = 1  # 150 px
            arr[38:39, 38:39] = 1  # 1 px (removed by min_size=10)
            tif = _write_geotiff(
                os.path.join(tmpdir, "rg.tif"), arr, origin=(0.0, 40.0)
            )
            da_orig = rxr.open_rasterio(tif).sel(band=1)
            snapshot = da_orig.values.copy()

            out_vec = os.path.join(tmpdir, "rg.gpkg")
            da_out, gdf = region_groups(da_orig, min_size=10, out_vector=out_vec)

            # Caller's DataArray must not be mutated in place
            np.testing.assert_array_equal(da_orig.values, snapshot)
            self.assertEqual(len(gdf), 2)
            # Merge on "label" must succeed with no NaN areas
            self.assertFalse(gdf["area"].isna().any())
            self.assertEqual(sorted(gdf["area"].tolist()), [100.0, 150.0])

        # Guard missing axis columns and zero axis_minor_length (1-px line)
        line_arr = np.zeros((20, 20), dtype=np.uint8)
        line_arr[5, 2:15] = 1  # 1-px thin horizontal line -> axis_minor_length == 0
        _, df_line = region_groups(line_arr, min_size=1)
        self.assertEqual(len(df_line), 1)
        self.assertTrue(np.isnan(df_line["elongation"].iloc[0]))

        _, df_custom = region_groups(line_arr, min_size=1, properties=["label", "area"])
        self.assertNotIn("elongation", df_custom.columns)

        # With spacing=(2, 2), a 4-pixel region has measured area 16:
        # kept when min_size=10, removed when max_size=12.
        sp_arr = np.zeros((10, 10), dtype=np.uint8)
        sp_arr[2:4, 2:4] = 1  # 4 pixels -> area 16 with spacing=(2, 2)
        _, df_sp_keep = region_groups(
            sp_arr, min_size=10, properties=["label", "area"], spacing=(2, 2)
        )
        self.assertEqual(len(df_sp_keep), 1)
        self.assertAlmostEqual(float(df_sp_keep["area"].iloc[0]), 16.0)

        _, df_sp_scalar = region_groups(
            sp_arr, min_size=10, properties=["label", "area"], spacing=2
        )
        self.assertEqual(len(df_sp_scalar), 1)
        self.assertAlmostEqual(float(df_sp_scalar["area"].iloc[0]), 16.0)

        _, df_sp_max = region_groups(
            sp_arr,
            min_size=1,
            max_size=12,
            properties=["label", "area"],
            spacing=(2, 2),
        )
        self.assertEqual(len(df_sp_max), 0)

        # Without spacing, 4-pixel region is removed by min_size=10
        _, df_no_sp = region_groups(sp_arr, min_size=10, properties=["label", "area"])
        self.assertEqual(len(df_no_sp), 0)


class TestTrainingUtilsFixes(unittest.TestCase):
    """Group 4: geoai/utils/training.py fixes."""

    def test_batch_coco_unique_ann_ids_and_pascal_voc_without_lxml(self):
        from geoai.utils.training import export_geotiff_tiles_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            imgs_dir = os.path.join(tmpdir, "imgs")
            msks_dir = os.path.join(tmpdir, "msks")
            os.makedirs(imgs_dir)
            os.makedirs(msks_dir)

            for name, x0 in [("a", 0.0), ("b", 200.0)]:
                _write_geotiff(
                    os.path.join(imgs_dir, f"{name}.tif"),
                    np.full((3, 64, 64), 128, dtype=np.uint8),
                    origin=(x0, 100.0),
                )
                msk = np.zeros((64, 64), dtype=np.uint8)
                msk[5:20, 5:20] = 1
                msk[35:50, 35:50] = 1
                _write_geotiff(
                    os.path.join(msks_dir, f"{name}.tif"),
                    msk,
                    origin=(x0, 100.0),
                )

            out_coco = os.path.join(tmpdir, "out_coco")
            export_geotiff_tiles_batch(
                images_folder=imgs_dir,
                masks_folder=msks_dir,
                output_folder=out_coco,
                tile_size=64,
                stride=64,
                metadata_format="COCO",
                quiet=True,
            )
            with open(os.path.join(out_coco, "annotations", "instances.json")) as fh:
                coco = json.load(fh)
            ann_ids = [ann["id"] for ann in coco["annotations"]]
            self.assertEqual(len(ann_ids), 4)
            self.assertEqual(len(set(ann_ids)), 4)

            # PASCAL_VOC must work without lxml installed
            out_voc = os.path.join(tmpdir, "out_voc")
            with patch.dict(sys.modules, {"lxml": None, "lxml.etree": None}):
                export_geotiff_tiles_batch(
                    images_folder=imgs_dir,
                    masks_folder=msks_dir,
                    output_folder=out_voc,
                    tile_size=64,
                    stride=64,
                    metadata_format="PASCAL_VOC",
                    quiet=True,
                )
            xml_files = [
                f
                for f in os.listdir(os.path.join(out_voc, "annotations"))
                if f.endswith(".xml")
            ]
            self.assertEqual(len(xml_files), 2)

    def test_small_raster_guard_and_compute_tile_window(self):
        from geoai.utils.training import (
            _compute_tile_window,
            export_geotiff_tiles,
            export_geotiff_tiles_batch,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, "imgs")
            os.makedirs(img_dir)
            small_tif = _write_geotiff(
                os.path.join(img_dir, "tiny.tif"),
                np.full((3, 32, 32), 77, dtype=np.uint8),
                origin=(0.0, 32.0),
            )
            with rasterio.open(small_tif) as src:
                win, *_ = _compute_tile_window(0, 0, 32, 32, 64, 64, src)
                self.assertEqual(win.col_off, 0)
                self.assertEqual(win.row_off, 0)

            out_single = os.path.join(tmpdir, "single")
            export_geotiff_tiles(
                in_raster=small_tif,
                out_folder=out_single,
                tile_size=64,
                stride=32,
                quiet=True,
            )
            tiles = os.listdir(os.path.join(out_single, "images"))
            self.assertEqual(len(tiles), 1)
            with rasterio.open(os.path.join(out_single, "images", tiles[0])) as ds:
                self.assertEqual((ds.height, ds.width), (64, 64))

            out_batch = os.path.join(tmpdir, "batch")
            bstats = export_geotiff_tiles_batch(
                images_folder=img_dir,
                output_folder=out_batch,
                tile_size=64,
                stride=32,
                quiet=True,
            )
            self.assertEqual(bstats["total_tiles"], 1)

    def test_load_class_data_raster_scans_blocks_for_rare_class(self):
        from geoai.utils.training import _load_class_data_raster

        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((2048, 2048), dtype=np.uint8)
            arr[:1024, :] = 1
            # Single-pixel rare class skipped by 1000x1000 nearest-neighbor decimation
            arr[1501, 1501] = 9
            tif = _write_geotiff(os.path.join(tmpdir, "rare.tif"), arr)

            class_to_id, _ = _load_class_data_raster(
                tif, rasterio.crs.CRS.from_epsg(32617), quiet=True
            )
            self.assertIn(9, class_to_id)
            self.assertEqual(class_to_id, {1: 1, 9: 2})

            # uint64 class values exceeding int64 max (2**63 - 1) must not overflow
            big_val = (1 << 63) + 1
            arr_u64 = np.zeros((16, 16), dtype=np.uint64)
            arr_u64[2:4, 2:4] = np.uint64(big_val)
            tif_u64 = _write_geotiff(os.path.join(tmpdir, "u64.tif"), arr_u64)
            class_to_id_u64, cats_u64 = _load_class_data_raster(
                tif_u64,
                rasterio.crs.CRS.from_epsg(32617),
                quiet=True,
                metadata_format="COCO",
            )
            self.assertEqual(class_to_id_u64, {big_val: 1})
            self.assertEqual(cats_u64[0]["name"], str(big_val))

    def test_rasterize_label_from_vector_single_call_and_precedence(self):
        from geoai.utils.training import _rasterize_label_from_vector

        transform = from_origin(0.0, 32.0, 1.0, 1.0)
        win_bounds = box(0.0, 0.0, 32.0, 32.0)
        gdf = gpd.GeoDataFrame(
            {
                "cls": ["b", "a", "c"],
                "geometry": [
                    box(4.0, 4.0, 20.0, 20.0),
                    box(8.0, 8.0, 24.0, 24.0),
                    box(10.0, 10.0, 16.0, 16.0),
                ],
            },
            crs="EPSG:32617",
        )
        class_to_id = {"a": 1, "b": 2, "c": 3}

        real_rasterize = rasterio.features.rasterize
        call_count = [0]

        def spy_rasterize(*args, **kwargs):
            call_count[0] += 1
            return real_rasterize(*args, **kwargs)

        with patch(
            "geoai.utils.training.features.rasterize", side_effect=spy_rasterize
        ):
            mask, has_feat, _, errs = _rasterize_label_from_vector(
                gdf, win_bounds, transform, 32, "cls", class_to_id
            )

        self.assertEqual(errs, 0)
        self.assertTrue(has_feat)
        self.assertEqual(call_count[0], 1)
        # Pixel inside all three boxes (col=12, row=32-12=20 -> y=12) must get max class ID 3
        self.assertEqual(int(mask[19, 12]), 3)

    def test_export_geotiff_tiles_feature_pixels_no_double_count(self):
        from geoai.utils.training import export_geotiff_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            img = _write_geotiff(
                os.path.join(tmpdir, "img.tif"),
                np.full((3, 64, 64), 100, dtype=np.uint8),
                origin=(0.0, 64.0),
            )
            msk_arr = np.zeros((64, 64), dtype=np.uint8)
            msk_arr[10:20, 10:20] = 1  # exactly 100 non-zero pixels
            msk = _write_geotiff(
                os.path.join(tmpdir, "msk.tif"), msk_arr, origin=(0.0, 64.0)
            )
            out_dir = os.path.join(tmpdir, "out")
            # Capture stats by spying on _log_export_summary or inspecting logger
            captured = {}
            from geoai.utils import training as tr_mod

            orig_summary = tr_mod._log_export_summary

            def spy_summary(stats, *args, **kwargs):
                captured.update(stats)
                return orig_summary(stats, *args, **kwargs)

            with patch.object(tr_mod, "_log_export_summary", side_effect=spy_summary):
                export_geotiff_tiles(
                    in_raster=img,
                    out_folder=out_dir,
                    in_class_data=msk,
                    tile_size=64,
                    stride=64,
                    quiet=False,
                )
            self.assertEqual(captured["feature_pixels"], 100)

    def test_export_training_data_point_loc_and_geotiff_transforms(self):
        from geoai.utils.training import export_training_data

        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = _write_geotiff(
                os.path.join(tmpdir, "r1.tif"),
                np.full((3, 64, 64), 100, dtype=np.uint8),
                origin=(0.0, 64.0),
            )
            img2 = _write_geotiff(
                os.path.join(tmpdir, "r2.tif"),
                np.full((1, 64, 64), 50, dtype=np.uint8),
                origin=(0.0, 64.0),
            )

            # 1) Point geometry (vector_bounds.area == 0) must not ZeroDivisionError
            pts = gpd.GeoDataFrame(
                {"class": [1]}, geometry=[Point(32.0, 32.0)], crs="EPSG:32617"
            )
            pts_path = os.path.join(tmpdir, "pts.geojson")
            pts.to_file(pts_path, driver="GeoJSON")
            stats_pt, _ = export_training_data(
                in_raster=img1,
                out_folder=os.path.join(tmpdir, "out_pt"),
                in_class_data=pts_path,
                tile_size_x=64,
                tile_size_y=64,
                quiet=True,
            )
            self.assertEqual(stats_pt["total_tiles"], 1)

            # 2) Non-RangeIndex GeoDataFrame with CENTERED_ON_FEATURE must use .loc
            poly_gdf = gpd.GeoDataFrame(
                {"class": [1]},
                geometry=[box(16.0, 16.0, 48.0, 48.0)],
                crs="EPSG:32617",
                index=[42],
            )
            with patch("geoai.utils.training.gpd.read_file", return_value=poly_gdf):
                stats_loc, _ = export_training_data(
                    in_raster=img1,
                    out_folder=os.path.join(tmpdir, "out_loc"),
                    in_class_data="unused.geojson",
                    tile_size_x=64,
                    tile_size_y=64,
                    crop_mode="CENTERED_ON_FEATURE",
                    quiet=True,
                )
            self.assertEqual(stats_loc["total_tiles"], 1)

            # 3) save_geotiff=True must write in_raster2 + blacken_around_feature
            poly_path = os.path.join(tmpdir, "poly.geojson")
            poly_gdf.reset_index(drop=True).to_file(poly_path, driver="GeoJSON")
            out_xf = os.path.join(tmpdir, "out_xf")
            export_training_data(
                in_raster=img1,
                in_raster2=img2,
                out_folder=out_xf,
                in_class_data=poly_path,
                tile_size_x=64,
                tile_size_y=64,
                blacken_around_feature=True,
                save_geotiff=True,
                quiet=True,
            )
            tile_tif = os.path.join(out_xf, "images", "tile_000000.tif")
            with rasterio.open(tile_tif) as ds:
                arr = ds.read()
            self.assertEqual(arr.shape, (4, 64, 64))
            # Corner pixel (0, 0) is outside feature box -> blackened to 0
            self.assertEqual(int(arr[:, 0, 0].sum()), 0)
            # Center pixel (32, 32) is inside feature box -> retains band values (100 and 50)
            self.assertEqual(int(arr[0, 32, 32]), 100)
            self.assertEqual(int(arr[3, 32, 32]), 50)


class TestLandcoverUtilsFixes(unittest.TestCase):
    """Group 5: geoai/landcover_utils.py fixes."""

    def test_export_landcover_tiles_no_mutation_ratio_decoupled_and_cleanup(self):
        from geoai.landcover_utils import export_landcover_tiles

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = _write_geotiff(
                os.path.join(tmpdir, "img.tif"),
                np.full((3, 64, 128), 100, dtype=np.uint8),
                origin=(0.0, 64.0),
            )
            # Left tile (0..64): small 10x10 box (~2.4% coverage)
            # Right tile (64..128): large 56x56 box (~76.6% coverage)
            gdf = gpd.GeoDataFrame(
                {"class": [1, 1]},
                geometry=[box(10.0, 10.0, 20.0, 20.0), box(68.0, 4.0, 124.0, 60.0)],
                crs="EPSG:32617",
            )
            orig_wkt = [g.wkt for g in gdf.geometry]

            stats = export_landcover_tiles(
                in_raster=img_path,
                out_folder=os.path.join(tmpdir, "out"),
                in_class_data=gdf,
                tile_size=64,
                stride=64,
                buffer_radius=2.0,
                skip_empty_tiles=False,
                min_feature_ratio=0.5,
                quiet=True,
            )
            # Caller GeoDataFrame must remain unmutated
            self.assertEqual([g.wkt for g in gdf.geometry], orig_wkt)
            # min_feature_ratio must filter even when skip_empty_tiles=False
            self.assertEqual(stats["tiles_exported"], 1)
            self.assertEqual(stats["tiles_skipped_ratio"], 1)

            # ExitStack must close class_src even if an exception occurs mid-export
            msk_path = _write_geotiff(
                os.path.join(tmpdir, "msk.tif"),
                np.ones((64, 128), dtype=np.uint8),
                origin=(0.0, 64.0),
            )
            opened = []
            real_open = rasterio.open

            def spy_open(path, mode="r", *args, **kwargs):
                if mode == "w":
                    raise RuntimeError("disk full")
                h = real_open(path, mode, *args, **kwargs)
                opened.append(h)
                return h

            with patch("geoai.landcover_utils.rasterio.open", side_effect=spy_open):
                with self.assertRaises(RuntimeError):
                    export_landcover_tiles(
                        in_raster=img_path,
                        out_folder=os.path.join(tmpdir, "out_err"),
                        in_class_data=msk_path,
                        tile_size=64,
                        stride=64,
                        quiet=True,
                    )
            self.assertGreaterEqual(len(opened), 2)
            self.assertTrue(all(h.closed for h in opened))

    def test_lirrn_fresh_rounds_and_compute_distances_truncation(self):
        from geoai.landcover_utils import _compute_distances, _compute_sample

        # Fallback truncation when safe_indices is empty and candidate lengths differ
        a1 = np.array([1.0, 2.0, 3.0])
        b1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        sub, ref = _compute_distances(10, a1, b1, np.array([99, 100]))
        self.assertEqual(len(sub), 3)
        self.assertEqual(len(ref), 3)

        # Fresh random indices across sampling rounds (rounds must not all be identical)
        rng = np.random.default_rng(123)
        a_big = np.linspace(1.0, 100.0, 200)
        b_big = np.linspace(2.0, 200.0, 200)
        sub_s, ref_s = _compute_sample(
            100, a_big, b_big, None, num_sampling_rounds=3, rng=rng, subsample_ratio=0.1
        )
        self.assertEqual(len(sub_s), 30)
        r0, r1, r2 = sub_s[:10], sub_s[10:20], sub_s[20:30]
        self.assertFalse(np.array_equal(r0, r1) and np.array_equal(r1, r2))


class TestDeviceModelsVisualizationFixes(unittest.TestCase):
    """Group 6: device.py, models.py, and visualization.py fixes."""

    def test_install_package_safe_argv_and_error_check(self):
        from geoai.utils.device import install_package

        with patch("subprocess.Popen") as mock_popen:
            proc = mock_popen.return_value
            proc.stdout.readline.side_effect = [b""]
            proc.poll.return_value = 0
            proc.wait.return_value = 0

            # Whitespace in argument must stay a single argv element after "--"
            install_package("pkg --index-url http://evil")
            cmd = mock_popen.call_args[0][0]
            self.assertEqual(
                cmd,
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--",
                    "pkg --index-url http://evil",
                ],
            )

        with patch("subprocess.Popen") as mock_popen:
            proc = mock_popen.return_value
            proc.stdout.readline.side_effect = [b""]
            proc.poll.return_value = 0
            proc.wait.return_value = 0

            install_package("https://github.com/opengeos/geoai")
            cmd = mock_popen.call_args[0][0]
            self.assertEqual(cmd[-1], "git+https://github.com/opengeos/geoai")

        with patch("subprocess.Popen") as mock_popen:
            proc = mock_popen.return_value
            proc.stdout.readline.side_effect = [b""]
            proc.poll.return_value = 1
            proc.wait.return_value = 1

            with self.assertRaises(subprocess.CalledProcessError):
                install_package("bad-pkg")

    def test_get_device_honors_geoai_device_env_var(self):
        from geoai.utils.device import get_device

        with (
            patch.dict(os.environ, {"GEOAI_DEVICE": "cpu"}),
            patch("torch.cuda.is_available", return_value=True),
        ):
            self.assertEqual(get_device(), torch.device("cpu"))

        # Invalid device string raises a clear ValueError
        with patch.dict(os.environ, {"GEOAI_DEVICE": "not_a_real_device"}):
            with self.assertRaises(ValueError):
                get_device()

        # Unavailable CUDA accelerator raises ValueError
        with (
            patch.dict(os.environ, {"GEOAI_DEVICE": "cuda"}),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaises(ValueError):
                get_device()

        # Unavailable MPS accelerator raises ValueError
        fake_mps = MagicMock()
        fake_mps.is_available.return_value = False
        with (
            patch.dict(os.environ, {"GEOAI_DEVICE": "mps"}),
            patch.object(torch.backends, "mps", fake_mps, create=True),
        ):
            with self.assertRaises(ValueError):
                get_device()

        # Unavailable XPU accelerator raises ValueError
        fake_xpu = MagicMock()
        fake_xpu.is_available.return_value = False
        with (
            patch.dict(os.environ, {"GEOAI_DEVICE": "xpu"}),
            patch.object(torch, "xpu", fake_xpu, create=True),
        ):
            with self.assertRaises(ValueError):
                get_device()

    def test_models_and_visualization_use_weights_only_true(self):
        import matplotlib

        matplotlib.use("Agg")
        from geoai.utils.models import inspect_pth_file
        from geoai.utils.visualization import plot_performance_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) inspect_pth_file tries weights_only=True first
            pth_path = os.path.join(tmpdir, "weights.pth")
            torch.save({"conv.weight": torch.zeros(2, 2)}, pth_path)

            real_load = torch.load
            seen_models_kwargs = []

            def spy_models_load(*args, **kwargs):
                seen_models_kwargs.append(dict(kwargs))
                return real_load(*args, **kwargs)

            with (
                patch("geoai.utils.models.torch.load", side_effect=spy_models_load),
                patch("geoai.utils.models.try_common_architectures"),
            ):
                inspect_pth_file(pth_path)

            self.assertEqual(len(seen_models_kwargs), 1)
            self.assertTrue(seen_models_kwargs[0].get("weights_only"))

            # 2) inspect_pth_file does NOT auto-retry with weights_only=False without opt-in
            fallback_calls = []

            def failing_first_load(*args, **kwargs):
                fallback_calls.append(dict(kwargs))
                if kwargs.get("weights_only") is True:
                    raise pickle.UnpicklingError("legacy full model")
                return {"conv.weight": torch.zeros(2, 2)}

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("geoai.utils.models.torch.load", side_effect=failing_first_load),
                patch("geoai.utils.models.try_common_architectures"),
            ):
                os.environ.pop("GEOAI_ALLOW_UNSAFE_TORCH_LOAD", None)
                with self.assertRaises(RuntimeError) as ctx:
                    inspect_pth_file(pth_path)

            self.assertEqual(len(fallback_calls), 1)
            self.assertIs(fallback_calls[0].get("weights_only"), True)
            self.assertIn("allow_unsafe=True", str(ctx.exception))
            self.assertIn("GEOAI_ALLOW_UNSAFE_TORCH_LOAD=1", str(ctx.exception))

            # 3) Explicit allow_unsafe=True opts in to fallback with warning
            fallback_calls.clear()
            with (
                patch("geoai.utils.models.torch.load", side_effect=failing_first_load),
                patch("geoai.utils.models.try_common_architectures"),
                self.assertLogs("geoai.utils.models", level="WARNING") as logs,
            ):
                inspect_pth_file(pth_path, allow_unsafe=True)

            self.assertEqual(len(fallback_calls), 2)
            self.assertIs(fallback_calls[0].get("weights_only"), True)
            self.assertIs(fallback_calls[1].get("weights_only"), False)
            self.assertTrue(any("weights_only=False" in m for m in logs.output))

            # 4) Explicit GEOAI_ALLOW_UNSAFE_TORCH_LOAD=1 opts in to fallback
            fallback_calls.clear()
            with (
                patch.dict(os.environ, {"GEOAI_ALLOW_UNSAFE_TORCH_LOAD": "1"}),
                patch("geoai.utils.models.torch.load", side_effect=failing_first_load),
                patch("geoai.utils.models.try_common_architectures"),
                self.assertLogs("geoai.utils.models", level="WARNING"),
            ):
                inspect_pth_file(pth_path)

            self.assertEqual(len(fallback_calls), 2)
            self.assertIs(fallback_calls[0].get("weights_only"), True)
            self.assertIs(fallback_calls[1].get("weights_only"), False)

            # 5) plot_performance_metrics calls torch.load with weights_only=True
            hist_path = os.path.join(tmpdir, "history.pth")
            torch.save(
                {
                    "train_losses": [0.5, 0.3],
                    "val_losses": [0.6, 0.4],
                    "val_ious": [0.5, 0.7],
                },
                hist_path,
            )
            seen_viz_kwargs = []

            def spy_viz_load(*args, **kwargs):
                seen_viz_kwargs.append(dict(kwargs))
                return real_load(*args, **kwargs)

            with (
                patch("geoai.utils.visualization.torch.load", side_effect=spy_viz_load),
                patch("matplotlib.pyplot.show"),
            ):
                df = plot_performance_metrics(hist_path, verbose=False)

            self.assertEqual(len(df), 2)
            self.assertEqual(len(seen_viz_kwargs), 1)
            self.assertTrue(seen_viz_kwargs[0].get("weights_only"))


if __name__ == "__main__":
    unittest.main()
