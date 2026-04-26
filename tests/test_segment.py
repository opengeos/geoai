#!/usr/bin/env python

"""Tests for `geoai.segment` module."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from geoai import segment
from geoai.segment import BoundingBox, DetectionResult

# ---------------------------------------------------------------------------
# Import / exports
# ---------------------------------------------------------------------------


class TestSegmentImport(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(segment, "__name__"))

    def test_all_exports(self):
        expected = ["BoundingBox", "DetectionResult", "GroundedSAM", "CLIPSegmentation"]
        for name in expected:
            self.assertIn(name, segment.__all__)

    def test_classes_exist(self):
        self.assertTrue(inspect.isclass(segment.GroundedSAM))
        self.assertTrue(inspect.isclass(segment.CLIPSegmentation))
        self.assertTrue(inspect.isclass(segment.BoundingBox))
        self.assertTrue(inspect.isclass(segment.DetectionResult))


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox(unittest.TestCase):
    def test_creation(self):
        bbox = BoundingBox(xmin=10, ymin=20, xmax=100, ymax=200)
        self.assertEqual(bbox.xmin, 10)
        self.assertEqual(bbox.ymin, 20)
        self.assertEqual(bbox.xmax, 100)
        self.assertEqual(bbox.ymax, 200)

    def test_xyxy_property(self):
        bbox = BoundingBox(xmin=5, ymin=10, xmax=50, ymax=100)
        self.assertEqual(bbox.xyxy, [5, 10, 50, 100])

    def test_zero_area_box(self):
        bbox = BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0)
        self.assertEqual(bbox.xyxy, [0, 0, 0, 0])

    def test_float_coords(self):
        bbox = BoundingBox(xmin=1.5, ymin=2.5, xmax=10.5, ymax=20.5)
        self.assertEqual(bbox.xyxy, [1.5, 2.5, 10.5, 20.5])


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult(unittest.TestCase):
    def test_creation(self):
        bbox = BoundingBox(xmin=0, ymin=0, xmax=50, ymax=50)
        result = DetectionResult(score=0.95, label="building", box=bbox)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.label, "building")
        self.assertIsNone(result.mask)

    def test_creation_with_mask(self):
        bbox = BoundingBox(xmin=0, ymin=0, xmax=50, ymax=50)
        mask = np.ones((50, 50), dtype=np.uint8)
        result = DetectionResult(score=0.9, label="tree", box=bbox, mask=mask)
        self.assertIsNotNone(result.mask)
        np.testing.assert_array_equal(result.mask, mask)

    def test_from_dict(self):
        d = {
            "score": 0.85,
            "label": "car",
            "box": {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200},
        }
        result = DetectionResult.from_dict(d)
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.label, "car")
        self.assertEqual(result.box.xmin, 10)
        self.assertEqual(result.box.ymax, 200)
        self.assertIsNone(result.mask)

    def test_from_dict_missing_key_raises(self):
        with self.assertRaises(KeyError):
            DetectionResult.from_dict({"score": 0.5, "label": "x"})

    def test_from_dict_missing_box_field_raises(self):
        with self.assertRaises(KeyError):
            DetectionResult.from_dict(
                {"score": 0.5, "label": "x", "box": {"xmin": 0, "ymin": 0}}
            )


# ---------------------------------------------------------------------------
# GroundedSAM signatures
# ---------------------------------------------------------------------------


class TestGroundedSAMSignatures(unittest.TestCase):
    def test_init_params(self):
        sig = inspect.signature(segment.GroundedSAM.__init__)
        expected = [
            "detector_id",
            "segmenter_id",
            "device",
            "tile_size",
            "overlap",
            "threshold",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_segment_image_params(self):
        sig = inspect.signature(segment.GroundedSAM.segment_image)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("text_prompts", sig.parameters)
        self.assertIn("output_path", sig.parameters)

    def test_init_defaults(self):
        sig = inspect.signature(segment.GroundedSAM.__init__)
        self.assertEqual(
            sig.parameters["detector_id"].default,
            "IDEA-Research/grounding-dino-tiny",
        )
        self.assertEqual(sig.parameters["tile_size"].default, 1024)
        self.assertEqual(sig.parameters["overlap"].default, 128)
        self.assertEqual(sig.parameters["threshold"].default, 0.3)


# ---------------------------------------------------------------------------
# CLIPSegmentation signatures
# ---------------------------------------------------------------------------


class TestCLIPSegmentationSignatures(unittest.TestCase):
    def test_init_params(self):
        sig = inspect.signature(segment.CLIPSegmentation.__init__)
        self.assertIn("model_name", sig.parameters)
        self.assertIn("device", sig.parameters)
        self.assertIn("tile_size", sig.parameters)
        self.assertIn("overlap", sig.parameters)

    def test_segment_image_params(self):
        sig = inspect.signature(segment.CLIPSegmentation.segment_image)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("text_prompt", sig.parameters)
        self.assertIn("threshold", sig.parameters)

    def test_segment_image_batch_params(self):
        sig = inspect.signature(segment.CLIPSegmentation.segment_image_batch)
        self.assertIn("input_paths", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("text_prompt", sig.parameters)


# ---------------------------------------------------------------------------
# GroundedSAM internal helpers (mocked)
# ---------------------------------------------------------------------------


class TestGroundedSAMHelpers(unittest.TestCase):
    def _make_mock_sam(self):
        with patch.object(segment.GroundedSAM, "_load_models"):
            sam = segment.GroundedSAM.__new__(segment.GroundedSAM)
            sam.detector_id = "IDEA-Research/grounding-dino-tiny"
            sam.segmenter_id = "facebook/sam-vit-base"
            sam.device = "cpu"
            sam.tile_size = 1024
            sam.overlap = 128
            sam.threshold = 0.3
            sam.object_detector = MagicMock()
            sam.segmentator = MagicMock()
            sam.processor = MagicMock()
        return sam

    def test_get_boxes(self):
        sam = self._make_mock_sam()
        bbox = BoundingBox(xmin=10, ymin=20, xmax=100, ymax=200)
        results = [DetectionResult(score=0.9, label="a", box=bbox)]
        boxes = sam._get_boxes(results)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], [[10, 20, 100, 200]])

    def test_get_boxes_empty(self):
        sam = self._make_mock_sam()
        boxes = sam._get_boxes([])
        # _get_boxes always wraps result in outer list
        self.assertEqual(boxes, [[]])

    def test_mask_to_polygon(self):
        sam = self._make_mock_sam()
        # Create a simple rectangular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        polygon = sam._mask_to_polygon(mask)
        self.assertIsInstance(polygon, list)
        self.assertGreater(len(polygon), 0)

    def test_mask_to_polygon_empty_mask(self):
        sam = self._make_mock_sam()
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygon = sam._mask_to_polygon(mask)
        self.assertIsInstance(polygon, list)

    def test_polygon_to_mask(self):
        sam = self._make_mock_sam()
        # Polygon as list of (x, y) pairs
        polygon = [[10, 10], [10, 50], [50, 50], [50, 10]]
        mask = sam._polygon_to_mask(polygon, (100, 100))
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        # Should have some filled pixels
        self.assertGreater(mask.sum(), 0)

    def test_apply_nms(self):
        sam = self._make_mock_sam()
        # Create overlapping detections
        bbox1 = BoundingBox(xmin=0, ymin=0, xmax=50, ymax=50)
        bbox2 = BoundingBox(xmin=5, ymin=5, xmax=55, ymax=55)
        results = [
            DetectionResult(score=0.9, label="a", box=bbox1),
            DetectionResult(score=0.7, label="a", box=bbox2),
        ]
        filtered = sam._apply_nms(results, iou_threshold=0.5)
        self.assertLessEqual(len(filtered), len(results))

    def test_apply_nms_no_overlap(self):
        sam = self._make_mock_sam()
        bbox1 = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        bbox2 = BoundingBox(xmin=100, ymin=100, xmax=110, ymax=110)
        results = [
            DetectionResult(score=0.9, label="a", box=bbox1),
            DetectionResult(score=0.8, label="b", box=bbox2),
        ]
        filtered = sam._apply_nms(results, iou_threshold=0.5)
        self.assertEqual(len(filtered), 2)

    def test_apply_nms_empty(self):
        sam = self._make_mock_sam()
        filtered = sam._apply_nms([], iou_threshold=0.5)
        self.assertEqual(len(filtered), 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    def test_bounding_box_negative_coords(self):
        """Negative coordinates should be allowed (pixel space may need them)."""
        bbox = BoundingBox(xmin=-10, ymin=-20, xmax=100, ymax=200)
        self.assertEqual(bbox.xyxy, [-10, -20, 100, 200])

    def test_detection_result_zero_score(self):
        bbox = BoundingBox(xmin=0, ymin=0, xmax=50, ymax=50)
        result = DetectionResult(score=0.0, label="unknown", box=bbox)
        self.assertEqual(result.score, 0.0)

    def test_detection_result_empty_label(self):
        bbox = BoundingBox(xmin=0, ymin=0, xmax=50, ymax=50)
        result = DetectionResult(score=0.5, label="", box=bbox)
        self.assertEqual(result.label, "")


if __name__ == "__main__":
    unittest.main()
