#!/usr/bin/env python

"""Tests for dataclasses in `geoai.segment` module."""

import unittest

import numpy as np

from geoai.segment import BoundingBox, DetectionResult


class TestBoundingBox(unittest.TestCase):
    """Tests for the BoundingBox dataclass."""

    def test_construction(self):
        """Test basic construction with integer coordinates."""
        bbox = BoundingBox(xmin=10, ymin=20, xmax=100, ymax=200)
        self.assertEqual(bbox.xmin, 10)
        self.assertEqual(bbox.ymin, 20)
        self.assertEqual(bbox.xmax, 100)
        self.assertEqual(bbox.ymax, 200)

    def test_xyxy_property(self):
        """Test that xyxy property returns list in correct order."""
        bbox = BoundingBox(xmin=5, ymin=10, xmax=50, ymax=100)
        self.assertEqual(bbox.xyxy, [5, 10, 50, 100])

    def test_xyxy_returns_list(self):
        """Test that xyxy returns a list type."""
        bbox = BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
        self.assertIsInstance(bbox.xyxy, list)
        self.assertEqual(len(bbox.xyxy), 4)

    def test_zero_size_box(self):
        """Test bounding box with zero dimensions."""
        bbox = BoundingBox(xmin=50, ymin=50, xmax=50, ymax=50)
        self.assertEqual(bbox.xyxy, [50, 50, 50, 50])

    def test_float_coordinates(self):
        """Test bounding box with float-like coordinates."""
        bbox = BoundingBox(xmin=0, ymin=0, xmax=512, ymax=512)
        self.assertEqual(bbox.xmax - bbox.xmin, 512)


class TestDetectionResult(unittest.TestCase):
    """Tests for the DetectionResult dataclass."""

    def test_construction_without_mask(self):
        """Test construction with score, label, and box but no mask."""
        bbox = BoundingBox(xmin=10, ymin=20, xmax=100, ymax=200)
        result = DetectionResult(score=0.95, label="building", box=bbox)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.label, "building")
        self.assertIsNone(result.mask)

    def test_construction_with_mask(self):
        """Test construction including a numpy mask array."""
        bbox = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        mask = np.ones((10, 10), dtype=np.uint8)
        result = DetectionResult(score=0.8, label="tree", box=bbox, mask=mask)
        self.assertIsNotNone(result.mask)
        self.assertEqual(result.mask.shape, (10, 10))

    def test_from_dict_classmethod(self):
        """Test creating DetectionResult from a dictionary."""
        detection_dict = {
            "score": 0.92,
            "label": "car",
            "box": {"xmin": 5, "ymin": 10, "xmax": 50, "ymax": 60},
        }
        result = DetectionResult.from_dict(detection_dict)
        self.assertEqual(result.score, 0.92)
        self.assertEqual(result.label, "car")
        self.assertIsInstance(result.box, BoundingBox)
        self.assertEqual(result.box.xmin, 5)
        self.assertEqual(result.box.ymax, 60)

    def test_from_dict_preserves_box_coordinates(self):
        """Test that from_dict correctly maps all box coordinates."""
        detection_dict = {
            "score": 0.75,
            "label": "pool",
            "box": {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400},
        }
        result = DetectionResult.from_dict(detection_dict)
        self.assertEqual(result.box.xyxy, [100, 200, 300, 400])

    def test_from_dict_no_mask(self):
        """Test that from_dict produces result with None mask."""
        detection_dict = {
            "score": 0.5,
            "label": "road",
            "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        }
        result = DetectionResult.from_dict(detection_dict)
        self.assertIsNone(result.mask)

    def test_score_range(self):
        """Test DetectionResult with various score values."""
        bbox = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        for score in [0.0, 0.5, 1.0]:
            result = DetectionResult(score=score, label="test", box=bbox)
            self.assertEqual(result.score, score)

    def test_label_string(self):
        """Test DetectionResult with various label strings."""
        bbox = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        for label in ["building", "solar_panel", "parking lot", ""]:
            result = DetectionResult(score=0.9, label=label, box=bbox)
            self.assertEqual(result.label, label)


if __name__ == "__main__":
    unittest.main()
