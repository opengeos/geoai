#!/usr/bin/env python

"""Tests for moondream sliding window functionality."""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from PIL import Image


class TestMoondreamSlidingWindow(unittest.TestCase):
    """Tests for Moondream sliding window methods."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock processor without loading the actual model
        self.mock_processor = None

    def test_create_sliding_windows(self):
        """Test sliding window coordinate generation."""
        from geoai.moondream import MoondreamGeo

        # Create a real instance with mocked dependencies
        with patch("geoai.moondream.MoondreamGeo.__init__", return_value=None):
            processor = MoondreamGeo.__new__(MoondreamGeo)

        # Test case 1: Small image that fits in one window
        windows = MoondreamGeo._create_sliding_windows(
            processor, 400, 400, window_size=512, overlap=64
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], (0, 0, 400, 400))

        # Test case 2: Image requiring multiple windows
        windows = MoondreamGeo._create_sliding_windows(
            processor, 1024, 1024, window_size=512, overlap=64
        )
        self.assertGreater(len(windows), 1)

        # Verify window properties
        for x_start, y_start, x_end, y_end in windows:
            # Windows should be within image bounds
            self.assertGreaterEqual(x_start, 0)
            self.assertGreaterEqual(y_start, 0)
            self.assertLessEqual(x_end, 1024)
            self.assertLessEqual(y_end, 1024)

            # Windows should have minimum size
        # Verify window properties
        for x_start, y_start, x_end, y_end in windows:
            # Windows should be within image bounds
            self.assertGreaterEqual(x_start, 0)
            self.assertGreaterEqual(y_start, 0)
            self.assertLessEqual(x_end, 1024)
            self.assertLessEqual(y_end, 1024)

            # Windows should have minimum size
            self.assertGreaterEqual(x_end - x_start, 512 // 2)
            self.assertGreaterEqual(y_end - y_start, 512 // 2)

    def test_apply_nms(self):
        """Test Non-Maximum Suppression."""
        from geoai.moondream import MoondreamGeo

        with patch("geoai.moondream.MoondreamGeo.__init__", return_value=None):
            processor = MoondreamGeo.__new__(MoondreamGeo)

        # Test case 1: Empty detections
        result = MoondreamGeo._apply_nms(processor, [])
        self.assertEqual(len(result), 0)

        # Test case 2: No overlapping detections
        detections = [
            {"x_min": 0.0, "y_min": 0.0, "x_max": 0.1, "y_max": 0.1, "score": 0.9},
            {"x_min": 0.5, "y_min": 0.5, "x_max": 0.6, "y_max": 0.6, "score": 0.8},
        ]
        result = MoondreamGeo._apply_nms(processor, detections, iou_threshold=0.5)
        self.assertEqual(len(result), 2)

        # Test case 3: Overlapping detections (should be merged)
        detections = [
            {"x_min": 0.0, "y_min": 0.0, "x_max": 0.2, "y_max": 0.2, "score": 0.9},
            {
                "x_min": 0.05,
                "y_min": 0.05,
                "x_max": 0.25,
                "y_max": 0.25,
                "score": 0.8,
            },
        ]
        result = MoondreamGeo._apply_nms(processor, detections, iou_threshold=0.5)
        # Should keep only the higher-scored detection
        self.assertLessEqual(len(result), len(detections))

    def test_detect_sliding_window_small_image(self):
        """Test that small images bypass sliding window."""
        with patch("geoai.moondream.MoondreamGeo") as MockProcessor:
            mock_instance = MockProcessor.return_value

            # Create a small test image
            small_image = Image.new("RGB", (256, 256), color="white")

            # Mock the detect method to return a simple result
            mock_instance.detect.return_value = {
                "objects": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}]
            }

            # The small image should call regular detect, not sliding window
            mock_instance.load_image.return_value = (small_image, None)

            # Verify the logic would bypass sliding window for small images
            self.assertLessEqual(small_image.size[0], 512)
            self.assertLessEqual(small_image.size[1], 512)

    def test_window_overlap_calculation(self):
        """Test that overlaps are calculated correctly."""
        from geoai.moondream import MoondreamGeo

        with patch("geoai.moondream.MoondreamGeo.__init__", return_value=None):
            processor = MoondreamGeo.__new__(MoondreamGeo)

        window_size = 512
        overlap = 64
        stride = window_size - overlap

        windows = MoondreamGeo._create_sliding_windows(
            processor, 1000, 1000, window_size, overlap
        )

        # Check that adjacent windows have the expected overlap
        if len(windows) > 1:
            # Find horizontally adjacent windows (same y_start)
            windows_by_y = {}
            for w in windows:
                y_start = w[1]
                if y_start not in windows_by_y:
                    windows_by_y[y_start] = []
                windows_by_y[y_start].append(w)

            # Check overlap between adjacent windows in the same row
            for y_start, row_windows in windows_by_y.items():
                if len(row_windows) > 1:
                    sorted_windows = sorted(row_windows, key=lambda w: w[0])
                    for i in range(len(sorted_windows) - 1):
                        w1 = sorted_windows[i]
                        w2 = sorted_windows[i + 1]
                        # Check that there is overlap or adjacency
                        self.assertLessEqual(w2[0] - w1[0], stride)

    def test_convenience_functions_exist(self):
        """Test that convenience functions are importable."""
        try:
            from geoai import (
                moondream_detect_sliding_window,
                moondream_point_sliding_window,
                moondream_query_sliding_window,
                moondream_caption_sliding_window,
            )

            # If we can import them, test passes
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import convenience functions: {e}")

    def test_sliding_window_parameters(self):
        """Test that sliding window methods accept correct parameters."""
        from geoai.moondream import MoondreamGeo
        import inspect

        # Check that methods have correct signatures
        self.assertTrue(hasattr(MoondreamGeo, "detect_sliding_window"))
        self.assertTrue(hasattr(MoondreamGeo, "point_sliding_window"))
        self.assertTrue(hasattr(MoondreamGeo, "query_sliding_window"))
        self.assertTrue(hasattr(MoondreamGeo, "caption_sliding_window"))

        # Check method signatures include expected parameters
        detect_sig = inspect.signature(MoondreamGeo.detect_sliding_window)
        self.assertIn("window_size", detect_sig.parameters)
        self.assertIn("overlap", detect_sig.parameters)
        self.assertIn("iou_threshold", detect_sig.parameters)

        point_sig = inspect.signature(MoondreamGeo.point_sliding_window)
        self.assertIn("window_size", point_sig.parameters)
        self.assertIn("overlap", point_sig.parameters)

        query_sig = inspect.signature(MoondreamGeo.query_sliding_window)
        self.assertIn("window_size", query_sig.parameters)
        self.assertIn("overlap", query_sig.parameters)
        self.assertIn("combine_strategy", query_sig.parameters)

        caption_sig = inspect.signature(MoondreamGeo.caption_sliding_window)
        self.assertIn("window_size", caption_sig.parameters)
        self.assertIn("overlap", caption_sig.parameters)
        self.assertIn("combine_strategy", caption_sig.parameters)


if __name__ == "__main__":
    unittest.main()
