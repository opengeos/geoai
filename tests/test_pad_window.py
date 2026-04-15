"""Tests for _pad_window helper and edge-tile padding in inference functions."""

import numpy as np
import pytest

from geoai.train import _pad_window


class TestPadWindow:
    """Tests for the _pad_window helper function."""

    def test_no_padding_when_full_size(self):
        window = np.ones((3, 256, 256), dtype=np.uint8)
        result, h, w = _pad_window(window, 256)
        assert result.shape == (3, 256, 256)
        assert h == 256
        assert w == 256
        np.testing.assert_array_equal(result, window)

    def test_pads_smaller_height(self):
        window = np.ones((3, 100, 256), dtype=np.uint8)
        result, h, w = _pad_window(window, 256)
        assert result.shape == (3, 256, 256)
        assert h == 100
        assert w == 256
        # Original data preserved in top-left
        np.testing.assert_array_equal(result[:, :100, :256], 1)
        # Padded region is zero
        np.testing.assert_array_equal(result[:, 100:, :], 0)

    def test_pads_smaller_width(self):
        window = np.ones((3, 256, 100), dtype=np.uint8)
        result, h, w = _pad_window(window, 256)
        assert result.shape == (3, 256, 256)
        assert h == 256
        assert w == 100
        np.testing.assert_array_equal(result[:, :256, :100], 1)
        np.testing.assert_array_equal(result[:, :, 100:], 0)

    def test_pads_both_dimensions(self):
        window = np.ones((3, 80, 120), dtype=np.float32)
        result, h, w = _pad_window(window, 256)
        assert result.shape == (3, 256, 256)
        assert h == 80
        assert w == 120
        np.testing.assert_array_equal(result[:, :80, :120], 1)
        np.testing.assert_array_equal(result[:, 80:, :], 0)
        np.testing.assert_array_equal(result[:, :, 120:], 0)

    def test_preserves_dtype(self):
        for dtype in [np.uint8, np.float32, np.float64]:
            window = np.ones((3, 50, 50), dtype=dtype)
            result, _, _ = _pad_window(window, 256)
            assert result.dtype == dtype

    def test_single_channel(self):
        window = np.ones((1, 64, 128), dtype=np.float32)
        result, h, w = _pad_window(window, 256)
        assert result.shape == (1, 256, 256)
        assert h == 64
        assert w == 128

    def test_non_square_input_one_dim_smaller(self):
        window = np.ones((3, 256, 100), dtype=np.uint8)
        result, h, w = _pad_window(window, 256)
        # Height == window_size but width < window_size
        assert result.shape == (3, 256, 256)
        assert h == 256
        assert w == 100

    def test_returns_same_array_when_no_padding_needed(self):
        window = np.ones((3, 256, 256), dtype=np.uint8)
        result, h, w = _pad_window(window, 256)
        # Should return the original array (not a copy)
        assert result is window
