"""Tests for stitching super-resolved patches in geoai.tools.sr."""

import numpy as np
import pytest
import torch

pytest.importorskip("omegaconf")  # geoai.tools.sr imports it at module level

from geoai.tools.sr import (
    _create_blend_weights,
    _process_patched,
    _process_patched_uncertainty,
)


class NearestUpsampler:
    """Stands in for the OpenSR model: nearest-neighbor upsampling.

    Every patch gets the exact upsampled values, so the stitched image must
    equal the upsampled input everywhere, whatever the blend weights are.
    """

    def __init__(self, scale):
        self.scale = scale

    def forward(self, x, sampling_steps=100):
        return x.repeat_interleave(self.scale, -2).repeat_interleave(self.scale, -1)

    def uncertainty_map(self, x, n_variations=25, sampling_steps=100):
        h, w = x.shape[-2] * self.scale, x.shape[-1] * self.scale
        return torch.full((1, 1, h, w), 0.5)


# (height, width, patch_size, overlap, scale): the patch grid fits exactly, the
# last patch is shifted back, one side is shorter than a patch, no overlap, a
# one-pixel overlap, and the super_resolution() defaults.
CASES = [
    (56, 56, 32, 8, 2),
    (70, 45, 32, 8, 2),
    (20, 90, 32, 8, 3),
    (64, 96, 32, 0, 2),
    (40, 50, 12, 1, 1),
    (200, 300, 128, 16, 4),
]


@pytest.mark.parametrize("h, w, patch_size, overlap, scale", CASES)
def test_patched_sr_matches_model_output_everywhere(h, w, patch_size, overlap, scale):
    lr = torch.rand(1, 4, h, w, generator=torch.Generator().manual_seed(0)) + 0.1
    model = NearestUpsampler(scale)
    out = _process_patched(model, lr, patch_size, overlap, scale, sampling_steps=1)
    assert out.shape == (4, h * scale, w * scale)
    np.testing.assert_allclose(out, model.forward(lr)[0].numpy(), rtol=1e-5)


@pytest.mark.parametrize("h, w, patch_size, overlap, scale", CASES)
def test_patched_uncertainty_matches_model_output_everywhere(
    h, w, patch_size, overlap, scale
):
    lr = torch.rand(1, 4, h, w, generator=torch.Generator().manual_seed(0))
    model = NearestUpsampler(scale)
    unc = _process_patched_uncertainty(
        model, lr, patch_size, overlap, scale, n_variations=4, sampling_steps=1
    )
    assert unc.shape == (h * scale, w * scale)
    np.testing.assert_allclose(unc, 0.5, rtol=1e-5)


def test_blend_weights_taper_only_the_requested_ends():
    ramp = np.linspace(0, 1, 4, dtype=np.float32)
    both = _create_blend_weights(16, 2, 2)
    np.testing.assert_array_equal(both[:4], ramp)
    np.testing.assert_array_equal(both[-4:], ramp[::-1])
    np.testing.assert_array_equal(both[4:-4], 1)

    start_only = _create_blend_weights(16, 2, 2, taper_end=False)
    np.testing.assert_array_equal(start_only[:4], ramp)
    np.testing.assert_array_equal(start_only[4:], 1)

    end_only = _create_blend_weights(16, 2, 2, taper_start=False)
    np.testing.assert_array_equal(end_only[:-4], 1)
    np.testing.assert_array_equal(end_only[-4:], ramp[::-1])

    neither = _create_blend_weights(16, 2, 2, taper_start=False, taper_end=False)
    np.testing.assert_array_equal(neither, 1)

    # A one-pixel overlap has no room for a ramp and is not tapered.
    np.testing.assert_array_equal(_create_blend_weights(5, 1, 1), 1)
