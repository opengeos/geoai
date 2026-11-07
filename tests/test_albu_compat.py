"""Albumentations 2.x must receive the arguments it actually understands."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_albu_compat", Path(__file__).parent.parent / "geoai" / "_albu_compat.py"
)
compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat)


def test_gauss_noise_args_not_ignored():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = compat.make_gauss_noise(var_limit=(10.0, 50.0), p=1.0)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = t(image=img)["image"]
    std = float(out.astype(np.float32).std())
    # variance 10..50 -> std ~3.2..7.1 (not the 2.x default of ~0.2*255)
    assert 1.0 < std < 12.0


def test_coarse_dropout_args_not_ignored():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = compat.make_coarse_dropout(max_holes=8, max_height=32, max_width=32, p=1.0)
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    out = t(image=img)["image"]
    # 1.x behaviour: exactly 8 holes of 32x32 (<=12.5%, fewer if they overlap)
    frac = (out == 0).mean()
    assert 0.09 <= frac <= 8 * 32 * 32 / (256 * 256) + 1e-6
