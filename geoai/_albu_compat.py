"""Helpers to build Albumentations transforms across 1.x and 2.x APIs.

Albumentations 2.x renamed several arguments (``GaussNoise(var_limit=...)`` ->
``std_range`` and ``CoarseDropout(max_holes=...)`` -> ``num_holes_range``) and
silently ignores the old names, so they must be passed by the right name.
"""

import inspect
import math

import albumentations as A


def make_gauss_noise(
    var_limit: tuple[float, float] = (10.0, 50.0), p: float = 0.5
) -> "A.GaussNoise":
    """Create ``A.GaussNoise`` with a variance range given in 8-bit pixel units."""
    if "std_range" in inspect.signature(A.GaussNoise.__init__).parameters:
        # 2.x: std_range is a fraction of the max pixel value.
        std_range = (math.sqrt(var_limit[0]) / 255.0, math.sqrt(var_limit[1]) / 255.0)
        return A.GaussNoise(std_range=std_range, p=p)
    return A.GaussNoise(var_limit=var_limit, p=p)


def make_coarse_dropout(
    max_holes: int = 8, max_height: int = 32, max_width: int = 32, p: float = 0.5
) -> "A.CoarseDropout":
    """Create ``A.CoarseDropout`` matching the 1.x fixed-size behaviour."""
    if "num_holes_range" in inspect.signature(A.CoarseDropout.__init__).parameters:
        return A.CoarseDropout(
            num_holes_range=(max_holes, max_holes),
            hole_height_range=(max_height, max_height),
            hole_width_range=(max_width, max_width),
            p=p,
        )
    return A.CoarseDropout(
        max_holes=max_holes, max_height=max_height, max_width=max_width, p=p
    )
