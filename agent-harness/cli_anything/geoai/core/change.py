"""Change detection operations.

Wraps geoai change detection functions: ChangeStar and AnyChange (ChangeDetection)
for CLI consumption.
"""

import os
from typing import Any, Dict, Union

CHANGE_METHODS = ["changestar", "anychange"]


def _resolve_change_threshold(confidence_threshold: Union[int, float]) -> float:
    """Convert a 0-255 integer or 0-1 float confidence threshold to [0, 1].

    Integers are always interpreted on the documented 0-255 scale (so integer 1
    maps to ``1 / 255``), while floats in ``[0.0, 1.0]`` retain their
    probability meaning (so float ``1.0`` maps to ``1.0``).
    """
    if isinstance(confidence_threshold, bool):
        raise ValueError(
            f"confidence_threshold must be an int (0-255) or float (0-1), got bool: {confidence_threshold}"
        )
    if isinstance(confidence_threshold, int):
        if not 0 <= confidence_threshold <= 255:
            raise ValueError(
                f"Integer confidence_threshold must be between 0 and 255, got {confidence_threshold}"
            )
        return float(confidence_threshold) / 255.0

    val = float(confidence_threshold)
    if 0.0 <= val <= 1.0:
        return val
    if 1.0 < val <= 255.0:
        return val / 255.0
    raise ValueError(
        f"confidence_threshold must be in [0.0, 1.0] or [0, 255], got {confidence_threshold}"
    )


def _filter_min_area_raster(output: str, min_area: int) -> None:
    """Zero out connected nonzero regions smaller than ``min_area`` pixels."""
    if min_area <= 0 or not os.path.isfile(output):
        return

    import numpy as np
    import rasterio

    with rasterio.open(output) as src:
        profile = src.profile.copy()
        data = src.read()

    mask = data[0] != 0
    if not np.any(mask):
        return

    try:
        from scipy.ndimage import label as nd_label

        labeled, num_features = nd_label(mask)
        if num_features == 0:
            return
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_area
        keep[0] = False
        filtered_mask = keep[labeled]
    except ImportError:
        # Pure-numpy/Python 4-connected component fallback when scipy is absent.
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        filtered_mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    component = []
                    while stack:
                        cr, cc = stack.pop()
                        component.append((cr, cc))
                        for nr, nc in (
                            (cr - 1, cc),
                            (cr + 1, cc),
                            (cr, cc - 1),
                            (cr, cc + 1),
                        ):
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and mask[nr, nc]
                                and not visited[nr, nc]
                            ):
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    if len(component) >= min_area:
                        for cr, cc in component:
                            filtered_mask[cr, cc] = True

    data[0] = np.where(filtered_mask, data[0], 0).astype(data.dtype)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data)


def detect_changes(
    image1: str,
    image2: str,
    output: str,
    method: str = "changestar",
    confidence_threshold: Union[int, float] = 155,
    min_area: int = 0,
) -> Dict[str, Any]:
    """Run change detection between two temporal images.

    Args:
        image1: Path to the earlier image.
        image2: Path to the later image.
        output: Output change map file path.
        method: Detection method (changestar, anychange).
        confidence_threshold: Confidence threshold for change pixels (0-255 int, or 0-1 float).
        min_area: Minimum area threshold for change regions in pixels.

    Returns:
        Result dict with output path and change statistics.

    Raises:
        FileNotFoundError: If input images do not exist.
        ValueError: If method, threshold, or min_area is invalid.
    """
    image1 = os.path.abspath(image1)
    image2 = os.path.abspath(image2)
    output = os.path.abspath(output)

    if not os.path.isfile(image1):
        raise FileNotFoundError(f"Image 1 not found: {image1}")
    if not os.path.isfile(image2):
        raise FileNotFoundError(f"Image 2 not found: {image2}")

    if method not in CHANGE_METHODS:
        raise ValueError(
            f"Unknown method: {method}. Choose from: {', '.join(CHANGE_METHODS)}"
        )

    if min_area < 0:
        raise ValueError(f"min_area must be non-negative, got {min_area}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    if method == "changestar":
        return _run_changestar(image1, image2, output, confidence_threshold, min_area)
    return _run_anychange(image1, image2, output, min_area)


def _run_changestar(
    image1: str,
    image2: str,
    output: str,
    confidence_threshold: Union[int, float],
    min_area: int,
) -> Dict[str, Any]:
    """Run ChangeStar change detection.

    Args:
        image1: Path to the earlier image.
        image2: Path to the later image.
        output: Output change map path.
        confidence_threshold: Confidence threshold (0-255 integer or 0-1 float).
        min_area: Minimum area threshold in pixels.

    Returns:
        Result dict.
    """
    from geoai.change_detection import changestar_detect

    threshold = _resolve_change_threshold(confidence_threshold)

    changestar_detect(
        image1_path=image1,
        image2_path=image2,
        output_change=output,
        threshold=threshold,
    )

    if min_area > 0 and os.path.isfile(output):
        _filter_min_area_raster(output, min_area)

    result: Dict[str, Any] = {
        "image1": image1,
        "image2": image2,
        "output": output,
        "method": "changestar",
        "threshold": round(threshold, 4),
        "confidence_threshold": confidence_threshold,
        "min_area": min_area,
    }

    if os.path.isfile(output):
        result["output_size_bytes"] = os.path.getsize(output)
        result = _add_change_stats(result, output)

    return result


def _run_anychange(
    image1: str,
    image2: str,
    output: str,
    min_area: int,
) -> Dict[str, Any]:
    """Run AnyChange (ChangeDetection) change detection.

    Args:
        image1: Path to the earlier image.
        image2: Path to the later image.
        output: Output change map path.
        min_area: Minimum area threshold in pixels.

    Returns:
        Result dict.
    """
    from geoai.change_detection import ChangeDetection

    detector = ChangeDetection()
    detector.detect_changes(
        image1_path=image1,
        image2_path=image2,
        output_path=output,
    )

    if min_area > 0 and os.path.isfile(output):
        _filter_min_area_raster(output, min_area)

    result: Dict[str, Any] = {
        "image1": image1,
        "image2": image2,
        "output": output,
        "method": "anychange",
        "min_area": min_area,
    }

    if os.path.isfile(output):
        result["output_size_bytes"] = os.path.getsize(output)
        result = _add_change_stats(result, output)

    return result


def _add_change_stats(result: Dict[str, Any], output: str) -> Dict[str, Any]:
    """Add change statistics to a result dict by reading the output raster.

    Args:
        result: Existing result dict.
        output: Path to the change map raster.

    Returns:
        Updated result dict with change pixel statistics.
    """
    try:
        import numpy as np
        import rasterio

        with rasterio.open(output) as src:
            data = src.read(1)
            total = int(data.size)
            changed = int(np.count_nonzero(data))
            result["total_pixels"] = total
            result["changed_pixels"] = changed
            result["change_percentage"] = (
                round(changed / total * 100, 2) if total > 0 else 0.0
            )
    except Exception:
        pass

    return result


def list_methods() -> list:
    """List available change detection methods.

    Returns:
        List of method info dicts.
    """
    descriptions = {
        "changestar": "ChangeStar bi-temporal change detection (torchange backend)",
        "anychange": "AnyChange generic multi-temporal change detector",
    }
    return [{"name": m, "description": descriptions.get(m, "")} for m in CHANGE_METHODS]
