"""Change detection operations.

Wraps geoai change detection functions: ChangeStar, AnyChange
for CLI consumption.
"""

import os
from typing import Any, Dict, Optional


CHANGE_METHODS = ["changestar", "anychange"]


def detect_changes(
    image1: str,
    image2: str,
    output: str,
    method: str = "changestar",
    confidence_threshold: int = 155,
    min_area: int = 0,
) -> Dict[str, Any]:
    """Run change detection between two temporal images.

    Args:
        image1: Path to the earlier image.
        image2: Path to the later image.
        output: Output change map file path.
        method: Detection method (changestar, anychange).
        confidence_threshold: Confidence threshold for change pixels (0-255).
        min_area: Minimum area threshold for change regions in pixels.

    Returns:
        Result dict with output path and change statistics.

    Raises:
        FileNotFoundError: If input images do not exist.
        ValueError: If method is not recognized.
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

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    if method == "changestar":
        return _run_changestar(
            image1, image2, output, confidence_threshold, min_area
        )
    else:
        return _run_anychange(image1, image2, output, min_area)


def _run_changestar(
    image1: str,
    image2: str,
    output: str,
    confidence_threshold: int,
    min_area: int,
) -> Dict[str, Any]:
    """Run ChangeStar change detection.

    Args:
        image1: Path to the earlier image.
        image2: Path to the later image.
        output: Output change map path.
        confidence_threshold: Confidence threshold.
        min_area: Minimum area threshold.

    Returns:
        Result dict.
    """
    from geoai import changestar_detect

    changestar_detect(
        image1=image1,
        image2=image2,
        output=output,
        change_confidence_threshold=confidence_threshold,
        min_area=min_area,
    )

    result = {
        "image1": image1,
        "image2": image2,
        "output": output,
        "method": "changestar",
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
    """Run AnyChange change detection.

    Args:
        image1: Path to the earlier image.
        image2: Path to the later image.
        output: Output change map path.
        min_area: Minimum area threshold.

    Returns:
        Result dict.
    """
    from geoai.change_detection import AnyChangeDetection

    detector = AnyChangeDetection()
    detector.detect(image1=image1, image2=image2, output=output, min_area=min_area)

    result = {
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
            result["change_percentage"] = round(changed / total * 100, 2) if total > 0 else 0.0
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
    return [
        {"name": m, "description": descriptions.get(m, "")}
        for m in CHANGE_METHODS
    ]
