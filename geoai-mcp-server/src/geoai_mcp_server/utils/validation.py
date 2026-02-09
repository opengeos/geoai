"""Input validation utilities for GeoAI MCP Server.

Provides validation functions for various input types.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Any
import logging

from .error_handling import InputValidationError

logger = logging.getLogger("geoai_mcp.validation")


def validate_bbox(
    bbox: List[float] | Tuple[float, ...],
    crs: str = "EPSG:4326",
) -> Tuple[float, float, float, float]:
    """Validate a bounding box.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        crs: Coordinate reference system (for validation limits)

    Returns:
        Validated bounding box tuple

    Raises:
        InputValidationError: If bounding box is invalid
    """
    if not isinstance(bbox, (list, tuple)):
        raise InputValidationError(
            message="Bounding box must be a list or tuple",
            parameter_name="bbox",
            received_value=type(bbox).__name__,
            expected="[min_lon, min_lat, max_lon, max_lat]",
        )

    if len(bbox) != 4:
        raise InputValidationError(
            message=f"Bounding box must have exactly 4 values, got {len(bbox)}",
            parameter_name="bbox",
            received_value=str(bbox),
            expected="[min_lon, min_lat, max_lon, max_lat]",
        )

    try:
        min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    except (TypeError, ValueError) as e:
        raise InputValidationError(
            message=f"Bounding box values must be numbers: {e}",
            parameter_name="bbox",
            received_value=str(bbox),
            expected="Numeric values",
        )

    # Validate coordinate ranges for WGS84
    if crs.upper() == "EPSG:4326":
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise InputValidationError(
                message="Longitude must be between -180 and 180",
                parameter_name="bbox",
                received_value=f"lon: [{min_lon}, {max_lon}]",
                expected="Longitude in range [-180, 180]",
            )
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise InputValidationError(
                message="Latitude must be between -90 and 90",
                parameter_name="bbox",
                received_value=f"lat: [{min_lat}, {max_lat}]",
                expected="Latitude in range [-90, 90]",
            )

    # Validate min < max
    if min_lon >= max_lon:
        raise InputValidationError(
            message="min_lon must be less than max_lon",
            parameter_name="bbox",
            received_value=f"min_lon={min_lon}, max_lon={max_lon}",
            expected="min_lon < max_lon",
        )
    if min_lat >= max_lat:
        raise InputValidationError(
            message="min_lat must be less than max_lat",
            parameter_name="bbox",
            received_value=f"min_lat={min_lat}, max_lat={max_lat}",
            expected="min_lat < max_lat",
        )

    return (min_lon, min_lat, max_lon, max_lat)


def validate_image_path(
    path: str | Path,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """Validate an image file path.

    Args:
        path: Path to the image file
        allowed_extensions: List of allowed extensions (e.g., ['.tif', '.png'])

    Returns:
        Validated Path object

    Raises:
        InputValidationError: If path is invalid
    """
    if allowed_extensions is None:
        allowed_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2"]

    path = Path(path)

    # Check extension
    if allowed_extensions:
        ext = path.suffix.lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise InputValidationError(
                message=f"Unsupported image format: {ext}",
                parameter_name="image_path",
                received_value=str(path),
                expected=f"One of: {', '.join(allowed_extensions)}",
            )

    return path


def validate_text_prompts(
    prompts: str | List[str],
    max_length: int = 500,
    max_prompts: int = 20,
) -> List[str]:
    """Validate text prompts for segmentation/detection.

    Args:
        prompts: Single prompt or list of prompts
        max_length: Maximum length per prompt
        max_prompts: Maximum number of prompts

    Returns:
        List of validated prompts

    Raises:
        InputValidationError: If prompts are invalid
    """
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]

    if not isinstance(prompts, list):
        raise InputValidationError(
            message="Prompts must be a string or list of strings",
            parameter_name="prompts",
            received_value=type(prompts).__name__,
            expected="String or list of strings",
        )

    if len(prompts) == 0:
        raise InputValidationError(
            message="At least one prompt is required",
            parameter_name="prompts",
            received_value="empty list",
            expected="At least one text prompt",
        )

    if len(prompts) > max_prompts:
        raise InputValidationError(
            message=f"Too many prompts (max {max_prompts})",
            parameter_name="prompts",
            received_value=f"{len(prompts)} prompts",
            expected=f"At most {max_prompts} prompts",
        )

    validated = []
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise InputValidationError(
                message=f"Prompt {i+1} must be a string",
                parameter_name=f"prompts[{i}]",
                received_value=type(prompt).__name__,
                expected="String",
            )

        # Strip and validate
        prompt = prompt.strip()
        if not prompt:
            raise InputValidationError(
                message=f"Prompt {i+1} is empty",
                parameter_name=f"prompts[{i}]",
                received_value="empty string",
                expected="Non-empty text",
            )

        if len(prompt) > max_length:
            raise InputValidationError(
                message=f"Prompt {i+1} is too long ({len(prompt)} chars, max {max_length})",
                parameter_name=f"prompts[{i}]",
                received_value=f"{len(prompt)} characters",
                expected=f"At most {max_length} characters",
            )

        # Basic sanitization (remove control characters)
        prompt = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", prompt)

        validated.append(prompt)

    return validated


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename to be safe for the filesystem.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove/replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Truncate if too long (preserving extension)
    if len(filename) > max_length:
        base, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        max_base = max_length - len(ext) - 1
        filename = base[:max_base] + ("." + ext if ext else "")

    # Fallback for empty result
    if not filename:
        filename = "unnamed_file"

    return filename


def validate_confidence_threshold(threshold: float) -> float:
    """Validate a confidence threshold value.

    Args:
        threshold: Threshold value to validate

    Returns:
        Validated threshold

    Raises:
        InputValidationError: If threshold is invalid
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise InputValidationError(
            message="Confidence threshold must be a number",
            parameter_name="threshold",
            received_value=str(threshold),
            expected="A number between 0 and 1",
        )

    if not 0 <= threshold <= 1:
        raise InputValidationError(
            message=f"Confidence threshold must be between 0 and 1, got {threshold}",
            parameter_name="threshold",
            received_value=str(threshold),
            expected="A value between 0 and 1",
        )

    return threshold


def validate_tile_size(tile_size: int) -> int:
    """Validate tile size for processing.

    Args:
        tile_size: Tile size in pixels

    Returns:
        Validated tile size

    Raises:
        InputValidationError: If tile size is invalid
    """
    try:
        tile_size = int(tile_size)
    except (TypeError, ValueError):
        raise InputValidationError(
            message="Tile size must be an integer",
            parameter_name="tile_size",
            received_value=str(tile_size),
            expected="An integer (e.g., 256, 512, 1024)",
        )

    if tile_size < 64:
        raise InputValidationError(
            message="Tile size too small (minimum 64)",
            parameter_name="tile_size",
            received_value=str(tile_size),
            expected="At least 64 pixels",
        )

    if tile_size > 4096:
        raise InputValidationError(
            message="Tile size too large (maximum 4096)",
            parameter_name="tile_size",
            received_value=str(tile_size),
            expected="At most 4096 pixels",
        )

    return tile_size


def validate_num_classes(num_classes: int) -> int:
    """Validate number of classes for classification/segmentation.

    Args:
        num_classes: Number of classes

    Returns:
        Validated number

    Raises:
        InputValidationError: If invalid
    """
    try:
        num_classes = int(num_classes)
    except (TypeError, ValueError):
        raise InputValidationError(
            message="Number of classes must be an integer",
            parameter_name="num_classes",
            received_value=str(num_classes),
            expected="A positive integer",
        )

    if num_classes < 2:
        raise InputValidationError(
            message="Number of classes must be at least 2",
            parameter_name="num_classes",
            received_value=str(num_classes),
            expected="At least 2 classes",
        )

    if num_classes > 1000:
        raise InputValidationError(
            message="Number of classes too large (maximum 1000)",
            parameter_name="num_classes",
            received_value=str(num_classes),
            expected="At most 1000 classes",
        )

    return num_classes
