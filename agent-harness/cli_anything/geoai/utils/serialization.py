"""Shared JSON serialization, sanitization, and atomic file I/O helpers."""

import json
import math
import os
import tempfile
from typing import Any, Dict


def sanitize_json_value(value: Any) -> Any:
    """Recursively replace non-finite floats (NaN, Inf, -Inf) with None.

    Also converts numpy scalar/array values into plain JSON-serializable
    Python types so ``json.dumps(..., allow_nan=False)`` always succeeds.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # Handle numpy scalars and arrays without requiring numpy as a hard import
    if (
        hasattr(value, "item")
        and callable(value.item)
        and getattr(value, "ndim", 1) == 0
    ):
        scalar = value.item()
        if scalar is value or type(scalar) is type(value):
            # Extended-precision scalars (e.g. np.longdouble) return another
            # NumPy scalar from .item(), so convert directly to terminate.
            try:
                fval = float(scalar)
                return fval if math.isfinite(fval) else None
            except (TypeError, ValueError, OverflowError):
                return str(scalar)
        return sanitize_json_value(scalar)

    if hasattr(value, "tolist") and callable(value.tolist):
        converted = value.tolist()
        if converted is not value:
            return sanitize_json_value(converted)
        return str(value)

    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]

    return value


def serialize_info(info: Any) -> Dict[str, Any]:
    """Convert geoai info output into a clean, JSON-serializable dict."""
    if isinstance(info, dict):
        result: Dict[str, Any] = {}
        for k, v in info.items():
            if hasattr(v, "__geo_interface__"):
                result[str(k)] = str(v)
            else:
                try:
                    cleaned = sanitize_json_value(v)
                    json.dumps(cleaned, allow_nan=False)
                    result[str(k)] = cleaned
                except (TypeError, ValueError, RecursionError):
                    result[str(k)] = str(v)
        return result
    return {"info": str(info)}


def atomic_write_json(data: Any, path: str) -> str:
    """Atomically write JSON data to ``path`` via a sibling temp file.

    Args:
        data: JSON-serializable payload.
        path: Destination file path.

    Returns:
        Absolute path to the written file.
    """
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path) or "."
    os.makedirs(parent, exist_ok=True)

    payload = sanitize_json_value(data)
    existing_mode: int | None = None
    try:
        existing_mode = os.stat(abs_path).st_mode & 0o777
    except OSError:
        existing_mode = None

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=".geoai_tmp_",
            suffix=".json",
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(payload, tmp_file, indent=2, default=str, allow_nan=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        if existing_mode is not None:
            try:
                os.chmod(tmp_path, existing_mode)
            except OSError:
                pass
        os.replace(tmp_path, abs_path)
        tmp_path = None
        return abs_path
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
