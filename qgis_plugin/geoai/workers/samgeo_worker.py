#!/usr/bin/env python3
"""SamGeo inference worker process for the GeoAI QGIS plugin.

This script is launched by the plugin using the managed venv Python so torch and
samgeo load outside the QGIS process.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

_PROTO_STDOUT = sys.stdout

_STATE = {
    "sam": None,
    "model_name": None,
    "supported_methods": set(),
}


def _bootstrap_windows_torch_dll_dirs() -> None:
    """Register venv PyTorch DLL directories before importing SamGeo on Windows."""
    if sys.platform != "win32":
        return

    try:
        import sysconfig

        site_packages = sysconfig.get_paths().get(
            "platlib"
        ) or sysconfig.get_paths().get("purelib")
    except Exception:
        site_packages = None

    if not site_packages:
        return

    dll_dirs = [
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "torch", "bin"),
        os.path.join(site_packages, "torchvision"),
    ]
    path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    _dll_handles: list = []
    for dll_dir in dll_dirs:
        if not os.path.isdir(dll_dir):
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                _dll_handles.append(os.add_dll_directory(dll_dir))
            except OSError:
                pass
        if dll_dir not in path_parts:
            path_parts.insert(0, dll_dir)
    os.environ["PATH"] = os.pathsep.join(path_parts)


_bootstrap_windows_torch_dll_dirs()


def _send_ok(result: Any = None) -> None:
    _PROTO_STDOUT.write(json.dumps({"type": "ok", "result": result}) + "\n")
    _PROTO_STDOUT.flush()


def _send_error(message: str) -> None:
    _PROTO_STDOUT.write(json.dumps({"type": "error", "message": message}) + "\n")
    _PROTO_STDOUT.flush()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__fspath__"):
        try:
            return os.fspath(value)
        except Exception:
            pass
    return str(value)


def _run_quiet_stdout(fn, *args, **kwargs):
    # Third-party libs sometimes print to stdout. Keep stdout JSON-only.
    with redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _ensure_pkg_resources_shim() -> bool:
    try:
        plugin_parent = Path(__file__).resolve().parents[2]
        plugin_parent_str = str(plugin_parent)
        if plugin_parent_str not in sys.path:
            sys.path.insert(0, plugin_parent_str)

        from geoai._pkg_resources_compat import ensure_pkg_resources

        return bool(ensure_pkg_resources())
    except Exception:
        return False


def _get_sam():
    sam = _STATE.get("sam")
    if sam is None:
        raise RuntimeError("SamGeo model is not initialized")
    return sam


def _mask_count() -> int:
    sam = _STATE.get("sam")
    if sam is None:
        return 0
    try:
        masks = getattr(sam, "masks", None)
        return len(masks) if masks is not None else 0
    except Exception:
        return 0


def _supported_methods_for(sam) -> set[str]:
    methods = set()
    for name in (
        "generate_masks",
        "generate_masks_by_points",
        "predict",
        "generate_masks_by_boxes",
        "generate_masks_by_points_patch",
        "generate_masks_tiled",
        "save_masks",
        "set_image",
    ):
        if hasattr(sam, name):
            methods.add(name)
    return methods


def _resolve_device(device: Any) -> Any:
    """Pick an accelerator for automatic device selection.

    The QGIS process used to choose the device, but it can only inspect its own
    torch, which may be missing or a different build from the venv's. This
    process owns the torch that actually runs the model, so resolve here. MPS
    is checked before CUDA to match the previous in-process behavior.

    Args:
        device: Requested device, or None/"auto" to detect one.

    Returns:
        A concrete device string, or the caller's value if it was explicit.
    """
    if device and device != "auto":
        return device

    try:
        import torch
    except Exception:
        return None

    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _handle_init(req: dict) -> Any:
    shim_installed = _ensure_pkg_resources_shim()

    model_version = str(req.get("model_version") or "")
    backend = req.get("backend")
    device = _resolve_device(req.get("device"))
    confidence = req.get("confidence")
    enable_interactive = bool(req.get("enable_interactive", False))

    if "SamGeo3" in model_version:
        from samgeo import SamGeo3

        sam = _run_quiet_stdout(
            SamGeo3,
            backend=backend,
            device=device,
            confidence_threshold=confidence,
            enable_inst_interactivity=enable_interactive,
        )
        model_name = "SamGeo3"
    elif "SamGeo2" in model_version:
        from samgeo import SamGeo2

        sam = _run_quiet_stdout(SamGeo2, device=device)
        model_name = "SamGeo2"
    else:
        from samgeo import SamGeo

        sam = _run_quiet_stdout(SamGeo, device=device)
        model_name = "SamGeo"

    _STATE["sam"] = sam
    _STATE["model_name"] = model_name
    _STATE["supported_methods"] = _supported_methods_for(sam)

    return _json_safe(
        {
            "status": "ready",
            "model_name": model_name,
            "mask_count": _mask_count(),
            "supported_methods": sorted(_STATE["supported_methods"]),
            "pkg_resources_shim_installed": shim_installed,
        }
    )


def _handle_set_image(req: dict) -> Any:
    sam = _get_sam()
    _run_quiet_stdout(sam.set_image, req["source"], bands=req.get("bands"))
    return {"mask_count": _mask_count()}


def _handle_generate_masks(req: dict) -> Any:
    sam = _get_sam()
    _run_quiet_stdout(
        sam.generate_masks,
        req["prompt"],
        min_size=req.get("min_size"),
        max_size=req.get("max_size"),
    )
    return {"mask_count": _mask_count()}


def _to_numpy_if_possible(value):
    if value is None:
        return None
    try:
        import numpy as np

        return np.array(value)
    except Exception:
        return value


def _predict_with_compat(sam, kwargs: dict) -> None:
    call_kwargs = dict(kwargs)

    if "point_coords" in call_kwargs:
        call_kwargs["point_coords"] = _to_numpy_if_possible(call_kwargs["point_coords"])
    if "point_labels" in call_kwargs:
        call_kwargs["point_labels"] = _to_numpy_if_possible(call_kwargs["point_labels"])
    if "box" in call_kwargs:
        call_kwargs["box"] = _to_numpy_if_possible(call_kwargs["box"])

    try:
        _run_quiet_stdout(sam.predict, **call_kwargs)
    except TypeError:
        # Older SamGeo versions may not support multimask_output.
        if "multimask_output" in call_kwargs:
            call_kwargs.pop("multimask_output", None)
            _run_quiet_stdout(sam.predict, **call_kwargs)
        else:
            raise


def _handle_generate_masks_by_points(req: dict) -> Any:
    sam = _get_sam()
    if hasattr(sam, "generate_masks_by_points"):
        _run_quiet_stdout(
            sam.generate_masks_by_points,
            point_coords=req.get("point_coords"),
            point_labels=req.get("point_labels"),
            point_crs=req.get("point_crs"),
            multimask_output=req.get("multimask_output"),
        )
    else:
        _predict_with_compat(
            sam,
            {
                "point_coords": req.get("point_coords"),
                "point_labels": req.get("point_labels"),
                "point_crs": req.get("point_crs"),
                "multimask_output": req.get("multimask_output"),
            },
        )
    return {"mask_count": _mask_count()}


def _handle_predict(req: dict) -> Any:
    sam = _get_sam()
    _predict_with_compat(sam, req.get("kwargs") or {})
    return {"mask_count": _mask_count()}


def _handle_generate_masks_by_boxes(req: dict) -> Any:
    sam = _get_sam()
    boxes = req.get("boxes") or []
    if hasattr(sam, "generate_masks_by_boxes"):
        _run_quiet_stdout(sam.generate_masks_by_boxes, boxes=boxes)
    else:
        if not boxes:
            raise ValueError("No boxes provided")
        _predict_with_compat(sam, {"box": boxes[0]})
    return {"mask_count": _mask_count()}


def _handle_generate_masks_by_points_patch(req: dict) -> Any:
    sam = _get_sam()
    if not hasattr(sam, "generate_masks_by_points_patch"):
        raise AttributeError("generate_masks_by_points_patch is not supported")

    _run_quiet_stdout(
        sam.generate_masks_by_points_patch,
        point_coords_batch=req.get("point_coords_batch"),
        point_crs=req.get("point_crs"),
        output=req.get("output"),
        unique=bool(req.get("unique", False)),
        min_size=req.get("min_size"),
        max_size=req.get("max_size"),
    )
    return {"mask_count": _mask_count()}


def _mask_scores_of(sam) -> Any:
    """Read the per-mask confidence scores left by ``save_masks``.

    samgeo 1.4.2+ sets ``SamGeo3.mask_scores`` to a mapping of each unique
    raster value to SAM's predicted confidence for that mask (None when the
    masks were saved with ``unique=False``). Older samgeo has no such
    attribute.

    Args:
        sam: The SamGeo model object.

    Returns:
        A JSON-safe ``{str(mask id): float score}`` dict, or None when no
        scores are available.
    """
    scores = getattr(sam, "mask_scores", None)
    if not scores:
        return None
    try:
        items = list(scores.items())
    except AttributeError:
        return None
    result = {}
    for key, value in items:
        try:
            score = float(value)
            mask_id = str(int(float(key)))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(score):
            continue  # NaN/inf would make the JSON response invalid
        result[mask_id] = score
    return result or None


def _handle_save_masks(req: dict) -> Any:
    """Save the current masks to a raster and report their confidence scores.

    Args:
        req: Request with ``output`` (raster path) and ``unique``.

    Returns:
        A dict with ``mask_count`` and ``mask_scores`` (mask id -> SAM
        confidence, or None when unavailable or ``unique`` is False).
    """
    sam = _get_sam()
    unique = bool(req.get("unique", False))
    _run_quiet_stdout(
        sam.save_masks,
        output=req["output"],
        unique=unique,
    )
    return {
        "mask_count": _mask_count(),
        "mask_scores": _mask_scores_of(sam) if unique else None,
    }


def _iter_raster_blocks(path: str):
    """Yield band 1 of a raster one internal block at a time.

    Args:
        path: Raster path.

    Yields:
        Arrays holding one block window each.
    """
    import rasterio

    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            yield src.read(1, window=window)


def _unique_values(block: Any) -> set:
    """Return the distinct values of one raster block.

    Args:
        block: A numpy array, or nested lists in tests.

    Returns:
        The set of values in the block.
    """
    if hasattr(block, "ravel"):
        import numpy as np

        return set(np.unique(block).tolist())
    return {value for row in block for value in row}


def _count_raster_labels(path: str, unique: bool) -> Any:
    """Count the objects in a labelled mask raster without loading it whole.

    Reads the raster block by block and collects the labels in a set, so
    memory grows with the number of objects rather than the raster size.

    Args:
        path: Path to the mask raster written by ``generate_masks_tiled``.
        unique: Whether each object has its own label.

    Returns:
        The number of distinct non-zero labels when ``unique`` is True, None
        for binary masks (objects cannot be told apart) or if the raster
        cannot be read.
    """
    if not unique:
        return None
    try:
        labels = set()
        for block in _iter_raster_blocks(path):
            labels.update(_unique_values(block))
        labels.discard(0)
        return len(labels)
    except Exception:
        return None


def _handle_generate_masks_tiled(req: dict) -> Any:
    """Run text-prompted segmentation tile by tile on a large GeoTIFF.

    ``SamGeo3.generate_masks_tiled`` writes the merged mask raster to
    ``output`` itself and clears ``masks``/``scores`` after each tile, so the
    result is only available as that raster (no per-mask scores).

    Args:
        req: Request with ``source``, ``prompt``, ``output``, ``tile_size``,
            ``overlap``, ``min_size``, ``max_size``, ``unique`` and ``bands``.

    Returns:
        A dict with ``output`` (the written raster), ``object_count`` and
        ``mask_count`` (always 0: no in-memory masks remain).
    """
    sam = _get_sam()
    if not hasattr(sam, "generate_masks_tiled"):
        raise AttributeError("generate_masks_tiled is not supported")
    unique = bool(req.get("unique", True))
    kwargs = {
        "source": req["source"],
        "prompt": req["prompt"],
        "output": req["output"],
        "tile_size": int(req.get("tile_size") or 1024),
        "overlap": int(req.get("overlap") or 0),
        "min_size": int(req.get("min_size") or 0),
        "max_size": req.get("max_size"),
        "unique": unique,
    }
    if req.get("bands"):
        kwargs["bands"] = req.get("bands")
    output = _run_quiet_stdout(sam.generate_masks_tiled, **kwargs) or req["output"]
    return {
        "output": output,
        "object_count": _count_raster_labels(output, unique),
        "mask_count": _mask_count(),
    }


def _handle_supports(req: dict) -> Any:
    method_name = str(req.get("method_name") or "")
    return {"supported": method_name in _STATE.get("supported_methods", set())}


def _cleanup() -> None:
    sam = _STATE.get("sam")
    if sam is not None:
        try:
            if hasattr(sam, "model"):
                try:
                    sam.model = None
                except Exception:
                    pass
        except Exception:
            pass
    _STATE["sam"] = None
    _STATE["model_name"] = None
    _STATE["supported_methods"] = set()


def _dispatch(req: dict) -> Any:
    action = req.get("action")
    if action == "init":
        return _handle_init(req)
    if action == "set_image":
        return _handle_set_image(req)
    if action == "generate_masks":
        return _handle_generate_masks(req)
    if action == "generate_masks_by_points":
        return _handle_generate_masks_by_points(req)
    if action == "predict":
        return _handle_predict(req)
    if action == "generate_masks_by_boxes":
        return _handle_generate_masks_by_boxes(req)
    if action == "generate_masks_by_points_patch":
        return _handle_generate_masks_by_points_patch(req)
    if action == "generate_masks_tiled":
        return _handle_generate_masks_tiled(req)
    if action == "save_masks":
        return _handle_save_masks(req)
    if action == "supports":
        return _handle_supports(req)
    if action == "shutdown":
        _cleanup()
        return {"status": "shutdown"}
    raise ValueError(f"Unknown action: {action}")


def main() -> int:
    while True:
        line = sys.stdin.readline()
        if not line:
            _cleanup()
            return 0

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _send_error(f"Invalid JSON request: {exc}")
            continue

        try:
            result = _dispatch(req)
            _send_ok(_json_safe(result))
            if req.get("action") == "shutdown":
                return 0
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send_error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
