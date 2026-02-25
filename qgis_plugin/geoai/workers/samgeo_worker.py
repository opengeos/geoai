#!/usr/bin/env python3
"""SamGeo inference worker process for the GeoAI QGIS plugin.

This script is launched by the plugin using the managed venv Python so torch and
samgeo load outside the QGIS process.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import json
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
        "save_masks",
        "set_image",
    ):
        if hasattr(sam, name):
            methods.add(name)
    return methods


def _handle_init(req: dict) -> Any:
    shim_installed = _ensure_pkg_resources_shim()

    model_version = str(req.get("model_version") or "")
    backend = req.get("backend")
    device = req.get("device")
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


def _handle_save_masks(req: dict) -> Any:
    sam = _get_sam()
    _run_quiet_stdout(
        sam.save_masks,
        output=req["output"],
        unique=bool(req.get("unique", False)),
    )
    return {"mask_count": _mask_count()}


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
