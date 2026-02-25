#!/usr/bin/env python3
"""Moondream inference worker process for the GeoAI QGIS plugin.

This script is launched by the plugin using the managed venv Python so
torch loads outside the QGIS process.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
import sys
import tempfile
import traceback
from typing import Any

_STATE = {
    "moondream": None,
    "temp_vectors": set(),
}


def _send_ok(result: Any = None) -> None:
    print(json.dumps({"type": "ok", "result": result}), flush=True)


def _send_error(message: str) -> None:
    print(json.dumps({"type": "error", "message": message}), flush=True)


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


def _store_vector_result(result: dict) -> dict:
    """Replace GeoDataFrame in result with a temp GeoJSON path."""
    output = dict(result)
    gdf = output.pop("gdf", None)

    if gdf is not None:
        try:
            feature_count = len(gdf)
        except Exception:
            feature_count = 0
        output["vector_feature_count"] = feature_count

        if feature_count > 0:
            fd, path = tempfile.mkstemp(prefix="geoai_moondream_", suffix=".geojson")
            os.close(fd)
            try:
                gdf.to_file(path, driver="GeoJSON")
                _STATE["temp_vectors"].add(path)
                output["vector_path"] = path
            except Exception:
                try:
                    os.remove(path)
                except OSError:
                    pass
                raise

    return _json_safe(output)


def _run_quiet_stdout(fn, *args, **kwargs):
    """Run a callable while redirecting stdout away from the JSON protocol."""
    with redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _handle_init(req: dict) -> Any:
    from geoai import MoondreamGeo

    _STATE["moondream"] = _run_quiet_stdout(
        MoondreamGeo,
        model_name=req.get("model_name") or "vikhyatk/moondream2",
        device=req.get("device"),
    )
    return {"status": "ready"}


def _get_moondream():
    moondream = _STATE.get("moondream")
    if moondream is None:
        raise RuntimeError("Moondream model is not initialized")
    return moondream


def _handle_load_image(req: dict) -> Any:
    moondream = _get_moondream()
    _image, metadata = _run_quiet_stdout(
        moondream.load_image, req["source"], bands=req.get("bands")
    )
    return _json_safe({"metadata": metadata})


def _handle_caption(req: dict) -> Any:
    moondream = _get_moondream()
    kwargs = req.get("kwargs") or {}
    result = _run_quiet_stdout(
        moondream.caption,
        req["source"],
        length=req.get("length", "normal"),
        stream=bool(req.get("stream", False)),
        bands=req.get("bands"),
        settings=req.get("settings"),
        **kwargs,
    )
    return _json_safe(result)


def _handle_query(req: dict) -> Any:
    moondream = _get_moondream()
    kwargs = req.get("kwargs") or {}
    result = _run_quiet_stdout(
        moondream.query,
        req["question"],
        source=req.get("source"),
        reasoning=req.get("reasoning"),
        stream=bool(req.get("stream", False)),
        bands=req.get("bands"),
        settings=req.get("settings"),
        **kwargs,
    )
    return _json_safe(result)


def _handle_detect(req: dict) -> Any:
    moondream = _get_moondream()
    kwargs = req.get("kwargs") or {}
    result = _run_quiet_stdout(
        moondream.detect,
        req["source"],
        req["object_type"],
        bands=req.get("bands"),
        output_path=req.get("output_path"),
        settings=req.get("settings"),
        stream=bool(req.get("stream", False)),
        **kwargs,
    )
    return _store_vector_result(result)


def _handle_point(req: dict) -> Any:
    moondream = _get_moondream()
    kwargs = req.get("kwargs") or {}
    result = _run_quiet_stdout(
        moondream.point,
        req["source"],
        req["object_description"],
        bands=req.get("bands"),
        output_path=req.get("output_path"),
        **kwargs,
    )
    return _store_vector_result(result)


def _cleanup() -> None:
    moondream = _STATE.get("moondream")
    if moondream is not None:
        try:
            if hasattr(moondream, "model") and moondream.model is not None:
                try:
                    moondream.model = None
                except Exception:
                    pass
        except Exception:
            pass
    _STATE["moondream"] = None

    for path in list(_STATE.get("temp_vectors", ())):
        try:
            os.remove(path)
        except OSError:
            pass
        finally:
            _STATE["temp_vectors"].discard(path)


def _dispatch(req: dict) -> Any:
    action = req.get("action")
    if action == "init":
        return _handle_init(req)
    if action == "load_image":
        return _handle_load_image(req)
    if action == "caption":
        return _handle_caption(req)
    if action == "query":
        return _handle_query(req)
    if action == "detect":
        return _handle_detect(req)
    if action == "point":
        return _handle_point(req)
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
            _send_ok(result)
            if req.get("action") == "shutdown":
                return 0
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send_error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
