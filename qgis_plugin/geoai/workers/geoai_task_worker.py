#!/usr/bin/env python3
"""One-shot worker process for running geoai tasks outside the QGIS process."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Callable, Dict

_PROTO_STDOUT = sys.stdout


def _emit(message: Dict[str, Any]) -> None:
    print(json.dumps(message), file=_PROTO_STDOUT, flush=True)


def _progress(message: str) -> None:
    _emit({"type": "progress", "message": message})


def _ok(result: Any) -> None:
    _emit({"type": "ok", "result": result})


def _error(message: str) -> None:
    _emit({"type": "error", "message": message})


class _ProgressStdout:
    """Capture printed output and re-emit line-by-line as progress events."""

    def __init__(self, callback: Callable[[str], None]):
        self._callback = callback
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._callback(line)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._callback(self._buf)
        self._buf = ""


def _run_with_progress(fn: Callable[[], Any]) -> Any:
    old_stdout = sys.stdout
    sys.stdout = _ProgressStdout(_progress)
    try:
        return fn()
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = old_stdout


def _geoai():
    import geoai

    return geoai


def _driver_for_path(path: str):
    lower = path.lower()
    if lower.endswith(".geojson"):
        return "GeoJSON"
    if lower.endswith(".gpkg"):
        return "GPKG"
    if lower.endswith(".shp"):
        return "ESRI Shapefile"
    return None


def _task_train_segmentation_model(params: Dict[str, Any]) -> Dict[str, Any]:
    geoai = _geoai()
    _progress("Starting training...")
    _run_with_progress(lambda: geoai.train_segmentation_model(**params))
    model_path = os.path.join(params["output_dir"], "best_model.pth")
    return {"model_path": model_path}


def _task_train_instance_segmentation_model(params: Dict[str, Any]) -> Dict[str, Any]:
    geoai = _geoai()
    _progress("Starting Mask R-CNN training...")
    _run_with_progress(lambda: geoai.train_instance_segmentation_model(**params))
    model_path = os.path.join(params["output_dir"], "best_model.pth")
    return {"model_path": model_path}


def _task_export_geotiff_tiles(params: Dict[str, Any]) -> Dict[str, Any]:
    from geoai.utils.training import export_geotiff_tiles

    _progress("Exporting tiles...")
    _run_with_progress(lambda: export_geotiff_tiles(**params))
    return {"output_dir": params["out_folder"]}


def _task_semantic_segmentation(params: Dict[str, Any]) -> Dict[str, Any]:
    geoai = _geoai()
    _progress("Running inference...")
    _run_with_progress(lambda: geoai.semantic_segmentation(**params))
    return {"output_path": params["output_path"]}


def _task_instance_segmentation(params: Dict[str, Any]) -> Dict[str, Any]:
    geoai = _geoai()
    _progress("Running instance segmentation inference...")
    _run_with_progress(lambda: geoai.instance_segmentation(**params))
    return {"output_path": params["output_path"]}


def _task_vectorize_mask(params: Dict[str, Any]) -> Dict[str, Any]:
    from geoai.utils.geometry import orthogonalize
    from geoai.utils.vector import add_geometric_properties

    mask_path = params["mask_path"]
    output_path = params["output_path"]
    epsilon = params.get("epsilon", 2.0)
    min_area = params.get("min_area")

    _progress("Vectorizing mask...")
    gdf = orthogonalize(mask_path, output_path, epsilon=epsilon)

    if min_area is not None and min_area > 0:
        _progress(f"Filtering by area (min: {min_area})...")
        gdf = add_geometric_properties(gdf, area_unit="m2")
        gdf = gdf[gdf["area_m2"] >= min_area]
        driver = _driver_for_path(output_path)
        gdf.to_file(output_path, driver=driver)

    return {"output_path": output_path}


def _task_smooth_vector(params: Dict[str, Any]) -> Dict[str, Any]:
    from geoai.utils.raster import raster_to_vector
    from geoai.utils.vector import smooth_vector

    mask_path = params["mask_path"]
    output_path = params["output_path"]
    smooth_iterations = params.get("smooth_iterations", 3)
    min_area = params.get("min_area")
    simplify_tolerance = params.get("simplify_tolerance")

    _progress("Converting raster to vector...")
    min_area_value = min_area if min_area is not None else 0
    gdf = raster_to_vector(
        mask_path,
        min_area=min_area_value,
        simplify_tolerance=simplify_tolerance,
    )

    _progress(f"Smoothing vector (iterations: {smooth_iterations})...")
    smooth_vector(
        gdf,
        smooth_iterations=smooth_iterations,
        output_path=output_path,
    )
    return {"output_path": output_path}


def _task_segment_water(params: Dict[str, Any]) -> Dict[str, Any]:
    geoai = _geoai()
    _progress("Starting water segmentation...")
    result = _run_with_progress(lambda: geoai.segment_water(**params))

    input_path = params["input_path"]
    output_raster = params.get("output_raster")
    output_vector = params.get("output_vector") or ""

    if isinstance(result, str):
        final_raster = result
    elif output_raster:
        final_raster = output_raster
    else:
        stem = os.path.splitext(input_path)[0]
        final_raster = f"{stem}_water_mask.tif"

    return {"output_raster": final_raster, "output_vector": output_vector}


_TASKS = {
    "train_segmentation_model": _task_train_segmentation_model,
    "train_instance_segmentation_model": _task_train_instance_segmentation_model,
    "export_geotiff_tiles": _task_export_geotiff_tiles,
    "semantic_segmentation": _task_semantic_segmentation,
    "instance_segmentation": _task_instance_segmentation,
    "vectorize_mask": _task_vectorize_mask,
    "smooth_vector": _task_smooth_vector,
    "segment_water": _task_segment_water,
}


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        _error("No request received")
        return 1

    try:
        req = json.loads(line)
    except json.JSONDecodeError as exc:
        _error(f"Invalid JSON request: {exc}")
        return 1

    action = req.get("action")
    params = req.get("params") or {}
    handler = _TASKS.get(action)
    if handler is None:
        _error(f"Unknown action: {action}")
        return 1

    try:
        result = handler(params)
        _ok(result)
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
