#!/usr/bin/env python3
"""One-shot worker process for running geoai tasks outside the QGIS process."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager, redirect_stdout
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


class _ProgressLoggingHandler(logging.Handler):
    """Route logging output through the progress callback."""

    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._callback(msg)
        except Exception:
            self.handleError(record)


def _run_with_progress(fn: Callable[[], Any]) -> Any:
    old_stdout = sys.stdout
    sys.stdout = _ProgressStdout(_progress)

    # Capture logging output (logger.info, etc.) that would otherwise be lost
    log_handler = _ProgressLoggingHandler(_progress)
    log_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    original_level = root_logger.level
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_handler)

    try:
        return fn()
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = old_stdout
        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)


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
    output_path = params["output_path"]
    num_instances = None
    try:
        import rasterio

        if os.path.exists(output_path):
            with rasterio.open(output_path) as src:
                data = src.read(1)
                num_instances = int(data.max()) if getattr(data, "size", 0) > 0 else 0
    except Exception:
        num_instances = None
    return {"output_path": output_path, "num_instances": num_instances}


def _task_clip_raster_by_bbox(params: Dict[str, Any]) -> Dict[str, Any]:
    from geoai.utils.raster import clip_raster_by_bbox

    _progress("Clipping raster to inference range...")
    output_path = clip_raster_by_bbox(**params)
    return {"output_path": output_path}


def _task_raster_to_vector(params: Dict[str, Any]) -> Dict[str, Any]:
    from geoai.utils.raster import raster_to_vector

    mask_path = params["mask_path"]
    output_path = params["output_path"]
    min_area = params.get("min_area") or 0
    output_format = params.get("output_format", "geojson")
    simplify_tolerance = params.get("simplify_tolerance")

    _progress("Converting raster to vector...")
    raster_to_vector(
        mask_path,
        output_path=output_path,
        min_area=min_area,
        simplify_tolerance=simplify_tolerance,
        output_format=output_format,
    )
    return {"output_path": output_path}


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


@contextmanager
def _temp_dir_ignore_cleanup_errors():
    """Create a temporary directory and ignore cleanup errors (Python 3.9 compatible)."""
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _task_detect_changes(params: Dict[str, Any]) -> Dict[str, Any]:
    from geoai.change_detection import ChangeDetection
    from geoai.utils.raster import raster_to_vector

    output_raster = params["output_raster"]
    output_vector = params["output_vector"]

    # Create missing folders now, so a bad path fails before the slow steps.
    # geoai-py 0.43.1 and older don't create the SAM checkpoint folder before
    # downloading, so make the one download_checkpoint() uses.
    folders = [
        os.environ.get(
            "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
        ),
        os.path.dirname(output_raster),
        os.path.dirname(output_vector),
    ]
    for folder in folders:
        if folder:
            os.makedirs(folder, exist_ok=True)

    _progress(f"Loading AnyChange (SAM {params['sam_model_type']})...")
    detector = _run_with_progress(
        lambda: ChangeDetection(sam_model_type=params["sam_model_type"])
    )
    detector.set_hyperparameters(
        change_confidence_threshold=params["change_confidence_threshold"],
        auto_threshold=params["auto_threshold"],
    )

    # The instance raster (and the score raster detect_changes writes next to
    # it) only feed the polygons, so keep them out of the user's folders.
    with _temp_dir_ignore_cleanup_errors() as tmp_dir:
        # Each change instance has its own ID here, so it becomes its own polygon.
        instance_raster = (
            os.path.join(tmp_dir, "instances.tif") if output_vector else None
        )

        _progress("Detecting changes...")
        _run_with_progress(
            lambda: detector.detect_changes(
                params["before_path"],
                params["after_path"],
                output_path=output_raster,
                return_results=False,
                export_instance_masks=bool(output_vector),
                instance_masks_output_path=instance_raster,
            )
        )

        if output_vector:
            _progress("Converting change instances to polygons...")
            raster_to_vector(
                instance_raster,
                output_path=output_vector,
                min_area=params["min_area"],
            )

    return {"output_raster": output_raster, "output_vector": output_vector}


def _redact(text: Any, secret: str) -> str:
    """Replace every occurrence of ``secret`` in ``text`` with ``***``.

    Args:
        text: Message that might contain the secret.
        secret: The secret (an access token).

    Returns:
        The redacted message.
    """
    text = "" if text is None else str(text)
    return text.replace(secret, "***") if secret else text


def _huggingface_hub():
    """Import ``huggingface_hub`` from the plugin environment.

    Returns:
        The ``huggingface_hub`` module.

    Raises:
        RuntimeError: If the package is not installed.
    """
    try:
        import huggingface_hub
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed in the plugin environment. "
            "Reinstall the plugin dependencies."
        ) from None
    return huggingface_hub


def _hf_token_files(hub) -> list:
    """List the files where ``huggingface_hub`` stores access tokens.

    Args:
        hub: The ``huggingface_hub`` module.

    Returns:
        Paths of the active token file and the stored-tokens file (those the
        installed version defines, honoring ``HF_HOME``/``HF_TOKEN_PATH``).
    """
    paths = []
    constants = getattr(hub, "constants", None)
    for name in ("HF_TOKEN_PATH", "HF_STORED_TOKENS_PATH"):
        value = getattr(constants, name, None) if constants is not None else None
        if value:
            paths.append(str(value))
    return paths


def _restrict_token_files(hub) -> None:
    """Make the saved token files readable only by their owner (best effort).

    Does nothing on Windows, where POSIX modes do not apply.

    Args:
        hub: The ``huggingface_hub`` module.
    """
    if os.name == "nt":
        return
    for path in _hf_token_files(hub):
        try:
            if os.path.isfile(path):
                os.chmod(path, 0o600)
        except OSError:
            pass


def _task_hf_login(params: Dict[str, Any]) -> Dict[str, Any]:
    """Save a Hugging Face access token for the plugin environment.

    Equivalent to ``hf auth login`` (the token is written to the Hugging Face
    token cache, not to git credentials). On POSIX systems the token files are
    created with mode 0600 and tightened to 0600 if they already existed. The
    token is never included in progress messages, results or errors.

    Args:
        params: ``{"token": <access token>}``.

    Returns:
        ``{"logged_in": True, "user": <user name or None>}``. ``user`` is None
        when the token was saved but the account name could not be looked up.

    Raises:
        ValueError: If no token was given.
        RuntimeError: If the token could not be validated or saved (message
            redacted).
    """
    token = str(params.get("token") or "").strip()
    if not token:
        raise ValueError("No Hugging Face token was provided.")
    hub = _huggingface_hub()
    old_umask = os.umask(0o077) if os.name != "nt" else None
    try:
        # Libraries may print; stdout is the JSON protocol channel.
        with redirect_stdout(sys.stderr):
            hub.login(token=token, add_to_git_credential=False)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face login failed: {_redact(exc, token)}"
        ) from None
    finally:
        if old_umask is not None:
            os.umask(old_umask)
    _restrict_token_files(hub)

    # The token is saved now; a failed name lookup is not a login failure.
    user = None
    try:
        with redirect_stdout(sys.stderr):
            info = hub.whoami(token=token)
        if isinstance(info, dict):
            user = info.get("name")
    except Exception:
        user = None
    return {"logged_in": True, "user": user}


def _task_hf_whoami(params: Dict[str, Any]) -> Dict[str, Any]:
    """Report which Hugging Face account the plugin environment uses.

    Args:
        params: Unused.

    Returns:
        ``{"logged_in": bool, "user": name or None, "credentials_found": bool,
        "message": str}``. ``credentials_found`` is True when a token is stored or
        set through ``HF_TOKEN`` even if it could not be verified.
    """
    hub = _huggingface_hub()
    get_token = getattr(hub, "get_token", None)
    token = None
    if callable(get_token):
        try:
            token = get_token()
        except Exception:
            token = None
    if not token:
        return {
            "logged_in": False,
            "user": None,
            "credentials_found": False,
            "message": "Not logged in to Hugging Face.",
        }
    try:
        with redirect_stdout(sys.stderr):
            info = hub.whoami(token=token)
    except Exception as exc:
        return {
            "logged_in": False,
            "user": None,
            "credentials_found": True,
            "message": "A Hugging Face token is set but could not be verified: "
            + _redact(exc, token),
        }
    user = info.get("name") if isinstance(info, dict) else None
    return {
        "logged_in": True,
        "user": user,
        "credentials_found": True,
        "message": f"Logged in to Hugging Face as {user}.",
    }


_TASKS = {
    "train_segmentation_model": _task_train_segmentation_model,
    "train_instance_segmentation_model": _task_train_instance_segmentation_model,
    "export_geotiff_tiles": _task_export_geotiff_tiles,
    "semantic_segmentation": _task_semantic_segmentation,
    "instance_segmentation": _task_instance_segmentation,
    "clip_raster_by_bbox": _task_clip_raster_by_bbox,
    "raster_to_vector": _task_raster_to_vector,
    "vectorize_mask": _task_vectorize_mask,
    "smooth_vector": _task_smooth_vector,
    "segment_water": _task_segment_water,
    "detect_changes": _task_detect_changes,
    "hf_login": _task_hf_login,
    "hf_whoami": _task_hf_whoami,
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
