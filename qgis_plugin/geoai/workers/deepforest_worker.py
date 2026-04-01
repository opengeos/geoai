#!/usr/bin/env python3
"""DeepForest inference worker process for the GeoAI QGIS plugin."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import json
import math
import os
import sys
import traceback
from typing import Any

_PROTO_STDOUT = sys.stdout

_STATE = {
    "model": None,
    "model_name": None,
}


def _send_ok(result: Any = None) -> None:
    _PROTO_STDOUT.write(json.dumps({"type": "ok", "result": result}) + "\n")
    _PROTO_STDOUT.flush()


def _send_error(message: str) -> None:
    _PROTO_STDOUT.write(json.dumps({"type": "error", "message": message}) + "\n")
    _PROTO_STDOUT.flush()


def _run_quiet_stdout(fn, *args, **kwargs):
    with redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _configure_process_env() -> None:
    # Prevent PyTorch Lightning distributed process spawning.
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["PL_TRAINER_STRATEGY"] = "auto"
    os.environ["PL_ACCELERATOR"] = "gpu"


def _get_model():
    model = _STATE.get("model")
    if model is None:
        raise RuntimeError("DeepForest model is not initialized")
    return model


def _serialize_predictions(result) -> dict:
    if result is None:
        return {"columns": [], "records": []}

    try:
        if getattr(result, "empty", False):
            columns = [str(c) for c in getattr(result, "columns", [])]
            if "geometry" in columns:
                columns = [c for c in columns if c != "geometry"]
            return {"columns": columns, "records": []}
    except Exception:
        pass

    try:
        df = result.copy()
    except Exception:
        df = result

    try:
        if "geometry" in getattr(df, "columns", []):
            df = df.drop(columns=["geometry"])
    except Exception:
        pass

    columns = [str(c) for c in getattr(df, "columns", [])]

    try:
        raw_records = df.to_dict(orient="records")
    except Exception:
        raw_records = []
        try:
            for _, row in df.iterrows():
                raw_records.append(dict(row))
        except Exception:
            pass

    records = []
    for rec in raw_records:
        records.append({str(k): _json_safe(v) for k, v in rec.items()})
    return {"columns": columns, "records": records}


def _handle_init(req: dict) -> Any:
    _configure_process_env()

    try:
        import torch

        torch.set_float32_matmul_precision("medium")
        if torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    except Exception:
        pass

    with redirect_stdout(sys.stderr):
        main = importlib.import_module("deepforest.main")

    model_name = str(req.get("model_name") or "weecology/deepforest-tree")
    revision = str(req.get("revision") or "main")
    device = req.get("device")

    model = _run_quiet_stdout(main.deepforest)
    _run_quiet_stdout(model.load_model, model_name=model_name, revision=revision)

    if device and device != "auto":
        if device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    _run_quiet_stdout(model.model.to, "cuda")
            except Exception:
                pass
        elif device == "cpu":
            try:
                _run_quiet_stdout(model.model.to, "cpu")
            except Exception:
                pass

    _STATE["model"] = model
    _STATE["model_name"] = model_name
    return {"status": "ready", "model_name": model_name}


def _prepare_image_for_deepforest(image_path: str) -> tuple[str, str | None]:
    temp_rgb_path = None
    try:
        import rasterio

        with rasterio.open(image_path) as src:
            if src.count >= 4:
                profile = src.profile.copy()
                profile.update(count=3)
                temp_rgb_path = image_path + ".deepforest_rgb.tif"
                with rasterio.open(temp_rgb_path, "w", **profile) as dst:
                    for band in range(1, 4):
                        dst.write(src.read(band), band)
                return temp_rgb_path, temp_rgb_path
    except Exception:
        pass
    return image_path, None


def _is_cuda_oom(exc: Exception) -> bool:
    """Check whether *exc* is a CUDA out-of-memory error."""
    try:
        import torch

        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        # String fallback: only match when CUDA is actually in use to
        # avoid misclassifying CPU OOM or raster I/O errors.
        if torch.cuda.is_available():
            msg = str(exc).lower()
            if "out of memory" in msg and ("cuda" in msg or "cublas" in msg):
                return True
    except Exception:
        pass
    return False


def _predict_tile_with_oom_retry(
    model,
    path: str,
    *,
    patch_size: int,
    patch_overlap: float,
    iou_threshold: float,
    dataloader_strategy: str,
    batch_size: int,
    accelerator: str,
) -> Any:
    """Run predict_tile with automatic OOM recovery.

    Model config is restored after retries so the singleton model in
    ``_STATE`` is not permanently degraded for subsequent requests.
    """
    tile_kwargs = dict(
        path=path,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        iou_threshold=iou_threshold,
    )

    # Snapshot config for restoration after retries.
    orig_batch = model.config.get("batch_size")
    orig_accel = model.config.get("accelerator")

    try:
        return _run_quiet_stdout(
            model.predict_tile,
            **tile_kwargs,
            dataloader_strategy=dataloader_strategy,
        )
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise

    # Retry 1: reduce batch, switch to window strategy
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    reduced_batch = max(1, batch_size // 2)
    model.config["batch_size"] = reduced_batch
    try:
        _run_quiet_stdout(model.create_trainer, accelerator=accelerator, devices=1)
    except Exception:
        pass
    try:
        return _run_quiet_stdout(
            model.predict_tile, **tile_kwargs, dataloader_strategy="window"
        )
    except RuntimeError as exc2:
        if not _is_cuda_oom(exc2):
            raise
    finally:
        model.config["batch_size"] = orig_batch

    # Retry 2: fall back to CPU
    try:
        import torch

        torch.cuda.empty_cache()
        model.model.to("cpu")
    except Exception:
        pass
    model.config["accelerator"] = "cpu"
    model.config["batch_size"] = 1
    try:
        _run_quiet_stdout(model.create_trainer, accelerator="cpu", devices=1)
    except Exception:
        pass
    try:
        return _run_quiet_stdout(
            model.predict_tile, **tile_kwargs, dataloader_strategy="window"
        )
    except Exception as cpu_exc:
        raise RuntimeError(
            f"DeepForest prediction failed on GPU and CPU fallback: {cpu_exc}"
        ) from cpu_exc
    finally:
        # Restore original config so the singleton model is not degraded.
        model.config["batch_size"] = orig_batch
        model.config["accelerator"] = orig_accel
        if orig_accel != "cpu":
            try:
                model.model.to(orig_accel.replace("gpu", "cuda"))
            except Exception:
                pass
            try:
                _run_quiet_stdout(
                    model.create_trainer, accelerator=orig_accel, devices=1
                )
            except Exception:
                pass


def _handle_predict(req: dict) -> Any:
    _configure_process_env()
    model = _get_model()

    image_path = req["image_path"]
    mode = req.get("mode") or "Single Image"
    patch_size = int(req.get("patch_size") or 400)
    patch_overlap = float(req.get("patch_overlap") or 0.25)
    iou_threshold = float(req.get("iou_threshold") or 0.15)
    dataloader_strategy = req.get("dataloader_strategy") or "batch"
    score_threshold = float(req.get("score_threshold") or 0.3)
    batch_size = int(req.get("batch_size") or 4)

    # Configure model for safe in-process behavior (in the worker process).
    try:
        import torch

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        accelerator = "cpu"

    try:
        model.config["workers"] = 0
        model.config["batch_size"] = batch_size
        model.config["devices"] = 1
        model.config["accelerator"] = accelerator
    except Exception:
        pass

    try:
        _run_quiet_stdout(model.create_trainer, accelerator=accelerator, devices=1)
    except Exception:
        pass

    pred_path, temp_rgb_path = _prepare_image_for_deepforest(image_path)
    try:
        if mode == "Single Image":
            result = _run_quiet_stdout(model.predict_image, path=pred_path)
            pred_mode = "single"
        else:
            result = _predict_tile_with_oom_retry(
                model,
                pred_path,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                iou_threshold=iou_threshold,
                dataloader_strategy=dataloader_strategy,
                batch_size=batch_size,
                accelerator=accelerator,
            )
            pred_mode = "tile"
    finally:
        if temp_rgb_path and os.path.exists(temp_rgb_path):
            try:
                os.remove(temp_rgb_path)
            except Exception:
                pass

    try:
        if (
            score_threshold > 0
            and result is not None
            and not result.empty
            and "score" in result.columns
        ):
            result = result[result.score >= score_threshold]
    except Exception:
        pass

    return {
        "pred_mode": pred_mode,
        "predictions": _serialize_predictions(result),
    }


def _cleanup() -> None:
    model = _STATE.get("model")
    if model is not None:
        try:
            if hasattr(model, "model"):
                try:
                    model.model = None
                except Exception:
                    pass
        except Exception:
            pass
    _STATE["model"] = None
    _STATE["model_name"] = None


def _dispatch(req: dict) -> Any:
    action = req.get("action")
    if action == "init":
        return _handle_init(req)
    if action == "predict":
        return _handle_predict(req)
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
