"""Subprocess-backed SamGeo client for QGIS.

Runs SamGeo/SAM3 (torch-dependent) model loading and inference in the plugin
venv's Python process to avoid DLL conflicts with QGIS's embedded Python on
Windows.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from .subprocess_client import PersistentSubprocessClient
from .venv_manager import (
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    get_venv_python_path,
    venv_exists,
)

# Hugging Face gives these variables priority over the token saved by
# ``huggingface_hub.login``. After a panel login they are dropped from the
# SamGeo worker's environment only, leaving the QGIS process untouched.
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_PREFER_SAVED_HF_TOKEN = False


def hf_token_env_overrides() -> tuple:
    """Return the token variables new venv subprocesses should not inherit.

    Returns:
        The Hugging Face token variable names after a panel login, otherwise
        an empty tuple. Used so the account-status check describes the same
        credential that new SamGeo workers use.
    """
    return _HF_TOKEN_ENV_VARS if _PREFER_SAVED_HF_TOKEN else ()


def prefer_saved_hf_token() -> list:
    """Make SamGeo workers started from now on use the saved HF token.

    Returns:
        Names of the token variables set in the QGIS environment that new
        SamGeo workers will no longer inherit.
    """
    global _PREFER_SAVED_HF_TOKEN
    _PREFER_SAVED_HF_TOKEN = True
    return [name for name in _HF_TOKEN_ENV_VARS if os.environ.get(name)]


class _MaskCountProxy:
    """Lightweight list-like placeholder so existing UI len()/truthiness checks work."""

    def __init__(self, count: int):
        self.count = max(int(count or 0), 0)

    def __len__(self):
        return self.count

    def __bool__(self):
        return self.count > 0

    def __repr__(self):
        return f"<MaskCountProxy count={self.count}>"


class SamGeoSubprocessClient(PersistentSubprocessClient):
    """Thin stateful proxy for SamGeo running in a persistent subprocess."""

    _WORKER_LABEL = "SamGeo"
    _TIMEOUT_INIT = 900
    _TIMEOUT_SET_IMAGE = 180
    _TIMEOUT_REQUEST = 600
    _STDERR_TAIL_CHARS = 2000
    _EMPTY_RESULT_IS_DICT = True

    def __init__(
        self,
        model_version: str,
        backend: str,
        device: Optional[str],
        confidence: float,
        enable_interactive: bool,
    ):
        self.model_version = model_version
        self.backend = backend
        self.device = device
        self.confidence = float(confidence)
        self.enable_interactive = bool(enable_interactive)

        self.model_name = "SamGeo"
        self.supported_methods = set()
        self.masks = None
        self.mask_scores = None
        self.objects = None

        plugin_root = Path(__file__).resolve().parents[1]
        super().__init__(plugin_root / "workers" / "samgeo_worker.py")

    def _prepare_env(self) -> dict:
        env = super()._prepare_env()
        if _PREFER_SAVED_HF_TOKEN:
            for name in _HF_TOKEN_ENV_VARS:
                env.pop(name, None)
        return env

    def initialize(self) -> None:
        with self._lock:
            if self._is_running():
                return

            self._start_process_locked()
            try:
                result = self._request_locked(
                    {
                        "action": "init",
                        "model_version": self.model_version,
                        "backend": self.backend,
                        "device": self.device,
                        "confidence": self.confidence,
                        "enable_interactive": self.enable_interactive,
                    },
                    timeout=self._TIMEOUT_INIT,
                )
            except Exception:
                self.close()
                raise

            self.model_name = str(result.get("model_name") or "SamGeo")
            self.supported_methods = set(result.get("supported_methods") or [])
            self._set_mask_count(result.get("mask_count", 0))

    def supports_method(self, method_name: str) -> bool:
        return method_name in self.supported_methods

    def set_image(self, source: str, bands=None):
        result = self._request(
            {"action": "set_image", "source": source, "bands": bands},
            timeout=self._TIMEOUT_SET_IMAGE,
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def generate_masks(self, prompt: str, min_size=None, max_size=None):
        result = self._request(
            {
                "action": "generate_masks",
                "prompt": prompt,
                "min_size": min_size,
                "max_size": max_size,
            }
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def generate_masks_by_points(
        self,
        point_coords,
        point_labels=None,
        point_crs=None,
        multimask_output=None,
    ):
        result = self._request(
            {
                "action": "generate_masks_by_points",
                "point_coords": self._json_payload(point_coords),
                "point_labels": self._json_payload(point_labels),
                "point_crs": point_crs,
                "multimask_output": multimask_output,
            }
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def predict(self, **kwargs):
        result = self._request(
            {
                "action": "predict",
                "kwargs": self._json_payload(kwargs),
            }
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def generate_masks_by_boxes(self, boxes):
        result = self._request(
            {"action": "generate_masks_by_boxes", "boxes": self._json_payload(boxes)}
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def generate_masks_by_points_patch(
        self,
        point_coords_batch,
        point_crs=None,
        output=None,
        unique=False,
        min_size=None,
        max_size=None,
    ):
        result = self._request(
            {
                "action": "generate_masks_by_points_patch",
                "point_coords_batch": self._json_payload(point_coords_batch),
                "point_crs": point_crs,
                "output": output,
                "unique": bool(unique),
                "min_size": min_size,
                "max_size": max_size,
            },
            timeout=max(self._TIMEOUT_REQUEST, 1800),
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def generate_masks_tiled(
        self,
        source: str,
        prompt: str,
        output: str,
        tile_size: int = 1024,
        overlap: int = 128,
        min_size: int = 0,
        max_size=None,
        unique: bool = True,
        bands=None,
    ):
        """Segment a large GeoTIFF tile by tile with a text prompt.

        The worker's ``SamGeo3.generate_masks_tiled`` writes the merged mask
        raster to ``output`` and keeps no masks in memory afterwards.

        Args:
            source: Path to the input GeoTIFF.
            prompt: Text prompt describing the objects.
            output: Path of the mask GeoTIFF to write.
            tile_size: Tile size in pixels.
            overlap: Overlap between tiles in pixels.
            min_size: Minimum mask size in pixels.
            max_size: Maximum mask size in pixels, or None for no limit.
            unique: Give each object its own raster value.
            bands: Optional 1-based RGB band indices.

        Returns:
            The worker result dict with ``output`` and ``object_count``.
        """
        result = self._request(
            {
                "action": "generate_masks_tiled",
                "source": source,
                "prompt": prompt,
                "output": output,
                "tile_size": int(tile_size),
                "overlap": int(overlap),
                "min_size": min_size,
                "max_size": max_size,
                "unique": bool(unique),
                "bands": self._json_payload(bands),
            },
            timeout=max(self._TIMEOUT_REQUEST, 6 * 3600),
        )
        self._set_mask_count(result.get("mask_count", 0))
        self.mask_scores = None
        return result

    def save_masks(self, output: str, unique: bool = False):
        """Save the current masks to a raster in the worker process.

        Also stores the per-mask confidence scores the worker reports in
        ``self.mask_scores`` (mask id -> SAM confidence), mirroring
        ``SamGeo3.mask_scores``; None when unavailable or ``unique`` is False.

        Args:
            output: Path of the mask raster to write.
            unique: Give each mask its own raster value.

        Returns:
            The worker result dict with ``mask_count`` and ``mask_scores``.
        """
        result = self._request(
            {
                "action": "save_masks",
                "output": output,
                "unique": bool(unique),
            },
            timeout=max(self._TIMEOUT_REQUEST, 1800),
        )
        self._set_mask_count(result.get("mask_count", 0))
        self.mask_scores = result.get("mask_scores") or None
        return result

    def close(self) -> None:
        with self._lock:
            self.masks = None
            super().close()

    def _set_mask_count(self, count: Any) -> None:
        count = int(count or 0)
        self.masks = _MaskCountProxy(count) if count > 0 else None

    def _json_payload(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._json_payload(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_payload(v) for k, v in value.items()}
        if hasattr(value, "tolist"):
            try:
                return self._json_payload(value.tolist())
            except Exception:
                pass
        if hasattr(value, "__fspath__"):
            try:
                return os.fspath(value)
            except Exception:
                pass
        return str(value)
