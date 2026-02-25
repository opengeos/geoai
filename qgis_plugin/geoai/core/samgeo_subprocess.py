"""Subprocess-backed SamGeo client for QGIS.

Runs SamGeo/SAM3 (torch-dependent) model loading and inference in the plugin
venv's Python process to avoid DLL conflicts with QGIS's embedded Python on
Windows.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from qgis.core import Qgis, QgsMessageLog

from .venv_manager import (
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    get_venv_python_path,
    venv_exists,
)


def _log(message: str, level=Qgis.Info) -> None:
    QgsMessageLog.logMessage(message, "GeoAI", level=level)


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


class SamGeoSubprocessClient:
    """Thin stateful proxy for SamGeo running in a persistent subprocess."""

    _TIMEOUT_INIT = 900
    _TIMEOUT_SET_IMAGE = 180
    _TIMEOUT_REQUEST = 600

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
        self.process: Optional[subprocess.Popen] = None
        self._stderr_file = None
        self._lock = threading.RLock()
        self.masks = None

        plugin_root = Path(__file__).resolve().parents[1]
        self.worker_script = plugin_root / "workers" / "samgeo_worker.py"

    def initialize(self) -> None:
        with self._lock:
            if self._is_running():
                return

            if not venv_exists():
                raise RuntimeError(
                    "GeoAI virtual environment not found. Install dependencies first."
                )

            python_path = get_venv_python_path()
            if not os.path.exists(python_path):
                raise FileNotFoundError(f"Venv Python not found: {python_path}")
            if not self.worker_script.exists():
                raise FileNotFoundError(
                    f"SamGeo worker script not found: {self.worker_script}"
                )

            cmd = [python_path, "-u", str(self.worker_script)]
            env = _get_clean_env_for_venv()
            popen_kwargs = dict(_get_subprocess_kwargs())

            try:
                self._stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            except Exception:
                self._stderr_file = None

            _log(f"Starting SamGeo worker subprocess: {python_path}", Qgis.Info)
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=(
                    self._stderr_file
                    if self._stderr_file is not None
                    else subprocess.DEVNULL
                ),
                text=True,
                bufsize=1,
                env=env,
                **popen_kwargs,
            )

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

    def save_masks(self, output: str, unique: bool = False):
        result = self._request(
            {
                "action": "save_masks",
                "output": output,
                "unique": bool(unique),
            },
            timeout=max(self._TIMEOUT_REQUEST, 1800),
        )
        self._set_mask_count(result.get("mask_count", 0))
        return result

    def close(self) -> None:
        with self._lock:
            proc = self.process
            self.process = None
            self.masks = None
            if proc is None:
                self._close_stderr_file()
                return

            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass

            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            self._close_stderr_file()

    def _set_mask_count(self, count: Any) -> None:
        count = int(count or 0)
        self.masks = _MaskCountProxy(count) if count > 0 else None

    def _close_stderr_file(self) -> None:
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def _request(self, payload: Dict[str, Any], timeout: Optional[int] = None):
        with self._lock:
            if not self._is_running():
                self.initialize()
            return self._request_locked(payload, timeout=timeout)

    def _request_locked(self, payload: Dict[str, Any], timeout: Optional[int] = None):
        if not self._is_running():
            raise RuntimeError("SamGeo worker process is not running")

        proc = self.process
        assert proc is not None
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("SamGeo worker pipes are not available")

        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to send request to SamGeo worker: {exc}"
            ) from exc

        line = self._read_response_line(timeout or self._TIMEOUT_REQUEST)
        while True:
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                line = self._read_response_line(timeout or self._TIMEOUT_REQUEST)

        if response.get("type") == "error":
            message = response.get("message", "Unknown SamGeo worker error")
            stderr_output = self._read_stderr()
            if stderr_output:
                message = f"{message}\nWorker stderr:\n{stderr_output[:2000]}"
            raise RuntimeError(message)

        if response.get("type") != "ok":
            raise RuntimeError(f"Unexpected SamGeo worker response: {response}")

        return response.get("result") or {}

    def _read_response_line(self, timeout_seconds: int) -> str:
        proc = self.process
        if proc is None or proc.poll() is not None:
            raise RuntimeError("SamGeo worker process is not running")
        if proc.stdout is None:
            raise RuntimeError("SamGeo worker stdout is not available")

        result = [None]
        error = [None]

        def _reader():
            try:
                result[0] = proc.stdout.readline()
            except Exception as exc:  # pragma: no cover - defensive
                error[0] = exc

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise TimeoutError(
                f"SamGeo worker did not respond within {timeout_seconds}s"
            )
        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            stderr_output = self._read_stderr()
            msg = "SamGeo worker closed stdout unexpectedly"
            if stderr_output:
                msg = f"{msg}\nWorker stderr:\n{stderr_output[:2000]}"
            raise RuntimeError(msg)
        return line.strip()

    def _read_stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.seek(0)
            return self._stderr_file.read()
        except Exception:
            return ""

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
