"""Subprocess-backed DeepForest client for QGIS.

Runs DeepForest (torch-dependent) model loading and prediction in the plugin
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


class DeepForestSubprocessClient:
    """Thin stateful proxy for DeepForest running in a persistent subprocess."""

    _TIMEOUT_INIT = 1200
    _TIMEOUT_REQUEST = 1800

    def __init__(self, model_name: str, revision: str, device: Optional[str]):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.process: Optional[subprocess.Popen] = None
        self._stderr_file = None
        self._lock = threading.RLock()

        plugin_root = Path(__file__).resolve().parents[1]
        self.worker_script = plugin_root / "workers" / "deepforest_worker.py"

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
                    f"DeepForest worker script not found: {self.worker_script}"
                )

            cmd = [python_path, "-u", str(self.worker_script)]
            env = _get_clean_env_for_venv()
            popen_kwargs = dict(_get_subprocess_kwargs())

            try:
                self._stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            except Exception:
                self._stderr_file = None

            _log(f"Starting DeepForest worker subprocess: {python_path}", Qgis.Info)
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
                        "model_name": self.model_name,
                        "revision": self.revision,
                        "device": self.device,
                    },
                    timeout=self._TIMEOUT_INIT,
                )
            except Exception:
                self.close()
                raise

            self.model_name = str(result.get("model_name") or self.model_name)

    def predict_subprocess(
        self,
        image_path: str,
        mode: str,
        patch_size: int,
        patch_overlap: float,
        iou_threshold: float,
        dataloader_strategy: str,
        score_threshold: float,
        batch_size: int,
    ) -> Dict[str, Any]:
        return self._request(
            {
                "action": "predict",
                "image_path": image_path,
                "mode": mode,
                "patch_size": int(patch_size),
                "patch_overlap": float(patch_overlap),
                "iou_threshold": float(iou_threshold),
                "dataloader_strategy": dataloader_strategy,
                "score_threshold": float(score_threshold),
                "batch_size": int(batch_size),
            },
            timeout=max(self._TIMEOUT_REQUEST, 3600),
        )

    def close(self) -> None:
        with self._lock:
            proc = self.process
            self.process = None
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
            raise RuntimeError("DeepForest worker process is not running")

        proc = self.process
        assert proc is not None
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("DeepForest worker pipes are not available")

        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to send request to DeepForest worker: {exc}"
            ) from exc

        line = self._read_response_line(timeout or self._TIMEOUT_REQUEST)
        while True:
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                line = self._read_response_line(timeout or self._TIMEOUT_REQUEST)

        if response.get("type") == "error":
            message = response.get("message", "Unknown DeepForest worker error")
            stderr_output = self._read_stderr()
            if stderr_output:
                message = f"{message}\nWorker stderr:\n{stderr_output[:3000]}"
            raise RuntimeError(message)

        if response.get("type") != "ok":
            raise RuntimeError(f"Unexpected DeepForest worker response: {response}")

        return response.get("result") or {}

    def _read_response_line(self, timeout_seconds: int) -> str:
        proc = self.process
        if proc is None or proc.poll() is not None:
            raise RuntimeError("DeepForest worker process is not running")
        if proc.stdout is None:
            raise RuntimeError("DeepForest worker stdout is not available")

        result = [None]
        error = [None]

        def _reader():
            try:
                result[0] = proc.stdout.readline()
            except Exception as exc:  # pragma: no cover
                error[0] = exc

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise TimeoutError(
                f"DeepForest worker did not respond within {timeout_seconds}s"
            )
        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            stderr_output = self._read_stderr()
            msg = "DeepForest worker closed stdout unexpectedly"
            if stderr_output:
                msg = f"{msg}\nWorker stderr:\n{stderr_output[:3000]}"
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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
