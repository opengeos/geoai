"""Subprocess-backed Moondream client for QGIS.

Runs torch-dependent Moondream inference in the plugin venv's Python process
to avoid DLL conflicts with QGIS's embedded Python on Windows.
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


class MoondreamSubprocessClient:
    """Thin client that proxies Moondream calls to a persistent subprocess."""

    _TIMEOUT_INIT = 600
    _TIMEOUT_LOAD_IMAGE = 120
    _TIMEOUT_REQUEST = 300

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.process: Optional[subprocess.Popen] = None
        self._stderr_file = None
        self._lock = threading.RLock()

        plugin_root = Path(__file__).resolve().parents[1]
        self.worker_script = plugin_root / "workers" / "moondream_worker.py"

    def initialize(self) -> None:
        """Start the worker process and load the requested model."""
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
                    f"Moondream worker script not found: {self.worker_script}"
                )

            cmd = [python_path, "-u", str(self.worker_script)]
            env = _get_clean_env_for_venv()
            popen_kwargs = dict(_get_subprocess_kwargs())

            try:
                self._stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            except Exception:
                self._stderr_file = None

            _log(
                f"Starting Moondream worker subprocess: {python_path}",
                Qgis.Info,
            )
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
                self._request_locked(
                    {
                        "action": "init",
                        "model_name": self.model_name,
                        "device": self.device,
                    },
                    timeout=self._TIMEOUT_INIT,
                )
            except Exception:
                self.close()
                raise

    def load_image(self, source: str, bands=None):
        return self._request(
            {
                "action": "load_image",
                "source": source,
                "bands": bands,
            },
            timeout=self._TIMEOUT_LOAD_IMAGE,
        )

    def caption(
        self,
        source,
        length: str = "normal",
        stream: bool = False,
        bands=None,
        settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        payload = {
            "action": "caption",
            "source": source,
            "length": length,
            "stream": bool(stream),
            "bands": bands,
            "settings": settings,
            "kwargs": kwargs,
        }
        return self._request(payload)

    def query(
        self,
        question: str,
        source=None,
        reasoning=None,
        stream: bool = False,
        bands=None,
        settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        payload = {
            "action": "query",
            "question": question,
            "source": source,
            "reasoning": reasoning,
            "stream": bool(stream),
            "bands": bands,
            "settings": settings,
            "kwargs": kwargs,
        }
        return self._request(payload)

    def detect(
        self,
        source,
        object_type: str,
        bands=None,
        output_path=None,
        settings: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ):
        payload = {
            "action": "detect",
            "source": source,
            "object_type": object_type,
            "bands": bands,
            "output_path": output_path,
            "settings": settings,
            "stream": bool(stream),
            "kwargs": kwargs,
        }
        return self._request(payload)

    def point(
        self,
        source,
        object_description: str,
        bands=None,
        output_path=None,
        **kwargs,
    ):
        payload = {
            "action": "point",
            "source": source,
            "object_description": object_description,
            "bands": bands,
            "output_path": output_path,
            "kwargs": kwargs,
        }
        return self._request(payload)

    def close(self) -> None:
        """Shut down the worker process."""
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
            raise RuntimeError("Moondream worker process is not running")

        proc = self.process
        assert proc is not None
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Moondream worker pipes are not available")

        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to send request to Moondream worker: {exc}"
            ) from exc

        line = self._read_response_line(
            timeout or self._TIMEOUT_REQUEST,
        )
        while True:
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                # Be tolerant of accidental stdout noise from third-party libs.
                line = self._read_response_line(timeout or self._TIMEOUT_REQUEST)

        if response.get("type") == "error":
            message = response.get("message", "Unknown Moondream worker error")
            stderr_output = self._read_stderr()
            if stderr_output:
                message = f"{message}\nWorker stderr:\n{stderr_output[:1000]}"
            raise RuntimeError(message)

        if response.get("type") != "ok":
            raise RuntimeError(f"Unexpected Moondream worker response: {response}")

        return response.get("result")

    def _read_response_line(self, timeout_seconds: int) -> str:
        proc = self.process
        if proc is None or proc.poll() is not None:
            raise RuntimeError("Moondream worker process is not running")
        if proc.stdout is None:
            raise RuntimeError("Moondream worker stdout is not available")

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
                f"Moondream worker did not respond within {timeout_seconds}s"
            )
        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            stderr_output = self._read_stderr()
            msg = "Moondream worker closed stdout unexpectedly"
            if stderr_output:
                msg = f"{msg}\nWorker stderr:\n{stderr_output[:1000]}"
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
