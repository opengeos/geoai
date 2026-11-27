"""Subprocess-backed Moondream client for QGIS.

Runs torch-dependent Moondream inference in the plugin venv's Python process
to avoid DLL conflicts with QGIS's embedded Python on Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .subprocess_client import PersistentSubprocessClient, SubprocessAbortedError

__all__ = ["MoondreamSubprocessClient"]


class MoondreamSubprocessClient(PersistentSubprocessClient):
    """Thin client that proxies Moondream calls to a persistent subprocess."""

    _WORKER_LABEL = "Moondream"
    _TIMEOUT_INIT = 900
    _TIMEOUT_LOAD_IMAGE = 300
    _TIMEOUT_REQUEST = 600
    _STDERR_TAIL_CHARS = 2000
    _EMPTY_RESULT_IS_DICT = False

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        plugin_root = Path(__file__).resolve().parents[1]
        super().__init__(plugin_root / "workers" / "moondream_worker.py")

    def initialize(self) -> None:
        """Start the worker process and load the requested model."""
        with self._lock:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker start cancelled"
                )
            if self._is_running():
                return

            self._start_process_locked()
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
        return self._request(payload, timeout=max(self._TIMEOUT_REQUEST, 1800))

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
        return self._request(payload, timeout=max(self._TIMEOUT_REQUEST, 1800))
