"""Subprocess-backed DeepForest client for QGIS.

Runs DeepForest (torch-dependent) model loading and prediction in the plugin
venv's Python process to avoid DLL conflicts with QGIS's embedded Python on
Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .subprocess_client import PersistentSubprocessClient, SubprocessAbortedError

__all__ = ["DeepForestSubprocessClient"]


class DeepForestSubprocessClient(PersistentSubprocessClient):
    """Thin stateful proxy for DeepForest running in a persistent subprocess."""

    _WORKER_LABEL = "DeepForest"
    _TIMEOUT_INIT = 1200
    _TIMEOUT_REQUEST = 1800
    _STDERR_TAIL_CHARS = 3000
    _EMPTY_RESULT_IS_DICT = True

    def __init__(self, model_name: str, revision: str, device: Optional[str]):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        plugin_root = Path(__file__).resolve().parents[1]
        super().__init__(plugin_root / "workers" / "deepforest_worker.py")

    def initialize(self) -> None:
        with self._lock:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker start cancelled"
                )
            if self._is_running():
                return

            self._start_process_locked()
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
