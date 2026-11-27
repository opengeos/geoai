"""Shared persistent subprocess client base for QGIS worker processes.

Provides a common JSON-lines IPC client for long-lived worker subprocesses
(SamGeo, DeepForest, Moondream) with consistent pipe management, stderr tail
capture, timeout cleanup, and a lock-free ``abort()`` method that unblocks
waiting worker threads without deadlocking on the request lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
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
    """Log a message to the QGIS message log."""
    QgsMessageLog.logMessage(message, "GeoAI", level=level)


class PersistentSubprocessClient:
    """Base class for stateful proxies backed by a persistent venv subprocess."""

    _WORKER_LABEL = "Worker"
    _TIMEOUT_INIT = 900
    _TIMEOUT_REQUEST = 600
    _STDERR_TAIL_CHARS = 3000
    _EMPTY_RESULT_IS_DICT = True

    def __init__(self, worker_script: Path):
        self.worker_script = Path(worker_script)
        self.process: Optional[subprocess.Popen] = None
        self._stderr_file = None
        self._lock = threading.RLock()

    def _module_attr(self, name: str, default: Any) -> Any:
        """Look up a module attribute on the subclass's module if present.

        This preserves test monkeypatches applied directly to subclass modules
        (for example ``monkeypatch.setattr(samgeo_subprocess, "venv_exists", ...)``).
        """
        mod = sys.modules.get(type(self).__module__)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
        return default

    def _prepare_env(self) -> dict:
        """Build the subprocess environment dictionary."""
        get_clean_env = self._module_attr(
            "_get_clean_env_for_venv", _get_clean_env_for_venv
        )
        return get_clean_env()

    def _start_process_locked(self) -> None:
        """Spawn the worker subprocess. Caller must hold ``self._lock``."""
        check_venv = self._module_attr("venv_exists", venv_exists)
        if not check_venv():
            raise RuntimeError(
                "GeoAI virtual environment not found. Install dependencies first."
            )

        get_python = self._module_attr("get_venv_python_path", get_venv_python_path)
        python_path = get_python()
        if not os.path.exists(python_path):
            raise FileNotFoundError(f"Venv Python not found: {python_path}")
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"{self._WORKER_LABEL} worker script not found: {self.worker_script}"
            )

        cmd = [python_path, "-u", str(self.worker_script)]
        env = self._prepare_env()
        get_kwargs = self._module_attr("_get_subprocess_kwargs", _get_subprocess_kwargs)
        popen_kwargs = dict(get_kwargs())
        subproc_mod = self._module_attr("subprocess", subprocess)

        try:
            self._stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        except Exception:
            self._stderr_file = None

        _log(
            f"Starting {self._WORKER_LABEL} worker subprocess: {python_path}",
            Qgis.Info,
        )
        self.process = subproc_mod.Popen(
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

    def initialize(self) -> None:
        """Start the worker subprocess if not running."""
        with self._lock:
            if self._is_running():
                return
            self._start_process_locked()

    def abort(self) -> None:
        """Terminate the worker subprocess immediately without acquiring ``_lock``.

        Safe to invoke from the UI thread while a background worker thread is
        blocked in ``_request()`` holding ``_lock``. Killing the child closes
        its stdout pipe so ``stdout.readline()`` returns EOF immediately,
        unblocking the worker thread without deadlocking.
        """
        proc = self.process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def close(self) -> None:
        """Gracefully shut down the worker process and release resources."""
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
                        proc.wait(timeout=2)
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
            try:
                return self._request_locked(payload, timeout=timeout)
            except TimeoutError:
                self.close()
                raise

    def _request_locked(self, payload: Dict[str, Any], timeout: Optional[int] = None):
        if not self._is_running():
            raise RuntimeError(f"{self._WORKER_LABEL} worker process is not running")

        proc = self.process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError(f"{self._WORKER_LABEL} worker pipes are not available")

        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to send request to {self._WORKER_LABEL} worker: {exc}"
            ) from exc

        effective_timeout = timeout or self._TIMEOUT_REQUEST
        # Total deadline caps overall runtime even when heartbeats keep arriving.
        deadline = time.monotonic() + effective_timeout * 3
        try:
            line = self._read_response_line(effective_timeout)
            while True:
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    line = self._read_response_line(effective_timeout)
                    continue
                if not isinstance(response, dict):
                    line = self._read_response_line(effective_timeout)
                    continue
                if response.get("type") == "heartbeat":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"{self._WORKER_LABEL} worker exceeded total deadline "
                            f"({effective_timeout * 3}s)"
                        )
                    next_timeout = min(effective_timeout, remaining)
                    line = self._read_response_line(int(max(next_timeout, 1)))
                    continue
                break
        except TimeoutError:
            self.close()
            raise

        if response.get("type") == "error":
            message = response.get(
                "message", f"Unknown {self._WORKER_LABEL} worker error"
            )
            stderr_output = self._read_stderr()
            if stderr_output:
                message = (
                    f"{message}\nWorker stderr:\n"
                    f"{stderr_output[-self._STDERR_TAIL_CHARS:]}"
                )
            raise RuntimeError(message)

        if response.get("type") != "ok":
            raise RuntimeError(
                f"Unexpected {self._WORKER_LABEL} worker response: {response}"
            )

        result = response.get("result")
        if self._EMPTY_RESULT_IS_DICT:
            return result or {}
        return result

    def _read_response_line(self, timeout_seconds: int) -> str:
        proc = self.process
        if proc is None or proc.poll() is not None:
            raise RuntimeError(f"{self._WORKER_LABEL} worker process is not running")
        if proc.stdout is None:
            raise RuntimeError(f"{self._WORKER_LABEL} worker stdout is not available")

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
            stderr_output = self._read_stderr()
            msg = (
                f"{self._WORKER_LABEL} worker did not respond within "
                f"{timeout_seconds}s"
            )
            if stderr_output:
                msg = (
                    f"{msg}\nWorker stderr:\n"
                    f"{stderr_output[-self._STDERR_TAIL_CHARS:]}"
                )
            raise TimeoutError(msg)
        if error[0] is not None:
            raise error[0]

        line = result[0]
        if not line:
            stderr_output = self._read_stderr()
            msg = f"{self._WORKER_LABEL} worker closed stdout unexpectedly"
            if stderr_output:
                msg = (
                    f"{msg}\nWorker stderr:\n"
                    f"{stderr_output[-self._STDERR_TAIL_CHARS:]}"
                )
            raise RuntimeError(msg)
        return line.strip()

    def _read_stderr(self, max_chars: Optional[int] = None) -> str:
        if self._stderr_file is None:
            return ""
        limit = self._STDERR_TAIL_CHARS if max_chars is None else max_chars
        try:
            self._stderr_file.seek(0)
            return self._stderr_file.read()[-limit:]
        except Exception:
            return ""

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
