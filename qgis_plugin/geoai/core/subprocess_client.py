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


class SubprocessAbortedError(RuntimeError):
    """Raised when a persistent subprocess request or startup is aborted."""


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
        self._stderr_path: Optional[str] = None
        self._lock = threading.RLock()
        self._abort_lock = threading.Lock()
        self._aborted = False

    def reset_abort(self) -> None:
        """Clear cancellation state before explicitly starting a new session."""
        with self._abort_lock:
            self._aborted = False

    def _is_aborted(self) -> bool:
        with self._abort_lock:
            return self._aborted

    def _prepare_env(self) -> dict:
        """Build the subprocess environment dictionary."""
        return _get_clean_env_for_venv()

    def _start_process_locked(self) -> None:
        """Spawn the worker subprocess. Caller must hold ``self._lock``."""
        if self._is_aborted():
            raise SubprocessAbortedError(f"{self._WORKER_LABEL} worker start cancelled")

        if not venv_exists():
            raise RuntimeError(
                "GeoAI virtual environment not found. Install dependencies first."
            )

        python_path = get_venv_python_path()
        if not os.path.exists(python_path):
            raise FileNotFoundError(f"Venv Python not found: {python_path}")
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"{self._WORKER_LABEL} worker script not found: {self.worker_script}"
            )

        cmd = [python_path, "-u", str(self.worker_script)]
        env = self._prepare_env()
        popen_kwargs = dict(_get_subprocess_kwargs())

        self._close_stderr_file()
        try:
            stderr_handle = tempfile.NamedTemporaryFile(
                mode="ab",
                prefix="geoai_worker_",
                suffix=".stderr.log",
                delete=False,
            )
            self._stderr_file = stderr_handle
            self._stderr_path = stderr_handle.name
        except Exception:
            self._stderr_file = None
            self._stderr_path = None

        _log(
            f"Starting {self._WORKER_LABEL} worker subprocess: {python_path}",
            Qgis.Info,
        )
        try:
            proc = subprocess.Popen(
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
        except Exception:
            self._close_stderr_file()
            raise

        aborted_during_spawn = False
        with self._abort_lock:
            if self._aborted:
                aborted_during_spawn = True
            else:
                self.process = proc

        if aborted_during_spawn:
            self._terminate_process(proc)
            self._close_stderr_file()
            raise SubprocessAbortedError(f"{self._WORKER_LABEL} worker start cancelled")

    @staticmethod
    def _terminate_process(proc: subprocess.Popen, wait_timeout: float = 0.2) -> None:
        """Terminate and reap a child process with a short escalation timeout."""
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=wait_timeout)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=wait_timeout)
                    except Exception:
                        pass
            else:
                proc.wait(timeout=wait_timeout)
        except Exception:
            pass

    def initialize(self) -> None:
        """Start the worker subprocess if not running."""
        with self._lock:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker start cancelled"
                )
            if self._is_running():
                return
            self._start_process_locked()

    def abort(self) -> None:
        """Terminate the worker subprocess immediately without acquiring ``_lock``.

        Safe to invoke from the UI thread while a background worker thread is
        blocked in ``_request()`` holding ``_lock``. Recording ``_aborted``
        under ``_abort_lock`` ensures cancellation persists even if ``abort()``
        runs before ``Popen`` publishes ``self.process``.
        """
        with self._abort_lock:
            self._aborted = True
            # Reading self.process under _abort_lock into a local ref avoids a
            # race with close() clearing self.process.
            proc = self.process
        if proc is None:
            return
        self._terminate_process(proc, wait_timeout=0.2)

    def close(self) -> None:
        """Gracefully shut down the worker process and release resources."""
        with self._lock:
            with self._abort_lock:
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
        if self._stderr_path is not None:
            try:
                if os.path.exists(self._stderr_path):
                    os.remove(self._stderr_path)
            except OSError:
                pass
            self._stderr_path = None

    def _request(self, payload: Dict[str, Any], timeout: Optional[int] = None):
        with self._lock:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker request cancelled"
                )
            if not self._is_running():
                self.initialize()
            try:
                return self._request_locked(payload, timeout=timeout)
            except TimeoutError:
                self.close()
                raise

    def _request_locked(self, payload: Dict[str, Any], timeout: Optional[int] = None):
        if self._is_aborted():
            raise SubprocessAbortedError(
                f"{self._WORKER_LABEL} worker request cancelled"
            )
        if not self._is_running():
            raise RuntimeError(f"{self._WORKER_LABEL} worker process is not running")

        proc = self.process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError(f"{self._WORKER_LABEL} worker pipes are not available")

        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker request cancelled"
                ) from exc
            raise RuntimeError(
                f"Failed to send request to {self._WORKER_LABEL} worker: {exc}"
            ) from exc

        effective_timeout = timeout or self._TIMEOUT_REQUEST
        # Total deadline caps overall runtime across heartbeats and noisy output.
        deadline = time.monotonic() + effective_timeout * 3

        def _next_line() -> str:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self._WORKER_LABEL} worker exceeded total deadline "
                    f"({effective_timeout * 3}s)"
                )
            return self._read_response_line(min(float(effective_timeout), remaining))

        try:
            line = _next_line()
            while True:
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    line = _next_line()
                    continue
                if not isinstance(response, dict):
                    line = _next_line()
                    continue
                if response.get("type") == "heartbeat":
                    line = _next_line()
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

    def _read_response_line(self, timeout_seconds: float) -> str:
        proc = self.process
        if self._is_aborted():
            raise SubprocessAbortedError(
                f"{self._WORKER_LABEL} worker request cancelled"
            )
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
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker request cancelled"
                )
            stderr_output = self._read_stderr()
            msg = (
                f"{self._WORKER_LABEL} worker did not respond within "
                f"{timeout_seconds:g}s"
            )
            if stderr_output:
                msg = (
                    f"{msg}\nWorker stderr:\n"
                    f"{stderr_output[-self._STDERR_TAIL_CHARS:]}"
                )
            raise TimeoutError(msg)
        if error[0] is not None:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker request cancelled"
                ) from error[0]
            raise error[0]

        line = result[0]
        if not line:
            if self._is_aborted():
                raise SubprocessAbortedError(
                    f"{self._WORKER_LABEL} worker request cancelled"
                )
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
        """Read a bounded tail of the worker stderr log without moving child offset."""
        limit = self._STDERR_TAIL_CHARS if max_chars is None else max_chars
        if limit <= 0:
            return ""
        # Read via an independent read-only handle so the shared append-mode
        # descriptor inherited by the child is never seeked or perturbed.
        stderr_path = self._stderr_path
        if stderr_path is not None and os.path.exists(stderr_path):
            try:
                byte_limit = limit * 4
                with open(stderr_path, "rb") as reader:
                    reader.seek(0, os.SEEK_END)
                    size = reader.tell()
                    reader.seek(max(0, size - byte_limit), os.SEEK_SET)
                    data = reader.read(byte_limit)
                return data.decode("utf-8", errors="replace")[-limit:]
            except Exception:
                return ""
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - limit * 4), os.SEEK_SET)
            raw = self._stderr_file.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")[-limit:]
            return raw[-limit:]
        except Exception:
            return ""

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
