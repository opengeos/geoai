"""Run geoai tasks in a subprocess using the plugin-managed venv Python."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .venv_manager import (
    _get_clean_env_for_venv,
    _get_subprocess_kwargs,
    get_venv_python_path,
    venv_exists,
)

ProgressCallback = Optional[Callable[[str], None]]


def export_geotiff_tiles(
    params: Dict[str, Any], progress_callback: ProgressCallback = None
) -> None:
    """Export training tiles with ``geoai.export_geotiff_tiles``.

    Runs in the managed venv's subprocess when it exists, so the geoai library
    (which imports PyTorch) is never loaded into QGIS. Without a managed venv,
    geoai can only come from the interpreter running QGIS, so it is called
    in-process.

    Args:
        params: Keyword arguments for ``export_geotiff_tiles`` (JSON
            serializable).
        progress_callback: Optional callable receiving progress lines.

    Raises:
        RuntimeError: If the export fails or geoai is not available.
    """
    if venv_exists():
        run_geoai_task("export_geotiff_tiles", params, progress_callback)
        return

    from .._geoai_lib import get_geoai

    geoai = get_geoai()
    if geoai is None:
        raise RuntimeError("The geoai package is not available. Install dependencies.")
    geoai.export_geotiff_tiles(**params)


def run_geoai_task(
    action: str,
    params: Dict[str, Any],
    progress_callback: ProgressCallback = None,
    drop_env: Tuple[str, ...] = (),
) -> Any:
    """Run a geoai task in a subprocess and return its JSON-serializable result.

    Args:
        action: Name of the task in the worker's task table.
        params: JSON-serializable task parameters.
        progress_callback: Optional callable receiving progress lines.
        drop_env: Environment variable names to remove from the worker's
            environment only (the QGIS process environment is unchanged).

    Returns:
        The task's result.
    """
    if not venv_exists():
        raise RuntimeError("GeoAI virtual environment not found. Install dependencies.")

    python_path = get_venv_python_path()
    if not os.path.exists(python_path):
        raise FileNotFoundError(f"Venv Python not found: {python_path}")

    worker_script = (
        Path(__file__).resolve().parents[1] / "workers" / "geoai_task_worker.py"
    )
    if not worker_script.exists():
        raise FileNotFoundError(f"geoai task worker not found: {worker_script}")

    cmd = [python_path, "-u", str(worker_script)]
    env = _get_clean_env_for_venv()
    for name in drop_env:
        env.pop(name, None)
    popen_kwargs = dict(_get_subprocess_kwargs())

    stderr_file = None
    try:
        stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    except Exception:
        stderr_file = None

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file if stderr_file is not None else subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=env,
        **popen_kwargs,
    )

    try:
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Failed to create pipes for geoai task worker")

        proc.stdin.write(json.dumps({"action": action, "params": params}) + "\n")
        proc.stdin.flush()
        proc.stdin.close()

        final_result = None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                if progress_callback is not None:
                    progress_callback(line)
                continue

            msg_type = msg.get("type")
            if msg_type == "progress":
                text = msg.get("message", "")
                if text and progress_callback is not None:
                    progress_callback(text)
            elif msg_type == "ok":
                final_result = msg.get("result")
            elif msg_type == "error":
                error_msg = msg.get("message", "Unknown geoai task worker error")
                error_msg_lower = error_msg.lower()
                if "permission denied" in error_msg_lower:
                    raise PermissionError(error_msg)
                if (
                    "no such file or directory" in error_msg_lower
                    or "not found" in error_msg_lower
                ):
                    raise FileNotFoundError(error_msg)
                raise RuntimeError(error_msg)

        returncode = proc.wait()
        if returncode != 0:
            stderr_text = ""
            if stderr_file is not None:
                try:
                    stderr_file.seek(0)
                    stderr_text = stderr_file.read()
                except Exception:
                    pass
            if final_result is None:
                msg = f"geoai task worker exited with code {returncode}"
                if returncode == -9:
                    msg += (
                        " (SIGKILL). This usually means the OS killed the worker "
                        "process (often due to out-of-memory)."
                    )
                if stderr_text:
                    msg += f"\n{stderr_text[:2000]}"
                raise RuntimeError(msg)

        if final_result is None:
            stderr_text = ""
            if stderr_file is not None:
                try:
                    stderr_file.seek(0)
                    stderr_text = stderr_file.read()
                except Exception:
                    pass
            msg = "geoai task worker produced no result"
            if stderr_text:
                msg += f"\n{stderr_text[:2000]}"
            raise RuntimeError(msg)

        return final_result
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass
