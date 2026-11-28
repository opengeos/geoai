"""Unit tests for QGIS plugin architecture and reliability improvements (#1013)."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
import types
import zipfile
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest
from qgis.PyQt.QtCore import QThread
from qgis.PyQt.QtWidgets import QApplication

# ===========================================================================
# 1. venv_manager: no --proxy argv, strict-first TLS, cancel before CPU fallback
# ===========================================================================


def test_pip_proxy_args_returns_empty_and_proxy_stays_in_env(monkeypatch):
    from geoai.core import venv_manager

    proxy_url = "http://alice:s3cr3t@proxy.corp.local:8080"
    monkeypatch.setattr(venv_manager, "_get_qgis_proxy_settings", lambda: proxy_url)

    assert venv_manager._get_pip_proxy_args() == []

    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    env = venv_manager._get_clean_env_for_venv()
    assert env["HTTP_PROXY"] == proxy_url
    assert env["HTTPS_PROXY"] == proxy_url


def test_clean_env_enforces_strict_tls_by_default(monkeypatch):
    from geoai.core import venv_manager

    monkeypatch.setattr(venv_manager, "_get_qgis_proxy_settings", lambda: None)
    monkeypatch.delenv("UV_INSECURE_HOST", raising=False)
    monkeypatch.delenv("PIP_TRUSTED_HOST", raising=False)
    monkeypatch.delenv("UV_NATIVE_TLS", raising=False)

    env = venv_manager._get_clean_env_for_venv()

    assert env["UV_NATIVE_TLS"] == "true"
    assert "UV_INSECURE_HOST" not in env
    assert "PIP_TRUSTED_HOST" not in env


def test_pip_ssl_retry_only_runs_after_ssl_error_and_logs_warning(monkeypatch):
    from geoai.core import venv_manager

    calls = []
    warnings = []

    def fake_run_pip_install(**kwargs):
        calls.append((list(kwargs["cmd"]), dict(kwargs["env"])))
        return venv_manager._PipResult(0, "ok", "")

    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)
    monkeypatch.setattr(
        venv_manager, "_log", lambda msg, level=None: warnings.append(msg)
    )

    # Non-SSL error does not retry
    non_ssl = venv_manager._PipResult(1, "", "No matching distribution found")
    res_skip = venv_manager._retry_pip_install_with_insecure_hosts(
        result=non_ssl,
        cmd=["python", "-m", "pip", "install", "torch"],
        timeout=10,
        env={},
        subprocess_kwargs={},
        label="torch",
        progress_start=0,
        progress_end=10,
    )
    assert res_skip is non_ssl
    assert calls == []

    # SSL error retries with --trusted-host + PIP_TRUSTED_HOST and logs a warning
    ssl_err = venv_manager._PipResult(1, "", "CERTIFICATE_VERIFY_FAILED")
    res_ok = venv_manager._retry_pip_install_with_insecure_hosts(
        result=ssl_err,
        cmd=["python", "-m", "pip", "install", "torch"],
        timeout=10,
        env={},
        subprocess_kwargs={},
        label="torch",
        progress_start=0,
        progress_end=10,
    )
    assert res_ok.returncode == 0
    assert len(calls) == 1
    retry_cmd, retry_env = calls[0]
    assert "--trusted-host" in retry_cmd
    assert "pypi.org" in retry_env.get("PIP_TRUSTED_HOST", "").split()
    assert any("TLS" in w or "trusted-host" in w for w in warnings)


def test_uv_ssl_retry_sets_uv_insecure_host_in_retry_env_only(monkeypatch):
    from geoai.core import venv_manager

    captured_envs = []

    def fake_run_pip_install(**kwargs):
        captured_envs.append(dict(kwargs["env"]))
        return venv_manager._PipResult(0, "ok", "")

    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)

    base_env = {"UV_NATIVE_TLS": "true"}
    res = venv_manager._retry_uv_install_with_insecure_hosts(
        result=venv_manager._PipResult(1, "", "CERTIFICATE_VERIFY_FAILED"),
        cmd=["uv", "pip", "install", "--native-tls", "torch"],
        timeout=10,
        env=base_env,
        subprocess_kwargs={},
        label="torch",
        progress_start=0,
        progress_end=10,
    )

    assert res.returncode == 0
    assert "UV_INSECURE_HOST" not in base_env
    assert "pypi.org" in captured_envs[0].get("UV_INSECURE_HOST", "").split()


def test_pip_first_attempt_omits_trusted_host_and_proxy_argv(tmp_path, monkeypatch):
    from geoai.core import uv_manager, venv_manager

    venv_dir = tmp_path / "venv"
    py = venv_manager.get_venv_python_path(str(venv_dir))
    Path(py).parent.mkdir(parents=True, exist_ok=True)
    Path(py).write_text("", encoding="utf-8")

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: False)
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _d=None: True)
    monkeypatch.setattr(
        venv_manager,
        "_get_qgis_proxy_settings",
        lambda: "http://user:pass@proxy.local:8080",
    )
    monkeypatch.setattr(venv_manager, "detect_nvidia_gpu", lambda: (False, {}))
    monkeypatch.setattr(venv_manager, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(
        venv_manager.subprocess,
        "run",
        lambda cmd, *_a, **_kw: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr=""
        ),
    )

    seen_cmds = []
    seen_envs = []

    def fake_run_pip_install(**kwargs):
        seen_cmds.append(list(kwargs["cmd"]))
        seen_envs.append(dict(kwargs["env"]))
        return venv_manager._PipResult(0, "ok", "")

    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)

    ok, _msg = venv_manager.install_dependencies(
        venv_dir=str(venv_dir), cuda_enabled=False
    )

    assert ok is True
    assert seen_cmds
    for cmd in seen_cmds:
        assert "--proxy" not in cmd
        assert "--trusted-host" not in cmd
    for env in seen_envs:
        assert "PIP_TRUSTED_HOST" not in env
        assert "UV_INSECURE_HOST" not in env


def test_cancel_during_cuda_retry_prevents_cpu_fallback(tmp_path, monkeypatch):
    from geoai.core import uv_manager, venv_manager

    venv_dir = tmp_path / "venv"
    py = venv_manager.get_venv_python_path(str(venv_dir))
    Path(py).parent.mkdir(parents=True, exist_ok=True)
    Path(py).write_text("", encoding="utf-8")

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: True)
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "/usr/bin/uv")
    monkeypatch.setattr(venv_manager, "venv_exists", lambda _d=None: True)
    monkeypatch.setattr(venv_manager, "_get_qgis_proxy_settings", lambda: None)
    monkeypatch.setattr(venv_manager, "detect_nvidia_gpu", lambda: (True, {}))
    monkeypatch.setattr(venv_manager, "_select_cuda_index", lambda _g: "cu128")
    monkeypatch.setattr(venv_manager, "_is_cpu_torch_installed", lambda *_a: False)

    cancelled = {"value": False}
    labels_seen = []

    def fake_run_pip_install(**kwargs):
        labels_seen.append(kwargs["label"])
        # First CUDA attempt fails with a hash mismatch; user cancels during retry
        if len(labels_seen) == 1:
            return venv_manager._PipResult(
                1, "", "THESE PACKAGES DO NOT MATCH THE HASHES"
            )
        cancelled["value"] = True
        return venv_manager._PipResult(-1, "", "Installation cancelled")

    monkeypatch.setattr(venv_manager, "_run_pip_install", fake_run_pip_install)

    ok, msg = venv_manager.install_dependencies(
        venv_dir=str(venv_dir),
        cuda_enabled=True,
        cancel_check=lambda: cancelled["value"],
    )

    assert ok is False
    assert "cancelled" in msg.lower()
    assert not any("CPU fallback" in label for label in labels_seen)


# ===========================================================================
# 2. PersistentSubprocessClient + lock-free abort() + geoai_task cancel
# ===========================================================================


def test_persistent_subprocess_clients_inherit_from_shared_base():
    from geoai.core.deepforest_subprocess import DeepForestSubprocessClient
    from geoai.core.moondream_subprocess import MoondreamSubprocessClient
    from geoai.core.samgeo_subprocess import SamGeoSubprocessClient
    from geoai.core.subprocess_client import PersistentSubprocessClient

    assert issubclass(DeepForestSubprocessClient, PersistentSubprocessClient)
    assert issubclass(MoondreamSubprocessClient, PersistentSubprocessClient)
    assert issubclass(SamGeoSubprocessClient, PersistentSubprocessClient)


def test_persistent_subprocess_client_abort_unblocks_request_without_deadlock(tmp_path):
    from geoai.core.subprocess_client import (
        PersistentSubprocessClient,
        SubprocessAbortedError,
    )

    ready_marker = tmp_path / "request_received.flag"
    # Real child script that reads one request line, signals receipt, then sleeps.
    worker_script = tmp_path / "hanging_worker.py"
    worker_script.write_text(
        "import pathlib, sys, time\n"
        "line = sys.stdin.readline()\n"
        f"pathlib.Path({str(ready_marker)!r}).write_text(line, encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    client = PersistentSubprocessClient(worker_script)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    client.process = proc

    errors = []

    def _run_request():
        try:
            client._request({"action": "hang"}, timeout=30)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=_run_request)
    try:
        t.start()
        # Wait until the child confirms receipt of the in-flight request.
        deadline = time.monotonic() + 5.0
        while not ready_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready_marker.exists(), "child never received the request"

        t0 = time.monotonic()
        client.abort()
        t.join(timeout=5.0)
        elapsed = time.monotonic() - t0

        assert not t.is_alive(), "abort() failed to unblock worker thread"
        assert elapsed < 4.0
        assert errors and isinstance(errors[0], SubprocessAbortedError)
        assert isinstance(errors[0], RuntimeError)

        # Subsequent close() must acquire _lock cleanly without deadlocking.
        client.close()
        assert client.process is None
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        if t.is_alive():
            t.join(timeout=5.0)
        client.close()


def test_run_geoai_task_cooperative_cancel_terminates_worker(tmp_path, monkeypatch):
    from geoai.core import geoai_task_subprocess as gts

    sleeping_worker = tmp_path / "geoai_task_worker.py"
    sleeping_worker.write_text(
        "import sys, time\nsys.stdin.readline()\ntime.sleep(60)\n",
        encoding="utf-8",
    )

    # Point worker_script lookup at our sleeping script by patching Popen cmd
    real_popen = subprocess.Popen
    spawned = []

    def patched_popen(cmd, **kwargs):
        proc = real_popen([sys.executable, "-u", str(sleeping_worker)], **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(gts, "venv_exists", lambda: True)
    monkeypatch.setattr(gts, "get_venv_python_path", lambda: sys.executable)
    monkeypatch.setattr(gts, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(gts, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(gts.subprocess, "Popen", patched_popen)

    cancel_event = threading.Event()
    errors = []

    def _runner():
        try:
            gts.run_geoai_task("slow", {}, cancel_event=cancel_event)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=_runner)
    try:
        t.start()
        deadline = time.monotonic() + 3.0
        while not spawned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert spawned

        t0 = time.monotonic()
        cancel_event.set()
        t.join(timeout=5.0)
        elapsed = time.monotonic() - t0

        assert not t.is_alive()
        assert elapsed < 4.0
        assert spawned[0].poll() is not None
        assert errors and "cancelled" in str(errors[0]).lower()
    finally:
        cancel_event.set()
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        if t.is_alive():
            t.join(timeout=5.0)


# ===========================================================================
# 3. Dialog lifecycle safety, old-client closing, no QThread.terminate()
# ===========================================================================


def test_no_qthread_terminate_or_dead_output_capture_in_dialogs():
    dialogs_dir = Path(__file__).resolve().parents[1] / "geoai" / "dialogs"
    for name in (
        "change_detection.py",
        "deepforest_panel.py",
        "instance_segmentation.py",
        "moondream.py",
        "samgeo.py",
        "segmentation.py",
        "update_checker.py",
        "water_segmentation.py",
    ):
        src = (dialogs_dir / name).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=name)
        terminate_calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "terminate"
        ]
        assert (
            not terminate_calls
        ), f"{name} still calls .terminate() on line(s) {terminate_calls}"

    for name in (
        "water_segmentation.py",
        "segmentation.py",
        "instance_segmentation.py",
    ):
        src = (dialogs_dir / name).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=name)
        dead_classes = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "OutputCapture"
        ]
        assert not dead_classes, f"{name} still has dead OutputCapture"


def test_moondream_and_deepforest_close_previous_client_and_abort_on_cleanup(
    monkeypatch, tmp_path
):
    from geoai.dialogs import deepforest_panel, moondream

    # Moondream reload closes previous client
    old_md = MagicMock()
    new_md = MagicMock()
    md_stub = types.SimpleNamespace(
        moondream=old_md,
        model_combo=MagicMock(
            currentText=MagicMock(return_value="vikhyatk/moondream2")
        ),
        model_status=MagicMock(),
        progress_bar=MagicMock(),
        load_model_btn=MagicMock(),
        current_image_path=None,
        run_btn=MagicMock(),
    )
    moondream.MoondreamDockWidget.on_model_loaded(md_stub, new_md)
    old_md.close.assert_called_once()
    assert md_stub.moondream is new_md

    # Moondream cleanup calls abort() before waiting and then close()
    order = []
    client = types.SimpleNamespace(
        abort=lambda: order.append("abort"),
        close=lambda: order.append("close"),
    )
    worker = types.SimpleNamespace(
        isRunning=lambda: True,
        wait=lambda ms=0: (order.append(("wait", ms)) or True),
    )
    md_cleanup_stub = types.SimpleNamespace(
        moondream=client,
        worker=worker,
        model_load_worker=None,
    )
    moondream.MoondreamDockWidget.cleanup(md_cleanup_stub)
    assert order == ["abort", ("wait", 5000), "close"]
    assert md_cleanup_stub.moondream is None

    # DeepForest reload closes previous model
    old_df = MagicMock()
    new_df = MagicMock()
    df_stub = types.SimpleNamespace(
        deepforest=old_df,
        model_status=MagicMock(),
        progress_bar=MagicMock(),
        load_model_btn=MagicMock(),
        log_message=MagicMock(),
    )
    deepforest_panel.DeepForestDockWidget.on_model_loaded(df_stub, new_df, "tree")
    old_df.close.assert_called_once()
    assert df_stub.deepforest is new_df

    # DeepForest cleanup keeps active layer temp files and removes unused ones
    active_gpkg = tmp_path / "active.gpkg"
    stale_tif = tmp_path / "stale.tif"
    active_gpkg.write_bytes(b"gpkg")
    stale_tif.write_bytes(b"tif")

    active_layer = types.SimpleNamespace(
        source=lambda: f"{active_gpkg}|layername=trees"
    )
    monkeypatch.setattr(
        deepforest_panel,
        "QgsProject",
        types.SimpleNamespace(
            instance=lambda: types.SimpleNamespace(
                mapLayers=lambda: {"1": active_layer}
            )
        ),
    )

    df_order = []
    df_client = types.SimpleNamespace(
        abort=lambda: df_order.append("abort"),
        close=lambda: df_order.append("close"),
    )
    df_worker = types.SimpleNamespace(
        isRunning=lambda: True,
        wait=lambda ms=0: (df_order.append(("wait", ms)) or True),
    )
    df_cleanup_stub = types.SimpleNamespace(
        deepforest=df_client,
        model_load_worker=None,
        predict_worker=df_worker,
        _temp_files=[str(active_gpkg), str(stale_tif)],
        log_message=MagicMock(),
    )
    deepforest_panel.DeepForestDockWidget.cleanup(df_cleanup_stub)
    assert df_order == ["abort", ("wait", 5000), "close"]
    assert active_gpkg.exists(), "active map layer temp file must not be deleted"
    assert not stale_tif.exists(), "unused temp file should be cleaned up"


def test_samgeo_run_with_progress_rejects_overlap_and_closes_old_model():
    from geoai.dialogs import samgeo

    QApplication.instance() or QApplication([])

    # Close previous model on reload
    old_sam = MagicMock()
    new_sam = MagicMock()
    stub = types.SimpleNamespace(
        sam=old_sam,
        _set_tiled_result=MagicMock(),
        _image_state_stale=True,
        model_status=MagicMock(),
        progress_bar=MagicMock(),
        load_model_btn=MagicMock(),
        _update_tiled_support=MagicMock(),
        log_message=MagicMock(),
    )
    samgeo.SamGeoDockWidget.on_model_loaded(stub, new_sam, "SamGeo3")
    old_sam.close.assert_called_once()
    assert stub.sam is new_sam

    # Overlapping _run_with_progress call is rejected cleanly
    running_worker = types.SimpleNamespace(isRunning=lambda: True)
    errors = []
    dock_stub = types.SimpleNamespace(
        _operation_worker=running_worker,
        progress_bar=MagicMock(),
        log_message=MagicMock(),
        show_error=MagicMock(),
    )
    started = samgeo.SamGeoDockWidget._run_with_progress(
        dock_stub, lambda: 1, lambda _r: None, on_error=errors.append
    )
    assert started is False
    assert errors and "already in progress" in errors[0]


# ===========================================================================
# 4. QGIS 4.0 geometry/writer compat, DeepForest export, DepsInstall worker
# ===========================================================================


def test_qgis4_geometry_helpers_prefer_qgis_enum_with_wkb_fallback(monkeypatch):
    from geoai.dialogs import map_tools, samgeo

    # QGIS 4 style: Qgis.GeometryType.{Polygon, Point}
    fake_geom_type = types.SimpleNamespace(Polygon="QGIS4_POLY", Point="QGIS4_PT")
    monkeypatch.setattr(
        map_tools, "Qgis", types.SimpleNamespace(GeometryType=fake_geom_type)
    )
    monkeypatch.setattr(
        samgeo, "Qgis", types.SimpleNamespace(GeometryType=fake_geom_type)
    )
    assert map_tools._polygon_geometry_type() == "QGIS4_POLY"
    assert samgeo._point_geometry_type() == "QGIS4_PT"

    # Older QGIS 3 fallback: no GeometryType attribute on Qgis
    monkeypatch.setattr(map_tools, "Qgis", types.SimpleNamespace())
    monkeypatch.setattr(
        map_tools, "QgsWkbTypes", types.SimpleNamespace(PolygonGeometry="WKB_POLY")
    )
    monkeypatch.setattr(samgeo, "Qgis", types.SimpleNamespace())
    monkeypatch.setattr(
        samgeo, "QgsWkbTypes", types.SimpleNamespace(PointGeometry="WKB_PT")
    )
    assert map_tools._polygon_geometry_type() == "WKB_POLY"
    assert samgeo._point_geometry_type() == "WKB_PT"


def test_moondream_save_vector_prefers_write_as_vector_format_v3(monkeypatch):
    from geoai.dialogs import moondream

    v3_calls = []
    v1_calls = []

    class FakeOptions:
        driverName = ""
        fileEncoding = ""

    class FakeWriterV3:
        NoError = 0
        SaveVectorOptions = FakeOptions

        @staticmethod
        def writeAsVectorFormatV3(layer, path, ctx, options):
            v3_calls.append((path, ctx, options.driverName, options.fileEncoding))
            return (0, "")

        @staticmethod
        def writeAsVectorFormat(*args):
            v1_calls.append(args)
            return (0, "")

    fake_layer = types.SimpleNamespace(isValid=lambda: True, crs=lambda: "CRS")
    monkeypatch.setattr(moondream, "QgsVectorLayer", lambda *_a: fake_layer)
    monkeypatch.setattr(moondream, "QgsVectorFileWriter", FakeWriterV3)
    monkeypatch.setattr(
        moondream,
        "QgsProject",
        types.SimpleNamespace(
            instance=lambda: types.SimpleNamespace(transformContext=lambda: "CTX")
        ),
    )

    dock = types.SimpleNamespace()
    moondream.MoondreamDockWidget._save_vector_result(dock, "in.geojson", "out.gpkg")
    assert v3_calls == [("out.gpkg", "CTX", "GPKG", "UTF-8")]
    assert v1_calls == []

    # Fallback when writeAsVectorFormatV3 is absent (independent of QgsProject)
    class FakeWriterV1:
        NoError = 0

        @staticmethod
        def writeAsVectorFormat(layer, path, enc, crs, driver):
            v1_calls.append((path, driver, enc))
            return 0

    monkeypatch.setattr(moondream, "QgsProject", types.SimpleNamespace())
    monkeypatch.setattr(moondream, "QgsVectorFileWriter", FakeWriterV1)
    moondream.MoondreamDockWidget._save_vector_result(dock, "in.geojson", "out.shp")
    assert v1_calls == [("out.shp", "ESRI Shapefile", "UTF-8")]


def test_deepforest_worker_temp_rgb_uses_tempdir_for_readonly_input(
    tmp_path, monkeypatch
):
    from geoai.workers import deepforest_worker

    ro_dir = tmp_path / "readonly_input"
    ro_dir.mkdir()
    img_path = ro_dir / "rgba.tif"
    img_path.write_bytes(b"placeholder")

    written_paths = []

    class FakeSrc:
        count = 4
        profile = {"driver": "GTiff", "count": 4}

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self, band):
            return band

    class FakeDst:
        def __init__(self, path):
            written_paths.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def write(self, data, band):
            pass

    def fake_open(path, mode="r", **_kw):
        if mode == "w":
            return FakeDst(path)
        return FakeSrc()

    fake_rasterio = types.ModuleType("rasterio")
    fake_rasterio.open = fake_open
    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)

    os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        pred_path, temp_rgb = deepforest_worker._prepare_image_for_deepforest(
            str(img_path)
        )
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)

    try:
        assert temp_rgb is not None
        assert pred_path == temp_rgb
        assert Path(temp_rgb).parent != ro_dir
        assert not temp_rgb.endswith(".deepforest_rgb.tif")
        assert written_paths == [temp_rgb]
    finally:
        if temp_rgb and os.path.exists(temp_rgb):
            os.remove(temp_rgb)


def test_deepforest_save_as_raster_uses_uint32_at_65535_and_image_dims_helper(
    monkeypatch,
):
    from geoai.dialogs import deepforest_panel

    # Fake rasterio + rasterio.features
    captured = {}

    class FakeDataset:
        profile = {"driver": "GTiff", "count": 3, "dtype": "uint8"}
        transform = "TRANSFORM"
        width = 10
        height = 10

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def write(self, data, band):
            captured["written_band"] = band

    def fake_open(path, mode="r", **kw):
        if mode == "w":
            captured["profile_dtype"] = kw.get("dtype")
        return FakeDataset()

    def fake_rasterize(shapes, out_shape, transform, fill, dtype):
        captured["rasterize_dtype"] = dtype
        return [[0]]

    fake_rasterio = types.ModuleType("rasterio")
    fake_rasterio.uint16 = "uint16"
    fake_rasterio.uint32 = "uint32"
    fake_rasterio.open = fake_open
    fake_features = types.ModuleType("rasterio.features")
    fake_features.rasterize = fake_rasterize
    fake_shapely = types.ModuleType("shapely")
    fake_shapely_geom = types.ModuleType("shapely.geometry")
    fake_shapely_geom.box = lambda *args: "BOX"
    fake_gpd = types.ModuleType("geopandas")
    fake_np = types.ModuleType("numpy")
    fake_np.uint16 = "uint16"
    fake_np.uint32 = "uint32"
    fake_np.zeros = lambda shape, dtype=None: [[0]]
    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)
    monkeypatch.setitem(sys.modules, "rasterio.features", fake_features)
    monkeypatch.setitem(sys.modules, "shapely", fake_shapely)
    monkeypatch.setitem(sys.modules, "shapely.geometry", fake_shapely_geom)
    monkeypatch.setitem(sys.modules, "geopandas", fake_gpd)
    if "numpy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "numpy", fake_np)

    class FakePreds:
        def __init__(self, count):
            self._count = count

        def __len__(self):
            return self._count

        def iterrows(self):
            yield (
                0,
                types.SimpleNamespace(xmin=0, ymin=0, xmax=1, ymax=1),
            )

    dock = types.SimpleNamespace(
        current_image_path="in.tif",
        predictions=FakePreds(65535),
        _pixel_to_geo_box=lambda *_a: "POLY",
    )
    deepforest_panel.DeepForestDockWidget._save_as_raster(dock, "out.tif")
    assert captured["rasterize_dtype"] == "uint32"
    assert captured["profile_dtype"] == "uint32"

    # Below threshold stays uint16
    captured.clear()
    dock.predictions = FakePreds(100)
    deepforest_panel.DeepForestDockWidget._save_as_raster(dock, "out.tif")
    assert captured["rasterize_dtype"] == "uint16"
    assert captured["profile_dtype"] == "uint16"

    # _get_image_dimensions prefers rasterio, then falls back to PIL with MAX_IMAGE_PIXELS = None
    class FakeDimSrc:
        width = 4096
        height = 2048
        count = 4

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    fake_rasterio.open = lambda _p: FakeDimSrc()
    assert deepforest_panel._get_image_dimensions("img.tif") == (4096, 2048, 4)

    def broken_open(_p):
        raise RuntimeError("no rasterio")

    fake_rasterio.open = broken_open

    class FakePILImg:
        size = (1024, 512)

        def getbands(self):
            return ("R", "G", "B")

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    fake_pil_image = types.SimpleNamespace(
        MAX_IMAGE_PIXELS=12345, open=lambda _p: FakePILImg()
    )
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_pil_image
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    assert deepforest_panel._get_image_dimensions("img.tif") == (1024, 512, 3)
    assert fake_pil_image.MAX_IMAGE_PIXELS == 12345


def test_deps_install_dialog_runs_remove_venv_off_main_thread(monkeypatch):
    from geoai.core import venv_manager
    from geoai.dialogs import deps_install_dialog

    app = QApplication.instance() or QApplication([])
    main_tid = threading.get_ident()
    caller_tids = []

    def fake_remove_venv():
        caller_tids.append(threading.get_ident())
        return True, "removed"

    monkeypatch.setattr(venv_manager, "remove_venv", fake_remove_venv)
    monkeypatch.setattr(venv_manager, "detect_nvidia_gpu", lambda: (False, {}))

    dock = deps_install_dialog.DepsInstallDockWidget()
    emitted = []
    dock.install_requested.connect(lambda: emitted.append(True))
    try:
        dock._on_reinstall_clicked()
        deadline = time.monotonic() + 5.0
        while not emitted and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

        assert emitted == [True]
        assert caller_tids and caller_tids[0] != main_tid
    finally:
        dock.close()


# ===========================================================================
# 5. Runtime integrity (SHA-256, update_checker) & processing_monitor fallback
# ===========================================================================


@pytest.mark.parametrize("mod_name", ["python_manager", "uv_manager"])
def test_verify_sha256_accepts_match_and_deletes_mismatch(tmp_path, mod_name):
    import importlib

    mod = importlib.import_module(f"geoai.core.{mod_name}")
    payload = b"trusted archive bytes"
    digest = hashlib.sha256(payload).hexdigest()

    good = tmp_path / "good.tar.gz"
    good.write_bytes(payload)
    assert mod.verify_sha256(str(good), digest.upper()) is True
    assert good.exists()

    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        mod.verify_sha256(str(bad), digest)
    assert not bad.exists()


def test_download_uv_and_python_reject_known_sha256_mismatch(tmp_path, monkeypatch):
    from geoai.core import python_manager, uv_manager

    class FakeBytes:
        def __init__(self, data):
            self._data = data

        def __len__(self):
            return len(self._data)

        def data(self):
            return self._data

    class FakeReply:
        def content(self):
            return FakeBytes(b"tampered-payload")

    class FakeRequest:
        NoError = 0

        def get(self, _req):
            return self.NoError

        def reply(self):
            return FakeReply()

    wrong_digest = "0" * 64

    # python_manager
    monkeypatch.setattr(python_manager, "STANDALONE_DIR", str(tmp_path / "py"))
    monkeypatch.setattr(
        python_manager,
        "get_download_url",
        lambda: "https://example.test/cpython-test.tar.gz",
    )
    monkeypatch.setitem(
        python_manager.KNOWN_SHA256, "cpython-test.tar.gz", wrong_digest
    )
    monkeypatch.setattr(python_manager, "QgsBlockingNetworkRequest", FakeRequest)
    monkeypatch.setattr(python_manager, "QNetworkRequest", lambda u: u)
    monkeypatch.setattr(python_manager, "QUrl", lambda u: u)

    ok, msg = python_manager.download_python_standalone()
    assert ok is False
    assert "SHA-256 checksum mismatch" in msg

    # uv_manager
    monkeypatch.setattr(uv_manager, "UV_DIR", str(tmp_path / "uv"))
    monkeypatch.setattr(
        uv_manager,
        "get_uv_download_url",
        lambda: "https://example.test/uv-test.tar.gz",
    )
    monkeypatch.setitem(uv_manager.KNOWN_SHA256, "uv-test.tar.gz", wrong_digest)
    monkeypatch.setattr(uv_manager, "QgsBlockingNetworkRequest", FakeRequest)
    monkeypatch.setattr(uv_manager, "QNetworkRequest", lambda u: u)
    monkeypatch.setattr(uv_manager, "QUrl", lambda u: u)

    ok, msg = uv_manager.download_uv()
    assert ok is False
    assert "SHA-256 checksum mismatch" in msg


def test_update_checker_https_validation_tag_fallback_timeout_and_cancel(
    tmp_path, monkeypatch
):
    from geoai.dialogs import update_checker

    assert "urlretrieve" not in Path(update_checker.__file__).read_text(
        encoding="utf-8"
    )

    # Non-HTTPS URL rejected
    with pytest.raises(ValueError, match="non-HTTPS"):
        update_checker._validate_https_url("http://example.com/plugin.zip")
    with pytest.raises(ValueError, match="non-HTTPS"):
        update_checker._validate_https_url("file:///etc/passwd")

    # Tag URL preferred over branch archive
    urls = update_checker._get_archive_urls("1.2.3")
    assert urls[0].endswith("/archive/refs/tags/v1.2.3.zip")
    assert urls[1] == update_checker.ZIP_URL

    # Streaming download passes timeout=60 and falls back on 404
    seen_calls = []

    class FakeResp:
        headers = {"Content-Length": "5"}

        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, n=-1):
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def fake_urlopen(url, timeout=None):
        seen_calls.append((url, timeout))
        if "refs/tags/" in url:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return FakeResp(b"hello")

    monkeypatch.setattr(update_checker, "urlopen", fake_urlopen)
    dest = tmp_path / "out.zip"
    used = update_checker._download_archive_streaming(
        urls, str(dest), threading.Event()
    )
    assert used == update_checker.ZIP_URL
    assert dest.read_bytes() == b"hello"
    assert all(timeout == 60 for _, timeout in seen_calls)

    # Cooperative cancel before filesystem replace leaves existing plugin intact
    plugin_dir = tmp_path / "plugins" / "geoai"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "metadata.txt").write_text("version=1.0.0\n", encoding="utf-8")
    (plugin_dir / "geoai_plugin.py").write_text("# old\n", encoding="utf-8")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("geoai-v1.2.3/qgis_plugin/geoai/metadata.txt", "version=1.2.3\n")
        zf.writestr("geoai-v1.2.3/qgis_plugin/geoai/geoai_plugin.py", "# new\n")
    zip_bytes = zip_buf.getvalue()

    worker = update_checker.DownloadWorker(str(plugin_dir), version="1.2.3")
    hook_calls = []

    def fake_download_then_cancel(urls, dest_path, cancel_event, **_kw):
        hook_calls.append(True)
        Path(dest_path).write_bytes(zip_bytes)
        worker.cancel()
        return urls[0]

    monkeypatch.setattr(
        update_checker, "_download_archive_streaming", fake_download_then_cancel
    )
    worker.run()
    assert hook_calls == [True], "download hook never ran"
    assert worker._cancel_event.is_set(), "cancellation event was not set"
    assert (plugin_dir / "metadata.txt").read_text(
        encoding="utf-8"
    ) == "version=1.0.0\n"


def test_processing_monitor_gpu_metrics_falls_back_without_torch(monkeypatch):
    from geoai.core import processing_monitor, venv_manager

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        venv_manager,
        "detect_nvidia_gpu",
        lambda: (True, {"name": "NVIDIA RTX 4090", "memory_mb": 24564}),
    )

    metrics = processing_monitor.get_gpu_metrics()

    assert "torch" not in sys.modules
    assert metrics.available is True
    assert "NVIDIA RTX 4090" in (metrics.device_name or "")
    assert metrics.total_bytes == 24564 * 1024 * 1024


# ===========================================================================
# 6. geoai_task_worker returns num_instances & panel uses it without rasterio
# ===========================================================================


def test_task_instance_segmentation_returns_num_instances(tmp_path, monkeypatch):
    from geoai.workers import geoai_task_worker

    out_path = tmp_path / "instances.tif"
    fake_geoai = types.SimpleNamespace(
        instance_segmentation=lambda **kw: Path(kw["output_path"]).write_bytes(b"tif")
    )
    monkeypatch.setattr(geoai_task_worker, "_geoai", lambda: fake_geoai)
    monkeypatch.setattr(geoai_task_worker, "_emit", lambda _m: None)

    class FakeArray:
        size = 5

        def __iter__(self):
            return iter([0, 1, 1, 3, 0])

        def ravel(self):
            return [0, 1, 1, 3, 0]

        def __array__(self, dtype=None):
            import numpy as np

            return np.array([0, 1, 1, 3, 0], dtype=dtype)

    class FakeSrc:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self, _band):
            return FakeArray()

    fake_rasterio = types.ModuleType("rasterio")
    fake_rasterio.open = lambda _p: FakeSrc()
    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)

    result = geoai_task_worker._task_instance_segmentation(
        {"input_path": "in.tif", "output_path": str(out_path)}
    )
    # Distinct positive labels in [0, 1, 1, 3, 0] are {1, 3} -> 2 (not max 3).
    assert result == {"output_path": str(out_path), "num_instances": 2}


def test_instance_panel_uses_worker_num_instances_without_local_rasterio(monkeypatch):
    from geoai.dialogs import instance_segmentation

    class BlockRasterio:
        def find_spec(self, name, path=None, target=None):
            if name == "rasterio":
                raise ImportError("rasterio not installed in QGIS")
            return None

    monkeypatch.delitem(sys.modules, "rasterio", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockRasterio(), *sys.meta_path])

    warnings = []
    infos = []
    monkeypatch.setattr(
        instance_segmentation.QMessageBox,
        "warning",
        lambda _p, title, text: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        instance_segmentation.QMessageBox,
        "information",
        lambda _p, title, text: infos.append((title, text)),
    )

    dock = types.SimpleNamespace(
        inference_progress=MagicMock(),
        run_inference_btn=MagicMock(),
        vectorize_btn=MagicMock(),
        last_output_path=None,
        log=MagicMock(),
        add_to_map_check=types.SimpleNamespace(isChecked=lambda: False),
        inference_worker=types.SimpleNamespace(num_instances=5),
    )
    instance_segmentation.InstanceSegmentationDockWidget.on_inference_finished(
        dock, "out.tif"
    )
    assert warnings == []
    assert infos and "Detected 5 instances." in infos[0][1]

    # When num_instances is None and rasterio is missing, do not show a false 0-instance warning
    warnings.clear()
    infos.clear()
    dock.inference_worker = types.SimpleNamespace(num_instances=None)
    instance_segmentation.InstanceSegmentationDockWidget.on_inference_finished(
        dock, "out.tif"
    )
    assert warnings == []
    assert infos and "out.tif" in infos[0][1]


# ===========================================================================
# 7. PR review follow-up regression tests
# ===========================================================================


def test_subprocess_client_abort_before_popen_prevents_init_and_reaps_child(
    tmp_path, monkeypatch
):
    from geoai.core import subprocess_client
    from geoai.core.subprocess_client import (
        PersistentSubprocessClient,
        SubprocessAbortedError,
    )

    worker_script = tmp_path / "dummy_worker.py"
    worker_script.write_text("import time; time.sleep(60)\n", encoding="utf-8")
    client = PersistentSubprocessClient(worker_script)

    spawned = []
    real_popen = subprocess.Popen

    def slow_popen(cmd, **kwargs):
        proc = real_popen(cmd, **kwargs)
        spawned.append(proc)
        # Simulate UI calling abort() before Popen returns.
        client.abort()
        return proc

    monkeypatch.setattr(subprocess_client, "venv_exists", lambda: True)
    monkeypatch.setattr(
        subprocess_client, "get_venv_python_path", lambda: sys.executable
    )
    monkeypatch.setattr(subprocess_client, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(subprocess_client, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(subprocess_client.subprocess, "Popen", slow_popen)

    try:
        with pytest.raises(SubprocessAbortedError):
            client.initialize()
        assert client.process is None
        assert spawned and spawned[0].poll() is not None
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        client.close()


def test_subprocess_client_total_deadline_caps_malformed_json_and_stderr_is_bounded(
    tmp_path,
):
    from geoai.core.subprocess_client import PersistentSubprocessClient

    worker_script = tmp_path / "noisy_worker.py"
    worker_script.write_text(
        "import sys, time\n"
        "sys.stdin.readline()\n"
        "sys.stderr.write('X' * 10000 + 'TAIL_MARKER')\n"
        "sys.stderr.flush()\n"
        "while True:\n"
        "    print('not-json', flush=True)\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )

    client = PersistentSubprocessClient(worker_script)
    stderr_path = tmp_path / "stderr.log"
    stderr_handle = open(stderr_path, "ab")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
        bufsize=1,
    )
    client.process = proc
    client._stderr_file = stderr_handle
    client._stderr_path = str(stderr_path)
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            client._request({"action": "noisy"}, timeout=0.1)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0
        assert client._read_stderr(max_chars=50) == ""  # closed on timeout
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        client.close()


def test_keep_thread_alive_releases_cancelled_workers_via_native_qthread_finished():
    from geoai.dialogs import change_detection, segmentation, water_segmentation

    app = QApplication.instance() or QApplication([])

    class CancelledChangeWorker(change_detection.ChangeDetectionWorker):
        def run(self):
            # Exits without emitting custom finished or error signals.
            time.sleep(0.05)

    w1 = CancelledChangeWorker({})
    w1.start()
    change_detection._keep_thread_alive(w1)
    assert w1 in change_detection._LIVE_THREADS
    assert w1.wait(2000)
    for _ in range(50):
        app.processEvents()
        if w1 not in change_detection._LIVE_THREADS:
            break
    assert w1 not in change_detection._LIVE_THREADS

    class CancelledWaterWorker(water_segmentation.WaterSegmentationWorker):
        def run(self):
            time.sleep(0.05)

    w2 = CancelledWaterWorker(
        "in.tif",
        "RGB",
        None,
        None,
        1,
        None,
        "float32",
        0,
        256,
        32,
        False,
        False,
        False,
        0,
        False,
        1,
    )
    w2.start()
    water_segmentation._keep_thread_alive(w2)
    assert w2 in water_segmentation._LIVE_THREADS
    assert w2.wait(2000)
    for _ in range(50):
        app.processEvents()
        if w2 not in water_segmentation._LIVE_THREADS:
            break
    assert w2 not in water_segmentation._LIVE_THREADS

    class CancelledTrainWorker(segmentation.TrainingWorker):
        def run(self):
            time.sleep(0.05)

    w3 = CancelledTrainWorker(
        "imgs", "lbls", "out", "unet", "resnet34", "imagenet", 3, 2, 1, 1, 1e-3, 0.2
    )
    w3.start()
    segmentation._keep_thread_alive(w3)
    assert w3 in segmentation._LIVE_THREADS
    assert w3.wait(2000)
    for _ in range(50):
        app.processEvents()
        if w3 not in segmentation._LIVE_THREADS:
            break
    assert w3 not in segmentation._LIVE_THREADS


def test_dialog_cleanups_restore_idle_controls():
    from geoai.dialogs import (
        change_detection,
        instance_segmentation,
        segmentation,
        water_segmentation,
    )

    running_worker = types.SimpleNamespace(
        isRunning=lambda: True,
        cancel=MagicMock(),
        wait=lambda ms=0: True,
    )

    cd_dock = types.SimpleNamespace(
        worker=running_worker,
        progress_bar=MagicMock(),
        run_btn=MagicMock(),
        status_label=MagicMock(),
    )
    change_detection.ChangeDetectionDockWidget.cleanup(cd_dock)
    cd_dock.run_btn.setEnabled.assert_called_with(True)
    cd_dock.progress_bar.setVisible.assert_called_with(False)

    ws_dock = types.SimpleNamespace(
        worker=running_worker,
        progress_bar=MagicMock(),
        run_btn=MagicMock(),
        status_label=MagicMock(),
        _temp_files=[],
    )
    water_segmentation.WaterSegmentationDockWidget.cleanup(ws_dock)
    ws_dock.run_btn.setEnabled.assert_called_with(True)
    ws_dock.progress_bar.setVisible.assert_called_with(False)

    seg_dock = types.SimpleNamespace(
        train_worker=running_worker,
        tile_worker=None,
        inference_worker=None,
        vectorize_worker=None,
        smooth_worker=None,
        train_btn=MagicMock(),
        train_progress=MagicMock(),
        export_tiles_btn=MagicMock(),
        tile_progress=MagicMock(),
        run_inference_btn=MagicMock(),
        vectorize_btn=MagicMock(),
        inference_progress=MagicMock(),
        last_output_path=None,
        _cancel_monitored_worker=MagicMock(),
    )
    segmentation.SegmentationDockWidget.cleanup(seg_dock)
    seg_dock.train_btn.setEnabled.assert_called_with(True)
    seg_dock.train_progress.setRange.assert_called_with(0, 100)
    seg_dock.train_progress.setValue.assert_called_with(0)
    seg_dock.train_progress.setFormat.assert_called_with("")

    inst_dock = types.SimpleNamespace(
        train_worker=None,
        tile_worker=None,
        inference_worker=running_worker,
        vectorize_worker=None,
        smooth_worker=None,
        train_btn=MagicMock(),
        train_progress=MagicMock(),
        export_tiles_btn=MagicMock(),
        tile_progress=MagicMock(),
        run_inference_btn=MagicMock(),
        vectorize_btn=MagicMock(),
        inference_progress=MagicMock(),
        last_output_path=None,
    )
    instance_segmentation.InstanceSegmentationDockWidget.cleanup(inst_dock)
    inst_dock.run_inference_btn.setEnabled.assert_called_with(True)
    inst_dock.inference_progress.setRange.assert_called_with(0, 100)
    inst_dock.inference_progress.setValue.assert_called_with(0)


def test_deps_install_dialog_cleanup_detaches_running_worker(monkeypatch):
    from geoai.core import venv_manager
    from geoai.dialogs import deps_install_dialog

    app = QApplication.instance() or QApplication([])
    release = threading.Event()
    started = threading.Event()

    def slow_remove():
        started.set()
        release.wait(timeout=3.0)
        return True, "removed"

    monkeypatch.setattr(venv_manager, "remove_venv", slow_remove)
    monkeypatch.setattr(venv_manager, "detect_nvidia_gpu", lambda: (False, {}))

    dock = deps_install_dialog.DepsInstallDockWidget()
    emitted = []
    dock.install_requested.connect(lambda: emitted.append(True))
    try:
        dock._on_reinstall_clicked()
        assert started.wait(timeout=2.0)
        worker = dock._remove_worker
        assert worker is not None
        # Force wait() timeout to exercise retention path cleanly.
        monkeypatch.setattr(worker, "wait", lambda _ms=0: False)
        dock.cleanup()
        assert dock._remove_worker is None
        assert worker.parent() is None
        assert worker in deps_install_dialog._LIVE_THREADS
    finally:
        release.set()
        monkeypatch.undo()
        if worker.isRunning():
            worker.wait(2000)
        for _ in range(50):
            app.processEvents()
            if worker not in deps_install_dialog._LIVE_THREADS:
                break
        dock.close()

    assert worker not in deps_install_dialog._LIVE_THREADS
    assert emitted == []


def test_samgeo_overlapping_set_image_preserves_tiled_source(tmp_path):
    from geoai.dialogs import samgeo

    img_b = tmp_path / "b.tif"
    img_b.write_bytes(b"tif")
    orig_tiled = {"image": "a.tif", "source": "a.tif", "bands": None}
    dock = types.SimpleNamespace(
        sam=MagicMock(),
        _operation_worker=types.SimpleNamespace(isRunning=lambda: True),
        _tiled_source=orig_tiled,
        image_path_edit=types.SimpleNamespace(text=lambda: str(img_b)),
        log_message=MagicMock(),
        show_error=MagicMock(),
        _has_active_operation=lambda: True,
    )
    samgeo.SamGeoDockWidget.set_image_from_file(dock)
    assert dock._tiled_source is orig_tiled
    dock.show_error.assert_called_once()


def test_known_sha256_populated_and_verify_sha256_deduplicated():
    from geoai.core import python_manager, uv_manager

    assert uv_manager.verify_sha256 is python_manager.verify_sha256
    assert len(uv_manager.KNOWN_SHA256) == 5
    assert len(python_manager.KNOWN_SHA256) == 25
    assert all(len(v) == 64 for v in uv_manager.KNOWN_SHA256.values())
    assert all(len(v) == 64 for v in python_manager.KNOWN_SHA256.values())
