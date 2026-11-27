"""Regression tests for verified QGIS plugin bugs."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qgis.PyQt.QtCore import Qt

# ===========================================================================
# 1 & 2. GeoAIPlugin: processing monitor toggle and dock cleanup on unload
# ===========================================================================


def test_toggle_processing_monitor_dock_uses_dock_visibility():
    from geoai.geoai_plugin import GeoAIPlugin

    plugin = GeoAIPlugin(MagicMock())
    dock = MagicMock()
    action = MagicMock()
    plugin._processing_monitor_dock = dock
    plugin.processing_monitor_action = action

    # When a checkable QAction is clicked to open a hidden dock, Qt sets
    # isChecked() to True before emitting triggered, while the dock is still
    # not visible. The toggle must inspect dock.isVisible(), not isChecked().
    dock.isVisible.return_value = False
    action.isChecked.return_value = True
    plugin.toggle_processing_monitor_dock()
    dock.show.assert_called_once()
    dock.raise_.assert_called_once()
    dock.hide.assert_not_called()

    dock.reset_mock()
    dock.isVisible.return_value = True
    action.isChecked.return_value = False
    plugin.toggle_processing_monitor_dock()
    dock.hide.assert_called_once()
    dock.show.assert_not_called()


def test_unload_calls_cleanup_on_all_dock_widgets():
    from geoai.geoai_plugin import GeoAIPlugin

    iface = MagicMock()
    plugin = GeoAIPlugin(iface)

    dock_attrs = [
        "_deps_dock",
        "_moondream_dock",
        "_segmentation_dock",
        "_samgeo_dock",
        "_deepforest_dock",
        "_water_segmentation_dock",
        "_change_detection_dock",
        "_instance_segmentation_dock",
        "_processing_monitor_dock",
    ]
    cleaned = []

    class DockWithCleanup:
        def __init__(self, name):
            self.name = name
            self.deleted = False

        def cleanup(self):
            cleaned.append(self.name)

        def deleteLater(self):
            self.deleted = True

    class DockWithoutCleanup:
        def __init__(self):
            self.deleted = False

        def deleteLater(self):
            self.deleted = True

    docks = {}
    for idx, attr in enumerate(dock_attrs):
        dock = DockWithoutCleanup() if idx == 0 else DockWithCleanup(attr)
        setattr(plugin, attr, dock)
        docks[attr] = dock

    plugin.unload()

    assert cleaned == dock_attrs[1:]
    for attr, dock in docks.items():
        assert getattr(plugin, attr) is None
        assert dock.deleted is True
        iface.removeDockWidget.assert_any_call(dock)


# ===========================================================================
# 3. samgeo_worker._ensure_pkg_resources_shim does not mutate sys.path or
#    import qgis
# ===========================================================================


def test_samgeo_worker_pkg_resources_shim_without_sys_path_mutation(monkeypatch):
    from geoai.workers import samgeo_worker

    plugin_parent = str(Path(samgeo_worker.__file__).resolve().parents[2])

    class BlockPkgResourcesAndQgis:
        def find_spec(self, name, path=None, target=None):
            if name == "pkg_resources" or name == "qgis" or name.startswith("qgis."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    clean_path = [
        p for p in sys.path if os.path.realpath(p) != os.path.realpath(plugin_parent)
    ]
    monkeypatch.setattr(sys, "path", list(clean_path))
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    for key in [k for k in sys.modules if k == "qgis" or k.startswith("qgis.")]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    for key in [k for k in sys.modules if k == "geoai" or k.startswith("geoai.")]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockPkgResourcesAndQgis(), *sys.meta_path])

    installed = samgeo_worker._ensure_pkg_resources_shim()

    assert installed is True
    assert "pkg_resources" in sys.modules
    assert sys.path == clean_path
    assert plugin_parent not in sys.path


# ===========================================================================
# 4. deepforest_worker preserves explicit 0.0 float thresholds
# ===========================================================================


def test_deepforest_worker_preserves_zero_thresholds(monkeypatch):
    from geoai.workers import deepforest_worker

    class FakeFrame:
        columns = ["score"]
        empty = False

        def __init__(self, scores):
            self.scores = list(scores)
            self.filtered = False

        @property
        def score(self):
            return self

        def __ge__(self, threshold):
            self.filtered = True
            return FakeFrame([s for s in self.scores if s >= threshold])

        def __getitem__(self, mask):
            return mask

    fake_model = types.SimpleNamespace(
        config={},
        create_trainer=lambda **_kw: None,
    )
    captured_tile_kwargs = {}
    frame = FakeFrame([0.05, 0.5])

    def fake_predict_tile(_model, _path, **kwargs):
        captured_tile_kwargs.update(kwargs)
        return frame

    monkeypatch.setattr(deepforest_worker, "_configure_process_env", lambda: None)
    monkeypatch.setattr(deepforest_worker, "_get_model", lambda: fake_model)
    monkeypatch.setattr(
        deepforest_worker, "_prepare_image_for_deepforest", lambda p: (p, None)
    )
    monkeypatch.setattr(
        deepforest_worker, "_predict_tile_with_oom_retry", fake_predict_tile
    )
    monkeypatch.setattr(
        deepforest_worker, "_serialize_predictions", lambda df: list(df.scores)
    )

    out = deepforest_worker._handle_predict(
        {
            "image_path": "img.tif",
            "mode": "Large Tile",
            "patch_size": 400,
            "patch_overlap": 0.0,
            "iou_threshold": 0.0,
            "score_threshold": 0.0,
            "batch_size": 2,
        }
    )

    assert captured_tile_kwargs["patch_overlap"] == 0.0
    assert captured_tile_kwargs["iou_threshold"] == 0.0
    assert out["predictions"] == [0.05, 0.5]
    assert frame.filtered is False

    # Omitting them (or passing None) still applies the documented defaults.
    captured_tile_kwargs.clear()
    frame2 = FakeFrame([0.05, 0.5])
    monkeypatch.setattr(
        deepforest_worker,
        "_predict_tile_with_oom_retry",
        lambda _m, _p, **kw: (captured_tile_kwargs.update(kw) or frame2),
    )
    out_default = deepforest_worker._handle_predict(
        {"image_path": "img.tif", "mode": "Large Tile"}
    )
    assert captured_tile_kwargs["patch_overlap"] == 0.25
    assert captured_tile_kwargs["iou_threshold"] == 0.15
    assert out_default["predictions"] == [0.5]


# ===========================================================================
# 5. deepforest_panel._get_gpu_memory_mb uses props.total_memory
# ===========================================================================


def test_deepforest_gpu_memory_uses_total_memory(monkeypatch):
    from geoai.dialogs import deepforest_panel

    fake_props = types.SimpleNamespace(total_memory=8 * 1024 * 1024 * 1024)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _idx: fake_props,
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)

    monkeypatch.setattr(deepforest_panel, "loaded_torch", lambda: fake_torch)
    monkeypatch.setattr(
        "geoai.core.venv_manager.detect_nvidia_gpu",
        lambda: (False, {}),
    )

    assert deepforest_panel._get_gpu_memory_mb() == 8192


# ===========================================================================
# 6. samgeo._on_custom_bands_changed handles PyQt6 int state and CheckState enum
# ===========================================================================


def test_samgeo_on_custom_bands_changed_accepts_int_and_enum():
    from geoai.dialogs.samgeo import SamGeoDockWidget

    class FakeSpin:
        def __init__(self):
            self.enabled = None

        def setEnabled(self, value):
            self.enabled = value

    stub = types.SimpleNamespace(
        red_band_spin=FakeSpin(),
        green_band_spin=FakeSpin(),
        blue_band_spin=FakeSpin(),
    )

    # PyQt6 stateChanged emits int 2 for Checked and 0 for Unchecked.
    SamGeoDockWidget._on_custom_bands_changed(stub, 2)
    assert [
        stub.red_band_spin.enabled,
        stub.green_band_spin.enabled,
        stub.blue_band_spin.enabled,
    ] == [True, True, True]

    SamGeoDockWidget._on_custom_bands_changed(stub, 0)
    assert [
        stub.red_band_spin.enabled,
        stub.green_band_spin.enabled,
        stub.blue_band_spin.enabled,
    ] == [False, False, False]

    SamGeoDockWidget._on_custom_bands_changed(stub, Qt.CheckState.Checked)
    assert [
        stub.red_band_spin.enabled,
        stub.green_band_spin.enabled,
        stub.blue_band_spin.enabled,
    ] == [True, True, True]

    SamGeoDockWidget._on_custom_bands_changed(stub, Qt.CheckState.Unchecked)
    assert [
        stub.red_band_spin.enabled,
        stub.green_band_spin.enabled,
        stub.blue_band_spin.enabled,
    ] == [False, False, False]


# ===========================================================================
# 7. python_manager: safe tar link validation on <3.12 and cleanup on failure
# ===========================================================================


def _make_tar_bytes(entries) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for entry in entries:
            kind = entry[0]
            if kind == "file":
                _, name, data = entry
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "sym":
                _, name, target = entry
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)
            elif kind == "lnk":
                _, name, target = entry
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.LNKTYPE
                info.linkname = target
                tar.addfile(info)
    return buf.getvalue()


def test_safe_extract_tar_allows_safe_links_and_blocks_escapes_pre_312(
    monkeypatch, tmp_path
):
    from geoai.core import python_manager

    monkeypatch.setattr(python_manager.sys, "version_info", (3, 11, 9))

    good_tar = _make_tar_bytes(
        [
            ("file", "python/bin/python3.11", b"#!/bin/sh\n"),
            ("sym", "python/bin/python3", "python3.11"),
            ("lnk", "python/bin/python", "python/bin/python3.11"),
        ]
    )
    dest = tmp_path / "good"
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(good_tar), mode="r:gz") as tar:
        python_manager._safe_extract_tar(tar, str(dest))

    assert (dest / "python" / "bin" / "python3.11").is_file()
    assert (dest / "python" / "bin" / "python3").is_symlink()
    assert (dest / "python" / "bin" / "python").is_file()

    # Escaping relative symlink must be rejected.
    bad_rel_sym = _make_tar_bytes([("sym", "python/bin/bad", "../../../../etc/passwd")])
    bad_dest1 = tmp_path / "bad1"
    bad_dest1.mkdir()
    with tarfile.open(fileobj=io.BytesIO(bad_rel_sym), mode="r:gz") as tar:
        with pytest.raises(ValueError):
            python_manager._safe_extract_tar(tar, str(bad_dest1))

    # Absolute symlink must be rejected.
    bad_abs_sym = _make_tar_bytes([("sym", "python/bin/bad", "/etc/passwd")])
    bad_dest2 = tmp_path / "bad2"
    bad_dest2.mkdir()
    with tarfile.open(fileobj=io.BytesIO(bad_abs_sym), mode="r:gz") as tar:
        with pytest.raises(ValueError):
            python_manager._safe_extract_tar(tar, str(bad_dest2))

    # Escaping hardlink must be rejected.
    bad_lnk = _make_tar_bytes([("lnk", "python/bin/bad", "../../outside.txt")])
    bad_dest3 = tmp_path / "bad3"
    bad_dest3.mkdir()
    with tarfile.open(fileobj=io.BytesIO(bad_lnk), mode="r:gz") as tar:
        with pytest.raises(ValueError):
            python_manager._safe_extract_tar(tar, str(bad_dest3))


def test_download_standalone_python_cleans_up_on_verify_or_extract_failure(
    monkeypatch, tmp_path
):
    from geoai.core import python_manager

    standalone_dir = tmp_path / "python_standalone"
    monkeypatch.setattr(python_manager, "STANDALONE_DIR", str(standalone_dir))
    monkeypatch.setattr(
        python_manager,
        "get_download_url",
        lambda: ("https://example.test/py.tar.gz", "3.12.8"),
    )

    payload = _make_tar_bytes([("file", "python/bin/python3", b"not-a-binary")])

    class FakeBytes:
        def __init__(self, data):
            self._data = data

        def __len__(self):
            return len(self._data)

        def data(self):
            return self._data

    class FakeReply:
        def content(self):
            return FakeBytes(payload)

    class FakeRequest:
        NoError = 0

        def get(self, _req):
            return self.NoError

        def reply(self):
            return FakeReply()

    monkeypatch.setattr(python_manager, "QgsBlockingNetworkRequest", FakeRequest)
    monkeypatch.setattr(python_manager, "QNetworkRequest", lambda url: url)
    monkeypatch.setattr(python_manager, "QUrl", lambda url: url)

    # Case 1: verify_standalone_python fails after extraction.
    monkeypatch.setattr(
        python_manager, "verify_standalone_python", lambda: (False, "broken binary")
    )
    ok, msg = python_manager.download_python_standalone()
    assert ok is False
    assert "Verification failed" in msg
    assert not standalone_dir.exists()
    assert python_manager.standalone_python_exists() is False

    # Case 2: extraction raises after writing partial files.
    def partial_failing_extract(_tar, dest_dir):
        partial = Path(dest_dir) / "python" / "bin"
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "python3").write_bytes(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(python_manager, "_safe_extract_tar", partial_failing_extract)
    ok, msg = python_manager.download_python_standalone()
    assert ok is False
    assert "disk full" in msg
    assert not standalone_dir.exists()
    assert python_manager.standalone_python_exists() is False


# ===========================================================================
# 8. geoai_task_worker temp dir compatibility on Python 3.9
# ===========================================================================


def test_geoai_task_worker_detect_changes_python39_tempdir(monkeypatch, tmp_path):
    from geoai.workers import geoai_task_worker

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch_checkpoints"))

    class Py39TemporaryDirectory:
        def __init__(self, *args, **kwargs):
            if "ignore_cleanup_errors" in kwargs:
                raise TypeError(
                    "TemporaryDirectory.__init__() got an unexpected keyword "
                    "argument 'ignore_cleanup_errors'"
                )
            self._real = tempfile.TemporaryDirectory(*args, **kwargs)

        def __enter__(self):
            return self._real.__enter__()

        def __exit__(self, *exc):
            return self._real.__exit__(*exc)

    monkeypatch.setattr(
        geoai_task_worker.tempfile, "TemporaryDirectory", Py39TemporaryDirectory
    )

    created_dirs = []

    class FakeDetector:
        def __init__(self, sam_model_type):
            self.sam_model_type = sam_model_type

        def set_hyperparameters(self, **_kw):
            pass

        def detect_changes(self, _b, _a, **kwargs):
            inst = kwargs["instance_masks_output_path"]
            created_dirs.append(os.path.dirname(inst))
            Path(inst).write_bytes(b"mask")

    change_mod = types.ModuleType("geoai.change_detection")
    change_mod.ChangeDetection = FakeDetector
    raster_mod = types.ModuleType("geoai.utils.raster")
    raster_mod.raster_to_vector = lambda _path, **_kw: None
    monkeypatch.setitem(sys.modules, "geoai.change_detection", change_mod)
    monkeypatch.setitem(sys.modules, "geoai.utils.raster", raster_mod)
    monkeypatch.setattr(geoai_task_worker, "_emit", lambda _msg: None)

    result = geoai_task_worker._task_detect_changes(
        {
            "before_path": "b.tif",
            "after_path": "a.tif",
            "output_raster": str(tmp_path / "out.tif"),
            "output_vector": str(tmp_path / "out.gpkg"),
            "sam_model_type": "vit_b",
            "change_confidence_threshold": 150,
            "auto_threshold": False,
            "min_area": 0,
        }
    )

    assert result["output_vector"] == str(tmp_path / "out.gpkg")
    assert len(created_dirs) == 1
    assert not os.path.exists(created_dirs[0])


# ===========================================================================
# 9. venv_manager._get_linux_system_python refuses /usr/bin/qgis.bin
# ===========================================================================


def test_linux_system_python_skips_non_python_sys_executable(monkeypatch, tmp_path):
    from geoai.core import venv_manager

    fake_qgis = tmp_path / "qgis.bin"
    fake_qgis.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(venv_manager.sys, "platform", "linux")
    monkeypatch.setattr(venv_manager.shutil, "which", lambda _name: None)
    monkeypatch.setattr(venv_manager.sys, "executable", str(fake_qgis))

    ran = []
    monkeypatch.setattr(
        venv_manager.subprocess,
        "run",
        lambda cmd, **_kw: (
            ran.append(cmd)
            or subprocess.CompletedProcess(cmd, 0, stdout="3 12 0", stderr="")
        ),
    )

    assert venv_manager._get_linux_system_python() is None
    assert ran == []


# ===========================================================================
# 10. browse_labels_dir opens JSON file dialog for COCO format
# ===========================================================================


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("geoai.dialogs.segmentation", "SegmentationDockWidget"),
        ("geoai.dialogs.instance_segmentation", "InstanceSegmentationDockWidget"),
    ],
)
def test_browse_labels_dir_uses_file_dialog_for_coco(
    monkeypatch, module_name, class_name
):
    import importlib

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    calls = []
    monkeypatch.setattr(
        mod.QFileDialog,
        "getOpenFileName",
        lambda _parent, title, _dir, filt: (
            calls.append(("file", title, filt)) or ("/data/annotations.json", filt)
        ),
    )
    monkeypatch.setattr(
        mod.QFileDialog,
        "getExistingDirectory",
        lambda _parent, title: (calls.append(("dir", title)) or "/data/labels"),
    )

    class FakeCombo:
        def __init__(self, text):
            self._text = text

        def currentText(self):
            return self._text

    class FakeEdit:
        def __init__(self):
            self.value = ""

        def setText(self, text):
            self.value = text

    # COCO mode -> open JSON file dialog
    stub_coco = types.SimpleNamespace(
        input_format_combo=FakeCombo("coco"),
        labels_dir_edit=FakeEdit(),
    )
    cls.browse_labels_dir(stub_coco)
    assert stub_coco.labels_dir_edit.value == "/data/annotations.json"
    assert calls == [
        ("file", "Select COCO Annotations File", "JSON Files (*.json);;All Files (*)")
    ]

    # Directory mode -> open directory dialog
    calls.clear()
    stub_dir = types.SimpleNamespace(
        input_format_combo=FakeCombo("directory"),
        labels_dir_edit=FakeEdit(),
    )
    cls.browse_labels_dir(stub_dir)
    assert stub_dir.labels_dir_edit.value == "/data/labels"
    assert calls == [("dir", "Select Labels Directory")]


# ===========================================================================
# 11. Subprocess clients: non-dict JSON noise, reap after kill, stderr tail,
#     close on TimeoutError
# ===========================================================================


class _FakePopen:
    def __init__(self, stdout_lines=None, poll_code=None, block_stdout=False):
        self.stdin = io.StringIO()
        self._stdout_lines = list(stdout_lines or [])
        self._poll_code = poll_code
        self._block_stdout = block_stdout
        self.events = []
        self.stdout = self._build_stdout()

    def _build_stdout(self):
        proc = self

        class _Stdout:
            def readline(self):
                if proc._block_stdout:
                    time.sleep(2)
                    return ""
                if proc._stdout_lines:
                    return proc._stdout_lines.pop(0)
                return ""

            def __iter__(self):
                return self

            def __next__(self):
                line = self.readline()
                if not line:
                    raise StopIteration
                return line

        return _Stdout()

    def poll(self):
        return self._poll_code

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        if self._poll_code is None:
            return 0
        return self._poll_code

    def terminate(self):
        self.events.append(("terminate",))

    def kill(self):
        self.events.append(("kill",))
        self._poll_code = -9


def _make_persistent_clients():
    from geoai.core.deepforest_subprocess import DeepForestSubprocessClient
    from geoai.core.moondream_subprocess import MoondreamSubprocessClient
    from geoai.core.samgeo_subprocess import SamGeoSubprocessClient

    return [
        DeepForestSubprocessClient("model", "main", "cpu"),
        MoondreamSubprocessClient("vikhyatk/moondream2", "cpu"),
        SamGeoSubprocessClient("sam3", "meta", "cpu", 0.5, False),
    ]


def test_subprocess_clients_ignore_non_dict_json_stdout_noise(monkeypatch):
    noise_then_ok = [
        "123\n",
        '["list", "noise"]\n',
        '"plain string"\n',
        "true\n",
        "null\n",
        json.dumps({"type": "ok", "result": {"ok": 1}}) + "\n",
    ]
    for client in _make_persistent_clients():
        client.process = _FakePopen(stdout_lines=noise_then_ok)
        res = client._request_locked({"action": "ping"}, timeout=5)
        assert res == {"ok": 1}

    # Also test geoai_task_subprocess.run_geoai_task
    from geoai.core import geoai_task_subprocess as gts

    fake_proc = _FakePopen(stdout_lines=noise_then_ok)
    monkeypatch.setattr(gts, "venv_exists", lambda: True)
    monkeypatch.setattr(gts, "get_venv_python_path", lambda: sys.executable)
    monkeypatch.setattr(gts, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(gts, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(gts.subprocess, "Popen", lambda *a, **kw: fake_proc)

    progress = []
    result = gts.run_geoai_task("ping", {}, progress_callback=progress.append)
    assert result == {"ok": 1}
    assert "123" in progress


def test_subprocess_clients_reap_process_after_kill(monkeypatch):
    class _StubbornPopen(_FakePopen):
        def __init__(self, stdout_lines=None):
            super().__init__(stdout_lines=stdout_lines)
            self._killed = False

        def poll(self):
            return -9 if self._killed else None

        def wait(self, timeout=None):
            self.events.append(("wait", timeout))
            if not self._killed:
                raise subprocess.TimeoutExpired("worker", timeout or 2)
            return -9

        def kill(self):
            self.events.append(("kill",))
            self._killed = True

    for client in _make_persistent_clients():
        proc = _StubbornPopen()
        client.process = proc
        client.close()
        kill_idx = proc.events.index(("kill",))
        assert ("wait", 2) in proc.events[kill_idx + 1 :]

    # Also test geoai_task_subprocess finally block reaps after terminate/kill.
    from geoai.core import geoai_task_subprocess as gts

    stubborn = _StubbornPopen(
        stdout_lines=[json.dumps({"type": "error", "message": "boom"}) + "\n"]
    )
    monkeypatch.setattr(gts, "venv_exists", lambda: True)
    monkeypatch.setattr(gts, "get_venv_python_path", lambda: sys.executable)
    monkeypatch.setattr(gts, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(gts, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(gts.subprocess, "Popen", lambda *a, **kw: stubborn)

    with pytest.raises(RuntimeError, match="boom"):
        gts.run_geoai_task("fail", {})

    assert ("terminate",) in stubborn.events
    kill_idx = stubborn.events.index(("kill",))
    assert ("wait", 2) in stubborn.events[kill_idx + 1 :]


def test_subprocess_clients_read_stderr_tail(monkeypatch):
    head_noise = "WARNING startup noise\n" * 300
    tail_marker = "RuntimeError: LATE_TRACEBACK_MARKER"
    full_stderr = head_noise + tail_marker

    for client in _make_persistent_clients():
        tf = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        tf.write(full_stderr)
        tf.flush()
        client._stderr_file = tf
        try:
            text = client._read_stderr()
            assert tail_marker in text
            assert len(text) <= 3000
        finally:
            client._close_stderr_file()

    # Also test geoai_task_subprocess error formatting uses tail of stderr.
    from geoai.core import geoai_task_subprocess as gts

    class _FailPopen(_FakePopen):
        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    def fake_popen(*_a, **kw):
        stderr_f = kw["stderr"]
        stderr_f.write(full_stderr)
        stderr_f.flush()
        return _FailPopen(stdout_lines=[])

    monkeypatch.setattr(gts, "venv_exists", lambda: True)
    monkeypatch.setattr(gts, "get_venv_python_path", lambda: sys.executable)
    monkeypatch.setattr(gts, "_get_clean_env_for_venv", dict)
    monkeypatch.setattr(gts, "_get_subprocess_kwargs", dict)
    monkeypatch.setattr(gts.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError) as exc_info:
        gts.run_geoai_task("fail", {})

    assert tail_marker in str(exc_info.value)


def test_subprocess_clients_close_on_timeout():
    for client in _make_persistent_clients():
        proc = _FakePopen(block_stdout=True)
        client.process = proc
        close_called = []
        orig_close = client.close

        def wrapped_close(orig=orig_close, flag=close_called):
            flag.append(True)
            orig()

        client.close = wrapped_close
        client._read_response_line = lambda _timeout: (_ for _ in ()).throw(
            TimeoutError("timed out")
        )

        with pytest.raises(TimeoutError):
            client._request({"action": "slow"}, timeout=1)

        assert close_called
        assert client.process is None
