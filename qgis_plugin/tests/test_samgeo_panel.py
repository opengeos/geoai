"""Widget-level tests for the Segment Anything panel's save, tile and HF controls.

``qgis.core`` is a MagicMock here (see conftest.py), so the subprocess
vectorization and QGIS attribute writing are replaced with recorders.
"""

import os
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from geoai.core import venv_manager
from geoai.dialogs import samgeo


class FakeSam:
    """In-memory model stand-in that writes an empty raster on save."""

    def __init__(self):
        self.masks = [1, 2]
        self.mask_scores = None

    def save_masks(self, output, unique):
        """Write a placeholder raster and record scores like SamGeo3.

        Args:
            output: Raster path.
            unique: Whether masks get unique values.
        """
        open(output, "wb").close()
        self.mask_scores = {"1": 0.9} if unique else None


@pytest.fixture(scope="module")
def qapp():
    """Keep one QApplication alive for the module's widgets."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dock(qapp, monkeypatch, tmp_path):
    """A panel with an image set and synchronous save workers."""
    monkeypatch.setattr(venv_manager, "venv_exists", lambda *_a, **_k: False)
    widget = samgeo.SamGeoDockWidget(MagicMock())
    image = tmp_path / "scene.tif"
    image.write_bytes(b"x")
    widget.current_image_path = str(image)
    widget.current_source_path = str(image)
    widget.add_to_map_check.setChecked(False)
    monkeypatch.setattr(widget, "_run_with_progress", lambda w, d, e: d(w()))
    calls = {}
    monkeypatch.setattr(
        widget,
        "_vectorize_mask_raster",
        lambda raster, out, settings: calls.setdefault("vec", (raster, out)),
    )
    monkeypatch.setattr(
        samgeo,
        "add_mask_attributes",
        lambda path, **kw: calls.setdefault("attrs", (path, kw)),
    )
    widget._test_calls = calls
    yield widget
    widget.cleanup()


def test_hf_controls(dock):
    assert dock.hf_token_edit.echoMode() == dock.hf_token_edit.EchoMode.Password
    assert "install" in dock.hf_status_label.text()


def test_tiled_controls_follow_checkbox_and_model_support(dock):
    assert not dock.tile_size_spin.isEnabled()
    dock.tiled_check.setChecked(True)
    assert dock.tile_size_spin.isEnabled()
    assert dock.tile_size_spin.value() == 1024
    assert dock.tile_overlap_spin.value() == 128

    class NoTiles:
        masks = None

        def supports_method(self, name):
            return name != "generate_masks_tiled"

    dock.sam = NoTiles()
    dock._update_tiled_support()
    assert not dock.tiled_check.isEnabled()
    assert not dock.tiled_check.isChecked()


def test_resolve_output_path_default_and_explicit(dock, tmp_path):
    assert dock._resolve_output_path(".gpkg") == (
        str(tmp_path / "scene_sam_masks.gpkg"),
        "scene_sam_masks",
    )
    dock.output_path_edit.setText("/x/y.shp")
    assert dock._resolve_output_path(".shp") == ("/x/y.shp", "y.shp")


def test_save_vector_passes_scores_source_and_prompt(dock, tmp_path):
    dock.sam = FakeSam()
    dock._last_prompt = "text: tree"
    dock.output_format_combo.setCurrentText("Vector (GeoJSON)")

    dock.save_masks()

    raster, out = dock._test_calls["vec"]
    assert out == str(tmp_path / "scene_sam_masks.geojson")
    assert not os.path.exists(raster)  # temporary mask raster removed
    path, kwargs = dock._test_calls["attrs"]
    assert path == out
    assert kwargs["mask_scores"] == {"1": 0.9}
    assert kwargs["source"] == "scene.tif"
    assert kwargs["prompt"] == "text: tree"


def test_save_vector_skips_scores_when_not_unique(dock):
    dock.sam = FakeSam()
    dock.unique_check.setChecked(False)
    dock.output_format_combo.setCurrentText("Vector (GeoPackage)")

    dock.save_masks()

    assert dock._test_calls["attrs"][1]["mask_scores"] is None


def test_save_reuses_tiled_raster(dock, tmp_path):
    dock.sam = FakeSam()
    dock.sam.masks = None
    tiled = tmp_path / "tiled.tif"
    tiled.write_bytes(b"tiles")
    dock._tiled_result = {"raster": str(tiled), "unique": True, "count": 3}
    assert dock._has_results()

    dock.output_format_combo.setCurrentText("Vector (Shapefile)")
    dock.save_masks()
    assert dock._test_calls["vec"][0] == str(tiled)
    assert dock._test_calls["attrs"][1]["mask_scores"] is None

    dock.output_format_combo.setCurrentText("Raster (GeoTIFF)")
    dock.save_masks()
    assert (tmp_path / "scene_sam_masks.tif").read_bytes() == b"tiles"


def test_new_image_clears_tiled_state(dock, tmp_path):
    dock._tiled_result = {"raster": str(tmp_path / "t.tif"), "unique": True}
    dock._image_state_stale = True
    dock._last_prompt = "box"

    dock._on_image_changed([3, 2, 1])

    assert dock._tiled_result is None
    assert dock._image_state_stale is False
    assert dock._last_prompt is None
    assert dock._current_bands == [3, 2, 1]


def test_restore_image_after_tiled_run(dock):
    sam = MagicMock()
    dock.sam = sam
    dock._current_bands = [1, 2, 3]
    dock._image_state_stale = True

    dock._restore_image_if_needed()
    dock._restore_image_if_needed()

    sam.set_image.assert_called_once_with(dock.current_image_path, bands=[1, 2, 3])


# ---------------------------------------------------------------------------
# Hugging Face login
# ---------------------------------------------------------------------------

TOKEN = "hf_PanelSecretToken0123456789"


class _BusyWorker:
    """Stand-in for a running Hugging Face worker thread."""

    def __init__(self):
        self.running = True

    def isRunning(self):
        """Report whether the fake thread is running.

        Returns:
            bool: The running flag.
        """
        return self.running

    def wait(self, *_args):
        """Pretend the thread finished.

        Returns:
            bool: Always True.
        """
        return True


@pytest.fixture
def hf_dock(dock, monkeypatch):
    """A panel whose HF tasks are captured instead of run."""
    monkeypatch.setattr(dock, "_hf_env_available", lambda: True)
    errors = []
    monkeypatch.setattr(dock, "show_error", errors.append)
    dock._test_errors = errors
    return dock


def _capture_hf_tasks(dock, monkeypatch):
    tasks = []
    monkeypatch.setattr(dock, "_run_hf_task", lambda *args: tasks.append(args))
    return tasks


def test_login_is_queued_behind_running_status_check(hf_dock, monkeypatch):
    hf_dock._hf_worker = _BusyWorker()
    hf_dock.hf_token_edit.setText(TOKEN)

    hf_dock.hf_login()

    assert hf_dock._hf_queued is not None
    assert hf_dock._hf_queued[0] == "hf_login"
    assert not hf_dock.hf_login_btn.isEnabled()
    assert hf_dock.hf_token_edit.isReadOnly()

    # The status check finishes; the queued login runs and completes.
    hf_dock._hf_worker.running = False
    ran = []

    def _run(action, params, on_done, on_error):
        ran.append(action)
        on_done({"logged_in": True, "user": "jdoe"})

    monkeypatch.setattr(hf_dock, "_run_hf_task", _run)
    hf_dock._start_queued_hf_task()

    assert ran == ["hf_login"]
    assert hf_dock._hf_queued is None
    assert hf_dock.hf_login_btn.isEnabled()
    assert not hf_dock.hf_token_edit.isReadOnly()
    assert hf_dock.hf_token_edit.text() == ""
    assert "jdoe" in hf_dock.hf_status_label.text()


def test_login_success_clears_only_the_submitted_token(hf_dock, monkeypatch):
    tasks = _capture_hf_tasks(hf_dock, monkeypatch)
    hf_dock.hf_token_edit.setText(TOKEN)
    hf_dock.hf_login()
    hf_dock.hf_token_edit.setText("hf_SomethingElse")

    tasks[0][2]({"logged_in": True, "user": "jdoe"})

    assert hf_dock.hf_token_edit.text() == "hf_SomethingElse"


def test_login_success_drops_inherited_hf_token_for_samgeo_workers_only(
    hf_dock, monkeypatch
):
    from geoai.core import samgeo_subprocess

    monkeypatch.setattr(samgeo_subprocess, "_PREFER_SAVED_HF_TOKEN", False)
    monkeypatch.setenv("HF_TOKEN", "hf_OldEnvironmentToken")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_OldLegacyToken")
    tasks = _capture_hf_tasks(hf_dock, monkeypatch)
    hf_dock.hf_token_edit.setText(TOKEN)
    hf_dock.hf_login()

    tasks[0][2]({"logged_in": True, "user": None})

    # The QGIS process environment (shared with other plugins) is untouched;
    # only SamGeo workers started from now on drop the variables.
    assert os.environ["HF_TOKEN"] == "hf_OldEnvironmentToken"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_OldLegacyToken"
    assert samgeo_subprocess._PREFER_SAVED_HF_TOKEN is True
    assert "unknown user" in hf_dock.hf_status_label.text()


def test_login_failure_keeps_env_and_redacts(hf_dock, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_OldEnvironmentToken")
    tasks = _capture_hf_tasks(hf_dock, monkeypatch)
    hf_dock.hf_token_edit.setText(TOKEN)
    hf_dock.hf_login()

    from geoai.core import samgeo_subprocess

    monkeypatch.setattr(samgeo_subprocess, "_PREFER_SAVED_HF_TOKEN", False)
    tasks[0][3](f"Hugging Face login failed: bad token {TOKEN}")

    assert os.environ["HF_TOKEN"] == "hf_OldEnvironmentToken"
    assert samgeo_subprocess._PREFER_SAVED_HF_TOKEN is False
    assert hf_dock.hf_login_btn.isEnabled()
    assert not hf_dock.hf_token_edit.isReadOnly()
    assert TOKEN not in hf_dock._test_errors[0]
    assert hf_dock.hf_token_edit.text() == TOKEN  # kept for a retry


# ---------------------------------------------------------------------------
# Save guard and temporary rasters
# ---------------------------------------------------------------------------


def test_second_save_is_rejected_while_one_runs(dock, monkeypatch):
    shown = []
    monkeypatch.setattr(
        samgeo.QMessageBox, "information", lambda *args: shown.append(args)
    )
    dock.sam = FakeSam()
    dock._save_busy = True

    dock.save_masks()
    dock._auto_show_results()

    assert "vec" not in dock._test_calls
    assert len(shown) == 1


def test_save_re_enables_button_when_done(dock):
    dock.sam = FakeSam()
    dock.output_format_combo.setCurrentText("Vector (GeoJSON)")

    dock.save_masks()

    assert dock._save_busy is False
    assert dock.save_btn.isEnabled()


def test_superseded_tiled_raster_is_deleted_after_running_save(dock, tmp_path):
    raster = tmp_path / "old_tiles.tif"
    raster.write_bytes(b"t")
    dock._temp_files.append(str(raster))
    dock._set_tiled_result({"raster": str(raster), "unique": True})

    dock._save_busy = True
    dock._set_tiled_result(None)
    assert raster.exists()  # a save may still be reading it

    dock._end_save()
    assert not raster.exists()
    assert str(raster) not in dock._temp_files


# ---------------------------------------------------------------------------
# Tiled segmentation without a full-image setup
# ---------------------------------------------------------------------------


class TiledSam:
    """Model stand-in that supports generate_masks_tiled."""

    masks = None

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def supports_method(self, name):
        """Report support for tiled generation.

        Args:
            name: Method name.

        Returns:
            bool: True for ``generate_masks_tiled``.
        """
        return name == "generate_masks_tiled"

    def generate_masks_tiled(self, **kwargs):
        """Record the call and write the output raster.

        Args:
            **kwargs: Tiling arguments.

        Returns:
            dict: A worker-style result.
        """
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        with open(kwargs["output"], "wb") as handle:
            handle.write(b"tiles")
        return {"output": kwargs["output"], "object_count": 2}


def _sync_runner(work, done, error):
    try:
        result = work()
    except Exception as exc:
        error(str(exc))
        return
    done(result)


def test_tiled_text_runs_on_recorded_source_without_full_image(
    dock, monkeypatch, tmp_path
):
    big = tmp_path / "big.tif"
    big.write_bytes(b"x")
    dock.current_image_path = None
    dock.current_source_path = None
    dock._tiled_source = {"image": str(big), "source": str(big), "bands": [3, 2, 1]}
    dock.sam = TiledSam()
    dock.tiled_check.setChecked(True)
    dock.text_prompt_edit.setText("tree")
    monkeypatch.setattr(dock, "_auto_show_results", lambda: None)

    dock.segment_by_text()

    call = dock.sam.calls[0]
    assert call["source"] == str(big)
    assert call["bands"] == [3, 2, 1]
    assert dock._tiled_result["source"] == str(big)
    assert dock._tiled_result["count"] == 2

    # Saving names the output after the tiled source image.
    dock.output_format_combo.setCurrentText("Vector (GeoPackage)")
    dock.save_masks()
    assert dock._test_calls["vec"][1] == str(tmp_path / "big_sam_masks.gpkg")
    assert dock._test_calls["attrs"][1]["source"] == "big.tif"


def test_untiled_text_still_needs_full_image(dock, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(dock, "show_error", errors.append)
    dock.current_image_path = None
    dock._tiled_source = {"image": str(tmp_path / "big.tif"), "source": None}
    dock.sam = TiledSam()
    dock.text_prompt_edit.setText("tree")

    dock.segment_by_text()

    assert errors == ["Please set an image first."]
    assert dock.sam.calls == []


def test_failed_tiled_run_removes_its_raster(dock, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(dock, "show_error", errors.append)
    monkeypatch.setattr(dock, "_run_with_progress", _sync_runner)
    big = tmp_path / "big.tif"
    big.write_bytes(b"x")
    dock._tiled_source = {"image": str(big), "source": str(big), "bands": None}
    dock.sam = TiledSam(fail=True)
    dock.tiled_check.setChecked(True)
    dock.text_prompt_edit.setText("tree")
    before = list(dock._temp_files)

    dock.segment_by_text()

    output = dock.sam.calls[0]["output"]
    assert not os.path.exists(output)
    assert dock._temp_files == before
    assert "CUDA out of memory" in errors[0]


# ---------------------------------------------------------------------------
# Worker threads on cleanup
# ---------------------------------------------------------------------------


def test_cleanup_keeps_running_thread_alive_until_it_finishes(dock, qapp):
    import threading
    import time

    release = threading.Event()
    worker = samgeo.SamGeoOperationWorker(lambda: release.wait(10))
    worker.start()
    real_wait = worker.wait
    calls = []

    def _wait(*args):
        calls.append(args)
        return False if len(calls) == 1 else real_wait(*args)

    worker.wait = _wait
    dock._hf_worker = worker

    dock._stop_worker_threads()

    assert dock._hf_worker is None
    assert worker in samgeo._LIVE_THREADS

    release.set()
    deadline = time.monotonic() + 5
    while worker in samgeo._LIVE_THREADS and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert worker not in samgeo._LIVE_THREADS
    assert not worker.isRunning()


def test_dropping_a_queued_login_unlocks_the_controls(hf_dock):
    hf_dock.hf_login_btn.setEnabled(False)
    hf_dock.hf_token_edit.setReadOnly(True)
    hf_dock._hf_queued = ("hf_login", {"token": TOKEN}, None, None)

    hf_dock._drop_queued_hf_task()

    assert hf_dock._hf_queued is None
    assert hf_dock.hf_login_btn.isEnabled()
    assert not hf_dock.hf_token_edit.isReadOnly()
