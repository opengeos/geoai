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
