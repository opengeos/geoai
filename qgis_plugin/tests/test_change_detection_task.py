import os
import sys
import types

import pytest

from geoai.workers import geoai_task_worker


class FakeChangeDetection:
    instances = []

    def __init__(self, sam_model_type):
        self.sam_model_type = sam_model_type
        self.hyperparameters = None
        self.detect_args = None
        FakeChangeDetection.instances.append(self)

    def set_hyperparameters(self, **kwargs):
        self.hyperparameters = kwargs

    def detect_changes(self, image1_path, image2_path, **kwargs):
        self.detect_args = (image1_path, image2_path, kwargs)


def _install_fakes(monkeypatch, vectorized):
    change_module = types.ModuleType("geoai.change_detection")
    change_module.ChangeDetection = FakeChangeDetection
    raster_module = types.ModuleType("geoai.utils.raster")
    raster_module.raster_to_vector = lambda path, **kwargs: vectorized.append(
        (path, kwargs)
    )
    monkeypatch.setitem(sys.modules, "geoai.change_detection", change_module)
    monkeypatch.setitem(sys.modules, "geoai.utils.raster", raster_module)
    monkeypatch.setattr(geoai_task_worker, "_emit", lambda message: None)
    FakeChangeDetection.instances.clear()


def _params(**overrides):
    params = {
        "before_path": "before.tif",
        "after_path": "after.tif",
        "output_raster": "changes.tif",
        "output_vector": "",
        "sam_model_type": "vit_b",
        "change_confidence_threshold": 150,
        "auto_threshold": False,
        "min_area": 0,
    }
    params.update(overrides)
    return params


def test_detect_changes_is_registered():
    assert (
        geoai_task_worker._TASKS["detect_changes"]
        is geoai_task_worker._task_detect_changes
    )


def test_detect_changes_passes_settings_to_anychange(monkeypatch):
    vectorized = []
    _install_fakes(monkeypatch, vectorized)

    result = geoai_task_worker._task_detect_changes(_params())

    detector = FakeChangeDetection.instances[0]
    assert detector.sam_model_type == "vit_b"
    assert detector.hyperparameters == {
        "change_confidence_threshold": 150,
        "auto_threshold": False,
    }
    assert detector.detect_args == (
        "before.tif",
        "after.tif",
        {
            "output_path": "changes.tif",
            "return_results": False,
            "export_instance_masks": False,
            "instance_masks_output_path": None,
        },
    )
    assert vectorized == []
    assert result == {"output_raster": "changes.tif", "output_vector": ""}


def test_detect_changes_vectorizes_when_requested(monkeypatch):
    vectorized = []
    _install_fakes(monkeypatch, vectorized)

    result = geoai_task_worker._task_detect_changes(
        _params(output_vector="changes.gpkg", min_area=50)
    )

    detect_kwargs = FakeChangeDetection.instances[0].detect_args[2]
    instance_raster = detect_kwargs["instance_masks_output_path"]
    assert detect_kwargs["export_instance_masks"] is True
    assert os.path.basename(instance_raster) == "instances.tif"
    assert vectorized == [
        (instance_raster, {"output_path": "changes.gpkg", "min_area": 50})
    ]
    assert not os.path.exists(os.path.dirname(instance_raster))
    assert result["output_vector"] == "changes.gpkg"


def test_detect_changes_keeps_user_files_next_to_outputs(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, [])
    user_file = tmp_path / "changes_instances.tif"
    user_file.write_bytes(b"keep")

    geoai_task_worker._task_detect_changes(
        _params(
            output_raster=str(tmp_path / "changes.tif"),
            output_vector=str(tmp_path / "changes.gpkg"),
        )
    )

    assert user_file.read_bytes() == b"keep"


def test_detect_changes_cleans_up_when_vectorization_fails(monkeypatch):
    _install_fakes(monkeypatch, [])
    instance_rasters = []

    def failing_raster_to_vector(path, **kwargs):
        instance_rasters.append(path)
        raise RuntimeError("vectorization failed")

    monkeypatch.setattr(
        sys.modules["geoai.utils.raster"], "raster_to_vector", failing_raster_to_vector
    )

    with pytest.raises(RuntimeError, match="vectorization failed"):
        geoai_task_worker._task_detect_changes(_params(output_vector="changes.gpkg"))

    assert not os.path.exists(os.path.dirname(instance_rasters[0]))
