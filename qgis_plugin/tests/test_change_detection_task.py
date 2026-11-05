import sys
import types

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
    assert detect_kwargs["export_instance_masks"] is True
    assert detect_kwargs["instance_masks_output_path"] == "changes_instances.tif"
    assert vectorized == [
        ("changes_instances.tif", {"output_path": "changes.gpkg", "min_area": 50})
    ]
    assert result["output_vector"] == "changes.gpkg"


def test_detect_changes_removes_intermediate_rasters(monkeypatch, tmp_path):
    vectorized = []
    _install_fakes(monkeypatch, vectorized)
    instance_raster = tmp_path / "changes_instances.tif"
    scores_raster = tmp_path / "changes_instances_scores.tif"
    instance_raster.write_bytes(b"")
    scores_raster.write_bytes(b"")

    geoai_task_worker._task_detect_changes(
        _params(
            output_raster=str(tmp_path / "changes.tif"),
            output_vector=str(tmp_path / "changes.gpkg"),
        )
    )

    assert not instance_raster.exists()
    assert not scores_raster.exists()
