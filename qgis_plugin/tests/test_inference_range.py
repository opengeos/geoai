from qgis.PyQt.QtWidgets import QApplication

from geoai.dialogs.instance_segmentation import (
    InstanceInferenceWorker,
    InstanceSegmentationDockWidget,
)
from geoai.dialogs.segmentation import InferenceWorker, SegmentationDockWidget


def test_semantic_bbox_intersection_clips_to_layer_extent():
    assert SegmentationDockWidget._intersect_bboxes([0, 0, 10, 10], [5, -5, 20, 6]) == [
        5,
        0,
        10,
        6,
    ]
    assert SegmentationDockWidget._intersect_bboxes([0, 0, 1, 1], [2, 2, 3, 3]) is None


def test_instance_bbox_intersection_clips_to_layer_extent():
    assert InstanceSegmentationDockWidget._intersect_bboxes(
        [0, 0, 10, 10], [-2, 2, 8, 12]
    ) == [0, 2, 8, 10]
    assert (
        InstanceSegmentationDockWidget._intersect_bboxes([0, 0, 1, 1], [1, 1, 2, 2])
        is None
    )


def test_semantic_clear_inference_range_resets_state():
    QApplication.instance() or QApplication([])
    widget = SegmentationDockWidget.__new__(SegmentationDockWidget)
    widget.inference_bbox = [1, 2, 3, 4]
    widget.inference_bbox_crs = "EPSG:3857"
    widget.inference_range_layer_source = "image.tif"
    widget.inference_range_tool = None

    widget.clear_inference_range()

    assert widget.inference_bbox is None
    assert widget.inference_bbox_crs is None
    assert widget.inference_range_layer_source is None


def test_instance_clear_inference_range_resets_state():
    QApplication.instance() or QApplication([])
    widget = InstanceSegmentationDockWidget.__new__(InstanceSegmentationDockWidget)
    widget.inference_bbox = [1, 2, 3, 4]
    widget.inference_bbox_crs = "EPSG:3857"
    widget.inference_range_layer_source = "image.tif"
    widget.inference_range_tool = None

    widget.clear_inference_range()

    assert widget.inference_bbox is None
    assert widget.inference_bbox_crs is None
    assert widget.inference_range_layer_source is None


def test_semantic_worker_clips_before_inference(monkeypatch):
    calls = []

    def fake_run_geoai_task(action, params, progress_callback=None):
        calls.append((action, dict(params)))
        if action == "clip_raster_by_bbox":
            with open(params["output_raster"], "wb") as fp:
                fp.write(b"temporary")
        return {"output_path": params.get("output_path")}

    monkeypatch.setattr(
        "geoai.core.geoai_task_subprocess.run_geoai_task", fake_run_geoai_task
    )

    worker = InferenceWorker(
        "input.tif",
        "output.tif",
        "model.pth",
        "unet",
        "resnet34",
        3,
        2,
        512,
        256,
        4,
        inference_bbox=[1.0, 2.0, 3.0, 4.0],
        inference_bbox_crs="EPSG:4326",
    )

    worker.run()

    assert [call[0] for call in calls] == [
        "clip_raster_by_bbox",
        "semantic_segmentation",
    ]
    assert calls[0][1]["input_raster"] == "input.tif"
    assert calls[0][1]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert calls[0][1]["bbox_crs"] == "EPSG:4326"
    assert calls[1][1]["input_path"] == calls[0][1]["output_raster"]
    assert calls[1][1]["output_path"] == "output.tif"


def test_instance_worker_clips_before_inference(monkeypatch):
    calls = []

    def fake_run_geoai_task(action, params, progress_callback=None):
        calls.append((action, dict(params)))
        if action == "clip_raster_by_bbox":
            with open(params["output_raster"], "wb") as fp:
                fp.write(b"temporary")
        return {"output_path": params.get("output_path")}

    monkeypatch.setattr(
        "geoai.core.geoai_task_subprocess.run_geoai_task", fake_run_geoai_task
    )

    worker = InstanceInferenceWorker(
        "input.tif",
        "output.tif",
        "model.pth",
        3,
        2,
        512,
        256,
        0.5,
        4,
        inference_bbox=[1.0, 2.0, 3.0, 4.0],
        inference_bbox_crs="EPSG:4326",
    )

    worker.run()

    assert [call[0] for call in calls] == [
        "clip_raster_by_bbox",
        "instance_segmentation",
    ]
    assert calls[0][1]["input_raster"] == "input.tif"
    assert calls[0][1]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert calls[0][1]["bbox_crs"] == "EPSG:4326"
    assert calls[1][1]["input_path"] == calls[0][1]["output_raster"]
    assert calls[1][1]["output_path"] == "output.tif"
