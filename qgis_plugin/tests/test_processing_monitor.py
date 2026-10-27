"""Qt integration tests for the QGIS processing monitor."""

from types import SimpleNamespace
from uuid import uuid4

from qgis.PyQt.QtGui import QTextCursor
from qgis.PyQt.QtWidgets import QApplication

from geoai.core.processing_monitor import TaskStatus, get_processing_monitor
from geoai.dialogs.processing_monitor import ProcessingMonitorDockWidget
from geoai.dialogs.segmentation import SegmentationDockWidget


def _app():
    return QApplication.instance() or QApplication([])


def _task_id(prefix):
    return f"{prefix}-{uuid4().hex}"


def test_refresh_reuses_items_and_preserves_selection():
    _app()
    monitor = get_processing_monitor()
    first_id = _task_id("first")
    second_id = _task_id("second")
    monitor.submit(first_id, "First task")

    dock = ProcessingMonitorDockWidget()
    try:
        dock.refresh()
        dock.task_list.setCurrentRow(0)
        item = dock.task_list.currentItem()
        monitor.start(first_id)
        monitor.progress(first_id, "Updated")
        dock.refresh()

        assert dock.task_list.currentItem() is item
        assert dock._selected_task_id == first_id
        assert item.text().endswith("RUNNING")

        monitor.submit(second_id, "Second task")
        dock.refresh()
        assert dock.task_list.count() == 2
        assert dock.task_list.item(0) is item

        monitor.complete(first_id)
        monitor.complete(second_id)
        monitor.clear_finished()
        dock.refresh()
        assert dock.task_list.count() == 0
        assert dock._selected_task_id is None
    finally:
        dock.close()


def test_details_text_is_not_reset_when_content_is_unchanged():
    _app()
    monitor = get_processing_monitor()
    task_id = _task_id("details")
    monitor.submit(task_id, "Details task")
    monitor.progress(task_id, "Same details")

    dock = ProcessingMonitorDockWidget()
    try:
        dock.refresh()
        cursor = dock.details_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        dock.details_text.setTextCursor(cursor)
        cursor_position = cursor.position()
        dock.refresh()
        assert dock.details_text.toPlainText() == "Same details"
        assert dock.details_text.textCursor().position() == cursor_position

        monitor.progress(task_id, "Changed details")
        dock.refresh()
        assert dock.details_text.toPlainText() == "Changed details"
    finally:
        monitor.complete(task_id)
        monitor.clear_finished()
        dock.close()


def test_timer_tracks_dock_visibility():
    _app()
    dock = ProcessingMonitorDockWidget()
    try:
        assert not dock._timer.isActive()
        dock.show()
        QApplication.processEvents()
        assert dock._timer.isActive()
        dock.hide()
        QApplication.processEvents()
        assert not dock._timer.isActive()
        dock.show()
        QApplication.processEvents()
        assert dock._timer.isActive()
    finally:
        dock.close()
        assert not dock._timer.isActive()


def test_panel_shutdown_cancels_monitor_task():
    monitor = get_processing_monitor()
    task_id = _task_id("shutdown")
    monitor.submit(task_id, "Shutdown task")
    worker = SimpleNamespace(_processing_monitor_task_id=task_id)
    panel = SimpleNamespace(_processing_monitor=monitor)

    SegmentationDockWidget._cancel_monitored_worker(panel, worker)

    assert monitor.get(task_id).status == TaskStatus.CANCELLED
    monitor.clear_finished()


def test_panel_shutdown_handles_missing_and_terminal_tasks():
    monitor = get_processing_monitor()
    panel = SimpleNamespace(_processing_monitor=monitor)
    SegmentationDockWidget._cancel_monitored_worker(panel, None)
    SegmentationDockWidget._cancel_monitored_worker(panel, SimpleNamespace())

    task_id = _task_id("terminal")
    monitor.submit(task_id, "Terminal task")
    monitor.complete(task_id)
    SegmentationDockWidget._cancel_monitored_worker(
        panel, SimpleNamespace(_processing_monitor_task_id=task_id)
    )
    assert monitor.get(task_id).status == TaskStatus.COMPLETED
    monitor.clear_finished()
