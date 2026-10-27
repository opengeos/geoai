"""Qt integration tests for the QGIS processing monitor."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from qgis.PyQt.QtCore import QObject, QThread, pyqtSignal
from qgis.PyQt.QtGui import QTextCursor
from qgis.PyQt.QtWidgets import QApplication

from geoai.core import processing_monitor as processing_monitor_core
from geoai.core.processing_monitor import ProcessingMonitor, TaskStatus
from geoai.dialogs.processing_monitor import ProcessingMonitorDockWidget
from geoai.dialogs.segmentation import SegmentationDockWidget


@pytest.fixture(autouse=True)
def isolated_processing_monitor(monkeypatch):
    monitor = ProcessingMonitor()
    monkeypatch.setattr(processing_monitor_core, "_monitor", monitor)
    yield monitor


@pytest.fixture(scope="module", autouse=True)
def qt_application():
    app = QApplication.instance() or QApplication([])
    yield app


def _app():
    return QApplication.instance() or QApplication([])


def _task_id(prefix):
    return f"{prefix}-{uuid4().hex}"


def test_refresh_reuses_items_and_preserves_selection():
    _app()
    monitor = processing_monitor_core.get_processing_monitor()
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
    monitor = processing_monitor_core.get_processing_monitor()
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
    monitor = processing_monitor_core.get_processing_monitor()
    task_id = _task_id("shutdown")
    monitor.submit(task_id, "Shutdown task")
    worker = SimpleNamespace(_processing_monitor_task_id=task_id)
    panel = SimpleNamespace(_processing_monitor=monitor)

    SegmentationDockWidget._cancel_monitored_worker(panel, worker)

    assert monitor.get(task_id).status == TaskStatus.CANCELLED
    monitor.clear_finished()


def test_panel_shutdown_handles_missing_and_terminal_tasks():
    monitor = processing_monitor_core.get_processing_monitor()
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


def test_worker_exception_signal_preserves_exception_for_monitor():
    class Worker(QObject):
        started = pyqtSignal()
        progress = pyqtSignal(str)
        finished = pyqtSignal(str)
        error = pyqtSignal(str)
        exception = pyqtSignal(object)

    monitor = processing_monitor_core.get_processing_monitor()
    worker = Worker()

    class Panel(QObject):
        pass

    panel = Panel()
    panel._processing_monitor = monitor
    task_id = SegmentationDockWidget._monitor_worker(panel, worker, "Test task")

    worker.started.emit()
    worker.error.emit("missing input\n\ntraceback")
    worker.exception.emit(FileNotFoundError("missing input"))

    task = monitor.get(task_id)
    assert task.status == TaskStatus.FAILED
    assert task.error_summary == "Missing or invalid input file: missing input"
    assert "FileNotFoundError: missing input" in task.error_details


@pytest.mark.parametrize(
    ("error", "summary"),
    [
        (
            ImportError("model dependency"),
            "Missing Python dependency: model dependency",
        ),
        (
            PermissionError("cannot save output"),
            "Output or file access failure: cannot save output",
        ),
        ("first line\ntechnical detail", "Processing failed: first line"),
        ("", "Processing failed"),
    ],
)
def test_worker_exception_signal_preserves_error_classification(error, summary):
    class Worker(QObject):
        started = pyqtSignal()
        progress = pyqtSignal(str)
        finished = pyqtSignal(str)
        exception = pyqtSignal(object)

    monitor = processing_monitor_core.get_processing_monitor()
    worker = Worker()

    class Panel(QObject):
        pass

    panel = Panel()
    panel._processing_monitor = monitor
    task_id = SegmentationDockWidget._monitor_worker(panel, worker, "Test task")

    worker.started.emit()
    worker.exception.emit(error)

    task = monitor.get(task_id)
    assert task.status == TaskStatus.FAILED
    assert task.error_summary == summary


def test_worker_monitor_callbacks_are_delivered_to_dock_thread():
    callback_threads = []

    class Monitor(ProcessingMonitor):
        def start(self, task_id):
            callback_threads.append(QThread.currentThread())
            super().start(task_id)

        def progress(self, task_id, message="", percent=None):
            callback_threads.append(QThread.currentThread())
            super().progress(task_id, message, percent)

        def fail(self, task_id, error):
            callback_threads.append(QThread.currentThread())
            super().fail(task_id, error)

    class Worker(QThread):
        started = pyqtSignal()
        progress = pyqtSignal(str)
        finished = pyqtSignal(str)
        exception = pyqtSignal(object)

        def run(self):
            self.progress.emit("Working")
            self.exception.emit(FileNotFoundError("missing input"))

    monitor = Monitor()

    class Panel(QObject):
        pass

    panel = Panel()
    panel._processing_monitor = monitor
    worker = Worker()
    task_id = SegmentationDockWidget._monitor_worker(panel, worker, "Threaded task")

    worker.start()
    assert worker.wait(5000)
    for _ in range(100):
        QApplication.processEvents()
        if monitor.get(task_id).status == TaskStatus.FAILED:
            break

    task = monitor.get(task_id)
    assert task.status == TaskStatus.FAILED
    assert task.progress_message == "Working"
    assert task.error_summary == "Missing or invalid input file: missing input"
    assert callback_threads
    assert all(thread == panel.thread() for thread in callback_threads)
    worker.deleteLater()


@pytest.mark.parametrize(
    ("status", "progress", "visible", "minimum", "maximum", "value"),
    [
        (TaskStatus.QUEUED, None, True, 0, 0, 0),
        (TaskStatus.RUNNING, None, True, 0, 0, 0),
        (TaskStatus.RUNNING, 42, True, 0, 100, 42),
        (TaskStatus.COMPLETED, None, False, 0, 0, 0),
        (TaskStatus.FAILED, None, False, 0, 0, 0),
        (TaskStatus.CANCELLED, None, False, 0, 0, 0),
        (TaskStatus.COMPLETED, 100, True, 0, 100, 100),
    ],
)
def test_progress_bar_reflects_task_status_and_measurement(
    status, progress, visible, minimum, maximum, value
):
    _app()
    monitor = processing_monitor_core.get_processing_monitor()
    task_id = _task_id("progress")
    task = monitor.submit(task_id, "Progress task")
    task.status = status
    task.progress = progress

    dock = ProcessingMonitorDockWidget()
    try:
        dock._show_task(task)
        assert dock.progress_bar.isHidden() is not visible
        assert dock.progress_bar.minimum() == minimum
        assert dock.progress_bar.maximum() == maximum
        if progress is not None:
            assert dock.progress_bar.value() == value
    finally:
        dock.close()


def test_no_selected_task_hides_progress_bar():
    _app()
    dock = ProcessingMonitorDockWidget()
    try:
        dock._show_task(None)
        assert dock.progress_bar.isHidden()
    finally:
        dock.close()


def test_processing_monitor_tests_use_an_isolated_singleton():
    assert processing_monitor_core.get_processing_monitor().snapshot() == []
