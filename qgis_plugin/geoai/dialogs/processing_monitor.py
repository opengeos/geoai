"""Dock widget for observing GeoAI processing tasks."""

from __future__ import annotations

from typing import Optional

from qgis.PyQt.QtCore import QTimer, Qt
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.processing_monitor import (
    ProcessingTask,
    TaskStatus,
    format_elapsed,
    get_processing_monitor,
)


class ProcessingMonitorDockWidget(QDockWidget):
    """Display lifecycle, timing, progress, errors and optional GPU metrics."""

    def __init__(self, parent=None):
        super().__init__("AI Processing Monitor", parent)
        self.setObjectName("GeoAIProcessingMonitorDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._monitor = get_processing_monitor()
        self._selected_task_id: Optional[str] = None
        self._items = {}
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def _build_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        self.task_list = QListWidget()
        self.task_list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.task_list)

        self.status_label = QLabel("No task selected")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.elapsed_label = QLabel("Elapsed: 00:00:00")
        layout.addWidget(self.elapsed_label)
        self.time_label = QLabel("Start: -\nEnd: -")
        layout.addWidget(self.time_label)
        self.gpu_label = QLabel("GPU: unavailable")
        self.gpu_label.setWordWrap(True)
        layout.addWidget(self.gpu_label)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)

        buttons = QHBoxLayout()
        clear_button = QPushButton("Clear finished")
        clear_button.clicked.connect(self._monitor.clear_finished)
        buttons.addWidget(clear_button)
        layout.addLayout(buttons)
        self.setWidget(widget)

    def refresh(self):
        tasks = self._monitor.snapshot()
        selected = self._selected_task_id
        task_ids = {task.task_id for task in tasks}
        self.task_list.blockSignals(True)
        for task in tasks:
            item = self._items.get(task.task_id)
            if item is None:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, task.task_id)
                self.task_list.addItem(item)
                self._items[task.task_id] = item
            item.setText(f"{task.name} - {task.status.value}")
        for task_id in tuple(self._items):
            if task_id not in task_ids:
                item = self._items.pop(task_id)
                row = self.task_list.row(item)
                if row >= 0:
                    self.task_list.takeItem(row)
        self.task_list.blockSignals(False)
        if selected in self._items:
            self._selected_task_id = selected
        elif tasks:
            self._selected_task_id = tasks[-1].task_id
        else:
            self._selected_task_id = None
            self._show_task(None)
        if self._selected_task_id is not None:
            self.task_list.blockSignals(True)
            self.task_list.setCurrentItem(self._items[self._selected_task_id])
            self.task_list.blockSignals(False)
            self._show_task(self._monitor.get(self._selected_task_id))

    def _selection_changed(self, current, _previous):
        self._selected_task_id = (
            current.data(Qt.ItemDataRole.UserRole) if current else None
        )
        self._show_task(self._monitor.get(self._selected_task_id))

    def _show_task(self, task: Optional[ProcessingTask]):
        if task is None:
            self.status_label.setText("No task selected")
            self.progress_bar.setVisible(False)
            self.elapsed_label.setText("Elapsed: 00:00:00")
            self.time_label.setText("Start: -\nEnd: -")
            self.gpu_label.setText("GPU: unavailable")
            self.details_text.clear()
            return

        self.status_label.setText(f"Status: {task.status.value}")
        if task.progress is None:
            if task.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(task.progress)
        self.elapsed_label.setText(f"Elapsed: {format_elapsed(task.elapsed_seconds)}")
        self.time_label.setText(
            f"Start: {self._format_timestamp(task.started_timestamp)}\n"
            f"End: {self._format_timestamp(task.ended_timestamp)}"
        )
        if task.gpu and task.gpu.available:
            self.gpu_label.setText(
                f"GPU: {task.gpu.device_name}\n"
                f"Allocated: {self._format_bytes(task.gpu.allocated_bytes)} | "
                f"Reserved: {self._format_bytes(task.gpu.reserved_bytes)}\n"
                f"Peak allocated: {self._format_bytes(task.gpu.peak_allocated_bytes)}"
            )
        else:
            reason = task.gpu.unavailable_reason if task.gpu else "not measured"
            self.gpu_label.setText(f"GPU: unavailable ({reason})")
        details = task.progress_message
        if task.error_summary:
            details = f"{details}\n\nError: {task.error_summary}".strip()
        if task.error_details:
            details = f"{details}\n\nTechnical details:\n{task.error_details}".strip()
        if self.details_text.toPlainText() != details:
            self.details_text.setPlainText(details)

    @staticmethod
    def _format_timestamp(value) -> str:
        if value is None:
            return "-"
        return value.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_bytes(value: Optional[int]) -> str:
        if value is None:
            return "-"
        return f"{value / (1024 ** 2):.1f} MB"

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)
