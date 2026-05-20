"""Diagnostics report dialog for the GeoAI QGIS plugin."""

import os
from datetime import datetime

from qgis.PyQt.QtCore import QThread, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class _DiagnosticsWorker(QThread):
    """Background worker that builds the diagnostics report off the UI thread."""

    finished_report = pyqtSignal(str)

    def run(self):
        try:
            from ..core.diagnostics import generate_diagnostics_report

            report = generate_diagnostics_report()
        except Exception as exc:
            report = f"Failed to generate diagnostics report:\n{exc}"
        self.finished_report.emit(report)


class DiagnosticsDialog(QDialog):
    """Dialog that displays and exports a GeoAI diagnostics report."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GeoAI Diagnostics Report")
        self.resize(900, 650)

        self._worker = None

        layout = QVBoxLayout(self)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        try:
            no_wrap = QTextEdit.LineWrapMode.NoWrap
        except AttributeError:
            no_wrap = QTextEdit.NoWrap
        self.report_text.setLineWrapMode(no_wrap)
        self.report_text.setPlainText("Generating diagnostics report...")
        layout.addWidget(self.report_text)

        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.copy_button = QPushButton("Copy Markdown")
        self.save_button = QPushButton("Save Markdown...")
        self.close_button = QPushButton("Close")

        self.refresh_button.clicked.connect(self.refresh_report)
        self.copy_button.clicked.connect(self.copy_report)
        self.save_button.clicked.connect(self.save_report)
        self.close_button.clicked.connect(self.close)

        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        QTimer.singleShot(0, self.refresh_report)

    def refresh_report(self):
        """Regenerate the diagnostics report on a background thread."""
        if self._worker is not None and self._worker.isRunning():
            return

        self.refresh_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.report_text.setPlainText(
            "Generating diagnostics report... this may take up to 90 seconds while "
            "the managed environment is probed."
        )
        QApplication.processEvents()

        worker = _DiagnosticsWorker()
        worker.finished_report.connect(self._on_report_ready)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_report_ready(self, report):
        """Handle the completed diagnostics report."""
        self.report_text.setPlainText(report)
        self.refresh_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def closeEvent(self, event):
        """Detach from the worker so a late report does not touch a dead dialog."""
        if self._worker is not None and self._worker.isRunning():
            try:
                self._worker.finished_report.disconnect(self._on_report_ready)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)

    def copy_report(self):
        """Copy the diagnostics report to the clipboard."""
        QApplication.clipboard().setText(self.report_text.toPlainText())

    def save_report(self):
        """Save the diagnostics report to a text file."""
        try:
            from ..core.venv_manager import CACHE_DIR

            default_dir = CACHE_DIR
        except Exception:
            default_dir = os.path.expanduser("~")

        default_name = "geoai_diagnostics_{}.md".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GeoAI Diagnostics Report",
            default_path,
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        try:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.report_text.toPlainText())
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Save Diagnostics Report",
                f"Could not save diagnostics report:\n{exc}",
            )
