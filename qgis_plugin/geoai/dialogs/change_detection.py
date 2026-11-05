"""
Change Detection Dock Widget for GeoAI Plugin

This dock widget provides an interface for detecting changes between two
co-registered images using the AnyChange model (torchange + SAM) via
geoai.change_detection.ChangeDetection.
"""

import os
import traceback

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsFillSymbol,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRasterTransparency,
    QgsVectorLayer,
    Qgis,
)

# AnyChange resizes both images to this size before inference.
ANYCHANGE_TARGET_SIZE = 1024


class ChangeDetectionWorker(QThread):
    """Worker thread for running AnyChange change detection."""

    finished = pyqtSignal(str, str)  # (output_raster_path, output_vector_path)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        """Initialize the change detection worker.

        Args:
            params: Keyword arguments for the ``detect_changes`` task.
        """
        super().__init__()
        self.params = params

    def run(self):
        """Run change detection in background thread."""
        try:
            from ..core.geoai_task_subprocess import run_geoai_task

            result = run_geoai_task(
                "detect_changes",
                self.params,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result["output_raster"], result.get("output_vector", ""))

        except Exception as e:
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ChangeDetectionDockWidget(QDockWidget):
    """Dock widget for bi-temporal change detection using AnyChange."""

    def __init__(self, iface, parent=None):
        """Initialize the Change Detection dock widget.

        Args:
            iface: The QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("Change Detection", parent)
        self.iface = iface
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        header_label = QLabel(
            "<b>Change Detection</b> — "
            '<a href="https://github.com/Z-Zheng/pytorch-change-models">AnyChange</a>'
        )
        header_label.setOpenExternalLinks(True)
        main_layout.addWidget(header_label)

        main_layout.addWidget(self._create_input_group())
        main_layout.addWidget(self._create_model_group())
        main_layout.addWidget(self._create_vectorization_group())
        main_layout.addWidget(self._create_output_group())

        self.run_btn = QPushButton("Run Change Detection")
        self.run_btn.clicked.connect(self.run_detection)
        main_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        log_layout.addWidget(self.results_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        main_layout.addStretch()

        scroll_area.setWidget(main_widget)
        self.setWidget(scroll_area)

    def _create_input_group(self):
        """Create the before/after layer selection group.

        Returns:
            QGroupBox with input selection controls.
        """
        group = QGroupBox("Input")
        layout = QVBoxLayout()

        self.before_combo = QComboBox()
        self.after_combo = QComboBox()

        for label, combo in [
            ("Before:", self.before_combo),
            ("After:", self.after_combo),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(combo)
            layout.addLayout(row)

        refresh_btn = QPushButton("Refresh layers")
        refresh_btn.clicked.connect(self.refresh_layers)
        layout.addWidget(refresh_btn)

        note = QLabel(
            "Both images must share the same CRS. Only the first three bands are used."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

        self.refresh_layers()
        group.setLayout(layout)
        return group

    def _create_model_group(self):
        """Create the AnyChange model settings group.

        Returns:
            QGroupBox with model parameter controls.
        """
        group = QGroupBox("Model Settings")
        layout = QVBoxLayout()

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("SAM Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["vit_b", "vit_l", "vit_h"])
        self.model_combo.setToolTip(
            "SAM backbone. vit_b is the smallest and fastest on CPU; "
            "vit_h is the largest and slowest. AnyChange is zero-shot, so "
            "expect coarse change masks. Weights are downloaded on first use "
            "(vit_b ~375 MB, vit_h ~2.5 GB)."
        )
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Change Angle (°):"))
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(0, 180)
        self.angle_spin.setValue(145)
        self.angle_spin.setToolTip(
            "AnyChange change confidence threshold in degrees. Higher values "
            "detect more changes; lower values keep only the strongest ones."
        )
        angle_row.addWidget(self.angle_spin)
        layout.addLayout(angle_row)

        self.auto_threshold_check = QCheckBox("Auto threshold")
        self.auto_threshold_check.setToolTip(
            "Let AnyChange pick the change angle from the image pair."
        )
        self.auto_threshold_check.toggled.connect(
            lambda checked: self.angle_spin.setEnabled(not checked)
        )
        layout.addWidget(self.auto_threshold_check)

        group.setLayout(layout)
        return group

    def _create_vectorization_group(self):
        """Create the vectorization options group.

        Returns:
            QGroupBox with vectorization controls.
        """
        group = QGroupBox("Vectorization (Optional)")
        layout = QVBoxLayout()

        self.vectorize_check = QCheckBox("Generate change polygons")
        self.vectorize_check.toggled.connect(self._on_vectorize_toggled)
        layout.addWidget(self.vectorize_check)

        min_area_row = QHBoxLayout()
        min_area_row.addWidget(QLabel("Min Area (map units²):"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 1000000)
        self.min_area_spin.setValue(0)
        self.min_area_spin.setToolTip(
            "Drop polygons smaller than this. Square metres if the layer CRS is "
            "geographic, otherwise square map units."
        )
        min_area_row.addWidget(self.min_area_spin)
        self.min_area_widget = QWidget()
        min_area_row.setContentsMargins(0, 0, 0, 0)
        self.min_area_widget.setLayout(min_area_row)
        self.min_area_widget.setEnabled(False)
        layout.addWidget(self.min_area_widget)

        group.setLayout(layout)
        return group

    def _create_output_group(self):
        """Create the output path selection group.

        Returns:
            QGroupBox with output path controls.
        """
        group = QGroupBox("Output")
        layout = QVBoxLayout()

        raster_row = QHBoxLayout()
        raster_row.addWidget(QLabel("Raster:"))
        self.output_raster_edit = QLineEdit()
        self.output_raster_edit.setPlaceholderText("Change mask output path (.tif)")
        raster_row.addWidget(self.output_raster_edit)
        raster_browse_btn = QPushButton("...")
        raster_browse_btn.setMaximumWidth(30)
        raster_browse_btn.clicked.connect(self.browse_output_raster)
        raster_row.addWidget(raster_browse_btn)
        layout.addLayout(raster_row)

        self.vector_output_row = QWidget()
        vec_out_layout = QHBoxLayout()
        vec_out_layout.setContentsMargins(0, 0, 0, 0)
        vec_out_layout.addWidget(QLabel("Vector:"))
        self.output_vector_edit = QLineEdit()
        self.output_vector_edit.setPlaceholderText(
            "Change polygons output (.gpkg, .geojson, .shp)"
        )
        vec_out_layout.addWidget(self.output_vector_edit)
        vector_browse_btn = QPushButton("...")
        vector_browse_btn.setMaximumWidth(30)
        vector_browse_btn.clicked.connect(self.browse_output_vector)
        vec_out_layout.addWidget(vector_browse_btn)
        self.vector_output_row.setLayout(vec_out_layout)
        self.vector_output_row.setEnabled(False)
        layout.addWidget(self.vector_output_row)

        self.add_to_map_check = QCheckBox("Add results to map")
        self.add_to_map_check.setChecked(True)
        layout.addWidget(self.add_to_map_check)

        group.setLayout(layout)
        return group

    # ---- Slot methods ----

    def _on_vectorize_toggled(self, checked):
        """Enable/disable vectorization settings and vector output path.

        Args:
            checked: Whether vectorization is enabled.
        """
        self.min_area_widget.setEnabled(checked)
        self.vector_output_row.setEnabled(checked)

    def refresh_layers(self):
        """Refresh the list of raster layers in both combo boxes."""
        for combo in (self.before_combo, self.after_combo):
            combo.clear()
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsRasterLayer):
                    combo.addItem(layer.name(), layer.id())
        if self.after_combo.count() > 1:
            self.after_combo.setCurrentIndex(1)

    def browse_output_raster(self):
        """Browse for output raster file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Change Mask Raster",
            "",
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if file_path:
            self.output_raster_edit.setText(file_path)

    def browse_output_vector(self):
        """Browse for output vector file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Change Polygons",
            "",
            "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;Shapefile (*.shp);;All files (*.*)",
        )
        if file_path:
            self.output_vector_edit.setText(file_path)

    # ---- Run detection ----

    def _selected_layers(self):
        """Return the before/after layers if AnyChange can use them.

        Returns:
            Tuple of (before_layer, after_layer).

        Raises:
            ValueError: If the layers cannot be compared.
        """
        project = QgsProject.instance()
        before = project.mapLayer(self.before_combo.currentData() or "")
        after = project.mapLayer(self.after_combo.currentData() or "")

        if before is None or after is None:
            raise ValueError("Please select both a before and an after layer.")
        if before.id() == after.id():
            raise ValueError("Before and after layers must be different.")
        # ChangeDetection intersects the two extents without reprojecting.
        if before.crs() != after.crs():
            raise ValueError(
                f"Before layer is in {before.crs().userFriendlyIdentifier()} but after "
                f"layer is in {after.crs().userFriendlyIdentifier()}. Reproject one "
                "so both use the same CRS."
            )
        for layer in (before, after):
            if not os.path.isfile(layer.source()):
                raise ValueError(
                    f"'{layer.name()}' is not a raster file on disk. "
                    "Export it to GeoTIFF first."
                )
        if before.extent().intersect(after.extent()).isEmpty():
            raise ValueError("Before and after layers do not overlap.")
        return before, after

    def run_detection(self):
        """Validate inputs and run change detection in a worker thread."""
        if self.worker is not None and self.worker.isRunning():
            self.show_error("Change detection is already running.")
            return

        try:
            before, after = self._selected_layers()
        except ValueError as e:
            self.show_error(str(e))
            return

        # ChangeDetection reads the overlap at the before layer's resolution.
        overlap = before.extent().intersect(after.extent())
        overlap_px = round(
            max(
                overlap.width() / before.rasterUnitsPerPixelX(),
                overlap.height() / before.rasterUnitsPerPixelY(),
            )
        )
        if overlap_px > ANYCHANGE_TARGET_SIZE:
            reply = QMessageBox.question(
                self,
                "Large Raster",
                f"The overlapping area is {overlap_px} pixels across. AnyChange "
                f"resizes it to {ANYCHANGE_TARGET_SIZE}×{ANYCHANGE_TARGET_SIZE} "
                "pixels, so small changes may be lost.\n\nContinue?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        base = os.path.splitext(after.source())[0]
        output_raster = self.output_raster_edit.text().strip()
        if not output_raster:
            output_raster = f"{base}_changes.tif"
            self.output_raster_edit.setText(output_raster)

        output_vector = ""
        if self.vectorize_check.isChecked():
            output_vector = self.output_vector_edit.text().strip()
            if not output_vector:
                output_vector = f"{base}_changes.gpkg"
                self.output_vector_edit.setText(output_vector)

        inputs = {
            os.path.normcase(os.path.abspath(layer.source()))
            for layer in (before, after)
        }
        for path in (output_raster, output_vector):
            if path and os.path.normcase(os.path.abspath(path)) in inputs:
                self.show_error(
                    "Output paths must differ from the before and after images."
                )
                return

        params = {
            "before_path": before.source(),
            "after_path": after.source(),
            "output_raster": output_raster,
            "output_vector": output_vector,
            "sam_model_type": self.model_combo.currentText(),
            "change_confidence_threshold": self.angle_spin.value(),
            "auto_threshold": self.auto_threshold_check.isChecked(),
            "min_area": self.min_area_spin.value(),
        }

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running change detection...")
        self.status_label.setStyleSheet("color: orange;")
        self.results_text.clear()

        self.worker = ChangeDetectionWorker(params)
        self.worker.finished.connect(self._on_detection_finished)
        self.worker.error.connect(self._on_detection_error)
        self.worker.progress.connect(self._on_detection_progress)
        self.worker.start()

    def _on_detection_finished(self, raster_path, vector_path):
        """Handle successful change detection.

        Args:
            raster_path: Path to the output change mask raster.
            vector_path: Path to the output change polygons (may be empty).
        """
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Change detection complete!")
        self.status_label.setStyleSheet("color: green;")

        self.results_text.append(f"Raster output: {raster_path}")
        if vector_path:
            self.results_text.append(f"Vector output: {vector_path}")

        if self.add_to_map_check.isChecked():
            raster_layer = QgsRasterLayer(raster_path, "Change Mask")
            if raster_layer.isValid():
                self._apply_change_raster_style(raster_layer)
                QgsProject.instance().addMapLayer(raster_layer)
            else:
                self.log_message(
                    f"Failed to load raster layer: {raster_path}", level=Qgis.Warning
                )

            if vector_path:
                vector_layer = QgsVectorLayer(vector_path, "Changes", "ogr")
                if vector_layer.isValid():
                    self._apply_change_vector_style(vector_layer)
                    QgsProject.instance().addMapLayer(vector_layer)
                else:
                    self.log_message(
                        f"Failed to load vector layer: {vector_path}",
                        level=Qgis.Warning,
                    )

        self.log_message("Change detection completed successfully.")

    def _on_detection_error(self, error_message):
        """Handle change detection error.

        Args:
            error_message: The error message string.
        """
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Change detection failed")
        self.status_label.setStyleSheet("color: red;")
        self.results_text.append(f"Error: {error_message}")
        self.show_error(f"Change detection failed:\n{error_message}")

    def _on_detection_progress(self, message):
        """Handle progress updates from the worker.

        Args:
            message: Progress message string.
        """
        self.status_label.setText(message)
        self.results_text.append(message)

    # ---- Result styling ----

    def _apply_change_raster_style(self, layer):
        """Make unchanged pixels (value 0) transparent.

        Args:
            layer: The QgsRasterLayer to style.
        """
        renderer = layer.renderer()
        if renderer is None:
            return
        transparency = QgsRasterTransparency()
        tr_pixel = QgsRasterTransparency.TransparentSingleValuePixel()
        tr_pixel.min = 0.0
        tr_pixel.max = 0.0
        tr_pixel.percentTransparent = 100.0
        transparency.setTransparentSingleValuePixelList([tr_pixel])
        renderer.setRasterTransparency(transparency)
        renderer.setOpacity(0.7)
        layer.triggerRepaint()

    def _apply_change_vector_style(self, layer):
        """Apply a semi-transparent red style to change polygons.

        Args:
            layer: The QgsVectorLayer to style.
        """
        symbol = QgsFillSymbol.createSimple(
            {
                "color": "230,50,50,80",
                "outline_color": "200,0,0,255",
                "outline_width": "0.4",
            }
        )
        layer.renderer().setSymbol(symbol)
        layer.triggerRepaint()

    # ---- Utility methods ----

    def show_error(self, message):
        """Show an error message dialog.

        Args:
            message: The error message to display.
        """
        QMessageBox.critical(self, "Change Detection Error", message)
        self.log_message(message, level=Qgis.Critical)

    def log_message(self, message, level=Qgis.Info):
        """Log a message to the QGIS message log.

        Args:
            message: The message to log.
            level: The log level (default: Qgis.Info).
        """
        QgsMessageLog.logMessage(message, "GeoAI - Change Detection", level)

    def cleanup(self):
        """Clean up resources when the dock is closed."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

    def closeEvent(self, event):
        """Handle dock widget close event.

        Args:
            event: The close event.
        """
        self.cleanup()
        super().closeEvent(event)
