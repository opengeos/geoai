"""
Moondream Dock Widget for GeoAI Plugin

This dock widget provides an interactive interface for using the Moondream
vision-language model for geospatial image analysis in QGIS.
"""

import os
import tempfile

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QMessageBox,
    QCheckBox,
    QColorDialog,
    QScrollArea,
)
from qgis.PyQt.QtGui import QColor

from qgis.core import (
    QgsProject,
    QgsVectorLayer,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel


class MoondreamWorker(QThread):
    """Worker thread for running Moondream operations."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        moondream,
        mode: str,
        source_path: str,
        prompt: str = "",
        caption_length: str = "normal",
    ):
        super().__init__()
        self.moondream = moondream
        self.mode = mode
        self.source_path = source_path
        self.prompt = prompt
        self.caption_length = caption_length

    def run(self):
        """Execute the Moondream operation."""
        try:
            if self.mode == "Caption":
                self.progress.emit(f"Generating {self.caption_length} caption...")
                result = self.moondream.caption(
                    self.source_path,
                    length=self.caption_length,
                    stream=False,
                )
                self.finished.emit({"type": "caption", "result": result})

            elif self.mode == "Query":
                self.progress.emit(f"Processing query: {self.prompt}")
                result = self.moondream.query(
                    self.prompt,
                    source=self.source_path,
                    stream=False,
                )
                self.finished.emit(
                    {"type": "query", "result": result, "question": self.prompt}
                )

            elif self.mode == "Detect":
                self.progress.emit(f"Detecting: {self.prompt}")
                result = self.moondream.detect(
                    self.source_path,
                    self.prompt,
                )
                self.finished.emit(
                    {"type": "detect", "result": result, "object_type": self.prompt}
                )

            elif self.mode == "Point":
                self.progress.emit(f"Locating: {self.prompt}")
                result = self.moondream.point(
                    self.source_path,
                    self.prompt,
                )
                self.finished.emit(
                    {"type": "point", "result": result, "description": self.prompt}
                )

        except Exception as e:
            self.error.emit(str(e))


class MoondreamDockWidget(QDockWidget):
    """Dockable widget for Moondream vision-language model interaction."""

    def __init__(self, iface, parent=None):
        """Initialize the Moondream dock widget.

        Args:
            iface: QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("GeoAI - Moondream", parent)
        self.iface = iface
        self.moondream = None
        self.current_image_path = None
        self.last_result = None
        self.worker = None

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Set up the user interface."""
        # Main widget with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Consistent height for all input elements (24px)
        self.input_height = 24
        combo_style = f"QComboBox {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"
        line_style = f"QLineEdit {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"
        btn_style = f"QPushButton {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"
        spin_style = f"QSpinBox, QDoubleSpinBox {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"

        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        model_layout.setSpacing(5)

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(combo_style)
        self.model_combo.addItems(
            [
                "vikhyatk/moondream2",
                "moondream/moondream3-preview",
            ]
        )
        model_layout.addRow("Model:", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.setStyleSheet(combo_style)
        self.device_combo.addItems(["Auto", "cuda", "cpu", "mps"])
        self.device_combo.setToolTip(
            'Device to run the model on. "Auto" will select CUDA if available, otherwise CPU.'
        )
        model_layout.addRow("Device:", self.device_combo)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setStyleSheet(btn_style)
        model_layout.addRow("", self.load_model_btn)

        self.model_status = QLabel("Model not loaded")
        self.model_status.setStyleSheet("color: gray;")
        self.model_status.setWordWrap(True)
        model_layout.addRow("Status:", self.model_status)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Image Input Group
        image_group = QGroupBox("Image Input")
        image_layout = QVBoxLayout()
        image_layout.setSpacing(5)

        # Layer selection
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Layer:"))
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setFixedHeight(self.input_height)
        layer_layout.addWidget(self.layer_combo)
        image_layout.addLayout(layer_layout)

        # Or file input
        file_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("Or path to image file...")
        self.image_path_edit.setStyleSheet(line_style)
        file_layout.addWidget(self.image_path_edit)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedSize(30, self.input_height)
        file_layout.addWidget(self.browse_btn)
        image_layout.addLayout(file_layout)

        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.setStyleSheet(btn_style)
        image_layout.addWidget(self.load_image_btn)

        self.image_status = QLabel("No image loaded")
        self.image_status.setStyleSheet("color: gray;")
        self.image_status.setWordWrap(True)
        image_layout.addWidget(self.image_status)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Analysis Settings Group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QFormLayout()
        analysis_layout.setSpacing(5)

        self.mode_combo = QComboBox()
        self.mode_combo.setStyleSheet(combo_style)
        self.mode_combo.addItems(["Caption", "Query", "Detect", "Point"])
        analysis_layout.addRow("Mode:", self.mode_combo)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Enter prompt...")
        self.prompt_edit.setStyleSheet(line_style)
        analysis_layout.addRow("Prompt:", self.prompt_edit)

        self.caption_length_combo = QComboBox()
        self.caption_length_combo.setStyleSheet(combo_style)
        self.caption_length_combo.addItems(["short", "normal", "long"])
        self.caption_length_combo.setCurrentText("normal")
        analysis_layout.addRow("Length:", self.caption_length_combo)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Output Settings Group
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()
        output_layout.setSpacing(5)

        # Color picker
        color_layout = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(25, 25)
        self.result_color = QColor(255, 0, 0)
        self.color_btn.setStyleSheet(f"background-color: {self.result_color.name()};")
        color_layout.addWidget(self.color_btn)
        color_layout.addWidget(QLabel("Result Color"))
        color_layout.addStretch()
        output_layout.addRow("", color_layout)

        self.add_to_map_check = QCheckBox("Add results to map")
        self.add_to_map_check.setChecked(True)
        output_layout.addRow("", self.add_to_map_check)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(btn_style)
        btn_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(btn_style)
        btn_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet(btn_style)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("")
        self.progress_bar.setMaximumHeight(20)
        layout.addWidget(self.progress_bar)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(100)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        self.setWidget(scroll)

        # Set minimum width
        self.setMinimumWidth(320)

        # Initial UI state
        self.update_ui_for_mode()

    def connect_signals(self):
        """Connect widget signals to slots."""
        self.load_model_btn.clicked.connect(self.load_model)
        self.browse_btn.clicked.connect(self.browse_image)
        self.load_image_btn.clicked.connect(self.load_image)
        self.mode_combo.currentTextChanged.connect(self.update_ui_for_mode)
        self.color_btn.clicked.connect(self.pick_color)
        self.run_btn.clicked.connect(self.run_analysis)
        self.save_btn.clicked.connect(self.save_results)
        self.reset_btn.clicked.connect(self.reset)
        self.layer_combo.layerChanged.connect(self.on_layer_changed)

    def update_ui_for_mode(self):
        """Update UI elements based on selected mode."""
        mode = self.mode_combo.currentText()

        if mode == "Caption":
            self.prompt_edit.setEnabled(False)
            self.prompt_edit.setPlaceholderText("Not needed for caption mode")
            self.caption_length_combo.setEnabled(True)
        elif mode == "Query":
            self.prompt_edit.setEnabled(True)
            self.prompt_edit.setPlaceholderText("Ask a question about the image...")
            self.caption_length_combo.setEnabled(False)
        elif mode == "Detect":
            self.prompt_edit.setEnabled(True)
            self.prompt_edit.setPlaceholderText("Object type (e.g., building, car)...")
            self.caption_length_combo.setEnabled(False)
        elif mode == "Point":
            self.prompt_edit.setEnabled(True)
            self.prompt_edit.setPlaceholderText("Object description to locate...")
            self.caption_length_combo.setEnabled(False)

    def load_model(self):
        """Load the Moondream model."""
        try:
            self.model_status.setText("Loading model...")
            self.model_status.setStyleSheet("color: orange;")
            self.progress_bar.setRange(0, 0)
            self.load_model_btn.setEnabled(False)

            from qgis.PyQt.QtWidgets import QApplication

            QApplication.processEvents()

            # Lazy import
            from .._geoai_lib import get_geoai

            geoai = get_geoai()
            MoondreamGeo = geoai.MoondreamGeo

            model_name = self.model_combo.currentText()
            device = self.device_combo.currentText()
            if device == "Auto":
                device = None

            self.moondream = MoondreamGeo(model_name=model_name, device=device)

            self.model_status.setText(f"Loaded: {model_name.split('/')[-1]}")
            self.model_status.setStyleSheet("color: green;")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

            if self.current_image_path:
                self.run_btn.setEnabled(True)

        except Exception as e:
            self.model_status.setText(f"Error: {str(e)[:40]}...")
            self.model_status.setStyleSheet("color: red;")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")

        finally:
            self.load_model_btn.setEnabled(True)

    def browse_image(self):
        """Open file dialog to browse for an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)",
        )
        if file_path:
            self.image_path_edit.setText(file_path)

    def on_layer_changed(self, layer):
        """Handle layer selection change."""
        if layer:
            self.image_path_edit.clear()

    def load_image(self):
        """Load the selected image."""
        try:
            layer = self.layer_combo.currentLayer()
            if layer and not self.image_path_edit.text():
                source = layer.source()
            else:
                source = self.image_path_edit.text()

            if not source:
                QMessageBox.warning(
                    self, "Warning", "Please select a layer or specify an image file."
                )
                return

            if not os.path.exists(source):
                QMessageBox.warning(self, "Warning", f"File not found: {source}")
                return

            self.current_image_path = source

            if self.moondream:
                self.moondream.load_image(source)
                self.run_btn.setEnabled(True)

            self.image_status.setText(f"Loaded: {os.path.basename(source)}")
            self.image_status.setStyleSheet("color: green;")

        except Exception as e:
            self.image_status.setText(f"Error: {str(e)[:40]}...")
            self.image_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def pick_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(self.result_color, self, "Select Color")
        if color.isValid():
            self.result_color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")

    def run_analysis(self):
        """Run the Moondream analysis."""
        if not self.moondream:
            QMessageBox.warning(self, "Warning", "Please load the model first.")
            return

        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        mode = self.mode_combo.currentText()
        prompt = self.prompt_edit.text()

        if mode in ["Query", "Detect", "Point"] and not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a prompt.")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.results_text.clear()

        self.worker = MoondreamWorker(
            self.moondream,
            mode,
            self.current_image_path,
            prompt,
            self.caption_length_combo.currentText(),
        )
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.progress.connect(self.on_analysis_progress)
        self.worker.start()

    def on_analysis_progress(self, message: str):
        """Handle progress updates from worker."""
        self.progress_bar.setFormat(message)

    def on_analysis_finished(self, data: dict):
        """Handle analysis completion."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete")
        self.run_btn.setEnabled(True)

        result_type = data.get("type")
        result = data.get("result", {})
        self.last_result = data

        if result_type == "caption":
            caption = result.get("caption", str(result))
            self.results_text.setPlainText(f"Caption:\n{caption}")
            self.save_btn.setEnabled(True)

        elif result_type == "query":
            question = data.get("question", "")
            answer = result.get("answer", str(result))
            self.results_text.setPlainText(f"Q: {question}\n\nA: {answer}")
            self.save_btn.setEnabled(True)

        elif result_type == "detect":
            object_type = data.get("object_type", "")
            objects = result.get("objects", [])
            gdf = result.get("gdf")

            text = f"Detected '{object_type}': {len(objects)} object(s)"
            if gdf is not None and len(gdf) > 0 and self.add_to_map_check.isChecked():
                text += "\nBounding boxes added to map."
                self.add_detection_layer(gdf, object_type)

            self.results_text.setPlainText(text)
            self.save_btn.setEnabled(gdf is not None and len(gdf) > 0)

        elif result_type == "point":
            description = data.get("description", "")
            points = result.get("points", [])
            gdf = result.get("gdf")

            text = f"Located '{description}': {len(points)} point(s)"
            if gdf is not None and len(gdf) > 0 and self.add_to_map_check.isChecked():
                text += "\nPoints added to map."
                self.add_point_layer(gdf, description)

            self.results_text.setPlainText(text)
            self.save_btn.setEnabled(gdf is not None and len(gdf) > 0)

    def on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error")
        self.run_btn.setEnabled(True)

        self.results_text.setPlainText(f"Error:\n{error_message}")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_message}")

    def add_detection_layer(self, gdf, object_type: str):
        """Add detection results as a vector layer to QGIS."""
        try:
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(
                temp_dir, f"moondream_detect_{object_type.replace(' ', '_')}.geojson"
            )
            gdf.to_file(temp_file, driver="GeoJSON")

            layer = QgsVectorLayer(temp_file, f"Detections: {object_type}", "ogr")
            if layer.isValid():
                symbol = layer.renderer().symbol()
                symbol.setColor(self.result_color)
                symbol.symbolLayer(0).setStrokeColor(self.result_color)
                symbol.symbolLayer(0).setStrokeWidth(1.5)
                symbol.symbolLayer(0).setFillColor(
                    QColor(
                        self.result_color.red(),
                        self.result_color.green(),
                        self.result_color.blue(),
                        50,
                    )
                )
                QgsProject.instance().addMapLayer(layer)
                self.iface.mapCanvas().refresh()

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to add layer:\n{str(e)}")

    def add_point_layer(self, gdf, description: str):
        """Add point results as a vector layer to QGIS."""
        try:
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(
                temp_dir, f"moondream_points_{description.replace(' ', '_')}.geojson"
            )
            gdf.to_file(temp_file, driver="GeoJSON")

            layer = QgsVectorLayer(temp_file, f"Points: {description}", "ogr")
            if layer.isValid():
                symbol = layer.renderer().symbol()
                symbol.setColor(self.result_color)
                symbol.setSize(4)
                QgsProject.instance().addMapLayer(layer)
                self.iface.mapCanvas().refresh()

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to add layer:\n{str(e)}")

    def save_results(self):
        """Save the current results to a file."""
        if not self.last_result:
            QMessageBox.warning(self, "Warning", "No results to save.")
            return

        result_type = self.last_result.get("type")
        result = self.last_result.get("result", {})

        if result_type in ["detect", "point"]:
            gdf = result.get("gdf")
            if gdf is not None and len(gdf) > 0:
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Results",
                    "",
                    "GeoJSON (*.geojson);;Shapefile (*.shp);;GeoPackage (*.gpkg)",
                )
                if file_path:
                    try:
                        if file_path.endswith(".geojson"):
                            gdf.to_file(file_path, driver="GeoJSON")
                        elif file_path.endswith(".shp"):
                            gdf.to_file(file_path, driver="ESRI Shapefile")
                        else:
                            gdf.to_file(file_path)
                        QMessageBox.information(
                            self, "Success", f"Saved to:\n{file_path}"
                        )
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Error", f"Failed to save:\n{str(e)}"
                        )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Results",
                "",
                "Text File (*.txt)",
            )
            if file_path:
                try:
                    with open(file_path, "w") as f:
                        f.write(self.results_text.toPlainText())
                    QMessageBox.information(self, "Success", f"Saved to:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def reset(self):
        """Reset the widget state."""
        self.results_text.clear()
        self.prompt_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.last_result = None
        self.save_btn.setEnabled(False)

    def closeEvent(self, event):
        """Handle widget close event."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()
