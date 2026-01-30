"""
DeepForest Dock Widget for GeoAI Plugin

This dock widget provides an interface for tree crown detection and forest analysis
using the DeepForest library with pretrained models for various detection tasks.
"""

import os
import json
import tempfile

try:
    import torch
except ImportError:
    torch = None

from qgis.PyQt.QtCore import Qt, QCoreApplication
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QTextEdit,
    QScrollArea,
)
from qgis.core import (
    QgsFillSymbol,
    QgsProject,
    QgsRasterFileWriter,
    QgsRasterLayer,
    QgsRasterPipe,
    QgsVectorLayer,
    Qgis,
    QgsMessageLog,
)

from qgis.PyQt.QtCore import QThread, pyqtSignal


class DeepForestModelLoadWorker(QThread):
    """Worker thread for loading DeepForest model."""

    finished = pyqtSignal(object, str)  # Emits (model, model_name)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model_name: str, revision: str, device: str):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self.device = device

    def run(self):
        """Load the DeepForest model in background."""
        try:
            self.progress.emit("Importing DeepForest...")
            from deepforest import main

            self.progress.emit("Initializing DeepForest model...")
            model = main.deepforest()

            self.progress.emit(f"Loading pretrained model: {self.model_name}...")
            model.load_model(model_name=self.model_name, revision=self.revision)

            # Move to appropriate device if CUDA is available and requested
            if self.device and self.device != "auto":
                if (
                    self.device == "cuda"
                    and torch is not None
                    and torch.cuda.is_available()
                ):
                    try:
                        model.model.to("cuda")
                    except Exception as e:
                        self.progress.emit(
                            f"Warning: Could not move model to CUDA: {e}"
                        )
                elif self.device == "cpu":
                    try:
                        model.model.to("cpu")
                    except Exception:
                        pass  # CPU is default

            self.finished.emit(model, self.model_name)

        except Exception as e:
            self.error.emit(str(e))


class DeepForestDockWidget(QDockWidget):
    """Dock widget for DeepForest tree detection operations."""

    def __init__(self, iface, parent=None):
        """Initialize the DeepForest dock widget.

        Args:
            iface: The QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("DeepForest Tree Detection", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # DeepForest model instance
        self.deepforest = None
        self.current_layer = None
        self.current_image_path = None

        # Store predictions and mode
        self.predictions = None
        self.prediction_mode = None  # "single" or "tile"

        # Track temporary files for cleanup
        self._temp_files = []

        # Model loading worker
        self.model_load_worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Header with link to DeepForest repo
        header_label = QLabel(
            "<b>DeepForest</b> — "
            '<a href="https://github.com/weecology/DeepForest">GitHub</a> · '
            '<a href="https://deepforest.readthedocs.io">Docs</a>'
        )
        header_label.setOpenExternalLinks(True)
        main_layout.addWidget(header_label)

        # Tab widget for different modes
        self.tab_widget = QTabWidget()

        # === Model Settings Tab ===
        model_tab = self._create_model_tab()
        self.tab_widget.addTab(model_tab, "Model")

        # === Predict Tab ===
        predict_tab = self._create_predict_tab()
        self.tab_widget.addTab(predict_tab, "Predict")

        # === Output Tab ===
        output_tab = self._create_output_tab()
        self.tab_widget.addTab(output_tab, "Output")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        scroll_area.setWidget(main_widget)
        self.setWidget(scroll_area)

    def _create_model_tab(self):
        """Create the model settings tab."""
        model_tab = QWidget()
        model_layout = QVBoxLayout()
        model_tab.setLayout(model_layout)

        # Model selection
        model_group = QGroupBox("Model Settings")
        model_layout_inner = QVBoxLayout()

        # Model selection dropdown
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            [
                "weecology/deepforest-tree",
                "weecology/deepforest-bird",
                "weecology/deepforest-livestock",
                "weecology/everglades-nest-detection",
                "weecology/cropmodel-deadtrees",
            ]
        )
        model_row.addWidget(self.model_combo)
        model_layout_inner.addLayout(model_row)

        # Model revision
        revision_row = QHBoxLayout()
        revision_row.addWidget(QLabel("Revision:"))
        self.revision_edit = QLineEdit()
        self.revision_edit.setText("main")
        revision_row.addWidget(self.revision_edit)
        model_layout_inner.addLayout(revision_row)

        # Device selection
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        device_row.addWidget(self.device_combo)
        model_layout_inner.addLayout(device_row)

        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout_inner.addWidget(self.load_model_btn)

        # Model status
        self.model_status = QLabel("Model: Not loaded")
        self.model_status.setStyleSheet("color: gray;")
        model_layout_inner.addWidget(self.model_status)

        model_group.setLayout(model_layout_inner)
        model_layout.addWidget(model_group)

        # Input Layer section
        layer_group = QGroupBox("Input Layer")
        layer_layout = QVBoxLayout()

        # Layer selection
        layer_row = QHBoxLayout()
        self.layer_combo = QComboBox()
        self.refresh_layers()
        layer_row.addWidget(self.layer_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self.refresh_layers)
        layer_row.addWidget(refresh_btn)
        layer_layout.addLayout(layer_row)

        self.set_layer_btn = QPushButton("Set Image from Layer")
        self.set_layer_btn.clicked.connect(self.set_image_from_layer)
        layer_layout.addWidget(self.set_layer_btn)

        # Or load from file
        file_row = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("Or select image file...")
        file_row.addWidget(self.image_path_edit)

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self.browse_image)
        file_row.addWidget(browse_btn)
        layer_layout.addLayout(file_row)

        self.set_file_btn = QPushButton("Set Image from File")
        self.set_file_btn.clicked.connect(self.set_image_from_file)
        layer_layout.addWidget(self.set_file_btn)

        self.image_status = QLabel("Image: Not set")
        self.image_status.setStyleSheet("color: gray;")
        layer_layout.addWidget(self.image_status)

        layer_group.setLayout(layer_layout)
        model_layout.addWidget(layer_group)

        model_layout.addStretch()
        return model_tab

    def _create_predict_tab(self):
        """Create the prediction tab."""
        predict_tab = QWidget()
        predict_layout = QVBoxLayout()
        predict_tab.setLayout(predict_layout)

        # Prediction mode
        mode_group = QGroupBox("Prediction Mode")
        mode_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Image", "Large Tile"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_layout.addLayout(mode_row)

        mode_group.setLayout(mode_layout)
        predict_layout.addWidget(mode_group)

        # Large Tile settings (initially hidden)
        self.tile_settings_group = QGroupBox("Large Tile Settings")
        tile_settings_layout = QVBoxLayout()

        # Patch size
        patch_size_row = QHBoxLayout()
        patch_size_row.addWidget(QLabel("Patch Size:"))
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(64, 2048)
        self.patch_size_spin.setValue(400)
        self.patch_size_spin.setSingleStep(64)
        patch_size_row.addWidget(self.patch_size_spin)
        tile_settings_layout.addLayout(patch_size_row)

        # Patch overlap
        patch_overlap_row = QHBoxLayout()
        patch_overlap_row.addWidget(QLabel("Patch Overlap:"))
        self.patch_overlap_spin = QDoubleSpinBox()
        self.patch_overlap_spin.setRange(0.0, 0.9)
        self.patch_overlap_spin.setValue(0.25)
        self.patch_overlap_spin.setSingleStep(0.05)
        patch_overlap_row.addWidget(self.patch_overlap_spin)
        tile_settings_layout.addLayout(patch_overlap_row)

        # IoU threshold
        iou_threshold_row = QHBoxLayout()
        iou_threshold_row.addWidget(QLabel("IoU Threshold:"))
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.0, 1.0)
        self.iou_threshold_spin.setValue(0.15)
        self.iou_threshold_spin.setSingleStep(0.05)
        iou_threshold_row.addWidget(self.iou_threshold_spin)
        tile_settings_layout.addLayout(iou_threshold_row)

        # Dataloader strategy
        dataloader_row = QHBoxLayout()
        dataloader_row.addWidget(QLabel("Dataloader Strategy:"))
        self.dataloader_combo = QComboBox()
        self.dataloader_combo.addItems(["single", "batch", "window"])
        dataloader_row.addWidget(self.dataloader_combo)
        tile_settings_layout.addLayout(dataloader_row)

        self.tile_settings_group.setLayout(tile_settings_layout)
        self.tile_settings_group.setVisible(False)
        predict_layout.addWidget(self.tile_settings_group)

        # Score threshold filter
        filter_group = QGroupBox("Filtering")
        filter_layout = QVBoxLayout()

        score_threshold_row = QHBoxLayout()
        score_threshold_row.addWidget(QLabel("Score Threshold:"))
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.0, 1.0)
        self.score_threshold_spin.setValue(0.3)
        self.score_threshold_spin.setSingleStep(0.05)
        score_threshold_row.addWidget(self.score_threshold_spin)
        filter_layout.addLayout(score_threshold_row)

        filter_group.setLayout(filter_layout)
        predict_layout.addWidget(filter_group)

        # Run prediction button
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.clicked.connect(self.run_prediction)
        predict_layout.addWidget(self.predict_btn)

        # Results status
        self.predict_status_label = QLabel("")
        predict_layout.addWidget(self.predict_status_label)

        predict_layout.addStretch()
        return predict_tab

    def _create_output_tab(self):
        """Create the output settings tab."""
        output_tab = QWidget()
        output_layout = QVBoxLayout()
        output_tab.setLayout(output_layout)

        # Output format
        format_group = QGroupBox("Output Format")
        format_layout = QVBoxLayout()

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(
            [
                "Vector (GeoPackage)",
                "Vector (Shapefile)",
                "Vector (GeoJSON)",
                "Raster (GeoTIFF)",
            ]
        )
        format_row.addWidget(self.output_format_combo)
        format_layout.addLayout(format_row)

        # Add to map checkbox
        self.add_to_map_check = QCheckBox("Add to map")
        self.add_to_map_check.setChecked(True)
        format_layout.addWidget(self.add_to_map_check)

        # Auto-show results after prediction
        self.auto_show_check = QCheckBox("Auto-show results after prediction")
        self.auto_show_check.setChecked(True)
        self.auto_show_check.setToolTip(
            "Automatically save and display results on the map after running prediction"
        )
        format_layout.addWidget(self.auto_show_check)

        # Output path
        output_path_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output file path (optional)...")
        output_path_row.addWidget(self.output_path_edit)

        output_browse_btn = QPushButton("...")
        output_browse_btn.setMaximumWidth(30)
        output_browse_btn.clicked.connect(self.browse_output)
        output_path_row.addWidget(output_browse_btn)
        format_layout.addLayout(output_path_row)

        format_group.setLayout(format_layout)
        output_layout.addWidget(format_group)

        # Save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        output_layout.addWidget(self.save_btn)

        # Export Training Data
        export_group = QGroupBox("Export Training Data")
        export_layout = QVBoxLayout()

        # Export format
        export_format_row = QHBoxLayout()
        export_format_row.addWidget(QLabel("Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["COCO", "YOLO"])
        export_format_row.addWidget(self.export_format_combo)
        export_layout.addLayout(export_format_row)

        # Output directory
        export_dir_row = QHBoxLayout()
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("Output directory...")
        export_dir_row.addWidget(self.export_dir_edit)

        export_browse_btn = QPushButton("...")
        export_browse_btn.setMaximumWidth(30)
        export_browse_btn.clicked.connect(self.browse_export_dir)
        export_dir_row.addWidget(export_browse_btn)
        export_layout.addLayout(export_dir_row)

        # Export button
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_training_data)
        export_layout.addWidget(self.export_btn)

        export_group.setLayout(export_layout)
        output_layout.addWidget(export_group)

        # Results text area
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        output_layout.addWidget(results_group)

        output_layout.addStretch()
        return output_tab

    def _on_mode_changed(self, mode):
        """Handle prediction mode change."""
        is_tile_mode = mode == "Large Tile"
        self.tile_settings_group.setVisible(is_tile_mode)

    def refresh_layers(self):
        """Refresh the list of raster layers."""
        self.layer_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.layer_combo.addItem(layer.name(), layer.id())

    def browse_image(self):
        """Browse for an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.tif *.tiff *.jpg *.jpeg *.png);;All files (*.*)",
        )
        if file_path:
            self.image_path_edit.setText(file_path)

    def browse_output(self):
        """Browse for output file location."""
        format_text = self.output_format_combo.currentText()
        if "GeoJSON" in format_text:
            filter_str = "GeoJSON (*.geojson)"
        elif "GeoPackage" in format_text:
            filter_str = "GeoPackage (*.gpkg)"
        elif "Shapefile" in format_text:
            filter_str = "Shapefile (*.shp)"
        else:
            filter_str = "GeoTIFF (*.tif)"

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Output", "", filter_str)
        if file_path:
            self.output_path_edit.setText(file_path)

    def browse_export_dir(self):
        """Browse for export directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory", "")
        if dir_path:
            self.export_dir_edit.setText(dir_path)

    def check_cuda_devices(self):
        """Check CUDA device availability and fix CUDA_VISIBLE_DEVICES if needed.

        Returns:
            tuple: (is_cuda_available: bool, warning_message: str or None)
        """
        if torch is None:
            return (
                False,
                "PyTorch is not installed. Please install PyTorch to use CUDA acceleration.",
            )

        # Check if CUDA is available
        try:
            cuda_available = torch.cuda.is_available()

            if not cuda_available:
                # Check if CUDA_VISIBLE_DEVICES is the issue
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

                if cuda_visible is not None:
                    # Log the issue
                    self.log_message(
                        f"CUDA not available. CUDA_VISIBLE_DEVICES is set to: {cuda_visible}"
                    )

                    # Try to fix by resetting to device 0
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        self.log_message(
                            "Reset CUDA_VISIBLE_DEVICES to '0', checking again..."
                        )

                        # Force PyTorch to reinitialize CUDA
                        if hasattr(torch.cuda, "_lazy_init"):
                            torch.cuda._lazy_init()

                        cuda_available = torch.cuda.is_available()

                        if cuda_available:
                            device_count = torch.cuda.device_count()
                            device_name = (
                                torch.cuda.get_device_name(0)
                                if device_count > 0
                                else "Unknown"
                            )
                            warning_msg = (
                                f"Fixed CUDA issue: Reset CUDA_VISIBLE_DEVICES from '{cuda_visible}' to '0'. "
                                f"Detected {device_count} GPU(s): {device_name}"
                            )
                            self.log_message(warning_msg)
                            return True, warning_msg
                        else:
                            warning_msg = (
                                f"CUDA_VISIBLE_DEVICES was set to '{cuda_visible}' but you may only have GPU 0. "
                                f"Attempted to reset to '0' but CUDA still not available. "
                                f"Try restarting QGIS or set CUDA_VISIBLE_DEVICES=0 before launching QGIS."
                            )
                            return False, warning_msg
                    except Exception as e:
                        warning_msg = f"Failed to reset CUDA_VISIBLE_DEVICES: {str(e)}"
                        self.log_message(warning_msg)
                        return False, warning_msg
                else:
                    warning_msg = (
                        "CUDA is not available. Possible reasons:\n"
                        "1. No NVIDIA GPU detected\n"
                        "2. CUDA drivers not installed\n"
                        "3. PyTorch not compiled with CUDA support\n"
                        "Please check your CUDA installation or use CPU mode."
                    )
                    return False, warning_msg
            else:
                # CUDA is available
                device_count = torch.cuda.device_count()
                device_name = (
                    torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                )
                self.log_message(
                    f"CUDA available: {device_count} GPU(s) detected - {device_name}"
                )
                return True, None

        except Exception as e:
            error_msg = f"Error checking CUDA availability: {str(e)}"
            self.log_message(error_msg)
            return False, error_msg

    def load_model(self):
        """Load the DeepForest model asynchronously."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.model_status.setText("Loading model...")
        self.model_status.setStyleSheet("color: orange;")
        self.load_model_btn.setEnabled(False)

        model_name = self.model_combo.currentText()
        revision = self.revision_edit.text().strip() or "main"
        device = self.device_combo.currentText()

        if device == "auto":
            device = None

        # Check CUDA availability if using CUDA or auto device selection
        if device == "cuda" or device is None:
            cuda_available, warning_message = self.check_cuda_devices()

            if not cuda_available:
                if device == "cuda":
                    # User explicitly requested CUDA but it's not available
                    self.progress_bar.setVisible(False)
                    self.load_model_btn.setEnabled(True)
                    error_msg = (
                        f"CUDA device requested but not available.\n\n{warning_message}\n\n"
                        "Please select 'cpu' from the Device dropdown or fix your CUDA installation."
                    )
                    self.show_error(error_msg)
                    self.model_status.setText("Model: Failed to load")
                    self.model_status.setStyleSheet("color: red;")
                    return
                else:
                    # Auto mode - fall back to CPU
                    device = "cpu"
                    self.log_message(
                        f"Auto mode: CUDA not available, using CPU. Reason: {warning_message}"
                    )
                    QMessageBox.information(
                        self,
                        "Using CPU Mode",
                        f"CUDA is not available. Automatically using CPU mode.\n\n{warning_message}",
                    )
            elif warning_message:
                # CUDA is now available but there was a warning
                QMessageBox.information(self, "CUDA Issue Fixed", warning_message)

        # Create and start the model loading worker thread
        self.model_load_worker = DeepForestModelLoadWorker(model_name, revision, device)
        self.model_load_worker.finished.connect(self.on_model_loaded)
        self.model_load_worker.error.connect(self.on_model_load_error)
        self.model_load_worker.progress.connect(self.on_model_load_progress)
        self.model_load_worker.start()

    def on_model_load_progress(self, message: str):
        """Handle model loading progress updates."""
        self.model_status.setText(message)

    def on_model_loaded(self, model, model_name: str):
        """Handle successful model loading."""
        self.deepforest = model
        self.model_status.setText(f"Model: {model_name} loaded")
        self.model_status.setStyleSheet("color: green;")
        self.progress_bar.setVisible(False)
        self.load_model_btn.setEnabled(True)
        self.log_message(f"{model_name} model loaded successfully")

    def on_model_load_error(self, error_message: str):
        """Handle model loading error."""
        self.model_status.setText("Model: Failed to load")
        self.model_status.setStyleSheet("color: red;")
        self.progress_bar.setVisible(False)
        self.load_model_btn.setEnabled(True)
        self.show_error(f"Failed to load model: {error_message}")

    def _is_geopackage_raster(self, source):
        """Check if the layer source is a raster inside a GeoPackage."""
        if source.upper().startswith("GPKG:"):
            return True
        if ".gpkg" in source.lower() and "|" in source:
            after_pipe = source.split("|", 1)[1]
            if after_pipe.startswith("layername=") or "layername=" in after_pipe:
                return True
        if ".gpkg" in source.lower():
            gpkg_path = source.split("|", 1)[0]
            if gpkg_path.lower().endswith(".gpkg") and os.path.exists(gpkg_path):
                return True
        return False

    def _export_geopackage_raster(self, layer):
        """Export a GeoPackage raster layer to a temporary GeoTIFF file."""
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False, prefix="deepforest_gpkg_"
            )
            temp_path = temp_file.name
            temp_file.close()

            # Set up the raster pipe
            pipe = QgsRasterPipe()
            provider = layer.dataProvider()

            if not pipe.set(provider.clone()):
                self.log_message(
                    "Failed to set up raster pipe for GeoPackage export",
                    level=Qgis.Warning,
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None

            # Create the file writer
            file_writer = QgsRasterFileWriter(temp_path)
            file_writer.setOutputFormat("GTiff")

            # Write the raster
            error = file_writer.writeRaster(
                pipe,
                provider.xSize(),
                provider.ySize(),
                provider.extent(),
                provider.crs(),
            )

            if error == QgsRasterFileWriter.NoError:
                self.log_message(f"Exported GeoPackage raster to: {temp_path}")
                # Track temp file for cleanup
                self._temp_files.append(temp_path)
                return temp_path
            else:
                self.log_message(
                    f"Failed to export GeoPackage raster: error code {error}",
                    level=Qgis.Warning,
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None

        except Exception as e:
            self.log_message(
                f"Exception exporting GeoPackage raster: {str(e)}", level=Qgis.Warning
            )
            return None

    def set_image_from_layer(self):
        """Set the image from the selected QGIS layer."""
        if self.deepforest is None:
            self.show_error("Please load the model first.")
            return

        layer_id = self.layer_combo.currentData()
        if not layer_id:
            self.show_error("Please select a raster layer.")
            return

        layer = QgsProject.instance().mapLayer(layer_id)
        if not layer:
            self.show_error("Layer not found.")
            return

        # Get the layer's file path
        source = layer.source()

        # Check if this is a GeoPackage raster
        is_gpkg = self._is_geopackage_raster(source)
        temp_export_path = None

        if is_gpkg:
            # Export GeoPackage raster to a temporary file
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.image_status.setText("Exporting GeoPackage raster...")
            QCoreApplication.processEvents()

            temp_export_path = self._export_geopackage_raster(layer)
            if temp_export_path is None:
                self.progress_bar.setVisible(False)
                self.image_status.setText("Image: Failed to export")
                self.image_status.setStyleSheet("color: red;")
                self.show_error(
                    "Failed to export GeoPackage raster. "
                    "Try exporting the layer to a GeoTIFF file manually."
                )
                return
            image_path = temp_export_path
        else:
            # Regular file path
            if not os.path.exists(source):
                self.show_error(f"Layer source file not found: {source}")
                return
            image_path = source

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.image_status.setText("Setting image...")
            QCoreApplication.processEvents()

            self.current_layer = layer
            self.current_image_path = image_path

            # Build status message
            status_msg = f"Image: {layer.name()}"
            if is_gpkg:
                status_msg += " (from GeoPackage)"
            self.image_status.setText(status_msg)
            self.image_status.setStyleSheet("color: green;")

            log_msg = f"Image set from layer: {layer.name()}"
            if is_gpkg:
                log_msg += f" (exported from GeoPackage to {image_path})"
            self.log_message(log_msg)

        except Exception as e:
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            # Clean up temp file if export succeeded but set_image failed
            if temp_export_path and os.path.exists(temp_export_path):
                try:
                    os.remove(temp_export_path)
                    if hasattr(self, "_temp_files"):
                        try:
                            self._temp_files.remove(temp_export_path)
                        except ValueError:
                            pass  # File was not tracked; safe to ignore
                except Exception as cleanup_error:
                    self.log_message(
                        f"Failed to clean up temporary export file '{temp_export_path}': {cleanup_error}",
                        level=Qgis.Warning,
                    )
            self.show_error(f"Failed to set image: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def set_image_from_file(self):
        """Set the image from the file path."""
        if self.deepforest is None:
            self.show_error("Please load the model first.")
            return

        file_path = self.image_path_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.show_error("Please select a valid image file.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.image_status.setText("Setting image...")
            QCoreApplication.processEvents()

            # Optionally add the layer to the map
            layer = QgsRasterLayer(file_path, os.path.basename(file_path))
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.current_layer = layer
                self.current_image_path = file_path

                status_msg = f"Image: {os.path.basename(file_path)}"
                self.image_status.setText(status_msg)
                self.image_status.setStyleSheet("color: green;")

                self.log_message(f"Image set from file: {file_path}")
                self.refresh_layers()
            else:
                self.current_layer = None
                self.current_image_path = None
                self.image_status.setText("Image: Failed to set (invalid layer)")
                self.image_status.setStyleSheet("color: red;")
                self.show_error(
                    "Failed to add image layer: The raster layer is invalid."
                )

        except Exception as e:
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            self.show_error(f"Failed to set image: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def run_prediction(self):
        """Run DeepForest prediction on the current image."""
        if self.deepforest is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.predict_status_label.setText("Running prediction...")
            self.predict_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            mode = self.mode_combo.currentText()

            if mode == "Single Image":
                # Use predict_image for single images
                result = self.deepforest.predict_image(path=self.current_image_path)
                self.prediction_mode = "single"
            else:
                # Use predict_tile for large geospatial tiles
                patch_size = self.patch_size_spin.value()
                patch_overlap = self.patch_overlap_spin.value()
                iou_threshold = self.iou_threshold_spin.value()

                dataloader_strategy = self.dataloader_combo.currentText()

                result = self.deepforest.predict_tile(
                    path=self.current_image_path,
                    patch_size=patch_size,
                    patch_overlap=patch_overlap,
                    iou_threshold=iou_threshold,
                    dataloader_strategy=dataloader_strategy,
                )
                self.prediction_mode = "tile"

            # Apply score threshold filter
            score_threshold = self.score_threshold_spin.value()
            if score_threshold > 0 and result is not None and not result.empty:
                if "score" in result.columns:
                    result = result[result.score >= score_threshold]

            self.predictions = result

            # Update results
            if result is not None and not result.empty:
                num_detections = len(result)
                self.predict_status_label.setText(
                    f"Found {num_detections} detection(s)."
                )
                self.predict_status_label.setStyleSheet("color: green;")

                # Show summary in results text
                summary = f"DeepForest Prediction Results:\n"
                summary += f"Mode: {mode}\n"
                summary += f"Image: {os.path.basename(self.current_image_path)}\n"
                summary += f"Detections: {num_detections}\n"

                if "score" in result.columns:
                    summary += f"Score range: {result.score.min():.3f} - {result.score.max():.3f}\n"

                if "label" in result.columns:
                    labels = result.label.value_counts()
                    summary += f"Labels: {dict(labels)}\n"

                self.results_text.setText(summary)
                self.log_message(
                    f"Prediction complete. Found {num_detections} detections."
                )

                # Auto-show results on map
                self._auto_show_results()
            else:
                self.predict_status_label.setText("No detections found.")
                self.predict_status_label.setStyleSheet("color: orange;")
                self.results_text.setText(
                    "No detections found. Try adjusting the score threshold."
                )

        except Exception as e:
            self.predict_status_label.setText("Prediction failed!")
            self.predict_status_label.setStyleSheet("color: red;")
            self.show_error(f"Prediction failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _auto_show_results(self):
        """Automatically save and show results on the map after prediction."""
        if not self.auto_show_check.isChecked():
            return

        if self.predictions is None or self.predictions.empty:
            return

        try:
            # Always auto-show as vector (GeoPackage) for best map display
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".gpkg", delete=False, prefix="deepforest_"
            )
            temp_path = temp_file.name
            temp_file.close()
            self._temp_files.append(temp_path)

            self._save_as_vector(temp_path, "GeoPackage")

            layer = QgsVectorLayer(temp_path, "deepforest_detections", "ogr")
            if layer.isValid():
                self._apply_semi_transparent_style(layer)
                QgsProject.instance().addMapLayer(layer)
                self.results_text.append(f"\nAuto-saved to: {temp_path}")
                self.log_message(f"Auto-saved detections to: {temp_path}")

        except Exception as e:
            self.log_message(f"Auto-show failed: {str(e)}", level=Qgis.Warning)

    def save_results(self):
        """Save the prediction results."""
        if self.predictions is None or self.predictions.empty:
            self.show_error("No predictions to save. Please run prediction first.")
            return

        output_path = self.output_path_edit.text().strip()
        format_text = self.output_format_combo.currentText()

        # Generate temp file path if not specified
        use_temp_file = False
        if not output_path:
            use_temp_file = True
            if "GeoJSON" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "GeoPackage" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "Shapefile" in format_text:
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "deepforest_detections.shp")
            else:  # Raster
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                output_path = temp_file.name
                temp_file.close()

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            QCoreApplication.processEvents()

            if "Raster" in format_text:
                # Save as raster - rasterize the bounding boxes
                self._save_as_raster(output_path)
            else:
                # Save as vector
                self._save_as_vector(output_path, format_text)

            if self.add_to_map_check.isChecked():
                layer_name = (
                    "deepforest_detections"
                    if use_temp_file
                    else os.path.basename(output_path)
                )

                if "Raster" in format_text:
                    layer = QgsRasterLayer(output_path, layer_name)
                else:
                    layer = QgsVectorLayer(output_path, layer_name, "ogr")

                if layer.isValid():
                    if isinstance(layer, QgsVectorLayer):
                        self._apply_semi_transparent_style(layer)
                    elif isinstance(layer, QgsRasterLayer):
                        self._apply_raster_transparency(layer)
                    QgsProject.instance().addMapLayer(layer)

            self.results_text.append(f"\nSaved to: {output_path}")
            self.log_message(f"Results saved to: {output_path}")

        except Exception as e:
            self.show_error(f"Failed to save results: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _pixel_to_geo_box(self, xmin, ymin, xmax, ymax, transform):
        """Convert pixel bounding box coordinates to geographic coordinates.

        DeepForest predict_image returns pixel-space boxes where:
          xmin/xmax = column indices, ymin/ymax = row indices.
        Rasterio's affine transform maps (col, row) → (x, y).

        Args:
            xmin, ymin, xmax, ymax: Pixel-space bounding box
                (xmin=col_min, ymin=row_min, xmax=col_max, ymax=row_max).
            transform: Affine transform from rasterio.

        Returns:
            shapely.geometry.box in geographic coordinates.
        """
        from shapely.geometry import box

        # (col, row) → (x, y) via affine transform
        # top-left pixel corner
        geo_x1, geo_y1 = transform * (xmin, ymin)
        # bottom-right pixel corner
        geo_x2, geo_y2 = transform * (xmax, ymax)
        # box() expects (minx, miny, maxx, maxy)
        return box(
            min(geo_x1, geo_x2),
            min(geo_y1, geo_y2),
            max(geo_x1, geo_x2),
            max(geo_y1, geo_y2),
        )

    def _save_as_vector(self, output_path, format_text):
        """Save predictions as vector format.

        predict_tile returns a GeoDataFrame with geometry already in map coords.
        predict_image returns a plain DataFrame with pixel-space xmin/ymin/xmax/ymax
        that must be converted to geographic coordinates via the source raster's
        affine transform.
        """
        import geopandas as gpd
        from shapely.geometry import box
        import pandas as pd

        preds = self.predictions

        # Determine if we already have usable geometry from predict_tile
        has_geo_geometry = (
            isinstance(preds, gpd.GeoDataFrame)
            and "geometry" in preds.columns
            and self.prediction_mode == "tile"
        )

        if has_geo_geometry:
            gdf = preds.copy()
            # Ensure it's a proper GeoDataFrame
            if not isinstance(gdf, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
        else:
            # Build geometry from bounding box columns.
            # For predict_image on georeferenced rasters, convert pixel coords
            # to geographic coords using the source raster's affine transform.
            transform = None
            if self.current_image_path:
                try:
                    import rasterio

                    with rasterio.open(self.current_image_path) as src:
                        transform = src.transform
                except Exception:
                    pass  # Fall back to raw pixel coords

            geometries = []
            for _, row in preds.iterrows():
                if transform is not None:
                    geom = self._pixel_to_geo_box(
                        row.xmin, row.ymin, row.xmax, row.ymax, transform
                    )
                else:
                    geom = box(row.xmin, row.ymin, row.xmax, row.ymax)
                geometries.append(geom)

            # Drop any existing non-shapely 'geometry' column to avoid conflict
            df = pd.DataFrame(preds)
            if "geometry" in df.columns:
                df = df.drop(columns=["geometry"])
            gdf = gpd.GeoDataFrame(df, geometry=geometries)

        # Set CRS if available from the current layer
        if self.current_layer and self.current_layer.crs().isValid():
            gdf.set_crs(self.current_layer.crs().toWkt(), inplace=True)

        # Determine driver
        if "GeoJSON" in format_text:
            driver = "GeoJSON"
        elif "GeoPackage" in format_text:
            driver = "GPKG"
        else:  # Shapefile
            driver = "ESRI Shapefile"

        gdf.to_file(output_path, driver=driver)

    def _save_as_raster(self, output_path):
        """Save predictions as raster by rasterizing bounding boxes.

        For predict_tile results, geometries are already in map coordinates and
        can be rasterized directly with the source raster's transform.
        For predict_image results, pixel-space bounding boxes are converted to
        geographic coordinates first so rasterize() works correctly with the
        source raster's affine transform.
        """
        try:
            import rasterio
            from rasterio.features import rasterize
            from shapely.geometry import box
            import numpy as np
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "Rasterio, shapely, and geopandas are required for raster output"
            )

        # Get source raster properties
        with rasterio.open(self.current_image_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            width = src.width
            height = src.height

        preds = self.predictions
        has_geo_geometry = (
            isinstance(preds, gpd.GeoDataFrame)
            and "geometry" in preds.columns
            and self.prediction_mode == "tile"
        )

        # Build (geometry, value) pairs for rasterize()
        shapes = []
        for i, (idx, row) in enumerate(preds.iterrows()):
            if has_geo_geometry:
                # predict_tile: geometry already in map coords
                geom = row.geometry
            else:
                # predict_image: pixel-space coords → convert to map coords
                geom = self._pixel_to_geo_box(
                    row.xmin, row.ymin, row.xmax, row.ymax, transform
                )
            # Use sequential value (1-based, 0 is background)
            shapes.append((geom, i + 1))

        # Rasterize
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint16,
        )

        # Update profile for single band output
        profile.update({"count": 1, "dtype": rasterio.uint16, "compress": "lzw"})

        # Write raster
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(raster, 1)

    def export_training_data(self):
        """Export predictions to training data format."""
        if self.predictions is None or self.predictions.empty:
            self.show_error("No predictions to export. Please run prediction first.")
            return

        if not self.current_image_path:
            self.show_error("No image set.")
            return

        export_dir = self.export_dir_edit.text().strip()
        if not export_dir:
            self.show_error("Please specify an export directory.")
            return

        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        format_type = self.export_format_combo.currentText()

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            QCoreApplication.processEvents()

            if format_type == "COCO":
                self._export_coco(export_dir)
            elif format_type == "YOLO":
                self._export_yolo(export_dir)

            self.results_text.append(f"\nTraining data exported to: {export_dir}")
            self.log_message(f"Training data exported to: {export_dir}")

        except Exception as e:
            self.show_error(f"Failed to export training data: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _export_coco(self, export_dir):
        """Export to COCO format."""
        import shutil
        from PIL import Image

        # Copy image to export directory
        image_name = os.path.basename(self.current_image_path)
        image_dest = os.path.join(export_dir, image_name)
        shutil.copy2(self.current_image_path, image_dest)

        # Get image dimensions
        with Image.open(self.current_image_path) as img:
            img_width, img_height = img.size

        # Create COCO annotation structure
        coco_data = {
            "info": {
                "description": "DeepForest Detection Export",
                "version": "1.0",
                "year": 2024,
            },
            "licenses": [],
            "images": [
                {
                    "id": 1,
                    "width": img_width,
                    "height": img_height,
                    "file_name": image_name,
                }
            ],
            "annotations": [],
            "categories": [],
        }

        # Get unique labels
        if "label" in self.predictions.columns:
            unique_labels = self.predictions.label.unique()
        else:
            unique_labels = ["detection"]

        # Create categories
        for idx, label in enumerate(unique_labels):
            coco_data["categories"].append(
                {"id": idx + 1, "name": label, "supercategory": "object"}
            )

        # Create annotations
        for ann_id, (_, row) in enumerate(self.predictions.iterrows()):
            bbox_width = row.xmax - row.xmin
            bbox_height = row.ymax - row.ymin
            area = bbox_width * bbox_height

            category_id = 1  # Default
            if "label" in self.predictions.columns:
                category_id = list(unique_labels).index(row.label) + 1

            annotation = {
                "id": ann_id + 1,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [row.xmin, row.ymin, bbox_width, bbox_height],
                "area": area,
                "iscrowd": 0,
            }

            if "score" in row:
                annotation["score"] = float(row.score)

            coco_data["annotations"].append(annotation)

        # Save COCO JSON
        coco_path = os.path.join(export_dir, "annotations.json")
        with open(coco_path, "w") as f:
            json.dump(coco_data, f, indent=2)

    def _export_yolo(self, export_dir):
        """Export to YOLO format."""
        import shutil
        from PIL import Image

        # Copy image to export directory
        image_name = os.path.basename(self.current_image_path)
        image_dest = os.path.join(export_dir, image_name)
        shutil.copy2(self.current_image_path, image_dest)

        # Get image dimensions
        with Image.open(self.current_image_path) as img:
            img_width, img_height = img.size

        # Get unique labels
        if "label" in self.predictions.columns:
            unique_labels = list(self.predictions.label.unique())
        else:
            unique_labels = ["detection"]

        # Create classes.txt
        classes_path = os.path.join(export_dir, "classes.txt")
        with open(classes_path, "w") as f:
            for label in unique_labels:
                f.write(f"{label}\n")

        # Create YOLO annotation file
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(export_dir, label_name)

        with open(label_path, "w") as f:
            for _, row in self.predictions.iterrows():
                # Get class index
                class_idx = 0  # Default
                if "label" in self.predictions.columns:
                    class_idx = unique_labels.index(row.label)

                # Convert to YOLO format (normalized xywh)
                center_x = ((row.xmin + row.xmax) / 2) / img_width
                center_y = ((row.ymin + row.ymax) / 2) / img_height
                width = (row.xmax - row.xmin) / img_width
                height = (row.ymax - row.ymin) / img_height

                f.write(
                    f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                )

    def _apply_raster_transparency(self, layer):
        """Make value 0 fully transparent and other values semi-transparent."""
        try:
            from qgis.core import (
                QgsRasterTransparency,
                QgsRasterRenderer,
            )

            renderer = layer.renderer()
            if renderer is None:
                return

            # Make value 0 fully transparent via no-data
            # Set the transparent pixel list for band 1
            transparency = QgsRasterTransparency()
            tr_pixel = QgsRasterTransparency.TransparentSingleValuePixel()
            tr_pixel.min = 0.0
            tr_pixel.max = 0.0
            tr_pixel.percentTransparent = 100.0
            transparency.setTransparentSingleValuePixelList([tr_pixel])
            renderer.setRasterTransparency(transparency)

            # Make all other values semi-transparent (50% opacity)
            renderer.setOpacity(0.5)

            layer.triggerRepaint()
        except Exception:
            pass  # Non-critical; default rendering is acceptable

    def _apply_semi_transparent_style(self, layer):
        """Apply a semi-transparent fill style to a vector layer."""
        try:
            symbol = QgsFillSymbol.createSimple(
                {
                    "color": "0,255,0,50",  # green fill, 50% transparent
                    "outline_color": "255,0,0,255",  # solid green outline
                    "outline_width": "0.4",
                }
            )
            layer.renderer().setSymbol(symbol)
            layer.triggerRepaint()
        except Exception:
            pass  # Non-critical; default style is acceptable

    def show_error(self, message):
        """Show an error message."""
        QMessageBox.critical(self, "DeepForest Error", message)
        self.log_message(message, level=Qgis.Critical)

    def log_message(self, message, level=Qgis.Info):
        """Log a message to QGIS."""
        QgsMessageLog.logMessage(message, "GeoAI - DeepForest", level)

    def cleanup(self):
        """Clean up resources when the dock is closed."""
        # Stop model loading worker if running
        if self.model_load_worker is not None and self.model_load_worker.isRunning():
            self.model_load_worker.terminate()
            self.model_load_worker.wait()

        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.log_message(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.log_message(
                    f"Failed to clean up temp file {temp_file}: {e}", level=Qgis.Warning
                )
        self._temp_files.clear()

        # Clean up model
        if self.deepforest is not None:
            del self.deepforest
            self.deepforest = None

    def closeEvent(self, event):
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)
