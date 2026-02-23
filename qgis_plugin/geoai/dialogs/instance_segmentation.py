"""
Instance Segmentation Dock Widget for GeoAI Plugin

This dock widget provides an interface for training Mask R-CNN instance
segmentation models and running inference, combined in a single dockable panel.
"""

import os
from typing import Optional

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
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTabWidget,
    QScrollArea,
    QSplitter,
)

from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel


class OutputCapture:
    """Capture stdout and emit lines to a callback in real-time."""

    def __init__(self, callback, original_stdout):
        self.callback = callback
        self.original_stdout = original_stdout
        self.buffer = ""

    def write(self, text):
        # Also write to original stdout
        if self.original_stdout:
            self.original_stdout.write(text)

        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.callback(line)

    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()
        if self.buffer.strip():
            self.callback(self.buffer)
            self.buffer = ""


class InstanceTrainingWorker(QThread):
    """Worker thread for training Mask R-CNN instance segmentation models."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    epoch_progress = pyqtSignal(int, int, str)  # current_epoch, total_epochs, metrics

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        num_channels: int,
        num_classes: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        val_split: float,
        input_format: str = "directory",
        visualize: bool = False,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.input_format = input_format
        self.visualize = visualize
        self._epoch_pattern = None

    def _parse_output_line(self, line: str):
        """Parse a line of output for epoch progress."""
        import re

        if self._epoch_pattern is None:
            self._epoch_pattern = re.compile(
                r"Epoch (\d+)/(\d+): Train Loss: ([\d.]+|inf), "
                r"Val Loss: ([\d.]+|inf), Val IoU: ([\d.]+|inf)"
            )

        match = self._epoch_pattern.search(line)
        if match:
            current_epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            metrics = f"Loss: {match.group(3)}, IoU: {match.group(5)}"
            self.epoch_progress.emit(current_epoch, total_epochs, metrics)

    def run(self):
        """Execute the training."""
        import sys

        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Starting Mask R-CNN training...")

            # Capture stdout to parse epoch progress in real-time
            old_stdout = sys.stdout
            sys.stdout = OutputCapture(self._parse_output_line, old_stdout)

            try:
                geoai.train_instance_segmentation_model(
                    images_dir=self.images_dir,
                    labels_dir=self.labels_dir,
                    output_dir=self.output_dir,
                    input_format=self.input_format,
                    num_channels=self.num_channels,
                    num_classes=self.num_classes,
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    val_split=self.val_split,
                    visualize=self.visualize,
                    verbose=True,
                )
            finally:
                sys.stdout = old_stdout

            model_path = os.path.join(self.output_dir, "best_model.pth")
            self.finished.emit(model_path)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class InstanceTileExportWorker(QThread):
    """Worker thread for exporting training tiles."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        raster_path: str,
        vector_path: str,
        output_dir: str,
        tile_size: int,
        stride: int,
        buffer_radius: float,
        metadata_format: str = "PASCAL_VOC",
    ):
        super().__init__()
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.stride = stride
        self.buffer_radius = buffer_radius
        self.metadata_format = metadata_format

    def run(self):
        """Execute tile export."""
        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit(f"Exporting tiles in {self.metadata_format} format...")

            geoai.export_geotiff_tiles(
                in_raster=self.raster_path,
                out_folder=self.output_dir,
                in_class_data=self.vector_path,
                tile_size=self.tile_size,
                stride=self.stride,
                buffer_radius=self.buffer_radius,
                metadata_format=self.metadata_format,
            )

            self.finished.emit(self.output_dir)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class InstanceInferenceWorker(QThread):
    """Worker thread for running instance segmentation inference."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_path: str,
        num_channels: int,
        num_classes: int,
        window_size: int,
        overlap: int,
        confidence_threshold: float,
        batch_size: int,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.window_size = window_size
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

    def _forward_line(self, line: str):
        """Forward stdout lines as progress messages."""
        self.progress.emit(line)

    def run(self):
        """Execute the inference."""
        import sys

        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Running instance segmentation inference...")

            # Capture stdout to forward progress messages to the log
            old_stdout = sys.stdout
            sys.stdout = OutputCapture(self._forward_line, old_stdout)

            try:
                geoai.instance_segmentation(
                    input_path=self.input_path,
                    output_path=self.output_path,
                    model_path=self.model_path,
                    num_channels=self.num_channels,
                    num_classes=self.num_classes,
                    window_size=self.window_size,
                    overlap=self.overlap,
                    confidence_threshold=self.confidence_threshold,
                    batch_size=self.batch_size,
                )
            finally:
                sys.stdout = old_stdout

            self.finished.emit(self.output_path)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class VectorizeWorker(QThread):
    """Worker thread for vectorizing raster masks."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        mask_path: str,
        output_path: str,
        epsilon: float = 2.0,
        min_area: Optional[float] = None,
    ):
        super().__init__()
        self.mask_path = mask_path
        self.output_path = output_path
        self.epsilon = epsilon
        self.min_area = min_area

    def run(self):
        """Execute vectorization."""
        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Vectorizing mask...")

            gdf = geoai.orthogonalize(
                self.mask_path,
                self.output_path,
                epsilon=self.epsilon,
            )

            if self.min_area is not None and self.min_area > 0:
                self.progress.emit(f"Filtering by area (min: {self.min_area})...")
                gdf = geoai.add_geometric_properties(gdf, area_unit="m2")
                gdf = gdf[gdf["area_m2"] >= self.min_area]
                driver = "GeoJSON" if self.output_path.endswith(".geojson") else None
                gdf.to_file(self.output_path, driver=driver)

            self.finished.emit(self.output_path)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class SmoothVectorWorker(QThread):
    """Worker thread for smoothing vector data."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        mask_path: str,
        output_path: str,
        smooth_iterations: int = 3,
        min_area: Optional[float] = None,
        simplify_tolerance: Optional[float] = None,
    ):
        super().__init__()
        self.mask_path = mask_path
        self.output_path = output_path
        self.smooth_iterations = smooth_iterations
        self.min_area = min_area
        self.simplify_tolerance = simplify_tolerance

    def run(self):
        """Execute smoothing."""
        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Converting raster to vector...")

            # First convert raster to vector (use 0 for min_area if None)
            min_area_value = self.min_area if self.min_area is not None else 0
            gdf = geoai.raster_to_vector(
                self.mask_path,
                min_area=min_area_value,
                simplify_tolerance=self.simplify_tolerance,
            )

            self.progress.emit(
                f"Smoothing vector (iterations: {self.smooth_iterations})..."
            )

            # Apply smoothing
            geoai.smooth_vector(
                gdf,
                smooth_iterations=self.smooth_iterations,
                output_path=self.output_path,
            )

            self.finished.emit(self.output_path)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class InstanceSegmentationDockWidget(QDockWidget):
    """Dockable widget for instance segmentation training and inference."""

    def __init__(self, iface, parent=None):
        """Initialize the instance segmentation dock widget.

        Args:
            iface: QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("GeoAI - Instance Segmentation", parent)
        self.iface = iface
        self.train_worker = None
        self.tile_worker = None
        self.inference_worker = None
        self.vectorize_worker = None
        self.smooth_worker = None
        self.last_output_path = None

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Set up the user interface."""
        # Consistent height for all input elements (24px)
        self.input_height = 24
        self.combo_style = f"QComboBox {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"
        self.line_style = f"QLineEdit {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"
        self.btn_style = f"QPushButton {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"
        self.spin_style = f"QSpinBox, QDoubleSpinBox {{ min-height: {self.input_height}px; max-height: {self.input_height}px; }}"

        # Vertical splitter: tabs on top, log on bottom (user can drag to resize)
        splitter = QSplitter(Qt.Vertical)

        # Top part: scrollable tabs
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        tab_container = QWidget()
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Tab 1: Create Training Data
        self.create_data_tab = QWidget()
        self.setup_create_data_tab()
        self.tab_widget.addTab(self.create_data_tab, "1. Create Data")

        # Tab 2: Train Model
        self.train_tab = QWidget()
        self.setup_train_tab()
        self.tab_widget.addTab(self.train_tab, "2. Train")

        # Tab 3: Inference
        self.inference_tab = QWidget()
        self.setup_inference_tab()
        self.tab_widget.addTab(self.inference_tab, "3. Inference")

        tab_layout.addWidget(self.tab_widget)
        tab_container.setLayout(tab_layout)
        self.scroll_area.setWidget(tab_container)
        splitter.addWidget(self.scroll_area)

        # Bottom part: log output (resizable via splitter)
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 2, 5, 2)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)

        # Give tabs ~70% and log ~30% of available space
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        self.setWidget(splitter)

        self.setMinimumWidth(350)

    def setup_create_data_tab(self):
        """Set up the Create Training Data tab."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Input Data Group
        input_group = QGroupBox("Input Data")
        input_layout = QFormLayout()
        input_layout.setSpacing(5)

        # Raster input
        raster_layout = QHBoxLayout()
        self.raster_layer_combo = QgsMapLayerComboBox()
        self.raster_layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.raster_layer_combo.setFixedHeight(self.input_height)
        raster_layout.addWidget(self.raster_layer_combo)
        self.raster_browse_btn = QPushButton("...")
        self.raster_browse_btn.setFixedSize(30, self.input_height)
        raster_layout.addWidget(self.raster_browse_btn)
        input_layout.addRow("Raster:", raster_layout)

        self.raster_path_edit = QLineEdit()
        self.raster_path_edit.setPlaceholderText("Or path to raster...")
        self.raster_path_edit.setStyleSheet(self.line_style)
        input_layout.addRow("", self.raster_path_edit)

        # Vector input
        vector_layout = QHBoxLayout()
        self.vector_layer_combo = QgsMapLayerComboBox()
        self.vector_layer_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.vector_layer_combo.setFixedHeight(self.input_height)
        vector_layout.addWidget(self.vector_layer_combo)
        self.vector_browse_btn = QPushButton("...")
        self.vector_browse_btn.setFixedSize(30, self.input_height)
        vector_layout.addWidget(self.vector_browse_btn)
        input_layout.addRow("Labels:", vector_layout)

        self.vector_path_edit = QLineEdit()
        self.vector_path_edit.setPlaceholderText("Or path to vector...")
        self.vector_path_edit.setStyleSheet(self.line_style)
        input_layout.addRow("", self.vector_path_edit)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Tile Settings Group
        tile_group = QGroupBox("Tile Settings")
        tile_layout = QFormLayout()
        tile_layout.setSpacing(5)

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(64, 2048)
        self.tile_size_spin.setValue(512)
        self.tile_size_spin.setSingleStep(64)
        self.tile_size_spin.setStyleSheet(self.spin_style)
        tile_layout.addRow("Tile Size:", self.tile_size_spin)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(32, 1024)
        self.stride_spin.setValue(256)
        self.stride_spin.setSingleStep(32)
        self.stride_spin.setStyleSheet(self.spin_style)
        tile_layout.addRow("Stride:", self.stride_spin)

        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(0, 100)
        self.buffer_spin.setValue(0)
        self.buffer_spin.setDecimals(2)
        self.buffer_spin.setStyleSheet(self.spin_style)
        tile_layout.addRow("Buffer:", self.buffer_spin)

        # Export format selection
        self.tile_export_format_combo = QComboBox()
        self.tile_export_format_combo.setStyleSheet(self.combo_style)
        self.tile_export_format_combo.addItems(["PASCAL_VOC", "COCO", "YOLO"])
        self.tile_export_format_combo.setToolTip(
            "Export format for training data:\n"
            "- PASCAL_VOC: XML annotation files (default)\n"
            "- COCO: JSON annotation file (for instance segmentation)\n"
            "- YOLO: Text files with normalized coordinates"
        )
        tile_layout.addRow("Format:", self.tile_export_format_combo)

        tile_group.setLayout(tile_layout)
        layout.addWidget(tile_group)

        # Output Group
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()
        output_layout.setSpacing(5)

        output_dir_layout = QHBoxLayout()
        self.tile_output_dir_edit = QLineEdit()
        self.tile_output_dir_edit.setPlaceholderText("Output directory...")
        self.tile_output_dir_edit.setStyleSheet(self.line_style)
        output_dir_layout.addWidget(self.tile_output_dir_edit)
        self.tile_output_browse_btn = QPushButton("...")
        self.tile_output_browse_btn.setFixedSize(30, self.input_height)
        output_dir_layout.addWidget(self.tile_output_browse_btn)
        output_layout.addRow("Output:", output_dir_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Export button
        btn_layout = QHBoxLayout()
        self.export_tiles_btn = QPushButton("Export Tiles")
        self.export_tiles_btn.setStyleSheet(self.btn_style)
        btn_layout.addWidget(self.export_tiles_btn)
        self.tile_progress = QProgressBar()
        self.tile_progress.setMaximumHeight(18)
        btn_layout.addWidget(self.tile_progress)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.create_data_tab.setLayout(layout)

    def setup_train_tab(self):
        """Set up the Train Model tab."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Training Data Group
        data_group = QGroupBox("Training Data")
        data_layout = QFormLayout()
        data_layout.setSpacing(5)

        self.input_format_combo = QComboBox()
        self.input_format_combo.setStyleSheet(self.combo_style)
        self.input_format_combo.addItems(["directory", "coco", "yolo"])
        data_layout.addRow("Format:", self.input_format_combo)

        images_layout = QHBoxLayout()
        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setPlaceholderText("Images directory...")
        self.images_dir_edit.setStyleSheet(self.line_style)
        images_layout.addWidget(self.images_dir_edit)
        self.images_browse_btn = QPushButton("...")
        self.images_browse_btn.setFixedSize(30, self.input_height)
        images_layout.addWidget(self.images_browse_btn)
        data_layout.addRow("Images:", images_layout)

        labels_layout = QHBoxLayout()
        self.labels_dir_edit = QLineEdit()
        self.labels_dir_edit.setPlaceholderText("Labels directory...")
        self.labels_dir_edit.setStyleSheet(self.line_style)
        labels_layout.addWidget(self.labels_dir_edit)
        self.labels_browse_btn = QPushButton("...")
        self.labels_browse_btn.setFixedSize(30, self.input_height)
        labels_layout.addWidget(self.labels_browse_btn)
        data_layout.addRow("Labels:", labels_layout)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Model Settings Group
        model_group = QGroupBox("Model (Mask R-CNN)")
        model_layout = QFormLayout()
        model_layout.setSpacing(5)

        self.num_channels_spin = QSpinBox()
        self.num_channels_spin.setRange(1, 12)
        self.num_channels_spin.setValue(3)
        self.num_channels_spin.setStyleSheet(self.spin_style)
        model_layout.addRow("Channels:", self.num_channels_spin)

        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 100)
        self.num_classes_spin.setValue(2)
        self.num_classes_spin.setStyleSheet(self.spin_style)
        self.num_classes_spin.setToolTip("Number of classes including background")
        model_layout.addRow("Classes:", self.num_classes_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Training Parameters Group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        params_layout.setSpacing(5)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        self.batch_size_spin.setStyleSheet(self.spin_style)
        params_layout.addRow("Batch Size:", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setStyleSheet(self.spin_style)
        params_layout.addRow("Epochs:", self.epochs_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setValue(0.005)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setStyleSheet(self.spin_style)
        params_layout.addRow("LR:", self.learning_rate_spin)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setValue(0.2)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setStyleSheet(self.spin_style)
        params_layout.addRow("Val Split:", self.val_split_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Output Group
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()
        output_layout.setSpacing(5)

        model_output_layout = QHBoxLayout()
        self.model_output_dir_edit = QLineEdit()
        self.model_output_dir_edit.setPlaceholderText("Output directory...")
        self.model_output_dir_edit.setStyleSheet(self.line_style)
        model_output_layout.addWidget(self.model_output_dir_edit)
        self.model_output_browse_btn = QPushButton("...")
        self.model_output_browse_btn.setFixedSize(30, self.input_height)
        model_output_layout.addWidget(self.model_output_browse_btn)
        output_layout.addRow("Output:", model_output_layout)

        self.visualize_check = QCheckBox("Save prediction visualizations")
        self.visualize_check.setChecked(False)
        output_layout.addRow("", self.visualize_check)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Train button
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet(self.btn_style)
        btn_layout.addWidget(self.train_btn)
        self.train_progress = QProgressBar()
        self.train_progress.setMaximumHeight(18)
        btn_layout.addWidget(self.train_progress)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.train_tab.setLayout(layout)

    def setup_inference_tab(self):
        """Set up the Inference tab."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Input Group
        input_group = QGroupBox("Input")
        input_layout = QFormLayout()
        input_layout.setSpacing(5)

        raster_layout = QHBoxLayout()
        self.inf_raster_layer_combo = QgsMapLayerComboBox()
        self.inf_raster_layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.inf_raster_layer_combo.setFixedHeight(self.input_height)
        raster_layout.addWidget(self.inf_raster_layer_combo)
        self.inf_raster_browse_btn = QPushButton("...")
        self.inf_raster_browse_btn.setFixedSize(30, self.input_height)
        raster_layout.addWidget(self.inf_raster_browse_btn)
        input_layout.addRow("Raster:", raster_layout)

        self.inf_raster_path_edit = QLineEdit()
        self.inf_raster_path_edit.setPlaceholderText("Or path to raster...")
        self.inf_raster_path_edit.setStyleSheet(self.line_style)
        input_layout.addRow("", self.inf_raster_path_edit)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Model Group
        model_group = QGroupBox("Model")
        model_layout = QFormLayout()
        model_layout.setSpacing(5)

        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Model path (.pth)...")
        self.model_path_edit.setStyleSheet(self.line_style)
        model_path_layout.addWidget(self.model_path_edit)
        self.model_browse_btn = QPushButton("...")
        self.model_browse_btn.setFixedSize(30, self.input_height)
        model_path_layout.addWidget(self.model_browse_btn)
        model_layout.addRow("Model:", model_path_layout)

        self.inf_num_channels_spin = QSpinBox()
        self.inf_num_channels_spin.setRange(1, 12)
        self.inf_num_channels_spin.setValue(3)
        self.inf_num_channels_spin.setStyleSheet(self.spin_style)
        model_layout.addRow("Channels:", self.inf_num_channels_spin)

        self.inf_num_classes_spin = QSpinBox()
        self.inf_num_classes_spin.setRange(2, 100)
        self.inf_num_classes_spin.setValue(2)
        self.inf_num_classes_spin.setStyleSheet(self.spin_style)
        model_layout.addRow("Classes:", self.inf_num_classes_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Inference Settings Group
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        settings_layout.setSpacing(5)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(64, 2048)
        self.window_size_spin.setValue(512)
        self.window_size_spin.setStyleSheet(self.spin_style)
        settings_layout.addRow("Window:", self.window_size_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 1024)
        self.overlap_spin.setValue(256)
        self.overlap_spin.setStyleSheet(self.spin_style)
        settings_layout.addRow("Overlap:", self.overlap_spin)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setStyleSheet(self.spin_style)
        self.confidence_spin.setToolTip("Confidence threshold for detections")
        settings_layout.addRow("Confidence:", self.confidence_spin)

        self.inf_batch_size_spin = QSpinBox()
        self.inf_batch_size_spin.setRange(1, 64)
        self.inf_batch_size_spin.setValue(4)
        self.inf_batch_size_spin.setStyleSheet(self.spin_style)
        settings_layout.addRow("Batch:", self.inf_batch_size_spin)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Output Group
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()
        output_layout.setSpacing(5)

        output_path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output mask path...")
        self.output_path_edit.setStyleSheet(self.line_style)
        output_path_layout.addWidget(self.output_path_edit)
        self.output_browse_btn = QPushButton("...")
        self.output_browse_btn.setFixedSize(30, self.input_height)
        output_path_layout.addWidget(self.output_browse_btn)
        output_layout.addRow("Output:", output_path_layout)

        self.add_to_map_check = QCheckBox("Add result to map")
        self.add_to_map_check.setChecked(True)
        output_layout.addRow("", self.add_to_map_check)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_inference_btn = QPushButton("Run Inference")
        self.run_inference_btn.setStyleSheet(self.btn_style)
        btn_layout.addWidget(self.run_inference_btn)
        self.inference_progress = QProgressBar()
        self.inference_progress.setMaximumHeight(18)
        btn_layout.addWidget(self.inference_progress)
        layout.addLayout(btn_layout)

        # Vectorize Group
        vectorize_group = QGroupBox("Vectorize (Optional)")
        vectorize_layout = QFormLayout()
        vectorize_layout.setSpacing(5)

        # Mode selection: Regularize vs Smooth
        self.vectorize_mode_combo = QComboBox()
        self.vectorize_mode_combo.setStyleSheet(self.combo_style)
        self.vectorize_mode_combo.addItems(
            ["Regularize (buildings)", "Smooth (natural features)"]
        )
        self.vectorize_mode_combo.currentIndexChanged.connect(
            self.on_vectorize_mode_changed
        )
        vectorize_layout.addRow("Mode:", self.vectorize_mode_combo)

        # Regularize options (visible by default)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 100.0)
        self.epsilon_spin.setValue(2.0)
        self.epsilon_spin.setStyleSheet(self.spin_style)
        self.epsilon_spin.setToolTip(
            "Douglas-Peucker tolerance for orthogonalization (suitable for buildings)"
        )
        vectorize_layout.addRow("Epsilon:", self.epsilon_spin)

        # Smooth options (hidden by default)
        self.smooth_iterations_spin = QSpinBox()
        self.smooth_iterations_spin.setRange(1, 20)
        self.smooth_iterations_spin.setValue(3)
        self.smooth_iterations_spin.setStyleSheet(self.spin_style)
        self.smooth_iterations_spin.setToolTip(
            "Number of smoothing iterations (suitable for natural features like water)"
        )
        self.smooth_iterations_spin.setVisible(False)
        vectorize_layout.addRow("Iterations:", self.smooth_iterations_spin)
        # Store reference to the label for visibility toggle
        self._smooth_iterations_label = vectorize_layout.labelForField(
            self.smooth_iterations_spin
        )
        self._smooth_iterations_label.setVisible(False)

        # Store reference to epsilon label for visibility toggle
        self._epsilon_label = vectorize_layout.labelForField(self.epsilon_spin)

        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.0, 10000.0)
        self.min_area_spin.setValue(0.0)
        self.min_area_spin.setSuffix(" m\u00b2")
        self.min_area_spin.setStyleSheet(self.spin_style)
        vectorize_layout.addRow("Min Area:", self.min_area_spin)

        vector_output_layout = QHBoxLayout()
        self.vector_output_edit = QLineEdit()
        self.vector_output_edit.setPlaceholderText("Vector output...")
        self.vector_output_edit.setStyleSheet(self.line_style)
        vector_output_layout.addWidget(self.vector_output_edit)
        self.vector_output_browse_btn = QPushButton("...")
        self.vector_output_browse_btn.setFixedSize(30, self.input_height)
        vector_output_layout.addWidget(self.vector_output_browse_btn)
        vectorize_layout.addRow("Output:", vector_output_layout)

        self.vectorize_btn = QPushButton("Vectorize")
        self.vectorize_btn.setEnabled(False)
        self.vectorize_btn.setStyleSheet(self.btn_style)
        vectorize_layout.addRow("", self.vectorize_btn)

        vectorize_group.setLayout(vectorize_layout)
        layout.addWidget(vectorize_group)

        layout.addStretch()
        self.inference_tab.setLayout(layout)

    def connect_signals(self):
        """Connect widget signals to slots."""
        # Tab changes - auto-fill from previous steps
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Create Data tab
        self.raster_browse_btn.clicked.connect(self.browse_raster)
        self.vector_browse_btn.clicked.connect(self.browse_vector)
        self.tile_output_browse_btn.clicked.connect(self.browse_tile_output)
        self.export_tiles_btn.clicked.connect(self.export_tiles)

        # Train tab
        self.images_browse_btn.clicked.connect(self.browse_images_dir)
        self.labels_browse_btn.clicked.connect(self.browse_labels_dir)
        self.model_output_browse_btn.clicked.connect(self.browse_model_output)
        self.train_btn.clicked.connect(self.start_training)

        # Inference tab
        self.inf_raster_layer_combo.layerChanged.connect(
            self.on_inf_raster_layer_changed
        )
        self.inf_raster_path_edit.textChanged.connect(self.on_inf_raster_path_changed)
        self.inf_raster_browse_btn.clicked.connect(self.browse_inf_raster)
        self.model_browse_btn.clicked.connect(self.browse_model)
        self.output_browse_btn.clicked.connect(self.browse_output)
        self.vector_output_browse_btn.clicked.connect(self.browse_vector_output)
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.vectorize_btn.clicked.connect(self.vectorize_mask)

        # Sync model settings between train and inference tabs
        self.num_channels_spin.valueChanged.connect(self.sync_channels_to_inference)
        self.num_classes_spin.valueChanged.connect(self.sync_classes_to_inference)

    def on_tab_changed(self, index):
        """Handle tab change - auto-fill inputs from previous steps."""
        if index == 1:  # Train tab
            # Auto-fill from Create Data tab if empty
            if not self.images_dir_edit.text() and self.tile_output_dir_edit.text():
                output_dir = self.tile_output_dir_edit.text()
                self.images_dir_edit.setText(os.path.join(output_dir, "images"))
                self.labels_dir_edit.setText(os.path.join(output_dir, "labels"))

            # Auto-suggest model output directory
            if (
                not self.model_output_dir_edit.text()
                and self.tile_output_dir_edit.text()
            ):
                output_dir = self.tile_output_dir_edit.text()
                parent_dir = os.path.dirname(output_dir)
                self.model_output_dir_edit.setText(os.path.join(parent_dir, "models"))

        elif index == 2:  # Inference tab
            # Auto-fill model path from training if available
            if not self.model_path_edit.text() and self.model_output_dir_edit.text():
                model_dir = self.model_output_dir_edit.text()
                model_path = os.path.join(model_dir, "best_model.pth")
                if os.path.exists(model_path):
                    self.model_path_edit.setText(model_path)
                else:
                    self.model_path_edit.setText(model_dir)

            # Auto-suggest output path based on input
            layer = self.inf_raster_layer_combo.currentLayer()
            if layer and not self.output_path_edit.text():
                source = layer.source()
                base, ext = os.path.splitext(source)
                self.output_path_edit.setText(f"{base}_instance_seg.tif")

    def _update_output_from_source(self, source: str):
        """Update output path based on a raster source path."""
        if source:
            base, _ = os.path.splitext(source)
            self.output_path_edit.setText(f"{base}_instance_seg.tif")

    def on_inf_raster_layer_changed(self, layer):
        """Update output path when the inference raster layer changes."""
        if layer and not self.inf_raster_path_edit.text():
            self._update_output_from_source(layer.source())

    def on_inf_raster_path_changed(self, text):
        """Update output path when the inference raster path changes."""
        if text:
            self._update_output_from_source(text)

    def sync_channels_to_inference(self, value):
        """Sync channels to inference tab."""
        self.inf_num_channels_spin.setValue(value)

    def sync_classes_to_inference(self, value):
        """Sync classes to inference tab."""
        self.inf_num_classes_spin.setValue(value)

    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # Browse methods
    def browse_raster(self):
        """Browse for a raster file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raster", "", "GeoTIFF (*.tif *.tiff);;All (*)"
        )
        if file_path:
            self.raster_path_edit.setText(file_path)

    def browse_vector(self):
        """Browse for a vector file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Vector", "", "GeoJSON (*.geojson);;Shapefile (*.shp);;All (*)"
        )
        if file_path:
            self.vector_path_edit.setText(file_path)

    def browse_tile_output(self):
        """Browse for tile output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.tile_output_dir_edit.setText(dir_path)

    def browse_images_dir(self):
        """Browse for images directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if dir_path:
            self.images_dir_edit.setText(dir_path)

    def browse_labels_dir(self):
        """Browse for labels directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Labels Directory")
        if dir_path:
            self.labels_dir_edit.setText(dir_path)

    def browse_model_output(self):
        """Browse for model output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.model_output_dir_edit.setText(dir_path)

    def browse_inf_raster(self):
        """Browse for inference input raster."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raster", "", "GeoTIFF (*.tif *.tiff);;All (*)"
        )
        if file_path:
            self.inf_raster_path_edit.setText(file_path)

    def browse_model(self):
        """Browse for a trained model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pth *.pt);;All (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_output(self):
        """Browse for inference output path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "", "GeoTIFF (*.tif)"
        )
        if file_path:
            if not file_path.endswith(".tif"):
                file_path += ".tif"
            self.output_path_edit.setText(file_path)

    def browse_vector_output(self):
        """Browse for vector output path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Vector", "", "GeoJSON (*.geojson);;Shapefile (*.shp)"
        )
        if file_path:
            self.vector_output_edit.setText(file_path)

    def on_vectorize_mode_changed(self, index):
        """Handle vectorize mode change between Regularize and Smooth."""
        is_smooth = index == 1  # 1 = Smooth mode

        # Toggle visibility of Regularize options
        self.epsilon_spin.setVisible(not is_smooth)
        self._epsilon_label.setVisible(not is_smooth)

        # Toggle visibility of Smooth options
        self.smooth_iterations_spin.setVisible(is_smooth)
        self._smooth_iterations_label.setVisible(is_smooth)

    # Export tiles
    def export_tiles(self):
        """Export training tiles from raster and vector data."""
        raster_layer = self.raster_layer_combo.currentLayer()
        raster_path = (
            raster_layer.source()
            if raster_layer and not self.raster_path_edit.text()
            else self.raster_path_edit.text()
        )

        vector_layer = self.vector_layer_combo.currentLayer()
        vector_path = (
            vector_layer.source()
            if vector_layer and not self.vector_path_edit.text()
            else self.vector_path_edit.text()
        )

        output_dir = self.tile_output_dir_edit.text()

        if not raster_path or not vector_path or not output_dir:
            QMessageBox.warning(self, "Warning", "Please fill in all fields.")
            return

        self.export_tiles_btn.setEnabled(False)
        self.tile_progress.setRange(0, 0)
        export_format = self.tile_export_format_combo.currentText()
        self.log(f"Starting tile export in {export_format} format...")

        self.tile_worker = InstanceTileExportWorker(
            raster_path,
            vector_path,
            output_dir,
            self.tile_size_spin.value(),
            self.stride_spin.value(),
            self.buffer_spin.value(),
            export_format,
        )
        self.tile_worker.finished.connect(self.on_tile_export_finished)
        self.tile_worker.error.connect(self.on_tile_export_error)
        self.tile_worker.progress.connect(self.log)
        self.tile_worker.start()

    def on_tile_export_finished(self, output_dir: str):
        """Handle tile export completion."""
        self.tile_progress.setRange(0, 100)
        self.tile_progress.setValue(100)
        self.export_tiles_btn.setEnabled(True)

        export_format = self.tile_export_format_combo.currentText()
        self.log(f"Tiles exported to: {output_dir} (format: {export_format})")

        # Auto-fill training tab
        self.images_dir_edit.setText(os.path.join(output_dir, "images"))
        self.labels_dir_edit.setText(os.path.join(output_dir, "labels"))

        # Auto-suggest model output directory
        parent_dir = os.path.dirname(output_dir)
        self.model_output_dir_edit.setText(os.path.join(parent_dir, "models"))

        # Switch to Train tab
        self.tab_widget.setCurrentIndex(1)

        QMessageBox.information(
            self,
            "Success",
            f"Tiles exported to:\n{output_dir}\n\n"
            f"Format: {export_format}\n\nSwitched to Train tab.",
        )

    def on_tile_export_error(self, error: str):
        """Handle tile export error."""
        self.tile_progress.setRange(0, 100)
        self.tile_progress.setValue(0)
        self.export_tiles_btn.setEnabled(True)
        self.log(f"Error: {error}")
        QMessageBox.critical(self, "Error", f"Export failed:\n{error}")

    # Training
    def start_training(self):
        """Start Mask R-CNN training."""
        images_dir = self.images_dir_edit.text()
        labels_dir = self.labels_dir_edit.text()
        output_dir = self.model_output_dir_edit.text()

        if not images_dir or not labels_dir or not output_dir:
            QMessageBox.warning(self, "Warning", "Please fill in all fields.")
            return

        self.train_btn.setEnabled(False)
        num_epochs = self.epochs_spin.value()
        self.train_progress.setRange(0, num_epochs)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("Epoch %v/%m")
        self.log("Starting Mask R-CNN training...")
        self.log(f"  Channels: {self.num_channels_spin.value()}")
        self.log(f"  Classes: {self.num_classes_spin.value()}")
        self.log(f"  Epochs: {num_epochs}, Batch size: {self.batch_size_spin.value()}")

        self.train_worker = InstanceTrainingWorker(
            images_dir,
            labels_dir,
            output_dir,
            self.num_channels_spin.value(),
            self.num_classes_spin.value(),
            self.batch_size_spin.value(),
            num_epochs,
            self.learning_rate_spin.value(),
            self.val_split_spin.value(),
            self.input_format_combo.currentText(),
            self.visualize_check.isChecked(),
        )
        self.train_worker.finished.connect(self.on_training_finished)
        self.train_worker.error.connect(self.on_training_error)
        self.train_worker.progress.connect(self.log)
        self.train_worker.epoch_progress.connect(self.on_epoch_progress)
        self.train_worker.start()

    def on_epoch_progress(self, current_epoch: int, total_epochs: int, metrics: str):
        """Handle epoch progress updates."""
        self.train_progress.setValue(current_epoch)
        self.log(f"Epoch {current_epoch}/{total_epochs}: {metrics}")

    def on_training_finished(self, model_path: str):
        """Handle training completion."""
        self.train_progress.setValue(self.train_progress.maximum())
        self.train_progress.setFormat("Complete")
        self.train_btn.setEnabled(True)

        self.log("=" * 40)
        self.log("Training Complete!")

        # Read and display training summary
        output_dir = os.path.dirname(model_path)
        summary_path = os.path.join(output_dir, "training_summary.txt")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = f.read()
                self.log("-" * 40)
                for line in summary.strip().split("\n"):
                    self.log(line)
                self.log("-" * 40)
            except Exception:
                pass

        self.log(f"Model saved to: {model_path}")
        self.log("=" * 40)

        # Auto-fill inference tab
        self.model_path_edit.setText(model_path)

        # Sync model settings to inference tab
        self.inf_num_channels_spin.setValue(self.num_channels_spin.value())
        self.inf_num_classes_spin.setValue(self.num_classes_spin.value())

        # Switch to Inference tab
        self.tab_widget.setCurrentIndex(2)

        QMessageBox.information(
            self,
            "Success",
            f"Model saved to:\n{model_path}\n\nSwitched to Inference tab.",
        )

    def on_training_error(self, error: str):
        """Handle training error."""
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("")
        self.train_btn.setEnabled(True)
        self.log(f"Training error: {error}")
        QMessageBox.critical(self, "Error", f"Training failed:\n{error}")

    # Inference
    def run_inference(self):
        """Run instance segmentation inference."""
        layer = self.inf_raster_layer_combo.currentLayer()
        input_path = (
            layer.source()
            if layer and not self.inf_raster_path_edit.text()
            else self.inf_raster_path_edit.text()
        )

        model_path = self.model_path_edit.text()
        output_path = self.output_path_edit.text()

        if not input_path or not model_path or not output_path:
            QMessageBox.warning(self, "Warning", "Please fill in all fields.")
            return

        self.run_inference_btn.setEnabled(False)
        self.inference_progress.setRange(0, 0)
        self.log("Starting instance segmentation inference...")

        self.inference_worker = InstanceInferenceWorker(
            input_path,
            output_path,
            model_path,
            self.inf_num_channels_spin.value(),
            self.inf_num_classes_spin.value(),
            self.window_size_spin.value(),
            self.overlap_spin.value(),
            self.confidence_spin.value(),
            self.inf_batch_size_spin.value(),
        )
        self.inference_worker.finished.connect(self.on_inference_finished)
        self.inference_worker.error.connect(self.on_inference_error)
        self.inference_worker.progress.connect(self.log)
        self.inference_worker.start()

    def on_inference_finished(self, output_path: str):
        """Handle inference completion."""
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(100)
        self.run_inference_btn.setEnabled(True)
        self.vectorize_btn.setEnabled(True)
        self.last_output_path = output_path

        self.log(f"Instance segmentation complete: {output_path}")

        # Check if result contains any instances
        num_instances = 0
        try:
            import numpy as np

            try:
                import rasterio

                with rasterio.open(output_path) as src:
                    data = src.read(1)
                    num_instances = int(data.max())
                    self.log(f"Detected instances: {num_instances}")
            except ImportError:
                pass
        except Exception:
            pass

        if self.add_to_map_check.isChecked():
            layer = QgsRasterLayer(output_path, "Instance Segmentation Result")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.iface.mapCanvas().refresh()

        if num_instances == 0:
            QMessageBox.warning(
                self,
                "Warning",
                f"Output saved to:\n{output_path}\n\n"
                "No instances were detected. Try:\n"
                "- Lowering the confidence threshold\n"
                "- Using a model trained on similar data\n"
                "- Checking that num_classes and num_channels match the trained model",
            )
        else:
            QMessageBox.information(
                self,
                "Success",
                f"Output saved to:\n{output_path}\n\n"
                f"Detected {num_instances} instances.",
            )

    def on_inference_error(self, error: str):
        """Handle inference error."""
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(0)
        self.run_inference_btn.setEnabled(True)
        self.log(f"Inference error: {error}")
        QMessageBox.critical(self, "Error", f"Inference failed:\n{error}")

    # Vectorize
    def vectorize_mask(self):
        """Vectorize the inference output mask."""
        mask_path = self.last_output_path or self.output_path_edit.text()
        vector_output = self.vector_output_edit.text()

        if not mask_path or not os.path.exists(mask_path):
            QMessageBox.warning(self, "Warning", "Please run inference first.")
            return

        if not vector_output:
            base, _ = os.path.splitext(mask_path)
            vector_output = f"{base}_vector.geojson"
            self.vector_output_edit.setText(vector_output)

        self.vectorize_btn.setEnabled(False)
        self.inference_progress.setRange(0, 0)

        min_area = (
            self.min_area_spin.value() if self.min_area_spin.value() > 0 else None
        )

        # Check vectorize mode: 0 = Regularize, 1 = Smooth
        is_smooth_mode = self.vectorize_mode_combo.currentIndex() == 1

        if is_smooth_mode:
            self.log("Smoothing vector (for natural features)...")
            self.smooth_worker = SmoothVectorWorker(
                mask_path,
                vector_output,
                smooth_iterations=self.smooth_iterations_spin.value(),
                min_area=min_area,
            )
            self.smooth_worker.finished.connect(self.on_vectorize_finished)
            self.smooth_worker.error.connect(self.on_vectorize_error)
            self.smooth_worker.progress.connect(self.log)
            self.smooth_worker.start()
        else:
            self.log("Regularizing vector (for buildings)...")
            self.vectorize_worker = VectorizeWorker(
                mask_path,
                vector_output,
                self.epsilon_spin.value(),
                min_area,
            )
            self.vectorize_worker.finished.connect(self.on_vectorize_finished)
            self.vectorize_worker.error.connect(self.on_vectorize_error)
            self.vectorize_worker.progress.connect(self.log)
            self.vectorize_worker.start()

    def on_vectorize_finished(self, output_path: str):
        """Handle vectorization completion."""
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(100)
        self.vectorize_btn.setEnabled(True)

        self.log(f"Vectorized: {output_path}")

        if self.add_to_map_check.isChecked():
            layer = QgsVectorLayer(output_path, "Vectorized Result", "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.iface.mapCanvas().refresh()

        QMessageBox.information(self, "Success", f"Saved to:\n{output_path}")

    def on_vectorize_error(self, error: str):
        """Handle vectorization error."""
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(0)
        self.vectorize_btn.setEnabled(True)
        self.log(f"Vectorize error: {error}")
        QMessageBox.critical(self, "Error", f"Vectorization failed:\n{error}")

    def closeEvent(self, event):
        """Handle widget close event."""
        for worker in [
            self.train_worker,
            self.tile_worker,
            self.inference_worker,
            self.vectorize_worker,
            self.smooth_worker,
        ]:
            if worker and worker.isRunning():
                worker.terminate()
                worker.wait()
        event.accept()
