"""
Segmentation Dock Widget for GeoAI Plugin

This dock widget provides an interface for training semantic segmentation models
and running inference, combined in a single dockable panel.
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


class TrainingWorker(QThread):
    """Worker thread for training segmentation models."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    epoch_progress = pyqtSignal(int, int, str)  # current_epoch, total_epochs, metrics

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        architecture: str,
        encoder_name: str,
        encoder_weights: str,
        num_channels: int,
        num_classes: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        val_split: float,
        input_format: str = "directory",
        plot_curves: bool = False,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.input_format = input_format
        self.plot_curves = plot_curves
        self._epoch_pattern = None

    def _parse_output_line(self, line: str):
        """Parse a line of output for epoch progress."""
        import re

        if self._epoch_pattern is None:
            self._epoch_pattern = re.compile(
                r"Epoch (\d+)/(\d+): Train Loss: ([\d.]+), Val Loss: ([\d.]+), "
                r"Val IoU: ([\d.]+), Val F1: ([\d.]+)"
            )

        match = self._epoch_pattern.search(line)
        if match:
            current_epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            metrics = (
                f"Loss: {match.group(3)}, IoU: {match.group(5)}, F1: {match.group(6)}"
            )
            self.epoch_progress.emit(current_epoch, total_epochs, metrics)

    def run(self):
        """Execute the training."""
        import sys

        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Starting training...")

            # Capture stdout to parse epoch progress in real-time
            old_stdout = sys.stdout
            sys.stdout = OutputCapture(self._parse_output_line, old_stdout)

            try:
                geoai.train_segmentation_model(
                    images_dir=self.images_dir,
                    labels_dir=self.labels_dir,
                    output_dir=self.output_dir,
                    input_format=self.input_format,
                    architecture=self.architecture,
                    encoder_name=self.encoder_name,
                    encoder_weights=(
                        self.encoder_weights if self.encoder_weights != "None" else None
                    ),
                    num_channels=self.num_channels,
                    num_classes=self.num_classes,
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    val_split=self.val_split,
                    verbose=True,
                    plot_curves=self.plot_curves,
                )
            finally:
                sys.stdout = old_stdout

            model_path = os.path.join(self.output_dir, "best_model.pth")
            self.finished.emit(model_path)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class TileExportWorker(QThread):
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
    ):
        super().__init__()
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.stride = stride
        self.buffer_radius = buffer_radius

    def run(self):
        """Execute tile export."""
        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Exporting tiles...")

            geoai.export_geotiff_tiles(
                in_raster=self.raster_path,
                out_folder=self.output_dir,
                in_class_data=self.vector_path,
                tile_size=self.tile_size,
                stride=self.stride,
                buffer_radius=self.buffer_radius,
            )

            self.finished.emit(self.output_dir)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class InferenceWorker(QThread):
    """Worker thread for running segmentation inference."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_path: str,
        architecture: str,
        encoder_name: str,
        num_channels: int,
        num_classes: int,
        window_size: int,
        overlap: int,
        batch_size: int,
        probability_path: Optional[str] = None,
        probability_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.probability_path = probability_path
        self.probability_threshold = probability_threshold

    def run(self):
        """Execute the inference."""
        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Running inference...")

            kwargs = {}
            if self.probability_path:
                kwargs["probability_path"] = self.probability_path
            if self.probability_threshold is not None:
                kwargs["probability_threshold"] = self.probability_threshold

            geoai.semantic_segmentation(
                input_path=self.input_path,
                output_path=self.output_path,
                model_path=self.model_path,
                architecture=self.architecture,
                encoder_name=self.encoder_name,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                window_size=self.window_size,
                overlap=self.overlap,
                batch_size=self.batch_size,
                **kwargs,
            )

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
            smoothed_gdf = geoai.smooth_vector(
                gdf,
                smooth_iterations=self.smooth_iterations,
                output_path=self.output_path,
            )

            self.finished.emit(self.output_path)

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class SegmentationDockWidget(QDockWidget):
    """Dockable widget for segmentation training and inference."""

    def __init__(self, iface, parent=None):
        """Initialize the segmentation dock widget.

        Args:
            iface: QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("GeoAI - Segmentation", parent)
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

        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

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

        layout.addWidget(self.tab_widget)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)
        self.setWidget(scroll)

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

        # Model Architecture Group
        arch_group = QGroupBox("Model")
        arch_layout = QFormLayout()
        arch_layout.setSpacing(5)

        self.architecture_combo = QComboBox()
        self.architecture_combo.setStyleSheet(self.combo_style)
        self.architecture_combo.addItems(
            [
                "unet",
                "unetplusplus",
                "deeplabv3",
                "deeplabv3plus",
                "fpn",
                "pspnet",
                "linknet",
                "manet",
                "pan",
                "upernet",
                "segformer",
            ]
        )
        arch_layout.addRow("Architecture:", self.architecture_combo)

        self.encoder_combo = QComboBox()
        self.encoder_combo.setStyleSheet(self.combo_style)
        self.encoder_combo.addItems(
            [
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
                "resnext50_32x4d",
                "resnext101_32x4d",
                "efficientnet-b0",
                "efficientnet-b1",
                "efficientnet-b2",
                "efficientnet-b3",
                "efficientnet-b4",
                "efficientnet-b5",
                "mobilenet_v2",
                "mobileone_s0",
                "mobileone_s1",
                "vgg16",
                "vgg19",
                "densenet121",
                "densenet169",
                "mit_b0",
                "mit_b1",
                "mit_b2",
            ]
        )
        self.encoder_combo.setCurrentText("resnet34")
        arch_layout.addRow("Encoder:", self.encoder_combo)

        self.encoder_weights_combo = QComboBox()
        self.encoder_weights_combo.setStyleSheet(self.combo_style)
        self.encoder_weights_combo.addItems(
            ["imagenet", "ssl", "swsl", "advprop", "noisy-student", "None"]
        )
        arch_layout.addRow("Weights:", self.encoder_weights_combo)

        self.num_channels_spin = QSpinBox()
        self.num_channels_spin.setRange(1, 12)
        self.num_channels_spin.setValue(3)
        self.num_channels_spin.setStyleSheet(self.spin_style)
        arch_layout.addRow("Channels:", self.num_channels_spin)

        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 100)
        self.num_classes_spin.setValue(2)
        self.num_classes_spin.setStyleSheet(self.spin_style)
        arch_layout.addRow("Classes:", self.num_classes_spin)

        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group)

        # Training Parameters Group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        params_layout.setSpacing(5)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        self.batch_size_spin.setStyleSheet(self.spin_style)
        params_layout.addRow("Batch Size:", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setStyleSheet(self.spin_style)
        params_layout.addRow("Epochs:", self.epochs_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)  # Must set decimals before setValue
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setSingleStep(0.0001)
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

        self.plot_curves_check = QCheckBox("Save training curves plot")
        self.plot_curves_check.setChecked(True)
        output_layout.addRow("", self.plot_curves_check)

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

        self.inf_architecture_combo = QComboBox()
        self.inf_architecture_combo.setStyleSheet(self.combo_style)
        self.inf_architecture_combo.addItems(
            [
                "unet",
                "unetplusplus",
                "deeplabv3",
                "deeplabv3plus",
                "fpn",
                "pspnet",
                "linknet",
                "manet",
                "pan",
                "upernet",
                "segformer",
            ]
        )
        model_layout.addRow("Arch:", self.inf_architecture_combo)

        self.inf_encoder_combo = QComboBox()
        self.inf_encoder_combo.setStyleSheet(self.combo_style)
        self.inf_encoder_combo.addItems(
            [
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
                "resnext50_32x4d",
                "resnext101_32x4d",
                "efficientnet-b0",
                "efficientnet-b1",
                "efficientnet-b2",
                "efficientnet-b3",
                "efficientnet-b4",
                "efficientnet-b5",
                "mobilenet_v2",
                "mobileone_s0",
                "mobileone_s1",
                "vgg16",
                "vgg19",
                "densenet121",
                "densenet169",
                "mit_b0",
                "mit_b1",
                "mit_b2",
            ]
        )
        self.inf_encoder_combo.setCurrentText("resnet34")
        model_layout.addRow("Encoder:", self.inf_encoder_combo)

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

        self.inf_batch_size_spin = QSpinBox()
        self.inf_batch_size_spin.setRange(1, 64)
        self.inf_batch_size_spin.setValue(4)
        self.inf_batch_size_spin.setStyleSheet(self.spin_style)
        settings_layout.addRow("Batch:", self.inf_batch_size_spin)

        self.threshold_check = QCheckBox("Use threshold")
        settings_layout.addRow("", self.threshold_check)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.setStyleSheet(self.spin_style)
        settings_layout.addRow("Threshold:", self.threshold_spin)

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

        self.save_probability_check = QCheckBox("Save probability map")
        self.save_probability_check.setChecked(False)
        output_layout.addRow("", self.save_probability_check)

        prob_output_layout = QHBoxLayout()
        self.probability_path_edit = QLineEdit()
        self.probability_path_edit.setPlaceholderText("Probability output path...")
        self.probability_path_edit.setStyleSheet(self.line_style)
        self.probability_path_edit.setEnabled(False)
        prob_output_layout.addWidget(self.probability_path_edit)
        self.probability_browse_btn = QPushButton("...")
        self.probability_browse_btn.setFixedSize(30, self.input_height)
        self.probability_browse_btn.setEnabled(False)
        prob_output_layout.addWidget(self.probability_browse_btn)
        output_layout.addRow("Prob:", prob_output_layout)

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
        self.min_area_spin.setSuffix(" m²")
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
        self.inf_raster_browse_btn.clicked.connect(self.browse_inf_raster)
        self.model_browse_btn.clicked.connect(self.browse_model)
        self.output_browse_btn.clicked.connect(self.browse_output)
        self.vector_output_browse_btn.clicked.connect(self.browse_vector_output)
        self.threshold_check.toggled.connect(self.threshold_spin.setEnabled)
        self.save_probability_check.toggled.connect(self.on_probability_check_toggled)
        self.probability_browse_btn.clicked.connect(self.browse_probability)
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.vectorize_btn.clicked.connect(self.vectorize_mask)

        # Sync architecture and encoder between train and inference tabs
        self.architecture_combo.currentTextChanged.connect(
            self.sync_architecture_to_inference
        )
        self.encoder_combo.currentTextChanged.connect(self.sync_encoder_to_inference)
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
                self.output_path_edit.setText(f"{base}_segmentation.tif")

    def sync_architecture_to_inference(self, text):
        """Sync architecture selection to inference tab."""
        self.inf_architecture_combo.setCurrentText(text)

    def sync_encoder_to_inference(self, text):
        """Sync encoder selection to inference tab."""
        self.inf_encoder_combo.setCurrentText(text)

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raster", "", "GeoTIFF (*.tif *.tiff);;All (*)"
        )
        if file_path:
            self.raster_path_edit.setText(file_path)

    def browse_vector(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Vector", "", "GeoJSON (*.geojson);;Shapefile (*.shp);;All (*)"
        )
        if file_path:
            self.vector_path_edit.setText(file_path)

    def browse_tile_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.tile_output_dir_edit.setText(dir_path)

    def browse_images_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if dir_path:
            self.images_dir_edit.setText(dir_path)

    def browse_labels_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Labels Directory")
        if dir_path:
            self.labels_dir_edit.setText(dir_path)

    def browse_model_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.model_output_dir_edit.setText(dir_path)

    def browse_inf_raster(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raster", "", "GeoTIFF (*.tif *.tiff);;All (*)"
        )
        if file_path:
            self.inf_raster_path_edit.setText(file_path)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pth *.pt);;All (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "", "GeoTIFF (*.tif)"
        )
        if file_path:
            if not file_path.endswith(".tif"):
                file_path += ".tif"
            self.output_path_edit.setText(file_path)

    def browse_vector_output(self):
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

    def on_probability_check_toggled(self, checked):
        """Enable/disable probability path input based on checkbox."""
        self.probability_path_edit.setEnabled(checked)
        self.probability_browse_btn.setEnabled(checked)
        # Auto-generate probability path from output path
        if checked and not self.probability_path_edit.text():
            output_path = self.output_path_edit.text()
            if output_path:
                base, ext = os.path.splitext(output_path)
                self.probability_path_edit.setText(f"{base}_probability{ext}")

    def browse_probability(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Probability Map", "", "GeoTIFF (*.tif)"
        )
        if file_path:
            if not file_path.endswith(".tif"):
                file_path += ".tif"
            self.probability_path_edit.setText(file_path)

    # Export tiles
    def export_tiles(self):
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
        self.log("Starting tile export...")

        self.tile_worker = TileExportWorker(
            raster_path,
            vector_path,
            output_dir,
            self.tile_size_spin.value(),
            self.stride_spin.value(),
            self.buffer_spin.value(),
        )
        self.tile_worker.finished.connect(self.on_tile_export_finished)
        self.tile_worker.error.connect(self.on_tile_export_error)
        self.tile_worker.progress.connect(self.log)
        self.tile_worker.start()

    def on_tile_export_finished(self, output_dir: str):
        self.tile_progress.setRange(0, 100)
        self.tile_progress.setValue(100)
        self.export_tiles_btn.setEnabled(True)

        self.log(f"Tiles exported to: {output_dir}")

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
            f"Tiles exported to:\n{output_dir}\n\nSwitched to Train tab.",
        )

    def on_tile_export_error(self, error: str):
        self.tile_progress.setRange(0, 100)
        self.tile_progress.setValue(0)
        self.export_tiles_btn.setEnabled(True)
        self.log(f"Error: {error}")
        QMessageBox.critical(self, "Error", f"Export failed:\n{error}")

    # Training
    def start_training(self):
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
        self.log(f"Starting training: {self.architecture_combo.currentText()}")
        self.log(f"  Encoder: {self.encoder_combo.currentText()}")
        self.log(f"  Epochs: {num_epochs}, Batch size: {self.batch_size_spin.value()}")

        self.train_worker = TrainingWorker(
            images_dir,
            labels_dir,
            output_dir,
            self.architecture_combo.currentText(),
            self.encoder_combo.currentText(),
            self.encoder_weights_combo.currentText(),
            self.num_channels_spin.value(),
            self.num_classes_spin.value(),
            self.batch_size_spin.value(),
            num_epochs,
            self.learning_rate_spin.value(),
            self.val_split_spin.value(),
            self.input_format_combo.currentText(),
            self.plot_curves_check.isChecked(),
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

        # Check for training curves plot
        if self.plot_curves_check.isChecked():
            curves_path = os.path.join(output_dir, "training_curves.png")
            if os.path.exists(curves_path):
                self.log(f"Training curves saved to: {curves_path}")

        self.log("=" * 40)

        # Auto-fill inference tab
        self.model_path_edit.setText(model_path)

        # Sync model settings to inference tab
        self.inf_architecture_combo.setCurrentText(
            self.architecture_combo.currentText()
        )
        self.inf_encoder_combo.setCurrentText(self.encoder_combo.currentText())
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
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("")
        self.train_btn.setEnabled(True)
        self.log(f"Training error: {error}")
        QMessageBox.critical(self, "Error", f"Training failed:\n{error}")

    # Inference
    def run_inference(self):
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
        self.log("Starting inference...")

        threshold = (
            self.threshold_spin.value() if self.threshold_check.isChecked() else None
        )

        probability_path = None
        if self.save_probability_check.isChecked():
            probability_path = self.probability_path_edit.text()
            if not probability_path:
                # Auto-generate from output path
                base, ext = os.path.splitext(output_path)
                probability_path = f"{base}_probability{ext}"
                self.probability_path_edit.setText(probability_path)

        self.inference_worker = InferenceWorker(
            input_path,
            output_path,
            model_path,
            self.inf_architecture_combo.currentText(),
            self.inf_encoder_combo.currentText(),
            self.inf_num_channels_spin.value(),
            self.inf_num_classes_spin.value(),
            self.window_size_spin.value(),
            self.overlap_spin.value(),
            self.inf_batch_size_spin.value(),
            probability_path=probability_path,
            probability_threshold=threshold,
        )
        self.inference_worker.finished.connect(self.on_inference_finished)
        self.inference_worker.error.connect(self.on_inference_error)
        self.inference_worker.progress.connect(self.log)
        self.inference_worker.start()

    def on_inference_finished(self, output_path: str):
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(100)
        self.run_inference_btn.setEnabled(True)
        self.vectorize_btn.setEnabled(True)
        self.last_output_path = output_path

        self.log(f"Inference complete: {output_path}")

        # Log probability path if saved
        if self.save_probability_check.isChecked():
            prob_path = self.probability_path_edit.text()
            if prob_path and os.path.exists(prob_path):
                self.log(f"Probability map saved: {prob_path}")

        if self.add_to_map_check.isChecked():
            layer = QgsRasterLayer(output_path, "Segmentation Result")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.iface.mapCanvas().refresh()

        message = f"Output saved to:\n{output_path}"
        if self.save_probability_check.isChecked():
            prob_path = self.probability_path_edit.text()
            message += f"\n\nProbability map:\n{prob_path}"

        QMessageBox.information(self, "Success", message)

    def on_inference_error(self, error: str):
        self.inference_progress.setRange(0, 100)
        self.inference_progress.setValue(0)
        self.run_inference_btn.setEnabled(True)
        self.log(f"Inference error: {error}")
        QMessageBox.critical(self, "Error", f"Inference failed:\n{error}")

    # Vectorize
    def vectorize_mask(self):
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
