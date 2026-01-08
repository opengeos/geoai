"""
SamGeo Dock Widget for GeoAI Plugin

This dock widget provides an interface for remote sensing image segmentation
using the SamGeo library (SAM, SAM2, and SAM3 models).
"""

import os
import tempfile

try:
    import torch
except ImportError:
    torch = None

from qgis.PyQt.QtCore import Qt, QCoreApplication
from qgis.PyQt.QtGui import QColor
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
    QListWidget,
    QListWidgetItem,
    QScrollArea,
)
from qgis.core import (
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterFileWriter,
    QgsRasterLayer,
    QgsRasterPipe,
    QgsVectorLayer,
    QgsWkbTypes,
    Qgis,
    QgsMessageLog,
)

from .map_tools import PointPromptTool, BoxPromptTool


class SamGeoDockWidget(QDockWidget):
    """Dock widget for SamGeo segmentation operations."""

    def __init__(self, iface, parent=None):
        """Initialize the SamGeo dock widget.

        Args:
            iface: The QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("SamGeo Segmentation", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # SamGeo model instance
        self.sam = None
        self.current_layer = None
        self.current_image_path = None

        # Point and box prompts
        self.point_coords = []
        self.point_labels = []
        self.box_coords = None

        # Batch point prompts
        self.batch_point_coords = []
        self.batch_point_coords_map = []  # Map coordinates for display

        # Map tools
        self.point_tool = None
        self.batch_point_tool = None
        self.box_tool = None
        self.previous_tool = None

        # Track temporary files for cleanup
        self._temp_files = []

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

        # Tab widget for different modes
        self.tab_widget = QTabWidget()

        # === Model Settings Tab ===
        model_tab = self._create_model_tab()
        self.tab_widget.addTab(model_tab, "Model")

        # === Text Prompts Tab ===
        text_tab = self._create_text_tab()
        self.tab_widget.addTab(text_tab, "Text")

        # === Interactive Tab ===
        interactive_tab = self._create_interactive_tab()
        self.tab_widget.addTab(interactive_tab, "Interactive")

        # === Batch Tab ===
        batch_tab = self._create_batch_tab()
        self.tab_widget.addTab(batch_tab, "Batch")

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

        # Backend selection
        backend_group = QGroupBox("Model Settings")
        backend_layout = QVBoxLayout()

        # Model version selection
        version_row = QHBoxLayout()
        version_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["SamGeo3 (SAM3)"])
        version_row.addWidget(self.model_combo)
        backend_layout.addLayout(version_row)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["meta", "transformers"])
        backend_row.addWidget(self.backend_combo)
        backend_layout.addLayout(backend_row)

        # Device selection
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        device_row.addWidget(self.device_combo)
        backend_layout.addLayout(device_row)

        # Confidence threshold
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        conf_row.addWidget(self.conf_spin)
        backend_layout.addLayout(conf_row)

        # Interactive mode checkbox
        self.interactive_check = QCheckBox(
            "Enable Interactive Mode (Point/Box Prompts)"
        )
        self.interactive_check.setChecked(True)
        backend_layout.addWidget(self.interactive_check)

        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        backend_layout.addWidget(self.load_model_btn)

        # Model status
        self.model_status = QLabel("Model: Not loaded")
        self.model_status.setStyleSheet("color: gray;")
        backend_layout.addWidget(self.model_status)

        backend_group.setLayout(backend_layout)
        model_layout.addWidget(backend_group)

        # Layer selection
        layer_group = QGroupBox("Input Layer")
        layer_layout = QVBoxLayout()

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

        # Band selection for multi-band GeoTIFFs
        self.custom_bands_check = QCheckBox("Use custom RGB bands")
        self.custom_bands_check.setChecked(False)
        self.custom_bands_check.setToolTip(
            "Enable to specify custom bands for RGB display.\n"
            "Useful for multi-band GeoTIFF files (with more than 3 bands).\n"
            "Example: [4, 3, 2] for NIR-R-G false color composite."
        )
        self.custom_bands_check.stateChanged.connect(self._on_custom_bands_changed)
        layer_layout.addWidget(self.custom_bands_check)

        # RGB band spinboxes
        bands_row = QHBoxLayout()
        bands_row.addWidget(QLabel("R:"))
        self.red_band_spin = QSpinBox()
        self.red_band_spin.setRange(1, 100)
        self.red_band_spin.setValue(1)
        self.red_band_spin.setToolTip("Red band index (1-based)")
        self.red_band_spin.setEnabled(False)
        bands_row.addWidget(self.red_band_spin)

        bands_row.addWidget(QLabel("G:"))
        self.green_band_spin = QSpinBox()
        self.green_band_spin.setRange(1, 100)
        self.green_band_spin.setValue(2)
        self.green_band_spin.setToolTip("Green band index (1-based)")
        self.green_band_spin.setEnabled(False)
        bands_row.addWidget(self.green_band_spin)

        bands_row.addWidget(QLabel("B:"))
        self.blue_band_spin = QSpinBox()
        self.blue_band_spin.setRange(1, 100)
        self.blue_band_spin.setValue(3)
        self.blue_band_spin.setToolTip("Blue band index (1-based)")
        self.blue_band_spin.setEnabled(False)
        bands_row.addWidget(self.blue_band_spin)
        layer_layout.addLayout(bands_row)

        self.image_status = QLabel("Image: Not set")
        self.image_status.setStyleSheet("color: gray;")
        layer_layout.addWidget(self.image_status)

        layer_group.setLayout(layer_layout)
        model_layout.addWidget(layer_group)

        model_layout.addStretch()
        return model_tab

    def _create_text_tab(self):
        """Create the text prompts tab."""
        text_tab = QWidget()
        text_layout = QVBoxLayout()
        text_tab.setLayout(text_layout)

        text_group = QGroupBox("Text-Based Segmentation")
        text_group_layout = QVBoxLayout()

        text_group_layout.addWidget(QLabel("Describe objects to segment:"))
        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setPlaceholderText("e.g., tree, building, road...")
        text_group_layout.addWidget(self.text_prompt_edit)

        # Size filters
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Min size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 1000000)
        self.min_size_spin.setValue(0)
        size_row.addWidget(self.min_size_spin)

        size_row.addWidget(QLabel("Max size:"))
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(0, 10000000)
        self.max_size_spin.setValue(0)
        self.max_size_spin.setSpecialValueText("No limit")
        size_row.addWidget(self.max_size_spin)
        text_group_layout.addLayout(size_row)

        self.text_segment_btn = QPushButton("Segment by Text")
        self.text_segment_btn.clicked.connect(self.segment_by_text)
        text_group_layout.addWidget(self.text_segment_btn)

        # Text segmentation status
        self.text_status_label = QLabel("")
        text_group_layout.addWidget(self.text_status_label)

        text_group.setLayout(text_group_layout)
        text_layout.addWidget(text_group)

        text_layout.addStretch()
        return text_tab

    def _create_interactive_tab(self):
        """Create the interactive prompts tab."""
        interactive_tab = QWidget()
        interactive_layout = QVBoxLayout()
        interactive_tab.setLayout(interactive_layout)

        # Point prompts
        point_group = QGroupBox("Point Prompts")
        point_layout = QVBoxLayout()

        point_btn_row = QHBoxLayout()
        self.add_fg_point_btn = QPushButton("Add Foreground Points")
        self.add_fg_point_btn.setCheckable(True)
        self.add_fg_point_btn.clicked.connect(
            lambda: self.start_point_tool(foreground=True)
        )
        point_btn_row.addWidget(self.add_fg_point_btn)

        self.add_bg_point_btn = QPushButton("Add Background Points")
        self.add_bg_point_btn.setCheckable(True)
        self.add_bg_point_btn.clicked.connect(
            lambda: self.start_point_tool(foreground=False)
        )
        point_btn_row.addWidget(self.add_bg_point_btn)
        point_layout.addLayout(point_btn_row)

        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(100)
        point_layout.addWidget(self.points_list)

        point_action_row = QHBoxLayout()
        clear_points_btn = QPushButton("Clear Points")
        clear_points_btn.clicked.connect(self.clear_points)
        point_action_row.addWidget(clear_points_btn)

        self.point_segment_btn = QPushButton("Segment by Points")
        self.point_segment_btn.clicked.connect(self.segment_by_points)
        point_action_row.addWidget(self.point_segment_btn)
        point_layout.addLayout(point_action_row)

        # Point segmentation status
        self.point_status_label = QLabel("")
        point_layout.addWidget(self.point_status_label)

        point_group.setLayout(point_layout)
        interactive_layout.addWidget(point_group)

        # Box prompts
        box_group = QGroupBox("Box Prompts")
        box_layout = QVBoxLayout()

        self.draw_box_btn = QPushButton("Draw Box")
        self.draw_box_btn.setCheckable(True)
        self.draw_box_btn.clicked.connect(self.start_box_tool)
        box_layout.addWidget(self.draw_box_btn)

        self.box_label = QLabel("Box: Not set")
        box_layout.addWidget(self.box_label)

        box_action_row = QHBoxLayout()
        clear_box_btn = QPushButton("Clear Box")
        clear_box_btn.clicked.connect(self.clear_box)
        box_action_row.addWidget(clear_box_btn)

        self.box_segment_btn = QPushButton("Segment by Box")
        self.box_segment_btn.clicked.connect(self.segment_by_box)
        box_action_row.addWidget(self.box_segment_btn)
        box_layout.addLayout(box_action_row)

        # Box segmentation status
        self.box_status_label = QLabel("")
        box_layout.addWidget(self.box_status_label)

        box_group.setLayout(box_layout)
        interactive_layout.addWidget(box_group)

        interactive_layout.addStretch()
        return interactive_tab

    def _create_batch_tab(self):
        """Create the batch processing tab."""
        batch_tab = QWidget()
        batch_layout = QVBoxLayout()
        batch_tab.setLayout(batch_layout)

        # Interactive Points for Batch
        batch_interactive_group = QGroupBox("Create Points Interactively")
        batch_interactive_layout = QVBoxLayout()

        batch_point_btn_row = QHBoxLayout()
        self.batch_add_point_btn = QPushButton("Add Points on Map")
        self.batch_add_point_btn.setCheckable(True)
        self.batch_add_point_btn.clicked.connect(self.start_batch_point_tool)
        batch_point_btn_row.addWidget(self.batch_add_point_btn)

        self.batch_clear_points_btn = QPushButton("Clear Points")
        self.batch_clear_points_btn.clicked.connect(self.clear_batch_points)
        batch_point_btn_row.addWidget(self.batch_clear_points_btn)
        batch_interactive_layout.addLayout(batch_point_btn_row)

        # Batch points list
        self.batch_points_list = QListWidget()
        self.batch_points_list.setMaximumHeight(80)
        batch_interactive_layout.addWidget(self.batch_points_list)

        self.batch_points_count_label = QLabel("Points: 0")
        batch_interactive_layout.addWidget(self.batch_points_count_label)

        batch_interactive_group.setLayout(batch_interactive_layout)
        batch_layout.addWidget(batch_interactive_group)

        # Or Load from File/Layer
        batch_file_group = QGroupBox("Or Load Points from File/Layer")
        batch_file_layout = QVBoxLayout()

        # Vector layer selection
        vector_layer_row = QHBoxLayout()
        vector_layer_row.addWidget(QLabel("Layer:"))
        self.vector_layer_combo = QComboBox()
        self.refresh_vector_layers()
        vector_layer_row.addWidget(self.vector_layer_combo)

        refresh_vector_btn = QPushButton("↻")
        refresh_vector_btn.setMaximumWidth(30)
        refresh_vector_btn.clicked.connect(self.refresh_vector_layers)
        vector_layer_row.addWidget(refresh_vector_btn)
        batch_file_layout.addLayout(vector_layer_row)

        # Or load from file
        vector_file_row = QHBoxLayout()
        self.vector_file_edit = QLineEdit()
        self.vector_file_edit.setPlaceholderText("Or select vector file...")
        vector_file_row.addWidget(self.vector_file_edit)

        browse_vector_btn = QPushButton("...")
        browse_vector_btn.setMaximumWidth(30)
        browse_vector_btn.clicked.connect(self.browse_vector_file)
        vector_file_row.addWidget(browse_vector_btn)
        batch_file_layout.addLayout(vector_file_row)

        # CRS selection
        crs_row = QHBoxLayout()
        crs_row.addWidget(QLabel("CRS:"))
        self.point_crs_edit = QLineEdit()
        self.point_crs_edit.setPlaceholderText("e.g., EPSG:4326 (auto-detect if empty)")
        crs_row.addWidget(self.point_crs_edit)
        batch_file_layout.addLayout(crs_row)

        batch_file_group.setLayout(batch_file_layout)
        batch_layout.addWidget(batch_file_group)

        # Batch Settings
        batch_settings_group = QGroupBox("Batch Settings")
        batch_settings_layout = QVBoxLayout()

        # Batch size filters
        batch_size_row = QHBoxLayout()
        batch_size_row.addWidget(QLabel("Min size:"))
        self.batch_min_size_spin = QSpinBox()
        self.batch_min_size_spin.setRange(0, 1000000)
        self.batch_min_size_spin.setValue(0)
        batch_size_row.addWidget(self.batch_min_size_spin)

        batch_size_row.addWidget(QLabel("Max size:"))
        self.batch_max_size_spin = QSpinBox()
        self.batch_max_size_spin.setRange(0, 10000000)
        self.batch_max_size_spin.setValue(0)
        self.batch_max_size_spin.setSpecialValueText("No limit")
        batch_size_row.addWidget(self.batch_max_size_spin)
        batch_settings_layout.addLayout(batch_size_row)

        # Output options for batch
        batch_output_row = QHBoxLayout()
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("Output raster file (optional)...")
        batch_output_row.addWidget(self.batch_output_edit)

        browse_batch_output_btn = QPushButton("...")
        browse_batch_output_btn.setMaximumWidth(30)
        browse_batch_output_btn.clicked.connect(self.browse_batch_output)
        batch_output_row.addWidget(browse_batch_output_btn)
        batch_settings_layout.addLayout(batch_output_row)

        # Batch unique values
        self.batch_unique_check = QCheckBox("Unique values for each object")
        self.batch_unique_check.setChecked(True)
        batch_settings_layout.addWidget(self.batch_unique_check)

        batch_settings_group.setLayout(batch_settings_layout)
        batch_layout.addWidget(batch_settings_group)

        # Segment button
        self.batch_segment_btn = QPushButton("Run Batch Segmentation")
        self.batch_segment_btn.clicked.connect(self.segment_by_points_batch)
        batch_layout.addWidget(self.batch_segment_btn)

        # Batch status
        self.batch_status_label = QLabel("")
        batch_layout.addWidget(self.batch_status_label)

        # Info text
        batch_info = QLabel(
            "<i>Batch mode processes each point as a separate prompt, "
            "generating individual masks for each point.</i>"
        )
        batch_info.setWordWrap(True)
        batch_layout.addWidget(batch_info)

        batch_layout.addStretch()
        return batch_tab

    def _create_output_tab(self):
        """Create the output settings tab."""
        output_tab = QWidget()
        output_layout = QVBoxLayout()
        output_tab.setLayout(output_layout)

        output_group = QGroupBox("Output Settings")
        output_group_layout = QVBoxLayout()

        # Output format
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(
            [
                "Raster (GeoTIFF)",
                "Vector (GeoPackage)",
                "Vector (Shapefile)",
                "Vector (GeoJSON)",
            ]
        )
        self.output_format_combo.currentIndexChanged.connect(
            self._on_output_format_changed
        )
        format_row.addWidget(self.output_format_combo)
        output_group_layout.addLayout(format_row)

        # Unique values
        self.unique_check = QCheckBox("Unique values for each object")
        self.unique_check.setChecked(True)
        output_group_layout.addWidget(self.unique_check)

        # Add to map
        self.add_to_map_check = QCheckBox("Add result to map")
        self.add_to_map_check.setChecked(True)
        output_group_layout.addWidget(self.add_to_map_check)

        # Auto-show results after segmentation
        self.auto_show_check = QCheckBox("Auto-show results after segmentation")
        self.auto_show_check.setChecked(True)
        self.auto_show_check.setToolTip(
            "Automatically save and display results after running segmentation\n"
            "in the Text, Interactive, or Batch tabs"
        )
        output_group_layout.addWidget(self.auto_show_check)

        # Output path
        output_path_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText(
            "Output file path (optional, uses temp file if empty)..."
        )
        output_path_row.addWidget(self.output_path_edit)

        output_browse_btn = QPushButton("...")
        output_browse_btn.setMaximumWidth(30)
        output_browse_btn.clicked.connect(self.browse_output)
        output_path_row.addWidget(output_browse_btn)
        output_group_layout.addLayout(output_path_row)

        output_group.setLayout(output_group_layout)
        output_layout.addWidget(output_group)

        # Vector processing options (for vector output)
        self.vector_options_group = QGroupBox("Vector Processing Options (Vector Only)")
        vector_options_layout = QVBoxLayout()

        # Mode selection: Simple, Regularize, or Smooth
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.vector_mode_combo = QComboBox()
        self.vector_mode_combo.addItems(
            [
                "Simple (no processing)",
                "Regularize (buildings)",
                "Smooth (natural features)",
            ]
        )
        self.vector_mode_combo.currentIndexChanged.connect(self._on_vector_mode_changed)
        mode_row.addWidget(self.vector_mode_combo)
        vector_options_layout.addLayout(mode_row)

        # Regularize options
        self.regularize_options_widget = QWidget()
        regularize_options_layout = QVBoxLayout()
        regularize_options_layout.setContentsMargins(0, 0, 0, 0)

        # Epsilon (Douglas-Peucker tolerance)
        epsilon_row = QHBoxLayout()
        epsilon_row.addWidget(QLabel("Epsilon:"))
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 100.0)
        self.epsilon_spin.setValue(2.0)
        self.epsilon_spin.setSingleStep(0.5)
        self.epsilon_spin.setToolTip(
            "Douglas-Peucker simplification tolerance for orthogonalization (suitable for buildings)"
        )
        epsilon_row.addWidget(self.epsilon_spin)
        regularize_options_layout.addLayout(epsilon_row)

        self.regularize_options_widget.setLayout(regularize_options_layout)
        self.regularize_options_widget.setVisible(
            False
        )  # Hidden by default (Simple mode)
        vector_options_layout.addWidget(self.regularize_options_widget)

        # Smooth options
        self.smooth_options_widget = QWidget()
        smooth_options_layout = QVBoxLayout()
        smooth_options_layout.setContentsMargins(0, 0, 0, 0)

        # Smooth iterations
        smooth_iterations_row = QHBoxLayout()
        smooth_iterations_row.addWidget(QLabel("Iterations:"))
        self.smooth_iterations_spin = QSpinBox()
        self.smooth_iterations_spin.setRange(1, 20)
        self.smooth_iterations_spin.setValue(3)
        self.smooth_iterations_spin.setToolTip(
            "Number of smoothing iterations (suitable for natural features like water, vegetation)"
        )
        smooth_iterations_row.addWidget(self.smooth_iterations_spin)
        smooth_options_layout.addLayout(smooth_iterations_row)

        self.smooth_options_widget.setLayout(smooth_options_layout)
        self.smooth_options_widget.setVisible(False)  # Hidden by default
        vector_options_layout.addWidget(self.smooth_options_widget)

        # Min Area filter (common to both modes)
        min_area_row = QHBoxLayout()
        min_area_row.addWidget(QLabel("Min Area:"))
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.0, 100000.0)
        self.min_area_spin.setValue(0.0)
        self.min_area_spin.setSuffix(" m²")
        self.min_area_spin.setToolTip(
            "Minimum area filter - polygons smaller than this will be removed"
        )
        min_area_row.addWidget(self.min_area_spin)
        vector_options_layout.addLayout(min_area_row)

        self.vector_options_group.setLayout(vector_options_layout)
        self.vector_options_group.setVisible(
            False
        )  # Hidden by default (raster selected)
        output_layout.addWidget(self.vector_options_group)

        # Save button
        self.save_btn = QPushButton("Save Masks")
        self.save_btn.clicked.connect(self.save_masks)
        output_layout.addWidget(self.save_btn)

        # Results info
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        output_layout.addWidget(results_group)

        output_layout.addStretch()
        return output_tab

    def _on_output_format_changed(self, index):
        """Handle output format combo box change."""
        format_text = self.output_format_combo.currentText()
        is_vector = "Vector" in format_text
        self.vector_options_group.setVisible(is_vector)

    def _on_vector_mode_changed(self, index):
        """Handle vector processing mode change: Simple, Regularize, or Smooth."""
        # 0 = Simple (no processing), 1 = Regularize, 2 = Smooth
        is_simple = index == 0
        is_regularize = index == 1
        is_smooth = index == 2
        self.regularize_options_widget.setVisible(is_regularize)
        self.smooth_options_widget.setVisible(is_smooth)

    def _on_custom_bands_changed(self, state):
        """Handle custom bands checkbox state change."""
        enabled = state == Qt.Checked
        self.red_band_spin.setEnabled(enabled)
        self.green_band_spin.setEnabled(enabled)
        self.blue_band_spin.setEnabled(enabled)

    def _get_bands(self):
        """Get the bands list for set_image if custom bands are enabled.

        Returns:
            list[int] or None: A list of exactly three integers [R, G, B], representing
                the band indices for red, green, and blue channels in that order, if custom bands
                are enabled. Returns None if custom bands are disabled.
        """
        if self.custom_bands_check.isChecked():
            bands = [
                self.red_band_spin.value(),
                self.green_band_spin.value(),
                self.blue_band_spin.value(),
            ]
            if len(set(bands)) < 3:
                QMessageBox.warning(
                    self,
                    "Invalid Band Selection",
                    "Please select three different bands for R, G, and B channels.",
                )
                return None
            return bands
        return None

    def _is_geopackage_raster(self, source):
        """Check if the layer source is a raster inside a GeoPackage.

        Args:
            source: The layer source string from QGIS.

        Returns:
            bool: True if the source is a GeoPackage raster, False otherwise.
        """
        # GeoPackage raster sources can have various formats:
        # - GPKG:/path/to/file.gpkg:layername
        # - /path/to/file.gpkg|layername=...
        # - /path/to/file.gpkg (with sublayer info)
        if source.upper().startswith("GPKG:"):
            return True
        # For non-prefixed sources, be stricter about GDAL-style layer syntax.
        # Only treat `.gpkg|...` as GeoPackage when the part after `|` contains
        # a `layername=` token commonly used in GDAL connection strings.
        if ".gpkg" in source.lower() and "|" in source:
            after_pipe = source.split("|", 1)[1]
            if after_pipe.startswith("layername=") or "layername=" in after_pipe:
                return True
        # Check if it's a .gpkg file that exists (plain GeoPackage path)
        if ".gpkg" in source.lower():
            # Extract potential gpkg path (ignore any GDAL-style pipe options)
            gpkg_path = source.split("|", 1)[0]
            if gpkg_path.lower().endswith(".gpkg") and os.path.exists(gpkg_path):
                return True
        return False

    def _export_geopackage_raster(self, layer):
        """Export a GeoPackage raster layer to a temporary GeoTIFF file.

        Args:
            layer: The QgsRasterLayer to export.

        Returns:
            str: Path to the temporary GeoTIFF file, or None if export failed.
        """
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False, prefix="samgeo_gpkg_"
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

    def refresh_layers(self):
        """Refresh the list of raster layers."""
        self.layer_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.layer_combo.addItem(layer.name(), layer.id())

    def refresh_vector_layers(self):
        """Refresh the list of vector layers (for batch point mode)."""
        self.vector_layer_combo.clear()
        self.vector_layer_combo.addItem("-- Select from file instead --", None)
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                # Only include point layers
                if layer.geometryType() == QgsWkbTypes.PointGeometry:
                    self.vector_layer_combo.addItem(layer.name(), layer.id())

    def browse_vector_file(self):
        """Browse for a vector file containing points."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vector File",
            "",
            "Vector files (*.geojson *.json *.shp *.gpkg *.kml);;All files (*.*)",
        )
        if file_path:
            self.vector_file_edit.setText(file_path)

    def browse_batch_output(self):
        """Browse for batch output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Output", "", "GeoTIFF (*.tif)"
        )
        if file_path:
            self.batch_output_edit.setText(file_path)

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
                        # Note: This may not work in all cases as CUDA is initialized once per process
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
                            # Still not available after fix attempt
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
                    # CUDA not available and CUDA_VISIBLE_DEVICES is not set
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
        """Load the SamGeo model."""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.model_status.setText("Loading model...")
            self.model_status.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            model_version = self.model_combo.currentText()
            backend = self.backend_combo.currentText()
            device = self.device_combo.currentText()

            if device == "auto":
                device = None

            # Check CUDA availability if using CUDA or auto device selection
            if device == "cuda" or device is None:
                cuda_available, warning_message = self.check_cuda_devices()

                if not cuda_available:
                    # CUDA is not available
                    if device == "cuda":
                        # User explicitly requested CUDA but it's not available
                        self.progress_bar.setVisible(False)
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
                    # CUDA is now available but there was a warning (e.g., fixed CUDA_VISIBLE_DEVICES)
                    QMessageBox.information(self, "CUDA Issue Fixed", warning_message)

            confidence = self.conf_spin.value()
            enable_interactive = self.interactive_check.isChecked()

            # Import and initialize the appropriate model
            if "SamGeo3" in model_version:
                from samgeo import SamGeo3

                self.sam = SamGeo3(
                    backend=backend,
                    device=device,
                    confidence_threshold=confidence,
                    enable_inst_interactivity=enable_interactive,
                )
                model_name = "SamGeo3"
            elif "SamGeo2" in model_version:
                from samgeo import SamGeo2

                self.sam = SamGeo2(
                    device=device,
                )
                model_name = "SamGeo2"
            else:
                from samgeo import SamGeo

                self.sam = SamGeo(
                    device=device,
                )
                model_name = "SamGeo"

            self.model_status.setText(f"Model: {model_name} loaded")
            self.model_status.setStyleSheet("color: green;")
            self.log_message(f"{model_name} model loaded successfully")

        except Exception as e:
            self.model_status.setText("Model: Failed to load")
            self.model_status.setStyleSheet("color: red;")
            self.show_error(f"Failed to load model: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def set_image_from_layer(self):
        """Set the image from the selected QGIS layer."""
        if self.sam is None:
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

            # Get bands if custom bands are enabled
            bands = self._get_bands()
            self.sam.set_image(image_path, bands=bands)
            self.current_layer = layer
            self.current_image_path = image_path

            # Build status message
            status_msg = f"Image: {layer.name()}"
            if is_gpkg:
                status_msg += " (from GeoPackage)"
            if bands:
                status_msg += f" (Bands: {bands})"
            self.image_status.setText(status_msg)
            self.image_status.setStyleSheet("color: green;")

            log_msg = f"Image set from layer: {layer.name()}"
            if is_gpkg:
                log_msg += f" (exported from GeoPackage to {image_path})"
            if bands:
                log_msg += f" with bands {bands}"
            self.log_message(log_msg)

        except Exception as e:
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            # Clean up temp file if export succeeded but set_image failed
            if temp_export_path and os.path.exists(temp_export_path):
                try:
                    os.remove(temp_export_path)
                    # Keep internal temp file tracking consistent, if present
                    if hasattr(self, "_temp_files"):
                        try:
                            self._temp_files.remove(temp_export_path)
                        except ValueError:
                            # It was not tracked; ignore
                            pass
                except Exception as cleanup_error:
                    # Log cleanup failure but do not mask the original error
                    self.log_message(
                        f"Failed to clean up temporary export file '{temp_export_path}': {cleanup_error}",
                        level=Qgis.Warning,
                    )
            self.show_error(f"Failed to set image: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def set_image_from_file(self):
        """Set the image from the file path."""
        if self.sam is None:
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

            # Get bands if custom bands are enabled
            bands = self._get_bands()
            self.sam.set_image(file_path, bands=bands)

            # Optionally add the layer to the map
            layer = QgsRasterLayer(file_path, os.path.basename(file_path))
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.current_layer = layer
                self.current_image_path = file_path

                # Build status message
                status_msg = f"Image: {os.path.basename(file_path)}"
                if bands:
                    status_msg += f" (Bands: {bands})"
                self.image_status.setText(status_msg)
                self.image_status.setStyleSheet("color: green;")

                log_msg = f"Image set from file: {file_path}"
                if bands:
                    log_msg += f" with bands {bands}"
                self.log_message(log_msg)
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

    def segment_by_text(self):
        """Segment the image using text prompt."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        prompt = self.text_prompt_edit.text().strip()
        if not prompt:
            self.show_error("Please enter a text prompt.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.text_status_label.setText("Processing...")
            self.text_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            min_size = self.min_size_spin.value()
            max_size = (
                self.max_size_spin.value() if self.max_size_spin.value() > 0 else None
            )

            self.sam.generate_masks(prompt, min_size=min_size, max_size=max_size)

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Text Segmentation Results:\n"
                f"Prompt: {prompt}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.text_status_label.setText(f"Found {num_masks} object(s).")
                else:
                    self.text_status_label.setText(
                        f"Found {num_masks} object(s). Go to Output tab to save."
                    )
                self.text_status_label.setStyleSheet("color: green;")
            else:
                self.text_status_label.setText(
                    "No objects found. Try a different prompt."
                )
                self.text_status_label.setStyleSheet("color: orange;")

            self.log_message(f"Text segmentation complete. Found {num_masks} objects.")

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.text_status_label.setText("Segmentation failed!")
            self.text_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def start_point_tool(self, foreground=True):
        """Start the point prompt tool."""
        if self.current_layer is None:
            self.show_error("Please set an image first.")
            self.add_fg_point_btn.setChecked(False)
            self.add_bg_point_btn.setChecked(False)
            return

        # Uncheck the other button
        if foreground:
            self.add_bg_point_btn.setChecked(False)
        else:
            self.add_fg_point_btn.setChecked(False)

        if self.point_tool is None:
            self.point_tool = PointPromptTool(self.canvas, self)

        self.point_tool.set_foreground(foreground)
        self.previous_tool = self.canvas.mapTool()
        self.canvas.setMapTool(self.point_tool)

    def add_point(self, point, foreground):
        """Add a point prompt."""
        # Convert map coordinates to pixel coordinates
        if self.current_layer is not None:
            extent = self.current_layer.extent()
            width = self.current_layer.width()
            height = self.current_layer.height()

            # Transform point from canvas CRS to layer CRS if they differ
            canvas_crs = self.canvas.mapSettings().destinationCrs()
            layer_crs = self.current_layer.crs()
            if canvas_crs != layer_crs:
                transform = QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance()
                )
                point = transform.transform(point)

            # Calculate pixel coordinates
            px = (point.x() - extent.xMinimum()) / extent.width() * width
            py = (extent.yMaximum() - point.y()) / extent.height() * height

            self.point_coords.append([px, py])
            self.point_labels.append(1 if foreground else 0)

            # Update list widget
            label_text = "FG" if foreground else "BG"
            item = QListWidgetItem(f"{label_text}: ({px:.1f}, {py:.1f})")
            item.setForeground(QColor("green") if foreground else QColor("red"))
            self.points_list.addItem(item)

    def clear_points(self):
        """Clear all point prompts."""
        self.point_coords = []
        self.point_labels = []
        self.points_list.clear()

        # Clear rubber bands from point tool
        if self.point_tool is not None:
            self.point_tool.clear_markers()

    def start_batch_point_tool(self):
        """Start the batch point tool for adding multiple points."""
        if self.current_layer is None:
            self.show_error("Please set an image first.")
            self.batch_add_point_btn.setChecked(False)
            return

        if self.batch_point_tool is None:
            self.batch_point_tool = PointPromptTool(self.canvas, self, batch_mode=True)

        self.batch_point_tool.set_foreground(True)  # All batch points are foreground
        self.previous_tool = self.canvas.mapTool()
        self.canvas.setMapTool(self.batch_point_tool)

    def add_batch_point(self, point):
        """Add a batch point prompt."""
        if self.current_layer is not None:
            extent = self.current_layer.extent()
            width = self.current_layer.width()
            height = self.current_layer.height()

            # Transform point from canvas CRS to layer CRS if they differ
            canvas_crs = self.canvas.mapSettings().destinationCrs()
            layer_crs = self.current_layer.crs()
            if canvas_crs != layer_crs:
                transform = QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance()
                )
                point = transform.transform(point)

            # Calculate pixel coordinates
            px = (point.x() - extent.xMinimum()) / extent.width() * width
            py = (extent.yMaximum() - point.y()) / extent.height() * height

            self.batch_point_coords.append([px, py])
            self.batch_point_coords_map.append([point.x(), point.y()])

            # Update list widget
            item = QListWidgetItem(
                f"Point {len(self.batch_point_coords)}: ({px:.1f}, {py:.1f})"
            )
            item.setForeground(QColor("green"))
            self.batch_points_list.addItem(item)

            # Update count label
            self.batch_points_count_label.setText(
                f"Points: {len(self.batch_point_coords)}"
            )

    def clear_batch_points(self):
        """Clear all batch point prompts."""
        self.batch_point_coords = []
        self.batch_point_coords_map = []
        self.batch_points_list.clear()
        self.batch_points_count_label.setText("Points: 0")

        # Clear markers from batch point tool
        if self.batch_point_tool is not None:
            self.batch_point_tool.clear_markers()

    def segment_by_points(self):
        """Segment using point prompts."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        if not self.point_coords:
            self.show_error("Please add at least one point.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.point_status_label.setText("Processing...")
            self.point_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            import numpy as np

            point_coords = np.array(self.point_coords)
            point_labels = np.array(self.point_labels)

            # Use multimask_output=False when there are multiple points or
            # background points, as it gives better results for non-ambiguous prompts
            has_background = 0 in point_labels
            use_multimask = len(point_coords) == 1 and not has_background

            # Use appropriate method based on model type
            # point_crs=None because coordinates are already in pixel space
            # (transformed from canvas CRS to layer CRS in add_point())
            if hasattr(self.sam, "generate_masks_by_points"):
                self.sam.generate_masks_by_points(
                    point_coords=point_coords.tolist(),
                    point_labels=point_labels.tolist(),
                    point_crs=None,
                    multimask_output=use_multimask,
                )
            else:
                # Fallback for older SamGeo versions
                # Try to pass multimask_output for consistency; if not supported, fall back without it.
                try:
                    self.sam.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        point_crs=None,
                        multimask_output=use_multimask,
                    )
                except TypeError:
                    # Older SamGeo versions do not support multimask_output; behavior may differ.
                    self.sam.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        point_crs=None,
                    )

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Point Segmentation Results:\n"
                f"Points used: {len(self.point_coords)}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.point_status_label.setText(f"Found {num_masks} object(s).")
                else:
                    self.point_status_label.setText(
                        f"Found {num_masks} object(s). Go to Output tab to save."
                    )
                self.point_status_label.setStyleSheet("color: green;")
            else:
                self.point_status_label.setText(
                    "No objects found. Try different points."
                )
                self.point_status_label.setStyleSheet("color: orange;")

            self.log_message(f"Point segmentation complete. Found {num_masks} objects.")

            # Deactivate tool
            self.add_fg_point_btn.setChecked(False)
            self.add_bg_point_btn.setChecked(False)
            if self.previous_tool:
                self.canvas.setMapTool(self.previous_tool)

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.point_status_label.setText("Segmentation failed!")
            self.point_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def start_box_tool(self):
        """Start the box prompt tool."""
        if self.current_layer is None:
            self.show_error("Please set an image first.")
            self.draw_box_btn.setChecked(False)
            return

        if self.box_tool is None:
            self.box_tool = BoxPromptTool(self.canvas, self)

        self.previous_tool = self.canvas.mapTool()
        self.canvas.setMapTool(self.box_tool)

    def set_box(self, rect):
        """Set the box prompt from a rectangle."""
        if self.current_layer is not None:
            extent = self.current_layer.extent()
            width = self.current_layer.width()
            height = self.current_layer.height()

            # Transform rectangle from canvas CRS to layer CRS if they differ
            canvas_crs = self.canvas.mapSettings().destinationCrs()
            layer_crs = self.current_layer.crs()
            if canvas_crs != layer_crs:
                transform = QgsCoordinateTransform(
                    canvas_crs, layer_crs, QgsProject.instance()
                )
                rect = transform.transformBoundingBox(rect)

            # Convert to pixel coordinates
            x1 = (rect.xMinimum() - extent.xMinimum()) / extent.width() * width
            y1 = (extent.yMaximum() - rect.yMaximum()) / extent.height() * height
            x2 = (rect.xMaximum() - extent.xMinimum()) / extent.width() * width
            y2 = (extent.yMaximum() - rect.yMinimum()) / extent.height() * height

            self.box_coords = [x1, y1, x2, y2]
            self.box_label.setText(f"Box: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")

    def clear_box(self):
        """Clear the box prompt."""
        self.box_coords = None
        self.box_label.setText("Box: Not set")

        if self.box_tool is not None:
            self.box_tool.clear_rubber_band()

    def segment_by_box(self):
        """Segment using box prompt."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        if self.box_coords is None:
            self.show_error("Please draw a box first.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.box_status_label.setText("Processing...")
            self.box_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            # Use appropriate method based on model type
            if hasattr(self.sam, "generate_masks_by_boxes"):
                self.sam.generate_masks_by_boxes(boxes=[self.box_coords])
            else:
                # Fallback for older SamGeo versions
                import numpy as np

                self.sam.predict(box=np.array(self.box_coords))

            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Box Segmentation Results:\n"
                f"Box: {self.box_coords}\n"
                f"Objects found: {num_masks}\n"
            )

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.box_status_label.setText(f"Found {num_masks} object(s).")
                else:
                    self.box_status_label.setText(
                        f"Found {num_masks} object(s). Go to Output tab to save."
                    )
                self.box_status_label.setStyleSheet("color: green;")
            else:
                self.box_status_label.setText("No objects found. Try a different box.")
                self.box_status_label.setStyleSheet("color: orange;")

            self.log_message(f"Box segmentation complete. Found {num_masks} objects.")

            # Deactivate tool
            self.draw_box_btn.setChecked(False)
            if self.previous_tool:
                self.canvas.setMapTool(self.previous_tool)

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.box_status_label.setText("Segmentation failed!")
            self.box_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def segment_by_points_batch(self):
        """Segment using batch point prompts from interactive points or vector file/layer."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        # Check if batch method is available
        if not hasattr(self.sam, "generate_masks_by_points_patch"):
            self.show_error(
                "Batch point mode requires SamGeo3 with enable_inst_interactivity=True.\n"
                "Please reload the model with the correct settings."
            )
            return

        # Check for interactive batch points first
        use_interactive_points = len(self.batch_point_coords) > 0

        # Get point source (interactive, layer, or file)
        point_source = None
        point_crs = None
        source_description = ""

        if use_interactive_points:
            # Use interactive points (already in pixel coordinates)
            point_source = self.batch_point_coords
            point_crs = None  # Already in pixel coordinates
            source_description = f"{len(self.batch_point_coords)} interactive points"
        else:
            # Check if a layer is selected
            layer_id = self.vector_layer_combo.currentData()
            if layer_id:
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer and layer.isValid():
                    point_source = layer.source()
                    source_description = layer.name()
                    # Get CRS from layer
                    if layer.crs().isValid():
                        point_crs = layer.crs().authid()

            # If no layer, check for file path
            if point_source is None:
                vector_file = self.vector_file_edit.text().strip()
                if vector_file and os.path.exists(vector_file):
                    point_source = vector_file
                    source_description = os.path.basename(vector_file)
                else:
                    self.show_error(
                        "Please add points interactively or select a vector layer/file."
                    )
                    return

            # Get CRS from input field if specified
            crs_text = self.point_crs_edit.text().strip()
            if crs_text:
                point_crs = crs_text

        # Get output path
        output_path = self.batch_output_edit.text().strip()
        if not output_path:
            output_path = None  # Will only store in memory

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.batch_status_label.setText("Processing batch segmentation...")
            self.batch_status_label.setStyleSheet("color: orange;")
            QCoreApplication.processEvents()

            # Get size filters
            min_size = self.batch_min_size_spin.value()
            max_size = (
                self.batch_max_size_spin.value()
                if self.batch_max_size_spin.value() > 0
                else None
            )
            unique = self.batch_unique_check.isChecked()

            # Run batch segmentation
            self.sam.generate_masks_by_points_patch(
                point_coords_batch=point_source,
                point_crs=point_crs,
                output=output_path,
                unique=unique,
                min_size=min_size,
                max_size=max_size,
            )

            num_masks = len(self.sam.masks) if self.sam.masks else 0

            # Update results
            result_text = (
                f"Batch Point Segmentation Results:\n"
                f"Source: {source_description}\n"
                f"Objects found: {num_masks}\n"
            )
            if output_path:
                result_text += f"Output saved to: {output_path}\n"

            self.results_text.setText(result_text)

            # Update status label
            if num_masks > 0:
                if self.auto_show_check.isChecked():
                    self.batch_status_label.setText(f"Found {num_masks} object(s).")
                else:
                    self.batch_status_label.setText(
                        f"Found {num_masks} object(s). Go to Output tab to save."
                    )
                self.batch_status_label.setStyleSheet("color: green;")
            else:
                self.batch_status_label.setText(
                    "No objects found. Try different points."
                )
                self.batch_status_label.setStyleSheet("color: orange;")

            self.log_message(
                f"Batch point segmentation complete. Found {num_masks} objects."
            )

            # Deactivate batch point tool if active
            self.batch_add_point_btn.setChecked(False)

            # Add output to map if saved and option is checked (for batch-specific output)
            if output_path and self.add_to_map_check.isChecked():
                layer = QgsRasterLayer(output_path, os.path.basename(output_path))
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    self.results_text.append("Added result layer to map.")

            # Auto-show results if enabled
            if num_masks > 0:
                self._auto_show_results()

        except Exception as e:
            self.batch_status_label.setText("Failed!")
            self.batch_status_label.setStyleSheet("color: red;")
            self.show_error(f"Batch segmentation failed: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def _auto_show_results(self):
        """Automatically save and show results based on Output tab settings.

        This method is called after successful segmentation to automatically
        display results without requiring the user to manually click on the
        Output tab and Save Masks button.
        """
        if not self.auto_show_check.isChecked():
            return

        if self.sam is None or self.sam.masks is None or len(self.sam.masks) == 0:
            return

        import tempfile

        format_text = self.output_format_combo.currentText()
        output_path = self.output_path_edit.text().strip()

        # Generate temp file path if not specified
        use_temp_file = False
        if not output_path:
            use_temp_file = True
            if "Raster" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "GeoJSON" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "GeoPackage" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
                output_path = temp_file.name
                temp_file.close()
            else:  # Shapefile
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "masks.shp")

        try:
            unique = self.unique_check.isChecked()

            if "Raster" in format_text:
                # Save as raster
                self.sam.save_masks(output=output_path, unique=unique)

                if self.add_to_map_check.isChecked():
                    layer_name = (
                        "samgeo_masks"
                        if use_temp_file
                        else os.path.basename(output_path)
                    )
                    layer = QgsRasterLayer(output_path, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
            else:
                # Save as vector - first save as raster, then convert
                temp_raster = tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False
                ).name
                try:
                    self.sam.save_masks(output=temp_raster, unique=unique)

                    from .._geoai_lib import get_geoai

                    geoai = get_geoai()

                    min_area = self.min_area_spin.value()

                    # Check vector processing mode: 0 = Simple, 1 = Regularize, 2 = Smooth
                    vector_mode = self.vector_mode_combo.currentIndex()

                    if vector_mode == 0:
                        # Simple mode - just convert raster to vector
                        gdf = geoai.raster_to_vector(
                            temp_raster,
                            output_path=output_path,
                            min_area=min_area if min_area > 0 else 0,
                            simplify_tolerance=None,
                        )
                    elif vector_mode == 2:
                        # Use smooth_vector for natural features
                        smooth_iterations = self.smooth_iterations_spin.value()

                        # First convert raster to vector (min_area=0 means no filtering)
                        gdf = geoai.raster_to_vector(
                            temp_raster,
                            min_area=min_area if min_area > 0 else 0,
                            simplify_tolerance=None,
                        )

                        # Apply smoothing
                        gdf = geoai.smooth_vector(
                            gdf,
                            smooth_iterations=smooth_iterations,
                            output_path=output_path,
                        )
                    else:
                        # Use orthogonalize for regularization (buildings)
                        epsilon = self.epsilon_spin.value()

                        gdf = geoai.orthogonalize(
                            temp_raster,
                            output_path,
                            epsilon=epsilon,
                        )

                        # Apply min area filter if specified
                        if min_area > 0:
                            gdf = geoai.add_geometric_properties(gdf, area_unit="m2")
                            gdf = gdf[gdf["area_m2"] >= min_area]
                            # Determine driver based on output format
                            if output_path.endswith(".geojson"):
                                driver = "GeoJSON"
                            elif output_path.endswith(".gpkg"):
                                driver = "GPKG"
                            elif output_path.endswith(".shp"):
                                driver = "ESRI Shapefile"
                            else:
                                driver = None
                            gdf.to_file(output_path, driver=driver)

                    if self.add_to_map_check.isChecked():
                        layer_name = (
                            "samgeo_masks"
                            if use_temp_file
                            else os.path.basename(output_path)
                        )
                        layer = QgsVectorLayer(output_path, layer_name, "ogr")
                        if layer.isValid():
                            QgsProject.instance().addMapLayer(layer)
                finally:
                    if os.path.exists(temp_raster):
                        os.remove(temp_raster)

            self.results_text.append(f"\nAuto-saved to: {output_path}")
            self.log_message(f"Auto-saved masks to: {output_path}")

        except Exception as e:
            self.log_message(f"Auto-show failed: {str(e)}", level=Qgis.Warning)
            self.show_error(f"Auto-show failed: {str(e)}")

    def save_masks(self):
        """Save the segmentation masks."""
        if self.sam is None or self.sam.masks is None or len(self.sam.masks) == 0:
            self.show_error("No masks to save. Please run segmentation first.")
            return

        import tempfile

        output_path = self.output_path_edit.text().strip()
        format_text = self.output_format_combo.currentText()

        # Generate temp file path if not specified
        use_temp_file = False
        if not output_path:
            use_temp_file = True
            if "Raster" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "GeoJSON" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
                output_path = temp_file.name
                temp_file.close()
            elif "GeoPackage" in format_text:
                temp_file = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
                output_path = temp_file.name
                temp_file.close()
            else:  # Shapefile
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "masks.shp")

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            QCoreApplication.processEvents()

            unique = self.unique_check.isChecked()

            if "Raster" in format_text:
                # Save as raster
                self.sam.save_masks(output=output_path, unique=unique)

                if self.add_to_map_check.isChecked():
                    layer_name = (
                        "samgeo_masks"
                        if use_temp_file
                        else os.path.basename(output_path)
                    )
                    layer = QgsRasterLayer(output_path, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
            else:
                # Save as vector - first save as raster, then convert
                temp_raster = tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False
                ).name
                try:
                    self.sam.save_masks(output=temp_raster, unique=unique)

                    from .._geoai_lib import get_geoai

                    geoai = get_geoai()

                    min_area = self.min_area_spin.value()

                    # Check vector processing mode: 0 = Simple, 1 = Regularize, 2 = Smooth
                    vector_mode = self.vector_mode_combo.currentIndex()

                    if vector_mode == 0:
                        # Simple mode - just convert raster to vector
                        gdf = geoai.raster_to_vector(
                            temp_raster,
                            output_path=output_path,
                            min_area=min_area if min_area > 0 else 0,
                            simplify_tolerance=None,
                        )
                    elif vector_mode == 2:
                        # Use smooth_vector for natural features
                        smooth_iterations = self.smooth_iterations_spin.value()

                        # First convert raster to vector (min_area=0 means no filtering)
                        gdf = geoai.raster_to_vector(
                            temp_raster,
                            min_area=min_area if min_area > 0 else 0,
                            simplify_tolerance=None,
                        )

                        # Apply smoothing
                        gdf = geoai.smooth_vector(
                            gdf,
                            smooth_iterations=smooth_iterations,
                            output_path=output_path,
                        )
                    else:
                        # Use orthogonalize for regularization (buildings)
                        epsilon = self.epsilon_spin.value()

                        gdf = geoai.orthogonalize(
                            temp_raster,
                            output_path,
                            epsilon=epsilon,
                        )

                        # Apply min area filter if specified
                        if min_area > 0:
                            gdf = geoai.add_geometric_properties(gdf, area_unit="m2")
                            gdf = gdf[gdf["area_m2"] >= min_area]
                            # Determine driver based on output format
                            if output_path.endswith(".geojson"):
                                driver = "GeoJSON"
                            elif output_path.endswith(".gpkg"):
                                driver = "GPKG"
                            elif output_path.endswith(".shp"):
                                driver = "ESRI Shapefile"
                            else:
                                driver = None
                            gdf.to_file(output_path, driver=driver)

                    if self.add_to_map_check.isChecked():
                        layer_name = (
                            "samgeo_masks"
                            if use_temp_file
                            else os.path.basename(output_path)
                        )
                        layer = QgsVectorLayer(output_path, layer_name, "ogr")
                        if layer.isValid():
                            QgsProject.instance().addMapLayer(layer)
                finally:
                    if os.path.exists(temp_raster):
                        os.remove(temp_raster)
            self.results_text.append(f"\nSaved to: {output_path}")
            self.log_message(f"Masks saved to: {output_path}")

        except Exception as e:
            self.show_error(f"Failed to save masks: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def show_error(self, message):
        """Show an error message."""
        QMessageBox.critical(self, "SamGeo Error", message)
        self.log_message(message, level=Qgis.Critical)

    def log_message(self, message, level=Qgis.Info):
        """Log a message to QGIS."""
        QgsMessageLog.logMessage(message, "GeoAI - SamGeo", level)

    def cleanup(self):
        """Clean up resources when the dock is closed."""
        # Clear map tools markers
        if self.point_tool is not None:
            self.point_tool.clear_markers()

        if self.batch_point_tool is not None:
            self.batch_point_tool.clear_markers()

        if self.box_tool is not None:
            self.box_tool.clear_rubber_band()

        # Clean up temporary files (e.g., exported GeoPackage rasters)
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
        if self.sam is not None:
            del self.sam
            self.sam = None

    def closeEvent(self, event):
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)
