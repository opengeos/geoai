"""
Water Segmentation Dock Widget for GeoAI Plugin

This dock widget provides an interface for water body segmentation from
satellite and aerial imagery using the OmniWaterMask model via the
geoai.segment_water() function.
"""

import os
import sys
import tempfile
import traceback

try:
    import torch
except ImportError:
    torch = None

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
    QgsRasterFileWriter,
    QgsRasterLayer,
    QgsRasterPipe,
    QgsRasterTransparency,
    QgsVectorLayer,
    Qgis,
)


class OutputCapture:
    """Capture stdout and emit lines to a callback in real-time."""

    def __init__(self, callback, original_stdout):
        """Initialize the output capture.

        Args:
            callback: Function to call for each line of output.
            original_stdout: The original stdout stream to also write to.
        """
        self.callback = callback
        self.original_stdout = original_stdout
        self.buffer = ""

    def write(self, text):
        """Write text to buffer, emitting complete lines via callback.

        Args:
            text: Text to write.
        """
        if self.original_stdout:
            self.original_stdout.write(text)

        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.callback(line)

    def flush(self):
        """Flush the buffer, emitting any remaining content."""
        if self.original_stdout:
            self.original_stdout.flush()
        if self.buffer.strip():
            self.callback(self.buffer)
            self.buffer = ""


class WaterSegmentationWorker(QThread):
    """Worker thread for running water segmentation via geoai.segment_water()."""

    finished = pyqtSignal(str, str)  # (output_raster_path, output_vector_path)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        input_path,
        band_order,
        output_raster,
        output_vector,
        batch_size,
        device,
        dtype,
        no_data_value,
        patch_size,
        overlap_size,
        use_osm_water,
        use_osm_building,
        use_osm_roads,
        min_size,
        smooth,
        smooth_iterations,
    ):
        """Initialize the water segmentation worker.

        Args:
            input_path: Path to the input raster file.
            band_order: Band order preset string or list of 4 band indices.
            output_raster: Path for the output water mask raster (or None).
            output_vector: Path for the output vector polygons (or None).
            batch_size: Number of scenes to process in parallel.
            device: Inference device string (or None for auto).
            dtype: Inference precision string.
            no_data_value: No-data pixel value.
            patch_size: Patch size for sliding-window inference.
            overlap_size: Overlap between patches.
            use_osm_water: Whether to use OSM water features.
            use_osm_building: Whether to use OSM building data.
            use_osm_roads: Whether to use OSM road data.
            min_size: Minimum polygon size in pixels for vectorization.
            smooth: Whether to smooth vectorized polygons.
            smooth_iterations: Number of smoothing iterations.
        """
        super().__init__()
        self.input_path = input_path
        self.band_order = band_order
        self.output_raster = output_raster
        self.output_vector = output_vector
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.no_data_value = no_data_value
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.use_osm_water = use_osm_water
        self.use_osm_building = use_osm_building
        self.use_osm_roads = use_osm_roads
        self.min_size = min_size
        self.smooth = smooth
        self.smooth_iterations = smooth_iterations

    def run(self):
        """Run water segmentation in background thread."""
        try:
            from .._geoai_lib import get_geoai

            geoai = get_geoai()

            self.progress.emit("Starting water segmentation...")

            kwargs = {
                "input_path": self.input_path,
                "band_order": self.band_order,
                "batch_size": self.batch_size,
                "dtype": self.dtype,
                "no_data_value": self.no_data_value,
                "patch_size": self.patch_size,
                "overlap_size": self.overlap_size,
                "use_osm_water": self.use_osm_water,
                "use_osm_building": self.use_osm_building,
                "use_osm_roads": self.use_osm_roads,
                "min_size": self.min_size,
                "smooth": self.smooth,
                "smooth_iterations": self.smooth_iterations,
                "overwrite": True,
                "verbose": True,
            }

            if self.output_raster:
                kwargs["output_raster"] = self.output_raster
            if self.output_vector:
                kwargs["output_vector"] = self.output_vector
            if self.device:
                kwargs["device"] = self.device

            # Capture stdout to forward verbose messages
            old_stdout = sys.stdout
            sys.stdout = OutputCapture(
                lambda line: self.progress.emit(line), old_stdout
            )
            try:
                result = geoai.segment_water(**kwargs)
            finally:
                sys.stdout = old_stdout

            # Determine output paths.
            # segment_water returns a raster path (str) when no vector output
            # is requested, or a GeoDataFrame when vectorization is enabled.
            # The raster is always produced regardless, so derive its path.
            if isinstance(result, str):
                output_raster = result
            elif self.output_raster:
                output_raster = self.output_raster
            else:
                # Auto-derived by segment_water: <stem>_water_mask.tif
                stem = os.path.splitext(self.input_path)[0]
                output_raster = f"{stem}_water_mask.tif"

            output_vector = self.output_vector or ""

            self.finished.emit(output_raster, output_vector)

        except Exception as e:
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class WaterSegmentationDockWidget(QDockWidget):
    """Dock widget for water body segmentation using OmniWaterMask."""

    def __init__(self, iface, parent=None):
        """Initialize the Water Segmentation dock widget.

        Args:
            iface: The QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("Water Segmentation", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.worker = None
        self._temp_files = []

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Header
        header_label = QLabel(
            "<b>Water Segmentation</b> — "
            '<a href="https://github.com/DPIRD-DMA/OmniWaterMask">OmniWaterMask</a>'
        )
        header_label.setOpenExternalLinks(True)
        main_layout.addWidget(header_label)

        # Input group
        main_layout.addWidget(self._create_input_group())

        # Band order group
        main_layout.addWidget(self._create_band_order_group())

        # Processing settings group
        main_layout.addWidget(self._create_processing_group())

        # OSM refinement group
        main_layout.addWidget(self._create_osm_group())

        # Vectorization group
        main_layout.addWidget(self._create_vectorization_group())

        # Output group
        main_layout.addWidget(self._create_output_group())

        # Run button
        self.run_btn = QPushButton("Run Water Segmentation")
        self.run_btn.clicked.connect(self.run_segmentation)
        main_layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        # Log area
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
        """Create the input selection group.

        Returns:
            QGroupBox with input selection controls.
        """
        group = QGroupBox("Input")
        layout = QVBoxLayout()

        # Raster layer combo
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Layer:"))
        self.layer_combo = QComboBox()
        self.refresh_layers()
        layer_row.addWidget(self.layer_combo)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self.refresh_layers)
        layer_row.addWidget(refresh_btn)
        layout.addLayout(layer_row)

        # File path alternative
        file_row = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Or select raster file...")
        file_row.addWidget(self.input_path_edit)

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self.browse_input)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        group.setLayout(layout)
        return group

    def _create_band_order_group(self):
        """Create the sensor/band order selection group.

        Returns:
            QGroupBox with band order controls.
        """
        group = QGroupBox("Sensor / Band Order")
        layout = QVBoxLayout()

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.band_order_combo = QComboBox()
        self.band_order_combo.addItems(["NAIP", "Sentinel-2", "Landsat", "Custom"])
        self.band_order_combo.currentTextChanged.connect(self._on_band_order_changed)
        preset_row.addWidget(self.band_order_combo)
        layout.addLayout(preset_row)

        # Custom band order (hidden by default)
        self.custom_band_widget = QWidget()
        custom_layout = QHBoxLayout()
        custom_layout.setContentsMargins(0, 0, 0, 0)

        self.band_r_spin = QSpinBox()
        self.band_r_spin.setRange(1, 20)
        self.band_r_spin.setValue(1)

        self.band_g_spin = QSpinBox()
        self.band_g_spin.setRange(1, 20)
        self.band_g_spin.setValue(2)

        self.band_b_spin = QSpinBox()
        self.band_b_spin.setRange(1, 20)
        self.band_b_spin.setValue(3)

        self.band_nir_spin = QSpinBox()
        self.band_nir_spin.setRange(1, 20)
        self.band_nir_spin.setValue(4)

        for label, spin in [
            ("R:", self.band_r_spin),
            ("G:", self.band_g_spin),
            ("B:", self.band_b_spin),
            ("NIR:", self.band_nir_spin),
        ]:
            custom_layout.addWidget(QLabel(label))
            custom_layout.addWidget(spin)

        self.custom_band_widget.setLayout(custom_layout)
        self.custom_band_widget.setVisible(False)
        layout.addWidget(self.custom_band_widget)

        group.setLayout(layout)
        return group

    def _create_processing_group(self):
        """Create the processing settings group.

        Returns:
            QGroupBox with processing parameter controls.
        """
        group = QGroupBox("Processing Settings")
        layout = QVBoxLayout()

        # Device
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        device_row.addWidget(self.device_combo)
        layout.addLayout(device_row)

        # Dtype
        dtype_row = QHBoxLayout()
        dtype_row.addWidget(QLabel("Dtype:"))
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["float32", "float16", "bfloat16"])
        self.dtype_combo.setToolTip(
            "Inference precision. float16 is faster but requires GPU support."
        )
        dtype_row.addWidget(self.dtype_combo)
        layout.addLayout(dtype_row)

        # Batch size
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        batch_row.addWidget(self.batch_size_spin)
        layout.addLayout(batch_row)

        # Patch size
        patch_row = QHBoxLayout()
        patch_row.addWidget(QLabel("Patch Size:"))
        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(100, 4000)
        self.patch_size_spin.setValue(1000)
        self.patch_size_spin.setSingleStep(100)
        patch_row.addWidget(self.patch_size_spin)
        layout.addLayout(patch_row)

        # Overlap size
        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("Overlap Size:"))
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 1000)
        self.overlap_spin.setValue(300)
        self.overlap_spin.setSingleStep(50)
        overlap_row.addWidget(self.overlap_spin)
        layout.addLayout(overlap_row)

        # No data value
        nodata_row = QHBoxLayout()
        nodata_row.addWidget(QLabel("No Data Value:"))
        self.nodata_spin = QSpinBox()
        self.nodata_spin.setRange(-9999, 9999)
        self.nodata_spin.setValue(0)
        nodata_row.addWidget(self.nodata_spin)
        layout.addLayout(nodata_row)

        group.setLayout(layout)
        return group

    def _create_osm_group(self):
        """Create the OSM refinement group.

        Returns:
            QGroupBox with OSM option checkboxes.
        """
        group = QGroupBox("OSM Refinement")
        layout = QVBoxLayout()

        self.osm_water_check = QCheckBox("Use OSM Water features")
        self.osm_water_check.setChecked(True)
        layout.addWidget(self.osm_water_check)

        self.osm_building_check = QCheckBox(
            "Use OSM Buildings (reduce false positives)"
        )
        self.osm_building_check.setChecked(True)
        layout.addWidget(self.osm_building_check)

        self.osm_roads_check = QCheckBox("Use OSM Roads (reduce false positives)")
        self.osm_roads_check.setChecked(True)
        layout.addWidget(self.osm_roads_check)

        group.setLayout(layout)
        return group

    def _create_vectorization_group(self):
        """Create the vectorization options group.

        Returns:
            QGroupBox with vectorization controls.
        """
        group = QGroupBox("Vectorization (Optional)")
        layout = QVBoxLayout()

        self.vectorize_check = QCheckBox("Generate vector polygons")
        self.vectorize_check.setChecked(False)
        self.vectorize_check.toggled.connect(self._on_vectorize_toggled)
        layout.addWidget(self.vectorize_check)

        # Vectorization settings (initially disabled)
        self.vectorize_settings_widget = QWidget()
        vec_layout = QVBoxLayout()
        vec_layout.setContentsMargins(0, 0, 0, 0)

        # Min area
        min_size_row = QHBoxLayout()
        min_size_row.addWidget(QLabel("Min Size (pixels):"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 100000)
        self.min_size_spin.setValue(10)
        self.min_size_spin.setSingleStep(5)
        min_size_row.addWidget(self.min_size_spin)
        vec_layout.addLayout(min_size_row)

        # Smooth
        self.smooth_check = QCheckBox("Smooth polygons")
        self.smooth_check.setChecked(True)
        self.smooth_check.toggled.connect(self._on_smooth_toggled)
        vec_layout.addWidget(self.smooth_check)

        # Smooth iterations
        smooth_iter_row = QHBoxLayout()
        smooth_iter_row.addWidget(QLabel("Smooth Iterations:"))
        self.smooth_iterations_spin = QSpinBox()
        self.smooth_iterations_spin.setRange(1, 20)
        self.smooth_iterations_spin.setValue(3)
        smooth_iter_row.addWidget(self.smooth_iterations_spin)
        vec_layout.addLayout(smooth_iter_row)

        self.vectorize_settings_widget.setLayout(vec_layout)
        self.vectorize_settings_widget.setEnabled(False)
        layout.addWidget(self.vectorize_settings_widget)

        group.setLayout(layout)
        return group

    def _create_output_group(self):
        """Create the output path selection group.

        Returns:
            QGroupBox with output path controls.
        """
        group = QGroupBox("Output")
        layout = QVBoxLayout()

        # Raster output
        raster_row = QHBoxLayout()
        raster_row.addWidget(QLabel("Raster:"))
        self.output_raster_edit = QLineEdit()
        self.output_raster_edit.setPlaceholderText("Water mask output path (.tif)")
        raster_row.addWidget(self.output_raster_edit)

        raster_browse_btn = QPushButton("...")
        raster_browse_btn.setMaximumWidth(30)
        raster_browse_btn.clicked.connect(self.browse_output_raster)
        raster_row.addWidget(raster_browse_btn)
        layout.addLayout(raster_row)

        # Vector output
        self.vector_output_row = QWidget()
        vec_out_layout = QHBoxLayout()
        vec_out_layout.setContentsMargins(0, 0, 0, 0)
        vec_out_layout.addWidget(QLabel("Vector:"))
        self.output_vector_edit = QLineEdit()
        self.output_vector_edit.setPlaceholderText(
            "Water polygons output (.gpkg, .geojson, .shp)"
        )
        vec_out_layout.addWidget(self.output_vector_edit)

        vector_browse_btn = QPushButton("...")
        vector_browse_btn.setMaximumWidth(30)
        vector_browse_btn.clicked.connect(self.browse_output_vector)
        vec_out_layout.addWidget(vector_browse_btn)
        self.vector_output_row.setLayout(vec_out_layout)
        self.vector_output_row.setEnabled(False)
        layout.addWidget(self.vector_output_row)

        # Add to map
        self.add_to_map_check = QCheckBox("Add results to map")
        self.add_to_map_check.setChecked(True)
        layout.addWidget(self.add_to_map_check)

        group.setLayout(layout)
        return group

    # ---- Slot methods ----

    def _on_band_order_changed(self, text):
        """Show/hide custom band order spinboxes.

        Args:
            text: The selected band order preset text.
        """
        self.custom_band_widget.setVisible(text == "Custom")

    def _on_vectorize_toggled(self, checked):
        """Enable/disable vectorization settings and vector output path.

        Args:
            checked: Whether vectorization is enabled.
        """
        self.vectorize_settings_widget.setEnabled(checked)
        self.vector_output_row.setEnabled(checked)

    def _on_smooth_toggled(self, checked):
        """Enable/disable smooth iterations spinbox.

        Args:
            checked: Whether smoothing is enabled.
        """
        self.smooth_iterations_spin.setEnabled(checked)

    # ---- Browse methods ----

    def refresh_layers(self):
        """Refresh the list of raster layers in the combo box."""
        self.layer_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.layer_combo.addItem(layer.name(), layer.id())

    def browse_input(self):
        """Browse for an input raster file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Raster",
            "",
            "Raster files (*.tif *.tiff);;All files (*.*)",
        )
        if file_path:
            self.input_path_edit.setText(file_path)

    def browse_output_raster(self):
        """Browse for output raster file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Water Mask Raster",
            "",
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if file_path:
            self.output_raster_edit.setText(file_path)

    def browse_output_vector(self):
        """Browse for output vector file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Water Body Polygons",
            "",
            "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;Shapefile (*.shp);;All files (*.*)",
        )
        if file_path:
            self.output_vector_edit.setText(file_path)

    # ---- Input resolution ----

    def _resolve_input_path(self):
        """Resolve input raster path from either file path edit or layer combo.

        Returns:
            Tuple of (file_path, error_message). If successful, error_message
            is None. If failed, file_path is None.
        """
        # Priority: file path edit > layer combo
        file_path = self.input_path_edit.text().strip()
        if file_path:
            if not os.path.exists(file_path):
                return None, f"Input file not found: {file_path}"
            return file_path, None

        layer_id = self.layer_combo.currentData()
        if not layer_id:
            return None, "Please select a raster layer or provide a file path."

        layer = QgsProject.instance().mapLayer(layer_id)
        if not layer:
            return None, "Selected layer not found."

        source = layer.source()

        # Handle GeoPackage rasters
        if self._is_geopackage_raster(source):
            temp_path = self._export_geopackage_raster(layer)
            if temp_path is None:
                return None, "Failed to export GeoPackage raster."
            return temp_path, None

        if not os.path.exists(source):
            return None, f"Layer source file not found: {source}"
        return source, None

    def _is_geopackage_raster(self, source):
        """Check if the layer source is a raster inside a GeoPackage.

        Args:
            source: The layer source string.

        Returns:
            True if the source is a GeoPackage raster.
        """
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
        """Export a GeoPackage raster layer to a temporary GeoTIFF file.

        Args:
            layer: The QGIS raster layer to export.

        Returns:
            Path to the exported temporary GeoTIFF, or None on failure.
        """
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False, prefix="water_seg_gpkg_"
            )
            temp_path = temp_file.name
            temp_file.close()

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

            file_writer = QgsRasterFileWriter(temp_path)
            file_writer.setOutputFormat("GTiff")

            error = file_writer.writeRaster(
                pipe,
                provider.xSize(),
                provider.ySize(),
                provider.extent(),
                provider.crs(),
            )

            if error == QgsRasterFileWriter.NoError:
                self.log_message(f"Exported GeoPackage raster to: {temp_path}")
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
                f"Exception exporting GeoPackage raster: {str(e)}",
                level=Qgis.Warning,
            )
            return None

    def _resolve_band_order(self):
        """Resolve band_order parameter from UI controls.

        Returns:
            Band order string preset or list of 4 band indices.
        """
        preset = self.band_order_combo.currentText()
        if preset == "NAIP":
            return "naip"
        elif preset == "Sentinel-2":
            return "sentinel2"
        elif preset == "Landsat":
            return "landsat"
        else:
            return [
                self.band_r_spin.value(),
                self.band_g_spin.value(),
                self.band_b_spin.value(),
                self.band_nir_spin.value(),
            ]

    # ---- Run segmentation ----

    def run_segmentation(self):
        """Validate inputs and run water segmentation in a worker thread."""
        if self.worker is not None and self.worker.isRunning():
            self.show_error("Segmentation is already running.")
            return

        input_path, error = self._resolve_input_path()
        if error:
            self.show_error(error)
            return

        band_order = self._resolve_band_order()
        output_raster = self.output_raster_edit.text().strip() or None

        output_vector = None
        if self.vectorize_check.isChecked():
            output_vector = self.output_vector_edit.text().strip()
            if not output_vector:
                # Auto-derive from input path
                base = os.path.splitext(input_path)[0]
                output_vector = f"{base}_water_bodies.gpkg"
                self.output_vector_edit.setText(output_vector)

        device = self.device_combo.currentText()
        if device == "auto":
            device = None

        # Disable UI and show progress
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running water segmentation...")
        self.status_label.setStyleSheet("color: orange;")
        self.results_text.clear()

        self.worker = WaterSegmentationWorker(
            input_path=input_path,
            band_order=band_order,
            output_raster=output_raster,
            output_vector=output_vector,
            batch_size=self.batch_size_spin.value(),
            device=device,
            dtype=self.dtype_combo.currentText(),
            no_data_value=self.nodata_spin.value(),
            patch_size=self.patch_size_spin.value(),
            overlap_size=self.overlap_spin.value(),
            use_osm_water=self.osm_water_check.isChecked(),
            use_osm_building=self.osm_building_check.isChecked(),
            use_osm_roads=self.osm_roads_check.isChecked(),
            min_size=self.min_size_spin.value(),
            smooth=self.smooth_check.isChecked(),
            smooth_iterations=self.smooth_iterations_spin.value(),
        )
        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)
        self.worker.progress.connect(self._on_segmentation_progress)
        self.worker.start()

    def _on_segmentation_finished(self, raster_path, vector_path):
        """Handle successful segmentation completion.

        Args:
            raster_path: Path to the output water mask raster.
            vector_path: Path to the output vector polygons (may be empty).
        """
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Segmentation complete!")
        self.status_label.setStyleSheet("color: green;")

        if raster_path:
            self.output_raster_edit.setText(raster_path)
            self.results_text.append(f"Raster output: {raster_path}")
        if vector_path:
            self.results_text.append(f"Vector output: {vector_path}")

        if self.add_to_map_check.isChecked():
            # Add raster result
            if raster_path and os.path.exists(raster_path):
                raster_layer = QgsRasterLayer(raster_path, "Water Mask")
                if raster_layer.isValid():
                    self._apply_water_raster_style(raster_layer)
                    QgsProject.instance().addMapLayer(raster_layer)
                    self.log_message(f"Water mask raster added to map: {raster_path}")
                else:
                    self.log_message(
                        f"Failed to load raster layer: {raster_path}",
                        level=Qgis.Warning,
                    )

            # Add vector result
            if vector_path and os.path.exists(vector_path):
                vector_layer = QgsVectorLayer(vector_path, "Water Bodies", "ogr")
                if vector_layer.isValid():
                    self._apply_water_vector_style(vector_layer)
                    QgsProject.instance().addMapLayer(vector_layer)
                    self.log_message(f"Water body polygons added to map: {vector_path}")
                else:
                    self.log_message(
                        f"Failed to load vector layer: {vector_path}",
                        level=Qgis.Warning,
                    )

        self.log_message("Water segmentation completed successfully.")

    def _on_segmentation_error(self, error_message):
        """Handle segmentation error.

        Args:
            error_message: The error message string.
        """
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Segmentation failed")
        self.status_label.setStyleSheet("color: red;")
        self.results_text.append(f"Error: {error_message}")
        self.show_error(f"Water segmentation failed:\n{error_message}")

    def _on_segmentation_progress(self, message):
        """Handle progress updates from the worker.

        Args:
            message: Progress message string.
        """
        self.status_label.setText(message)
        self.results_text.append(message)

    # ---- Result styling ----

    def _apply_water_raster_style(self, layer):
        """Apply water-appropriate styling to the raster mask layer.

        Makes value 0 (non-water) transparent and sets overall opacity.

        Args:
            layer: The QgsRasterLayer to style.
        """
        try:
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
        except Exception:
            pass

    def _apply_water_vector_style(self, layer):
        """Apply a water-themed semi-transparent style to vector polygons.

        Args:
            layer: The QgsVectorLayer to style.
        """
        try:
            symbol = QgsFillSymbol.createSimple(
                {
                    "color": "0,100,255,80",
                    "outline_color": "0,50,200,255",
                    "outline_width": "0.4",
                }
            )
            layer.renderer().setSymbol(symbol)
            layer.triggerRepaint()
        except Exception:
            pass

    # ---- Utility methods ----

    def show_error(self, message):
        """Show an error message dialog.

        Args:
            message: The error message to display.
        """
        QMessageBox.critical(self, "Water Segmentation Error", message)
        self.log_message(message, level=Qgis.Critical)

    def log_message(self, message, level=Qgis.Info):
        """Log a message to the QGIS message log.

        Args:
            message: The message to log.
            level: The log level (default: Qgis.Info).
        """
        QgsMessageLog.logMessage(message, "GeoAI - Water Segmentation", level)

    def cleanup(self):
        """Clean up resources when the dock is closed."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.log_message(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.log_message(
                    f"Failed to clean up temp file {temp_file}: {e}",
                    level=Qgis.Warning,
                )

    def closeEvent(self, event):
        """Handle dock widget close event.

        Args:
            event: The close event.
        """
        self.cleanup()
        super().closeEvent(event)
