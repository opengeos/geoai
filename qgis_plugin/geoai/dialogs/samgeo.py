"""
SamGeo Dock Widget for GeoAI Plugin

This dock widget provides an interface for remote sensing image segmentation
using the SamGeo library (SAM, SAM2, and SAM3 models).
"""

import os
import shutil
import sys
import tempfile
import threading

from qgis.PyQt.QtCore import Qt, QCoreApplication, QTimer
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

from qgis.PyQt.QtCore import QThread, pyqtSignal

from .map_tools import PointPromptTool, BoxPromptTool
from .._pkg_resources_compat import ensure_pkg_resources
from ..core.lazy_torch import import_torch
from ..core.samgeo_outputs import (
    add_mask_attributes,
    count_raster_labels,
    default_output_path,
    format_prompt_label,
    hf_access_help,
    output_extension,
    redact,
    source_file_path,
    suggested_output_name,
    temp_output_path,
)


def _use_samgeo_subprocess() -> bool:
    """Decide whether to run SamGeo in the managed venv subprocess.

    Importing the managed venv's packages into the QGIS process is unsafe on
    every platform, not just Windows. QGIS ships its own NumPy and imports it
    at startup, so prepending the venv's site-packages to ``sys.path`` cannot
    displace it: the venv's SciPy/scikit stack (built for NumPy 2) then runs
    against QGIS's NumPy 1.x and fails on NumPy-2-only names --
    ``numpy.lib.array_utils`` (issue #688) and ``np.long`` reached via
    ``scipy.sparse`` (issue #854). This was gated to Windows, where the
    subprocess is also required for an unrelated PyTorch DLL conflict, which
    left macOS and Linux exposed to the skew.

    Windows always uses the subprocess because of that DLL conflict. Elsewhere
    it is used whenever the managed venv exists, which is exactly when the
    version skew is possible. Without a managed venv there is nothing to skew
    against -- SamGeo can only come from the interpreter running QGIS -- so the
    in-process path stays available for users who launch QGIS from an
    environment that already provides SamGeo.

    Returns:
        True if model loading and inference should run in a subprocess.
    """
    if os.name == "nt":
        return True

    from ..core.venv_manager import venv_exists

    return venv_exists()


# Worker threads still running when their dock was cleaned up. Holding a
# reference here keeps Qt from destroying a running QThread (which aborts the
# process); each entry drops out once its thread has finished.
_LIVE_THREADS = set()


def _keep_thread_alive(thread):
    """Detach a still-running worker thread from its dock and keep it referenced.

    The thread's result signals are disconnected (their slots belong to the
    dock being released) and the thread is dropped from ``_LIVE_THREADS``
    once it finishes.

    Args:
        thread: A ``SamGeoOperationWorker`` or other QThread with ``finished``
            and ``error`` signals.
    """
    for signal_name in ("finished", "error", "progress"):
        signal = getattr(thread, signal_name, None)
        if signal is None:
            continue
        try:
            signal.disconnect()
        except (TypeError, RuntimeError):
            pass

    def _drop(*_args):
        _LIVE_THREADS.discard(thread)

    _LIVE_THREADS.add(thread)
    if isinstance(thread, QThread):
        QThread.finished.__get__(thread, QThread).connect(_drop)
    else:
        for signal_name in ("finished", "error"):
            signal = getattr(thread, signal_name, None)
            if signal is not None:
                signal.connect(_drop)
    if not thread.isRunning():
        _LIVE_THREADS.discard(thread)


def _retire_client_in_background(client, wait_worker=None) -> None:
    """Close a superseded or busy SamGeo client off the Qt GUI thread."""

    def _close():
        if wait_worker is not None:
            try:
                wait_worker.wait()
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass

    threading.Thread(target=_close, daemon=True).start()


def _point_geometry_type():
    """Return the point geometry enum compatible with QGIS 3 and QGIS 4."""
    geometry_type = getattr(Qgis, "GeometryType", None)
    if geometry_type is not None and hasattr(geometry_type, "Point"):
        return geometry_type.Point
    return QgsWkbTypes.PointGeometry


class SamGeoModelLoadWorker(QThread):
    """Worker thread for loading SamGeo model."""

    finished = pyqtSignal(object, str)  # Emits (model, model_name)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        model_version: str,
        backend: str,
        device: str,
        confidence: float,
        enable_interactive: bool,
    ):
        super().__init__()
        self.model_version = model_version
        self.backend = backend
        self.device = device
        self.confidence = confidence
        self.enable_interactive = enable_interactive
        self._client = None
        self._abort_lock = threading.Lock()
        self._aborted = False

    def abort(self):
        """Record cancellation and abort any active subprocess load without locking."""
        with self._abort_lock:
            self._aborted = True
            client = self._client
        if client is not None and hasattr(client, "abort"):
            try:
                client.abort()
            except Exception:
                pass

    def _is_aborted(self) -> bool:
        with self._abort_lock:
            return self._aborted

    def run(self):
        """Load the SamGeo model in background."""
        try:
            if self._is_aborted():
                return
            self.progress.emit("Initializing SamGeo...")
            if _use_samgeo_subprocess():
                self.progress.emit("Starting SamGeo subprocess worker...")
                from ..core.samgeo_subprocess import SamGeoSubprocessClient

                model = SamGeoSubprocessClient(
                    model_version=self.model_version,
                    backend=self.backend,
                    device=self.device,
                    confidence=self.confidence,
                    enable_interactive=self.enable_interactive,
                )
                with self._abort_lock:
                    if self._aborted:
                        return
                    self._client = model
                model.initialize()
                if self._is_aborted():
                    try:
                        model.close()
                    except Exception:
                        pass
                    return
                self.finished.emit(model, getattr(model, "model_name", "SamGeo"))
                return

            shim_installed = ensure_pkg_resources()
            if shim_installed:
                self.progress.emit(
                    "Installed pkg_resources compatibility shim for runtime imports..."
                )

            if "SamGeo3" in self.model_version:
                from samgeo import SamGeo3

                self.progress.emit("Loading SamGeo3 model...")
                model = SamGeo3(
                    backend=self.backend,
                    device=self.device,
                    confidence_threshold=self.confidence,
                    enable_inst_interactivity=self.enable_interactive,
                )
                model_name = "SamGeo3"
            elif "SamGeo2" in self.model_version:
                from samgeo import SamGeo2

                self.progress.emit("Loading SamGeo2 model...")
                model = SamGeo2(device=self.device)
                model_name = "SamGeo2"
            else:
                from samgeo import SamGeo

                self.progress.emit("Loading SamGeo model...")
                model = SamGeo(device=self.device)
                model_name = "SamGeo"

            if self._is_aborted():
                if hasattr(model, "close") and callable(getattr(model, "close")):
                    try:
                        model.close()
                    except Exception:
                        pass
                return
            self.finished.emit(model, model_name)

        except Exception as e:
            if self._is_aborted():
                client = self._client
                if client is not None and hasattr(client, "close"):
                    try:
                        client.close()
                    except Exception:
                        pass
                return
            self.error.emit(str(e))


class SamGeoOperationWorker(QThread):
    """Generic worker thread for blocking SamGeo operations.

    Runs an arbitrary callable in a background thread so the Qt event loop
    (and the spinning progress bar) remains responsive on the main thread.
    """

    finished = pyqtSignal(object)  # result returned by the callable
    error = pyqtSignal(str)  # error message on exception

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class SamGeoDockWidget(QDockWidget):
    """Dock widget for SamGeo segmentation operations."""

    def __init__(self, iface, parent=None):
        """Initialize the SamGeo dock widget.

        Args:
            iface: The QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("Segment Anything", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

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

        # Model loading worker
        self.model_load_worker = None

        # Original file behind the current image (e.g. the GeoPackage when the
        # image was exported to a temporary GeoTIFF) and the RGB bands used.
        self.current_source_path = None
        self._current_bands = None

        # Description of the prompt behind the current masks, written to the
        # ``prompt`` field of vector outputs.
        self._last_prompt = None

        # Result of a tiled text run: {"raster", "unique", "count"}. Tiled runs
        # keep no masks in memory, only this raster.
        self._tiled_result = None
        # Latest image that passed validation in a set-image attempt, recorded
        # even if the full-image set_image then fails (e.g. a huge GeoTIFF), so
        # tiled text segmentation can still use it:
        # {"image", "source", "bands"}.
        self._tiled_source = None
        # True while a save runs; temp rasters released meanwhile are deleted
        # once it finishes.
        self._save_busy = False
        self._pending_deletes = []
        # generate_masks_tiled leaves the model's image state on the last tile,
        # so the full image must be set again before other prompts.
        self._image_state_stale = False

        # Background thread for Hugging Face login/status checks, and a task
        # queued behind it (action, params, on_done, on_error).
        self._hf_worker = None
        self._hf_queued = None

        self._setup_ui()
        self.check_hf_status()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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
        if sys.platform == "darwin":
            self.backend_combo.setCurrentText("transformers")
        backend_row.addWidget(self.backend_combo)
        backend_layout.addLayout(backend_row)

        # Device selection
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "mps"])
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

        model_layout.addWidget(self._create_hf_group())

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

    def _create_hf_group(self):
        """Create the Hugging Face login group for the gated SAM 3 checkpoint.

        The ``hf`` CLI lives in the plugin's hidden environment (users on
        Windows cannot easily run ``hf auth login``), so the token can be
        saved from here instead.

        Returns:
            QGroupBox: The group box.
        """
        hf_group = QGroupBox("Hugging Face")
        hf_layout = QVBoxLayout()

        info = QLabel(
            'SAM 3 is gated: request access at <a href="https://huggingface.co/'
            'facebook/sam3">huggingface.co/facebook/sam3</a>, then paste a '
            '<a href="https://huggingface.co/settings/tokens">Read token</a>.'
        )
        info.setWordWrap(True)
        info.setOpenExternalLinks(True)
        hf_layout.addWidget(info)

        token_row = QHBoxLayout()
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText("hf_... access token")
        self.hf_token_edit.returnPressed.connect(self.hf_login)
        token_row.addWidget(self.hf_token_edit)

        self.hf_login_btn = QPushButton("Log in")
        self.hf_login_btn.setToolTip(
            "Save the token for the plugin's Python environment "
            "(same as 'hf auth login')."
        )
        self.hf_login_btn.clicked.connect(self.hf_login)
        token_row.addWidget(self.hf_login_btn)
        hf_layout.addLayout(token_row)

        self.hf_status_label = QLabel("Hugging Face: status unknown")
        self.hf_status_label.setStyleSheet("color: gray;")
        self.hf_status_label.setWordWrap(True)
        hf_layout.addWidget(self.hf_status_label)

        hf_group.setLayout(hf_layout)
        return hf_group

    def _hf_env_available(self) -> bool:
        """Check whether the managed venv needed for Hugging Face tasks exists.

        Returns:
            bool: True if the plugin's virtual environment is installed.
        """
        try:
            from ..core.venv_manager import venv_exists

            return bool(venv_exists())
        except Exception:
            return False

    def _run_hf_task(self, action, params, on_done, on_error):
        """Run a Hugging Face task in the venv without blocking the UI.

        Uses its own worker thread so it never replaces a running
        segmentation worker. If a task is already running, this one is queued
        (replacing any earlier queued task) and starts when it finishes.

        Args:
            action: ``"hf_login"`` or ``"hf_whoami"``.
            params: Task parameters.
            on_done: Called with the task result on the main thread.
            on_error: Called with the error message on the main thread.
        """
        if self._hf_worker is not None and self._hf_worker.isRunning():
            self._drop_queued_hf_task()
            self._hf_queued = (action, params, on_done, on_error)
            return

        def _work():
            from ..core.geoai_task_subprocess import run_geoai_task
            from ..core.samgeo_subprocess import hf_token_env_overrides

            # Check the account with the same token policy as new SamGeo
            # workers, so the status names the credential model loads use.
            return run_geoai_task(action, params, drop_env=hf_token_env_overrides())

        self._hf_worker = SamGeoOperationWorker(_work)
        self._hf_worker.finished.connect(on_done)
        self._hf_worker.error.connect(on_error)
        self._hf_worker.finished.connect(self._start_queued_hf_task)
        self._hf_worker.error.connect(self._start_queued_hf_task)
        self._hf_worker.start()

    def _drop_queued_hf_task(self):
        """Discard the queued Hugging Face task without running it.

        A queued login has already locked the Log in button and the token
        field; its callbacks would normally unlock them, so unlock them here
        when it is dropped instead.
        """
        queued = self._hf_queued
        self._hf_queued = None
        if queued is not None and queued[0] == "hf_login":
            try:
                self.hf_login_btn.setEnabled(True)
                self.hf_token_edit.setReadOnly(False)
            except RuntimeError:
                pass  # Widgets already deleted during teardown

    def _start_queued_hf_task(self, *_args):
        """Start the queued Hugging Face task, if any, after one finishes.

        Args:
            *_args: The finished task's result or error message (ignored).
        """
        worker = self._hf_worker
        if worker is not None and worker.isRunning():
            # Signals are delivered just before run() returns. Retry shortly
            # instead of blocking the GUI thread with wait(); the task stays
            # queued until the worker has actually exited.
            if self._hf_queued is not None:
                QTimer.singleShot(50, self._start_queued_hf_task)
            return
        queued = self._hf_queued
        self._hf_queued = None
        if queued is None:
            return
        self._run_hf_task(*queued)

    def _set_hf_status(self, text, color):
        """Update the Hugging Face status label.

        Args:
            text: Status text.
            color: CSS color name.
        """
        self.hf_status_label.setText(text)
        self.hf_status_label.setStyleSheet(f"color: {color};")

    def check_hf_status(self):
        """Show which Hugging Face account (if any) the plugin environment uses."""
        if not self._hf_env_available():
            self._set_hf_status(
                "Hugging Face: install the plugin dependencies first.", "gray"
            )
            return

        def _done(result):
            if self._hf_queued is not None:
                return  # A login was queued behind this check; it sets the status.
            result = result or {}
            if result.get("logged_in"):
                user = result.get("user") or "unknown user"
                self._set_hf_status(f"Hugging Face: logged in as {user}", "green")
            elif result.get("credentials_found"):
                self._set_hf_status(
                    result.get("message") or "Hugging Face: token not verified",
                    "orange",
                )
            else:
                self._set_hf_status("Hugging Face: not logged in", "gray")

        def _error(msg):
            if self._hf_queued is None:
                self._set_hf_status("Hugging Face: status unavailable", "gray")
            self.log_message(f"Hugging Face status check failed: {msg}")

        self._set_hf_status("Hugging Face: checking...", "gray")
        self._run_hf_task("hf_whoami", {}, _done, _error)

    def hf_login(self):
        """Log in to Hugging Face with the token typed in the panel.

        The token is sent to the venv worker over stdin and never written to
        logs or messages. The field is read-only while the login runs and is
        cleared on success. A login clicked during the startup account check
        is queued behind it.
        """
        token = self.hf_token_edit.text().strip()
        if not token:
            self.show_error("Please paste a Hugging Face access token.")
            return
        if not self._hf_env_available():
            self.show_error(
                "The plugin's Python environment is not installed. "
                "Install the dependencies first, or log in with the hf command "
                "line tool (see the plugin documentation)."
            )
            return

        self.hf_login_btn.setEnabled(False)
        self.hf_token_edit.setReadOnly(True)
        self._set_hf_status("Hugging Face: logging in...", "orange")

        def _finish():
            self.hf_login_btn.setEnabled(True)
            self.hf_token_edit.setReadOnly(False)

        def _done(result):
            _finish()
            if self.hf_token_edit.text().strip() == token:
                self.hf_token_edit.clear()
            self._prefer_saved_hf_token()
            user = (result or {}).get("user") or "unknown user"
            self._set_hf_status(f"Hugging Face: logged in as {user}", "green")
            self.log_message(f"Logged in to Hugging Face as {user}.")

        def _error(msg):
            _finish()
            safe = redact(msg, token)
            self._set_hf_status("Hugging Face: login failed", "red")
            self.show_error(safe)

        self._run_hf_task("hf_login", {"token": token}, _done, _error)

    def _prefer_saved_hf_token(self):
        """Make later model loads use the token saved by the panel login.

        Hugging Face gives ``HF_TOKEN`` (and the legacy
        ``HUGGING_FACE_HUB_TOKEN``) priority over the saved token. Those
        variables are dropped only from the environment of SamGeo workers
        started after the login; the QGIS process environment (shared with
        other plugins) is left unchanged. A model that is already loaded
        keeps working; the next Load Model starts a fresh worker.
        """
        from ..core.samgeo_subprocess import prefer_saved_hf_token

        overridden = prefer_saved_hf_token()
        if overridden:
            self.log_message(
                f"New SamGeo workers will ignore {', '.join(overridden)} so the "
                "token saved by the panel login is used."
            )

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

        # Tiled processing for images too large to segment in one pass
        self.tiled_check = QCheckBox("Process in tiles (large images)")
        self.tiled_check.setToolTip(
            "Segment a large GeoTIFF tile by tile with SamGeo3's\n"
            "generate_masks_tiled to limit GPU memory use. The merged\n"
            "mask raster is used directly for saving; per-mask confidence\n"
            "scores are not available in this mode."
        )
        self.tiled_check.toggled.connect(self._on_tiled_toggled)
        text_group_layout.addWidget(self.tiled_check)

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("Tile size:"))
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(256, 8192)
        self.tile_size_spin.setSingleStep(128)
        self.tile_size_spin.setValue(1024)
        self.tile_size_spin.setSuffix(" px")
        tile_row.addWidget(self.tile_size_spin)
        tile_row.addWidget(QLabel("Overlap:"))
        self.tile_overlap_spin = QSpinBox()
        self.tile_overlap_spin.setRange(0, 4096)
        self.tile_overlap_spin.setSingleStep(32)
        self.tile_overlap_spin.setValue(128)
        self.tile_overlap_spin.setSuffix(" px")
        tile_row.addWidget(self.tile_overlap_spin)
        text_group_layout.addLayout(tile_row)
        self.tile_size_spin.setEnabled(False)
        self.tile_overlap_spin.setEnabled(False)

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

        # Export Training Data Group
        export_group = QGroupBox("Export Training Data")
        export_group_layout = QVBoxLayout()

        # Export format
        export_format_row = QHBoxLayout()
        export_format_row.addWidget(QLabel("Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PASCAL_VOC", "COCO", "YOLO"])
        self.export_format_combo.setToolTip(
            "Export format for training data:\n"
            "- PASCAL_VOC: XML annotation files (semantic segmentation)\n"
            "- COCO: JSON annotation file (instance segmentation)\n"
            "- YOLO: Text files with normalized coordinates"
        )
        export_format_row.addWidget(self.export_format_combo)
        export_group_layout.addLayout(export_format_row)

        # Tile size
        tile_size_row = QHBoxLayout()
        tile_size_row.addWidget(QLabel("Tile Size:"))
        self.export_tile_size_spin = QSpinBox()
        self.export_tile_size_spin.setRange(64, 2048)
        self.export_tile_size_spin.setValue(512)
        self.export_tile_size_spin.setSingleStep(64)
        tile_size_row.addWidget(self.export_tile_size_spin)
        export_group_layout.addLayout(tile_size_row)

        # Stride
        stride_row = QHBoxLayout()
        stride_row.addWidget(QLabel("Stride:"))
        self.export_stride_spin = QSpinBox()
        self.export_stride_spin.setRange(32, 1024)
        self.export_stride_spin.setValue(256)
        self.export_stride_spin.setSingleStep(32)
        stride_row.addWidget(self.export_stride_spin)
        export_group_layout.addLayout(stride_row)

        # Output directory
        export_dir_row = QHBoxLayout()
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("Export output directory...")
        export_dir_row.addWidget(self.export_dir_edit)
        export_dir_browse_btn = QPushButton("...")
        export_dir_browse_btn.setMaximumWidth(30)
        export_dir_browse_btn.clicked.connect(self.browse_export_dir)
        export_dir_row.addWidget(export_dir_browse_btn)
        export_group_layout.addLayout(export_dir_row)

        # Export button
        self.export_btn = QPushButton("Export Training Tiles")
        self.export_btn.clicked.connect(self.export_training_data)
        self.export_btn.setToolTip(
            "Export image tiles with segmentation masks.\n"
            "Requires: image set and segmentation results."
        )
        export_group_layout.addWidget(self.export_btn)

        export_group.setLayout(export_group_layout)
        output_layout.addWidget(export_group)

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

    def _on_tiled_toggled(self, checked):
        """Enable the tile size and overlap inputs only when tiling is on.

        Args:
            checked: Whether "Process in tiles" is checked.
        """
        self.tile_size_spin.setEnabled(bool(checked))
        self.tile_overlap_spin.setEnabled(bool(checked))

    def _update_tiled_support(self):
        """Enable the tiled option only if the loaded model supports it."""
        supported = self._sam_supports("generate_masks_tiled")
        if self.sam is not None and not supported:
            self.tiled_check.setChecked(False)
            self.tiled_check.setEnabled(False)
            self.tiled_check.setToolTip(
                "Tiled processing needs SamGeo3 with generate_masks_tiled "
                "(update samgeo)."
            )
        else:
            self.tiled_check.setEnabled(True)
            self.tiled_check.setToolTip(
                "Segment a large GeoTIFF tile by tile with SamGeo3's\n"
                "generate_masks_tiled to limit GPU memory use. The merged\n"
                "mask raster is used directly for saving; per-mask confidence\n"
                "scores are not available in this mode."
            )

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
        checked_value = getattr(Qt.CheckState.Checked, "value", 2)
        enabled = state in (Qt.CheckState.Checked, checked_value)
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
                if layer.geometryType() == _point_geometry_type():
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
        """Browse for output file location.

        Opens in the source image's folder with a suggested, non-overwriting
        name such as ``<image>_sam_masks.gpkg``.
        """
        format_text = self.output_format_combo.currentText()
        if "GeoJSON" in format_text:
            filter_str = "GeoJSON (*.geojson)"
        elif "GeoPackage" in format_text:
            filter_str = "GeoPackage (*.gpkg)"
        elif "Shapefile" in format_text:
            filter_str = "Shapefile (*.shp)"
        else:
            filter_str = "GeoTIFF (*.tif)"

        ext = output_extension(format_text)
        start = self.output_path_edit.text().strip()
        if not start:
            start = default_output_path(self._source_image_path(), ext) or ""
        if not start:
            start = suggested_output_name(self._source_image_path(), ext)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", start, filter_str
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def _source_image_path(self):
        """Return the file that the current image came from.

        Returns:
            str or None: The original file (for example the GeoPackage behind
            an exported temporary GeoTIFF), else the image path, else None.
        """
        return self.current_source_path or self.current_image_path

    def check_cuda_devices(self):
        """Check CUDA device availability and fix CUDA_VISIBLE_DEVICES if needed.

        Only used when SamGeo runs inside the QGIS process, so PyTorch is
        imported here rather than when the panel opens.

        Returns:
            tuple: (is_cuda_available: bool, warning_message: str or None)
        """
        torch, torch_error = import_torch()
        if torch is None:
            detail = ""
            if torch_error is not None:
                detail = f"\n\nPyTorch import error in QGIS process:\n{torch_error}"
            return (
                False,
                "PyTorch is not available in this QGIS session. "
                "Please reinstall dependencies or use a subprocess-backed panel."
                f"{detail}",
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

    def _sam_supports(self, method_name: str) -> bool:
        """Check feature support on either a real SamGeo object or subprocess proxy."""
        if self.sam is None:
            return False
        supports_method = getattr(self.sam, "supports_method", None)
        if callable(supports_method):
            try:
                return bool(supports_method(method_name))
            except Exception:
                return False
        return hasattr(self.sam, method_name)

    def load_model(self):
        """Load the SamGeo model asynchronously."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate (spinning)
        self.model_status.setText("Loading model...")
        self.model_status.setStyleSheet("color: orange;")
        self.load_model_btn.setEnabled(False)

        model_version = self.model_combo.currentText()
        backend = self.backend_combo.currentText()
        device = self.device_combo.currentText()
        use_subprocess = _use_samgeo_subprocess()

        # The subprocess worker picks the device itself; PyTorch is imported
        # into QGIS only when the model runs in-process.
        torch = None
        if not use_subprocess:
            torch, torch_error = import_torch()
            if torch is None:
                self.progress_bar.setVisible(False)
                self.load_model_btn.setEnabled(True)
                err = "PyTorch is not available in the QGIS process."
                if torch_error is not None:
                    err += (
                        "\n\nImport error:\n"
                        f"{torch_error}\n\n"
                        "This is typically a Windows QGIS/PyTorch DLL incompatibility."
                    )
                self.show_error(err)
                self.model_status.setText("Model: Failed to load")
                self.model_status.setStyleSheet("color: red;")
                return

        if device == "auto":
            device = None

        # Prefer MPS on Apple Silicon in auto mode before checking CUDA.
        if not use_subprocess and device is None and torch is not None:
            try:
                if (
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                ):
                    device = "mps"
                    self.log_message("Auto mode: using Apple MPS backend")
            except Exception as e:
                self.log_message(f"MPS detection failed: {e}", level=Qgis.Warning)

        # Check CUDA availability if using CUDA or auto device selection
        if not use_subprocess and (device == "cuda" or device is None):
            cuda_available, warning_message = self.check_cuda_devices()

            if not cuda_available:
                # CUDA is not available
                if device == "cuda":
                    # User explicitly requested CUDA but it's not available
                    self.progress_bar.setVisible(False)
                    self.load_model_btn.setEnabled(True)
                    error_msg = (
                        f"CUDA device requested but not available.\n\n{warning_message}\n\n"  # nosec B608
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
                    # Avoid modal dialogs during auto device selection; these can
                    # interrupt UI flow and have caused freezes on some platforms.
                    self.model_status.setText("Loading model (CPU fallback)...")
            elif warning_message:
                # CUDA is now available but there was a warning (e.g., fixed CUDA_VISIBLE_DEVICES)
                self.log_message(
                    f"CUDA issue fixed: {warning_message}", level=Qgis.Warning
                )

        confidence = self.conf_spin.value()
        enable_interactive = self.interactive_check.isChecked()

        # Abort any previous in-flight model load before starting a new one
        prev_loader = getattr(self, "model_load_worker", None)
        if prev_loader is not None:
            self.model_load_worker = None
            if hasattr(prev_loader, "abort"):
                try:
                    prev_loader.abort()
                except Exception:
                    pass
            try:
                if prev_loader.isRunning():
                    _keep_thread_alive(prev_loader)
            except RuntimeError:
                pass

        # Create and start the model loading worker thread
        worker = SamGeoModelLoadWorker(
            model_version, backend, device, confidence, enable_interactive
        )
        self.model_load_worker = worker

        def _on_loaded(model, loaded_name, w=worker):
            if getattr(self, "model_load_worker", None) is not w:
                if hasattr(model, "close") and callable(getattr(model, "close")):
                    _retire_client_in_background(model)
                return
            self.model_load_worker = None
            self.on_model_loaded(model, loaded_name)

        def _on_error(err, w=worker):
            if getattr(self, "model_load_worker", None) is not w:
                return
            self.model_load_worker = None
            self.on_model_load_error(err)

        def _on_progress(msg, w=worker):
            if getattr(self, "model_load_worker", None) is not w:
                return
            self.on_model_load_progress(msg)

        worker.finished.connect(_on_loaded)
        worker.error.connect(_on_error)
        worker.progress.connect(_on_progress)
        worker.start()

    def on_model_load_progress(self, message: str):
        """Handle model loading progress updates."""
        self.model_status.setText(message)

    def on_model_loaded(self, model, model_name: str):
        """Handle successful model loading."""
        old_sam = self.sam
        if (
            old_sam is not None
            and old_sam is not model
            and hasattr(old_sam, "close")
            and callable(getattr(old_sam, "close"))
        ):
            op_worker = getattr(self, "_operation_worker", None)
            if op_worker is not None and op_worker.isRunning():
                _retire_client_in_background(old_sam, wait_worker=op_worker)
            else:
                try:
                    old_sam.close()
                except Exception:
                    pass
        self.sam = model
        self._set_tiled_result(None)
        self._image_state_stale = False
        self.model_status.setText(f"Model: {model_name} loaded")
        self.model_status.setStyleSheet("color: green;")
        self.progress_bar.setVisible(False)
        self.load_model_btn.setEnabled(True)
        self._update_tiled_support()
        self.log_message(f"{model_name} model loaded successfully")

    def on_model_load_error(self, error_message: str):
        """Handle model loading error."""
        self.model_status.setText("Model: Failed to load")
        self.model_status.setStyleSheet("color: red;")
        self.progress_bar.setVisible(False)
        self.load_model_btn.setEnabled(True)
        help_text = hf_access_help(error_message)
        if help_text:
            self._set_hf_status("Hugging Face: access to SAM 3 denied", "red")
            self.log_message(
                f"Failed to load model: {error_message}", level=Qgis.Critical
            )
            QMessageBox.critical(
                self,
                "Segment Anything: Hugging Face access needed",
                f"{help_text}\n\nOriginal error:\n{error_message[:1500]}",
            )
            self.hf_token_edit.setFocus()
            return
        self.show_error(f"Failed to load model: {error_message}")

    # ------------------------------------------------------------------
    # Generic threaded-operation helper
    # ------------------------------------------------------------------

    def _has_active_operation(self) -> bool:
        """Return True when a background SamGeo operation is still active."""
        existing = getattr(self, "_operation_worker", None)
        return existing is not None and existing.isRunning()

    def _run_with_progress(self, func, on_done, on_error=None):
        """Run *func* in a background thread while the progress bar spins.

        Rejects overlapping calls while an operation worker is still running,
        and retains each running worker in ``_LIVE_THREADS`` until its thread
        exits cleanly.
        """
        existing = getattr(self, "_operation_worker", None)
        if existing is not None and existing.isRunning():
            self.log_message(
                "An operation is already running; ignoring overlapping request.",
                level=Qgis.Warning,
            )
            if on_error:
                on_error("An operation is already in progress. Please wait.")
            return False

        worker = SamGeoOperationWorker(func)

        def _on_finished(result):
            if getattr(self, "_operation_worker", None) is not worker:
                return
            self._operation_worker = None
            self.progress_bar.setVisible(False)
            try:
                on_done(result)
            except Exception as exc:
                if on_error:
                    on_error(str(exc))
                else:
                    self.show_error(str(exc))

        def _on_error(msg):
            if getattr(self, "_operation_worker", None) is not worker:
                return
            self._operation_worker = None
            self.progress_bar.setVisible(False)
            if on_error:
                on_error(msg)
            else:
                self.show_error(msg)

        def _drop_thread(*_args):
            _LIVE_THREADS.discard(worker)

        _LIVE_THREADS.add(worker)
        self._operation_worker = worker
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        if isinstance(worker, QThread):
            QThread.finished.__get__(worker, QThread).connect(_drop_thread)
        else:
            worker.finished.connect(_drop_thread)
            worker.error.connect(_drop_thread)
        worker.start()
        return True

    def set_image_from_layer(self):
        """Set the image from the selected QGIS layer."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self._has_active_operation():
            self.log_message(
                "An operation is already running; ignoring overlapping set_image request.",
                level=Qgis.Warning,
            )
            self.show_error("An operation is already in progress. Please wait.")
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
            # GeoPackage export uses QGIS internals; keep on main thread.
            # It is brief; processEvents() keeps the UI alive during it.
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.image_status.setText("Exporting GeoPackage raster...")
            QCoreApplication.processEvents()

            temp_export_path = self._export_geopackage_raster(layer)
            self.progress_bar.setVisible(False)
            if temp_export_path is None:
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

        # Run the (potentially slow) model embedding step in a background thread
        # so the spinning progress bar remains animated.
        bands = self._get_bands()
        prev_tiled_source = getattr(self, "_tiled_source", None)
        new_tiled_source = {
            "image": image_path,
            "source": source_file_path(source) if is_gpkg else image_path,
            "bands": bands,
        }
        self._tiled_source = new_tiled_source

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.image_status.setText("Setting image...")

        def _work():
            self.sam.set_image(image_path, bands=bands)

        def _done(_):
            self.current_layer = layer
            self.current_image_path = image_path
            self.current_source_path = (
                source_file_path(source) if is_gpkg else image_path
            )
            self._on_image_changed(bands)
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

        def _error(msg):
            # Keep a GeoPackage export: tiled text segmentation can still use
            # it, and it is tracked in _temp_files for cleanup.
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            self.show_error(
                f"Failed to set image: {msg}{self._tiled_fallback_hint(image_path)}"
            )

        if not self._run_with_progress(_work, _done, _error):
            self._tiled_source = prev_tiled_source

    def set_image_from_file(self):
        """Set the image from the file path."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self._has_active_operation():
            self.log_message(
                "An operation is already running; ignoring overlapping set_image request.",
                level=Qgis.Warning,
            )
            self.show_error("An operation is already in progress. Please wait.")
            return

        file_path = self.image_path_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.show_error("Please select a valid image file.")
            return

        bands = self._get_bands()
        prev_tiled_source = getattr(self, "_tiled_source", None)
        self._tiled_source = {"image": file_path, "source": file_path, "bands": bands}
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.image_status.setText("Setting image...")

        def _work():
            self.sam.set_image(file_path, bands=bands)

        def _done(_):
            layer = QgsRasterLayer(file_path, os.path.basename(file_path))
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.current_layer = layer
                self.current_image_path = file_path
                self.current_source_path = file_path
                self._on_image_changed(bands)
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
                self.current_source_path = None
                self._on_image_changed(None)
                self.image_status.setText("Image: Failed to set (invalid layer)")
                self.image_status.setStyleSheet("color: red;")
                self.show_error(
                    "Failed to add image layer: The raster layer is invalid."
                )

        def _error(msg):
            self.image_status.setText("Image: Failed to set")
            self.image_status.setStyleSheet("color: red;")
            self.show_error(
                f"Failed to set image: {msg}{self._tiled_fallback_hint(file_path)}"
            )

        if not self._run_with_progress(_work, _done, _error):
            self._tiled_source = prev_tiled_source

    def segment_by_text(self):
        """Segment the image using text prompt."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        tiled = self.tiled_check.isChecked()
        if self.current_image_path is None and not (tiled and self._tiled_source):
            self.show_error("Please set an image first.")
            return

        prompt = self.text_prompt_edit.text().strip()
        if not prompt:
            self.show_error("Please enter a text prompt.")
            return

        min_size = self.min_size_spin.value()
        max_size = (
            self.max_size_spin.value() if self.max_size_spin.value() > 0 else None
        )

        if tiled:
            self._segment_by_text_tiled(prompt, min_size, max_size)
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.text_status_label.setText("Processing...")
        self.text_status_label.setStyleSheet("color: orange;")

        def _work():
            self._restore_image_if_needed()
            self.sam.generate_masks(prompt, min_size=min_size, max_size=max_size)

        def _done(_):
            self._on_new_masks(format_prompt_label("text", prompt))
            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Text Segmentation Results:\n"
                f"Prompt: {prompt}\n"
                f"Objects found: {num_masks}\n"
            )
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
            if num_masks > 0:
                self._auto_show_results()

        def _error(msg):
            self.text_status_label.setText("Segmentation failed!")
            self.text_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {msg}")

        self._run_with_progress(_work, _done, _error)

    def _on_image_changed(self, bands):
        """Reset per-image state after a new image is set.

        Args:
            bands: The RGB bands used for the image, or None.
        """
        self._current_bands = bands
        self._set_tiled_result(None)
        self._image_state_stale = False
        self._last_prompt = None

    def _on_new_masks(self, prompt_label):
        """Record a finished in-memory segmentation run.

        Args:
            prompt_label: Description of the prompt (for the ``prompt`` field).
        """
        self._last_prompt = prompt_label
        self._set_tiled_result(None)

    def _tiled_fallback_hint(self, image_path):
        """Explain that tiled text segmentation can still use an image.

        Args:
            image_path: The image whose full-image setup failed.

        Returns:
            str: A hint to append to the error, or an empty string when tiled
            processing does not apply (model unsupported or not a GeoTIFF).
        """
        if not self._sam_supports("generate_masks_tiled"):
            return ""
        if not str(image_path).lower().endswith((".tif", ".tiff")):
            return ""
        return (
            "\n\nFor large GeoTIFFs you can still segment by text: check "
            "'Process in tiles (large images)' on the Text tab."
        )

    def _set_tiled_result(self, result):
        """Replace the current tiled result and release the previous raster.

        Args:
            result: The new tiled result dict, or None to clear it.
        """
        old = self._tiled_result
        self._tiled_result = result
        if old and (result is None or old.get("raster") != result.get("raster")):
            self._discard_temp_raster(old.get("raster"))

    def _discard_temp_raster(self, path):
        """Delete a temporary raster now, or after the running save finishes.

        Args:
            path: Raster path (ignored if empty).
        """
        if not path:
            return
        if self._save_busy:
            self._pending_deletes.append(path)
            return
        if path in self._temp_files:
            self._temp_files.remove(path)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            self.log_message(
                f"Failed to remove temporary raster {path}: {exc}", level=Qgis.Warning
            )

    def _restore_image_if_needed(self):
        """Set the full image again after a tiled run moved the model off it.

        ``generate_masks_tiled`` leaves the model's image and inference state
        on the last tile, so point, box, batch and untiled text prompts need
        the image to be set again first. Runs in the worker thread.
        """
        if not self._image_state_stale or self.current_image_path is None:
            return
        self.sam.set_image(self.current_image_path, bands=self._current_bands)
        self._image_state_stale = False

    def _has_results(self):
        """Tell whether there are masks to save.

        Returns:
            bool: True if the model holds masks or a tiled run produced a raster.
        """
        if self._tiled_result is not None and os.path.exists(
            self._tiled_result.get("raster") or ""
        ):
            return True
        return bool(
            self.sam is not None
            and self.sam.masks is not None
            and len(self.sam.masks) > 0
        )

    def _segment_by_text_tiled(self, prompt, min_size, max_size):
        """Segment a large GeoTIFF tile by tile with a text prompt.

        Uses ``SamGeo3.generate_masks_tiled``, which writes the merged mask
        raster itself and keeps no masks or scores in memory. The raster is
        kept as the current result and reused by the save flows.

        Args:
            prompt: Text prompt.
            min_size: Minimum mask size in pixels.
            max_size: Maximum mask size in pixels, or None.
        """
        if not self._sam_supports("generate_masks_tiled"):
            self.show_error(
                "Tiled processing requires SamGeo3 with generate_masks_tiled. "
                "Update samgeo or uncheck 'Process in tiles'."
            )
            return
        tiled_source = self._tiled_source or {
            "image": self.current_image_path,
            "source": self._source_image_path(),
            "bands": self._current_bands,
        }
        source = tiled_source["image"]
        if not str(source).lower().endswith((".tif", ".tiff")):
            self.show_error("Tiled processing requires a GeoTIFF image.")
            return
        tile_size = self.tile_size_spin.value()
        overlap = self.tile_overlap_spin.value()
        if overlap >= tile_size:
            self.show_error("Tile overlap must be smaller than the tile size.")
            return

        unique = self.unique_check.isChecked()
        bands = tiled_source["bands"]
        source_path = tiled_source["source"]
        output = temp_output_path(".tif")
        self._temp_files.append(output)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.text_status_label.setText("Processing tiles...")
        self.text_status_label.setStyleSheet("color: orange;")

        def _work():
            kwargs = dict(
                source=source,
                prompt=prompt,
                output=output,
                tile_size=tile_size,
                overlap=overlap,
                min_size=min_size,
                max_size=max_size,
                unique=unique,
            )
            if bands:
                kwargs["bands"] = bands
            # The model's image state now points at the last tile.
            self._image_state_stale = True
            result = self.sam.generate_masks_tiled(**kwargs)
            if isinstance(result, dict):
                return result.get("object_count")
            return count_raster_labels(output, unique)

        def _done(count):
            self._set_tiled_result(
                {
                    "raster": output,
                    "unique": unique,
                    "count": count,
                    "source": source_path,
                    # The image this raster was segmented from; it can differ
                    # from current_image_path when its full-image setup failed.
                    "image": source,
                }
            )
            self._last_prompt = format_prompt_label("text", prompt)
            found = "unknown" if count is None else str(count)
            self.results_text.setText(
                f"Tiled Text Segmentation Results:\n"
                f"Prompt: {prompt}\n"
                f"Tile size: {tile_size}, overlap: {overlap}\n"
                f"Objects found: {found}\n"
            )
            if count == 0:
                self.text_status_label.setText(
                    "No objects found. Try a different prompt."
                )
                self.text_status_label.setStyleSheet("color: orange;")
                self.log_message("Tiled text segmentation complete. Found 0 objects.")
                return
            if self.auto_show_check.isChecked():
                self.text_status_label.setText(f"Found {found} object(s).")
            else:
                self.text_status_label.setText(
                    f"Found {found} object(s). Go to Output tab to save."
                )
            self.text_status_label.setStyleSheet("color: green;")
            self.log_message(
                f"Tiled text segmentation complete. Found {found} objects."
            )
            self._auto_show_results()

        def _error(msg):
            self._discard_temp_raster(output)
            self.text_status_label.setText("Segmentation failed!")
            self.text_status_label.setStyleSheet("color: red;")
            self.show_error(f"Tiled segmentation failed: {msg}")

        self._run_with_progress(_work, _done, _error)

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

        import numpy as np

        point_coords = np.array(self.point_coords)
        point_labels = np.array(self.point_labels)

        # Use multimask_output=False when there are multiple points or
        # background points, as it gives better results for non-ambiguous prompts
        has_background = 0 in point_labels
        use_multimask = len(point_coords) == 1 and not has_background
        num_points = len(self.point_coords)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.point_status_label.setText("Processing...")
        self.point_status_label.setStyleSheet("color: orange;")

        def _work():
            self._restore_image_if_needed()
            # Use appropriate method based on model type
            # point_crs=None because coordinates are already in pixel space
            # (transformed from canvas CRS to layer CRS in add_point())
            if self._sam_supports("generate_masks_by_points"):
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

        def _done(_):
            self._on_new_masks(format_prompt_label("point"))
            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Point Segmentation Results:\n"
                f"Points used: {num_points}\n"
                f"Objects found: {num_masks}\n"
            )
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
            if num_masks > 0:
                self._auto_show_results()

        def _error(msg):
            self.point_status_label.setText("Segmentation failed!")
            self.point_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {msg}")

        self._run_with_progress(_work, _done, _error)

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

        box_coords = self.box_coords  # capture for closure

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.box_status_label.setText("Processing...")
        self.box_status_label.setStyleSheet("color: orange;")

        def _work():
            self._restore_image_if_needed()
            # Use appropriate method based on model type
            if self._sam_supports("generate_masks_by_boxes"):
                self.sam.generate_masks_by_boxes(boxes=[box_coords])
            else:
                # Fallback for older SamGeo versions
                import numpy as np

                self.sam.predict(box=np.array(box_coords))

        def _done(_):
            self._on_new_masks(format_prompt_label("box"))
            num_masks = len(self.sam.masks) if self.sam.masks else 0
            self.results_text.setText(
                f"Box Segmentation Results:\n"
                f"Box: {box_coords}\n"
                f"Objects found: {num_masks}\n"
            )
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
            if num_masks > 0:
                self._auto_show_results()

        def _error(msg):
            self.box_status_label.setText("Segmentation failed!")
            self.box_status_label.setStyleSheet("color: red;")
            self.show_error(f"Segmentation failed: {msg}")

        self._run_with_progress(_work, _done, _error)

    def segment_by_points_batch(self):
        """Segment using batch point prompts from interactive points or vector file/layer."""
        if self.sam is None:
            self.show_error("Please load the model first.")
            return

        if self.current_image_path is None:
            self.show_error("Please set an image first.")
            return

        # Check if batch method is available
        if not self._sam_supports("generate_masks_by_points_patch"):
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

        # Get size filters
        min_size = self.batch_min_size_spin.value()
        max_size = (
            self.batch_max_size_spin.value()
            if self.batch_max_size_spin.value() > 0
            else None
        )
        unique = self.batch_unique_check.isChecked()
        add_to_map = self.add_to_map_check.isChecked()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.batch_status_label.setText("Processing batch segmentation...")
        self.batch_status_label.setStyleSheet("color: orange;")

        def _work():
            self._restore_image_if_needed()
            self.sam.generate_masks_by_points_patch(
                point_coords_batch=point_source,
                point_crs=point_crs,
                output=output_path,
                unique=unique,
                min_size=min_size,
                max_size=max_size,
            )

        def _done(_):
            self._on_new_masks(format_prompt_label("batch points"))
            num_masks = len(self.sam.masks) if self.sam.masks else 0
            result_text = (
                f"Batch Point Segmentation Results:\n"
                f"Source: {source_description}\n"
                f"Objects found: {num_masks}\n"
            )
            if output_path:
                result_text += f"Output saved to: {output_path}\n"
            self.results_text.setText(result_text)
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
            if output_path and add_to_map:
                layer = QgsRasterLayer(output_path, os.path.basename(output_path))
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    self.results_text.append("Added result layer to map.")
            if num_masks > 0:
                self._auto_show_results()

        def _error(msg):
            self.batch_status_label.setText("Failed!")
            self.batch_status_label.setStyleSheet("color: red;")
            self.show_error(f"Batch segmentation failed: {msg}")

        self._run_with_progress(_work, _done, _error)

    def _auto_show_results(self):
        """Automatically save and show results based on Output tab settings.

        This method is called after successful segmentation to automatically
        display results without requiring the user to manually click on the
        Output tab and Save Masks button.
        """
        if not self.auto_show_check.isChecked():
            return
        if not self._has_results():
            return
        self._save_results(auto=True)

    def save_masks(self):
        """Save the segmentation masks."""
        if not self._has_results():
            self.show_error("No masks to save. Please run segmentation first.")
            return
        self._save_results(auto=False)

    def _resolve_output_path(self, ext, source_path=None):
        """Decide where to write the masks.

        Uses the Output tab path when set. Otherwise picks a non-overwriting
        ``<image>_sam_masks<ext>`` next to the source image, and falls back to
        a temporary file when the image is not a local file or its folder is
        not writable.

        Args:
            ext: Output extension including the leading dot.
            source_path: Image the masks came from; defaults to the current
                image.

        Returns:
            tuple: ``(output_path, layer_name)``.
        """
        output_path = self.output_path_edit.text().strip()
        if output_path:
            return output_path, os.path.basename(output_path)
        output_path = default_output_path(source_path or self._source_image_path(), ext)
        if output_path:
            return output_path, os.path.splitext(os.path.basename(output_path))[0]
        return temp_output_path(ext), "samgeo_masks"

    def _vectorize_mask_raster(self, mask_raster, output_path, settings):
        """Convert a mask raster to polygons in the managed venv subprocess.

        geoai.utils (which requires numpy>=2) must not be imported into the
        QGIS process, which may already have an older numpy loaded (issue
        #688). The mask's raster value ends up in ``class`` (Simple and
        Smooth modes) or ``value`` (Regularize mode).

        Args:
            mask_raster: Path to the mask GeoTIFF.
            output_path: Path of the vector file to write.
            settings: Dict with ``vector_mode``, ``min_area``,
                ``smooth_iterations`` and ``epsilon``.
        """
        from ..core.geoai_task_subprocess import run_geoai_task

        lower = output_path.lower()
        if lower.endswith(".gpkg"):
            vec_format = "gpkg"
        elif lower.endswith(".shp"):
            vec_format = "shapefile"
        else:
            vec_format = "geojson"

        min_area = settings["min_area"]
        vector_mode = settings["vector_mode"]
        if vector_mode == 0:
            # Simple mode - just convert raster to vector
            run_geoai_task(
                "raster_to_vector",
                {
                    "mask_path": mask_raster,
                    "output_path": output_path,
                    "min_area": min_area if min_area > 0 else 0,
                    "output_format": vec_format,
                },
            )
        elif vector_mode == 2:
            # Use smooth_vector for natural features
            run_geoai_task(
                "smooth_vector",
                {
                    "mask_path": mask_raster,
                    "output_path": output_path,
                    "smooth_iterations": settings["smooth_iterations"],
                    "min_area": min_area if min_area > 0 else 0,
                },
            )
        else:
            # Use orthogonalize for regularization (buildings)
            run_geoai_task(
                "vectorize_mask",
                {
                    "mask_path": mask_raster,
                    "output_path": output_path,
                    "epsilon": settings["epsilon"],
                    "min_area": min_area if min_area > 0 else None,
                },
            )

    def _save_results(self, auto=False):
        """Save the current masks as a raster or vector file and load it.

        Shared by the Save Masks button and auto-show. Vector outputs get
        ``score`` (SAM's predicted confidence for the polygon's mask, only when
        "Unique values" is on and the model reports scores), ``source`` (image
        file name) and ``prompt`` fields; see
        :func:`~geoai.core.samgeo_outputs.add_mask_attributes`.

        Only one save runs at a time: the Save button is disabled meanwhile
        and further requests are rejected, so two saves can never pick the
        same default file name.

        Args:
            auto: True when called by auto-show (changes the messages).
        """
        if self._save_busy:
            if auto:
                self.log_message(
                    "A save is already running; skipping auto-show.",
                    level=Qgis.Warning,
                )
            else:
                QMessageBox.information(
                    self,
                    "Segment Anything",
                    "A save is already running. Please wait for it to finish.",
                )
            return

        tiled = self._tiled_result
        source_path = (tiled or {}).get("source") or self._source_image_path()
        format_text = self.output_format_combo.currentText()
        ext = output_extension(format_text)
        output_path, layer_name = self._resolve_output_path(ext, source_path)

        # Capture UI values needed in the worker
        unique = tiled["unique"] if tiled else self.unique_check.isChecked()
        add_to_map = self.add_to_map_check.isChecked()
        is_raster = ext == ".tif"
        vector_mode = self.vector_mode_combo.currentIndex()
        settings = {
            "vector_mode": vector_mode,
            "min_area": self.min_area_spin.value(),
            "smooth_iterations": (
                self.smooth_iterations_spin.value() if vector_mode == 2 else 0
            ),
            "epsilon": self.epsilon_spin.value() if vector_mode == 1 else 0.0,
        }
        source_name = os.path.basename(source_path) if source_path else None
        prompt_label = self._last_prompt

        def _work():
            scores = None
            if is_raster:
                if tiled:
                    shutil.copyfile(tiled["raster"], output_path)
                else:
                    self.sam.save_masks(output=output_path, unique=unique)
                return ("raster", output_path, layer_name, None)

            if tiled:
                self._vectorize_mask_raster(tiled["raster"], output_path, settings)
                return ("vector", output_path, layer_name, None)

            temp_raster = temp_output_path(".tif")
            try:
                self.sam.save_masks(output=temp_raster, unique=unique)
                if unique:
                    scores = getattr(self.sam, "mask_scores", None)
                self._vectorize_mask_raster(temp_raster, output_path, settings)
            finally:
                if os.path.exists(temp_raster):
                    os.remove(temp_raster)
            return ("vector", output_path, layer_name, scores)

        def _done(result):
            self._end_save()
            kind, out_path, lname, mask_scores = result
            if kind == "vector":
                if tiled:
                    self.log_message(
                        "Confidence scores are not available for tiled runs; "
                        "skipping the score field."
                    )
                elif not unique:
                    self.log_message("Unique values are off; skipping the score field.")
                elif not mask_scores:
                    self.log_message(
                        "This SamGeo version does not report mask scores; "
                        "skipping the score field."
                    )
                try:
                    add_mask_attributes(
                        out_path,
                        mask_scores=mask_scores or None,
                        source=source_name,
                        prompt=prompt_label,
                        log=self.log_message,
                    )
                except Exception as exc:  # never block loading the saved file
                    self.log_message(
                        f"Could not add attributes to {out_path}: {exc}",
                        level=Qgis.Warning,
                    )
            if add_to_map:
                if kind == "raster":
                    layer = QgsRasterLayer(out_path, lname)
                else:
                    layer = QgsVectorLayer(out_path, lname, "ogr")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
            if auto:
                self.results_text.append(f"\nAuto-saved to: {out_path}")
                self.log_message(f"Auto-saved masks to: {out_path}")
            else:
                self.results_text.append(f"\nSaved to: {out_path}")
                self.log_message(f"Masks saved to: {out_path}")

        def _error(msg):
            self._end_save()
            if auto:
                self.log_message(f"Auto-show failed: {msg}", level=Qgis.Warning)
                self.show_error(f"Auto-show failed: {msg}")
            else:
                self.show_error(f"Failed to save masks: {msg}")

        self._save_busy = True
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self._run_with_progress(_work, _done, _error)

    def _end_save(self):
        """Re-enable saving and delete temp rasters released during the save."""
        self._save_busy = False
        self.save_btn.setEnabled(True)
        pending, self._pending_deletes = self._pending_deletes, []
        for path in pending:
            self._discard_temp_raster(path)

    def browse_export_dir(self):
        """Open directory dialog for export output."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
        )
        if dir_path:
            self.export_dir_edit.setText(dir_path)

    def export_training_data(self):
        """Export image tiles with segmentation masks for training."""
        tiled = self._tiled_result
        tiled_image = tiled.get("image") if tiled is not None else None
        if self.current_image_path is None and not tiled_image:
            self.show_error("Please set an image first.")
            return

        if not self._has_results():
            self.show_error(
                "No segmentation masks to export. Please run segmentation first."
            )
            return

        output_dir = self.export_dir_edit.text().strip()
        if not output_dir:
            self.show_error("Please specify an export output directory.")
            return

        temp_mask = None
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)

            # Stage the mask and capture its image before any processEvents():
            # a set-image completion handled there can change
            # current_image_path or delete a superseded tiled raster. A tiled
            # mask is paired with the image it was segmented from.
            temp_mask = tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False, prefix="samgeo_export_"
            ).name
            image_path = self.current_image_path
            if tiled is not None:
                image_path = tiled_image or image_path
                shutil.copyfile(tiled["raster"], temp_mask)
            else:
                unique = self.unique_check.isChecked()
                self.sam.save_masks(output=temp_mask, unique=unique)
            QCoreApplication.processEvents()

            # Get export parameters
            export_format = self.export_format_combo.currentText()
            tile_size = self.export_tile_size_spin.value()
            stride = self.export_stride_spin.value()

            # Runs in the venv subprocess when it exists, keeping the geoai
            # library (and PyTorch) out of the QGIS process. It blocks until
            # done without pumping events, so a second export cannot start
            # while this one is still writing the same tiles.
            from ..core.geoai_task_subprocess import export_geotiff_tiles

            self.results_text.append(f"\nExporting tiles in {export_format} format...")
            QCoreApplication.processEvents()

            export_geotiff_tiles(
                {
                    "in_raster": image_path,
                    "out_folder": output_dir,
                    "in_class_data": temp_mask,
                    "tile_size": tile_size,
                    "stride": stride,
                    "metadata_format": export_format,
                    "skip_empty_tiles": True,
                },
            )

            self.results_text.append(f"Training data exported to: {output_dir}")
            self.results_text.append(
                f"Format: {export_format}, Tile size: {tile_size}x{tile_size}"
            )
            self.log_message(f"Training data exported to: {output_dir}")

            QMessageBox.information(
                self,
                "Export Complete",
                f"Training data exported to:\n{output_dir}\n\n"
                f"Format: {export_format}\n"
                f"Tile size: {tile_size}x{tile_size}\n"
                f"Stride: {stride}",
            )

        except Exception as e:
            self.show_error(f"Failed to export training data: {str(e)}")
        finally:
            # Remove the staged mask on success and failure alike, so a
            # partial copy (e.g. a full disk) does not linger.
            if temp_mask and os.path.exists(temp_mask):
                try:
                    os.remove(temp_mask)
                except OSError:
                    pass
            self.progress_bar.setVisible(False)

    def show_error(self, message):
        """Show an error message."""
        QMessageBox.critical(self, "Segment Anything Error", message)
        self.log_message(message, level=Qgis.Critical)

    def log_message(self, message, level=Qgis.Info):
        """Log a message to QGIS."""
        QgsMessageLog.logMessage(message, "GeoAI - Segment Anything", level)

    def cleanup(self):
        """Clean up resources when the dock is closed."""
        # Clear map tools markers
        if self.point_tool is not None:
            self.point_tool.clear_markers()

        if self.batch_point_tool is not None:
            self.batch_point_tool.clear_markers()

        if self.box_tool is not None:
            self.box_tool.clear_rubber_band()

        # Stop model loading worker if running (detach + lock-free abort + wait)
        load_worker = getattr(self, "model_load_worker", None)
        self.model_load_worker = None
        if load_worker is not None:
            if hasattr(load_worker, "abort"):
                try:
                    load_worker.abort()
                except Exception:
                    pass
            for sig_name in ("finished", "error", "progress"):
                sig = getattr(load_worker, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except (TypeError, RuntimeError):
                        pass
            try:
                if load_worker.isRunning() and not load_worker.wait(5000):
                    _keep_thread_alive(load_worker)
            except RuntimeError:
                pass

        retained_op_worker = self._stop_worker_threads()

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

        # Clean up model (defer off GUI thread if a retained operation worker
        # may still hold the client's request lock).
        sam = self.sam
        self.sam = None
        if (
            sam is not None
            and hasattr(sam, "close")
            and callable(getattr(sam, "close"))
        ):
            if retained_op_worker is not None:
                _retire_client_in_background(sam, wait_worker=retained_op_worker)
            else:
                try:
                    sam.close()
                except Exception:
                    pass

    def _stop_worker_threads(self):
        """Make sure no running dock-owned QThread is destroyed with the dock.

        A running operation backed by the SamGeo subprocess is unblocked by
        killing that process. Each worker then gets a short wait; one that is
        still running is detached and kept referenced until it finishes.

        Returns:
            The retained operation worker if its wait timed out, else None.
        """
        op_worker = getattr(self, "_operation_worker", None)
        if op_worker is not None and op_worker.isRunning():
            if self.sam is not None and hasattr(self.sam, "abort"):
                try:
                    self.sam.abort()
                except Exception:
                    pass
            else:
                proc = getattr(self.sam, "process", None)
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        retained_op_worker = None
        for attr in ("_operation_worker", "_hf_worker"):
            worker = getattr(self, attr, None)
            setattr(self, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning() and not worker.wait(3000):
                    _keep_thread_alive(worker)
                    if attr == "_operation_worker":
                        retained_op_worker = worker
            except RuntimeError:
                pass  # Underlying C++ object already gone
        self._drop_queued_hf_task()
        return retained_op_worker

    def closeEvent(self, event):
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)
