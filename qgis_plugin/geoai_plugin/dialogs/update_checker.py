"""
Update Checker Dialog for GeoAI Plugin

This dialog provides functionality to check for updates from GitHub
and download/install the latest version of the plugin.
"""

import os
import re
import shutil
import tempfile
import zipfile
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError, HTTPError

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QTextEdit,
)
from qgis.PyQt.QtGui import QFont


# GitHub URLs for the plugin
GITHUB_REPO = "opengeos/geoai"
GITHUB_BRANCH = "main"
PLUGIN_PATH = "qgis_plugin/geoai_plugin"
METADATA_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{PLUGIN_PATH}/metadata.txt"
ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"


class VersionCheckWorker(QThread):
    """Worker thread for checking the latest version from GitHub."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def run(self):
        """Fetch the latest metadata from GitHub."""
        try:
            with urlopen(METADATA_URL, timeout=15) as response:
                content = response.read().decode("utf-8")

            # Parse version from metadata
            version_match = re.search(r"^version=(.+)$", content, re.MULTILINE)
            if version_match:
                latest_version = version_match.group(1).strip()
            else:
                self.error.emit("Could not parse version from remote metadata")
                return

            # Parse changelog
            changelog_match = re.search(
                r"^changelog=(.+?)(?=^[a-zA-Z0-9_]+=|\Z)",
                content,
                re.MULTILINE | re.DOTALL,
            )
            changelog = ""
            if changelog_match:
                changelog = changelog_match.group(1).strip()

            self.finished.emit(
                {"version": latest_version, "changelog": changelog, "metadata": content}
            )

        except HTTPError as e:
            self.error.emit(f"HTTP Error: {e.code} - {e.reason}")
        except URLError as e:
            self.error.emit(f"URL Error: {e.reason}")
        except Exception as e:
            self.error.emit(f"Error checking for updates: {str(e)}")


class DownloadWorker(QThread):
    """Worker thread for downloading and installing the plugin update."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, plugin_dir):
        super().__init__()
        self.plugin_dir = plugin_dir

    def run(self):
        """Download and install the latest plugin version."""
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="geoai_update_")
            zip_path = os.path.join(temp_dir, "geoai.zip")

            # Download the zip file
            self.progress.emit(10, "Downloading plugin from GitHub...")

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(int((downloaded / total_size) * 50), 50)
                    self.progress.emit(10 + percent, "Downloading...")

            urlretrieve(ZIP_URL, zip_path, reporthook)

            self.progress.emit(60, "Extracting files...")

            # Extract the zip file
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Secure extraction to prevent zip slip
                for member in zip_ref.namelist():
                    member_path = os.path.realpath(os.path.join(extract_dir, member))
                    extract_dir_real = os.path.realpath(extract_dir)
                    if not member_path.startswith(extract_dir_real + os.sep):
                        raise Exception(
                            f"Attempted Path Traversal in Zip File: {member}"
                        )
                zip_ref.extractall(extract_dir)

            self.progress.emit(70, "Locating plugin files...")

            # Find the plugin directory in the extracted files
            # The structure will be: geoai-main/qgis_plugin/geoai_plugin/
            extracted_plugin_dir = None
            for root, dirs, files in os.walk(extract_dir):
                if "metadata.txt" in files and "geoai_plugin.py" in files:
                    extracted_plugin_dir = root
                    break

            if not extracted_plugin_dir:
                self.error.emit("Could not find plugin files in downloaded archive")
                return

            self.progress.emit(80, "Installing update...")

            # Get the parent directory of the current plugin (QGIS plugins folder)
            plugins_folder = os.path.dirname(self.plugin_dir)
            current_plugin_name = os.path.basename(self.plugin_dir)
            target_dir = os.path.join(plugins_folder, current_plugin_name)

            # Backup current plugin (optional - we'll just replace)
            backup_dir = os.path.join(temp_dir, "backup")

            # If target exists, move it to backup
            if os.path.exists(target_dir):
                shutil.move(target_dir, backup_dir)

            try:
                # Copy new plugin files
                shutil.copytree(extracted_plugin_dir, target_dir)
                self.progress.emit(100, "Update complete!")
                self.finished.emit(target_dir)

            except Exception as e:
                # Restore backup if copy fails
                def is_valid_plugin_dir(path):
                    return os.path.exists(
                        os.path.join(path, "metadata.txt")
                    ) and os.path.exists(os.path.join(path, "geoai_plugin.py"))

                if os.path.exists(backup_dir) and is_valid_plugin_dir(backup_dir):
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    shutil.move(backup_dir, target_dir)
                else:
                    # Backup is missing or invalid, cannot restore
                    self.error.emit(
                        "Backup missing or invalid, cannot restore previous plugin version."
                    )
                raise e

        except HTTPError as e:
            self.error.emit(f"HTTP Error downloading: {e.code} - {e.reason}")
        except URLError as e:
            self.error.emit(f"URL Error downloading: {e.reason}")
        except Exception as e:
            self.error.emit(f"Error installing update: {str(e)}")
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass


class UpdateCheckerDialog(QDialog):
    """Dialog for checking and installing plugin updates."""

    def __init__(self, plugin_dir, parent=None):
        super().__init__(parent)
        self.plugin_dir = plugin_dir
        self.current_version = self._get_current_version()
        self.latest_version = None
        self.changelog = ""
        self.check_worker = None
        self.download_worker = None

        self.setWindowTitle("GeoAI Plugin Update")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._setup_ui()

    def _get_current_version(self):
        """Read the current version from local metadata.txt."""
        metadata_path = os.path.join(self.plugin_dir, "metadata.txt")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                content = f.read()
            version_match = re.search(r"^version=(.+)$", content, re.MULTILINE)
            if version_match:
                return version_match.group(1).strip()
        except (FileNotFoundError, OSError, IOError):
            pass
        return "Unknown"

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header_label = QLabel("GeoAI Plugin Update Checker")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Version info group
        version_group = QGroupBox("Version Information")
        version_layout = QFormLayout(version_group)

        self.current_version_label = QLabel(self.current_version)
        self.current_version_label.setStyleSheet("font-weight: bold;")
        version_layout.addRow("Current Version:", self.current_version_label)

        self.latest_version_label = QLabel("Not checked")
        self.latest_version_label.setStyleSheet("color: gray;")
        version_layout.addRow("Latest Version:", self.latest_version_label)

        self.status_label = QLabel(
            "Click 'Check for Updates' to check for new versions"
        )
        self.status_label.setWordWrap(True)
        version_layout.addRow("Status:", self.status_label)

        layout.addWidget(version_group)

        # Changelog group
        changelog_group = QGroupBox("Changelog")
        changelog_layout = QVBoxLayout(changelog_group)

        self.changelog_text = QTextEdit()
        self.changelog_text.setReadOnly(True)
        self.changelog_text.setMaximumHeight(150)
        self.changelog_text.setPlaceholderText(
            "Changelog will appear here after checking for updates..."
        )
        changelog_layout.addWidget(self.changelog_text)

        layout.addWidget(changelog_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.check_btn = QPushButton("Check for Updates")
        self.check_btn.clicked.connect(self.check_for_updates)
        button_layout.addWidget(self.check_btn)

        self.install_btn = QPushButton("Download and Install Update")
        self.install_btn.setEnabled(False)
        self.install_btn.clicked.connect(self.download_and_install)
        button_layout.addWidget(self.install_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Info label
        info_label = QLabel(
            "<small>Updates are downloaded from "
            f'<a href="https://github.com/{GITHUB_REPO}">GitHub</a>. '
            "QGIS must be restarted after installing an update.</small>"
        )
        info_label.setWordWrap(True)
        info_label.setOpenExternalLinks(True)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

    def check_for_updates(self):
        """Check for updates from GitHub."""
        self.check_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        self.status_label.setText("Checking for updates...")
        self.status_label.setStyleSheet("color: blue;")
        self.latest_version_label.setText("Checking...")
        self.latest_version_label.setStyleSheet("color: blue;")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        self.check_worker = VersionCheckWorker()
        self.check_worker.finished.connect(self._on_check_finished)
        self.check_worker.error.connect(self._on_check_error)
        self.check_worker.start()

    def _on_check_finished(self, result):
        """Handle successful version check."""
        self.progress_bar.setVisible(False)
        self.check_btn.setEnabled(True)

        self.latest_version = result["version"]
        self.changelog = result.get("changelog", "")

        self.latest_version_label.setText(self.latest_version)

        # Show changelog
        if self.changelog:
            self.changelog_text.setPlainText(self.changelog)

        # Compare versions
        if self._is_newer_version(self.latest_version, self.current_version):
            self.latest_version_label.setStyleSheet("color: green; font-weight: bold;")
            self.status_label.setText(
                f"A new version ({self.latest_version}) is available! "
                "Click 'Download and Install Update' to update."
            )
            self.status_label.setStyleSheet("color: green;")
            self.install_btn.setEnabled(True)
        elif self.latest_version == self.current_version:
            self.latest_version_label.setStyleSheet("color: green;")
            self.status_label.setText("You are running the latest version!")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.latest_version_label.setStyleSheet("color: orange;")
            self.status_label.setText(
                "Your version appears to be newer than the released version. "
                "You may be running a development build."
            )
            self.status_label.setStyleSheet("color: orange;")

    def _on_check_error(self, error_msg):
        """Handle version check error."""
        self.progress_bar.setVisible(False)
        self.check_btn.setEnabled(True)

        self.latest_version_label.setText("Error")
        self.latest_version_label.setStyleSheet("color: red;")
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: red;")

    def _is_newer_version(self, latest, current):
        """Compare version strings to determine if latest is newer than current."""
        try:
            # Parse versions like "0.2.0" into tuples of integers
            def parse_version(v):
                # Remove any non-numeric prefix/suffix and split by dots
                parts = re.findall(r"\d+", v)
                return tuple(int(p) for p in parts)

            latest_parts = parse_version(latest)
            current_parts = parse_version(current)

            return latest_parts > current_parts
        except Exception:
            # Fall back to string comparison
            return latest > current

    def download_and_install(self):
        """Download and install the latest version."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Confirm Update",
            f"This will download and install version {self.latest_version}.\n\n"
            "IMPORTANT: You will need to restart QGIS after the update completes.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        self.check_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        self.close_btn.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting download...")

        self.status_label.setText("Downloading and installing update...")
        self.status_label.setStyleSheet("color: blue;")

        self.download_worker = DownloadWorker(self.plugin_dir)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.error.connect(self._on_download_error)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.start()

    def _on_download_progress(self, percent, message):
        """Update progress bar during download."""
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _on_download_finished(self, install_path):
        """Handle successful download and installation."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.close_btn.setEnabled(True)

        self.status_label.setText(
            f"Update installed successfully to:\n{install_path}\n\n"
            "Please restart QGIS to apply the update."
        )
        self.status_label.setStyleSheet("color: green;")

        self.current_version_label.setText(
            f"{self.current_version} → {self.latest_version}"
        )

        QMessageBox.information(
            self,
            "Update Complete",
            f"GeoAI Plugin has been updated to version {self.latest_version}.\n\n"
            "Please restart QGIS to apply the changes.",
        )

    def _on_download_error(self, error_msg):
        """Handle download/installation error."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.check_btn.setEnabled(True)
        self.install_btn.setEnabled(True)
        self.close_btn.setEnabled(True)

        self.status_label.setText(f"Update failed: {error_msg}")
        self.status_label.setStyleSheet("color: red;")

        QMessageBox.critical(
            self,
            "Update Failed",
            f"Failed to update the plugin:\n\n{error_msg}\n\n"
            "You can try again or manually download from GitHub.",
        )

    def closeEvent(self, event):
        """Handle dialog close event."""
        # Stop any running workers
        if self.check_worker and self.check_worker.isRunning():
            self.check_worker.terminate()
            self.check_worker.wait()

        if self.download_worker and self.download_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Download in Progress",
                "A download is in progress. Are you sure you want to cancel?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.download_worker.terminate()
            self.download_worker.wait()

        event.accept()
