"""
GeoAI Plugin for QGIS - Main Plugin Class

This module contains the main plugin class that manages the QGIS interface
integration, menu items, and toolbar buttons.
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QToolBar, QMessageBox


class GeoAIPlugin:
    """GeoAI Plugin implementation class for QGIS."""

    def __init__(self, iface):
        """Constructor.

        Args:
            iface: An interface instance that provides the hook to QGIS.
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = None
        self.toolbar = None

        # Dock widgets (lazy loaded)
        self._moondream_dock = None
        self._segmentation_dock = None
        self._samgeo_dock = None

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        checkable=False,
        parent=None,
    ):
        """Add a toolbar icon to the toolbar.

        Args:
            icon_path: Path to the icon for this action.
            text: Text that appears in the menu for this action.
            callback: Function to be called when the action is triggered.
            enabled_flag: A flag indicating if the action should be enabled.
            add_to_menu: Flag indicating whether action should be added to menu.
            add_to_toolbar: Flag indicating whether action should be added to toolbar.
            status_tip: Optional text to show in status bar when mouse hovers over action.
            checkable: Whether the action is checkable (toggle).
            parent: Parent widget for the new action.

        Returns:
            The action that was created.
        """
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        action.setCheckable(checkable)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.menu.addAction(action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        # Create menu
        self.menu = QMenu("&GeoAI")
        self.iface.mainWindow().menuBar().addMenu(self.menu)

        # Create toolbar
        self.toolbar = QToolBar("GeoAI Toolbar")
        self.toolbar.setObjectName("GeoAIToolbar")
        self.iface.addToolBar(self.toolbar)

        # Get icon paths
        icon_base = os.path.join(self.plugin_dir, "icons")

        # Icon paths with SVG support
        moondream_icon = os.path.join(icon_base, "moondream.svg")
        if not os.path.exists(moondream_icon):
            moondream_icon = ":/images/themes/default/mIconRaster.svg"

        segment_icon = os.path.join(icon_base, "segment.svg")
        if not os.path.exists(segment_icon):
            segment_icon = ":/images/themes/default/mActionAddRasterLayer.svg"

        samgeo_icon = os.path.join(icon_base, "samgeo.png")
        if not os.path.exists(samgeo_icon):
            samgeo_icon = ":/images/themes/default/mIconPolygonLayer.svg"

        about_icon = os.path.join(icon_base, "about.svg")
        if not os.path.exists(about_icon):
            about_icon = ":/images/themes/default/mActionHelpContents.svg"

        # Add Moondream action (checkable for dock toggle)
        self.moondream_action = self.add_action(
            moondream_icon,
            "Moondream VLM",
            self.toggle_moondream_dock,
            status_tip="Toggle Moondream Vision-Language Model panel",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        # Add Segmentation action (checkable for dock toggle)
        self.segmentation_action = self.add_action(
            segment_icon,
            "Segmentation",
            self.toggle_segmentation_dock,
            status_tip="Toggle Segmentation Training & Inference panel",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        # Add SamGeo action (checkable for dock toggle)
        self.samgeo_action = self.add_action(
            samgeo_icon,
            "SamGeo",
            self.toggle_samgeo_dock,
            status_tip="Toggle SamGeo Segmentation panel (supports SAM1, SAM2, and SAM3 models)",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        # Add separator to toolbar
        self.toolbar.addSeparator()

        # GPU cleanup icon - use QGIS default refresh/clear icon
        gpu_icon = os.path.join(icon_base, "gpu.svg")
        if not os.path.exists(gpu_icon):
            gpu_icon = ":/images/themes/default/mActionRefresh.svg"

        # Add Clear GPU action
        self.add_action(
            gpu_icon,
            "Clear GPU Memory",
            self.clear_gpu_memory,
            status_tip="Release GPU memory and clear CUDA cache",
            parent=self.iface.mainWindow(),
        )

        # Add separator
        self.menu.addSeparator()

        # Update icon - use QGIS default download/update icon
        update_icon = ":/images/themes/default/mActionRefresh.svg"

        # Add Check for Updates action (menu only)
        self.add_action(
            update_icon,
            "Check for Updates...",
            self.show_update_checker,
            add_to_toolbar=False,
            status_tip="Check for plugin updates from GitHub",
            parent=self.iface.mainWindow(),
        )

        # Add About action (menu only)
        self.add_action(
            about_icon,
            "About GeoAI",
            self.show_about,
            add_to_toolbar=False,
            status_tip="About GeoAI Plugin",
            parent=self.iface.mainWindow(),
        )

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        # Remove dock widgets
        if self._moondream_dock:
            self.iface.removeDockWidget(self._moondream_dock)
            self._moondream_dock.deleteLater()
            self._moondream_dock = None

        if self._segmentation_dock:
            self.iface.removeDockWidget(self._segmentation_dock)
            self._segmentation_dock.deleteLater()
            self._segmentation_dock = None

        if self._samgeo_dock:
            self.iface.removeDockWidget(self._samgeo_dock)
            self._samgeo_dock.deleteLater()
            self._samgeo_dock = None

        # Remove actions from menu
        for action in self.actions:
            self.iface.removePluginMenu("&GeoAI", action)

        # Remove toolbar
        if self.toolbar:
            del self.toolbar

        # Remove menu
        if self.menu:
            self.menu.deleteLater()

    def toggle_moondream_dock(self):
        """Toggle the Moondream dock widget visibility."""
        if self._moondream_dock is None:
            try:
                from .dialogs.moondream import MoondreamDockWidget

                self._moondream_dock = MoondreamDockWidget(
                    self.iface, self.iface.mainWindow()
                )
                self._moondream_dock.setObjectName("GeoAIMoondreamDock")
                self._moondream_dock.visibilityChanged.connect(
                    self._on_moondream_visibility_changed
                )
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self._moondream_dock)
                self._moondream_dock.show()
                self._moondream_dock.raise_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create Moondream panel:\n{str(e)}",
                )
                self.moondream_action.setChecked(False)
                return

        # Toggle visibility
        if self._moondream_dock.isVisible():
            self._moondream_dock.hide()
        else:
            self._moondream_dock.show()
            self._moondream_dock.raise_()

    def _on_moondream_visibility_changed(self, visible):
        """Handle Moondream dock visibility change."""
        self.moondream_action.setChecked(visible)

    def toggle_segmentation_dock(self):
        """Toggle the Segmentation dock widget visibility."""
        if self._segmentation_dock is None:
            try:
                from .dialogs.segmentation import SegmentationDockWidget

                self._segmentation_dock = SegmentationDockWidget(
                    self.iface, self.iface.mainWindow()
                )
                self._segmentation_dock.setObjectName("GeoAISegmentationDock")
                self._segmentation_dock.visibilityChanged.connect(
                    self._on_segmentation_visibility_changed
                )
                self.iface.addDockWidget(
                    Qt.RightDockWidgetArea, self._segmentation_dock
                )
                self._segmentation_dock.show()
                self._segmentation_dock.raise_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create Segmentation panel:\n{str(e)}",
                )
                self.segmentation_action.setChecked(False)
                return

        # Toggle visibility
        if self._segmentation_dock.isVisible():
            self._segmentation_dock.hide()
        else:
            self._segmentation_dock.show()
            self._segmentation_dock.raise_()

    def _on_segmentation_visibility_changed(self, visible):
        """Handle Segmentation dock visibility change."""
        self.segmentation_action.setChecked(visible)

    def toggle_samgeo_dock(self):
        """Toggle the SamGeo dock widget visibility."""
        if self._samgeo_dock is None:
            try:
                from .dialogs.samgeo import SamGeoDockWidget

                self._samgeo_dock = SamGeoDockWidget(
                    self.iface, self.iface.mainWindow()
                )
                self._samgeo_dock.setObjectName("GeoAISamGeoDock")
                self._samgeo_dock.visibilityChanged.connect(
                    self._on_samgeo_visibility_changed
                )
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self._samgeo_dock)
                self._samgeo_dock.show()
                self._samgeo_dock.raise_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create SamGeo panel:\n{str(e)}",
                )
                self.samgeo_action.setChecked(False)
                return

        # Toggle visibility
        if self._samgeo_dock.isVisible():
            self._samgeo_dock.hide()
        else:
            self._samgeo_dock.show()
            self._samgeo_dock.raise_()

    def _on_samgeo_visibility_changed(self, visible):
        """Handle SamGeo dock visibility change."""
        self.samgeo_action.setChecked(visible)

    def clear_gpu_memory(self):
        """Clear GPU memory and release CUDA resources."""
        import gc

        cleared_items = []

        # Import torch early to use for cleanup
        torch = None
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            # PyTorch is optional; continue without GPU memory clearing if not installed.
            pass

        # Clear Moondream model if loaded
        if self._moondream_dock is not None:
            try:
                if hasattr(self._moondream_dock, "moondream"):
                    moondream_obj = self._moondream_dock.moondream
                    if moondream_obj is not None:
                        # Move model to CPU first to free GPU memory
                        if (
                            hasattr(moondream_obj, "model")
                            and moondream_obj.model is not None
                        ):
                            try:
                                moondream_obj.model.cpu()
                            except Exception:
                                pass
                            # Clear all parameters
                            try:
                                for param in moondream_obj.model.parameters():
                                    param.data = None
                                    if param.grad is not None:
                                        param.grad = None
                            except Exception:
                                pass
                            del moondream_obj.model
                            moondream_obj.model = None

                        # Clear any other attributes
                        for attr in list(vars(moondream_obj).keys()):
                            try:
                                setattr(moondream_obj, attr, None)
                            except Exception:
                                pass

                        # Delete the moondream object
                        self._moondream_dock.moondream = None
                        del moondream_obj
                        cleared_items.append("Moondream model")

                        # Update UI status
                        if hasattr(self._moondream_dock, "model_status"):
                            self._moondream_dock.model_status.setText(
                                "Model not loaded"
                            )
                            self._moondream_dock.model_status.setStyleSheet(
                                "color: gray;"
                            )
                        if hasattr(self._moondream_dock, "run_btn"):
                            self._moondream_dock.run_btn.setEnabled(False)
            except Exception:
                pass

        # Clear SamGeo model if loaded
        if self._samgeo_dock is not None:
            try:
                if hasattr(self._samgeo_dock, "sam"):
                    sam_obj = self._samgeo_dock.sam
                    if sam_obj is not None:
                        # Clear the model
                        if hasattr(sam_obj, "model") and sam_obj.model is not None:
                            try:
                                sam_obj.model.cpu()
                            except Exception:
                                pass
                            try:
                                for param in sam_obj.model.parameters():
                                    param.data = None
                                    if param.grad is not None:
                                        param.grad = None
                            except Exception:
                                pass
                            del sam_obj.model
                            sam_obj.model = None

                        # Clear any other attributes
                        for attr in list(vars(sam_obj).keys()):
                            try:
                                setattr(sam_obj, attr, None)
                            except Exception:
                                # Ignore errors when clearing attributes; some may be read-only or protected.
                                pass

                        # Delete the sam object
                        self._samgeo_dock.sam = None
                        del sam_obj
                        cleared_items.append("SamGeo model")

                        # Update UI status
                        if hasattr(self._samgeo_dock, "model_status"):
                            self._samgeo_dock.model_status.setText("Model: Not loaded")
                            self._samgeo_dock.model_status.setStyleSheet("color: gray;")
                        if hasattr(self._samgeo_dock, "image_status"):
                            self._samgeo_dock.image_status.setText("Image: Not set")
                            self._samgeo_dock.image_status.setStyleSheet("color: gray;")
                        # Also clear internal state for image and layer
                        if hasattr(self._samgeo_dock, "current_image_path"):
                            self._samgeo_dock.current_image_path = None
                        if hasattr(self._samgeo_dock, "current_layer"):
                            self._samgeo_dock.current_layer = None
            except Exception:
                pass

        # Run garbage collection multiple times
        for _ in range(5):
            gc.collect()

        # Clear PyTorch CUDA cache
        if torch is not None and torch.cuda.is_available():
            try:
                # Synchronize first
                torch.cuda.synchronize()
                # Empty cache
                torch.cuda.empty_cache()
                # IPC collect if available
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                # Run gc and empty cache again
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                cleared_items.append("CUDA cache")

                # Get memory info for display
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                memory_info = f"\n\nGPU Memory:\n  Allocated: {allocated:.1f} MB\n  Reserved: {reserved:.1f} MB"

                if allocated > 100:  # More than 100MB still allocated
                    memory_info += "\n\nNote: Some GPU memory may still be held by PyTorch's memory allocator. Restart QGIS to fully release all GPU memory."
            except Exception as e:
                memory_info = f"\n\nError clearing CUDA: {str(e)}"
        elif torch is None:
            memory_info = "\n\nPyTorch not installed."
        else:
            memory_info = "\n\nNo CUDA GPU available."

        if cleared_items:
            message = f"Cleared: {', '.join(cleared_items)}{memory_info}"
        else:
            message = f"No models loaded to clear.{memory_info}"

        self.iface.statusBarIface().showMessage("GPU memory cleared", 3000)
        QMessageBox.information(
            self.iface.mainWindow(),
            "Clear GPU Memory",
            message,
        )

    def show_about(self):
        """Display the about dialog."""
        # Read version from metadata.txt
        version = "Unknown"
        try:
            metadata_path = os.path.join(self.plugin_dir, "metadata.txt")
            with open(metadata_path, "r", encoding="utf-8") as f:
                import re

                content = f.read()
                version_match = re.search(r"^version=(.+)$", content, re.MULTILINE)
                if version_match:
                    version = version_match.group(1).strip()
        except Exception as e:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "GeoAI Plugin",
                f"Could not read version from metadata.txt:\n{str(e)}",
            )

        about_text = f"""
<h2>GeoAI Plugin for QGIS</h2>
<p>Version: {version}</p>
<p>Author: Qiusheng Wu</p>

<h3>Features:</h3>
<ul>
<li><b>Moondream Vision-Language Model:</b> AI-powered image captioning, querying, object detection, and point localization</li>
<li><b>Semantic Segmentation:</b> Train and run inference with deep learning models (U-Net, DeepLabV3+, FPN, etc.)</li>
<li><b>SamGeo:</b> Segment Anything Model (SAM, SAM2, SAM3) for geospatial data with text, point, and box prompts</li>
</ul>

<h3>Links:</h3>
<ul>
<li><a href="https://opengeoai.org">Documentation</a></li>
<li><a href="https://github.com/opengeos/geoai">GitHub Repository</a></li>
<li><a href="https://github.com/opengeos/geoai/issues">Report Issues</a></li>
</ul>

<p>Licensed under MIT License</p>
"""
        QMessageBox.about(
            self.iface.mainWindow(),
            "About GeoAI Plugin",
            about_text,
        )

    def show_update_checker(self):
        """Display the update checker dialog."""
        try:
            from .dialogs.update_checker import UpdateCheckerDialog
        except ImportError as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Error",
                f"Failed to import update checker dialog:\n{str(e)}",
            )
            return

        try:
            dialog = UpdateCheckerDialog(self.plugin_dir, self.iface.mainWindow())
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Error",
                f"Failed to open update checker:\n{str(e)}",
            )
