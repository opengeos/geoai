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
        self._instance_segmentation_dock = None
        self._samgeo_dock = None
        self._deepforest_dock = None
        self._water_segmentation_dock = None

        # Dependency installation state
        self._deps_available = False
        self._deps_install_worker = None
        self._pending_dock_action = None
        self._deps_dock = None

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

        deepforest_icon = os.path.join(icon_base, "deepforest.svg")
        if not os.path.exists(deepforest_icon):
            deepforest_icon = ":/images/themes/default/mIconPointLayer.svg"

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

        # Add Semantic Segmentation action (checkable for dock toggle)
        self.segmentation_action = self.add_action(
            segment_icon,
            "Semantic Segmentation",
            self.toggle_segmentation_dock,
            status_tip="Toggle Semantic Segmentation Training & Inference panel",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        # Add Instance Segmentation action (checkable for dock toggle)
        instance_seg_icon = os.path.join(icon_base, "instance_segmentation.svg")
        if not os.path.exists(instance_seg_icon):
            instance_seg_icon = ":/images/themes/default/mIconPolygonLayer.svg"

        self.instance_segmentation_action = self.add_action(
            instance_seg_icon,
            "Instance Segmentation",
            self.toggle_instance_segmentation_dock,
            status_tip="Toggle Instance Segmentation panel (Mask R-CNN training & inference)",
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

        # Add DeepForest action (checkable for dock toggle)
        self.deepforest_action = self.add_action(
            deepforest_icon,
            "DeepForest",
            self.toggle_deepforest_dock,
            status_tip="Toggle DeepForest Tree Detection panel (tree crown detection and forest analysis)",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        # Add Water Segmentation action (checkable for dock toggle)
        water_icon = os.path.join(icon_base, "water.svg")
        if not os.path.exists(water_icon):
            water_icon = ":/images/themes/default/mIconPolygonLayer.svg"

        self.water_segmentation_action = self.add_action(
            water_icon,
            "Water Segmentation",
            self.toggle_water_segmentation_dock,
            status_tip="Toggle Water Segmentation panel (OmniWaterMask water body detection)",
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
        # Stop install worker if running
        if self._deps_install_worker:
            if self._deps_install_worker.isRunning():
                try:
                    self._deps_install_worker.cancel()
                    self._deps_install_worker.terminate()
                    self._deps_install_worker.wait(5000)
                except RuntimeError:
                    pass
            self._deps_install_worker = None

        # Remove deps dock
        if self._deps_dock:
            self.iface.removeDockWidget(self._deps_dock)
            self._deps_dock.deleteLater()
            self._deps_dock = None

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

        if self._deepforest_dock:
            self.iface.removeDockWidget(self._deepforest_dock)
            self._deepforest_dock.deleteLater()
            self._deepforest_dock = None

        if self._water_segmentation_dock:
            self.iface.removeDockWidget(self._water_segmentation_dock)
            self._water_segmentation_dock.deleteLater()
            self._water_segmentation_dock = None

        if self._instance_segmentation_dock:
            self.iface.removeDockWidget(self._instance_segmentation_dock)
            self._instance_segmentation_dock.deleteLater()
            self._instance_segmentation_dock = None

        # Remove actions from menu
        for action in self.actions:
            self.iface.removePluginMenu("&GeoAI", action)

        # Remove toolbar
        if self.toolbar:
            del self.toolbar

        # Remove menu
        if self.menu:
            self.menu.deleteLater()

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def _ensure_dependencies(self, action_name):
        """Check if dependencies are installed, show installer if not.

        Args:
            action_name: Name of the dock to open after installation
                (e.g., 'moondream', 'samgeo').

        Returns:
            True if dependencies are available, False if installer was shown.
        """
        if self._deps_available:
            return True

        try:
            from .core.venv_manager import (
                ensure_venv_packages_available,
                get_venv_status,
            )

            is_ready, message = get_venv_status()
            if is_ready:
                ensure_venv_packages_available()
                self._deps_available = True
                return True
        except Exception:
            pass

        # Dependencies not available -- show the installer dock
        self._pending_dock_action = action_name
        self._show_deps_install_dock()
        return False

    def _show_deps_install_dock(self):
        """Create and show the dependency installation dock widget."""
        if self._deps_dock is not None:
            self._deps_dock.show()
            self._deps_dock.raise_()
            return

        from .dialogs.deps_install_dialog import DepsInstallDockWidget

        self._deps_dock = DepsInstallDockWidget(self.iface.mainWindow())
        self._deps_dock.install_requested.connect(self._on_install_requested)
        self._deps_dock.cancel_requested.connect(self._on_cancel_install)

        self.iface.addDockWidget(Qt.RightDockWidgetArea, self._deps_dock)
        self._deps_dock.show()
        self._deps_dock.raise_()

    def _on_install_requested(self):
        """Handle install button click from the deps dock."""
        if self._deps_install_worker and self._deps_install_worker.isRunning():
            return

        from .core.venv_manager import detect_nvidia_gpu
        from .workers.deps_install_worker import DepsInstallWorker

        has_gpu, _ = detect_nvidia_gpu()

        self._deps_install_worker = DepsInstallWorker(cuda_enabled=has_gpu)
        self._deps_install_worker.progress.connect(self._on_install_progress)
        self._deps_install_worker.finished.connect(self._on_install_finished)

        if self._deps_dock:
            self._deps_dock.show_progress_ui()

        self._deps_install_worker.start()

    def _on_install_progress(self, percent, message):
        """Handle progress updates from the install worker.

        Args:
            percent: Progress percentage (0-100).
            message: Status message.
        """
        if self._deps_dock:
            self._deps_dock.set_progress(percent, message)

    def _on_install_finished(self, success, message):
        """Handle installation completion.

        Args:
            success: Whether installation succeeded.
            message: Completion message.
        """
        if self._deps_dock:
            self._deps_dock.show_complete_ui(success, message)

        if success:
            self._deps_available = True

            # Ensure venv packages are on sys.path
            try:
                from .core.venv_manager import ensure_venv_packages_available

                ensure_venv_packages_available()
            except Exception:
                pass

            # Close the deps dock and open the originally requested panel
            if self._deps_dock:
                self.iface.removeDockWidget(self._deps_dock)
                self._deps_dock.deleteLater()
                self._deps_dock = None

            if self._pending_dock_action:
                action_map = {
                    "moondream": self.toggle_moondream_dock,
                    "segmentation": self.toggle_segmentation_dock,
                    "instance_segmentation": self.toggle_instance_segmentation_dock,
                    "samgeo": self.toggle_samgeo_dock,
                    "deepforest": self.toggle_deepforest_dock,
                    "water_segmentation": self.toggle_water_segmentation_dock,
                }
                callback = action_map.get(self._pending_dock_action)
                self._pending_dock_action = None
                if callback:
                    callback()

    def _on_cancel_install(self):
        """Handle cancel button click during installation."""
        if self._deps_install_worker and self._deps_install_worker.isRunning():
            self._deps_install_worker.cancel()
        if self._deps_dock:
            self._deps_dock.show_install_ui()

    def toggle_moondream_dock(self):
        """Toggle the Moondream dock widget visibility."""
        if not self._ensure_dependencies("moondream"):
            return

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
        """Toggle the Semantic Segmentation dock widget visibility."""
        if not self._ensure_dependencies("segmentation"):
            return

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
                    f"Failed to create Semantic Segmentation panel:\n{str(e)}",
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
        """Handle Semantic Segmentation dock visibility change."""
        self.segmentation_action.setChecked(visible)

    def toggle_samgeo_dock(self):
        """Toggle the SamGeo dock widget visibility."""
        if not self._ensure_dependencies("samgeo"):
            return

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

    def toggle_deepforest_dock(self):
        """Toggle the DeepForest dock widget visibility."""
        if not self._ensure_dependencies("deepforest"):
            return

        if self._deepforest_dock is None:
            try:
                from .dialogs.deepforest_panel import DeepForestDockWidget

                self._deepforest_dock = DeepForestDockWidget(
                    self.iface, self.iface.mainWindow()
                )
                self._deepforest_dock.setObjectName("GeoAIDeepForestDock")
                self._deepforest_dock.visibilityChanged.connect(
                    self._on_deepforest_visibility_changed
                )
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self._deepforest_dock)
                self._deepforest_dock.show()
                self._deepforest_dock.raise_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create DeepForest panel:\n{str(e)}",
                )
                self.deepforest_action.setChecked(False)
                return

        # Toggle visibility
        if self._deepforest_dock.isVisible():
            self._deepforest_dock.hide()
        else:
            self._deepforest_dock.show()
            self._deepforest_dock.raise_()

    def _on_deepforest_visibility_changed(self, visible):
        """Handle DeepForest dock visibility change."""
        self.deepforest_action.setChecked(visible)

    def toggle_water_segmentation_dock(self):
        """Toggle the Water Segmentation dock widget visibility."""
        if not self._ensure_dependencies("water_segmentation"):
            return

        if self._water_segmentation_dock is None:
            try:
                from .dialogs.water_segmentation import (
                    WaterSegmentationDockWidget,
                )

                self._water_segmentation_dock = WaterSegmentationDockWidget(
                    self.iface, self.iface.mainWindow()
                )
                self._water_segmentation_dock.setObjectName(
                    "GeoAIWaterSegmentationDock"
                )
                self._water_segmentation_dock.visibilityChanged.connect(
                    self._on_water_segmentation_visibility_changed
                )
                self.iface.addDockWidget(
                    Qt.RightDockWidgetArea, self._water_segmentation_dock
                )
                self._water_segmentation_dock.show()
                self._water_segmentation_dock.raise_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create Water Segmentation panel:\n{str(e)}",
                )
                self.water_segmentation_action.setChecked(False)
                return

        # Toggle visibility
        if self._water_segmentation_dock.isVisible():
            self._water_segmentation_dock.hide()
        else:
            self._water_segmentation_dock.show()
            self._water_segmentation_dock.raise_()

    def _on_water_segmentation_visibility_changed(self, visible):
        """Handle Water Segmentation dock visibility change."""
        self.water_segmentation_action.setChecked(visible)

    def toggle_instance_segmentation_dock(self):
        """Toggle the Instance Segmentation dock widget visibility."""
        if not self._ensure_dependencies("instance_segmentation"):
            return

        if self._instance_segmentation_dock is None:
            try:
                from .dialogs.instance_segmentation import (
                    InstanceSegmentationDockWidget,
                )

                self._instance_segmentation_dock = InstanceSegmentationDockWidget(
                    self.iface, self.iface.mainWindow()
                )
                self._instance_segmentation_dock.setObjectName(
                    "GeoAIInstanceSegmentationDock"
                )
                self._instance_segmentation_dock.visibilityChanged.connect(
                    self._on_instance_segmentation_visibility_changed
                )
                self.iface.addDockWidget(
                    Qt.RightDockWidgetArea, self._instance_segmentation_dock
                )
                self._instance_segmentation_dock.show()
                self._instance_segmentation_dock.raise_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create Instance Segmentation panel:\n{str(e)}",
                )
                self.instance_segmentation_action.setChecked(False)
                return

        # Toggle visibility
        if self._instance_segmentation_dock.isVisible():
            self._instance_segmentation_dock.hide()
        else:
            self._instance_segmentation_dock.show()
            self._instance_segmentation_dock.raise_()

    def _on_instance_segmentation_visibility_changed(self, visible):
        """Handle Instance Segmentation dock visibility change."""
        self.instance_segmentation_action.setChecked(visible)

    def clear_gpu_memory(self):
        """Clear accelerator memory and release model resources."""
        import gc

        cleared_items = []

        # Import torch early to use for cleanup
        torch = None
        try:
            import torch as _torch

            torch = _torch
        except (ImportError, OSError):
            # PyTorch is optional; continue without GPU memory clearing if not installed.
            pass

        # Clear Moondream model if loaded
        if self._moondream_dock is not None:
            try:
                if hasattr(self._moondream_dock, "moondream"):
                    moondream_obj = self._moondream_dock.moondream
                    if moondream_obj is not None:
                        if hasattr(moondream_obj, "close") and callable(
                            getattr(moondream_obj, "close")
                        ):
                            try:
                                moondream_obj.close()
                            except Exception:
                                pass
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
                        if hasattr(sam_obj, "close") and callable(
                            getattr(sam_obj, "close")
                        ):
                            try:
                                sam_obj.close()
                            except Exception:
                                pass
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

        # Clear DeepForest model if loaded
        if self._deepforest_dock is not None:
            try:
                if hasattr(self._deepforest_dock, "deepforest"):
                    deepforest_obj = self._deepforest_dock.deepforest
                    if deepforest_obj is not None:
                        if hasattr(deepforest_obj, "close") and callable(
                            getattr(deepforest_obj, "close")
                        ):
                            try:
                                deepforest_obj.close()
                            except Exception:
                                pass
                        # Clear the model
                        if (
                            hasattr(deepforest_obj, "model")
                            and deepforest_obj.model is not None
                        ):
                            try:
                                deepforest_obj.model.cpu()
                            except Exception:
                                pass  # Model may not be on GPU or already freed
                            try:
                                for param in deepforest_obj.model.parameters():
                                    param.data = None
                                    if param.grad is not None:
                                        param.grad = None
                            except Exception:
                                pass  # Some parameters may be read-only or already cleared
                            del deepforest_obj.model
                            deepforest_obj.model = None

                        # Clear any other attributes
                        for attr in list(vars(deepforest_obj).keys()):
                            try:
                                setattr(deepforest_obj, attr, None)
                            except Exception:
                                pass  # Some attributes may be read-only or protected

                        # Delete the deepforest object
                        self._deepforest_dock.deepforest = None
                        del deepforest_obj
                        cleared_items.append("DeepForest model")

                        # Update UI status
                        if hasattr(self._deepforest_dock, "model_status"):
                            self._deepforest_dock.model_status.setText(
                                "Model: Not loaded"
                            )
                            self._deepforest_dock.model_status.setStyleSheet(
                                "color: gray;"
                            )
                        if hasattr(self._deepforest_dock, "image_status"):
                            self._deepforest_dock.image_status.setText("Image: Not set")
                            self._deepforest_dock.image_status.setStyleSheet(
                                "color: gray;"
                            )
                        # Clear internal state
                        if hasattr(self._deepforest_dock, "current_image_path"):
                            self._deepforest_dock.current_image_path = None
                        if hasattr(self._deepforest_dock, "current_layer"):
                            self._deepforest_dock.current_layer = None
                        if hasattr(self._deepforest_dock, "predictions"):
                            self._deepforest_dock.predictions = None
            except Exception:
                pass  # Best-effort cleanup; continue to garbage collection

        # Run garbage collection multiple times
        for _ in range(5):
            gc.collect()

        # Clear PyTorch accelerator cache (CUDA or Apple MPS)
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
        elif (
            torch is not None
            and hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            try:
                if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()

                mps_cache_cleared = False
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    mps_cache_cleared = True

                gc.collect()

                if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()

                if mps_cache_cleared:
                    cleared_items.append("MPS cache")
                    memory_info = "\n\nApple Metal (MPS) cache cleared."
                else:
                    memory_info = (
                        "\n\nApple Metal (MPS) is available, but this PyTorch build "
                        "does not expose torch.mps.empty_cache(). Models were released "
                        "and garbage collection was run."
                    )
            except Exception as e:
                memory_info = f"\n\nError clearing MPS cache: {str(e)}"
        elif torch is None:
            memory_info = "\n\nPyTorch not installed."
        else:
            memory_info = "\n\nNo CUDA or MPS accelerator available."

        if cleared_items:
            message = f"Cleared: {', '.join(cleared_items)}{memory_info}"
        else:
            message = f"No models loaded to clear.{memory_info}"

        self.iface.statusBarIface().showMessage("Accelerator memory cleared", 3000)
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
<li><b>Instance Segmentation:</b> Train and run Mask R-CNN models for instance-level object detection and segmentation</li>
<li><b>SamGeo:</b> Segment Anything Model (SAM, SAM2, SAM3) for geospatial data with text, point, and box prompts</li>
<li><b>DeepForest:</b> Tree crown detection and forest analysis using pretrained deep learning models</li>
<li><b>Water Segmentation:</b> Water body detection from satellite/aerial imagery using OmniWaterMask</li>
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
