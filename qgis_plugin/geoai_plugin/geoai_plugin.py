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

        # Add separator
        self.menu.addSeparator()

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

    def show_about(self):
        """Display the about dialog."""
        about_text = """
<h2>GeoAI Plugin for QGIS</h2>
<p>Version: 0.1.0</p>
<p>Author: Qiusheng Wu</p>

<h3>Features:</h3>
<ul>
<li><b>Moondream Vision-Language Model:</b> AI-powered image captioning, querying, object detection, and point localization</li>
<li><b>Semantic Segmentation:</b> Train and run inference with deep learning models (U-Net, DeepLabV3+, FPN, etc.)</li>
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
