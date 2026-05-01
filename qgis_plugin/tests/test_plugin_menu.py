from qgis.PyQt.QtWidgets import QApplication, QMainWindow, QAction

from geoai.geoai_plugin import GeoAIPlugin


class FakeIface:
    def __init__(self):
        self._main_window = QMainWindow()

    def mainWindow(self):
        return self._main_window

    def addToolBar(self, toolbar):
        self._main_window.addToolBar(toolbar)

    def removePluginMenu(self, _menu, _action):
        pass

    def removeDockWidget(self, dock_widget):
        self._main_window.removeDockWidget(dock_widget)


def _no_menu_role():
    try:
        return QAction.MenuRole.NoRole
    except AttributeError:
        return QAction.NoRole


def test_plugin_actions_stay_out_of_application_menu():
    app = QApplication.instance() or QApplication([])
    iface = FakeIface()
    plugin = GeoAIPlugin(iface)

    try:
        plugin.initGui()

        about_actions = [
            action for action in plugin.actions if action.text() == "About GeoAI"
        ]
        assert len(about_actions) == 1
        assert about_actions[0].menuRole() == _no_menu_role()
        assert all(action.menuRole() == _no_menu_role() for action in plugin.actions)
    finally:
        plugin.unload()
        iface.mainWindow().close()
        app.processEvents()
