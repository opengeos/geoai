"""Shared pytest fixtures.

Stubs the ``qgis`` package so the plugin's modules can be imported without a
running QGIS instance. The stub reproduces the real ``qgis.PyQt`` shim
behavior on Qt6: it re-exports ``QAction``, ``QActionGroup`` and ``QShortcut``
from ``PyQt6.QtGui`` under ``qgis.PyQt.QtWidgets`` (they moved out of
``QtWidgets`` in Qt6).

Also forces ``qgis_plugin/`` to the front of ``sys.path`` and drops any
pre-imported top-level ``geoai`` entries so that ``import geoai`` resolves
to the plugin package (``qgis_plugin/geoai``) instead of the unrelated
top-level library package at the repo root.
"""

import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PLUGIN_PARENT = pathlib.Path(__file__).resolve().parent.parent
if str(PLUGIN_PARENT) in sys.path:
    sys.path.remove(str(PLUGIN_PARENT))
sys.path.insert(0, str(PLUGIN_PARENT))
for _mod in [m for m in sys.modules if m == "geoai" or m.startswith("geoai.")]:
    del sys.modules[_mod]

import PyQt6.QtCore
import PyQt6.QtGui
import PyQt6.QtNetwork
import PyQt6.QtWidgets


def _install_qgis_stub() -> None:
    qgis = types.ModuleType("qgis")
    qgis.__path__ = []
    sys.modules["qgis"] = qgis

    qgis_pyqt = types.ModuleType("qgis.PyQt")
    qgis_pyqt.__path__ = []
    sys.modules["qgis.PyQt"] = qgis_pyqt
    qgis.PyQt = qgis_pyqt

    pyqt_submodules = {
        "QtCore": PyQt6.QtCore,
        "QtGui": PyQt6.QtGui,
        "QtNetwork": PyQt6.QtNetwork,
        "QtWidgets": PyQt6.QtWidgets,
    }
    for name, real in pyqt_submodules.items():
        alias = types.ModuleType(f"qgis.PyQt.{name}")
        for attr in dir(real):
            if not attr.startswith("_"):
                setattr(alias, attr, getattr(real, attr))
        sys.modules[f"qgis.PyQt.{name}"] = alias
        setattr(qgis_pyqt, name, alias)

    # Qt6: QAction, QActionGroup, and QShortcut live in QtGui. The real
    # qgis.PyQt.QtWidgets shim re-exports them, so mirror that here.
    qtwidgets_alias = sys.modules["qgis.PyQt.QtWidgets"]
    for attr in ("QAction", "QActionGroup", "QShortcut"):
        setattr(qtwidgets_alias, attr, getattr(PyQt6.QtGui, attr))

    for submodule in ("QtSvg", "QtWebEngineWidgets"):
        alias = MagicMock()
        sys.modules[f"qgis.PyQt.{submodule}"] = alias
        setattr(qgis_pyqt, submodule, alias)

    for name in ("core", "gui", "utils"):
        stub = MagicMock()
        stub.__spec__ = None
        sys.modules[f"qgis.{name}"] = stub
        setattr(qgis, name, stub)


_install_qgis_stub()
