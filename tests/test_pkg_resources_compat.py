"""Tests for QGIS plugin pkg_resources compatibility shim."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


def _load_compat_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "qgis_plugin" / "geoai_plugin" / "_pkg_resources_compat.py"
    spec = importlib.util.spec_from_file_location("_pkg_resources_compat_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ensure_pkg_resources_installs_shim_when_missing(monkeypatch):
    compat = _load_compat_module()

    # Simulate missing pkg_resources even if available in the test environment.
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)

    real_import_module = compat.importlib.import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "pkg_resources":
            raise ModuleNotFoundError("No module named 'pkg_resources'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(compat.importlib, "import_module", _fake_import_module)

    installed = compat.ensure_pkg_resources()
    assert installed is True
    assert "pkg_resources" in sys.modules

    shim = sys.modules["pkg_resources"]
    assert hasattr(shim, "get_distribution")
    assert hasattr(shim, "parse_version")

    parsed = shim.parse_version("1.2.3")
    assert parsed is not None


def test_ensure_pkg_resources_noop_when_present(monkeypatch):
    compat = _load_compat_module()

    class DummyPkgResources:
        pass

    dummy = DummyPkgResources()
    monkeypatch.setitem(sys.modules, "pkg_resources", dummy)

    installed = compat.ensure_pkg_resources()
    assert installed is False
    assert sys.modules["pkg_resources"] is dummy
