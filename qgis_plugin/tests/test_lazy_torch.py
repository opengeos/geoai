"""PyTorch must not be imported into QGIS when a panel opens.

With conda-forge QGIS on macOS, importing torch into the QGIS process loads a
second libomp and aborts QGIS ("OMP: Error #15"). On Windows it can hit DLL
conflicts. Models run in the plugin venv's subprocess, so the panels must not
import torch at module import time, and cleanup must not import it either.
"""

import pathlib
import subprocess
import sys
import textwrap

import pytest

from geoai.core import lazy_torch

TESTS_DIR = pathlib.Path(__file__).resolve().parent

# Fails the import of torch (and records it) in a fresh interpreter, then
# imports the panel modules with the same qgis stub as the test suite.
_PROBE = textwrap.dedent(
    """
    import importlib, sys
    sys.path.insert(0, {tests_dir!r})

    attempts = []

    class BlockTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                attempts.append(name)
                raise ImportError("torch import blocked by test")
            return None

    sys.meta_path.insert(0, BlockTorch())
    import conftest  # noqa: F401  (installs the qgis stub)

    for mod in {modules!r}:
        importlib.import_module(mod)
    print("ATTEMPTS=" + ",".join(attempts))
    """
)

PANEL_MODULES = [
    "geoai.dialogs.samgeo",
    "geoai.dialogs.deepforest_panel",
    "geoai.dialogs.water_segmentation",
    "geoai.geoai_plugin",
]


def test_opening_panel_modules_does_not_import_torch():
    """Regression test: importing the panel modules must not touch torch."""
    code = _PROBE.format(tests_dir=str(TESTS_DIR), modules=PANEL_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(TESTS_DIR),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line == "ATTEMPTS=", f"torch was imported: {last_line}"


def test_loaded_torch_never_imports(monkeypatch):
    """``loaded_torch`` only reports a torch module that is already loaded."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert lazy_torch.loaded_torch() is None
    assert "torch" not in sys.modules

    fake = object()
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert lazy_torch.loaded_torch() is fake


def test_import_torch_reports_failure(monkeypatch):
    """A failing import returns the error instead of raising."""

    class BlockTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "torch":
                raise OSError("DLL load failed")
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockTorch(), *sys.meta_path])

    torch, error = lazy_torch.import_torch()

    assert torch is None
    assert isinstance(error, OSError)


@pytest.mark.parametrize("module", ["samgeo", "deepforest_panel"])
def test_panels_have_no_module_level_torch(module):
    """The panels must not keep a module-level ``torch`` reference."""
    import importlib

    mod = importlib.import_module(f"geoai.dialogs.{module}")
    assert not hasattr(mod, "torch")
    assert not hasattr(mod, "_TORCH_IMPORT_ERROR")
