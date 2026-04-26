"""Import-smoke tests that verify every plugin module loads under PyQt6.

This catches short-form Qt enum regressions (for example ``Qt.AlignCenter``
instead of ``Qt.AlignmentFlag.AlignCenter``) which raise ``AttributeError`` in
PyQt6 during class-body evaluation.

The plugin package is auto-discovered: the first sibling directory of
``tests/`` that contains a ``metadata.txt`` is treated as the plugin root,
so this file does not need to be edited per-plugin.
"""

import importlib
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _find_plugin_root() -> pathlib.Path:
    """Return the plugin package directory (the one with metadata.txt)."""
    candidates = [
        p.parent for p in REPO_ROOT.glob("*/metadata.txt") if p.parent.name != "tests"
    ]
    if not candidates:
        raise RuntimeError(
            "Could not locate a QGIS plugin package (no */metadata.txt found "
            f"under {REPO_ROOT})."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple plugin packages found under {REPO_ROOT}: {candidates}. "
            "Set PLUGIN_ROOT explicitly in this file."
        )
    return candidates[0]


PLUGIN_ROOT = _find_plugin_root()


def _module_names():
    """Yield dotted module names for every .py file under the plugin package."""
    for path in sorted(PLUGIN_ROOT.rglob("*.py")):
        rel = path.relative_to(PLUGIN_ROOT.parent).with_suffix("")
        parts = rel.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts)


@pytest.mark.parametrize("module_name", list(_module_names()))
def test_module_imports_under_pyqt6(module_name):
    """Each plugin module must import cleanly when qgis.PyQt maps to PyQt6."""
    importlib.import_module(module_name)
