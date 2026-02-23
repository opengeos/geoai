"""
Helper to import the *external* geoai Python library from within the QGIS plugin.

Why this exists:
- When the plugin folder is named "geoai" (as required by the official QGIS plugin repo),
  importing `geoai` from plugin code resolves to the plugin package itself, shadowing the
  external `geoai` library (distributed as "geoai-py").
- Some plugin dialogs need the external library (e.g., `MoondreamGeo`, segmentation helpers).

This module provides `get_geoai()` which returns the external library module even when
shadowed by the plugin package name.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Optional

import importlib
import importlib.metadata
import importlib.util
import logging
import sys

_CACHED: Optional[ModuleType] = None
_log = logging.getLogger(__name__)


def _is_plugin_module(mod: ModuleType) -> bool:
    """
    Returns True if *mod* is a QGIS plugin package rather than the external geoai library.

    Detection uses the ``classFactory`` function that every QGIS plugin exposes in its
    ``__init__.py``.  This is more robust than path-based checks because the plugin can
    live at the installed QGIS plugins directory **or** inside the source repository
    (e.g. when the repo root is on ``sys.path`` via an editable install).
    """
    return callable(getattr(mod, "classFactory", None))


def _save_geoai_modules() -> dict:
    """Remove and return all ``geoai`` and ``geoai.*`` entries from ``sys.modules``."""
    saved = {}
    for key in list(sys.modules.keys()):
        if key == "geoai" or key.startswith("geoai."):
            saved[key] = sys.modules.pop(key)
    return saved


def _restore_geoai_modules(saved: dict) -> None:
    """Remove current ``geoai`` entries from ``sys.modules`` and restore *saved* ones."""
    for key in list(sys.modules.keys()):
        if key == "geoai" or key.startswith("geoai."):
            del sys.modules[key]
    sys.modules.update(saved)


def _find_geoai_init_from_dist(dist_name: str) -> Optional[Path]:
    """
    Find the ``__init__.py`` of the external ``geoai`` package from an installed distribution.

    Handles both regular and editable installs:
    - Regular installs list all files in ``dist.files`` (RECORD).
    - Editable installs only list the dist-info files, but register a meta-path finder
      whose ``MAPPING`` dict contains the package directory path.
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    # Strategy A: look in dist.files (works for regular installs)
    for f in dist.files or []:
        if str(f).replace("\\", "/").endswith("geoai/__init__.py"):
            init_path = Path(dist.locate_file(f)).resolve()
            if init_path.exists():
                return init_path

    # Strategy B: look in editable-install finder modules.
    # Setuptools editable installs register a meta-path finder class whose *module*
    # has a top-level MAPPING dict (e.g. {'geoai': '/path/to/geoai'}).
    for finder in sys.meta_path:
        finder_mod = sys.modules.get(getattr(finder, "__module__", ""), None)
        if finder_mod is None:
            continue
        mapping = getattr(finder_mod, "MAPPING", None)
        if not isinstance(mapping, dict):
            continue
        pkg_dir = mapping.get("geoai")
        if pkg_dir is None:
            continue
        init_path = Path(pkg_dir).resolve() / "__init__.py"
        if init_path.exists():
            _log.debug("Found geoai via editable finder MAPPING: %s", init_path)
            return init_path

    return None


def _load_geoai_from_path(init_path: Path) -> Optional[ModuleType]:
    """
    Load the external geoai package from an explicit ``__init__.py`` path.

    The module is loaded under its real name ``geoai`` so that relative imports
    inside the library (e.g. ``from .water import segment_water``) resolve
    correctly.  We temporarily take over ``sys.modules["geoai"]`` during loading
    and restore the plugin's entries afterward.
    """
    pkg_dir = init_path.parent

    saved = _save_geoai_modules()
    try:
        spec = importlib.util.spec_from_file_location(
            "geoai",
            str(init_path),
            submodule_search_locations=[str(pkg_dir)],
        )
        if spec is None or spec.loader is None:
            _log.debug("spec_from_file_location returned None for %s", init_path)
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules["geoai"] = module
        spec.loader.exec_module(module)

        if _is_plugin_module(module):
            _log.debug(
                "Loaded module from %s is the plugin, not the library", init_path
            )
            return None

        _log.debug("Loaded external geoai from %s", init_path)
        return module
    except Exception as exc:
        _log.debug("Failed to load geoai from %s: %s", init_path, exc)
        return None
    finally:
        _restore_geoai_modules(saved)


def get_geoai() -> ModuleType:
    """
    Return the external ``geoai`` library module.

    Tries two strategies in order:

    1. Normal ``import geoai`` — works when the plugin is NOT named ``geoai``.
    2. Locate the installed distribution (``geoai-py`` or ``geoai``) via
       ``importlib.metadata`` (including editable-install finders) and load the
       library from its ``__init__.py`` file path.

    The result is cached after the first successful resolution.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    # Strategy 1: normal import (works when plugin package is NOT named `geoai`)
    try:
        imported = importlib.import_module("geoai")
        if not _is_plugin_module(imported):
            _CACHED = imported
            return imported
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # Strategy 2: find the external library via distribution metadata / editable
    # finders and load it by explicit file path.
    for dist_name in ("geoai-py", "geoai"):
        init_path = _find_geoai_init_from_dist(dist_name)
        if init_path is None:
            continue
        ext = _load_geoai_from_path(init_path)
        if ext is not None:
            _CACHED = ext
            return ext

    py = getattr(sys, "executable", "python")
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    raise ImportError(
        "GeoAI plugin could not import the external 'geoai' library because the plugin "
        "package name shadows it.\n\n"
        f"QGIS Python:\n  executable: {py}\n  version: {ver}\n\n"
        "Fix: install the GeoAI Python package into *this same Python environment*.\n"
        "In QGIS, open Python Console and run:\n"
        "  import sys, subprocess\n"
        "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'geoai-py'])\n"
        "If you encounter issues, please consult the QGIS documentation for your "
        "platform regarding installing Python packages for use in QGIS.\n"
    )
