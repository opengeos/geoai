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
import sys


_CACHED: Optional[ModuleType] = None


def _is_module_from_dir(mod: ModuleType, directory: Path) -> bool:
    """
    Returns True if the given module's __file__ is located within the specified directory.

    Returns False if the module has no __file__, or if path resolution fails (e.g., due to
    being on different filesystems or invalid paths). Only ValueError and OSError from
    path resolution/comparison are caught and treated as a negative result.
    """
    try:
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            return False
        return Path(mod_file).resolve().is_relative_to(directory.resolve())
    except (ValueError, OSError):
        return False


def _import_geoai_without_plugin_shadow(plugin_pkg_dir: Path) -> Optional[ModuleType]:
    """
    Try to import `geoai` while temporarily removing the plugin path from sys.path.

    This avoids relying on `importlib.metadata` (which can be incomplete in some QGIS
    Python setups) and is the most direct way to bypass the shadowing.
    """
    plugin_parent = plugin_pkg_dir.parent
    orig_sys_path = list(sys.path)
    orig_geoai_mod = sys.modules.get("geoai")

    try:
        # Remove the directory that contains the plugin package named `geoai`
        sys.path = [p for p in sys.path if Path(p).resolve() != plugin_parent.resolve()]

        # Remove shadowed module entry so import attempts re-resolution
        if "geoai" in sys.modules:
            del sys.modules["geoai"]

        imported = importlib.import_module("geoai")

        # If we somehow still got the plugin, treat as failure.
        if _is_module_from_dir(imported, plugin_pkg_dir):
            if orig_geoai_mod is not None:
                sys.modules["geoai"] = orig_geoai_mod
            return None
        return imported
    except (ImportError, ModuleNotFoundError):
        return None
    finally:
        # Restore sys.path and the original shadow module (if any)
        sys.path = orig_sys_path
        if orig_geoai_mod is not None:
            sys.modules["geoai"] = orig_geoai_mod


def _load_external_geoai_from_dist(dist_name: str) -> Optional[ModuleType]:
    """
    Load the external geoai package from an installed distribution.

    We load it under an alias module name (geoai_external) to avoid conflicting with the
    plugin package (which may also be named `geoai`).
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    files = list(dist.files or [])
    init_rel = None
    for f in files:
        # dist.files entries are pathlib-ish objects; compare as posix
        if str(f).replace("\\", "/").endswith("geoai/__init__.py"):
            init_rel = f
            break
    if init_rel is None:
        return None

    init_path = Path(dist.locate_file(init_rel)).resolve()
    pkg_dir = init_path.parent

    alias_name = "geoai_external"
    existing = sys.modules.get(alias_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        alias_name,
        str(init_path),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = module
    spec.loader.exec_module(module)
    return module


def get_geoai() -> ModuleType:
    """
    Return the external `geoai` library module.

    If the plugin package name shadows `geoai`, we load the external library under the
    alias module name `geoai_external`.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    plugin_dir = Path(__file__).resolve().parent

    # First try a normal import. This works when the plugin package is NOT named `geoai`.
    try:
        imported = importlib.import_module("geoai")
        # If `geoai` resolves to the plugin itself, it is shadowed.
        if not _is_module_from_dir(imported, plugin_dir):
            _CACHED = imported
            return imported
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # Shadowed: try to import by temporarily removing the plugin from sys.path.
    imported = _import_geoai_without_plugin_shadow(plugin_dir)
    if imported is not None:
        _CACHED = imported
        return imported

    # Shadowed: try loading from installed distributions.
    for dist_name in ("geoai-py", "geoai"):
        ext = _load_external_geoai_from_dist(dist_name)
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
        "If you encounter issues, please consult the QGIS documentation for your platform regarding installing Python packages for use in QGIS.\n"
    )
