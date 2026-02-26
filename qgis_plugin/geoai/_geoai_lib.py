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

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Optional

_CACHED: Optional[ModuleType] = None
_log = logging.getLogger(__name__)

# Diagnostics collected during get_geoai() to help debug failures.
_diag: list[str] = []


def _install_torch_import_blocker():
    """Monkey-patch ``builtins.__import__`` to block ``import torch``.

    On Windows, torch DLLs may fail to load inside QGIS's process with
    ``OSError`` (WinError 127).  Most code only catches ``ImportError``,
    so the ``OSError`` propagates and crashes the entire import chain.

    This patches ``builtins.__import__`` (the lowest-level import function)
    so that any ``import torch`` raises a clean ``ImportError``.  This works
    even when QGIS replaces ``builtins.__import__`` with its own hook
    (``qgis.utils._import``), because QGIS's hook eventually calls
    ``_builtin_import`` which is the original — we patch before QGIS does,
    or we patch QGIS's replacement directly.

    Returns:
        A callable that, when called, restores the original import function.
    """
    import builtins

    original_import = builtins.__import__

    def _torch_blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(
                f"{name} is not available (torch DLLs incompatible with QGIS process)"
            )
        return original_import(name, *args, **kwargs)

    builtins.__import__ = _torch_blocking_import

    def _restore():
        builtins.__import__ = original_import

    return _restore


def _is_plugin_module(mod: ModuleType) -> bool:
    """Returns True if *mod* is a QGIS plugin package rather than the external library.

    Detection uses the ``classFactory`` function that every QGIS plugin exposes in its
    ``__init__.py``.
    """
    return callable(getattr(mod, "classFactory", None))


def _find_geoai_init_from_dist(dist_name: str) -> Optional[Path]:
    """Find the ``__init__.py`` of the external ``geoai`` package via distribution metadata.

    Handles both regular and editable installs.
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        _diag.append(f"  dist '{dist_name}': not found")
        return None

    # Strategy A: look in dist.files (works for regular installs)
    for f in dist.files or []:
        if str(f).replace("\\", "/").endswith("geoai/__init__.py"):
            init_path = Path(dist.locate_file(f)).resolve()
            if init_path.exists():
                _diag.append(f"  dist '{dist_name}': found via RECORD: {init_path}")
                return init_path

    # Strategy B: look in editable-install finder modules.
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
            _diag.append(
                f"  dist '{dist_name}': found via editable MAPPING: {init_path}"
            )
            return init_path

    _diag.append(f"  dist '{dist_name}': found distribution but no geoai/__init__.py")
    return None


def _find_geoai_init_from_sys_path() -> Optional[Path]:
    """Scan ``sys.path`` for a ``geoai/__init__.py`` that belongs to the external library."""
    plugin_init = Path(__file__).resolve().parent / "__init__.py"
    for entry in sys.path:
        candidate = Path(entry) / "geoai" / "__init__.py"
        try:
            if candidate.exists() and candidate.resolve() != plugin_init:
                _diag.append(f"  sys.path scan: found {candidate}")
                return candidate
        except (OSError, ValueError):
            continue
    _diag.append(
        f"  sys.path scan: no geoai/__init__.py found (plugin at {plugin_init})"
    )
    return None


def _fix_proj_for_qgis() -> None:
    """Ensure pyproj can find a PROJ database (proj.db) inside QGIS.

    QGIS's pyproj native extensions (``_crs.so``, ``_context.so``) are already
    loaded and cannot be replaced by the venv's copies (Python caches native
    extensions per-process).  We must therefore keep QGIS's pyproj and make
    sure its PROJ context has a valid database path.

    Strategy:
    1. If pyproj is already imported and working, do nothing.
    2. Otherwise, find a valid ``proj.db`` (QGIS conda env, or the venv's
       bundled copy) and call ``pyproj.datadir.set_data_dir()``.
    """
    # Quick check: does pyproj.CRS already work?
    try:
        import pyproj

        pyproj.CRS("epsg:4326")
        _diag.append("  pyproj already working, no fix needed")
        return
    except Exception:
        pass

    # Find a proj.db that the already-loaded libproj.so can use.
    # Prefer QGIS's own PROJ data (matching its libproj version), then
    # fall back to the venv's bundled copy.
    qgis_python = sys.executable
    proj_search_paths = []
    # QGIS conda env share/proj
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        proj_search_paths.append(os.path.join(conda_prefix, "share", "proj"))
    # Derive from sys.executable (e.g. .../miniconda3/envs/qgis/bin/python3)
    exe_prefix = os.path.dirname(os.path.dirname(qgis_python))
    proj_search_paths.append(os.path.join(exe_prefix, "share", "proj"))
    # sys.prefix
    proj_search_paths.append(os.path.join(sys.prefix, "share", "proj"))

    for proj_dir in proj_search_paths:
        if os.path.exists(os.path.join(proj_dir, "proj.db")):
            try:
                import pyproj.datadir

                pyproj.datadir.set_data_dir(proj_dir)
                os.environ["PROJ_DATA"] = proj_dir
                os.environ["PROJ_LIB"] = proj_dir
                _diag.append(f"  set pyproj data_dir={proj_dir}")
                # Verify it works
                pyproj.CRS("epsg:4326")
                _diag.append("  pyproj fix verified OK")
                return
            except Exception as exc:
                _diag.append(f"  pyproj fix attempt {proj_dir} failed: {exc}")

    _diag.append("  WARNING: could not fix pyproj PROJ database")


def _add_windows_dll_directories(site_packages: Path) -> None:
    """Register DLL search directories on Windows for native packages.

    On Windows, torch and other native packages ship DLLs in subdirectories
    (e.g. ``torch/lib/``) that the OS loader doesn't search by default when
    loading from a foreign venv inside QGIS's process.  We must add them
    explicitly via ``os.add_dll_directory()`` *and* prepend them to ``PATH``
    (the latter covers legacy ``LoadLibrary`` calls without search flags).
    """
    if sys.platform != "win32":
        return

    dll_dirs = [
        site_packages / "torch" / "lib",
        site_packages / "torch" / "bin",
        site_packages / "torchvision",
    ]

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in dll_dirs:
        if dll_dir.is_dir():
            dll_dir_str = str(dll_dir)
            try:
                os.add_dll_directory(dll_dir_str)
                _diag.append(f"  add_dll_directory: {dll_dir_str}")
            except OSError as exc:
                _diag.append(f"  add_dll_directory {dll_dir_str} failed: {exc}")
            if dll_dir_str not in path_parts:
                path_parts.insert(0, dll_dir_str)

    os.environ["PATH"] = os.pathsep.join(path_parts)


def _load_geoai_from_path(init_path: Path) -> Optional[ModuleType]:
    """Load the external geoai package from an explicit ``__init__.py`` path.

    The external library **permanently** takes over ``sys.modules["geoai"]``
    (and its submodule entries).  This is safe because the plugin's own code
    uses only relative imports.
    """
    pkg_dir = init_path.parent

    # Remove the plugin's geoai entries from sys.modules so the external
    # library can claim the "geoai" namespace.
    saved = {}
    for key in list(sys.modules.keys()):
        if key == "geoai" or key.startswith("geoai."):
            saved[key] = sys.modules.pop(key)

    # Fix pyproj PROJ database context.  We must NOT clear pyproj/rasterio
    # from sys.modules — their native extensions (.so) are cached per-process
    # and cannot be reloaded from a different path.  Instead, configure the
    # already-loaded QGIS pyproj to find its own proj.db.
    _fix_proj_for_qgis()

    # On Windows, register DLL directories for native packages (torch, etc.)
    # before loading any modules that depend on them.
    site_packages = pkg_dir.parent  # geoai/__init__.py -> geoai -> site-packages
    _add_windows_dll_directories(site_packages)

    def _attempt_load():
        """Try to load the geoai module once. Returns module or raises."""
        spec = importlib.util.spec_from_file_location(
            "geoai",
            str(init_path),
            submodule_search_locations=[str(pkg_dir)],
        )
        if spec is None or spec.loader is None:
            _diag.append(f"  load {init_path}: spec_from_file_location returned None")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules["geoai"] = module
        spec.loader.exec_module(module)

        if _is_plugin_module(module):
            _diag.append(f"  load {init_path}: is plugin module, skipping")
            for key in list(sys.modules.keys()):
                if key == "geoai" or key.startswith("geoai."):
                    del sys.modules[key]
            return None

        return module

    def _cleanup_and_restore():
        """Remove any partial geoai entries and restore saved modules."""
        for key in list(sys.modules.keys()):
            if key == "geoai" or key.startswith("geoai."):
                try:
                    del sys.modules[key]
                except KeyError:
                    pass
        sys.modules.update(saved)

    def _finalize(module):
        """Re-register plugin sub-modules and adjust __path__."""
        _diag.append(f"  load {init_path}: SUCCESS")
        plugin_dir = str(Path(__file__).resolve().parent)
        for key, mod in saved.items():
            if key.startswith("geoai.") and key not in sys.modules:
                sys.modules[key] = mod
        if hasattr(module, "__path__"):
            if plugin_dir not in module.__path__:
                module.__path__.append(plugin_dir)
        return module

    try:
        module = _attempt_load()
        if module is None:
            sys.modules.update(saved)
            return None
        return _finalize(module)
    except OSError as exc:
        # On Windows, torch DLL loading can fail with WinError 127.
        # Install a torch import blocker so ALL transitive torch imports
        # raise ImportError, then retry.
        _diag.append(f"  load {init_path}: OSError: {exc}")
        _cleanup_and_restore()

        if sys.platform == "win32" and "torch" in str(exc).lower():
            _diag.append("  Installing torch import blocker and retrying")

            # Remove any partial torch entries from sys.modules
            for key in list(sys.modules.keys()):
                if key == "torch" or key.startswith("torch."):
                    del sys.modules[key]

            # Patch builtins.__import__ to block torch imports
            restore_import = _install_torch_import_blocker()

            # Re-remove geoai entries for fresh attempt
            for key in list(sys.modules.keys()):
                if key == "geoai" or key.startswith("geoai."):
                    sys.modules.pop(key, None)

            try:
                module = _attempt_load()
                if module is not None:
                    for key, mod in saved.items():
                        if key.startswith("geoai.") and key not in sys.modules:
                            sys.modules[key] = mod
                    _diag.append("  retry with torch blocker: SUCCESS")
                    return _finalize(module)
            except Exception as retry_exc:
                tb_str = traceback.format_exc()
                _diag.append(
                    f"  retry with torch blocker FAILED: {retry_exc}\n{tb_str}"
                )
            finally:
                # Always restore the original __import__
                restore_import()

            # Restore on retry failure
            _cleanup_and_restore()
        return None
    except Exception as exc:
        tb_str = traceback.format_exc()
        _diag.append(f"  load {init_path}: FAILED: {exc}\n{tb_str}")
        _cleanup_and_restore()
        return None


def _try_ensure_venv_available():
    """Add the GeoAI plugin venv site-packages to sys.path if available."""
    try:
        from .core.venv_manager import ensure_venv_packages_available, venv_exists

        if venv_exists():
            ensure_venv_packages_available()
            _diag.append("Strategy 0: venv exists, site-packages added to sys.path")
        else:
            _diag.append("Strategy 0: venv does not exist")
    except Exception as exc:
        _diag.append(f"Strategy 0: exception: {exc}")


def get_geoai() -> ModuleType:
    """Return the external ``geoai`` library module.

    Tries four strategies in order:

    0. Ensure the plugin's managed venv site-packages is on ``sys.path``.
    1. Normal ``import geoai`` — works when the plugin is NOT named ``geoai``.
    2. Locate the installed distribution (``geoai-py`` or ``geoai``) via
       ``importlib.metadata`` and load the library from its file path.
    3. Direct filesystem scan of ``sys.path`` for ``geoai/__init__.py``.

    The result is cached after the first successful resolution.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    _diag.clear()
    _diag.append(f"Plugin __file__: {__file__}")

    # Strategy 0: ensure venv packages are available on sys.path
    _try_ensure_venv_available()

    # Strategy 1: normal import (works when plugin package is NOT named `geoai`)
    _diag.append("Strategy 1: importlib.import_module('geoai')")
    try:
        imported = importlib.import_module("geoai")
        if not _is_plugin_module(imported):
            _diag.append("  result: SUCCESS (not plugin module)")
            _CACHED = imported
            return imported
        else:
            _diag.append(
                f"  result: found plugin module at {getattr(imported, '__file__', '?')}"
            )
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        _diag.append(f"  result: import error: {exc}")

    # Strategy 2: find via distribution metadata
    _diag.append("Strategy 2: distribution metadata lookup")
    for dist_name in ("geoai-py", "geoai"):
        init_path = _find_geoai_init_from_dist(dist_name)
        if init_path is None:
            continue
        ext = _load_geoai_from_path(init_path)
        if ext is not None:
            _CACHED = ext
            return ext

    # Strategy 3: direct sys.path scan
    _diag.append("Strategy 3: sys.path filesystem scan")
    # Log the first few sys.path entries containing "geoai" or the venv
    _cache_marker = (
        os.environ.get("GEOAI_CACHE_DIR")
        or os.environ.get("GEOAI_VENV_DIR")
        or ".qgis_geoai"
    )
    for idx, entry in enumerate(sys.path[:20]):
        if "geoai" in entry.lower() or _cache_marker in entry:
            _diag.append(f"  sys.path[{idx}]: {entry}")
    init_path = _find_geoai_init_from_sys_path()
    if init_path is not None:
        ext = _load_geoai_from_path(init_path)
        if ext is not None:
            _CACHED = ext
            return ext

    # All strategies failed — build a diagnostic error message
    py = getattr(sys, "executable", "python")
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    diag_text = "\n".join(_diag)

    # Check if the venv geoai package exists on disk
    venv_check = ""
    try:
        from .core.venv_manager import VENV_DIR, get_venv_site_packages

        sp = get_venv_site_packages()
        geoai_in_venv = os.path.join(sp, "geoai", "__init__.py")
        venv_check = (
            f"\n\nVenv check:\n"
            f"  VENV_DIR: {VENV_DIR}\n"
            f"  site-packages: {sp}\n"
            f"  exists: {os.path.exists(sp)}\n"
            f"  geoai/__init__.py exists: {os.path.exists(geoai_in_venv)}"
        )
    except Exception as exc:
        venv_check = f"\n\nVenv check failed: {exc}"

    raise ImportError(
        "GeoAI plugin could not import the external 'geoai' library.\n\n"
        f"QGIS Python:\n  executable: {py}\n  version: {ver}\n"
        f"{venv_check}\n\n"
        f"Strategy diagnostics:\n{diag_text}\n\n"
        "Fix: Click any GeoAI toolbar button to open the dependency installer,\n"
        "which will automatically download and install all required packages.\n\n"
        "Alternatively, install the GeoAI Python package manually:\n"
        "  In QGIS Python Console, run:\n"
        "  import sys, subprocess\n"
        "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'geoai-py'])\n"
    )
