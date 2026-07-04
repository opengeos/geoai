#!/usr/bin/env python
"""Dry-run resolver for GeoAI QGIS plugin dependencies.

This script is intentionally lightweight: it stubs the tiny part of QGIS needed
by ``venv_manager`` and asks uv to resolve the plugin dependency set for a
target Python/platform without installing the large AI packages.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types


def _install_qgis_stub() -> None:
    """Install the minimal qgis.core stub required by venv_manager imports."""
    if "qgis" in sys.modules:
        return
    qgis = types.ModuleType("qgis")
    qgis.__path__ = []
    core = types.ModuleType("qgis.core")

    class _Qgis:
        Info = 0
        Warning = 1
        Critical = 2
        Success = 3
        QGIS_VERSION = "resolver-stub"

    class _QgsMessageLog:
        @staticmethod
        def logMessage(message, tag="GeoAI", level=0):  # noqa: N802 - QGIS API
            return None

    setattr(core, "Qgis", _Qgis)
    setattr(core, "QgsMessageLog", _QgsMessageLog)
    setattr(qgis, "core", core)
    sys.modules["qgis"] = qgis
    sys.modules["qgis.core"] = core


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run GeoAI QGIS plugin dependency resolution."
    )
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument(
        "--platform",
        default="win32",
        choices=["win32", "darwin", "linux"],
        help="Target Python platform to resolve for.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--cuda-index",
        default=None,
        help=(
            "PyTorch CUDA wheel index to mirror a GPU install, e.g. 'cu126'. "
            "When set, the resolve adds the PyTorch --extra-index-url and the "
            "multi-index strategy, reproducing the real install (issue #829)."
        ),
    )
    args = parser.parse_args(argv)

    _install_qgis_stub()
    plugin_root = pathlib.Path(__file__).resolve().parents[1]
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

    venv_manager_path = plugin_root / "geoai" / "core" / "venv_manager.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "geoai_qgis_venv_manager", venv_manager_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {venv_manager_path}")
    venv_manager = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = venv_manager
    spec.loader.exec_module(venv_manager)

    success, message = venv_manager.resolve_qgis_dependencies(
        python_version=args.python_version,
        platform_name=args.platform,
        timeout=args.timeout,
        cuda_index=args.cuda_index,
    )
    print(message)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
