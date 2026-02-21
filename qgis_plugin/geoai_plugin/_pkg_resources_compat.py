"""Compatibility shim for environments where ``pkg_resources`` is unavailable.

Some third-party packages still import ``pkg_resources`` at runtime.
On newer setuptools installations (or stripped Python runtimes), that module may
be missing, which can break imports even when modern alternatives are available.

This shim provides a minimal subset commonly used by downstream libraries.
It is intentionally conservative and only activated when ``pkg_resources`` cannot
be imported.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import importlib.metadata
import importlib.resources
import sys
import types


class DistributionNotFound(Exception):
    """Raised when a requested distribution cannot be found."""


def _build_module() -> types.ModuleType:
    module = types.ModuleType("pkg_resources")
    module.__dict__["__doc__"] = "Lightweight pkg_resources compatibility shim."

    module.DistributionNotFound = DistributionNotFound

    def get_distribution(dist_name: str):
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DistributionNotFound(str(exc)) from exc

        class _Dist:
            def __init__(self, project_name: str, version: str):
                self.project_name = project_name
                self.version = version

            def __str__(self) -> str:
                return f"{self.project_name} {self.version}"

        return _Dist(dist_name, version)

    def parse_version(version: str):
        try:
            from packaging.version import parse as _parse

            return _parse(version)
        except Exception:
            # Basic fallback: preserve sortable semantics for typical X.Y.Z strings.
            return tuple(int(p) if p.isdigit() else p for p in str(version).split("."))

    def resource_filename(package_or_requirement: str, resource_name: str) -> str:
        package = (
            package_or_requirement.project_name
            if hasattr(package_or_requirement, "project_name")
            else str(package_or_requirement)
        )
        if not package:
            raise ValueError("package_or_requirement must be non-empty")

        resource = importlib.resources.files(package).joinpath(resource_name)
        return str(Path(resource))

    def require(*_args, **_kwargs):
        # Commonly called for side effects in legacy code; no-op for shim.
        return []

    module.get_distribution = get_distribution
    module.parse_version = parse_version
    module.resource_filename = resource_filename
    module.require = require
    return module


def ensure_pkg_resources() -> bool:
    """Ensure ``pkg_resources`` is importable.

    Returns:
        bool: True if a shim was installed, False if a real module already exists.
    """
    if "pkg_resources" in sys.modules:
        return False

    try:
        importlib.import_module("pkg_resources")
        return False
    except Exception:
        sys.modules["pkg_resources"] = _build_module()
        return True
