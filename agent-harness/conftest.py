"""Conftest for cli-anything-geoai tests.

Ensures the real geoai package is imported before pytest's test collection
resolves our cli_anything.geoai package as the top-level 'geoai'.
"""

import importlib
import sys


def pytest_configure(config):
    """Pre-load the real geoai package to prevent namespace shadowing.

    pytest walks up from test files looking for __init__.py to determine
    package names. Since cli_anything/ is a namespace package (no __init__.py),
    pytest treats cli_anything/geoai/ as the top-level 'geoai' package.
    This pre-loads the real geoai so sys.modules['geoai'] is correct.
    """
    # Force-load the real geoai package before test collection
    if "geoai" not in sys.modules:
        # Import the real geoai package
        real_geoai = importlib.import_module("geoai")
        sys.modules["geoai"] = real_geoai
    # Also ensure geoai.utils resolves correctly
    if "geoai.utils" not in sys.modules:
        real_utils = importlib.import_module("geoai.utils")
        sys.modules["geoai.utils"] = real_utils
