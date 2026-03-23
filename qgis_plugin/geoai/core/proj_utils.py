"""Utilities for managing PROJ environment variables in the QGIS plugin.

pyogrio (the default geopandas I/O engine since geopandas >= 1.0) validates
that ``PROJ_DATA`` points to its own bundled data files.  When ``PROJ_DATA``
points elsewhere (e.g. pyproj or QGIS PROJ data), pyogrio raises::

    Could not correctly detect PROJ data files installed by pyogrio wheel

This module provides a context manager and helper that temporarily clear
``PROJ_DATA`` / ``PROJ_LIB`` around ``GeoDataFrame.to_file()`` calls so
pyogrio can auto-detect its own bundled PROJ data.
"""

import os
from contextlib import contextmanager


@contextmanager
def clean_proj_env():
    """Temporarily remove PROJ_DATA/PROJ_LIB from the environment.

    Use around ``gdf.to_file()`` calls to prevent pyogrio PROJ data
    detection errors.  The original values are restored on exit.

    Yields:
        None
    """
    saved = {}
    for var in ("PROJ_DATA", "PROJ_LIB"):
        if var in os.environ:
            saved[var] = os.environ.pop(var)
    try:
        yield
    finally:
        os.environ.update(saved)


def safe_to_file(gdf, output_path, **kwargs):
    """Write a GeoDataFrame to file with PROJ env vars temporarily cleared.

    Wrapper around ``gdf.to_file()`` that avoids pyogrio PROJ data
    detection errors.

    Args:
        gdf: GeoDataFrame to save.
        output_path: Path for the output file.
        **kwargs: Additional keyword arguments passed to ``gdf.to_file()``.
    """
    with clean_proj_env():
        gdf.to_file(output_path, **kwargs)
