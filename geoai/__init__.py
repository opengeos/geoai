"""Top-level package for geoai."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "0.3.5"


import os
import sys


def set_proj_lib_path():
    """Set the PROJ_LIB environment variable based on the current conda environment."""
    try:
        # Get conda environment path
        conda_env_path = os.environ.get("CONDA_PREFIX") or sys.prefix

        # Set PROJ_LIB environment variable
        proj_path = os.path.join(conda_env_path, "share", "proj")
        gdal_path = os.path.join(conda_env_path, "share", "gdal")

        # Check if the directory exists before setting
        if os.path.exists(proj_path):
            os.environ["PROJ_LIB"] = proj_path
        if os.path.exists(gdal_path):
            os.environ["GDAL_DATA"] = gdal_path
    except Exception as e:
        print(e)
        return


if "google.colab" not in sys.modules:
    set_proj_lib_path()

from .geoai import *
