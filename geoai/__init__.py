"""Top-level package for geoai."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "0.6.0"


import os
import sys


def set_proj_lib_path(verbose=False):
    """
    Set the PROJ_LIB and GDAL_DATA environment variables based on the current conda environment.

    This function attempts to locate and set the correct paths for PROJ_LIB and GDAL_DATA
    by checking multiple possible locations within the conda environment structure.

    Args:
        verbose (bool): If True, print additional information during the process.

    Returns:
        bool: True if both paths were set successfully, False otherwise.
    """
    try:
        from rasterio.env import set_gdal_config

        # Get conda environment path
        conda_env_path = os.environ.get("CONDA_PREFIX") or sys.prefix

        # Define possible paths for PROJ_LIB
        possible_proj_paths = [
            os.path.join(conda_env_path, "share", "proj"),
            os.path.join(conda_env_path, "Library", "share", "proj"),
            os.path.join(conda_env_path, "Library", "share"),
        ]

        # Define possible paths for GDAL_DATA
        possible_gdal_paths = [
            os.path.join(conda_env_path, "share", "gdal"),
            os.path.join(conda_env_path, "Library", "share", "gdal"),
            os.path.join(conda_env_path, "Library", "data", "gdal"),
            os.path.join(conda_env_path, "Library", "share"),
        ]

        # Set PROJ_LIB environment variable
        proj_set = False
        for proj_path in possible_proj_paths:
            if os.path.exists(proj_path) and os.path.isdir(proj_path):
                # Verify it contains projection data
                if os.path.exists(os.path.join(proj_path, "proj.db")):
                    os.environ["PROJ_LIB"] = proj_path
                    if verbose:
                        print(f"PROJ_LIB set to: {proj_path}")
                    proj_set = True
                    break

        # Set GDAL_DATA environment variable
        gdal_set = False
        for gdal_path in possible_gdal_paths:
            if os.path.exists(gdal_path) and os.path.isdir(gdal_path):
                # Verify it contains the header.dxf file or other critical GDAL files
                if os.path.exists(
                    os.path.join(gdal_path, "header.dxf")
                ) or os.path.exists(os.path.join(gdal_path, "gcs.csv")):
                    os.environ["GDAL_DATA"] = gdal_path
                    if verbose:
                        print(f"GDAL_DATA set to: {gdal_path}")
                    gdal_set = True
                    break

        # If paths still not found, try a last-resort approach
        if not proj_set or not gdal_set:
            # Try a deep search in the conda environment
            for root, dirs, files in os.walk(conda_env_path):
                if not gdal_set and "header.dxf" in files:
                    os.environ["GDAL_DATA"] = root
                    if verbose:
                        print(f"GDAL_DATA set to: {root} (deep search)")
                    gdal_set = True

                if not proj_set and "proj.db" in files:
                    os.environ["PROJ_LIB"] = root
                    if verbose:
                        print(f"PROJ_LIB set to: {root} (deep search)")
                    proj_set = True

                if proj_set and gdal_set:
                    break

        set_gdal_config("PROJ_LIB", os.environ["PROJ_LIB"])
        set_gdal_config("GDAL_DATA", os.environ["GDAL_DATA"])

    except Exception as e:
        print(f"Error setting projection library paths: {e}")
        return


# if ("google.colab" not in sys.modules) and (sys.platform != "windows"):
#     set_proj_lib_path()

from .geoai import *
