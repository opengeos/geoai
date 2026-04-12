"""GeoAI backend -- verifies geoai is installed and provides utility wrappers.

Since geoai is a Python library (not a GUI app with a separate process),
the backend is invoked via direct imports. This module validates the
installation and provides helper functions for the CLI.
"""

import os
import sys
from typing import Any, Dict, Optional


def check_geoai_installed() -> Dict[str, Any]:
    """Check if geoai is installed and return version info.

    Returns:
        Dict with installation status and version.

    Raises:
        RuntimeError: If geoai is not installed.
    """
    try:
        import geoai

        return {
            "installed": True,
            "version": getattr(geoai, "__version__", "unknown"),
            "path": os.path.dirname(geoai.__file__),
        }
    except ImportError:
        raise RuntimeError(
            "GeoAI is not installed. Install it with:\n"
            "  pip install geoai-py\n"
            "  # or\n"
            "  conda install -c conda-forge geoai"
        )


def check_torch_available() -> Dict[str, Any]:
    """Check if PyTorch is available and report device info.

    Returns:
        Dict with torch version, CUDA availability, and device info.
    """
    try:
        import torch

        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": (
                torch.backends.mps.is_available()
                if hasattr(torch.backends, "mps")
                else False
            ),
        }

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )

        if info.get("cuda_available"):
            info["default_device"] = "cuda"
        elif info.get("mps_available"):
            info["default_device"] = "mps"
        else:
            info["default_device"] = "cpu"

        return info
    except ImportError:
        return {
            "torch_version": None,
            "cuda_available": False,
            "mps_available": False,
            "default_device": "cpu",
        }


def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics.

    Returns:
        Dict with Python version, platform, and dependency versions.
    """
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
    }

    geoai_info = check_geoai_installed()
    info["geoai_version"] = geoai_info["version"]
    info["geoai_path"] = geoai_info["path"]

    torch_info = check_torch_available()
    info.update(torch_info)

    deps = [
        "rasterio",
        "geopandas",
        "shapely",
        "numpy",
        "transformers",
        "segmentation_models_pytorch",
    ]
    for dep in deps:
        try:
            mod = __import__(dep)
            info[f"{dep}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[f"{dep}_version"] = None

    return info
