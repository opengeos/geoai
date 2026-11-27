"""Access PyTorch from the QGIS process only when it is really needed.

Importing ``torch`` into QGIS is not free. On Windows its DLLs can conflict
with QGIS's own. On macOS with QGIS installed from conda-forge, QGIS has
already loaded conda's ``libomp`` and torch brings a second copy, which aborts
the whole process ("OMP: Error #15"). The panels therefore never import torch
when they open: models run in the plugin venv's subprocess, and only code that
actually runs a model inside QGIS imports torch.
"""

import sys
from typing import Any, Optional, Tuple


def import_torch() -> Tuple[Optional[Any], Optional[BaseException]]:
    """Import PyTorch into the QGIS process.

    Call this only on code paths that run a model inside QGIS.

    Returns:
        ``(torch, None)`` on success, or ``(None, error)`` if the import
        failed with ``ImportError`` or ``OSError`` (for example a DLL
        conflict).
    """
    try:
        import torch
    except (ImportError, OSError) as exc:
        return None, exc
    return torch, None


def loaded_torch() -> Optional[Any]:
    """Return PyTorch if something already imported it into this process.

    Never imports it. Use this for cleanup and error checks, which have
    nothing to do when no model ever ran inside QGIS.

    Returns:
        The ``torch`` module, or None if it is not loaded.
    """
    return sys.modules.get("torch")
