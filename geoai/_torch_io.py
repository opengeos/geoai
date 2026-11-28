"""Internal PyTorch checkpoint I/O and raster metadata helpers."""

import logging
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_UNSAFE_LOAD_ENV_VAR = "GEOAI_ALLOW_UNSAFE_TORCH_LOAD"


def _is_weights_only_error(exc: BaseException) -> bool:
    """Return True if exc indicates a restricted weights_only=True load failure."""
    if isinstance(exc, pickle.UnpicklingError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "weights only" in msg
            or "weights_only" in msg
            or "unsupported global" in msg
        )
    return False


def _safe_torch_load(
    path: str,
    map_location: Any = None,
    allow_unsafe: Optional[bool] = None,
) -> Any:
    """Load a PyTorch checkpoint using ``weights_only=True`` by default.

    Unrestricted pickle loading (``weights_only=False``) is never attempted
    unless explicitly opted in via ``allow_unsafe=True`` or the environment
    variable ``GEOAI_ALLOW_UNSAFE_TORCH_LOAD=1``. Pre-2.0 PyTorch versions
    that do not accept the ``weights_only`` keyword argument fall back to
    plain ``torch.load``.

    Args:
        path (str): Path to the checkpoint file.
        map_location (Any, optional): Device mapping passed to ``torch.load``.
        allow_unsafe (bool, optional): If True, allow retrying with
            ``weights_only=False`` when a legacy checkpoint fails restricted
            loading. If None, checks ``GEOAI_ALLOW_UNSAFE_TORCH_LOAD == "1"``.

    Returns:
        Any: Loaded checkpoint object or state dictionary.
    """
    unsafe_allowed = (
        bool(allow_unsafe)
        if allow_unsafe is not None
        else os.environ.get(_UNSAFE_LOAD_ENV_VAR) == "1"
    )
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" in str(exc):
            return torch.load(path, map_location=map_location)
        raise
    except Exception as exc:
        if not _is_weights_only_error(exc):
            raise
        if not unsafe_allowed:
            err_cls = (
                pickle.UnpicklingError
                if isinstance(exc, pickle.UnpicklingError)
                else RuntimeError
            )
            raise err_cls(
                f"Failed to load checkpoint {path!r} with weights_only=True "
                f"({exc}). If you trust this legacy checkpoint, pass "
                f"allow_unsafe=True or set {_UNSAFE_LOAD_ENV_VAR}=1."
            ) from exc
        logger.warning(
            "Retrying legacy checkpoint %s with weights_only=False because "
            "explicit unsafe loading opt-in is enabled: %s",
            path,
            exc,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def _clear_incompatible_nodata(meta: Dict[str, Any], dtype: str) -> None:
    """Clear inherited nodata if it cannot be represented in the target dtype."""
    nodata = meta.get("nodata")
    if nodata is None:
        return
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        try:
            if (
                (
                    isinstance(nodata, float)
                    and (not np.isfinite(nodata) or not nodata.is_integer())
                )
                or nodata < info.min
                or nodata > info.max
            ):
                meta["nodata"] = None
        except TypeError:
            meta["nodata"] = None
