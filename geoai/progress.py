"""Centralized progress bar handling for geoai.

Every progress bar in geoai is created through the :class:`tqdm` wrapper defined
in this module. The wrapper behaves exactly like ``tqdm.auto.tqdm`` but honors a
package-wide switch, so users can silence all progress reporting at once::

    import geoai

    geoai.disable_progress_bars()

The switch can also be set before importing geoai with the
``GEOAI_DISABLE_PROGRESS`` environment variable::

    import os

    os.environ["GEOAI_DISABLE_PROGRESS"] = "1"

Using ``tqdm.auto`` means notebooks (Jupyter, Colab) get widget-based progress
bars that update in place instead of streaming thousands of text updates to the
front end, which could freeze or crash the browser tab.
"""

from __future__ import annotations

import os
from typing import Any

from tqdm.auto import tqdm as _tqdm

__all__ = [
    "tqdm",
    "disable_progress_bars",
    "enable_progress_bars",
    "set_progress_bars",
    "progress_bars_disabled",
]

_TRUTHY = {"1", "true", "t", "yes", "y", "on"}

# Package-wide switch. ``None`` means "not explicitly set", in which case the
# ``GEOAI_DISABLE_PROGRESS`` environment variable decides.
_DISABLED: bool | None = None


def _env_disabled() -> bool:
    """Check whether progress bars are disabled via environment variable.

    Returns:
        bool: True if ``GEOAI_DISABLE_PROGRESS`` is set to a truthy value.
    """
    return os.environ.get("GEOAI_DISABLE_PROGRESS", "").strip().lower() in _TRUTHY


def progress_bars_disabled() -> bool:
    """Report whether geoai progress bars are currently disabled.

    Returns:
        bool: True if progress bars are globally disabled, False otherwise.
    """
    if _DISABLED is not None:
        return _DISABLED
    return _env_disabled()


def set_progress_bars(enabled: bool) -> None:
    """Enable or disable all geoai progress bars.

    Args:
        enabled (bool): If True, progress bars are shown. If False, every
            progress bar created by geoai is suppressed.

    Returns:
        None
    """
    global _DISABLED
    _DISABLED = not bool(enabled)


def disable_progress_bars() -> None:
    """Disable all geoai progress bars.

    Useful in notebooks (e.g., Google Colab) where a long-running progress bar
    can flood the output and freeze the browser tab.

    Returns:
        None
    """
    set_progress_bars(False)


def enable_progress_bars() -> None:
    """Re-enable all geoai progress bars.

    Returns:
        None
    """
    set_progress_bars(True)


class tqdm(_tqdm):  # noqa: N801 - drop-in replacement for tqdm.auto.tqdm
    """Drop-in replacement for ``tqdm.auto.tqdm`` that honors the global switch.

    The only difference from ``tqdm.auto.tqdm`` is that the progress bar is
    forced off when progress bars have been disabled with
    :func:`disable_progress_bars`, :func:`set_progress_bars`, or the
    ``GEOAI_DISABLE_PROGRESS`` environment variable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the progress bar.

        Args:
            *args: Positional arguments forwarded to ``tqdm.auto.tqdm``.
            **kwargs: Keyword arguments forwarded to ``tqdm.auto.tqdm``. The
                ``disable`` keyword is forced to True when progress bars are
                globally disabled.
        """
        if progress_bars_disabled():
            kwargs["disable"] = True
        super().__init__(*args, **kwargs)
