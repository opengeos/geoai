"""
GeoAI Plugin Dialogs

This module contains the dialog and dock widget classes for the GeoAI plugin.
"""

from .moondream import MoondreamDockWidget
from .segmentation import SegmentationDockWidget

__all__ = [
    "MoondreamDockWidget",
    "SegmentationDockWidget",
]
