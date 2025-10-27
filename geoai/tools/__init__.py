"""
GeoAI Tools - Utility functions and integrations for geospatial AI workflows.

This subpackage contains various tools and integrations for enhancing
geospatial AI workflows, including post-processing utilities and
third-party library integrations.
"""

# MultiClean integration (optional dependency)
try:
    from .multiclean import (
        clean_segmentation_mask,
        clean_raster,
        clean_raster_batch,
        compare_masks,
        check_multiclean_available,
    )

    __all__ = [
        "clean_segmentation_mask",
        "clean_raster",
        "clean_raster_batch",
        "compare_masks",
        "check_multiclean_available",
    ]
except ImportError:
    # MultiClean not installed - functions will not be available
    __all__ = []
