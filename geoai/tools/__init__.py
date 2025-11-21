"""
GeoAI Tools - Utility functions and integrations for geospatial AI workflows.

This subpackage contains various tools and integrations for enhancing
geospatial AI workflows, including post-processing utilities and
third-party library integrations.
"""

__all__ = []

# MultiClean integration (optional dependency)
try:
    from .multiclean import (
        clean_segmentation_mask,
        clean_raster,
        clean_raster_batch,
        compare_masks,
        check_multiclean_available,
    )

    __all__.extend(
        [
            "clean_segmentation_mask",
            "clean_raster",
            "clean_raster_batch",
            "compare_masks",
            "check_multiclean_available",
        ]
    )
except ImportError:
    # MultiClean not installed - functions will not be available
    pass

# OmniCloudMask integration (optional dependency)
try:
    from .cloudmask import (
        predict_cloud_mask,
        predict_cloud_mask_from_raster,
        predict_cloud_mask_batch,
        calculate_cloud_statistics,
        create_cloud_free_mask,
        check_omnicloudmask_available,
        CLEAR,
        THICK_CLOUD,
        THIN_CLOUD,
        CLOUD_SHADOW,
    )

    __all__.extend(
        [
            "predict_cloud_mask",
            "predict_cloud_mask_from_raster",
            "predict_cloud_mask_batch",
            "calculate_cloud_statistics",
            "create_cloud_free_mask",
            "check_omnicloudmask_available",
            "CLEAR",
            "THICK_CLOUD",
            "THIN_CLOUD",
            "CLOUD_SHADOW",
        ]
    )
except ImportError:
    # OmniCloudMask not installed - functions will not be available
    pass


# Super Resolution integration (optional dependency)

try:
    from .sr import super_resolution

    __all__.extend(["super_resolution"])
except ImportError:
    # Super resolution not installed - function will not be available
    pass
