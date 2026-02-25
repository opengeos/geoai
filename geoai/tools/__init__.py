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
except (ImportError, OSError):
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
except (ImportError, OSError):
    # OmniCloudMask not installed or torch DLL load failure
    pass


# Super Resolution integration (optional dependency)

try:
    from .sr import super_resolution

    __all__.extend(["super_resolution"])
except (ImportError, OSError):
    # Super resolution not installed or torch DLL load failure
    pass

# Time-series analysis utilities (rasterio required)
try:
    from .timeseries import (
        validate_temporal_stack,
        create_temporal_composite,
        create_cloud_free_composite,
        calculate_spectral_index_timeseries,
        detect_change,
        calculate_temporal_statistics,
        extract_dates_from_filenames,
        sort_by_date,
        COMPOSITE_METHODS,
        SPECTRAL_INDICES,
        CHANGE_METHODS,
        TEMPORAL_STATISTICS,
        SENTINEL2_DATE_PATTERN,
        LANDSAT_DATE_PATTERN,
    )

    __all__.extend(
        [
            "validate_temporal_stack",
            "create_temporal_composite",
            "create_cloud_free_composite",
            "calculate_spectral_index_timeseries",
            "detect_change",
            "calculate_temporal_statistics",
            "extract_dates_from_filenames",
            "sort_by_date",
            "COMPOSITE_METHODS",
            "SPECTRAL_INDICES",
            "CHANGE_METHODS",
            "TEMPORAL_STATISTICS",
            "SENTINEL2_DATE_PATTERN",
            "LANDSAT_DATE_PATTERN",
        ]
    )
except (ImportError, OSError):
    # rasterio not installed - time-series functions will not be available
    pass
