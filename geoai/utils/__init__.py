"""The utils module contains common functions and classes used by the other modules.

This package is organized into focused submodules:

- ``device``: Device detection and environment utilities
- ``download``: File download utilities
- ``metrics``: Segmentation metrics and evaluation
- ``models``: Model inspection and loading
- ``geometry``: Geometry processing and regularization
- ``conversion``: Data conversion and coordinate transforms
- ``visualization``: Raster and vector visualization
- ``raster``: Raster I/O and processing
- ``vector``: Vector I/O and processing
- ``training``: Training data export and augmentation

All public functions are re-exported here for backward compatibility,
so ``from geoai.utils import <function>`` continues to work.

Submodules that require PyTorch (``metrics``, ``models``, ``training``,
``visualization``) are loaded lazily on first access so that importing
``geoai.utils`` does not require torch to be available — which allows the
library to load inside environments where torch DLLs cannot be initialized
(e.g. the QGIS process on Windows).
"""

import importlib

# ---------------------------------------------------------------------------
# Eager imports — submodules with no PyTorch dependency
# ---------------------------------------------------------------------------

from .conversion import (
    bbox_to_xy,
    coords_to_xy,
    dict_to_image,
    dict_to_rioxarray,
    rowcol_to_xy,
)
from .device import (
    empty_cache,
    get_device,
    install_package,
    temp_file_path,
)
from .download import (
    download_file,
    download_model_from_hf,
)
from .geometry import (
    adaptive_regularization,
    hybrid_regularization,
    orthogonalize,
    region_groups,
    regularization,
    regularize,
)
from .raster import (
    batch_vector_to_raster,
    calc_stats,
    clip_raster_by_bbox,
    get_raster_info,
    get_raster_info_gdal,
    get_raster_resolution,
    get_raster_stats,
    masks_to_vector,
    mosaic_geotiffs,
    print_raster_info,
    raster_to_vector,
    raster_to_vector_batch,
    read_raster,
    read_vector,
    stack_bands,
    vector_to_raster,
    write_colormap,
)
from .vector import (
    add_geometric_properties,
    analyze_vector_attributes,
    boxes_to_vector,
    export_tiles_to_geojson,
    geojson_to_coords,
    geojson_to_xy,
    get_vector_info,
    get_vector_info_ogr,
    print_vector_info,
    smooth_vector,
    vector_to_geojson,
    visualize_vector_by_attribute,
)

# ---------------------------------------------------------------------------
# Lazy imports — submodules that import torch at the top level.
# These are resolved on first attribute access via __getattr__ so that
# importing geoai.utils never forces torch to load.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict = {
    # .metrics
    "calc_f1_score": "metrics",
    "calc_iou": "metrics",
    "calc_segmentation_metrics": "metrics",
    # .models
    "inspect_pth_file": "models",
    "try_common_architectures": "models",
    # .training
    "export_flipnslide_tiles": "training",
    "export_geotiff_tiles": "training",
    "export_geotiff_tiles_batch": "training",
    "export_training_data": "training",
    "flipnslide_augmentation": "training",
    "get_default_augmentation_transforms": "training",
    # .visualization
    "create_overview_image": "visualization",
    "create_split_map": "visualization",
    "display_image_with_vector": "visualization",
    "display_training_tiles": "visualization",
    "plot_batch": "visualization",
    "plot_images": "visualization",
    "plot_masks": "visualization",
    "plot_performance_metrics": "visualization",
    "plot_prediction_comparison": "visualization",
    "view_image": "visualization",
    "view_raster": "visualization",
    "view_vector": "visualization",
    "view_vector_interactive": "visualization",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        submod_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(f".{submod_name}", package=__name__)
        val = getattr(mod, name)
        # Cache in module globals so subsequent accesses don't call __getattr__
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "view_raster",
    "view_image",
    "plot_images",
    "plot_masks",
    "plot_batch",
    "calc_stats",
    "calc_iou",
    "calc_segmentation_metrics",
    "dict_to_rioxarray",
    "dict_to_image",
    "view_vector",
    "view_vector_interactive",
    "regularization",
    "hybrid_regularization",
    "adaptive_regularization",
    "install_package",
    "create_split_map",
    "download_file",
    "get_raster_info",
    "get_raster_stats",
    "print_raster_info",
    "get_raster_info_gdal",
    "get_vector_info",
    "print_vector_info",
    "get_vector_info_ogr",
    "analyze_vector_attributes",
    "visualize_vector_by_attribute",
    "clip_raster_by_bbox",
    "raster_to_vector",
    "raster_to_vector_batch",
    "vector_to_raster",
    "batch_vector_to_raster",
    "get_default_augmentation_transforms",
    "export_geotiff_tiles",
    "export_geotiff_tiles_batch",
    "display_training_tiles",
    "display_image_with_vector",
    "create_overview_image",
    "export_tiles_to_geojson",
    "export_training_data",
    "masks_to_vector",
    "read_vector",
    "read_raster",
    "temp_file_path",
    "region_groups",
    "add_geometric_properties",
    "orthogonalize",
    "inspect_pth_file",
    "try_common_architectures",
    "mosaic_geotiffs",
    "download_model_from_hf",
    "regularize",
    "vector_to_geojson",
    "geojson_to_coords",
    "coords_to_xy",
    "boxes_to_vector",
    "rowcol_to_xy",
    "bbox_to_xy",
    "geojson_to_xy",
    "write_colormap",
    "plot_performance_metrics",
    "get_device",
    "plot_prediction_comparison",
    "get_raster_resolution",
    "stack_bands",
    "empty_cache",
    "smooth_vector",
    "flipnslide_augmentation",
    "export_flipnslide_tiles",
    "calc_f1_score",
]
