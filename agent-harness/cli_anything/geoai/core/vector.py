"""Vector file inspection and operations.

Wraps geoai vector utility functions for CLI consumption.
All functions return JSON-serializable dicts.
"""

import os
from typing import Any, Dict, List, Optional

from cli_anything.geoai.utils.serialization import sanitize_json_value, serialize_info


def get_vector_info(path: str) -> Dict[str, Any]:
    """Get metadata for a vector file.

    Args:
        path: Path to the vector file.

    Returns:
        Dict with feature count, geometry types, CRS, bounds, columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Vector file not found: {path}")

    try:
        from geoai.utils import get_vector_info as _get_info

        info = _get_info(path)
        return serialize_info(info)
    except ImportError:
        return _get_vector_info_geopandas(path)


def rasterize_vector(
    path: str,
    template: str,
    output: str,
    attribute: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a vector file to a raster using a template for resolution/extent.

    Args:
        path: Input vector file path.
        template: Template raster for CRS, resolution, and extent.
        output: Output raster file path.
        attribute: Optional attribute column to burn into raster values.

    Returns:
        Dict with output path and dimensions.

    Raises:
        FileNotFoundError: If the input file or template does not exist.
    """
    path = os.path.abspath(path)
    template = os.path.abspath(template)
    output = os.path.abspath(output)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Vector file not found: {path}")
    if not os.path.isfile(template):
        raise FileNotFoundError(f"Template raster not found: {template}")

    from geoai.utils import vector_to_raster

    kwargs = {"reference_raster": template}
    if attribute is not None:
        kwargs["attribute_field"] = attribute

    vector_to_raster(path, output, **kwargs)

    import rasterio

    with rasterio.open(output) as src:
        return {
            "input": path,
            "template": template,
            "output": output,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs) if src.crs else None,
        }


def _get_vector_info_geopandas(path: str) -> Dict[str, Any]:
    """Fallback vector info using geopandas directly.

    Args:
        path: Path to vector file.

    Returns:
        Info dict.
    """
    import geopandas as gpd

    gdf = gpd.read_file(path)
    bounds = gdf.total_bounds if len(gdf) > 0 else [None, None, None, None]

    return sanitize_json_value(
        {
            "path": path,
            "feature_count": len(gdf),
            "geometry_types": (
                list(gdf.geometry.geom_type.dropna().unique()) if len(gdf) > 0 else []
            ),
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": {
                "minx": float(bounds[0]) if bounds[0] is not None else None,
                "miny": float(bounds[1]) if bounds[1] is not None else None,
                "maxx": float(bounds[2]) if bounds[2] is not None else None,
                "maxy": float(bounds[3]) if bounds[3] is not None else None,
            },
            "columns": list(gdf.columns),
            "file_size_bytes": os.path.getsize(path),
        }
    )
