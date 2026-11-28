"""Raster inspection and operations.

Wraps geoai raster utility functions for CLI consumption.
All functions return JSON-serializable dicts.
"""

import math
import os
from typing import Any, Dict, List, Optional

from cli_anything.geoai.utils.serialization import sanitize_json_value, serialize_info


def get_raster_info(path: str) -> Dict[str, Any]:
    """Get metadata for a raster file.

    Args:
        path: Path to the raster file.

    Returns:
        Dict with CRS, shape, bands, bounds, dtype, etc.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    try:
        from geoai.utils import get_raster_info as _get_info

        info = _get_info(path)
        return serialize_info(info)
    except ImportError:
        return _get_raster_info_rasterio(path)


def get_raster_stats(path: str, band: int = 1) -> Dict[str, Any]:
    """Compute statistics for a raster band.

    Args:
        path: Path to the raster file.
        band: Band number (1-indexed).

    Returns:
        Dict with min, max, mean, std, nodata count.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"Band {band} out of range (1-{src.count})")
        data = src.read(band)
        nodata = src.nodata

        mask = np.isfinite(data)
        if nodata is not None and not (
            isinstance(nodata, float) and math.isnan(nodata)
        ):
            mask &= data != nodata

        valid = data[mask]

        return sanitize_json_value(
            {
                "band": band,
                "min": float(np.min(valid)) if valid.size > 0 else None,
                "max": float(np.max(valid)) if valid.size > 0 else None,
                "mean": float(np.mean(valid)) if valid.size > 0 else None,
                "std": float(np.std(valid)) if valid.size > 0 else None,
                "valid_pixels": int(valid.size),
                "nodata_pixels": int(data.size - valid.size),
                "total_pixels": int(data.size),
                "nodata_value": float(nodata) if nodata is not None else None,
            }
        )


def tile_raster(
    path: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 0,
) -> Dict[str, Any]:
    """Split a raster into tiles.

    Args:
        path: Input raster path.
        output_dir: Output directory for tiles.
        tile_size: Tile width and height in pixels.
        overlap: Overlap between tiles in pixels.

    Returns:
        Dict with tile count and output directory.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    output_dir = os.path.abspath(output_dir)

    if overlap >= tile_size:
        raise ValueError(
            f"Overlap ({overlap}) must be less than tile_size ({tile_size})"
        )

    from geoai.utils import export_geotiff_tiles

    stride = tile_size - overlap
    export_geotiff_tiles(
        in_raster=path,
        out_folder=output_dir,
        tile_size=tile_size,
        stride=stride,
    )

    tile_files = []
    if os.path.isdir(output_dir):
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith((".tif", ".tiff")):
                    tile_files.append(os.path.join(root, f))

    return {
        "input": path,
        "output_dir": output_dir,
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(tile_files),
        "tiles": tile_files[:20],
    }


def vectorize_raster(
    path: str,
    output: str,
    simplify_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert a raster to vector polygons.

    Args:
        path: Input raster path.
        output: Output vector file path.
        simplify_tolerance: Optional geometry simplification tolerance.

    Returns:
        Dict with output path and feature count.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    path = os.path.abspath(path)
    output = os.path.abspath(output)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    from geoai.utils import raster_to_vector

    kwargs = {}
    if simplify_tolerance is not None:
        kwargs["simplify_tolerance"] = simplify_tolerance

    raster_to_vector(path, output, **kwargs)

    import geopandas as gpd

    gdf = gpd.read_file(output)
    return {
        "input": path,
        "output": output,
        "feature_count": len(gdf),
        "geometry_types": list(gdf.geometry.geom_type.unique()),
    }


def _get_raster_info_rasterio(path: str) -> Dict[str, Any]:
    """Fallback raster info using rasterio directly.

    Args:
        path: Path to raster file.

    Returns:
        Info dict.
    """
    import rasterio

    with rasterio.open(path) as src:
        return sanitize_json_value(
            {
                "path": path,
                "driver": src.driver,
                "dtype": str(src.dtypes[0]),
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs) if src.crs else None,
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "resolution": {"x": src.res[0], "y": src.res[1]},
                "nodata": src.nodata,
                "file_size_bytes": os.path.getsize(path),
            }
        )


def extract_band_subset(
    raster_path: str,
    band_indexes: List[int],
    temp_dir: Optional[str] = None,
) -> str:
    """Write a temporary GeoTIFF containing only ``band_indexes`` in order.

    Args:
        raster_path: Path to the source raster.
        band_indexes: Non-empty list of 0-indexed band indices to extract.
        temp_dir: Directory in which to create the temporary GeoTIFF.

    Returns:
        Absolute path to the newly created temporary GeoTIFF.

    Raises:
        ValueError: If ``band_indexes`` is empty, contains invalid types, or
            references a band outside the source raster's band count.
    """
    if not band_indexes:
        raise ValueError(
            "band_indexes must be a non-empty list of 0-based band indices"
        )

    validated: List[int] = []
    for idx in band_indexes:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError(f"Band index must be an integer, got {idx!r}")
        if idx < 0:
            raise ValueError(f"Band index must be non-negative, got {idx}")
        validated.append(idx)

    import tempfile
    import rasterio

    with rasterio.open(raster_path) as src:
        for idx in validated:
            if idx >= src.count:
                raise ValueError(
                    f"Band index {idx} out of range for raster with {src.count} band(s) "
                    f"(valid 0-based indices: 0-{src.count - 1})"
                )

        one_based = [idx + 1 for idx in validated]
        data = src.read(one_based)
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=len(one_based))

    fd, subset_path = tempfile.mkstemp(
        suffix=".tif",
        prefix="geoai_bands_",
        dir=temp_dir or os.path.dirname(os.path.abspath(raster_path)) or ".",
    )
    os.close(fd)
    try:
        with rasterio.open(subset_path, "w", **profile) as dst:
            dst.write(data)
    except Exception:
        if os.path.exists(subset_path):
            try:
                os.unlink(subset_path)
            except OSError:
                pass
        raise

    return os.path.abspath(subset_path)
