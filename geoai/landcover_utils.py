"""
Landcover Classification Utilities - Enhanced Tile Export Module

This module extends the base geoai functionality with specialized utilities
for discrete landcover classification. It provides enhanced tile generation
with background filtering capabilities to improve training efficiency.

Key Features:
- Enhanced tile filtering with configurable feature ratio thresholds
- Separate statistics tracking for different skip reasons
- Maintains full compatibility with base geoai workflow
- Optimized for discrete landcover classification tasks

Date: November 2025
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
from tqdm import tqdm


def export_landcover_tiles(
    in_raster: str,
    out_folder: str,
    in_class_data: Optional[Union[str, gpd.GeoDataFrame]] = None,
    tile_size: int = 256,
    stride: int = 128,
    class_value_field: str = "class",
    buffer_radius: float = 0,
    max_tiles: Optional[int] = None,
    quiet: bool = False,
    all_touched: bool = True,
    create_overview: bool = False,
    skip_empty_tiles: bool = False,
    min_feature_ratio: Union[bool, float] = False,
    metadata_format: str = "PASCAL_VOC",
) -> Dict[str, Any]:
    """
    Export GeoTIFF tiles optimized for landcover classification training.

    This function extends the base export_geotiff_tiles with enhanced filtering
    capabilities specifically designed for discrete landcover classification.
    It can filter out tiles dominated by background pixels to improve training
    data quality and reduce dataset size.

    Args:
        in_raster: Path to input raster (image to tile)
        out_folder: Output directory for tiles
        in_class_data: Path to vector mask or GeoDataFrame (optional for image-only export)
        tile_size: Size of output tiles in pixels (default: 256)
        stride: Stride for sliding window (default: 128)
        class_value_field: Field name containing class values (default: "class")
        buffer_radius: Buffer radius around features in pixels (default: 0)
        max_tiles: Maximum number of tiles to export (default: None)
        quiet: Suppress progress output (default: False)
        all_touched: Include pixels touched by geometry (default: True)
        create_overview: Create overview image showing tile locations (default: False)
        skip_empty_tiles: Skip tiles with no features (default: False)
        min_feature_ratio: Minimum ratio of non-background pixels required to keep tile
            - False: Disable ratio filtering (default)
            - 0.0-1.0: Minimum ratio threshold (e.g., 0.1 = 10% features required)
        metadata_format: Annotation format ("PASCAL_VOC" or "YOLO")

    Returns:
        Dictionary containing:
            - tiles_exported: Number of tiles successfully exported
            - tiles_skipped_empty: Number of completely empty tiles skipped
            - tiles_skipped_ratio: Number of tiles filtered by min_feature_ratio
            - output_dirs: Dictionary with paths to images and labels directories

    Examples:
        # Original behavior (no filtering)
        export_landcover_tiles(
            "input.tif",
            "output",
            "mask.shp",
            skip_empty_tiles=True
        )

        # Light filtering (keep tiles with ≥5% features)
        export_landcover_tiles(
            "input.tif",
            "output",
            "mask.shp",
            skip_empty_tiles=True,
            min_feature_ratio=0.05
        )

        # Moderate filtering (keep tiles with ≥15% features)
        export_landcover_tiles(
            "input.tif",
            "output",
            "mask.shp",
            skip_empty_tiles=True,
            min_feature_ratio=0.15
        )

    Note:
        This function is designed for discrete landcover classification where
        class 0 typically represents background/no data. The min_feature_ratio
        parameter counts non-zero pixels as "features".
    """

    # Validate min_feature_ratio parameter
    if min_feature_ratio is not False:
        if not isinstance(min_feature_ratio, (int, float)):
            warnings.warn(
                f"min_feature_ratio must be a number between 0.0 and 1.0, got {type(min_feature_ratio)}. "
                "Disabling ratio filtering."
            )
            min_feature_ratio = False
        elif not (0.0 <= min_feature_ratio <= 1.0):
            warnings.warn(
                f"min_feature_ratio must be between 0.0 and 1.0, got {min_feature_ratio}. "
                "Disabling ratio filtering."
            )
            min_feature_ratio = False

    # Create output directories
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    images_dir = out_folder / "images"
    labels_dir = out_folder / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    if metadata_format == "PASCAL_VOC":
        ann_dir = out_folder / "annotations"
        ann_dir.mkdir(exist_ok=True)

    # Initialize statistics
    stats = {
        "tiles_exported": 0,
        "tiles_skipped_empty": 0,
        "tiles_skipped_ratio": 0,
        "output_dirs": {"images": str(images_dir), "labels": str(labels_dir)},
    }

    # Open raster
    with rasterio.open(in_raster) as src:
        height, width = src.shape

        # Detect if in_class_data is raster or vector
        is_class_data_raster = False
        class_src = None
        gdf = None
        mask_array = None

        if in_class_data is not None:
            if isinstance(in_class_data, str):
                file_ext = Path(in_class_data).suffix.lower()
                if file_ext in [
                    ".tif",
                    ".tiff",
                    ".img",
                    ".jp2",
                    ".png",
                    ".bmp",
                    ".gif",
                ]:
                    try:
                        # Try to open as raster
                        class_src = rasterio.open(in_class_data)
                        is_class_data_raster = True

                        # Verify CRS match
                        if class_src.crs != src.crs:
                            if not quiet:
                                print(
                                    f"Warning: CRS mismatch between image ({src.crs}) and mask ({class_src.crs})"
                                )
                    except Exception as e:
                        is_class_data_raster = False
                        if not quiet:
                            print(f"Could not open as raster, trying vector: {e}")

                # If not raster or raster open failed, try vector
                if not is_class_data_raster:
                    gdf = gpd.read_file(in_class_data)

                    # Reproject if needed
                    if gdf.crs != src.crs:
                        if not quiet:
                            print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
                        gdf = gdf.to_crs(src.crs)

                    # Apply buffer if requested
                    if buffer_radius > 0:
                        gdf.geometry = gdf.geometry.buffer(buffer_radius)

                    # For vector data, rasterize entire mask up front for efficiency
                    shapes = [
                        (geom, value)
                        for geom, value in zip(gdf.geometry, gdf[class_value_field])
                    ]
                    mask_array = features.rasterize(
                        shapes,
                        out_shape=(height, width),
                        transform=src.transform,
                        all_touched=all_touched,
                        fill=0,
                        dtype=np.uint8,
                    )
            else:
                # Assume GeoDataFrame passed directly
                gdf = in_class_data

                # Reproject if needed
                if gdf.crs != src.crs:
                    if not quiet:
                        print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
                    gdf = gdf.to_crs(src.crs)

                # Apply buffer if requested
                if buffer_radius > 0:
                    gdf.geometry = gdf.geometry.buffer(buffer_radius)

                # Rasterize entire mask up front
                shapes = [
                    (geom, value)
                    for geom, value in zip(gdf.geometry, gdf[class_value_field])
                ]
                mask_array = features.rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=src.transform,
                    all_touched=all_touched,
                    fill=0,
                    dtype=np.uint8,
                )

        # Calculate tile positions
        tile_positions = []
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                tile_positions.append((x, y))

        if max_tiles:
            tile_positions = tile_positions[:max_tiles]

        # Process tiles
        pbar = tqdm(tile_positions, desc="Exporting tiles", disable=quiet)

        for tile_idx, (x, y) in enumerate(pbar):
            window = Window(x, y, tile_size, tile_size)

            # Read image tile
            image_tile = src.read(window=window)

            # Read mask tile based on data type
            mask_tile = None
            has_features = False

            if is_class_data_raster and class_src is not None:
                # For raster masks, read directly from the raster source
                # Get window transform and bounds
                window_transform = src.window_transform(window)
                minx = window_transform[2]
                maxy = window_transform[5]
                maxx = minx + tile_size * window_transform[0]
                miny = maxy + tile_size * window_transform[4]

                # Get corresponding window in class raster
                window_class = rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, class_src.transform
                )

                try:
                    # Read label data from raster
                    mask_tile = class_src.read(
                        1,
                        window=window_class,
                        boundless=True,
                        out_shape=(tile_size, tile_size),
                    )

                    # Check if tile has features
                    has_features = np.any(mask_tile > 0)
                except Exception as e:
                    if not quiet:
                        pbar.write(f"Error reading mask tile at ({x}, {y}): {e}")
                    continue

            elif mask_array is not None:
                # For vector masks (pre-rasterized)
                mask_tile = mask_array[y : y + tile_size, x : x + tile_size]
                has_features = np.any(mask_tile > 0)

            # Skip empty tiles if requested
            if skip_empty_tiles and not has_features:
                stats["tiles_skipped_empty"] += 1
                continue

            # Apply min_feature_ratio filtering if enabled
            if skip_empty_tiles and has_features and min_feature_ratio is not False:
                # Calculate ratio of non-background pixels
                total_pixels = mask_tile.size
                feature_pixels = np.sum(mask_tile > 0)
                feature_ratio = feature_pixels / total_pixels

                # Skip tile if below threshold
                if feature_ratio < min_feature_ratio:
                    stats["tiles_skipped_ratio"] += 1
                    continue

            # Save image tile
            tile_name = f"tile_{tile_idx:06d}.tif"
            image_path = images_dir / tile_name

            # Get transform for this tile
            tile_transform = src.window_transform(window)

            # Write image
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                height=tile_size,
                width=tile_size,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=tile_transform,
                compress="lzw",
            ) as dst:
                dst.write(image_tile)

            # Save mask tile if available
            if mask_tile is not None:
                mask_path = labels_dir / tile_name
                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype=np.uint8,
                    crs=src.crs,
                    transform=tile_transform,
                    compress="lzw",
                ) as dst:
                    dst.write(mask_tile, 1)

            stats["tiles_exported"] += 1

            # Update progress bar description with selection count
            if not quiet:
                pbar.set_description(
                    f"Exporting tiles ({stats['tiles_exported']}/{tile_idx + 1})"
                )

    # Close raster class source if opened
    if class_src is not None:
        class_src.close()

    # Print summary
    if not quiet:
        print(f"\n{'='*60}")
        print("TILE EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Tiles exported: {stats['tiles_exported']}/{len(tile_positions)}")
        if skip_empty_tiles:
            print(f"Tiles skipped (empty): {stats['tiles_skipped_empty']}")
        if min_feature_ratio is not False:
            print(
                f"Tiles skipped (low feature ratio < {min_feature_ratio}): {stats['tiles_skipped_ratio']}"
            )
        print(f"\nOutput directories:")
        print(f"  Images: {stats['output_dirs']['images']}")
        print(f"  Labels: {stats['output_dirs']['labels']}")
        print(f"{'='*60}\n")

    return stats
