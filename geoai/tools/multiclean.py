"""
MultiClean integration utilities for cleaning segmentation results.

This module provides functions to use MultiClean (https://github.com/DPIRD-DMA/MultiClean)
for post-processing segmentation masks and classification rasters. MultiClean performs
morphological operations to smooth edges, remove noise islands, and fill gaps.
"""

import os
from typing import Optional, List, Union, Tuple
import numpy as np

try:
    from multiclean import clean_array

    MULTICLEAN_AVAILABLE = True
except ImportError:
    MULTICLEAN_AVAILABLE = False

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def check_multiclean_available():
    """
    Check if multiclean is installed.

    Raises:
        ImportError: If multiclean is not installed.
    """
    if not MULTICLEAN_AVAILABLE:
        raise ImportError(
            "multiclean is not installed. "
            "Please install it with: pip install multiclean "
            "or: pip install geoai-py[extra]"
        )


def clean_segmentation_mask(
    mask: np.ndarray,
    class_values: Optional[Union[int, List[int]]] = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 8,
    max_workers: Optional[int] = None,
    fill_nan: bool = False,
) -> np.ndarray:
    """
    Clean a segmentation mask using MultiClean morphological operations.

    This function applies three cleaning operations:
    1. Edge smoothing - Uses morphological opening to reduce jagged boundaries
    2. Island removal - Eliminates small connected components (noise)
    3. Gap filling - Replaces invalid pixels with nearest valid class

    Args:
        mask (np.ndarray): 2D numpy array containing segmentation classes.
            Can be int or float. NaN values are treated as nodata.
        class_values (int, list of int, or None): Target class values to process.
            If None, auto-detects unique values from the mask. Defaults to None.
        smooth_edge_size (int): Kernel width in pixels for edge smoothing.
            Set to 0 to disable smoothing. Defaults to 2.
        min_island_size (int): Minimum area (in pixels) for connected components.
            Components with area strictly less than this are removed. Defaults to 100.
        connectivity (int): Connectivity for component detection. Use 4 or 8.
            8-connectivity considers diagonal neighbors. Defaults to 8.
        max_workers (int, optional): Thread pool size for parallel processing.
            If None, uses default threading. Defaults to None.
        fill_nan (bool): Whether to fill NaN pixels with nearest valid class.
            Defaults to False.

    Returns:
        np.ndarray: Cleaned 2D segmentation mask with same shape as input.

    Raises:
        ImportError: If multiclean is not installed.
        ValueError: If mask is not 2D or if connectivity is not 4 or 8.

    Example:
        >>> import numpy as np
        >>> from geoai.tools.multiclean import clean_segmentation_mask
        >>> mask = np.random.randint(0, 3, (512, 512))
        >>> cleaned = clean_segmentation_mask(
        ...     mask,
        ...     class_values=[0, 1, 2],
        ...     smooth_edge_size=2,
        ...     min_island_size=50
        ... )
    """
    check_multiclean_available()

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

    if connectivity not in [4, 8]:
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    # Apply MultiClean
    cleaned = clean_array(
        mask,
        class_values=class_values,
        smooth_edge_size=smooth_edge_size,
        min_island_size=min_island_size,
        connectivity=connectivity,
        max_workers=max_workers,
        fill_nan=fill_nan,
    )

    return cleaned


def clean_raster(
    input_path: str,
    output_path: str,
    class_values: Optional[Union[int, List[int]]] = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 8,
    max_workers: Optional[int] = None,
    fill_nan: bool = False,
    band: int = 1,
    nodata: Optional[float] = None,
) -> None:
    """
    Clean a classification raster (GeoTIFF) and save the result.

    Reads a GeoTIFF file, applies MultiClean morphological operations,
    and saves the cleaned result while preserving geospatial metadata
    (CRS, transform, nodata value).

    Args:
        input_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save cleaned GeoTIFF file.
        class_values (int, list of int, or None): Target class values to process.
            If None, auto-detects unique values. Defaults to None.
        smooth_edge_size (int): Kernel width in pixels for edge smoothing.
            Defaults to 2.
        min_island_size (int): Minimum area (in pixels) for components.
            Defaults to 100.
        connectivity (int): Connectivity for component detection (4 or 8).
            Defaults to 8.
        max_workers (int, optional): Thread pool size. Defaults to None.
        fill_nan (bool): Whether to fill NaN/nodata pixels. Defaults to False.
        band (int): Band index to read (1-indexed). Defaults to 1.
        nodata (float, optional): Nodata value to use. If None, uses value
            from input file. Defaults to None.

    Returns:
        None: Writes cleaned raster to output_path.

    Raises:
        ImportError: If multiclean or rasterio is not installed.
        FileNotFoundError: If input_path does not exist.

    Example:
        >>> from geoai.tools.multiclean import clean_raster
        >>> clean_raster(
        ...     "segmentation_raw.tif",
        ...     "segmentation_cleaned.tif",
        ...     class_values=[0, 1, 2],
        ...     smooth_edge_size=3,
        ...     min_island_size=50
        ... )
    """
    check_multiclean_available()

    if not RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for raster operations. "
            "Please install it with: pip install rasterio"
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read input raster
    with rasterio.open(input_path) as src:
        # Read the specified band
        mask = src.read(band)

        # Get metadata
        profile = src.profile.copy()

        # Handle nodata
        if nodata is None:
            nodata = src.nodata

        # Convert nodata to NaN if specified
        if nodata is not None:
            mask = mask.astype(np.float32)
            mask[mask == nodata] = np.nan

    # Clean the mask
    cleaned = clean_segmentation_mask(
        mask,
        class_values=class_values,
        smooth_edge_size=smooth_edge_size,
        min_island_size=min_island_size,
        connectivity=connectivity,
        max_workers=max_workers,
        fill_nan=fill_nan,
    )

    # Convert NaN back to nodata if needed
    if nodata is not None:
        # Convert any remaining NaN values back to nodata value
        if np.isnan(cleaned).any():
            cleaned = np.nan_to_num(cleaned, nan=nodata)

    # Update profile for output
    profile.update(
        dtype=cleaned.dtype,
        count=1,
        compress="lzw",
        nodata=nodata,
    )

    # Write cleaned raster
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and output_dir != os.path.abspath(os.sep):
        os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cleaned, 1)


def clean_raster_batch(
    input_paths: List[str],
    output_dir: str,
    class_values: Optional[Union[int, List[int]]] = None,
    smooth_edge_size: int = 2,
    min_island_size: int = 100,
    connectivity: int = 8,
    max_workers: Optional[int] = None,
    fill_nan: bool = False,
    band: int = 1,
    suffix: str = "_cleaned",
    verbose: bool = True,
) -> List[str]:
    """
    Clean multiple classification rasters in batch.

    Processes multiple GeoTIFF files with the same cleaning parameters
    and saves results to an output directory.

    Args:
        input_paths (list of str): List of paths to input GeoTIFF files.
        output_dir (str): Directory to save cleaned files.
        class_values (int, list of int, or None): Target class values.
            Defaults to None (auto-detect).
        smooth_edge_size (int): Kernel width for edge smoothing. Defaults to 2.
        min_island_size (int): Minimum component area. Defaults to 100.
        connectivity (int): Connectivity (4 or 8). Defaults to 8.
        max_workers (int, optional): Thread pool size. Defaults to None.
        fill_nan (bool): Whether to fill NaN pixels. Defaults to False.
        band (int): Band index to read (1-indexed). Defaults to 1.
        suffix (str): Suffix to add to output filenames. Defaults to "_cleaned".
        verbose (bool): Whether to print progress. Defaults to True.

    Returns:
        list of str: Paths to cleaned output files.

    Raises:
        ImportError: If multiclean or rasterio is not installed.

    Example:
        >>> from geoai.tools.multiclean import clean_raster_batch
        >>> input_files = ["mask1.tif", "mask2.tif", "mask3.tif"]
        >>> outputs = clean_raster_batch(
        ...     input_files,
        ...     output_dir="cleaned_masks",
        ...     min_island_size=50
        ... )
    """
    check_multiclean_available()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    for i, input_path in enumerate(input_paths):
        if verbose:
            print(f"Processing {i+1}/{len(input_paths)}: {input_path}")

        # Generate output filename
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        output_filename = f"{name}{suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Clean the raster
            clean_raster(
                input_path,
                output_path,
                class_values=class_values,
                smooth_edge_size=smooth_edge_size,
                min_island_size=min_island_size,
                connectivity=connectivity,
                max_workers=max_workers,
                fill_nan=fill_nan,
                band=band,
            )

            output_paths.append(output_path)

            if verbose:
                print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue

    return output_paths


def compare_masks(
    original: np.ndarray,
    cleaned: np.ndarray,
) -> Tuple[int, int, float]:
    """
    Compare original and cleaned masks to quantify changes.

    Args:
        original (np.ndarray): Original segmentation mask.
        cleaned (np.ndarray): Cleaned segmentation mask.

    Returns:
        tuple: (pixels_changed, total_pixels, change_percentage)
            - pixels_changed: Number of pixels that changed value
            - total_pixels: Total number of valid pixels
            - change_percentage: Percentage of pixels changed

    Example:
        >>> import numpy as np
        >>> from geoai.tools.multiclean import compare_masks
        >>> original = np.random.randint(0, 3, (512, 512))
        >>> cleaned = original.copy()
        >>> changed, total, pct = compare_masks(original, cleaned)
        >>> print(f"Changed: {pct:.2f}%")
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(original) | np.isnan(cleaned))

    # Count changed pixels
    pixels_changed = np.sum((original != cleaned) & valid_mask)
    total_pixels = np.sum(valid_mask)

    # Calculate percentage
    change_percentage = (pixels_changed / total_pixels * 100) if total_pixels > 0 else 0

    return pixels_changed, total_pixels, change_percentage
