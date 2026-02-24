"""
Time-series analysis utilities for multi-temporal satellite imagery.

This module provides functions for temporal compositing, spectral index
time-series computation, bi-temporal change detection, and temporal
statistics on aligned raster stacks. It works with GeoTIFF files on disk
using numpy and rasterio, and optionally integrates with OmniCloudMask
for cloud-aware compositing.

Supports Sentinel-2, Landsat, NAIP, and any other GeoTIFF-based imagery
with consistent spatial reference and resolution.
"""

import os
import re
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# Supported compositing methods
COMPOSITE_METHODS = ("median", "mean", "min", "max", "medoid")

# Supported spectral indices
SPECTRAL_INDICES = {
    "NDVI": {"formula": "(NIR - RED) / (NIR + RED)", "bands": ("nir", "red")},
    "NDWI": {"formula": "(GREEN - NIR) / (GREEN + NIR)", "bands": ("green", "nir")},
    "NDBI": {
        "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "bands": ("swir1", "nir"),
    },
    "EVI": {
        "formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
        "bands": ("nir", "red", "blue"),
    },
    "SAVI": {
        "formula": "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
        "bands": ("nir", "red"),
    },
    "MNDWI": {
        "formula": "(GREEN - SWIR1) / (GREEN + SWIR1)",
        "bands": ("green", "swir1"),
    },
}

# Supported change detection methods
CHANGE_METHODS = ("difference", "ratio", "normalized_difference")

# Supported temporal statistics
TEMPORAL_STATISTICS = ("mean", "std", "min", "max", "range", "count", "median")

# Common satellite filename date patterns
SENTINEL2_DATE_PATTERN = r"(\d{8})T\d{6}"
LANDSAT_DATE_PATTERN = r"_(\d{8})_"
GENERIC_DATE_PATTERN = r"(\d{4}[-_]?\d{2}[-_]?\d{2})"


def check_rasterio_available():
    """Check if rasterio is installed.

    Raises:
        ImportError: If rasterio is not installed.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for time-series raster operations. "
            "Please install it with: pip install rasterio"
        )


def extract_dates_from_filenames(
    input_paths: List[str],
    date_pattern: Optional[str] = None,
    date_format: str = "%Y%m%d",
) -> List[Optional[datetime]]:
    """Extract dates from raster filenames using a regex pattern.

    Parses the filename (not the full path) of each input file to extract
    a date string, then converts it to a datetime object.

    Args:
        input_paths (list of str): Paths to raster files.
        date_pattern (str, optional): Regex pattern with one capture group
            that matches the date portion of the filename. If None, tries
            common satellite naming conventions in order:
            1. Sentinel-2: ``(\\d{8})T\\d{6}``
            2. Landsat: ``_(\\d{8})_``
            3. Generic: ``(\\d{4}[-_]?\\d{2}[-_]?\\d{2})``
            Defaults to None.
        date_format (str): strptime format string for parsing the captured
            date string. Defaults to '%Y%m%d'. Common alternatives:
            '%Y-%m-%d' for dates like '2023-01-15',
            '%Y_%m_%d' for dates like '2023_01_15'.

    Returns:
        list of datetime or None: Parsed datetime objects for each file.
            Returns None for files where no date could be extracted.

    Raises:
        ValueError: If date_pattern is provided but contains no capture
            group.

    Example:
        >>> from geoai.tools.timeseries import extract_dates_from_filenames
        >>> files = [
        ...     "S2A_MSIL2A_20230115T101301_N0509.tif",
        ...     "S2A_MSIL2A_20230315T101301_N0509.tif",
        ... ]
        >>> dates = extract_dates_from_filenames(files)
        >>> print(dates[0].strftime('%Y-%m-%d'))
        '2023-01-15'
    """
    if date_pattern is not None:
        compiled = re.compile(date_pattern)
        if compiled.groups < 1:
            raise ValueError(
                f"date_pattern must contain at least one capture group, "
                f"got: '{date_pattern}'"
            )

    auto_patterns = [
        (SENTINEL2_DATE_PATTERN, "%Y%m%d"),
        (LANDSAT_DATE_PATTERN, "%Y%m%d"),
        (GENERIC_DATE_PATTERN, None),  # format determined dynamically
    ]

    results = []
    for path in input_paths:
        basename = os.path.basename(path)
        parsed_date = None

        if date_pattern is not None:
            match = re.search(date_pattern, basename)
            if match:
                date_str = match.group(1)
                try:
                    parsed_date = datetime.strptime(date_str, date_format)
                except ValueError:
                    parsed_date = None
        else:
            # Try each auto-detection pattern
            for pattern, auto_fmt in auto_patterns:
                match = re.search(pattern, basename)
                if match:
                    date_str = match.group(1)
                    # For generic pattern, strip separators and use %Y%m%d
                    if auto_fmt is None:
                        cleaned = date_str.replace("-", "").replace("_", "")
                        fmt = "%Y%m%d"
                    else:
                        cleaned = date_str
                        fmt = auto_fmt
                    try:
                        parsed_date = datetime.strptime(cleaned, fmt)
                        break
                    except ValueError:
                        continue

        results.append(parsed_date)

    return results


def sort_by_date(
    input_paths: List[str],
    dates: Optional[List[Optional[datetime]]] = None,
    date_pattern: Optional[str] = None,
    date_format: str = "%Y%m%d",
) -> Tuple[List[str], List[datetime]]:
    """Sort file paths chronologically by date.

    If dates are not provided, extracts them from filenames using
    ``extract_dates_from_filenames()``.

    Args:
        input_paths (list of str): Paths to raster files.
        dates (list of datetime, optional): Pre-computed dates for each
            file. Must be the same length as input_paths. Files with
            None dates are placed at the end. Defaults to None.
        date_pattern (str, optional): Regex pattern for date extraction
            (passed to extract_dates_from_filenames if dates is None).
            Defaults to None.
        date_format (str): Date format string (passed to
            extract_dates_from_filenames if dates is None).
            Defaults to '%Y%m%d'.

    Returns:
        tuple: A tuple of (sorted_paths, sorted_dates) where:
            - sorted_paths (list of str): File paths sorted chronologically.
            - sorted_dates (list of datetime): Corresponding dates, sorted.

    Raises:
        ValueError: If dates is provided but has different length
            than input_paths.
        ValueError: If no dates could be extracted from any filename.

    Example:
        >>> from geoai.tools.timeseries import sort_by_date
        >>> files = ["scene_20230601.tif", "scene_20230101.tif", "scene_20230315.tif"]
        >>> sorted_files, sorted_dates = sort_by_date(files)
        >>> print([os.path.basename(f) for f in sorted_files])
        ['scene_20230101.tif', 'scene_20230315.tif', 'scene_20230601.tif']
    """
    if dates is None:
        dates = extract_dates_from_filenames(input_paths, date_pattern, date_format)
    elif len(dates) != len(input_paths):
        raise ValueError(
            f"Length of dates ({len(dates)}) must match length of "
            f"input_paths ({len(input_paths)})"
        )

    # Separate valid and None dates
    valid_pairs = []
    none_pairs = []
    for path, date in zip(input_paths, dates):
        if date is not None:
            valid_pairs.append((path, date))
        else:
            none_pairs.append((path, date))

    if not valid_pairs:
        raise ValueError(
            "No dates could be extracted from any filename. "
            "Provide dates explicitly or check the date_pattern."
        )

    # Sort valid pairs by date
    valid_pairs.sort(key=lambda x: x[1])

    # Combine: valid first, then None-dated at the end
    all_pairs = valid_pairs + none_pairs
    sorted_paths = [p for p, _ in all_pairs]
    sorted_dates = [d for _, d in all_pairs]

    return sorted_paths, sorted_dates


def validate_temporal_stack(
    input_paths: List[str],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Validate that raster files are compatible for temporal stacking.

    Checks that all rasters share the same CRS, pixel resolution, dimensions
    (width and height), and spatial bounds. Returns metadata from the
    reference (first) raster if validation passes.

    Args:
        input_paths (list of str): Paths to GeoTIFF files to validate.
        tolerance (float): Tolerance for floating-point comparison of
            resolution and bounds. Defaults to 1e-6.

    Returns:
        dict: Reference metadata dictionary containing:
            - crs: The coordinate reference system (rasterio.crs.CRS).
            - width: Raster width in pixels (int).
            - height: Raster height in pixels (int).
            - transform: Affine transform (rasterio.transform.Affine).
            - bounds: Bounding box as (left, bottom, right, top).
            - resolution: Pixel size as (x_res, y_res).
            - count: Number of bands in the reference raster (int).
            - dtype: Data type of the reference raster (str).
            - num_files: Number of files validated (int).

    Raises:
        ImportError: If rasterio is not installed.
        ValueError: If fewer than 2 files are provided.
        ValueError: If any file does not exist.
        ValueError: If CRS, resolution, dimensions, or bounds differ
            between any file and the reference.

    Example:
        >>> from geoai.tools.timeseries import validate_temporal_stack
        >>> info = validate_temporal_stack([
        ...     "scene_2023_01.tif",
        ...     "scene_2023_06.tif",
        ...     "scene_2023_09.tif",
        ... ])
        >>> print(f"Stack: {info['num_files']} files, "
        ...       f"{info['width']}x{info['height']} pixels")
    """
    check_rasterio_available()

    if len(input_paths) < 2:
        raise ValueError(
            f"At least 2 files are required for temporal stacking, "
            f"got {len(input_paths)}"
        )

    for path in input_paths:
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")

    # Read reference metadata from the first file
    with rasterio.open(input_paths[0]) as ref:
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_transform = ref.transform
        ref_bounds = ref.bounds
        ref_res = ref.res
        ref_count = ref.count
        ref_dtype = ref.dtypes[0]

    # Validate all other files against the reference
    for path in input_paths[1:]:
        with rasterio.open(path) as src:
            # Check CRS
            if src.crs != ref_crs:
                raise ValueError(
                    f"CRS mismatch: '{os.path.basename(path)}' has "
                    f"{src.crs}, expected {ref_crs}"
                )

            # Check dimensions
            if src.width != ref_width or src.height != ref_height:
                raise ValueError(
                    f"Dimension mismatch: '{os.path.basename(path)}' has "
                    f"{src.width}x{src.height}, expected "
                    f"{ref_width}x{ref_height}"
                )

            # Check resolution
            if (
                abs(src.res[0] - ref_res[0]) > tolerance
                or abs(src.res[1] - ref_res[1]) > tolerance
            ):
                raise ValueError(
                    f"Resolution mismatch: '{os.path.basename(path)}' has "
                    f"{src.res}, expected {ref_res}"
                )

            # Check bounds
            src_bounds = src.bounds
            if (
                abs(src_bounds.left - ref_bounds.left) > tolerance
                or abs(src_bounds.bottom - ref_bounds.bottom) > tolerance
                or abs(src_bounds.right - ref_bounds.right) > tolerance
                or abs(src_bounds.top - ref_bounds.top) > tolerance
            ):
                raise ValueError(
                    f"Bounds mismatch: '{os.path.basename(path)}' has "
                    f"bounds {src_bounds}, expected {ref_bounds}"
                )

    return {
        "crs": ref_crs,
        "width": ref_width,
        "height": ref_height,
        "transform": ref_transform,
        "bounds": ref_bounds,
        "resolution": ref_res,
        "count": ref_count,
        "dtype": ref_dtype,
        "num_files": len(input_paths),
    }


def create_temporal_composite(
    input_paths: List[str],
    output_path: str,
    method: str = "median",
    cloud_masks: Optional[List[str]] = None,
    cloud_clear_value: int = 0,
    bands: Optional[List[int]] = None,
    nodata: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Create a pixel-wise temporal composite from multiple aligned rasters.

    Stacks multiple co-registered rasters along a temporal axis and reduces
    each pixel to a single value using the specified compositing method.
    Optionally excludes cloudy pixels using pre-computed cloud masks.

    Args:
        input_paths (list of str): Paths to input GeoTIFF files. All files
            must share the same CRS, resolution, and spatial extent.
        output_path (str): Path to save the output composite GeoTIFF.
        method (str): Compositing method. One of:
            - 'median': Pixel-wise median (robust to outliers).
            - 'mean': Pixel-wise mean.
            - 'min': Pixel-wise minimum (useful for shadow-free composites).
            - 'max': Pixel-wise maximum (useful for peak NDVI).
            - 'medoid': Selects the observation closest to the median across
              all bands (preserves spectral consistency).
            Defaults to 'median'.
        cloud_masks (list of str, optional): Paths to cloud mask GeoTIFFs
            corresponding to each input file. Must be same length as
            input_paths. Pixels where mask != cloud_clear_value are excluded.
            Defaults to None (no cloud masking).
        cloud_clear_value (int): Value in cloud masks that indicates clear
            sky. Defaults to 0 (matching OmniCloudMask CLEAR constant).
        bands (list of int, optional): 1-based band indices to composite.
            If None, all bands from the first raster are used.
            Defaults to None.
        nodata (float, optional): Nodata value. Pixels with this value in
            any input are excluded from compositing. If None, uses the
            nodata value from the first input file. Defaults to None.
        verbose (bool): Print progress messages. Defaults to True.

    Returns:
        str: Path to the output composite GeoTIFF file.

    Raises:
        ImportError: If rasterio is not installed.
        ValueError: If method is not one of the supported methods.
        ValueError: If cloud_masks is provided but its length differs
            from input_paths.
        ValueError: If input rasters are not spatially aligned.
        FileNotFoundError: If any input file does not exist.

    Example:
        >>> from geoai.tools.timeseries import create_temporal_composite
        >>> scenes = ["scene_jan.tif", "scene_mar.tif", "scene_jun.tif"]
        >>> create_temporal_composite(
        ...     scenes,
        ...     "median_composite.tif",
        ...     method="median",
        ... )
        'median_composite.tif'
    """
    check_rasterio_available()

    if method not in COMPOSITE_METHODS:
        raise ValueError(
            f"Unknown compositing method '{method}'. "
            f"Supported methods: {COMPOSITE_METHODS}"
        )

    if cloud_masks is not None and len(cloud_masks) != len(input_paths):
        raise ValueError(
            f"Length of cloud_masks ({len(cloud_masks)}) must match "
            f"length of input_paths ({len(input_paths)})"
        )

    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    if cloud_masks is not None:
        for path in cloud_masks:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cloud mask file not found: {path}")

    # Validate spatial alignment
    ref_info = validate_temporal_stack(input_paths)
    height = ref_info["height"]
    width = ref_info["width"]

    # Determine bands to read
    if bands is None:
        bands = list(range(1, ref_info["count"] + 1))

    num_bands = len(bands)
    num_scenes = len(input_paths)

    # Determine nodata value
    if nodata is None:
        with rasterio.open(input_paths[0]) as src:
            nodata = src.nodata

    if verbose:
        print(
            f"Creating {method} composite from {num_scenes} scenes, "
            f"{num_bands} bands, {width}x{height} pixels"
        )

    # Read all scenes into a 4D array (num_scenes, num_bands, H, W)
    stack = np.empty((num_scenes, num_bands, height, width), dtype=np.float32)

    for i, path in enumerate(input_paths):
        if verbose:
            print(f"  Reading {i + 1}/{num_scenes}: {os.path.basename(path)}")
        with rasterio.open(path) as src:
            for j, band_idx in enumerate(bands):
                stack[i, j] = src.read(band_idx).astype(np.float32)

    # Build valid-pixel mask (num_scenes, H, W)
    valid = np.ones((num_scenes, height, width), dtype=bool)

    # Mark nodata pixels as invalid
    if nodata is not None:
        for i in range(num_scenes):
            for j in range(num_bands):
                valid[i] &= stack[i, j] != nodata

    # Apply cloud masks
    if cloud_masks is not None:
        for i, mask_path in enumerate(cloud_masks):
            with rasterio.open(mask_path) as src:
                cloud_mask = src.read(1)
            valid[i] &= cloud_mask == cloud_clear_value

    # Set invalid pixels to NaN
    for i in range(num_scenes):
        for j in range(num_bands):
            stack[i, j][~valid[i]] = np.nan

    if verbose:
        total_pixels = num_scenes * height * width
        valid_pixels = valid.sum()
        print(
            f"  Valid observations: {valid_pixels}/{total_pixels} "
            f"({valid_pixels / total_pixels * 100:.1f}%)"
        )

    # Compute composite
    if method == "median":
        composite = np.nanmedian(stack, axis=0)
    elif method == "mean":
        composite = np.nanmean(stack, axis=0)
    elif method == "min":
        composite = np.nanmin(stack, axis=0)
    elif method == "max":
        composite = np.nanmax(stack, axis=0)
    elif method == "medoid":
        composite = _compute_medoid(stack)

    # Replace remaining NaN with nodata value
    output_nodata = nodata if nodata is not None else 0
    nan_mask = np.isnan(composite)
    composite[nan_mask] = output_nodata

    # Write output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_paths[0]) as ref:
        profile = ref.profile.copy()

    profile.update(
        dtype="float32",
        count=num_bands,
        compress="lzw",
        nodata=output_nodata,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for j in range(num_bands):
            dst.write(composite[j], j + 1)

    if verbose:
        print(f"  Composite saved to: {output_path}")

    return output_path


def _compute_medoid(stack: np.ndarray) -> np.ndarray:
    """Compute the medoid composite from a temporal stack.

    For each pixel location, selects the temporal observation whose spectral
    vector is closest (Euclidean distance) to the per-pixel median vector.
    This preserves spectral consistency unlike per-band median.

    Args:
        stack (np.ndarray): 4D array of shape (num_scenes, num_bands, H, W).

    Returns:
        np.ndarray: Composite of shape (num_bands, H, W).
    """
    num_scenes, num_bands, height, width = stack.shape

    # Compute the per-pixel median across time
    median_vals = np.nanmedian(stack, axis=0)  # (num_bands, H, W)

    # Compute Euclidean distance from each observation to the median
    # diff shape: (num_scenes, num_bands, H, W)
    diff = stack - median_vals[np.newaxis, :, :, :]
    # distances shape: (num_scenes, H, W)
    distances = np.sqrt(np.nansum(diff**2, axis=1))

    # For pixels where an observation is all-NaN, set distance to inf
    all_nan = np.all(np.isnan(stack), axis=1)  # (num_scenes, H, W)
    distances[all_nan] = np.inf

    # Find the index of the closest observation per pixel
    best_idx = np.argmin(distances, axis=0)  # (H, W)

    # Gather the medoid values using advanced indexing
    composite = np.empty((num_bands, height, width), dtype=np.float32)
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    for j in range(num_bands):
        composite[j] = stack[best_idx, j, rows, cols]

    return composite


def create_cloud_free_composite(
    input_paths: List[str],
    output_path: str,
    red_band: int = 1,
    green_band: int = 2,
    nir_band: int = 3,
    method: str = "median",
    bands: Optional[List[int]] = None,
    nodata: Optional[float] = None,
    include_thin_clouds: bool = False,
    include_shadows: bool = False,
    cloud_mask_dir: Optional[str] = None,
    batch_size: int = 1,
    inference_device: str = "cpu",
    inference_dtype: str = "fp32",
    patch_size: int = 1000,
    model_version: int = 3,
    verbose: bool = True,
) -> str:
    """Create a cloud-free temporal composite using OmniCloudMask.

    This is a convenience function that:
    1. Generates cloud masks for each input scene using OmniCloudMask.
    2. Creates binary clear-sky masks from the cloud predictions.
    3. Creates a temporal composite excluding cloudy pixels.

    Requires the omnicloudmask package to be installed.

    Args:
        input_paths (list of str): Paths to input GeoTIFF files.
        output_path (str): Path to save the cloud-free composite GeoTIFF.
        red_band (int): Band index for Red (1-indexed). Defaults to 1.
        green_band (int): Band index for Green (1-indexed). Defaults to 2.
        nir_band (int): Band index for NIR (1-indexed). Defaults to 3.
        method (str): Compositing method ('median', 'mean', 'min', 'max',
            'medoid'). Defaults to 'median'.
        bands (list of int, optional): Bands to include in the composite.
            If None, all bands are used. Defaults to None.
        nodata (float, optional): Nodata value. Defaults to None.
        include_thin_clouds (bool): If True, thin clouds are treated as
            clear for masking purposes. Defaults to False.
        include_shadows (bool): If True, cloud shadows are treated as
            clear for masking purposes. Defaults to False.
        cloud_mask_dir (str, optional): Directory to save intermediate
            cloud masks. If None, masks are saved to a temporary directory
            and cleaned up after compositing. Defaults to None.
        batch_size (int): OmniCloudMask batch size. Defaults to 1.
        inference_device (str): Device for inference. Defaults to 'cpu'.
        inference_dtype (str): Inference dtype. Defaults to 'fp32'.
        patch_size (int): OmniCloudMask patch size. Defaults to 1000.
        model_version (int): OmniCloudMask model version. Defaults to 3.
        verbose (bool): Print progress messages. Defaults to True.

    Returns:
        str: Path to the output cloud-free composite GeoTIFF.

    Raises:
        ImportError: If omnicloudmask is not installed.
        ImportError: If rasterio is not installed.
        ValueError: If input rasters are not spatially aligned.

    Example:
        >>> from geoai.tools.timeseries import create_cloud_free_composite
        >>> scenes = ["S2_20230101.tif", "S2_20230315.tif", "S2_20230601.tif"]
        >>> create_cloud_free_composite(
        ...     scenes,
        ...     "cloud_free_composite.tif",
        ...     red_band=4,
        ...     green_band=3,
        ...     nir_band=8,
        ...     method="median",
        ... )
        'cloud_free_composite.tif'
    """
    from .cloudmask import (
        check_omnicloudmask_available,
        create_cloud_free_mask,
        predict_cloud_mask,
    )

    check_omnicloudmask_available()
    check_rasterio_available()

    # Determine where to store cloud masks
    use_temp_dir = cloud_mask_dir is None
    if use_temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="geoai_cloudmask_")
        mask_dir = temp_dir
    else:
        mask_dir = cloud_mask_dir
        os.makedirs(mask_dir, exist_ok=True)

    cloud_mask_paths = []

    try:
        if verbose:
            print(f"Generating cloud masks for {len(input_paths)} scenes...")

        for i, path in enumerate(input_paths):
            if verbose:
                print(
                    f"  Cloud masking {i + 1}/{len(input_paths)}: "
                    f"{os.path.basename(path)}"
                )

            # Read the RGB+NIR bands for cloud detection
            with rasterio.open(path) as src:
                red = src.read(red_band).astype(np.float32)
                green = src.read(green_band).astype(np.float32)
                nir = src.read(nir_band).astype(np.float32)
                profile = src.profile.copy()

            # Stack into (3, H, W) for OmniCloudMask
            image = np.stack([red, green, nir], axis=0)

            # Predict cloud mask
            cloud_pred = predict_cloud_mask(
                image,
                batch_size=batch_size,
                inference_device=inference_device,
                inference_dtype=inference_dtype,
                patch_size=patch_size,
                model_version=model_version,
            )

            # Convert to binary clear-sky mask (1 = usable, 0 = not usable)
            clear_mask = create_cloud_free_mask(
                cloud_pred,
                include_thin_clouds=include_thin_clouds,
                include_shadows=include_shadows,
            )

            # Save the binary mask as a GeoTIFF
            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)
            mask_path = os.path.join(mask_dir, f"{name}_clearmask{ext}")

            mask_profile = profile.copy()
            mask_profile.update(
                dtype="uint8",
                count=1,
                compress="lzw",
                nodata=None,
            )

            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(clear_mask, 1)

            cloud_mask_paths.append(mask_path)

        if verbose:
            print("Creating cloud-free composite...")

        # Create composite using clear-sky masks
        # The clear mask has 1 = usable, so cloud_clear_value=1
        result = create_temporal_composite(
            input_paths=input_paths,
            output_path=output_path,
            method=method,
            cloud_masks=cloud_mask_paths,
            cloud_clear_value=1,
            bands=bands,
            nodata=nodata,
            verbose=verbose,
        )

        return result

    finally:
        # Clean up temporary cloud mask files
        if use_temp_dir:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def calculate_spectral_index_timeseries(
    input_paths: List[str],
    output_path: str,
    index_type: str = "NDVI",
    red_band: int = 1,
    green_band: int = 2,
    blue_band: int = 3,
    nir_band: int = 4,
    swir1_band: Optional[int] = None,
    nodata: Optional[float] = None,
    scale_factor: float = 1.0,
    verbose: bool = True,
) -> str:
    """Calculate a spectral index across a temporal stack of rasters.

    Computes the specified spectral index for each input scene and writes
    the results as a multi-band GeoTIFF where each band corresponds to one
    timestep in the time series.

    Args:
        input_paths (list of str): Paths to input GeoTIFF files, each
            containing multi-spectral bands. All must be spatially aligned.
        output_path (str): Path to save the output multi-band GeoTIFF.
            Each band in the output represents one timestep.
        index_type (str): Spectral index to compute. One of:
            'NDVI', 'NDWI', 'NDBI', 'EVI', 'SAVI', 'MNDWI'.
            Defaults to 'NDVI'.
        red_band (int): Band index for Red (1-indexed). Defaults to 1.
        green_band (int): Band index for Green (1-indexed). Defaults to 2.
        blue_band (int): Band index for Blue (1-indexed). Defaults to 3.
        nir_band (int): Band index for NIR (1-indexed). Defaults to 4.
        swir1_band (int, optional): Band index for SWIR1 (1-indexed).
            Required for NDBI and MNDWI. Defaults to None.
        nodata (float, optional): Nodata value in the input. Pixels with
            this value are set to NaN in the output. If None, uses the
            nodata value from the first input file. Defaults to None.
        scale_factor (float): Factor to divide raw pixel values by to
            convert to reflectance (e.g., 10000.0 for Sentinel-2 L2A).
            Defaults to 1.0 (no scaling).
        verbose (bool): Print progress. Defaults to True.

    Returns:
        str: Path to the output spectral index time-series GeoTIFF.

    Raises:
        ImportError: If rasterio is not installed.
        ValueError: If index_type is not supported.
        ValueError: If required bands for the index are not provided.
        ValueError: If input rasters are not spatially aligned.

    Example:
        >>> from geoai.tools.timeseries import calculate_spectral_index_timeseries
        >>> scenes = ["S2_20230101.tif", "S2_20230315.tif", "S2_20230601.tif"]
        >>> calculate_spectral_index_timeseries(
        ...     scenes,
        ...     "ndvi_timeseries.tif",
        ...     index_type="NDVI",
        ...     red_band=4,
        ...     nir_band=8,
        ...     scale_factor=10000.0,
        ... )
        'ndvi_timeseries.tif'
    """
    check_rasterio_available()

    index_upper = index_type.upper()
    if index_upper not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unknown spectral index '{index_type}'. "
            f"Supported indices: {list(SPECTRAL_INDICES.keys())}"
        )

    required_bands = SPECTRAL_INDICES[index_upper]["bands"]
    if "swir1" in required_bands and swir1_band is None:
        raise ValueError(
            f"swir1_band is required for {index_upper} but was not provided."
        )

    # Validate spatial alignment
    ref_info = validate_temporal_stack(input_paths)
    height = ref_info["height"]
    width = ref_info["width"]
    num_scenes = len(input_paths)

    # Determine nodata
    if nodata is None:
        with rasterio.open(input_paths[0]) as src:
            nodata = src.nodata

    if verbose:
        print(
            f"Computing {index_upper} time-series for {num_scenes} scenes, "
            f"{width}x{height} pixels"
        )

    # Map band names to band indices
    band_map = {
        "red": red_band,
        "green": green_band,
        "blue": blue_band,
        "nir": nir_band,
        "swir1": swir1_band,
    }

    eps = 1e-10

    # Compute index for each scene
    index_stack = np.empty((num_scenes, height, width), dtype=np.float32)

    for i, path in enumerate(input_paths):
        if verbose:
            print(f"  Processing {i + 1}/{num_scenes}: {os.path.basename(path)}")

        with rasterio.open(path) as src:
            # Read required bands
            band_data = {}
            for band_name in required_bands:
                bidx = band_map[band_name]
                data = src.read(bidx).astype(np.float32)
                # Apply scale factor
                if scale_factor != 1.0:
                    data = data / scale_factor
                # Mask nodata
                if nodata is not None:
                    data[src.read(bidx) == nodata] = np.nan
                band_data[band_name] = data

        # Compute the spectral index
        if index_upper == "NDVI":
            nir = band_data["nir"]
            red = band_data["red"]
            index_val = (nir - red) / (nir + red + eps)
        elif index_upper == "NDWI":
            green_arr = band_data["green"]
            nir = band_data["nir"]
            index_val = (green_arr - nir) / (green_arr + nir + eps)
        elif index_upper == "NDBI":
            swir1 = band_data["swir1"]
            nir = band_data["nir"]
            index_val = (swir1 - nir) / (swir1 + nir + eps)
        elif index_upper == "EVI":
            nir = band_data["nir"]
            red = band_data["red"]
            blue = band_data["blue"]
            index_val = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
        elif index_upper == "SAVI":
            nir = band_data["nir"]
            red = band_data["red"]
            index_val = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
        elif index_upper == "MNDWI":
            green_arr = band_data["green"]
            swir1 = band_data["swir1"]
            index_val = (green_arr - swir1) / (green_arr + swir1 + eps)

        index_stack[i] = index_val

    # Write output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_paths[0]) as ref:
        profile = ref.profile.copy()

    profile.update(
        dtype="float32",
        count=num_scenes,
        compress="lzw",
        nodata=np.nan,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(num_scenes):
            dst.write(index_stack[i], i + 1)

    if verbose:
        print(f"  {index_upper} time-series saved to: {output_path}")

    return output_path


def detect_change(
    image1_path: str,
    image2_path: str,
    output_path: str,
    method: str = "difference",
    threshold: Optional[float] = None,
    bands: Optional[List[int]] = None,
    nodata: Optional[float] = None,
    absolute: bool = True,
    verbose: bool = True,
) -> str:
    """Detect changes between two aligned raster images.

    Computes a change map between two co-registered rasters using simple
    image algebra methods. The output can be a continuous change magnitude
    map or a binary change/no-change mask (if threshold is provided).

    Args:
        image1_path (str): Path to the first (earlier) GeoTIFF image.
        image2_path (str): Path to the second (later) GeoTIFF image.
        output_path (str): Path to save the change detection GeoTIFF.
        method (str): Change detection method. One of:
            - 'difference': image2 - image1 (or absolute difference).
            - 'ratio': image2 / image1.
            - 'normalized_difference': (image2 - image1) / (image2 + image1).
            Defaults to 'difference'.
        threshold (float, optional): If provided, the continuous change map
            is thresholded to produce a binary mask where 1 = change and
            0 = no change. The threshold is applied to the absolute value
            of the change metric. Defaults to None (continuous output).
        bands (list of int, optional): 1-based band indices to use for
            change detection. If multiple bands are specified, change
            magnitude is computed as the Euclidean norm across bands.
            If None, uses band 1 only. Defaults to None.
        nodata (float, optional): Nodata value. Defaults to None.
        absolute (bool): If True, output the absolute value of the change
            metric (for 'difference' and 'normalized_difference' methods).
            Defaults to True.
        verbose (bool): Print progress. Defaults to True.

    Returns:
        str: Path to the output change detection GeoTIFF.

    Raises:
        ImportError: If rasterio is not installed.
        ValueError: If method is not supported.
        ValueError: If images are not spatially aligned.
        FileNotFoundError: If either image file does not exist.

    Example:
        >>> from geoai.tools.timeseries import detect_change
        >>> detect_change(
        ...     "scene_2020.tif",
        ...     "scene_2023.tif",
        ...     "change_map.tif",
        ...     method="normalized_difference",
        ...     bands=[4],
        ...     threshold=0.3,
        ... )
        'change_map.tif'
    """
    check_rasterio_available()

    if method not in CHANGE_METHODS:
        raise ValueError(
            f"Unknown change detection method '{method}'. "
            f"Supported methods: {CHANGE_METHODS}"
        )

    if not os.path.exists(image1_path):
        raise FileNotFoundError(f"Image file not found: {image1_path}")
    if not os.path.exists(image2_path):
        raise FileNotFoundError(f"Image file not found: {image2_path}")

    # Validate spatial alignment
    validate_temporal_stack([image1_path, image2_path])

    if bands is None:
        bands = [1]

    if verbose:
        print(
            f"Detecting changes ({method}) between "
            f"{os.path.basename(image1_path)} and "
            f"{os.path.basename(image2_path)}, bands={bands}"
        )

    # Determine nodata
    if nodata is None:
        with rasterio.open(image1_path) as src:
            nodata = src.nodata

    eps = 1e-10

    # Read bands from both images
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        profile = src1.profile.copy()
        height = src1.height
        width = src1.width

        change_per_band = []
        nodata_mask = np.zeros((height, width), dtype=bool)

        for band_idx in bands:
            data1 = src1.read(band_idx).astype(np.float32)
            data2 = src2.read(band_idx).astype(np.float32)

            # Track nodata pixels
            if nodata is not None:
                nodata_mask |= (data1 == nodata) | (data2 == nodata)

            # Compute change metric
            if method == "difference":
                change = data2 - data1
                if absolute:
                    change = np.abs(change)
            elif method == "ratio":
                change = data2 / (data1 + eps)
            elif method == "normalized_difference":
                change = (data2 - data1) / (data2 + data1 + eps)
                if absolute:
                    change = np.abs(change)

            change_per_band.append(change)

    # Combine bands
    if len(change_per_band) == 1:
        magnitude = change_per_band[0]
    else:
        # Euclidean magnitude across bands
        squared_sum = np.zeros((height, width), dtype=np.float32)
        for change in change_per_band:
            squared_sum += change**2
        magnitude = np.sqrt(squared_sum)

    # Apply nodata mask
    magnitude[nodata_mask] = np.nan

    # Apply threshold if provided
    if threshold is not None:
        binary = (np.abs(magnitude) > threshold).astype(np.uint8)
        binary[nodata_mask] = 255  # Use 255 as nodata for uint8
        output_dtype = "uint8"
        output_data = binary
        output_nodata = 255
    else:
        output_dtype = "float32"
        output_data = magnitude
        output_nodata = np.nan

    # Write output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    profile.update(
        dtype=output_dtype,
        count=1,
        compress="lzw",
        nodata=output_nodata,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_data, 1)

    if verbose:
        if threshold is not None:
            changed_pixels = (output_data == 1).sum()
            total_valid = (~nodata_mask).sum()
            pct = changed_pixels / total_valid * 100 if total_valid > 0 else 0
            print(f"  Changed pixels: {changed_pixels} ({pct:.1f}%)")
        print(f"  Change map saved to: {output_path}")

    return output_path


def calculate_temporal_statistics(
    input_paths: List[str],
    output_path: str,
    statistics: Optional[List[str]] = None,
    bands: Optional[List[int]] = None,
    nodata: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Calculate per-pixel temporal statistics across a stack of rasters.

    For each pixel, computes specified statistics across all temporal
    observations. The output is a multi-band GeoTIFF where each band
    corresponds to one statistic. If multiple input bands are requested,
    statistics are computed independently for each band, and bands are
    interleaved in the output as: [band1_stat1, band1_stat2, ...,
    band2_stat1, band2_stat2, ...].

    Args:
        input_paths (list of str): Paths to input GeoTIFF files. All must
            be spatially aligned.
        output_path (str): Path to save the output statistics GeoTIFF.
        statistics (list of str, optional): Statistics to compute. Any
            combination of: 'mean', 'std', 'min', 'max', 'range', 'count',
            'median'. If None, computes all. Defaults to None.
        bands (list of int, optional): 1-based band indices to analyze.
            If None, uses band 1 only. Defaults to None.
        nodata (float, optional): Nodata value. Defaults to None.
        verbose (bool): Print progress. Defaults to True.

    Returns:
        str: Path to the output temporal statistics GeoTIFF.

    Raises:
        ImportError: If rasterio is not installed.
        ValueError: If any statistic name is not recognized.
        ValueError: If input rasters are not spatially aligned.

    Example:
        >>> from geoai.tools.timeseries import calculate_temporal_statistics
        >>> ndvi_files = ["ndvi_jan.tif", "ndvi_mar.tif", "ndvi_jun.tif",
        ...               "ndvi_sep.tif", "ndvi_dec.tif"]
        >>> calculate_temporal_statistics(
        ...     ndvi_files,
        ...     "ndvi_stats.tif",
        ...     statistics=["mean", "std", "min", "max", "count"],
        ... )
        'ndvi_stats.tif'
    """
    check_rasterio_available()

    if statistics is None:
        statistics = list(TEMPORAL_STATISTICS)
    else:
        for stat in statistics:
            if stat not in TEMPORAL_STATISTICS:
                raise ValueError(
                    f"Unknown statistic '{stat}'. " f"Supported: {TEMPORAL_STATISTICS}"
                )

    # Validate spatial alignment
    ref_info = validate_temporal_stack(input_paths)
    height = ref_info["height"]
    width = ref_info["width"]
    num_scenes = len(input_paths)

    if bands is None:
        bands = [1]

    # Determine nodata
    if nodata is None:
        with rasterio.open(input_paths[0]) as src:
            nodata = src.nodata

    num_output_bands = len(bands) * len(statistics)

    if verbose:
        print(
            f"Computing temporal statistics for {num_scenes} scenes, "
            f"{len(bands)} band(s), {len(statistics)} statistic(s)"
        )

    # Compute statistics for each band
    results = []

    for band_idx in bands:
        if verbose:
            print(f"  Reading band {band_idx}...")

        # Read all scenes for this band
        stack = np.empty((num_scenes, height, width), dtype=np.float32)
        for i, path in enumerate(input_paths):
            with rasterio.open(path) as src:
                stack[i] = src.read(band_idx).astype(np.float32)

        # Mask nodata as NaN
        if nodata is not None:
            stack[stack == nodata] = np.nan

        # Compute each requested statistic
        for stat in statistics:
            if verbose:
                print(f"  Computing {stat} for band {band_idx}...")

            if stat == "mean":
                result = np.nanmean(stack, axis=0)
            elif stat == "std":
                result = np.nanstd(stack, axis=0)
            elif stat == "min":
                result = np.nanmin(stack, axis=0)
            elif stat == "max":
                result = np.nanmax(stack, axis=0)
            elif stat == "range":
                result = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
            elif stat == "count":
                result = np.sum(~np.isnan(stack), axis=0).astype(np.float32)
            elif stat == "median":
                result = np.nanmedian(stack, axis=0)

            results.append(result)

    # Write output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_paths[0]) as ref:
        profile = ref.profile.copy()

    profile.update(
        dtype="float32",
        count=num_output_bands,
        compress="lzw",
        nodata=np.nan,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for i, result in enumerate(results):
            dst.write(result, i + 1)

    if verbose:
        print(f"  Temporal statistics saved to: {output_path}")

    return output_path
