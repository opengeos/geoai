"""
Landcover Classification Utilities

This module provides utilities for discrete landcover classification workflows,
including tile export with background filtering and radiometric normalization
for multi-temporal image comparability.

Key Features:
- Enhanced tile filtering with configurable feature ratio thresholds
- Separate statistics tracking for different skip reasons
- LIRRN (Location-Independent Relative Radiometric Normalization)
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
from skimage.filters import threshold_multiotsu
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

__all__ = [
    "export_landcover_tiles",
    "normalize_radiometric",
]


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


# ---------------------------------------------------------------------------
# Radiometric Normalization
# ---------------------------------------------------------------------------


def _load_raster(filepath: str) -> Tuple[np.ndarray, dict]:
    """Load a multi-band raster as a (H, W, B) float64 array.

    Args:
        filepath: Path to the raster file.

    Returns:
        Tuple of (image_array, profile) where image_array has shape (H, W, B)
        and profile is the rasterio dataset profile dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Raster file not found: {filepath}")
    with rasterio.open(filepath) as src:
        img = src.read()  # (B, H, W)
        profile = src.profile.copy()
    img = np.moveaxis(img, 0, -1).astype(np.float64)
    img = np.abs(img)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img, profile


def _save_raster(filepath: str, img: np.ndarray, profile: dict) -> None:
    """Save a (H, W, B) array as a multi-band GeoTIFF.

    Args:
        filepath: Output file path.
        img: Image array with shape (H, W, B).
        profile: Rasterio profile dict (from ``_load_raster``).

    Raises:
        ValueError: If *img* is not 3-dimensional.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3-D array (H, W, B), got shape {img.shape}")
    out_profile = profile.copy()
    out_profile.update(
        dtype="float64",
        count=img.shape[2],
        height=img.shape[0],
        width=img.shape[1],
        compress="lzw",
    )
    with rasterio.open(filepath, "w", **out_profile) as dst:
        for i in range(img.shape[2]):
            dst.write(img[:, :, i], i + 1)


def _compute_distances(
    p_n: int,
    a1: np.ndarray,
    b1: np.ndarray,
    id_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select *p_n* samples closest to the maximum value, then subsample.

    Args:
        p_n: Number of candidate samples to draw from each array.
        a1: Non-zero reference pixel values (1-D).
        b1: Non-zero subject pixel values (1-D).
        id_indices: Random indices used to subsample from the candidates.

    Returns:
        Tuple of (sub_samples, ref_samples) after subsampling.
    """
    max_ref = np.max(a1)
    idx_ref = np.argsort(np.abs(a1 - max_ref))
    ref_candidates = a1[idx_ref[:p_n]]

    max_sub = np.max(b1)
    idx_sub = np.argsort(np.abs(b1 - max_sub))
    sub_candidates = b1[idx_sub[:p_n]]

    safe_indices = id_indices[
        id_indices < min(len(sub_candidates), len(ref_candidates))
    ]
    if len(safe_indices) == 0:
        return sub_candidates, ref_candidates
    return sub_candidates[safe_indices], ref_candidates[safe_indices]


def _compute_sample(
    p_n: int,
    a1: np.ndarray,
    b1: np.ndarray,
    id_indices: np.ndarray,
    num_sampling_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiple rounds of distance-based sampling, concatenated (non-zero only).

    Args:
        p_n: Number of samples per round.
        a1: Non-zero reference pixel values (1-D).
        b1: Non-zero subject pixel values (1-D).
        id_indices: Random subsample indices.
        num_sampling_rounds: Number of sampling rounds.

    Returns:
        Tuple of (sub_combined, ref_combined).
    """
    pairs = [
        _compute_distances(p_n, a1, b1, id_indices) for _ in range(num_sampling_rounds)
    ]
    sub_combined = np.concatenate([s[s != 0] for s, _ in pairs])
    ref_combined = np.concatenate([r[r != 0] for _, r in pairs])
    return sub_combined, ref_combined


def _sample_selection(
    p_n: int,
    a: np.ndarray,
    b: np.ndarray,
    id_indices: np.ndarray,
    num_sampling_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select representative sample pairs from a single quantization level.

    Args:
        p_n: Number of sample points.
        a: Flattened reference pixels (masked by quantization level; 0 = outside).
        b: Flattened subject pixels (masked by quantization level; 0 = outside).
        id_indices: Random sub-sampling indices.
        num_sampling_rounds: Number of sampling rounds.

    Returns:
        Tuple of (sub_samples, ref_samples), each 1-D.
    """
    a1 = a[a != 0]
    b1 = b[b != 0]

    if len(a1) == 0 or len(b1) == 0:
        return np.array([0.0]), np.array([0.0])

    if len(a1) < p_n or len(b1) < p_n:
        min_len = min(len(a1), len(b1))
        return b1[:min_len], a1[:min_len]

    sub_1, ref_1 = _compute_sample(p_n, a1, b1, id_indices, num_sampling_rounds)

    sub = np.concatenate([sub_1[sub_1 != 0], sub_1[sub_1 == 0]])
    ref = np.concatenate([ref_1[ref_1 != 0], ref_1[ref_1 == 0]])
    return sub, ref


def _linear_reg(
    sub_samples: np.ndarray,
    ref_samples: np.ndarray,
    image_band: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """OLS linear regression to normalize one image band.

    Fits ``ref = intercept + slope * sub`` and applies the transformation
    to *image_band*.

    Args:
        sub_samples: Subject sample values (1-D).
        ref_samples: Reference sample values (1-D).
        image_band: Full subject band to normalize (H, W).

    Returns:
        Tuple of (normalized_band, adjusted_r_squared, rmse).

    Raises:
        ValueError: If fewer than 2 valid samples are available for regression.
    """
    mask = (sub_samples != 0) & (ref_samples != 0)
    sub_clean = sub_samples[mask]
    ref_clean = ref_samples[mask]

    if len(sub_clean) < 2:
        raise ValueError(
            f"Insufficient samples for regression: got {len(sub_clean)}, need >= 2"
        )

    X = sub_clean.reshape(-1, 1)
    y = ref_clean

    model = LinearRegression().fit(X, y)
    intercept, slope = model.intercept_, model.coef_[0]

    norm_band = intercept + slope * image_band

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    n = len(y)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    r_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else 0.0
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return norm_band, r_adj, rmse


def _lirrn(
    p_n: int,
    sub_img: np.ndarray,
    ref_img: np.ndarray,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core LIRRN algorithm.

    Implements Location-Independent Relative Radiometric Normalization.
    Processes each band independently via multi-Otsu thresholding,
    stratified sampling and per-band linear regression.

    Args:
        p_n: Number of sample points per quantization level.
        sub_img: Subject image (H, W, B) float64.
        ref_img: Reference image (H, W, B) float64.
        num_quantisation_classes: Number of brightness strata (default 3).
        num_sampling_rounds: Number of sampling rounds (default 3).
        subsample_ratio: Fraction of candidates retained (default 0.1).
        rng: Numpy random Generator for reproducibility.

    Returns:
        Tuple of (normalized_image, rmse_per_band, r_adj_per_band).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_bands = sub_img.shape[2]

    id_indices = rng.integers(0, p_n, size=max(1, round(subsample_ratio * p_n)))

    norm_img = np.zeros_like(sub_img, dtype=np.float64)
    rmse = np.zeros(num_bands)
    r_adj = np.zeros(num_bands)

    # Quantize each band into brightness levels via multi-Otsu
    sub_labels = np.zeros_like(sub_img, dtype=np.int32)
    ref_labels = np.zeros_like(ref_img, dtype=np.int32)

    for j in range(num_bands):
        for img, labels in [(sub_img, sub_labels), (ref_img, ref_labels)]:
            nonzero = img[:, :, j][img[:, :, j] != 0]
            if len(nonzero) > 0:
                try:
                    thresh = threshold_multiotsu(
                        nonzero, classes=num_quantisation_classes
                    )
                    labels[:, :, j] = np.digitize(img[:, :, j], bins=thresh) + 1
                except ValueError:
                    labels[:, :, j] = 1

    # For each band: sample from quantization levels then regress
    for j in range(num_bands):
        sub_list, ref_list = [], []

        for level in range(1, num_quantisation_classes + 1):
            a = np.where(ref_labels[:, :, j] == level, ref_img[:, :, j], 0).ravel()
            b = np.where(sub_labels[:, :, j] == level, sub_img[:, :, j], 0).ravel()
            sub_s, ref_s = _sample_selection(p_n, a, b, id_indices, num_sampling_rounds)
            sub_list.append(sub_s)
            ref_list.append(ref_s)

        all_sub = np.concatenate(sub_list)
        all_ref = np.concatenate(ref_list)

        try:
            norm_img[:, :, j], r_adj[j], rmse[j] = _linear_reg(
                all_sub, all_ref, sub_img[:, :, j]
            )
        except ValueError:
            warnings.warn(
                f"Band {j}: insufficient samples for regression, "
                "returning band unchanged.",
                stacklevel=2,
            )
            norm_img[:, :, j] = sub_img[:, :, j]
            continue

        ref_band = ref_img[:, :, j]
        norm_img[:, :, j] = np.clip(norm_img[:, :, j], ref_band.min(), ref_band.max())

    return norm_img, rmse, r_adj


def normalize_radiometric(
    subject_image: Union[str, np.ndarray],
    reference_image: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    method: str = "lirrn",
    p_n: int = 500,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize subject image radiometry to match a reference image.

    Adjusts brightness and contrast of the subject image so that its pixel
    value distribution matches the reference image. This is essential for
    multi-temporal analysis where images are acquired under different
    atmospheric conditions, sensor calibrations, or illumination angles.

    Currently supports the LIRRN (Location-Independent Relative Radiometric
    Normalization) method, which uses multi-Otsu thresholding and linear
    regression to identify pseudo-invariant features and transform pixel
    values band-by-band.

    Reference: doi:10.3390/s24072272

    Args:
        subject_image: Path to the subject GeoTIFF or numpy array with
            shape (H, W, B). The image to be normalized.
        reference_image: Path to the reference GeoTIFF or numpy array with
            shape (H, W, B). The target radiometry to match.
        output_path: Path to save the normalized image as GeoTIFF. Only
            applicable when *subject_image* is a file path (so spatial
            metadata is available). If None, the array is returned without
            saving. Default: None.
        method: Normalization method. Currently only ``"lirrn"`` is
            supported. Default: ``"lirrn"``.
        p_n: Number of pseudo-invariant feature samples per quantization
            level. Higher values increase accuracy but slow computation.
            Default: 500.
        num_quantisation_classes: Number of brightness strata for stratified
            sampling. Default: 3.
        num_sampling_rounds: Number of iterative refinement rounds for
            sample selection. Default: 3.
        subsample_ratio: Fraction of candidates retained for regression.
            Default: 0.1.
        random_state: Seed or numpy Generator for reproducible results.
            Default: None (non-deterministic).

    Returns:
        Tuple of (normalized_image, metrics) where:
            - normalized_image: numpy array (H, W, B) float64.
            - metrics: dict with keys ``"rmse"`` and ``"r_adj"``, each a
              numpy array of length B.

    Raises:
        ValueError: If *method* is not ``"lirrn"``.
        ValueError: If *p_n* < 1 or *num_sampling_rounds* < 1.
        ValueError: If subject and reference have different band counts.
        ValueError: If input arrays are not 3-dimensional.
        ValueError: If *output_path* is set but *subject_image* is an array.
        FileNotFoundError: If file paths do not point to existing files.

    Examples:
        Normalize a satellite image using file paths:

        >>> from geoai import normalize_radiometric
        >>> norm_img, metrics = normalize_radiometric(
        ...     "subject.tif",
        ...     "reference.tif",
        ...     output_path="normalized.tif",
        ... )
        >>> print(f"RMSE per band: {metrics['rmse']}")

        Normalize using numpy arrays:

        >>> import numpy as np
        >>> subject = np.random.rand(100, 100, 4)
        >>> reference = np.random.rand(120, 120, 4)
        >>> norm_img, metrics = normalize_radiometric(subject, reference)
        >>> norm_img.shape
        (100, 100, 4)

    Note:
        The subject and reference images must have the same number of bands
        but may have different spatial dimensions (height and width).
    """
    # --- Validate parameters ---
    if method != "lirrn":
        raise ValueError(
            f"Unsupported normalization method {method!r}. "
            "Currently only 'lirrn' is supported."
        )
    if p_n < 1:
        raise ValueError(f"p_n must be >= 1, got {p_n}")
    if num_sampling_rounds < 1:
        raise ValueError(f"num_sampling_rounds must be >= 1, got {num_sampling_rounds}")
    if subsample_ratio <= 0 or subsample_ratio > 1:
        raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")

    # --- Resolve inputs ---
    profile = None
    if isinstance(subject_image, str):
        sub_arr, profile = _load_raster(subject_image)
    else:
        sub_arr = np.asarray(subject_image, dtype=np.float64)
        if sub_arr.ndim != 3:
            raise ValueError(
                f"subject_image must be 3-D (H, W, B), got {sub_arr.ndim}-D"
            )

    if isinstance(reference_image, str):
        ref_arr, _ = _load_raster(reference_image)
    else:
        ref_arr = np.asarray(reference_image, dtype=np.float64)
        if ref_arr.ndim != 3:
            raise ValueError(
                f"reference_image must be 3-D (H, W, B), got {ref_arr.ndim}-D"
            )

    if output_path is not None and profile is None:
        raise ValueError(
            "output_path requires subject_image to be a file path "
            "(not an array) so that spatial metadata is available."
        )

    # Band count check
    if sub_arr.shape[2] != ref_arr.shape[2]:
        raise ValueError(
            f"Band count mismatch: subject has {sub_arr.shape[2]} bands, "
            f"reference has {ref_arr.shape[2]} bands."
        )

    # Handle NaN / inf
    if np.any(~np.isfinite(sub_arr)):
        warnings.warn(
            "subject_image contains NaN or infinite values; " "replacing with 0.",
            stacklevel=2,
        )
        sub_arr = np.nan_to_num(sub_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(~np.isfinite(ref_arr)):
        warnings.warn(
            "reference_image contains NaN or infinite values; " "replacing with 0.",
            stacklevel=2,
        )
        ref_arr = np.nan_to_num(ref_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Build RNG ---
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # --- Run normalization ---
    norm_img, rmse, r_adj = _lirrn(
        p_n,
        sub_arr,
        ref_arr,
        num_quantisation_classes=num_quantisation_classes,
        num_sampling_rounds=num_sampling_rounds,
        subsample_ratio=subsample_ratio,
        rng=rng,
    )

    metrics = {"rmse": rmse, "r_adj": r_adj}

    if output_path is not None:
        _save_raster(output_path, norm_img, profile)

    return norm_img, metrics
