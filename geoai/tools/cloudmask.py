"""
OmniCloudMask integration for cloud and cloud shadow detection in satellite imagery.

This module provides functions to use OmniCloudMask (https://github.com/DPIRD-DMA/OmniCloudMask)
for detecting clouds and cloud shadows in satellite imagery. OmniCloudMask performs semantic
segmentation to classify pixels into: Clear (0), Thick Cloud (1), Thin Cloud (2), Cloud Shadow (3).

Supports Sentinel-2, Landsat 8, PlanetScope, and Maxar imagery at 10-50m resolution.
"""

import os
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

try:
    from omnicloudmask import predict_from_array

    OMNICLOUDMASK_AVAILABLE = True
except ImportError:
    OMNICLOUDMASK_AVAILABLE = False

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# Cloud mask class values
CLEAR = 0
THICK_CLOUD = 1
THIN_CLOUD = 2
CLOUD_SHADOW = 3


def check_omnicloudmask_available():
    """
    Check if omnicloudmask is installed.

    Raises:
        ImportError: If omnicloudmask is not installed.
    """
    if not OMNICLOUDMASK_AVAILABLE:
        raise ImportError(
            "omnicloudmask is not installed. "
            "Please install it with: pip install omnicloudmask "
            "or: pip install geoai-py[extra]"
        )


def predict_cloud_mask(
    image: np.ndarray,
    batch_size: int = 1,
    inference_device: str = "cpu",
    inference_dtype: str = "fp32",
    patch_size: int = 1000,
    export_confidence: bool = False,
    model_version: int = 3,
) -> np.ndarray:
    """
    Predict cloud mask from a numpy array using OmniCloudMask.

    This function classifies each pixel into one of four categories:
    - 0: Clear
    - 1: Thick Cloud
    - 2: Thin Cloud
    - 3: Cloud Shadow

    Args:
        image (np.ndarray): Input image array with shape (3, height, width) or (height, width, 3).
            Should contain Red, Green, and NIR bands. Values should be in reflectance (0-1)
            or digital numbers (0-10000 typical for Sentinel-2/Landsat).
        batch_size (int): Number of patches to process per inference batch. Defaults to 1.
        inference_device (str): Device for inference ('cpu', 'cuda', or 'mps'). Defaults to 'cpu'.
        inference_dtype (str): Data type for inference ('fp32', 'fp16', or 'bf16').
            'bf16' recommended for speed on compatible hardware. Defaults to 'fp32'.
        patch_size (int): Size of patches for processing large images. Defaults to 1000.
        export_confidence (bool): If True, also returns confidence map. Defaults to False.
        model_version (int): Model version to use (1, 2, or 3). Defaults to 3.

    Returns:
        np.ndarray: Cloud mask array with shape (height, width) containing class predictions.
            If export_confidence=True, returns tuple of (mask, confidence).

    Raises:
        ImportError: If omnicloudmask is not installed.
        ValueError: If image has wrong shape or number of channels.

    Example:
        >>> import numpy as np
        >>> from geoai.tools.cloudmask import predict_cloud_mask
        >>> # Create synthetic image (3 bands: R, G, NIR)
        >>> image = np.random.rand(3, 512, 512) * 10000
        >>> mask = predict_cloud_mask(image)
        >>> print(f"Clear pixels: {(mask == 0).sum()}")
    """
    check_omnicloudmask_available()

    # Ensure image has correct shape (3, H, W)
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape {image.shape}")

    # Convert (H, W, 3) to (3, H, W) if needed
    if image.shape[2] == 3 and image.shape[0] != 3:
        image = np.transpose(image, (2, 0, 1))

    if image.shape[0] != 3:
        raise ValueError(
            f"Image must have 3 channels (R, G, NIR), got {image.shape[0]} channels"
        )

    # Call OmniCloudMask
    result = predict_from_array(
        image,
        batch_size=batch_size,
        inference_device=inference_device,
        inference_dtype=inference_dtype,
        patch_size=patch_size,
        export_confidence=export_confidence,
        model_version=model_version,
    )

    # Handle output shape - omnicloudmask returns (1, H, W) or ((1, H, W), (1, H, W))
    if export_confidence:
        mask, confidence = result
        # Squeeze batch dimension
        mask = mask.squeeze(0) if mask.ndim == 3 else mask
        confidence = confidence.squeeze(0) if confidence.ndim == 3 else confidence
        return mask, confidence
    else:
        # Squeeze batch dimension
        return result.squeeze(0) if result.ndim == 3 else result


def predict_cloud_mask_from_raster(
    input_path: str,
    output_path: str,
    red_band: int = 1,
    green_band: int = 2,
    nir_band: int = 3,
    batch_size: int = 1,
    inference_device: str = "cpu",
    inference_dtype: str = "fp32",
    patch_size: int = 1000,
    export_confidence: bool = False,
    model_version: int = 3,
) -> None:
    """
    Predict cloud mask from a GeoTIFF file and save the result.

    Reads a multi-band raster, extracts RGB+NIR bands, applies OmniCloudMask,
    and saves the result while preserving geospatial metadata.

    Args:
        input_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save cloud mask GeoTIFF.
        red_band (int): Band index for Red (1-indexed). Defaults to 1.
        green_band (int): Band index for Green (1-indexed). Defaults to 2.
        nir_band (int): Band index for NIR (1-indexed). Defaults to 3.
        batch_size (int): Patches per inference batch. Defaults to 1.
        inference_device (str): Device ('cpu', 'cuda', 'mps'). Defaults to 'cpu'.
        inference_dtype (str): Dtype ('fp32', 'fp16', 'bf16'). Defaults to 'fp32'.
        patch_size (int): Patch size for large images. Defaults to 1000.
        export_confidence (bool): Export confidence map. Defaults to False.
        model_version (str): Model version ('1.0', '2.0', '3.0'). Defaults to '3.0'.

    Returns:
        None: Writes cloud mask to output_path.

    Raises:
        ImportError: If omnicloudmask or rasterio not installed.
        FileNotFoundError: If input_path doesn't exist.

    Example:
        >>> from geoai.tools.cloudmask import predict_cloud_mask_from_raster
        >>> predict_cloud_mask_from_raster(
        ...     "sentinel2_image.tif",
        ...     "cloud_mask.tif",
        ...     red_band=4,   # Sentinel-2 band order
        ...     green_band=3,
        ...     nir_band=8
        ... )
    """
    check_omnicloudmask_available()

    if not RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for raster operations. "
            "Please install it with: pip install rasterio"
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read input raster
    with rasterio.open(input_path) as src:
        # Read required bands
        red = src.read(red_band).astype(np.float32)
        green = src.read(green_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)

        # Stack into (3, H, W)
        image = np.stack([red, green, nir], axis=0)

        # Get metadata
        profile = src.profile.copy()

    # Predict cloud mask
    result = predict_cloud_mask(
        image,
        batch_size=batch_size,
        inference_device=inference_device,
        inference_dtype=inference_dtype,
        patch_size=patch_size,
        export_confidence=export_confidence,
        model_version=model_version,
    )

    # Handle confidence output
    if export_confidence:
        mask, confidence = result
    else:
        mask = result

    # Update profile for output
    profile.update(
        dtype=np.uint8,
        count=1,
        compress="lzw",
        nodata=None,
    )

    # Write cloud mask
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and output_dir != os.path.abspath(os.sep):
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

    # Optionally write confidence map
    if export_confidence:
        confidence_path = output_path.replace(".tif", "_confidence.tif")
        profile.update(dtype=np.float32)
        with rasterio.open(confidence_path, "w", **profile) as dst:
            dst.write(confidence, 1)


def predict_cloud_mask_batch(
    input_paths: List[str],
    output_dir: str,
    red_band: int = 1,
    green_band: int = 2,
    nir_band: int = 3,
    batch_size: int = 1,
    inference_device: str = "cpu",
    inference_dtype: str = "fp32",
    patch_size: int = 1000,
    export_confidence: bool = False,
    model_version: int = 3,
    suffix: str = "_cloudmask",
    verbose: bool = True,
) -> List[str]:
    """
    Predict cloud masks for multiple rasters in batch.

    Processes multiple GeoTIFF files with the same cloud detection parameters
    and saves results to an output directory.

    Args:
        input_paths (list of str): Paths to input GeoTIFF files.
        output_dir (str): Directory to save cloud masks.
        red_band (int): Red band index. Defaults to 1.
        green_band (int): Green band index. Defaults to 2.
        nir_band (int): NIR band index. Defaults to 3.
        batch_size (int): Patches per batch. Defaults to 1.
        inference_device (str): Device. Defaults to 'cpu'.
        inference_dtype (str): Dtype. Defaults to 'fp32'.
        patch_size (int): Patch size. Defaults to 1000.
        export_confidence (bool): Export confidence. Defaults to False.
        model_version (str): Model version. Defaults to '3.0'.
        suffix (str): Suffix for output filenames. Defaults to '_cloudmask'.
        verbose (bool): Print progress. Defaults to True.

    Returns:
        list of str: Paths to output cloud mask files.

    Raises:
        ImportError: If omnicloudmask or rasterio not installed.

    Example:
        >>> from geoai.tools.cloudmask import predict_cloud_mask_batch
        >>> files = ["scene1.tif", "scene2.tif", "scene3.tif"]
        >>> outputs = predict_cloud_mask_batch(
        ...     files,
        ...     output_dir="cloud_masks",
        ...     inference_device="cuda"
        ... )
    """
    check_omnicloudmask_available()

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
            # Predict cloud mask
            predict_cloud_mask_from_raster(
                input_path,
                output_path,
                red_band=red_band,
                green_band=green_band,
                nir_band=nir_band,
                batch_size=batch_size,
                inference_device=inference_device,
                inference_dtype=inference_dtype,
                patch_size=patch_size,
                export_confidence=export_confidence,
                model_version=model_version,
            )

            output_paths.append(output_path)

            if verbose:
                print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue

    return output_paths


def calculate_cloud_statistics(
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate statistics from a cloud mask.

    Args:
        mask (np.ndarray): Cloud mask array with values 0-3.

    Returns:
        dict: Statistics including:
            - total_pixels: Total number of pixels
            - clear_pixels: Number of clear pixels
            - thick_cloud_pixels: Number of thick cloud pixels
            - thin_cloud_pixels: Number of thin cloud pixels
            - shadow_pixels: Number of cloud shadow pixels
            - clear_percent: Percentage of clear pixels
            - cloud_percent: Percentage of cloudy pixels (thick + thin)
            - shadow_percent: Percentage of shadow pixels

    Example:
        >>> from geoai.tools.cloudmask import calculate_cloud_statistics
        >>> import numpy as np
        >>> mask = np.random.randint(0, 4, (512, 512))
        >>> stats = calculate_cloud_statistics(mask)
        >>> print(f"Clear: {stats['clear_percent']:.1f}%")
    """
    total_pixels = mask.size

    clear_pixels = (mask == CLEAR).sum()
    thick_cloud_pixels = (mask == THICK_CLOUD).sum()
    thin_cloud_pixels = (mask == THIN_CLOUD).sum()
    shadow_pixels = (mask == CLOUD_SHADOW).sum()

    cloud_pixels = thick_cloud_pixels + thin_cloud_pixels

    return {
        "total_pixels": int(total_pixels),
        "clear_pixels": int(clear_pixels),
        "thick_cloud_pixels": int(thick_cloud_pixels),
        "thin_cloud_pixels": int(thin_cloud_pixels),
        "shadow_pixels": int(shadow_pixels),
        "clear_percent": float(clear_pixels / total_pixels * 100),
        "cloud_percent": float(cloud_pixels / total_pixels * 100),
        "shadow_percent": float(shadow_pixels / total_pixels * 100),
    }


def create_cloud_free_mask(
    mask: np.ndarray,
    include_thin_clouds: bool = False,
    include_shadows: bool = False,
) -> np.ndarray:
    """
    Create a binary mask of cloud-free pixels.

    Args:
        mask (np.ndarray): Cloud mask with values 0-3.
        include_thin_clouds (bool): If True, treats thin clouds as acceptable.
            Defaults to False.
        include_shadows (bool): If True, treats shadows as acceptable.
            Defaults to False.

    Returns:
        np.ndarray: Binary mask where 1 = usable, 0 = not usable.

    Example:
        >>> from geoai.tools.cloudmask import create_cloud_free_mask
        >>> import numpy as np
        >>> mask = np.random.randint(0, 4, (512, 512))
        >>> cloud_free = create_cloud_free_mask(mask)
        >>> print(f"Usable pixels: {cloud_free.sum()}")
    """
    # Start with clear pixels
    usable = mask == CLEAR

    # Optionally include thin clouds
    if include_thin_clouds:
        usable = usable | (mask == THIN_CLOUD)

    # Optionally include shadows
    if include_shadows:
        usable = usable | (mask == CLOUD_SHADOW)

    return usable.astype(np.uint8)
