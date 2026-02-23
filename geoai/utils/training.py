"""Training data export and augmentation utilities."""

# Standard Library
import glob
import json
import logging
import math
import os
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Libraries
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
import torch
import torchgeo
from PIL import Image
from rasterio import features
from rasterio.windows import Window
from shapely.affinity import rotate
from shapely.geometry import box
from torchvision.transforms import RandomRotation
from tqdm import tqdm

__all__ = [
    "get_default_augmentation_transforms",
    "export_geotiff_tiles",
    "export_geotiff_tiles_batch",
    "export_training_data",
    "flipnslide_augmentation",
    "export_flipnslide_tiles",
]


def get_default_augmentation_transforms(
    tile_size: int = 256,
    include_normalize: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    """
    Get default data augmentation transforms for geospatial imagery using albumentations.

    This function returns a composition of augmentation transforms commonly used
    for remote sensing and geospatial data. The transforms include geometric
    transformations (flips, rotations) and photometric adjustments (brightness,
    contrast, saturation).

    Args:
        tile_size (int): Target size for tiles. Defaults to 256.
        include_normalize (bool): Whether to include normalization transform.
            Defaults to False. Set to True if using for training with pretrained models.
        mean (tuple): Mean values for normalization (RGB). Defaults to ImageNet values.
        std (tuple): Standard deviation for normalization (RGB). Defaults to ImageNet values.

    Returns:
        albumentations.Compose: A composition of augmentation transforms.

    Example:
        >>> import albumentations as A
        >>> # Get default transforms
        >>> transform = get_default_augmentation_transforms()
        >>> # Apply to image and mask
        >>> augmented = transform(image=image, mask=mask)
        >>> aug_image = augmented['image']
        >>> aug_mask = augmented['mask']
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install it with: pip install albumentations"
        )

    transforms_list = [
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5,
        ),
        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]

    # Add normalization if requested
    if include_normalize:
        transforms_list.append(A.Normalize(mean=mean, std=std))

    return A.Compose(transforms_list)


def flipnslide_augmentation(
    image,
    tile_size=256,
    output_format="numpy",
    crop_to_multiple=True,
):
    """
    Apply Flip-n-Slide tiling strategy for geospatial imagery data augmentation.

    This function implements the Flip-n-Slide algorithm from "A Concise Tiling
    Strategy for Preserving Spatial Context in Earth Observation Imagery" by
    Abrahams et al., presented at the ML4RS workshop at ICLR 2024 (best short
    paper). The strategy generates overlapping tiles with diverse augmentations
    while eliminating redundant pixel representations.

    The algorithm produces two sets of tiles:

    1. **Standard overlapping tiles** with half-stride (stride = tile_size / 2)
       and rotational augmentations determined by grid position:

       - Even row, even col: identity (no augmentation)
       - Odd row, even col: 180 degree rotation
       - Even row, odd col: 90 degree rotation
       - Odd row, odd col: 270 degree rotation

    2. **Inner offset tiles** extracted from the image interior (inset by
       tile_size / 2 from each edge) with the same half-stride, applying flip
       and rotation augmentations:

       - Even row, even col: horizontal flip
       - Even row, odd col: vertical flip
       - Odd row, odd col: 90 degree rotation + horizontal flip
       - Odd row, even col: 90 degree rotation + vertical flip

    Args:
        image (Union[str, numpy.ndarray]): Input image as a numpy array of
            shape ``(channels, height, width)`` or a file path to a raster
            readable by rasterio.
        tile_size (int, optional): Size of each square tile in pixels.
            Defaults to 256.
        output_format (str, optional): ``"numpy"`` to return a
            :class:`numpy.ndarray` or ``"torch"`` to return a
            :class:`torch.Tensor`. Defaults to ``"numpy"``.
        crop_to_multiple (bool, optional): If ``True``, crop the image to the
            nearest dimensions that are multiples of *tile_size* before tiling.
            Defaults to ``True``.

    Returns:
        Tuple[Union[numpy.ndarray, torch.Tensor], List[int]]:
            - **tiles** -- Array of shape
              ``(num_tiles, channels, tile_size, tile_size)``.
            - **augmentation_indices** -- List of integers indicating the
              augmentation applied to each tile:

              - 0: Identity (no augmentation)
              - 1: 180 degree rotation
              - 2: 90 degree rotation
              - 3: 270 degree rotation
              - 4: Horizontal flip
              - 5: Vertical flip
              - 6: 90 degree rotation + horizontal flip
              - 7: 90 degree rotation + vertical flip

    Example:
        >>> import numpy as np
        >>> image = np.random.rand(3, 512, 512)
        >>> tiles, aug_indices = flipnslide_augmentation(image, tile_size=256)
        >>> print(f"Generated {tiles.shape[0]} tiles of shape {tiles.shape[1:]}")
        >>> print(f"Augmentation types used: {sorted(set(aug_indices))}")

    References:
        Abrahams, E., Snow, T., Siegfried, M. R., & Perez, F. (2024).
        *A Concise Tiling Strategy for Preserving Spatial Context in Earth
        Observation Imagery*. ML4RS Workshop @ ICLR 2024.
        https://doi.org/10.48550/arXiv.2404.10927
    """
    # Load image if path provided
    if isinstance(image, str):
        import rasterio

        with rasterio.open(image) as src:
            image = src.read()

    # Ensure numpy array
    if not isinstance(image, np.ndarray):
        try:
            image = image.cpu().numpy()
        except AttributeError:
            image = np.asarray(image)

    # Validate input shape
    if image.ndim != 3:
        raise ValueError(
            f"Image must be a 3-D array (channels, height, width), "
            f"got shape {image.shape}"
        )

    channels, height, width = image.shape

    # Crop to nearest multiple of tile_size if requested
    if crop_to_multiple:
        new_height = (height // tile_size) * tile_size
        new_width = (width // tile_size) * tile_size
        if new_height == 0 or new_width == 0:
            raise ValueError(
                f"Image size ({height}x{width}) is smaller than "
                f"tile_size ({tile_size})"
            )
        if new_height < height or new_width < width:
            image = image[:, :new_height, :new_width]
            height, width = new_height, new_width

    # Check if image is large enough for tiling
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"Image size ({height}x{width}) is smaller than " f"tile_size ({tile_size})"
        )

    tiles = []
    augmentation_indices = []

    # Half-stride for overlapping tiles
    stride = tile_size // 2

    # ------------------------------------------------------------------
    # Stage 1: Standard overlapping tiles with rotational augmentations
    # ------------------------------------------------------------------
    for idx_h, row in enumerate(range(0, height - tile_size + 1, stride)):
        for idx_w, col in enumerate(range(0, width - tile_size + 1, stride)):
            tile = image[:, row : row + tile_size, col : col + tile_size]

            if idx_h % 2 == 0 and idx_w % 2 == 0:
                # Identity
                aug_tile = tile.copy()
                aug_idx = 0
            elif idx_h % 2 == 1 and idx_w % 2 == 0:
                # 180 degree rotation
                aug_tile = np.rot90(tile, 2, axes=(1, 2)).copy()
                aug_idx = 1
            elif idx_h % 2 == 0 and idx_w % 2 == 1:
                # 90 degree rotation
                aug_tile = np.rot90(tile, 1, axes=(1, 2)).copy()
                aug_idx = 2
            else:
                # 270 degree rotation
                aug_tile = np.rot90(tile, 3, axes=(1, 2)).copy()
                aug_idx = 3

            tiles.append(aug_tile)
            augmentation_indices.append(aug_idx)

    # ------------------------------------------------------------------
    # Stage 2: Inner offset tiles with flip + rotation augmentations
    # The inner image is inset by tile_size/2 from each edge, matching the
    # original FlipnSlide implementation.
    # ------------------------------------------------------------------
    inset = tile_size // 2
    if height - 2 * inset >= tile_size and width - 2 * inset >= tile_size:
        inner_image = image[:, inset : height - inset, inset : width - inset]
        inner_height = inner_image.shape[1]
        inner_width = inner_image.shape[2]

        for idx_h, row in enumerate(range(0, inner_height - tile_size + 1, stride)):
            for idx_w, col in enumerate(range(0, inner_width - tile_size + 1, stride)):
                tile = inner_image[:, row : row + tile_size, col : col + tile_size]

                if idx_h % 2 == 0 and idx_w % 2 == 0:
                    # Horizontal flip
                    aug_tile = tile[:, :, ::-1].copy()
                    aug_idx = 4
                elif idx_h % 2 == 0 and idx_w % 2 == 1:
                    # Vertical flip
                    aug_tile = tile[:, ::-1, :].copy()
                    aug_idx = 5
                elif idx_h % 2 == 1 and idx_w % 2 == 1:
                    # 90 degree rotation + horizontal flip
                    aug_tile = np.rot90(tile, 1, axes=(1, 2))
                    aug_tile = aug_tile[:, :, ::-1].copy()
                    aug_idx = 6
                else:
                    # 90 degree rotation + vertical flip
                    aug_tile = np.rot90(tile, 1, axes=(1, 2))
                    aug_tile = aug_tile[:, ::-1, :].copy()
                    aug_idx = 7

                tiles.append(aug_tile)
                augmentation_indices.append(aug_idx)

    # Stack into array
    tiles_array = np.stack(tiles, axis=0)

    # Optionally convert to torch tensor
    if output_format == "torch":
        tiles_array = torch.from_numpy(tiles_array)
    elif output_format != "numpy":
        raise ValueError(
            f"output_format must be 'numpy' or 'torch', got '{output_format}'"
        )

    return tiles_array, augmentation_indices


def export_flipnslide_tiles(
    in_raster,
    out_folder,
    in_class_data=None,
    tile_size=256,
    output_format="tif",
    crop_to_multiple=True,
    quiet=False,
):
    """
    Export georeferenced tiles using the Flip-n-Slide augmentation strategy.

    This function applies the Flip-n-Slide tiling algorithm to an image raster
    (and optionally a corresponding label/mask raster), preserving spatial
    relationships and geospatial information. Each tile is saved as an
    individual GeoTIFF file with proper CRS and geotransform.

    Args:
        in_raster (str): Path to the input raster image.
        out_folder (str): Path to the output folder where tiles will be saved.
        in_class_data (str, optional): Path to a classification/mask file.
            Can be a raster file (GeoTIFF, etc.) or vector file (GeoJSON,
            Shapefile, etc.). When provided, matching mask tiles are generated
            with identical augmentations. Vector files are rasterized to match
            the input raster dimensions and CRS. Defaults to None.
        tile_size (int, optional): Size of each square tile in pixels.
            Defaults to 256.
        output_format (str, optional): File extension for the output tiles
            (e.g., ``"tif"``). Defaults to ``"tif"``.
        crop_to_multiple (bool, optional): If ``True``, crop the image to the
            nearest dimensions that are multiples of *tile_size* before tiling.
            Defaults to ``True``.
        quiet (bool, optional): If ``True``, suppress progress output.
            Defaults to ``False``.

    Returns:
        dict: Statistics dictionary with keys:

            - ``'total_tiles'`` -- Number of tiles generated.
            - ``'tile_size'`` -- Size of each tile.
            - ``'augmentation_counts'`` -- Count of each augmentation type.
            - ``'output_folder'`` -- Path to the output folder.
            - ``'has_labels'`` -- Whether label tiles were generated.

    Example:
        >>> stats = export_flipnslide_tiles("image.tif", "output_tiles/")
        >>> print(f"Generated {stats['total_tiles']} tiles")

        >>> stats = export_flipnslide_tiles(
        ...     "image.tif", "output_tiles/",
        ...     in_class_data="mask.tif", tile_size=512,
        ... )

    Notes:
        Both input raster and class data (if provided) must share the same CRS
        and spatial extent for proper alignment.
    """
    import logging
    from collections import Counter

    logging.getLogger("rasterio").setLevel(logging.ERROR)

    # Create output directories
    os.makedirs(out_folder, exist_ok=True)
    image_dir = os.path.join(out_folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    has_labels = in_class_data is not None
    if has_labels:
        label_dir = os.path.join(out_folder, "labels")
        os.makedirs(label_dir, exist_ok=True)

    with rasterio.open(in_raster) as src:
        if not quiet:
            print(f"Input raster: {in_raster}")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bands: {src.count}")

        image_data = src.read()

        # Read class data if provided (handle both raster and vector)
        class_data = None
        if has_labels:
            # Detect if input is raster or vector
            is_class_data_raster = False
            file_ext = Path(in_class_data).suffix.lower()

            # Common raster extensions
            if file_ext in [".tif", ".tiff", ".img", ".jp2", ".png", ".bmp", ".gif"]:
                try:
                    with rasterio.open(in_class_data) as test_src:
                        is_class_data_raster = True
                except Exception:
                    is_class_data_raster = False
            else:
                # Common vector extensions
                vector_extensions = [
                    ".geojson",
                    ".json",
                    ".shp",
                    ".gpkg",
                    ".fgb",
                    ".parquet",
                    ".geoparquet",
                ]
                if file_ext in vector_extensions:
                    is_class_data_raster = False
                else:
                    # Unknown extension - try raster first, then vector
                    try:
                        with rasterio.open(in_class_data) as test_src:
                            is_class_data_raster = True
                    except Exception:
                        is_class_data_raster = False

            if is_class_data_raster:
                # Handle raster class data
                with rasterio.open(in_class_data) as class_src:
                    if class_src.crs != src.crs:
                        raise ValueError(
                            f"CRS mismatch: image ({src.crs}) vs mask ({class_src.crs})"
                        )
                    if (class_src.width != src.width) or (
                        class_src.height != src.height
                    ):
                        raise ValueError(
                            f"Dimension mismatch: image "
                            f"({src.width}x{src.height}) vs mask "
                            f"({class_src.width}x{class_src.height})"
                        )
                    class_data = class_src.read()
                    if not quiet:
                        print(f"  Class data (raster): {in_class_data}")
            else:
                # Handle vector class data
                try:
                    gdf = gpd.read_file(in_class_data)
                    if not quiet:
                        print(f"  Class data (vector): {in_class_data}")
                        print(f"    Loaded {len(gdf)} features")
                        print(f"    Vector CRS: {gdf.crs}")

                    # Reproject to match raster CRS if needed
                    if gdf.crs != src.crs:
                        if not quiet:
                            print(f"    Reprojecting from {gdf.crs} to {src.crs}")
                        gdf = gdf.to_crs(src.crs)

                    # Rasterize vector data to match raster dimensions
                    if len(gdf) > 0:
                        # Create binary mask: 1 for features, 0 for background
                        geometries = [
                            (geom, 1) for geom in gdf.geometry if geom is not None
                        ]
                        if geometries:
                            rasterized = features.rasterize(
                                geometries,
                                out_shape=(src.height, src.width),
                                transform=src.transform,
                                fill=0,
                                all_touched=True,
                                dtype=np.uint8,
                            )
                            # Reshape to match expected format (bands, height, width)
                            class_data = rasterized.reshape(1, src.height, src.width)
                        else:
                            # No valid geometries, create empty mask
                            class_data = np.zeros(
                                (1, src.height, src.width), dtype=np.uint8
                            )
                    else:
                        # Empty geodataframe, create empty mask
                        class_data = np.zeros(
                            (1, src.height, src.width), dtype=np.uint8
                        )

                except Exception as e:
                    raise ValueError(
                        f"Could not read {in_class_data} as vector file: {e}"
                    )

        # Apply FlipnSlide augmentation
        image_tiles, aug_indices = flipnslide_augmentation(
            image_data, tile_size=tile_size, crop_to_multiple=crop_to_multiple
        )

        class_tiles = None
        if has_labels:
            class_tiles, _ = flipnslide_augmentation(
                class_data, tile_size=tile_size, crop_to_multiple=crop_to_multiple
            )

        if not quiet:
            print(
                f"\nGenerated {len(image_tiles)} tiles with "
                f"Flip-n-Slide augmentation"
            )

        # Determine cropped dimensions for geo-transform calculation
        _channels, orig_h, orig_w = image_data.shape
        if crop_to_multiple:
            height = (orig_h // tile_size) * tile_size
            width = (orig_w // tile_size) * tile_size
        else:
            height, width = orig_h, orig_w

        transform = src.transform
        stride = tile_size // 2
        inset = tile_size // 2

        # Build tile positions mirroring flipnslide_augmentation order
        tile_positions = []

        # Stage 1 - standard overlapping tiles
        for row in range(0, height - tile_size + 1, stride):
            for col in range(0, width - tile_size + 1, stride):
                tile_positions.append((row, col))

        # Stage 2 - inner offset tiles
        inner_h = height - 2 * inset
        inner_w = width - 2 * inset
        if inner_h >= tile_size and inner_w >= tile_size:
            for row in range(0, inner_h - tile_size + 1, stride):
                for col in range(0, inner_w - tile_size + 1, stride):
                    tile_positions.append((inset + row, inset + col))

        # Augmentation label names for filenames
        aug_names = {
            0: "identity",
            1: "rot180",
            2: "rot90",
            3: "rot270",
            4: "hflip",
            5: "vflip",
            6: "rot90_hflip",
            7: "rot90_vflip",
        }

        # Save tiles
        for i, (tile_row, tile_col) in enumerate(tile_positions):
            tile_transform = rasterio.transform.from_bounds(
                transform.c + tile_col * transform.a,
                transform.f + (tile_row + tile_size) * transform.e,
                transform.c + (tile_col + tile_size) * transform.a,
                transform.f + tile_row * transform.e,
                tile_size,
                tile_size,
            )

            image_profile = src.profile.copy()
            image_profile.update(
                {
                    "height": tile_size,
                    "width": tile_size,
                    "count": image_tiles.shape[1],
                    "transform": tile_transform,
                }
            )

            aug_label = aug_names.get(aug_indices[i], str(aug_indices[i]))
            fname = f"tile_{i:06d}_{aug_label}.{output_format}"

            image_path = os.path.join(image_dir, fname)
            with rasterio.open(image_path, "w", **image_profile) as dst:
                dst.write(image_tiles[i])

            if has_labels:
                class_profile = image_profile.copy()
                class_profile.update(
                    {
                        "count": class_tiles.shape[1],
                        "dtype": class_tiles.dtype,
                    }
                )
                class_path = os.path.join(label_dir, fname)
                with rasterio.open(class_path, "w", **class_profile) as dst:
                    dst.write(class_tiles[i])

    # Statistics
    aug_counts = Counter(aug_indices)
    stats = {
        "total_tiles": len(image_tiles),
        "tile_size": tile_size,
        "augmentation_counts": dict(aug_counts),
        "output_folder": out_folder,
        "has_labels": has_labels,
    }

    if not quiet:
        print("\n--- Flip-n-Slide Export Summary ---")
        print(f"  Total tiles : {stats['total_tiles']}")
        print(f"  Tile size   : {tile_size}x{tile_size}")
        print(f"  Augmentations: {dict(aug_counts)}")
        if has_labels:
            print("  Exported both image and label tiles")
        else:
            print("  Exported image tiles only")

    return stats


def export_geotiff_tiles(
    in_raster,
    out_folder,
    in_class_data=None,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    quiet=False,
    all_touched=True,
    create_overview=False,
    skip_empty_tiles=False,
    metadata_format="PASCAL_VOC",
    apply_augmentation=False,
    augmentation_count=3,
    augmentation_transforms=None,
    tiling_strategy="grid",
):
    """
    Export georeferenced GeoTIFF tiles and labels from raster and classification data.

    Args:
        in_raster (str): Path to input raster image
        out_folder (str): Path to output folder
        in_class_data (str, optional): Path to classification data - can be vector file or raster.
            If None, only image tiles will be exported without labels. Defaults to None.
        tile_size (int): Size of tiles in pixels (square)
        stride (int): Step size between tiles
        class_value_field (str): Field containing class values (for vector data)
        buffer_radius (float): Buffer to add around features (in units of the CRS)
        max_tiles (int): Maximum number of tiles to process (None for all)
        quiet (bool): If True, suppress non-essential output
        all_touched (bool): Whether to use all_touched=True in rasterization (for vector data)
        create_overview (bool): Whether to create an overview image of all tiles
        skip_empty_tiles (bool): If True, skip tiles with no features
        metadata_format (str): Output metadata format (PASCAL_VOC, COCO, YOLO). Default: PASCAL_VOC
        apply_augmentation (bool): If True, generate augmented versions of each tile.
            This will create multiple variants of each tile using data augmentation techniques.
            Defaults to False.
        augmentation_count (int): Number of augmented versions to generate per tile
            (only used if apply_augmentation=True). Defaults to 3.
        augmentation_transforms (albumentations.Compose, optional): Custom augmentation transforms.
            If None and apply_augmentation=True, uses default transforms from
            get_default_augmentation_transforms(). Should be an albumentations.Compose object.
            Defaults to None.
        tiling_strategy (str, optional): Tiling strategy to use. Options are:
            - "grid": Regular grid tiling with specified stride (default behavior)
            - "flipnslide": Flip-n-Slide augmentation strategy with overlapping tiles
            Defaults to "grid".

    Returns:
        None: Tiles and labels are saved to out_folder.

    Example:
        >>> # Export tiles without augmentation
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif')
        >>>
        >>> # Export tiles with default augmentation (3 augmented versions per tile)
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif',
        ...                      apply_augmentation=True)
        >>>
        >>> # Export with custom augmentation
        >>> import albumentations as A
        >>> custom_transform = A.Compose([
        ...     A.HorizontalFlip(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif',
        ...                      apply_augmentation=True,
        ...                      augmentation_count=5,
        ...                      augmentation_transforms=custom_transform)
        >>>
        >>> # Export with Flip-n-Slide tiling strategy
        >>> export_geotiff_tiles('image.tif', 'output/', 'labels.tif',
        ...                      tiling_strategy='flipnslide')
    """

    import logging

    logging.getLogger("rasterio").setLevel(logging.ERROR)

    # Handle FlipnSlide tiling strategy
    if tiling_strategy == "flipnslide":
        if apply_augmentation:
            warnings.warn(
                "apply_augmentation is ignored when using tiling_strategy='flipnslide'. "
                "FlipnSlide applies its own augmentation scheme."
            )
        if stride != 128:
            warnings.warn(
                "stride parameter is ignored when using tiling_strategy='flipnslide'. "
                "FlipnSlide uses its own stride pattern (tile_size/2)."
            )

        # Use the dedicated FlipnSlide export function
        stats = export_flipnslide_tiles(
            in_raster=in_raster,
            out_folder=out_folder,
            in_class_data=in_class_data,
            tile_size=tile_size,
            quiet=quiet,
        )

        if not quiet:
            print("Used Flip-n-Slide tiling strategy")
            print(
                f"Generated {stats['total_tiles']} tiles with spatial context preservation"
            )

        return
    elif tiling_strategy != "grid":
        raise ValueError(
            f"Unknown tiling_strategy '{tiling_strategy}'. Must be 'grid' or 'flipnslide'."
        )

    # Initialize augmentation transforms if needed
    if apply_augmentation:
        if augmentation_transforms is None:
            augmentation_transforms = get_default_augmentation_transforms(
                tile_size=tile_size
            )
        if not quiet:
            print(
                f"Data augmentation enabled: generating {augmentation_count} augmented versions per tile"
            )

    # Create output directories
    os.makedirs(out_folder, exist_ok=True)
    image_dir = os.path.join(out_folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Only create label and annotation directories if class data is provided
    if in_class_data is not None:
        label_dir = os.path.join(out_folder, "labels")
        os.makedirs(label_dir, exist_ok=True)

        # Create annotation directory based on metadata format
        if metadata_format in ["PASCAL_VOC", "COCO"]:
            ann_dir = os.path.join(out_folder, "annotations")
            os.makedirs(ann_dir, exist_ok=True)

        # Initialize COCO annotations dictionary
        if metadata_format == "COCO":
            coco_annotations = {"images": [], "annotations": [], "categories": []}
            ann_id = 0

    # Determine if class data is raster or vector (only if class data provided)
    is_class_data_raster = False
    if in_class_data is not None:
        if isinstance(in_class_data, str):
            file_ext = Path(in_class_data).suffix.lower()
            # Common raster extensions
            if file_ext in [".tif", ".tiff", ".img", ".jp2", ".png", ".bmp", ".gif"]:
                try:
                    with rasterio.open(in_class_data) as src:
                        is_class_data_raster = True
                        if not quiet:
                            print(f"Detected in_class_data as raster: {in_class_data}")
                            print(f"Raster CRS: {src.crs}")
                            print(f"Raster dimensions: {src.width} x {src.height}")
                except Exception:
                    is_class_data_raster = False
                    if not quiet:
                        print(
                            f"Unable to open {in_class_data} as raster, trying as vector"
                        )

    # Open the input raster
    with rasterio.open(in_raster) as src:
        if not quiet:
            print(f"\nRaster info for {in_raster}:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Resolution: {src.res}")
            print(f"  Bands: {src.count}")
            print(f"  Bounds: {src.bounds}")

        # Calculate number of tiles
        num_tiles_x = math.ceil((src.width - tile_size) / stride) + 1
        num_tiles_y = math.ceil((src.height - tile_size) / stride) + 1
        total_tiles = num_tiles_x * num_tiles_y

        if max_tiles is None:
            max_tiles = total_tiles

        # Process classification data (only if class data provided)
        class_to_id = {}

        if in_class_data is not None and is_class_data_raster:
            # Load raster class data
            with rasterio.open(in_class_data) as class_src:
                # Check if raster CRS matches
                if class_src.crs != src.crs:
                    warnings.warn(
                        f"CRS mismatch: Class raster ({class_src.crs}) doesn't match input raster ({src.crs}). "
                        f"Results may be misaligned."
                    )

                # Get unique values from raster
                # Sample to avoid loading huge rasters
                sample_data = class_src.read(
                    1,
                    out_shape=(
                        1,
                        min(class_src.height, 1000),
                        min(class_src.width, 1000),
                    ),
                )

                unique_classes = np.unique(sample_data)
                unique_classes = unique_classes[
                    unique_classes > 0
                ]  # Remove 0 as it's typically background

                if not quiet:
                    print(
                        f"Found {len(unique_classes)} unique classes in raster: {unique_classes}"
                    )

                # Create class mapping
                class_to_id = {int(cls): i + 1 for i, cls in enumerate(unique_classes)}

                # Populate COCO categories
                if metadata_format == "COCO":
                    for cls_val in unique_classes:
                        coco_annotations["categories"].append(
                            {
                                "id": class_to_id[int(cls_val)],
                                "name": str(int(cls_val)),
                                "supercategory": "object",
                            }
                        )
        elif in_class_data is not None:
            # Load vector class data
            try:
                gdf = gpd.read_file(in_class_data)
                if not quiet:
                    print(f"Loaded {len(gdf)} features from {in_class_data}")
                    print(f"Vector CRS: {gdf.crs}")

                # Always reproject to match raster CRS
                if gdf.crs != src.crs:
                    if not quiet:
                        print(f"Reprojecting features from {gdf.crs} to {src.crs}")
                    gdf = gdf.to_crs(src.crs)

                # Apply buffer if specified
                if buffer_radius > 0:
                    gdf["geometry"] = gdf.buffer(buffer_radius)
                    if not quiet:
                        print(f"Applied buffer of {buffer_radius} units")

                # Check if class_value_field exists
                if class_value_field in gdf.columns:
                    unique_classes = gdf[class_value_field].unique()
                    if not quiet:
                        print(
                            f"Found {len(unique_classes)} unique classes: {unique_classes}"
                        )
                    # Create class mapping
                    class_to_id = {cls: i + 1 for i, cls in enumerate(unique_classes)}

                    # Populate COCO categories
                    if metadata_format == "COCO":
                        for cls_val in unique_classes:
                            coco_annotations["categories"].append(
                                {
                                    "id": class_to_id[cls_val],
                                    "name": str(cls_val),
                                    "supercategory": "object",
                                }
                            )
                else:
                    if not quiet:
                        print(
                            f"WARNING: '{class_value_field}' not found in vector data. Using default class ID 1."
                        )
                    class_to_id = {1: 1}  # Default mapping

                    # Populate COCO categories with default
                    if metadata_format == "COCO":
                        coco_annotations["categories"].append(
                            {
                                "id": 1,
                                "name": "object",
                                "supercategory": "object",
                            }
                        )
            except Exception as e:
                raise ValueError(f"Error processing vector data: {e}")

        # Create progress bar
        pbar = tqdm(
            total=min(total_tiles, max_tiles),
            desc="Generating tiles",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # Track statistics for summary
        stats = {
            "total_tiles": 0,
            "tiles_with_features": 0,
            "feature_pixels": 0,
            "errors": 0,
            "tile_coordinates": [],  # For overview image
        }

        # Process tiles
        tile_index = 0
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                if tile_index >= max_tiles:
                    break

                # Calculate window coordinates
                window_x = x * stride
                window_y = y * stride

                # Adjust for edge cases
                if window_x + tile_size > src.width:
                    window_x = src.width - tile_size
                if window_y + tile_size > src.height:
                    window_y = src.height - tile_size

                # Define window
                window = Window(window_x, window_y, tile_size, tile_size)

                # Get window transform and bounds
                window_transform = src.window_transform(window)

                # Calculate window bounds
                minx = window_transform[2]  # Upper left x
                maxy = window_transform[5]  # Upper left y
                maxx = minx + tile_size * window_transform[0]  # Add width
                miny = maxy + tile_size * window_transform[4]  # Add height

                window_bounds = box(minx, miny, maxx, maxy)

                # Store tile coordinates for overview
                if create_overview:
                    stats["tile_coordinates"].append(
                        {
                            "index": tile_index,
                            "x": window_x,
                            "y": window_y,
                            "bounds": [minx, miny, maxx, maxy],
                            "has_features": False,
                        }
                    )

                # Create label mask
                label_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                has_features = False

                # Process classification data to create labels (only if class data provided)
                if in_class_data is not None and is_class_data_raster:
                    # For raster class data
                    with rasterio.open(in_class_data) as class_src:
                        # Calculate window in class raster
                        src_bounds = src.bounds
                        class_bounds = class_src.bounds

                        # Check if windows overlap
                        if (
                            src_bounds.left > class_bounds.right
                            or src_bounds.right < class_bounds.left
                            or src_bounds.bottom > class_bounds.top
                            or src_bounds.top < class_bounds.bottom
                        ):
                            warnings.warn(
                                "Class raster and input raster do not overlap."
                            )
                        else:
                            # Get corresponding window in class raster
                            window_class = rasterio.windows.from_bounds(
                                minx, miny, maxx, maxy, class_src.transform
                            )

                            # Read label data
                            try:
                                label_data = class_src.read(
                                    1,
                                    window=window_class,
                                    boundless=True,
                                    out_shape=(tile_size, tile_size),
                                )

                                # Remap class values if needed
                                if class_to_id:
                                    remapped_data = np.zeros_like(label_data)
                                    for orig_val, new_val in class_to_id.items():
                                        remapped_data[label_data == orig_val] = new_val
                                    label_mask = remapped_data
                                else:
                                    label_mask = label_data

                                # Check if we have any features
                                if np.any(label_mask > 0):
                                    has_features = True
                                    stats["feature_pixels"] += np.count_nonzero(
                                        label_mask
                                    )
                            except Exception as e:
                                pbar.write(f"Error reading class raster window: {e}")
                                stats["errors"] += 1
                elif in_class_data is not None:
                    # For vector class data
                    # Find features that intersect with window
                    window_features = gdf[gdf.intersects(window_bounds)]

                    if len(window_features) > 0:
                        for idx, feature in window_features.iterrows():
                            # Get class value
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                class_id = class_to_id.get(class_val, 1)
                            else:
                                class_id = 1

                            # Get geometry in window coordinates
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                try:
                                    # Rasterize feature
                                    feature_mask = features.rasterize(
                                        [(geom, class_id)],
                                        out_shape=(tile_size, tile_size),
                                        transform=window_transform,
                                        fill=0,
                                        all_touched=all_touched,
                                    )

                                    # Add to label mask
                                    label_mask = np.maximum(label_mask, feature_mask)

                                    # Check if the feature was actually rasterized
                                    if np.any(feature_mask):
                                        has_features = True
                                        if create_overview and tile_index < len(
                                            stats["tile_coordinates"]
                                        ):
                                            stats["tile_coordinates"][tile_index][
                                                "has_features"
                                            ] = True
                                except Exception as e:
                                    pbar.write(f"Error rasterizing feature {idx}: {e}")
                                    stats["errors"] += 1

                # Skip tile if no features and skip_empty_tiles is True (only when class data provided)
                if in_class_data is not None and skip_empty_tiles and not has_features:
                    pbar.update(1)
                    tile_index += 1
                    continue

                # Read image data
                image_data = src.read(window=window)

                # Helper function to save a single tile (original or augmented)
                def save_tile(
                    img_data,
                    lbl_mask,
                    tile_id,
                    img_profile,
                    window_trans,
                    is_augmented=False,
                ):
                    """Save a single image and label tile."""
                    # Export image as GeoTIFF
                    image_path = os.path.join(image_dir, f"tile_{tile_id:06d}.tif")

                    # Update profile
                    img_profile_copy = img_profile.copy()
                    img_profile_copy.update(
                        {
                            "height": tile_size,
                            "width": tile_size,
                            "count": img_data.shape[0],
                            "transform": window_trans,
                        }
                    )

                    # Save image as GeoTIFF
                    try:
                        with rasterio.open(image_path, "w", **img_profile_copy) as dst:
                            dst.write(img_data)
                        stats["total_tiles"] += 1
                    except Exception as e:
                        pbar.write(f"ERROR saving image GeoTIFF: {e}")
                        stats["errors"] += 1
                        return

                    # Export label as GeoTIFF (only if class data provided)
                    if in_class_data is not None:
                        # Create profile for label GeoTIFF
                        label_profile = {
                            "driver": "GTiff",
                            "height": tile_size,
                            "width": tile_size,
                            "count": 1,
                            "dtype": "uint8",
                            "crs": src.crs,
                            "transform": window_trans,
                        }

                        label_path = os.path.join(label_dir, f"tile_{tile_id:06d}.tif")
                        try:
                            with rasterio.open(label_path, "w", **label_profile) as dst:
                                dst.write(lbl_mask.astype(np.uint8), 1)

                            if not is_augmented and np.any(lbl_mask > 0):
                                stats["tiles_with_features"] += 1
                                stats["feature_pixels"] += np.count_nonzero(lbl_mask)
                        except Exception as e:
                            pbar.write(f"ERROR saving label GeoTIFF: {e}")
                            stats["errors"] += 1

                # Save original tile
                save_tile(
                    image_data,
                    label_mask,
                    tile_index,
                    src.profile,
                    window_transform,
                    is_augmented=False,
                )

                # Generate and save augmented tiles if enabled
                if apply_augmentation:
                    for aug_idx in range(augmentation_count):
                        # Prepare image for augmentation (convert from CHW to HWC)
                        img_for_aug = np.transpose(image_data, (1, 2, 0))

                        # Ensure uint8 data type for albumentations
                        # Albumentations expects uint8 for most transforms
                        if not np.issubdtype(img_for_aug.dtype, np.uint8):
                            # If image is float, scale to 0-255 and convert to uint8
                            if np.issubdtype(img_for_aug.dtype, np.floating):
                                img_for_aug = (
                                    (img_for_aug * 255).clip(0, 255).astype(np.uint8)
                                )
                            else:
                                img_for_aug = img_for_aug.astype(np.uint8)

                        # Apply augmentation
                        try:
                            if in_class_data is not None:
                                # Augment both image and mask
                                augmented = augmentation_transforms(
                                    image=img_for_aug, mask=label_mask
                                )
                                aug_image = augmented["image"]
                                aug_mask = augmented["mask"]
                            else:
                                # Augment only image
                                augmented = augmentation_transforms(image=img_for_aug)
                                aug_image = augmented["image"]
                                aug_mask = label_mask

                            # Convert back from HWC to CHW
                            aug_image = np.transpose(aug_image, (2, 0, 1))

                            # Ensure correct dtype for saving
                            aug_image = aug_image.astype(image_data.dtype)

                            # Generate unique tile ID for augmented version
                            # Use a collision-free numbering scheme: (tile_index * (augmentation_count + 1)) + aug_idx + 1
                            aug_tile_id = (
                                (tile_index * (augmentation_count + 1)) + aug_idx + 1
                            )

                            # Save augmented tile
                            save_tile(
                                aug_image,
                                aug_mask,
                                aug_tile_id,
                                src.profile,
                                window_transform,
                                is_augmented=True,
                            )

                        except Exception as e:
                            pbar.write(
                                f"ERROR applying augmentation {aug_idx} to tile {tile_index}: {e}"
                            )
                            stats["errors"] += 1

                # Create annotations for object detection if using vector class data
                if (
                    in_class_data is not None
                    and not is_class_data_raster
                    and "gdf" in locals()
                    and len(window_features) > 0
                ):
                    if metadata_format == "PASCAL_VOC":
                        # Create XML annotation
                        root = ET.Element("annotation")
                        ET.SubElement(root, "folder").text = "images"
                        ET.SubElement(root, "filename").text = (
                            f"tile_{tile_index:06d}.tif"
                        )

                        size = ET.SubElement(root, "size")
                        ET.SubElement(size, "width").text = str(tile_size)
                        ET.SubElement(size, "height").text = str(tile_size)
                        ET.SubElement(size, "depth").text = str(image_data.shape[0])

                        # Add georeference information
                        geo = ET.SubElement(root, "georeference")
                        ET.SubElement(geo, "crs").text = str(src.crs)
                        ET.SubElement(geo, "transform").text = str(
                            window_transform
                        ).replace("\n", "")
                        ET.SubElement(geo, "bounds").text = (
                            f"{minx}, {miny}, {maxx}, {maxy}"
                        )

                        # Add objects
                        for idx, feature in window_features.iterrows():
                            # Get feature class
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                            else:
                                class_val = "object"

                            # Get geometry bounds in pixel coordinates
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                # Get bounds in world coordinates
                                minx_f, miny_f, maxx_f, maxy_f = geom.bounds

                                # Convert to pixel coordinates
                                col_min, row_min = ~window_transform * (minx_f, maxy_f)
                                col_max, row_max = ~window_transform * (maxx_f, miny_f)

                                # Ensure coordinates are within tile bounds
                                xmin = max(0, min(tile_size, int(col_min)))
                                ymin = max(0, min(tile_size, int(row_min)))
                                xmax = max(0, min(tile_size, int(col_max)))
                                ymax = max(0, min(tile_size, int(row_max)))

                                # Only add if the box has non-zero area
                                if xmax > xmin and ymax > ymin:
                                    obj = ET.SubElement(root, "object")
                                    ET.SubElement(obj, "name").text = str(class_val)
                                    ET.SubElement(obj, "difficult").text = "0"

                                    bbox = ET.SubElement(obj, "bndbox")
                                    ET.SubElement(bbox, "xmin").text = str(xmin)
                                    ET.SubElement(bbox, "ymin").text = str(ymin)
                                    ET.SubElement(bbox, "xmax").text = str(xmax)
                                    ET.SubElement(bbox, "ymax").text = str(ymax)

                        # Save XML
                        tree = ET.ElementTree(root)
                        xml_path = os.path.join(ann_dir, f"tile_{tile_index:06d}.xml")
                        tree.write(xml_path)

                    elif metadata_format == "COCO":
                        # Add image info
                        image_id = tile_index
                        coco_annotations["images"].append(
                            {
                                "id": image_id,
                                "file_name": f"tile_{tile_index:06d}.tif",
                                "width": tile_size,
                                "height": tile_size,
                                "crs": str(src.crs),
                                "transform": str(window_transform),
                            }
                        )

                        # Add annotations for each feature
                        for _, feature in window_features.iterrows():
                            # Get feature class
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                category_id = class_to_id.get(class_val, 1)
                            else:
                                category_id = 1

                            # Get geometry bounds
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                # Get bounds in world coordinates
                                minx_f, miny_f, maxx_f, maxy_f = geom.bounds

                                # Convert to pixel coordinates
                                col_min, row_min = ~window_transform * (minx_f, maxy_f)
                                col_max, row_max = ~window_transform * (maxx_f, miny_f)

                                # Ensure coordinates are within tile bounds
                                xmin = max(0, min(tile_size, int(col_min)))
                                ymin = max(0, min(tile_size, int(row_min)))
                                xmax = max(0, min(tile_size, int(col_max)))
                                ymax = max(0, min(tile_size, int(row_max)))

                                # Skip if box is too small
                                if xmax - xmin < 1 or ymax - ymin < 1:
                                    continue

                                width = xmax - xmin
                                height = ymax - ymin

                                # Add annotation
                                ann_id += 1
                                coco_annotations["annotations"].append(
                                    {
                                        "id": ann_id,
                                        "image_id": image_id,
                                        "category_id": category_id,
                                        "bbox": [xmin, ymin, width, height],
                                        "area": width * height,
                                        "iscrowd": 0,
                                    }
                                )

                    elif metadata_format == "YOLO":
                        # Create YOLO format annotations
                        yolo_annotations = []

                        for _, feature in window_features.iterrows():
                            # Get feature class
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                # YOLO uses 0-indexed class IDs
                                class_id = class_to_id.get(class_val, 1) - 1
                            else:
                                class_id = 0

                            # Get geometry bounds
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                # Get bounds in world coordinates
                                minx_f, miny_f, maxx_f, maxy_f = geom.bounds

                                # Convert to pixel coordinates
                                col_min, row_min = ~window_transform * (minx_f, maxy_f)
                                col_max, row_max = ~window_transform * (maxx_f, miny_f)

                                # Ensure coordinates are within tile bounds
                                xmin = max(0, min(tile_size, col_min))
                                ymin = max(0, min(tile_size, row_min))
                                xmax = max(0, min(tile_size, col_max))
                                ymax = max(0, min(tile_size, row_max))

                                # Skip if box is too small
                                if xmax - xmin < 1 or ymax - ymin < 1:
                                    continue

                                # Calculate normalized coordinates (YOLO format)
                                x_center = ((xmin + xmax) / 2) / tile_size
                                y_center = ((ymin + ymax) / 2) / tile_size
                                width = (xmax - xmin) / tile_size
                                height = (ymax - ymin) / tile_size

                                # Add YOLO annotation line
                                yolo_annotations.append(
                                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                                )

                        # Save YOLO annotations to text file
                        if yolo_annotations:
                            yolo_path = os.path.join(
                                label_dir, f"tile_{tile_index:06d}.txt"
                            )
                            with open(yolo_path, "w") as f:
                                f.write("\n".join(yolo_annotations))

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Generated: {stats['total_tiles']}, With features: {stats['tiles_with_features']}"
                )

                tile_index += 1
                if tile_index >= max_tiles:
                    break

            if tile_index >= max_tiles:
                break

        # Close progress bar
        pbar.close()

        # Save COCO annotations if applicable (only if class data provided)
        if in_class_data is not None and metadata_format == "COCO":
            try:
                with open(os.path.join(ann_dir, "instances.json"), "w") as f:
                    json.dump(coco_annotations, f, indent=2)
                if not quiet:
                    print(
                        f"Saved COCO annotations: {len(coco_annotations['images'])} images, "
                        f"{len(coco_annotations['annotations'])} annotations, "
                        f"{len(coco_annotations['categories'])} categories"
                    )
            except Exception as e:
                if not quiet:
                    print(f"ERROR saving COCO annotations: {e}")
                stats["errors"] += 1

        # Save YOLO classes file if applicable (only if class data provided)
        if in_class_data is not None and metadata_format == "YOLO":
            try:
                # Create classes.txt with class names
                classes_path = os.path.join(out_folder, "classes.txt")
                # Sort by class ID to ensure correct order
                sorted_classes = sorted(class_to_id.items(), key=lambda x: x[1])
                with open(classes_path, "w") as f:
                    for class_val, _ in sorted_classes:
                        f.write(f"{class_val}\n")
                if not quiet:
                    print(f"Saved YOLO classes file with {len(class_to_id)} classes")
            except Exception as e:
                if not quiet:
                    print(f"ERROR saving YOLO classes file: {e}")
                stats["errors"] += 1

        # Create overview image if requested
        if create_overview and stats["tile_coordinates"]:
            try:
                from geoai.utils import create_overview_image

                create_overview_image(
                    src,
                    stats["tile_coordinates"],
                    os.path.join(out_folder, "overview.png"),
                    tile_size,
                    stride,
                )
            except Exception as e:
                print(f"Failed to create overview image: {e}")

        # Report results
        if not quiet:
            print("\n------- Export Summary -------")
            print(f"Total tiles exported: {stats['total_tiles']}")
            if in_class_data is not None:
                print(
                    f"Tiles with features: {stats['tiles_with_features']} ({stats['tiles_with_features']/max(1, stats['total_tiles'])*100:.1f}%)"
                )
                if stats["tiles_with_features"] > 0:
                    print(
                        f"Average feature pixels per tile: {stats['feature_pixels']/stats['tiles_with_features']:.1f}"
                    )
            if stats["errors"] > 0:
                print(f"Errors encountered: {stats['errors']}")
            print(f"Output saved to: {out_folder}")

            # Verify georeference in a sample image and label
            if stats["total_tiles"] > 0:
                print("\n------- Georeference Verification -------")
                sample_image = os.path.join(image_dir, f"tile_0.tif")

                if os.path.exists(sample_image):
                    try:
                        with rasterio.open(sample_image) as img:
                            print(f"Image CRS: {img.crs}")
                            print(f"Image transform: {img.transform}")
                            print(
                                f"Image has georeference: {img.crs is not None and img.transform is not None}"
                            )
                            print(
                                f"Image dimensions: {img.width}x{img.height}, {img.count} bands, {img.dtypes[0]} type"
                            )
                    except Exception as e:
                        print(f"Error verifying image georeference: {e}")

                # Only verify label if class data was provided
                if in_class_data is not None:
                    sample_label = os.path.join(label_dir, f"tile_0.tif")
                    if os.path.exists(sample_label):
                        try:
                            with rasterio.open(sample_label) as lbl:
                                print(f"Label CRS: {lbl.crs}")
                                print(f"Label transform: {lbl.transform}")
                                print(
                                    f"Label has georeference: {lbl.crs is not None and lbl.transform is not None}"
                                )
                                print(
                                    f"Label dimensions: {lbl.width}x{lbl.height}, {lbl.count} bands, {lbl.dtypes[0]} type"
                                )
                        except Exception as e:
                            print(f"Error verifying label georeference: {e}")

        # Return statistics dictionary for further processing if needed
        return stats


def export_geotiff_tiles_batch(
    images_folder,
    masks_folder=None,
    masks_file=None,
    output_folder=None,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    quiet=False,
    all_touched=True,
    skip_empty_tiles=False,
    image_extensions=None,
    mask_extensions=None,
    match_by_name=False,
    metadata_format="PASCAL_VOC",
) -> Dict[str, Any]:
    """
    Export georeferenced GeoTIFF tiles from images and optionally masks.

    This function supports four modes:
    1. Images only (no masks) - when neither masks_file nor masks_folder is provided
    2. Single vector file covering all images (masks_file parameter)
    3. Multiple vector files, one per image (masks_folder parameter)
    4. Multiple raster mask files (masks_folder parameter)

    For mode 1 (images only), only image tiles will be exported without labels.

    For mode 2 (single vector file), specify masks_file path. The function will
    use spatial intersection to determine which features apply to each image.

    For mode 3/4 (multiple mask files), specify masks_folder path. Images and masks
    are paired either by matching filenames (match_by_name=True) or by sorted order
    (match_by_name=False).

    All image tiles are saved to a single 'images' folder and all mask tiles (if provided)
    to a single 'masks' folder within the output directory.

    Args:
        images_folder (str): Path to folder containing raster images
        masks_folder (str, optional): Path to folder containing classification masks/vectors.
            Use this for multiple mask files (one per image or raster masks). If not provided
            and masks_file is also not provided, only image tiles will be exported.
        masks_file (str, optional): Path to a single vector file covering all images.
            Use this for a single GeoJSON/Shapefile that covers multiple images. If not provided
            and masks_folder is also not provided, only image tiles will be exported.
        output_folder (str, optional): Path to output folder. If None, creates 'tiles'
            subfolder in images_folder.
        tile_size (int): Size of tiles in pixels (square)
        stride (int): Step size between tiles
        class_value_field (str): Field containing class values (for vector data)
        buffer_radius (float): Buffer to add around features (in units of the CRS)
        max_tiles (int): Maximum number of tiles to process per image (None for all)
        quiet (bool): If True, suppress non-essential output
        all_touched (bool): Whether to use all_touched=True in rasterization (for vector data)
        create_overview (bool): Whether to create an overview image of all tiles
        skip_empty_tiles (bool): If True, skip tiles with no features
        image_extensions (list): List of image file extensions to process (default: common raster formats)
        mask_extensions (list): List of mask file extensions to process (default: common raster/vector formats)
        match_by_name (bool): If True, match image and mask files by base filename.
            If False, match by sorted order (alphabetically). Only applies when masks_folder is used.
        metadata_format (str): Annotation format - "PASCAL_VOC" (XML), "COCO" (JSON), or "YOLO" (TXT).
            Default is "PASCAL_VOC".

    Returns:
        Dict[str, Any]: Dictionary containing batch processing statistics

    Raises:
        ValueError: If no images found, or if masks_folder and masks_file are both specified,
            or if counts don't match when using masks_folder with match_by_name=False.

    Examples:
        # Images only (no masks)
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     output_folder='output/tiles'
        ... )

        # Single vector file covering all images
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     masks_file='data/buildings.geojson',
        ...     output_folder='output/tiles'
        ... )

        # Multiple vector files, matched by filename
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     masks_folder='data/masks',
        ...     output_folder='output/tiles',
        ...     match_by_name=True
        ... )

        # Multiple mask files, matched by sorted order
        >>> stats = export_geotiff_tiles_batch(
        ...     images_folder='data/images',
        ...     masks_folder='data/masks',
        ...     output_folder='output/tiles',
        ...     match_by_name=False
        ... )
    """

    import logging

    logging.getLogger("rasterio").setLevel(logging.ERROR)

    # Validate input parameters
    if masks_folder is not None and masks_file is not None:
        raise ValueError(
            "Cannot specify both masks_folder and masks_file. Please use only one."
        )

    # Default output folder if not specified
    if output_folder is None:
        output_folder = os.path.join(images_folder, "tiles")

    # Default extensions if not provided
    if image_extensions is None:
        image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".img"]
    if mask_extensions is None:
        mask_extensions = [
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
            ".png",
            ".jp2",
            ".img",
            ".shp",
            ".geojson",
            ".gpkg",
            ".geoparquet",
            ".json",
        ]

    # Convert extensions to lowercase for comparison
    image_extensions = [ext.lower() for ext in image_extensions]
    mask_extensions = [ext.lower() for ext in mask_extensions]

    # Create output folder structure
    os.makedirs(output_folder, exist_ok=True)
    output_images_dir = os.path.join(output_folder, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # Only create masks directory if masks are provided
    output_masks_dir = None
    if masks_folder is not None or masks_file is not None:
        output_masks_dir = os.path.join(output_folder, "masks")
        os.makedirs(output_masks_dir, exist_ok=True)

    # Create annotation directory based on metadata format (only if masks are provided)
    ann_dir = None
    if (masks_folder is not None or masks_file is not None) and metadata_format in [
        "PASCAL_VOC",
        "COCO",
    ]:
        ann_dir = os.path.join(output_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)

    # Initialize COCO annotations dictionary (only if masks are provided)
    coco_annotations = None
    if (
        masks_folder is not None or masks_file is not None
    ) and metadata_format == "COCO":
        coco_annotations = {"images": [], "annotations": [], "categories": []}

    # Initialize YOLO class set (only if masks are provided)
    yolo_classes = (
        set()
        if (masks_folder is not None or masks_file is not None)
        and metadata_format == "YOLO"
        else None
    )

    # Get list of image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(images_folder, f"*{ext}")
        image_files.extend(glob.glob(pattern))

    # Sort files for consistent processing
    image_files.sort()

    if not image_files:
        raise ValueError(
            f"No image files found in {images_folder} with extensions {image_extensions}"
        )

    # Handle different mask input modes
    use_single_mask_file = masks_file is not None
    has_masks = masks_file is not None or masks_folder is not None
    mask_files = []
    image_mask_pairs = []

    if not has_masks:
        # Mode 0: No masks - create pairs with None for mask
        for image_file in image_files:
            image_mask_pairs.append((image_file, None, None))

    elif use_single_mask_file:
        # Mode 1: Single vector file covering all images
        if not os.path.exists(masks_file):
            raise ValueError(f"Mask file not found: {masks_file}")

        # Load the single mask file once - will be spatially filtered per image
        single_mask_gdf = gpd.read_file(masks_file)

        if not quiet:
            print(f"Using single mask file: {masks_file}")
            print(
                f"Mask contains {len(single_mask_gdf)} features in CRS: {single_mask_gdf.crs}"
            )

        # Create pairs with the same mask file for all images
        for image_file in image_files:
            image_mask_pairs.append((image_file, masks_file, single_mask_gdf))

    else:
        # Mode 2/3: Multiple mask files (vector or raster)
        # Get list of mask files
        for ext in mask_extensions:
            pattern = os.path.join(masks_folder, f"*{ext}")
            mask_files.extend(glob.glob(pattern))

        # Sort files for consistent processing
        mask_files.sort()

        if not mask_files:
            raise ValueError(
                f"No mask files found in {masks_folder} with extensions {mask_extensions}"
            )

        # Match images to masks
        if match_by_name:
            # Match by base filename
            image_dict = {
                os.path.splitext(os.path.basename(f))[0]: f for f in image_files
            }
            mask_dict = {
                os.path.splitext(os.path.basename(f))[0]: f for f in mask_files
            }

            # Find matching pairs
            for img_base, img_path in image_dict.items():
                if img_base in mask_dict:
                    image_mask_pairs.append((img_path, mask_dict[img_base], None))
                else:
                    if not quiet:
                        print(f"Warning: No mask found for image {img_base}")

            if not image_mask_pairs:
                # Provide detailed error message with found files
                image_bases = list(image_dict.keys())
                mask_bases = list(mask_dict.keys())
                error_msg = (
                    "No matching image-mask pairs found when matching by filename. "
                    "Check that image and mask files have matching base names.\n"
                    f"Found {len(image_bases)} image(s): "
                    f"{', '.join(image_bases[:5]) if image_bases else 'None found'}"
                    f"{'...' if len(image_bases) > 5 else ''}\n"
                    f"Found {len(mask_bases)} mask(s): "
                    f"{', '.join(mask_bases[:5]) if mask_bases else 'None found'}"
                    f"{'...' if len(mask_bases) > 5 else ''}\n"
                    "Tip: Set match_by_name=False to match by sorted order, or ensure filenames match."
                )
                raise ValueError(error_msg)

        else:
            # Match by sorted order
            if len(image_files) != len(mask_files):
                raise ValueError(
                    f"Number of image files ({len(image_files)}) does not match "
                    f"number of mask files ({len(mask_files)}) when matching by sorted order. "
                    f"Use match_by_name=True for filename-based matching."
                )

            # Create pairs by sorted order
            for image_file, mask_file in zip(image_files, mask_files):
                image_mask_pairs.append((image_file, mask_file, None))

    # Initialize batch statistics
    batch_stats = {
        "total_image_pairs": 0,
        "processed_pairs": 0,
        "total_tiles": 0,
        "tiles_with_features": 0,
        "errors": 0,
        "processed_files": [],
        "failed_files": [],
    }

    if not quiet:
        if not has_masks:
            print(
                f"Found {len(image_files)} image files to process (images only, no masks)"
            )
        elif use_single_mask_file:
            print(f"Found {len(image_files)} image files to process")
            print(f"Using single mask file: {masks_file}")
        else:
            print(f"Found {len(image_mask_pairs)} matching image-mask pairs to process")
            print(f"Processing batch from {images_folder} and {masks_folder}")
        print(f"Output folder: {output_folder}")
        print("-" * 60)

    # Global tile counter for unique naming
    global_tile_counter = 0

    # Process each image-mask pair
    for idx, (image_file, mask_file, mask_gdf) in enumerate(
        tqdm(
            image_mask_pairs,
            desc="Processing image pairs",
            disable=quiet,
        )
    ):
        batch_stats["total_image_pairs"] += 1

        # Get base filename without extension for naming (use image filename)
        base_name = os.path.splitext(os.path.basename(image_file))[0]

        try:
            if not quiet:
                print(f"\nProcessing: {base_name}")
                print(f"  Image: {os.path.basename(image_file)}")
                if mask_file is not None:
                    if use_single_mask_file:
                        print(
                            f"  Mask: {os.path.basename(mask_file)} (spatially filtered)"
                        )
                    else:
                        print(f"  Mask: {os.path.basename(mask_file)}")
                else:
                    print(f"  Mask: None (images only)")

            # Process the image-mask pair
            tiles_generated = _process_image_mask_pair(
                image_file=image_file,
                mask_file=mask_file,
                base_name=base_name,
                output_images_dir=output_images_dir,
                output_masks_dir=output_masks_dir,
                global_tile_counter=global_tile_counter,
                tile_size=tile_size,
                stride=stride,
                class_value_field=class_value_field,
                buffer_radius=buffer_radius,
                max_tiles=max_tiles,
                all_touched=all_touched,
                skip_empty_tiles=skip_empty_tiles,
                quiet=quiet,
                mask_gdf=mask_gdf,  # Pass pre-loaded GeoDataFrame if using single mask
                use_single_mask_file=use_single_mask_file,
                metadata_format=metadata_format,
                ann_dir=(
                    ann_dir
                    if "ann_dir" in locals()
                    and metadata_format in ["PASCAL_VOC", "COCO"]
                    else None
                ),
            )

            # Update counters
            global_tile_counter += tiles_generated["total_tiles"]

            # Update batch statistics
            batch_stats["processed_pairs"] += 1
            batch_stats["total_tiles"] += tiles_generated["total_tiles"]
            batch_stats["tiles_with_features"] += tiles_generated["tiles_with_features"]
            batch_stats["errors"] += tiles_generated["errors"]

            batch_stats["processed_files"].append(
                {
                    "image": image_file,
                    "mask": mask_file,
                    "base_name": base_name,
                    "tiles_generated": tiles_generated["total_tiles"],
                    "tiles_with_features": tiles_generated["tiles_with_features"],
                }
            )

            # Aggregate COCO annotations
            if metadata_format == "COCO" and "coco_data" in tiles_generated:
                coco_data = tiles_generated["coco_data"]
                # Add images and annotations
                coco_annotations["images"].extend(coco_data.get("images", []))
                coco_annotations["annotations"].extend(coco_data.get("annotations", []))
                # Merge categories (avoid duplicates)
                for cat in coco_data.get("categories", []):
                    if not any(
                        c["id"] == cat["id"] for c in coco_annotations["categories"]
                    ):
                        coco_annotations["categories"].append(cat)

            # Aggregate YOLO classes
            if metadata_format == "YOLO" and "yolo_classes" in tiles_generated:
                yolo_classes.update(tiles_generated["yolo_classes"])

        except Exception as e:
            if not quiet:
                print(f"ERROR processing {base_name}: {e}")
            batch_stats["failed_files"].append(
                {"image": image_file, "mask": mask_file, "error": str(e)}
            )
            batch_stats["errors"] += 1

    # Save aggregated COCO annotations
    if metadata_format == "COCO" and coco_annotations:
        import json

        coco_path = os.path.join(ann_dir, "instances.json")
        with open(coco_path, "w") as f:
            json.dump(coco_annotations, f, indent=2)
        if not quiet:
            print(f"\nSaved COCO annotations: {coco_path}")
            print(
                f"  Images: {len(coco_annotations['images'])}, "
                f"Annotations: {len(coco_annotations['annotations'])}, "
                f"Categories: {len(coco_annotations['categories'])}"
            )

    # Save aggregated YOLO classes
    if metadata_format == "YOLO" and yolo_classes:
        classes_path = os.path.join(output_folder, "labels", "classes.txt")
        os.makedirs(os.path.dirname(classes_path), exist_ok=True)
        sorted_classes = sorted(yolo_classes)
        with open(classes_path, "w") as f:
            for cls in sorted_classes:
                f.write(f"{cls}\n")
        if not quiet:
            print(f"\nSaved YOLO classes: {classes_path}")
            print(f"  Total classes: {len(sorted_classes)}")

    # Print batch summary
    if not quiet:
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total image pairs found: {batch_stats['total_image_pairs']}")
        print(f"Successfully processed: {batch_stats['processed_pairs']}")
        print(f"Failed to process: {len(batch_stats['failed_files'])}")
        print(f"Total tiles generated: {batch_stats['total_tiles']}")
        print(f"Tiles with features: {batch_stats['tiles_with_features']}")

        if batch_stats["total_tiles"] > 0:
            feature_percentage = (
                batch_stats["tiles_with_features"] / batch_stats["total_tiles"]
            ) * 100
            print(f"Feature percentage: {feature_percentage:.1f}%")

        if batch_stats["errors"] > 0:
            print(f"Total errors: {batch_stats['errors']}")

        print(f"Output saved to: {output_folder}")
        print(f"  Images: {output_images_dir}")
        if output_masks_dir is not None:
            print(f"  Masks: {output_masks_dir}")
            if metadata_format in ["PASCAL_VOC", "COCO"] and ann_dir is not None:
                print(f"  Annotations: {ann_dir}")
            elif metadata_format == "YOLO":
                print(f"  Labels: {os.path.join(output_folder, 'labels')}")

        # List failed files if any
        if batch_stats["failed_files"]:
            print(f"\nFailed files:")
            for failed in batch_stats["failed_files"]:
                print(f"  - {os.path.basename(failed['image'])}: {failed['error']}")

    return batch_stats


def _process_image_mask_pair(
    image_file,
    mask_file,
    base_name,
    output_images_dir,
    output_masks_dir,
    global_tile_counter,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    all_touched=True,
    skip_empty_tiles=False,
    quiet=False,
    mask_gdf=None,
    use_single_mask_file=False,
    metadata_format="PASCAL_VOC",
    ann_dir=None,
):
    """
    Process a single image-mask pair and save tiles directly to output directories.

    Args:
        mask_gdf (GeoDataFrame, optional): Pre-loaded GeoDataFrame when using single mask file
        use_single_mask_file (bool): If True, spatially filter mask_gdf to image bounds

    Returns:
        dict: Statistics for this image-mask pair
    """
    import warnings

    # Determine if mask data is raster or vector (only if mask_file is provided)
    is_class_data_raster = False
    if mask_file is not None and isinstance(mask_file, str):
        file_ext = Path(mask_file).suffix.lower()
        # Common raster extensions
        if file_ext in [".tif", ".tiff", ".img", ".jp2", ".png", ".bmp", ".gif"]:
            try:
                with rasterio.open(mask_file) as src:
                    is_class_data_raster = True
            except Exception:
                is_class_data_raster = False

    # Track statistics
    stats = {
        "total_tiles": 0,
        "tiles_with_features": 0,
        "errors": 0,
    }

    # Initialize COCO/YOLO tracking for this image
    if metadata_format == "COCO":
        stats["coco_data"] = {"images": [], "annotations": [], "categories": []}
        coco_ann_id = 0
    if metadata_format == "YOLO":
        stats["yolo_classes"] = set()

    # Open the input raster
    with rasterio.open(image_file) as src:
        # Calculate number of tiles
        num_tiles_x = math.ceil((src.width - tile_size) / stride) + 1
        num_tiles_y = math.ceil((src.height - tile_size) / stride) + 1
        total_tiles = num_tiles_x * num_tiles_y

        if max_tiles is None:
            max_tiles = total_tiles

        # Process classification data (only if mask_file is provided)
        class_to_id = {}

        if mask_file is not None and is_class_data_raster:
            # Load raster class data
            with rasterio.open(mask_file) as class_src:
                # Check if raster CRS matches
                if class_src.crs != src.crs:
                    warnings.warn(
                        f"CRS mismatch: Class raster ({class_src.crs}) doesn't match input raster ({src.crs}). "
                        f"Results may be misaligned."
                    )

                # Get unique values from raster
                sample_data = class_src.read(
                    1,
                    out_shape=(
                        1,
                        min(class_src.height, 1000),
                        min(class_src.width, 1000),
                    ),
                )

                unique_classes = np.unique(sample_data)
                unique_classes = unique_classes[
                    unique_classes > 0
                ]  # Remove 0 as it's typically background

                # Create class mapping
                class_to_id = {int(cls): i + 1 for i, cls in enumerate(unique_classes)}
        elif mask_file is not None:
            # Load vector class data
            try:
                if use_single_mask_file and mask_gdf is not None:
                    # Using pre-loaded single mask file - spatially filter to image bounds
                    # Get image bounds
                    image_bounds = box(*src.bounds)
                    image_gdf = gpd.GeoDataFrame(
                        {"geometry": [image_bounds]}, crs=src.crs
                    )

                    # Reproject mask if needed
                    if mask_gdf.crs != src.crs:
                        mask_gdf_reprojected = mask_gdf.to_crs(src.crs)
                    else:
                        mask_gdf_reprojected = mask_gdf

                    # Spatially filter features that intersect with image bounds
                    gdf = mask_gdf_reprojected[
                        mask_gdf_reprojected.intersects(image_bounds)
                    ].copy()

                    if not quiet and len(gdf) > 0:
                        print(
                            f"  Filtered to {len(gdf)} features intersecting image bounds"
                        )
                else:
                    # Load individual mask file
                    gdf = gpd.read_file(mask_file)

                    # Always reproject to match raster CRS
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)

                # Apply buffer if specified
                if buffer_radius > 0:
                    gdf["geometry"] = gdf.buffer(buffer_radius)

                # Check if class_value_field exists
                if class_value_field in gdf.columns:
                    unique_classes = gdf[class_value_field].unique()
                    # Create class mapping
                    class_to_id = {cls: i + 1 for i, cls in enumerate(unique_classes)}
                else:
                    class_to_id = {1: 1}  # Default mapping
            except Exception as e:
                raise ValueError(f"Error processing vector data: {e}")

        # Process tiles
        tile_index = 0
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate window coordinates
                window_x = x * stride
                window_y = y * stride

                # Adjust for edge cases
                if window_x + tile_size > src.width:
                    window_x = src.width - tile_size
                if window_y + tile_size > src.height:
                    window_y = src.height - tile_size

                # Define window
                window = Window(window_x, window_y, tile_size, tile_size)

                # Get window transform and bounds
                window_transform = src.window_transform(window)

                # Calculate window bounds
                minx = window_transform[2]  # Upper left x
                maxy = window_transform[5]  # Upper left y
                maxx = minx + tile_size * window_transform[0]  # Add width
                miny = maxy + tile_size * window_transform[4]  # Add height

                window_bounds = box(minx, miny, maxx, maxy)

                # Create label mask (only if mask_file is provided)
                label_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                has_features = False

                # Process classification data to create labels (only if mask_file is provided)
                if mask_file is not None and is_class_data_raster:
                    # For raster class data
                    with rasterio.open(mask_file) as class_src:
                        # Get corresponding window in class raster
                        window_class = rasterio.windows.from_bounds(
                            minx, miny, maxx, maxy, class_src.transform
                        )

                        # Read label data
                        try:
                            label_data = class_src.read(
                                1,
                                window=window_class,
                                boundless=True,
                                out_shape=(tile_size, tile_size),
                            )

                            # Remap class values if needed
                            if class_to_id:
                                remapped_data = np.zeros_like(label_data)
                                for orig_val, new_val in class_to_id.items():
                                    remapped_data[label_data == orig_val] = new_val
                                label_mask = remapped_data
                            else:
                                label_mask = label_data

                            # Check if we have any features
                            if np.any(label_mask > 0):
                                has_features = True
                        except Exception as e:
                            if not quiet:
                                print(f"Error reading class raster window: {e}")
                            stats["errors"] += 1
                elif mask_file is not None:
                    # For vector class data
                    # Find features that intersect with window
                    window_features = gdf[gdf.intersects(window_bounds)]

                    if len(window_features) > 0:
                        for idx, feature in window_features.iterrows():
                            # Get class value
                            if class_value_field in feature:
                                class_val = feature[class_value_field]
                                class_id = class_to_id.get(class_val, 1)
                            else:
                                class_id = 1

                            # Get geometry in window coordinates
                            geom = feature.geometry.intersection(window_bounds)
                            if not geom.is_empty:
                                try:
                                    # Rasterize feature
                                    feature_mask = features.rasterize(
                                        [(geom, class_id)],
                                        out_shape=(tile_size, tile_size),
                                        transform=window_transform,
                                        fill=0,
                                        all_touched=all_touched,
                                    )

                                    # Add to label mask
                                    label_mask = np.maximum(label_mask, feature_mask)

                                    # Check if the feature was actually rasterized
                                    if np.any(feature_mask):
                                        has_features = True
                                except Exception as e:
                                    if not quiet:
                                        print(f"Error rasterizing feature {idx}: {e}")
                                    stats["errors"] += 1

                # Skip tile if no features and skip_empty_tiles is True (only applies when masks are provided)
                if mask_file is not None and skip_empty_tiles and not has_features:
                    continue

                # Check if we've reached max_tiles before saving
                if tile_index >= max_tiles:
                    break

                # Generate unique tile name
                tile_name = f"{base_name}_{global_tile_counter + tile_index:06d}"

                # Read image data
                image_data = src.read(window=window)

                # Export image as GeoTIFF
                image_path = os.path.join(output_images_dir, f"{tile_name}.tif")

                # Create profile for image GeoTIFF
                image_profile = src.profile.copy()
                image_profile.update(
                    {
                        "height": tile_size,
                        "width": tile_size,
                        "count": image_data.shape[0],
                        "transform": window_transform,
                    }
                )

                # Save image as GeoTIFF
                try:
                    with rasterio.open(image_path, "w", **image_profile) as dst:
                        dst.write(image_data)
                    stats["total_tiles"] += 1
                except Exception as e:
                    if not quiet:
                        print(f"ERROR saving image GeoTIFF: {e}")
                    stats["errors"] += 1

                # Export label as GeoTIFF (only if mask_file and output_masks_dir are provided)
                if mask_file is not None and output_masks_dir is not None:
                    # Create profile for label GeoTIFF
                    label_profile = {
                        "driver": "GTiff",
                        "height": tile_size,
                        "width": tile_size,
                        "count": 1,
                        "dtype": "uint8",
                        "crs": src.crs,
                        "transform": window_transform,
                    }

                    label_path = os.path.join(output_masks_dir, f"{tile_name}.tif")
                    try:
                        with rasterio.open(label_path, "w", **label_profile) as dst:
                            dst.write(label_mask.astype(np.uint8), 1)

                        if has_features:
                            stats["tiles_with_features"] += 1
                    except Exception as e:
                        if not quiet:
                            print(f"ERROR saving label GeoTIFF: {e}")
                        stats["errors"] += 1

                # Generate annotation metadata based on format (only if mask_file is provided)
                if (
                    mask_file is not None
                    and metadata_format == "PASCAL_VOC"
                    and ann_dir
                ):
                    # Create PASCAL VOC XML annotation
                    from lxml import etree as ET

                    annotation = ET.Element("annotation")
                    ET.SubElement(annotation, "folder").text = os.path.basename(
                        output_images_dir
                    )
                    ET.SubElement(annotation, "filename").text = f"{tile_name}.tif"
                    ET.SubElement(annotation, "path").text = image_path

                    source = ET.SubElement(annotation, "source")
                    ET.SubElement(source, "database").text = "GeoAI"

                    size = ET.SubElement(annotation, "size")
                    ET.SubElement(size, "width").text = str(tile_size)
                    ET.SubElement(size, "height").text = str(tile_size)
                    ET.SubElement(size, "depth").text = str(image_data.shape[0])

                    ET.SubElement(annotation, "segmented").text = "1"

                    # Find connected components for instance segmentation
                    from scipy import ndimage

                    for class_id in np.unique(label_mask):
                        if class_id == 0:
                            continue

                        class_mask = (label_mask == class_id).astype(np.uint8)
                        labeled_array, num_features = ndimage.label(class_mask)

                        for instance_id in range(1, num_features + 1):
                            instance_mask = labeled_array == instance_id
                            coords = np.argwhere(instance_mask)

                            if len(coords) == 0:
                                continue

                            ymin, xmin = coords.min(axis=0)
                            ymax, xmax = coords.max(axis=0)

                            obj = ET.SubElement(annotation, "object")
                            class_name = next(
                                (k for k, v in class_to_id.items() if v == class_id),
                                str(class_id),
                            )
                            ET.SubElement(obj, "name").text = str(class_name)
                            ET.SubElement(obj, "pose").text = "Unspecified"
                            ET.SubElement(obj, "truncated").text = "0"
                            ET.SubElement(obj, "difficult").text = "0"

                            bndbox = ET.SubElement(obj, "bndbox")
                            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                            ET.SubElement(bndbox, "ymax").text = str(int(ymax))

                    # Save XML file
                    xml_path = os.path.join(ann_dir, f"{tile_name}.xml")
                    tree = ET.ElementTree(annotation)
                    tree.write(xml_path, pretty_print=True, encoding="utf-8")

                elif mask_file is not None and metadata_format == "COCO":
                    # Add COCO image entry
                    image_id = int(global_tile_counter + tile_index)
                    stats["coco_data"]["images"].append(
                        {
                            "id": image_id,
                            "file_name": f"{tile_name}.tif",
                            "width": int(tile_size),
                            "height": int(tile_size),
                        }
                    )

                    # Add COCO categories (only once per unique class)
                    for class_val, class_id in class_to_id.items():
                        if not any(
                            c["id"] == class_id
                            for c in stats["coco_data"]["categories"]
                        ):
                            stats["coco_data"]["categories"].append(
                                {
                                    "id": int(class_id),
                                    "name": str(class_val),
                                    "supercategory": "object",
                                }
                            )

                    # Add COCO annotations (instance segmentation)
                    from scipy import ndimage
                    from skimage import measure

                    for class_id in np.unique(label_mask):
                        if class_id == 0:
                            continue

                        class_mask = (label_mask == class_id).astype(np.uint8)
                        labeled_array, num_features = ndimage.label(class_mask)

                        for instance_id in range(1, num_features + 1):
                            instance_mask = (labeled_array == instance_id).astype(
                                np.uint8
                            )
                            coords = np.argwhere(instance_mask)

                            if len(coords) == 0:
                                continue

                            ymin, xmin = coords.min(axis=0)
                            ymax, xmax = coords.max(axis=0)

                            bbox = [
                                int(xmin),
                                int(ymin),
                                int(xmax - xmin),
                                int(ymax - ymin),
                            ]
                            area = int(np.sum(instance_mask))

                            # Find contours for segmentation
                            contours = measure.find_contours(instance_mask, 0.5)
                            segmentation = []
                            for contour in contours:
                                contour = np.flip(contour, axis=1)
                                segmentation_points = contour.ravel().tolist()
                                if len(segmentation_points) >= 6:
                                    segmentation.append(segmentation_points)

                            if segmentation:
                                stats["coco_data"]["annotations"].append(
                                    {
                                        "id": int(coco_ann_id),
                                        "image_id": int(image_id),
                                        "category_id": int(class_id),
                                        "bbox": bbox,
                                        "area": area,
                                        "segmentation": segmentation,
                                        "iscrowd": 0,
                                    }
                                )
                                coco_ann_id += 1

                elif mask_file is not None and metadata_format == "YOLO":
                    # Create YOLO labels directory if needed
                    labels_dir = os.path.join(
                        os.path.dirname(output_images_dir), "labels"
                    )
                    os.makedirs(labels_dir, exist_ok=True)

                    # Generate YOLO annotation file
                    yolo_path = os.path.join(labels_dir, f"{tile_name}.txt")
                    from scipy import ndimage

                    with open(yolo_path, "w") as yolo_file:
                        for class_id in np.unique(label_mask):
                            if class_id == 0:
                                continue

                            # Track class for classes.txt
                            class_name = next(
                                (k for k, v in class_to_id.items() if v == class_id),
                                str(class_id),
                            )
                            stats["yolo_classes"].add(class_name)

                            class_mask = (label_mask == class_id).astype(np.uint8)
                            labeled_array, num_features = ndimage.label(class_mask)

                            for instance_id in range(1, num_features + 1):
                                instance_mask = labeled_array == instance_id
                                coords = np.argwhere(instance_mask)

                                if len(coords) == 0:
                                    continue

                                ymin, xmin = coords.min(axis=0)
                                ymax, xmax = coords.max(axis=0)

                                # Convert to YOLO format (normalized center coordinates)
                                x_center = ((xmin + xmax) / 2) / tile_size
                                y_center = ((ymin + ymax) / 2) / tile_size
                                width = (xmax - xmin) / tile_size
                                height = (ymax - ymin) / tile_size

                                # YOLO uses 0-based class indices
                                yolo_class_id = class_id - 1
                                yolo_file.write(
                                    f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                )

                tile_index += 1
                if tile_index >= max_tiles:
                    break

            if tile_index >= max_tiles:
                break

    return stats


def export_training_data(
    in_raster,
    out_folder,
    in_class_data,
    image_chip_format="GEOTIFF",
    tile_size_x=256,
    tile_size_y=256,
    stride_x=None,
    stride_y=None,
    output_nofeature_tiles=True,
    metadata_format="PASCAL_VOC",
    start_index=0,
    class_value_field="class",
    buffer_radius=0,
    in_mask_polygons=None,
    rotation_angle=0,
    reference_system=None,
    blacken_around_feature=False,
    crop_mode="FIXED_SIZE",  # Implemented but not fully used yet
    in_raster2=None,
    in_instance_data=None,
    instance_class_value_field=None,  # Implemented but not fully used yet
    min_polygon_overlap_ratio=0.0,
    all_touched=True,
    save_geotiff=True,
    quiet=False,
):
    """
    Export training data for deep learning using TorchGeo with progress bar.

    Args:
        in_raster (str): Path to input raster image.
        out_folder (str): Output folder path where chips and labels will be saved.
        in_class_data (str): Path to vector file containing class polygons.
        image_chip_format (str): Output image format (PNG, JPEG, TIFF, GEOTIFF).
        tile_size_x (int): Width of image chips in pixels.
        tile_size_y (int): Height of image chips in pixels.
        stride_x (int): Horizontal stride between chips. If None, uses tile_size_x.
        stride_y (int): Vertical stride between chips. If None, uses tile_size_y.
        output_nofeature_tiles (bool): Whether to export chips without features.
        metadata_format (str): Output metadata format (PASCAL_VOC, KITTI, COCO).
        start_index (int): Starting index for chip filenames.
        class_value_field (str): Field name in in_class_data containing class values.
        buffer_radius (float): Buffer radius around features (in CRS units).
        in_mask_polygons (str): Path to vector file containing mask polygons.
        rotation_angle (float): Rotation angle in degrees.
        reference_system (str): Reference system code.
        blacken_around_feature (bool): Whether to mask areas outside of features.
        crop_mode (str): Crop mode (FIXED_SIZE, CENTERED_ON_FEATURE).
        in_raster2 (str): Path to secondary raster image.
        in_instance_data (str): Path to vector file containing instance polygons.
        instance_class_value_field (str): Field name in in_instance_data for instance classes.
        min_polygon_overlap_ratio (float): Minimum overlap ratio for polygons.
        all_touched (bool): Whether to use all_touched=True in rasterization.
        save_geotiff (bool): Whether to save as GeoTIFF with georeferencing.
        quiet (bool): If True, suppress most output messages.
    """
    # Create output directories
    image_dir = os.path.join(out_folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    label_dir = os.path.join(out_folder, "labels")
    os.makedirs(label_dir, exist_ok=True)

    # Define annotation directories based on metadata format
    if metadata_format == "PASCAL_VOC":
        ann_dir = os.path.join(out_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
    elif metadata_format == "COCO":
        ann_dir = os.path.join(out_folder, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        # Initialize COCO annotations dictionary
        coco_annotations = {"images": [], "annotations": [], "categories": []}

    # Initialize statistics dictionary
    stats = {
        "total_tiles": 0,
        "tiles_with_features": 0,
        "feature_pixels": 0,
        "errors": 0,
    }

    # Open raster
    with rasterio.open(in_raster) as src:
        if not quiet:
            print(f"\nRaster info for {in_raster}:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bounds: {src.bounds}")

        # Set defaults for stride if not provided
        if stride_x is None:
            stride_x = tile_size_x
        if stride_y is None:
            stride_y = tile_size_y

        # Calculate number of tiles in x and y directions
        num_tiles_x = math.ceil((src.width - tile_size_x) / stride_x) + 1
        num_tiles_y = math.ceil((src.height - tile_size_y) / stride_y) + 1
        total_tiles = num_tiles_x * num_tiles_y

        # Read class data
        gdf = gpd.read_file(in_class_data)
        if not quiet:
            print(f"Loaded {len(gdf)} features from {in_class_data}")
            print(f"Available columns: {gdf.columns.tolist()}")
            print(f"GeoJSON CRS: {gdf.crs}")

        # Check if class_value_field exists
        if class_value_field not in gdf.columns:
            if not quiet:
                print(
                    f"WARNING: '{class_value_field}' field not found in the input data. Using default class value 1."
                )
            # Add a default class column
            gdf[class_value_field] = 1
            unique_classes = [1]
        else:
            # Print unique classes for debugging
            unique_classes = gdf[class_value_field].unique()
            if not quiet:
                print(f"Found {len(unique_classes)} unique classes: {unique_classes}")

        # CRITICAL: Always reproject to match raster CRS to ensure proper alignment
        if gdf.crs != src.crs:
            if not quiet:
                print(f"Reprojecting features from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        elif reference_system and gdf.crs != reference_system:
            if not quiet:
                print(
                    f"Reprojecting features to specified reference system {reference_system}"
                )
            gdf = gdf.to_crs(reference_system)

        # Check overlap between raster and vector data
        raster_bounds = box(*src.bounds)
        vector_bounds = box(*gdf.total_bounds)
        if not raster_bounds.intersects(vector_bounds):
            if not quiet:
                print(
                    "WARNING: The vector data doesn't intersect with the raster extent!"
                )
                print(f"Raster bounds: {src.bounds}")
                print(f"Vector bounds: {gdf.total_bounds}")
        else:
            overlap = (
                raster_bounds.intersection(vector_bounds).area / vector_bounds.area
            )
            if not quiet:
                print(f"Overlap between raster and vector: {overlap:.2%}")

        # Apply buffer if specified
        if buffer_radius > 0:
            gdf["geometry"] = gdf.buffer(buffer_radius)

        # Initialize class mapping (ensure all classes are mapped to non-zero values)
        class_to_id = {cls: i + 1 for i, cls in enumerate(unique_classes)}

        # Store category info for COCO format
        if metadata_format == "COCO":
            for cls_val in unique_classes:
                coco_annotations["categories"].append(
                    {
                        "id": class_to_id[cls_val],
                        "name": str(cls_val),
                        "supercategory": "object",
                    }
                )

        # Load mask polygons if provided
        mask_gdf = None
        if in_mask_polygons:
            mask_gdf = gpd.read_file(in_mask_polygons)
            if reference_system:
                mask_gdf = mask_gdf.to_crs(reference_system)
            elif mask_gdf.crs != src.crs:
                mask_gdf = mask_gdf.to_crs(src.crs)

        # Process instance data if provided
        instance_gdf = None
        if in_instance_data:
            instance_gdf = gpd.read_file(in_instance_data)
            if reference_system:
                instance_gdf = instance_gdf.to_crs(reference_system)
            elif instance_gdf.crs != src.crs:
                instance_gdf = instance_gdf.to_crs(src.crs)

        # Load secondary raster if provided
        src2 = None
        if in_raster2:
            src2 = rasterio.open(in_raster2)

        # Set up augmentation if rotation is specified
        augmentation = None
        if rotation_angle != 0:
            # Fixed: Added data_keys parameter to AugmentationSequential
            augmentation = torchgeo.transforms.AugmentationSequential(
                torch.nn.ModuleList([RandomRotation(rotation_angle)]),
                data_keys=["image"],  # Add data_keys parameter
            )

        # Initialize annotation ID for COCO format
        ann_id = 0

        # Create progress bar
        pbar = tqdm(
            total=total_tiles,
            desc=f"Generating tiles (with features: 0)",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # Generate tiles
        chip_index = start_index
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate window coordinates
                window_x = x * stride_x
                window_y = y * stride_y

                # Adjust for edge cases
                if window_x + tile_size_x > src.width:
                    window_x = src.width - tile_size_x
                if window_y + tile_size_y > src.height:
                    window_y = src.height - tile_size_y

                # Adjust window based on crop_mode
                if crop_mode == "CENTERED_ON_FEATURE" and len(gdf) > 0:
                    # Find the nearest feature to the center of this window
                    window_center_x = window_x + tile_size_x // 2
                    window_center_y = window_y + tile_size_y // 2

                    # Convert center to world coordinates
                    center_x, center_y = src.xy(window_center_y, window_center_x)
                    center_point = gpd.points_from_xy([center_x], [center_y])[0]

                    # Find nearest feature
                    distances = gdf.geometry.distance(center_point)
                    nearest_idx = distances.idxmin()
                    nearest_feature = gdf.iloc[nearest_idx]

                    # Get centroid of nearest feature
                    feature_centroid = nearest_feature.geometry.centroid

                    # Convert feature centroid to pixel coordinates
                    feature_row, feature_col = src.index(
                        feature_centroid.x, feature_centroid.y
                    )

                    # Adjust window to center on feature
                    window_x = max(
                        0, min(src.width - tile_size_x, feature_col - tile_size_x // 2)
                    )
                    window_y = max(
                        0, min(src.height - tile_size_y, feature_row - tile_size_y // 2)
                    )

                # Define window
                window = Window(window_x, window_y, tile_size_x, tile_size_y)

                # Get window transform and bounds in source CRS
                window_transform = src.window_transform(window)

                # Calculate window bounds more explicitly and accurately
                minx = window_transform[2]  # Upper left x
                maxy = window_transform[5]  # Upper left y
                maxx = minx + tile_size_x * window_transform[0]  # Add width
                miny = (
                    maxy + tile_size_y * window_transform[4]
                )  # Add height (note: transform[4] is typically negative)

                window_bounds = box(minx, miny, maxx, maxy)

                # Apply rotation if specified
                if rotation_angle != 0:
                    window_bounds = rotate(
                        window_bounds, rotation_angle, origin="center"
                    )

                # Find features that intersect with window
                window_features = gdf[gdf.intersects(window_bounds)]

                # Process instance data if provided
                window_instances = None
                if instance_gdf is not None and instance_class_value_field is not None:
                    window_instances = instance_gdf[
                        instance_gdf.intersects(window_bounds)
                    ]
                    if len(window_instances) > 0:
                        if not quiet:
                            pbar.write(
                                f"Found {len(window_instances)} instances in tile {chip_index}"
                            )

                # Skip if no features and output_nofeature_tiles is False
                if not output_nofeature_tiles and len(window_features) == 0:
                    pbar.update(1)  # Still update progress bar
                    continue

                # Check polygon overlap ratio if specified
                if min_polygon_overlap_ratio > 0 and len(window_features) > 0:
                    valid_features = []
                    for _, feature in window_features.iterrows():
                        overlap_ratio = (
                            feature.geometry.intersection(window_bounds).area
                            / feature.geometry.area
                        )
                        if overlap_ratio >= min_polygon_overlap_ratio:
                            valid_features.append(feature)

                    if len(valid_features) > 0:
                        window_features = gpd.GeoDataFrame(valid_features)
                    elif not output_nofeature_tiles:
                        pbar.update(1)  # Still update progress bar
                        continue

                # Apply mask if provided
                if mask_gdf is not None:
                    mask_features = mask_gdf[mask_gdf.intersects(window_bounds)]
                    if len(mask_features) == 0:
                        pbar.update(1)  # Still update progress bar
                        continue

                # Read image data - keep original for GeoTIFF export
                orig_image_data = src.read(window=window)

                # Create a copy for processing
                image_data = orig_image_data.copy().astype(np.float32)

                # Normalize image data for processing
                for band in range(image_data.shape[0]):
                    band_min, band_max = np.percentile(image_data[band], (1, 99))
                    if band_max > band_min:
                        image_data[band] = np.clip(
                            (image_data[band] - band_min) / (band_max - band_min), 0, 1
                        )

                # Read secondary image data if provided
                if src2:
                    image_data2 = src2.read(window=window)
                    # Stack the two images
                    image_data = np.vstack((image_data, image_data2))

                # Apply blacken_around_feature if needed
                if blacken_around_feature and len(window_features) > 0:
                    mask = np.zeros((tile_size_y, tile_size_x), dtype=bool)
                    for _, feature in window_features.iterrows():
                        # Project feature to pixel coordinates
                        feature_pixels = features.rasterize(
                            [(feature.geometry, 1)],
                            out_shape=(tile_size_y, tile_size_x),
                            transform=window_transform,
                        )
                        mask = np.logical_or(mask, feature_pixels.astype(bool))

                    # Apply mask to image
                    for band in range(image_data.shape[0]):
                        temp = image_data[band, :, :]
                        temp[~mask] = 0
                        image_data[band, :, :] = temp

                # Apply rotation if specified
                if augmentation:
                    # Convert to torch tensor for augmentation
                    image_tensor = torch.from_numpy(image_data).unsqueeze(
                        0
                    )  # Add batch dimension
                    # Apply augmentation with proper data format
                    augmented = augmentation({"image": image_tensor})
                    image_data = (
                        augmented["image"].squeeze(0).numpy()
                    )  # Remove batch dimension

                # Create a processed version for regular image formats
                processed_image = (image_data * 255).astype(np.uint8)

                # Create label mask
                label_mask = np.zeros((tile_size_y, tile_size_x), dtype=np.uint8)
                has_features = False

                if len(window_features) > 0:
                    for idx, feature in window_features.iterrows():
                        # Get class value
                        class_val = (
                            feature[class_value_field]
                            if class_value_field in feature
                            else 1
                        )
                        if isinstance(class_val, str):
                            # If class is a string, use its position in the unique classes list
                            class_id = class_to_id.get(class_val, 1)
                        else:
                            # If class is already a number, use it directly
                            class_id = int(class_val) if class_val > 0 else 1

                        # Get the geometry in pixel coordinates
                        geom = feature.geometry.intersection(window_bounds)
                        if not geom.is_empty:
                            try:
                                # Rasterize the feature
                                feature_mask = features.rasterize(
                                    [(geom, class_id)],
                                    out_shape=(tile_size_y, tile_size_x),
                                    transform=window_transform,
                                    fill=0,
                                    all_touched=all_touched,
                                )

                                # Update mask with higher class values taking precedence
                                label_mask = np.maximum(label_mask, feature_mask)

                                # Check if any pixels were added
                                if np.any(feature_mask):
                                    has_features = True
                            except Exception as e:
                                if not quiet:
                                    pbar.write(f"Error rasterizing feature {idx}: {e}")
                                stats["errors"] += 1

                # Save as GeoTIFF if requested
                if save_geotiff or image_chip_format.upper() in [
                    "TIFF",
                    "TIF",
                    "GEOTIFF",
                ]:
                    # Standardize extension to .tif for GeoTIFF files
                    image_filename = f"tile_{chip_index:06d}.tif"
                    image_path = os.path.join(image_dir, image_filename)

                    # Create profile for the GeoTIFF
                    profile = src.profile.copy()
                    profile.update(
                        {
                            "height": tile_size_y,
                            "width": tile_size_x,
                            "count": orig_image_data.shape[0],
                            "transform": window_transform,
                        }
                    )

                    # Save the GeoTIFF with original data
                    try:
                        with rasterio.open(image_path, "w", **profile) as dst:
                            dst.write(orig_image_data)
                        stats["total_tiles"] += 1
                    except Exception as e:
                        if not quiet:
                            pbar.write(
                                f"ERROR saving image GeoTIFF for tile {chip_index}: {e}"
                            )
                        stats["errors"] += 1
                else:
                    # For non-GeoTIFF formats, use PIL to save the image
                    image_filename = (
                        f"tile_{chip_index:06d}.{image_chip_format.lower()}"
                    )
                    image_path = os.path.join(image_dir, image_filename)

                    # Create PIL image for saving
                    if processed_image.shape[0] == 1:
                        img = Image.fromarray(processed_image[0])
                    elif processed_image.shape[0] == 3:
                        # For RGB, need to transpose and make sure it's the right data type
                        rgb_data = np.transpose(processed_image, (1, 2, 0))
                        img = Image.fromarray(rgb_data)
                    else:
                        # For multiband images, save only RGB or first three bands
                        rgb_data = np.transpose(processed_image[:3], (1, 2, 0))
                        img = Image.fromarray(rgb_data)

                    # Save image
                    try:
                        img.save(image_path)
                        stats["total_tiles"] += 1
                    except Exception as e:
                        if not quiet:
                            pbar.write(f"ERROR saving image for tile {chip_index}: {e}")
                        stats["errors"] += 1

                # Save label as GeoTIFF
                label_filename = f"tile_{chip_index:06d}.tif"
                label_path = os.path.join(label_dir, label_filename)

                # Create profile for label GeoTIFF
                label_profile = {
                    "driver": "GTiff",
                    "height": tile_size_y,
                    "width": tile_size_x,
                    "count": 1,
                    "dtype": "uint8",
                    "crs": src.crs,
                    "transform": window_transform,
                }

                # Save label GeoTIFF
                try:
                    with rasterio.open(label_path, "w", **label_profile) as dst:
                        dst.write(label_mask, 1)

                    if has_features:
                        pixel_count = np.count_nonzero(label_mask)
                        stats["tiles_with_features"] += 1
                        stats["feature_pixels"] += pixel_count
                except Exception as e:
                    if not quiet:
                        pbar.write(f"ERROR saving label for tile {chip_index}: {e}")
                    stats["errors"] += 1

                # Also save a PNG version for easy visualization if requested
                if metadata_format == "PASCAL_VOC":
                    try:
                        # Ensure correct data type for PIL
                        png_label = label_mask.astype(np.uint8)
                        label_img = Image.fromarray(png_label)
                        label_png_path = os.path.join(
                            label_dir, f"tile_{chip_index:06d}.png"
                        )
                        label_img.save(label_png_path)
                    except Exception as e:
                        if not quiet:
                            pbar.write(
                                f"ERROR saving PNG label for tile {chip_index}: {e}"
                            )
                            pbar.write(
                                f"  Label mask shape: {label_mask.shape}, dtype: {label_mask.dtype}"
                            )
                            # Try again with explicit conversion
                            try:
                                # Alternative approach for problematic arrays
                                png_data = np.zeros(
                                    (tile_size_y, tile_size_x), dtype=np.uint8
                                )
                                np.copyto(png_data, label_mask, casting="unsafe")
                                label_img = Image.fromarray(png_data)
                                label_img.save(label_png_path)
                                pbar.write(
                                    f"  Succeeded using alternative conversion method"
                                )
                            except Exception as e2:
                                pbar.write(f"  Second attempt also failed: {e2}")
                                stats["errors"] += 1

                # Generate annotations
                if metadata_format == "PASCAL_VOC" and len(window_features) > 0:
                    # Create XML annotation
                    root = ET.Element("annotation")
                    ET.SubElement(root, "folder").text = "images"
                    ET.SubElement(root, "filename").text = image_filename

                    size = ET.SubElement(root, "size")
                    ET.SubElement(size, "width").text = str(tile_size_x)
                    ET.SubElement(size, "height").text = str(tile_size_y)
                    ET.SubElement(size, "depth").text = str(min(image_data.shape[0], 3))

                    # Add georeference information
                    geo = ET.SubElement(root, "georeference")
                    ET.SubElement(geo, "crs").text = str(src.crs)
                    ET.SubElement(geo, "transform").text = str(
                        window_transform
                    ).replace("\n", "")
                    ET.SubElement(geo, "bounds").text = (
                        f"{minx}, {miny}, {maxx}, {maxy}"
                    )

                    for _, feature in window_features.iterrows():
                        # Convert feature geometry to pixel coordinates
                        feature_bounds = feature.geometry.intersection(window_bounds)
                        if feature_bounds.is_empty:
                            continue

                        # Get pixel coordinates of bounds
                        minx_f, miny_f, maxx_f, maxy_f = feature_bounds.bounds

                        # Convert to pixel coordinates
                        col_min, row_min = ~window_transform * (minx_f, maxy_f)
                        col_max, row_max = ~window_transform * (maxx_f, miny_f)

                        # Ensure coordinates are within bounds
                        xmin = max(0, min(tile_size_x, int(col_min)))
                        ymin = max(0, min(tile_size_y, int(row_min)))
                        xmax = max(0, min(tile_size_x, int(col_max)))
                        ymax = max(0, min(tile_size_y, int(row_max)))

                        # Skip if box is too small
                        if xmax - xmin < 1 or ymax - ymin < 1:
                            continue

                        obj = ET.SubElement(root, "object")
                        ET.SubElement(obj, "name").text = str(
                            feature[class_value_field]
                        )
                        ET.SubElement(obj, "difficult").text = "0"

                        bbox = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bbox, "xmin").text = str(xmin)
                        ET.SubElement(bbox, "ymin").text = str(ymin)
                        ET.SubElement(bbox, "xmax").text = str(xmax)
                        ET.SubElement(bbox, "ymax").text = str(ymax)

                    # Save XML
                    try:
                        tree = ET.ElementTree(root)
                        xml_path = os.path.join(ann_dir, f"tile_{chip_index:06d}.xml")
                        tree.write(xml_path)
                    except Exception as e:
                        if not quiet:
                            pbar.write(
                                f"ERROR saving XML annotation for tile {chip_index}: {e}"
                            )
                        stats["errors"] += 1

                elif metadata_format == "COCO" and len(window_features) > 0:
                    # Add image info
                    image_id = chip_index
                    coco_annotations["images"].append(
                        {
                            "id": image_id,
                            "file_name": image_filename,
                            "width": tile_size_x,
                            "height": tile_size_y,
                            "crs": str(src.crs),
                            "transform": str(window_transform),
                        }
                    )

                    # Add annotations for each feature
                    for _, feature in window_features.iterrows():
                        feature_bounds = feature.geometry.intersection(window_bounds)
                        if feature_bounds.is_empty:
                            continue

                        # Get pixel coordinates of bounds
                        minx_f, miny_f, maxx_f, maxy_f = feature_bounds.bounds

                        # Convert to pixel coordinates
                        col_min, row_min = ~window_transform * (minx_f, maxy_f)
                        col_max, row_max = ~window_transform * (maxx_f, miny_f)

                        # Ensure coordinates are within bounds
                        xmin = max(0, min(tile_size_x, int(col_min)))
                        ymin = max(0, min(tile_size_y, int(row_min)))
                        xmax = max(0, min(tile_size_x, int(col_max)))
                        ymax = max(0, min(tile_size_y, int(row_max)))

                        # Skip if box is too small
                        if xmax - xmin < 1 or ymax - ymin < 1:
                            continue

                        width = xmax - xmin
                        height = ymax - ymin

                        # Add annotation
                        ann_id += 1
                        category_id = class_to_id[feature[class_value_field]]

                        coco_annotations["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [xmin, ymin, width, height],
                                "area": width * height,
                                "iscrowd": 0,
                            }
                        )

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Generated: {stats['total_tiles']}, With features: {stats['tiles_with_features']}"
                )

                chip_index += 1

        # Close progress bar
        pbar.close()

        # Save COCO annotations if applicable
        if metadata_format == "COCO":
            try:
                with open(os.path.join(ann_dir, "instances.json"), "w") as f:
                    json.dump(coco_annotations, f)
            except Exception as e:
                if not quiet:
                    print(f"ERROR saving COCO annotations: {e}")
                stats["errors"] += 1

        # Close secondary raster if opened
        if src2:
            src2.close()

    # Print summary
    if not quiet:
        print("\n------- Export Summary -------")
        print(f"Total tiles exported: {stats['total_tiles']}")
        print(
            f"Tiles with features: {stats['tiles_with_features']} ({stats['tiles_with_features']/max(1, stats['total_tiles'])*100:.1f}%)"
        )
        if stats["tiles_with_features"] > 0:
            print(
                f"Average feature pixels per tile: {stats['feature_pixels']/stats['tiles_with_features']:.1f}"
            )
        if stats["errors"] > 0:
            print(f"Errors encountered: {stats['errors']}")
        print(f"Output saved to: {out_folder}")

        # Verify georeference in a sample image and label
        if stats["total_tiles"] > 0:
            print("\n------- Georeference Verification -------")
            sample_image = os.path.join(image_dir, f"tile_{start_index}.tif")
            sample_label = os.path.join(label_dir, f"tile_{start_index}.tif")

            if os.path.exists(sample_image):
                try:
                    with rasterio.open(sample_image) as img:
                        print(f"Image CRS: {img.crs}")
                        print(f"Image transform: {img.transform}")
                        print(
                            f"Image has georeference: {img.crs is not None and img.transform is not None}"
                        )
                        print(
                            f"Image dimensions: {img.width}x{img.height}, {img.count} bands, {img.dtypes[0]} type"
                        )
                except Exception as e:
                    print(f"Error verifying image georeference: {e}")

            if os.path.exists(sample_label):
                try:
                    with rasterio.open(sample_label) as lbl:
                        print(f"Label CRS: {lbl.crs}")
                        print(f"Label transform: {lbl.transform}")
                        print(
                            f"Label has georeference: {lbl.crs is not None and lbl.transform is not None}"
                        )
                        print(
                            f"Label dimensions: {lbl.width}x{lbl.height}, {lbl.count} bands, {lbl.dtypes[0]} type"
                        )
                except Exception as e:
                    print(f"Error verifying label georeference: {e}")

    # Return statistics
    return stats, out_folder
