"""Utilities for downloading and preparing the Fields of The World (FTW) dataset.

The Fields of The World (FTW) dataset is a large-scale benchmark for
agricultural field boundary instance segmentation. It contains Sentinel-2
imagery (4 bands: Red, Green, Blue, NIR at 10 m resolution) paired with
instance segmentation masks across 25 countries.

Reference:
    Kerner et al., "Fields of The World: A Machine Learning Benchmark Dataset
    For Global Agricultural Field Boundary Delineation", 2024.
    https://fieldsofthe.world/
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# All 25 countries available in the FTW dataset
FTW_COUNTRIES = [
    "austria",
    "belgium",
    "brazil",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "india",
    "kenya",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "portugal",
    "rwanda",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]

FTW_BASE_URL = (
    "https://data.source.coop/kerner-lab/fields-of-the-world-archive/{country}.zip"
)


def download_ftw(
    countries: Optional[List[str]] = None,
    output_dir: str = "ftw_data",
    overwrite: bool = False,
) -> str:
    """Download the Fields of The World (FTW) dataset for specified countries.

    Downloads Sentinel-2 imagery and instance segmentation masks from the
    FTW dataset hosted on Source Cooperative. Each country subset includes
    256x256 pixel chips with 4-band (Red, Green, Blue, NIR) GeoTIFF images
    captured at two different dates (``window_a`` and ``window_b``) and
    corresponding instance mask GeoTIFFs.  The two temporal windows allow
    models to exploit seasonal vegetation differences for better field
    boundary detection.

    Args:
        countries: List of country names to download. If None, downloads
            Luxembourg (smallest European subset). Use ``FTW_COUNTRIES`` for
            the full list of available countries.
        output_dir: Directory to save downloaded data. Defaults to "ftw_data".
        overwrite: If True, re-download even if data already exists.
            Defaults to False.

    Returns:
        Path to the output directory containing downloaded country subsets.

    Raises:
        ValueError: If any country name is not in the list of available
            countries.

    Example:
        >>> import geoai
        >>> geoai.download_ftw(countries=["luxembourg"], output_dir="ftw_data")
        'ftw_data'
    """
    from .utils.download import download_file

    if countries is None:
        countries = ["luxembourg"]

    # Validate country names
    invalid = [c for c in countries if c.lower() not in FTW_COUNTRIES]
    if invalid:
        raise ValueError(
            f"Invalid country names: {invalid}. "
            f"Available countries: {FTW_COUNTRIES}"
        )

    os.makedirs(output_dir, exist_ok=True)

    for country in countries:
        country = country.lower()
        country_dir = os.path.join(output_dir, country)

        if os.path.exists(country_dir) and not overwrite:
            logger.info("FTW %s already exists at %s, skipping.", country, country_dir)
            continue

        # Remove existing directory when overwriting
        if os.path.exists(country_dir) and overwrite:
            shutil.rmtree(country_dir)

        url = FTW_BASE_URL.format(country=country)
        zip_path = os.path.join(output_dir, f"{country}.zip")

        # Download the zip file
        logger.info("Downloading FTW %s dataset...", country)
        download_file(url, output_path=zip_path, overwrite=overwrite, unzip=False)

        # Extract into a temporary directory, then move to country_dir
        import zipfile

        if os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path):
            # Extract into a temp dir first to handle flat zip structure
            tmp_dir = os.path.join(output_dir, f"_tmp_{country}")
            os.makedirs(tmp_dir, exist_ok=True)

            logger.info("Extracting %s...", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)

            # Check if the zip contained a single top-level directory
            top_items = os.listdir(tmp_dir)
            if len(top_items) == 1 and os.path.isdir(
                os.path.join(tmp_dir, top_items[0])
            ):
                # Single directory — move it
                shutil.move(os.path.join(tmp_dir, top_items[0]), country_dir)
                os.rmdir(tmp_dir)
            else:
                # Flat structure — rename the temp dir
                shutil.move(tmp_dir, country_dir)

            logger.info("Extracted to %s", country_dir)

    return output_dir


def prepare_ftw(
    data_dir: str,
    country: str = "luxembourg",
    output_dir: Optional[str] = None,
    window: str = "window_a",
    clip_value: int = 3000,
    num_test: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Prepare FTW data for training with geoai's instance segmentation pipeline.

    Rescales Sentinel-2 reflectance images from 0-10000 to uint8 (0-255) and
    organizes them into ``images/`` and ``labels/`` directories compatible with
    ``geoai.train_instance_segmentation_model()``.

    Args:
        data_dir: Path to the root FTW data directory (containing country
            subdirectories as downloaded by ``download_ftw``).
        country: Country subset to prepare. Defaults to "luxembourg".
        output_dir: Directory to write prepared images and labels. If None,
            defaults to ``"field_boundaries"``.
        window: Which temporal window to use for imagery. The FTW dataset
            provides two Sentinel-2 acquisitions from different dates for
            each chip so that seasonal vegetation differences can help
            delineate field boundaries.  ``"window_a"`` and ``"window_b"``
            correspond to these two dates.  Use one window for 4-band
            input or stack both externally for 8-band input.
            Defaults to ``"window_a"``.
        clip_value: Upper bound for Sentinel-2 reflectance clipping before
            rescaling to 0-255. Defaults to 3000.
        num_test: Number of test chips to prepare for inference. Set to 0 to
            skip test data preparation. Defaults to 5.
        verbose: If True, print progress information. Defaults to True.

    Returns:
        Dictionary with keys:
            - ``images_dir``: Path to prepared training images.
            - ``labels_dir``: Path to prepared training labels.
            - ``test_dir``: Path to prepared test images (or None if
              ``num_test=0``).
            - ``num_train``: Number of training chips prepared.
            - ``num_test``: Number of test chips prepared.

    Raises:
        FileNotFoundError: If the country directory or parquet metadata
            file is not found.

    Example:
        >>> import geoai
        >>> geoai.download_ftw(countries=["luxembourg"])
        >>> result = geoai.prepare_ftw("ftw_data", country="luxembourg")
        >>> print(result["images_dir"], result["num_train"])
    """
    import rasterio

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for prepare_ftw. Install it with: pip install pandas"
        )

    if output_dir is None:
        output_dir = "field_boundaries"

    country = country.lower()
    country_dir = os.path.join(data_dir, country)

    # Handle case where ftw-tools creates an extra subdirectory
    if not os.path.isdir(country_dir):
        alt_dir = os.path.join(data_dir, "ftw", country)
        if os.path.isdir(alt_dir):
            country_dir = alt_dir
        else:
            raise FileNotFoundError(
                f"Country directory not found: {country_dir}. "
                f"Run download_ftw(countries=['{country}'], "
                f"output_dir='{data_dir}') first."
            )

    parquet_path = os.path.join(country_dir, f"chips_{country}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Metadata file not found: {parquet_path}. "
            f"The FTW download may be incomplete."
        )

    chips_df = pd.read_parquet(parquet_path)

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    test_dir = os.path.join(output_dir, "test") if num_test > 0 else None

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    if test_dir:
        os.makedirs(test_dir, exist_ok=True)

    # Get split AOI IDs
    train_aois = chips_df[chips_df["split"] == "train"]["aoi_id"].tolist()
    val_aois = chips_df[chips_df["split"] == "val"]["aoi_id"].tolist()
    test_aois = chips_df[chips_df["split"] == "test"]["aoi_id"].tolist()

    # Use train + val for training (geoai handles its own val_split)
    all_train_aois = train_aois + val_aois

    if verbose:
        logger.info("FTW %s: %d total chips", country, len(chips_df))
        logger.info(
            "  Train: %d, Val: %d, Test: %d",
            len(train_aois),
            len(val_aois),
            len(test_aois),
        )
        logger.info("  Using %d chips for training", len(all_train_aois))
        logger.info("Preparing training data...")

    # Process training chips
    prepared_train = 0
    for aoi in all_train_aois:
        src_img = os.path.join(country_dir, "s2_images", window, f"{aoi}.tif")
        src_mask = os.path.join(country_dir, "label_masks", "instance", f"{aoi}.tif")

        if not os.path.exists(src_img) or not os.path.exists(src_mask):
            continue

        _rescale_sentinel2_image(
            src_img, os.path.join(images_dir, f"{aoi}.tif"), clip_value
        )
        shutil.copy2(src_mask, os.path.join(labels_dir, f"{aoi}.tif"))
        prepared_train += 1

    if verbose:
        skipped = len(all_train_aois) - prepared_train
        logger.info("Prepared %d training chips (skipped %d)", prepared_train, skipped)

    # Process test chips
    prepared_test = 0
    if test_dir and num_test > 0:
        for aoi in test_aois[:num_test]:
            src_img = os.path.join(country_dir, "s2_images", window, f"{aoi}.tif")
            if os.path.exists(src_img):
                _rescale_sentinel2_image(
                    src_img, os.path.join(test_dir, f"{aoi}.tif"), clip_value
                )
                prepared_test += 1

        if verbose:
            logger.info("Prepared %d test chips", prepared_test)

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "test_dir": test_dir,
        "num_train": prepared_train,
        "num_test": prepared_test,
    }


def _rescale_sentinel2_image(
    src_path: str,
    dst_path: str,
    clip_value: int = 3000,
) -> None:
    """Rescale a Sentinel-2 image from raw reflectance to uint8 (0-255).

    Args:
        src_path: Path to the source GeoTIFF.
        dst_path: Path to write the rescaled GeoTIFF.
        clip_value: Upper bound for reflectance clipping.
    """
    import rasterio

    with rasterio.open(src_path) as src:
        data = src.read().astype(np.float32)
        data = np.clip(data, 0, clip_value)
        data = (data / clip_value * 255).astype(np.uint8)

        profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=0)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)


def display_ftw_samples(
    data_dir: str,
    country: str = "luxembourg",
    num_samples: int = 4,
    split: str = "train",
    window: str = "window_a",
    clip_value: int = 3000,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "tab20",
    save_path: Optional[str] = None,
) -> None:
    """Display FTW image-mask pairs from the raw dataset.

    Shows Sentinel-2 RGB images alongside their corresponding instance
    segmentation masks for visual inspection of the training data.

    Args:
        data_dir: Path to the root FTW data directory (containing country
            subdirectories as downloaded by ``download_ftw``).
        country: Country subset to display. Defaults to "luxembourg".
        num_samples: Number of image-mask pairs to display. Defaults to 4.
        split: Dataset split to sample from ("train", "val", or "test").
            Defaults to "train".
        window: Which temporal acquisition to display. The FTW dataset
            provides two Sentinel-2 images per chip from different dates
            (``"window_a"`` and ``"window_b"``).  Defaults to
            ``"window_a"``.
        clip_value: Upper bound for Sentinel-2 reflectance used for RGB
            visualization. Defaults to 3000.
        figsize: Figure size as (width, height) in inches. If None,
            auto-calculated based on ``num_samples``.
        cmap: Colormap for instance mask display. Defaults to "tab20".
        save_path: If provided, save figure to this path instead of
            displaying. Defaults to None.

    Example:
        >>> import geoai
        >>> geoai.download_ftw(countries=["luxembourg"])
        >>> geoai.display_ftw_samples("ftw_data", country="luxembourg", num_samples=6)
    """
    import matplotlib.pyplot as plt
    import rasterio

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for display_ftw_samples. "
            "Install it with: pip install pandas"
        )

    country = country.lower()
    country_dir = os.path.join(data_dir, country)

    # Handle alternate directory structure from ftw-tools
    if not os.path.isdir(country_dir):
        alt_dir = os.path.join(data_dir, "ftw", country)
        if os.path.isdir(alt_dir):
            country_dir = alt_dir
        else:
            raise FileNotFoundError(
                f"Country directory not found: {country_dir}. "
                f"Run download_ftw(countries=['{country}'], "
                f"output_dir='{data_dir}') first."
            )

    parquet_path = os.path.join(country_dir, f"chips_{country}.parquet")
    chips_df = pd.read_parquet(parquet_path)

    aois = chips_df[chips_df["split"] == split]["aoi_id"].tolist()
    num_samples = min(num_samples, len(aois))

    if num_samples == 0:
        logger.warning("No samples found for split='%s'", split)
        return

    if figsize is None:
        figsize = (3 * num_samples, 6)

    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        aoi = aois[i]
        img_path = os.path.join(country_dir, "s2_images", window, f"{aoi}.tif")
        mask_path = os.path.join(country_dir, "label_masks", "instance", f"{aoi}.tif")

        with rasterio.open(img_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            img = np.clip(img / float(clip_value), 0, 1)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask, cmap=cmap, interpolation="nearest")
        axes[1, i].set_title(f"Mask {i + 1}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Sentinel-2 RGB", fontsize=12)
    axes[1, 0].set_ylabel("Instance Mask", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
