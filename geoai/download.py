"""This module provides functions to download data, including NAIP imagery and building data from Overture Maps."""

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import requests
import rioxarray
import xarray as xr
from pystac_client import Client
from shapely.geometry import box
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_naip(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    year: Optional[int] = None,
    max_items: int = 10,
    overwrite: bool = False,
    preview: bool = False,
    **kwargs: Any,
) -> List[str]:
    """Download NAIP imagery from Planetary Computer based on a bounding box.

    This function searches for NAIP (National Agriculture Imagery Program) imagery
    from Microsoft's Planetary Computer that intersects with the specified bounding box.
    It downloads the imagery and saves it as GeoTIFF files.

    Args:
        bbox: Bounding box in the format (min_lon, min_lat, max_lon, max_lat) in WGS84 coordinates.
        output_dir: Directory to save the downloaded imagery.
        year: Specific year of NAIP imagery to download (e.g., 2020). If None, returns imagery from all available years.
        max_items: Maximum number of items to download.
        overwrite: If True, overwrite existing files with the same name.
        preview: If True, display a preview of the downloaded imagery.

    Returns:
        List of downloaded file paths.

    Raises:
        Exception: If there is an error downloading or saving the imagery.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a geometry from the bounding box
    geometry = box(*bbox)

    # Connect to Planetary Computer STAC API
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Build query for NAIP data
    search_params = {
        "collections": ["naip"],
        "intersects": geometry,
        "limit": max_items,
    }

    # Add year filter if specified
    if year:
        search_params["query"] = {"naip:year": {"eq": year}}

    for key, value in kwargs.items():
        search_params[key] = value

    # Search for NAIP imagery
    search_results = catalog.search(**search_params)
    items = list(search_results.items())

    if len(items) > max_items:
        items = items[:max_items]

    if not items:
        print("No NAIP imagery found for the specified region and parameters.")
        return []

    print(f"Found {len(items)} NAIP items.")

    # Download and save each item
    downloaded_files = []
    for i, item in enumerate(items):
        # Sign the assets (required for Planetary Computer)
        signed_item = pc.sign(item)

        # Get the RGB asset URL
        rgb_asset = signed_item.assets.get("image")
        if not rgb_asset:
            print(f"No RGB asset found for item {i+1}")
            continue

        # Use the original filename from the asset
        original_filename = os.path.basename(
            rgb_asset.href.split("?")[0]
        )  # Remove query parameters
        output_path = os.path.join(output_dir, original_filename)
        if not overwrite and os.path.exists(output_path):
            print(f"Skipping existing file: {output_path}")
            downloaded_files.append(output_path)
            continue

        print(f"Downloading item {i+1}/{len(items)}: {original_filename}")

        try:
            # Open and save the data with progress bar
            # For direct file download with progress bar
            if rgb_asset.href.startswith("http"):
                download_with_progress(rgb_asset.href, output_path)
                #
            else:
                # Fallback to direct rioxarray opening (less common case)
                data = rioxarray.open_rasterio(rgb_asset.href)
                data.rio.to_raster(output_path)

            downloaded_files.append(output_path)
            print(f"Successfully saved to {output_path}")

            # Optional: Display a preview (uncomment if needed)
            if preview:
                data = rioxarray.open_rasterio(output_path)
                preview_raster(data)

        except Exception as e:
            print(f"Error downloading item {i+1}: {str(e)}")

    return downloaded_files


def download_with_progress(url: str, output_path: str) -> None:
    """Download a file with a progress bar.

    Args:
        url: URL of the file to download.
        output_path: Path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with (
        open(output_path, "wb") as file,
        tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def preview_raster(data: Any, title: str = None) -> None:
    """Display a preview of the downloaded imagery.

    This function creates a visualization of the downloaded NAIP imagery
    by converting it to an RGB array and displaying it with matplotlib.

    Args:
        data: The raster data as a rioxarray object.
        title: The title for the preview plot.
    """
    # Convert to 8-bit RGB for display
    rgb_data = data.transpose("y", "x", "band").values[:, :, 0:3]
    rgb_data = np.where(rgb_data > 255, 255, rgb_data).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_data)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


# Helper function to convert NumPy types to native Python types for JSON serialization
def json_serializable(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization.

    Args:
        obj: Any object to convert.

    Returns:
        JSON serializable version of the object.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def download_overture_buildings(
    bbox: Tuple[float, float, float, float],
    output_file: str,
    output_format: str = "geojson",
    data_type: str = "building",
    verbose: bool = True,
) -> str:
    """Download building data from Overture Maps for a given bounding box using the overturemaps CLI tool.

    Args:
        bbox: Bounding box in the format (min_lon, min_lat, max_lon, max_lat) in WGS84 coordinates.
        output_file: Path to save the output file.
        output_format: Format to save the output, one of "geojson", "geojsonseq", or "geoparquet".
        data_type: The Overture Maps data type to download (building, place, etc.).
        verbose: Whether to print verbose output.

    Returns:
        Path to the output file.
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Format the bounding box string for the command
    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"

    # Build the command
    cmd = [
        "overturemaps",
        "download",
        "--bbox",
        bbox_str,
        "-f",
        output_format,
        "--type",
        data_type,
        "--output",
        output_file,
    ]

    if verbose:
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("Downloading %s data for area: %s", data_type, bbox_str)

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
            logger.info(
                f"Successfully downloaded data to {output_file} ({file_size:.2f} MB)"
            )

            # Optionally show some stats about the downloaded data
            if output_format == "geojson" and os.path.getsize(output_file) > 0:
                try:
                    gdf = gpd.read_file(output_file)
                    logger.info(f"Downloaded {len(gdf)} features")

                    if len(gdf) > 0 and verbose:
                        # Show a sample of the attribute names
                        attrs = list(gdf.columns)
                        attrs.remove("geometry")
                        logger.info(f"Available attributes: {', '.join(attrs[:10])}...")
                except Exception as e:
                    logger.warning(f"Could not read the GeoJSON file: {str(e)}")

            return output_file
        else:
            logger.error(f"Command completed but file {output_file} was not created")
            if result.stderr:
                logger.error(f"Command error output: {result.stderr}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running overturemaps command: {str(e)}")
        if e.stderr:
            logger.error(f"Command error output: {e.stderr}")
        raise RuntimeError(f"Failed to download Overture Maps data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


def convert_vector_format(
    input_file: str,
    output_format: str = "geojson",
    filter_expression: Optional[str] = None,
) -> str:
    """Convert the downloaded data to a different format or filter it.

    Args:
        input_file: Path to the input file.
        output_format: Format to convert to, one of "geojson", "parquet", "shapefile", "csv".
        filter_expression: Optional GeoDataFrame query expression to filter the data.

    Returns:
        Path to the converted file.
    """
    try:
        # Read the input file
        logger.info(f"Reading {input_file}")
        gdf = gpd.read_file(input_file)

        # Apply filter if specified
        if filter_expression:
            logger.info(f"Filtering data using expression: {filter_expression}")
            gdf = gdf.query(filter_expression)
            logger.info(f"After filtering: {len(gdf)} features")

        # Define output file path
        base_path = os.path.splitext(input_file)[0]

        if output_format == "geojson":
            output_file = f"{base_path}.geojson"
            logger.info(f"Converting to GeoJSON: {output_file}")
            gdf.to_file(output_file, driver="GeoJSON")
        elif output_format == "parquet":
            output_file = f"{base_path}.parquet"
            logger.info(f"Converting to Parquet: {output_file}")
            gdf.to_parquet(output_file)
        elif output_format == "shapefile":
            output_file = f"{base_path}.shp"
            logger.info(f"Converting to Shapefile: {output_file}")
            gdf.to_file(output_file)
        elif output_format == "csv":
            output_file = f"{base_path}.csv"
            logger.info(f"Converting to CSV: {output_file}")

            # For CSV, we need to convert geometry to WKT
            gdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt)

            # Save to CSV with geometry as WKT
            gdf.drop(columns=["geometry"]).to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return output_file

    except Exception as e:
        logger.error(f"Error converting data: {str(e)}")
        raise


def extract_building_stats(geojson_file: str) -> Dict[str, Any]:
    """Extract statistics from the building data.

    Args:
        geojson_file: Path to the GeoJSON file.

    Returns:
        Dictionary with statistics.
    """
    try:
        # Read the GeoJSON file
        gdf = gpd.read_file(geojson_file)

        # Calculate statistics
        bbox = gdf.total_bounds.tolist()
        # Convert numpy values to Python native types
        bbox = [float(x) for x in bbox]

        stats = {
            "total_buildings": int(len(gdf)),
            "has_height": (
                int(gdf["height"].notna().sum()) if "height" in gdf.columns else 0
            ),
            "has_name": (
                int(gdf["names.common.value"].notna().sum())
                if "names.common.value" in gdf.columns
                else 0
            ),
            "bbox": bbox,
        }

        return stats

    except Exception as e:
        logger.error(f"Error extracting statistics: {str(e)}")
        return {"error": str(e)}


def download_pc_stac_item(
    item_url,
    bands=None,
    output_dir=None,
    show_progress=True,
    merge_bands=False,
    merged_filename=None,
    overwrite=False,
    cell_size=None,
):
    """
    Downloads a STAC item from Microsoft Planetary Computer with specified bands.

    This function fetches a STAC item by URL, signs the assets using Planetary Computer
    credentials, and downloads the specified bands with a progress bar. Can optionally
    merge bands into a single multi-band GeoTIFF.

    Args:
        item_url (str): The URL of the STAC item to download.
        bands (list, optional): List of specific bands to download (e.g., ['B01', 'B02']).
                               If None, all available bands will be downloaded.
        output_dir (str, optional): Directory to save downloaded bands. If None,
                                   bands are returned as xarray DataArrays.
        show_progress (bool, optional): Whether to display a progress bar. Default is True.
        merge_bands (bool, optional): Whether to merge downloaded bands into a single
                                     multi-band GeoTIFF file. Default is False.
        merged_filename (str, optional): Filename for the merged bands. If None and
                                        merge_bands is True, uses "{item_id}_merged.tif".
        overwrite (bool, optional): Whether to overwrite existing files. Default is False.
        cell_size (float, optional): Resolution in meters for the merged output. If None,
                                    uses the resolution of the first band.

    Returns:
        dict: Dictionary mapping band names to their corresponding xarray DataArrays
              or file paths if output_dir is provided. If merge_bands is True, also
              includes a 'merged' key with the path to the merged file.

    Raises:
        ValueError: If the item cannot be retrieved or a requested band is not available.
    """
    from rasterio.enums import Resampling

    # Get the item ID from the URL
    item_id = item_url.split("/")[-1]
    collection = item_url.split("/collections/")[1].split("/items/")[0]

    # Connect to the Planetary Computer STAC API
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    # Search for the specific item
    search = catalog.search(collections=[collection], ids=[item_id])

    # Get the first item from the search results
    items = list(search.get_items())
    if not items:
        raise ValueError(f"Item with ID {item_id} not found")

    item = items[0]

    # Determine which bands to download
    available_assets = list(item.assets.keys())

    if bands is None:
        # If no bands specified, download all band assets
        bands_to_download = [
            asset for asset in available_assets if asset.startswith("B")
        ]
    else:
        # Verify all requested bands exist
        missing_bands = [band for band in bands if band not in available_assets]
        if missing_bands:
            raise ValueError(
                f"The following bands are not available: {missing_bands}. "
                f"Available assets are: {available_assets}"
            )
        bands_to_download = bands

    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result = {}
    band_data_arrays = []
    resampled_arrays = []
    band_names = []  # Track band names in order

    # Set up progress bar
    progress_iter = (
        tqdm(bands_to_download, desc="Downloading bands")
        if show_progress
        else bands_to_download
    )

    # Download each requested band
    for band in progress_iter:
        if band not in item.assets:
            if show_progress and not isinstance(progress_iter, list):
                progress_iter.write(
                    f"Warning: Band {band} not found in assets, skipping."
                )
            continue

        band_url = item.assets[band].href

        if output_dir:
            file_path = os.path.join(output_dir, f"{item.id}_{band}.tif")

            # Check if file exists and skip if overwrite is False
            if os.path.exists(file_path) and not overwrite:
                if show_progress and not isinstance(progress_iter, list):
                    progress_iter.write(
                        f"File {file_path} already exists, skipping (use overwrite=True to force download)."
                    )
                # Still need to open the file to get the data for merging
                if merge_bands:
                    band_data = rioxarray.open_rasterio(file_path)
                    band_data_arrays.append((band, band_data))
                    band_names.append(band)
                result[band] = file_path
                continue

        if show_progress and not isinstance(progress_iter, list):
            progress_iter.set_description(f"Downloading {band}")

        band_data = rioxarray.open_rasterio(band_url)

        # Store the data array for potential merging later
        if merge_bands:
            band_data_arrays.append((band, band_data))
            band_names.append(band)

        if output_dir:
            file_path = os.path.join(output_dir, f"{item.id}_{band}.tif")
            band_data.rio.to_raster(file_path)
            result[band] = file_path
        else:
            result[band] = band_data

    # Merge bands if requested
    if merge_bands and output_dir:
        if merged_filename is None:
            merged_filename = f"{item.id}_merged.tif"

        merged_path = os.path.join(output_dir, merged_filename)

        # Check if merged file exists and skip if overwrite is False
        if os.path.exists(merged_path) and not overwrite:
            if show_progress:
                print(
                    f"Merged file {merged_path} already exists, skipping (use overwrite=True to force creation)."
                )
            result["merged"] = merged_path
        else:
            if show_progress:
                print("Resampling and merging bands...")

            # Determine target cell size if not provided
            if cell_size is None and band_data_arrays:
                # Use the resolution of the first band (usually 10m for B02, B03, B04, B08)
                # Get the affine transform (containing resolution info)
                first_band_data = band_data_arrays[0][1]
                # Extract resolution from transform
                cell_size = abs(first_band_data.rio.transform()[0])
                if show_progress:
                    print(f"Using detected resolution: {cell_size}m")
            elif cell_size is None:
                # Default to 10m if no bands are available
                cell_size = 10
                if show_progress:
                    print(f"Using default resolution: {cell_size}m")

            # Process bands in memory-efficient way
            for i, (band_name, data_array) in enumerate(band_data_arrays):
                if show_progress:
                    print(f"Processing band: {band_name}")

                # Get current resolution
                current_res = abs(data_array.rio.transform()[0])

                # Resample if needed
                if (
                    abs(current_res - cell_size) > 0.01
                ):  # Small tolerance for floating point comparison
                    if show_progress:
                        print(
                            f"Resampling {band_name} from {current_res}m to {cell_size}m"
                        )

                    # Use bilinear for downsampling (higher to lower resolution)
                    # Use nearest for upsampling (lower to higher resolution)
                    resampling_method = (
                        Resampling.bilinear
                        if current_res < cell_size
                        else Resampling.nearest
                    )

                    resampled = data_array.rio.reproject(
                        data_array.rio.crs,
                        resolution=(cell_size, cell_size),
                        resampling=resampling_method,
                    )
                    resampled_arrays.append(resampled)
                else:
                    resampled_arrays.append(data_array)

            if show_progress:
                print("Stacking bands...")

            # Concatenate all resampled arrays along the band dimension
            try:
                merged_data = xr.concat(resampled_arrays, dim="band")

                if show_progress:
                    print(f"Writing merged data to {merged_path}...")

                # Add description metadata
                merged_data.attrs["description"] = (
                    f"Multi-band image containing {', '.join(band_names)}"
                )

                # Create a dictionary mapping band indices to band names
                band_descriptions = {}
                for i, name in enumerate(band_names):
                    band_descriptions[i + 1] = name

                # Write the merged data to file with band descriptions
                merged_data.rio.to_raster(
                    merged_path,
                    tags={"BAND_NAMES": ",".join(band_names)},
                    descriptions=band_names,
                )

                result["merged"] = merged_path

                if show_progress:
                    print(f"Merged bands saved to: {merged_path}")
                    print(f"Band order in merged file: {', '.join(band_names)}")
            except Exception as e:
                if show_progress:
                    print(f"Error during merging: {str(e)}")
                    print(f"Error details: {type(e).__name__}: {str(e)}")
                raise

    return result
