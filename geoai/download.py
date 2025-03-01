"""This module provides functions to download data, including NAIP imagery and building data from Overture Maps."""

import os
from typing import List, Tuple, Optional, Dict, Any
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import requests
import subprocess
import logging

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
