import json
import math
import os
from PIL import Image
from pathlib import Path
import requests
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.windows import Window
from rasterio import features
from rasterio.plot import show
from shapely.geometry import box, shape, mapping, Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import RandomRotation
from shapely.affinity import rotate
import torch
import cv2

try:
    import torchgeo
except ImportError as e:
    raise ImportError(
        "Your torchgeo version is too old. Please upgrade to the latest version using 'pip install -U torchgeo'."
    )


def download_file(url, output_path=None, overwrite=False):
    """
    Download a file from a given URL with a progress bar.

    Args:
        url (str): The URL of the file to download.
        output_path (str, optional): The path where the downloaded file will be saved.
            If not provided, the filename from the URL will be used.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.

    Returns:
        str: The path to the downloaded file.
    """
    # Get the filename from the URL if output_path is not provided
    if output_path is None:
        output_path = os.path.basename(url)

    # Check if the file already exists
    if os.path.exists(output_path) and not overwrite:
        print(f"File already exists: {output_path}")
        return output_path

    # Send a streaming GET request
    response = requests.get(url, stream=True, timeout=50)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Get the total file size if available
    total_size = int(response.headers.get("content-length", 0))

    # Open the output file
    with (
        open(output_path, "wb") as file,
        tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar,
    ):

        # Download the file in chunks and update the progress bar
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                progress_bar.update(len(chunk))

    return output_path


def get_raster_info(raster_path):
    """Display basic information about a raster dataset.

    Args:
        raster_path (str): Path to the raster file

    Returns:
        dict: Dictionary containing the basic information about the raster
    """
    # Open the raster dataset
    with rasterio.open(raster_path) as src:
        # Get basic metadata
        info = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else "No CRS defined",
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": (src.transform[0], -src.transform[4]),
            "nodata": src.nodata,
        }

        # Calculate statistics for each band
        stats = []
        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)
            band_stats = {
                "band": i,
                "min": float(band.min()),
                "max": float(band.max()),
                "mean": float(band.mean()),
                "std": float(band.std()),
            }
            stats.append(band_stats)

        info["band_stats"] = stats

    return info


def get_raster_stats(raster_path, divide_by=1.0):
    """Calculate statistics for each band in a raster dataset.

    This function computes min, max, mean, and standard deviation values
    for each band in the provided raster, returning results in a dictionary
    with lists for each statistic type.

    Args:
        raster_path (str): Path to the raster file
        divide_by (float, optional): Value to divide pixel values by.
            Defaults to 1.0, which keeps the original pixel

    Returns:
        dict: Dictionary containing lists of statistics with keys:
            - 'min': List of minimum values for each band
            - 'max': List of maximum values for each band
            - 'mean': List of mean values for each band
            - 'std': List of standard deviation values for each band
    """
    # Initialize the results dictionary with empty lists
    stats = {"min": [], "max": [], "mean": [], "std": []}

    # Open the raster dataset
    with rasterio.open(raster_path) as src:
        # Calculate statistics for each band
        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)

            # Append statistics for this band to each list
            stats["min"].append(float(band.min()) / divide_by)
            stats["max"].append(float(band.max()) / divide_by)
            stats["mean"].append(float(band.mean()) / divide_by)
            stats["std"].append(float(band.std()) / divide_by)

    return stats


def print_raster_info(raster_path, show_preview=True, figsize=(10, 8)):
    """Print formatted information about a raster dataset and optionally show a preview.

    Args:
        raster_path (str): Path to the raster file
        show_preview (bool, optional): Whether to display a visual preview of the raster.
            Defaults to True.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).

    Returns:
        dict: Dictionary containing raster information if successful, None otherwise
    """
    try:
        info = get_raster_info(raster_path)

        # Print basic information
        print(f"===== RASTER INFORMATION: {raster_path} =====")
        print(f"Driver: {info['driver']}")
        print(f"Dimensions: {info['width']} x {info['height']} pixels")
        print(f"Number of bands: {info['count']}")
        print(f"Data type: {info['dtype']}")
        print(f"Coordinate Reference System: {info['crs']}")
        print(f"Georeferenced Bounds: {info['bounds']}")
        print(f"Pixel Resolution: {info['resolution'][0]}, {info['resolution'][1]}")
        print(f"NoData Value: {info['nodata']}")

        # Print band statistics
        print("\n----- Band Statistics -----")
        for band_stat in info["band_stats"]:
            print(f"Band {band_stat['band']}:")
            print(f"  Min: {band_stat['min']:.2f}")
            print(f"  Max: {band_stat['max']:.2f}")
            print(f"  Mean: {band_stat['mean']:.2f}")
            print(f"  Std Dev: {band_stat['std']:.2f}")

        # Show a preview if requested
        if show_preview:
            with rasterio.open(raster_path) as src:
                # For multi-band images, show RGB composite or first band
                if src.count >= 3:
                    # Try to show RGB composite
                    rgb = np.dstack([src.read(i) for i in range(1, 4)])
                    plt.figure(figsize=figsize)
                    plt.imshow(rgb)
                    plt.title(f"RGB Preview: {raster_path}")
                else:
                    # Show first band for single-band images
                    plt.figure(figsize=figsize)
                    show(
                        src.read(1),
                        cmap="viridis",
                        title=f"Band 1 Preview: {raster_path}",
                    )
                    plt.colorbar(label="Pixel Value")
                plt.show()

    except Exception as e:
        print(f"Error reading raster: {str(e)}")


def get_raster_info_gdal(raster_path):
    """Get basic information about a raster dataset using GDAL.

    Args:
        raster_path (str): Path to the raster file

    Returns:
        dict: Dictionary containing the basic information about the raster,
            or None if the file cannot be opened
    """

    from osgeo import gdal

    # Open the dataset
    ds = gdal.Open(raster_path)
    if ds is None:
        print(f"Error: Could not open {raster_path}")
        return None

    # Get basic information
    info = {
        "driver": ds.GetDriver().ShortName,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "count": ds.RasterCount,
        "projection": ds.GetProjection(),
        "geotransform": ds.GetGeoTransform(),
    }

    # Calculate resolution
    gt = ds.GetGeoTransform()
    if gt:
        info["resolution"] = (abs(gt[1]), abs(gt[5]))
        info["origin"] = (gt[0], gt[3])

    # Get band information
    bands_info = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        stats = band.GetStatistics(True, True)
        band_info = {
            "band": i,
            "datatype": gdal.GetDataTypeName(band.DataType),
            "min": stats[0],
            "max": stats[1],
            "mean": stats[2],
            "std": stats[3],
            "nodata": band.GetNoDataValue(),
        }
        bands_info.append(band_info)

    info["bands"] = bands_info

    # Close the dataset
    ds = None

    return info


def get_vector_info(vector_path):
    """Display basic information about a vector dataset using GeoPandas.

    Args:
        vector_path (str): Path to the vector file

    Returns:
        dict: Dictionary containing the basic information about the vector dataset
    """
    # Open the vector dataset
    gdf = (
        gpd.read_parquet(vector_path)
        if vector_path.endswith(".parquet")
        else gpd.read_file(vector_path)
    )

    # Get basic metadata
    info = {
        "file_path": vector_path,
        "driver": os.path.splitext(vector_path)[1][1:].upper(),  # Format from extension
        "feature_count": len(gdf),
        "crs": str(gdf.crs),
        "geometry_type": str(gdf.geom_type.value_counts().to_dict()),
        "attribute_count": len(gdf.columns) - 1,  # Subtract the geometry column
        "attribute_names": list(gdf.columns[gdf.columns != "geometry"]),
        "bounds": gdf.total_bounds.tolist(),
    }

    # Add statistics about numeric attributes
    numeric_columns = gdf.select_dtypes(include=["number"]).columns
    attribute_stats = {}
    for col in numeric_columns:
        if col != "geometry":
            attribute_stats[col] = {
                "min": gdf[col].min(),
                "max": gdf[col].max(),
                "mean": gdf[col].mean(),
                "std": gdf[col].std(),
                "null_count": gdf[col].isna().sum(),
            }

    info["attribute_stats"] = attribute_stats

    return info


def print_vector_info(vector_path, show_preview=True, figsize=(10, 8)):
    """Print formatted information about a vector dataset and optionally show a preview.

    Args:
        vector_path (str): Path to the vector file
        show_preview (bool, optional): Whether to display a visual preview of the vector data.
            Defaults to True.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).

    Returns:
        dict: Dictionary containing vector information if successful, None otherwise
    """
    try:
        info = get_vector_info(vector_path)

        # Print basic information
        print(f"===== VECTOR INFORMATION: {vector_path} =====")
        print(f"Driver: {info['driver']}")
        print(f"Feature count: {info['feature_count']}")
        print(f"Geometry types: {info['geometry_type']}")
        print(f"Coordinate Reference System: {info['crs']}")
        print(f"Bounds: {info['bounds']}")
        print(f"Number of attributes: {info['attribute_count']}")
        print(f"Attribute names: {', '.join(info['attribute_names'])}")

        # Print attribute statistics
        if info["attribute_stats"]:
            print("\n----- Attribute Statistics -----")
            for attr, stats in info["attribute_stats"].items():
                print(f"Attribute: {attr}")
                for stat_name, stat_value in stats.items():
                    print(
                        f"  {stat_name}: {stat_value:.4f}"
                        if isinstance(stat_value, float)
                        else f"  {stat_name}: {stat_value}"
                    )

        # Show a preview if requested
        if show_preview:
            gdf = (
                gpd.read_parquet(vector_path)
                if vector_path.endswith(".parquet")
                else gpd.read_file(vector_path)
            )
            fig, ax = plt.subplots(figsize=figsize)
            gdf.plot(ax=ax, cmap="viridis")
            ax.set_title(f"Preview: {vector_path}")
            plt.tight_layout()
            plt.show()

            # # Show a sample of the attribute table
            # if not gdf.empty:
            #     print("\n----- Sample of attribute table (first 5 rows) -----")
            #     print(gdf.head().to_string())

    except Exception as e:
        print(f"Error reading vector data: {str(e)}")


# Alternative implementation using OGR directly
def get_vector_info_ogr(vector_path):
    """Get basic information about a vector dataset using OGR.

    Args:
        vector_path (str): Path to the vector file

    Returns:
        dict: Dictionary containing the basic information about the vector dataset,
            or None if the file cannot be opened
    """
    from osgeo import ogr

    # Register all OGR drivers
    ogr.RegisterAll()

    # Open the dataset
    ds = ogr.Open(vector_path)
    if ds is None:
        print(f"Error: Could not open {vector_path}")
        return None

    # Basic dataset information
    info = {
        "file_path": vector_path,
        "driver": ds.GetDriver().GetName(),
        "layer_count": ds.GetLayerCount(),
        "layers": [],
    }

    # Extract information for each layer
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayer(i)
        layer_info = {
            "name": layer.GetName(),
            "feature_count": layer.GetFeatureCount(),
            "geometry_type": ogr.GeometryTypeToName(layer.GetGeomType()),
            "spatial_ref": (
                layer.GetSpatialRef().ExportToWkt() if layer.GetSpatialRef() else "None"
            ),
            "extent": layer.GetExtent(),
            "fields": [],
        }

        # Get field information
        defn = layer.GetLayerDefn()
        for j in range(defn.GetFieldCount()):
            field_defn = defn.GetFieldDefn(j)
            field_info = {
                "name": field_defn.GetName(),
                "type": field_defn.GetTypeName(),
                "width": field_defn.GetWidth(),
                "precision": field_defn.GetPrecision(),
            }
            layer_info["fields"].append(field_info)

        info["layers"].append(layer_info)

    # Close the dataset
    ds = None

    return info


def analyze_vector_attributes(vector_path, attribute_name):
    """Analyze a specific attribute in a vector dataset and create a histogram.

    Args:
        vector_path (str): Path to the vector file
        attribute_name (str): Name of the attribute to analyze

    Returns:
        dict: Dictionary containing analysis results for the attribute
    """
    try:
        gdf = gpd.read_file(vector_path)

        # Check if attribute exists
        if attribute_name not in gdf.columns:
            print(f"Attribute '{attribute_name}' not found in the dataset")
            return None

        # Get the attribute series
        attr = gdf[attribute_name]

        # Perform different analyses based on data type
        if pd.api.types.is_numeric_dtype(attr):
            # Numeric attribute
            analysis = {
                "attribute": attribute_name,
                "type": "numeric",
                "count": attr.count(),
                "null_count": attr.isna().sum(),
                "min": attr.min(),
                "max": attr.max(),
                "mean": attr.mean(),
                "median": attr.median(),
                "std": attr.std(),
                "unique_values": attr.nunique(),
            }

            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(attr.dropna(), bins=20, alpha=0.7, color="blue")
            plt.title(f"Histogram of {attribute_name}")
            plt.xlabel(attribute_name)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.show()

        else:
            # Categorical attribute
            analysis = {
                "attribute": attribute_name,
                "type": "categorical",
                "count": attr.count(),
                "null_count": attr.isna().sum(),
                "unique_values": attr.nunique(),
                "value_counts": attr.value_counts().to_dict(),
            }

            # Create bar plot for top categories
            top_n = min(10, attr.nunique())
            plt.figure(figsize=(10, 6))
            attr.value_counts().head(top_n).plot(kind="bar", color="skyblue")
            plt.title(f"Top {top_n} values for {attribute_name}")
            plt.xlabel(attribute_name)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return analysis

    except Exception as e:
        print(f"Error analyzing attribute: {str(e)}")
        return None


def visualize_vector_by_attribute(
    vector_path, attribute_name, cmap="viridis", figsize=(10, 8)
):
    """Create a thematic map visualization of vector data based on an attribute.

    Args:
        vector_path (str): Path to the vector file
        attribute_name (str): Name of the attribute to visualize
        cmap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).

    Returns:
        bool: True if visualization was successful, False otherwise
    """
    try:
        # Read the vector data
        gdf = gpd.read_file(vector_path)

        # Check if attribute exists
        if attribute_name not in gdf.columns:
            print(f"Attribute '{attribute_name}' not found in the dataset")
            return False

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Determine plot type based on data type
        if pd.api.types.is_numeric_dtype(gdf[attribute_name]):
            # Continuous data
            gdf.plot(column=attribute_name, cmap=cmap, legend=True, ax=ax)
        else:
            # Categorical data
            gdf.plot(column=attribute_name, categorical=True, legend=True, ax=ax)

        # Add title and labels
        ax.set_title(f"{os.path.basename(vector_path)} - {attribute_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Add basemap or additional elements if available
        # Note: Additional options could be added here for more complex maps

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error visualizing data: {str(e)}")


def clip_raster_by_bbox(
    input_raster, output_raster, bbox, bands=None, bbox_type="geo", bbox_crs=None
):
    """
    Clip a raster dataset using a bounding box and optionally select specific bands.

    Args:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path where the clipped raster will be saved.
        bbox (tuple): Bounding box coordinates either as:
                     - Geographic coordinates (minx, miny, maxx, maxy) if bbox_type="geo"
                     - Pixel indices (min_row, min_col, max_row, max_col) if bbox_type="pixel"
        bands (list, optional): List of band indices to keep (1-based indexing).
                               If None, all bands will be kept.
        bbox_type (str, optional): Type of bounding box coordinates. Either "geo" for
                                  geographic coordinates or "pixel" for row/column indices.
                                  Default is "geo".
        bbox_crs (str or dict, optional): CRS of the bbox if different from the raster CRS.
                                         Can be provided as EPSG code (e.g., "EPSG:4326") or
                                         as a proj4 string. Only applies when bbox_type="geo".
                                         If None, assumes bbox is in the same CRS as the raster.

    Returns:
        str: Path to the clipped output raster.

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If the bbox is invalid, bands are out of range, or bbox_type is invalid.
        RuntimeError: If the clipping operation fails.

    Examples:
        # Clip using geographic coordinates in the same CRS as the raster
        >>> clip_raster_by_bbox('input.tif', 'clipped_geo.tif', (100, 200, 300, 400))
        'clipped_geo.tif'

        # Clip using WGS84 coordinates when the raster is in a different CRS
        >>> clip_raster_by_bbox('input.tif', 'clipped_wgs84.tif', (-122.5, 37.7, -122.4, 37.8),
        ...                     bbox_crs="EPSG:4326")
        'clipped_wgs84.tif'

        # Clip using row/column indices
        >>> clip_raster_by_bbox('input.tif', 'clipped_pixel.tif', (50, 100, 150, 200),
        ...                     bbox_type="pixel")
        'clipped_pixel.tif'

        # Clip with band selection
        >>> clip_raster_by_bbox('input.tif', 'clipped_bands.tif', (100, 200, 300, 400),
        ...                     bands=[1, 3])
        'clipped_bands.tif'
    """
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    # Validate bbox_type
    if bbox_type not in ["geo", "pixel"]:
        raise ValueError("bbox_type must be either 'geo' or 'pixel'")

    # Validate bbox
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")

    # Open the source raster
    with rasterio.open(input_raster) as src:
        # Get the source CRS
        src_crs = src.crs

        # Handle different bbox types
        if bbox_type == "geo":
            minx, miny, maxx, maxy = bbox

            # Validate geographic bbox
            if minx >= maxx or miny >= maxy:
                raise ValueError(
                    "Invalid geographic bbox. Expected (minx, miny, maxx, maxy) where minx < maxx and miny < maxy"
                )

            # If bbox_crs is provided and different from the source CRS, transform the bbox
            if bbox_crs is not None and bbox_crs != src_crs:
                try:
                    # Transform bbox coordinates from bbox_crs to src_crs
                    minx, miny, maxx, maxy = transform_bounds(
                        bbox_crs, src_crs, minx, miny, maxx, maxy
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to transform bbox from {bbox_crs} to {src_crs}: {str(e)}"
                    )

            # Calculate the pixel window from geographic coordinates
            window = src.window(minx, miny, maxx, maxy)

            # Use the same bounds for the output transform
            output_bounds = (minx, miny, maxx, maxy)

        else:  # bbox_type == "pixel"
            min_row, min_col, max_row, max_col = bbox

            # Validate pixel bbox
            if min_row >= max_row or min_col >= max_col:
                raise ValueError(
                    "Invalid pixel bbox. Expected (min_row, min_col, max_row, max_col) where min_row < max_row and min_col < max_col"
                )

            if (
                min_row < 0
                or min_col < 0
                or max_row > src.height
                or max_col > src.width
            ):
                raise ValueError(
                    f"Pixel indices out of bounds. Raster dimensions are {src.height} rows x {src.width} columns"
                )

            # Create a window from pixel coordinates
            window = Window(min_col, min_row, max_col - min_col, max_row - min_row)

            # Calculate the geographic bounds for this window
            window_transform = src.window_transform(window)
            output_bounds = rasterio.transform.array_bounds(
                window.height, window.width, window_transform
            )
            # Reorder to (minx, miny, maxx, maxy)
            output_bounds = (
                output_bounds[0],
                output_bounds[1],
                output_bounds[2],
                output_bounds[3],
            )

        # Get window dimensions
        window_width = int(window.width)
        window_height = int(window.height)

        # Check if the window is valid
        if window_width <= 0 or window_height <= 0:
            raise ValueError("Bounding box results in an empty window")

        # Handle band selection
        if bands is None:
            # Use all bands
            bands_to_read = list(range(1, src.count + 1))
        else:
            # Validate band indices
            if not all(1 <= b <= src.count for b in bands):
                raise ValueError(f"Band indices must be between 1 and {src.count}")
            bands_to_read = bands

        # Calculate new transform for the clipped raster
        new_transform = from_bounds(
            output_bounds[0],
            output_bounds[1],
            output_bounds[2],
            output_bounds[3],
            window_width,
            window_height,
        )

        # Create a metadata dictionary for the output
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": window_height,
                "width": window_width,
                "transform": new_transform,
                "count": len(bands_to_read),
            }
        )

        # Read the data for the selected bands
        data = []
        for band_idx in bands_to_read:
            band_data = src.read(band_idx, window=window)
            data.append(band_data)

        # Stack the bands into a single array
        if len(data) > 1:
            clipped_data = np.stack(data)
        else:
            clipped_data = data[0][np.newaxis, :, :]

        # Write the output raster
        with rasterio.open(output_raster, "w", **out_meta) as dst:
            dst.write(clipped_data)

    return output_raster


def raster_to_vector(
    raster_path,
    output_path=None,
    threshold=0,
    min_area=10,
    simplify_tolerance=None,
    class_values=None,
    attribute_name="class",
    output_format="geojson",
    plot_result=False,
):
    """
    Convert a raster label mask to vector polygons.

    Args:
        raster_path (str): Path to the input raster file (e.g., GeoTIFF).
        output_path (str): Path to save the output vector file. If None, returns GeoDataFrame without saving.
        threshold (int/float): Pixel values greater than this threshold will be vectorized.
        min_area (float): Minimum polygon area in square map units to keep.
        simplify_tolerance (float): Tolerance for geometry simplification. None for no simplification.
        class_values (list): Specific pixel values to vectorize. If None, all values > threshold are vectorized.
        attribute_name (str): Name of the attribute field for the class values.
        output_format (str): Format for output file - 'geojson', 'shapefile', 'gpkg'.
        plot_result (bool): Whether to plot the resulting polygons overlaid on the raster.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the vectorized polygons.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the data
        data = src.read(1)

        # Get metadata
        transform = src.transform
        crs = src.crs

        # Create mask based on threshold and class values
        if class_values is not None:
            # Create a mask for each specified class value
            masks = {val: (data == val) for val in class_values}
        else:
            # Create a mask for values above threshold
            masks = {1: (data > threshold)}
            class_values = [1]  # Default class

        # Initialize list to store features
        all_features = []

        # Process each class value
        for class_val in class_values:
            mask = masks[class_val]

            # Vectorize the mask
            for geom, value in features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform
            ):
                # Convert to shapely geometry
                geom = shape(geom)

                # Skip small polygons
                if geom.area < min_area:
                    continue

                # Simplify geometry if requested
                if simplify_tolerance is not None:
                    geom = geom.simplify(simplify_tolerance)

                # Add to features list with class value
                all_features.append({"geometry": geom, attribute_name: class_val})

        # Create GeoDataFrame
        if all_features:
            gdf = gpd.GeoDataFrame(all_features, crs=crs)
        else:
            print("Warning: No features were extracted from the raster.")
            # Return empty GeoDataFrame with correct CRS
            gdf = gpd.GeoDataFrame([], geometry=[], crs=crs)

        # Save to file if requested
        if output_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Save to file based on format
            if output_format.lower() == "geojson":
                gdf.to_file(output_path, driver="GeoJSON")
            elif output_format.lower() == "shapefile":
                gdf.to_file(output_path)
            elif output_format.lower() == "gpkg":
                gdf.to_file(output_path, driver="GPKG")
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            print(f"Vectorized data saved to {output_path}")

        # Plot result if requested
        if plot_result:
            fig, ax = plt.subplots(figsize=(12, 12))

            # Plot raster
            raster_img = src.read()
            if raster_img.shape[0] == 1:
                plt.imshow(raster_img[0], cmap="viridis", alpha=0.7)
            else:
                # Use first 3 bands for RGB display
                rgb = raster_img[:3].transpose(1, 2, 0)
                # Normalize for display
                rgb = np.clip(rgb / rgb.max(), 0, 1)
                plt.imshow(rgb)

            # Plot vector boundaries
            if not gdf.empty:
                gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)

            plt.title("Raster with Vectorized Boundaries")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return gdf


def batch_raster_to_vector(
    input_dir,
    output_dir,
    pattern="*.tif",
    threshold=0,
    min_area=10,
    simplify_tolerance=None,
    class_values=None,
    attribute_name="class",
    output_format="geojson",
    merge_output=False,
    merge_filename="merged_vectors",
):
    """
    Batch convert multiple raster files to vector polygons.

    Args:
        input_dir (str): Directory containing input raster files.
        output_dir (str): Directory to save output vector files.
        pattern (str): Pattern to match raster files (e.g., '*.tif').
        threshold (int/float): Pixel values greater than this threshold will be vectorized.
        min_area (float): Minimum polygon area in square map units to keep.
        simplify_tolerance (float): Tolerance for geometry simplification. None for no simplification.
        class_values (list): Specific pixel values to vectorize. If None, all values > threshold are vectorized.
        attribute_name (str): Name of the attribute field for the class values.
        output_format (str): Format for output files - 'geojson', 'shapefile', 'gpkg'.
        merge_output (bool): Whether to merge all output vectors into a single file.
        merge_filename (str): Filename for the merged output (without extension).

    Returns:
        geopandas.GeoDataFrame or None: If merge_output is True, returns the merged GeoDataFrame.
    """
    import glob

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of raster files
    raster_files = glob.glob(os.path.join(input_dir, pattern))

    if not raster_files:
        print(f"No files matching pattern '{pattern}' found in {input_dir}")
        return None

    print(f"Found {len(raster_files)} raster files to process")

    # Process each raster file
    gdfs = []
    for raster_file in tqdm(raster_files, desc="Processing rasters"):
        # Get output filename
        base_name = os.path.splitext(os.path.basename(raster_file))[0]
        if output_format.lower() == "geojson":
            out_file = os.path.join(output_dir, f"{base_name}.geojson")
        elif output_format.lower() == "shapefile":
            out_file = os.path.join(output_dir, f"{base_name}.shp")
        elif output_format.lower() == "gpkg":
            out_file = os.path.join(output_dir, f"{base_name}.gpkg")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Convert raster to vector
        if merge_output:
            # Don't save individual files if merging
            gdf = raster_to_vector(
                raster_file,
                output_path=None,
                threshold=threshold,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance,
                class_values=class_values,
                attribute_name=attribute_name,
            )

            # Add filename as attribute
            if not gdf.empty:
                gdf["source_file"] = base_name
                gdfs.append(gdf)
        else:
            # Save individual files
            raster_to_vector(
                raster_file,
                output_path=out_file,
                threshold=threshold,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance,
                class_values=class_values,
                attribute_name=attribute_name,
                output_format=output_format,
            )

    # Merge output if requested
    if merge_output and gdfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        # Set CRS to the CRS of the first GeoDataFrame
        if merged_gdf.crs is None and gdfs:
            merged_gdf.crs = gdfs[0].crs

        # Save merged output
        if output_format.lower() == "geojson":
            merged_file = os.path.join(output_dir, f"{merge_filename}.geojson")
            merged_gdf.to_file(merged_file, driver="GeoJSON")
        elif output_format.lower() == "shapefile":
            merged_file = os.path.join(output_dir, f"{merge_filename}.shp")
            merged_gdf.to_file(merged_file)
        elif output_format.lower() == "gpkg":
            merged_file = os.path.join(output_dir, f"{merge_filename}.gpkg")
            merged_gdf.to_file(merged_file, driver="GPKG")

        print(f"Merged vector data saved to {merged_file}")
        return merged_gdf

    return None


def vector_to_raster(
    vector_path,
    output_path=None,
    reference_raster=None,
    attribute_field=None,
    output_shape=None,
    transform=None,
    pixel_size=None,
    bounds=None,
    crs=None,
    all_touched=False,
    fill_value=0,
    dtype=np.uint8,
    nodata=None,
    plot_result=False,
):
    """
    Convert vector data to a raster.

    Args:
        vector_path (str or GeoDataFrame): Path to the input vector file or a GeoDataFrame.
        output_path (str): Path to save the output raster file. If None, returns the array without saving.
        reference_raster (str): Path to a reference raster for dimensions, transform and CRS.
        attribute_field (str): Field name in the vector data to use for pixel values.
            If None, all vector features will be burned with value 1.
        output_shape (tuple): Shape of the output raster as (height, width).
            Required if reference_raster is not provided.
        transform (affine.Affine): Affine transformation matrix.
            Required if reference_raster is not provided.
        pixel_size (float or tuple): Pixel size (resolution) as single value or (x_res, y_res).
            Used to calculate transform if transform is not provided.
        bounds (tuple): Bounds of the output raster as (left, bottom, right, top).
            Used to calculate transform if transform is not provided.
        crs (str or CRS): Coordinate reference system of the output raster.
            Required if reference_raster is not provided.
        all_touched (bool): If True, all pixels touched by geometries will be burned in.
            If False, only pixels whose center is within the geometry will be burned in.
        fill_value (int): Value to fill the raster with before burning in features.
        dtype (numpy.dtype): Data type of the output raster.
        nodata (int): No data value for the output raster.
        plot_result (bool): Whether to plot the resulting raster.

    Returns:
        numpy.ndarray: The rasterized data array if output_path is None, else None.
    """
    # Load vector data
    if isinstance(vector_path, gpd.GeoDataFrame):
        gdf = vector_path
    else:
        gdf = gpd.read_file(vector_path)

    # Check if vector data is empty
    if gdf.empty:
        warnings.warn("The input vector data is empty. Creating an empty raster.")

    # Get CRS from vector data if not provided
    if crs is None and reference_raster is None:
        crs = gdf.crs

    # Get transform and output shape from reference raster if provided
    if reference_raster is not None:
        with rasterio.open(reference_raster) as src:
            transform = src.transform
            output_shape = src.shape
            crs = src.crs
            if nodata is None:
                nodata = src.nodata
    else:
        # Check if we have all required parameters
        if transform is None:
            if pixel_size is None or bounds is None:
                raise ValueError(
                    "Either reference_raster, transform, or both pixel_size and bounds must be provided."
                )

            # Calculate transform from pixel size and bounds
            if isinstance(pixel_size, (int, float)):
                x_res = y_res = float(pixel_size)
            else:
                x_res, y_res = pixel_size
                y_res = abs(y_res) * -1  # Convert to negative for north-up raster

            left, bottom, right, top = bounds
            transform = rasterio.transform.from_bounds(
                left,
                bottom,
                right,
                top,
                int((right - left) / x_res),
                int((top - bottom) / abs(y_res)),
            )

        if output_shape is None:
            # Calculate output shape from bounds and pixel size
            if bounds is None or pixel_size is None:
                raise ValueError(
                    "output_shape must be provided if reference_raster is not provided and "
                    "cannot be calculated from bounds and pixel_size."
                )

            if isinstance(pixel_size, (int, float)):
                x_res = y_res = float(pixel_size)
            else:
                x_res, y_res = pixel_size

            left, bottom, right, top = bounds
            width = int((right - left) / x_res)
            height = int((top - bottom) / abs(y_res))
            output_shape = (height, width)

    # Ensure CRS is set
    if crs is None:
        raise ValueError(
            "CRS must be provided either directly, from reference_raster, or from input vector data."
        )

    # Reproject vector data if its CRS doesn't match the output CRS
    if gdf.crs != crs:
        print(f"Reprojecting vector data from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)

    # Create empty raster filled with fill_value
    raster_data = np.full(output_shape, fill_value, dtype=dtype)

    # Burn vector features into raster
    if not gdf.empty:
        # Prepare shapes for burning
        if attribute_field is not None and attribute_field in gdf.columns:
            # Use attribute field for values
            shapes = [
                (geom, value) for geom, value in zip(gdf.geometry, gdf[attribute_field])
            ]
        else:
            # Burn with value 1
            shapes = [(geom, 1) for geom in gdf.geometry]

        # Burn shapes into raster
        burned = features.rasterize(
            shapes=shapes,
            out_shape=output_shape,
            transform=transform,
            fill=fill_value,
            all_touched=all_touched,
            dtype=dtype,
        )

        # Update raster data
        raster_data = burned

    # Save raster if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Define metadata
        metadata = {
            "driver": "GTiff",
            "height": output_shape[0],
            "width": output_shape[1],
            "count": 1,
            "dtype": raster_data.dtype,
            "crs": crs,
            "transform": transform,
        }

        # Add nodata value if provided
        if nodata is not None:
            metadata["nodata"] = nodata

        # Write raster
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(raster_data, 1)

        print(f"Rasterized data saved to {output_path}")

    # Plot result if requested
    if plot_result:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot raster
        im = ax.imshow(raster_data, cmap="viridis")
        plt.colorbar(im, ax=ax, label=attribute_field if attribute_field else "Value")

        # Plot vector boundaries for reference
        if output_path is not None:
            # Get the extent of the raster
            with rasterio.open(output_path) as src:
                bounds = src.bounds
                raster_bbox = box(*bounds)
        else:
            # Calculate extent from transform and shape
            height, width = output_shape
            left, top = transform * (0, 0)
            right, bottom = transform * (width, height)
            raster_bbox = box(left, bottom, right, top)

        # Clip vector to raster extent for clarity in plot
        if not gdf.empty:
            gdf_clipped = gpd.clip(gdf, raster_bbox)
            if not gdf_clipped.empty:
                gdf_clipped.boundary.plot(ax=ax, color="red", linewidth=1)

        plt.title("Rasterized Vector Data")
        plt.tight_layout()
        plt.show()

    return raster_data


def batch_vector_to_raster(
    vector_path,
    output_dir,
    attribute_field=None,
    reference_rasters=None,
    bounds_list=None,
    output_filename_pattern="{vector_name}_{index}",
    pixel_size=1.0,
    all_touched=False,
    fill_value=0,
    dtype=np.uint8,
    nodata=None,
):
    """
    Batch convert vector data to multiple rasters based on different extents or reference rasters.

    Args:
        vector_path (str or GeoDataFrame): Path to the input vector file or a GeoDataFrame.
        output_dir (str): Directory to save output raster files.
        attribute_field (str): Field name in the vector data to use for pixel values.
        reference_rasters (list): List of paths to reference rasters for dimensions, transform and CRS.
        bounds_list (list): List of bounds tuples (left, bottom, right, top) to use if reference_rasters not provided.
        output_filename_pattern (str): Pattern for output filenames.
            Can include {vector_name} and {index} placeholders.
        pixel_size (float or tuple): Pixel size to use if reference_rasters not provided.
        all_touched (bool): If True, all pixels touched by geometries will be burned in.
        fill_value (int): Value to fill the raster with before burning in features.
        dtype (numpy.dtype): Data type of the output raster.
        nodata (int): No data value for the output raster.

    Returns:
        list: List of paths to the created raster files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load vector data if it's a path
    if isinstance(vector_path, str):
        gdf = gpd.read_file(vector_path)
        vector_name = os.path.splitext(os.path.basename(vector_path))[0]
    else:
        gdf = vector_path
        vector_name = "vector"

    # Check input parameters
    if reference_rasters is None and bounds_list is None:
        raise ValueError("Either reference_rasters or bounds_list must be provided.")

    # Use reference_rasters if provided, otherwise use bounds_list
    if reference_rasters is not None:
        sources = reference_rasters
        is_raster_reference = True
    else:
        sources = bounds_list
        is_raster_reference = False

    # Create output filenames
    output_files = []

    # Process each source (reference raster or bounds)
    for i, source in enumerate(tqdm(sources, desc="Processing")):
        # Generate output filename
        output_filename = output_filename_pattern.format(
            vector_name=vector_name, index=i
        )
        if not output_filename.endswith(".tif"):
            output_filename += ".tif"
        output_path = os.path.join(output_dir, output_filename)

        if is_raster_reference:
            # Use reference raster
            vector_to_raster(
                vector_path=gdf,
                output_path=output_path,
                reference_raster=source,
                attribute_field=attribute_field,
                all_touched=all_touched,
                fill_value=fill_value,
                dtype=dtype,
                nodata=nodata,
            )
        else:
            # Use bounds
            vector_to_raster(
                vector_path=gdf,
                output_path=output_path,
                bounds=source,
                pixel_size=pixel_size,
                attribute_field=attribute_field,
                all_touched=all_touched,
                fill_value=fill_value,
                dtype=dtype,
                nodata=nodata,
            )

        output_files.append(output_path)

    return output_files


def export_geotiff_tiles(
    in_raster,
    out_folder,
    in_class_data,
    tile_size=256,
    stride=128,
    class_value_field="class",
    buffer_radius=0,
    max_tiles=None,
    quiet=False,
    all_touched=True,
    create_overview=False,
    skip_empty_tiles=False,
):
    """
    Export georeferenced GeoTIFF tiles and labels from raster and classification data.

    Args:
        in_raster (str): Path to input raster image
        out_folder (str): Path to output folder
        in_class_data (str): Path to classification data - can be vector file or raster
        tile_size (int): Size of tiles in pixels (square)
        stride (int): Step size between tiles
        class_value_field (str): Field containing class values (for vector data)
        buffer_radius (float): Buffer to add around features (in units of the CRS)
        max_tiles (int): Maximum number of tiles to process (None for all)
        quiet (bool): If True, suppress non-essential output
        all_touched (bool): Whether to use all_touched=True in rasterization (for vector data)
        create_overview (bool): Whether to create an overview image of all tiles
        skip_empty_tiles (bool): If True, skip tiles with no features
    """
    # Create output directories
    os.makedirs(out_folder, exist_ok=True)
    image_dir = os.path.join(out_folder, "images")
    os.makedirs(image_dir, exist_ok=True)
    label_dir = os.path.join(out_folder, "labels")
    os.makedirs(label_dir, exist_ok=True)
    ann_dir = os.path.join(out_folder, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # Determine if class data is raster or vector
    is_class_data_raster = False
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
                    print(f"Unable to open {in_class_data} as raster, trying as vector")

    # Open the input raster
    with rasterio.open(in_raster) as src:
        if not quiet:
            print(f"\nRaster info for {in_raster}:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bounds: {src.bounds}")

        # Calculate number of tiles
        num_tiles_x = math.ceil((src.width - tile_size) / stride) + 1
        num_tiles_y = math.ceil((src.height - tile_size) / stride) + 1
        total_tiles = num_tiles_x * num_tiles_y

        if max_tiles is None:
            max_tiles = total_tiles

        # Process classification data
        class_to_id = {}

        if is_class_data_raster:
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
        else:
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
                else:
                    if not quiet:
                        print(
                            f"WARNING: '{class_value_field}' not found in vector data. Using default class ID 1."
                        )
                    class_to_id = {1: 1}  # Default mapping
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

                # Process classification data to create labels
                if is_class_data_raster:
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
                else:
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

                # Skip tile if no features and skip_empty_tiles is True
                if skip_empty_tiles and not has_features:
                    pbar.update(1)
                    tile_index += 1
                    continue

                # Read image data
                image_data = src.read(window=window)

                # Export image as GeoTIFF
                image_path = os.path.join(image_dir, f"tile_{tile_index:06d}.tif")

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
                    pbar.write(f"ERROR saving image GeoTIFF: {e}")
                    stats["errors"] += 1

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

                # Export label as GeoTIFF
                label_path = os.path.join(label_dir, f"tile_{tile_index:06d}.tif")
                try:
                    with rasterio.open(label_path, "w", **label_profile) as dst:
                        dst.write(label_mask.astype(np.uint8), 1)

                    if has_features:
                        stats["tiles_with_features"] += 1
                        stats["feature_pixels"] += np.count_nonzero(label_mask)
                except Exception as e:
                    pbar.write(f"ERROR saving label GeoTIFF: {e}")
                    stats["errors"] += 1

                # Create XML annotation for object detection if using vector class data
                if (
                    not is_class_data_raster
                    and "gdf" in locals()
                    and len(window_features) > 0
                ):
                    # Create XML annotation
                    root = ET.Element("annotation")
                    ET.SubElement(root, "folder").text = "images"
                    ET.SubElement(root, "filename").text = f"tile_{tile_index:06d}.tif"

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

        # Create overview image if requested
        if create_overview and stats["tile_coordinates"]:
            try:
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
                sample_label = os.path.join(label_dir, f"tile_0.tif")

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

        # Return statistics dictionary for further processing if needed
        return stats


def create_overview_image(
    src, tile_coordinates, output_path, tile_size, stride, geojson_path=None
):
    """Create an overview image showing all tiles and their status, with optional GeoJSON export.

    Args:
        src (rasterio.io.DatasetReader): The source raster dataset.
        tile_coordinates (list): A list of dictionaries containing tile information.
        output_path (str): The path where the overview image will be saved.
        tile_size (int): The size of each tile in pixels.
        stride (int): The stride between tiles in pixels. Controls overlap between adjacent tiles.
        geojson_path (str, optional): If provided, exports the tile rectangles as GeoJSON to this path.

    Returns:
        str: Path to the saved overview image.
    """
    # Read a reduced version of the source image
    overview_scale = max(
        1, int(max(src.width, src.height) / 2000)
    )  # Scale to max ~2000px
    overview_width = src.width // overview_scale
    overview_height = src.height // overview_scale

    # Read downsampled image
    overview_data = src.read(
        out_shape=(src.count, overview_height, overview_width),
        resampling=rasterio.enums.Resampling.average,
    )

    # Create RGB image for display
    if overview_data.shape[0] >= 3:
        rgb = np.moveaxis(overview_data[:3], 0, -1)
    else:
        # For single band, create grayscale RGB
        rgb = np.stack([overview_data[0], overview_data[0], overview_data[0]], axis=-1)

    # Normalize for display
    for i in range(rgb.shape[-1]):
        band = rgb[..., i]
        non_zero = band[band > 0]
        if len(non_zero) > 0:
            p2, p98 = np.percentile(non_zero, (2, 98))
            rgb[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)

    # Create figure
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)

    # If GeoJSON export is requested, prepare GeoJSON structures
    if geojson_path:
        features = []

    # Draw tile boundaries
    for tile in tile_coordinates:
        # Convert bounds to pixel coordinates in overview
        bounds = tile["bounds"]
        # Calculate scaled pixel coordinates
        x_min = int((tile["x"]) / overview_scale)
        y_min = int((tile["y"]) / overview_scale)
        width = int(tile_size / overview_scale)
        height = int(tile_size / overview_scale)

        # Draw rectangle
        color = "lime" if tile["has_features"] else "red"
        rect = plt.Rectangle(
            (x_min, y_min), width, height, fill=False, edgecolor=color, linewidth=0.5
        )
        plt.gca().add_patch(rect)

        # Add tile number if not too crowded
        if width > 20 and height > 20:
            plt.text(
                x_min + width / 2,
                y_min + height / 2,
                str(tile["index"]),
                color="white",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Add to GeoJSON features if exporting
        if geojson_path:
            # Create a polygon from the bounds (already in geo-coordinates)
            minx, miny, maxx, maxy = bounds
            polygon = box(minx, miny, maxx, maxy)

            # Calculate overlap with neighboring tiles
            overlap = 0
            if stride < tile_size:
                overlap = tile_size - stride

            # Create a GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "index": tile["index"],
                    "has_features": tile["has_features"],
                    "bounds_pixel": [
                        tile["x"],
                        tile["y"],
                        tile["x"] + tile_size,
                        tile["y"] + tile_size,
                    ],
                    "tile_size_px": tile_size,
                    "stride_px": stride,
                    "overlap_px": overlap,
                },
            }

            # Add any additional properties from the tile
            for key, value in tile.items():
                if key not in ["x", "y", "index", "has_features", "bounds"]:
                    feature["properties"][key] = value

            features.append(feature)

    plt.title("Tile Overview (Green = Contains Features, Red = Empty)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Overview image saved to {output_path}")

    # Export GeoJSON if requested
    if geojson_path:
        geojson_collection = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "crs": (
                    src.crs.to_string()
                    if hasattr(src.crs, "to_string")
                    else str(src.crs)
                ),
                "total_tiles": len(features),
                "source_raster_dimensions": [src.width, src.height],
            },
        }

        # Save to file
        with open(geojson_path, "w") as f:
            json.dump(geojson_collection, f)

        print(f"GeoJSON saved to {geojson_path}")

    return output_path


def export_tiles_to_geojson(
    tile_coordinates, src, output_path, tile_size=None, stride=None
):
    """
    Export tile rectangles directly to GeoJSON without creating an overview image.

    Args:
        tile_coordinates (list): A list of dictionaries containing tile information.
        src (rasterio.io.DatasetReader): The source raster dataset.
        output_path (str): The path where the GeoJSON will be saved.
        tile_size (int, optional): The size of each tile in pixels. Only needed if not in tile_coordinates.
        stride (int, optional): The stride between tiles in pixels. Used to calculate overlaps between tiles.

    Returns:
        str: Path to the saved GeoJSON file.
    """
    features = []

    for tile in tile_coordinates:
        # Get the size from the tile or use the provided parameter
        tile_width = tile.get("width", tile.get("size", tile_size))
        tile_height = tile.get("height", tile.get("size", tile_size))

        if tile_width is None or tile_height is None:
            raise ValueError(
                "Tile size not found in tile data and no tile_size parameter provided"
            )

        # Get bounds from the tile
        if "bounds" in tile:
            # If bounds are already in geo coordinates
            minx, miny, maxx, maxy = tile["bounds"]
        else:
            # Try to calculate bounds from transform if available
            if hasattr(src, "transform"):
                # Convert pixel coordinates to geo coordinates
                window_transform = src.transform
                x, y = tile["x"], tile["y"]
                minx = window_transform[2] + x * window_transform[0]
                maxy = window_transform[5] + y * window_transform[4]
                maxx = minx + tile_width * window_transform[0]
                miny = maxy + tile_height * window_transform[4]
            else:
                raise ValueError(
                    "Cannot determine bounds. Neither 'bounds' in tile nor transform in src."
                )

        # Calculate overlap with neighboring tiles if stride is provided
        overlap = 0
        if stride is not None and stride < tile_width:
            overlap = tile_width - stride

        # Create a polygon from the bounds
        polygon = box(minx, miny, maxx, maxy)

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "index": tile["index"],
                "has_features": tile.get("has_features", False),
                "tile_width_px": tile_width,
                "tile_height_px": tile_height,
            },
        }

        # Add overlap information if stride is provided
        if stride is not None:
            feature["properties"]["stride_px"] = stride
            feature["properties"]["overlap_px"] = overlap

        # Add additional properties from the tile
        for key, value in tile.items():
            if key not in ["bounds", "geometry"]:
                feature["properties"][key] = value

        features.append(feature)

    # Create the GeoJSON collection
    geojson_collection = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "crs": (
                src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
            ),
            "total_tiles": len(features),
            "source_raster_dimensions": (
                [src.width, src.height] if hasattr(src, "width") else None
            ),
        },
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(geojson_collection, f)

    print(f"GeoJSON saved to {output_path}")
    return output_path


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


def masks_to_vector(
    mask_path,
    output_path=None,
    simplify_tolerance=1.0,
    mask_threshold=0.5,
    min_object_area=100,
    max_object_area=None,
    nms_iou_threshold=0.5,
):
    """
    Convert a building mask GeoTIFF to vector polygons and save as a vector dataset.

    Args:
        mask_path: Path to the building masks GeoTIFF
        output_path: Path to save the output GeoJSON (default: mask_path with .geojson extension)
        simplify_tolerance: Tolerance for polygon simplification (default: self.simplify_tolerance)
        mask_threshold: Threshold for mask binarization (default: self.mask_threshold)
        min_object_area: Minimum area in pixels to keep a building (default: self.min_object_area)
        max_object_area: Maximum area in pixels to keep a building (default: self.max_object_area)
        nms_iou_threshold: IoU threshold for non-maximum suppression (default: self.nms_iou_threshold)

    Returns:
        GeoDataFrame with building footprints
    """
    # Set default output path if not provided
    # if output_path is None:
    #     output_path = os.path.splitext(mask_path)[0] + ".geojson"

    print(f"Converting mask to GeoJSON with parameters:")
    print(f"- Mask threshold: {mask_threshold}")
    print(f"- Min building area: {min_object_area}")
    print(f"- Simplify tolerance: {simplify_tolerance}")
    print(f"- NMS IoU threshold: {nms_iou_threshold}")

    # Open the mask raster
    with rasterio.open(mask_path) as src:
        # Read the mask data
        mask_data = src.read(1)
        transform = src.transform
        crs = src.crs

        # Print mask statistics
        print(f"Mask dimensions: {mask_data.shape}")
        print(f"Mask value range: {mask_data.min()} to {mask_data.max()}")

        # Prepare for connected component analysis
        # Binarize the mask based on threshold
        binary_mask = (mask_data > (mask_threshold * 255)).astype(np.uint8)

        # Apply morphological operations for better results (optional)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        print(f"Found {num_labels-1} potential buildings")  # Subtract 1 for background

        # Create list to store polygons and confidence values
        all_polygons = []
        all_confidences = []

        # Process each component (skip the first one which is background)
        for i in tqdm(range(1, num_labels)):
            # Extract this building
            area = stats[i, cv2.CC_STAT_AREA]

            # Skip if too small
            if area < min_object_area:
                continue

            # Skip if too large
            if max_object_area is not None and area > max_object_area:
                continue

            # Create a mask for this building
            building_mask = (labels == i).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process each contour
            for contour in contours:
                # Skip if too few points
                if contour.shape[0] < 3:
                    continue

                # Simplify contour if it has many points
                if contour.shape[0] > 50 and simplify_tolerance > 0:
                    epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                # Convert to list of (x, y) coordinates
                polygon_points = contour.reshape(-1, 2)

                # Convert pixel coordinates to geographic coordinates
                geo_points = []
                for x, y in polygon_points:
                    gx, gy = transform * (x, y)
                    geo_points.append((gx, gy))

                # Create Shapely polygon
                if len(geo_points) >= 3:
                    try:
                        shapely_poly = Polygon(geo_points)
                        if shapely_poly.is_valid and shapely_poly.area > 0:
                            all_polygons.append(shapely_poly)

                            # Calculate "confidence" as normalized size
                            # This is a proxy since we don't have model confidence scores
                            normalized_size = min(1.0, area / 1000)  # Cap at 1.0
                            all_confidences.append(normalized_size)
                    except Exception as e:
                        print(f"Error creating polygon: {e}")

        print(f"Created {len(all_polygons)} valid polygons")

        # Create GeoDataFrame
        if not all_polygons:
            print("No valid polygons found")
            return None

        gdf = gpd.GeoDataFrame(
            {
                "geometry": all_polygons,
                "confidence": all_confidences,
                "class": 1,  # Building class
            },
            crs=crs,
        )

        def filter_overlapping_polygons(gdf, **kwargs):
            """
            Filter overlapping polygons using non-maximum suppression.

            Args:
                gdf: GeoDataFrame with polygons
                **kwargs: Optional parameters:
                    nms_iou_threshold: IoU threshold for filtering

            Returns:
                Filtered GeoDataFrame
            """
            if len(gdf) <= 1:
                return gdf

            # Get parameters from kwargs or use instance defaults
            iou_threshold = kwargs.get("nms_iou_threshold", nms_iou_threshold)

            # Sort by confidence
            gdf = gdf.sort_values("confidence", ascending=False)

            # Fix any invalid geometries
            gdf["geometry"] = gdf["geometry"].apply(
                lambda geom: geom.buffer(0) if not geom.is_valid else geom
            )

            keep_indices = []
            polygons = gdf.geometry.values

            for i in range(len(polygons)):
                if i in keep_indices:
                    continue

                keep = True
                for j in keep_indices:
                    # Skip invalid geometries
                    if not polygons[i].is_valid or not polygons[j].is_valid:
                        continue

                    # Calculate IoU
                    try:
                        intersection = polygons[i].intersection(polygons[j]).area
                        union = polygons[i].area + polygons[j].area - intersection
                        iou = intersection / union if union > 0 else 0

                        if iou > iou_threshold:
                            keep = False
                            break
                    except Exception:
                        # Skip on topology exceptions
                        continue

                if keep:
                    keep_indices.append(i)

            return gdf.iloc[keep_indices]

        # Apply non-maximum suppression to remove overlapping polygons
        gdf = filter_overlapping_polygons(gdf, nms_iou_threshold=nms_iou_threshold)

        print(f"Final building count after filtering: {len(gdf)}")

        # Save to file
        if output_path is not None:
            gdf.to_file(output_path)
            print(f"Saved {len(gdf)} building footprints to {output_path}")

        return gdf
