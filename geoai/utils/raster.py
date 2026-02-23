"""Raster I/O and processing utilities."""

# Standard Library
import os
import subprocess
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-Party Libraries
import geopandas as gpd
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio import features
from rasterio.plot import show
from rasterio.windows import Window
from shapely.geometry import Polygon, box, shape
from tqdm import tqdm

__all__ = [
    "calc_stats",
    "get_raster_info",
    "get_raster_stats",
    "print_raster_info",
    "get_raster_info_gdal",
    "clip_raster_by_bbox",
    "raster_to_vector",
    "raster_to_vector_batch",
    "vector_to_raster",
    "batch_vector_to_raster",
    "masks_to_vector",
    "read_raster",
    "read_vector",
    "mosaic_geotiffs",
    "write_colormap",
    "get_raster_resolution",
    "stack_bands",
]


def calc_stats(dataset, divide_by: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the statistics (mean and std) for the entire dataset.

    This function is adapted from the plot_batch() function in the torchgeo library at
    https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html.
    Credit to the torchgeo developers for the original implementation.

    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.

    Args:
        dataset (RasterDataset): The dataset to calculate statistics for.
        divide_by (float, optional): The value to divide the image data by. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean and standard deviation for each band.
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    files = [
        item.object
        for item in dataset.index.intersection(dataset.index.bounds, objects=True)
    ]

    # Resetting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        img = rasterio.open(file).read() / divide_by  # type: ignore
        accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
        accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with length n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)


def get_raster_info(raster_path: str) -> Dict[str, Any]:
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


def get_raster_stats(raster_path: str, divide_by: float = 1.0) -> Dict[str, Any]:
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


def print_raster_info(
    raster_path: str, show_preview: bool = True, figsize: Tuple[int, int] = (10, 8)
) -> Optional[Dict[str, Any]]:
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


def get_raster_info_gdal(raster_path: str) -> Optional[Dict[str, Any]]:
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


def clip_raster_by_bbox(
    input_raster: str,
    output_raster: str,
    bbox: List[float],
    bands: Optional[List[int]] = None,
    bbox_type: str = "geo",
    bbox_crs: Optional[str] = None,
) -> str:
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
        Clip using geographic coordinates in the same CRS as the raster
        >>> clip_raster_by_bbox('input.tif', 'clipped_geo.tif', (100, 200, 300, 400))
        'clipped_geo.tif'

        Clip using WGS84 coordinates when the raster is in a different CRS
        >>> clip_raster_by_bbox('input.tif', 'clipped_wgs84.tif', (-122.5, 37.7, -122.4, 37.8),
        ...                     bbox_crs="EPSG:4326")
        'clipped_wgs84.tif'

        Clip using row/column indices
        >>> clip_raster_by_bbox('input.tif', 'clipped_pixel.tif', (50, 100, 150, 200),
        ...                     bbox_type="pixel")
        'clipped_pixel.tif'

        Clip with band selection
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
    raster_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0,
    min_area: float = 10,
    simplify_tolerance: Optional[float] = None,
    class_values: Optional[List[int]] = None,
    attribute_name: str = "class",
    unique_attribute_value: bool = False,
    output_format: str = "geojson",
    plot_result: bool = False,
) -> gpd.GeoDataFrame:
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
        unique_attribute_value (bool): Whether to generate unique values for each shape within a class.
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

        # Check if CRS is geographic (area in sq degrees, not sq meters)
        is_geographic = crs is not None and crs.is_geographic

        # Initialize list to store features
        all_features = []

        # Process each class value
        for class_val in class_values:
            mask = masks[class_val]
            shape_count = 1
            # Vectorize the mask
            for geom, value in features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform
            ):
                # Convert to shapely geometry
                geom = shape(geom)

                # Skip small polygons (area check deferred for geographic CRS)
                if not is_geographic and geom.area < min_area:
                    continue

                # Simplify geometry if requested
                if simplify_tolerance is not None:
                    geom = geom.simplify(simplify_tolerance)

                # Add to features list with class value
                if unique_attribute_value:
                    all_features.append(
                        {"geometry": geom, attribute_name: class_val * shape_count}
                    )
                else:
                    all_features.append({"geometry": geom, attribute_name: class_val})

                shape_count += 1

        # Create GeoDataFrame
        if all_features:
            gdf = gpd.GeoDataFrame(all_features, crs=crs)
        else:
            print("Warning: No features were extracted from the raster.")
            # Return empty GeoDataFrame with correct CRS
            gdf = gpd.GeoDataFrame([], geometry=[], crs=crs)

        # For geographic CRS, filter by area using a projected CRS
        if is_geographic and min_area > 0 and len(gdf) > 0:
            utm_crs = gdf.estimate_utm_crs()
            projected_area = gdf.to_crs(utm_crs).geometry.area
            gdf = gdf[projected_area >= min_area].reset_index(drop=True)

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


def raster_to_vector_batch(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.tif",
    threshold: float = 0,
    min_area: float = 10,
    simplify_tolerance: Optional[float] = None,
    class_values: Optional[List[int]] = None,
    attribute_name: str = "class",
    output_format: str = "geojson",
    merge_output: bool = False,
    merge_filename: str = "merged_vectors",
) -> Optional[gpd.GeoDataFrame]:
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
    vector_path: Union[str, gpd.GeoDataFrame],
    output_path: Optional[str] = None,
    reference_raster: Optional[str] = None,
    attribute_field: Optional[str] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    transform: Optional[Any] = None,
    pixel_size: Optional[float] = None,
    bounds: Optional[List[float]] = None,
    crs: Optional[str] = None,
    all_touched: bool = False,
    fill_value: Union[int, float] = 0,
    dtype: Any = np.uint8,
    nodata: Optional[Union[int, float]] = None,
    plot_result: bool = False,
) -> np.ndarray:
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
) -> List[str]:
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
        List[str]: List of paths to the created raster files.
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


def masks_to_vector(
    mask_path: str,
    output_path: Optional[str] = None,
    simplify_tolerance: float = 1.0,
    mask_threshold: float = 0.5,
    min_object_area: int = 100,
    max_object_area: Optional[int] = None,
    nms_iou_threshold: float = 0.5,
) -> Any:
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
        Any: GeoDataFrame with building footprints
    """
    import cv2  # Lazy import to avoid QGIS opencv conflicts

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


def read_vector(
    source: str, layer: Optional[str] = None, **kwargs: Any
) -> gpd.GeoDataFrame:
    """Reads vector data from various formats including GeoParquet.

    This function dynamically determines the file type based on extension
    and reads it into a GeoDataFrame. It supports both local files and HTTP/HTTPS URLs.

    Args:
        source: String path to the vector file or URL.
        layer: String or integer specifying which layer to read from multi-layer
            files (only applicable for formats like GPKG, GeoJSON, etc.).
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the underlying reader.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the vector data.

    Raises:
        ValueError: If the file format is not supported or source cannot be accessed.

    Examples:
        Read a local shapefile
        >>> gdf = read_vector("path/to/data.shp")
        >>>
        Read a GeoParquet file from URL
        >>> gdf = read_vector("https://example.com/data.parquet")
        >>>
        Read a specific layer from a GeoPackage
        >>> gdf = read_vector("path/to/data.gpkg", layer="layer_name")
    """

    import urllib.parse

    import fiona

    # Determine if source is a URL or local file
    parsed_url = urllib.parse.urlparse(source)
    is_url = parsed_url.scheme in ["http", "https"]

    # If it's a local file, check if it exists
    if not is_url and not os.path.exists(source):
        raise ValueError(f"File does not exist: {source}")

    # Get file extension
    _, ext = os.path.splitext(source)
    ext = ext.lower()

    # Handle GeoParquet files
    if ext in [".parquet", ".pq", ".geoparquet"]:
        return gpd.read_parquet(source, **kwargs)

    # Handle common vector formats
    if ext in [".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml", ".gpx"]:
        # For formats that might have multiple layers
        if ext in [".gpkg", ".gml"] and layer is not None:
            return gpd.read_file(source, layer=layer, **kwargs)
        return gpd.read_file(source, **kwargs)

    # Try to use fiona to identify valid layers for formats that might have them
    # Only attempt this for local files as fiona.listlayers might not work with URLs
    if layer is None and ext in [".gpkg", ".gml"] and not is_url:
        try:
            layers = fiona.listlayers(source)
            if layers:
                return gpd.read_file(source, layer=layers[0], **kwargs)
        except Exception:
            # If listing layers fails, we'll fall through to the generic read attempt
            pass

    # For other formats or when layer listing fails, attempt to read using GeoPandas
    try:
        return gpd.read_file(source, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not read from source '{source}': {str(e)}")


def read_raster(
    source: str,
    band: Optional[Union[int, List[int]]] = None,
    masked: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Reads raster data from various formats using rioxarray.

    This function reads raster data from local files or URLs into a rioxarray
    data structure with preserved geospatial metadata.

    Args:
        source: String path to the raster file or URL.
        band: Integer or list of integers specifying which band(s) to read.
            Defaults to None (all bands).
        masked: Boolean indicating whether to mask nodata values.
            Defaults to True.
        **kwargs: Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns:
        xarray.DataArray: A DataArray containing the raster data with geospatial
            metadata preserved.

    Raises:
        ValueError: If the file format is not supported or source cannot be accessed.

    Examples:
        Read a local GeoTIFF
        >>> raster = read_raster("path/to/data.tif")
        >>>
        Read only band 1 from a remote GeoTIFF
        >>> raster = read_raster("https://example.com/data.tif", band=1)
        >>>
        Read a raster without masking nodata values
        >>> raster = read_raster("path/to/data.tif", masked=False)
    """
    import urllib.parse

    from rasterio.errors import RasterioIOError

    # Determine if source is a URL or local file
    parsed_url = urllib.parse.urlparse(source)
    is_url = parsed_url.scheme in ["http", "https"]

    # If it's a local file, check if it exists
    if not is_url and not os.path.exists(source):
        raise ValueError(f"Raster file does not exist: {source}")

    try:
        # Open the raster with rioxarray
        raster = rxr.open_rasterio(source, masked=masked, **kwargs)

        # Handle band selection if specified
        if band is not None:
            if isinstance(band, (list, tuple)):
                # Convert from 1-based indexing to 0-based indexing
                band_indices = [b - 1 for b in band]
                raster = raster.isel(band=band_indices)
            else:
                # Single band selection (convert from 1-based to 0-based indexing)
                raster = raster.isel(band=band - 1)

        return raster

    except RasterioIOError as e:
        raise ValueError(f"Could not read raster from source '{source}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading raster data: {str(e)}")


def mosaic_geotiffs(
    input_dir: str, output_file: str, mask_file: Optional[str] = None
) -> None:
    """Create a mosaic from all GeoTIFF files as a Cloud Optimized GeoTIFF (COG).

    This function identifies all GeoTIFF files in the specified directory,
    creates a seamless mosaic with proper handling of nodata values, and saves
    as a Cloud Optimized GeoTIFF format. If a mask file is provided, the output
    will be clipped to the extent of the mask.

    Args:
        input_dir (str): Path to the directory containing GeoTIFF files.
        output_file (str): Path to the output Cloud Optimized GeoTIFF file.
        mask_file (str, optional): Path to a mask file to clip the output.
            If provided, the output will be clipped to the extent of this mask.
            Defaults to None.

    Returns:
        bool: True if the mosaic was created successfully, False otherwise.

    Examples:
        >>> mosaic_geotiffs('naip', 'merged_naip.tif')
        True
        >>> mosaic_geotiffs('naip', 'merged_naip.tif', 'boundary.tif')
        True
    """
    import glob

    from osgeo import gdal

    gdal.UseExceptions()
    # Get all tif files in the directory
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))

    if not tif_files:
        print("No GeoTIFF files found in the specified directory.")
        return False

    # Analyze the first input file to determine compression and nodata settings
    ds = gdal.Open(tif_files[0])
    if ds is None:
        print(f"Unable to open {tif_files[0]}")
        return False

    # Get driver metadata from the first file
    driver = ds.GetDriver()
    creation_options = []

    # Check compression type
    metadata = ds.GetMetadata("IMAGE_STRUCTURE")
    if "COMPRESSION" in metadata:
        compression = metadata["COMPRESSION"]
        creation_options.append(f"COMPRESS={compression}")
    else:
        # Default compression if none detected
        creation_options.append("COMPRESS=LZW")

    # Add COG-specific creation options
    creation_options.extend(["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])

    # Check for nodata value in the first band of the first file
    band = ds.GetRasterBand(1)
    has_nodata = band.GetNoDataValue() is not None
    nodata_value = band.GetNoDataValue() if has_nodata else None

    # Close the dataset
    ds = None

    # Create a temporary VRT (Virtual Dataset)
    vrt_path = os.path.join(input_dir, "temp_mosaic.vrt")

    # Build VRT from input files with proper nodata handling
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg="nearest",
        srcNodata=nodata_value if has_nodata else None,
        VRTNodata=nodata_value if has_nodata else None,
    )
    vrt_dataset = gdal.BuildVRT(vrt_path, tif_files, options=vrt_options)

    # Close the VRT dataset to flush it to disk
    vrt_dataset = None

    # Create temp mosaic
    temp_mosaic = output_file + ".temp.tif"

    # Convert VRT to GeoTIFF with the same compression as input
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=creation_options,
        noData=nodata_value if has_nodata else None,
    )
    gdal.Translate(temp_mosaic, vrt_path, options=translate_options)

    # Apply mask if provided
    if mask_file and os.path.exists(mask_file):
        print(f"Clipping mosaic to mask: {mask_file}")

        # Create a temporary clipped file
        clipped_mosaic = output_file + ".clipped.tif"

        # Open mask file
        mask_ds = gdal.Open(mask_file)
        if mask_ds is None:
            print(f"Unable to open mask file: {mask_file}")
            # Continue without clipping
        else:
            # Get mask extent
            mask_geotransform = mask_ds.GetGeoTransform()
            mask_projection = mask_ds.GetProjection()
            mask_ulx = mask_geotransform[0]
            mask_uly = mask_geotransform[3]
            mask_lrx = mask_ulx + (mask_geotransform[1] * mask_ds.RasterXSize)
            mask_lry = mask_uly + (mask_geotransform[5] * mask_ds.RasterYSize)

            # Close mask dataset
            mask_ds = None

            # Use warp options to clip
            warp_options = gdal.WarpOptions(
                format="GTiff",
                outputBounds=[mask_ulx, mask_lry, mask_lrx, mask_uly],
                dstSRS=mask_projection,
                creationOptions=creation_options,
                srcNodata=nodata_value if has_nodata else None,
                dstNodata=nodata_value if has_nodata else None,
            )

            # Apply clipping
            gdal.Warp(clipped_mosaic, temp_mosaic, options=warp_options)

            # Remove the unclipped temp mosaic and use the clipped one
            os.remove(temp_mosaic)
            temp_mosaic = clipped_mosaic

    # Create internal overviews for the temp mosaic
    ds = gdal.Open(temp_mosaic, gdal.GA_Update)
    overview_list = [2, 4, 8, 16, 32]
    ds.BuildOverviews("NEAREST", overview_list)
    ds = None  # Close the dataset to ensure overviews are written

    # Convert the temp mosaic to a proper COG
    cog_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COPY_SRC_OVERVIEWS=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
        ],
        noData=nodata_value if has_nodata else None,
    )
    gdal.Translate(output_file, temp_mosaic, options=cog_options)

    # Clean up temporary files
    if os.path.exists(vrt_path):
        os.remove(vrt_path)
    if os.path.exists(temp_mosaic):
        os.remove(temp_mosaic)

    print(f"Cloud Optimized GeoTIFF mosaic created successfully: {output_file}")
    return True


def write_colormap(
    image: Union[str, np.ndarray],
    colormap: Union[str, Dict],
    output: Optional[str] = None,
) -> Optional[str]:
    """Write a colormap to an image.

    Args:
        image: The image to write the colormap to.
        colormap: The colormap to write to the image.
        output: The output file path.
    """
    if isinstance(colormap, str):
        colormap = leafmap.get_image_colormap(colormap)
    leafmap.write_image_colormap(image, colormap, output)


def get_raster_resolution(image_path: str) -> Tuple[float, float]:
    """Get pixel resolution from the raster using rasterio.

    Args:
        image_path: The path to the raster image.

    Returns:
        A tuple of (x resolution, y resolution).
    """
    with rasterio.open(image_path) as src:
        res = src.res
    return res


def stack_bands(
    input_files: List[str],
    output_file: str,
    resolution: Optional[float] = None,
    dtype: Optional[str] = None,  # e.g., "UInt16", "Float32"
    temp_vrt: str = "stack.vrt",
    overwrite: bool = False,
    compress: str = "DEFLATE",
    output_format: str = "COG",
    extra_gdal_translate_args: Optional[List[str]] = None,
) -> str:
    """
    Stack bands from multiple images into a single multi-band GeoTIFF.

    Args:
        input_files (List[str]): List of input image paths.
        output_file (str): Path to the output stacked image.
        resolution (float, optional): Output resolution. If None, inferred from first image.
        dtype (str, optional): Output data type (e.g., "UInt16", "Float32").
        temp_vrt (str): Temporary VRT filename.
        overwrite (bool): Whether to overwrite the output file.
        compress (str): Compression method.
        output_format (str): GDAL output format (default is "COG").
        extra_gdal_translate_args (List[str], optional): Extra arguments for gdal_translate.

    Returns:
        str: Path to the output file.
    """
    import leafmap

    if not input_files:
        raise ValueError("No input files provided.")
    elif isinstance(input_files, str):
        input_files = leafmap.find_files(input_files, ".tif")

    if os.path.exists(output_file) and not overwrite:
        print(f"Output file already exists: {output_file}")
        return output_file

    # Infer resolution if not provided
    if resolution is None:
        resolution_x, resolution_y = get_raster_resolution(input_files[0])
    else:
        resolution_x = resolution_y = resolution

    # Step 1: Build VRT
    vrt_cmd = ["gdalbuildvrt", "-separate", temp_vrt] + input_files
    subprocess.run(vrt_cmd, check=True)

    # Step 2: Translate VRT to output GeoTIFF
    translate_cmd = [
        "gdal_translate",
        "-tr",
        str(resolution_x),
        str(resolution_y),
        temp_vrt,
        output_file,
        "-of",
        output_format,
        "-co",
        f"COMPRESS={compress}",
    ]

    if dtype:
        translate_cmd.insert(1, "-ot")
        translate_cmd.insert(2, dtype)

    if extra_gdal_translate_args:
        translate_cmd += extra_gdal_translate_args

    subprocess.run(translate_cmd, check=True)

    # Step 3: Clean up VRT
    if os.path.exists(temp_vrt):
        os.remove(temp_vrt)

    return output_file
