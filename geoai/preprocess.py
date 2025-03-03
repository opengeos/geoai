import json
import math
import os
from PIL import Image
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.windows import Window
from rasterio import features
from shapely.geometry import box, shape
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import RandomRotation
from shapely.affinity import rotate
import torchgeo
import torch


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


# # Example usage
# if __name__ == "__main__":
#     # Single file conversion example
#     gdf = raster_to_vector(
#         raster_path="output/labels/tile_000001.tif",
#         output_path="output/labels/tile_000001.geojson",
#         threshold=0,
#         min_area=10,
#         simplify_tolerance=0.5,
#         class_values=[1],  # For a binary mask, use [1]
#         attribute_name='class',
#         plot_result=True
#     )

# Batch conversion example
# batch_raster_to_vector(
#     input_dir="path/to/labels",
#     output_dir="path/to/vectors",
#     pattern="*.tif",
#     threshold=0,
#     min_area=10,
#     class_values=[1, 2, 3],  # For a multiclass mask
#     merge_output=True
# )


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


# # Example usage
# if __name__ == "__main__":
#     # Single file conversion example
#     raster_data = vector_to_raster(
#         vector_path="buildings_train.geojson",
#         output_path="buildings_train.tif",
#         reference_raster="naip_train.tif",  # Optional, can use other parameters instead
#         # attribute_field="class",  # Optional, uses field values for pixel values
#         all_touched=True,  # Ensures small features are captured
#         plot_result=True
#     )

# Example with custom dimensions
# raster_data = vector_to_raster(
#     vector_path="path/to/buildings.geojson",
#     output_path="path/to/rasterized_buildings.tif",
#     pixel_size=0.5,  # 0.5 meter resolution
#     bounds=(454780, 5277567, 456282, 5278242),  # from original data
#     crs="EPSG:26911",
#     output_shape=(1350, 3000),  # custom dimensions
#     attribute_field="class"
# )

# Batch conversion example
# output_files = batch_vector_to_raster(
#     vector_path="path/to/buildings.geojson",
#     output_dir="path/to/output",
#     reference_rasters=["path/to/ref1.tif", "path/to/ref2.tif"],
#     attribute_field="class",
#     all_touched=True
# )


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


def create_overview_image(src, tile_coordinates, output_path, tile_size, stride):
    """Create an overview image showing all tiles and their status."""
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

    plt.title("Tile Overview (Green = Contains Features, Red = Empty)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Overview image saved to {output_path}")


# # Example usage
# if __name__ == "__main__":
#     # Try to install tqdm if not available
#     try:
#         import tqdm
#     except ImportError:
#         print("Installing tqdm progress bar library...")
#         import sys
#         import subprocess

#         subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
#         import tqdm

#     # Example with vector class data
#     export_geotiff_tiles(
#         in_raster="naip_train.tif",
#         out_folder="geotiff_output_vector",
#         in_class_data="buildings_train.geojson",
#         tile_size=256,
#         stride=128,
#         class_value_field="class",
#         buffer_radius=2,
#         create_overview=True,
#     )

#     # Example with raster class data
#     export_geotiff_tiles(
#         in_raster="naip_train.tif",
#         out_folder="geotiff_output_raster",
#         in_class_data="buildings_train.tif",  # This would be a raster mask
#         tile_size=256,
#         stride=128,
#         create_overview=True,
#         skip_empty_tiles=True,
#     )


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


# if __name__ == "__main__":
#     # Example parameters
#     export_training_data(
#         in_raster="naip_train.tif",
#         out_folder="output",
#         in_class_data="buildings_train.geojson",
#         image_chip_format="GEOTIFF",  # Use GeoTIFF format to preserve georeference
#         tile_size_x=256,
#         tile_size_y=256,
#         stride_x=128,  # Use overlapping tiles to increase chance of capturing features
#         stride_y=128,
#         metadata_format="PASCAL_VOC",
#         class_value_field="class",
#         buffer_radius=2,  # Add small buffer to buildings to ensure they're captured
#         all_touched=True,  # Ensure small features are rasterized
#         save_geotiff=True,  # Always save as GeoTIFF regardless of image_chip_format
#     )
