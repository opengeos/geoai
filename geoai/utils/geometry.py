"""Geometry processing and regularization utilities."""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio import features
from shapely.geometry import Polygon

from ..progress import tqdm
from ._orthogonalize import orthogonalize
from .device import install_package, temp_file_path

__all__ = [
    "regularization",
    "hybrid_regularization",
    "adaptive_regularization",
    "orthogonalize",
    "region_groups",
    "regularize",
]


def regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
    angle_tolerance: float = 10,
    simplify_tolerance: float = 0.5,
    orthogonalize: bool = True,
    preserve_topology: bool = True,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    Regularizes building footprint polygons with multiple techniques beyond minimum
    rotated rectangles.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons containing building footprints
        angle_tolerance: Degrees within which angles will be regularized to 90/180 degrees
        simplify_tolerance: Distance tolerance for Douglas-Peucker simplification
        orthogonalize: Whether to enforce orthogonal angles in the final polygons
        preserve_topology: Whether to preserve topology during simplification

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely import wkt
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, shape

    regularized_buildings = []

    # Check if we're dealing with a GeoDataFrame
    if isinstance(building_polygons, gpd.GeoDataFrame):
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    for building in geom_objects:
        # Handle potential string representations of geometries
        if isinstance(building, str):
            try:
                # Try to parse as WKT
                building = wkt.loads(building)
            except (ValueError, TypeError):
                logger.warning("Failed to parse geometry string: %s...", building[:30])
                continue

        # Ensure we have a valid geometry
        if not hasattr(building, "simplify"):
            logger.warning("Invalid geometry type: %s", type(building))
            continue

        # Step 1: Simplify to remove noise and small vertices
        simplified = building.simplify(
            simplify_tolerance, preserve_topology=preserve_topology
        )

        if orthogonalize:
            # Make sure we have a valid polygon with an exterior
            if not hasattr(simplified, "exterior") or simplified.exterior is None:
                logger.warning("Simplified geometry has no exterior: %s", simplified)
                regularized_buildings.append(building)  # Use original instead
                continue

            # Step 2: Get the dominant angle to rotate building
            coords = np.array(simplified.exterior.coords)

            # Make sure we have enough coordinates for angle calculation
            if len(coords) < 3:
                logger.warning(
                    "Not enough coordinates for angle calculation: %d", len(coords)
                )
                regularized_buildings.append(building)  # Use original instead
                continue

            segments = np.diff(coords, axis=0)
            angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

            # Find most common angle classes (0, 90, 180, 270 degrees)
            binned_angles = np.round(angles / 90) * 90
            dominant_angle = np.bincount(binned_angles.astype(int) % 180).argmax()

            # Step 3: Rotate to align with axes, regularize, then rotate back
            rotated = rotate(simplified, -dominant_angle, origin="centroid")

            # Step 4: Rectify coordinates to enforce right angles
            ext_coords = np.array(rotated.exterior.coords)
            rect_coords = []

            # Regularize each vertex to create orthogonal corners
            for i in range(len(ext_coords) - 1):
                rect_coords.append(ext_coords[i])

                # Check if we need to add a right-angle vertex
                angle = (
                    np.arctan2(
                        ext_coords[(i + 1) % (len(ext_coords) - 1), 1]
                        - ext_coords[i, 1],
                        ext_coords[(i + 1) % (len(ext_coords) - 1), 0]
                        - ext_coords[i, 0],
                    )
                    * 180
                    / np.pi
                )

                if abs(angle % 90) > angle_tolerance and abs(angle % 90) < (
                    90 - angle_tolerance
                ):
                    # Add intermediate point to create right angle
                    rect_coords.append(
                        [
                            ext_coords[(i + 1) % (len(ext_coords) - 1), 0],
                            ext_coords[i, 1],
                        ]
                    )

            # Close the polygon by adding the first point again
            rect_coords.append(rect_coords[0])

            # Create regularized polygon and rotate back
            regularized = Polygon(rect_coords)
            final_building = rotate(regularized, dominant_angle, origin="centroid")
        else:
            final_building = simplified

        regularized_buildings.append(final_building)

    # If input was a GeoDataFrame, return a GeoDataFrame
    if isinstance(building_polygons, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(
            geometry=regularized_buildings, crs=building_polygons.crs
        )
    else:
        return regularized_buildings


def hybrid_regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    A comprehensive hybrid approach to building footprint regularization.

    Applies different strategies based on building characteristics.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons containing building footprints

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    # Use minimum_rotated_rectangle instead of oriented_envelope
    try:
        from shapely.minimum_rotated_rectangle import minimum_rotated_rectangle
    except ImportError:
        # For older Shapely versions
        def minimum_rotated_rectangle(geom):
            """Calculate the minimum rotated rectangle for a geometry"""
            # For older Shapely versions, implement a simple version
            return geom.minimum_rotated_rectangle

    # Determine input type for correct return
    is_gdf = isinstance(building_polygons, gpd.GeoDataFrame)

    # Extract geometries if GeoDataFrame
    if is_gdf:
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    results = []

    for building in geom_objects:
        # 1. Analyze building characteristics
        if not hasattr(building, "exterior") or building.is_empty:
            results.append(building)
            continue

        # Calculate shape complexity metrics
        complexity = building.length / (4 * np.sqrt(building.area))

        # Calculate dominant angle
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        segment_angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

        # Weight angles by segment length
        hist, bins = np.histogram(
            segment_angles % 180, bins=36, range=(0, 180), weights=segment_lengths
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        dominant_angle = bin_centers[np.argmax(hist)]

        # Check if building is close to orthogonal
        is_orthogonal = min(dominant_angle % 45, 45 - (dominant_angle % 45)) < 5

        # 2. Apply appropriate regularization strategy
        if complexity > 1.5:
            # Complex buildings: use minimum rotated rectangle
            result = minimum_rotated_rectangle(building)
        elif is_orthogonal:
            # Near-orthogonal buildings: orthogonalize in place
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Create orthogonal hull in rotated space
            bounds = rotated.bounds
            ortho_hull = Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            )

            result = rotate(ortho_hull, dominant_angle, origin="centroid")
        else:
            # Diagonal buildings: use custom approach for diagonal buildings
            # Rotate to align with axes
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Simplify in rotated space
            simplified = rotated.simplify(0.3, preserve_topology=True)

            # Get the bounds in rotated space
            bounds = simplified.bounds
            min_x, min_y, max_x, max_y = bounds

            # Create a rectangular hull in rotated space
            rect_poly = Polygon(
                [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            )

            # Rotate back to original orientation
            result = rotate(rect_poly, dominant_angle, origin="centroid")

        results.append(result)

    # Return in same format as input
    if is_gdf:
        return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)
    else:
        return results


def adaptive_regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
    simplify_tolerance: float = 0.5,
    area_threshold: float = 0.9,
    preserve_shape: bool = True,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    Adaptively regularizes building footprints based on their characteristics.

    This approach determines the best regularization method for each building.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons
        simplify_tolerance: Distance tolerance for simplification
        area_threshold: Minimum acceptable area ratio
        preserve_shape: Whether to preserve overall shape for complex buildings

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    # Analyze the overall dataset to set appropriate parameters
    if is_gdf := isinstance(building_polygons, gpd.GeoDataFrame):
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    results = []

    for building in geom_objects:
        # Skip invalid geometries
        if not hasattr(building, "exterior") or building.is_empty:
            results.append(building)
            continue

        # Measure building complexity
        complexity = building.length / (4 * np.sqrt(building.area))

        # Determine if the building has a clear principal direction
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

        # Normalize angles to 0-180 range and get histogram
        norm_angles = angles % 180
        hist, bins = np.histogram(
            norm_angles, bins=18, range=(0, 180), weights=segment_lengths
        )

        # Calculate direction clarity (ratio of longest direction to total)
        direction_clarity = np.max(hist) / np.sum(hist) if np.sum(hist) > 0 else 0

        # Choose regularization method based on building characteristics
        if complexity < 1.2 and direction_clarity > 0.5:
            # Simple building with clear direction: use rotated rectangle
            bin_max = np.argmax(hist)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            dominant_angle = bin_centers[bin_max]

            # Rotate to align with coordinate system
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Create bounding box in rotated space
            bounds = rotated.bounds
            rect = Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            )

            # Rotate back
            result = rotate(rect, dominant_angle, origin="centroid")

            # Quality check
            if (
                result.area / building.area < area_threshold
                or result.area / building.area > (1.0 / area_threshold)
            ):
                # Too much area change, use simplified original
                result = building.simplify(simplify_tolerance, preserve_topology=True)

        else:
            # Complex building or no clear direction: preserve shape
            if preserve_shape:
                # Simplify with topology preservation
                result = building.simplify(simplify_tolerance, preserve_topology=True)
            else:
                # Fall back to convex hull for very complex shapes
                result = building.convex_hull

        results.append(result)

    # Return in same format as input
    if is_gdf:
        return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)
    else:
        return results


def region_groups(
    image: Union[str, "xr.DataArray", np.ndarray],
    connectivity: int = 1,
    min_size: int = 10,
    max_size: Optional[int] = None,
    threshold: Optional[int] = None,
    properties: Optional[List[str]] = None,
    intensity_image: Optional[Union[str, "xr.DataArray", np.ndarray]] = None,
    out_csv: Optional[str] = None,
    out_vector: Optional[str] = None,
    out_image: Optional[str] = None,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, "pd.DataFrame"], Tuple["xr.DataArray", "pd.DataFrame"]]:
    """
    Segment regions in an image and filter them based on size.

    Args:
        image (Union[str, xr.DataArray, np.ndarray]): Input image, can be a file
            path, xarray DataArray, or numpy array.
        connectivity (int, optional): Connectivity for labeling. Defaults to 1
            for 4-connectivity. Use 2 for 8-connectivity.
        min_size (int, optional): Minimum size of regions to keep. Defaults to 10.
        max_size (Optional[int], optional): Maximum size of regions to keep.
            Defaults to None.
        threshold (Optional[int], optional): Threshold for filling holes.
            Defaults to None, which is equal to min_size.
        properties (Optional[List[str]], optional): List of properties to measure.
            See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
            Defaults to None.
        intensity_image (Optional[Union[str, xr.DataArray, np.ndarray]], optional):
            Intensity image to measure properties. Defaults to None.
        out_csv (Optional[str], optional): Path to save the properties as a CSV file.
            Defaults to None.
        out_vector (Optional[str], optional): Path to save the vector file.
            Defaults to None.
        out_image (Optional[str], optional): Path to save the output image.
            Defaults to None.

    Returns:
        Union[Tuple[np.ndarray, pd.DataFrame], Tuple[xr.DataArray, pd.DataFrame]]: Labeled image and properties DataFrame.
    """
    import scipy.ndimage as ndi
    from skimage import measure

    # Import raster_to_vector lazily to avoid circular imports
    from geoai.utils import raster_to_vector

    if isinstance(image, str):
        ds = rxr.open_rasterio(image)
        da = ds.sel(band=1)
        array = da.values.squeeze()
    elif isinstance(image, xr.DataArray):
        da = image
        array = image.values.squeeze()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise ValueError(
            "The input image must be a file path, xarray DataArray, or numpy array."
        )

    if threshold is None:
        threshold = min_size

    # Define a custom function to calculate median intensity
    def intensity_median(region, intensity_image):
        # Extract the intensity values for the region
        return np.median(intensity_image[region])

    # Add your custom function to the list of extra properties
    if intensity_image is not None:
        extra_props = (intensity_median,)
    else:
        extra_props = None

    if properties is None:
        properties = [
            "label",
            "area",
            "area_bbox",
            "area_convex",
            "area_filled",
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "diameter_areagth",
            "extent",
            "orientation",
            "perimeter",
            "solidity",
        ]

        if intensity_image is not None:

            properties += [
                "intensity_max",
                "intensity_mean",
                "intensity_min",
                "intensity_std",
            ]

    if intensity_image is not None:
        if isinstance(intensity_image, str):
            ds = rxr.open_rasterio(intensity_image)
            intensity_da = ds.sel(band=1)
            intensity_image = intensity_da.values.squeeze()
        elif isinstance(intensity_image, xr.DataArray):
            intensity_image = intensity_image.values.squeeze()
        elif isinstance(intensity_image, np.ndarray):
            pass
        else:
            raise ValueError(
                "The intensity_image must be a file path, xarray DataArray, or numpy array."
            )

    label_image = measure.label(array, connectivity=connectivity)
    props = measure.regionprops_table(
        label_image, properties=properties, intensity_image=intensity_image, **kwargs
    )

    df = pd.DataFrame(props)

    # Get the labels of regions with area smaller than the threshold
    small_regions = df[df["area"] < min_size]["label"].values
    # Set the corresponding labels in the label_image to zero
    for region_label in small_regions:
        label_image[label_image == region_label] = 0

    if max_size is not None:
        large_regions = df[df["area"] > max_size]["label"].values
        for region_label in large_regions:
            label_image[label_image == region_label] = 0

    # Find the background (holes) which are zeros
    holes = label_image == 0

    # Label the holes (connected components in the background)
    labeled_holes, _ = ndi.label(holes)

    # Measure properties of the labeled holes, including area and bounding box
    hole_props = measure.regionprops(labeled_holes)

    # Loop through each hole and fill it if it is smaller than the threshold
    for prop in hole_props:
        if prop.area < threshold:
            # Get the coordinates of the small hole
            coords = prop.coords

            # Find the surrounding region's ID (non-zero value near the hole)
            surrounding_region_values = []
            for coord in coords:
                x, y = coord
                # Get a 3x3 neighborhood around the hole pixel
                neighbors = label_image[max(0, x - 1) : x + 2, max(0, y - 1) : y + 2]
                # Exclude the hole pixels (zeros) and get region values
                region_values = neighbors[neighbors != 0]
                if region_values.size > 0:
                    surrounding_region_values.append(
                        region_values[0]
                    )  # Take the first non-zero value

            if surrounding_region_values:
                # Fill the hole with the mode (most frequent) of the surrounding region values
                fill_value = max(
                    set(surrounding_region_values), key=surrounding_region_values.count
                )
                label_image[coords[:, 0], coords[:, 1]] = fill_value

    label_image, num_labels = measure.label(
        label_image, connectivity=connectivity, return_num=True
    )
    props = measure.regionprops_table(
        label_image,
        properties=properties,
        intensity_image=intensity_image,
        extra_properties=extra_props,
        **kwargs,
    )

    df = pd.DataFrame(props)
    df["elongation"] = df["axis_major_length"] / df["axis_minor_length"]

    dtype = "uint8"
    if num_labels > 255 and num_labels <= 65535:
        dtype = "uint16"
    elif num_labels > 65535:
        dtype = "uint32"

    if out_csv is not None:
        df.to_csv(out_csv, index=False)

    if isinstance(image, np.ndarray):
        return label_image, df
    else:
        da.values = label_image
        if out_image is not None:
            da.rio.to_raster(out_image, dtype=dtype)

        if out_vector is not None:
            tmp_raster = None
            tmp_vector = None
            try:
                if out_image is None:
                    tmp_raster = temp_file_path(".tif")
                    da.rio.to_raster(tmp_raster, dtype=dtype)
                    tmp_vector = temp_file_path(".gpkg")
                    raster_to_vector(
                        tmp_raster,
                        tmp_vector,
                        attribute_name="value",
                        unique_attribute_value=True,
                    )
                else:
                    tmp_vector = temp_file_path(".gpkg")
                    raster_to_vector(
                        out_image,
                        tmp_vector,
                        attribute_name="value",
                        unique_attribute_value=True,
                    )
                gdf = gpd.read_file(tmp_vector)
                gdf["label"] = gdf["value"].astype(int)
                gdf.drop(columns=["value"], inplace=True)
                gdf2 = pd.merge(gdf, df, on="label", how="left")
                gdf2.to_file(out_vector)
                gdf2.sort_values("label", inplace=True)
                df = gdf2
            finally:
                try:
                    if tmp_raster is not None and os.path.exists(tmp_raster):
                        os.remove(tmp_raster)
                    if tmp_vector is not None and os.path.exists(tmp_vector):
                        os.remove(tmp_vector)
                except OSError as e:
                    logger.warning("Failed to delete temporary files: %s", str(e))

        return da, df


def regularize(
    data: Union[gpd.GeoDataFrame, str],
    parallel_threshold: float = 1.0,
    target_crs: Optional[Union[str, "pyproj.CRS"]] = None,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 1,
    include_metadata: bool = False,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Regularizes polygon geometries in a GeoDataFrame by aligning edges.

    Aligns edges to be parallel or perpendicular (optionally also 45 degrees)
    to their main direction. Handles reprojection, initial simplification,
    regularization, geometry cleanup, and parallel processing.

    This function is a wrapper around the `regularize_geodataframe` function
    from the `buildingregulariser` package. Credits to the original author
    Nick Wright. Check out the repo at https://github.com/DPIRD-DMA/Building-Regulariser.

    Args:
        data (Union[gpd.GeoDataFrame, str]): Input GeoDataFrame with polygon or multipolygon geometries,
            or a file path to the GeoDataFrame.
        parallel_threshold (float, optional): Distance threshold for merging nearly parallel adjacent edges
            during regularization. Defaults to 1.0.
        target_crs (Optional[Union[str, "pyproj.CRS"]], optional): Target Coordinate Reference System for
            processing. If None, uses the input GeoDataFrame's CRS. Processing is more reliable in a
            projected CRS. Defaults to None.
        simplify (bool, optional): If True, applies initial simplification to the geometry before
            regularization. Defaults to True.
        simplify_tolerance (float, optional): Tolerance for the initial simplification step (if `simplify`
            is True). Also used for geometry cleanup steps. Defaults to 0.5.
        allow_45_degree (bool, optional): If True, allows edges to be oriented at 45-degree angles relative
            to the main direction during regularization. Defaults to True.
        diagonal_threshold_reduction (float, optional): Reduction factor in degrees to reduce the likelihood
            of diagonal edges being created. Larger values reduce the likelihood of diagonal edges.
            Defaults to 15.
        allow_circles (bool, optional): If True, attempts to detect polygons that are nearly circular and
            replaces them with perfect circles. Defaults to True.
        circle_threshold (float, optional): Intersection over Union (IoU) threshold used for circle detection
            (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
        num_cores (int, optional): Number of CPU cores to use for parallel processing. If 1, processing is
            done sequentially. Defaults to 1.
        include_metadata (bool, optional): If True, includes metadata about the regularization process in the
            output GeoDataFrame. Defaults to False.
        output_path (Optional[str], optional): Path to save the output GeoDataFrame. If None, the output is
            not saved. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the `to_file` method when saving the output.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with regularized polygon geometries. Original attributes are
        preserved. Geometries that failed processing might be dropped.

    Raises:
        ValueError: If the input data is not a GeoDataFrame or a file path, or if the input GeoDataFrame is empty.
    """
    try:
        from buildingregulariser import regularize_geodataframe
    except ImportError:
        install_package("buildingregulariser")
        from buildingregulariser import regularize_geodataframe

    if isinstance(data, str):
        data = gpd.read_file(data)
    elif not isinstance(data, gpd.GeoDataFrame):
        raise ValueError("Input data must be a GeoDataFrame or a file path.")

    # Check if the input data is empty
    if data.empty:
        raise ValueError("Input GeoDataFrame is empty.")

    gdf = regularize_geodataframe(
        data,
        parallel_threshold=parallel_threshold,
        target_crs=target_crs,
        simplify=simplify,
        simplify_tolerance=simplify_tolerance,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        num_cores=num_cores,
        include_metadata=include_metadata,
    )

    if output_path:
        gdf.to_file(output_path, **kwargs)

    return gdf
