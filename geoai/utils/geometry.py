"""Geometry processing and regularization utilities."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio import features
from shapely.geometry import Polygon
from tqdm import tqdm

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
            except Exception:
                print(f"Failed to parse geometry string: {building[:30]}...")
                continue

        # Ensure we have a valid geometry
        if not hasattr(building, "simplify"):
            print(f"Invalid geometry type: {type(building)}")
            continue

        # Step 1: Simplify to remove noise and small vertices
        simplified = building.simplify(
            simplify_tolerance, preserve_topology=preserve_topology
        )

        if orthogonalize:
            # Make sure we have a valid polygon with an exterior
            if not hasattr(simplified, "exterior") or simplified.exterior is None:
                print(f"Simplified geometry has no exterior: {simplified}")
                regularized_buildings.append(building)  # Use original instead
                continue

            # Step 2: Get the dominant angle to rotate building
            coords = np.array(simplified.exterior.coords)

            # Make sure we have enough coordinates for angle calculation
            if len(coords) < 3:
                print(f"Not enough coordinates for angle calculation: {len(coords)}")
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
                except Exception as e:
                    print(f"Warning: Failed to delete temporary files: {str(e)}")

        return da, df


def orthogonalize(
    input_path,
    output_path=None,
    epsilon=0.2,
    min_area=10,
    min_segments=4,
    area_tolerance=0.7,
    detect_triangles=True,
) -> Any:
    """
    Orthogonalizes object masks in a GeoTIFF file.

    This function reads a GeoTIFF containing object masks (binary or labeled regions),
    converts the raster masks to vector polygons, applies orthogonalization to each polygon,
    and optionally writes the result to a GeoJSON file.
    The source code is adapted from the Solar Panel Detection algorithm by Esri.
    See https://www.arcgis.com/home/item.html?id=c2508d72f2614104bfcfd5ccf1429284.
    Credits to Esri for the original code.

    Args:
        input_path (str): Path to the input GeoTIFF file.
        output_path (str, optional): Path to save the output GeoJSON file. If None, no file is saved.
        epsilon (float, optional): Simplification tolerance for the Douglas-Peucker algorithm.
            Higher values result in more simplification. Default is 0.2.
        min_area (float, optional): Minimum area of polygons to process (smaller ones are kept as-is).
        min_segments (int, optional): Minimum number of segments to keep after simplification.
            Default is 4 (for rectangular shapes).
        area_tolerance (float, optional): Allowed ratio of area change. Values less than 1.0 restrict
            area change. Default is 0.7 (allows reduction to 70% of original area).
        detect_triangles (bool, optional): If True, performs additional check to avoid creating triangular shapes.

    Returns:
        Any: A GeoDataFrame containing the orthogonalized features.
    """
    import cv2  # Lazy import to avoid QGIS opencv conflicts

    from functools import partial

    def orthogonalize_ring(ring, epsilon=0.2, min_segments=4):
        """
        Orthogonalizes a ring (list of coordinates).

        Args:
            ring (list): List of [x, y] coordinates forming a ring
            epsilon (float, optional): Simplification tolerance
            min_segments (int, optional): Minimum number of segments to keep

        Returns:
            list: Orthogonalized list of coordinates
        """
        if len(ring) <= 3:
            return ring

        # Convert to numpy array
        ring_arr = np.array(ring)

        # Get orientation
        angle = math.degrees(get_orientation(ring_arr))

        # Simplify using Ramer-Douglas-Peucker algorithm
        ring_arr = simplify(ring_arr, eps=epsilon)

        # If simplified too much, adjust epsilon to maintain minimum segments
        if len(ring_arr) < min_segments:
            # Try with smaller epsilon until we get at least min_segments points
            for adjust_factor in [0.75, 0.5, 0.25, 0.1]:
                test_arr = simplify(np.array(ring), eps=epsilon * adjust_factor)
                if len(test_arr) >= min_segments:
                    ring_arr = test_arr
                    break

        # Convert to dataframe for processing
        df = to_dataframe(ring_arr)

        # Add orientation information
        add_orientation(df, angle)

        # Align segments to orthogonal directions
        df = align(df)

        # Merge collinear line segments
        df = merge_lines(df)

        if len(df) == 0:
            return ring

        # If we have a triangle-like result (3 segments or less), return the original shape
        if len(df) <= 3:
            return ring

        # Join the orthogonalized segments back into a ring
        joined_ring = join_ring(df)

        # If the join operation didn't produce a valid ring, return the original
        if len(joined_ring) == 0 or len(joined_ring[0]) < 3:
            return ring

        # Enhanced validation: check for triangular result and geometric validity
        result_coords = joined_ring[0]

        # If result has 3 or fewer points (triangle), use original
        if len(result_coords) <= 3:  # 2 points + closing point (degenerate)
            return ring

        # Additional validation: check for degenerate geometry
        # Calculate area ratio to detect if the shape got severely distorted
        def calculate_polygon_area(coords):
            if len(coords) < 3:
                return 0
            area = 0
            n = len(coords)
            for i in range(n):
                j = (i + 1) % n
                area += coords[i][0] * coords[j][1]
                area -= coords[j][0] * coords[i][1]
            return abs(area) / 2

        original_area = calculate_polygon_area(ring)
        result_area = calculate_polygon_area(result_coords)

        # If the area changed dramatically (more than 30% shrinkage or 300% growth), use original
        if original_area > 0 and result_area > 0:
            area_ratio = result_area / original_area
            if area_ratio < 0.3 or area_ratio > 3.0:
                return ring

        # Check for triangular spikes and problematic artifacts
        very_acute_angle_count = 0
        triangular_spike_detected = False

        for i in range(len(result_coords) - 1):  # -1 to exclude closing point
            p1 = result_coords[i - 1]
            p2 = result_coords[i]
            p3 = result_coords[(i + 1) % (len(result_coords) - 1)]

            # Calculate angle at p2
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                # Count very acute angles (< 20 degrees) - these are likely spikes
                if angle < np.pi / 9:  # 20 degrees
                    very_acute_angle_count += 1
                    # If it's very acute with short sides, it's definitely a spike
                    if v1_norm < 5 or v2_norm < 5:
                        triangular_spike_detected = True

        # Check for excessively long edges that might be artifacts
        edge_lengths = []
        for i in range(len(result_coords) - 1):
            edge_len = np.sqrt(
                (result_coords[i + 1][0] - result_coords[i][0]) ** 2
                + (result_coords[i + 1][1] - result_coords[i][1]) ** 2
            )
            edge_lengths.append(edge_len)

        excessive_edge_detected = False
        if len(edge_lengths) > 0:
            avg_edge_length = np.mean(edge_lengths)
            max_edge_length = np.max(edge_lengths)
            # Only reject if edge is extremely disproportionate (8x average)
            if max_edge_length > avg_edge_length * 8:
                excessive_edge_detected = True

        # Check for triangular artifacts by detecting spikes that extend beyond bounds
        # Calculate original bounds
        orig_xs = [p[0] for p in ring]
        orig_ys = [p[1] for p in ring]
        orig_min_x, orig_max_x = min(orig_xs), max(orig_xs)
        orig_min_y, orig_max_y = min(orig_ys), max(orig_ys)
        orig_width = orig_max_x - orig_min_x
        orig_height = orig_max_y - orig_min_y

        # Calculate result bounds
        result_xs = [p[0] for p in result_coords]
        result_ys = [p[1] for p in result_coords]
        result_min_x, result_max_x = min(result_xs), max(result_xs)
        result_min_y, result_max_y = min(result_ys), max(result_ys)

        # Stricter bounds checking to catch triangular artifacts
        bounds_extension_detected = False
        # More conservative: only allow 10% extension
        tolerance_x = max(orig_width * 0.1, 1.0)  # 10% tolerance, at least 1 unit
        tolerance_y = max(orig_height * 0.1, 1.0)  # 10% tolerance, at least 1 unit

        if (
            result_min_x < orig_min_x - tolerance_x
            or result_max_x > orig_max_x + tolerance_x
            or result_min_y < orig_min_y - tolerance_y
            or result_max_y > orig_max_y + tolerance_y
        ):
            bounds_extension_detected = True

        # Reject if we detect triangular spikes, excessive edges, or bounds violations
        if (
            triangular_spike_detected
            or very_acute_angle_count > 2  # Multiple very acute angles
            or excessive_edge_detected
            or bounds_extension_detected
        ):  # Any significant bounds extension
            return ring

        # Convert back to a list and ensure it's closed
        result = joined_ring[0].tolist()
        if len(result) > 0 and (result[0] != result[-1]):
            result.append(result[0])

        return result

    def vectorize_mask(mask, transform):
        """
        Converts a binary mask to vector polygons.

        Args:
            mask (numpy.ndarray): Binary mask where non-zero values represent objects
            transform (rasterio.transform.Affine): Affine transformation matrix

        Returns:
            list: List of GeoJSON features
        """
        shapes = features.shapes(mask, transform=transform)
        features_list = []

        for shape, value in shapes:
            if value > 0:  # Only process non-zero values (actual objects)
                features_list.append(
                    {
                        "type": "Feature",
                        "properties": {"value": int(value)},
                        "geometry": shape,
                    }
                )

        return features_list

    def rasterize_features(features, shape, transform, dtype=np.uint8):
        """
        Converts vector features back to a raster mask.

        Args:
            features (list): List of GeoJSON features
            shape (tuple): Shape of the output raster (height, width)
            transform (rasterio.transform.Affine): Affine transformation matrix
            dtype (numpy.dtype, optional): Data type of the output raster

        Returns:
            numpy.ndarray: Rasterized mask
        """
        mask = features.rasterize(
            [
                (feature["geometry"], feature["properties"]["value"])
                for feature in features
            ],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=dtype,
        )

        return mask

    # The following helper functions are from the original code
    def get_orientation(contour):
        """
        Calculate the orientation angle of a contour.

        Args:
            contour (numpy.ndarray): Array of shape (n, 2) containing point coordinates

        Returns:
            float: Orientation angle in radians
        """
        box = cv2.minAreaRect(contour.astype(int))
        (cx, cy), (w, h), angle = box
        return math.radians(angle)

    def simplify(contour, eps=0.2):
        """
        Simplify a contour using the Ramer-Douglas-Peucker algorithm.

        Args:
            contour (numpy.ndarray): Array of shape (n, 2) containing point coordinates
            eps (float, optional): Epsilon value for simplification

        Returns:
            numpy.ndarray: Simplified contour
        """
        return rdp(contour, epsilon=eps)

    def to_dataframe(ring):
        """
        Convert a ring to a pandas DataFrame with line segment information.

        Args:
            ring (numpy.ndarray): Array of shape (n, 2) containing point coordinates

        Returns:
            pandas.DataFrame: DataFrame with line segment information
        """
        df = pd.DataFrame(ring, columns=["x1", "y1"])
        df["x2"] = df["x1"].shift(-1)
        df["y2"] = df["y1"].shift(-1)
        df.dropna(inplace=True)
        df["angle_atan"] = np.arctan2((df["y2"] - df["y1"]), (df["x2"] - df["x1"]))
        df["angle_atan_deg"] = df["angle_atan"] * 57.2958
        df["len"] = np.sqrt((df["y2"] - df["y1"]) ** 2 + (df["x2"] - df["x1"]) ** 2)
        df["cx"] = (df["x2"] + df["x1"]) / 2.0
        df["cy"] = (df["y2"] + df["y1"]) / 2.0
        return df

    def add_orientation(df, angle):
        """
        Add orientation information to the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with line segment information
            angle (float): Orientation angle in degrees

        Returns:
            None: Modifies the DataFrame in-place
        """
        rtangle = angle + 90
        is_parallel = (
            (df["angle_atan_deg"] > (angle - 45))
            & (df["angle_atan_deg"] < (angle + 45))
        ) | (
            (df["angle_atan_deg"] + 180 > (angle - 45))
            & (df["angle_atan_deg"] + 180 < (angle + 45))
        )
        df["angle"] = math.radians(angle)
        df["angle"] = df["angle"].where(is_parallel, math.radians(rtangle))

    def align(df):
        """
        Align line segments to their nearest orthogonal direction.

        Args:
            df (pandas.DataFrame): DataFrame with line segment information

        Returns:
            pandas.DataFrame: DataFrame with aligned line segments
        """
        # Handle edge case with empty dataframe
        if len(df) == 0:
            return df.copy()

        df_clone = df.copy()

        # Ensure angle column exists and has valid values
        if "angle" not in df_clone.columns or df_clone["angle"].isna().any():
            # If angle data is missing, add default angles based on atan2
            df_clone["angle"] = df_clone["angle_atan"]

        # Ensure length and center point data is valid
        if "len" not in df_clone.columns or df_clone["len"].isna().any():
            # Recalculate lengths if missing
            df_clone["len"] = np.sqrt(
                (df_clone["x2"] - df_clone["x1"]) ** 2
                + (df_clone["y2"] - df_clone["y1"]) ** 2
            )

        if "cx" not in df_clone.columns or df_clone["cx"].isna().any():
            df_clone["cx"] = (df_clone["x1"] + df_clone["x2"]) / 2.0

        if "cy" not in df_clone.columns or df_clone["cy"].isna().any():
            df_clone["cy"] = (df_clone["y1"] + df_clone["y2"]) / 2.0

        # Apply orthogonal alignment
        df_clone["x1"] = df_clone["cx"] - ((df_clone["len"] / 2) * np.cos(df["angle"]))
        df_clone["x2"] = df_clone["cx"] + ((df_clone["len"] / 2) * np.cos(df["angle"]))
        df_clone["y1"] = df_clone["cy"] - ((df_clone["len"] / 2) * np.sin(df["angle"]))
        df_clone["y2"] = df_clone["cy"] + ((df_clone["len"] / 2) * np.sin(df["angle"]))

        return df_clone

    def merge_lines(df_aligned):
        """
        Merge collinear line segments.

        Args:
            df_aligned (pandas.DataFrame): DataFrame with aligned line segments

        Returns:
            pandas.DataFrame: DataFrame with merged line segments
        """
        ortho_lines = []
        groups = df_aligned.groupby(
            (df_aligned["angle"].shift() != df_aligned["angle"]).cumsum()
        )
        for x, y in groups:
            group_cx = (y["cx"] * y["len"]).sum() / y["len"].sum()
            group_cy = (y["cy"] * y["len"]).sum() / y["len"].sum()
            cumlen = y["len"].sum()

            ortho_lines.append((group_cx, group_cy, cumlen, y["angle"].iloc[0]))

        ortho_list = []
        for cx, cy, length, rot_angle in ortho_lines:
            X1 = cx - (length / 2) * math.cos(rot_angle)
            X2 = cx + (length / 2) * math.cos(rot_angle)
            Y1 = cy - (length / 2) * math.sin(rot_angle)
            Y2 = cy + (length / 2) * math.sin(rot_angle)

            ortho_list.append(
                {
                    "x1": X1,
                    "y1": Y1,
                    "x2": X2,
                    "y2": Y2,
                    "len": length,
                    "cx": cx,
                    "cy": cy,
                    "angle": rot_angle,
                }
            )

        # Improved fix: Prevent merging that would create triangular or problematic shapes
        if (
            len(ortho_list) > 3 and ortho_list[0]["angle"] == ortho_list[-1]["angle"]
        ):  # join first and last segment if they're in same direction
            # Check if merging would result in 3 or 4 segments (potentially triangular)
            resulting_segments = len(ortho_list) - 1
            if resulting_segments <= 4:
                # For very small polygons, be extra cautious about merging
                # Calculate the spatial relationship between first and last segments
                first_center = np.array([ortho_list[0]["cx"], ortho_list[0]["cy"]])
                last_center = np.array([ortho_list[-1]["cx"], ortho_list[-1]["cy"]])
                center_distance = np.linalg.norm(first_center - last_center)

                # Get average segment length for comparison
                avg_length = sum(seg["len"] for seg in ortho_list) / len(ortho_list)

                # Only merge if segments are close enough and it won't create degenerate shapes
                if center_distance > avg_length * 1.5:
                    # Skip merging - segments are too far apart
                    pass
                else:
                    # Proceed with merging only for well-connected segments
                    totlen = ortho_list[0]["len"] + ortho_list[-1]["len"]
                    merge_cx = (
                        (ortho_list[0]["cx"] * ortho_list[0]["len"])
                        + (ortho_list[-1]["cx"] * ortho_list[-1]["len"])
                    ) / totlen

                    merge_cy = (
                        (ortho_list[0]["cy"] * ortho_list[0]["len"])
                        + (ortho_list[-1]["cy"] * ortho_list[-1]["len"])
                    ) / totlen

                    rot_angle = ortho_list[0]["angle"]
                    X1 = merge_cx - (totlen / 2) * math.cos(rot_angle)
                    X2 = merge_cx + (totlen / 2) * math.cos(rot_angle)
                    Y1 = merge_cy - (totlen / 2) * math.sin(rot_angle)
                    Y2 = merge_cy + (totlen / 2) * math.sin(rot_angle)

                    ortho_list[-1] = {
                        "x1": X1,
                        "y1": Y1,
                        "x2": X2,
                        "y2": Y2,
                        "len": totlen,
                        "cx": merge_cx,
                        "cy": merge_cy,
                        "angle": rot_angle,
                    }
                    ortho_list = ortho_list[1:]
            else:
                # For larger polygons, proceed with standard merging
                totlen = ortho_list[0]["len"] + ortho_list[-1]["len"]
                merge_cx = (
                    (ortho_list[0]["cx"] * ortho_list[0]["len"])
                    + (ortho_list[-1]["cx"] * ortho_list[-1]["len"])
                ) / totlen

                merge_cy = (
                    (ortho_list[0]["cy"] * ortho_list[0]["len"])
                    + (ortho_list[-1]["cy"] * ortho_list[-1]["len"])
                ) / totlen

                rot_angle = ortho_list[0]["angle"]
                X1 = merge_cx - (totlen / 2) * math.cos(rot_angle)
                X2 = merge_cx + (totlen / 2) * math.cos(rot_angle)
                Y1 = merge_cy - (totlen / 2) * math.sin(rot_angle)
                Y2 = merge_cy + (totlen / 2) * math.sin(rot_angle)

                ortho_list[-1] = {
                    "x1": X1,
                    "y1": Y1,
                    "x2": X2,
                    "y2": Y2,
                    "len": totlen,
                    "cx": merge_cx,
                    "cy": merge_cy,
                    "angle": rot_angle,
                }
                ortho_list = ortho_list[1:]
        ortho_df = pd.DataFrame(ortho_list)
        return ortho_df

    def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Find the intersection point of two line segments.

        Args:
            x1, y1, x2, y2: Coordinates of the first line segment
            x3, y3, x4, y4: Coordinates of the second line segment

        Returns:
            list: [x, y] coordinates of the intersection point

        Raises:
            ZeroDivisionError: If the lines are parallel or collinear
        """
        # Calculate the denominator of the intersection formula
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Check if lines are parallel or collinear (denominator close to zero)
        if abs(denominator) < 1e-10:
            raise ZeroDivisionError("Lines are parallel or collinear")

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator

        # Check if the intersection point is within a reasonable distance
        # from both line segments to avoid extreme extrapolation
        def point_on_segment(x, y, x1, y1, x2, y2, tolerance=2.0):
            # Check if point (x,y) is near the line segment from (x1,y1) to (x2,y2)
            # First check if it's near the infinite line
            line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_len < 1e-10:
                return np.sqrt((x - x1) ** 2 + (y - y1) ** 2) <= tolerance

            t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_len**2)

            # Check distance to the infinite line
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            dist_to_line = np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

            # Check if the projection is near the segment, not just the infinite line
            if t < -tolerance or t > 1 + tolerance:
                # If far from the segment, compute distance to the nearest endpoint
                dist_to_start = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                dist_to_end = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                return min(dist_to_start, dist_to_end) <= tolerance * 2

            return dist_to_line <= tolerance

        # Check if intersection is reasonably close to both line segments
        if not (
            point_on_segment(px, py, x1, y1, x2, y2)
            and point_on_segment(px, py, x3, y3, x4, y4)
        ):
            # If intersection is far from segments, it's probably extrapolating too much
            raise ValueError("Intersection point too far from line segments")

        return [px, py]

    def join_ring(merged_df):
        """
        Join line segments to form a closed ring.

        Args:
            merged_df (pandas.DataFrame): DataFrame with merged line segments

        Returns:
            numpy.ndarray: Array of shape (1, n, 2) containing the ring coordinates
        """
        # Handle edge cases
        if len(merged_df) < 3:
            # Not enough segments to form a valid polygon
            return np.array([[]])

        ring = []

        # Find intersections between adjacent line segments
        for i in range(len(merged_df) - 1):
            x1, y1, x2, y2, *_ = merged_df.iloc[i]
            x3, y3, x4, y4, *_ = merged_df.iloc[i + 1]

            try:
                intersection = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)

                # Check if the intersection point is too far from either line segment
                # This helps prevent extending edges beyond reasonable bounds
                dist_to_seg1 = min(
                    np.sqrt((intersection[0] - x1) ** 2 + (intersection[1] - y1) ** 2),
                    np.sqrt((intersection[0] - x2) ** 2 + (intersection[1] - y2) ** 2),
                )
                dist_to_seg2 = min(
                    np.sqrt((intersection[0] - x3) ** 2 + (intersection[1] - y3) ** 2),
                    np.sqrt((intersection[0] - x4) ** 2 + (intersection[1] - y4) ** 2),
                )

                # Use the maximum of line segment lengths as a reference
                max_len = max(
                    np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                    np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2),
                )

                # Improved intersection validation
                # Calculate angle between segments to detect sharp corners
                v1 = np.array([x2 - x1, y2 - y1])
                v2 = np.array([x4 - x3, y4 - y3])
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)

                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)

                    # Check for very sharp angles that could create triangular artifacts
                    is_sharp_angle = (
                        angle < np.pi / 6 or angle > 5 * np.pi / 6
                    )  # <30° or >150°
                else:
                    is_sharp_angle = False

                # Determine whether to use intersection or segment endpoint
                if (
                    dist_to_seg1 > max_len * 0.5
                    or dist_to_seg2 > max_len * 0.5
                    or is_sharp_angle
                ):
                    # Use a more conservative approach for problematic intersections
                    # Use the closer endpoint between segments
                    dist_x2_to_seg2 = min(
                        np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
                        np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
                    )
                    dist_x3_to_seg1 = min(
                        np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2),
                        np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2),
                    )

                    if dist_x2_to_seg2 <= dist_x3_to_seg1:
                        ring.append([x2, y2])
                    else:
                        ring.append([x3, y3])
                else:
                    ring.append(intersection)
            except Exception:
                # If intersection calculation fails, use the endpoint of the first segment
                ring.append([x2, y2])

        # Connect last segment with first segment
        x1, y1, x2, y2, *_ = merged_df.iloc[-1]
        x3, y3, x4, y4, *_ = merged_df.iloc[0]

        try:
            intersection = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)

            # Check if the intersection point is too far from either line segment
            dist_to_seg1 = min(
                np.sqrt((intersection[0] - x1) ** 2 + (intersection[1] - y1) ** 2),
                np.sqrt((intersection[0] - x2) ** 2 + (intersection[1] - y2) ** 2),
            )
            dist_to_seg2 = min(
                np.sqrt((intersection[0] - x3) ** 2 + (intersection[1] - y3) ** 2),
                np.sqrt((intersection[0] - x4) ** 2 + (intersection[1] - y4) ** 2),
            )

            max_len = max(
                np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2),
            )

            # Apply same sharp angle detection for closing segment
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x4 - x3, y4 - y3])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                is_sharp_angle = angle < np.pi / 6 or angle > 5 * np.pi / 6
            else:
                is_sharp_angle = False

            if (
                dist_to_seg1 > max_len * 0.5
                or dist_to_seg2 > max_len * 0.5
                or is_sharp_angle
            ):
                # Use conservative approach for closing segment
                dist_x2_to_seg2 = min(
                    np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
                    np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
                )
                dist_x3_to_seg1 = min(
                    np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2),
                    np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2),
                )

                if dist_x2_to_seg2 <= dist_x3_to_seg1:
                    ring.append([x2, y2])
                else:
                    ring.append([x3, y3])
            else:
                ring.append(intersection)
        except Exception:
            # If intersection calculation fails, use the endpoint of the last segment
            ring.append([x2, y2])

        # Ensure the ring is closed
        if len(ring) > 0 and (ring[0][0] != ring[-1][0] or ring[0][1] != ring[-1][1]):
            ring.append(ring[0])

        return np.array([ring])

    def rdp(M, epsilon=0, dist=None, algo="iter", return_mask=False):
        """
        Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            epsilon (float, optional): Epsilon value for simplification
            dist (callable, optional): Distance function
            algo (str, optional): Algorithm to use ('iter' or 'rec')
            return_mask (bool, optional): Whether to return a mask instead of the simplified array

        Returns:
            numpy.ndarray or list: Simplified points or mask
        """
        if dist is None:
            dist = pldist

        if algo == "iter":
            algo = partial(rdp_iter, return_mask=return_mask)
        elif algo == "rec":
            if return_mask:
                raise NotImplementedError(
                    'return_mask=True not supported with algo="rec"'
                )
            algo = rdp_rec

        if "numpy" in str(type(M)):
            return algo(M, epsilon, dist)

        return algo(np.array(M), epsilon, dist).tolist()

    def pldist(point, start, end):
        """
        Calculates the distance from 'point' to the line given by 'start' and 'end'.

        Args:
            point (numpy.ndarray): Point coordinates
            start (numpy.ndarray): Start point of the line
            end (numpy.ndarray): End point of the line

        Returns:
            float: Distance from point to line
        """
        if np.all(np.equal(start, end)):
            return np.linalg.norm(point - start)

        # Fix for NumPy 2.0 deprecation warning - handle 2D vectors properly
        # Instead of using cross product directly, calculate the area of the
        # parallelogram formed by the vectors and divide by the length of the line
        line_vec = end - start
        point_vec = point - start

        # Area of parallelogram = |a|*|b|*sin(θ)
        # For 2D vectors: |a×b| = |a|*|b|*sin(θ) = determinant([ax, ay], [bx, by])
        area = abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0])

        # Distance = Area / |line_vec|
        return area / np.linalg.norm(line_vec)

    def rdp_rec(M, epsilon, dist=pldist):
        """
        Recursive implementation of the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            epsilon (float): Epsilon value for simplification
            dist (callable, optional): Distance function

        Returns:
            numpy.ndarray: Simplified points
        """
        dmax = 0.0
        index = -1

        for i in range(1, M.shape[0]):
            d = dist(M[i], M[0], M[-1])

            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            r1 = rdp_rec(M[: index + 1], epsilon, dist)
            r2 = rdp_rec(M[index:], epsilon, dist)

            return np.vstack((r1[:-1], r2))
        else:
            return np.vstack((M[0], M[-1]))

    def _rdp_iter(M, start_index, last_index, epsilon, dist=pldist):
        """
        Internal iterative implementation of the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            start_index (int): Start index
            last_index (int): Last index
            epsilon (float): Epsilon value for simplification
            dist (callable, optional): Distance function

        Returns:
            numpy.ndarray: Boolean mask of points to keep
        """
        stk = []
        stk.append([start_index, last_index])
        global_start_index = start_index
        indices = np.ones(last_index - start_index + 1, dtype=bool)

        while stk:
            start_index, last_index = stk.pop()

            dmax = 0.0
            index = start_index

            for i in range(index + 1, last_index):
                if indices[i - global_start_index]:
                    d = dist(M[i], M[start_index], M[last_index])
                    if d > dmax:
                        index = i
                        dmax = d

            if dmax > epsilon:
                stk.append([start_index, index])
                stk.append([index, last_index])
            else:
                for i in range(start_index + 1, last_index):
                    indices[i - global_start_index] = False

        return indices

    def rdp_iter(M, epsilon, dist=pldist, return_mask=False):
        """
        Iterative implementation of the Ramer-Douglas-Peucker algorithm.

        Args:
            M (numpy.ndarray): Array of shape (n, d) containing point coordinates
            epsilon (float): Epsilon value for simplification
            dist (callable, optional): Distance function
            return_mask (bool, optional): Whether to return a mask instead of the simplified array

        Returns:
            numpy.ndarray: Simplified points or boolean mask
        """
        mask = _rdp_iter(M, 0, len(M) - 1, epsilon, dist)

        if return_mask:
            return mask

        return M[mask]

    # Read the raster data
    with rasterio.open(input_path) as src:
        # Read the first band (assuming it contains the mask)
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

        # Extract shapes from the raster mask
        shapes = list(features.shapes(mask, transform=transform))

        # Initialize progress bar
        print(f"Processing {len(shapes)} features...")

        # Convert shapes to GeoJSON features
        features_list = []
        for shape, value in tqdm(shapes, desc="Converting features", unit="shape"):
            if value > 0:  # Only process non-zero values (actual objects)
                # Convert GeoJSON geometry to Shapely polygon
                polygon = Polygon(shape["coordinates"][0])

                # Skip tiny polygons
                if polygon.area < min_area:
                    features_list.append(
                        {
                            "type": "Feature",
                            "properties": {"value": int(value)},
                            "geometry": shape,
                        }
                    )
                    continue

                # Check if shape is triangular and if we want to avoid triangular shapes
                if detect_triangles:
                    # Create a simplified version to check number of vertices
                    simple_polygon = polygon.simplify(epsilon)
                    if (
                        len(simple_polygon.exterior.coords) <= 4
                    ):  # 3 points + closing point
                        # Likely a triangular shape - skip orthogonalization
                        features_list.append(
                            {
                                "type": "Feature",
                                "properties": {"value": int(value)},
                                "geometry": shape,
                            }
                        )
                        continue

                # Process larger, non-triangular polygons
                try:
                    # Convert shapely polygon to a ring format for orthogonalization
                    exterior_ring = list(polygon.exterior.coords)
                    interior_rings = [
                        list(interior.coords) for interior in polygon.interiors
                    ]

                    # Calculate bounding box aspect ratio to help with parameter tuning
                    minx, miny, maxx, maxy = polygon.bounds
                    width = maxx - minx
                    height = maxy - miny
                    aspect_ratio = max(width, height) / max(1.0, min(width, height))

                    # Determine if this shape is likely to be a building/rectangular object
                    # Long thin objects might require different treatment
                    is_rectangular = aspect_ratio < 3.0

                    # Rectangular objects usually need more careful orthogonalization
                    epsilon_adjusted = epsilon
                    min_segments_adjusted = min_segments

                    if is_rectangular:
                        # For rectangular objects, use more conservative epsilon
                        epsilon_adjusted = epsilon * 0.75
                        # Ensure we get at least 4 points for a proper rectangle
                        min_segments_adjusted = max(4, min_segments)

                    # Orthogonalize the exterior and interior rings
                    orthogonalized_exterior = orthogonalize_ring(
                        exterior_ring,
                        epsilon=epsilon_adjusted,
                        min_segments=min_segments_adjusted,
                    )

                    orthogonalized_interiors = [
                        orthogonalize_ring(
                            ring,
                            epsilon=epsilon_adjusted,
                            min_segments=min_segments_adjusted,
                        )
                        for ring in interior_rings
                    ]

                    # Validate the result - calculate area change
                    original_area = polygon.area
                    orthogonalized_poly = Polygon(orthogonalized_exterior)

                    if orthogonalized_poly.is_valid:
                        area_ratio = (
                            orthogonalized_poly.area / original_area
                            if original_area > 0
                            else 0
                        )

                        # If area changed too much, revert to original
                        if area_ratio < area_tolerance or area_ratio > (
                            1.0 / area_tolerance
                        ):
                            # Use original polygon instead
                            geometry = shape
                        else:
                            # Create a new geometry with orthogonalized rings
                            geometry = {
                                "type": "Polygon",
                                "coordinates": [orthogonalized_exterior],
                            }

                            # Add interior rings if they exist
                            if orthogonalized_interiors:
                                geometry["coordinates"].extend(
                                    [ring for ring in orthogonalized_interiors]
                                )
                    else:
                        # If resulting polygon is invalid, use original
                        geometry = shape

                    # Add the feature to the list
                    features_list.append(
                        {
                            "type": "Feature",
                            "properties": {"value": int(value)},
                            "geometry": geometry,
                        }
                    )
                except Exception as e:
                    # Keep the original shape if orthogonalization fails
                    features_list.append(
                        {
                            "type": "Feature",
                            "properties": {"value": int(value)},
                            "geometry": shape,
                        }
                    )

        # Create the final GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": str(crs)}},
            "features": features_list,
        }

        # Convert to GeoDataFrame and set the CRS
        gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs=crs)

        # Save to file if output_path is provided
        if output_path:
            print(f"Saving to {output_path}...")
            gdf.to_file(output_path)
            print("Done!")

        return gdf


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
