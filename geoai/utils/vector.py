"""Vector I/O and processing utilities."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import MultiPolygon, Polygon, box, mapping

from .conversion import coords_to_xy
from .device import install_package
from .download import download_file

__all__ = [
    "get_vector_info",
    "print_vector_info",
    "get_vector_info_ogr",
    "analyze_vector_attributes",
    "visualize_vector_by_attribute",
    "export_tiles_to_geojson",
    "add_geometric_properties",
    "vector_to_geojson",
    "geojson_to_coords",
    "boxes_to_vector",
    "geojson_to_xy",
    "smooth_vector",
]


def get_vector_info(vector_path: str) -> Optional[Dict[str, Any]]:
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


def print_vector_info(
    vector_path: str, show_preview: bool = True, figsize: Tuple[int, int] = (10, 8)
) -> Optional[Dict[str, Any]]:
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
def get_vector_info_ogr(vector_path: str) -> Optional[Dict[str, Any]]:
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


def analyze_vector_attributes(
    vector_path: str, attribute_name: str
) -> Optional[Dict[str, Any]]:
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
    vector_path: str,
    attribute_name: str,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
) -> bool:
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


def export_tiles_to_geojson(
    tile_coordinates, src, output_path, tile_size=None, stride=None
) -> str:
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


def add_geometric_properties(
    data: gpd.GeoDataFrame,
    properties: Optional[List[str]] = None,
    area_unit: str = "m2",
    length_unit: str = "m",
) -> gpd.GeoDataFrame:
    """Calculates geometric properties and adds them to the GeoDataFrame.

    This function calculates various geometric properties of features in a
    GeoDataFrame and adds them as new columns without modifying existing attributes.

    Args:
        data: GeoDataFrame containing vector features.
        properties: List of geometric properties to calculate. Options include:
            'area', 'length', 'perimeter', 'centroid_x', 'centroid_y', 'bounds',
            'convex_hull_area', 'orientation', 'complexity', 'area_bbox',
            'area_convex', 'area_filled', 'major_length', 'minor_length',
            'eccentricity', 'diameter_area', 'extent', 'solidity',
            'elongation'.
            Defaults to ['area', 'length'] if None.
        area_unit: String specifying the unit for area calculation ('m2', 'km2',
            'ha'). Defaults to 'm2'.
        length_unit: String specifying the unit for length calculation ('m', 'km').
            Defaults to 'm'.

    Returns:
        geopandas.GeoDataFrame: A copy of the input GeoDataFrame with added
        geometric property columns.
    """
    from shapely.ops import unary_union

    if isinstance(data, str):
        from .raster import read_vector

        data = read_vector(data)

    # Make a copy to avoid modifying the original
    result = data.copy()

    # Default properties to calculate
    if properties is None:
        properties = [
            "area",
            "length",
            "perimeter",
            "convex_hull_area",
            "orientation",
            "complexity",
            "area_bbox",
            "area_convex",
            "area_filled",
            "major_length",
            "minor_length",
            "eccentricity",
            "diameter_area",
            "extent",
            "solidity",
            "elongation",
        ]

    # Make sure we're working with a GeoDataFrame with a valid CRS

    if not isinstance(result, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")

    if result.crs is None:
        raise ValueError(
            "GeoDataFrame must have a defined coordinate reference system (CRS)"
        )

    # Ensure we're working with a projected CRS for accurate measurements
    if result.crs.is_geographic:
        # Reproject to a suitable projected CRS for accurate measurements
        result = result.to_crs(result.estimate_utm_crs())

    # Basic area calculation with unit conversion
    if "area" in properties:
        # Calculate area (only for polygons)
        result["area"] = result.geometry.apply(
            lambda geom: geom.area if isinstance(geom, (Polygon, MultiPolygon)) else 0
        )

        # Convert to requested units
        if area_unit == "km2":
            result["area"] = result["area"] / 1_000_000  # m² to km²
            result.rename(columns={"area": "area_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area"] = result["area"] / 10_000  # m² to hectares
            result.rename(columns={"area": "area_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area": "area_m2"}, inplace=True)

    # Length calculation with unit conversion
    if "length" in properties:
        # Calculate length (works for lines and polygon boundaries)
        result["length"] = result.geometry.length

        # Convert to requested units
        if length_unit == "km":
            result["length"] = result["length"] / 1_000  # m to km
            result.rename(columns={"length": "length_km"}, inplace=True)
        else:  # Default is m
            result.rename(columns={"length": "length_m"}, inplace=True)

    # Perimeter calculation (for polygons)
    if "perimeter" in properties:
        result["perimeter"] = result.geometry.apply(
            lambda geom: (
                geom.boundary.length if isinstance(geom, (Polygon, MultiPolygon)) else 0
            )
        )

        # Convert to requested units
        if length_unit == "km":
            result["perimeter"] = result["perimeter"] / 1_000  # m to km
            result.rename(columns={"perimeter": "perimeter_km"}, inplace=True)
        else:  # Default is m
            result.rename(columns={"perimeter": "perimeter_m"}, inplace=True)

    # Centroid coordinates
    if "centroid_x" in properties or "centroid_y" in properties:
        centroids = result.geometry.centroid

        if "centroid_x" in properties:
            result["centroid_x"] = centroids.x

        if "centroid_y" in properties:
            result["centroid_y"] = centroids.y

    # Bounding box properties
    if "bounds" in properties:
        bounds = result.geometry.bounds
        result["minx"] = bounds.minx
        result["miny"] = bounds.miny
        result["maxx"] = bounds.maxx
        result["maxy"] = bounds.maxy

    # Area of bounding box
    if "area_bbox" in properties:
        bounds = result.geometry.bounds
        result["area_bbox"] = (bounds.maxx - bounds.minx) * (bounds.maxy - bounds.miny)

        # Convert to requested units
        if area_unit == "km2":
            result["area_bbox"] = result["area_bbox"] / 1_000_000
            result.rename(columns={"area_bbox": "area_bbox_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area_bbox"] = result["area_bbox"] / 10_000
            result.rename(columns={"area_bbox": "area_bbox_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area_bbox": "area_bbox_m2"}, inplace=True)

    # Area of convex hull
    if "area_convex" in properties or "convex_hull_area" in properties:
        result["area_convex"] = result.geometry.convex_hull.area

        # Convert to requested units
        if area_unit == "km2":
            result["area_convex"] = result["area_convex"] / 1_000_000
            result.rename(columns={"area_convex": "area_convex_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area_convex"] = result["area_convex"] / 10_000
            result.rename(columns={"area_convex": "area_convex_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area_convex": "area_convex_m2"}, inplace=True)

        # For backward compatibility
        if "convex_hull_area" in properties and "area_convex" not in properties:
            result["convex_hull_area"] = result["area_convex"]
            if area_unit == "km2":
                result.rename(
                    columns={"convex_hull_area": "convex_hull_area_km2"}, inplace=True
                )
            elif area_unit == "ha":
                result.rename(
                    columns={"convex_hull_area": "convex_hull_area_ha"}, inplace=True
                )
            else:
                result.rename(
                    columns={"convex_hull_area": "convex_hull_area_m2"}, inplace=True
                )

    # Area of filled geometry (no holes)
    if "area_filled" in properties:

        def get_filled_area(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return 0

            if isinstance(geom, MultiPolygon):
                # For MultiPolygon, fill all constituent polygons
                filled_polys = [Polygon(p.exterior) for p in geom.geoms]
                return unary_union(filled_polys).area
            else:
                # For single Polygon, create a new one with just the exterior ring
                return Polygon(geom.exterior).area

        result["area_filled"] = result.geometry.apply(get_filled_area)

        # Convert to requested units
        if area_unit == "km2":
            result["area_filled"] = result["area_filled"] / 1_000_000
            result.rename(columns={"area_filled": "area_filled_km2"}, inplace=True)
        elif area_unit == "ha":
            result["area_filled"] = result["area_filled"] / 10_000
            result.rename(columns={"area_filled": "area_filled_ha"}, inplace=True)
        else:  # Default is m²
            result.rename(columns={"area_filled": "area_filled_m2"}, inplace=True)

    # Axes lengths, eccentricity, orientation, and elongation
    if any(
        p in properties
        for p in [
            "major_length",
            "minor_length",
            "eccentricity",
            "orientation",
            "elongation",
        ]
    ):

        def get_axes_properties(geom):
            # Skip non-polygons
            if not isinstance(geom, (Polygon, MultiPolygon)):
                return None, None, None, None, None

            # Handle multipolygons by using the largest polygon
            if isinstance(geom, MultiPolygon):
                # Get the polygon with the largest area
                geom = sorted(list(geom.geoms), key=lambda p: p.area, reverse=True)[0]

            try:
                # Get the minimum rotated rectangle
                rect = geom.minimum_rotated_rectangle

                # Extract coordinates
                coords = list(rect.exterior.coords)[
                    :-1
                ]  # Remove the duplicated last point

                if len(coords) < 4:
                    return None, None, None, None, None

                # Calculate lengths of all four sides
                sides = []
                for i in range(len(coords)):
                    p1 = coords[i]
                    p2 = coords[(i + 1) % len(coords)]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = np.sqrt(dx**2 + dy**2)
                    angle = np.degrees(np.arctan2(dy, dx)) % 180
                    sides.append((length, angle, p1, p2))

                # Group sides by length (allowing for small differences due to floating point precision)
                # This ensures we correctly identify the rectangle's dimensions
                sides_grouped = {}
                tolerance = 1e-6  # Tolerance for length comparison

                for s in sides:
                    length, angle = s[0], s[1]
                    matched = False

                    for key in sides_grouped:
                        if abs(length - key) < tolerance:
                            sides_grouped[key].append(s)
                            matched = True
                            break

                    if not matched:
                        sides_grouped[length] = [s]

                # Get unique lengths (should be 2 for a rectangle, parallel sides have equal length)
                unique_lengths = sorted(sides_grouped.keys(), reverse=True)

                if len(unique_lengths) != 2:
                    # If we don't get exactly 2 unique lengths, something is wrong with the rectangle
                    # Fall back to simpler method using bounds
                    bounds = rect.bounds
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]
                    major_length = max(width, height)
                    minor_length = min(width, height)
                    orientation = 0 if width > height else 90
                else:
                    major_length = unique_lengths[0]
                    minor_length = unique_lengths[1]
                    # Get orientation from the major axis
                    orientation = sides_grouped[major_length][0][1]

                # Calculate eccentricity
                if major_length > 0:
                    # Eccentricity for an ellipse: e = sqrt(1 - (b²/a²))
                    # where a is the semi-major axis and b is the semi-minor axis
                    eccentricity = np.sqrt(
                        1 - ((minor_length / 2) ** 2 / (major_length / 2) ** 2)
                    )
                else:
                    eccentricity = 0

                # Calculate elongation (ratio of minor to major axis)
                elongation = major_length / minor_length if major_length > 0 else 1

                return major_length, minor_length, eccentricity, orientation, elongation

            except Exception as e:
                # For debugging
                # print(f"Error calculating axes: {e}")
                return None, None, None, None, None

        # Apply the function and split the results
        axes_data = result.geometry.apply(get_axes_properties)

        if "major_length" in properties:
            result["major_length"] = axes_data.apply(lambda x: x[0] if x else None)
            # Convert to requested units
            if length_unit == "km":
                result["major_length"] = result["major_length"] / 1_000
                result.rename(columns={"major_length": "major_length_km"}, inplace=True)
            else:
                result.rename(columns={"major_length": "major_length_m"}, inplace=True)

        if "minor_length" in properties:
            result["minor_length"] = axes_data.apply(lambda x: x[1] if x else None)
            # Convert to requested units
            if length_unit == "km":
                result["minor_length"] = result["minor_length"] / 1_000
                result.rename(columns={"minor_length": "minor_length_km"}, inplace=True)
            else:
                result.rename(columns={"minor_length": "minor_length_m"}, inplace=True)

        if "eccentricity" in properties:
            result["eccentricity"] = axes_data.apply(lambda x: x[2] if x else None)

        if "orientation" in properties:
            result["orientation"] = axes_data.apply(lambda x: x[3] if x else None)

        if "elongation" in properties:
            result["elongation"] = axes_data.apply(lambda x: x[4] if x else None)

    # Equivalent diameter based on area
    if "diameter_area" in properties:

        def get_equivalent_diameter(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.area <= 0:
                return None
            # Diameter of a circle with the same area: d = 2 * sqrt(A / π)
            return 2 * np.sqrt(geom.area / np.pi)

        result["diameter_area"] = result.geometry.apply(get_equivalent_diameter)

        # Convert to requested units
        if length_unit == "km":
            result["diameter_area"] = result["diameter_area"] / 1_000
            result.rename(
                columns={"diameter_area": "equivalent_diameter_area_km"},
                inplace=True,
            )
        else:
            result.rename(
                columns={"diameter_area": "equivalent_diameter_area_m"},
                inplace=True,
            )

    # Extent (ratio of shape area to bounding box area)
    if "extent" in properties:

        def get_extent(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.area <= 0:
                return None

            bounds = geom.bounds
            bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

            if bbox_area > 0:
                return geom.area / bbox_area
            return None

        result["extent"] = result.geometry.apply(get_extent)

    # Solidity (ratio of shape area to convex hull area)
    if "solidity" in properties:

        def get_solidity(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.area <= 0:
                return None

            convex_hull_area = geom.convex_hull.area

            if convex_hull_area > 0:
                return geom.area / convex_hull_area
            return None

        result["solidity"] = result.geometry.apply(get_solidity)

    # Complexity (ratio of perimeter to area)
    if "complexity" in properties:

        def calc_complexity(geom):
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.area > 0:
                # Shape index: P / (2 * sqrt(π * A))
                # Normalized to 1 for a circle, higher for more complex shapes
                return geom.boundary.length / (2 * np.sqrt(np.pi * geom.area))
            return None

        result["complexity"] = result.geometry.apply(calc_complexity)

    return result


def vector_to_geojson(
    filename: str, output: Optional[str] = None, **kwargs: Any
) -> str:
    """Converts a vector file to a geojson file.

    Args:
        filename (str): The vector file path.
        output (str, optional): The output geojson file path. Defaults to None.

    Returns:
        dict: The geojson dictionary.
    """

    if filename.startswith("http"):
        filename = download_file(filename)

    gdf = gpd.read_file(filename, **kwargs)
    if output is None:
        return gdf.__geo_interface__
    else:
        gdf.to_file(output, driver="GeoJSON")


def geojson_to_coords(
    geojson: str, src_crs: str = "epsg:4326", dst_crs: str = "epsg:4326"
) -> list:
    """Converts a geojson file or a dictionary of feature collection to a list of centroid coordinates.

    Args:
        geojson (str | dict): The geojson file path or a dictionary of feature collection.
        src_crs (str, optional): The source CRS. Defaults to "epsg:4326".
        dst_crs (str, optional): The destination CRS. Defaults to "epsg:4326".

    Returns:
        list: A list of centroid coordinates in the format of [[x1, y1], [x2, y2], ...]
    """

    import json
    import warnings

    warnings.filterwarnings("ignore")

    if isinstance(geojson, dict):
        geojson = json.dumps(geojson)
    gdf = gpd.read_file(geojson, driver="GeoJSON")
    centroids = gdf.geometry.centroid
    centroid_list = [[point.x, point.y] for point in centroids]
    if src_crs != dst_crs:
        centroid_list = transform_coords(
            [x[0] for x in centroid_list],
            [x[1] for x in centroid_list],
            src_crs,
            dst_crs,
        )
        centroid_list = [[x, y] for x, y in zip(centroid_list[0], centroid_list[1])]
    return centroid_list


def boxes_to_vector(
    coords: Union[List[List[float]], np.ndarray],
    src_crs: str,
    dst_crs: str = "EPSG:4326",
    output: Optional[str] = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """
    Convert a list of bounding box coordinates to vector data.

    Args:
        coords (list): A list of bounding box coordinates in the format [[left, top, right, bottom], [left, top, right, bottom], ...].
        src_crs (int or str): The EPSG code or proj4 string representing the source coordinate reference system (CRS) of the input coordinates.
        dst_crs (int or str, optional): The EPSG code or proj4 string representing the destination CRS to reproject the data (default is "EPSG:4326").
        output (str or None, optional): The full file path (including the directory and filename without the extension) where the vector data should be saved.
                                       If None (default), the function returns the GeoDataFrame without saving it to a file.
        **kwargs: Additional keyword arguments to pass to geopandas.GeoDataFrame.to_file() when saving the vector data.

    Returns:
        geopandas.GeoDataFrame or None: The GeoDataFrame with the converted vector data if output is None, otherwise None if the data is saved to a file.
    """

    from shapely.geometry import box

    # Create a list of Shapely Polygon objects based on the provided coordinates
    polygons = [box(*coord) for coord in coords]

    # Create a GeoDataFrame with the Shapely Polygon objects
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=src_crs)

    # Reproject the GeoDataFrame to the specified EPSG code
    gdf_reprojected = gdf.to_crs(dst_crs)

    if output is not None:
        gdf_reprojected.to_file(output, **kwargs)
    else:
        return gdf_reprojected


def geojson_to_xy(
    src_fp: str, geojson: str, coord_crs: str = "epsg:4326", **kwargs: Any
) -> List[List[float]]:
    """Converts a geojson file or a dictionary of feature collection to a list of pixel coordinates.

    Args:
        src_fp: The source raster file path.
        geojson: The geojson file path or a dictionary of feature collection.
        coord_crs: The coordinate CRS of the input coordinates. Defaults to "epsg:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.rowcol.

    Returns:
        A list of pixel coordinates in the format of [[x1, y1], [x2, y2], ...]
    """
    with rasterio.open(src_fp) as src:
        src_crs = src.crs
    coords = geojson_to_coords(geojson, coord_crs, src_crs)
    return coords_to_xy(src_fp, coords, src_crs, **kwargs)


def smooth_vector(
    vector_data: Union[str, gpd.GeoDataFrame],
    output_path: str = None,
    segment_length: float = None,
    smooth_iterations: int = 3,
    num_cores: int = 0,
    merge_collection: bool = True,
    merge_field: str = None,
    merge_multipolygons: bool = True,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Smooth a vector data using the smoothify library.
        See https://github.com/DPIRD-DMA/Smoothify for more details.

    Args:
        vector_data: The vector data to smooth.
        output_path: The path to save the smoothed vector data. If None, returns the smoothed vector data.
        segment_length: Resolution of the original raster data in map units. If None (default), automatically
            detects by finding the minimum segment length (from a data sample). Recommended to specify explicitly when known.
        smooth_iterations: The number of iterations to smooth the vector data.
        num_cores: Number of cores to use for parallel processing. If 0 (default), uses all available cores.
        merge_collection: Whether to merge/dissolve adjacent geometries in collections before smoothing.
        merge_field: Column name to use for dissolving geometries. Only valid when merge_collection=True.
            If None, dissolves all geometries together. If specified, dissolves geometries grouped by the column values.
        merge_multipolygons: Whether to merge adjacent polygons within MultiPolygons before smoothing
        preserve_area: Whether to restore original area after smoothing via buffering (applies to Polygons only)
        area_tolerance: Percentage of original area allowed as error (e.g., 0.01 = 0.01% error = 99.99% preservation).
            Only affects Polygons when preserve_area=True

    Returns:
        gpd.GeoDataFrame: The smoothed vector data.

    Examples:
        >>> import geoai
        >>> gdf = geoai.read_vector("path/to/vector.geojson")
        >>> smoothed_gdf = geoai.smooth_vector(gdf, smooth_iterations=3, output_path="path/to/smoothed_vector.geojson")
        >>> smoothed_gdf.head()
        >>> smoothed_gdf.explore()
    """
    try:
        from smoothify import smoothify
    except ImportError:
        install_package("smoothify")
        from smoothify import smoothify

    if isinstance(vector_data, str):
        vector_data = leafmap.read_vector(vector_data)

    smoothed_vector_data = smoothify(
        geom=vector_data,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        num_cores=num_cores,
        merge_collection=merge_collection,
        merge_field=merge_field,
        merge_multipolygons=merge_multipolygons,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
        **kwargs,
    )
    if output_path is not None:
        smoothed_vector_data.to_file(output_path)
    return smoothed_vector_data
