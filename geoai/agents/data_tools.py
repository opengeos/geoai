"""Tools for geospatial data processing and analysis within AI agents.

Provides agent-callable tools for raster/vector inspection, format conversion,
and basic data operations.  These complement the existing CatalogTools
(search), MapTools (visualization), and STACTools (STAC search) categories.
"""

import json
import os
from typing import Any, Dict, List, Optional

from strands import tool


class DataTools:
    """Collection of tools for inspecting and processing geospatial data."""

    @tool(
        description="Get metadata and band statistics for a raster file (GeoTIFF, etc.)"
    )
    def inspect_raster(self, raster_path: str) -> str:
        """Inspect a raster file and return its metadata and per-band statistics.

        Args:
            raster_path: Path to the raster file.

        Returns:
            JSON string with driver, dimensions, CRS, bounds, resolution,
            nodata, and per-band min/max/mean/std statistics.
        """
        from ..utils.raster import get_raster_info

        if not os.path.isfile(raster_path):
            return json.dumps({"error": f"File not found: {raster_path}"})
        info = get_raster_info(raster_path)
        # Make bounds serializable
        info["bounds"] = {
            "left": info["bounds"].left,
            "bottom": info["bounds"].bottom,
            "right": info["bounds"].right,
            "top": info["bounds"].top,
        }
        info["transform"] = list(info["transform"])[:6]
        return json.dumps(info, indent=2, default=str)

    @tool(description="Get metadata and attribute summary for a vector file")
    def inspect_vector(self, vector_path: str) -> str:
        """Inspect a vector file and return its metadata and attribute summary.

        Args:
            vector_path: Path to the vector file (GeoJSON, Shapefile, GeoPackage, etc.).

        Returns:
            JSON string with feature count, CRS, geometry type, bounds,
            and attribute column names and dtypes.
        """
        from ..utils.raster import read_vector

        if not os.path.isfile(vector_path):
            return json.dumps({"error": f"File not found: {vector_path}"})
        gdf = read_vector(vector_path)
        info = {
            "feature_count": len(gdf),
            "crs": str(gdf.crs) if gdf.crs else None,
            "geometry_type": (
                gdf.geometry.geom_type.unique().tolist() if len(gdf) > 0 else []
            ),
            "bounds": dict(
                zip(["minx", "miny", "maxx", "maxy"], gdf.total_bounds.tolist())
            ),
            "columns": {
                col: str(gdf[col].dtype) for col in gdf.columns if col != "geometry"
            },
        }
        return json.dumps(info, indent=2, default=str)

    @tool(description="Clip a raster to a bounding box and save the result")
    def clip_raster(
        self,
        input_raster: str,
        output_raster: str,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> str:
        """Clip a raster file to a geographic bounding box.

        Args:
            input_raster: Path to the input raster.
            output_raster: Path for the clipped output raster.
            min_lon: Western longitude of the bounding box.
            min_lat: Southern latitude of the bounding box.
            max_lon: Eastern longitude of the bounding box.
            max_lat: Northern latitude of the bounding box.

        Returns:
            JSON string confirming the output path and its dimensions.
        """
        from ..utils.raster import clip_raster_by_bbox

        bbox = [min_lon, min_lat, max_lon, max_lat]
        clip_raster_by_bbox(input_raster, output_raster, bbox)
        from ..utils.raster import get_raster_info

        info = get_raster_info(output_raster)
        return json.dumps(
            {
                "output_path": output_raster,
                "width": info["width"],
                "height": info["height"],
                "bands": info["count"],
            }
        )

    @tool(description="Convert a raster mask to vector polygons")
    def raster_to_vector(
        self,
        raster_path: str,
        output_path: str,
    ) -> str:
        """Convert a raster (e.g. segmentation mask) to vector polygons.

        Args:
            raster_path: Path to the input raster.
            output_path: Path for the output vector file (.geojson, .shp, .gpkg).

        Returns:
            JSON string with the output path and feature count.
        """
        from ..utils.raster import raster_to_vector as _r2v, read_vector

        _r2v(raster_path, output_path)
        gdf = read_vector(output_path)
        return json.dumps({"output_path": output_path, "feature_count": len(gdf)})

    @tool(description="List files in a directory, optionally filtered by extension")
    def list_files(
        self,
        directory: str,
        extension: Optional[str] = None,
    ) -> str:
        """List files in a directory with optional extension filtering.

        Args:
            directory: Directory path to list.
            extension: File extension filter (e.g. '.tif', '.geojson').

        Returns:
            JSON list of file paths.
        """
        if not os.path.isdir(directory):
            return json.dumps({"error": f"Directory not found: {directory}"})
        files = []
        for f in sorted(os.listdir(directory)):
            full = os.path.join(directory, f)
            if os.path.isfile(full):
                if extension is None or f.lower().endswith(extension.lower()):
                    files.append(full)
        return json.dumps(files[:100])  # cap at 100 entries
