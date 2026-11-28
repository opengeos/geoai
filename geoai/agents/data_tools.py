"""Tools for geospatial data processing and analysis within AI agents.

Provides agent-callable tools for raster/vector inspection, format conversion,
and basic data operations.  These complement the existing CatalogTools
(search), MapTools (visualization), and STACTools (STAC search) categories.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)


class DataTools:
    """Collection of tools for inspecting and processing geospatial data.

    Args:
        workspace_dir: Optional root workspace directory. When set, relative
            paths are resolved under it and all reads, listings, and writes
            are restricted to this directory (and any ``allowed_roots``).
        allowed_roots: Optional list of allowed root directories. When set,
            all file and directory paths must resolve inside one of these
            roots (after symlink resolution via ``os.path.realpath``). Passing
            an empty list ``[]`` denies all file paths.
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        allowed_roots: Optional[List[str]] = None,
    ) -> None:
        self.workspace_dir: Optional[str] = (
            os.path.realpath(workspace_dir) if workspace_dir is not None else None
        )
        if workspace_dir is None and allowed_roots is None:
            logger.warning(
                "DataTools initialized without workspace_dir or allowed_roots; "
                "file access is unrestricted. Pass workspace_dir or allowed_roots "
                "to restrict reads and writes to specific directories."
            )
            self.allowed_roots: Optional[List[str]] = None
        else:
            roots: List[str] = []
            if self.workspace_dir is not None:
                roots.append(self.workspace_dir)
            if allowed_roots is not None:
                roots.extend(os.path.realpath(r) for r in allowed_roots)
            self.allowed_roots = roots

    def _resolve_and_validate_path(self, path: str) -> str:
        """Resolve *path* via ``os.path.realpath`` and enforce root containment."""
        if self.workspace_dir is not None and not os.path.isabs(path):
            candidate = os.path.join(self.workspace_dir, path)
        else:
            candidate = path
        resolved = os.path.realpath(candidate)
        if self.allowed_roots is not None:
            allowed = False
            for root in self.allowed_roots:
                try:
                    if os.path.commonpath([root, resolved]) == root:
                        allowed = True
                        break
                except ValueError:
                    continue
            if not allowed:
                raise ValueError(
                    f"Access denied: path '{path}' resolves outside allowed directories."
                )
        return resolved

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

        resolved = self._resolve_and_validate_path(raster_path)
        if not os.path.isfile(resolved):
            return json.dumps({"error": f"File not found: {raster_path}"})
        info = get_raster_info(resolved)
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

        resolved = self._resolve_and_validate_path(vector_path)
        if not os.path.isfile(resolved):
            return json.dumps({"error": f"File not found: {vector_path}"})
        gdf = read_vector(resolved)
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
        from ..utils.raster import clip_raster_by_bbox, get_raster_info

        resolved_in = self._resolve_and_validate_path(input_raster)
        resolved_out = self._resolve_and_validate_path(output_raster)
        out_parent = os.path.dirname(resolved_out)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        bbox = [min_lon, min_lat, max_lon, max_lat]
        clip_raster_by_bbox(resolved_in, resolved_out, bbox)

        info = get_raster_info(resolved_out)
        return json.dumps(
            {
                "output_path": resolved_out,
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

        resolved_in = self._resolve_and_validate_path(raster_path)
        resolved_out = self._resolve_and_validate_path(output_path)
        out_parent = os.path.dirname(resolved_out)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        _r2v(resolved_in, resolved_out)
        gdf = read_vector(resolved_out)
        return json.dumps({"output_path": resolved_out, "feature_count": len(gdf)})

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
        resolved_dir = self._resolve_and_validate_path(directory)
        if not os.path.isdir(resolved_dir):
            return json.dumps({"error": f"Directory not found: {directory}"})
        files = []
        for f in sorted(os.listdir(resolved_dir)):
            full = os.path.join(resolved_dir, f)
            if os.path.isfile(full):
                if self.allowed_roots is not None:
                    try:
                        self._resolve_and_validate_path(full)
                    except ValueError:
                        continue
                if extension is None or f.lower().endswith(extension.lower()):
                    files.append(full)
        return json.dumps(files[:100])  # cap at 100 entries
