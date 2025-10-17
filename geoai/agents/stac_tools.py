"""Tools for STAC catalog search and interaction."""

import json
from typing import Any, Dict, List, Optional

from strands import tool

from ..download import pc_collection_list, pc_stac_search
from .stac_models import (
    LocationInfo,
    STACAssetInfo,
    STACCollectionInfo,
    STACItemInfo,
    STACSearchResult,
)


class STACTools:
    """Collection of tools for searching and interacting with STAC catalogs."""

    def __init__(
        self,
        endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    ) -> None:
        """Initialize STAC tools.

        Args:
            endpoint: STAC API endpoint URL. Defaults to Microsoft Planetary Computer.
        """
        self.endpoint = endpoint

    @tool(
        description="List and search available STAC collections from Planetary Computer"
    )
    def list_collections(
        self,
        filter_keyword: Optional[str] = None,
        detailed: bool = False,
    ) -> str:
        """List available STAC collections from Planetary Computer.

        Args:
            filter_keyword: Optional keyword to filter collections (searches in id, title, description).
            detailed: If True, return detailed information including temporal extent, license, etc.

        Returns:
            JSON string containing list of collections with their metadata.
        """
        try:
            # Get collections using existing function
            df = pc_collection_list(
                endpoint=self.endpoint,
                detailed=detailed,
                filter_by=None,
                sort_by="id",
            )

            # Apply keyword filtering if specified
            if filter_keyword:
                mask = df["id"].str.contains(filter_keyword, case=False, na=False) | df[
                    "title"
                ].str.contains(filter_keyword, case=False, na=False)
                if "description" in df.columns:
                    mask |= df["description"].str.contains(
                        filter_keyword, case=False, na=False
                    )
                df = df[mask]

            # Convert to list of dictionaries
            collections = df.to_dict("records")

            # Convert to structured models
            collection_models = []
            for col in collections:
                collection_models.append(
                    STACCollectionInfo(
                        id=col.get("id", ""),
                        title=col.get("title", ""),
                        description=col.get("description"),
                        license=col.get("license"),
                        temporal_extent=col.get("temporal_extent"),
                        spatial_extent=col.get("bbox"),
                        providers=col.get("providers"),
                        keywords=col.get("keywords"),
                    )
                )

            result = {
                "count": len(collection_models),
                "filter_keyword": filter_keyword,
                "collections": [c.model_dump() for c in collection_models],
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(
        description="Search for STAC items in a specific collection with optional filters"
    )
    def search_items(
        self,
        collection: str,
        bbox: Optional[List[float]] = None,
        time_range: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 10,
        max_items: Optional[int] = None,
    ) -> str:
        """Search for STAC items in the Planetary Computer catalog.

        Args:
            collection: Collection ID to search (e.g., "sentinel-2-l2a", "naip", "landsat-c2-l2").
            bbox: Bounding box as [west, south, east, north] in WGS84 coordinates.
                Example: [-122.5, 37.7, -122.3, 37.8] for San Francisco area.
            time_range: Time range as "start/end" string in ISO format.
                Example: "2024-09-01/2024-09-30" or "2024-09-01/2024-09-01" for single day.
            query: Query parameters for filtering.
                Example: {"eo:cloud_cover": {"lt": 10}} for cloud cover less than 10%.
            limit: Number of items to return per page.
                Example: 10 for 10 items per page.
            max_items: Maximum number of items to return (default: 10).

        Returns:
            JSON string containing search results with item details including IDs, URLs, and metadata.
        """
        try:
            # Search using existing function
            items = pc_stac_search(
                collection=collection,
                bbox=bbox,
                time_range=time_range,
                query=query,
                limit=limit,
                max_items=max_items,
                endpoint=self.endpoint,
            )

            # Convert to structured models
            item_models = []
            for item in items:
                # Extract assets
                assets = []
                for key, asset in item.assets.items():
                    assets.append(
                        STACAssetInfo(
                            key=key,
                            title=asset.title,
                        )
                    )

                item_models.append(
                    STACItemInfo(
                        id=item.id,
                        collection=item.collection_id,
                        datetime=str(item.datetime) if item.datetime else None,
                        bbox=list(item.bbox) if item.bbox else None,
                        assets=assets,
                        # properties=item.properties,
                    )
                )

            # Create search result
            result = STACSearchResult(
                query=f"Collection: {collection}",
                collection=collection,
                item_count=len(item_models),
                items=item_models,
                bbox=bbox,
                time_range=time_range,
            )

            return json.dumps(result.model_dump(), indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="Get detailed information about a specific STAC item")
    def get_item_info(
        self,
        item_id: str,
        collection: str,
    ) -> str:
        """Get detailed information about a specific STAC item.

        Args:
            item_id: The STAC item ID to retrieve.
            collection: The collection ID containing the item.

        Returns:
            JSON string with detailed item information including all assets and metadata.
        """
        try:
            # Search for the specific item
            items = pc_stac_search(
                collection=collection,
                bbox=None,
                time_range=None,
                query={"id": {"eq": item_id}},
                limit=1,
                max_items=1,
                endpoint=self.endpoint,
            )

            if not items:
                return json.dumps(
                    {"error": f"Item {item_id} not found in collection {collection}"}
                )

            item = items[0]

            # Extract all assets with full details
            assets = []
            for key, asset in item.assets.items():
                asset_info = {
                    "key": key,
                    "href": asset.href,
                    "type": asset.media_type,
                    "title": asset.title,
                    "description": getattr(asset, "description", None),
                    "roles": getattr(asset, "roles", None),
                }
                assets.append(asset_info)

            result = {
                "id": item.id,
                "collection": item.collection_id,
                "datetime": str(item.datetime) if item.datetime else None,
                "bbox": list(item.bbox) if item.bbox else None,
                # "properties": item.properties,
                "assets": assets,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="Parse a location name and return its bounding box coordinates")
    def geocode_location(self, location_name: str) -> str:
        """Convert a location name to geographic coordinates and bounding box.

        This tool uses a geocoding service to find the coordinates for a given location name.

        Args:
            location_name: Name of the location (e.g., "San Francisco", "New York", "Paris, France").

        Returns:
            JSON string with location info including bounding box and center coordinates.
        """
        try:
            import requests

            # Use Nominatim for geocoding (OpenStreetMap)
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location_name,
                "format": "json",
                "limit": 1,
            }
            headers = {"User-Agent": "GeoAI-STAC-Agent/1.0"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            results = response.json()

            if not results:
                return json.dumps({"error": f"Location '{location_name}' not found"})

            result = results[0]
            bbox = [
                float(result["boundingbox"][2]),  # west
                float(result["boundingbox"][0]),  # south
                float(result["boundingbox"][3]),  # east
                float(result["boundingbox"][1]),  # north
            ]
            center = [float(result["lon"]), float(result["lat"])]

            location_info = LocationInfo(
                name=result.get("display_name", location_name),
                bbox=bbox,
                center=center,
            )

            return json.dumps(location_info.model_dump(), indent=2)

        except Exception as e:
            return json.dumps({"error": f"Geocoding error: {str(e)}"})

    @tool(
        description="Get common STAC collection IDs for different satellite/aerial imagery types"
    )
    def get_common_collections(self) -> str:
        """Get a list of commonly used STAC collections from Planetary Computer.

        Returns:
            JSON string with collection IDs and descriptions for popular datasets.
        """
        common_collections = {
            "sentinel-2-l2a": "Sentinel-2 Level-2A - Multispectral imagery (10m-60m resolution, global coverage)",
            "landsat-c2-l2": "Landsat Collection 2 Level-2 - Multispectral imagery (30m resolution, global coverage)",
            "naip": "NAIP - National Agriculture Imagery Program (1m resolution, USA only)",
            "sentinel-1-grd": "Sentinel-1 GRD - Synthetic Aperture Radar imagery (global coverage)",
            "aster-l1t": "ASTER L1T - Multispectral and thermal imagery (15m-90m resolution)",
            "cop-dem-glo-30": "Copernicus DEM - Global Digital Elevation Model (30m resolution)",
            "hgb": "HGB - High Resolution Building Footprints",
            "io-lulc": "Impact Observatory Land Use/Land Cover - Annual 10m resolution land cover",
            "modis": "MODIS - Moderate Resolution Imaging Spectroradiometer (250m-1km resolution)",
            "daymet-daily-hi": "Daymet - Daily surface weather data for Hawaii",
        }

        result = {
            "count": len(common_collections),
            "collections": [
                {"id": k, "description": v} for k, v in common_collections.items()
            ],
        }

        return json.dumps(result, indent=2)
