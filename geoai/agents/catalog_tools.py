"""Tools for searching data catalogs."""

import io
import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from strands import tool

from .catalog_models import CatalogDatasetInfo, CatalogSearchResult, LocationInfo


class CatalogTools:
    """Collection of tools for searching and interacting with data catalogs."""

    # Common location cache to avoid repeated geocoding
    _LOCATION_CACHE = {
        "san francisco": {
            "name": "San Francisco",
            "bbox": [-122.5155, 37.7034, -122.3549, 37.8324],
            "center": [-122.4194, 37.7749],
        },
        "new york": {
            "name": "New York",
            "bbox": [-74.0479, 40.6829, -73.9067, 40.8820],
            "center": [-73.9352, 40.7306],
        },
        "new york city": {
            "name": "New York City",
            "bbox": [-74.0479, 40.6829, -73.9067, 40.8820],
            "center": [-73.9352, 40.7306],
        },
        "paris": {
            "name": "Paris",
            "bbox": [2.2241, 48.8156, 2.4698, 48.9022],
            "center": [2.3522, 48.8566],
        },
        "london": {
            "name": "London",
            "bbox": [-0.5103, 51.2868, 0.3340, 51.6919],
            "center": [-0.1276, 51.5074],
        },
        "tokyo": {
            "name": "Tokyo",
            "bbox": [139.5694, 35.5232, 139.9182, 35.8173],
            "center": [139.6917, 35.6895],
        },
        "los angeles": {
            "name": "Los Angeles",
            "bbox": [-118.6682, 33.7037, -118.1553, 34.3373],
            "center": [-118.2437, 34.0522],
        },
        "chicago": {
            "name": "Chicago",
            "bbox": [-87.9401, 41.6445, -87.5241, 42.0230],
            "center": [-87.6298, 41.8781],
        },
        "seattle": {
            "name": "Seattle",
            "bbox": [-122.4595, 47.4810, -122.2244, 47.7341],
            "center": [-122.3321, 47.6062],
        },
        "california": {
            "name": "California",
            "bbox": [-124.4820, 32.5288, -114.1315, 42.0095],
            "center": [-119.4179, 36.7783],
        },
        "las vegas": {
            "name": "Las Vegas",
            "bbox": [-115.3711, 35.9630, -114.9372, 36.2610],
            "center": [-115.1400, 36.1177],
        },
    }

    def __init__(
        self,
        catalog_url: Optional[str] = None,
        catalog_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize CatalogTools.

        Args:
            catalog_url: URL to a catalog file (TSV, CSV, or JSON). If None, must provide catalog_df.
            catalog_df: Pre-loaded catalog as a pandas DataFrame. If None, must provide catalog_url.
        """
        self.catalog_url = catalog_url
        self._catalog_df = catalog_df
        self._cache = {}
        # Runtime cache for geocoding results
        self._geocode_cache = {}

        # Load catalog if URL provided
        if catalog_url and catalog_df is None:
            self._catalog_df = self._load_catalog(catalog_url)

    def _load_catalog(self, url: str) -> pd.DataFrame:
        """Load catalog from a URL.

        Args:
            url: URL to catalog file (TSV, CSV, or JSON).

        Returns:
            DataFrame containing catalog data.
        """
        # Check cache first
        if url in self._cache:
            return self._cache[url]

        try:
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Determine file type and parse
            if url.endswith(".tsv"):
                df = pd.read_csv(io.StringIO(response.text), sep="\t")
            elif url.endswith(".csv"):
                df = pd.read_csv(io.StringIO(response.text))
            elif url.endswith(".json"):
                df = pd.read_json(io.StringIO(response.text))
            else:
                # Try to auto-detect (default to TSV)
                df = pd.read_csv(io.StringIO(response.text), sep="\t")

            # Cache the result
            self._cache[url] = df
            return df

        except Exception as e:
            raise ValueError(f"Failed to load catalog from {url}: {str(e)}")

    def _parse_bbox_string(self, bbox_str: str) -> Optional[List[float]]:
        """Parse a bbox string to a list of floats.

        Args:
            bbox_str: Bounding box string in format "minLon, minLat, maxLon, maxLat".

        Returns:
            List of floats [minLon, minLat, maxLon, maxLat] or None if parsing fails.
        """
        try:
            if pd.isna(bbox_str) or not bbox_str:
                return None
            parts = str(bbox_str).split(",")
            if len(parts) != 4:
                return None
            bbox = [float(p.strip()) for p in parts]
            return bbox
        except (ValueError, AttributeError):
            return None

    def _bbox_intersects(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes intersect.

        Args:
            bbox1: First bbox as [minLon, minLat, maxLon, maxLat].
            bbox2: Second bbox as [minLon, minLat, maxLon, maxLat].

        Returns:
            True if bboxes intersect, False otherwise.
        """
        # Check if boxes do NOT intersect, then negate
        # bbox1 is completely to the left, right, below, or above bbox2
        return not (
            bbox1[2] < bbox2[0]  # bbox1 maxLon < bbox2 minLon (left of)
            or bbox1[0] > bbox2[2]  # bbox1 minLon > bbox2 maxLon (right of)
            or bbox1[3] < bbox2[1]  # bbox1 maxLat < bbox2 minLat (below)
            or bbox1[1] > bbox2[3]  # bbox1 minLat > bbox2 maxLat (above)
        )

    def _bbox_contains_point(self, bbox: List[float], lon: float, lat: float) -> bool:
        """Check if a bounding box contains a point.

        Args:
            bbox: Bounding box as [minLon, minLat, maxLon, maxLat].
            lon: Longitude of the point.
            lat: Latitude of the point.

        Returns:
            True if bbox contains the point, False otherwise.
        """
        return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]

    def _search_dataframe(
        self,
        df: pd.DataFrame,
        keywords: Optional[str] = None,
        dataset_type: Optional[str] = None,
        provider: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 10,
    ) -> pd.DataFrame:
        """Search dataframe with filters.

        Args:
            df: DataFrame to search.
            keywords: Keywords to search for (searches in id, title, keywords, description).
            dataset_type: Filter by dataset type.
            provider: Filter by provider.
            start_date: Filter datasets that have data after this date (YYYY-MM-DD).
            end_date: Filter datasets that have data before this date (YYYY-MM-DD).
            max_results: Maximum number of results to return.

        Returns:
            Filtered DataFrame.
        """
        result_df = df.copy()

        # Apply keyword search
        if keywords:
            keyword_lower = keywords.lower()
            mask = pd.Series([False] * len(result_df), index=result_df.index)

            # Search in id
            if "id" in result_df.columns:
                mask |= (
                    result_df["id"]
                    .astype(str)
                    .str.lower()
                    .str.contains(keyword_lower, na=False)
                )

            # Search in title
            if "title" in result_df.columns:
                mask |= (
                    result_df["title"]
                    .astype(str)
                    .str.lower()
                    .str.contains(keyword_lower, na=False)
                )

            # Search in keywords
            if "keywords" in result_df.columns:
                mask |= (
                    result_df["keywords"]
                    .astype(str)
                    .str.lower()
                    .str.contains(keyword_lower, na=False)
                )

            # Search in description
            if "description" in result_df.columns:
                mask |= (
                    result_df["description"]
                    .astype(str)
                    .str.lower()
                    .str.contains(keyword_lower, na=False)
                )

            result_df = result_df[mask]

        # Filter by type
        if dataset_type and "type" in result_df.columns:
            result_df = result_df[
                result_df["type"]
                .astype(str)
                .str.lower()
                .str.contains(dataset_type.lower(), na=False)
            ]

        # Filter by provider
        if provider and "provider" in result_df.columns:
            result_df = result_df[
                result_df["provider"]
                .astype(str)
                .str.lower()
                .str.contains(provider.lower(), na=False)
            ]

        # Filter by temporal range
        if start_date and "end_date" in result_df.columns:
            # Keep datasets where end_date >= start_date (dataset has data after start_date)
            result_df = result_df[
                (result_df["end_date"].notna()) & (result_df["end_date"] >= start_date)
            ]

        if end_date and "start_date" in result_df.columns:
            # Keep datasets where start_date <= end_date (dataset has data before end_date)
            result_df = result_df[
                (result_df["start_date"].notna())
                & (result_df["start_date"] <= end_date)
            ]

        # Limit results
        if len(result_df) > max_results:
            result_df = result_df.head(max_results)

        return result_df

    @tool(
        description="Search for datasets in the catalog using keywords, filters, and date range"
    )
    def search_datasets(
        self,
        keywords: Optional[str] = None,
        dataset_type: Optional[str] = None,
        provider: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[Union[str, int]] = 10,
    ) -> str:
        """Search for datasets in the catalog.

        Args:
            keywords: Keywords to search for. Searches in id, title, keywords, and description fields.
                Example: "landcover" will find datasets with "landcover" in any searchable field.
            dataset_type: Filter by dataset type (e.g., "image", "image_collection", "table").
                Example: "image_collection" to find only image collections.
            provider: Filter by data provider name.
                Example: "NASA" to find only NASA datasets.
            start_date: Filter datasets that have data after this date in YYYY-MM-DD format.
                Example: "2020-01-01" to find datasets with data from 2020 onwards.
            end_date: Filter datasets that have data before this date in YYYY-MM-DD format.
                Example: "2023-12-31" to find datasets with data up to 2023.
            max_results: Maximum number of results to return (default: 10).

        Returns:
            JSON string containing search results with dataset information.
        """
        try:
            if self._catalog_df is None:
                return json.dumps(
                    {
                        "error": "No catalog loaded. Please provide catalog_url or catalog_df."
                    }
                )

            # Parse max_results if it's a string
            if isinstance(max_results, str):
                try:
                    max_results = int(max_results)
                except ValueError:
                    max_results = 10

            # Search the dataframe
            result_df = self._search_dataframe(
                self._catalog_df,
                keywords=keywords,
                dataset_type=dataset_type,
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
            )

            # Convert to models
            dataset_models = []
            for _, row in result_df.iterrows():
                dataset_models.append(
                    CatalogDatasetInfo(
                        id=str(row.get("id", "")),
                        title=str(row.get("title", "")),
                        type=(
                            str(row.get("type", ""))
                            if pd.notna(row.get("type"))
                            else None
                        ),
                        provider=(
                            str(row.get("provider", ""))
                            if pd.notna(row.get("provider"))
                            else None
                        ),
                        description=(
                            str(row.get("description", ""))
                            if pd.notna(row.get("description"))
                            else None
                        ),
                        keywords=(
                            str(row.get("keywords", ""))
                            if pd.notna(row.get("keywords"))
                            else None
                        ),
                        snippet=(
                            str(row.get("snippet", ""))
                            if pd.notna(row.get("snippet"))
                            else None
                        ),
                        start_date=(
                            str(row.get("start_date", ""))
                            if pd.notna(row.get("start_date"))
                            else None
                        ),
                        end_date=(
                            str(row.get("end_date", ""))
                            if pd.notna(row.get("end_date"))
                            else None
                        ),
                        bbox=(
                            str(row.get("bbox", ""))
                            if pd.notna(row.get("bbox"))
                            else None
                        ),
                        license=(
                            str(row.get("license", ""))
                            if pd.notna(row.get("license"))
                            else None
                        ),
                        url=(
                            str(row.get("url", ""))
                            if pd.notna(row.get("url"))
                            else None
                        ),
                        catalog=(
                            str(row.get("catalog", ""))
                            if pd.notna(row.get("catalog"))
                            else None
                        ),
                        deprecated=(
                            str(row.get("deprecated", ""))
                            if pd.notna(row.get("deprecated"))
                            else None
                        ),
                    )
                )

            # Create search result
            filters = {}
            if keywords:
                filters["keywords"] = keywords
            if dataset_type:
                filters["dataset_type"] = dataset_type
            if provider:
                filters["provider"] = provider

            query_parts = []
            if keywords:
                query_parts.append(f"keywords: {keywords}")
            if dataset_type:
                query_parts.append(f"type: {dataset_type}")
            if provider:
                query_parts.append(f"provider: {provider}")
            query_str = ", ".join(query_parts) if query_parts else "all datasets"

            result = CatalogSearchResult(
                query=query_str,
                dataset_count=len(dataset_models),
                datasets=dataset_models,
                filters=filters if filters else None,
            )

            return json.dumps(result.model_dump(), indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="Get detailed information about a specific dataset")
    def get_dataset_info(
        self,
        dataset_id: str,
    ) -> str:
        """Get detailed information about a specific dataset.

        Args:
            dataset_id: The dataset ID to retrieve.

        Returns:
            JSON string with detailed dataset information.
        """
        try:
            if self._catalog_df is None:
                return json.dumps({"error": "No catalog loaded."})

            # Find the dataset
            if "id" not in self._catalog_df.columns:
                return json.dumps({"error": "Catalog does not have 'id' column."})

            result_df = self._catalog_df[self._catalog_df["id"] == dataset_id]

            if len(result_df) == 0:
                return json.dumps(
                    {"error": f"Dataset '{dataset_id}' not found in catalog."}
                )

            row = result_df.iloc[0]

            # Convert to model
            dataset = CatalogDatasetInfo(
                id=str(row.get("id", "")),
                title=str(row.get("title", "")),
                type=str(row.get("type", "")) if pd.notna(row.get("type")) else None,
                provider=(
                    str(row.get("provider", ""))
                    if pd.notna(row.get("provider"))
                    else None
                ),
                description=(
                    str(row.get("description", ""))
                    if pd.notna(row.get("description"))
                    else None
                ),
                keywords=(
                    str(row.get("keywords", ""))
                    if pd.notna(row.get("keywords"))
                    else None
                ),
                snippet=(
                    str(row.get("snippet", ""))
                    if pd.notna(row.get("snippet"))
                    else None
                ),
                start_date=(
                    str(row.get("start_date", ""))
                    if pd.notna(row.get("start_date"))
                    else None
                ),
                end_date=(
                    str(row.get("end_date", ""))
                    if pd.notna(row.get("end_date"))
                    else None
                ),
                bbox=str(row.get("bbox", "")) if pd.notna(row.get("bbox")) else None,
                license=(
                    str(row.get("license", ""))
                    if pd.notna(row.get("license"))
                    else None
                ),
                url=str(row.get("url", "")) if pd.notna(row.get("url")) else None,
                catalog=(
                    str(row.get("catalog", ""))
                    if pd.notna(row.get("catalog"))
                    else None
                ),
                deprecated=(
                    str(row.get("deprecated", ""))
                    if pd.notna(row.get("deprecated"))
                    else None
                ),
            )

            return json.dumps(dataset.model_dump(), indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="List unique dataset types available in the catalog")
    def list_dataset_types(self) -> str:
        """List unique dataset types available in the catalog.

        Returns:
            JSON string with list of dataset types.
        """
        try:
            if self._catalog_df is None:
                return json.dumps({"error": "No catalog loaded."})

            if "type" not in self._catalog_df.columns:
                return json.dumps({"error": "Catalog does not have 'type' column."})

            types = self._catalog_df["type"].dropna().unique().tolist()
            types.sort()

            result = {
                "count": len(types),
                "types": types,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="List unique data providers in the catalog")
    def list_providers(self) -> str:
        """List unique data providers in the catalog.

        Returns:
            JSON string with list of providers.
        """
        try:
            if self._catalog_df is None:
                return json.dumps({"error": "No catalog loaded."})

            if "provider" not in self._catalog_df.columns:
                return json.dumps({"error": "Catalog does not have 'provider' column."})

            providers = self._catalog_df["provider"].dropna().unique().tolist()
            providers.sort()

            result = {
                "count": len(providers),
                "providers": providers,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="Get catalog statistics and summary information")
    def get_catalog_stats(self) -> str:
        """Get statistics about the catalog.

        Returns:
            JSON string with catalog statistics.
        """
        try:
            if self._catalog_df is None:
                return json.dumps({"error": "No catalog loaded."})

            stats = {
                "total_datasets": len(self._catalog_df),
                "columns": list(self._catalog_df.columns),
            }

            # Add type counts if available
            if "type" in self._catalog_df.columns:
                type_counts = self._catalog_df["type"].value_counts().to_dict()
                stats["dataset_types"] = type_counts

            # Add provider counts if available
            if "provider" in self._catalog_df.columns:
                # Get top 10 providers
                provider_counts = (
                    self._catalog_df["provider"].value_counts().head(10).to_dict()
                )
                stats["top_providers"] = provider_counts

            return json.dumps(stats, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool(description="Parse a location name and return its bounding box coordinates")
    def geocode_location(self, location_name: str) -> str:
        """Convert a location name to geographic coordinates and bounding box.

        This tool uses a geocoding service to find the coordinates for a given location name.

        Args:
            location_name: Name of the location (e.g., "San Francisco", "New York", "Paris, France", "California").

        Returns:
            JSON string with location info including bounding box and center coordinates.
        """
        try:
            # Check static cache first (common locations)
            location_key = location_name.lower().strip()
            if location_key in self._LOCATION_CACHE:
                cached = self._LOCATION_CACHE[location_key]
                location_info = LocationInfo(
                    name=cached["name"],
                    bbox=cached["bbox"],
                    center=cached["center"],
                )
                return json.dumps(location_info.model_dump(), indent=2)

            # Check runtime cache
            if location_key in self._geocode_cache:
                return self._geocode_cache[location_key]

            # Geocode using Nominatim
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location_name,
                "format": "json",
                "limit": 1,
            }
            headers = {"User-Agent": "GeoAI-Catalog-Agent/1.0"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            results = response.json()

            if not results:
                error_result = json.dumps(
                    {"error": f"Location '{location_name}' not found"}
                )
                self._geocode_cache[location_key] = error_result
                return error_result

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

            result_json = json.dumps(location_info.model_dump(), indent=2)
            # Cache the result
            self._geocode_cache[location_key] = result_json

            return result_json

        except Exception as e:
            return json.dumps({"error": f"Geocoding error: {str(e)}"})

    @tool(
        description="Search for datasets by geographic region, keywords, and date range"
    )
    def search_by_region(
        self,
        bbox: Optional[Union[str, List[float]]] = None,
        location: Optional[str] = None,
        keywords: Optional[str] = None,
        dataset_type: Optional[str] = None,
        provider: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[Union[str, int]] = 10,
    ) -> str:
        """Search for datasets that cover a specific geographic region.

        Args:
            bbox: Bounding box as [west, south, east, north] or comma-separated string.
                Example: [-122.5, 37.5, -122.0, 38.0] for San Francisco Bay Area.
            location: Location name to geocode into a bounding box.
                Example: "California", "San Francisco", "New York City".
            keywords: Additional keywords to search for in dataset metadata.
            dataset_type: Filter by dataset type (e.g., "image", "image_collection").
            provider: Filter by data provider name.
            start_date: Filter datasets that have data after this date in YYYY-MM-DD format.
                Example: "2020-01-01" to find datasets with data from 2020 onwards.
            end_date: Filter datasets that have data before this date in YYYY-MM-DD format.
                Example: "2023-12-31" to find datasets with data up to 2023.
            max_results: Maximum number of results to return (default: 10).

        Returns:
            JSON string containing search results with datasets that intersect the search region.
        """
        try:
            if self._catalog_df is None:
                return json.dumps({"error": "No catalog loaded."})

            # Parse max_results if it's a string
            if isinstance(max_results, str):
                try:
                    max_results = int(max_results)
                except ValueError:
                    max_results = 10

            # Determine search bbox
            search_bbox = None

            if bbox is not None:
                # Parse bbox if it's a string
                if isinstance(bbox, str):
                    search_bbox = self._parse_bbox_string(bbox)
                    if search_bbox is None:
                        return json.dumps({"error": f"Invalid bbox format: {bbox}"})
                else:
                    search_bbox = bbox

            elif location is not None:
                # Geocode location to bbox
                geocode_result = json.loads(self.geocode_location(location))
                if "error" in geocode_result:
                    return json.dumps(geocode_result)
                search_bbox = geocode_result["bbox"]

            if search_bbox is None:
                return json.dumps(
                    {"error": "Either bbox or location must be provided."}
                )

            # Validate search bbox
            if len(search_bbox) != 4:
                return json.dumps(
                    {
                        "error": "Bbox must have 4 values [minLon, minLat, maxLon, maxLat]"
                    }
                )

            # Filter by spatial intersection
            if "bbox" not in self._catalog_df.columns:
                return json.dumps(
                    {
                        "error": "Catalog does not have 'bbox' column. Try using a JSON format catalog."
                    }
                )

            # Create mask for spatial intersection
            spatial_mask = pd.Series(
                [False] * len(self._catalog_df), index=self._catalog_df.index
            )

            for idx, row in self._catalog_df.iterrows():
                dataset_bbox = self._parse_bbox_string(row.get("bbox"))
                if dataset_bbox and self._bbox_intersects(dataset_bbox, search_bbox):
                    spatial_mask[idx] = True

            result_df = self._catalog_df[spatial_mask]

            # Apply additional filters using existing _search_dataframe logic
            result_df = self._search_dataframe(
                result_df,
                keywords=keywords,
                dataset_type=dataset_type,
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
            )

            # Convert to models
            dataset_models = []
            for _, row in result_df.iterrows():
                dataset_models.append(
                    CatalogDatasetInfo(
                        id=str(row.get("id", "")),
                        title=str(row.get("title", "")),
                        type=(
                            str(row.get("type", ""))
                            if pd.notna(row.get("type"))
                            else None
                        ),
                        provider=(
                            str(row.get("provider", ""))
                            if pd.notna(row.get("provider"))
                            else None
                        ),
                        description=(
                            str(row.get("description", ""))
                            if pd.notna(row.get("description"))
                            else None
                        ),
                        keywords=(
                            str(row.get("keywords", ""))
                            if pd.notna(row.get("keywords"))
                            else None
                        ),
                        snippet=(
                            str(row.get("snippet", ""))
                            if pd.notna(row.get("snippet"))
                            else None
                        ),
                        start_date=(
                            str(row.get("start_date", ""))
                            if pd.notna(row.get("start_date"))
                            else None
                        ),
                        end_date=(
                            str(row.get("end_date", ""))
                            if pd.notna(row.get("end_date"))
                            else None
                        ),
                        bbox=(
                            str(row.get("bbox", ""))
                            if pd.notna(row.get("bbox"))
                            else None
                        ),
                        license=(
                            str(row.get("license", ""))
                            if pd.notna(row.get("license"))
                            else None
                        ),
                        url=(
                            str(row.get("url", ""))
                            if pd.notna(row.get("url"))
                            else None
                        ),
                        catalog=(
                            str(row.get("catalog", ""))
                            if pd.notna(row.get("catalog"))
                            else None
                        ),
                        deprecated=(
                            str(row.get("deprecated", ""))
                            if pd.notna(row.get("deprecated"))
                            else None
                        ),
                    )
                )

            # Create search result
            filters = {"search_bbox": search_bbox}
            if keywords:
                filters["keywords"] = keywords
            if dataset_type:
                filters["dataset_type"] = dataset_type
            if provider:
                filters["provider"] = provider

            query_parts = []
            if location:
                query_parts.append(f"location: {location}")
            elif bbox:
                query_parts.append(f"bbox: {search_bbox}")
            if keywords:
                query_parts.append(f"keywords: {keywords}")
            if dataset_type:
                query_parts.append(f"type: {dataset_type}")
            if provider:
                query_parts.append(f"provider: {provider}")
            query_str = ", ".join(query_parts) if query_parts else "spatial search"

            result = CatalogSearchResult(
                query=query_str,
                dataset_count=len(dataset_models),
                datasets=dataset_models,
                filters=filters,
            )

            return json.dumps(result.model_dump(), indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})
