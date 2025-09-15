from typing import List, Optional, Union

import leafmap.maplibregl as leafmap
from strands import tool


class MapSession:
    """Manages a leafmap session with map instance."""

    def __init__(self, m: Optional[leafmap.Map] = None) -> None:
        """Initialize map session.

        Args:
            m: Optional existing map instance. If None, creates a default map.
        """
        # allow user to pass a map, otherwise create a default
        self.m: leafmap.Map = m or leafmap.Map(style="liberty", projection="globe")


class MapTools:
    """Collection of tools for interacting with leafmap instances."""

    def __init__(self, session: Optional[MapSession] = None) -> None:
        """Initialize map tools.

        Args:
            session: Optional MapSession instance. If None, creates a default session.
        """
        self.session: MapSession = session or MapSession()

    @tool(
        description="Create or reset a Leafmap map with optional center/zoom and basemap."
    )
    def create_map(
        self,
        center_lat: float = 20.0,
        center_lon: float = 0.0,
        zoom: int = 2,
        style: str = "liberty",
        projection: str = "globe",
        use_message_queue: bool = True,
    ) -> str:
        """Create or reset a Leafmap map with specified parameters.

        Args:
            center_lat: Latitude for map center (default: 20.0).
            center_lon: Longitude for map center (default: 0.0).
            zoom: Initial zoom level (default: 2).
            style: Map style name (default: "liberty").
            projection: Map projection (default: "globe").
            use_message_queue: Whether to use message queue (default: True).

        Returns:
            Confirmation message.
        """
        self.session.m = leafmap.Map(
            center=[center_lon, center_lat],
            zoom=zoom,
            style=style,
            projection=projection,
            use_message_queue=use_message_queue,
        )
        self.session.m.create_container()
        return "Map created."

    @tool(description="Add a basemap by name")
    def add_basemap(self, name: str) -> str:
        """Add a basemap to the map by name.

        Args:
            name: Name of the basemap to add.

        Returns:
            Confirmation message with basemap name.
        """
        self.session.m.add_basemap(name)
        return f"Basemap added: {name}"

    @tool(description="Add a vector dataset (GeoJSON, Shapefile, etc.)")
    def add_vector(self, data: str, name: Optional[str] = None) -> str:
        """Add a vector dataset to the map.

        Args:
            data: Path or URL to the vector data file.
            name: Optional name for the layer.

        Returns:
            Confirmation message with layer name.
        """
        self.session.m.add_vector(data=data, name=name)
        return f"Vector added: {name}"

    @tool(description="Fly to a specific location")
    def fly_to(self, longitude: float, latitude: float, zoom: int = 12) -> str:
        """Fly to a specific geographic location.

        Args:
            longitude: Target longitude coordinate.
            latitude: Target latitude coordinate.
            zoom: Zoom level for the target location (default: 12).

        Returns:
            Confirmation message with coordinates and zoom level.
        """
        self.session.m.fly_to(longitude, latitude, zoom)
        return f"Flown to: {longitude}, {latitude}, zoom {zoom}"

    @tool(description="Add Cloud Optimized GeoTIFF (COG) to the map")
    def add_cog_layer(
        self,
        url: str,
        name: Optional[str] = None,
        attribution: str = "TiTiler",
        opacity: float = 1.0,
        visible: bool = True,
        bands: Optional[List[int]] = None,
        nodata: Optional[Union[int, float]] = 0,
        titiler_endpoint: str = "https://titiler.xyz",
    ) -> str:
        """Add a Cloud Optimized GeoTIFF (COG) layer to the map.

        Args:
            url: URL to the COG file.
            name: Optional name for the layer.
            attribution: Attribution text (default: "TiTiler").
            opacity: Layer opacity from 0.0 to 1.0 (default: 1.0).
            visible: Whether the layer is initially visible (default: True).
            bands: Optional list of band indices to display.
            nodata: No data value (default: 0).
            titiler_endpoint: TiTiler endpoint URL (default: "https://titiler.xyz").

        Returns:
            Confirmation message with COG URL.
        """
        self.session.m.add_cog_layer(
            url, name, attribution, opacity, visible, bands, nodata, titiler_endpoint
        )
        return f"COG layer added: {url}"

    @tool(description="Remove a layer by name")
    def remove_layer(self, name: str) -> str:
        """Remove a layer from the map by name.

        Args:
            name: Name of the layer to remove.

        Returns:
            Confirmation message with removed layer name.
        """
        self.session.m.remove_layer(name)
        return f"Removed: {name}"
