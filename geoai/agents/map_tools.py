from typing import Any, Dict, List, Optional, Union

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
        titiler_endpoint: str = None,
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
            titiler_endpoint: TiTiler endpoint URL (default: "https://giswqs-titiler-endpoint.hf.space").

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
        layer_names = self.session.m.get_layer_names()
        if name in layer_names:
            self.session.m.remove_layer(name)
            return f"Removed: {name}"
        else:
            for layer_name in layer_names:
                if name.lower() in layer_name.lower():
                    self.session.m.remove_layer(layer_name)
                    return f"Removed: {layer_name}"
            return f"Layer {name} not found"

    @tool(description="Add 3D buildings from Overture Maps to the map")
    def add_overture_3d_buildings(
        self,
        release: Optional[str] = None,
        style: Optional[Dict[str, Any]] = None,
        values: Optional[List[int]] = None,
        colors: Optional[List[str]] = None,
        visible: bool = True,
        opacity: float = 1.0,
        tooltip: bool = True,
        template: str = "simple",
        fit_bounds: bool = False,
        **kwargs: Any,
    ) -> None:
        """Add 3D buildings from Overture Maps to the map.

        Args:
            release (Optional[str], optional): The release date of the Overture Maps data.
                Defaults to the latest release. For more info, see
                https://github.com/OvertureMaps/overture-tiles.
            style (Optional[Dict[str, Any]], optional): The style dictionary for
                the buildings. Defaults to None.
            values (Optional[List[int]], optional): List of height values for
                color interpolation. Defaults to None.
            colors (Optional[List[str]], optional): List of colors corresponding
                to the height values. Defaults to None.
            visible (bool, optional): Whether the buildings layer is visible.
                Defaults to True.
            opacity (float, optional): The opacity of the buildings layer.
                Defaults to 1.0.
            tooltip (bool, optional): Whether to show tooltips on the buildings.
                Defaults to True.
            template (str, optional): The template for the tooltip. It can be
                "simple" or "all". Defaults to "simple".
            fit_bounds (bool, optional): Whether to fit the map bounds to the
                buildings layer. Defaults to False.

        Raises:
            ValueError: If the length of values and colors lists are not the same.
        """
        self.session.m.add_overture_3d_buildings(
            release=release,
            style=style,
            values=values,
            colors=colors,
            visible=visible,
            opacity=opacity,
            tooltip=tooltip,
            template=template,
            fit_bounds=fit_bounds,
            **kwargs,
        )
        return f"Overture 3D buildings added: {release}"

    @tool(description="Set the pitch of the map")
    def set_pitch(self, pitch: float) -> None:
        """
        Sets the pitch of the map.

        This function sets the pitch of the map to the specified value. The pitch is the
        angle of the camera measured in degrees where 0 is looking straight down, and 60 is
        looking towards the horizon. Additional keyword arguments can be provided to control
        the pitch. For more information, see https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/#setpitch

        Args:
            pitch (float): The pitch value to set.

        Returns:
            None
        """
        self.session.m.set_pitch(pitch=pitch)
        return f"Map pitched to: {pitch}"

    @tool
    def add_draw_control(
        self,
        options: Optional[Dict[str, Any]] = None,
        controls: Optional[Dict[str, Any]] = None,
        position: str = "top-right",
        geojson: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Adds a drawing control to the map.

        This method enables users to add interactive drawing controls to the map,
        allowing for the creation, editing, and deletion of geometric shapes on
        the map. The options, position, and initial GeoJSON can be customized.

        Args:
            options (Optional[Dict[str, Any]]): Configuration options for the
                drawing control. Defaults to None.
            controls (Optional[Dict[str, Any]]): The drawing controls to enable.
                Can be one or more of the following: 'polygon', 'line_string',
                'point', 'trash', 'combine_features', 'uncombine_features'.
                Defaults to None.
            position (str): The position of the control on the map. Defaults
                to "top-right".
            geojson (Optional[Dict[str, Any]]): Initial GeoJSON data to load
                into the drawing control. Defaults to None.
            **kwargs (Any): Additional keyword arguments to be passed to the
                drawing control.

        Returns:
            None
        """
        self.session.m.add_draw_control(
            options=options,
            controls=controls,
            position=position,
            geojson=geojson,
            **kwargs,
        )
        return f"Draw control added: {position}"

    @tool
    def add_vector_tile(
        self,
        url: str,
        layer_id: str,
        layer_type: str = "fill",
        source_layer: Optional[str] = None,
        name: Optional[str] = None,
        paint: Optional[Dict] = None,
        layout: Optional[Dict] = None,
        filter: Optional[Dict] = None,
        minzoom: Optional[int] = None,
        maxzoom: Optional[int] = None,
        visible: bool = True,
        opacity: float = 1.0,
        add_popup: bool = True,
        before_id: Optional[str] = None,
        source_args: Dict = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Adds a vector tile layer to the map.

        This method adds a vector tile layer to the map using a vector tile source.
        Vector tiles are a data format for efficiently storing and transmitting
        vector map data.

        Args:
            url (str): The URL template for the vector tiles. Should contain {z}, {x},
                and {y} placeholders for tile coordinates.
            layer_id (str): The ID of the layer within the vector tile source.
            layer_type (str, optional): The type of layer to create. Can be 'fill',
                'line', 'symbol', 'circle', etc. Defaults to 'fill'.
            source_layer (str, optional): The name of the source layer within the
                vector tiles. If None, uses layer_id.
            name (str, optional): The name to use for the layer. If None, uses layer_id.
            paint (dict, optional): Paint properties for the layer. If None, uses
                default styling based on layer_type.
            layout (dict, optional): Layout properties for the layer.
            filter (dict, optional): Filter expression for the layer.
            minzoom (int, optional): Minimum zoom level for the layer.
            maxzoom (int, optional): Maximum zoom level for the layer.
            visible (bool, optional): Whether the layer should be visible by default.
                Defaults to True.
            opacity (float, optional): The opacity of the layer. Defaults to 1.0.
            add_popup (bool, optional): Whether to add a popup to the layer. Defaults to True.
            before_id (str, optional): The ID of an existing layer before which the
                new layer should be inserted.
            source_args (dict, optional): Additional keyword arguments passed to the
                vector tile source.
            overwrite (bool, optional): Whether to overwrite an existing layer with
                the same name. Defaults to False.
            **kwargs: Additional keyword arguments passed to the Layer class.

        Returns:
            None

        Example:
            >>> m = Map()
            >>> m.add_vector_tile(
            ...     url="https://api.maptiler.com/tiles/contours/tiles.json?key={api_key}",
            ...     layer_id="contour-lines",
            ...     layer_type="line",
            ...     source_layer="contour",
            ...     paint={"line-color": "#ff69b4", "line-width": 1}
            ... )
        """
        self.session.m.add_vector_tile(
            url=url,
            layer_id=layer_id,
            layer_type=layer_type,
            source_layer=source_layer,
            name=name,
            paint=paint,
            layout=layout,
            filter=filter,
            minzoom=minzoom,
            maxzoom=maxzoom,
            visible=visible,
            opacity=opacity,
            add_popup=add_popup,
            before_id=before_id,
            source_args=source_args,
            overwrite=overwrite,
            **kwargs,
        )
        return f"Vector tile layer added: {url}"

    @tool
    def add_wms_layer(
        self,
        url: str,
        layers: str,
        format: str = "image/png",
        name: str = "WMS Layer",
        attribution: str = "",
        opacity: float = 1.0,
        visible: bool = True,
        tile_size: int = 256,
        before_id: Optional[str] = None,
        source_args: Dict = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Adds a WMS layer to the map.

        This method adds a WMS layer to the map. The WMS  is created from
            the specified URL, and it is added to the map with the specified
            name, attribution, visibility, and tile size.

        Args:
            url (str): The URL of the tile layer.
            layers (str): The layers to include in the WMS request.
            format (str, optional): The format of the tiles in the layer.
            name (str, optional): The name to use for the layer. Defaults to
                'WMS Layer'.
            attribution (str, optional): The attribution to use for the layer.
                Defaults to ''.
            visible (bool, optional): Whether the layer should be visible by
                default. Defaults to True.
            tile_size (int, optional): The size of the tiles in the layer.
                Defaults to 256.
            before_id (str, optional): The ID of an existing layer before which
                the new layer should be inserted.
            source_args (dict, optional): Additional keyword arguments that are
                passed to the RasterTileSource class.
            overwrite (bool, optional): Whether to overwrite an existing layer with the same name.
                Defaults to False.
            **kwargs: Additional keyword arguments that are passed to the Layer class.
                See https://eodagmbh.github.io/py-maplibregl/api/layer/ for more information.

        Returns:
            None
        """
        self.session.m.add_wms_layer(
            url=url,
            layers=layers,
            format=format,
            name=name,
            attribution=attribution,
            opacity=opacity,
            visible=visible,
            tile_size=tile_size,
            before_id=before_id,
            source_args=source_args,
            overwrite=overwrite,
            **kwargs,
        )
        return f"WMS layer added: {url}"

    @tool
    def add_nwi_basemap(
        self,
        name: str = "NWI Wetlands",
        format: str = "image/png",
        attribution: str = "USFWS",
        opacity: float = 1.0,
        visible: bool = True,
        tile_size: int = 256,
        before_id: Optional[str] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Adds a NWI Wetlands basemap to the map.

        This method adds a NWI Wetlands basemap to the map. The NWI Wetlands basemap is created from
            the specified URL, and it is added to the map with the specified
            name, attribution, visibility, and tile size.

        Args:
            name (str, optional): The name to use for the layer. Defaults to
                'NWI Wetlands'.
            format (str, optional): The format of the tiles in the layer.
            attribution (str, optional): The attribution to use for the layer.
                Defaults to ''.
            visible (bool, optional): Whether the layer should be visible by
                default. Defaults to True.
            tile_size (int, optional): The size of the tiles in the layer.
                Defaults to 256.
            before_id (str, optional): The ID of an existing layer before which
                the new layer should be inserted.
            overwrite (bool, optional): Whether to overwrite an existing layer with the same name.
                Defaults to False.
            **kwargs: Additional keyword arguments that are passed to the Layer class.
                See https://eodagmbh.github.io/py-maplibregl/api/layer/ for more information.

        Returns:
            None
        """
        self.session.m.add_nwi_basemap(
            name=name,
            format=format,
            attribution=attribution,
            opacity=opacity,
            visible=visible,
            tile_size=tile_size,
            before_id=before_id,
            overwrite=overwrite,
            **kwargs,
        )
        return f"NWI Wetlands basemap added: {name}"

    @tool
    def add_raster(
        self,
        source,
        indexes=None,
        colormap=None,
        vmin=None,
        vmax=None,
        nodata=None,
        name="Raster",
        before_id=None,
        fit_bounds=True,
        visible=True,
        opacity=1.0,
        array_args=None,
        client_args={"cors_all": True},
        overwrite: bool = True,
        **kwargs: Any,
    ):
        """Add a local raster dataset to the map.
            If you are using this function in JupyterHub on a remote server
            (e.g., Binder, Microsoft Planetary Computer) and if the raster
            does not render properly, try installing jupyter-server-proxy using
            `pip install jupyter-server-proxy`, then running the following code
            before calling this function. For more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band.
                See https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            opacity (float, optional): The opacity of the layer. Defaults to 1.0.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            client_args (dict, optional): Additional arguments to pass to
                localtileserver.TileClient. Defaults to { "cors_all": False }.
            overwrite (bool, optional): Whether to overwrite an existing layer with the same name.
                Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the underlying
                `add_tile_layer` method.
        """
        self.session.m.add_raster(
            source=source,
            indexes=indexes,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            name=name,
            before_id=before_id,
            fit_bounds=fit_bounds,
            visible=visible,
            opacity=opacity,
            array_args=array_args,
            client_args=client_args,
            overwrite=overwrite,
            **kwargs,
        )
        return f"Raster added: {source}"

    @tool
    def save_map(
        self,
        output: str = "map.html",
        title: str = "My Awesome Map",
        width: str = "100%",
        height: str = "100%",
        replace_key: bool = False,
        remove_port: bool = True,
        preview: bool = False,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        """Render the map to an HTML page.

        Args:
            output (str, optional): The output HTML file. If None, the HTML content
                is returned as a string. Defaults to 'map.html'.
            title (str, optional): The title of the HTML page. Defaults to 'My Awesome Map'.
            width (str, optional): The width of the map. Defaults to '100%'.
            height (str, optional): The height of the map. Defaults to '100%'.
            replace_key (bool, optional): Whether to replace the API key in the HTML.
                If True, the API key is replaced with the public API key.
                The API key is read from the environment variable `MAPTILER_KEY`.
                The public API key is read from the environment variable `MAPTILER_KEY_PUBLIC`.
                Defaults to False.
            remove_port (bool, optional): Whether to remove the port number from the HTML.
            preview (bool, optional): Whether to preview the HTML file in a web browser.
                Defaults to False.
            overwrite (bool, optional): Whether to overwrite the output file if it already exists.
            **kwargs: Additional keyword arguments that are passed to the
                `maplibre.ipywidget.MapWidget.to_html()` method.

        Returns:
            str: The HTML content of the map.
        """
        self.session.m.to_html(
            output=output,
            title=title,
            width=width,
            height=height,
            replace_key=replace_key,
            remove_port=remove_port,
            preview=preview,
            overwrite=overwrite,
            **kwargs,
        )
        return f"HTML file created: {output}"

    @tool
    def set_paint_property(self, name: str, prop: str, value: Any) -> None:
        """
        Set the paint property of a layer.

        This method sets the opacity of the specified layer to the specified value.

        Args:
            name (str): The name of the layer.
            prop (str): The paint property to set.
            value (Any): The value to set.

        Returns:
            None
        """
        self.session.m.set_paint_property(name=name, prop=prop, value=value)
        return f"Paint property set: {name}, {prop}, {value}"

    @tool
    def set_layout_property(self, name: str, prop: str, value: Any) -> None:
        """
        Set the layout property of a layer.

        This method sets the layout property of the specified layer to the specified value.

        Args:
            name (str): The name of the layer.
            prop (str): The layout property to set.
            value (Any): The value to set.

        Returns:
            None
        """
        self.session.m.set_layout_property(name=name, prop=prop, value=value)
        return f"Layout property set: {name}, {prop}, {value}"

    @tool
    def set_color(self, name: str, color: str) -> None:
        """
        Set the color of a layer.

        This method sets the color of the specified layer to the specified value.

        Args:
            name (str): The name of the layer.
            color (str): The color value to set.

        Returns:
            None
        """
        self.session.m.set_color(name=name, color=color)
        return f"Color set: {name}, {color}"

    @tool
    def set_opacity(self, name: str, opacity: float) -> None:
        """
        Set the opacity of a layer.

        This method sets the opacity of the specified layer to the specified value.

        Args:
            name (str): The name of the layer.
            opacity (float): The opacity value to set.

        Returns:
            None
        """
        self.session.m.set_opacity(name=name, opacity=opacity)
        return f"Opacity set: {name}, {opacity}"

    @tool
    def set_visibility(self, name: str, visible: bool) -> None:
        """
        Set the visibility of a layer.

        This method sets the visibility of the specified layer to the specified value.

        Args:
            name (str): The name of the layer.
            visible (bool): The visibility value to set.

        Returns:
            None
        """
        self.session.m.set_visibility(name=name, visible=visible)
        return f"Visibility set: {name}, {visible}"

    @tool
    def add_pmtiles(
        self,
        url: str,
        style: Optional[Dict] = None,
        visible: bool = True,
        opacity: float = 1.0,
        exclude_mask: bool = False,
        tooltip: bool = True,
        properties: Optional[Dict] = None,
        template: Optional[str] = None,
        attribution: str = "PMTiles",
        fit_bounds: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Adds a PMTiles layer to the map.

        Args:
            url (str): The URL of the PMTiles file.
            style (dict, optional): The CSS style to apply to the layer. Defaults to None.
                See https://docs.mapbox.com/style-spec/reference/layers/ for more info.
            visible (bool, optional): Whether the layer should be shown initially. Defaults to True.
            opacity (float, optional): The opacity of the layer. Defaults to 1.0.
            exclude_mask (bool, optional): Whether to exclude the mask layer. Defaults to False.
            tooltip (bool, optional): Whether to show tooltips on the layer. Defaults to True.
            properties (dict, optional): The properties to use for the tooltips. Defaults to None.
            template (str, optional): The template to use for the tooltips. Defaults to None.
            attribution (str, optional): The attribution to use for the layer. Defaults to 'PMTiles'.
            fit_bounds (bool, optional): Whether to zoom to the layer extent. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the PMTilesLayer constructor.

        Returns:
            None
        """
        self.session.m.add_pmtiles(
            url=url,
            style=style,
            visible=visible,
            opacity=opacity,
            exclude_mask=exclude_mask,
            tooltip=tooltip,
            properties=properties,
            template=template,
            attribution=attribution,
            fit_bounds=fit_bounds,
            **kwargs,
        )
        return f"PMTiles layer added: {url}"

    @tool
    def add_marker(
        self,
        lng_lat: List[Union[float, float]],
        popup: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> None:
        """
        Adds a marker to the map.

        Args:
            lng_lat (List[Union[float, float]]): A list of two floats
                representing the longitude and latitude of the marker.
            popup (Optional[str], optional): The text to display in a popup when
                the marker is clicked. Defaults to None.
            options (Optional[Dict], optional): A dictionary of options to
                customize the marker. Defaults to None.

        Returns:
            None
        """
        self.session.m.add_marker(lng_lat=lng_lat, popup=popup, options=options)
        return f"Marker added: {lng_lat}"

    @tool
    def add_image(
        self,
        id: str = None,
        image: Union[str, Dict] = None,
        width: int = None,
        height: int = None,
        coordinates: List[float] = None,
        position: str = None,
        icon_size: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Add an image to the map.

        Args:
            id (str): The layer ID of the image.
            image (Union[str, Dict, np.ndarray]): The URL or local file path to
                the image, or a dictionary containing image data, or a numpy
                array representing the image.
            width (int, optional): The width of the image. Defaults to None.
            height (int, optional): The height of the image. Defaults to None.
            coordinates (List[float], optional): The longitude and latitude
                coordinates to place the image.
            position (str, optional): The position of the image. Defaults to None.
                Can be one of 'top-right', 'top-left', 'bottom-right', 'bottom-left'.
            icon_size (float, optional): The size of the icon. Defaults to 1.0.

        Returns:
            None
        """
        self.session.m.add_image(
            id=id,
            image=image,
            width=width,
            height=height,
            coordinates=coordinates,
            position=position,
            icon_size=icon_size,
            **kwargs,
        )
        return f"Image added: {id}"

    @tool
    def rotate_to(
        self, bearing: float, options: Dict[str, Any] = {}, **kwargs: Any
    ) -> None:
        """
        Rotate the map to a specified bearing.

        This function rotates the map to a specified bearing. The bearing is specified in degrees
        counter-clockwise from true north. If the bearing is not specified, the map will rotate to
        true north. Additional options and keyword arguments can be provided to control the rotation.
        For more information, see https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/#rotateto

        Args:
            bearing (float): The bearing to rotate to, in degrees counter-clockwise from true north.
            options (Dict[str, Any], optional): Additional options to control the rotation. Defaults to {}.
            **kwargs (Any): Additional keyword arguments to control the rotation.

        Returns:
            None
        """
        self.session.m.rotate_to(bearing=bearing, options=options, **kwargs)
        return f"Map rotated to: {bearing}"

    @tool
    def pan_to(
        self,
        lnglat: List[float],
        options: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """
        Pans the map to a specified location.

        This function pans the map to the specified longitude and latitude coordinates.
        Additional options and keyword arguments can be provided to control the panning.
        For more information, see https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/#panto

        Args:
            lnglat (List[float, float]): The longitude and latitude coordinates to pan to.
            options (Dict[str, Any], optional): Additional options to control the panning. Defaults to {}.
            **kwargs (Any): Additional keyword arguments to control the panning.

        Returns:
            None
        """
        self.session.m.pan_to(lnglat=lnglat, options=options, **kwargs)
        return f"Map panned to: {lnglat}"

    @tool
    def jump_to(self, options: Dict[str, Any] = {}, **kwargs: Any) -> None:
        """
        Jumps the map to a specified location.

        This function jumps the map to the specified location with the specified options.
        Additional keyword arguments can be provided to control the jump. For more information,
        see https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/#jumpto

        Args:
            options (Dict[str, Any], optional): Additional options to control the jump. Defaults to {}.
            **kwargs (Any): Additional keyword arguments to control the jump.

        Returns:
            None
        """
        self.session.m.jump_to(options=options, **kwargs)
        return f"Map jumped to: {options}"

    @tool
    def zoom_to(self, zoom: float, options: Dict[str, Any] = {}) -> None:
        """
        Zooms the map to a specified zoom level.

        This function zooms the map to the specified zoom level. Additional options and keyword
        arguments can be provided to control the zoom. For more information, see
        https://maplibre.org/maplibre-gl-js/docs/API/classes/Map/#zoomto

        Args:
            zoom (float): The zoom level to zoom to.
            options (Dict[str, Any], optional): Additional options to control the zoom. Defaults to {}.

        Returns:
            None
        """
        self.session.m.zoom_to(zoom=zoom, options=options)
        return f"Map zoomed to: {zoom}"

    @tool
    def first_symbol_layer_id(self) -> Optional[str]:
        """
        Get the ID of the first symbol layer in the map's current style.
        """
        return self.session.m.first_symbol_layer_id

    @tool
    def add_text(
        self,
        text: str,
        fontsize: int = 20,
        fontcolor: str = "black",
        bold: bool = False,
        padding: str = "5px",
        bg_color: str = "white",
        border_radius: str = "5px",
        position: str = "bottom-right",
        **kwargs: Any,
    ) -> None:
        """
        Adds text to the map with customizable styling.

        This method allows adding a text widget to the map with various styling options such as font size, color,
        background color, and more. The text's appearance can be further customized using additional CSS properties
        passed through kwargs.

        Args:
            text (str): The text to add to the map.
            fontsize (int, optional): The font size of the text. Defaults to 20.
            fontcolor (str, optional): The color of the text. Defaults to "black".
            bold (bool, optional): If True, the text will be bold. Defaults to False.
            padding (str, optional): The padding around the text. Defaults to "5px".
            bg_color (str, optional): The background color of the text widget. Defaults to "white".
                To make the background transparent, set this to "transparent".
                To make the background half transparent, set this to "rgba(255, 255, 255, 0.5)".
            border_radius (str, optional): The border radius of the text widget. Defaults to "5px".
            position (str, optional): The position of the text widget on the map. Defaults to "bottom-right".
            **kwargs (Any): Additional CSS properties to apply to the text widget.

        Returns:
            None
        """
        self.session.m.add_text(
            text=text,
            fontsize=fontsize,
            fontcolor=fontcolor,
            bold=bold,
            padding=padding,
            bg_color=bg_color,
            border_radius=border_radius,
            position=position,
            **kwargs,
        )
        return f"Text added: {text}"

    @tool
    def add_html(
        self,
        html: str,
        bg_color: str = "white",
        position: str = "bottom-right",
        **kwargs: Union[str, int, float],
    ) -> None:
        """
        Add HTML content to the map.

        This method allows for the addition of arbitrary HTML content to the map, which can be used to display
        custom information or controls. The background color and position of the HTML content can be customized.

        Args:
            html (str): The HTML content to add.
            bg_color (str, optional): The background color of the HTML content. Defaults to "white".
                To make the background transparent, set this to "transparent".
                To make the background half transparent, set this to "rgba(255, 255, 255, 0.5)".
            position (str, optional): The position of the HTML content on the map. Can be one of "top-left",
                "top-right", "bottom-left", "bottom-right". Defaults to "bottom-right".
            **kwargs: Additional keyword arguments for future use.

        Returns:
            None
        """
        self.session.m.add_html(
            html=html, bg_color=bg_color, position=position, **kwargs
        )
        return f"HTML added: {html}"

    @tool
    def add_legend(
        self,
        title: str = "Legend",
        legend_dict: Optional[Dict[str, str]] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        fontsize: int = 15,
        bg_color: str = "white",
        position: str = "bottom-right",
        builtin_legend: Optional[str] = None,
        shape_type: str = "rectangle",
        **kwargs: Union[str, int, float],
    ) -> None:
        """
        Adds a legend to the map.

        This method allows for the addition of a legend to the map. The legend can be customized with a title,
        labels, colors, and more. A built-in legend can also be specified.

        Args:
            title (str, optional): The title of the legend. Defaults to "Legend".
            legend_dict (Optional[Dict[str, str]], optional): A dictionary with legend items as keys and colors as values.
                If provided, `labels` and `colors` will be ignored. Defaults to None.
            labels (Optional[List[str]], optional): A list of legend labels. Defaults to None.
            colors (Optional[List[str]], optional): A list of colors corresponding to the labels. Defaults to None.
            fontsize (int, optional): The font size of the legend text. Defaults to 15.
            bg_color (str, optional): The background color of the legend. Defaults to "white".
                To make the background transparent, set this to "transparent".
                To make the background half transparent, set this to "rgba(255, 255, 255, 0.5)".
            position (str, optional): The position of the legend on the map. Can be one of "top-left",
                "top-right", "bottom-left", "bottom-right". Defaults to "bottom-right".
            builtin_legend (Optional[str], optional): The name of a built-in legend to use. Defaults to None.
            shape_type (str, optional): The shape type of the legend items. Can be one of "rectangle", "circle", or "line".
            **kwargs: Additional keyword arguments for future use.

        Returns:
            None
        """
        self.session.m.add_legend(
            title=title,
            legend_dict=legend_dict,
            labels=labels,
            colors=colors,
            fontsize=fontsize,
            bg_color=bg_color,
            position=position,
            builtin_legend=builtin_legend,
            shape_type=shape_type,
            **kwargs,
        )
        return f"Legend added: {title}"

    @tool
    def add_colorbar(
        self,
        width: Optional[float] = 3.0,
        height: Optional[float] = 0.2,
        vmin: Optional[float] = 0,
        vmax: Optional[float] = 1.0,
        palette: Optional[List[str]] = None,
        vis_params: Optional[Dict[str, Union[str, float, int]]] = None,
        cmap: Optional[str] = "gray",
        discrete: Optional[bool] = False,
        label: Optional[str] = None,
        label_size: Optional[int] = 10,
        label_weight: Optional[str] = "normal",
        tick_size: Optional[int] = 8,
        bg_color: Optional[str] = "white",
        orientation: Optional[str] = "horizontal",
        dpi: Optional[Union[str, float]] = "figure",
        transparent: Optional[bool] = False,
        position: str = "bottom-right",
        **kwargs: Any,
    ) -> str:
        """
        Add a colorbar to the map.

        This function uses matplotlib to generate a colorbar, saves it as a PNG file, and adds it to the map using
        the Map.add_html() method. The colorbar can be customized in various ways including its size, color palette,
        label, and orientation.

        Args:
            width (Optional[float]): Width of the colorbar in inches. Defaults to 3.0.
            height (Optional[float]): Height of the colorbar in inches. Defaults to 0.2.
            vmin (Optional[float]): Minimum value of the colorbar. Defaults to 0.
            vmax (Optional[float]): Maximum value of the colorbar. Defaults to 1.0.
            palette (Optional[List[str]]): List of colors or a colormap name for the colorbar. Defaults to None.
            vis_params (Optional[Dict[str, Union[str, float, int]]]): Visualization parameters as a dictionary.
            cmap (Optional[str]): Matplotlib colormap name. Defaults to "gray".
            discrete (Optional[bool]): Whether to create a discrete colorbar. Defaults to False.
            label (Optional[str]): Label for the colorbar. Defaults to None.
            label_size (Optional[int]): Font size for the colorbar label. Defaults to 10.
            label_weight (Optional[str]): Font weight for the colorbar label. Defaults to "normal".
            tick_size (Optional[int]): Font size for the colorbar tick labels. Defaults to 8.
            bg_color (Optional[str]): Background color for the colorbar. Defaults to "white".
            orientation (Optional[str]): Orientation of the colorbar ("vertical" or "horizontal"). Defaults to "horizontal".
            dpi (Optional[Union[str, float]]): Resolution in dots per inch. If 'figure', uses the figure's dpi value. Defaults to "figure".
            transparent (Optional[bool]): Whether the background is transparent. Defaults to False.
            position (str): Position of the colorbar on the map. Defaults to "bottom-right".
            **kwargs: Additional keyword arguments passed to matplotlib.pyplot.savefig().

        Returns:
            str: Path to the generated colorbar image.
        """
        self.session.m.add_colorbar(
            width=width,
            height=height,
            vmin=vmin,
            vmax=vmax,
            palette=palette,
            vis_params=vis_params,
            cmap=cmap,
            discrete=discrete,
            label=label,
            label_size=label_size,
            label_weight=label_weight,
            tick_size=tick_size,
            bg_color=bg_color,
            orientation=orientation,
            dpi=dpi,
            transparent=transparent,
            position=position,
            **kwargs,
        )
        return f"Colorbar added: {position}"

    @tool
    def add_layer_control(
        self,
        layer_ids: Optional[List[str]] = None,
        theme: str = "default",
        css_text: Optional[str] = None,
        position: str = "top-left",
        bg_layers: Optional[Union[bool, List[str]]] = False,
    ) -> None:
        """
        Adds a layer control to the map.

        This function creates and adds a layer switcher control to the map, allowing users to toggle the visibility
        of specified layers. The appearance and functionality of the layer control can be customized with parameters
        such as theme, CSS styling, and position on the map.

        Args:
            layer_ids (Optional[List[str]]): A list of layer IDs to include in the control. If None, all layers
                in the map will be included. Defaults to None.
            theme (str): The theme for the layer switcher control. Can be "default" or other custom themes. Defaults to "default".
            css_text (Optional[str]): Custom CSS text for styling the layer control. If None, a default style will be applied.
                Defaults to None.
            position (str): The position of the layer control on the map. Can be "top-left", "top-right", "bottom-left",
                or "bottom-right". Defaults to "top-left".
            bg_layers (bool): If True, background layers will be included in the control. Defaults to False.

        Returns:
            None
        """
        self.session.m.add_layer_control(
            layer_ids=layer_ids,
            theme=theme,
            css_text=css_text,
            position=position,
            bg_layers=bg_layers,
        )
        return f"Layer control added: {position}"

    @tool
    def add_video(
        self,
        urls: Union[str, List[str]],
        coordinates: List[List[float]],
        layer_id: str = "video",
        before_id: Optional[str] = None,
    ) -> None:
        """
        Adds a video layer to the map.

        This method allows embedding a video into the map by specifying the video URLs and the geographical coordinates
        that the video should cover. The video will be stretched and fitted into the specified coordinates.

        Args:
            urls (Union[str, List[str]]): A single video URL or a list of video URLs. These URLs must be accessible
                from the client's location.
            coordinates (List[List[float]]): A list of four coordinates in [longitude, latitude] format, specifying
                the corners of the video. The coordinates order should be top-left, top-right, bottom-right, bottom-left.
            layer_id (str): The ID for the video layer. Defaults to "video".
            before_id (Optional[str]): The ID of an existing layer to insert the new layer before. If None, the layer
                will be added on top. Defaults to None.

        Returns:
            None
        """
        self.session.m.add_video(
            urls=urls,
            coordinates=coordinates,
            layer_id=layer_id,
            before_id=before_id,
        )
        return f"Video added: {layer_id}"

    @tool
    def add_nlcd(
        self, years: list = [2023], add_legend: bool = True, **kwargs: Any
    ) -> None:
        """
        Adds National Land Cover Database (NLCD) data to the map.

        Args:
            years (list): A list of years to add. It can be any of 1985-2023. Defaults to [2023].
            add_legend (bool): Whether to add a legend to the map. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the add_cog_layer method.

        Returns:
            None
        """
        self.session.m.add_nlcd(
            years=years,
            add_legend=add_legend,
            **kwargs,
        )
        return f"NLCD added: {years}"

    @tool
    def add_data(
        self,
        data: Union[str],
        column: str,
        cmap: Optional[str] = None,
        colors: Optional[str] = None,
        labels: Optional[str] = None,
        scheme: Optional[str] = "Quantiles",
        k: int = 5,
        add_legend: Optional[bool] = True,
        legend_title: Optional[bool] = None,
        legend_position: Optional[str] = "bottom-right",
        legend_kwds: Optional[dict] = None,
        classification_kwds: Optional[dict] = None,
        legend_args: Optional[dict] = None,
        layer_type: Optional[str] = None,
        extrude: Optional[bool] = False,
        scale_factor: Optional[float] = 1.0,
        filter: Optional[Dict] = None,
        paint: Optional[Dict] = None,
        name: Optional[str] = None,
        fit_bounds: bool = True,
        visible: bool = True,
        opacity: float = 1.0,
        before_id: Optional[str] = None,
        source_args: Dict = {},
        **kwargs: Any,
    ) -> None:
        """Add vector data to the map with a variety of classification schemes.

        Args:
            data (str | pd.DataFrame | gpd.GeoDataFrame): The data to classify.
                It can be a filepath to a vector dataset, a pandas dataframe, or
                a geopandas geodataframe.
            column (str): The column to classify.
            cmap (str, optional): The name of a colormap recognized by matplotlib. Defaults to None.
            colors (list, optional): A list of colors to use for the classification. Defaults to None.
            labels (list, optional): A list of labels to use for the legend. Defaults to None.
            scheme (str, optional): Name of a choropleth classification scheme (requires mapclassify).
                Name of a choropleth classification scheme (requires mapclassify).
                A mapclassify.MapClassifier object will be used
                under the hood. Supported are all schemes provided by mapclassify (e.g.
                'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
                'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
                'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
                'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
                'UserDefined'). Arguments can be passed in classification_kwds.
            k (int, optional): Number of classes (ignored if scheme is None or if
                column is categorical). Default to 5.
            add_legend (bool, optional): Whether to add a legend to the map. Defaults to True.
            legend_title (str, optional): The title of the legend. Defaults to None.
            legend_position (str, optional): The position of the legend. Can be 'top-left',
                'top-right', 'bottom-left', or 'bottom-right'. Defaults to 'bottom-right'.
            legend_kwds (dict, optional): Keyword arguments to pass to :func:`matplotlib.pyplot.legend`
                or `matplotlib.pyplot.colorbar`. Defaults to None.
                Keyword arguments to pass to :func:`matplotlib.pyplot.legend` or
                Additional accepted keywords when `scheme` is specified:
                fmt : string
                    A formatting specification for the bin edges of the classes in the
                    legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
                labels : list-like
                    A list of legend labels to override the auto-generated labblels.
                    Needs to have the same number of elements as the number of
                    classes (`k`).
                interval : boolean (default False)
                    An option to control brackets from mapclassify legend.
                    If True, open/closed interval brackets are shown in the legend.
            classification_kwds (dict, optional): Keyword arguments to pass to mapclassify.
                Defaults to None.
            legend_args (dict, optional): Additional keyword arguments for the add_legend method. Defaults to None.
            layer_type (str, optional): The type of layer to add. Can be 'circle', 'line', or 'fill'. Defaults to None.
            filter (dict, optional): The filter to apply to the layer. If None,
                no filter is applied.
            paint (dict, optional): The paint properties to apply to the layer.
                If None, no paint properties are applied.
            name (str, optional): The name of the layer. If None, a random name
                is generated.
            fit_bounds (bool, optional): Whether to adjust the viewport of the
                map to fit the bounds of the GeoJSON data. Defaults to True.
            visible (bool, optional): Whether the layer is visible or not.
                Defaults to True.
            before_id (str, optional): The ID of an existing layer before which
                the new layer should be inserted.
            source_args (dict, optional): Additional keyword arguments that are
                passed to the GeoJSONSource class.
            **kwargs: Additional keyword arguments to pass to the GeoJSON class, such as
                fields, which can be a list of column names to be included in the popup.

        """
        self.session.m.add_data(
            data=data,
            column=column,
            cmap=cmap,
            colors=colors,
            labels=labels,
            scheme=scheme,
            k=k,
            add_legend=add_legend,
            legend_title=legend_title,
            legend_position=legend_position,
            legend_kwds=legend_kwds,
            classification_kwds=classification_kwds,
            legend_args=legend_args,
            layer_type=layer_type,
            extrude=extrude,
            scale_factor=scale_factor,
            filter=filter,
            paint=paint,
            name=name,
            fit_bounds=fit_bounds,
            visible=visible,
            opacity=opacity,
            before_id=before_id,
            source_args=source_args,
            **kwargs,
        )
        return f"Data added: {name}"

    @tool
    def add_mapillary(
        self,
        minzoom: int = 6,
        maxzoom: int = 14,
        sequence_lyr_name: str = "sequence",
        image_lyr_name: str = "image",
        before_id: str = None,
        sequence_paint: dict = None,
        image_paint: dict = None,
        image_minzoom: int = 17,
        add_popup: bool = True,
        access_token: str = None,
        opacity: float = 1.0,
        visible: bool = True,
        add_to_sidebar: bool = False,
        style: str = "photo",
        radius: float = 0.00005,
        height: int = 420,
        frame_border: int = 0,
        default_message: str = "No Mapillary image found",
        widget_icon: str = "mdi-image",
        widget_label: str = "Mapillary StreetView",
        **kwargs: Any,
    ) -> None:
        """
        Adds Mapillary layers to the map.

        Args:
            minzoom (int): Minimum zoom level for the Mapillary tiles. Defaults to 6.
            maxzoom (int): Maximum zoom level for the Mapillary tiles. Defaults to 14.
            sequence_lyr_name (str): Name of the sequence layer. Defaults to "sequence".
            image_lyr_name (str): Name of the image layer. Defaults to "image".
            before_id (str): The ID of an existing layer to insert the new layer before. Defaults to None.
            sequence_paint (dict, optional): Paint properties for the sequence layer. Defaults to None.
            image_paint (dict, optional): Paint properties for the image layer. Defaults to None.
            image_minzoom (int): Minimum zoom level for the image layer. Defaults to 17.
            add_popup (bool): Whether to add popups to the layers. Defaults to True.
            access_token (str, optional): Access token for Mapillary API. Defaults to None.
            opacity (float): Opacity of the Mapillary layers. Defaults to 1.0.
            visible (bool): Whether the Mapillary layers are visible. Defaults to True.

        Raises:
            ValueError: If no access token is provided.

        Returns:
            None
        """
        self.session.m.add_mapillary(
            minzoom=minzoom,
            maxzoom=maxzoom,
            sequence_lyr_name=sequence_lyr_name,
            image_lyr_name=image_lyr_name,
            before_id=before_id,
            sequence_paint=sequence_paint,
            image_paint=image_paint,
            image_minzoom=image_minzoom,
            add_popup=add_popup,
            access_token=access_token,
            opacity=opacity,
            visible=visible,
            add_to_sidebar=add_to_sidebar,
            style=style,
            radius=radius,
            height=height,
            frame_border=frame_border,
            default_message=default_message,
            widget_icon=widget_icon,
            widget_label=widget_label,
            **kwargs,
        )
        return f"Mapillary added: {sequence_lyr_name}"

    @tool
    def add_labels(
        self,
        source: Union[str, Dict[str, Any]],
        column: str,
        name: Optional[str] = None,
        text_size: int = 14,
        text_anchor: str = "center",
        text_color: str = "black",
        min_zoom: Optional[float] = None,
        max_zoom: Optional[float] = None,
        layout: Optional[Dict[str, Any]] = None,
        paint: Optional[Dict[str, Any]] = None,
        before_id: Optional[str] = None,
        opacity: float = 1.0,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Adds a label layer to the map.

        This method adds a label layer to the map using the specified source and column for text values.

        Args:
            source (Union[str, Dict[str, Any]]): The data source for the labels. It can be a GeoJSON file path
                or a dictionary containing GeoJSON data.
            column (str): The column name in the source data to use for the label text.
            name (Optional[str]): The name of the label layer. If None, a random name is generated. Defaults to None.
            text_size (int): The size of the label text. Defaults to 14.
            text_anchor (str): The anchor position of the text. Can be "center", "left", "right", etc. Defaults to "center".
            text_color (str): The color of the label text. Defaults to "black".
            min_zoom (Optional[float]): The minimum zoom level at which the labels are visible. Defaults to None.
            max_zoom (Optional[float]): The maximum zoom level at which the labels are visible. Defaults to None.
            layout (Optional[Dict[str, Any]]): Additional layout properties for the label layer. Defaults to None.
                For more information, refer to https://maplibre.org/maplibre-style-spec/layers/#symbol.
            paint (Optional[Dict[str, Any]]): Additional paint properties for the label layer. Defaults to None.
            before_id (Optional[str]): The ID of an existing layer before which the new layer should be inserted. Defaults to None.
            opacity (float): The opacity of the label layer. Defaults to 1.0.
            visible (bool): Whether the label layer is visible by default. Defaults to True.
            **kwargs (Any): Additional keyword arguments to customize the label layer.

        Returns:
            None
        """
        self.session.m.add_labels(
            source=source,
            column=column,
            name=name,
            text_size=text_size,
            text_anchor=text_anchor,
            text_color=text_color,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            layout=layout,
            paint=paint,
            before_id=before_id,
            opacity=opacity,
            visible=visible,
            **kwargs,
        )
        return f"Labels added: {name}"

    @tool
    def get_layer_names(self) -> list:
        """Gets layer names as a list.

        Returns:
            list: A list of layer names.
        """
        return self.session.m.get_layer_names()

    @tool
    def set_terrain(
        self,
        source: str = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png",
        exaggeration: float = 1.0,
        tile_size: int = 256,
        encoding: str = "terrarium",
        source_id: str = "terrain-dem",
    ) -> None:
        """Add terrain visualization to the map.

        Args:
            source: URL template for terrain tiles. Defaults to AWS elevation tiles.
            exaggeration: Terrain exaggeration factor. Defaults to 1.0.
            tile_size: Tile size in pixels. Defaults to 256.
            encoding: Encoding for the terrain tiles. Defaults to "terrarium".
            source_id: Unique identifier for the terrain source. Defaults to "terrain-dem".
        """
        self.session.m.set_terrain(
            source=source,
            exaggeration=exaggeration,
            tile_size=tile_size,
            encoding=encoding,
            source_id=source_id,
        )
        return f"Terrain added: {source}"

    @tool
    def remove_terrain(self) -> None:
        """Remove terrain visualization from the map."""
        self.session.m.remove_terrain()
        return "Terrain removed."
