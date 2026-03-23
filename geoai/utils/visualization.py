"""Visualization utilities for raster and vector data."""

# Standard Library
import json
import logging
import os
from collections.abc import Iterable
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# Third-Party Libraries
import geopandas as gpd
import leafmap
import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.plot import show
from shapely.geometry import box, mapping

# Cross-module imports
from .conversion import dict_to_image

__all__ = [
    "view_raster",
    "view_image",
    "plot_images",
    "plot_masks",
    "plot_batch",
    "view_vector",
    "view_vector_interactive",
    "create_split_map",
    "display_training_tiles",
    "display_image_with_vector",
    "create_overview_image",
    "plot_performance_metrics",
    "plot_prediction_comparison",
]


def view_raster(
    source: str,
    indexes: Optional[int] = None,
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nodata: Optional[float] = None,
    attribution: Optional[str] = None,
    layer_name: Optional[str] = "Raster",
    layer_index: Optional[int] = None,
    zoom_to_layer: Optional[bool] = True,
    visible: Optional[bool] = True,
    opacity: Optional[float] = 1.0,
    array_args: Optional[Dict] = None,
    client_args: Optional[Dict] = {"cors_all": False},
    legend_args: Optional[Dict] = None,
    basemap: Optional[str] = "OpenStreetMap",
    basemap_args: Optional[Dict] = None,
    backend: Optional[str] = "folium",
    **kwargs: Any,
) -> Any:
    """
    Visualize a raster using leafmap.

    Args:
        source (str): The source of the raster.
        indexes (Optional[int], optional): The band indexes to visualize. Defaults to None.
        colormap (Optional[str], optional): The colormap to apply. Defaults to None.
        vmin (Optional[float], optional): The minimum value for colormap scaling. Defaults to None.
        vmax (Optional[float], optional): The maximum value for colormap scaling. Defaults to None.
        nodata (Optional[float], optional): The nodata value. Defaults to None.
        attribution (Optional[str], optional): The attribution for the raster. Defaults to None.
        layer_name (Optional[str], optional): The name of the layer. Defaults to "Raster".
        layer_index (Optional[int], optional): The index of the layer. Defaults to None.
        zoom_to_layer (Optional[bool], optional): Whether to zoom to the layer. Defaults to True.
        visible (Optional[bool], optional): Whether the layer is visible. Defaults to True.
        opacity (Optional[float], optional): The opacity of the layer. Defaults to 1.0.
        array_args (Optional[Dict], optional): Additional arguments for array processing. Defaults to {}.
        client_args (Optional[Dict], optional): Additional arguments for the client. Defaults to {"cors_all": False}.
        legend_args (Optional[Dict], optional): Additional arguments for the legend. Defaults to None.
        basemap (Optional[str], optional): The basemap to use. Defaults to "OpenStreetMap".
        basemap_args (Optional[Dict], optional): Additional arguments for the basemap. Defaults to None.
        backend (Optional[str], optional): The backend to use. Defaults to "folium".
        **kwargs (Any): Additional keyword arguments.

    Returns:
        leafmap.Map: The map object with the raster layer added.
    """

    if backend == "folium":
        import leafmap.foliumap as leafmap
    else:
        import leafmap.leafmap as leafmap

    if basemap_args is None:
        basemap_args = {}

    if array_args is None:
        array_args = {}

    m = leafmap.Map()

    if isinstance(basemap, str):
        if basemap.lower().endswith(".tif"):
            if basemap.lower().startswith("http"):
                if "name" not in basemap_args:
                    basemap_args["name"] = "Basemap"
                m.add_cog_layer(basemap, **basemap_args)
            else:
                if "layer_name" not in basemap_args:
                    basemap_args["layer_name"] = "Basemap"
                m.add_raster(basemap, **basemap_args)
    else:
        m.add_basemap(basemap, **basemap_args)

    if isinstance(source, dict):
        source = dict_to_image(source)

    if isinstance(source, str) and source.startswith("http"):
        if backend == "folium":

            m.add_geotiff(
                url=source,
                name=layer_name,
                opacity=opacity,
                attribution=attribution,
                fit_bounds=zoom_to_layer,
                palette=colormap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            m.add_layer_control()
            m.add_opacity_control()

        else:
            if indexes is not None:
                kwargs["bidx"] = indexes
            if colormap is not None:
                kwargs["colormap_name"] = colormap
            if attribution is None:
                attribution = "TiTiler"

            m.add_cog_layer(
                source,
                name=layer_name,
                opacity=opacity,
                attribution=attribution,
                zoom_to_layer=zoom_to_layer,
                **kwargs,
            )
    else:
        m.add_raster(
            source=source,
            indexes=indexes,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            attribution=attribution,
            layer_name=layer_name,
            layer_index=layer_index,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            opacity=opacity,
            array_args=array_args,
            client_args=client_args,
            **kwargs,
        )
    if isinstance(legend_args, dict):
        m.add_legend(**legend_args)
    return m


def view_image(
    image: Union[np.ndarray, torch.Tensor],
    transpose: bool = False,
    bdx: Optional[int] = None,
    clip_percentiles: Optional[Tuple[float, float]] = (2, 98),
    gamma: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 5),
    axis_off: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Visualize an image using matplotlib.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The image to visualize.
        transpose (bool, optional): Whether to transpose the image. Defaults to False.
        bdx (Optional[int], optional): The band index to visualize. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 5).
        axis_off (bool, optional): Whether to turn off the axis. Defaults to True.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        **kwargs (Any): Additional keyword arguments for plt.imshow().

    Returns:
        None
    """

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    elif isinstance(image, str):
        image = rasterio.open(image).read().transpose(1, 2, 0)

    ax = plt.figure(figsize=figsize)

    if transpose:
        image = image.transpose(1, 2, 0)

    if bdx is not None:
        image = image[:, :, bdx]

    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, 0:3]

    if clip_percentiles is not None:
        p_low, p_high = clip_percentiles
        lower = np.percentile(image, p_low)
        upper = np.percentile(image, p_high)
        image = np.clip((image - lower) / (upper - lower), 0, 1)

    if gamma is not None:
        image = np.power(image, gamma)

    plt.imshow(image, **kwargs)
    if axis_off:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def plot_images(
    images: Iterable[torch.Tensor],
    axs: Iterable[plt.Axes],
    chnls: List[int] = [2, 1, 0],
    bright: float = 1.0,
) -> None:
    """
    Plot a list of images.

    Args:
        images (Iterable[torch.Tensor]): The images to plot.
        axs (Iterable[plt.Axes]): The axes to plot the images on.
        chnls (List[int], optional): The channels to use for RGB. Defaults to [2, 1, 0].
        bright (float, optional): The brightness factor. Defaults to 1.0.

    Returns:
        None
    """
    for img, ax in zip(images, axs):
        arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis("off")


def plot_masks(
    masks: Iterable[torch.Tensor], axs: Iterable[plt.Axes], cmap: str = "Blues"
) -> None:
    """
    Plot a list of masks.

    Args:
        masks (Iterable[torch.Tensor]): The masks to plot.
        axs (Iterable[plt.Axes]): The axes to plot the masks on.
        cmap (str, optional): The colormap to use. Defaults to "Blues".

    Returns:
        None
    """
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy(), cmap=cmap)
        ax.axis("off")


def plot_batch(
    batch: Dict[str, Any],
    bright: float = 1.0,
    cols: int = 4,
    width: int = 5,
    chnls: List[int] = [2, 1, 0],
    cmap: str = "Blues",
) -> None:
    """
    Plot a batch of images and masks. This function is adapted from the plot_batch()
    function in the torchgeo library at
    https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html
    Credit to the torchgeo developers for the original implementation.

    Args:
        batch (Dict[str, Any]): The batch containing images and masks.
        bright (float, optional): The brightness factor. Defaults to 1.0.
        cols (int, optional): The number of columns in the plot grid. Defaults to 4.
        width (int, optional): The width of each plot. Defaults to 5.
        chnls (List[int], optional): The channels to use for RGB. Defaults to [2, 1, 0].
        cmap (str, optional): The colormap to use for masks. Defaults to "Blues".

    Returns:
        None
    """

    try:
        from torchgeo.datasets import unbind_samples
    except ImportError as e:
        raise ImportError(
            "Your torchgeo version is too old. Please upgrade to the latest version using 'pip install -U torchgeo'."
        )

    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())

    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ("image" in batch) and ("mask" in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ("image" in batch) and ("mask" in batch):
        # plot the images on the even axis
        plot_images(
            images=map(lambda x: x["image"], samples),
            axs=axs.reshape(-1)[::2],
            chnls=chnls,
            bright=bright,
        )

        # plot the masks on the odd axis
        plot_masks(masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1)[1::2])

    else:
        if "image" in batch:
            plot_images(
                images=map(lambda x: x["image"], samples),
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )

        elif "mask" in batch:
            plot_masks(
                masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1), cmap=cmap
            )


def view_vector(
    vector_data: Union[str, gpd.GeoDataFrame],
    column: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    legend: bool = True,
    basemap: bool = False,
    basemap_type: str = "streets",
    alpha: float = 0.7,
    edge_color: str = "black",
    classification: str = "quantiles",
    n_classes: int = 5,
    highlight_index: Optional[int] = None,
    highlight_color: str = "red",
    scheme: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    raster_path: Optional[str] = None,
    raster_bands: Optional[Union[int, List[int]]] = None,
    raster_cmap: str = "gray",
    outline_only: bool = False,
    outline_linewidth: float = 1.0,
) -> Any:
    """
    Visualize vector datasets with options for styling, classification, basemaps and more.

    This function visualizes GeoDataFrame objects with customizable symbology.
    It supports different vector types (points, lines, polygons), attribute-based
    classification, background basemaps, and raster backgrounds with polygon
    outlines overlaid.

    Args:
        vector_data (Union[str, geopandas.GeoDataFrame]): The vector dataset to
            visualize. Can be a file path or a GeoDataFrame.
        column (str, optional): Column to use for choropleth mapping. If None,
            a single color will be used. Ignored when outline_only is True.
            Defaults to None.
        cmap (str or matplotlib.colors.Colormap, optional): Colormap to use for
            choropleth mapping. Defaults to "viridis".
        figsize (tuple, optional): Figure size as (width, height) in inches.
            Defaults to (10, 10).
        title (str, optional): Title for the plot. Defaults to None.
        legend (bool, optional): Whether to display a legend. Defaults to True.
        basemap (bool, optional): Whether to add a web basemap. Requires contextily.
            Ignored when raster_path is provided. Defaults to False.
        basemap_type (str, optional): Type of basemap to use. Options: 'streets',
            'satellite'. Defaults to 'streets'.
        alpha (float, optional): Transparency of the vector features, between 0-1.
            Defaults to 0.7.
        edge_color (str, optional): Color for feature edges. Defaults to "black".
        classification (str, optional): Classification method for choropleth maps.
            Options: "quantiles", "equal_interval", "natural_breaks".
            Defaults to "quantiles".
        n_classes (int, optional): Number of classes for choropleth maps.
            Defaults to 5.
        highlight_index (list, optional): List of indices to highlight.
            Defaults to None.
        highlight_color (str, optional): Color to use for highlighted features.
            Defaults to "red".
        scheme (str, optional): MapClassify classification scheme. Overrides
            classification parameter if provided. Defaults to None.
        save_path (str, optional): Path to save the figure. If None, the figure
            is not saved. Defaults to None.
        dpi (int, optional): DPI for saved figure. Defaults to 300.
        raster_path (str, optional): Path to a raster file to display as the
            background. The vector data will be reprojected to match the raster
            CRS if needed. When provided, the basemap option is ignored.
            Defaults to None.
        raster_bands (int or list of int, optional): Band index or list of band
            indices (1-indexed) to display from the raster. If None, all bands
            are shown (RGB if 3 bands). Defaults to None.
        raster_cmap (str, optional): Colormap for single-band raster display.
            Defaults to "gray".
        outline_only (bool, optional): If True, polygon features are drawn as
            outlines only (no fill), allowing the raster background to show
            through. Has no effect on point or line geometries. Defaults to False.
        outline_linewidth (float, optional): Line width for polygon outlines when
            outline_only is True. Defaults to 1.0.

    Returns:
        matplotlib.axes.Axes: The Axes object containing the plot.

    Examples:
        >>> import geopandas as gpd
        >>> cities = gpd.read_file("cities.shp")
        >>> view_vector(cities, "population", cmap="Reds", basemap=True)

        >>> roads = gpd.read_file("roads.shp")
        >>> view_vector(roads, "type", basemap=True, figsize=(12, 8))

        >>> parcels = gpd.read_file("parcels.shp")
        >>> view_vector(
        ...     parcels,
        ...     raster_path="imagery.tif",
        ...     outline_only=True,
        ...     edge_color="yellow",
        ...     outline_linewidth=2,
        ... )
    """
    import contextily as ctx

    if isinstance(vector_data, str):
        vector_data = gpd.read_file(vector_data)

    # Check if input is a GeoDataFrame
    if not isinstance(vector_data, gpd.GeoDataFrame):
        raise TypeError("Input data must be a GeoDataFrame")

    # Make a copy to avoid changing the original data
    gdf = vector_data.copy()

    # Set up figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Display raster as background if provided
    raster_crs = None
    if raster_path is not None:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            show_kwargs: Dict[str, Any] = {"ax": ax}

            if raster_bands is not None:
                show_kwargs["indexes"] = raster_bands
                n_display = 1 if isinstance(raster_bands, int) else len(raster_bands)
            else:
                n_display = src.count

            if n_display == 1:
                show_kwargs["cmap"] = raster_cmap

            show(src, **show_kwargs)

        # Reproject GDF to raster CRS if needed
        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

    # Determine geometry type
    geom_type = gdf.geometry.iloc[0].geom_type

    # Plotting parameters
    plot_kwargs: Dict[str, Any] = {"alpha": alpha, "ax": ax}

    # Set up keyword arguments based on geometry type
    if "Point" in geom_type:
        plot_kwargs["markersize"] = 50
        plot_kwargs["edgecolor"] = edge_color
    elif "Line" in geom_type:
        plot_kwargs["linewidth"] = 1
    elif "Polygon" in geom_type:
        if outline_only:
            plot_kwargs["facecolor"] = "none"
            plot_kwargs["edgecolor"] = edge_color
            plot_kwargs["linewidth"] = outline_linewidth
        else:
            plot_kwargs["edgecolor"] = edge_color

    # Classification options (skipped when outline_only is True)
    if column is not None and not outline_only:
        if scheme is not None:
            # Use mapclassify scheme if provided
            plot_kwargs["scheme"] = scheme
        else:
            # Use classification parameter
            if classification == "quantiles":
                plot_kwargs["scheme"] = "quantiles"
            elif classification == "equal_interval":
                plot_kwargs["scheme"] = "equal_interval"
            elif classification == "natural_breaks":
                plot_kwargs["scheme"] = "fisher_jenks"

        plot_kwargs["k"] = n_classes
        plot_kwargs["cmap"] = cmap
        plot_kwargs["column"] = column
        plot_kwargs["legend"] = legend

    # Plot the main data
    gdf.plot(**plot_kwargs)

    # Highlight specific features if requested
    if highlight_index is not None:
        gdf.iloc[highlight_index].plot(
            ax=ax, color=highlight_color, edgecolor="black", linewidth=2, zorder=5
        )

    if basemap and raster_path is None:
        try:
            basemap_options = {
                "streets": ctx.providers.OpenStreetMap.Mapnik,
                "satellite": ctx.providers.Esri.WorldImagery,
            }
            ctx.add_basemap(ax, crs=gdf.crs, source=basemap_options[basemap_type])
        except Exception as e:
            logger.debug(f"Could not add basemap: {e}")

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)

    # Remove axes if not needed
    ax.set_axis_off()

    # Adjust layout
    plt.tight_layout()

    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return ax


def view_vector_interactive(
    vector_data: Union[str, gpd.GeoDataFrame],
    layer_name: str = "Vector",
    tiles_args: Optional[Dict] = None,
    opacity: float = 0.7,
    **kwargs: Any,
) -> Any:
    """
    Visualize vector datasets with options for styling, classification, basemaps and more.

    This function visualizes GeoDataFrame objects with customizable symbology.
    It supports different vector types (points, lines, polygons), attribute-based
    classification, and background basemaps.

    Args:
        vector_data (geopandas.GeoDataFrame): The vector dataset to visualize.
        layer_name (str, optional): The name of the layer. Defaults to "Vector".
        tiles_args (dict, optional): Additional arguments for the localtileserver client.
            get_folium_tile_layer function. Defaults to None.
        opacity (float, optional): The opacity of the layer. Defaults to 0.7.
        **kwargs: Additional keyword arguments to pass to GeoDataFrame.explore() function.
            See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html

    Returns:
        folium.Map: The map object with the vector data added.

    Examples:
        >>> import geopandas as gpd
        >>> cities = gpd.read_file("cities.shp")
        >>> view_vector_interactive(cities)

        >>> roads = gpd.read_file("roads.shp")
        >>> view_vector_interactive(roads, figsize=(12, 8))
    """

    from leafmap.foliumap import Map
    from localtileserver import TileClient, get_folium_tile_layer

    google_tiles = {
        "Roadmap": {
            "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Maps",
        },
        "Satellite": {
            "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Satellite",
        },
        "Terrain": {
            "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Terrain",
        },
        "Hybrid": {
            "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "attribution": "Google",
            "name": "Google Hybrid",
        },
    }

    # Make it compatible with binder and JupyterHub
    if os.environ.get("JUPYTERHUB_SERVICE_PREFIX") is not None:
        os.environ["LOCALTILESERVER_CLIENT_PREFIX"] = (
            f"{os.environ['JUPYTERHUB_SERVICE_PREFIX'].lstrip('/')}/proxy/{{port}}"
        )

    basemap_layer_name = None
    raster_layer = None

    m = Map()

    if "tiles" in kwargs and isinstance(kwargs["tiles"], str):
        if kwargs["tiles"].title() in google_tiles:
            tile_key = kwargs["tiles"].title()
            basemap_layer_name = google_tiles[tile_key]["name"]
            url = google_tiles[tile_key]["url"]
            m.add_tile_layer(url=url, name=basemap_layer_name, attribution="Google")
            del kwargs["tiles"]
            if "attr" in kwargs:
                del kwargs["attr"]
        elif kwargs["tiles"].lower().endswith(".tif"):
            if tiles_args is None:
                tiles_args = {}
            if kwargs["tiles"].lower().startswith("http"):
                basemap_layer_name = "Remote Raster"
                m.add_geotiff(kwargs["tiles"], name=basemap_layer_name, **tiles_args)
            else:
                basemap_layer_name = "Local Raster"
                client = TileClient(kwargs["tiles"])
                raster_layer = get_folium_tile_layer(client, **tiles_args)
                m.add_tile_layer(
                    raster_layer.tiles,
                    name=basemap_layer_name,
                    attribution="localtileserver",
                    **tiles_args,
                )
            del kwargs["tiles"]
            if "attr" in kwargs:
                del kwargs["attr"]

    if "max_zoom" not in kwargs:
        kwargs["max_zoom"] = 30

    if isinstance(vector_data, str):
        if vector_data.endswith(".parquet"):
            vector_data = gpd.read_parquet(vector_data)
        else:
            vector_data = gpd.read_file(vector_data)

    # Check if input is a GeoDataFrame
    if not isinstance(vector_data, gpd.GeoDataFrame):
        raise TypeError("Input data must be a GeoDataFrame")

    if "column" in kwargs:
        if "legend_position" not in kwargs:
            kwargs["legend_position"] = "bottomleft"
        col = kwargs["column"]
        is_categorical = col in vector_data.columns and (
            pd.api.types.is_object_dtype(vector_data[col])
            or pd.api.types.is_string_dtype(vector_data[col])
            or isinstance(vector_data[col].dtype, pd.CategoricalDtype)
        )
        if is_categorical:
            # For categorical columns, use gdf.explore() directly with
            # categorical=True.  leafmap's add_data applies numeric
            # classification schemes that don't work with string columns.
            # We also build a ListedColormap with exactly N colors so
            # geopandas picks the first N distinct colors instead of
            # sampling at even intervals across a larger palette.
            from matplotlib.colors import ListedColormap

            kwargs["categorical"] = True
            kwargs.pop("legend_position", None)
            cmap_name = kwargs.pop("cmap", "Set1")
            cmap_src = plt.colormaps.get_cmap(cmap_name)
            n_cats = vector_data[col].nunique()
            cat_colors = [cmap_src(i % cmap_src.N) for i in range(n_cats)]
            kwargs["cmap"] = ListedColormap(cat_colors)
            vector_data.explore(
                m=m,
                name=layer_name,
                style_kwds={"opacity": opacity, "fillOpacity": opacity},
                **kwargs,
            )
            # Zoom to data extent (gdf.explore on an existing map
            # does not auto-fit bounds)
            bounds = vector_data.total_bounds  # [minx, miny, maxx, maxy]
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        else:
            if "cmap" not in kwargs:
                kwargs["cmap"] = "viridis"
            m.add_data(vector_data, layer_name=layer_name, opacity=opacity, **kwargs)

    else:
        m.add_gdf(vector_data, layer_name=layer_name, opacity=opacity, **kwargs)

    m.add_layer_control()
    m.add_opacity_control()

    return m


def create_split_map(
    left_layer: Optional[str] = "TERRAIN",
    right_layer: Optional[str] = "OpenTopoMap",
    left_args: Optional[dict] = None,
    right_args: Optional[dict] = None,
    left_array_args: Optional[dict] = None,
    right_array_args: Optional[dict] = None,
    zoom_control: Optional[bool] = True,
    fullscreen_control: Optional[bool] = True,
    layer_control: Optional[bool] = True,
    add_close_button: Optional[bool] = False,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
    left_position: Optional[str] = "bottomleft",
    right_position: Optional[str] = "bottomright",
    widget_layout: Optional[dict] = None,
    draggable: Optional[bool] = True,
    center: Optional[List[float]] = [20, 0],
    zoom: Optional[int] = 2,
    height: Optional[int] = "600px",
    basemap: Optional[str] = None,
    basemap_args: Optional[Dict] = None,
    m: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Adds split map.

    Args:
        left_layer (str, optional): The left tile layer. Can be a local file path, HTTP URL, or a basemap name. Defaults to 'TERRAIN'.
        right_layer (str, optional): The right tile layer. Can be a local file path, HTTP URL, or a basemap name. Defaults to 'OpenTopoMap'.
        left_args (dict, optional): The arguments for the left tile layer. Defaults to {}.
        right_args (dict, optional): The arguments for the right tile layer. Defaults to {}.
        left_array_args (dict, optional): The arguments for array_to_image for the left layer. Defaults to {}.
        right_array_args (dict, optional): The arguments for array_to_image for the right layer. Defaults to {}.
        zoom_control (bool, optional): Whether to add zoom control. Defaults to True.
        fullscreen_control (bool, optional): Whether to add fullscreen control. Defaults to True.
        layer_control (bool, optional): Whether to add layer control. Defaults to True.
        add_close_button (bool, optional): Whether to add a close button. Defaults to False.
        left_label (str, optional): The label for the left layer. Defaults to None.
        right_label (str, optional): The label for the right layer. Defaults to None.
        left_position (str, optional): The position for the left label. Defaults to "bottomleft".
        right_position (str, optional): The position for the right label. Defaults to "bottomright".
        widget_layout (dict, optional): The layout for the widget. Defaults to None.
        draggable (bool, optional): Whether the split map is draggable. Defaults to True.
    """

    if left_args is None:
        left_args = {}

    if right_args is None:
        right_args = {}

    if left_array_args is None:
        left_array_args = {}

    if right_array_args is None:
        right_array_args = {}

    if basemap_args is None:
        basemap_args = {}

    if m is None:
        m = leafmap.Map(center=center, zoom=zoom, height=height, **kwargs)
        m.clear_layers()
    if isinstance(basemap, str):
        if basemap.endswith(".tif"):
            if basemap.startswith("http"):
                m.add_cog_layer(basemap, name="Basemap", **basemap_args)
            else:
                m.add_raster(basemap, layer_name="Basemap", **basemap_args)
        else:
            m.add_basemap(basemap)
    m.split_map(
        left_layer=left_layer,
        right_layer=right_layer,
        left_args=left_args,
        right_args=right_args,
        left_array_args=left_array_args,
        right_array_args=right_array_args,
        zoom_control=zoom_control,
        fullscreen_control=fullscreen_control,
        layer_control=layer_control,
        add_close_button=add_close_button,
        left_label=left_label,
        right_label=right_label,
        left_position=left_position,
        right_position=right_position,
        widget_layout=widget_layout,
        draggable=draggable,
    )

    return m


def display_training_tiles(
    output_dir,
    num_tiles=6,
    figsize=(18, 6),
    cmap="gray",
    show_axes=True,
    save_path=None,
    image_subdir=None,
    mask_subdir=None,
):
    """
    Display image and mask tile pairs from training data output.

    Args:
        output_dir (str): Path to output directory containing image and mask
            subdirectories.
        num_tiles (int): Number of tile pairs to display (default: 6).
        figsize (tuple): Figure size as (width, height) in inches
            (default: (18, 6)).
        cmap (str): Colormap for mask display (default: 'gray').
        show_axes (bool): Whether to show row/column pixel labels on the
            axes. When True, axes display pixel-based row and column
            indices instead of CRS coordinates. Defaults to True.
        save_path (str, optional): If provided, save figure to this path
            instead of displaying.
        image_subdir (str, optional): Name of the subdirectory containing
            image tiles. If None, auto-detects by looking for 'images' or
            'image' subdirectories. Defaults to None.
        mask_subdir (str, optional): Name of the subdirectory containing
            mask/label tiles. If None, auto-detects by looking for 'masks',
            'mask', 'labels', or 'label' subdirectories. Defaults to None.

    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects.

    Example:
        >>> fig, axes = display_training_tiles('output/tiles', num_tiles=6)
        >>> # Show with row/col pixel labels
        >>> fig, axes = display_training_tiles('output/tiles', show_axes=True)
        >>> # Use custom subdirectory names
        >>> fig, axes = display_training_tiles('output/tiles', mask_subdir='labels')
        >>> # Or save to file
        >>> display_training_tiles('output/tiles', num_tiles=4, save_path='tiles_preview.png')
    """
    import matplotlib.pyplot as plt

    # Auto-detect image subdirectory
    if image_subdir is None:
        for candidate in ["images", "image"]:
            if os.path.isdir(os.path.join(output_dir, candidate)):
                image_subdir = candidate
                break
        if image_subdir is None:
            raise ValueError(
                f"Could not find image subdirectory in {output_dir}. "
                "Looked for 'images', 'image'. "
                "Specify image_subdir explicitly."
            )

    # Auto-detect mask subdirectory
    if mask_subdir is None:
        for candidate in ["masks", "mask", "labels", "label"]:
            if os.path.isdir(os.path.join(output_dir, candidate)):
                mask_subdir = candidate
                break
        if mask_subdir is None:
            raise ValueError(
                f"Could not find mask subdirectory in {output_dir}. "
                "Looked for 'masks', 'mask', 'labels', 'label'. "
                "Specify mask_subdir explicitly."
            )

    # Get list of image tiles
    images_dir = os.path.join(output_dir, image_subdir)
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")

    image_tiles = sorted(os.listdir(images_dir))[:num_tiles]

    if not image_tiles:
        raise ValueError(f"No image tiles found in {images_dir}")

    # Limit to available tiles
    num_tiles = min(num_tiles, len(image_tiles))

    # Create figure with subplots
    fig, axes = plt.subplots(2, num_tiles, figsize=figsize)

    # Handle case where num_tiles is 1
    if num_tiles == 1:
        axes = axes.reshape(2, 1)

    masks_dir = os.path.join(output_dir, mask_subdir)
    if not os.path.exists(masks_dir) or not os.path.isdir(masks_dir):
        raise ValueError(f"Mask directory not found: {masks_dir}")

    for idx, tile_name in enumerate(image_tiles):
        # Load and display image tile
        image_path = os.path.join(images_dir, tile_name)
        with rasterio.open(image_path) as src:
            if show_axes:
                data = src.read()
                if data.shape[0] >= 3:
                    axes[0, idx].imshow(data[:3].transpose(1, 2, 0))
                else:
                    axes[0, idx].imshow(data[0])
            else:
                show(src, ax=axes[0, idx])
        axes[0, idx].set_title(f"Image {idx+1}")
        if not show_axes:
            axes[0, idx].set_axis_off()

        # Load and display mask tile
        mask_path = os.path.join(masks_dir, tile_name)
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                if show_axes:
                    data = src.read(1)
                    axes[1, idx].imshow(data, cmap=cmap)
                else:
                    show(src, ax=axes[1, idx], cmap=cmap)
        else:
            axes[1, idx].text(
                0.5,
                0.5,
                "Mask not found",
                ha="center",
                va="center",
                transform=axes[1, idx].transAxes,
            )
        axes[1, idx].set_title(f"Mask {idx+1}")
        if not show_axes:
            axes[1, idx].set_axis_off()

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig, axes


def display_image_with_vector(
    image_path,
    vector_path,
    figsize=(16, 8),
    vector_color="red",
    vector_linewidth=1,
    vector_facecolor="none",
    save_path=None,
):
    """
    Display a raster image alongside the same image with vector overlay.

    Args:
        image_path (str): Path to raster image file
        vector_path (str): Path to vector file (GeoJSON, Shapefile, etc.)
        figsize (tuple): Figure size as (width, height) in inches (default: (16, 8))
        vector_color (str): Edge color for vector features (default: 'red')
        vector_linewidth (float): Line width for vector features (default: 1)
        vector_facecolor (str): Fill color for vector features (default: 'none')
        save_path (str, optional): If provided, save figure to this path instead of displaying

    Returns:
        tuple: (fig, axes, info_dict) where info_dict contains image and vector metadata

    Example:
        >>> fig, axes, info = display_image_with_vector(
        ...     'image.tif',
        ...     'buildings.geojson',
        ...     vector_color='blue'
        ... )
        >>> print(f"Number of features: {info['num_features']}")
    """
    import matplotlib.pyplot as plt

    # Validate inputs
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    if not os.path.exists(vector_path):
        raise ValueError(f"Vector file not found: {vector_path}")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Load and display image
    with rasterio.open(image_path) as src:
        # Plot image only
        show(src, ax=ax1, title="Image")

        # Load vector data
        vector_data = gpd.read_file(vector_path)

        # Reproject to image CRS if needed
        if vector_data.crs != src.crs:
            vector_data = vector_data.to_crs(src.crs)

        # Plot image with vector overlay
        show(
            src,
            ax=ax2,
            title=f"Image with {len(vector_data)} Vector Features",
        )
        vector_data.plot(
            ax=ax2,
            facecolor=vector_facecolor,
            edgecolor=vector_color,
            linewidth=vector_linewidth,
        )

        # Collect metadata
        info = {
            "image_shape": src.shape,
            "image_crs": src.crs,
            "image_bounds": src.bounds,
            "num_features": len(vector_data),
            "vector_crs": vector_data.crs,
            "vector_bounds": vector_data.total_bounds,
        }

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig, (ax1, ax2), info


def create_overview_image(
    src,
    tile_coordinates,
    output_path,
    tile_size,
    stride,
    geojson_path=None,
    in_class_data=None,
) -> str:
    """Create an overview image showing all tiles and their status, with optional GeoJSON export.

    Args:
        src (rasterio.io.DatasetReader): The source raster dataset.
        tile_coordinates (list): A list of dictionaries containing tile information.
        output_path (str): The path where the overview image will be saved.
        tile_size (int): The size of each tile in pixels.
        stride (int): The stride between tiles in pixels. Controls overlap between adjacent tiles.
        geojson_path (str, optional): If provided, exports the tile rectangles as GeoJSON to this path.
        in_class_data (str, optional): Path to classification data (vector or raster).
            If provided, shows the mask/label overlay instead of the raw satellite image.

    Returns:
        str: Path to the saved overview image.
    """
    # Read a reduced version of the source image
    overview_scale = max(
        1, int(max(src.width, src.height) / 2000)
    )  # Scale to max ~2000px
    overview_width = src.width // overview_scale
    overview_height = src.height // overview_scale

    # Try to load mask data if class data is provided
    mask_data = None
    if in_class_data is not None:
        try:
            from rasterio.transform import Affine

            file_ext = os.path.splitext(in_class_data)[1].lower()
            if file_ext in [".tif", ".tiff", ".img", ".jp2", ".png", ".bmp", ".gif"]:
                # Raster class data: read at overview scale
                with rasterio.open(in_class_data) as class_src:
                    mask_data = class_src.read(
                        1,
                        out_shape=(overview_height, overview_width),
                        resampling=rasterio.enums.Resampling.nearest,
                    )
            else:
                # Vector class data: rasterize at overview scale
                import rasterio.features

                gdf = gpd.read_file(in_class_data)
                if gdf.crs != src.crs:
                    gdf = gdf.to_crs(src.crs)

                # Compute overview transform
                overview_transform = Affine(
                    src.transform.a * overview_scale,
                    src.transform.b,
                    src.transform.c,
                    src.transform.d,
                    src.transform.e * overview_scale,
                    src.transform.f,
                )

                shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
                if shapes:
                    mask_data = rasterio.features.rasterize(
                        shapes,
                        out_shape=(overview_height, overview_width),
                        transform=overview_transform,
                        fill=0,
                        dtype=np.uint8,
                    )
        except Exception:
            mask_data = None

    if mask_data is not None:
        # Create colored mask image on dark background
        unique_vals = np.unique(mask_data)
        unique_vals = unique_vals[unique_vals > 0]

        # Use a colormap for class values
        cmap = plt.cm.get_cmap("tab10", max(len(unique_vals), 1))
        rgb = np.zeros((overview_height, overview_width, 3), dtype=np.float64)

        for i, val in enumerate(unique_vals):
            color = cmap(i % 10)[:3]
            mask = mask_data == val
            for c in range(3):
                rgb[mask, c] = color[c]
    else:
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
            rgb = np.stack(
                [overview_data[0], overview_data[0], overview_data[0]], axis=-1
            )

        # Normalize for display
        for i in range(rgb.shape[-1]):
            band = rgb[..., i]
            non_zero = band[band > 0]
            if len(non_zero) > 0:
                p2, p98 = np.percentile(non_zero, (2, 98))
                rgb[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)

        # Apply gamma correction to brighten dark imagery
        rgb = np.clip(np.power(rgb, 0.7), 0, 1)

    # Create figure with aspect-ratio-aware sizing
    max_fig_dim = 12
    aspect = overview_width / overview_height
    if aspect >= 1:
        fig_width = max_fig_dim
        fig_height = max_fig_dim / aspect
    else:
        fig_height = max_fig_dim
        fig_width = max_fig_dim * aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(rgb)

    # If GeoJSON export is requested, prepare GeoJSON structures
    if geojson_path:
        features = []

    # Draw tile boundaries
    for tile in tile_coordinates:
        # Convert bounds to pixel coordinates in overview
        bounds = tile["bounds"]
        # Calculate scaled pixel coordinates
        x_min = int((tile["x"]) / overview_scale)
        y_min = int((tile["y"]) / overview_scale)
        width = int(tile_size / overview_scale)
        height = int(tile_size / overview_scale)

        # Draw rectangle with semi-transparent fill
        if tile["has_features"]:
            edgecolor = "lime"
            facecolor = (0.0, 1.0, 0.0, 0.15)
        else:
            edgecolor = "red"
            facecolor = (1.0, 0.0, 0.0, 0.25)

        rect = plt.Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Add tile number if not too crowded
        if width > 20 and height > 20:
            ax.text(
                x_min + width / 2,
                y_min + height / 2,
                str(tile["index"]),
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
                ],
            )

        # Add to GeoJSON features if exporting
        if geojson_path:
            # Create a polygon from the bounds (already in geo-coordinates)
            minx, miny, maxx, maxy = bounds
            polygon = box(minx, miny, maxx, maxy)

            # Calculate overlap with neighboring tiles
            overlap = 0
            if stride < tile_size:
                overlap = tile_size - stride

            # Create a GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "index": tile["index"],
                    "has_features": tile["has_features"],
                    "bounds_pixel": [
                        tile["x"],
                        tile["y"],
                        tile["x"] + tile_size,
                        tile["y"] + tile_size,
                    ],
                    "tile_size_px": tile_size,
                    "stride_px": stride,
                    "overlap_px": overlap,
                },
            }

            # Add any additional properties from the tile
            for key, value in tile.items():
                if key not in ["x", "y", "index", "has_features", "bounds"]:
                    feature["properties"][key] = value

            features.append(feature)

    ax.set_title("Tile Overview", fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

    # Add legend
    legend_handles = [
        Patch(
            facecolor=(0.0, 1.0, 0.0, 0.3),
            edgecolor="lime",
            linewidth=1.5,
            label="Contains features",
        ),
        Patch(
            facecolor=(1.0, 0.0, 0.0, 0.4),
            edgecolor="red",
            linewidth=1.5,
            label="Empty",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
        framealpha=0.8,
        edgecolor="gray",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Overview image saved to {output_path}")

    # Export GeoJSON if requested
    if geojson_path:
        geojson_collection = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "crs": (
                    src.crs.to_string()
                    if hasattr(src.crs, "to_string")
                    else str(src.crs)
                ),
                "total_tiles": len(features),
                "source_raster_dimensions": [src.width, src.height],
            },
        }

        # Save to file
        with open(geojson_path, "w") as f:
            json.dump(geojson_collection, f)

        logger.info(f"GeoJSON saved to {geojson_path}")

    return output_path


def plot_performance_metrics(
    history_path: str,
    figsize: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Plot performance metrics from a training history object and return as DataFrame.

    This function loads training history, plots available metrics (loss, IoU, F1,
    precision, recall), optionally exports to CSV, and returns all metrics as a
    pandas DataFrame for further analysis.

    Args:
        history_path (str): Path to the saved training history (.pth file).
        figsize (Optional[Tuple[int, int]]): Figure size in inches. If None,
            automatically determined based on number of metrics.
        verbose (bool): Whether to print best and final metric values. Defaults to True.
        save_path (Optional[str]): Path to save the plot image. If None, plot is not saved.
        csv_path (Optional[str]): Path to export metrics as CSV. If None, CSV is not exported.
        kwargs (Optional[Dict]): Additional keyword arguments for plt.savefig().

    Returns:
        pd.DataFrame: DataFrame containing all metrics with columns for epoch and each metric.
            Columns include: 'epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1',
            'val_precision', 'val_recall' (depending on availability in history).

    Example:
        >>> df = plot_performance_metrics(
        ...     'training_history.pth',
        ...     save_path='metrics_plot.png',
        ...     csv_path='metrics.csv'
        ... )
        >>> print(df.head())
    """
    if kwargs is None:
        kwargs = {}
    history = torch.load(history_path)

    # Handle different key naming conventions
    train_loss_key = "train_losses" if "train_losses" in history else "train_loss"
    val_loss_key = "val_losses" if "val_losses" in history else "val_loss"
    val_iou_key = "val_ious" if "val_ious" in history else "val_iou"
    # Support both new (f1) and old (dice) key formats for backward compatibility
    val_f1_key = (
        "val_f1s"
        if "val_f1s" in history
        else ("val_dices" if "val_dices" in history else "val_dice")
    )
    # Add support for precision and recall
    val_precision_key = (
        "val_precisions" if "val_precisions" in history else "val_precision"
    )
    val_recall_key = "val_recalls" if "val_recalls" in history else "val_recall"

    # Collect available metrics for plotting
    available_metrics = []
    metric_info = {
        "Loss": (train_loss_key, val_loss_key, ["Train Loss", "Val Loss"]),
        "IoU": (val_iou_key, None, ["Val IoU"]),
        "F1": (val_f1_key, None, ["Val F1"]),
        "Precision": (val_precision_key, None, ["Val Precision"]),
        "Recall": (val_recall_key, None, ["Val Recall"]),
    }

    for metric_name, (key1, key2, labels) in metric_info.items():
        if key1 in history or (key2 and key2 in history):
            available_metrics.append((metric_name, key1, key2, labels))

    # Determine number of subplots and figure size
    n_plots = len(available_metrics)
    if figsize is None:
        figsize = (5 * n_plots, 5)

    # Create DataFrame for all metrics
    n_epochs = 0
    df_data = {}

    # Add epochs
    if "epochs" in history:
        df_data["epoch"] = history["epochs"]
        n_epochs = len(history["epochs"])
    elif train_loss_key in history:
        n_epochs = len(history[train_loss_key])
        df_data["epoch"] = list(range(1, n_epochs + 1))

    # Add all available metrics to DataFrame
    if train_loss_key in history:
        df_data["train_loss"] = history[train_loss_key]
    if val_loss_key in history:
        df_data["val_loss"] = history[val_loss_key]
    if val_iou_key in history:
        df_data["val_iou"] = history[val_iou_key]
    if val_f1_key in history:
        df_data["val_f1"] = history[val_f1_key]
    if val_precision_key in history:
        df_data["val_precision"] = history[val_precision_key]
    if val_recall_key in history:
        df_data["val_recall"] = history[val_recall_key]

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Export to CSV if requested
    if csv_path:
        df.to_csv(csv_path, index=False)
        if verbose:
            logger.info(f"Metrics exported to: {csv_path}")

    # Create plots
    if n_plots > 0:
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        # Use epoch numbers as x-axis values
        epochs = df_data.get("epoch", list(range(1, n_epochs + 1)))

        for idx, (metric_name, key1, key2, labels) in enumerate(available_metrics):
            ax = axes[idx]

            if metric_name == "Loss":
                # Special handling for loss (has both train and val)
                if key1 in history:
                    ax.plot(epochs, history[key1], label=labels[0])
                if key2 and key2 in history:
                    # Filter out non-finite values so the line still renders
                    vals = history[key2]
                    finite = [(e, v) for e, v in zip(epochs, vals) if np.isfinite(v)]
                    if finite:
                        ax.plot(*zip(*finite), label=labels[1])
            else:
                # Single metric plots
                if key1 in history:
                    ax.plot(epochs, history[key1], label=labels[0])

            ax.set_title(metric_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            if "dpi" not in kwargs:
                kwargs["dpi"] = 150
            if "bbox_inches" not in kwargs:
                kwargs["bbox_inches"] = "tight"
            plt.savefig(save_path, **kwargs)

        plt.show()

    # Print summary statistics
    if verbose:
        logger.info("=== Performance Metrics Summary ===")
        if val_iou_key in history:
            logger.info(
                f"IoU     - Best: {max(history[val_iou_key]):.4f} | Final: {history[val_iou_key][-1]:.4f}"
            )
        if val_f1_key in history:
            logger.info(
                f"F1      - Best: {max(history[val_f1_key]):.4f} | Final: {history[val_f1_key][-1]:.4f}"
            )
        if val_precision_key in history:
            logger.info(
                f"Precision - Best: {max(history[val_precision_key]):.4f} | Final: {history[val_precision_key][-1]:.4f}"
            )
        if val_recall_key in history:
            logger.info(
                f"Recall  - Best: {max(history[val_recall_key]):.4f} | Final: {history[val_recall_key][-1]:.4f}"
            )
        if val_loss_key in history:
            logger.info(
                f"Val Loss - Best: {min(history[val_loss_key]):.4f} | Final: {history[val_loss_key][-1]:.4f}"
            )
        logger.info("===================================\n")

    return df


def plot_prediction_comparison(
    original_image: Union[str, np.ndarray, Image.Image],
    prediction_image: Union[str, np.ndarray, Image.Image],
    ground_truth_image: Optional[Union[str, np.ndarray, Image.Image]] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    prediction_colormap: str = "gray",
    ground_truth_colormap: str = "gray",
    original_colormap: Optional[str] = None,
    indexes: Optional[List[int]] = None,
    divider: Optional[float] = None,
) -> None:
    """Plot original image, prediction, and optional ground truth side by side.

    Supports input as file paths, NumPy arrays, or PIL Images. For multi-band
    images, selected channels can be specified via `indexes`. If the image data
    is not normalized (e.g., Sentinel-2 [0, 10000]), the `divider` can be used
    to scale values for visualization.

    Args:
        original_image (Union[str, np.ndarray, Image.Image]):
            Original input image as a file path, NumPy array, or PIL Image.
        prediction_image (Union[str, np.ndarray, Image.Image]):
            Predicted segmentation mask image.
        ground_truth_image (Optional[Union[str, np.ndarray, Image.Image]], optional):
            Ground truth mask image. Defaults to None.
        titles (Optional[List[str]], optional):
            List of titles for the subplots. If not provided, default titles are used.
        figsize (Tuple[int, int], optional):
            Size of the entire figure in inches. Defaults to (15, 5).
        save_path (Optional[str], optional):
            If specified, saves the figure to this path. Defaults to None.
        show_plot (bool, optional):
            Whether to display the figure using plt.show(). Defaults to True.
        prediction_colormap (str, optional):
            Colormap to use for the prediction mask. Defaults to "gray".
        ground_truth_colormap (str, optional):
            Colormap to use for the ground truth mask. Defaults to "gray".
        original_colormap (Optional[str], optional):
            Colormap to use for the original image if it's grayscale. Defaults to None.
        indexes (Optional[List[int]], optional):
            List of band/channel indexes (0-based for NumPy, 1-based for rasterio) to extract from the original image.
            Useful for multi-band imagery like Sentinel-2. Defaults to None.
        divider (Optional[float], optional):
            Value to divide the original image by for normalization (e.g., 10000 for reflectance). Defaults to None.

    Returns:
        matplotlib.figure.Figure:
            The generated matplotlib figure object.
    """

    def _load_image(img_input, indexes=None):
        """Helper function to load image from various input types."""
        if isinstance(img_input, str):
            if img_input.lower().endswith((".tif", ".tiff")):
                with rasterio.open(img_input) as src:
                    if indexes:
                        img = src.read(indexes)  # 1-based
                        img = (
                            np.transpose(img, (1, 2, 0)) if len(indexes) > 1 else img[0]
                        )
                    else:
                        img = src.read()
                        if img.shape[0] == 1:
                            img = img[0]
                        else:
                            img = np.transpose(img, (1, 2, 0))
            else:
                img = np.array(Image.open(img_input))
        elif isinstance(img_input, Image.Image):
            img = np.array(img_input)
        elif isinstance(img_input, np.ndarray):
            img = img_input
            if indexes is not None and img.ndim == 3:
                img = img[:, :, indexes]
        else:
            raise ValueError(f"Unsupported image type: {type(img_input)}")
        return img

    # Load images
    original = _load_image(original_image, indexes=indexes)
    prediction = _load_image(prediction_image)
    ground_truth = (
        _load_image(ground_truth_image) if ground_truth_image is not None else None
    )

    # Apply divider normalization if requested
    if divider is not None and isinstance(original, np.ndarray) and original.ndim == 3:
        original = np.clip(original.astype(np.float32) / divider, 0, 1)

    # Determine layout
    num_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 2:
        axes = [axes[0], axes[1]]

    if titles is None:
        titles = ["Original Image", "Prediction"]
        if ground_truth is not None:
            titles.append("Ground Truth")

    # Plot original
    if original.ndim == 3 and original.shape[2] in [3, 4]:
        axes[0].imshow(original)
    else:
        axes[0].imshow(original, cmap=original_colormap)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(prediction, cmap=prediction_colormap)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    # Ground truth
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap=ground_truth_colormap)
        axes[2].set_title(titles[2])
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig
