"""The common module contains common functions and classes used by the other modules."""

import os
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import matplotlib.pyplot as plt
import geopandas as gpd
import leafmap
import torch
import numpy as np
import xarray as xr
import rioxarray
import rasterio as rio
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, utils
from torchgeo.samplers import RandomGeoSampler, Units
from torchgeo.transforms import indices


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
    array_args: Optional[Dict] = {},
    client_args: Optional[Dict] = {"cors_all": False},
    basemap: Optional[str] = "OpenStreetMap",
    **kwargs,
):
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
        basemap (Optional[str], optional): The basemap to use. Defaults to "OpenStreetMap".
        **kwargs (Any): Additional keyword arguments.

    Returns:
        leafmap.Map: The map object with the raster layer added.
    """

    m = leafmap.Map(basemap=basemap)

    if isinstance(source, dict):
        source = dict_to_image(source)

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
    return m


def view_image(
    image: Union[np.ndarray, torch.Tensor],
    transpose: bool = False,
    bdx: Optional[int] = None,
    scale_factor: float = 1.0,
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
        scale_factor (float, optional): The scale factor to apply to the image. Defaults to 1.0.
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
        image = rio.open(image).read().transpose(1, 2, 0)

    plt.figure(figsize=figsize)

    if transpose:
        image = image.transpose(1, 2, 0)

    if bdx is not None:
        image = image[:, :, bdx]

    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, 0:3]

    if scale_factor != 1.0:
        image = np.clip(image * scale_factor, 0, 1)

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


def calc_stats(
    dataset: RasterDataset, divide_by: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the statistics (mean and std) for the entire dataset.

    This function is adapted from the plot_batch() function in the torchgeo library at
    https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html.
    Credit to the torchgeo developers for the original implementation.

    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.

    Args:
        dataset (RasterDataset): The dataset to calculate statistics for.
        divide_by (float, optional): The value to divide the image data by. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean and standard deviation for each band.
    """
    import rasterio as rio

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    files = [
        item.object
        for item in dataset.index.intersection(dataset.index.bounds, objects=True)
    ]

    # Resetting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        img = rio.open(file).read() / divide_by  # type: ignore
        accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
        accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with length n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)


def dict_to_rioxarray(data_dict: Dict) -> xr.DataArray:
    """Convert a dictionary to a xarray DataArray. The dictionary should contain the
    following keys: "crs", "bounds", and "image". It can be generated from a TorchGeo
    dataset sampler.

    Args:
        data_dict (Dict): The dictionary containing the data.

    Returns:
        xr.DataArray: The xarray DataArray.
    """

    from affine import Affine

    # Extract components from the dictionary
    crs = data_dict["crs"]
    bounds = data_dict["bounds"]
    image_tensor = data_dict["image"]

    # Convert tensor to numpy array if needed
    if hasattr(image_tensor, "numpy"):
        # For PyTorch tensors
        image_array = image_tensor.numpy()
    else:
        # If it's already a numpy array or similar
        image_array = np.array(image_tensor)

    # Calculate pixel resolution
    width = image_array.shape[2]  # Width is the size of the last dimension
    height = image_array.shape[1]  # Height is the size of the middle dimension

    res_x = (bounds.maxx - bounds.minx) / width
    res_y = (bounds.maxy - bounds.miny) / height

    # Create the transform matrix
    transform = Affine(res_x, 0.0, bounds.minx, 0.0, -res_y, bounds.maxy)

    # Create dimensions
    x_coords = np.linspace(bounds.minx + res_x / 2, bounds.maxx - res_x / 2, width)
    y_coords = np.linspace(bounds.maxy - res_y / 2, bounds.miny + res_y / 2, height)

    # If time dimension exists in the bounds
    if hasattr(bounds, "mint") and hasattr(bounds, "maxt"):
        # Create a single time value or range if needed
        t_coords = [
            bounds.mint
        ]  # Or np.linspace(bounds.mint, bounds.maxt, num_time_steps)

        # Create DataArray with time dimension
        dims = (
            ("band", "y", "x")
            if image_array.shape[0] <= 10
            else ("time", "band", "y", "x")
        )

        if dims[0] == "band":
            # For multi-band single time
            da = xr.DataArray(
                image_array,
                dims=dims,
                coords={
                    "band": np.arange(1, image_array.shape[0] + 1),
                    "y": y_coords,
                    "x": x_coords,
                },
            )
        else:
            # For multi-time multi-band
            da = xr.DataArray(
                image_array,
                dims=dims,
                coords={
                    "time": t_coords,
                    "band": np.arange(1, image_array.shape[1] + 1),
                    "y": y_coords,
                    "x": x_coords,
                },
            )
    else:
        # Create DataArray without time dimension
        da = xr.DataArray(
            image_array,
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(1, image_array.shape[0] + 1),
                "y": y_coords,
                "x": x_coords,
            },
        )

    # Set spatial attributes
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)

    return da


def dict_to_image(
    data_dict: Dict[str, Any], output: Optional[str] = None, **kwargs
) -> rio.DatasetReader:
    """Convert a dictionary containing spatial data to a rasterio dataset or save it to
    a file. The dictionary should contain the following keys: "crs", "bounds", and "image".
    It can be generated from a TorchGeo dataset sampler.

    This function transforms a dictionary with CRS, bounding box, and image data
    into a rasterio DatasetReader using leafmap's array_to_image utility after
    first converting to a rioxarray DataArray.

    Args:
        data_dict: A dictionary containing:
            - 'crs': A pyproj CRS object
            - 'bounds': A BoundingBox object with minx, maxx, miny, maxy attributes
              and optionally mint, maxt for temporal bounds
            - 'image': A tensor or array-like object with image data
        output: Optional path to save the image to a file. If not provided, the image
            will be returned as a rasterio DatasetReader object.
        **kwargs: Additional keyword arguments to pass to leafmap.array_to_image.
            Common options include:
            - colormap: str, name of the colormap (e.g., 'viridis', 'terrain')
            - vmin: float, minimum value for colormap scaling
            - vmax: float, maximum value for colormap scaling

    Returns:
        A rasterio DatasetReader object that can be used for visualization or
        further processing.

    Examples:
        >>> image = dict_to_image(
        ...     {'crs': CRS.from_epsg(26911), 'bounds': bbox, 'image': tensor},
        ...     colormap='terrain'
        ... )
        >>> fig, ax = plt.subplots(figsize=(10, 10))
        >>> show(image, ax=ax)
    """
    da = dict_to_rioxarray(data_dict)

    if output is not None:
        out_dir = os.path.abspath(os.path.dirname(output))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        da.rio.to_raster(output)
        return output
    else:
        image = leafmap.array_to_image(da, **kwargs)
        return image


def view_vector(
    vector_data,
    column=None,
    cmap="viridis",
    figsize=(10, 10),
    title=None,
    legend=True,
    basemap=False,
    alpha=0.7,
    edge_color="black",
    classification="quantiles",
    n_classes=5,
    highlight_index=None,
    highlight_color="red",
    scheme=None,
    save_path=None,
    dpi=300,
):
    """
    Visualize vector datasets with options for styling, classification, basemaps and more.

    This function visualizes GeoDataFrame objects with customizable symbology.
    It supports different vector types (points, lines, polygons), attribute-based
    classification, and background basemaps.

    Args:
        vector_data (geopandas.GeoDataFrame): The vector dataset to visualize.
        column (str, optional): Column to use for choropleth mapping. If None,
            a single color will be used. Defaults to None.
        cmap (str or matplotlib.colors.Colormap, optional): Colormap to use for
            choropleth mapping. Defaults to "viridis".
        figsize (tuple, optional): Figure size as (width, height) in inches.
            Defaults to (10, 10).
        title (str, optional): Title for the plot. Defaults to None.
        legend (bool, optional): Whether to display a legend. Defaults to True.
        basemap (bool, optional): Whether to add a web basemap. Requires contextily.
            Defaults to False.
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

    Returns:
        matplotlib.axes.Axes: The Axes object containing the plot.

    Examples:
        >>> import geopandas as gpd
        >>> cities = gpd.read_file("cities.shp")
        >>> view_vector(cities, "population", cmap="Reds", basemap=True)

        >>> roads = gpd.read_file("roads.shp")
        >>> view_vector(roads, "type", basemap=True, figsize=(12, 8))
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

    # Determine geometry type
    geom_type = gdf.geometry.iloc[0].geom_type

    # Plotting parameters
    plot_kwargs = {"alpha": alpha, "ax": ax}

    # Set up keyword arguments based on geometry type
    if "Point" in geom_type:
        plot_kwargs["markersize"] = 50
        plot_kwargs["edgecolor"] = edge_color
    elif "Line" in geom_type:
        plot_kwargs["linewidth"] = 1
    elif "Polygon" in geom_type:
        plot_kwargs["edgecolor"] = edge_color

    # Classification options
    if column is not None:
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

    # Add basemap if requested
    if basemap:
        try:
            ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Could not add basemap: {e}")

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
    vector_data,
    **kwargs,
):
    """
    Visualize vector datasets with options for styling, classification, basemaps and more.

    This function visualizes GeoDataFrame objects with customizable symbology.
    It supports different vector types (points, lines, polygons), attribute-based
    classification, and background basemaps.

    Args:
        vector_data (geopandas.GeoDataFrame): The vector dataset to visualize.
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
    if isinstance(vector_data, str):
        vector_data = gpd.read_file(vector_data)

    # Check if input is a GeoDataFrame
    if not isinstance(vector_data, gpd.GeoDataFrame):
        raise TypeError("Input data must be a GeoDataFrame")

    return vector_data.explore(**kwargs)
