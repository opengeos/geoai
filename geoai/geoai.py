"""Main module."""

import logging

logging.getLogger("maplibre").setLevel(logging.ERROR)

import leafmap
import leafmap.maplibregl as maplibregl

from .classify import classify_image, classify_images, train_classifier
from .download import (
    download_naip,
    download_overture_buildings,
    download_pc_stac_item,
    extract_building_stats,
    get_overture_data,
    pc_collection_list,
    pc_item_asset_list,
    pc_stac_download,
    pc_stac_search,
    read_pc_item_asset,
    view_pc_item,
    view_pc_items,
)
from .extract import *
from .hf import *
from .segment import *
from .train import object_detection, object_detection_batch, train_MaskRCNN_model
from .utils import *


class Map(leafmap.Map):
    """A subclass of leafmap.Map for GeoAI applications."""

    def __init__(self, *args, **kwargs):
        """Initialize the Map class."""
        super().__init__(*args, **kwargs)


class MapLibre(maplibregl.Map):
    """A subclass of maplibregl.Map for GeoAI applications."""

    def __init__(self, *args, **kwargs):
        """Initialize the MapLibre class."""
        super().__init__(*args, **kwargs)


def create_vector_data(
    m: Optional[Map] = None,
    properties: Optional[Dict[str, List[Any]]] = None,
    time_format: str = "%Y%m%dT%H%M%S",
    column_widths: Optional[List[int]] = (9, 3),
    map_height: str = "600px",
    out_dir: Optional[str] = None,
    filename_prefix: str = "",
    file_ext: str = "geojson",
    add_mapillary: bool = False,
    style: str = "photo",
    radius: float = 0.00005,
    width: int = 300,
    height: int = 420,
    frame_border: int = 0,
    **kwargs: Any,
):
    """Generates a widget-based interface for creating and managing vector data on a map.

    This function creates an interactive widget interface that allows users to draw features
    (points, lines, polygons) on a map, assign properties to these features, and export them
    as GeoJSON files. The interface includes a map, a sidebar for property management, and
    buttons for saving, exporting, and resetting the data.

    Args:
        m (Map, optional): An existing Map object. If not provided, a default map with
            basemaps and drawing controls will be created. Defaults to None.
        properties (Dict[str, List[Any]], optional): A dictionary where keys are property names
            and values are lists of possible values for each property. These properties can be
            assigned to the drawn features. Defaults to None.
        time_format (str, optional): The format string for the timestamp used in the exported
            filename. Defaults to "%Y%m%dT%H%M%S".
        column_widths (Optional[List[int]], optional): A list of two integers specifying the
            relative widths of the map and sidebar columns. Defaults to (9, 3).
        map_height (str, optional): The height of the map widget. Defaults to "600px".
        out_dir (str, optional): The directory where the exported GeoJSON files will be saved.
            If not provided, the current working directory is used. Defaults to None.
        filename_prefix (str, optional): A prefix to be added to the exported filename.
            Defaults to "".
        file_ext (str, optional): The file extension for the exported file. Defaults to "geojson".
        add_mapillary (bool, optional): Whether to add a Mapillary image widget that displays the
            nearest image to the clicked point on the map. Defaults to False.
        style (str, optional): The style of the Mapillary image widget. Can be "classic", "photo",
            or "split". Defaults to "photo".
        radius (float, optional): The radius (in degrees) used to search for the nearest Mapillary
            image. Defaults to 0.00005 degrees.
        width (int, optional): The width of the Mapillary image widget. Defaults to 300.
        height (int, optional): The height of the Mapillary image widget. Defaults to 420.
        frame_border (int, optional): The width of the frame border for the Mapillary image widget.
            Defaults to 0.
        **kwargs (Any): Additional keyword arguments that may be passed to the function.

    Returns:
        widgets.VBox: A vertical box widget containing the map, sidebar, and control buttons.

    Example:
        >>> properties = {
        ...     "Type": ["Residential", "Commercial", "Industrial"],
        ...     "Area": [100, 200, 300],
        ... }
        >>> widget = create_vector_data(properties=properties)
        >>> display(widget)  # Display the widget in a Jupyter notebook
    """
    return maplibregl.create_vector_data(
        m=m,
        properties=properties,
        time_format=time_format,
        column_widths=column_widths,
        map_height=map_height,
        out_dir=out_dir,
        filename_prefix=filename_prefix,
        file_ext=file_ext,
        add_mapillary=add_mapillary,
        style=style,
        radius=radius,
        width=width,
        height=height,
        frame_border=frame_border,
        **kwargs,
    )


def edit_vector_data(
    m: Optional[Map] = None,
    filename: str = None,
    properties: Optional[Dict[str, List[Any]]] = None,
    time_format: str = "%Y%m%dT%H%M%S",
    column_widths: Optional[List[int]] = (9, 3),
    map_height: str = "600px",
    out_dir: Optional[str] = None,
    filename_prefix: str = "",
    file_ext: str = "geojson",
    add_mapillary: bool = False,
    style: str = "photo",
    radius: float = 0.00005,
    width: int = 300,
    height: int = 420,
    frame_border: int = 0,
    controls: Optional[List[str]] = None,
    position: str = "top-right",
    fit_bounds_options: Dict = None,
    **kwargs: Any,
):
    """Generates a widget-based interface for creating and managing vector data on a map.

    This function creates an interactive widget interface that allows users to draw features
    (points, lines, polygons) on a map, assign properties to these features, and export them
    as GeoJSON files. The interface includes a map, a sidebar for property management, and
    buttons for saving, exporting, and resetting the data.

    Args:
        m (Map, optional): An existing Map object. If not provided, a default map with
            basemaps and drawing controls will be created. Defaults to None.
        filename (str or gpd.GeoDataFrame): The path to a GeoJSON file or a GeoDataFrame
            containing the vector data to be edited. Defaults to None.
        properties (Dict[str, List[Any]], optional): A dictionary where keys are property names
            and values are lists of possible values for each property. These properties can be
            assigned to the drawn features. Defaults to None.
        time_format (str, optional): The format string for the timestamp used in the exported
            filename. Defaults to "%Y%m%dT%H%M%S".
        column_widths (Optional[List[int]], optional): A list of two integers specifying the
            relative widths of the map and sidebar columns. Defaults to (9, 3).
        map_height (str, optional): The height of the map widget. Defaults to "600px".
        out_dir (str, optional): The directory where the exported GeoJSON files will be saved.
            If not provided, the current working directory is used. Defaults to None.
        filename_prefix (str, optional): A prefix to be added to the exported filename.
            Defaults to "".
        file_ext (str, optional): The file extension for the exported file. Defaults to "geojson".
        add_mapillary (bool, optional): Whether to add a Mapillary image widget that displays the
            nearest image to the clicked point on the map. Defaults to False.
        style (str, optional): The style of the Mapillary image widget. Can be "classic", "photo",
            or "split". Defaults to "photo".
        radius (float, optional): The radius (in degrees) used to search for the nearest Mapillary
            image. Defaults to 0.00005 degrees.
        width (int, optional): The width of the Mapillary image widget. Defaults to 300.
        height (int, optional): The height of the Mapillary image widget. Defaults to 420.
        frame_border (int, optional): The width of the frame border for the Mapillary image widget.
            Defaults to 0.
        controls (Optional[List[str]], optional): The drawing controls to be added to the map.
            Defaults to ["point", "polygon", "line_string", "trash"].
        position (str, optional): The position of the drawing controls on the map. Defaults to "top-right".
        **kwargs (Any): Additional keyword arguments that may be passed to the function.

    Returns:
        widgets.VBox: A vertical box widget containing the map, sidebar, and control buttons.
    """
    return maplibregl.edit_vector_data(
        m=m,
        filename=filename,
        properties=properties,
        time_format=time_format,
        column_widths=column_widths,
        map_height=map_height,
        out_dir=out_dir,
        filename_prefix=filename_prefix,
        file_ext=file_ext,
        add_mapillary=add_mapillary,
        style=style,
        radius=radius,
        width=width,
        height=height,
        frame_border=frame_border,
        controls=controls,
        position=position,
        fit_bounds_options=fit_bounds_options,
        **kwargs,
    )
