"""Interactive widget for GeoAI."""

import os
import string
import random
import tempfile
from typing import Any, Optional

import ipywidgets as widgets

from .utils import dict_to_image, dict_to_rioxarray


def random_string(string_length: int = 6) -> str:
    """Generate a random string of fixed length.

    Args:
        string_length: The length of the random string. Defaults to 6.

    Returns:
        A random string of the specified length.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(string_length))


class DINOv3GUI(widgets.VBox):
    """Interactive widget for DINOv3."""

    def __init__(
        self,
        raster: str,
        processor=None,
        features=None,
        host_map=None,
        position="topright",
        colormap_options=None,
        raster_args=None,
    ):
        """Initialize the DINOv3 GUI.

        Args:
            raster (str): The path to the raster image.
            processor (DINOv3GeoProcessor): The DINOv3 processor.
            features (torch.Tensor): The features of the raster image.
            host_map (Map): The host map.
            position (str): The position of the widget.
            colormap_options (list): The colormap options.
            raster_args (dict): The raster arguments.

        Example:
            >>> processor = DINOv3GeoProcessor()
            >>> features, h_patches, w_patches = processor.extract_features(raster)
            >>> gui = DINOv3GUI(raster, processor, features, host_map=m)
        """
        super().__init__()

        if raster_args is None:
            raster_args = {}

        if "layer_name" not in raster_args:
            raster_args["layer_name"] = "Raster"

        if colormap_options is None:
            colormap_options = [
                "jet",
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
            ]

        main_widget = widgets.VBox(layout=widgets.Layout(width="230px"))
        style = {"description_width": "initial"}
        layout = widgets.Layout(width="95%", padding="0px 5px 0px 5px")

        interpolation_checkbox = widgets.Checkbox(
            value=True,
            description="Use interpolation",
            style=style,
            layout=layout,
        )

        threshold_slider = widgets.FloatSlider(
            value=0.7,
            min=0,
            max=1,
            step=0.01,
            description="Threshold",
            style=style,
            layout=layout,
        )

        opacity_slider = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.01,
            description="Opacity",
            style=style,
            layout=layout,
        )
        colormap_dropdown = widgets.Dropdown(
            options=colormap_options,
            value="jet",
            description="Colormap",
            style=style,
            layout=layout,
        )
        layer_name_input = widgets.Text(
            value="Similarity",
            description="Layer name",
            style=style,
            layout=layout,
        )

        save_button = widgets.Button(
            description="Save",
        )

        reset_button = widgets.Button(
            description="Reset",
        )

        output = widgets.Output()

        main_widget.children = [
            interpolation_checkbox,
            threshold_slider,
            opacity_slider,
            colormap_dropdown,
            layer_name_input,
            widgets.HBox([save_button, reset_button]),
            output,
        ]

        if host_map is not None:

            host_map.add_widget(main_widget, add_header=True, position=position)

            if raster is not None:
                host_map.add_raster(raster, **raster_args)

            def handle_map_interaction(**kwargs):
                try:
                    if kwargs.get("type") == "click":
                        latlon = kwargs.get("coordinates")
                        with output:
                            output.clear_output()

                            results = processor.compute_similarity(
                                source=raster,
                                features=features,
                                query_coords=latlon[::-1],
                                output_dir="dinov3_results",
                                use_interpolation=interpolation_checkbox.value,
                                coord_crs="EPSG:4326",
                            )
                            array = results["image_dict"]["image"]
                            binary_array = array > threshold_slider.value
                            image = dict_to_image(results["image_dict"])
                            binary_image = dict_to_image(
                                {
                                    "image": binary_array,
                                    "crs": results["image_dict"]["crs"],
                                    "bounds": results["image_dict"]["bounds"],
                                }
                            )
                            host_map.add_raster(
                                image,
                                colormap=colormap_dropdown.value,
                                opacity=opacity_slider.value,
                                layer_name=layer_name_input.value,
                                zoom_to_layer=False,
                                overwrite=True,
                            )
                            host_map.add_raster(
                                binary_image,
                                colormap="jet",
                                nodata=0,
                                opacity=opacity_slider.value,
                                layer_name="Foreground",
                                zoom_to_layer=False,
                                overwrite=True,
                                visible=False,
                            )
                except Exception as e:
                    with output:
                        print(e)

            host_map.on_interaction(handle_map_interaction)
            host_map.default_style = {"cursor": "crosshair"}


def moondream_gui(
    moondream,
    basemap: str = "SATELLITE",
    out_dir: Optional[str] = None,
    opacity: float = 0.5,
    **kwargs: Any,
):
    """Display an interactive GUI for using Moondream with leafmap.

    This function creates an interactive map interface for using Moondream
    vision language model capabilities including:
    - Image captioning (short, normal, long) with streaming output
    - Visual question answering (query) with streaming output
    - Object detection with bounding boxes displayed on map
    - Point detection for locating objects with markers on map

    Args:
        moondream (MoondreamGeo): The MoondreamGeo object with a loaded image.
            Must have called load_image() or load_geotiff() first.
        basemap (str, optional): The basemap to use. Defaults to "SATELLITE".
        out_dir (str, optional): The output directory for saving results.
            Defaults to None (uses temp directory).
        opacity (float, optional): The opacity of overlay layers. Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to leafmap.Map().

    Returns:
        leafmap.Map: The interactive map with the Moondream GUI.

    Example:
        >>> from geoai import MoondreamGeo, moondream_gui
        >>> moondream = MoondreamGeo()
        >>> moondream.load_image("image.tif")
        >>> m = moondream_gui(moondream)
        >>> m
    """
    try:
        import ipyevents
        import ipyleaflet
        import leafmap
        from ipyfilechooser import FileChooser
    except ImportError:
        raise ImportError(
            "The moondream_gui function requires additional packages. "
            "Please install them with: pip install leafmap ipyevents ipyfilechooser"
        )

    if out_dir is None:
        out_dir = tempfile.gettempdir()

    # Create the map
    m = leafmap.Map(**kwargs)
    m.default_style = {"cursor": "crosshair"}
    if basemap is not None:
        m.add_basemap(basemap, show=False)

    # Try to add the image layer if source is available
    if moondream._source_path is not None:
        try:
            m.add_raster(moondream._source_path, layer_name="Image")
        except Exception:
            pass

    # Initialize marker storage for detection results
    m.detection_markers = []
    m.point_markers = []

    # Removed unused LayerGroups for detections and points.
    m.last_result_as_gdf = None

    # Widget styling
    widget_width = "300px"
    button_width = "90px"
    padding = "0px 4px 0px 4px"
    style = {"description_width": "initial"}

    # Create toolbar buttons
    toolbar_button = widgets.ToggleButton(
        value=True,
        tooltip="Toolbar",
        icon="gear",
        layout=widgets.Layout(width="28px", height="28px", padding="0px 0px 0px 4px"),
    )

    close_button = widgets.ToggleButton(
        value=False,
        tooltip="Close the tool",
        icon="times",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding="0px 0px 0px 4px"),
    )

    # Mode selection
    mode_dropdown = widgets.Dropdown(
        options=["Caption", "Query", "Detect", "Point"],
        value="Caption",
        description="Mode:",
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    # Text prompt input
    text_prompt = widgets.Text(
        description="Prompt:",
        placeholder="Enter text prompt...",
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    # Caption length selector (only visible in Caption mode)
    caption_length = widgets.Dropdown(
        options=["short", "normal", "long"],
        value="normal",
        description="Length:",
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    # Opacity slider for overlays
    opacity_slider = widgets.FloatSlider(
        description="Opacity:",
        min=0,
        max=1,
        value=opacity,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    # Color picker for detection/point markers
    colorpicker = widgets.ColorPicker(
        concise=False,
        description="Color:",
        value="#ff0000",
        layout=widgets.Layout(width="150px", padding=padding),
        style=style,
    )

    # Action buttons
    run_button = widgets.ToggleButton(
        description="Run",
        value=False,
        button_style="primary",
        layout=widgets.Layout(padding=padding, width=button_width),
    )

    save_button = widgets.ToggleButton(
        description="Save",
        value=False,
        button_style="primary",
        layout=widgets.Layout(width=button_width),
    )

    reset_button = widgets.ToggleButton(
        description="Reset",
        value=False,
        button_style="primary",
        layout=widgets.Layout(width=button_width),
    )

    # Output area for displaying results - using HTML for better text display
    output_html = widgets.HTML(
        value="",
        layout=widgets.Layout(
            width=widget_width,
            padding=padding,
            max_width=widget_width,
            min_height="0px",
            max_height="300px",
            overflow="auto",
        ),
    )

    # Build the toolbar layout
    toolbar_header = widgets.HBox()
    toolbar_header.children = [close_button, toolbar_button]

    toolbar_footer = widgets.VBox()
    toolbar_footer.children = [
        mode_dropdown,
        text_prompt,
        caption_length,
        opacity_slider,
        colorpicker,
        widgets.HBox(
            [run_button, save_button, reset_button],
            layout=widgets.Layout(padding="0px 4px 0px 4px"),
        ),
        output_html,
    ]

    toolbar_widget = widgets.VBox()
    toolbar_widget.children = [toolbar_header, toolbar_footer]

    # Event handling for toolbar collapse/expand
    toolbar_event = ipyevents.Event(
        source=toolbar_widget, watched_events=["mouseenter", "mouseleave"]
    )

    def update_ui_visibility(change=None):
        """Update UI element visibility based on selected mode."""
        mode = mode_dropdown.value

        # Clear prompt and output when mode changes
        text_prompt.value = ""
        output_html.value = ""

        if mode == "Caption":
            text_prompt.layout.display = "none"
            caption_length.layout.display = "flex"
        elif mode == "Query":
            text_prompt.layout.display = "flex"
            text_prompt.placeholder = "Ask a question about the image..."
            caption_length.layout.display = "none"
        elif mode == "Detect":
            text_prompt.layout.display = "flex"
            text_prompt.placeholder = "Object type to detect (e.g., building, trees)..."
            caption_length.layout.display = "none"
        elif mode == "Point":
            text_prompt.layout.display = "flex"
            text_prompt.placeholder = "Object description to locate..."
            caption_length.layout.display = "none"

    mode_dropdown.observe(update_ui_visibility, "value")
    update_ui_visibility()  # Initial update

    def handle_toolbar_event(event):
        if event["type"] == "mouseenter":
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        elif event["type"] == "mouseleave":
            if not toolbar_button.value:
                toolbar_widget.children = [toolbar_button]
                toolbar_button.value = False
                close_button.value = False

    toolbar_event.on_dom_event(handle_toolbar_event)

    def toolbar_btn_click(change):
        if change["new"]:
            close_button.value = False
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        else:
            if not close_button.value:
                toolbar_widget.children = [toolbar_button]

    toolbar_button.observe(toolbar_btn_click, "value")

    def close_btn_click(change):
        if change["new"]:
            toolbar_button.value = False
            if m.toolbar_control in m.controls:
                m.remove_control(m.toolbar_control)
            toolbar_widget.close()

    close_button.observe(close_btn_click, "value")

    def clear_detections():
        """Clear all detection markers and layers."""
        if "Detections" in m.get_layer_names():
            m.remove_layer(m.find_layer("Detections"))

    def clear_points():
        """Clear all point markers."""
        if "Points" in m.get_layer_names():
            m.remove_layer(m.find_layer("Points"))

    def add_detection_boxes(result, color="#ff0000"):
        """Add bounding boxes from detection result to the map."""
        clear_detections()

        if "gdf" in result and len(result["gdf"]) > 0:
            gdf = result["gdf"].copy()
            m.add_gdf(
                gdf,
                layer_name="Detections",
                style={
                    "color": color,
                    "fillColor": color,
                    "fillOpacity": opacity_slider.value,
                    "weight": 2,
                },
                info_mode=None,
            )

    def add_point_markers(result, color="#ff0000", opacity=0.5):
        """Add point markers from point detection result to the map."""
        clear_points()

        if "gdf" in result and len(result["gdf"]) > 0:
            gdf = result["gdf"].copy().to_crs("EPSG:4326")
            gdf["x"] = gdf.geometry.centroid.x
            gdf["y"] = gdf.geometry.centroid.y

            m.add_circle_markers_from_xy(
                gdf,
                "x",
                "y",
                radius=6,
                color=color,
                fill_color=color,
                fill_opacity=opacity,
                layer_name="Points",
            )

    def update_output(text, append=False):
        """Update the output HTML widget."""
        # Escape HTML and convert newlines
        import html

        escaped = html.escape(text)
        formatted = escaped.replace("\n", "<br>")
        style = "font-family: monospace; font-size: 12px; word-wrap: break-word;"

        if append and output_html.value:
            # Extract existing content and append
            current = output_html.value
            if "<div" in current:
                # Find the content between div tags
                start = current.find(">") + 1
                end = current.rfind("</div>")
                existing = current[start:end]
                output_html.value = f'<div style="{style}">{existing}{formatted}</div>'
            else:
                output_html.value = f'<div style="{style}">{formatted}</div>'
        else:
            output_html.value = f'<div style="{style}">{formatted}</div>'

    def run_button_click(change):
        if change["new"]:
            run_button.value = False
            mode = mode_dropdown.value

            if moondream._source_path is None and moondream._metadata is None:
                update_output(
                    "Please load an image first using load_image() or load_geotiff()."
                )
                return

            try:

                if mode == "Caption":
                    update_output(f"Generating caption ({caption_length.value})...")

                    result = moondream.caption(
                        moondream._source_path,
                        length=caption_length.value,
                        stream=False,
                    )
                    caption_text = result.get("caption", str(result))
                    update_output(f"Caption ({caption_length.value}):\n{caption_text}")
                    m.last_result = result
                    m.last_result_as_gdf = None

                elif mode == "Query":
                    if len(text_prompt.value) == 0:
                        update_output("Please enter a question in the prompt field.")
                        return

                    update_output(f"Q: {text_prompt.value}\nGenerating answer...")

                    result = moondream.query(
                        text_prompt.value,
                        source=moondream._source_path,
                        stream=False,
                    )
                    answer_text = result.get("answer", str(result))
                    update_output(f"Q: {text_prompt.value}\nA: {answer_text}")
                    m.last_result = result

                elif mode == "Detect":
                    if len(text_prompt.value) == 0:
                        update_output("Please enter an object type to detect.")
                        return

                    update_output(f"Detecting: {text_prompt.value}...")

                    result = moondream.detect(
                        moondream._source_path,
                        text_prompt.value,
                    )
                    num_objects = len(result.get("objects", []))

                    # Show detection info
                    info_text = f"Detecting: {text_prompt.value}\nFound {num_objects} object(s)."
                    if "gdf" in result and len(result["gdf"]) > 0:
                        info_text += (
                            f"\nAdded {len(result['gdf'])} bounding box(es) to map."
                        )
                    update_output(info_text)

                    if num_objects > 0:
                        add_detection_boxes(result, colorpicker.value)
                    m.last_result = result
                    if "gdf" in result and len(result["gdf"]) > 0:
                        m.last_result_as_gdf = result["gdf"].to_crs("EPSG:4326")

                elif mode == "Point":
                    if len(text_prompt.value) == 0:
                        update_output("Please enter an object description to locate.")
                        return

                    update_output(f"Locating: {text_prompt.value}...")

                    result = moondream.point(
                        moondream._source_path,
                        text_prompt.value,
                    )
                    num_points = len(result.get("points", []))
                    update_output(
                        f"Locating: {text_prompt.value}\nFound {num_points} point(s)."
                    )

                    if num_points > 0:
                        add_point_markers(
                            result, colorpicker.value, opacity_slider.value
                        )
                    m.last_result = result
                    if "gdf" in result and len(result["gdf"]) > 0:
                        m.last_result_as_gdf = result["gdf"].to_crs("EPSG:4326")
            except Exception as e:
                import traceback

                update_output(f"Error: {e}\n\n{traceback.format_exc()}")

    run_button.observe(run_button_click, "value")

    def filechooser_callback(chooser):
        if chooser.selected is not None:
            try:
                filename = chooser.selected
                if hasattr(m, "last_result") and m.last_result:
                    result = m.last_result

                    # Save based on result type
                    if "gdf" in result and len(result["gdf"]) > 0:
                        gdf = result["gdf"]
                        ext = os.path.splitext(filename)[1].lower()
                        if ext == ".geojson":
                            gdf.to_file(filename, driver="GeoJSON")
                        elif ext == ".shp":
                            gdf.to_file(filename, driver="ESRI Shapefile")
                        elif ext == ".gpkg":
                            gdf.to_file(filename, driver="GPKG")
                        else:
                            gdf.to_file(filename)
                        update_output(f"Saved {len(gdf)} features to {filename}")

                    elif "caption" in result:
                        with open(filename, "w") as f:
                            f.write(result["caption"])
                        update_output(f"Saved caption to {filename}")

                    elif "answer" in result:
                        with open(filename, "w") as f:
                            f.write(f"Q: {text_prompt.value}\n")
                            f.write(f"A: {result['answer']}")
                        update_output(f"Saved Q&A to {filename}")

            except Exception as e:
                update_output(f"Error saving: {e}")

            if hasattr(m, "save_control") and m.save_control in m.controls:
                m.remove_control(m.save_control)
                delattr(m, "save_control")
            save_button.value = False

    def save_button_click(change):
        if change["new"]:
            if not hasattr(m, "last_result") or m.last_result is None:
                update_output("Please run an operation first.")
                save_button.value = False
                return

            result = m.last_result
            mode = mode_dropdown.value

            # Determine default filename and filter
            if mode in ["Detect", "Point"] and "gdf" in result:
                default_filename = f"{mode.lower()}_{random_string()}.geojson"
                filter_pattern = ["*.geojson", "*.gpkg", "*.shp"]
            else:
                default_filename = f"{mode.lower()}_{random_string()}.txt"
                filter_pattern = ["*.txt"]

            sandbox_path = os.environ.get("SANDBOX_PATH")
            filechooser = FileChooser(
                path=os.getcwd(),
                filename=default_filename,
                sandbox_path=sandbox_path,
                layout=widgets.Layout(width="454px"),
            )
            filechooser.use_dir_icons = True
            filechooser.filter_pattern = filter_pattern
            filechooser.register_callback(filechooser_callback)
            save_control = ipyleaflet.WidgetControl(
                widget=filechooser, position="topright"
            )
            m.add_control(save_control)
            m.save_control = save_control
        else:
            if hasattr(m, "save_control") and m.save_control in m.controls:
                m.remove_control(m.save_control)
                delattr(m, "save_control")

    save_button.observe(save_button_click, "value")

    def reset_button_click(change):
        if change["new"]:
            run_button.value = False
            save_button.value = False
            reset_button.value = False
            text_prompt.value = ""
            caption_length.value = "normal"
            opacity_slider.value = 0.5
            colorpicker.value = "#ff0000"
            output_html.value = ""

            # Clear all markers and detection boxes
            clear_detections()
            clear_points()

            # Clear last result
            if hasattr(m, "last_result"):
                m.last_result = None

    reset_button.observe(reset_button_click, "value")

    # Add the toolbar control to the map
    toolbar_control = ipyleaflet.WidgetControl(
        widget=toolbar_widget, position="topright"
    )
    m.add_control(toolbar_control)
    m.toolbar_control = toolbar_control

    return m
