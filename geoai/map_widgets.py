"""Interactive widget for GeoAI."""

import ipywidgets as widgets
from .utils import dict_to_image, dict_to_rioxarray


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

        Returns:
            None

        Example:
            >>> processor = DINOv3GeoProcessor()
            >>> features, h_patches, w_patches = processor.extract_features(raster)
            >>> gui = DINOv3GUI(raster, processor, features, host_map=m)
        """
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
