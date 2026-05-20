"""Shared inference-range helpers for segmentation dock widgets."""

from qgis.core import QgsCoordinateTransform, QgsProject
from qgis.PyQt.QtWidgets import QMessageBox

from .map_tools import RectangleRangeTool


class InferenceRangeMixin:
    """Mixin providing the inference-range drawing/state behavior.

    Dock widgets that mix this in are expected to initialize these
    attributes in their ``__init__``: ``inference_range_tool``,
    ``previous_map_tool``, ``inference_bbox``, ``inference_bbox_crs``,
    ``inference_range_layer_source``. They must also expose
    ``self.iface``, the ``inf_raster_layer_combo`` raster combo box,
    and the ``draw_range_btn``/``range_status_label`` widgets.
    """

    @staticmethod
    def _bbox_from_rect(rect):
        return [
            float(rect.xMinimum()),
            float(rect.yMinimum()),
            float(rect.xMaximum()),
            float(rect.yMaximum()),
        ]

    @staticmethod
    def _intersect_bboxes(first, second):
        minx = max(first[0], second[0])
        miny = max(first[1], second[1])
        maxx = min(first[2], second[2])
        maxy = min(first[3], second[3])
        if minx >= maxx or miny >= maxy:
            return None
        return [minx, miny, maxx, maxy]

    def _current_inference_layer(self):
        return self.inf_raster_layer_combo.currentLayer()

    def _layer_crs_authid(self, layer):
        crs = layer.crs()
        authid = getattr(crs, "authid", None)
        return authid() if callable(authid) else None

    def _range_bbox_for_layer(self, rect, layer):
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        layer_crs = layer.crs()
        if canvas_crs != layer_crs:
            transform = QgsCoordinateTransform(
                canvas_crs, layer_crs, QgsProject.instance()
            )
            rect = transform.transformBoundingBox(rect)

        rect_bbox = self._bbox_from_rect(rect)
        extent_bbox = self._bbox_from_rect(layer.extent())
        return self._intersect_bboxes(rect_bbox, extent_bbox)

    def start_inference_range_tool(self):
        """Start drawing the inference range on the map canvas."""
        layer = self._current_inference_layer()
        if layer is None:
            QMessageBox.warning(
                self,
                "Warning",
                "Please select a raster layer before drawing an inference range.",
            )
            self.draw_range_btn.setChecked(False)
            return

        if self.inference_range_tool is None:
            self.inference_range_tool = RectangleRangeTool(self.iface.mapCanvas())
            self.inference_range_tool.range_drawn.connect(self.set_inference_range)
            self.inference_range_tool.range_canceled.connect(
                self.cancel_inference_range_tool
            )

        self.previous_map_tool = self.iface.mapCanvas().mapTool()
        self.iface.mapCanvas().setMapTool(self.inference_range_tool)

    def cancel_inference_range_tool(self):
        """Handle range drawing cancellation without clearing the saved range."""
        self.draw_range_btn.setChecked(False)
        if self.previous_map_tool:
            self.iface.mapCanvas().setMapTool(self.previous_map_tool)

    def set_inference_range(self, rect):
        """Store the drawn inference range as a layer-CRS bbox."""
        layer = self._current_inference_layer()
        if layer is None:
            self.clear_inference_range()
            QMessageBox.warning(
                self, "Warning", "Selected raster layer is no longer available."
            )
            return

        bbox = self._range_bbox_for_layer(rect, layer)
        if bbox is None:
            self.clear_inference_range()
            QMessageBox.warning(
                self,
                "Warning",
                "The drawn range does not intersect the selected raster layer.",
            )
            return

        self.inference_bbox = bbox
        self.inference_bbox_crs = self._layer_crs_authid(layer)
        self.inference_range_layer_source = layer.source()
        self.range_status_label.setText(
            f"{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}"
        )
        self.draw_range_btn.setChecked(False)
        if self.previous_map_tool:
            self.iface.mapCanvas().setMapTool(self.previous_map_tool)

    def clear_inference_range(self, *args):
        """Clear the selected inference range."""
        self.inference_bbox = None
        self.inference_bbox_crs = None
        self.inference_range_layer_source = None
        if "range_status_label" in self.__dict__:
            self.range_status_label.setText("Full raster")
        if "draw_range_btn" in self.__dict__:
            self.draw_range_btn.setChecked(False)
        if self.inference_range_tool is not None:
            self.inference_range_tool.clear_rubber_band()

    def _validated_inference_bbox(self, input_path):
        if self.inference_bbox is None:
            return None, None, None
        if (
            self.inference_range_layer_source
            and input_path != self.inference_range_layer_source
        ):
            return (
                None,
                None,
                "The selected inference range belongs to a different raster layer. "
                "Clear the range or select the matching raster layer.",
            )
        return self.inference_bbox, self.inference_bbox_crs, None
