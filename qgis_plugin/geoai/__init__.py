"""
GeoAI Plugin for QGIS

This plugin provides AI-powered geospatial analysis tools including:
- Moondream vision-language model for image analysis
- Semantic segmentation model training and inference
"""

from .geoai_plugin import GeoAIPlugin


def classFactory(iface):
    """Load GeoAIPlugin class from file geoai_plugin.

    Args:
        iface: A QGIS interface instance.

    Returns:
        GeoAIPlugin: The plugin instance.
    """
    return GeoAIPlugin(iface)
