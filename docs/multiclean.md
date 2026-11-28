# multiclean module

Morphological post-processing utilities for cleaning segmentation masks and classification rasters using [MultiClean](https://github.com/DPIRD-DMA/MultiClean).

This module provides helpers to smooth polygon edges, remove small noise islands, and fill gaps in GeoTIFF segmentation outputs. Install the optional `extra` dependency group (`pip install "geoai-py[extra]"`) to use this module.

See the [Clean Segmentation Results](examples/clean_segmentation_results.md) example notebook for a complete walkthrough.

::: geoai.tools.multiclean
