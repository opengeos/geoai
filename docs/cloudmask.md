# cloudmask module

Cloud and cloud shadow detection utilities for satellite imagery powered by [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask).

OmniCloudMask performs semantic segmentation to classify pixels into four categories: Clear (`0`), Thick Cloud (`1`), Thin Cloud (`2`), and Cloud Shadow (`3`). It supports Sentinel-2, Landsat 8, PlanetScope, and Maxar imagery at 10 to 50 m resolution. Install the optional `extra` dependency group (`pip install "geoai-py[extra]"`) to use this module.

See the [Cloud Detection](examples/cloud_detection.md) example notebook for a complete walkthrough.

::: geoai.tools.cloudmask
