# inference module

Provides memory-efficient tiled inference with blending and test-time augmentation (TTA) for large geospatial rasters. This module processes imagery in overlapping tiles with smooth blending at boundaries, reducing memory usage and eliminating seam artifacts when running segmentation or detection models on high-resolution imagery.

!!! example "Related Examples"
    - [Smooth Inference](examples/smooth_inference.ipynb)

::: geoai.inference
