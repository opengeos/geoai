# Changelog

All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org/).

## v0.35.0 - Mar 17, 2026

- Preserve class labels and instance attributes in instance segmentation inference
- Cache OpenSR checkpoint in torch hub directory instead of cwd
- Add super-resolution notebook and improve SR module
- Improve cloud detection notebook and export cloudmask functions
- Fix clean_instance_mask to remove disconnected fragments
- Add field boundary instance segmentation with FTW dataset support
- Add legend_args parameter to view_raster
- Add semantic segmentation workshop and fix DataParallel state_dict loading
- Add error handling to push_detector_to_hub and predict_detector_from_hub

## v0.34.0 - Mar 15, 2026

- Add class_weights parameter to train_timm_segmentation_model()
- Fix val loss plot when validation loss contains non-finite values
- Add multi-architecture object detection support
- Avoid nested folder when extracting zip with single top-level directory
- Fix integer epoch ticks in training plots and remove deprecated verbose param
- Add push_classifier_to_hub and predict_images_from_hub functions
- Add probability output support to timm_semantic_segmentation
- Add impervious surface mapping example notebook

## v0.33.0 - Mar 10, 2026

- Add memory-efficient tiled inference with spline blending and D4 TTA
- Fix .tiff output filenames and update docstring in object_detection_batch
- Handle list and .tiff inputs in object_detection_batch
- Add image_subdir and mask_subdir params to display_training_tiles
- Fix inference_on_geotiff return type to Tuple[str, float]

## v0.32.0 - Mar 3, 2026

- Add CLIP-based zero-shot classification for vector features
- Fix SamGeo vector output for GeoPackage and Shapefile formats
- Add radiometric normalization (LIRRN) to landcover_utils
- Handle target no data values in rasterio loading
- Replace eager imports with PEP 562 lazy loading in __init__.py
- Fix multi-class segmentation mask handling

## v0.31.1 - Feb 25, 2026

- Fix tiles parameter ignored in view_vector_interactive
- Bump QGIS plugin to v1.0.0
- Add built-in dependency installer for QGIS plugin

## v0.31.0 - Feb 24, 2026

- Add multi-class object detection with NWPU-VHR-10 support
- Add image recognition module with EuroSAT notebook example
- Add Google Satellite Embedding download and interactive notebook
- Add time-series analysis support for multi-temporal satellite imagery
- Add batch processing pipeline framework
- Refactor utils.py into focused submodules, add CLI, expand tests, and add CI coverage

## v0.30.0 - Feb 22, 2026

- Add water segmentation panel to QGIS plugin
- Add OmniWaterMask integration for water body segmentation
- Add S2 surface water detection model and inference notebook
- Add WHU building detection model and HuggingFace Hub inference
- Add TESSERA support
- Add GeoAI MCP Server for AI agent integration
- Add support for torchgeo embeddings
- Fix ignore_index=0 handling and add sparse_labels IoU mode

## v0.29.1 - Feb 10, 2026

- Fix SAM3 load in QGIS when pkg_resources is unavailable
- Enable tests

## v0.29.0 - Feb 6, 2026

- Add canopy height estimation support
- Add ChangeStar building change detection support
- Fix: preserve class IDs in multi-class semantic segmentation masks
- Add JOSS paper

## v0.28.0 - Feb 3, 2026

- Add ONNX Runtime support for geospatial model inference
- Add Flip-n-Slide data augmentation strategy
- Add DeepForest segmentation panel to QGIS plugin
- Fix dependency conflicts and code quality improvements

## v0.27.0 - Jan 29, 2026

- Add support for pixel-level regression tasks
- Add support for exporting data in COCO and YOLO formats
- Add support for multiple Prithvi model variants
- Fix Windows PyQt5 dependency issue in pixi installation
- Fix: Use lazy cv2 imports to avoid QGIS opencv recursion error

## v0.26.0 - Jan 22, 2026

- Add Prithvi EO 2.0 Geospatial Foundation Model

## v0.25.0 - Jan 8, 2026

- Restore the utils module
- Implement sliding window method for Moondream VLM to handle large images
- Fix QGIS plugin CUDA issue

## v0.24.0 - Dec 21, 2025

- Add support for GPKG raster for SamGeo
- Add CPU installation instructions

## v0.23.0 - Dec 16, 2025

- Update QGIS plugin to v0.3.0
- Add timelapse notebook example
- Add macOS tests to GitHub actions

## v0.22.0 - Dec 13, 2025

- Add band selection support for SamGeo
- Add support for auto showing results for SamGeo
- Add QGIS plugin update checker
- Add video tutorials to docs

## v0.21.0 - Dec 12, 2025

- Add auto module for supporting HuggingFace models
- Add LICENSE file for QGIS plugin

## v0.20.0 - Dec 11, 2025

- Add SamGeo panel to the QGIS plugin
- Add QGIS plugin

## v0.19.0 - Dec 10, 2025

- Add Moondream VLM support
- Add moondream interactive GUI
- Add support for Gemini model integration into geoagents

## v0.18.0 - Nov 4, 2025

- Fix syntax error in scheduler initialization
- Add ASA workshop notebook

## v0.17.0 - Oct 19, 2025

- Show agent status during execution
- Add catalog search agent
- Add STAC agent for interactive search

## v0.16.0 - Oct 16, 2025

- Add support for exporting probability image for each class
- Add precision and recall metrics
- Add early stop patience for training
- Rename Dice score as F1 score

## v0.15.0 - Oct 5, 2025

- Add timm support for COCO and YOLO
- Add training support for COCO and YOLO
- Add support for exporting COCO and YOLO formats

## v0.14.0 - Oct 4, 2025

- Add support for creating training data from multiple vector files
- Add instance segmentation example

## v0.13.0 - Sep 30, 2025

- Add more model providers for AI agents
- Make torchange optional
- Add DINOv3 wetlands notebook

## v0.12.0 - Sep 17, 2025

- Add support for AI agents
- Add more tools for the agent
- Fix DINOv3 weights download error

## v0.11.0 - Sep 10, 2025

- Add support for DINOv3
- Add support for JPEG2000

## v0.10.0 - Sep 7, 2025

- Fix torchange python 3.10 error
- Add wetland mapping example

## v0.9.0 - Jul 17, 2025

- Add support for change detection
- Add multi-GPU support
- Fix GDAL installation issue

## v0.8.0 - Jul 8, 2025

- Add support for generating image chips from multiple images
- Add support for batch segmentation
- Add support for non-geospatial raster formats

## v0.7.0 - Jun 19, 2025

- Add support for land cover classification
- Add support for smp segmentation models
- Add support for customizing MaskRCNN model

## v0.6.0 - May 7, 2025

- Add Dockerfile
- Add samgeo module

## v0.5.0 - Mar 26, 2025

- Add water detection notebook example
- Improve pc_stac_download function
- Add get_overture_data function
- Add mosaic_geotiffs function

## v0.4.0 - Mar 12, 2025

- Add support for training object detection model
- Add parking spot detection notebook
- Add support for Hugging Face models

## v0.3.0 - Mar 4, 2025

- Rename common module to utils
- Add building regularization methods
- Add masks_to_vector function

## v0.2.0 - Mar 3, 2025

- Add building extraction pre-trained model
- Add preprocess module

## v0.1.0 - Oct 11, 2023

- Add samgeo support
- Add codespell

## v0.0.1 - Aug 11, 2023

- Initial release
