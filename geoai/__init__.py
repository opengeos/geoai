"""Top-level package for geoai.

Uses PEP 562 module-level ``__getattr__`` for lazy imports so that
``import geoai`` is fast and does not require torch / torchvision /
transformers / leafmap to be installed.  Heavy dependencies are loaded
only when a specific symbol is first accessed.
"""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "0.31.1"


import importlib
import os
import sys


def set_proj_lib_path(verbose=False):
    """
    Set the PROJ_LIB and GDAL_DATA environment variables based on the current conda environment.

    This function attempts to locate and set the correct paths for PROJ_LIB and GDAL_DATA
    by checking multiple possible locations within the conda environment structure.

    Args:
        verbose (bool): If True, print additional information during the process.

    Returns:
        bool: True if both paths were set successfully, False otherwise.
    """
    try:
        from rasterio.env import set_gdal_config

        # Get conda environment path
        conda_env_path = os.environ.get("CONDA_PREFIX") or sys.prefix

        # Define possible paths for PROJ_LIB
        possible_proj_paths = [
            os.path.join(conda_env_path, "share", "proj"),
            os.path.join(conda_env_path, "Library", "share", "proj"),
            os.path.join(conda_env_path, "Library", "share"),
        ]

        # Define possible paths for GDAL_DATA
        possible_gdal_paths = [
            os.path.join(conda_env_path, "share", "gdal"),
            os.path.join(conda_env_path, "Library", "share", "gdal"),
            os.path.join(conda_env_path, "Library", "data", "gdal"),
            os.path.join(conda_env_path, "Library", "share"),
        ]

        # Set PROJ_LIB environment variable
        proj_set = False
        for proj_path in possible_proj_paths:
            if os.path.exists(proj_path) and os.path.isdir(proj_path):
                # Verify it contains projection data
                if os.path.exists(os.path.join(proj_path, "proj.db")):
                    os.environ["PROJ_LIB"] = proj_path
                    if verbose:
                        print(f"PROJ_LIB set to: {proj_path}")
                    proj_set = True
                    break

        # Set GDAL_DATA environment variable
        gdal_set = False
        for gdal_path in possible_gdal_paths:
            if os.path.exists(gdal_path) and os.path.isdir(gdal_path):
                # Verify it contains the header.dxf file or other critical GDAL files
                if os.path.exists(
                    os.path.join(gdal_path, "header.dxf")
                ) or os.path.exists(os.path.join(gdal_path, "gcs.csv")):
                    os.environ["GDAL_DATA"] = gdal_path
                    if verbose:
                        print(f"GDAL_DATA set to: {gdal_path}")
                    gdal_set = True
                    break

        # If paths still not found, try a last-resort approach
        if not proj_set or not gdal_set:
            # Try a deep search in the conda environment
            for root, dirs, files in os.walk(conda_env_path):
                if not gdal_set and "header.dxf" in files:
                    os.environ["GDAL_DATA"] = root
                    if verbose:
                        print(f"GDAL_DATA set to: {root} (deep search)")
                    gdal_set = True

                if not proj_set and "proj.db" in files:
                    os.environ["PROJ_LIB"] = root
                    if verbose:
                        print(f"PROJ_LIB set to: {root} (deep search)")
                    proj_set = True

                if proj_set and gdal_set:
                    break

        set_gdal_config("PROJ_LIB", os.environ["PROJ_LIB"])
        set_gdal_config("GDAL_DATA", os.environ["GDAL_DATA"])

    except Exception as e:
        print(f"Error setting projection library paths: {e}")
        return


# if ("google.colab" not in sys.modules) and (sys.platform != "windows"):
#     set_proj_lib_path()

# =====================================================================
# Lazy import infrastructure (PEP 562)
#
# No eager imports of submodules — everything is resolved on first
# access via __getattr__.  This keeps ``import geoai`` fast and free
# of heavy dependencies (torch, geopandas, leafmap, etc.).
# =====================================================================

# Mapping: symbol_name -> (submodule_path, original_attr_name)
#   submodule_path is relative to the geoai package (e.g. "utils" -> geoai.utils)
#   original_attr_name is the attribute name on that module, or None if same as symbol_name
_LAZY_SYMBOL_MAP = {
    # --- geoai.pipeline ---
    "Pipeline": ("pipeline", None),
    "PipelineStep": ("pipeline", None),
    "FunctionStep": ("pipeline", None),
    "GlobStep": ("pipeline", None),
    "PipelineResult": ("pipeline", None),
    "load_pipeline": ("pipeline", None),
    "register_step": ("pipeline", None),
    # --- geoai.geoai (classes/functions defined in geoai.py) ---
    "LeafMap": ("geoai", None),
    "Map": ("geoai", None),
    "create_vector_data": ("geoai", None),
    "edit_vector_data": ("geoai", None),
    # --- geoai.map_widgets ---
    "DINOv3GUI": ("map_widgets", None),
    "moondream_gui": ("map_widgets", None),
    # --- geoai.classify ---
    "classify_image": ("classify", None),
    "classify_images": ("classify", None),
    "train_classifier": ("classify", None),
    # --- geoai.download ---
    "download_naip": ("download", None),
    "download_overture_buildings": ("download", None),
    "download_pc_stac_item": ("download", None),
    "extract_building_stats": ("download", None),
    "get_overture_data": ("download", None),
    "pc_collection_list": ("download", None),
    "pc_item_asset_list": ("download", None),
    "pc_stac_download": ("download", None),
    "pc_stac_search": ("download", None),
    "read_pc_item_asset": ("download", None),
    "view_pc_item": ("download", None),
    "view_pc_items": ("download", None),
    # --- geoai.extract ---
    "CustomDataset": ("extract", None),
    "ObjectDetector": ("extract", None),
    "BuildingFootprintExtractor": ("extract", None),
    "CarDetector": ("extract", None),
    "ShipDetector": ("extract", None),
    "SolarPanelDetector": ("extract", None),
    "ParkingSplotDetector": ("extract", None),
    "AgricultureFieldDelineator": ("extract", None),
    # --- geoai.hf ---
    "get_model_config": ("hf", None),
    "get_model_input_channels": ("hf", None),
    "image_segmentation": ("hf", None),
    "mask_generation": ("hf", None),
    # --- geoai.segment ---
    "BoundingBox": ("segment", None),
    "DetectionResult": ("segment", None),
    "GroundedSAM": ("segment", None),
    "CLIPSegmentation": ("segment", None),
    # --- geoai.train ---
    "COCODetectionDataset": ("train", None),
    "evaluate_coco_metrics": ("train", None),
    "get_instance_segmentation_model": ("train", None),
    "instance_segmentation": ("train", None),
    "instance_segmentation_batch": ("train", None),
    "instance_segmentation_inference_on_geotiff": ("train", None),
    "lightly_embed_images": ("train", None),
    "lightly_train_model": ("train", None),
    "load_lightly_pretrained_model": ("train", None),
    "multiclass_detection_inference_on_geotiff": ("train", None),
    "object_detection": ("train", None),
    "object_detection_batch": ("train", None),
    "semantic_segmentation": ("train", None),
    "semantic_segmentation_batch": ("train", None),
    "train_instance_segmentation_model": ("train", None),
    "train_MaskRCNN_model": ("train", None),
    "train_segmentation_model": ("train", None),
    # --- geoai.utils ---
    "orthogonalize": ("utils", None),
    "regularization": ("utils", None),
    "hybrid_regularization": ("utils", None),
    "adaptive_regularization": ("utils", None),
    "flipnslide_augmentation": ("utils", None),
    "export_flipnslide_tiles": ("utils", None),
    "view_raster": ("utils", None),
    "view_image": ("utils", None),
    "plot_images": ("utils", None),
    "plot_masks": ("utils", None),
    "plot_batch": ("utils", None),
    "calc_stats": ("utils", None),
    "calc_iou": ("utils", None),
    "calc_segmentation_metrics": ("utils", None),
    "dict_to_rioxarray": ("utils", None),
    "dict_to_image": ("utils", None),
    "view_vector": ("utils", None),
    "view_vector_interactive": ("utils", None),
    "install_package": ("utils", None),
    "create_split_map": ("utils", None),
    "download_file": ("utils", None),
    "get_raster_info": ("utils", None),
    "get_raster_stats": ("utils", None),
    "print_raster_info": ("utils", None),
    "get_raster_info_gdal": ("utils", None),
    "get_vector_info": ("utils", None),
    "print_vector_info": ("utils", None),
    "get_vector_info_ogr": ("utils", None),
    "analyze_vector_attributes": ("utils", None),
    "visualize_vector_by_attribute": ("utils", None),
    "clip_raster_by_bbox": ("utils", None),
    "raster_to_vector": ("utils", None),
    "raster_to_vector_batch": ("utils", None),
    "vector_to_raster": ("utils", None),
    "batch_vector_to_raster": ("utils", None),
    "get_default_augmentation_transforms": ("utils", None),
    "export_geotiff_tiles": ("utils", None),
    "export_geotiff_tiles_batch": ("utils", None),
    "display_training_tiles": ("utils", None),
    "display_image_with_vector": ("utils", None),
    "create_overview_image": ("utils", None),
    "export_tiles_to_geojson": ("utils", None),
    "export_training_data": ("utils", None),
    "masks_to_vector": ("utils", None),
    "read_vector": ("utils", None),
    "read_raster": ("utils", None),
    "temp_file_path": ("utils", None),
    "region_groups": ("utils", None),
    "add_geometric_properties": ("utils", None),
    "inspect_pth_file": ("utils", None),
    "try_common_architectures": ("utils", None),
    "mosaic_geotiffs": ("utils", None),
    "download_model_from_hf": ("utils", None),
    "regularize": ("utils", None),
    "vector_to_geojson": ("utils", None),
    "geojson_to_coords": ("utils", None),
    "coords_to_xy": ("utils", None),
    "boxes_to_vector": ("utils", None),
    "rowcol_to_xy": ("utils", None),
    "bbox_to_xy": ("utils", None),
    "geojson_to_xy": ("utils", None),
    "write_colormap": ("utils", None),
    "plot_performance_metrics": ("utils", None),
    "get_device": ("utils", None),
    "plot_prediction_comparison": ("utils", None),
    "get_raster_resolution": ("utils", None),
    "stack_bands": ("utils", None),
    "empty_cache": ("utils", None),
    "smooth_vector": ("utils", None),
    "calc_f1_score": ("utils", None),
    # --- geoai.landcover_utils ---
    "export_landcover_tiles": ("landcover_utils", None),
    # --- geoai.landcover_train ---
    "FocalLoss": ("landcover_train", None),
    "LandcoverCrossEntropyLoss": ("landcover_train", None),
    "landcover_iou": ("landcover_train", None),
    "get_landcover_loss_function": ("landcover_train", None),
    "compute_class_weights": ("landcover_train", None),
    "train_segmentation_landcover": ("landcover_train", None),
    "evaluate_sparse_iou": ("landcover_train", None),
    # --- geoai.dinov3 ---
    "DINOv3GeoProcessor": ("dinov3", None),
    "analyze_image_patches": ("dinov3", None),
    "create_similarity_map": ("dinov3", None),
    # --- geoai.timm_train ---
    "get_timm_model": ("timm_train", None),
    "modify_first_conv_for_channels": ("timm_train", None),
    "TimmClassifier": ("timm_train", None),
    "RemoteSensingDataset": ("timm_train", None),
    "train_timm_classifier": ("timm_train", None),
    "predict_with_timm": ("timm_train", None),
    "list_timm_models": ("timm_train", None),
    # --- geoai.recognize ---
    "ImageDataset": ("recognize", None),
    "load_image_dataset": ("recognize", None),
    "train_image_classifier": ("recognize", None),
    "predict_images": ("recognize", None),
    "evaluate_classifier": ("recognize", None),
    "plot_classification_history": ("recognize", "plot_training_history"),
    "plot_confusion_matrix": ("recognize", None),
    "plot_predictions": ("recognize", None),
    # --- geoai.timm_segment ---
    "TimmSegmentationModel": ("timm_segment", None),
    "SegmentationDataset": ("timm_segment", None),
    "train_timm_segmentation": ("timm_segment", None),
    "predict_segmentation": ("timm_segment", None),
    "train_timm_segmentation_model": ("timm_segment", None),
    "timm_semantic_segmentation": ("timm_segment", None),
    "push_timm_model_to_hub": ("timm_segment", None),
    "timm_segmentation_from_hub": ("timm_segment", None),
    # --- geoai.timm_regress ---
    "PixelRegressionModel": ("timm_regress", None),
    "PixelRegressionDataset": ("timm_regress", None),
    "create_regression_tiles": ("timm_regress", None),
    "train_pixel_regressor": ("timm_regress", None),
    "predict_raster": ("timm_regress", None),
    "evaluate_regression": ("timm_regress", None),
    "plot_regression_comparison": ("timm_regress", None),
    "plot_scatter": ("timm_regress", None),
    "plot_training_history": ("timm_regress", None),
    "visualize_prediction": ("timm_regress", None),
    "plot_regression_results": ("timm_regress", None),
    # Backward compatibility aliases
    "TimmRegressor": ("timm_regress", None),
    "RegressionDataset": ("timm_regress", None),
    "train_timm_regressor": ("timm_regress", None),
    "create_regression_patches": ("timm_regress", None),
    # --- geoai.tools ---
    "clean_segmentation_mask": ("tools", None),
    "clean_raster": ("tools", None),
    "clean_raster_batch": ("tools", None),
    "compare_masks": ("tools", None),
    "super_resolution": ("tools", None),
    # --- geoai.onnx ---
    "ONNXGeoModel": ("onnx", None),
    "export_to_onnx": ("onnx", None),
    "onnx_semantic_segmentation": ("onnx", None),
    "onnx_image_classification": ("onnx", None),
    # --- geoai.moondream ---
    "MoondreamGeo": ("moondream", None),
    "moondream_caption": ("moondream", None),
    "moondream_query": ("moondream", None),
    "moondream_detect": ("moondream", None),
    "moondream_point": ("moondream", None),
    "moondream_caption_sliding_window": ("moondream", None),
    "moondream_query_sliding_window": ("moondream", None),
    "moondream_detect_sliding_window": ("moondream", None),
    "moondream_point_sliding_window": ("moondream", None),
    # --- geoai.prithvi ---
    "PrithviProcessor": ("prithvi", None),
    "get_available_prithvi_models": ("prithvi", None),
    "load_prithvi_model": ("prithvi", None),
    "prithvi_inference": ("prithvi", None),
    # --- geoai.change_detection ---
    "ChangeStarDetection": ("change_detection", None),
    "changestar_detect": ("change_detection", None),
    "list_changestar_models": ("change_detection", None),
    # --- geoai.canopy ---
    "CanopyHeightEstimation": ("canopy", None),
    "canopy_height_estimation": ("canopy", None),
    "list_canopy_models": ("canopy", None),
    # --- geoai.tessera ---
    "tessera_download": ("tessera", None),
    "tessera_fetch_embeddings": ("tessera", None),
    "tessera_coverage": ("tessera", None),
    "tessera_visualize_rgb": ("tessera", None),
    "tessera_tile_count": ("tessera", None),
    "tessera_available_years": ("tessera", None),
    "tessera_sample_points": ("tessera", None),
    # --- geoai.embeddings ---
    "list_embedding_datasets": ("embeddings", None),
    "load_embedding_dataset": ("embeddings", None),
    "get_embedding_info": ("embeddings", None),
    "extract_patch_embeddings": ("embeddings", None),
    "extract_pixel_embeddings": ("embeddings", None),
    "visualize_embeddings": ("embeddings", None),
    "plot_embedding_vector": ("embeddings", None),
    "plot_embedding_raster": ("embeddings", None),
    "cluster_embeddings": ("embeddings", None),
    "embedding_similarity": ("embeddings", None),
    "train_embedding_classifier": ("embeddings", None),
    "compare_embeddings": ("embeddings", None),
    "embedding_to_geotiff": ("embeddings", None),
    "download_google_satellite_embedding": ("embeddings", None),
    "EMBEDDING_DATASETS": ("embeddings", None),
    # --- geoai.object_detect ---
    "NWPU_VHR10_CLASSES": ("object_detect", None),
    "download_nwpu_vhr10": ("object_detect", None),
    "download_nwpu_vhr10_model": ("object_detect", None),
    "prepare_nwpu_vhr10": ("object_detect", None),
    "train_multiclass_detector": ("object_detect", None),
    "multiclass_detection": ("object_detect", None),
    "detections_to_geodataframe": ("object_detect", None),
    "visualize_multiclass_detections": ("object_detect", None),
    "evaluate_multiclass_detector": ("object_detect", None),
    # --- geoai.water ---
    "segment_water": ("water", None),
    "BAND_ORDER_PRESETS": ("water", None),
}

# Submodules that can be imported via `from geoai import <submodule>`
_LAZY_SUBMODULES = {
    "pipeline",
    "tools",
    "utils",
    "classify",
    "download",
    "extract",
    "hf",
    "segment",
    "train",
    "agents",
    "landcover_utils",
    "landcover_train",
}


def __getattr__(name):
    """Lazily resolve symbols and submodules on first access."""
    # Lazy submodule imports (e.g. `from geoai import tools`)
    if name in _LAZY_SUBMODULES:
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except (ImportError, OSError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} "
                f"(failed to import {__name__}.{name}: {exc})"
            ) from exc
        globals()[name] = mod
        return mod

    # Lazy symbol imports (e.g. `geoai.semantic_segmentation`)
    if name in _LAZY_SYMBOL_MAP:
        module_rel, original_name = _LAZY_SYMBOL_MAP[name]
        attr_name = original_name if original_name is not None else name
        try:
            mod = importlib.import_module(f".{module_rel}", __name__)
        except (ImportError, OSError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} "
                f"(failed to import {__name__}.{module_rel}: {exc})"
            ) from exc
        try:
            val = getattr(mod, attr_name)
        except AttributeError:
            raise AttributeError(
                f"module {__name__}.{module_rel!r} has no attribute {attr_name!r}"
            )
        globals()[name] = val
        return val

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List all public attributes including lazily-loaded symbols."""
    module_attrs = list(globals().keys())
    return sorted(set(module_attrs) | set(_LAZY_SYMBOL_MAP.keys()) | _LAZY_SUBMODULES)


__all__ = [
    "set_proj_lib_path",
    *_LAZY_SYMBOL_MAP.keys(),
]
