"""Object detection operations.

Wraps geoai object detection functions (Mask R-CNN and friends via
``multiclass_detection`` / ``train_MaskRCNN_model``) for CLI consumption.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

DETECTION_MODELS = [
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
]

INPUT_FORMATS = ["directory", "coco", "yolo"]


def _detection_raster_to_vector(raster_path: str, output_vector: str) -> None:
    """Polygonize multiclass detection output from instance IDs (band 2).

    ``multiclass_detection`` writes class labels to band 1 and per-instance IDs
    to band 2. Polygonizing band 2 keeps touching same-class detections as
    separate polygons while attaching each instance's class label from band 1.
    Falls back to ``geoai.utils.raster_to_vector`` for single-band rasters.
    """
    import numpy as np
    import rasterio
    from rasterio.features import shapes

    with rasterio.open(raster_path) as src:
        band_count = src.count
        if band_count < 2:
            from geoai.utils import raster_to_vector

            raster_to_vector(raster_path, output_vector)
            return

        class_band = src.read(1)
        instance_band = src.read(2)
        transform = src.transform
        crs = src.crs

    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    geoms_by_instance: Dict[int, list] = {}
    for geom_dict, val in shapes(
        instance_band.astype(np.int32),
        mask=(instance_band > 0),
        transform=transform,
    ):
        inst_id = int(val)
        if inst_id <= 0:
            continue
        geoms_by_instance.setdefault(inst_id, []).append(shapely_shape(geom_dict))

    # Aggregate dominant positive class label per instance ID in a single pass
    # over the raster rather than rescanning both full bands per instance.
    valid_mask = (instance_band > 0) & (class_band > 0)
    inst_vals = instance_band[valid_mask].astype(np.int64, copy=False)
    cls_vals = class_band[valid_mask].astype(np.int64, copy=False)
    dominant_class_by_instance: Dict[int, int] = {}
    if inst_vals.size > 0:
        pairs = np.empty(inst_vals.size, dtype=[("inst", np.int64), ("cls", np.int64)])
        pairs["inst"] = inst_vals
        pairs["cls"] = cls_vals
        unique_pairs, pair_counts = np.unique(pairs, return_counts=True)
        best_count: Dict[int, int] = {}
        for (i_id, c_id), cnt in zip(unique_pairs.tolist(), pair_counts.tolist()):
            if cnt > best_count.get(i_id, -1):
                best_count[i_id] = cnt
                dominant_class_by_instance[i_id] = c_id

    records = []
    for inst_id, parts in sorted(geoms_by_instance.items()):
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        class_id = dominant_class_by_instance.get(inst_id, 0)
        records.append(
            {
                "instance_id": inst_id,
                "class": class_id,
                "geometry": geom,
            }
        )

    if records:
        gdf = gpd.GeoDataFrame(records, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            {"instance_id": [], "class": []},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    ext = os.path.splitext(output_vector)[1].lower()
    driver_map = {
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".gpkg": "GPKG",
        ".shp": "ESRI Shapefile",
    }
    driver = driver_map.get(ext)
    if driver:
        gdf.to_file(output_vector, driver=driver)
    else:
        gdf.to_file(output_vector)


def run_detection(
    raster: str,
    model_path: str,
    num_classes: int,
    output_vector: Optional[str] = None,
    output_raster: Optional[str] = None,
    band_indexes: Optional[List[int]] = None,
    confidence_threshold: float = 0.5,
    chip_size: int = 512,
    overlap: float = 0.25,
) -> Dict[str, Any]:
    """Run object detection on a raster image.

    Args:
        raster: Input raster file path.
        model_path: Path to trained model checkpoint.
        num_classes: Number of detection classes (including background).
        output_vector: Optional output vector path for detected objects.
        output_raster: Optional output raster path for detection mask.
        band_indexes: Band indices to use (0-indexed).
        confidence_threshold: Minimum confidence for detections.
        chip_size: Processing chip size in pixels.
        overlap: Overlap fraction (0-1) or overlap in pixels between chips.

    Returns:
        Result dict with output paths and detection count.

    Raises:
        FileNotFoundError: If input files do not exist.
        ValueError: If neither output_vector nor output_raster is provided.
    """
    raster = os.path.abspath(raster)
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(raster):
        raise FileNotFoundError(f"Raster file not found: {raster}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not output_vector and not output_raster:
        raise ValueError("Specify at least one of output_vector or output_raster")

    if output_vector:
        output_vector = os.path.abspath(output_vector)
        os.makedirs(os.path.dirname(output_vector) or ".", exist_ok=True)
    if output_raster:
        output_raster = os.path.abspath(output_raster)
        os.makedirs(os.path.dirname(output_raster) or ".", exist_ok=True)

    from geoai.object_detect import multiclass_detection

    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    overlap_px = (
        int(chip_size * overlap)
        if isinstance(overlap, float) and 0.0 < overlap < 1.0
        else int(overlap)
    )
    if overlap_px >= chip_size:
        raise ValueError(
            f"Resolved overlap ({overlap_px}px) must be smaller than chip_size ({chip_size}px)"
        )

    temp_raster: Optional[str] = None
    temp_input_raster: Optional[str] = None
    work_dir = os.path.dirname(output_raster or output_vector or ".") or "."
    if output_raster:
        target_raster = output_raster
    else:
        fd, temp_raster = tempfile.mkstemp(
            suffix=".tif",
            prefix="geoai_det_",
            dir=work_dir,
        )
        os.close(fd)
        target_raster = temp_raster

    try:
        inference_input = raster
        kwargs: Dict[str, Any] = {
            "output_path": target_raster,
            "model_path": model_path,
            "num_classes": num_classes,
            "confidence_threshold": confidence_threshold,
            "window_size": chip_size,
            "overlap": overlap_px,
        }
        if band_indexes is not None:
            from cli_anything.geoai.core.raster import extract_band_subset

            temp_input_raster = extract_band_subset(
                raster, band_indexes, temp_dir=work_dir
            )
            inference_input = temp_input_raster
            kwargs["num_channels"] = len(band_indexes)

        kwargs["input_path"] = inference_input
        multiclass_detection(**kwargs)

        if output_vector and os.path.isfile(target_raster):
            _detection_raster_to_vector(target_raster, output_vector)
    finally:
        for tmp in (temp_raster, temp_input_raster):
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    result: Dict[str, Any] = {
        "input": raster,
        "model": model_path,
        "num_classes": num_classes,
        "confidence_threshold": confidence_threshold,
    }

    if output_vector and os.path.isfile(output_vector):
        import geopandas as gpd

        gdf = gpd.read_file(output_vector)
        result["output_vector"] = output_vector
        result["detection_count"] = len(gdf)

    if output_raster and os.path.isfile(output_raster):
        result["output_raster"] = output_raster
        result["output_raster_size_bytes"] = os.path.getsize(output_raster)

    return result


def train_detector(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    model_name: str = "maskrcnn_resnet50_fpn",
    num_classes: int = 2,
    input_format: str = "directory",
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 0.005,
) -> Dict[str, Any]:
    """Train an object detection model.

    Args:
        images_dir: Directory with training images.
        labels_dir: Directory with training labels (masks or annotations).
        output_dir: Output directory for model checkpoints.
        model_name: Detection model architecture name.
        num_classes: Number of object classes (including background).
        input_format: Label format (directory, coco, yolo).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.

    Returns:
        Result dict with model path and training info.

    Raises:
        FileNotFoundError: If input directories do not exist.
        ValueError: If model name or format is invalid.
    """
    images_dir = os.path.abspath(images_dir)
    labels_dir = os.path.abspath(labels_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    if model_name not in DETECTION_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: {', '.join(DETECTION_MODELS)}"
        )
    if input_format not in INPUT_FORMATS:
        raise ValueError(
            f"Unknown input format: {input_format}. "
            f"Choose from: {', '.join(INPUT_FORMATS)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    from geoai.train import train_MaskRCNN_model

    result = train_MaskRCNN_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        model_name=model_name,
        num_classes=num_classes,
        input_format=input_format,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "output_dir": output_dir,
        "model_name": model_name,
        "num_classes": num_classes,
        "epochs": epochs,
        "result": str(result) if result else "completed",
    }


def list_models() -> List[Dict[str, str]]:
    """List available detection model architectures.

    Returns:
        List of model info dicts.
    """
    return [{"name": m} for m in DETECTION_MODELS]


def list_input_formats() -> List[Dict[str, str]]:
    """List supported input formats.

    Returns:
        List of format info dicts.
    """
    descriptions = {
        "directory": "One mask image per training image, matching filenames",
        "coco": "COCO JSON annotation format with instance polygons",
        "yolo": "YOLO format with .txt label files (class x y w h)",
    }
    return [{"name": f, "description": descriptions.get(f, "")} for f in INPUT_FORMATS]
