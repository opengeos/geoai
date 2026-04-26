"""Object detection operations.

Wraps geoai object detection functions: Mask R-CNN, GeoDeep, RF-DETR
for CLI consumption.
"""

import os
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
        overlap: Overlap fraction between chips.

    Returns:
        Result dict with output paths and detection count.

    Raises:
        FileNotFoundError: If input files do not exist.
    """
    raster = os.path.abspath(raster)
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(raster):
        raise FileNotFoundError(f"Raster file not found: {raster}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if output_vector:
        output_vector = os.path.abspath(output_vector)
        os.makedirs(os.path.dirname(output_vector) or ".", exist_ok=True)
    if output_raster:
        output_raster = os.path.abspath(output_raster)
        os.makedirs(os.path.dirname(output_raster) or ".", exist_ok=True)

    from geoai import multiclass_detection

    kwargs = {
        "raster": raster,
        "model_path": model_path,
        "num_classes": num_classes,
        "confidence_threshold": confidence_threshold,
        "chip_size": chip_size,
        "overlap": overlap,
    }
    if output_vector:
        kwargs["output_vector"] = output_vector
    if output_raster:
        kwargs["output_raster"] = output_raster
    if band_indexes is not None:
        kwargs["band_indexes"] = band_indexes

    multiclass_detection(**kwargs)

    result = {
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

    from geoai import train_MaskRCNN_model

    result = train_MaskRCNN_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        model_name=model_name,
        num_classes=num_classes,
        input_format=input_format,
        epochs=epochs,
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
