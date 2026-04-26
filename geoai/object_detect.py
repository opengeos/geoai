"""High-level functions for multi-class object detection.

This module provides convenience functions for training, evaluating, and
running inference with Mask R-CNN models on COCO-format datasets, including
support for the NWPU-VHR-10 remote sensing benchmark.
"""

import json
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import torch

from .train import (
    DETECTION_MODELS,
    COCODetectionDataset,
    collate_fn,
    evaluate_coco_metrics,
    get_detection_model,
    get_transform,
    model_has_masks,
    multiclass_detection_inference_on_geotiff,
    train_MaskRCNN_model,
)
from .utils import download_file, get_device

# NWPU-VHR-10 class definitions
NWPU_VHR10_CLASSES = [
    "background",
    "airplane",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "harbor",
    "bridge",
    "vehicle",
]

NWPU_VHR10_URL = "https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip"
NWPU_VHR10_HF_REPO = "giswqs/nwpu-vhr10-maskrcnn"
NWPU_VHR10_HF_FILENAME = "best_model.pth"


def download_nwpu_vhr10(
    output_dir: str = "NWPU-VHR-10",
    overwrite: bool = False,
) -> str:
    """Download and extract the NWPU-VHR-10 dataset.

    The NWPU-VHR-10 dataset contains 800 VHR (Very High Resolution) remote
    sensing images with 10 object classes: airplane, ship, storage_tank,
    baseball_diamond, tennis_court, basketball_court, ground_track_field,
    harbor, bridge, and vehicle. It has 3,775 annotated instances in COCO
    format (bounding boxes and instance segmentation masks).

    Args:
        output_dir (str): Path for the downloaded ZIP file and extracted
            dataset directory. Defaults to "NWPU-VHR-10".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.

    Returns:
        str: Path to the extracted dataset directory.
    """
    zip_path = output_dir + ".zip"
    data_path = download_file(NWPU_VHR10_URL, output_path=zip_path, overwrite=overwrite)
    return data_path


def download_nwpu_vhr10_model(
    repo_id: str = NWPU_VHR10_HF_REPO,
    filename: str = NWPU_VHR10_HF_FILENAME,
) -> str:
    """Download the pretrained NWPU-VHR-10 Mask R-CNN model from HuggingFace Hub.

    Downloads a Mask R-CNN (ResNet-50 FPN) model trained on the NWPU-VHR-10
    dataset for 10-class object detection on remote sensing imagery.

    The model achieves the following performance on the validation set:
        - mAP@0.5: 0.709
        - mAP@0.75: 0.518
        - mAP@[0.5:0.95]: 0.459

    Args:
        repo_id (str): HuggingFace Hub repository ID.
            Defaults to "giswqs/nwpu-vhr10-maskrcnn".
        filename (str): Model filename in the repository.
            Defaults to "best_model.pth".

    Returns:
        str: Local path to the downloaded model weights file.

    Example:
        >>> import geoai
        >>> model_path = geoai.download_nwpu_vhr10_model()
        >>> print(model_path)  # local cache path
    """
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(f"Model downloaded to: {model_path}")
    return model_path


def _parse_nwpu_gt_file(gt_path: str) -> List[Dict]:
    """Parse a NWPU-VHR-10 ground truth text file.

    Each line has format: (x1,y1),(x2,y2),class_id
    where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner.

    Args:
        gt_path (str): Path to the ground truth text file.

    Returns:
        List of dicts with keys: bbox (x1,y1,w,h in COCO format), category_id.
    """
    import re

    annotations = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse format: (x1,y1),(x2,y2),class_id
            match = re.match(r"\((\d+),(\d+)\),\((\d+),(\d+)\),(\d+)", line)
            if match:
                x1, y1, x2, y2, cat_id = [int(g) for g in match.groups()]
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    annotations.append(
                        {
                            "bbox": [x1, y1, w, h],
                            "category_id": cat_id,
                        }
                    )
    return annotations


def _convert_nwpu_to_coco(images_dir: str, gt_dir: str, output_path: str) -> str:
    """Convert NWPU-VHR-10 text annotations to COCO JSON format.

    Args:
        images_dir (str): Path to the positive image set directory.
        gt_dir (str): Path to the ground truth directory.
        output_path (str): Path to save the COCO JSON file.

    Returns:
        str: Path to the created COCO JSON file.
    """
    from PIL import Image as PILImage

    categories = [
        {"id": i + 1, "name": name}
        for i, name in enumerate(NWPU_VHR10_CLASSES[1:])  # Skip background
    ]

    images = []
    annotations = []
    ann_id = 1

    # Get all ground truth files
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".txt")])

    for gt_file in gt_files:
        # Derive image filename from gt filename (e.g., "001.txt" -> "001.jpg")
        base_name = os.path.splitext(gt_file)[0]

        # Find the corresponding image file
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp"]:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # Get image dimensions
        img = PILImage.open(img_path)
        width, height = img.size
        img_id = int(base_name)

        images.append(
            {
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height,
            }
        )

        # Parse ground truth
        gt_path = os.path.join(gt_dir, gt_file)
        gt_anns = _parse_nwpu_gt_file(gt_path)

        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            # Create polygon segmentation from bbox
            segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_path, "w") as f:
        json.dump(coco_data, f)

    return output_path


def prepare_nwpu_vhr10(
    data_dir: str,
    output_dir: Optional[str] = None,
    val_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Prepare NWPU-VHR-10 dataset for training.

    Converts the original text-based annotations to COCO JSON format,
    then splits the dataset into train/val sets. The original dataset uses
    text files with ``(x1,y1),(x2,y2),class_id`` per line for bounding boxes.

    Note: Only images with at least one annotation are included in the
    train/val splits. The 150 "negative" images in the NWPU-VHR-10 dataset
    (those without any target objects) are excluded from the splits.

    Args:
        data_dir (str): Path to the extracted NWPU-VHR-10 directory.
        output_dir (str, optional): Output directory for organized data.
            If None, creates files alongside the original data.
        val_split (float): Fraction of data for validation. Defaults to 0.2.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Dict with keys:
            - 'images_dir': Path to images directory
            - 'annotations_path': Path to the full annotations JSON
            - 'train_annotations': Path to train split annotations JSON
            - 'val_annotations': Path to val split annotations JSON
            - 'train_image_ids': List of training image IDs
            - 'val_image_ids': List of validation image IDs
            - 'class_names': List of class names (including background)
            - 'num_classes': Number of classes (including background)
    """
    from sklearn.model_selection import train_test_split

    if output_dir is None:
        output_dir = data_dir

    os.makedirs(output_dir, exist_ok=True)

    # Handle nested directory structure (NWPU-VHR-10/NWPU-VHR-10/)
    actual_dir = data_dir
    nested = os.path.join(data_dir, "NWPU-VHR-10")
    if os.path.isdir(nested):
        actual_dir = nested

    # Find images directory
    images_dir = None
    for candidate in ["positive image set", "positive_image_set", "images"]:
        path = os.path.join(actual_dir, candidate)
        if os.path.isdir(path):
            images_dir = path
            break

    if images_dir is None:
        raise FileNotFoundError(
            f"Could not find images directory in {actual_dir}. "
            "Expected 'positive image set' directory."
        )

    # Find ground truth directory
    gt_dir = None
    for candidate in ["ground truth", "ground_truth", "annotations", "labels"]:
        path = os.path.join(actual_dir, candidate)
        if os.path.isdir(path):
            gt_dir = path
            break

    # Check if COCO JSON already exists
    annotations_path = os.path.join(output_dir, "annotations.json")
    if os.path.exists(annotations_path):
        logger.info(f"Using existing COCO annotations: {annotations_path}")
    elif gt_dir is not None:
        # Convert text annotations to COCO JSON
        logger.info("Converting NWPU-VHR-10 text annotations to COCO JSON format...")
        _convert_nwpu_to_coco(images_dir, gt_dir, annotations_path)
        logger.info(f"COCO annotations saved to: {annotations_path}")
    else:
        # Look for existing JSON annotations
        for candidate_ann in [
            "annotations.json",
            "instances.json",
        ]:
            path = os.path.join(actual_dir, candidate_ann)
            if os.path.isfile(path):
                annotations_path = path
                break

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Could not find or create annotations for {data_dir}.")

    # Load annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Get image IDs that have annotations
    annotated_image_ids = set()
    for ann in coco_data.get("annotations", []):
        annotated_image_ids.add(ann["image_id"])

    image_ids = sorted(list(annotated_image_ids))

    # Split into train and validation
    train_ids, val_ids = train_test_split(
        image_ids, test_size=val_split, random_state=seed
    )

    # Create split annotation files
    train_ann_path = os.path.join(output_dir, "train_annotations.json")
    val_ann_path = os.path.join(output_dir, "val_annotations.json")

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    all_images = {img["id"]: img for img in coco_data["images"]}

    train_coco = {
        "images": [all_images[img_id] for img_id in train_ids if img_id in all_images],
        "annotations": [
            ann for ann in coco_data["annotations"] if ann["image_id"] in train_ids_set
        ],
        "categories": coco_data.get("categories", []),
    }

    val_coco = {
        "images": [all_images[img_id] for img_id in val_ids if img_id in all_images],
        "annotations": [
            ann for ann in coco_data["annotations"] if ann["image_id"] in val_ids_set
        ],
        "categories": coco_data.get("categories", []),
    }

    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f)

    with open(val_ann_path, "w") as f:
        json.dump(val_coco, f)

    class_names = NWPU_VHR10_CLASSES

    logger.info(f"Dataset prepared:")
    logger.info(f"  Images directory: {images_dir}")
    logger.info(f"  Total annotated images: {len(image_ids)}")
    logger.info(f"  Total annotations: {len(coco_data['annotations'])}")
    logger.info(f"  Training images: {len(train_ids)}")
    logger.info(f"  Validation images: {len(val_ids)}")
    logger.info(f"  Classes: {class_names[1:]}")

    return {
        "images_dir": images_dir,
        "annotations_path": annotations_path,
        "train_annotations": train_ann_path,
        "val_annotations": val_ann_path,
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
        "class_names": class_names,
        "num_classes": len(class_names),
    }


def train_multiclass_detector(
    images_dir: str,
    annotations_path: str,
    output_dir: str,
    model_name: str = "fasterrcnn_resnet50_fpn_v2",
    class_names: Optional[List[str]] = None,
    num_channels: int = 3,
    batch_size: int = 4,
    num_epochs: int = 50,
    learning_rate: float = 0.005,
    val_split: float = 0.2,
    seed: int = 42,
    pretrained: bool = True,
    pretrained_model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """Train a multi-class object detection model using COCO-format annotations.

    Supports multiple torchvision detection architectures including
    Faster R-CNN, RetinaNet, FCOS, and Mask R-CNN.

    Args:
        images_dir (str): Directory containing training images.
        annotations_path (str): Path to COCO-format annotations JSON file.
        output_dir (str): Directory for model outputs.
        model_name (str): Detection model architecture. One of
            ``"fasterrcnn_resnet50_fpn_v2"`` (default),
            ``"fasterrcnn_mobilenet_v3_large_fpn"``,
            ``"retinanet_resnet50_fpn_v2"``,
            ``"fcos_resnet50_fpn"``, or
            ``"maskrcnn_resnet50_fpn"``.
        class_names (list, optional): List of class names including background.
            If None, extracted from annotations.
        num_channels (int): Number of image channels. Defaults to 3.
        batch_size (int): Training batch size. Defaults to 4.
        num_epochs (int): Number of training epochs. Defaults to 50.
        learning_rate (float): Initial learning rate. Defaults to 0.005.
        val_split (float): Validation split fraction. Defaults to 0.2.
        seed (int): Random seed. Defaults to 42.
        pretrained (bool): Whether to use pretrained backbone. Defaults to True.
        pretrained_model_path (str, optional): Path to pretrained model.
        device (torch.device, optional): Compute device.
        num_workers (int, optional): Number of data loading workers.
        verbose (bool): Whether to print progress. Defaults to True.

    Returns:
        str: Path to the best model checkpoint.
    """
    # Determine num_classes from annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])
    num_classes = len(categories) + 1  # +1 for background

    if class_names is None:
        class_names = ["background"] + [
            cat["name"] for cat in sorted(categories, key=lambda c: c["id"])
        ]

    if verbose:
        logger.info(f"Training {model_name} with {num_classes} classes")
        logger.info(f"  Classes: {class_names[1:]}")

    # Save class names and model info to output directory for later use
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    class_info = {
        "class_names": class_names,
        "num_classes": num_classes,
        "model_name": model_name,
    }
    with open(os.path.join(output_dir, "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)

    train_MaskRCNN_model(
        images_dir=images_dir,
        labels_dir=annotations_path,
        output_dir=output_dir,
        input_format="coco_detection",
        num_channels=num_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_model_path=pretrained_model_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        seed=seed,
        device=device,
        num_workers=num_workers,
        verbose=verbose,
        model_name=model_name,
    )

    return os.path.join(output_dir, "best_model.pth")


def multiclass_detection(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    num_classes: int = 11,
    class_names: Optional[List[str]] = None,
    window_size: int = 512,
    overlap: int = 256,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    batch_size: int = 4,
    num_channels: int = 3,
    device: Optional[torch.device] = None,
    repo_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[str, float, List[Dict]]:
    """Perform multi-class object detection on a GeoTIFF or image.

    Loads a trained detection model and runs inference using a sliding
    window approach. Outputs a 2-band raster with class labels and
    instance IDs.

    If ``model_path`` is None, the pretrained NWPU-VHR-10 model is
    automatically downloaded from HuggingFace Hub with default
    ``num_classes=11`` and ``class_names`` set to NWPU-VHR-10 classes.

    Args:
        input_path (str): Path to input image (GeoTIFF, JPEG, PNG, etc.).
        output_path (str): Path to save output raster.
        model_path (str, optional): Path to trained model weights (.pth file).
            If None, downloads the pretrained NWPU-VHR-10 model from
            HuggingFace Hub.
        model_name (str, optional): Detection model architecture name. If
            None, auto-detected from ``class_info.json`` sidecar or
            checkpoint keys. Falls back to ``"maskrcnn_resnet50_fpn"``
            for backward compatibility.
        num_classes (int): Number of classes including background. Defaults
            to 11 (NWPU-VHR-10).
        class_names (list, optional): List of class names (index 0 = background).
        window_size (int): Sliding window size. Defaults to 512.
        overlap (int): Window overlap in pixels. Defaults to 256.
        confidence_threshold (float): Minimum detection score. Defaults to 0.5.
        nms_threshold (float): IoU threshold for NMS. Defaults to 0.3.
        batch_size (int): Inference batch size. Defaults to 4.
        num_channels (int): Number of input image channels. Defaults to 3.
        device (torch.device, optional): Compute device.
        repo_id (str, optional): HuggingFace Hub repository ID for
            downloading the model. Defaults to ``"giswqs/nwpu-vhr10-maskrcnn"``.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of (output_path, inference_time, detections_list) where each
        detection is a dict with mask, score, box, and label.
    """
    import rasterio

    if device is None:
        device = get_device()

    # Handle pretrained model download
    use_pretrained = model_path is None
    if use_pretrained:
        hf_repo = repo_id or NWPU_VHR10_HF_REPO
        model_path = download_nwpu_vhr10_model(
            repo_id=hf_repo, filename=NWPU_VHR10_HF_FILENAME
        )
        if class_names is None:
            class_names = NWPU_VHR10_CLASSES
        num_classes = len(NWPU_VHR10_CLASSES)
        if model_name is None:
            model_name = "maskrcnn_resnet50_fpn"

    # Convert non-GeoTIFF images to temporary GeoTIFF for processing
    temp_tif = None
    if not input_path.lower().endswith((".tif", ".tiff")):
        from PIL import Image as PILImage

        img = PILImage.open(input_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        temp_tif = output_path.replace(
            os.path.splitext(output_path)[1], "_temp_input.tif"
        )
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": w,
            "height": h,
            "count": img_array.shape[2] if img_array.ndim == 3 else 1,
            "crs": None,
            "transform": rasterio.transform.from_bounds(0, 0, w, h, w, h),
        }
        with rasterio.open(temp_tif, "w", **profile) as dst:
            if img_array.ndim == 3:
                for band in range(img_array.shape[2]):
                    dst.write(img_array[:, :, band], band + 1)
            else:
                dst.write(img_array, 1)
        input_path = temp_tif

    # Resolve model path (download if needed)
    if not os.path.exists(model_path):
        hf_repo = repo_id or NWPU_VHR10_HF_REPO
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=hf_repo, filename=model_path)

    # Try to load class_info.json sidecar
    class_info_path = os.path.join(os.path.dirname(model_path), "class_info.json")
    if not use_pretrained and os.path.exists(class_info_path):
        with open(class_info_path, "r") as f:
            class_info = json.load(f)
        num_classes = class_info.get("num_classes", num_classes)
        if class_names is None:
            class_names = class_info.get("class_names", class_names)
        if model_name is None:
            model_name = class_info.get("model_name", None)

    # Load checkpoint
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }

    # Auto-detect model_name from checkpoint keys if not set
    if model_name is None:
        if "roi_heads.mask_predictor.conv5_mask.weight" in state_dict:
            model_name = "maskrcnn_resnet50_fpn"
        elif "roi_heads.box_predictor.cls_score.weight" in state_dict:
            model_name = "fasterrcnn_resnet50_fpn_v2"
        elif "head.classification_head.cls_logits.weight" in state_dict:
            # Distinguish FCOS (anchor-free) from RetinaNet (anchor-based)
            # by checking for anchor_generator keys
            if any(k.startswith("anchor_generator.") for k in state_dict):
                model_name = "retinanet_resnet50_fpn_v2"
            else:
                model_name = "fcos_resnet50_fpn"
        else:
            model_name = "maskrcnn_resnet50_fpn"

    # Infer num_classes from checkpoint
    if not use_pretrained:
        rcnn_cls_key = "roi_heads.box_predictor.cls_score.weight"
        retina_cls_key = "head.classification_head.cls_logits.weight"
        if rcnn_cls_key in state_dict:
            inferred = state_dict[rcnn_cls_key].shape[0]
            if inferred != num_classes:
                num_classes = inferred
        elif retina_cls_key in state_dict:
            # For RetinaNet/FCOS: out_channels = num_anchors * num_classes
            # num_anchors is 9 for RetinaNet, 1 for FCOS
            out_channels = state_dict[retina_cls_key].shape[0]
            num_anchors = 9 if model_name == "retinanet_resnet50_fpn_v2" else 1
            inferred = out_channels // num_anchors
            if inferred != num_classes:
                num_classes = inferred

    # Load model
    model = get_detection_model(
        model_name=model_name,
        num_classes=num_classes,
        num_channels=num_channels,
        pretrained=False,
    )
    model.load_state_dict(state_dict)

    result = multiclass_detection_inference_on_geotiff(
        model=model,
        geotiff_path=input_path,
        output_path=output_path,
        class_names=class_names,
        window_size=window_size,
        overlap=overlap,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        batch_size=batch_size,
        num_channels=num_channels,
        device=device,
        **kwargs,
    )

    # Clean up temporary file
    if temp_tif and os.path.exists(temp_tif):
        os.remove(temp_tif)

    return result


def detections_to_geodataframe(
    detections: List[Dict],
    geotiff_path: str,
    class_names: Optional[List[str]] = None,
    use_mask_geometry: bool = False,
    simplify_tolerance: float = 0.0,
) -> Any:
    """Convert detections to a GeoDataFrame with geospatial coordinates.

    Converts pixel-space detections to geospatial coordinates using the
    CRS and transform from the source GeoTIFF. By default, bounding box
    rectangles are used as geometry. When ``use_mask_geometry=True``, the
    actual instance mask is vectorized into polygon geometry instead.

    Args:
        detections (list): List of detection dicts, each with keys:
            mask (np.ndarray), score (float), box (list of pixel coords),
            and optionally label (int), instance_id (int), and
            mask_offset (tuple of y, x, h, w) for compact masks.
        geotiff_path (str): Path to the source GeoTIFF (for CRS and transform).
        class_names (list, optional): List of class names (index 0 = background).
        use_mask_geometry (bool): If True, convert instance masks to polygon
            geometries using rasterio.features.shapes instead of using
            bounding boxes. Defaults to False.
        simplify_tolerance (float): Tolerance for polygon simplification
            in georeferenced units. Only used when use_mask_geometry=True.
            Set to 0 to disable simplification. Defaults to 0.0.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with columns: geometry, class_id,
        class_name, score, instance_id, area_pixels.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from shapely.geometry import Polygon

    if len(detections) == 0:
        return gpd.GeoDataFrame(
            columns=[
                "geometry",
                "class_id",
                "class_name",
                "score",
                "instance_id",
                "area_pixels",
            ]
        )

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    records = []
    for idx, det in enumerate(detections):
        bx = det["box"]
        label = det.get("label", 1)
        score = det["score"]
        instance_id = det.get("instance_id", idx + 1)
        area_pixels = int(det["mask"].sum()) if "mask" in det else 0

        geom = None

        if use_mask_geometry and "mask" in det:
            from rasterio.features import shapes as rasterio_shapes
            from rasterio.transform import Affine
            from shapely.geometry import shape
            from shapely.ops import unary_union

            mask = det["mask"]

            # Determine crop region from mask_offset (compact) or bbox
            if "mask_offset" in det:
                y_off, x_off, h, w = det["mask_offset"]
                cropped = mask[:h, :w].astype(np.uint8)
            else:
                # Use pixel bbox to crop (avoids full-image scan)
                c0, r0, c1, r1 = bx
                r0i = max(0, int(r0))
                c0i = max(0, int(c0))
                r1i = min(mask.shape[0], int(np.ceil(r1)))
                c1i = min(mask.shape[1], int(np.ceil(c1)))
                cropped = mask[r0i:r1i, c0i:c1i].astype(np.uint8)
                y_off, x_off = r0i, c0i

            if cropped.any():
                crop_transform = transform * Affine.translation(x_off, y_off)

                # Collect all polygon components and union them
                parts = []
                for geom_dict, value in rasterio_shapes(
                    cropped, mask=cropped, transform=crop_transform
                ):
                    if value == 1:
                        candidate = shape(geom_dict)
                        if candidate.is_valid and not candidate.is_empty:
                            parts.append(candidate)
                if parts:
                    merged = unary_union(parts) if len(parts) > 1 else parts[0]
                    if simplify_tolerance > 0:
                        simplified = merged.simplify(
                            simplify_tolerance, preserve_topology=True
                        )
                        merged = simplified if simplified.is_valid else merged
                    geom = merged

        # Fallback to bounding box geometry
        if geom is None:
            c0, r0, c1, r1 = bx  # (xmin, ymin, xmax, ymax) in pixel coords
            pts = []
            for c, r in [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]:
                x, y = transform * (c, r)
                pts.append((x, y))
            geom = Polygon(pts)

        name = "unknown"
        if class_names and label < len(class_names):
            name = class_names[label]
        elif class_names:
            name = f"class_{label}"

        records.append(
            {
                "geometry": geom,
                "class_id": label,
                "class_name": name,
                "score": score,
                "instance_id": instance_id,
                "area_pixels": area_pixels,
            }
        )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def visualize_multiclass_detections(
    image_path: str,
    detections: List[Dict],
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.0,
    figsize: Tuple[int, int] = (15, 10),
    output_path: Optional[str] = None,
    max_detections: int = 200,
) -> None:
    """Visualize multi-class detections overlaid on an image.

    Draws colored bounding boxes with class labels and confidence scores.

    Args:
        image_path (str): Path to the source image.
        detections (list): List of detection dicts with mask, score, box, label.
        class_names (list, optional): List of class names (index 0 = background).
        confidence_threshold (float): Minimum score to display. Defaults to 0.0.
        figsize (tuple): Figure size (width, height). Defaults to (15, 10).
        output_path (str, optional): Path to save the figure. If None, displays.
        max_detections (int): Maximum detections to display. Defaults to 200.
    """
    from PIL import Image as PILImage

    # Load image
    if image_path.lower().endswith((".tif", ".tiff")):
        import rasterio

        with rasterio.open(image_path) as src:
            image = src.read()
            if image.shape[0] >= 3:
                image = image[:3].transpose(1, 2, 0)
            else:
                image = image[0]
    else:
        image = np.array(PILImage.open(image_path).convert("RGB"))

    # Normalize for display
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Color map for classes
    cmap = plt.cm.get_cmap("tab20", 20)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)

    # Filter and sort detections
    filtered = [d for d in detections if d["score"] >= confidence_threshold]
    filtered.sort(key=lambda x: x["score"], reverse=True)
    filtered = filtered[:max_detections]

    legend_entries = {}

    for det in filtered:
        box = det["box"]
        label = det["label"]
        score = det["score"]

        color = cmap(label % 20)[:3]

        # Draw bounding box
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Class name
        name = f"class_{label}"
        if class_names and label < len(class_names):
            name = class_names[label]

        ax.text(
            box[0],
            box[1] - 5,
            f"{name}: {score:.2f}",
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
        )

        if name not in legend_entries:
            legend_entries[name] = color

    # Add legend
    if legend_entries:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=c, edgecolor=c, label=n)
            for n, c in sorted(legend_entries.items())
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.set_title(f"Detections: {len(filtered)}")
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def evaluate_multiclass_detector(
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    images_dir: str = "",
    annotations_path: str = "",
    num_classes: int = 11,
    class_names: Optional[List[str]] = None,
    num_channels: int = 3,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    num_workers: Optional[int] = None,
    repo_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a trained multi-class detection model on a dataset.

    Loads a trained model and computes COCO-style mAP metrics on the
    provided dataset.

    If ``model_path`` is None, the pretrained NWPU-VHR-10 model is
    automatically downloaded from HuggingFace Hub.

    Args:
        model_path (str, optional): Path to trained model weights.
            If None, downloads the pretrained NWPU-VHR-10 model.
        model_name (str, optional): Detection model architecture name.
            If None, auto-detected from sidecar or defaults to
            ``"maskrcnn_resnet50_fpn"``.
        images_dir (str): Directory containing evaluation images.
        annotations_path (str): Path to COCO-format annotations JSON.
        num_classes (int): Number of classes including background.
            Defaults to 11 (NWPU-VHR-10).
        class_names (list, optional): List of class names (excluding background).
        num_channels (int): Number of image channels. Defaults to 3.
        batch_size (int): Evaluation batch size. Defaults to 4.
        device (torch.device, optional): Compute device.
        num_workers (int, optional): Number of data loading workers.
        repo_id (str, optional): HuggingFace Hub repository ID for
            downloading the model. Defaults to ``"giswqs/nwpu-vhr10-maskrcnn"``.
        verbose (bool): Whether to print results. Defaults to True.

    Returns:
        Dict with mAP metrics.
    """
    import platform

    from torch.utils.data import DataLoader

    if device is None:
        device = get_device()

    # Handle pretrained model download
    if model_path is None:
        hf_repo = repo_id or NWPU_VHR10_HF_REPO
        model_path = download_nwpu_vhr10_model(
            repo_id=hf_repo, filename=NWPU_VHR10_HF_FILENAME
        )
        num_classes = len(NWPU_VHR10_CLASSES)
        if class_names is None:
            class_names = NWPU_VHR10_CLASSES[1:]  # Exclude background
        if model_name is None:
            model_name = "maskrcnn_resnet50_fpn"

    # Try to read model_name from sidecar
    if model_name is None:
        class_info_path = os.path.join(os.path.dirname(model_path), "class_info.json")
        if os.path.exists(class_info_path):
            with open(class_info_path, "r") as f:
                class_info = json.load(f)
            model_name = class_info.get("model_name", None)
    if model_name is None:
        model_name = "maskrcnn_resnet50_fpn"

    # Load model
    model = get_detection_model(
        model_name=model_name,
        num_classes=num_classes,
        num_channels=num_channels,
        pretrained=False,
    )

    if not os.path.exists(model_path):
        hf_repo = repo_id or NWPU_VHR10_HF_REPO
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=hf_repo, filename=model_path)

    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.to(device)

    # Create dataset and loader
    dataset = COCODetectionDataset(
        coco_json_path=annotations_path,
        images_dir=images_dir,
        transforms=get_transform(train=False),
        num_channels=num_channels,
        compute_masks=model_has_masks(model_name),
    )

    if num_workers is None:
        num_workers = 0 if platform.system() in ["Darwin", "Windows"] else 4

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # Evaluate
    results = evaluate_coco_metrics(
        model=model,
        data_loader=data_loader,
        device=device,
        class_names=class_names,
        verbose=verbose,
    )

    return results


def visualize_coco_annotations(
    annotations_path: str,
    images_dir: str,
    num_samples: int = 4,
    random: bool = False,
    seed: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 14),
    cols: int = 2,
    output_path: Optional[str] = None,
) -> None:
    """Visualize sample images with their COCO-format bounding box annotations.

    Loads a COCO JSON annotation file and displays a grid of sample images
    with colored bounding boxes and class labels overlaid.

    Args:
        annotations_path (str): Path to COCO-format annotations JSON file.
        images_dir (str): Directory containing the images referenced in the
            annotations file.
        num_samples (int): Number of sample images to display. Defaults to 4.
        random (bool): Whether to select images randomly instead of taking
            the first ``num_samples``. Defaults to False.
        seed (int, optional): Random seed for reproducibility when
            ``random=True``.
        figsize (tuple): Figure size (width, height). Defaults to (14, 14).
        cols (int): Number of columns in the grid layout. Defaults to 2.
        output_path (str, optional): Path to save the figure. If None,
            displays interactively.
    """
    import random as random_module

    from PIL import Image as PILImage

    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    all_images = coco_data["images"]
    if random:
        rng = random_module.Random(seed)
        sample_images = rng.sample(all_images, min(num_samples, len(all_images)))
    else:
        sample_images = all_images[:num_samples]
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    cmap = plt.cm.get_cmap("tab10", 10)

    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, img_info in enumerate(sample_images):
        img_path = os.path.join(images_dir, img_info["file_name"])
        img = PILImage.open(img_path)
        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(img_info["file_name"], fontsize=10)
        axes[ax_idx].axis("off")

        img_anns = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
        ]
        for ann in img_anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            color = cmap(cat_id % 10)
            rect = plt.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            axes[ax_idx].add_patch(rect)
            axes[ax_idx].text(
                x,
                y - 3,
                categories.get(cat_id, str(cat_id)),
                color="white",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
            )

    # Hide unused axes
    for ax_idx in range(num_samples, len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_detection_training_history(
    history_path: str,
    figsize: Tuple[int, int] = (15, 4),
    output_path: Optional[str] = None,
) -> None:
    """Plot training metrics from a detection model training history file.

    Loads a ``training_history.pth`` file saved during
    :func:`train_multiclass_detector` and plots up to three subplots:
    training/validation loss, validation IoU, and learning rate schedule.
    Subplots are skipped if the corresponding keys are missing.

    Args:
        history_path (str): Path to the ``training_history.pth`` file.
        figsize (tuple): Figure size (width, height). Defaults to (15, 4).
        output_path (str, optional): Path to save the figure. If None,
            displays interactively.
    """
    if not os.path.exists(history_path):
        logger.warning(f"Training history not found: {history_path}")
        return

    history = torch.load(history_path, weights_only=True)
    epochs = history.get("epochs", [])

    panels = []
    if "train_loss" in history:
        panels.append("loss")
    if "val_iou" in history:
        panels.append("iou")
    if "lr" in history:
        panels.append("lr")

    if not panels:
        logger.warning("No plottable metrics found in training history.")
        return

    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]

    panel_idx = 0
    if "loss" in panels:
        ax = axes[panel_idx]
        ax.plot(epochs, history["train_loss"], label="Train Loss")
        has_val_loss = "val_loss" in history and any(
            math.isfinite(v) for v in history["val_loss"]
        )
        if has_val_loss:
            ax.plot(epochs, history["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss" if has_val_loss else "Training Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        panel_idx += 1

    if "iou" in panels:
        ax = axes[panel_idx]
        ax.plot(epochs, history["val_iou"], label="Val IoU", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_title("Validation IoU")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        panel_idx += 1

    if "lr" in panels:
        ax = axes[panel_idx]
        ax.plot(epochs, history["lr"], label="Learning Rate", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate Schedule")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        panel_idx += 1

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def batch_multiclass_detection(
    image_paths: List[str],
    output_dir: str,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    num_classes: int = 11,
    class_names: Optional[List[str]] = None,
    window_size: int = 512,
    overlap: int = 256,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    batch_size: int = 4,
    num_channels: int = 3,
    device: Optional[torch.device] = None,
    visualize: bool = True,
    cols: int = 2,
    figsize: Tuple[int, int] = (16, 16),
    cleanup: bool = True,
    output_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    **kwargs: Any,
) -> List[Tuple[str, float, List[Dict]]]:
    """Run multi-class object detection on multiple images.

    Iterates over a list of image paths, calls :func:`multiclass_detection`
    for each, and optionally displays a grid of results with colored
    bounding boxes.

    Args:
        image_paths (list of str): Paths to input images.
        output_dir (str): Directory for intermediate detection output files.
        model_path (str, optional): Path to trained model weights. If None,
            downloads the pretrained NWPU-VHR-10 model.
        model_name (str, optional): Detection model architecture name. If
            None, auto-detected from checkpoint.
        num_classes (int): Number of classes including background. Defaults
            to 11.
        class_names (list, optional): List of class names (index 0 =
            background).
        window_size (int): Sliding window size. Defaults to 512.
        overlap (int): Window overlap in pixels. Defaults to 256.
        confidence_threshold (float): Minimum detection score. Defaults to
            0.5.
        nms_threshold (float): IoU threshold for NMS. Defaults to 0.3.
        batch_size (int): Inference batch size. Defaults to 4.
        num_channels (int): Number of input image channels. Defaults to 3.
        device (torch.device, optional): Compute device.
        visualize (bool): Whether to display a grid of results. Defaults to
            True.
        cols (int): Number of columns in the visualization grid. Defaults
            to 2.
        figsize (tuple): Figure size for the visualization grid. Defaults to
            (16, 16).
        cleanup (bool): Whether to remove intermediate output files after
            visualization. Defaults to True.
        output_path (str, optional): Path to save the visualization figure.
            If None, displays interactively.
        repo_id (str, optional): HuggingFace Hub repository ID for
            downloading the model.
        **kwargs: Additional keyword arguments passed to
            :func:`multiclass_detection`.

    Returns:
        list of tuple: Each tuple contains (result_path, inference_time,
        detections_list) from :func:`multiclass_detection`.
    """
    from PIL import Image as PILImage

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for idx, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{basename}_detection.tif")

        result = multiclass_detection(
            input_path=img_path,
            output_path=out_path,
            model_path=model_path,
            model_name=model_name,
            num_classes=num_classes,
            class_names=class_names,
            window_size=window_size,
            overlap=overlap,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            batch_size=batch_size,
            num_channels=num_channels,
            device=device,
            repo_id=repo_id,
            **kwargs,
        )
        results.append(result)

    if visualize:
        cmap = plt.cm.get_cmap("tab10", 10)
        n = len(image_paths)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (img_path, (_, _, dets)) in enumerate(zip(image_paths, results)):
            img = PILImage.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"{os.path.basename(img_path)} ({len(dets)} detections)",
                fontsize=10,
            )
            axes[idx].axis("off")

            for det in dets:
                box = det["box"]
                label = det["label"]
                score = det["score"]
                color = cmap(label % 10)
                rect = plt.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                axes[idx].add_patch(rect)
                name = (
                    class_names[label]
                    if class_names and label < len(class_names)
                    else str(label)
                )
                axes[idx].text(
                    box[0],
                    box[1] - 3,
                    f"{name}: {score:.2f}",
                    color="white",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                )

        for ax_idx in range(n, len(axes)):
            axes[ax_idx].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if cleanup:
        for result_path, _, _ in results:
            if os.path.exists(result_path):
                os.remove(result_path)

    return results


def push_detector_to_hub(
    model_path: str,
    repo_id: str,
    model_name: str = "fasterrcnn_resnet50_fpn_v2",
    num_classes: int = 11,
    num_channels: int = 3,
    class_names: Optional[List[str]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> Optional[str]:
    """Push a trained detection model to Hugging Face Hub.

    Uploads the model weights (``model.pth``) and a ``config.json`` file
    containing model metadata to the specified Hub repository. The
    repository is created automatically if it does not already exist.

    Args:
        model_path (str): Path to the trained model weights (``.pth`` file).
        repo_id (str): Hub repository in ``"username/repo-name"`` format.
        model_name (str): Detection model architecture name. Stored in
            ``config.json`` so the model can be reconstructed on download.
            Defaults to ``"fasterrcnn_resnet50_fpn_v2"``.
        num_classes (int): Number of classes including background. Defaults
            to 11.
        num_channels (int): Number of input image channels. Defaults to 3.
        class_names (list of str, optional): Ordered list of class name
            strings (index 0 should be ``"background"``). Stored in
            ``config.json`` so downstream users do not need the original
            dataset.
        commit_message (str, optional): Commit message for the Hub upload.
            Defaults to a descriptive string including ``model_name``.
        private (bool): Whether to create a private repository. Defaults to
            False.
        token (str, optional): Hugging Face API token with write access. If
            None, the token stored by ``huggingface-cli login`` is used.

    Returns:
        str: URL of the uploaded repository on Hugging Face Hub, or None
        if ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error(
            "huggingface_hub is required to push models. "
            "Install it with: pip install huggingface-hub"
        )
        return None

    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu")
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }

    # Build configuration dict
    config: Dict[str, Any] = {
        "model_type": "detection",
        "model_name": model_name,
        "num_classes": num_classes,
        "num_channels": num_channels,
        "class_names": class_names,
    }

    try:
        # Create Hub repository (no-op if it already exists)
        api = HfApi(token=token)
        create_repo(repo_id, private=private, token=token, exist_ok=True)

        if commit_message is None:
            commit_message = f"Upload {model_name} object detection model"

        with tempfile.TemporaryDirectory() as tmpdir:
            model_save_path = os.path.join(tmpdir, "model.pth")
            torch.save(state_dict, model_save_path)

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Model successfully pushed to: {url}")
        return url
    except Exception as e:
        logger.error(f"Failed to push model to Hub: {e}")
        return None


def predict_detector_from_hub(
    input_path: str,
    output_path: str,
    repo_id: str,
    window_size: int = 512,
    overlap: int = 256,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    token: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Tuple[str, float, List[Dict]]]:
    """Run object detection using a model downloaded from Hugging Face Hub.

    Downloads ``model.pth`` and ``config.json`` from the specified Hub
    repository and delegates to :func:`multiclass_detection` for inference.

    Args:
        input_path (str): Path to input image (GeoTIFF, JPEG, PNG, etc.).
        output_path (str): Path to save output raster.
        repo_id (str): Hub repository in ``"username/repo-name"`` format.
        window_size (int): Sliding window size. Defaults to 512.
        overlap (int): Window overlap in pixels. Defaults to 256.
        confidence_threshold (float): Minimum detection score. Defaults to
            0.5.
        nms_threshold (float): IoU threshold for NMS. Defaults to 0.3.
        batch_size (int): Inference batch size. Defaults to 4.
        device (torch.device, optional): Compute device.
        token (str, optional): Hugging Face API token for private
            repositories. If None, the token stored by
            ``huggingface-cli login`` is used.
        **kwargs: Additional keyword arguments passed to
            :func:`multiclass_detection`.

    Returns:
        Tuple of (output_path, inference_time, detections_list) where each
        detection is a dict with mask, score, box, and label, or None
        if ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is required. "
            "Install it with: pip install huggingface-hub"
        )
        return None

    try:
        logger.info(f"Downloading model from {repo_id}...")
        model_file = hf_hub_download(repo_id=repo_id, filename="model.pth", token=token)
        config_file = hf_hub_download(
            repo_id=repo_id, filename="config.json", token=token
        )
    except Exception as e:
        logger.error(f"Failed to download model from Hub: {e}")
        return None

    with open(config_file) as f:
        config = json.load(f)

    num_classes = config.get("num_classes", 11)
    num_channels = config.get("num_channels", 3)
    class_names = config.get("class_names", None)
    detected_model_name = config.get("model_name", "maskrcnn_resnet50_fpn")

    return multiclass_detection(
        input_path=input_path,
        output_path=output_path,
        model_path=model_file,
        model_name=detected_model_name,
        num_classes=num_classes,
        class_names=class_names,
        window_size=window_size,
        overlap=overlap,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        batch_size=batch_size,
        num_channels=num_channels,
        device=device,
        **kwargs,
    )
