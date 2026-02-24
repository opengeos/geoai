"""High-level functions for multi-class object detection.

This module provides convenience functions for training, evaluating, and
running inference with Mask R-CNN models on COCO-format datasets, including
support for the NWPU-VHR-10 remote sensing benchmark.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .train import (
    COCODetectionDataset,
    collate_fn,
    evaluate_coco_metrics,
    get_instance_segmentation_model,
    get_transform,
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
        output_dir (str): Directory to save the dataset. Defaults to "NWPU-VHR-10".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.

    Returns:
        str: Path to the extracted dataset directory.
    """
    data_path = download_file(NWPU_VHR10_URL, overwrite=overwrite)
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
    print(f"Model downloaded to: {model_path}")
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
        print(f"Using existing COCO annotations: {annotations_path}")
    elif gt_dir is not None:
        # Convert text annotations to COCO JSON
        print("Converting NWPU-VHR-10 text annotations to COCO JSON format...")
        _convert_nwpu_to_coco(images_dir, gt_dir, annotations_path)
        print(f"COCO annotations saved to: {annotations_path}")
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

    print(f"Dataset prepared:")
    print(f"  Images directory: {images_dir}")
    print(f"  Total annotated images: {len(image_ids)}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Training images: {len(train_ids)}")
    print(f"  Validation images: {len(val_ids)}")
    print(f"  Classes: {class_names[1:]}")

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

    This is a convenience wrapper around train_MaskRCNN_model that
    automatically sets up the COCODetectionDataset with proper class mapping.

    Args:
        images_dir (str): Directory containing training images.
        annotations_path (str): Path to COCO-format annotations JSON file.
        output_dir (str): Directory for model outputs.
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
        print(f"Training multi-class detector with {num_classes} classes")
        print(f"  Classes: {class_names[1:]}")

    # Save class names to output directory for later use
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    class_info = {
        "class_names": class_names,
        "num_classes": num_classes,
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
    )

    return os.path.join(output_dir, "best_model.pth")


def multiclass_detection(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
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

    Loads a trained Mask R-CNN model and runs inference using a sliding
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
            HuggingFace Hub. If the path does not exist locally, it is
            treated as a filename to download from ``repo_id``.
        num_classes (int): Number of classes including background. Defaults
            to 11 (NWPU-VHR-10).
        class_names (list, optional): List of class names (index 0 = background).
            If None and using the pretrained model, defaults to NWPU-VHR-10
            class names.
        window_size (int): Sliding window size. Defaults to 512.
        overlap (int): Window overlap in pixels. Defaults to 256.
        confidence_threshold (float): Minimum detection score. Defaults to 0.5.
        nms_threshold (float): IoU threshold for NMS. Defaults to 0.3.
        batch_size (int): Inference batch size. Defaults to 4.
        num_channels (int): Number of input image channels. Defaults to 3.
        device (torch.device, optional): Compute device.
        repo_id (str, optional): HuggingFace Hub repository ID for
            downloading the model. Defaults to "giswqs/nwpu-vhr10-maskrcnn".
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of (output_path, inference_time, detections_list) where each
        detection is a dict with mask, score, box, and label.

    Example:
        >>> import geoai
        >>> # Use pretrained model (auto-downloads from HuggingFace)
        >>> result_path, time, dets = geoai.multiclass_detection(
        ...     input_path="image.tif",
        ...     output_path="detections.tif",
        ... )
        >>> # Use a custom trained model
        >>> result_path, time, dets = geoai.multiclass_detection(
        ...     input_path="image.tif",
        ...     output_path="detections.tif",
        ...     model_path="my_model.pth",
        ...     num_classes=5,
        ... )
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

    # Load model
    model = get_instance_segmentation_model(
        num_classes=num_classes, num_channels=num_channels, pretrained=False
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
) -> Any:
    """Convert detections to a GeoDataFrame with geospatial coordinates.

    Converts pixel-space bounding boxes to geospatial coordinates using the
    CRS and transform from the source GeoTIFF.

    Args:
        detections (list): List of detection dicts, each with keys:
            mask, score, box (in pixel coords), label.
        geotiff_path (str): Path to the source GeoTIFF (for CRS and transform).
        class_names (list, optional): List of class names (index 0 = background).

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with columns: geometry, class_id,
        class_name, score, area_pixels.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box as shapely_box

    if len(detections) == 0:
        return gpd.GeoDataFrame(
            columns=["geometry", "class_id", "class_name", "score", "area_pixels"]
        )

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    records = []
    for det in detections:
        bx = det["box"]
        label = det["label"]
        score = det["score"]

        # Convert pixel coordinates to geographic coordinates
        x_min, y_max = transform * (bx[0], bx[1])
        x_max, y_min = transform * (bx[2], bx[3])

        geom = shapely_box(x_min, y_min, x_max, y_max)
        area_pixels = det["mask"].sum() if "mask" in det else 0

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
                "area_pixels": int(area_pixels),
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
            downloading the model. Defaults to "giswqs/nwpu-vhr10-maskrcnn".
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

    # Load model
    model = get_instance_segmentation_model(
        num_classes=num_classes, num_channels=num_channels, pretrained=False
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
