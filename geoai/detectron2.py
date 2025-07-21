"""Detectron2 integration for remote sensing image segmentation.
See https://github.com/facebookresearch/detectron2 for more details.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds

try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.config import LazyConfig, get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer

    HAS_DETECTRON2 = True
except ImportError:
    HAS_DETECTRON2 = False
    warnings.warn("Detectron2 not found. Please install detectron2 to use this module.")

try:
    from .utils import get_device
except ImportError:
    # Fallback device detection if utils is not available
    def get_device():
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def check_detectron2():
    """Check if detectron2 is available."""
    if not HAS_DETECTRON2:
        raise ImportError(
            "Detectron2 is required. Please install it with: pip install detectron2"
        )


def load_detectron2_model(
    model_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    model_weights: Optional[str] = None,
    score_threshold: float = 0.5,
    device: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> DefaultPredictor:
    """
    Load a Detectron2 model for instance segmentation.

    Args:
        model_config: Model configuration file path or name from model zoo
        model_weights: Path to model weights file. If None, uses model zoo weights
        score_threshold: Confidence threshold for predictions
        device: Device to use ('cpu', 'cuda', or None for auto-detection)
        num_classes: Number of classes for custom models

    Returns:
        DefaultPredictor: Configured Detectron2 predictor
    """
    check_detectron2()

    cfg = get_cfg()

    # Load model configuration
    if model_config.endswith(".yaml"):
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
    else:
        cfg.merge_from_file(model_config)

    # Set model weights
    if model_weights is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    else:
        cfg.MODEL.WEIGHTS = model_weights

    # Set score threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

    # Set device
    if device is None:
        device = get_device()

    # Ensure device is a string (detectron2 expects string, not torch.device)
    if hasattr(device, "type"):
        device = device.type
    elif not isinstance(device, str):
        device = str(device)

    cfg.MODEL.DEVICE = device

    # Set number of classes if specified
    if num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    return DefaultPredictor(cfg)


def detectron2_segment(
    image_path: str,
    output_dir: str = ".",
    model_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    model_weights: Optional[str] = None,
    score_threshold: float = 0.5,
    device: Optional[str] = None,
    save_masks: bool = True,
    save_probability: bool = True,
    mask_prefix: str = "instance_masks",
    prob_prefix: str = "probability_mask",
) -> Dict:
    """
    Perform instance segmentation on a remote sensing image using Detectron2.

    Args:
        image_path: Path to input image
        output_dir: Directory to save output files
        model_config: Model configuration file path or name from model zoo
        model_weights: Path to model weights file. If None, uses model zoo weights
        score_threshold: Confidence threshold for predictions
        device: Device to use ('cpu', 'cuda', or None for auto-detection)
        save_masks: Whether to save instance masks as GeoTIFF
        save_probability: Whether to save probability masks as GeoTIFF
        mask_prefix: Prefix for instance mask output file
        prob_prefix: Prefix for probability mask output file

    Returns:
        Dict containing segmentation results and output file paths
    """
    check_detectron2()

    # Load the model
    predictor = load_detectron2_model(
        model_config=model_config,
        model_weights=model_weights,
        score_threshold=score_threshold,
        device=device,
    )

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Run inference
    outputs = predictor(image)

    # Extract results
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    results = {
        "masks": masks,
        "scores": scores,
        "classes": classes,
        "boxes": boxes,
        "num_instances": len(masks),
    }

    # Get image geospatial information
    try:
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width
    except Exception:
        # If not a GeoTIFF, create a simple transform
        height, width = image.shape[:2]
        transform = from_bounds(0, 0, width, height, width, height)
        crs = CRS.from_epsg(4326)

    # Save instance masks as GeoTIFF
    if save_masks and len(masks) > 0:
        instance_mask_path = os.path.join(output_dir, f"{mask_prefix}.tif")
        instance_mask = create_instance_mask(masks)
        save_geotiff_mask(
            instance_mask, instance_mask_path, transform, crs, dtype="uint16"
        )
        results["instance_mask_path"] = instance_mask_path

    # Save probability masks as GeoTIFF
    if save_probability and len(masks) > 0:
        prob_mask_path = os.path.join(output_dir, f"{prob_prefix}.tif")
        probability_mask = create_probability_mask(masks, scores)
        save_geotiff_mask(
            probability_mask, prob_mask_path, transform, crs, dtype="float32"
        )
        results["probability_mask_path"] = prob_mask_path

    return results


def create_instance_mask(masks: np.ndarray) -> np.ndarray:
    """
    Create an instance mask from individual binary masks.

    Args:
        masks: Array of binary masks with shape (num_instances, height, width)

    Returns:
        Instance mask with unique ID for each instance
    """
    if len(masks) == 0:
        return np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint16)

    instance_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint16)

    for i, mask in enumerate(masks):
        # Assign unique instance ID (starting from 1)
        instance_mask[mask] = i + 1

    return instance_mask


def create_probability_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Create a probability mask from individual binary masks and their confidence scores.

    Args:
        masks: Array of binary masks with shape (num_instances, height, width)
        scores: Array of confidence scores for each mask

    Returns:
        Probability mask with maximum confidence score for each pixel
    """
    if len(masks) == 0:
        return np.zeros((masks.shape[1], masks.shape[2]), dtype=np.float32)

    probability_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.float32)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Update probability mask with higher confidence scores
        probability_mask = np.where(
            mask & (score > probability_mask), score, probability_mask
        )

    return probability_mask


def save_geotiff_mask(
    mask: np.ndarray,
    output_path: str,
    transform: rasterio.transform.Affine,
    crs: CRS,
    dtype: str = "uint16",
) -> None:
    """
    Save a mask as a GeoTIFF file.

    Args:
        mask: 2D numpy array representing the mask
        output_path: Path to save the GeoTIFF file
        transform: Rasterio transform for georeferencing
        crs: Coordinate reference system
        dtype: Data type for the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine numpy dtype
    if dtype == "uint16":
        np_dtype = np.uint16
    elif dtype == "float32":
        np_dtype = np.float32
    else:
        np_dtype = np.uint16

    # Convert mask to appropriate dtype
    mask = mask.astype(np_dtype)

    # Save as GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np_dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(mask, 1)


def visualize_detectron2_results(
    image_path: str,
    results: Dict,
    output_path: Optional[str] = None,
    show_scores: bool = True,
    show_classes: bool = True,
) -> np.ndarray:
    """
    Visualize Detectron2 segmentation results on the original image.

    Args:
        image_path: Path to the original image
        results: Results dictionary from detectron2_segment
        output_path: Path to save the visualization (optional)
        show_scores: Whether to show confidence scores
        show_classes: Whether to show class labels

    Returns:
        Visualization image as numpy array
    """
    check_detectron2()

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create visualizer
    v = Visualizer(image_rgb, scale=1.0)

    # Create instances object for visualization
    from detectron2.structures import Boxes, Instances

    instances = Instances((image.shape[0], image.shape[1]))
    instances.pred_masks = torch.from_numpy(results["masks"])
    instances.pred_boxes = Boxes(torch.from_numpy(results["boxes"]))
    instances.scores = torch.from_numpy(results["scores"])
    instances.pred_classes = torch.from_numpy(results["classes"])

    # Draw predictions
    out = v.draw_instance_predictions(instances)
    vis_image = out.get_image()

    # Save visualization if path provided
    if output_path is not None:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_image


def get_detectron2_models() -> List[str]:
    """
    Get a list of available Detectron2 models for instance segmentation.

    Returns:
        List of model configuration names
    """
    from detectron2.model_zoo.model_zoo import _ModelZooUrls

    configs = list(_ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys())
    models = [f"{config}.yaml" for config in configs]
    return models


def batch_detectron2_segment(
    image_paths: List[str],
    output_dir: str = ".",
    model_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    model_weights: Optional[str] = None,
    score_threshold: float = 0.5,
    device: Optional[str] = None,
    save_masks: bool = True,
    save_probability: bool = True,
) -> List[Dict]:
    """
    Perform batch instance segmentation on multiple images.

    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save output files
        model_config: Model configuration file path or name from model zoo
        model_weights: Path to model weights file. If None, uses model zoo weights
        score_threshold: Confidence threshold for predictions
        device: Device to use ('cpu', 'cuda', or None for auto-detection)
        save_masks: Whether to save instance masks as GeoTIFF
        save_probability: Whether to save probability masks as GeoTIFF

    Returns:
        List of results dictionaries for each image
    """
    check_detectron2()

    # Load the model once for batch processing
    predictor = load_detectron2_model(
        model_config=model_config,
        model_weights=model_weights,
        score_threshold=score_threshold,
        device=device,
    )

    results = []

    for i, image_path in enumerate(image_paths):
        try:
            # Generate unique output prefixes
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_prefix = f"{base_name}_instance_masks"
            prob_prefix = f"{base_name}_probability_mask"

            # Process image
            result = detectron2_segment(
                image_path=image_path,
                output_dir=output_dir,
                model_config=model_config,
                model_weights=model_weights,
                score_threshold=score_threshold,
                device=device,
                save_masks=save_masks,
                save_probability=save_probability,
                mask_prefix=mask_prefix,
                prob_prefix=prob_prefix,
            )

            result["image_path"] = image_path
            results.append(result)

            print(f"Processed {i+1}/{len(image_paths)}: {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({"image_path": image_path, "error": str(e)})

    return results


def get_class_id_name_mapping(config_path: str, lazy: bool = False) -> Dict[int, str]:
    """
    Get class ID to name mapping from a Detectron2 model config.

    Args:
        config_path (str): Path to the config file or model_zoo config name.
        lazy (bool): Whether the config is a LazyConfig (i.e., .py).

    Returns:
        dict: Mapping from class ID (int) to class name (str).
    """
    if lazy or config_path.endswith(".py"):
        cfg = LazyConfig.load(
            model_zoo.get_config_file(config_path)
            if not os.path.exists(config_path)
            else config_path
        )
        dataset_name = cfg.dataloader.train.mapper.dataset.names[0]
    else:
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(config_path)
            if not os.path.exists(config_path)
            else config_path
        )
        dataset_name = cfg.DATASETS.TRAIN[0]

    metadata = MetadataCatalog.get(dataset_name)

    classes = metadata.get("thing_classes", []) or metadata.get("stuff_classes", [])
    return {i: name for i, name in enumerate(classes)}
