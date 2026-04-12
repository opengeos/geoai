"""Image segmentation operations.

Wraps geoai segmentation functions: SAM, GroundedSAM, semantic segmentation,
and model training for CLI consumption.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

SAM_MODELS = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    "facebook/sam2.1-hiera-large",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-tiny",
]

SEGMENTATION_ARCHITECTURES = [
    "unet",
    "unetplusplus",
    "deeplabv3",
    "deeplabv3plus",
    "fpn",
    "pspnet",
    "linknet",
    "manet",
    "pan",
]

SEGMENTATION_BACKBONES = [
    "resnet50",
    "resnet101",
    "efficientnet-b0",
    "efficientnet-b4",
    "mobilenet_v2",
    "resnext50_32x4d",
]


def run_sam(
    raster: str,
    output: str,
    model: str = "facebook/sam-vit-huge",
    automatic: bool = True,
    foreground: bool = True,
    unique: bool = True,
) -> Dict[str, Any]:
    """Run SAM (Segment Anything Model) on a raster.

    Args:
        raster: Input raster file path.
        output: Output mask file path.
        model: SAM model ID.
        automatic: Whether to use automatic mask generation.
        foreground: Whether to extract foreground only.
        unique: Whether to assign unique labels to each segment.

    Returns:
        Result dict with output path, segment count, etc.

    Raises:
        FileNotFoundError: If the input raster does not exist.
    """
    raster = os.path.abspath(raster)
    output = os.path.abspath(output)
    if not os.path.isfile(raster):
        raise FileNotFoundError(f"Raster file not found: {raster}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    from geoai import SamGeo

    sam = SamGeo(model_id=model, automatic=automatic)
    sam.generate(raster, output, foreground=foreground, unique=unique)

    result = {
        "input": raster,
        "output": output,
        "model": model,
        "automatic": automatic,
        "foreground": foreground,
        "unique": unique,
    }

    if os.path.isfile(output):
        result["output_size_bytes"] = os.path.getsize(output)

    return result


def run_grounded_sam(
    raster: str,
    output: str,
    prompt: str,
    detector_model: str = "IDEA-Research/grounding-dino-tiny",
    segmenter_model: str = "facebook/sam-vit-huge",
    tile_size: int = 1024,
) -> Dict[str, Any]:
    """Run GroundedSAM text-prompted segmentation.

    Args:
        raster: Input raster file path.
        output: Output mask file path.
        prompt: Text prompt describing objects to segment.
        detector_model: Grounding DINO model ID.
        segmenter_model: SAM model ID.
        tile_size: Processing tile size in pixels.

    Returns:
        Result dict with output path.

    Raises:
        FileNotFoundError: If the input raster does not exist.
    """
    raster = os.path.abspath(raster)
    output = os.path.abspath(output)
    if not os.path.isfile(raster):
        raise FileNotFoundError(f"Raster file not found: {raster}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    from geoai import GroundedSAM

    gsam = GroundedSAM(
        detector_id=detector_model,
        segmenter_id=segmenter_model,
        tile_size=tile_size,
    )
    gsam.segment_image(raster, output, text_prompts=prompt)

    result = {
        "input": raster,
        "output": output,
        "prompt": prompt,
        "detector_model": detector_model,
        "segmenter_model": segmenter_model,
        "tile_size": tile_size,
    }

    if os.path.isfile(output):
        result["output_size_bytes"] = os.path.getsize(output)

    return result


def run_semantic_segmentation(
    raster: str,
    model_path: str,
    output: str,
    num_classes: int = 2,
    band_indexes: Optional[List[int]] = None,
    chip_size: int = 512,
    overlap: float = 0.25,
) -> Dict[str, Any]:
    """Run semantic segmentation with a trained model.

    Args:
        raster: Input raster file path.
        model_path: Path to the trained model checkpoint.
        output: Output segmentation raster path.
        num_classes: Number of classes in the model.
        band_indexes: Band indices to use (0-indexed). None for all bands.
        chip_size: Size of processing chips.
        overlap: Overlap fraction between chips.

    Returns:
        Result dict with output path.

    Raises:
        FileNotFoundError: If the input files do not exist.
    """
    raster = os.path.abspath(raster)
    model_path = os.path.abspath(model_path)
    output = os.path.abspath(output)

    if not os.path.isfile(raster):
        raise FileNotFoundError(f"Raster file not found: {raster}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    from geoai import semantic_segmentation

    kwargs = {
        "num_classes": num_classes,
        "chip_size": chip_size,
        "overlap": overlap,
    }
    if band_indexes is not None:
        kwargs["band_indexes"] = band_indexes

    semantic_segmentation(
        raster=raster,
        model_path=model_path,
        output=output,
        **kwargs,
    )

    result = {
        "input": raster,
        "model": model_path,
        "output": output,
        "num_classes": num_classes,
        "chip_size": chip_size,
        "overlap": overlap,
    }

    if os.path.isfile(output):
        result["output_size_bytes"] = os.path.getsize(output)

    return result


def train_segmentation(
    image_root: str,
    label_root: str,
    output_dir: str,
    architecture: str = "unet",
    backbone: str = "resnet50",
    num_classes: int = 2,
    in_channels: int = 4,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    loss: str = "ce",
) -> Dict[str, Any]:
    """Train a semantic segmentation model.

    Args:
        image_root: Directory with training images.
        label_root: Directory with training labels.
        output_dir: Output directory for model checkpoints.
        architecture: Model architecture name.
        backbone: Encoder backbone name.
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        loss: Loss function name (ce, jaccard, focal, dice).

    Returns:
        Result dict with model path and training metrics.

    Raises:
        FileNotFoundError: If input directories do not exist.
        ValueError: If architecture or backbone is invalid.
    """
    image_root = os.path.abspath(image_root)
    label_root = os.path.abspath(label_root)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not os.path.isdir(label_root):
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    if architecture not in SEGMENTATION_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: {', '.join(SEGMENTATION_ARCHITECTURES)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    from geoai import train_segmentation_model

    result = train_segmentation_model(
        image_root=image_root,
        label_root=label_root,
        output_dir=output_dir,
        model=architecture,
        backbone=backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss=loss,
    )

    return {
        "image_root": image_root,
        "label_root": label_root,
        "output_dir": output_dir,
        "architecture": architecture,
        "backbone": backbone,
        "num_classes": num_classes,
        "epochs": epochs,
        "result": str(result) if result else "completed",
    }


def list_sam_models() -> List[Dict[str, str]]:
    """List available SAM models.

    Returns:
        List of model info dicts.
    """
    return [
        {"model_id": m, "family": "SAM" if "sam-vit" in m else "SAM2"}
        for m in SAM_MODELS
    ]


def list_architectures() -> List[Dict[str, str]]:
    """List available segmentation architectures.

    Returns:
        List of architecture dicts.
    """
    return [{"name": a} for a in SEGMENTATION_ARCHITECTURES]
