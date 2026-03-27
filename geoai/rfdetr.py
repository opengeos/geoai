"""RF-DETR integration for object detection on geospatial imagery.

This module provides a Python interface to RF-DETR
(https://github.com/roboflow/rf-detr), Roboflow's state-of-the-art
real-time object detection model built on DINOv2 and DETR. Supports
tiled inference on GeoTIFF imagery with georeferenced output, batch
processing, training, and HuggingFace Hub integration.

Requirements:
    - rfdetr
    - supervision (installed with rfdetr)

Install with::

    pip install rfdetr
"""

import glob as _glob
import json
import logging
import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rfdetr as _rfdetr_pkg

    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

__all__ = [
    "RFDETR_MODELS",
    "check_rfdetr_available",
    "list_rfdetr_models",
    "rfdetr_detect",
    "rfdetr_detect_batch",
    "rfdetr_train",
    "push_rfdetr_to_hub",
    "rfdetr_detect_from_hub",
    "prepare_nwpu_for_rfdetr",
    "plot_rfdetr_metrics",
]

# ---------------------------------------------------------------------------
# Model registry -- all RF-DETR model variants
# ---------------------------------------------------------------------------

RFDETR_MODELS: Dict[str, Dict[str, Any]] = {
    # Detection models
    "base": {
        "class_name": "RFDETRBase",
        "resolution": 560,
        "description": "RF-DETR Base (29M params, 560px)",
    },
    "nano": {
        "class_name": "RFDETRNano",
        "resolution": 384,
        "description": "RF-DETR Nano (fastest, 384px)",
    },
    "small": {
        "class_name": "RFDETRSmall",
        "resolution": 512,
        "description": "RF-DETR Small (512px)",
    },
    "medium": {
        "class_name": "RFDETRMedium",
        "resolution": 576,
        "description": "RF-DETR Medium (576px)",
    },
    "large": {
        "class_name": "RFDETRLarge",
        "resolution": 704,
        "description": "RF-DETR Large (704px)",
    },
    # Segmentation models
    "seg-nano": {
        "class_name": "RFDETRSegNano",
        "resolution": 384,
        "description": "RF-DETR Seg Nano (instance segmentation, 384px)",
    },
    "seg-small": {
        "class_name": "RFDETRSegSmall",
        "resolution": 512,
        "description": "RF-DETR Seg Small (instance segmentation, 512px)",
    },
    "seg-medium": {
        "class_name": "RFDETRSegMedium",
        "resolution": 576,
        "description": "RF-DETR Seg Medium (instance segmentation, 576px)",
    },
    "seg-large": {
        "class_name": "RFDETRSegLarge",
        "resolution": 704,
        "description": "RF-DETR Seg Large (instance segmentation, 704px)",
    },
    "seg-xlarge": {
        "class_name": "RFDETRSegXLarge",
        "resolution": 700,
        "description": "RF-DETR Seg XLarge (instance segmentation, 700px)",
    },
    "seg-2xlarge": {
        "class_name": "RFDETRSeg2XLarge",
        "resolution": 880,
        "description": "RF-DETR Seg 2XLarge (instance segmentation, 880px)",
    },
}


def check_rfdetr_available() -> None:
    """Check if the rfdetr package is installed.

    Raises:
        ImportError: If rfdetr is not installed, with installation instructions.
    """
    if not RFDETR_AVAILABLE:
        raise ImportError(
            "The rfdetr package is required but not installed. "
            "Install it with: pip install rfdetr"
        )


def list_rfdetr_models() -> Dict[str, str]:
    """List available RF-DETR model variants.

    Returns:
        Dict[str, str]: Dictionary mapping model variant names to their
            descriptions.
    """
    return {name: info["description"] for name, info in RFDETR_MODELS.items()}


def _get_rfdetr_model_class(model_variant: str):
    """Get the RF-DETR model class for a given variant name.

    Args:
        model_variant: Name of the model variant (e.g., "base", "large",
            "seg-medium").

    Returns:
        The RF-DETR model class.

    Raises:
        ValueError: If the model variant is not recognized.
        ImportError: If rfdetr is not installed.
    """
    check_rfdetr_available()

    if model_variant not in RFDETR_MODELS:
        available = ", ".join(sorted(RFDETR_MODELS.keys()))
        raise ValueError(
            f"Unknown model variant '{model_variant}'. "
            f"Available variants: {available}"
        )

    class_name = RFDETR_MODELS[model_variant]["class_name"]
    try:
        return getattr(_rfdetr_pkg, class_name)
    except AttributeError:
        raise ValueError(
            f"The rfdetr package does not expose class '{class_name}' "
            f"for variant '{model_variant}'. You may need to update rfdetr: "
            f"pip install --upgrade rfdetr"
        )


def _create_rfdetr_model(
    model_variant: str = "base",
    pretrain_weights: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
):
    """Create an RF-DETR model instance.

    Args:
        model_variant: Name of the model variant. Defaults to "base".
        pretrain_weights: Path to custom pretrained weights. If None,
            uses the default COCO pretrained weights.
        device: Device to use ("cpu", "cuda", "mps"). If None, auto-detected.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An RF-DETR model instance.
    """
    model_cls = _get_rfdetr_model_class(model_variant)

    init_kwargs = {}
    if pretrain_weights is not None:
        init_kwargs["pretrain_weights"] = pretrain_weights
    if device is not None:
        init_kwargs["device"] = device
    init_kwargs.update(kwargs)

    return model_cls(**init_kwargs)


def rfdetr_detect(
    input_path: str,
    output_path: Optional[str] = None,
    model_variant: str = "base",
    pretrain_weights: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    batch_size: int = 4,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Perform object detection on a GeoTIFF using RF-DETR.

    Runs tiled inference with a sliding window approach on a GeoTIFF image
    using an RF-DETR model. Detections from overlapping tiles are merged
    using class-aware Non-Maximum Suppression (NMS). Results are returned
    as a georeferenced GeoDataFrame.

    Args:
        input_path: Path to input GeoTIFF image.
        output_path: Optional path to save the output GeoDataFrame
            (as GeoJSON, GPKG, or Shapefile based on extension). If None,
            results are only returned in memory.
        model_variant: RF-DETR model variant to use. See
            ``list_rfdetr_models()`` for available options. Defaults to
            "base".
        pretrain_weights: Path to custom pretrained weights file. If None,
            uses COCO pretrained weights.
        confidence_threshold: Minimum confidence score for detections.
            Defaults to 0.5.
        nms_threshold: IoU threshold for Non-Maximum Suppression across
            tiles. Defaults to 0.3.
        window_size: Size of the sliding window in pixels. Defaults to
            the model's native resolution.
        overlap: Overlap between adjacent windows in pixels. Defaults to
            ``window_size // 4``.
        batch_size: Number of tiles to process in each batch. Defaults to 4.
        class_names: Optional list of class names for labeling detections.
            If None and using COCO pretrained weights, COCO class names
            are used.
        device: Device to use ("cpu", "cuda", "mps"). If None, auto-detected.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with columns: geometry,
        class_id, class_name, confidence. The geometry column contains
        georeferenced bounding box polygons.
    """
    import geopandas as gpd
    import rasterio
    import torch
    import torchvision
    from PIL import Image
    from rasterio.windows import Window
    from shapely.geometry import Polygon
    from tqdm import tqdm

    check_rfdetr_available()

    model = _create_rfdetr_model(
        model_variant=model_variant,
        pretrain_weights=pretrain_weights,
        device=device,
        **kwargs,
    )

    # Resolve window_size and overlap
    model_resolution = RFDETR_MODELS[model_variant]["resolution"]
    if window_size is None:
        window_size = model_resolution
    if overlap is None:
        overlap = window_size // 4

    # Get class names from model if not provided
    if class_names is None:
        class_names = model.class_names

    with rasterio.open(input_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform

        all_boxes = []
        all_scores = []
        all_class_ids = []

        stride = window_size - overlap
        if stride <= 0:
            raise ValueError(
                f"overlap ({overlap}) must be less than window_size "
                f"({window_size})."
            )
        steps_y = max(1, math.ceil((height - window_size) / stride) + 1)
        steps_x = max(1, math.ceil((width - window_size) / stride) + 1)
        total_windows = steps_y * steps_x

        logger.info(
            "Processing %d windows (%dx%d, overlap=%d) on %s",
            total_windows,
            window_size,
            window_size,
            overlap,
            os.path.basename(input_path),
        )

        pbar = tqdm(total=total_windows, desc="RF-DETR detection")
        batch_images = []
        batch_positions = []
        start_time = time.time()

        for i in range(steps_y):
            y = min(i * stride, max(0, height - window_size))

            for j in range(steps_x):
                x = min(j * stride, max(0, width - window_size))

                actual_w = min(window_size, width - x)
                actual_h = min(window_size, height - y)

                window = src.read(window=Window(x, y, actual_w, actual_h))

                if window.shape[1] == 0 or window.shape[2] == 0:
                    pbar.update(1)
                    continue

                # Extract RGB bands (first 3) and transpose to HWC
                if window.shape[0] >= 3:
                    rgb = window[:3]
                else:
                    # Pad to 3 channels if needed
                    rgb = np.zeros(
                        (3, window.shape[1], window.shape[2]), dtype=window.dtype
                    )
                    rgb[: window.shape[0]] = window
                tile_hwc = np.transpose(rgb, (1, 2, 0))

                # Pad to full window_size if tile is smaller (edge tiles)
                if actual_h < window_size or actual_w < window_size:
                    padded = np.zeros(
                        (window_size, window_size, 3), dtype=tile_hwc.dtype
                    )
                    padded[:actual_h, :actual_w] = tile_hwc
                    tile_hwc = padded

                pil_image = Image.fromarray(tile_hwc.astype(np.uint8))
                batch_images.append(pil_image)
                batch_positions.append((x, y, actual_w, actual_h))

                if len(batch_images) == batch_size or (
                    i == steps_y - 1 and j == steps_x - 1
                ):
                    if batch_images:
                        detections_list = model.predict(
                            batch_images,
                            threshold=confidence_threshold,
                        )

                        # predict returns single Detections if single image
                        if not isinstance(detections_list, list):
                            detections_list = [detections_list]

                        for det, (x_pos, y_pos, w, h) in zip(
                            detections_list, batch_positions
                        ):
                            if len(det.xyxy) > 0:
                                # Scale from model resolution to actual tile
                                # size, then offset to global pixel coords.
                                # RF-DETR rescales detections back to input
                                # image size (window_size x window_size).
                                boxes = det.xyxy.copy()

                                # Clip boxes to the actual tile area (exclude
                                # padding region for edge tiles)
                                boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
                                boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
                                boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
                                boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

                                # Skip detections that collapsed after clipping
                                valid = (boxes[:, 2] > boxes[:, 0]) & (
                                    boxes[:, 3] > boxes[:, 1]
                                )
                                boxes = boxes[valid]
                                scores = det.confidence[valid]
                                class_ids = det.class_id[valid]

                                # Offset to global pixel coordinates
                                boxes[:, 0] += x_pos
                                boxes[:, 1] += y_pos
                                boxes[:, 2] += x_pos
                                boxes[:, 3] += y_pos

                                all_boxes.append(boxes)
                                all_scores.append(scores)
                                all_class_ids.append(class_ids)

                        pbar.update(len(batch_images))
                        batch_images = []
                        batch_positions = []

        pbar.close()
        elapsed = time.time() - start_time
        logger.info("Tiled inference completed in %.1f seconds", elapsed)

    # Merge detections from all tiles with class-aware NMS
    if all_boxes:
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_class_ids = np.concatenate(all_class_ids, axis=0)

        logger.info("Collected %d detections before NMS", len(all_boxes))

        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        labels_tensor = torch.tensor(all_class_ids, dtype=torch.int64)

        keep_indices = torchvision.ops.batched_nms(
            boxes_tensor, scores_tensor, labels_tensor, nms_threshold
        )

        final_boxes = all_boxes[keep_indices.numpy()]
        final_scores = all_scores[keep_indices.numpy()]
        final_class_ids = all_class_ids[keep_indices.numpy()]

        logger.info("After NMS: %d detections", len(final_boxes))
    else:
        final_boxes = np.empty((0, 4))
        final_scores = np.empty((0,))
        final_class_ids = np.empty((0,), dtype=int)

    # Convert pixel coordinates to georeferenced polygons
    records = []
    for idx in range(len(final_boxes)):
        c0, r0, c1, r1 = final_boxes[idx]
        class_id = int(final_class_ids[idx])
        score = float(final_scores[idx])

        # Convert pixel corners to georeferenced coordinates
        pts = []
        for c, r in [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]:
            geo_x, geo_y = transform * (float(c), float(r))
            pts.append((geo_x, geo_y))
        geom = Polygon(pts)

        # Resolve class name
        if class_names and class_id < len(class_names):
            name = class_names[class_id]
        else:
            name = f"class_{class_id}"

        records.append(
            {
                "geometry": geom,
                "class_id": class_id,
                "class_name": name,
                "confidence": score,
            }
        )

    if records:
        gdf = gpd.GeoDataFrame(records, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            columns=["geometry", "class_id", "class_name", "confidence"],
            geometry="geometry",
            crs=crs,
        )

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gdf.to_file(output_path)
        logger.info("Saved %d detections to %s", len(gdf), output_path)

    return gdf


def rfdetr_detect_batch(
    input_paths: Union[str, List[str]],
    output_dir: Optional[str] = None,
    model_variant: str = "base",
    pretrain_weights: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    batch_size: int = 4,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Perform batch object detection on multiple GeoTIFFs using RF-DETR.

    Processes multiple GeoTIFF images using tiled RF-DETR inference and
    returns a combined GeoDataFrame with all detections.

    Args:
        input_paths: Either a list of file paths or a glob pattern string
            (e.g., "images/*.tif") matching GeoTIFF files.
        output_dir: Optional directory to save per-image detection results.
            Each output file is named after the input with a "_detections"
            suffix. If None, results are only returned in memory.
        model_variant: RF-DETR model variant to use. Defaults to "base".
        pretrain_weights: Path to custom pretrained weights file.
        confidence_threshold: Minimum confidence score. Defaults to 0.5.
        nms_threshold: IoU threshold for NMS. Defaults to 0.3.
        window_size: Sliding window size in pixels.
        overlap: Overlap between adjacent windows in pixels.
        batch_size: Number of tiles per inference batch. Defaults to 4.
        class_names: Optional list of class names.
        device: Device to use. If None, auto-detected.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        geopandas.GeoDataFrame: Combined GeoDataFrame with detections from
        all images. Includes an additional "source_file" column indicating
        which image each detection came from.
    """
    import geopandas as gpd

    check_rfdetr_available()

    # Resolve input paths
    if isinstance(input_paths, str):
        if any(ch in input_paths for ch in ("*", "?", "[")):
            resolved_paths = sorted(_glob.glob(input_paths))
        elif os.path.isdir(input_paths):
            tif_paths = _glob.glob(os.path.join(input_paths, "*.tif"))
            tiff_paths = _glob.glob(os.path.join(input_paths, "*.tiff"))
            resolved_paths = sorted(set(tif_paths + tiff_paths))
        else:
            resolved_paths = [input_paths]
        if not resolved_paths:
            raise FileNotFoundError(f"No input files found for: {input_paths}")
    else:
        resolved_paths = list(input_paths)

    if not resolved_paths:
        raise ValueError("No input files provided.")

    logger.info("Processing %d images with RF-DETR", len(resolved_paths))

    all_gdfs = []
    for path in resolved_paths:
        logger.info("Processing: %s", os.path.basename(path))

        output_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(output_dir, f"{base}_detections.geojson")

        gdf = rfdetr_detect(
            input_path=path,
            output_path=output_path,
            model_variant=model_variant,
            pretrain_weights=pretrain_weights,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            window_size=window_size,
            overlap=overlap,
            batch_size=batch_size,
            class_names=class_names,
            device=device,
            **kwargs,
        )
        gdf["source_file"] = os.path.basename(path)
        all_gdfs.append(gdf)

    if all_gdfs:
        import pandas as pd

        combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        if all_gdfs[0].crs is not None:
            combined = combined.set_crs(all_gdfs[0].crs)
    else:
        combined = gpd.GeoDataFrame(
            columns=["geometry", "class_id", "class_name", "confidence", "source_file"]
        )

    logger.info(
        "Batch detection complete: %d total detections from %d images",
        len(combined),
        len(resolved_paths),
    )
    return combined


def rfdetr_train(
    dataset_dir: str,
    model_variant: str = "base",
    epochs: int = 100,
    batch_size: int = 4,
    output_dir: str = "output",
    pretrain_weights: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Train an RF-DETR model on a COCO or YOLO format dataset.

    Wraps the RF-DETR training API for fine-tuning on custom geospatial
    datasets. Supports COCO JSON and YOLO format annotations.

    Args:
        dataset_dir: Path to the dataset directory. For COCO format, must
            contain ``train/_annotations.coco.json``. For YOLO format,
            must contain ``data.yaml``.
        model_variant: RF-DETR model variant to use. Defaults to "base".
        epochs: Number of training epochs. Defaults to 100.
        batch_size: Training batch size. Defaults to 4.
        output_dir: Directory to save checkpoints and logs. Defaults to
            "output".
        pretrain_weights: Path to custom pretrained weights to start from.
            If None, uses COCO pretrained weights.
        device: Device to use ("cpu", "cuda", "mps"). If None, auto-detected.
        **kwargs: Additional keyword arguments forwarded to RF-DETR's
            TrainConfig (e.g., lr, lr_encoder, use_ema, grad_accum_steps,
            warmup_epochs, early_stopping, multi_scale).

    Returns:
        str: Path to the output directory containing the best checkpoint
        and training logs.
    """
    check_rfdetr_available()

    model = _create_rfdetr_model(
        model_variant=model_variant,
        pretrain_weights=pretrain_weights,
        device=device,
    )

    train_kwargs = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "output_dir": output_dir,
    }
    train_kwargs.update(kwargs)

    logger.info(
        "Starting RF-DETR training: variant=%s, epochs=%d, batch_size=%d",
        model_variant,
        epochs,
        batch_size,
    )
    model.train(**train_kwargs)

    logger.info("Training complete. Output saved to: %s", output_dir)
    return output_dir


def push_rfdetr_to_hub(
    model_path: str,
    repo_id: str,
    model_variant: str = "base",
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> Optional[str]:
    """Push a trained RF-DETR model to Hugging Face Hub.

    Uploads the model weights and a config.json file containing model
    metadata to the specified Hub repository. The repository is created
    automatically if it does not already exist.

    Args:
        model_path: Path to the trained model weights file.
        repo_id: Hub repository in "username/repo-name" format.
        model_variant: RF-DETR model variant name. Stored in config.json
            so the model can be reconstructed on download. Defaults to
            "base".
        num_classes: Number of detection classes. If None, inferred from
            class_names length.
        class_names: Ordered list of class name strings. Stored in
            config.json for downstream use.
        commit_message: Commit message for the Hub upload. Defaults to
            a descriptive string.
        private: Whether to create a private repository. Defaults to False.
        token: Hugging Face API token with write access. If None, uses
            the token stored by ``huggingface-cli login``.

    Returns:
        str: URL of the uploaded repository, or None if huggingface_hub
        is not installed.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error(
            "huggingface_hub is required to push models. "
            "Install it with: pip install huggingface-hub"
        )
        return None

    if num_classes is None and class_names is not None:
        num_classes = len(class_names)

    config: Dict[str, Any] = {
        "model_type": "rfdetr",
        "model_variant": model_variant,
        "num_classes": num_classes,
        "class_names": class_names,
        "resolution": RFDETR_MODELS.get(model_variant, {}).get("resolution"),
    }

    try:
        api = HfApi(token=token)
        create_repo(repo_id, private=private, token=token, exist_ok=True)

        if commit_message is None:
            commit_message = f"Upload RF-DETR {model_variant} object detection model"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy weights
            import shutil

            weights_dest = os.path.join(tmpdir, "weights.pth")
            shutil.copy2(model_path, weights_dest)

            # Write config
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
        logger.info("Model successfully pushed to: %s", url)
        return url
    except Exception as e:
        logger.error("Failed to push model to Hub: %s", e)
        return None


def rfdetr_detect_from_hub(
    input_path: str,
    repo_id: str,
    filename: str = "weights.pth",
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Run RF-DETR detection using a model from Hugging Face Hub.

    Downloads a trained RF-DETR model from HuggingFace Hub and runs
    tiled detection on a GeoTIFF image.

    Args:
        input_path: Path to input GeoTIFF image.
        repo_id: HuggingFace Hub repository in "username/repo-name" format.
        filename: Model weights filename in the repository. Defaults to
            "weights.pth".
        output_path: Optional path to save the output GeoDataFrame.
        confidence_threshold: Minimum confidence score. Defaults to 0.5.
        nms_threshold: IoU threshold for NMS. Defaults to 0.3.
        window_size: Sliding window size in pixels.
        overlap: Overlap between adjacent windows in pixels.
        batch_size: Number of tiles per inference batch. Defaults to 4.
        device: Device to use. If None, auto-detected.
        token: HuggingFace API token for private repositories.
        **kwargs: Additional keyword arguments passed to rfdetr_detect.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with georeferenced detections.
    """
    check_rfdetr_available()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. "
            "Install it with: pip install huggingface-hub"
        )

    # Download weights
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)

    # Try to download config
    model_variant = "base"
    class_names = None
    try:
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", token=token
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        model_variant = config.get("model_variant", "base")
        class_names = config.get("class_names", None)
    except Exception:
        logger.warning(
            "Could not download config.json from %s. Using defaults.", repo_id
        )

    return rfdetr_detect(
        input_path=input_path,
        output_path=output_path,
        model_variant=model_variant,
        pretrain_weights=weights_path,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        window_size=window_size,
        overlap=overlap,
        batch_size=batch_size,
        class_names=class_names,
        device=device,
        **kwargs,
    )


def prepare_nwpu_for_rfdetr(
    output_dir: str = "nwpu-rfdetr",
    val_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Download and prepare the NWPU-VHR-10 dataset for RF-DETR training.

    Downloads the NWPU-VHR-10 remote sensing dataset, converts its text
    annotations to COCO JSON format, and organizes files into the
    directory structure expected by RF-DETR (``train/`` and ``valid/``
    subdirectories with ``_annotations.coco.json`` files).

    The NWPU-VHR-10 dataset contains 650 annotated VHR remote sensing
    images with 10 object classes: airplane, ship, storage_tank,
    baseball_diamond, tennis_court, basketball_court, ground_track_field,
    harbor, bridge, and vehicle.

    Args:
        output_dir: Directory to create the RF-DETR dataset in. Defaults
            to "nwpu-rfdetr".
        val_split: Fraction of data to use for validation. Defaults to 0.2.
        seed: Random seed for train/val split. Defaults to 42.

    Returns:
        dict: Dictionary with keys:
            - dataset_dir (str): Path to the prepared dataset directory.
            - class_names (list): List of class name strings.
            - num_classes (int): Number of object classes.
            - train_images (int): Number of training images.
            - val_images (int): Number of validation images.
    """
    import shutil

    from .object_detect import download_nwpu_vhr10, prepare_nwpu_vhr10

    # Download dataset
    logger.info("Downloading NWPU-VHR-10 dataset...")
    data_dir = download_nwpu_vhr10()

    # Convert to COCO format
    logger.info("Converting annotations to COCO format...")
    coco_tmp = os.path.join(output_dir, "_coco_tmp")
    result = prepare_nwpu_vhr10(
        data_dir=data_dir,
        output_dir=coco_tmp,
        val_split=val_split,
        seed=seed,
    )

    # RF-DETR uses 0-indexed class IDs, so class_names should NOT
    # include "background" at index 0. The COCO categories in the
    # annotation files are 1-indexed, and RF-DETR remaps them to
    # 0-indexed internally during training.
    class_names = [name for name in result["class_names"] if name != "background"]
    images_dir = result.get("images_dir", os.path.join(data_dir, "positive_image_set"))

    # Create RF-DETR directory structure
    for split, ann_key in [
        ("train", "train_annotations"),
        ("valid", "val_annotations"),
    ]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        ann_path = result[ann_key]
        with open(ann_path, "r") as f:
            coco_data = json.load(f)

        # Copy images
        for img in coco_data["images"]:
            src = os.path.join(images_dir, img["file_name"])
            dst = os.path.join(split_dir, img["file_name"])
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        # Save annotations
        ann_dst = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_dst, "w") as f:
            json.dump(coco_data, f)

    # Get counts
    with open(os.path.join(output_dir, "train", "_annotations.coco.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(output_dir, "valid", "_annotations.coco.json")) as f:
        val_data = json.load(f)

    # Clean up temporary COCO files
    shutil.rmtree(coco_tmp, ignore_errors=True)

    logger.info(
        "NWPU-VHR-10 dataset prepared: %d train, %d val images, %d classes",
        len(train_data["images"]),
        len(val_data["images"]),
        len(class_names),
    )

    return {
        "dataset_dir": os.path.abspath(output_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_images": len(train_data["images"]),
        "val_images": len(val_data["images"]),
    }


def plot_rfdetr_metrics(
    metrics_path: str,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
) -> Any:
    """Plot training and validation metrics from an RF-DETR training run.

    Reads the ``metrics.csv`` file produced by RF-DETR training and plots
    training loss, validation mAP, F1 score, precision/recall, and
    per-class AP curves.

    Args:
        metrics_path: Path to the ``metrics.csv`` file in the RF-DETR
            output directory, or path to the output directory itself.
        figsize: Optional figure size as ``(width, height)`` in inches.
            If None, automatically determined based on number of subplots.
        save_path: Optional path to save the plot image. If None, the
            plot is displayed but not saved.

    Returns:
        pandas.DataFrame: DataFrame containing the validation metrics
        indexed by epoch.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pandas as pd

    # Resolve path
    if os.path.isdir(metrics_path):
        metrics_path = os.path.join(metrics_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    # Separate train and val rows
    train_df = df.dropna(subset=["train/loss"])
    val_df = df.dropna(subset=["val/mAP_50_95"])

    # Identify per-class AP columns
    ap_cols = [c for c in df.columns if c.startswith("val/AP/")]
    class_names = [c.replace("val/AP/", "") for c in ap_cols]

    # Determine subplot layout
    num_plots = 3  # loss, mAP, F1/precision/recall
    if ap_cols:
        num_plots += 1  # per-class AP

    if figsize is None:
        figsize = (14, 4 * num_plots)

    fig, axes = plt.subplots(num_plots, 1, figsize=figsize)
    if num_plots == 1:
        axes = [axes]
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    idx = 0

    # Plot 1: Training and Validation Loss
    ax = axes[idx]
    if not train_df.empty:
        ax.plot(
            train_df["epoch"], train_df["train/loss"], label="Train Loss", marker="."
        )
    if not val_df.empty and "val/loss" in val_df.columns:
        ax.plot(val_df["epoch"], val_df["val/loss"], label="Val Loss", marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    idx += 1

    # Plot 2: mAP Metrics
    ax = axes[idx]
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["val/mAP_50_95"], label="mAP@50:95", marker="o")
        ax.plot(val_df["epoch"], val_df["val/mAP_50"], label="mAP@50", marker="s")
        if "val/mAP_75" in val_df.columns:
            ax.plot(val_df["epoch"], val_df["val/mAP_75"], label="mAP@75", marker="^")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    idx += 1

    # Plot 3: F1, Precision, Recall
    ax = axes[idx]
    if not val_df.empty:
        if "val/F1" in val_df.columns:
            ax.plot(val_df["epoch"], val_df["val/F1"], label="F1", marker="o")
        if "val/precision" in val_df.columns:
            ax.plot(
                val_df["epoch"], val_df["val/precision"], label="Precision", marker="s"
            )
        if "val/recall" in val_df.columns:
            ax.plot(val_df["epoch"], val_df["val/recall"], label="Recall", marker="^")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation F1 / Precision / Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    idx += 1

    # Plot 4: Per-class AP
    if ap_cols and idx < num_plots:
        ax = axes[idx]
        if not val_df.empty:
            for col, name in zip(ap_cols, class_names):
                ax.plot(val_df["epoch"], val_df[col], label=name, marker=".")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AP@50:95")
        ax.set_title("Per-Class AP")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to: %s", save_path)

    plt.show()

    # Return val metrics as DataFrame
    if not val_df.empty:
        result_cols = ["epoch", "val/mAP_50_95", "val/mAP_50", "val/mAR", "val/F1"]
        result_cols += [
            c for c in ["val/precision", "val/recall"] if c in val_df.columns
        ]
        result_cols += ap_cols
        return val_df[result_cols].reset_index(drop=True)
    return pd.DataFrame()
