"""Auto classes for geospatial model inference with GeoTIFF support.

This module provides AutoGeoModel and AutoGeoImageProcessor that extend
Hugging Face transformers' AutoModel and AutoImageProcessor to support
processing geospatial data (GeoTIFF) and saving outputs as GeoTIFF or vector data.

Supported tasks:
    - Semantic segmentation (e.g., SegFormer, Mask2Former)
    - Image classification (e.g., ViT, ResNet)
    - Object detection (e.g., DETR, YOLOS)
    - Zero-shot object detection (e.g., Grounding DINO, OWL-ViT)
    - Depth estimation (e.g., Depth Anything, DPT)
    - Mask generation (e.g., SAM)

Example:
    >>> from geoai import AutoGeoModel
    >>> model = AutoGeoModel.from_pretrained(
    ...     "nvidia/segformer-b0-finetuned-ade-512-512",
    ...     task="semantic-segmentation"
    ... )
    >>> result = model.predict("input.tif", output_path="output.tif")
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import requests
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import box, shape
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForSemanticSegmentation,
    AutoModelForUniversalSegmentation,
    AutoModelForDepthEstimation,
    AutoModelForMaskGeneration,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from transformers.utils import logging as hf_logging

from .utils import get_device


hf_logging.set_verbosity_error()  # silence HF load reports


class AutoGeoImageProcessor:
    """
    Image processor for geospatial data that wraps AutoImageProcessor.

    This class provides functionality to load and preprocess GeoTIFF images
    while preserving geospatial metadata (CRS, transform, bounds). It wraps
    Hugging Face's AutoImageProcessor and adds geospatial capabilities.

    Use `from_pretrained` to instantiate this class, following the transformers pattern.

    Attributes:
        processor: The underlying AutoImageProcessor instance.
        device (str): The device being used ('cuda' or 'cpu').

    Example:
        >>> processor = AutoGeoImageProcessor.from_pretrained("facebook/sam-vit-base")
        >>> data, metadata = processor.load_geotiff("input.tif")
        >>> inputs = processor(data)
    """

    def __init__(
        self,
        processor: "AutoImageProcessor",
        processor_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the AutoGeoImageProcessor with an existing processor.

        Note: Use `from_pretrained` class method to load from Hugging Face Hub.

        Args:
            processor: An AutoImageProcessor instance.
            processor_name: Name or path of the processor (for reference).
            device: Device to use ('cuda', 'cpu'). If None, auto-detect.
        """
        self.processor = processor
        self.processor_name = processor_name

        if device is None:
            self.device = get_device()
        else:
            self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[str] = None,
        use_full_processor: bool = False,
        **kwargs: Any,
    ) -> "AutoGeoImageProcessor":
        """Load an AutoGeoImageProcessor from a pretrained processor.

        This method wraps AutoImageProcessor.from_pretrained and adds
        geospatial capabilities for processing GeoTIFF files.

        Args:
            pretrained_model_name_or_path: Hugging Face model/processor name or local path.
                Can be a model ID from huggingface.co or a local directory path.
            device: Device to use ('cuda', 'cpu'). If None, auto-detect.
            use_full_processor: If True, use AutoProcessor instead of AutoImageProcessor.
                Required for models that need text inputs (e.g., Grounding DINO).
            **kwargs: Additional arguments passed to AutoImageProcessor.from_pretrained.
                Common options include:
                - trust_remote_code (bool): Whether to trust remote code.
                - revision (str): Specific model version to use.
                - use_fast (bool): Whether to use fast tokenizer.

        Returns:
            AutoGeoImageProcessor instance with geospatial support.

        Example:
            >>> processor = AutoGeoImageProcessor.from_pretrained("facebook/sam-vit-base")
            >>> processor = AutoGeoImageProcessor.from_pretrained(
            ...     "nvidia/segformer-b0-finetuned-ade-512-512",
            ...     device="cuda"
            ... )
        """
        # Check if this is a model that needs the full processor (text + image)
        model_name_lower = pretrained_model_name_or_path.lower()
        needs_full_processor = use_full_processor or any(
            name in model_name_lower
            for name in ["grounding-dino", "owl", "clip", "blip"]
        )

        if needs_full_processor:
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        else:
            try:
                processor = AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
            except Exception:
                processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                )
        return cls(
            processor=processor,
            processor_name=pretrained_model_name_or_path,
            device=device,
        )

    def load_geotiff(
        self,
        source: Union[str, "rasterio.DatasetReader"],
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Load a GeoTIFF file and return data with metadata.

        Args:
            source: Path to GeoTIFF file or open rasterio DatasetReader.
            window: Optional rasterio Window for reading a subset.
            bands: List of band indices to read (1-indexed). If None, read all bands.

        Returns:
            Tuple of (image array in CHW format, metadata dict).

        Example:
            >>> processor = AutoGeoImageProcessor.from_pretrained("facebook/sam-vit-base")
            >>> data, metadata = processor.load_geotiff("input.tif")
            >>> print(data.shape)  # (C, H, W)
            >>> print(metadata['crs'])  # CRS info
        """
        should_close = False
        if isinstance(source, str):
            src = rasterio.open(source)
            should_close = True
        else:
            src = source

        try:
            # Read specified bands or all bands
            if bands:
                data = src.read(bands, window=window)
            else:
                data = src.read(window=window)

            # Get profile and update for window
            profile = src.profile.copy()
            if window:
                profile.update(
                    {
                        "height": window.height,
                        "width": window.width,
                        "transform": src.window_transform(window),
                    }
                )

            metadata = {
                "profile": profile,
                "crs": src.crs,
                "transform": profile["transform"],
                "bounds": (
                    src.bounds
                    if not window
                    else rasterio.windows.bounds(window, src.transform)
                ),
                "width": profile["width"],
                "height": profile["height"],
                "count": data.shape[0],
            }

        finally:
            if should_close:
                src.close()

        return data, metadata

    def load_image(
        self,
        source: Union[str, np.ndarray, Image.Image],
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load an image from various sources.

        Args:
            source: Path to image file, numpy array, or PIL Image.
            window: Optional rasterio Window (only for GeoTIFF).
            bands: List of band indices (only for GeoTIFF, 1-indexed).

        Returns:
            Tuple of (image array in CHW format, metadata dict or None).
        """
        if isinstance(source, str):
            # Check if GeoTIFF
            try:
                with rasterio.open(source) as src:
                    if src.crs is not None or source.lower().endswith(
                        (".tif", ".tiff")
                    ):
                        return self.load_geotiff(source, window, bands)
            except (rasterio.RasterioIOError, rasterio.errors.RasterioIOError):
                # If opening as GeoTIFF fails, fall back to loading as a regular image.
                pass

            # Load as regular image
            image = Image.open(source).convert("RGB")
            data = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
            return data, None

        elif isinstance(source, np.ndarray):
            # Ensure CHW format
            if source.ndim == 2:
                source = source[np.newaxis, :, :]
            elif source.ndim == 3 and source.shape[2] in [1, 3, 4]:
                source = source.transpose(2, 0, 1)
            return source, None

        elif isinstance(source, Image.Image):
            data = np.array(source.convert("RGB")).transpose(2, 0, 1)
            return data, None

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def prepare_for_model(
        self,
        data: np.ndarray,
        normalize: bool = True,
        to_rgb: bool = True,
        percentile_clip: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Prepare image data for model input.

        Args:
            data: Image array in CHW format.
            normalize: Whether to normalize pixel values.
            to_rgb: Whether to convert to 3-channel RGB.
            percentile_clip: Whether to use percentile clipping for normalization.
            return_tensors: Return format ('pt' for PyTorch, 'np' for numpy).

        Returns:
            Dictionary with processed inputs ready for model.
        """
        # Convert to HWC format
        if data.ndim == 3:
            img = data.transpose(1, 2, 0)  # CHW -> HWC
        else:
            img = data

        # Handle different band counts
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[..., :3]

        # Normalize
        if normalize:
            if percentile_clip:
                for i in range(img.shape[-1]):
                    band = img[..., i]
                    p2, p98 = np.percentile(band, [2, 98])
                    if p98 > p2:
                        img[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        img[..., i] = 0
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
                    np.uint8
                )

        # Convert to PIL Image
        pil_image = Image.fromarray(img)

        # Process with transformers processor
        inputs = self.processor(images=pil_image, return_tensors=return_tensors)

        return inputs

    def __call__(
        self,
        images: Union[str, np.ndarray, Image.Image, List],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process images for model input.

        Args:
            images: Single image or list of images (paths, arrays, or PIL Images).
            **kwargs: Additional arguments passed to the processor.

        Returns:
            Processed inputs ready for model.
        """
        if isinstance(images, (str, np.ndarray, Image.Image)):
            images = [images]

        all_pil_images = []
        for img in images:
            if isinstance(img, str):
                data, _ = self.load_image(img)
            elif isinstance(img, np.ndarray):
                data = img
                if data.ndim == 3 and data.shape[2] in [1, 3, 4]:
                    data = data.transpose(2, 0, 1)
            else:
                data = np.array(img.convert("RGB")).transpose(2, 0, 1)

            # Prepare PIL image
            if data.ndim == 3:
                img_arr = data.transpose(1, 2, 0)
            else:
                img_arr = data

            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)
            elif img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, axis=-1)
            elif img_arr.shape[-1] > 3:
                img_arr = img_arr[..., :3]

            # Normalize to uint8 if needed
            if img_arr.dtype != np.uint8:
                for i in range(img_arr.shape[-1]):
                    band = img_arr[..., i]
                    p2, p98 = np.percentile(band, [2, 98])
                    if p98 > p2:
                        img_arr[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        img_arr[..., i] = 0
                img_arr = (img_arr * 255).astype(np.uint8)

            all_pil_images.append(Image.fromarray(img_arr))

        return self.processor(images=all_pil_images, **kwargs)

    def save_geotiff(
        self,
        data: np.ndarray,
        output_path: str,
        metadata: Dict,
        dtype: Optional[str] = None,
        compress: str = "lzw",
        nodata: Optional[float] = None,
    ) -> str:
        """Save array as GeoTIFF with geospatial metadata.

        Args:
            data: Array to save (2D or 3D in CHW format).
            output_path: Output file path.
            metadata: Metadata dictionary from load_geotiff.
            dtype: Output data type. If None, infer from data.
            compress: Compression method.
            nodata: NoData value.

        Returns:
            Path to saved file.
        """
        profile = metadata["profile"].copy()

        if dtype is None:
            dtype = str(data.dtype)

        # Handle 2D vs 3D arrays
        if data.ndim == 2:
            count = 1
            height, width = data.shape
        else:
            count = data.shape[0]
            height, width = data.shape[1], data.shape[2]

        profile.update(
            {
                "dtype": dtype,
                "count": count,
                "height": height,
                "width": width,
                "compress": compress,
            }
        )

        if nodata is not None:
            profile["nodata"] = nodata

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                dst.write(data)

        return output_path


class AutoGeoModel:
    """
    Auto model for geospatial inference with GeoTIFF support.

    This class wraps Hugging Face transformers' AutoModel classes and adds
    geospatial capabilities for processing GeoTIFF images and saving results
    as georeferenced outputs (GeoTIFF or vector data).

    Use `from_pretrained` to instantiate this class, following the transformers pattern.

    Attributes:
        model: The underlying transformers model instance.
        processor: AutoGeoImageProcessor for preprocessing.
        device (str): The device being used ('cuda' or 'cpu').
        task (str): The task type.
        tile_size (int): Size of tiles for processing large images.
        overlap (int): Overlap between tiles.

    Example:
        >>> model = AutoGeoModel.from_pretrained(
        ...     "facebook/sam-vit-base",
        ...     task="mask-generation"
        ... )
        >>> result = model.predict("input.tif", output_path="output.tif")
    """

    TASK_MODEL_MAPPING = {
        "segmentation": AutoModelForSemanticSegmentation,
        "semantic-segmentation": AutoModelForSemanticSegmentation,
        "image-segmentation": AutoModelForImageSegmentation,
        "universal-segmentation": AutoModelForUniversalSegmentation,
        "depth-estimation": AutoModelForDepthEstimation,
        "mask-generation": AutoModelForMaskGeneration,
        "object-detection": AutoModelForObjectDetection,
        "zero-shot-object-detection": AutoModelForZeroShotObjectDetection,
        "classification": AutoModelForImageClassification,
        "image-classification": AutoModelForImageClassification,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Optional["AutoGeoImageProcessor"] = None,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        device: Optional[str] = None,
        tile_size: int = 1024,
        overlap: int = 128,
    ) -> None:
        """Initialize AutoGeoModel with an existing model.

        Note: Use `from_pretrained` class method to load from Hugging Face Hub.

        Args:
            model: A transformers model instance.
            processor: An AutoGeoImageProcessor instance (optional).
            model_name: Name or path of the model (for reference).
            task: Task type for the model.
            device: Device to use ('cuda', 'cpu'). If None, auto-detect.
            tile_size: Size of tiles for processing large images.
            overlap: Overlap between tiles.
        """
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.task = task
        self.tile_size = tile_size
        self.overlap = overlap

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # Ensure model is on the correct device and in eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        task: Optional[str] = None,
        device: Optional[str] = None,
        tile_size: int = 1024,
        overlap: int = 128,
        **kwargs: Any,
    ) -> "AutoGeoModel":
        """Load an AutoGeoModel from a pretrained model.

        This method wraps transformers' AutoModel.from_pretrained and adds
        geospatial capabilities for processing GeoTIFF files.

        Args:
            pretrained_model_name_or_path: Hugging Face model name or local path.
                Can be a model ID from huggingface.co or a local directory path.
            task: Task type for automatic model class selection. Options:
                - 'segmentation' or 'semantic-segmentation': Semantic segmentation
                - 'image-segmentation': General image segmentation
                - 'universal-segmentation': Universal segmentation (Mask2Former, etc.)
                - 'depth-estimation': Depth estimation
                - 'mask-generation': Mask generation (SAM, etc.)
                - 'object-detection': Object detection
                - 'zero-shot-object-detection': Zero-shot object detection
                - 'classification' or 'image-classification': Image classification
                If None, will try to infer from model config.
            device: Device to use ('cuda', 'cpu'). If None, auto-detect.
            tile_size: Size of tiles for processing large images.
            overlap: Overlap between tiles to avoid edge artifacts.
            **kwargs: Additional arguments passed to the model's from_pretrained.
                Common options include:
                - trust_remote_code (bool): Whether to trust remote code.
                - revision (str): Specific model version to use.
                - torch_dtype: Data type for model weights.

        Returns:
            AutoGeoModel instance with geospatial support.

        Example:
            >>> model = AutoGeoModel.from_pretrained("facebook/sam-vit-base", task="mask-generation")
            >>> model = AutoGeoModel.from_pretrained(
            ...     "nvidia/segformer-b0-finetuned-ade-512-512",
            ...     task="semantic-segmentation",
            ...     device="cuda"
            ... )
        """
        # Determine device
        if device is None:
            device = get_device()

        # Load model using appropriate auto class
        model = cls._load_model_from_pretrained(
            pretrained_model_name_or_path, task, **kwargs
        )

        # Load processor - use full processor for models that need text inputs
        needs_full_processor = task in (
            "zero-shot-object-detection",
            "object-detection",
        )
        try:
            processor = AutoGeoImageProcessor.from_pretrained(
                pretrained_model_name_or_path,
                device=device,
                use_full_processor=needs_full_processor,
            )
        except Exception:
            processor = None

        instance = cls(
            model=model,
            processor=processor,
            model_name=pretrained_model_name_or_path,
            task=task,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
        )

        print(f"Model loaded on {device}")
        return instance

    @classmethod
    def _load_model_from_pretrained(
        cls,
        model_name_or_path: str,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Load the appropriate model based on task using from_pretrained."""
        if task and task in cls.TASK_MODEL_MAPPING:
            model_class = cls.TASK_MODEL_MAPPING[task]
            return model_class.from_pretrained(model_name_or_path, **kwargs)

        # Try to infer from config
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            architectures = getattr(config, "architectures", [])

            if any("Segmentation" in arch for arch in architectures):
                return AutoModelForSemanticSegmentation.from_pretrained(
                    model_name_or_path, **kwargs
                )
            elif any("Detection" in arch for arch in architectures):
                return AutoModelForObjectDetection.from_pretrained(
                    model_name_or_path, **kwargs
                )
            elif any("Classification" in arch for arch in architectures):
                return AutoModelForImageClassification.from_pretrained(
                    model_name_or_path, **kwargs
                )
            else:
                return AutoModel.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            return AutoModel.from_pretrained(model_name_or_path, **kwargs)

    def predict(
        self,
        source: Union[str, np.ndarray, Image.Image],
        output_path: Optional[str] = None,
        output_vector_path: Optional[str] = None,
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
        threshold: float = 0.5,
        text: Optional[str] = None,
        labels: Optional[List[str]] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        min_object_area: int = 100,
        simplify_tolerance: float = 1.0,
        batch_size: int = 1,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run inference on a GeoTIFF or image.

        Args:
            source: Input image (path, array, or PIL Image). Can also be a URL.
            output_path: Path to save output GeoTIFF.
            output_vector_path: Path to save output vector file (GeoJSON, GPKG, etc.).
            window: Optional rasterio Window for processing a subset.
            bands: List of band indices to use (1-indexed).
            threshold: Threshold for binary masks (segmentation tasks).
            text: Text prompt for zero-shot detection models (e.g., "a cat. a dog.").
                For Grounding DINO, labels should be lowercase and end with a dot.
            labels: List of labels to detect (alternative to text).
                Will be converted to text prompt format automatically.
            box_threshold: Confidence threshold for bounding boxes (detection tasks).
            text_threshold: Text similarity threshold for zero-shot detection.
            min_object_area: Minimum object area in pixels for vectorization.
            simplify_tolerance: Tolerance for polygon simplification.
            batch_size: Batch size for processing tiles.
            return_probabilities: Whether to return probability maps.
            **kwargs: Additional arguments for specific tasks.

        Returns:
            Dictionary with results including mask/detections, metadata, and optional vector data.

        Example:
            >>> # Zero-shot object detection
            >>> model = AutoGeoModel.from_pretrained(
            ...     "IDEA-Research/grounding-dino-base",
            ...     task="zero-shot-object-detection"
            ... )
            >>> result = model.predict(
            ...     "image.jpg",
            ...     text="a building. a car. a tree.",
            ...     box_threshold=0.3
            ... )
            >>> print(result["boxes"], result["labels"])
        """
        # Convert labels list to text format if provided
        if labels is not None and text is None:
            text = " ".join(f"{label.lower().strip()}." for label in labels)

        # Handle zero-shot object detection separately
        if self.task in ("zero-shot-object-detection", "object-detection"):
            return self._predict_detection(
                source,
                text=text,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                output_vector_path=output_vector_path,
                **kwargs,
            )

        # Load image (handles URLs, local files, arrays, PIL Images)
        pil_image = None
        if isinstance(source, str):
            # Check if URL
            if source.startswith(("http://", "https://")):
                pil_image = Image.open(requests.get(source, stream=True).raw)
                metadata = None
                data = np.array(pil_image.convert("RGB")).transpose(2, 0, 1)
            else:
                # Local file - try to load with geospatial info
                if self.processor is not None:
                    data, metadata = self.processor.load_image(source, window, bands)
                else:
                    try:
                        with rasterio.open(source) as src:
                            data = src.read(bands) if bands else src.read()
                            profile = src.profile.copy()
                            metadata = {
                                "profile": profile,
                                "crs": src.crs,
                                "transform": src.transform,
                                "bounds": src.bounds,
                                "width": src.width,
                                "height": src.height,
                            }
                    except Exception:
                        # Fall back to PIL for regular images
                        pil_image = Image.open(source).convert("RGB")
                        data = np.array(pil_image).transpose(2, 0, 1)
                        metadata = None
        elif isinstance(source, Image.Image):
            pil_image = source
            data = np.array(source.convert("RGB")).transpose(2, 0, 1)
            metadata = None
        else:
            data = np.array(source)
            if data.ndim == 3 and data.shape[2] in [1, 3, 4]:
                data = data.transpose(2, 0, 1)
            metadata = None

        # Check if we need tiled processing (not for classification tasks)
        if data.ndim == 3:
            _, height, width = data.shape
        elif data.ndim == 2:
            height, width = data.shape
            _ = 1
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Classification tasks should not use tiled processing
        use_tiled = (
            height > self.tile_size or width > self.tile_size
        ) and self.task not in ("classification", "image-classification")

        if use_tiled:
            result = self._predict_tiled(
                source,
                data,
                metadata,
                threshold=threshold,
                batch_size=batch_size,
                return_probabilities=return_probabilities,
                **kwargs,
            )
        else:
            result = self._predict_single(
                data,
                metadata,
                threshold=threshold,
                return_probabilities=return_probabilities,
                **kwargs,
            )

        # Save GeoTIFF output
        if output_path and metadata:
            self.save_geotiff(
                result.get("mask", result.get("output")),
                output_path,
                metadata,
                nodata=0,
            )
            result["output_path"] = output_path

        # Save vector output
        if output_vector_path and metadata and "mask" in result:
            gdf = self.mask_to_vector(
                result["mask"],
                metadata,
                threshold=threshold,
                min_object_area=min_object_area,
                simplify_tolerance=simplify_tolerance,
            )
            if gdf is not None and len(gdf) > 0:
                gdf.to_file(output_vector_path)
                result["vector_path"] = output_vector_path
                result["geodataframe"] = gdf

        return result

    def _predict_single(
        self,
        data: np.ndarray,
        metadata: Optional[Dict],
        threshold: float = 0.5,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run inference on a single image."""
        # Prepare input
        if self.processor is not None:
            inputs = self.processor.prepare_for_model(data)
        else:
            # Fallback preparation
            img = data.transpose(1, 2, 0) if data.ndim == 3 else data
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] > 3:
                img = img[..., :3]

            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
                    np.uint8
                )

            pil_image = Image.fromarray(img)
            inputs = {
                "pixel_values": torch.from_numpy(
                    np.array(pil_image).transpose(2, 0, 1) / 255.0
                )
                .float()
                .unsqueeze(0)
            }

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs based on model type
        result = self._process_outputs(
            outputs, data.shape, threshold, return_probabilities
        )
        result["metadata"] = metadata

        return result

    def _predict_tiled(
        self,
        source: Union[str, np.ndarray],
        data: np.ndarray,
        metadata: Optional[Dict],
        threshold: float = 0.5,
        batch_size: int = 1,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run tiled inference for large images."""
        if data.ndim == 3:
            _, height, width = data.shape
        else:
            height, width = data.shape

        effective_tile_size = self.tile_size - 2 * self.overlap

        n_tiles_x = max(1, int(np.ceil(width / effective_tile_size)))
        n_tiles_y = max(1, int(np.ceil(height / effective_tile_size)))
        total_tiles = n_tiles_x * n_tiles_y

        # Initialize output arrays
        mask_output = np.zeros((height, width), dtype=np.float32)
        count_output = np.zeros((height, width), dtype=np.float32)

        print(f"Processing {total_tiles} tiles ({n_tiles_x}x{n_tiles_y})")

        with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
            for y in range(n_tiles_y):
                for x in range(n_tiles_x):
                    # Calculate tile coordinates
                    x_start = max(0, x * effective_tile_size - self.overlap)
                    y_start = max(0, y * effective_tile_size - self.overlap)
                    x_end = min(width, (x + 1) * effective_tile_size + self.overlap)
                    y_end = min(height, (y + 1) * effective_tile_size + self.overlap)

                    # Extract tile
                    if data.ndim == 3:
                        tile = data[:, y_start:y_end, x_start:x_end]
                    else:
                        tile = data[y_start:y_end, x_start:x_end]

                    try:
                        # Run inference on tile
                        tile_result = self._predict_single(
                            tile, None, threshold, return_probabilities
                        )

                        tile_mask = tile_result.get("mask", tile_result.get("output"))
                        if tile_mask is not None:
                            # Handle different dimensions
                            if tile_mask.ndim > 2:
                                tile_mask = tile_mask.squeeze()
                            if tile_mask.ndim > 2:
                                tile_mask = tile_mask[0]

                            # Resize if necessary
                            tile_h, tile_w = y_end - y_start, x_end - x_start
                            if tile_mask.shape != (tile_h, tile_w):
                                tile_mask = cv2.resize(
                                    tile_mask.astype(np.float32),
                                    (tile_w, tile_h),
                                    interpolation=cv2.INTER_LINEAR,
                                )

                            # Add to output with blending
                            mask_output[y_start:y_end, x_start:x_end] += tile_mask
                            count_output[y_start:y_end, x_start:x_end] += 1

                    except Exception as e:
                        print(f"Error processing tile ({x}, {y}): {e}")

                    pbar.update(1)

        # Average overlapping regions
        count_output = np.maximum(count_output, 1)
        mask_output = mask_output / count_output

        result = {
            "mask": (mask_output > threshold).astype(np.uint8),
            "probabilities": mask_output if return_probabilities else None,
            "metadata": metadata,
        }

        return result

    def _predict_detection(
        self,
        source: Union[str, np.ndarray, Image.Image],
        text: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        output_vector_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run object detection inference.

        Args:
            source: Input image (path, URL, array, or PIL Image).
            text: Text prompt for zero-shot detection (e.g., "a cat. a dog.").
            box_threshold: Confidence threshold for bounding boxes.
            text_threshold: Text similarity threshold for zero-shot detection.
            output_vector_path: Path to save detection results as vector file.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with boxes, scores, labels, and optional GeoDataFrame.
        """
        # Load image
        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                pil_image = Image.open(requests.get(source, stream=True).raw)
            else:
                try:
                    # Try loading with rasterio for GeoTIFF
                    with rasterio.open(source) as src:
                        data = src.read()
                        if data.shape[0] > 3:
                            data = data[:3]
                        elif data.shape[0] == 1:
                            data = np.repeat(data, 3, axis=0)
                        # Normalize to uint8
                        if data.dtype != np.uint8:
                            for i in range(data.shape[0]):
                                band = data[i].astype(np.float32)
                                p2, p98 = np.percentile(band, [2, 98])
                                if p98 > p2:
                                    data[i] = np.clip(
                                        (band - p2) / (p98 - p2) * 255, 0, 255
                                    )
                                else:
                                    data[i] = 0
                            data = data.astype(np.uint8)
                        pil_image = Image.fromarray(data.transpose(1, 2, 0))
                except Exception:
                    pil_image = Image.open(source)
        elif isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[0] in [1, 3, 4]:
                source = source.transpose(1, 2, 0)
            if source.dtype != np.uint8:
                source = (
                    (source - source.min()) / (source.max() - source.min()) * 255
                ).astype(np.uint8)
            pil_image = Image.fromarray(source)
        else:
            pil_image = source

        pil_image = pil_image.convert("RGB")
        image_size = pil_image.size[::-1]  # (height, width)

        # Get the underlying processor
        processor = self.processor.processor if self.processor else None
        if processor is None:
            # Load processor directly if not available
            processor = AutoProcessor.from_pretrained(self.model_name)

        # Prepare inputs based on task type
        if self.task == "zero-shot-object-detection":
            if text is None:
                raise ValueError(
                    "Text prompt is required for zero-shot object detection. "
                    "Provide text='a cat. a dog.' or labels=['cat', 'dog']"
                )
            # Use the processor to prepare inputs with text
            inputs = processor(images=pil_image, text=text, return_tensors="pt")
        else:
            # Standard object detection
            inputs = processor(images=pil_image, return_tensors="pt")

        # Move to device - use .to() method for BatchFeature objects
        inputs = inputs.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        result = self._process_detection_outputs(
            outputs,
            inputs,
            text,
            image_size,
            box_threshold,
            text_threshold,
            processor,
        )

        # Convert to GeoDataFrame if requested
        if output_vector_path and result.get("boxes") is not None:
            gdf = self._detections_to_geodataframe(result, pil_image.size)
            if gdf is not None and len(gdf) > 0:
                gdf.to_file(output_vector_path)
                result["vector_path"] = output_vector_path
                result["geodataframe"] = gdf

        return result

    def _process_detection_outputs(
        self,
        outputs: Any,
        inputs: Dict,
        text: Optional[str],
        image_size: Tuple[int, int],
        box_threshold: float,
        text_threshold: float,
        processor: Any = None,
    ) -> Dict[str, Any]:
        """Process detection model outputs."""
        result = {}

        if processor is None:
            processor = self.processor.processor if self.processor else None

        if self.task == "zero-shot-object-detection":
            # Use processor's post-processing for grounded detection
            try:
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=box_threshold,  # box confidence threshold
                    text_threshold=text_threshold,
                    target_sizes=[image_size],
                )
                if results and len(results) > 0:
                    r = results[0]
                    result["boxes"] = r["boxes"].cpu().numpy()
                    result["scores"] = r["scores"].cpu().numpy()
                    # Handle different output formats for labels
                    if "labels" in r:
                        result["labels"] = r["labels"]
                    elif "text_labels" in r:
                        result["labels"] = r["text_labels"]
                    else:
                        # Extract labels from logits if not provided
                        result["labels"] = [
                            f"object_{i}" for i in range(len(r["boxes"]))
                        ]
            except Exception as e:
                # Fallback for models without grounded post-processing
                print(f"Warning: Using fallback detection processing: {e}")
                if hasattr(outputs, "pred_boxes"):
                    boxes = outputs.pred_boxes[0].cpu().numpy()
                    logits = outputs.logits[0].cpu()
                    scores = logits.sigmoid().max(dim=-1).values.numpy()
                    mask = scores > box_threshold
                    result["boxes"] = boxes[mask]
                    result["scores"] = scores[mask]
                    result["labels"] = [
                        f"object_{i}" for i in range(len(result["boxes"]))
                    ]
        else:
            # Standard object detection post-processing
            if hasattr(outputs, "pred_boxes") and processor is not None:
                target_sizes = torch.tensor([image_size], device=self.device)
                results = processor.post_process_object_detection(
                    outputs, threshold=box_threshold, target_sizes=target_sizes
                )
                if results and len(results) > 0:
                    r = results[0]
                    result["boxes"] = r["boxes"].cpu().numpy()
                    result["scores"] = r["scores"].cpu().numpy()
                    result["labels"] = r["labels"].cpu().numpy()

        return result

    def _detections_to_geodataframe(
        self,
        detections: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Optional[gpd.GeoDataFrame]:
        """Convert detection results to a GeoDataFrame.

        Note: Without geospatial metadata, coordinates are in pixel space.
        """
        boxes = detections.get("boxes")
        if boxes is None or len(boxes) == 0:
            return None

        scores = detections.get("scores", [None] * len(boxes))
        labels = detections.get("labels", [None] * len(boxes))

        geometries = []
        for bbox in boxes:
            # Convert [x1, y1, x2, y2] to polygon
            x1, y1, x2, y2 = bbox
            geometries.append(box(x1, y1, x2, y2))

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "score": scores,
                "label": labels,
            }
        )

        return gdf

    def _process_outputs(
        self,
        outputs: Any,
        input_shape: Tuple,
        threshold: float = 0.5,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Process model outputs to extract masks or predictions."""
        result = {}

        # Handle different output types
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            if logits.dim() == 4:  # Segmentation output
                # Upsample if needed
                if logits.shape[2:] != input_shape[1:]:
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(input_shape[1], input_shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    )

                probs = torch.softmax(logits, dim=1)
                mask = probs.argmax(dim=1).squeeze().cpu().numpy()
                result["mask"] = mask.astype(np.uint8)

                if return_probabilities:
                    result["probabilities"] = probs.squeeze().cpu().numpy()

            elif logits.dim() == 2:  # Classification output
                probs = torch.softmax(logits, dim=-1)
                pred_class = probs.argmax(dim=-1).item()
                result["class"] = pred_class
                result["probabilities"] = probs.squeeze().cpu().numpy()

        elif hasattr(outputs, "pred_masks"):
            masks = outputs.pred_masks.squeeze().cpu().numpy()
            if masks.ndim == 3:
                mask = masks.max(axis=0)
            else:
                mask = masks
            result["mask"] = (mask > threshold).astype(np.uint8)

            if return_probabilities:
                result["probabilities"] = mask

        elif hasattr(outputs, "predicted_depth"):
            depth = outputs.predicted_depth.squeeze().cpu().numpy()
            result["output"] = depth
            result["depth"] = depth

        elif hasattr(outputs, "masks"):
            # SAM-like output
            masks = outputs.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if masks.ndim == 4:
                masks = masks.squeeze(0)
            if masks.ndim == 3:
                mask = masks.max(axis=0)
            else:
                mask = masks
            result["mask"] = (mask > threshold).astype(np.uint8)
            result["all_masks"] = masks

        else:
            # Generic output handling
            if hasattr(outputs, "last_hidden_state"):
                result["features"] = outputs.last_hidden_state.cpu().numpy()
            else:
                result["output"] = outputs

        return result

    def mask_to_vector(
        self,
        mask: np.ndarray,
        metadata: Dict,
        threshold: float = 0.5,
        min_object_area: int = 100,
        max_object_area: Optional[int] = None,
        simplify_tolerance: float = 1.0,
    ) -> Optional[gpd.GeoDataFrame]:
        """Convert a raster mask to vector polygons.

        Args:
            mask: Binary or probability mask array.
            metadata: Geospatial metadata dictionary.
            threshold: Threshold for binarizing probability masks.
            min_object_area: Minimum area in pixels for valid objects.
            max_object_area: Maximum area in pixels (optional).
            simplify_tolerance: Tolerance for polygon simplification.

        Returns:
            GeoDataFrame with polygon geometries, or None if no valid polygons.
        """
        if metadata is None or metadata.get("crs") is None:
            print("Warning: No CRS information available for vectorization")
            return None

        # Ensure binary mask
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask > threshold).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)

        # Get transform
        transform = metadata.get("transform")
        crs = metadata.get("crs")

        if transform is None:
            print("Warning: No transform available for vectorization")
            return None

        # Extract shapes using rasterio
        polygons = []
        values = []

        try:
            for geom, value in shapes(mask, transform=transform):
                if value > 0:  # Only keep non-background
                    poly = shape(geom)

                    # Filter by area
                    pixel_area = poly.area / (transform.a * abs(transform.e))
                    if pixel_area < min_object_area:
                        continue
                    if max_object_area and pixel_area > max_object_area:
                        continue

                    # Simplify
                    if simplify_tolerance > 0:
                        poly = poly.simplify(
                            simplify_tolerance * abs(transform.a),
                            preserve_topology=True,
                        )

                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)
                        values.append(value)

        except Exception as e:
            print(f"Error during vectorization: {e}")
            return None

        if not polygons:
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"geometry": polygons, "class": values},
            crs=crs,
        )

        return gdf

    def save_geotiff(
        self,
        data: np.ndarray,
        output_path: str,
        metadata: Dict,
        dtype: Optional[str] = None,
        compress: str = "lzw",
        nodata: Optional[float] = None,
    ) -> str:
        """Save array as GeoTIFF with geospatial metadata.

        Args:
            data: Array to save (2D or 3D in CHW format).
            output_path: Output file path.
            metadata: Metadata dictionary from load_geotiff.
            dtype: Output data type. If None, infer from data.
            compress: Compression method.
            nodata: NoData value.

        Returns:
            Path to saved file.
        """
        profile = metadata["profile"].copy()

        if dtype is None:
            dtype = str(data.dtype)

        # Handle 2D vs 3D arrays
        if data.ndim == 2:
            count = 1
            height, width = data.shape
        else:
            count = data.shape[0]
            height, width = data.shape[1], data.shape[2]

        profile.update(
            {
                "dtype": dtype,
                "count": count,
                "height": height,
                "width": width,
                "compress": compress,
            }
        )

        if nodata is not None:
            profile["nodata"] = nodata

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                dst.write(data)

        return output_path

    def save_vector(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str,
        driver: Optional[str] = None,
    ) -> str:
        """Save GeoDataFrame to file.

        Args:
            gdf: GeoDataFrame to save.
            output_path: Output file path.
            driver: File driver (auto-detected from extension if None).

        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if driver is None:
            ext = os.path.splitext(output_path)[1].lower()
            driver_map = {
                ".geojson": "GeoJSON",
                ".json": "GeoJSON",
                ".gpkg": "GPKG",
                ".shp": "ESRI Shapefile",
                ".parquet": "Parquet",
                ".fgb": "FlatGeobuf",
            }
            driver = driver_map.get(ext, "GeoJSON")

        gdf.to_file(output_path, driver=driver)
        return output_path


def semantic_segmentation(
    input_path: str,
    output_path: str,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    output_vector_path: Optional[str] = None,
    threshold: float = 0.5,
    tile_size: int = 1024,
    overlap: int = 128,
    min_object_area: int = 100,
    simplify_tolerance: float = 1.0,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform semantic segmentation on a GeoTIFF image.

    Args:
        input_path: Path to input GeoTIFF.
        output_path: Path to save output segmentation GeoTIFF.
        model_name: Hugging Face model name.
        output_vector_path: Optional path to save vectorized output.
        threshold: Threshold for binary masks.
        tile_size: Size of tiles for processing large images.
        overlap: Overlap between tiles.
        min_object_area: Minimum object area for vectorization.
        simplify_tolerance: Tolerance for polygon simplification.
        device: Device to use ('cuda', 'cpu').
        **kwargs: Additional arguments for prediction.

    Returns:
        Dictionary with results.

    Example:
        >>> result = semantic_segmentation(
        ...     "input.tif",
        ...     "output.tif",
        ...     model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        ...     output_vector_path="output.geojson"
        ... )
    """
    model = AutoGeoModel.from_pretrained(
        model_name,
        task="semantic-segmentation",
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )

    return model.predict(
        input_path,
        output_path=output_path,
        output_vector_path=output_vector_path,
        threshold=threshold,
        min_object_area=min_object_area,
        simplify_tolerance=simplify_tolerance,
        **kwargs,
    )


def depth_estimation(
    input_path: str,
    output_path: str,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    tile_size: int = 1024,
    overlap: int = 128,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform depth estimation on a GeoTIFF image.

    Args:
        input_path: Path to input GeoTIFF.
        output_path: Path to save output depth GeoTIFF.
        model_name: Hugging Face model name.
        tile_size: Size of tiles for processing large images.
        overlap: Overlap between tiles.
        device: Device to use ('cuda', 'cpu').
        **kwargs: Additional arguments for prediction.

    Returns:
        Dictionary with results.

    Example:
        >>> result = depth_estimation(
        ...     "input.tif",
        ...     "depth_output.tif",
        ...     model_name="depth-anything/Depth-Anything-V2-Small-hf"
        ... )
    """
    model = AutoGeoModel.from_pretrained(
        model_name,
        task="depth-estimation",
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )

    return model.predict(input_path, output_path=output_path, **kwargs)


def image_classification(
    input_path: str,
    model_name: str = "google/vit-base-patch16-224",
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform image classification on a GeoTIFF image.

    Args:
        input_path: Path to input GeoTIFF.
        model_name: Hugging Face model name.
        device: Device to use ('cuda', 'cpu').
        **kwargs: Additional arguments for prediction.

    Returns:
        Dictionary with classification results.

    Example:
        >>> result = image_classification(
        ...     "input.tif",
        ...     model_name="google/vit-base-patch16-224"
        ... )
        >>> print(result['class'], result['probabilities'])
    """
    model = AutoGeoModel.from_pretrained(
        model_name,
        task="image-classification",
        device=device,
    )

    return model.predict(input_path, **kwargs)


def object_detection(
    input_path: str,
    text: Optional[str] = None,
    labels: Optional[List[str]] = None,
    model_name: str = "IDEA-Research/grounding-dino-base",
    output_vector_path: Optional[str] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform object detection on an image using Grounding DINO or similar models.

    Args:
        input_path: Path to input image or URL.
        text: Text prompt for detection (e.g., "a building. a car.").
            Labels should be lowercase and end with a dot.
        labels: List of labels to detect (alternative to text).
            Will be converted to text format automatically.
        model_name: Hugging Face model name.
        output_vector_path: Optional path to save detection boxes as vector file.
        box_threshold: Confidence threshold for bounding boxes.
        text_threshold: Text similarity threshold for zero-shot detection.
        device: Device to use ('cuda', 'cpu').
        **kwargs: Additional arguments for prediction.

    Returns:
        Dictionary with detection results (boxes, scores, labels).

    Example:
        >>> result = object_detection(
        ...     "image.jpg",
        ...     labels=["car", "building", "tree"],
        ...     box_threshold=0.3
        ... )
        >>> print(result["boxes"], result["labels"])
    """
    # Determine task type based on model
    task = "zero-shot-object-detection"
    if "grounding-dino" not in model_name.lower() and "owl" not in model_name.lower():
        task = "object-detection"

    model = AutoGeoModel.from_pretrained(
        model_name,
        task=task,
        device=device,
    )

    return model.predict(
        input_path,
        text=text,
        labels=labels,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_vector_path=output_vector_path,
        **kwargs,
    )


def get_hf_tasks() -> List[str]:
    """Get all supported Hugging Face tasks for this module.

    Returns:
        List of supported task names.
    """
    from transformers.pipelines import SUPPORTED_TASKS

    return sorted(list(SUPPORTED_TASKS.keys()))


def get_hf_model_config(model_id: str) -> Dict[str, Any]:
    """Get the model configuration for a Hugging Face model.

    Args:
        model_id: The Hugging Face model ID (e.g., "facebook/sam-vit-base").

    Returns:
        Dictionary representation of the model config.

    Example:
        >>> config = get_hf_model_config("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> print(config.get("model_type"))
    """
    cfg = AutoConfig.from_pretrained(model_id)
    return cfg.to_dict()


# =============================================================================
# Visualization Functions
# =============================================================================


def _load_image_for_display(
    source: Union[str, np.ndarray, Image.Image],
) -> Tuple[np.ndarray, Optional[Dict]]:
    """Load an image for display purposes.

    Args:
        source: Image source (path, array, or PIL Image).

    Returns:
        Tuple of (RGB image array in HWC format, metadata dict or None).
    """
    metadata = None

    if isinstance(source, str):
        try:
            with rasterio.open(source) as src:
                data = src.read()
                metadata = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "bounds": src.bounds,
                }
                # Convert to HWC
                if data.shape[0] > 3:
                    data = data[:3]
                elif data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                img = data.transpose(1, 2, 0)

                # Normalize to uint8
                if img.dtype != np.uint8:
                    for i in range(img.shape[-1]):
                        band = img[..., i].astype(np.float32)
                        p2, p98 = np.percentile(band, [2, 98])
                        if p98 > p2:
                            img[..., i] = np.clip(
                                (band - p2) / (p98 - p2) * 255, 0, 255
                            )
                        else:
                            img[..., i] = 0
                    img = img.astype(np.uint8)
                return img, metadata
        except Exception:
            pass

        # Try as regular image
        img = np.array(Image.open(source).convert("RGB"))
        return img, None

    elif isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[0] in [1, 3, 4]:
            source = source.transpose(1, 2, 0)
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        elif source.shape[-1] > 3:
            source = source[..., :3]
        if source.dtype != np.uint8:
            source = (
                (source - source.min()) / (source.max() - source.min() + 1e-8) * 255
            ).astype(np.uint8)
        return source, None

    elif isinstance(source, Image.Image):
        return np.array(source.convert("RGB")), None

    else:
        raise TypeError(f"Unsupported source type: {type(source)}")


def show_image(
    source: Union[str, np.ndarray, Image.Image],
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    axis_off: bool = True,
    **kwargs: Any,
) -> "plt.Figure":
    """Display an image (GeoTIFF or regular image).

    Args:
        source: Image source (path to file, numpy array, or PIL Image).
        figsize: Figure size as (width, height).
        title: Optional title for the plot.
        axis_off: Whether to hide axes.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib figure object.

    Example:
        >>> fig = show_image("aerial.tif", title="Aerial Image")
    """
    import matplotlib.pyplot as plt

    img, _ = _load_image_for_display(source)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, **kwargs)

    if title:
        ax.set_title(title)
    if axis_off:
        ax.axis("off")

    plt.tight_layout()
    return fig


def show_detections(
    source: Union[str, np.ndarray, Image.Image],
    detections: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    box_color: str = "red",
    text_color: str = "white",
    linewidth: int = 2,
    fontsize: int = 10,
    show_scores: bool = True,
    axis_off: bool = True,
    **kwargs: Any,
) -> "plt.Figure":
    """Display an image with detection bounding boxes.

    Args:
        source: Image source (path to file, numpy array, or PIL Image).
        detections: Detection results dictionary with 'boxes', 'scores', 'labels'.
        figsize: Figure size as (width, height).
        title: Optional title for the plot.
        box_color: Color for bounding boxes (can be single color or list).
        text_color: Color for label text.
        linewidth: Width of bounding box lines.
        fontsize: Font size for labels.
        show_scores: Whether to show confidence scores.
        axis_off: Whether to hide axes.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib figure object.

    Example:
        >>> result = geoai.auto.object_detection("aerial.tif", labels=["building", "tree"])
        >>> fig = show_detections("aerial.tif", result)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img, _ = _load_image_for_display(source)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, **kwargs)

    boxes = detections.get("boxes", [])
    scores = detections.get("scores", [None] * len(boxes))
    labels = detections.get("labels", [None] * len(boxes))

    # Handle color list
    if isinstance(box_color, str):
        colors = [box_color] * len(boxes)
    else:
        colors = box_color

    for i, (bbox, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        color = colors[i % len(colors)]

        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label
        if label is not None:
            text = str(label)
            if show_scores and score is not None:
                text = f"{label}: {score:.2f}"
            ax.text(
                x1,
                y1 - 5,
                text,
                color=text_color,
                fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            )

    if title:
        ax.set_title(title)
    if axis_off:
        ax.axis("off")

    plt.tight_layout()
    return fig


def show_segmentation(
    source: Union[str, np.ndarray, Image.Image],
    mask: np.ndarray,
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None,
    alpha: float = 0.5,
    cmap: str = "tab20",
    show_original: bool = True,
    axis_off: bool = True,
    **kwargs: Any,
) -> "plt.Figure":
    """Display segmentation results overlaid on the original image.

    Args:
        source: Image source (path to file, numpy array, or PIL Image).
        mask: Segmentation mask array.
        figsize: Figure size as (width, height).
        title: Optional title for the plot.
        alpha: Transparency of the mask overlay.
        cmap: Colormap for the segmentation mask.
        show_original: Whether to show original image side-by-side.
        axis_off: Whether to hide axes.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib figure object.

    Example:
        >>> result = geoai.auto.semantic_segmentation("aerial.tif", output_path="seg.tif")
        >>> fig = show_segmentation("aerial.tif", result["mask"])
    """
    import matplotlib.pyplot as plt

    img, _ = _load_image_for_display(source)

    # Resize mask if necessary
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.float32),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(mask.dtype)

    if show_original:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].imshow(img, **kwargs)
        axes[0].set_title("Original Image")
        if axis_off:
            axes[0].axis("off")

        axes[1].imshow(img, **kwargs)
        axes[1].imshow(mask, alpha=alpha, cmap=cmap)
        axes[1].set_title(title or "Segmentation Overlay")
        if axis_off:
            axes[1].axis("off")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, **kwargs)
        ax.imshow(mask, alpha=alpha, cmap=cmap)
        if title:
            ax.set_title(title)
        if axis_off:
            ax.axis("off")

    plt.tight_layout()
    return fig


def show_depth(
    source: Union[str, np.ndarray, Image.Image],
    depth: np.ndarray,
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None,
    cmap: str = "plasma",
    show_original: bool = True,
    show_colorbar: bool = True,
    axis_off: bool = True,
    **kwargs: Any,
) -> "plt.Figure":
    """Display depth estimation results.

    Args:
        source: Image source (path to file, numpy array, or PIL Image).
        depth: Depth map array.
        figsize: Figure size as (width, height).
        title: Optional title for the plot.
        cmap: Colormap for the depth map.
        show_original: Whether to show original image side-by-side.
        show_colorbar: Whether to show a colorbar.
        axis_off: Whether to hide axes.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib figure object.

    Example:
        >>> result = geoai.auto.depth_estimation("aerial.tif", output_path="depth.tif")
        >>> fig = show_depth("aerial.tif", result["depth"])
    """
    import matplotlib.pyplot as plt

    img, _ = _load_image_for_display(source)

    # Resize depth if necessary
    if depth.shape[:2] != img.shape[:2]:
        depth = cv2.resize(
            depth.astype(np.float32),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if show_original:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].imshow(img, **kwargs)
        axes[0].set_title("Original Image")
        if axis_off:
            axes[0].axis("off")

        im = axes[1].imshow(depth, cmap=cmap)
        axes[1].set_title(title or "Depth Estimation")
        if axis_off:
            axes[1].axis("off")
        if show_colorbar:
            plt.colorbar(im, ax=axes[1], label="Relative Depth")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(depth, cmap=cmap)
        if title:
            ax.set_title(title)
        if axis_off:
            ax.axis("off")
        if show_colorbar:
            plt.colorbar(im, ax=ax, label="Relative Depth")

    plt.tight_layout()
    return fig
