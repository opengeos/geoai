"""ONNX Runtime support for geospatial model inference.

This module provides ONNXGeoModel for loading and running inference with
ONNX models on geospatial data (GeoTIFF), and export_to_onnx for converting
existing PyTorch/Hugging Face models to ONNX format.

Supported tasks:
    - Semantic segmentation (e.g., SegFormer, Mask2Former)
    - Image classification (e.g., ViT, ResNet)
    - Object detection (e.g., DETR, YOLOS)
    - Depth estimation (e.g., Depth Anything, DPT)

Requirements:
    - onnx
    - onnxruntime (or onnxruntime-gpu for GPU acceleration)

Install with::

    pip install geoai-py[onnx]

Example:
    >>> from geoai import export_to_onnx, ONNXGeoModel
    >>> # Export a HuggingFace model to ONNX
    >>> export_to_onnx(
    ...     "nvidia/segformer-b0-finetuned-ade-512-512",
    ...     "segformer.onnx",
    ...     task="semantic-segmentation",
    ... )
    >>> # Load and run inference with the ONNX model
    >>> model = ONNXGeoModel("segformer.onnx", task="semantic-segmentation")
    >>> result = model.predict("input.tif", output_path="output.tif")
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape
from tqdm import tqdm


def _check_onnx_deps() -> None:
    """Check that onnx and onnxruntime are installed.

    Raises:
        ImportError: If onnx or onnxruntime is not installed.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX support. "
            "Install it with: pip install geoai-py[onnx]"
        )

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'onnxruntime' package is required for ONNX support. "
            "Install it with: pip install geoai-py[onnx]  "
            "(use 'onnxruntime-gpu' for GPU acceleration)"
        )


def _check_torch_deps() -> None:
    """Check that torch and transformers are installed (needed for export).

    Raises:
        ImportError: If torch or transformers is not installed.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is required for exporting models to ONNX. "
            "Install it from https://pytorch.org/"
        )

    try:
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for exporting "
            "Hugging Face models to ONNX. "
            "Install it with: pip install transformers"
        )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_to_onnx(
    model_name_or_path: str,
    output_path: str,
    task: Optional[str] = None,
    input_height: int = 512,
    input_width: int = 512,
    input_channels: int = 3,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    simplify: bool = True,
    device: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Export a PyTorch / Hugging Face model to ONNX format.

    Args:
        model_name_or_path: Hugging Face model name or local checkpoint path.
        output_path: Path where the ``.onnx`` file will be saved.
        task: Model task.  One of ``"semantic-segmentation"``,
            ``"image-classification"``, ``"object-detection"``, or
            ``"depth-estimation"``.  If *None* the function tries to infer
            the task from the model configuration.
        input_height: Height of the dummy input tensor (pixels).
        input_width: Width of the dummy input tensor (pixels).
        input_channels: Number of input channels (default 3 for RGB).
        opset_version: ONNX opset version (default 17).
        dynamic_axes: Optional mapping of dynamic axes for variable-size
            inputs/outputs.  When *None* a sensible default is used so that
            batch size and spatial dimensions are dynamic.
        simplify: Whether to simplify the exported graph with
            ``onnxsim.simplify`` (requires the ``onnxsim`` package).
        device: Device used for tracing (``"cpu"`` recommended for export).
        **kwargs: Extra keyword arguments forwarded to
            ``AutoModel.from_pretrained``.

    Returns:
        Absolute path to the saved ONNX file.

    Raises:
        ImportError: If required packages are missing.
        ValueError: If the task cannot be determined.

    Example:
        >>> export_to_onnx(
        ...     "nvidia/segformer-b0-finetuned-ade-512-512",
        ...     "segformer.onnx",
        ...     task="semantic-segmentation",
        ... )
        'segformer.onnx'
    """
    _check_torch_deps()
    import onnx  # noqa: F811
    import torch
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        AutoModelForImageClassification,
        AutoModelForObjectDetection,
        AutoModelForSemanticSegmentation,
    )

    if device is None:
        device = "cpu"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    task_model_map = {
        "segmentation": AutoModelForSemanticSegmentation,
        "semantic-segmentation": AutoModelForSemanticSegmentation,
        "classification": AutoModelForImageClassification,
        "image-classification": AutoModelForImageClassification,
        "object-detection": AutoModelForObjectDetection,
        "depth-estimation": AutoModelForDepthEstimation,
    }

    if task and task in task_model_map:
        model_cls = task_model_map[task]
    else:
        # Try to infer from config
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            architectures = getattr(config, "architectures", [])
            if any("Segmentation" in a for a in architectures):
                model_cls = AutoModelForSemanticSegmentation
                task = task or "semantic-segmentation"
            elif any("Classification" in a for a in architectures):
                model_cls = AutoModelForImageClassification
                task = task or "image-classification"
            elif any("Detection" in a for a in architectures):
                model_cls = AutoModelForObjectDetection
                task = task or "object-detection"
            elif any("Depth" in a for a in architectures):
                model_cls = AutoModelForDepthEstimation
                task = task or "depth-estimation"
            else:
                raise ValueError(
                    f"Cannot infer task from model config. "
                    f"Found architectures: {architectures}. "
                    f"Please specify task= explicitly."
                )
        except Exception as exc:
            raise ValueError(
                "Cannot determine the model task. " "Please specify task= explicitly."
            ) from exc

    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    model = model.to(device).eval()

    # Try loading the image processor to get expected input size
    try:
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        if hasattr(processor, "size"):
            size = processor.size
            if isinstance(size, dict):
                input_height = size.get("height", input_height)
                input_width = size.get("width", input_width)
            elif isinstance(size, (list, tuple)) and len(size) == 2:
                input_height, input_width = size
    except Exception:
        pass  # processor introspection is optional; fall back to defaults

    # ------------------------------------------------------------------
    # Build dummy input & dynamic axes
    # ------------------------------------------------------------------
    dummy_input = torch.randn(
        1, input_channels, input_height, input_width, device=device
    )

    input_names = ["pixel_values"]

    if task in ("segmentation", "semantic-segmentation", "depth-estimation"):
        output_names = ["logits"]
    elif task in ("classification", "image-classification"):
        output_names = ["logits"]
    elif task == "object-detection":
        output_names = ["logits", "pred_boxes"]
    else:
        output_names = ["output"]

    if dynamic_axes is None:
        dynamic_axes = {
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        ({"pixel_values": dummy_input},),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Optional simplification
    if simplify:
        try:
            import onnxsim

            onnx_model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_simplified, output_path)
        except ImportError:
            pass  # onnxsim is optional
        except Exception:
            pass  # simplification can fail for some models; keep original

    # ------------------------------------------------------------------
    # Save metadata alongside the model
    # ------------------------------------------------------------------
    meta = {
        "model_name": model_name_or_path,
        "task": task,
        "input_height": input_height,
        "input_width": input_width,
        "input_channels": input_channels,
        "opset_version": opset_version,
        "output_names": output_names,
    }

    # Include id2label when available
    config = model.config if hasattr(model, "config") else None
    if config and hasattr(config, "id2label"):
        meta["id2label"] = {str(k): v for k, v in config.id2label.items()}
    if config and hasattr(config, "num_labels"):
        meta["num_labels"] = config.num_labels

    meta_path = output_path + ".json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"ONNX model exported to {output_path}")
    print(f"Metadata saved to {meta_path}")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# ONNXGeoModel
# ---------------------------------------------------------------------------


class ONNXGeoModel:
    """ONNX Runtime model for geospatial inference with GeoTIFF support.

    This class mirrors the :class:`~geoai.auto.AutoGeoModel` API but uses
    ONNX Runtime instead of PyTorch for inference, enabling deployment on
    edge devices and environments without GPU drivers.

    Attributes:
        session: The ``onnxruntime.InferenceSession`` instance.
        task (str): The model task (e.g. ``"semantic-segmentation"``).
        tile_size (int): Tile size used for processing large images.
        overlap (int): Overlap between adjacent tiles.
        metadata (dict): Model metadata loaded from the sidecar JSON file.

    Example:
        >>> model = ONNXGeoModel("segformer.onnx", task="semantic-segmentation")
        >>> result = model.predict("input.tif", output_path="output.tif")
    """

    def __init__(
        self,
        model_path: str,
        task: Optional[str] = None,
        providers: Optional[List[str]] = None,
        tile_size: int = 1024,
        overlap: int = 128,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load an ONNX model for geospatial inference.

        Args:
            model_path: Path to the ``.onnx`` model file.
            task: Model task.  One of ``"semantic-segmentation"``,
                ``"image-classification"``, ``"object-detection"``, or
                ``"depth-estimation"``.  If *None*, the task is read from the
                sidecar ``<model>.onnx.json`` metadata file.
            providers: ONNX Runtime execution providers in priority order.
                Defaults to ``["CUDAExecutionProvider",
                "CPUExecutionProvider"]``.
            tile_size: Tile size for processing large images.
            overlap: Overlap between adjacent tiles (in pixels).
            metadata: Optional pre-loaded metadata dict.  When *None* the
                constructor looks for ``<model_path>.json``.

        Raises:
            FileNotFoundError: If *model_path* does not exist.
            ImportError: If onnxruntime is not installed.
        """
        _check_onnx_deps()
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.tile_size = tile_size
        self.overlap = overlap

        # Load sidecar metadata
        if metadata is not None:
            self.metadata = metadata
        else:
            meta_path = model_path + ".json"
            if os.path.isfile(meta_path):
                with open(meta_path) as fh:
                    self.metadata = json.load(fh)
            else:
                self.metadata = {}

        # Resolve task
        self.task = task or self.metadata.get("task")

        # Label mapping
        self.id2label: Dict[int, str] = {}
        raw = self.metadata.get("id2label", {})
        if raw:
            self.id2label = {int(k): v for k, v in raw.items()}

        # Create session
        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Inspect inputs / outputs
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # may have str dims
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Determine expected spatial size from metadata or model input shape
        self._model_height = self.metadata.get("input_height")
        self._model_width = self.metadata.get("input_width")
        if self._model_height is None and isinstance(self.input_shape, list):
            if len(self.input_shape) == 4:
                h, w = self.input_shape[2], self.input_shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    self._model_height = h
                    self._model_width = w

        active = self.session.get_providers()
        print(f"ONNX model loaded from {model_path}")
        print(f"Execution providers: {active}")
        if self.task:
            print(f"Task: {self.task}")

    # ------------------------------------------------------------------
    # Image I/O helpers (mirrors AutoGeoImageProcessor)
    # ------------------------------------------------------------------

    @staticmethod
    def load_geotiff(
        source: Union[str, "rasterio.DatasetReader"],
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Load a GeoTIFF file and return data with metadata.

        Args:
            source: Path to GeoTIFF file or open rasterio DatasetReader.
            window: Optional rasterio Window for reading a subset.
            bands: List of band indices to read (1-indexed).

        Returns:
            Tuple of (image array in CHW format, metadata dict).
        """
        should_close = False
        if isinstance(source, str):
            src = rasterio.open(source)
            should_close = True
        else:
            src = source

        try:
            if bands:
                data = src.read(bands, window=window)
            else:
                data = src.read(window=window)

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

    @staticmethod
    def load_image(
        source: Union[str, np.ndarray, "Image.Image"],
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
            try:
                with rasterio.open(source) as src:
                    if src.crs is not None or source.lower().endswith(
                        (".tif", ".tiff")
                    ):
                        return ONNXGeoModel.load_geotiff(source, window, bands)
            except (rasterio.RasterioIOError, rasterio.errors.RasterioIOError):
                pass  # not a rasterio-compatible file; fall through to PIL

            image = Image.open(source).convert("RGB")
            data = np.array(image).transpose(2, 0, 1)
            return data, None

        elif isinstance(source, np.ndarray):
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

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        data: np.ndarray,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ) -> np.ndarray:
        """Prepare a CHW uint‑capable array for the ONNX model.

        The method converts to 3‑channel RGB, normalizes to ``[0, 1]``
        float32, resizes to the model's expected spatial dimensions and
        adds a batch dimension.

        Args:
            data: Image array in CHW format.
            target_height: Target height.  Defaults to model metadata.
            target_width: Target width.  Defaults to model metadata.

        Returns:
            Numpy array of shape ``(1, 3, H, W)`` ready for the ONNX
            session.
        """
        # Lazy import to avoid QGIS opencv conflicts
        import cv2

        # CHW → HWC
        if data.ndim == 3:
            img = data.transpose(1, 2, 0)
        else:
            img = data

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[..., :3]

        # Percentile normalization → uint8
        if img.dtype != np.uint8:
            for i in range(img.shape[-1]):
                band = img[..., i].astype(np.float32)
                p2, p98 = np.percentile(band, [2, 98])
                if p98 > p2:
                    img[..., i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    img[..., i] = 0
            img = (img * 255).astype(np.uint8)

        # Resize to model expected size if needed
        th = target_height or self._model_height
        tw = target_width or self._model_width
        if th and tw and (img.shape[0] != th or img.shape[1] != tw):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Normalize to float32 [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC → NCHW
        tensor = img.transpose(2, 0, 1)[np.newaxis, ...]
        return tensor

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        source: Union[str, np.ndarray, "Image.Image"],
        output_path: Optional[str] = None,
        output_vector_path: Optional[str] = None,
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
        threshold: float = 0.5,
        box_threshold: float = 0.3,
        min_object_area: int = 100,
        simplify_tolerance: float = 1.0,
        batch_size: int = 1,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run inference on a GeoTIFF or image.

        This method follows the same interface as
        :meth:`~geoai.auto.AutoGeoModel.predict`.

        Args:
            source: Input image path, numpy array, or PIL Image.
            output_path: Path to save output GeoTIFF (segmentation / depth).
            output_vector_path: Path to save vectorised output.
            window: Optional rasterio Window for reading a subset.
            bands: Band indices to read (1-indexed).
            threshold: Threshold for binary masks (segmentation).
            box_threshold: Confidence threshold for detections.
            min_object_area: Minimum polygon area in pixels for
                vectorization.
            simplify_tolerance: Tolerance for polygon simplification.
            batch_size: Batch size for tiled processing (reserved for
                future use).
            return_probabilities: Whether to return probability maps.
            **kwargs: Extra keyword arguments (currently unused).

        Returns:
            Dictionary with results (``mask``, ``class``, ``boxes`` etc.)
            depending on the task, plus ``metadata``.

        Example:
            >>> model = ONNXGeoModel("segformer.onnx",
            ...                      task="semantic-segmentation")
            >>> result = model.predict("input.tif", output_path="output.tif")
        """
        # Handle URL sources
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            import requests

            pil_image = Image.open(requests.get(source, stream=True).raw)
            data = np.array(pil_image.convert("RGB")).transpose(2, 0, 1)
            metadata = None
        else:
            data, metadata = self.load_image(source, window, bands)

        # Determine spatial size
        if data.ndim == 3:
            _, height, width = data.shape
        else:
            height, width = data.shape

        # Classification never uses tiled processing
        use_tiled = (
            height > self.tile_size or width > self.tile_size
        ) and self.task not in ("classification", "image-classification")

        if use_tiled:
            result = self._predict_tiled(
                data,
                metadata,
                threshold=threshold,
                return_probabilities=return_probabilities,
            )
        else:
            result = self._predict_single(
                data,
                metadata,
                threshold=threshold,
                return_probabilities=return_probabilities,
            )

        # Save GeoTIFF
        if output_path and metadata:
            out_data = result.get("mask", result.get("output"))
            if out_data is not None:
                self._save_geotiff(out_data, output_path, metadata, nodata=0)
                result["output_path"] = output_path

        # Save vector
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

    # ------------------------------------------------------------------
    # Internal prediction helpers
    # ------------------------------------------------------------------

    def _predict_single(
        self,
        data: np.ndarray,
        metadata: Optional[Dict],
        threshold: float = 0.5,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Run inference on a single (non-tiled) image."""
        # Lazy import to avoid QGIS opencv conflicts
        import cv2

        original_h = data.shape[1] if data.ndim == 3 else data.shape[0]
        original_w = data.shape[2] if data.ndim == 3 else data.shape[1]

        input_tensor = self._prepare_input(data)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        result = self._process_outputs(
            outputs, (original_h, original_w), threshold, return_probabilities
        )
        result["metadata"] = metadata
        return result

    def _predict_tiled(
        self,
        data: np.ndarray,
        metadata: Optional[Dict],
        threshold: float = 0.5,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Run tiled inference for large images."""
        # Lazy import to avoid QGIS opencv conflicts
        import cv2

        if data.ndim == 3:
            _, height, width = data.shape
        else:
            height, width = data.shape

        effective = self.tile_size - 2 * self.overlap
        n_x = max(1, int(np.ceil(width / effective)))
        n_y = max(1, int(np.ceil(height / effective)))
        total = n_x * n_y

        mask_output = np.zeros((height, width), dtype=np.float32)
        count_output = np.zeros((height, width), dtype=np.float32)

        print(f"Processing {total} tiles ({n_x}x{n_y})")

        with tqdm(total=total, desc="Processing tiles") as pbar:
            for iy in range(n_y):
                for ix in range(n_x):
                    x_start = max(0, ix * effective - self.overlap)
                    y_start = max(0, iy * effective - self.overlap)
                    x_end = min(width, (ix + 1) * effective + self.overlap)
                    y_end = min(height, (iy + 1) * effective + self.overlap)

                    if data.ndim == 3:
                        tile = data[:, y_start:y_end, x_start:x_end]
                    else:
                        tile = data[y_start:y_end, x_start:x_end]

                    try:
                        tile_result = self._predict_single(
                            tile, None, threshold, return_probabilities
                        )
                        tile_mask = tile_result.get("mask", tile_result.get("output"))
                        if tile_mask is not None:
                            if tile_mask.ndim > 2:
                                tile_mask = tile_mask.squeeze()
                            if tile_mask.ndim > 2:
                                tile_mask = tile_mask[0]

                            tile_h = y_end - y_start
                            tile_w = x_end - x_start
                            if tile_mask.shape != (tile_h, tile_w):
                                tile_mask = cv2.resize(
                                    tile_mask.astype(np.float32),
                                    (tile_w, tile_h),
                                    interpolation=cv2.INTER_LINEAR,
                                )

                            mask_output[y_start:y_end, x_start:x_end] += tile_mask
                            count_output[y_start:y_end, x_start:x_end] += 1
                    except Exception as e:
                        print(f"Error processing tile ({ix}, {iy}): {e}")

                    pbar.update(1)

        count_output = np.maximum(count_output, 1)
        mask_output = mask_output / count_output

        return {
            "mask": (mask_output > threshold).astype(np.uint8),
            "probabilities": mask_output if return_probabilities else None,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Output processing
    # ------------------------------------------------------------------

    def _process_outputs(
        self,
        outputs: List[np.ndarray],
        original_size: Tuple[int, int],
        threshold: float = 0.5,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Map raw ONNX outputs to a result dict.

        Args:
            outputs: List of numpy arrays returned by
                ``session.run()``.
            original_size: ``(height, width)`` of the input before
                resizing.
            threshold: Binary threshold for segmentation masks.
            return_probabilities: Whether to include probability maps.

        Returns:
            Result dictionary.
        """
        # Lazy import to avoid QGIS opencv conflicts
        import cv2

        result: Dict[str, Any] = {}
        oh, ow = original_size

        if self.task in ("segmentation", "semantic-segmentation"):
            logits = outputs[0]  # (1, C, H, W)
            if logits.ndim == 4:
                # Softmax → argmax
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                mask = probs.argmax(axis=1).squeeze()  # (H, W)

                if mask.shape != (oh, ow):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (ow, oh),
                        interpolation=cv2.INTER_NEAREST,
                    )

                result["mask"] = mask.astype(np.uint8)
                if return_probabilities:
                    result["probabilities"] = probs.squeeze()

        elif self.task in ("classification", "image-classification"):
            logits = outputs[0]  # (1, C)
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp / exp.sum(axis=-1, keepdims=True)
            pred = int(probs.argmax(axis=-1).squeeze())
            result["class"] = pred
            result["probabilities"] = probs.squeeze()
            if self.id2label:
                result["label"] = self.id2label.get(pred, str(pred))

        elif self.task == "object-detection":
            logits = outputs[0]  # (1, N, num_classes)
            pred_boxes = outputs[1] if len(outputs) > 1 else None  # (1, N, 4)
            if pred_boxes is not None:
                # Sigmoid scores
                scores_all = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
                scores = scores_all.max(axis=-1).squeeze()  # (N,)
                labels = scores_all.argmax(axis=-1).squeeze()  # (N,)
                boxes = pred_boxes.squeeze()  # (N, 4)

                keep = scores > threshold
                result["boxes"] = boxes[keep]
                result["scores"] = scores[keep]
                result["labels"] = labels[keep]

        elif self.task == "depth-estimation":
            depth = outputs[0].squeeze()
            if depth.shape != (oh, ow):
                depth = cv2.resize(
                    depth.astype(np.float32),
                    (ow, oh),
                    interpolation=cv2.INTER_LINEAR,
                )
            result["output"] = depth
            result["depth"] = depth

        else:
            # Fallback – expose raw outputs
            result["output"] = outputs[0]

        return result

    # ------------------------------------------------------------------
    # Vectorization
    # ------------------------------------------------------------------

    @staticmethod
    def mask_to_vector(
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
            min_object_area: Minimum polygon area in pixels.
            max_object_area: Maximum polygon area in pixels (optional).
            simplify_tolerance: Tolerance for polygon simplification.

        Returns:
            GeoDataFrame with polygon geometries, or *None* if no valid
            polygons are found.
        """
        if metadata is None or metadata.get("crs") is None:
            print("Warning: No CRS information available for vectorization")
            return None

        if mask.dtype in (np.float32, np.float64):
            mask = (mask > threshold).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)

        transform = metadata.get("transform")
        crs = metadata.get("crs")
        if transform is None:
            print("Warning: No transform available for vectorization")
            return None

        polygons: List = []
        values: List = []

        try:
            for geom, value in shapes(mask, transform=transform):
                if value > 0:
                    poly = shape(geom)
                    pixel_area = poly.area / (transform.a * abs(transform.e))
                    if pixel_area < min_object_area:
                        continue
                    if max_object_area and pixel_area > max_object_area:
                        continue
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

        return gpd.GeoDataFrame(
            {"geometry": polygons, "class": values},
            crs=crs,
        )

    # ------------------------------------------------------------------
    # GeoTIFF / vector save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_geotiff(
        data: np.ndarray,
        output_path: str,
        metadata: Dict,
        dtype: Optional[str] = None,
        compress: str = "lzw",
        nodata: Optional[float] = None,
    ) -> str:
        """Save an array as a GeoTIFF with geospatial metadata.

        Args:
            data: Array to save (2D or 3D in CHW format).
            output_path: Output file path.
            metadata: Metadata dictionary from :meth:`load_geotiff`.
            dtype: Output data type.  If *None*, inferred from *data*.
            compress: Compression method.
            nodata: NoData value.

        Returns:
            Path to the saved file.
        """
        profile = metadata["profile"].copy()
        if dtype is None:
            dtype = str(data.dtype)

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

    @staticmethod
    def save_vector(
        gdf: gpd.GeoDataFrame,
        output_path: str,
        driver: Optional[str] = None,
    ) -> str:
        """Save a GeoDataFrame to file.

        Args:
            gdf: GeoDataFrame to save.
            output_path: Output file path.
            driver: File driver (auto-detected from extension if *None*).

        Returns:
            Path to the saved file.
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

    def __repr__(self) -> str:
        return (
            f"ONNXGeoModel(path={self.model_path!r}, task={self.task!r}, "
            f"providers={self.session.get_providers()!r})"
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def onnx_semantic_segmentation(
    input_path: str,
    output_path: str,
    model_path: str,
    output_vector_path: Optional[str] = None,
    threshold: float = 0.5,
    tile_size: int = 1024,
    overlap: int = 128,
    min_object_area: int = 100,
    simplify_tolerance: float = 1.0,
    providers: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perform semantic segmentation using an ONNX model on a GeoTIFF.

    This is a convenience wrapper around :class:`ONNXGeoModel`.

    Args:
        input_path: Path to input GeoTIFF.
        output_path: Path to save output segmentation GeoTIFF.
        model_path: Path to the ONNX model file.
        output_vector_path: Optional path to save vectorised output.
        threshold: Threshold for binary masks.
        tile_size: Tile size for processing large images.
        overlap: Overlap between tiles.
        min_object_area: Minimum object area for vectorization.
        simplify_tolerance: Tolerance for polygon simplification.
        providers: ONNX Runtime execution providers.
        **kwargs: Additional arguments passed to :meth:`ONNXGeoModel.predict`.

    Returns:
        Dictionary with results.

    Example:
        >>> result = onnx_semantic_segmentation(
        ...     "input.tif",
        ...     "output.tif",
        ...     "segformer.onnx",
        ...     output_vector_path="output.geojson",
        ... )
    """
    model = ONNXGeoModel(
        model_path,
        task="semantic-segmentation",
        providers=providers,
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


def onnx_image_classification(
    input_path: str,
    model_path: str,
    providers: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Classify an image using an ONNX model.

    Args:
        input_path: Path to input image or GeoTIFF.
        model_path: Path to the ONNX model file.
        providers: ONNX Runtime execution providers.
        **kwargs: Additional arguments passed to :meth:`ONNXGeoModel.predict`.

    Returns:
        Dictionary with ``class``, ``label`` (if available), and
        ``probabilities``.

    Example:
        >>> result = onnx_image_classification("image.tif", "classifier.onnx")
        >>> print(result["class"], result["label"])
    """
    model = ONNXGeoModel(
        model_path,
        task="image-classification",
        providers=providers,
    )
    return model.predict(input_path, **kwargs)
