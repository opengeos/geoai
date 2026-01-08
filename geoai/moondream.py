"""Moondream Vision Language Model module for GeoAI.

This module provides an interface for using Moondream vision language models
(moondream2 and moondream3-preview) with geospatial imagery, supporting
GeoTIFF input and georeferenced output.

Supported models:
- moondream2: https://huggingface.co/vikhyatk/moondream2
- moondream3-preview: https://huggingface.co/moondream/moondream3-preview
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from shapely.geometry import Point, box
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from .utils import get_device


hf_logging.set_verbosity_error()  # silence HF load reports


class MoondreamGeo:
    """Moondream Vision Language Model processor with GeoTIFF support.

    This class provides an interface for using Moondream models for
    geospatial image analysis, including captioning, visual querying,
    object detection, and pointing.

    Attributes:
        model: The loaded Moondream model.
        model_name: Name of the model being used.
        device: Torch device for inference.
        model_version: Either "moondream2" or "moondream3".
    """

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        compile_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Moondream processor.

        Args:
            model_name: HuggingFace model name. Options:
                - "vikhyatk/moondream2" (default)
                - "moondream/moondream3-preview"
            revision: Model revision/checkpoint date. For moondream2, recommended
                to use a specific date like "2025-06-21" for reproducibility.
            device: Device for inference ("cuda", "mps", or "cpu").
                If None, automatically selects the best available device.
            compile_model: Whether to compile the model (recommended for
                moondream3-preview for faster inference).
            **kwargs: Additional arguments passed to from_pretrained.

        Raises:
            ImportError: If transformers is not installed.
            RuntimeError: If model loading fails.
        """
        self.model_name = model_name
        self.device = device or get_device()
        self._source_path: Optional[str] = None
        self._metadata: Optional[Dict] = None

        # Determine model version
        if "moondream3" in model_name.lower():
            self.model_version = "moondream3"
        else:
            self.model_version = "moondream2"

        # Load the model
        self.model = self._load_model(revision, compile_model, **kwargs)

    def _load_model(
        self,
        revision: Optional[str],
        compile_model: bool,
        **kwargs: Any,
    ) -> Any:
        """Load the Moondream model.

        Args:
            revision: Model revision/checkpoint.
            compile_model: Whether to compile the model.
            **kwargs: Additional arguments for from_pretrained.

        Returns:
            Loaded model instance.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            from transformers import AutoModelForCausalLM

            # Default kwargs
            load_kwargs = {
                "trust_remote_code": True,
            }

            # Try to use device_map with accelerate, fall back to manual device placement
            use_device_map = False
            try:
                import accelerate  # noqa: F401

                # Build device map
                if isinstance(self.device, str):
                    device_map = {"": self.device}
                else:
                    device_map = {"": str(self.device)}
                load_kwargs["device_map"] = device_map
                use_device_map = True
            except ImportError:
                # accelerate not available, will move model to device manually
                pass

            # Add revision if specified
            if revision:
                load_kwargs["revision"] = revision

            # For moondream3, use bfloat16
            if self.model_version == "moondream3":
                load_kwargs["torch_dtype"] = torch.bfloat16
                # Note: moondream3 uses dtype instead of torch_dtype in some versions
                load_kwargs["dtype"] = torch.bfloat16

            load_kwargs.update(kwargs)

            print(f"Loading {self.model_name}...")

            # Try to load with potential transformers 5.0 compatibility fix
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                )
            except AttributeError as attr_err:
                # Handle transformers 5.0 compatibility issue with custom models
                if "all_tied_weights_keys" in str(attr_err):
                    # print(
                    #     "Note: Detected transformers 5.0+ compatibility issue. "
                    #     "Attempting workaround..."
                    # )
                    # Try patching the model class
                    model = self._load_with_patch(load_kwargs)
                else:
                    raise

            # Move model to device if accelerate wasn't used
            if not use_device_map:
                device = (
                    self.device
                    if isinstance(self.device, torch.device)
                    else torch.device(self.device)
                )
                model = model.to(device)
                # print(f"Model moved to {device}")

            # Set model to evaluation mode
            model.eval()

            # Compile model if requested (recommended for moondream3)
            if compile_model and hasattr(model, "compile"):
                print("Compiling model for faster inference...")
                model.compile()

            print(f"Using device: {self.device}")
            return model

        except Exception as e:
            # Provide helpful error message
            error_msg = str(e)
            if "all_tied_weights_keys" in error_msg:
                error_msg = (
                    f"Failed to load Moondream model due to transformers version "
                    f"incompatibility. The model's custom code may not be compatible "
                    f"with your current transformers version. Try: "
                    f"1) Wait for an updated model revision, or "
                    f"2) Use a compatible transformers version. "
                    f"Original error: {e}"
                )
            raise RuntimeError(f"Failed to load Moondream model: {error_msg}") from e

    def _load_with_patch(self, load_kwargs: Dict) -> Any:
        """Load model with compatibility patch for transformers 5.0+.

        Args:
            load_kwargs: Keyword arguments for from_pretrained.

        Returns:
            Loaded model instance.
        """
        from transformers import AutoModelForCausalLM, PreTrainedModel

        # Patch the PreTrainedModel class to add missing attribute
        original_getattr = PreTrainedModel.__getattr__

        def patched_getattr(self, name):
            if name == "all_tied_weights_keys":
                # Return empty dict to satisfy the check
                if not hasattr(self, "_all_tied_weights_keys"):
                    self._all_tied_weights_keys = {}
                return self._all_tied_weights_keys
            return original_getattr(self, name)

        # Apply patch temporarily
        PreTrainedModel.__getattr__ = patched_getattr

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            return model
        finally:
            # Restore original
            PreTrainedModel.__getattr__ = original_getattr

    def load_geotiff(
        self,
        source: str,
        bands: Optional[List[int]] = None,
    ) -> Tuple[Image.Image, Dict]:
        """Load a GeoTIFF file and return a PIL Image with metadata.

        Args:
            source: Path to GeoTIFF file.
            bands: List of band indices to read (1-indexed). If None, reads
                first 3 bands for RGB or first band for grayscale.

        Returns:
            Tuple of (PIL Image, metadata dict).

        Raises:
            FileNotFoundError: If source file doesn't exist.
            RuntimeError: If loading fails.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        try:
            with rasterio.open(source) as src:
                # Store metadata
                metadata = {
                    "profile": src.profile.copy(),
                    "crs": src.crs,
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "width": src.width,
                    "height": src.height,
                }

                # Read bands
                if bands is None:
                    if src.count >= 3:
                        bands = [1, 2, 3]  # RGB
                    else:
                        bands = [1]  # Grayscale

                data = src.read(bands)

                # Convert to RGB image
                if len(bands) == 1:
                    # Grayscale to RGB
                    img_array = np.repeat(data[0:1], 3, axis=0)
                elif len(bands) >= 3:
                    img_array = data[:3]
                else:
                    # Pad to 3 channels
                    img_array = np.zeros((3, data.shape[1], data.shape[2]))
                    img_array[: data.shape[0]] = data

                # Normalize to 0-255 range
                img_array = self._normalize_image(img_array)

                # Convert to PIL Image (HWC format)
                img_array = np.transpose(img_array, (1, 2, 0))
                image = Image.fromarray(img_array.astype(np.uint8))

                self._source_path = source
                self._metadata = metadata

                return image, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load GeoTIFF: {e}") from e

    def load_image(
        self,
        source: Union[str, Image.Image, np.ndarray],
        bands: Optional[List[int]] = None,
    ) -> Tuple[Image.Image, Optional[Dict]]:
        """Load an image from various sources.

        Args:
            source: Image source - can be a file path (GeoTIFF, PNG, JPG),
                PIL Image, or numpy array.
            bands: Band indices for GeoTIFF (1-indexed).

        Returns:
            Tuple of (PIL Image, metadata dict or None).
        """
        if isinstance(source, Image.Image):
            self._source_path = None
            self._metadata = None
            return source, None

        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                # Grayscale
                source = np.stack([source] * 3, axis=-1)
            elif source.ndim == 3 and source.shape[0] <= 4:
                # CHW format
                source = np.transpose(source[:3], (1, 2, 0))

            source = self._normalize_image(source)
            image = Image.fromarray(source.astype(np.uint8))
            self._source_path = None
            self._metadata = None
            return image, None

        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                # URL - download and load
                from .utils import download_file

                source = download_file(source)

            # Check if it's a GeoTIFF
            try:
                with rasterio.open(source) as src:
                    if src.crs is not None or source.lower().endswith(
                        (".tif", ".tiff")
                    ):
                        return self.load_geotiff(source, bands)
            except rasterio.RasterioIOError:
                pass

            # Regular image
            image = Image.open(source).convert("RGB")
            self._source_path = source
            self._metadata = {
                "width": image.width,
                "height": image.height,
                "crs": None,
                "transform": None,
                "bounds": None,
            }
            return image, self._metadata

    def _normalize_image(self, data: np.ndarray) -> np.ndarray:
        """Normalize image data to 0-255 range using percentile stretching.

        Args:
            data: Input array (can be CHW or HWC format).

        Returns:
            Normalized array in uint8 range.
        """
        if data.dtype == np.uint8:
            return data

        # Determine if CHW or HWC
        if data.ndim == 3 and data.shape[0] <= 4:
            # CHW format - normalize each channel
            normalized = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[0]):
                band = data[i].astype(np.float32)
                p2, p98 = np.percentile(band, [2, 98])
                if p98 > p2:
                    normalized[i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    normalized[i] = np.clip(band, 0, 255)
        else:
            # HWC format or 2D
            data = data.astype(np.float32)
            p2, p98 = np.percentile(data, [2, 98])
            if p98 > p2:
                normalized = np.clip((data - p2) / (p98 - p2) * 255, 0, 255)
            else:
                normalized = np.clip(data, 0, 255)

        return normalized.astype(np.uint8)

    def encode_image(
        self,
        source: Union[str, Image.Image, np.ndarray],
        bands: Optional[List[int]] = None,
    ) -> Any:
        """Pre-encode an image for efficient multiple inferences.

        Use this when you plan to run multiple queries on the same image.

        Args:
            source: Image source.
            bands: Band indices for GeoTIFF.

        Returns:
            Encoded image that can be passed to query, caption, etc.
        """
        image, _ = self.load_image(source, bands)

        if hasattr(self.model, "encode_image"):
            return self.model.encode_image(image)
        return image

    def caption(
        self,
        source: Union[str, Image.Image, np.ndarray, Any],
        length: str = "normal",
        stream: bool = False,
        bands: Optional[List[int]] = None,
        settings: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a caption for an image.

        Args:
            source: Image source or pre-encoded image.
            length: Caption length - "short", "normal", or "long".
            stream: Whether to stream the output.
            bands: Band indices for GeoTIFF.
            settings: Additional settings (temperature, top_p, max_tokens).
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "caption" key containing the generated caption.
        """
        # Load image if not pre-encoded
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, _ = self.load_image(source, bands)
        else:
            image = source  # Pre-encoded

        call_kwargs = {"length": length, "stream": stream}
        if settings:
            call_kwargs["settings"] = settings
        call_kwargs.update(kwargs)

        return self.model.caption(image, **call_kwargs)

    def query(
        self,
        question: str,
        source: Optional[Union[str, Image.Image, np.ndarray, Any]] = None,
        reasoning: Optional[bool] = None,
        stream: bool = False,
        bands: Optional[List[int]] = None,
        settings: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Ask a question about an image or text-only query.

        Args:
            question: The question to ask.
            source: Image source or pre-encoded image. If None, performs
                text-only query (moondream3 only).
            reasoning: Enable reasoning mode for more complex tasks
                (moondream3 only, default True).
            stream: Whether to stream the output.
            bands: Band indices for GeoTIFF.
            settings: Additional settings (temperature, top_p, max_tokens).
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "answer" key containing the response.
        """
        call_kwargs = {"question": question, "stream": stream}

        if source is not None:
            if isinstance(source, (str, Image.Image, np.ndarray)):
                image, _ = self.load_image(source, bands)
            else:
                image = source
            call_kwargs["image"] = image

        if reasoning is not None and self.model_version == "moondream3":
            call_kwargs["reasoning"] = reasoning

        if settings:
            call_kwargs["settings"] = settings
        call_kwargs.update(kwargs)

        return self.model.query(**call_kwargs)

    def detect(
        self,
        source: Union[str, Image.Image, np.ndarray, Any],
        object_type: str,
        bands: Optional[List[int]] = None,
        output_path: Optional[str] = None,
        settings: Optional[Dict] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Detect objects of a specific type in an image.

        Args:
            source: Image source or pre-encoded image.
            object_type: Type of object to detect (e.g., "car", "building").
            bands: Band indices for GeoTIFF.
            output_path: Path to save results as GeoJSON/Shapefile/GeoPackage.
            settings: Additional settings (max_objects, etc.).
            stream: Whether to stream the output (moondream2 only).
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "objects" key containing list of bounding boxes
            with normalized coordinates (x_min, y_min, x_max, y_max).
            If georeferenced, also includes "gdf" (GeoDataFrame) and
            "crs", "bounds" keys.
        """
        # Load image
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, metadata = self.load_image(source, bands)
        else:
            image = source
            metadata = self._metadata

        call_kwargs = {}
        if settings:
            call_kwargs["settings"] = settings
        if self.model_version == "moondream2" and stream:
            call_kwargs["stream"] = stream
        call_kwargs.update(kwargs)

        result = self.model.detect(image, object_type, **call_kwargs)

        # Convert to georeferenced if possible
        if metadata and metadata.get("crs") and metadata.get("transform"):
            result = self._georef_detections(result, metadata)

            if output_path:
                self._save_vector(result["gdf"], output_path)

        return result

    def point(
        self,
        source: Union[str, Image.Image, np.ndarray, Any],
        object_description: str,
        bands: Optional[List[int]] = None,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Find points (x, y coordinates) for objects in an image.

        Args:
            source: Image source or pre-encoded image.
            object_description: Description of objects to find.
            bands: Band indices for GeoTIFF.
            output_path: Path to save results as GeoJSON/Shapefile/GeoPackage.
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "points" key containing list of points
            with normalized coordinates (x, y in 0-1 range).
            If georeferenced, also includes "gdf" (GeoDataFrame).
        """
        # Load image
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, metadata = self.load_image(source, bands)
        else:
            image = source
            metadata = self._metadata

        result = self.model.point(image, object_description, **kwargs)

        # Convert to georeferenced if possible
        if metadata and metadata.get("crs") and metadata.get("transform"):
            result = self._georef_points(result, metadata)

            if output_path:
                self._save_vector(result["gdf"], output_path)

        return result

    def _georef_detections(
        self,
        result: Dict[str, Any],
        metadata: Dict,
    ) -> Dict[str, Any]:
        """Convert detection results to georeferenced format.

        Args:
            result: Detection result from model.
            metadata: Image metadata with CRS and transform.

        Returns:
            Updated result dictionary with GeoDataFrame.
        """
        objects = result.get("objects", [])
        if not objects:
            result["gdf"] = gpd.GeoDataFrame(
                columns=["geometry", "x_min", "y_min", "x_max", "y_max"],
                crs=metadata["crs"],
            )
            result["crs"] = metadata["crs"]
            result["bounds"] = metadata["bounds"]
            return result

        width = metadata["width"]
        height = metadata["height"]
        transform = metadata["transform"]

        geometries = []
        records = []

        for obj in objects:
            # Convert normalized coords to pixel coords
            px_x_min = obj["x_min"] * width
            px_y_min = obj["y_min"] * height
            px_x_max = obj["x_max"] * width
            px_y_max = obj["y_max"] * height

            # Convert pixel coords to geographic coords
            geo_x_min, geo_y_min = transform * (px_x_min, px_y_max)
            geo_x_max, geo_y_max = transform * (px_x_max, px_y_min)

            # Create polygon
            geom = box(geo_x_min, geo_y_min, geo_x_max, geo_y_max)
            geometries.append(geom)

            records.append(
                {
                    "x_min": obj["x_min"],
                    "y_min": obj["y_min"],
                    "x_max": obj["x_max"],
                    "y_max": obj["y_max"],
                    "px_x_min": int(px_x_min),
                    "px_y_min": int(px_y_min),
                    "px_x_max": int(px_x_max),
                    "px_y_max": int(px_y_max),
                }
            )

        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=metadata["crs"])

        result["gdf"] = gdf
        result["crs"] = metadata["crs"]
        result["bounds"] = metadata["bounds"]

        return result

    def _georef_points(
        self,
        result: Dict[str, Any],
        metadata: Dict,
    ) -> Dict[str, Any]:
        """Convert point results to georeferenced format.

        Args:
            result: Point result from model.
            metadata: Image metadata with CRS and transform.

        Returns:
            Updated result dictionary with GeoDataFrame.
        """
        points = result.get("points", [])
        if not points:
            result["gdf"] = gpd.GeoDataFrame(
                columns=["geometry", "x", "y"],
                crs=metadata["crs"],
            )
            result["crs"] = metadata["crs"]
            result["bounds"] = metadata["bounds"]
            return result

        width = metadata["width"]
        height = metadata["height"]
        transform = metadata["transform"]

        geometries = []
        records = []

        for pt in points:
            # Convert normalized coords to pixel coords
            px_x = pt["x"] * width
            px_y = pt["y"] * height

            # Convert pixel coords to geographic coords
            geo_x, geo_y = transform * (px_x, px_y)

            # Create point
            geom = Point(geo_x, geo_y)
            geometries.append(geom)

            records.append(
                {
                    "x": pt["x"],
                    "y": pt["y"],
                    "px_x": int(px_x),
                    "px_y": int(px_y),
                }
            )

        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=metadata["crs"])

        result["gdf"] = gdf
        result["crs"] = metadata["crs"]
        result["bounds"] = metadata["bounds"]

        return result

    def _save_vector(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str,
    ) -> None:
        """Save GeoDataFrame to vector file.

        Args:
            gdf: GeoDataFrame to save.
            output_path: Output file path. Extension determines format:
                .geojson -> GeoJSON
                .shp -> Shapefile
                .gpkg -> GeoPackage
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".geojson":
            gdf.to_file(output_path, driver="GeoJSON")
        elif ext == ".shp":
            gdf.to_file(output_path, driver="ESRI Shapefile")
        elif ext == ".gpkg":
            gdf.to_file(output_path, driver="GPKG")
        else:
            gdf.to_file(output_path)

        print(f"Saved {len(gdf)} features to {output_path}")

    def create_detection_mask(
        self,
        source: Union[str, Image.Image, np.ndarray],
        object_type: str,
        output_path: Optional[str] = None,
        bands: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Create a binary mask from object detections.

        Args:
            source: Image source.
            object_type: Type of object to detect.
            output_path: Path to save mask as GeoTIFF.
            bands: Band indices for GeoTIFF.
            **kwargs: Additional arguments for detect().

        Returns:
            Tuple of (mask array, metadata dict).
        """
        # Load image to get dimensions
        image, metadata = self.load_image(source, bands)
        width, height = image.size

        # Detect objects
        result = self.detect(source, object_type, bands=bands, **kwargs)
        objects = result.get("objects", [])

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)

        for obj in objects:
            x_min = int(obj["x_min"] * width)
            y_min = int(obj["y_min"] * height)
            x_max = int(obj["x_max"] * width)
            y_max = int(obj["y_max"] * height)

            mask[y_min:y_max, x_min:x_max] = 255

        # Save as GeoTIFF if requested
        if output_path and metadata and metadata.get("crs"):
            self._save_mask_geotiff(mask, output_path, metadata)
        elif output_path:
            # Save as regular image
            Image.fromarray(mask).save(output_path)

        return mask, metadata

    def _save_mask_geotiff(
        self,
        mask: np.ndarray,
        output_path: str,
        metadata: Dict,
    ) -> None:
        """Save mask as GeoTIFF with georeferencing.

        Args:
            mask: 2D mask array.
            output_path: Output file path.
            metadata: Image metadata with profile.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        profile = metadata["profile"].copy()
        profile.update(
            {
                "dtype": "uint8",
                "count": 1,
                "height": mask.shape[0],
                "width": mask.shape[1],
                "compress": "lzw",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

        print(f"Saved mask to {output_path}")

    def show_gui(
        self,
        basemap: str = "SATELLITE",
        out_dir: Optional[str] = None,
        opacity: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        """Display an interactive GUI for using Moondream with leafmap.

        This method creates an interactive map interface for using Moondream
        vision language model capabilities including:
        - Image captioning (short, normal, long)
        - Visual question answering (query)
        - Object detection with bounding boxes
        - Point detection for locating objects

        Args:
            basemap: The basemap to use. Defaults to "SATELLITE".
            out_dir: The output directory for saving results.
                Defaults to None (uses temp directory).
            opacity: The opacity of overlay layers. Defaults to 0.5.
            **kwargs: Additional keyword arguments passed to leafmap.Map().

        Returns:
            leafmap.Map: The interactive map with the Moondream GUI.

        Example:
            >>> moondream = MoondreamGeo()
            >>> moondream.load_image("image.tif")
            >>> m = moondream.show_gui()
            >>> m
        """
        from .map_widgets import moondream_gui

        return moondream_gui(
            self,
            basemap=basemap,
            out_dir=out_dir,
            opacity=opacity,
            **kwargs,
        )

    def get_last_result(self) -> Dict[str, Any]:

        if hasattr(self, "last_result") and "gdf" in self.last_result:
            return self.last_result["gdf"]
        else:
            return None

    def _create_sliding_windows(
        self,
        image_width: int,
        image_height: int,
        window_size: int = 512,
        overlap: int = 64,
    ) -> List[Tuple[int, int, int, int]]:
        """Create sliding window coordinates for tiled processing.

        Args:
            image_width: Width of the full image.
            image_height: Height of the full image.
            window_size: Size of each window/tile.
            overlap: Overlap between adjacent windows.

        Returns:
            List of tuples (x_start, y_start, x_end, y_end) for each window.
        """
        windows = []
        stride = window_size - overlap

        for y in range(0, image_height, stride):
            for x in range(0, image_width, stride):
                x_start = x
                y_start = y
                x_end = min(x + window_size, image_width)
                y_end = min(y + window_size, image_height)

                # Only add windows that have sufficient size
                if (x_end - x_start) >= window_size // 2 and (
                    y_end - y_start
                ) >= window_size // 2:
                    windows.append((x_start, y_start, x_end, y_end))

        return windows

    def _apply_nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of detection dictionaries with bounding boxes.
            iou_threshold: IoU threshold for considering boxes as overlapping.

        Returns:
            Filtered list of detections after NMS.
        """
        if not detections:
            return []

        # Sort by confidence/score if available
        if "score" in detections[0]:
            detections = sorted(
                detections, key=lambda x: x.get("score", 1.0), reverse=True
            )

        # Convert to arrays for efficient computation
        boxes = np.array(
            [[d["x_min"], d["y_min"], d["x_max"], d["y_max"]] for d in detections]
        )

        # Calculate areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by y2 coordinate (bottom of box)
        order = y2.argsort()

        keep = []
        while order.size > 0:
            i = order[-1]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[:-1]] - intersection)

            # Keep only boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds]

        return [detections[i] for i in keep]

    def detect_sliding_window(
        self,
        source: Union[str, Image.Image, np.ndarray],
        object_type: str,
        window_size: int = 512,
        overlap: int = 64,
        iou_threshold: float = 0.5,
        bands: Optional[List[int]] = None,
        output_path: Optional[str] = None,
        settings: Optional[Dict] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Detect objects using sliding window for large images.

        This method processes large images by dividing them into overlapping
        windows/tiles, running detection on each tile, and merging results
        using Non-Maximum Suppression (NMS) to handle overlapping detections.

        Args:
            source: Image source or pre-encoded image.
            object_type: Type of object to detect (e.g., "car", "building").
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            iou_threshold: IoU threshold for NMS to merge overlapping detections.
            bands: Band indices for GeoTIFF.
            output_path: Path to save results as GeoJSON/Shapefile/GeoPackage.
            settings: Additional settings for the model.
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "objects" key containing list of bounding boxes
            with normalized coordinates. If georeferenced, also includes
            "gdf" (GeoDataFrame).
        """
        # Load image
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, metadata = self.load_image(source, bands)
        else:
            image = source
            metadata = self._metadata

        width, height = image.size

        # If image is smaller than window size, use regular detection
        if width <= window_size and height <= window_size:
            return self.detect(
                image,
                object_type,
                bands=bands,
                output_path=output_path,
                settings=settings,
                **kwargs,
            )

        # Create sliding windows
        windows = self._create_sliding_windows(width, height, window_size, overlap)

        all_detections = []

        # Progress bar setup
        iterator = (
            tqdm(windows, desc=f"Detecting {object_type}") if show_progress else windows
        )

        # Process each window
        for x_start, y_start, x_end, y_end in iterator:
            # Crop window from image
            window_img = image.crop((x_start, y_start, x_end, y_end))

            # Detect in window
            call_kwargs = {}
            if settings:
                call_kwargs["settings"] = settings
            call_kwargs.update(kwargs)

            try:
                result = self.model.detect(window_img, object_type, **call_kwargs)

                # Adjust coordinates to full image space
                window_width = x_end - x_start
                window_height = y_end - y_start

                for obj in result.get("objects", []):
                    # Convert from window-relative normalized coords to full image normalized coords
                    full_x_min = (x_start + obj["x_min"] * window_width) / width
                    full_y_min = (y_start + obj["y_min"] * window_height) / height
                    full_x_max = (x_start + obj["x_max"] * window_width) / width
                    full_y_max = (y_start + obj["y_max"] * window_height) / height

                    detection = {
                        "x_min": full_x_min,
                        "y_min": full_y_min,
                        "x_max": full_x_max,
                        "y_max": full_y_max,
                    }

                    # Preserve additional fields if present
                    for key in obj:
                        if key not in ["x_min", "y_min", "x_max", "y_max"]:
                            detection[key] = obj[key]

                    all_detections.append(detection)

            except Exception as e:
                if show_progress:
                    print(
                        f"Warning: Failed to process window ({x_start},{y_start})-({x_end},{y_end}): {e}"
                    )

        # Apply NMS to merge overlapping detections
        merged_detections = self._apply_nms(all_detections, iou_threshold)

        result = {"objects": merged_detections}

        # Convert to georeferenced if possible
        if metadata and metadata.get("crs") and metadata.get("transform"):
            result = self._georef_detections(result, metadata)

            if output_path:
                self._save_vector(result["gdf"], output_path)

        return result

    def point_sliding_window(
        self,
        source: Union[str, Image.Image, np.ndarray],
        object_description: str,
        window_size: int = 512,
        overlap: int = 64,
        bands: Optional[List[int]] = None,
        output_path: Optional[str] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Find points using sliding window for large images.

        This method processes large images by dividing them into overlapping
        windows/tiles and finding points in each tile.

        Args:
            source: Image source or pre-encoded image.
            object_description: Description of objects to find.
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            bands: Band indices for GeoTIFF.
            output_path: Path to save results as GeoJSON/Shapefile/GeoPackage.
            show_progress: Whether to show progress bar.
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "points" key containing list of points
            with normalized coordinates. If georeferenced, also includes
            "gdf" (GeoDataFrame).
        """
        # Load image
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, metadata = self.load_image(source, bands)
        else:
            image = source
            metadata = self._metadata

        width, height = image.size

        # If image is smaller than window size, use regular point detection
        if width <= window_size and height <= window_size:
            return self.point(
                image,
                object_description,
                bands=bands,
                output_path=output_path,
                **kwargs,
            )

        # Create sliding windows
        windows = self._create_sliding_windows(width, height, window_size, overlap)

        all_points = []

        # Progress bar setup
        iterator = (
            tqdm(windows, desc=f"Finding {object_description}")
            if show_progress
            else windows
        )

        # Process each window
        for x_start, y_start, x_end, y_end in iterator:
            # Crop window from image
            window_img = image.crop((x_start, y_start, x_end, y_end))

            # Find points in window
            try:
                result = self.model.point(window_img, object_description, **kwargs)

                # Adjust coordinates to full image space
                window_width = x_end - x_start
                window_height = y_end - y_start

                for pt in result.get("points", []):
                    # Convert from window-relative normalized coords to full image normalized coords
                    full_x = (x_start + pt["x"] * window_width) / width
                    full_y = (y_start + pt["y"] * window_height) / height

                    point = {"x": full_x, "y": full_y}

                    # Preserve additional fields if present
                    for key in pt:
                        if key not in ["x", "y"]:
                            point[key] = pt[key]

                    all_points.append(point)

            except Exception as e:
                if show_progress:
                    print(
                        f"Warning: Failed to process window ({x_start},{y_start})-({x_end},{y_end}): {e}"
                    )

        result = {"points": all_points}

        # Convert to georeferenced if possible
        if metadata and metadata.get("crs") and metadata.get("transform"):
            result = self._georef_points(result, metadata)

            if output_path:
                self._save_vector(result["gdf"], output_path)

        return result

    def query_sliding_window(
        self,
        question: str,
        source: Union[str, Image.Image, np.ndarray],
        window_size: int = 512,
        overlap: int = 64,
        reasoning: Optional[bool] = None,
        bands: Optional[List[int]] = None,
        settings: Optional[Dict] = None,
        show_progress: bool = True,
        combine_strategy: str = "concatenate",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Query image using sliding window for large images.

        This method processes large images by dividing them into overlapping
        windows/tiles, querying each tile, and combining the responses.

        Args:
            question: The question to ask about each window.
            source: Image source or pre-encoded image.
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            reasoning: Enable reasoning mode (moondream3 only).
            bands: Band indices for GeoTIFF.
            settings: Additional settings for the model.
            show_progress: Whether to show progress bar.
            combine_strategy: How to combine answers from different windows.
                Options: "concatenate", "summarize". Default "concatenate".
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "answer" key containing the combined response,
            and "tile_answers" with individual tile responses.
        """
        # Load image
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, _ = self.load_image(source, bands)
        else:
            image = source

        width, height = image.size

        # If image is smaller than window size, use regular query
        if width <= window_size and height <= window_size:
            return self.query(
                question,
                image,
                reasoning=reasoning,
                bands=bands,
                settings=settings,
                **kwargs,
            )

        # Create sliding windows
        windows = self._create_sliding_windows(width, height, window_size, overlap)

        tile_answers = []

        # Progress bar setup
        iterator = tqdm(windows, desc="Querying tiles") if show_progress else windows

        # Process each window
        for idx, (x_start, y_start, x_end, y_end) in enumerate(iterator):
            # Crop window from image
            window_img = image.crop((x_start, y_start, x_end, y_end))

            # Query window
            call_kwargs = {"question": question, "image": window_img}
            if reasoning is not None and self.model_version == "moondream3":
                call_kwargs["reasoning"] = reasoning
            if settings:
                call_kwargs["settings"] = settings
            call_kwargs.update(kwargs)

            try:
                result = self.model.query(**call_kwargs)
                tile_answers.append(
                    {
                        "tile_id": idx,
                        "bounds": (x_start, y_start, x_end, y_end),
                        "answer": result.get("answer", ""),
                    }
                )
            except Exception as e:
                if show_progress:
                    print(
                        f"Warning: Failed to process window ({x_start},{y_start})-({x_end},{y_end}): {e}"
                    )

        # Combine answers
        if combine_strategy == "concatenate":
            combined_answer = "\n\n".join(
                [
                    f"Tile {ta['tile_id']} (region {ta['bounds']}): {ta['answer']}"
                    for ta in tile_answers
                ]
            )
        elif combine_strategy == "summarize":
            # Use the model to summarize the tile answers
            summary_prompt = (
                f"Based on these regional observations about '{question}', "
                f"provide a comprehensive summary:\n\n"
            )
            for ta in tile_answers:
                summary_prompt += f"Region {ta['tile_id']}: {ta['answer']}\n"

            try:
                summary_result = self.model.query(question=summary_prompt)
                combined_answer = summary_result.get("answer", "")
            except:
                # Fall back to concatenation if summarization fails
                combined_answer = " ".join([ta["answer"] for ta in tile_answers])
        else:
            combined_answer = " ".join([ta["answer"] for ta in tile_answers])

        return {"answer": combined_answer, "tile_answers": tile_answers}

    def caption_sliding_window(
        self,
        source: Union[str, Image.Image, np.ndarray],
        window_size: int = 512,
        overlap: int = 64,
        length: str = "normal",
        bands: Optional[List[int]] = None,
        settings: Optional[Dict] = None,
        show_progress: bool = True,
        combine_strategy: str = "concatenate",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate caption using sliding window for large images.

        This method processes large images by dividing them into overlapping
        windows/tiles, generating captions for each tile, and combining them.

        Args:
            source: Image source or pre-encoded image.
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            length: Caption length - "short", "normal", or "long".
            bands: Band indices for GeoTIFF.
            settings: Additional settings for the model.
            show_progress: Whether to show progress bar.
            combine_strategy: How to combine captions from different windows.
                Options: "concatenate", "summarize". Default "concatenate".
            **kwargs: Additional arguments for the model.

        Returns:
            Dictionary with "caption" key containing the combined caption,
            and "tile_captions" with individual tile captions.
        """
        # Load image
        if isinstance(source, (str, Image.Image, np.ndarray)):
            image, _ = self.load_image(source, bands)
        else:
            image = source

        width, height = image.size

        # If image is smaller than window size, use regular caption
        if width <= window_size and height <= window_size:
            return self.caption(
                image, length=length, bands=bands, settings=settings, **kwargs
            )

        # Create sliding windows
        windows = self._create_sliding_windows(width, height, window_size, overlap)

        tile_captions = []

        # Progress bar setup
        iterator = (
            tqdm(windows, desc="Generating captions") if show_progress else windows
        )

        # Process each window
        for idx, (x_start, y_start, x_end, y_end) in enumerate(iterator):
            # Crop window from image
            window_img = image.crop((x_start, y_start, x_end, y_end))

            # Caption window
            call_kwargs = {"length": length}
            if settings:
                call_kwargs["settings"] = settings
            call_kwargs.update(kwargs)

            try:
                result = self.model.caption(window_img, **call_kwargs)
                tile_captions.append(
                    {
                        "tile_id": idx,
                        "bounds": (x_start, y_start, x_end, y_end),
                        "caption": result.get("caption", ""),
                    }
                )
            except Exception as e:
                if show_progress:
                    print(
                        f"Warning: Failed to process window ({x_start},{y_start})-({x_end},{y_end}): {e}"
                    )

        # Combine captions
        if combine_strategy == "concatenate":
            combined_caption = " ".join([tc["caption"] for tc in tile_captions])
        elif combine_strategy == "summarize":
            # Use the model to create a cohesive summary caption
            summary_prompt = (
                "Based on these descriptions of different regions of an image, "
                "create a single comprehensive caption for the entire image:\n\n"
            )
            for tc in tile_captions:
                summary_prompt += f"Region {tc['tile_id']}: {tc['caption']}\n"

            try:
                summary_result = self.model.query(question=summary_prompt)
                combined_caption = summary_result.get("answer", "")
            except:
                # Fall back to concatenation if summarization fails
                combined_caption = " ".join([tc["caption"] for tc in tile_captions])
        else:
            combined_caption = " ".join([tc["caption"] for tc in tile_captions])

        return {"caption": combined_caption, "tile_captions": tile_captions}


def moondream_caption(
    source: Union[str, Image.Image, np.ndarray],
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    length: str = "normal",
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Convenience function to generate a caption for an image.

    Args:
        source: Image source (file path, PIL Image, or numpy array).
        model_name: Moondream model name.
        revision: Model revision.
        length: Caption length ("short", "normal", "long").
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        **kwargs: Additional arguments.

    Returns:
        Generated caption string.
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    result = processor.caption(source, length=length, bands=bands, **kwargs)
    return result["caption"]


def moondream_query(
    question: str,
    source: Optional[Union[str, Image.Image, np.ndarray]] = None,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    reasoning: Optional[bool] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Convenience function to ask a question about an image.

    Args:
        question: Question to ask.
        source: Image source (optional for text-only queries).
        model_name: Moondream model name.
        revision: Model revision.
        reasoning: Enable reasoning mode (moondream3 only).
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        **kwargs: Additional arguments.

    Returns:
        Answer string.
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    result = processor.query(
        question, source=source, reasoning=reasoning, bands=bands, **kwargs
    )
    return result["answer"]


def moondream_detect(
    source: Union[str, Image.Image, np.ndarray],
    object_type: str,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to detect objects in an image.

    Args:
        source: Image source.
        object_type: Type of object to detect.
        model_name: Moondream model name.
        revision: Model revision.
        output_path: Path to save results as vector file.
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        **kwargs: Additional arguments.

    Returns:
        Detection results dictionary with "objects" and optionally "gdf".
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    return processor.detect(
        source, object_type, output_path=output_path, bands=bands, **kwargs
    )


def moondream_point(
    source: Union[str, Image.Image, np.ndarray],
    object_description: str,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to find points for objects in an image.

    Args:
        source: Image source.
        object_description: Description of objects to find.
        model_name: Moondream model name.
        revision: Model revision.
        output_path: Path to save results as vector file.
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        **kwargs: Additional arguments.

    Returns:
        Point results dictionary with "points" and optionally "gdf".
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    return processor.point(
        source, object_description, output_path=output_path, bands=bands, **kwargs
    )


def moondream_detect_sliding_window(
    source: Union[str, Image.Image, np.ndarray],
    object_type: str,
    window_size: int = 512,
    overlap: int = 64,
    iou_threshold: float = 0.5,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to detect objects using sliding window.

    This function is designed for large images where the standard detection
    may not work well. It divides the image into overlapping windows and
    merges detections using NMS.

    Args:
        source: Image source.
        object_type: Type of object to detect.
        window_size: Size of each processing window. Default 512.
        overlap: Overlap between windows. Default 64.
        iou_threshold: IoU threshold for NMS. Default 0.5.
        model_name: Moondream model name.
        revision: Model revision.
        output_path: Path to save results as vector file.
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        show_progress: Whether to show progress bar.
        **kwargs: Additional arguments.

    Returns:
        Detection results dictionary with "objects" and optionally "gdf".
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    return processor.detect_sliding_window(
        source,
        object_type,
        window_size=window_size,
        overlap=overlap,
        iou_threshold=iou_threshold,
        output_path=output_path,
        bands=bands,
        show_progress=show_progress,
        **kwargs,
    )


def moondream_point_sliding_window(
    source: Union[str, Image.Image, np.ndarray],
    object_description: str,
    window_size: int = 512,
    overlap: int = 64,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to find points using sliding window.

    This function is designed for large images. It divides the image
    into overlapping windows and aggregates all detected points.

    Args:
        source: Image source.
        object_description: Description of objects to find.
        window_size: Size of each processing window. Default 512.
        overlap: Overlap between windows. Default 64.
        model_name: Moondream model name.
        revision: Model revision.
        output_path: Path to save results as vector file.
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        show_progress: Whether to show progress bar.
        **kwargs: Additional arguments.

    Returns:
        Point results dictionary with "points" and optionally "gdf".
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    return processor.point_sliding_window(
        source,
        object_description,
        window_size=window_size,
        overlap=overlap,
        output_path=output_path,
        bands=bands,
        show_progress=show_progress,
        **kwargs,
    )


def moondream_query_sliding_window(
    question: str,
    source: Union[str, Image.Image, np.ndarray],
    window_size: int = 512,
    overlap: int = 64,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    reasoning: Optional[bool] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
    combine_strategy: str = "concatenate",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to query large images using sliding window.

    This function divides the image into overlapping windows, queries each,
    and combines the answers.

    Args:
        question: Question to ask about the image.
        source: Image source.
        window_size: Size of each processing window. Default 512.
        overlap: Overlap between windows. Default 64.
        model_name: Moondream model name.
        revision: Model revision.
        reasoning: Enable reasoning mode (moondream3 only).
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        show_progress: Whether to show progress bar.
        combine_strategy: How to combine answers ("concatenate" or "summarize").
        **kwargs: Additional arguments.

    Returns:
        Dictionary with "answer" and "tile_answers" keys.
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    return processor.query_sliding_window(
        question,
        source,
        window_size=window_size,
        overlap=overlap,
        reasoning=reasoning,
        bands=bands,
        show_progress=show_progress,
        combine_strategy=combine_strategy,
        **kwargs,
    )


def moondream_caption_sliding_window(
    source: Union[str, Image.Image, np.ndarray],
    window_size: int = 512,
    overlap: int = 64,
    length: str = "normal",
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
    combine_strategy: str = "concatenate",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to caption large images using sliding window.

    This function divides the image into overlapping windows, captions each,
    and combines the results.

    Args:
        source: Image source.
        window_size: Size of each processing window. Default 512.
        overlap: Overlap between windows. Default 64.
        length: Caption length ("short", "normal", "long").
        model_name: Moondream model name.
        revision: Model revision.
        bands: Band indices for GeoTIFF.
        device: Device for inference.
        show_progress: Whether to show progress bar.
        combine_strategy: How to combine captions ("concatenate" or "summarize").
        **kwargs: Additional arguments.

    Returns:
        Dictionary with "caption" and "tile_captions" keys.
    """
    processor = MoondreamGeo(model_name=model_name, revision=revision, device=device)
    return processor.caption_sliding_window(
        source,
        window_size=window_size,
        overlap=overlap,
        length=length,
        bands=bands,
        show_progress=show_progress,
        combine_strategy=combine_strategy,
        **kwargs,
    )
