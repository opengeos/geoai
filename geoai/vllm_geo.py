"""vLLM Vision Language Model module for GeoAI.

This module provides an interface for using open-source vision language
models served by vLLM (https://docs.vllm.ai) with geospatial imagery,
supporting GeoTIFF input and georeferenced output.

Two modes are supported:

- **Server mode** (default): connects to a running ``vllm serve <model>``
  OpenAI-compatible endpoint. Image tiles are sent as base64-encoded
  data URIs. Only requires ``requests``.
- **In-process mode** (``offline=True``): loads the model directly via
  ``vllm.LLM``. Requires ``vllm`` to be installed
  (``pip install geoai-py[vllm]``).

Suggested models:

- Qwen/Qwen2-VL-7B-Instruct (caption + VQA, best quality)
- llava-hf/llava-v1.6-mistral-7b-hf (general VQA)
- OpenGVLab/InternVL2-8B (scene understanding)
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import requests
from PIL import Image
from shapely.geometry import box
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

CAPTION_PROMPTS = {
    "short": "Describe this satellite/aerial image in one short sentence.",
    "normal": "Describe this satellite/aerial image, including land cover, "
    "structures, and notable features.",
    "long": "Provide a detailed description of this satellite/aerial image, "
    "including land cover types, structures, vegetation, water bodies, "
    "infrastructure, and spatial patterns.",
}

DETECT_PROMPT_TEMPLATE = (
    "Detect all instances of '{object_type}' in this image. "
    "Respond ONLY with a JSON array of bounding boxes with coordinates "
    "normalized to the 0-1 range, in this exact format: "
    '[{{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}}]. '
    "If no objects are found, respond with []."
)


def check_vllm_available() -> bool:
    """Check whether vLLM is installed in the current environment.

    Only required for in-process mode (``offline=True``). Server mode
    works without vLLM installed locally.

    Returns:
        True if ``vllm`` can be imported, False otherwise.

    Example:
        >>> import geoai
        >>> if geoai.check_vllm_available():
        ...     vlm = geoai.VLLMGeo(offline=True)
    """
    try:
        import vllm  # noqa: F401
    except Exception:  # noqa: BLE001 — catches broken installs
        return False
    else:
        return True


class VLLMGeo:
    """vLLM Vision Language Model processor with GeoTIFF support.

    Provides captioning, visual question answering, and prompt-based
    object detection for geospatial imagery using open-source VLMs
    served by vLLM.

    Attributes:
        model_id: HuggingFace model ID being served.
        base_url: vLLM server base URL (server mode).
        offline: Whether in-process mode is active.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        offline: bool = False,
        timeout: int = 120,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the vLLM processor.

        Args:
            model_id: HuggingFace model ID (e.g. "Qwen/Qwen2-VL-7B-Instruct").
            base_url: Base URL of the vLLM OpenAI-compatible server
                (server mode only). Defaults to "http://localhost:8000/v1".
            api_key: API key for the server. vLLM defaults to no auth,
                so "EMPTY" is the conventional placeholder.
            offline: If True, load the model in-process via ``vllm.LLM``
                instead of calling a server. Requires vLLM installed.
            timeout: Request timeout in seconds (server mode).
            max_tokens: Default maximum tokens to generate.
            temperature: Default sampling temperature.
            **kwargs: Additional arguments passed to ``vllm.LLM`` in
                offline mode. Ignored in server mode.

        Raises:
            ImportError: If ``offline=True`` and vLLM is not installed.
        """
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.offline = offline
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._source_path: Optional[str] = None
        self._metadata: Optional[Dict] = None
        self._llm = None

        if offline:
            if not check_vllm_available():
                raise ImportError(
                    "vLLM is required for in-process mode. "
                    "Install it with: pip install geoai-py[vllm]"
                )
            from vllm import LLM

            logger.info("Loading %s in-process via vLLM...", model_id)
            self._llm = LLM(model=model_id, **kwargs)

    # ------------------------------------------------------------------
    # Image loading (mirrors geoai.moondream)
    # ------------------------------------------------------------------

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
                metadata = {
                    "profile": src.profile.copy(),
                    "crs": src.crs,
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "width": src.width,
                    "height": src.height,
                }

                if bands is None:
                    bands = [1, 2, 3] if src.count >= 3 else [1]

                data = src.read(bands)

                if len(bands) == 1:
                    img_array = np.repeat(data[0:1], 3, axis=0)
                elif len(bands) >= 3:
                    img_array = data[:3]
                else:
                    img_array = np.zeros((3, data.shape[1], data.shape[2]))
                    img_array[: data.shape[0]] = data

                img_array = self._normalize_image(img_array)
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
            source: Image source - file path (GeoTIFF, PNG, JPG),
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
                source = np.stack([source] * 3, axis=-1)
            elif source.ndim == 3 and source.shape[0] <= 4:
                source = np.transpose(source[:3], (1, 2, 0))

            source = self._normalize_image(source)
            image = Image.fromarray(source.astype(np.uint8))
            self._source_path = None
            self._metadata = None
            return image, None

        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                from .utils import download_file

                source = download_file(source)

            try:
                with rasterio.open(source) as src:
                    if src.crs is not None or source.lower().endswith(
                        (".tif", ".tiff")
                    ):
                        return self.load_geotiff(source, bands)
            except rasterio.RasterioIOError:
                pass

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

        raise TypeError(f"Unsupported image source type: {type(source)}")

    def _normalize_image(self, data: np.ndarray) -> np.ndarray:
        """Normalize image data to 0-255 range using percentile stretching.

        Args:
            data: Input array (can be CHW or HWC format).

        Returns:
            Normalized array in uint8 range.
        """
        if data.dtype == np.uint8:
            return data

        if data.ndim == 3 and data.shape[0] <= 4:
            normalized = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[0]):
                band = data[i].astype(np.float32)
                p2, p98 = np.percentile(band, [2, 98])
                if p98 > p2:
                    normalized[i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    normalized[i] = np.clip(band, 0, 255)
        else:
            data = data.astype(np.float32)
            p2, p98 = np.percentile(data, [2, 98])
            if p98 > p2:
                normalized = np.clip((data - p2) / (p98 - p2) * 255, 0, 255)
            else:
                normalized = np.clip(data, 0, 255)

        return normalized.astype(np.uint8)

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode a PIL image as a base64 PNG data URI.

        Args:
            image: PIL Image to encode.

        Returns:
            Data URI string suitable for OpenAI-style image_url content.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        """Build OpenAI-style chat messages with optional image content.

        Args:
            prompt: Text prompt.
            image: Optional PIL image to attach.

        Returns:
            List of message dictionaries.
        """
        if image is None:
            content: Union[str, List[Dict[str, Any]]] = prompt
        else:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image_base64(image)},
                },
                {"type": "text", "text": prompt},
            ]
        return [{"role": "user", "content": content}]

    def _chat(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Run a single chat completion in server or offline mode.

        Args:
            prompt: Text prompt.
            image: Optional PIL image.
            max_tokens: Maximum tokens to generate (overrides default).
            temperature: Sampling temperature (overrides default).
            **kwargs: Additional sampling parameters.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If the request or generation fails.
        """
        messages = self._build_messages(prompt, image)
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        if self.offline:
            return self._chat_offline(messages, max_tokens, temperature, **kwargs)
        return self._chat_server(messages, max_tokens, temperature, **kwargs)

    def _chat_server(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Call the vLLM OpenAI-compatible chat completions endpoint.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional request body parameters.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If the request fails or returns an error.
        """
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"vLLM server request to {self.base_url} failed: {e}. "
                f"Make sure a server is running, e.g.: vllm serve {self.model_id}"
            ) from e

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected vLLM server response: {e}") from e

    def _chat_offline(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Run generation via an in-process ``vllm.LLM`` instance.

        Args:
            messages: Chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional SamplingParams arguments.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If generation fails.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )

        try:
            outputs = self._llm.chat(messages, sampling_params=sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            raise RuntimeError(f"vLLM in-process generation failed: {e}") from e

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def caption(
        self,
        source: Union[str, Image.Image, np.ndarray],
        length: str = "normal",
        prompt: Optional[str] = None,
        bands: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a caption for an image.

        Args:
            source: Image source (file path, PIL Image, or numpy array).
            length: Caption length - "short", "normal", or "long".
            prompt: Custom caption prompt (overrides ``length``).
            bands: Band indices for GeoTIFF.
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary with "caption" key containing the generated caption.
        """
        image, _ = self.load_image(source, bands)
        if prompt is None:
            prompt = CAPTION_PROMPTS.get(length, CAPTION_PROMPTS["normal"])
        answer = self._chat(prompt, image, **kwargs)
        return {"caption": answer}

    def query(
        self,
        question: str,
        source: Optional[Union[str, Image.Image, np.ndarray]] = None,
        bands: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Ask a question about an image, or a text-only question.

        Args:
            question: The question to ask.
            source: Image source. If None, performs a text-only query.
            bands: Band indices for GeoTIFF.
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary with "answer" key containing the response.
        """
        image = None
        if source is not None:
            image, _ = self.load_image(source, bands)
        answer = self._chat(question, image, **kwargs)
        return {"answer": answer}

    def detect(
        self,
        source: Union[str, Image.Image, np.ndarray],
        object_type: str,
        bands: Optional[List[int]] = None,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Detect objects of a specific type via JSON-prompted detection.

        The VLM is prompted to return bounding boxes as JSON with
        coordinates normalized to the 0-1 range. Detection quality
        depends on the model's spatial grounding ability.

        Args:
            source: Image source.
            object_type: Type of object to detect (e.g., "car", "building").
            bands: Band indices for GeoTIFF.
            output_path: Path to save results as GeoJSON/Shapefile/GeoPackage.
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary with "objects" key containing bounding boxes with
            normalized coordinates (x_min, y_min, x_max, y_max). If
            georeferenced, also includes "gdf" (GeoDataFrame), "crs",
            and "bounds" keys.
        """
        image, metadata = self.load_image(source, bands)

        prompt = DETECT_PROMPT_TEMPLATE.format(object_type=object_type)
        answer = self._chat(prompt, image, **kwargs)

        result = {"objects": self._parse_detections(answer)}

        if metadata and metadata.get("crs") and metadata.get("transform"):
            result = self._georef_detections(result, metadata)
            if output_path:
                self._save_vector(result["gdf"], output_path)

        return result

    def _parse_detections(self, text: str) -> List[Dict[str, float]]:
        """Parse bounding boxes from a model's JSON response.

        Args:
            text: Model response, expected to contain a JSON array of
                boxes with x_min/y_min/x_max/y_max in the 0-1 range.

        Returns:
            List of valid detection dictionaries (invalid entries dropped).
        """
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in detection response")
            return []

        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse detection JSON: %s", text[:200])
            return []

        detections = []
        for obj in raw:
            if not isinstance(obj, dict):
                continue
            try:
                det = {
                    "x_min": float(obj["x_min"]),
                    "y_min": float(obj["y_min"]),
                    "x_max": float(obj["x_max"]),
                    "y_max": float(obj["y_max"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            # Clamp to valid range and drop degenerate boxes
            det = {k: min(max(v, 0.0), 1.0) for k, v in det.items()}
            if det["x_max"] > det["x_min"] and det["y_max"] > det["y_min"]:
                detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Georeferencing (mirrors geoai.moondream)
    # ------------------------------------------------------------------

    def _georef_detections(
        self,
        result: Dict[str, Any],
        metadata: Dict,
    ) -> Dict[str, Any]:
        """Convert detection results to georeferenced format.

        Args:
            result: Detection result with normalized bounding boxes.
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
            px_x_min = obj["x_min"] * width
            px_y_min = obj["y_min"] * height
            px_x_max = obj["x_max"] * width
            px_y_max = obj["y_max"] * height

            geo_x_min, geo_y_min = transform * (px_x_min, px_y_max)
            geo_x_max, geo_y_max = transform * (px_x_max, px_y_min)

            geometries.append(box(geo_x_min, geo_y_min, geo_x_max, geo_y_max))
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

    def _save_vector(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str,
    ) -> None:
        """Save GeoDataFrame to vector file.

        Args:
            gdf: GeoDataFrame to save.
            output_path: Output file path. Extension determines format:
                .geojson -> GeoJSON, .shp -> Shapefile, .gpkg -> GeoPackage.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        ext = os.path.splitext(output_path)[1].lower()
        drivers = {".geojson": "GeoJSON", ".shp": "ESRI Shapefile", ".gpkg": "GPKG"}
        if ext in drivers:
            gdf.to_file(output_path, driver=drivers[ext])
        else:
            gdf.to_file(output_path)

        logger.info("Saved %d features to %s", len(gdf), output_path)

    # ------------------------------------------------------------------
    # Sliding window processing (mirrors geoai.moondream)
    # ------------------------------------------------------------------

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
                x_end = min(x + window_size, image_width)
                y_end = min(y + window_size, image_height)

                if (x_end - x) >= window_size // 2 and (y_end - y) >= window_size // 2:
                    windows.append((x, y, x_end, y_end))

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

        boxes = np.array(
            [[d["x_min"], d["y_min"], d["x_max"], d["y_max"]] for d in detections]
        )

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = y2.argsort()

        keep = []
        while order.size > 0:
            i = order[-1]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[:-1]] - intersection)
            order = order[np.where(iou <= iou_threshold)[0]]

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
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Detect objects using sliding window for large images.

        Divides large images into overlapping windows, runs detection on
        each tile, and merges results using Non-Maximum Suppression.

        Args:
            source: Image source.
            object_type: Type of object to detect.
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            iou_threshold: IoU threshold for NMS merging.
            bands: Band indices for GeoTIFF.
            output_path: Path to save results as vector file.
            show_progress: Whether to show progress bar.
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary with "objects" key containing merged bounding boxes.
            If georeferenced, also includes "gdf" (GeoDataFrame).
        """
        image, metadata = self.load_image(source, bands)
        width, height = image.size

        if width <= window_size and height <= window_size:
            return self.detect(image, object_type, output_path=output_path, **kwargs)

        windows = self._create_sliding_windows(width, height, window_size, overlap)
        all_detections = []

        iterator = (
            tqdm(windows, desc=f"Detecting {object_type}") if show_progress else windows
        )

        prompt = DETECT_PROMPT_TEMPLATE.format(object_type=object_type)

        for x_start, y_start, x_end, y_end in iterator:
            window_img = image.crop((x_start, y_start, x_end, y_end))

            try:
                answer = self._chat(prompt, window_img, **kwargs)
                window_width = x_end - x_start
                window_height = y_end - y_start

                for obj in self._parse_detections(answer):
                    all_detections.append(
                        {
                            "x_min": (x_start + obj["x_min"] * window_width) / width,
                            "y_min": (y_start + obj["y_min"] * window_height) / height,
                            "x_max": (x_start + obj["x_max"] * window_width) / width,
                            "y_max": (y_start + obj["y_max"] * window_height) / height,
                        }
                    )
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to process window (%d,%d)-(%d,%d): %s",
                    x_start,
                    y_start,
                    x_end,
                    y_end,
                    e,
                )

        merged = self._apply_nms(all_detections, iou_threshold)
        result = {"objects": merged}

        if metadata and metadata.get("crs") and metadata.get("transform"):
            result = self._georef_detections(result, metadata)
            if output_path:
                self._save_vector(result["gdf"], output_path)

        return result

    def query_sliding_window(
        self,
        question: str,
        source: Union[str, Image.Image, np.ndarray],
        window_size: int = 512,
        overlap: int = 64,
        bands: Optional[List[int]] = None,
        show_progress: bool = True,
        combine_strategy: str = "concatenate",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Query a large image using sliding window tiles.

        Args:
            question: The question to ask about each window.
            source: Image source.
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            bands: Band indices for GeoTIFF.
            show_progress: Whether to show progress bar.
            combine_strategy: How to combine answers - "concatenate" or
                "summarize" (uses the model to synthesize a summary).
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary with "answer" key containing the combined response,
            and "tile_answers" with individual tile responses.
        """
        image, _ = self.load_image(source, bands)
        width, height = image.size

        if width <= window_size and height <= window_size:
            return self.query(question, image, **kwargs)

        windows = self._create_sliding_windows(width, height, window_size, overlap)
        tile_answers = []

        iterator = tqdm(windows, desc="Querying tiles") if show_progress else windows

        for idx, (x_start, y_start, x_end, y_end) in enumerate(iterator):
            window_img = image.crop((x_start, y_start, x_end, y_end))

            try:
                answer = self._chat(question, window_img, **kwargs)
                tile_answers.append(
                    {
                        "tile_id": idx,
                        "bounds": (x_start, y_start, x_end, y_end),
                        "answer": answer,
                    }
                )
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to process window (%d,%d)-(%d,%d): %s",
                    x_start,
                    y_start,
                    x_end,
                    y_end,
                    e,
                )

        combined_answer = self._combine_tile_texts(
            tile_answers, "answer", question, combine_strategy
        )

        return {"answer": combined_answer, "tile_answers": tile_answers}

    def caption_sliding_window(
        self,
        source: Union[str, Image.Image, np.ndarray],
        window_size: int = 512,
        overlap: int = 64,
        length: str = "normal",
        bands: Optional[List[int]] = None,
        show_progress: bool = True,
        combine_strategy: str = "concatenate",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Caption a large image using sliding window tiles.

        Args:
            source: Image source.
            window_size: Size of each processing window/tile. Default 512.
            overlap: Overlap between adjacent windows. Default 64.
            length: Caption length - "short", "normal", or "long".
            bands: Band indices for GeoTIFF.
            show_progress: Whether to show progress bar.
            combine_strategy: How to combine captions - "concatenate" or
                "summarize" (uses the model to synthesize a summary).
            **kwargs: Additional sampling parameters.

        Returns:
            Dictionary with "caption" key containing the combined caption,
            and "tile_captions" with individual tile captions.
        """
        image, _ = self.load_image(source, bands)
        width, height = image.size

        if width <= window_size and height <= window_size:
            return self.caption(image, length=length, **kwargs)

        windows = self._create_sliding_windows(width, height, window_size, overlap)
        tile_captions = []

        prompt = CAPTION_PROMPTS.get(length, CAPTION_PROMPTS["normal"])
        iterator = (
            tqdm(windows, desc="Generating captions") if show_progress else windows
        )

        for idx, (x_start, y_start, x_end, y_end) in enumerate(iterator):
            window_img = image.crop((x_start, y_start, x_end, y_end))

            try:
                caption = self._chat(prompt, window_img, **kwargs)
                tile_captions.append(
                    {
                        "tile_id": idx,
                        "bounds": (x_start, y_start, x_end, y_end),
                        "caption": caption,
                    }
                )
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(
                    "Failed to process window (%d,%d)-(%d,%d): %s",
                    x_start,
                    y_start,
                    x_end,
                    y_end,
                    e,
                )

        combined_caption = self._combine_tile_texts(
            tile_captions, "caption", "the image content", combine_strategy
        )

        return {"caption": combined_caption, "tile_captions": tile_captions}

    def _combine_tile_texts(
        self,
        tiles: List[Dict[str, Any]],
        key: str,
        topic: str,
        combine_strategy: str,
    ) -> str:
        """Combine per-tile text results into a single response.

        Args:
            tiles: List of tile result dictionaries.
            key: Dictionary key holding the text ("answer" or "caption").
            topic: Topic description used in the summarization prompt.
            combine_strategy: "concatenate" or "summarize".

        Returns:
            Combined text string.
        """
        if combine_strategy == "summarize" and tiles:
            summary_prompt = (
                f"Based on these observations of different regions of an image "
                f"about '{topic}', provide a single comprehensive summary:\n\n"
            )
            for tile in tiles:
                summary_prompt += f"Region {tile['tile_id']}: {tile[key]}\n"

            try:
                return self._chat(summary_prompt)
            except RuntimeError:
                logger.warning("Summarization failed, falling back to concatenation")

        return "\n\n".join(
            f"Tile {tile['tile_id']} (region {tile['bounds']}): {tile[key]}"
            for tile in tiles
        )


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def vllm_caption(
    source: Union[str, Image.Image, np.ndarray],
    model_id: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    length: str = "normal",
    bands: Optional[List[int]] = None,
    **kwargs: Any,
) -> str:
    """Convenience function to generate a caption for an image via vLLM.

    Args:
        source: Image source (file path, PIL Image, or numpy array).
        model_id: HuggingFace model ID served by vLLM.
        base_url: vLLM server base URL.
        length: Caption length ("short", "normal", "long").
        bands: Band indices for GeoTIFF.
        **kwargs: Additional arguments for VLLMGeo.

    Returns:
        Generated caption string.
    """
    processor = VLLMGeo(model_id=model_id, base_url=base_url, **kwargs)
    result = processor.caption(source, length=length, bands=bands)
    return result["caption"]


def vllm_query(
    question: str,
    source: Optional[Union[str, Image.Image, np.ndarray]] = None,
    model_id: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    bands: Optional[List[int]] = None,
    **kwargs: Any,
) -> str:
    """Convenience function to ask a question about an image via vLLM.

    Args:
        question: Question to ask.
        source: Image source (optional for text-only queries).
        model_id: HuggingFace model ID served by vLLM.
        base_url: vLLM server base URL.
        bands: Band indices for GeoTIFF.
        **kwargs: Additional arguments for VLLMGeo.

    Returns:
        Answer string.
    """
    processor = VLLMGeo(model_id=model_id, base_url=base_url, **kwargs)
    result = processor.query(question, source=source, bands=bands)
    return result["answer"]


def vllm_detect(
    source: Union[str, Image.Image, np.ndarray],
    object_type: str,
    model_id: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    output_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to detect objects in an image via vLLM.

    Args:
        source: Image source.
        object_type: Type of object to detect.
        model_id: HuggingFace model ID served by vLLM.
        base_url: vLLM server base URL.
        output_path: Path to save results as vector file.
        bands: Band indices for GeoTIFF.
        **kwargs: Additional arguments for VLLMGeo.

    Returns:
        Detection results dictionary with "objects" and optionally "gdf".
    """
    processor = VLLMGeo(model_id=model_id, base_url=base_url, **kwargs)
    return processor.detect(source, object_type, output_path=output_path, bands=bands)
