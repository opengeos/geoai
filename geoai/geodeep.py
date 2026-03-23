"""GeoDeep integration for object detection and segmentation on geospatial imagery.

This module provides a Python interface to GeoDeep
(https://github.com/uav4geo/GeoDeep), a lightweight library for applying
ONNX AI models to geospatial rasters. Supports object detection (cars,
trees, birds, planes, utilities, aerovision) and semantic segmentation
(buildings, roads) with georeferenced output.

Requirements:
    - geodeep
    - onnxruntime (CPU) or onnxruntime-gpu (NVIDIA CUDA)

Install with::

    pip install geoai-py[geodeep]          # CPU only
    pip install geoai-py[geodeep-gpu]      # GPU (CUDA) support
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    from shapely.geometry import shape

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import geodeep as _geodeep
    from geodeep.segmentation import save_mask_to_raster as _save_mask_to_raster

    GEODEEP_AVAILABLE = True
except ImportError:
    GEODEEP_AVAILABLE = False

__all__ = [
    "GeoDeep",
    "geodeep_detect",
    "geodeep_segment",
    "geodeep_detect_batch",
    "geodeep_segment_batch",
    "list_geodeep_models",
    "check_geodeep_available",
    "GEODEEP_MODELS",
]

# ---------------------------------------------------------------------------
# Model registry — all built-in GeoDeep models
# ---------------------------------------------------------------------------

GEODEEP_MODELS: Dict[str, Dict[str, Any]] = {
    # Detection models
    "cars": {
        "type": "detection",
        "description": "Car detection using YOLOv7-m (10 cm/px)",
        "resolution": "10 cm/px",
        "classes": ["car"],
    },
    "trees": {
        "type": "detection",
        "description": "Tree crown detection using RetinaNet (10 cm/px)",
        "resolution": "10 cm/px",
        "classes": ["tree"],
    },
    "trees_yolov9": {
        "type": "detection",
        "description": "Tree crown detection using YOLOv9 (10 cm/px)",
        "resolution": "10 cm/px",
        "classes": ["tree"],
    },
    "birds": {
        "type": "detection",
        "description": "Bird detection using RetinaNet (2 cm/px)",
        "resolution": "2 cm/px",
        "classes": ["bird"],
    },
    "planes": {
        "type": "detection",
        "description": "Plane detection using YOLOv7-tiny (70 cm/px)",
        "resolution": "70 cm/px",
        "classes": ["plane"],
    },
    "aerovision": {
        "type": "detection",
        "description": ("Multi-class aerial detection using YOLOv8 (30 cm/px)"),
        "resolution": "30 cm/px",
        "classes": [
            "small-vehicle",
            "large-vehicle",
            "plane",
            "storage-tank",
            "boat",
            "dock",
            "track-field",
            "soccer-field",
            "tennis-court",
            "swimming-pool",
            "baseball-field",
            "road-circle",
            "basketball-court",
            "bridge",
            "helicopter",
            "crane",
        ],
    },
    "utilities": {
        "type": "detection",
        "description": ("Utility infrastructure detection using YOLOv8 (3 cm/px)"),
        "resolution": "3 cm/px",
        "classes": [
            "Gas",
            "Manhole",
            "Power",
            "Reclaimed",
            "Sewer",
            "Telecom",
            "Water",
        ],
    },
    # Segmentation models
    "buildings": {
        "type": "segmentation",
        "description": "Building segmentation using XUNet (50 cm/px)",
        "resolution": "50 cm/px",
        "classes": ["Background", "Building"],
    },
    "roads": {
        "type": "segmentation",
        "description": "Road segmentation (21 cm/px)",
        "resolution": "21 cm/px",
        "classes": ["not_road", "road"],
    },
}


def _get_onnx_device() -> str:
    """Detect the best available ONNX Runtime execution provider.

    Returns:
        str: ``"cuda"`` if CUDAExecutionProvider is available,
            otherwise ``"cpu"``.
    """
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def check_geodeep_available() -> None:
    """Check if geodeep is installed.

    Raises:
        ImportError: If geodeep is not installed.
    """
    if not GEODEEP_AVAILABLE:
        raise ImportError(
            "geodeep is not installed. "
            "Please install it with: pip install geodeep "
            "or: pip install geoai-py[geodeep]"
        )


def list_geodeep_models() -> Dict[str, str]:
    """List available GeoDeep built-in models.

    Returns:
        Dict[str, str]: Dictionary mapping model IDs to descriptions.

    Example:
        >>> from geoai import list_geodeep_models
        >>> models = list_geodeep_models()
        >>> for name, desc in models.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: info["description"] for name, info in GEODEEP_MODELS.items()}


class GeoDeep:
    """Object detection and segmentation on geospatial imagery using GeoDeep.

    GeoDeep is a lightweight library for applying ONNX AI models to
    geospatial rasters. It supports object detection (bounding boxes)
    and semantic segmentation (pixel masks) with built-in models for
    cars, trees, birds, planes, buildings, roads, and more.

    Models are automatically downloaded and cached on first use.

    Args:
        model_id (str): Built-in model name (see ``GEODEEP_MODELS``) or
            path to a custom ``.onnx`` file. Defaults to ``"cars"``.
        conf_threshold (float, optional): Override the default confidence
            threshold for detections. ``None`` uses the model default.
        classes (list, optional): Filter results to specific class names.
            Only applicable to multi-class models like ``"aerovision"``
            and ``"utilities"``. ``None`` keeps all classes.
        resolution (float, optional): Override the image resolution in
            cm/pixel. ``None`` lets GeoDeep estimate it from the raster.
        device (str, optional): Inference device — ``"auto"`` (default)
            selects CUDA if ``onnxruntime-gpu`` is installed, otherwise
            CPU. Use ``"cpu"`` to force CPU or ``"cuda"`` to require GPU.
        max_threads (int, optional): Maximum number of ONNX inference
            threads. ``None`` uses the default.

    Example:
        >>> from geoai import GeoDeep
        >>> gd = GeoDeep("cars")
        >>> detections = gd.detect("aerial_image.tif")
        >>> print(f"Found {len(detections)} cars")

    Example (GPU):
        >>> gd = GeoDeep("buildings", device="cuda")
        >>> print(gd.device)  # 'cuda'
    """

    def __init__(
        self,
        model_id: str = "cars",
        conf_threshold: Optional[float] = None,
        classes: Optional[List[str]] = None,
        resolution: Optional[float] = None,
        device: str = "auto",
        max_threads: Optional[int] = None,
    ) -> None:
        check_geodeep_available()

        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.resolution = resolution
        self.max_threads = max_threads

        # Resolve device
        if device == "auto":
            self._device = _get_onnx_device()
        else:
            self._device = device

        # Determine model type if it's a built-in model
        if model_id in GEODEEP_MODELS:
            self._model_info = GEODEEP_MODELS[model_id]
        else:
            # Custom model — metadata not available
            self._model_info = None

    @property
    def model_type(self) -> Optional[str]:
        """Return the model type (``'detection'`` or ``'segmentation'``)."""
        if self._model_info is not None:
            return self._model_info["type"]
        return None

    @property
    def model_info(self) -> Optional[Dict[str, Any]]:
        """Return model metadata dict (type, description, resolution, classes).

        Returns ``None`` for custom ONNX models not in the built-in registry.
        """
        return self._model_info

    @property
    def available_classes(self) -> Optional[List[str]]:
        """Return the list of class names this model can detect/segment.

        Returns ``None`` for custom ONNX models not in the built-in registry.
        """
        if self._model_info is not None:
            return list(self._model_info["classes"])
        return None

    @property
    def device(self) -> str:
        """Return the inference device (``'cpu'`` or ``'cuda'``)."""
        return self._device

    def _build_run_kwargs(
        self,
        verbose: bool = True,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Build keyword arguments for ``geodeep.run()``.

        Merges instance-level settings with per-call overrides.
        Per-call values take precedence over instance defaults.
        """
        kwargs: Dict[str, Any] = {}

        # Confidence threshold: per-call > instance > model default
        threshold = overrides.get("conf_threshold")
        if threshold is None:
            threshold = self.conf_threshold
        if threshold is not None:
            kwargs["conf_threshold"] = threshold

        # Class filter: per-call > instance
        cls_filter = overrides.get("classes")
        if cls_filter is None:
            cls_filter = self.classes
        if cls_filter is not None:
            kwargs["classes"] = cls_filter

        # Resolution: per-call > instance
        resolution = overrides.get("resolution")
        if resolution is None:
            resolution = self.resolution
        if resolution is not None:
            kwargs["resolution"] = resolution

        if self.max_threads is not None:
            kwargs["max_threads"] = self.max_threads
        if not verbose:
            kwargs["progress_callback"] = lambda *a: None

        return kwargs

    def detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[str]] = None,
        resolution: Optional[float] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> "gpd.GeoDataFrame":
        """Run object detection on a GeoTIFF image.

        Args:
            image_path (str): Path to the input GeoTIFF file.
            conf_threshold (float, optional): Confidence threshold override
                for this call. Falls back to the instance threshold, then
                the model default.
            classes (list, optional): Filter to specific class names for
                this call. Falls back to the instance setting.
            resolution (float, optional): Override image resolution in
                cm/pixel. Falls back to the instance setting.
            output_path (str, optional): Path to save results as a vector
                file (GeoJSON, GeoPackage, Shapefile). Format is inferred
                from the extension.
            verbose (bool): Print progress information. Defaults to True.

        Returns:
            geopandas.GeoDataFrame: Detection results with columns:
                - ``geometry``: Bounding box polygons (EPSG:4326).
                - ``score``: Confidence score (float).
                - ``class``: Class label (str).

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
            RuntimeError: If inference fails.

        Example:
            >>> gd = GeoDeep("cars")
            >>> detections = gd.detect("aerial.tif", conf_threshold=0.6)
            >>> print(detections[["class", "score"]].head())
        """
        self._validate_image_path(image_path)

        kwargs = self._build_run_kwargs(
            verbose=verbose,
            conf_threshold=conf_threshold,
            classes=classes,
            resolution=resolution,
        )
        kwargs["output_type"] = "geojson"

        try:
            geojson_str = _geodeep.run(image_path, self.model_id, **kwargs)
        except (RuntimeError, ValueError, OSError) as exc:
            raise RuntimeError(
                f"GeoDeep detection failed on '{image_path}': {exc}"
            ) from exc

        gdf = self._geojson_to_geodataframe(geojson_str)

        if output_path is not None:
            self._save_vector(gdf, output_path)
            if verbose:
                logger.info("Detections saved to: %s", output_path)

        return gdf

    def segment(
        self,
        image_path: str,
        output_raster_path: Optional[str] = None,
        output_vector_path: Optional[str] = None,
        resolution: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run semantic segmentation on a GeoTIFF image.

        Args:
            image_path (str): Path to the input GeoTIFF file.
            output_raster_path (str, optional): Path to save the
                segmentation mask as a georeferenced GeoTIFF.
            output_vector_path (str, optional): Path to save vectorized
                segmentation polygons (GeoJSON, GeoPackage, Shapefile).
            resolution (float, optional): Override image resolution in
                cm/pixel. Falls back to the instance setting.
            verbose (bool): Print progress information. Defaults to True.

        Returns:
            dict: Result dictionary with keys:
                - ``"mask"`` (numpy.ndarray): Segmentation mask of shape
                  ``(height, width)`` with ``uint8`` class indices.
                - ``"gdf"`` (geopandas.GeoDataFrame, optional): Vectorized
                  segmentation polygons. Present only when
                  ``output_vector_path`` is provided.

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
            RuntimeError: If inference fails.

        Example:
            >>> gd = GeoDeep("buildings")
            >>> result = gd.segment("city.tif",
            ...                     output_raster_path="mask.tif")
            >>> print(result["mask"].shape)
        """
        self._validate_image_path(image_path)

        run_kwargs = self._build_run_kwargs(
            verbose=verbose,
            resolution=resolution,
        )

        # Choose output_type based on what the caller needs.
        # If only vector is requested (no raster), use 'geojson' to avoid
        # running inference twice. Otherwise get 'raw' mask first.
        need_mask = output_raster_path is not None or output_vector_path is None
        need_vector = output_vector_path is not None

        result: Dict[str, Any] = {}

        if need_mask:
            try:
                mask = _geodeep.run(
                    image_path,
                    self.model_id,
                    output_type="raw",
                    **run_kwargs,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                raise RuntimeError(
                    f"GeoDeep segmentation failed on '{image_path}': {exc}"
                ) from exc

            result["mask"] = mask

            # Save raster mask
            if output_raster_path is not None:
                _save_mask_to_raster(image_path, mask, output_raster_path)
                result["raster_path"] = output_raster_path
                if verbose:
                    logger.info("Mask saved to: %s", output_raster_path)

        # Vectorize segmentation
        if need_vector:
            try:
                geojson_str = _geodeep.run(
                    image_path,
                    self.model_id,
                    output_type="geojson",
                    **run_kwargs,
                )
                gdf = self._geojson_to_geodataframe(geojson_str)
            except (RuntimeError, ValueError, OSError) as exc:
                raise RuntimeError(f"GeoDeep vectorization failed: {exc}") from exc

            self._save_vector(gdf, output_vector_path)
            result["gdf"] = gdf
            result["vector_path"] = output_vector_path
            if verbose:
                logger.info("Vectors saved to: %s", output_vector_path)

            # If mask wasn't computed above, still get it for completeness
            if "mask" not in result:
                try:
                    mask = _geodeep.run(
                        image_path,
                        self.model_id,
                        output_type="raw",
                        **run_kwargs,
                    )
                    result["mask"] = mask
                except (RuntimeError, ValueError, OSError):
                    pass  # Vector output was the priority

        return result

    def detect_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "gpd.GeoDataFrame":
        """Run object detection on multiple GeoTIFF images.

        Args:
            image_paths (list): List of paths to input GeoTIFF files.
            output_dir (str, optional): Directory to save per-image
                detection results as GeoJSON files.
            conf_threshold (float, optional): Confidence threshold override.
            classes (list, optional): Filter to specific class names.
            verbose (bool): Print progress information. Defaults to True.

        Returns:
            geopandas.GeoDataFrame: Combined detections from all images
                with an additional ``source_file`` column.

        Example:
            >>> gd = GeoDeep("trees")
            >>> results = gd.detect_batch(
            ...     ["area1.tif", "area2.tif"],
            ...     output_dir="results/"
            ... )
            >>> print(f"Total detections: {len(results)}")
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        all_gdfs = []
        total = len(image_paths)

        for idx, path in enumerate(image_paths, 1):
            if verbose:
                logger.info("Processing [%d/%d]: %s", idx, total, path)
            try:
                out_path = None
                if output_dir is not None:
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(output_dir, f"{base}_detections.geojson")

                gdf = self.detect(
                    path,
                    conf_threshold=conf_threshold,
                    classes=classes,
                    output_path=out_path,
                    verbose=False,
                )
                gdf["source_file"] = path
                all_gdfs.append(gdf)
            except (RuntimeError, ValueError, OSError) as exc:
                if verbose:
                    logger.warning("Failed on '%s': %s", path, exc)
                continue

        if all_gdfs:
            import pandas as pd

            combined = gpd.GeoDataFrame(
                pd.concat(all_gdfs, ignore_index=True),
                crs=all_gdfs[0].crs,
            )
        else:
            combined = gpd.GeoDataFrame(
                columns=["geometry", "class", "score", "source_file"],
                crs="EPSG:4326",
            )

        if verbose:
            logger.info("Total detections: %d", len(combined))

        return combined

    def segment_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        output_format: str = "raster",
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run semantic segmentation on multiple GeoTIFF images.

        Args:
            image_paths (list): List of paths to input GeoTIFF files.
            output_dir (str, optional): Directory to save results.
            output_format (str): Output format — ``"raster"`` for GeoTIFF
                masks, ``"vector"`` for GeoPackage polygons, or ``"both"``.
                Defaults to ``"raster"``.
            verbose (bool): Print progress information. Defaults to True.

        Returns:
            list: List of result dictionaries from ``segment()``,
                one per input image.

        Example:
            >>> gd = GeoDeep("roads")
            >>> results = gd.segment_batch(
            ...     ["tile1.tif", "tile2.tif"],
            ...     output_dir="masks/",
            ...     output_format="raster",
            ... )
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        total = len(image_paths)

        for idx, path in enumerate(image_paths, 1):
            if verbose:
                logger.info("Processing [%d/%d]: %s", idx, total, path)
            try:
                base = os.path.splitext(os.path.basename(path))[0]
                raster_out = None
                vector_out = None

                if output_dir is not None:
                    if output_format in ("raster", "both"):
                        raster_out = os.path.join(output_dir, f"{base}_mask.tif")
                    if output_format in ("vector", "both"):
                        vector_out = os.path.join(output_dir, f"{base}_segments.gpkg")

                result = self.segment(
                    path,
                    output_raster_path=raster_out,
                    output_vector_path=vector_out,
                    verbose=False,
                )
                result["source_file"] = path
                results.append(result)
            except (RuntimeError, ValueError, OSError) as exc:
                if verbose:
                    logger.warning("Failed on '%s': %s", path, exc)
                continue

        if verbose:
            logger.info("Processed %d/%d images", len(results), total)

        return results

    @staticmethod
    def _geojson_to_geodataframe(
        geojson_str: str,
        crs: str = "EPSG:4326",
    ) -> "gpd.GeoDataFrame":
        """Convert a GeoJSON string from GeoDeep to a GeoDataFrame.

        Args:
            geojson_str (str): GeoJSON FeatureCollection string.
            crs (str): Coordinate reference system. Defaults to EPSG:4326.

        Returns:
            geopandas.GeoDataFrame: Parsed features with geometry and
                attribute columns.
        """
        geojson_dict = json.loads(geojson_str)
        features = geojson_dict.get("features", [])

        if not features:
            return gpd.GeoDataFrame(
                columns=["geometry", "class", "score"],
                crs=crs,
            )

        geometries = []
        properties_list = []

        for feat in features:
            geom = shape(feat["geometry"])
            geometries.append(geom)
            props = dict(feat.get("properties", {}))
            properties_list.append(props)

        gdf = gpd.GeoDataFrame(
            properties_list,
            geometry=geometries,
            crs=crs,
        )

        return gdf

    @staticmethod
    def _save_vector(
        gdf: "gpd.GeoDataFrame",
        output_path: str,
        driver: Optional[str] = None,
    ) -> str:
        """Save a GeoDataFrame to a vector file.

        The file format is inferred from the extension unless ``driver``
        is specified explicitly.

        Args:
            gdf (geopandas.GeoDataFrame): Data to save.
            output_path (str): Output file path.
            driver (str, optional): OGR driver name. Inferred from
                extension if ``None``.

        Returns:
            str: The output path.
        """
        if driver is None:
            ext = os.path.splitext(output_path)[1].lower()
            driver_map = {
                ".geojson": "GeoJSON",
                ".json": "GeoJSON",
                ".gpkg": "GPKG",
                ".shp": "ESRI Shapefile",
                ".parquet": None,
                ".geoparquet": None,
            }
            driver = driver_map.get(ext, "GeoJSON")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if driver is None:
            gdf.to_parquet(output_path)
        else:
            gdf.to_file(output_path, driver=driver)

        return output_path

    @staticmethod
    def _validate_image_path(image_path: str) -> None:
        """Raise FileNotFoundError if the image does not exist."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: '{image_path}'")

    def __repr__(self) -> str:
        parts = [f"model_id={self.model_id!r}"]
        if self.conf_threshold is not None:
            parts.append(f"conf_threshold={self.conf_threshold}")
        if self.classes is not None:
            parts.append(f"classes={self.classes!r}")
        if self.resolution is not None:
            parts.append(f"resolution={self.resolution}")
        parts.append(f"device={self._device!r}")
        return f"GeoDeep({', '.join(parts)})"

    def __str__(self) -> str:
        model_type = self.model_type or "custom"
        desc = ""
        if self._model_info is not None:
            desc = f" - {self._model_info['description']}"
        return f"GeoDeep[{self.model_id}] ({model_type}, {self._device}){desc}"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def geodeep_detect(
    image_path: str,
    model_id: str = "cars",
    conf_threshold: Optional[float] = None,
    classes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    max_threads: Optional[int] = None,
    **kwargs: Any,
) -> "gpd.GeoDataFrame":
    """Run object detection on a GeoTIFF image using GeoDeep.

    Convenience function that creates a ``GeoDeep`` instance and runs
    detection in one call. For repeated use, instantiate ``GeoDeep``
    directly to avoid repeated initialization.

    Args:
        image_path (str): Path to the input GeoTIFF file.
        model_id (str): Model identifier. Defaults to ``"cars"``.
        conf_threshold (float, optional): Confidence threshold.
        classes (list, optional): Filter to specific class names.
        output_path (str, optional): Path to save vector results.
        max_threads (int, optional): Max ONNX inference threads.
        **kwargs: Additional keyword arguments passed to
            ``GeoDeep.detect()``.

    Returns:
        geopandas.GeoDataFrame: Detection results.

    Example:
        >>> from geoai import geodeep_detect
        >>> detections = geodeep_detect("image.tif", model_id="planes")
    """
    gd = GeoDeep(
        model_id=model_id,
        conf_threshold=conf_threshold,
        classes=classes,
        max_threads=max_threads,
    )
    return gd.detect(
        image_path,
        conf_threshold=conf_threshold,
        classes=classes,
        output_path=output_path,
        **kwargs,
    )


def geodeep_segment(
    image_path: str,
    model_id: str = "buildings",
    output_raster_path: Optional[str] = None,
    output_vector_path: Optional[str] = None,
    max_threads: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run semantic segmentation on a GeoTIFF image using GeoDeep.

    Convenience function that creates a ``GeoDeep`` instance and runs
    segmentation in one call.

    Args:
        image_path (str): Path to the input GeoTIFF file.
        model_id (str): Model identifier. Defaults to ``"buildings"``.
        output_raster_path (str, optional): Path to save mask GeoTIFF.
        output_vector_path (str, optional): Path to save vector polygons.
        max_threads (int, optional): Max ONNX inference threads.
        **kwargs: Additional keyword arguments passed to
            ``GeoDeep.segment()``.

    Returns:
        dict: Result dictionary with ``"mask"`` and optionally ``"gdf"``.

    Example:
        >>> from geoai import geodeep_segment
        >>> result = geodeep_segment("image.tif", model_id="roads",
        ...                          output_raster_path="roads.tif")
    """
    gd = GeoDeep(model_id=model_id, max_threads=max_threads)
    return gd.segment(
        image_path,
        output_raster_path=output_raster_path,
        output_vector_path=output_vector_path,
        **kwargs,
    )


def geodeep_detect_batch(
    image_paths: List[str],
    model_id: str = "cars",
    output_dir: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    classes: Optional[List[str]] = None,
    max_threads: Optional[int] = None,
    **kwargs: Any,
) -> "gpd.GeoDataFrame":
    """Run object detection on multiple GeoTIFF images using GeoDeep.

    Args:
        image_paths (list): List of paths to input GeoTIFF files.
        model_id (str): Model identifier. Defaults to ``"cars"``.
        output_dir (str, optional): Directory to save per-image results.
        conf_threshold (float, optional): Confidence threshold.
        classes (list, optional): Filter to specific class names.
        max_threads (int, optional): Max ONNX inference threads.
        **kwargs: Additional keyword arguments passed to
            ``GeoDeep.detect_batch()``.

    Returns:
        geopandas.GeoDataFrame: Combined detection results.

    Example:
        >>> from geoai import geodeep_detect_batch
        >>> results = geodeep_detect_batch(
        ...     ["img1.tif", "img2.tif"], model_id="trees"
        ... )
    """
    gd = GeoDeep(
        model_id=model_id,
        conf_threshold=conf_threshold,
        classes=classes,
        max_threads=max_threads,
    )
    return gd.detect_batch(
        image_paths,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        classes=classes,
        **kwargs,
    )


def geodeep_segment_batch(
    image_paths: List[str],
    model_id: str = "buildings",
    output_dir: Optional[str] = None,
    output_format: str = "raster",
    max_threads: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run semantic segmentation on multiple GeoTIFF images using GeoDeep.

    Args:
        image_paths (list): List of paths to input GeoTIFF files.
        model_id (str): Model identifier. Defaults to ``"buildings"``.
        output_dir (str, optional): Directory to save results.
        output_format (str): ``"raster"``, ``"vector"``, or ``"both"``.
            Defaults to ``"raster"``.
        max_threads (int, optional): Max ONNX inference threads.
        **kwargs: Additional keyword arguments passed to
            ``GeoDeep.segment_batch()``.

    Returns:
        list: List of result dictionaries.

    Example:
        >>> from geoai import geodeep_segment_batch
        >>> results = geodeep_segment_batch(
        ...     ["t1.tif", "t2.tif"],
        ...     model_id="roads",
        ...     output_dir="masks/",
        ... )
    """
    gd = GeoDeep(model_id=model_id, max_threads=max_threads)
    return gd.segment_batch(
        image_paths,
        output_dir=output_dir,
        output_format=output_format,
        **kwargs,
    )
