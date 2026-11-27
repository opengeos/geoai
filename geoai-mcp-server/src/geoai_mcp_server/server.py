"""GeoAI MCP Server - Main entry point.

This module implements the Model Context Protocol (MCP) server that exposes
GeoAI's geospatial AI capabilities to AI agents and LLM applications.

Usage:
    python -m geoai_mcp_server.server

For Claude Desktop integration, add to claude_desktop_config.json:
{
    "mcpServers": {
        "geoai": {
            "command": "python",
            "args": ["-m", "geoai_mcp_server.server"],
            "env": {
                "GEOAI_INPUT_DIR": "/path/to/input",
                "GEOAI_OUTPUT_DIR": "/path/to/output"
            }
        }
    }
}
"""

import asyncio
import builtins
import contextlib
import importlib
import logging
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:
    from mcp.server import MCPServer

from .config import GeoAIConfig, get_config, load_config, setup_logging
from .schemas import (
    # Segmentation
    SegmentObjectsInput,
    AutoSegmentInput,
    SegmentationResult,
    # Detection
    DetectionResult,
    ClassificationResult,
    # Change Detection
    ChangeDetectionResult,
    # Data Download
    DownloadImageryInput,
    PrepareTrainingDataInput,
    DownloadResult,
    TrainingDataResult,
    # Foundation Models
    FeatureExtractionResult,
    CanopyHeightResult,
    VLMResult,
    # Utilities
    CleanResultsResult,
    ListFilesResult,
    InspectRasterResult,
    InspectVectorResult,
    StacSearchResult,
    # Common
    OutputFormat,
    DataSource,
    FoundationModel,
    SegmentationModel,
    BoundingBox,
)
from .utils.error_handling import (
    GeoAIError,
    InputValidationError,
    FileAccessError,
    ModelLoadError,
    ProcessingError,
    OperationTimeoutError,
)
from .utils.file_management import (
    validate_output_path,
    validate_output_subdir,
    get_safe_file_path,
    get_safe_input_path,
    list_input_files,
    generate_output_filename,
)
from .utils.validation import (
    validate_confidence_threshold,
)

logger = logging.getLogger("geoai-mcp-server")

# Defer directory creation out of module import time.
# Tests may still patch ``geoai_mcp_server.server.config`` directly.
config: Optional[GeoAIConfig] = None

# Initialize MCP Server
mcp = MCPServer(
    "GeoAI Server",
    instructions="Geospatial AI tools for remote sensing and Earth observation analysis",
)


# =============================================================================
# Helper Functions
# =============================================================================


def _get_config() -> GeoAIConfig:
    """Return active configuration without creating directories at import time."""
    if config is not None:
        return config
    return get_config()


def _format_error_result(error: Exception, operation: str) -> dict[str, Any]:
    """Format an error into a consistent result structure."""
    return {
        "success": False,
        "message": f"Error during {operation}: {str(error)}",
        "output_files": [],
        "processing_time_seconds": None,
    }


def _get_geoai_module(module_name: str):
    """Lazily import GeoAI modules to avoid startup overhead."""
    try:
        return importlib.import_module(f"geoai.{module_name}")
    except ImportError as e:
        raise ModelLoadError(f"Failed to import geoai.{module_name}: {e}")


async def _run_blocking(fn, *args, **kwargs):
    """Execute a synchronous GeoAI call in a worker thread.

    Redirects stdout to stderr so stray print calls from libraries cannot
    corrupt MCP STDIO JSON-RPC framing, and enforces ``config.timeout``.
    """
    cfg = _get_config()
    raw_timeout = getattr(cfg, "timeout", 300)
    timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float)) else 300.0

    def _wrapped():
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)

    try:
        return await asyncio.wait_for(asyncio.to_thread(_wrapped), timeout=timeout)
    except (asyncio.TimeoutError, builtins.TimeoutError) as exc:
        raise OperationTimeoutError(
            message=f"Operation timed out after {int(timeout)} seconds",
            timeout_seconds=int(timeout),
            operation=getattr(fn, "__name__", str(fn)),
        ) from exc


def _sanitize_float(value: Any) -> Optional[float]:
    """Return a finite Python float or None if NaN/Inf/None."""
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return fval if math.isfinite(fval) else None


def _parse_output_format(output_format: str) -> OutputFormat:
    """Parse and validate an output format string."""
    try:
        return OutputFormat(output_format)
    except ValueError as exc:
        valid = ", ".join(fmt.value for fmt in OutputFormat)
        raise InputValidationError(
            message=f"Invalid output format '{output_format}'. Expected one of: {valid}",
            parameter_name="output_format",
            received_value=output_format,
            expected=valid,
        ) from exc


def _unlink_quietly(path: Optional[Path]) -> None:
    """Delete a temporary file if it exists, ignoring cleanup errors."""
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# =============================================================================
# SEGMENTATION TOOLS
# =============================================================================


@mcp.tool()
async def segment_objects_with_prompts(
    image_path: str,
    prompts: list[str],
    output_format: str = "geojson",
    model: str = "auto",
    confidence_threshold: float = 0.3,
    tile_size: int = 1024,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Segment objects in satellite/aerial imagery using text prompts.

    Uses foundation models (GroundedSAM, SAM, CLIPSeg) to find and segment
    objects matching natural language descriptions. Ideal for extracting
    buildings, roads, vegetation, water bodies, and other features.

    Args:
        image_path: Path to input image (relative to input directory).
                   Supports GeoTIFF, PNG, JPEG.
        prompts: Text descriptions of objects to find (e.g., ["building", "road"]).
        output_format: Output format: "geojson", "geotiff", "shapefile", "geopackage".
        model: Model to use: "sam", "sam2", "grounded_sam", "clipseg", "auto".
        confidence_threshold: Minimum confidence (0-1). Lower = more detections.
        tile_size: Tile size for large images (512-4096 pixels).
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with segmentation results, output file paths, and statistics.
    """
    start_time = time.time()
    cfg = _get_config()
    temp_files: list[Path] = []

    try:
        out_format = _parse_output_format(output_format)
        try:
            seg_model = SegmentationModel(model)
        except ValueError as exc:
            valid_models = ", ".join(m.value for m in SegmentationModel)
            raise InputValidationError(
                message=f"Invalid model '{model}'. Expected one of: {valid_models}",
                parameter_name="model",
                received_value=model,
                expected=valid_models,
            ) from exc

        input_data = SegmentObjectsInput(
            image_path=image_path,
            prompts=prompts,
            output_format=out_format,
            model=seg_model,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
            output_filename=output_filename,
        )

        full_input_path = get_safe_input_path(input_data.image_path, cfg)

        out_name = output_filename or generate_output_filename(
            input_data.image_path, "segmented", input_data.output_format.extension
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Segmenting objects in {full_input_path} with prompts: {prompts}")

        num_objects = 0
        output_files: list[str] = []

        if input_data.model in (SegmentationModel.GROUNDED_SAM, SegmentationModel.AUTO):
            # GroundedSAM writes a GeoTIFF and derives vector sidecars by replacing ".tif".
            segment_module = _get_geoai_module("segment")
            raster_path = output_path.with_suffix(".tif")

            def _run_grounded():
                segmenter = segment_module.GroundedSAM(
                    threshold=confidence_threshold,
                    tile_size=input_data.tile_size,
                )
                return segmenter.segment_image(
                    input_path=str(full_input_path),
                    output_path=str(raster_path),
                    text_prompts=prompts,
                    export_polygons=(input_data.output_format != OutputFormat.GEOTIFF),
                    export_boxes=(input_data.output_format == OutputFormat.GEOTIFF),
                )

            masks = await _run_blocking(_run_grounded)
            if isinstance(masks, dict):
                output_files = [str(p) for p in masks.values() if p]
                count_file = masks.get("polygons") or masks.get("boxes")
                if count_file:
                    try:
                        import geopandas as gpd

                        num_objects = len(gpd.read_file(count_file))
                    except Exception:
                        pass
            elif isinstance(masks, str):
                output_files = [masks]
            else:
                output_files = [str(raster_path)]

        elif input_data.model == SegmentationModel.CLIPSEG:
            segment_module = _get_geoai_module("segment")
            text_prompt = ", ".join(prompts)
            if input_data.output_format == OutputFormat.GEOTIFF:
                raster_path = output_path.with_suffix(".tif")
            else:
                fd, tmp_name = tempfile.mkstemp(
                    suffix=".tif", prefix="clipseg_", dir=str(output_path.parent)
                )
                import os

                os.close(fd)
                raster_path = Path(tmp_name)
                temp_files.append(raster_path)

            def _run_clipseg():
                segmenter = segment_module.CLIPSegmentation(
                    tile_size=input_data.tile_size,
                )
                return segmenter.segment_image(
                    input_path=str(full_input_path),
                    output_path=str(raster_path),
                    text_prompt=text_prompt,
                    threshold=confidence_threshold,
                )

            written_raster = await _run_blocking(_run_clipseg)
            if input_data.output_format != OutputFormat.GEOTIFF:
                utils_module = _get_geoai_module("utils")
                gdf = await _run_blocking(
                    utils_module.raster_to_vector,
                    str(raster_path),
                    str(output_path),
                )
                if hasattr(gdf, "__len__"):
                    num_objects = len(gdf)
                output_files = [str(output_path)]
            else:
                output_files = [
                    (
                        written_raster
                        if isinstance(written_raster, str)
                        else str(raster_path)
                    )
                ]

        else:
            # SAM / SAM2 automatic segmentation
            sam_module = _get_geoai_module("sam")
            sam_model_id = (
                "facebook/sam2.1-hiera-large"
                if input_data.model == SegmentationModel.SAM2
                else "facebook/sam-vit-huge"
            )
            if input_data.output_format == OutputFormat.GEOTIFF:
                raster_output_path = output_path
            else:
                fd, tmp_name = tempfile.mkstemp(
                    suffix=".tif", prefix="sam_", dir=str(output_path.parent)
                )
                import os

                os.close(fd)
                raster_output_path = Path(tmp_name)
                temp_files.append(raster_output_path)

            def _run_sam():
                segmenter = sam_module.SamGeo(model=sam_model_id, automatic=True)
                segmenter.generate(
                    source=str(full_input_path),
                    output=str(raster_output_path),
                )
                return segmenter

            segmenter = await _run_blocking(_run_sam)
            seg_masks = getattr(segmenter, "masks", None)
            if seg_masks is not None and hasattr(seg_masks, "__len__"):
                num_objects = len(seg_masks)

            if input_data.output_format != OutputFormat.GEOTIFF:
                utils_module = _get_geoai_module("utils")
                gdf = await _run_blocking(
                    utils_module.raster_to_vector,
                    str(raster_output_path),
                    str(output_path),
                )
                if num_objects == 0 and hasattr(gdf, "__len__"):
                    num_objects = len(gdf)

            output_files = [str(output_path)]

        processing_time = time.time() - start_time

        return SegmentationResult(
            success=True,
            message=f"Successfully segmented objects matching {prompts}",
            output_files=output_files,
            processing_time_seconds=processing_time,
            num_objects=num_objects,
            classes_found=prompts,
            statistics={
                "model_used": input_data.model.value,
                "confidence_threshold": confidence_threshold,
            },
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "text-prompted segmentation")
    except Exception as e:
        logger.exception("Unexpected error in segment_objects_with_prompts")
        return _format_error_result(e, "text-prompted segmentation")
    finally:
        for tmp_path in temp_files:
            _unlink_quietly(tmp_path)


@mcp.tool()
async def auto_segment_image(
    image_path: str,
    output_format: str = "geotiff",
    min_object_size: int = 100,
    max_object_size: Optional[int] = None,
    clean_results: bool = True,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Automatically segment all objects in an image without prompts.

    Uses Segment Anything Model (SAM) to detect and segment all distinct
    objects/regions in the image. Useful for exploratory analysis or when
    you don't know what objects are present.

    Args:
        image_path: Path to input image (relative to input directory).
        output_format: Output format: "geotiff", "geojson", "shapefile", "geopackage".
        min_object_size: Minimum object size in pixels to keep.
        max_object_size: Maximum object size in pixels (optional).
        clean_results: Whether to clean up small artifacts.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with segmentation masks and statistics.
    """
    start_time = time.time()
    cfg = _get_config()
    temp_files: list[Path] = []

    try:
        out_format = _parse_output_format(output_format)
        input_data = AutoSegmentInput(
            image_path=image_path,
            output_format=out_format,
            min_object_size=min_object_size,
            max_object_size=max_object_size,
            clean_results=clean_results,
            output_filename=output_filename,
        )

        full_input_path = get_safe_input_path(input_data.image_path, cfg)

        out_name = output_filename or generate_output_filename(
            input_data.image_path, "auto_segmented", input_data.output_format.extension
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Auto-segmenting {full_input_path}")

        sam_module = _get_geoai_module("sam")

        if input_data.output_format == OutputFormat.GEOTIFF and not clean_results:
            raster_output_path = output_path
        else:
            fd, tmp_name = tempfile.mkstemp(
                suffix=".tif", prefix="auto_seg_", dir=str(output_path.parent)
            )
            import os

            os.close(fd)
            raster_output_path = Path(tmp_name)
            temp_files.append(raster_output_path)

        def _run_sam_auto():
            segmenter = sam_module.SamGeo(model="facebook/sam-vit-huge", automatic=True)
            segmenter.generate(
                source=str(full_input_path),
                output=str(raster_output_path),
                min_size=min_object_size,
                max_size=max_object_size,
            )
            return segmenter

        segmenter = await _run_blocking(_run_sam_auto)
        seg_masks = getattr(segmenter, "masks", None)
        num_objects = (
            len(seg_masks)
            if seg_masks is not None and hasattr(seg_masks, "__len__")
            else 0
        )

        active_raster = raster_output_path
        if clean_results:
            utils_module = _get_geoai_module("utils")
            if input_data.output_format == OutputFormat.GEOTIFF:
                cleaned_raster = output_path
            else:
                fd, tmp_clean = tempfile.mkstemp(
                    suffix=".tif", prefix="auto_seg_clean_", dir=str(output_path.parent)
                )
                import os

                os.close(fd)
                cleaned_raster = Path(tmp_clean)
                temp_files.append(cleaned_raster)

            await _run_blocking(
                utils_module.clean_instance_mask,
                input_path=str(raster_output_path),
                output_path=str(cleaned_raster),
                min_area=min_object_size,
            )
            active_raster = cleaned_raster

        if input_data.output_format != OutputFormat.GEOTIFF:
            utils_module = _get_geoai_module("utils")
            gdf = await _run_blocking(
                utils_module.raster_to_vector,
                str(active_raster),
                str(output_path),
                min_area=min_object_size,
            )
            if hasattr(gdf, "__len__"):
                num_objects = len(gdf)

        processing_time = time.time() - start_time

        return SegmentationResult(
            success=True,
            message=f"Auto-segmentation complete, found {num_objects} objects",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            num_objects=num_objects,
            classes_found=["auto-detected"],
            statistics={"min_size_filter": min_object_size},
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "auto segmentation")
    except Exception as e:
        logger.exception("Unexpected error in auto_segment_image")
        return _format_error_result(e, "auto segmentation")
    finally:
        for tmp_path in temp_files:
            _unlink_quietly(tmp_path)


# =============================================================================
# DETECTION AND CLASSIFICATION TOOLS
# =============================================================================


@mcp.tool()
async def detect_and_classify_features(
    image_path: str,
    feature_types: list[str],
    confidence_threshold: float = 0.5,
    output_format: str = "geojson",
    custom_prompts: Optional[list[str]] = None,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Detect and classify specific features in geospatial imagery.

    Identifies and locates specific types of objects like buildings, vehicles,
    ships, solar panels, etc. Returns bounding boxes and classifications.

    Args:
        image_path: Path to input image.
        feature_types: Types to detect: "buildings", "vehicles", "ships",
                      "solar_panels", "trees", "water", "roads", "custom".
        confidence_threshold: Minimum detection confidence (0-1).
        output_format: Output format for results.
        custom_prompts: Custom text prompts when feature_type is "custom".
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with detection results and bounding boxes.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        if not feature_types:
            raise InputValidationError(
                message="At least one feature type is required",
                parameter_name="feature_types",
                received_value=feature_types,
                expected="Non-empty list of feature types",
            )
        validate_confidence_threshold(confidence_threshold)
        out_format = _parse_output_format(output_format)

        full_input_path = get_safe_input_path(image_path, cfg)

        out_name = output_filename or generate_output_filename(
            image_path, "detections", out_format.extension
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Detecting {feature_types} in {full_input_path}")

        prompt_map = {
            "buildings": ["building"],
            "vehicles": ["car", "truck", "vehicle"],
            "ships": ["ship", "boat", "vessel"],
            "solar_panels": ["solar panel"],
            "trees": ["tree"],
            "water": ["water", "pond", "lake", "river"],
            "roads": ["road"],
        }

        prompts: list[str] = []
        for ft in feature_types:
            if ft == "custom":
                if not custom_prompts:
                    raise InputValidationError(
                        message="custom_prompts is required when feature_types includes 'custom'",
                        parameter_name="custom_prompts",
                        expected="Non-empty list of text prompts",
                    )
                prompts.extend(custom_prompts)
            elif ft in prompt_map:
                prompts.extend(prompt_map[ft])
            else:
                prompts.append(ft)

        segment_module = _get_geoai_module("segment")
        raster_path = output_path.with_suffix(".tif")

        def _run_detect():
            detector = segment_module.GroundedSAM(threshold=confidence_threshold)
            return detector.segment_image(
                input_path=str(full_input_path),
                output_path=str(raster_path),
                text_prompts=prompts,
                export_boxes=True,
                export_polygons=(out_format != OutputFormat.GEOTIFF),
            )

        results = await _run_blocking(_run_detect)

        detections_by_class: dict[str, int] = {}
        num_detections = 0
        output_files: list[str] = []

        if isinstance(results, dict):
            output_files = [str(p) for p in results.values() if p]
            vector_file = results.get("boxes") or results.get("polygons")
            if vector_file and Path(vector_file).exists():
                import geopandas as gpd

                gdf = gpd.read_file(vector_file)
                num_detections = len(gdf)
                label_col = next(
                    (c for c in ("label", "class", "category") if c in gdf.columns),
                    None,
                )
                if label_col:
                    for cls_name, cnt in gdf[label_col].value_counts().items():
                        detections_by_class[str(cls_name)] = int(cnt)
                elif num_detections > 0:
                    detections_by_class["detected"] = num_detections
        else:
            output_files = [str(output_path)]

        processing_time = time.time() - start_time

        return DetectionResult(
            success=True,
            message=f"Detection complete: found {num_detections} objects",
            output_files=output_files,
            processing_time_seconds=processing_time,
            num_detections=num_detections,
            detections_by_class=detections_by_class,
            average_confidence=confidence_threshold,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "feature detection")
    except Exception as e:
        logger.exception("Unexpected error in detect_and_classify_features")
        return _format_error_result(e, "feature detection")


@mcp.tool()
async def classify_land_cover(
    image_path: str,
    model_path: Optional[str] = None,
    num_classes: int = 10,
    output_format: str = "geotiff",
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Classify land cover types in satellite/aerial imagery.

    Performs pixel-wise classification to identify land cover categories
    such as urban, forest, water, agriculture, etc.

    Args:
        image_path: Path to input multispectral image.
        model_path: Path to a trained classification model (required, within input directory).
        num_classes: Number of land cover classes (2-100).
        output_format: Output format for classified map.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with classification results and class distribution.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        out_format = _parse_output_format(output_format)

        if not model_path:
            raise InputValidationError(
                message="model_path is required for land cover classification",
                parameter_name="model_path",
                received_value=model_path,
                expected="A local model file path within the input directory",
            )

        if "://" in model_path or model_path.lower().startswith(
            ("http:", "https:", "ftp:", "file:", "s3:", "gs:")
        ):
            raise InputValidationError(
                message="Model path must be a local file within the input directory, not a URL",
                parameter_name="model_path",
                received_value=model_path,
                expected="A local model file path within the input directory",
            )

        full_input_path = get_safe_input_path(image_path, cfg)
        validated_model_path = str(get_safe_input_path(model_path, cfg))

        out_name = output_filename or generate_output_filename(
            image_path, "classified", out_format.extension
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Classifying land cover in {full_input_path}")

        classify_module = _get_geoai_module("classify")

        result = await _run_blocking(
            classify_module.classify_image,
            str(full_input_path),
            model_path=validated_model_path,
            output_path=str(output_path),
        )

        class_distribution: dict[str, float] = {}
        dominant_class: Optional[str] = None
        if hasattr(result, "class_distribution") and isinstance(
            result.class_distribution, dict
        ):
            class_distribution = result.class_distribution
            if class_distribution:
                dominant_class = max(class_distribution, key=class_distribution.get)
        elif output_path.exists() and out_format == OutputFormat.GEOTIFF:
            try:
                import numpy as np
                import rasterio

                with rasterio.open(output_path) as src:
                    arr = src.read(1)
                    vals, counts = np.unique(arr, return_counts=True)
                    total = int(arr.size)
                    if total > 0:
                        class_distribution = {
                            str(int(v)): round(float(c) / total * 100.0, 2)
                            for v, c in zip(vals, counts)
                        }
                        dominant_class = max(
                            class_distribution, key=class_distribution.get
                        )
            except Exception:
                pass

        processing_time = time.time() - start_time

        return ClassificationResult(
            success=True,
            message=f"Land cover classification complete with {num_classes} classes",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            class_distribution=class_distribution,
            dominant_class=dominant_class,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "land cover classification")
    except Exception as e:
        logger.exception("Unexpected error in classify_land_cover")
        return _format_error_result(e, "land cover classification")


# =============================================================================
# CHANGE DETECTION TOOLS
# =============================================================================


@mcp.tool()
async def detect_temporal_changes(
    image1_path: str,
    image2_path: str,
    change_threshold: float = 0.5,
    output_format: str = "geotiff",
    include_statistics: bool = True,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Detect changes between two temporal images of the same area.

    Compares two images from different times to identify areas of change,
    such as new construction, deforestation, urban expansion, or damage.

    Args:
        image1_path: Path to the earlier image.
        image2_path: Path to the later image.
        change_threshold: Sensitivity threshold (0-1). Higher = fewer changes detected.
        output_format: Output format for change map.
        include_statistics: Whether to calculate detailed change statistics.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with change map, statistics, and change percentages.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        validate_confidence_threshold(change_threshold)
        out_format = _parse_output_format(output_format)

        full_path1 = get_safe_input_path(image1_path, cfg)
        full_path2 = get_safe_input_path(image2_path, cfg)

        out_name = output_filename or generate_output_filename(
            image1_path, "change_map", out_format.extension
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Detecting changes between {full_path1} and {full_path2}")

        change_module = _get_geoai_module("change_detection")

        detect_kwargs: dict[str, Any] = {
            "image1_path": str(full_path1),
            "image2_path": str(full_path2),
            "threshold": change_threshold,
        }
        if out_format == OutputFormat.GEOTIFF:
            detect_kwargs["output_change"] = str(output_path)
        else:
            detect_kwargs["output_vector"] = str(output_path)

        result = await _run_blocking(change_module.changestar_detect, **detect_kwargs)

        change_percentage = 0.0
        change_area_sq_meters = None
        change_types: dict[str, float] = {}

        if include_statistics:
            if isinstance(result, dict) and "change" in result:
                import numpy as np

                change_arr = np.asarray(result["change"])
                if change_arr.size > 0:
                    change_percentage = round(
                        float(np.count_nonzero(change_arr)) / change_arr.size * 100.0,
                        2,
                    )
            elif hasattr(result, "statistics") and isinstance(result.statistics, dict):
                stats = result.statistics
                change_percentage = float(stats.get("change_percentage", 0.0))
                change_area_sq_meters = stats.get("change_area_sq_meters")
                change_types = stats.get("change_types", {})
            elif output_path.exists() and out_format == OutputFormat.GEOTIFF:
                try:
                    import numpy as np
                    import rasterio

                    with rasterio.open(output_path) as src:
                        data = src.read(1)
                        if data.size > 0:
                            change_percentage = round(
                                float(np.count_nonzero(data)) / data.size * 100.0, 2
                            )
                except Exception:
                    pass

        processing_time = time.time() - start_time

        return ChangeDetectionResult(
            success=True,
            message=f"Change detection complete: {change_percentage:.1f}% change detected",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            change_percentage=change_percentage,
            change_area_sq_meters=change_area_sq_meters,
            change_types=change_types,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "change detection")
    except Exception as e:
        logger.exception("Unexpected error in detect_temporal_changes")
        return _format_error_result(e, "change detection")


# =============================================================================
# DATA DOWNLOAD TOOLS
# =============================================================================


@mcp.tool()
async def download_satellite_imagery(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    data_source: str,
    date_range: Optional[str] = None,
    max_cloud_cover: float = 20,
    max_items: int = 10,
    output_subdir: Optional[str] = None,
) -> dict[str, Any]:
    """Download satellite or aerial imagery for an area of interest.

    Fetches imagery from supported sources including NAIP, Sentinel-2, Landsat,
    and Microsoft Planetary Computer.

    Args:
        min_lon: Minimum longitude (west boundary).
        min_lat: Minimum latitude (south boundary).
        max_lon: Maximum longitude (east boundary).
        max_lat: Maximum latitude (north boundary).
        data_source: Source: "naip", "sentinel2", "landsat", "planetary_computer".
        date_range: Date range as "YYYY-MM-DD/YYYY-MM-DD" (optional).
        max_cloud_cover: Maximum cloud cover percentage (0-100).
        max_items: Maximum number of images to download (1-100).
        output_subdir: Subdirectory for downloads (optional).

    Returns:
        Dictionary with download results and file paths.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        try:
            source_enum = DataSource(data_source)
        except ValueError as exc:
            valid_sources = ", ".join(s.value for s in DataSource)
            raise InputValidationError(
                message=f"Unsupported data_source '{data_source}'. Expected one of: {valid_sources}",
                parameter_name="data_source",
                received_value=data_source,
                expected=valid_sources,
            ) from exc

        bbox = BoundingBox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        )
        DownloadImageryInput(
            bbox=bbox,
            data_source=source_enum,
            date_range=date_range,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
            output_subdir=output_subdir,
        )

        output_dir = cfg.output_dir
        if output_subdir:
            output_dir = validate_output_subdir(output_subdir, cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloading {source_enum.value} imagery for bbox: {bbox.to_tuple()}"
        )

        download_module = _get_geoai_module("download")

        downloaded_files: list[str] = []
        total_size_mb = 0.0

        if source_enum == DataSource.NAIP:
            files = await _run_blocking(
                download_module.download_naip,
                bbox=bbox.to_tuple(),
                output_dir=str(output_dir),
                max_items=max_items,
            )
            if isinstance(files, list):
                downloaded_files = [str(f) for f in files if f]
            elif files:
                downloaded_files = [str(files)]
        else:
            collection_map = {
                DataSource.SENTINEL2: "sentinel-2-l2a",
                DataSource.LANDSAT: "landsat-c2-l2",
                DataSource.PLANETARY_COMPUTER: "sentinel-2-l2a",
            }
            collection = collection_map[source_enum]
            query = {"eo:cloud_cover": {"lt": max_cloud_cover}}

            items = await _run_blocking(
                download_module.pc_stac_search,
                collection=collection,
                bbox=list(bbox.to_tuple()),
                time_range=date_range,
                query=query,
                max_items=max_items,
                quiet=True,
            )
            if items:
                dl_map = await _run_blocking(
                    download_module.pc_stac_download,
                    items=items,
                    output_dir=str(output_dir),
                )
                if isinstance(dl_map, dict):
                    for asset_dict in dl_map.values():
                        if isinstance(asset_dict, dict):
                            downloaded_files.extend(
                                str(p) for p in asset_dict.values() if p
                            )
                        elif asset_dict:
                            downloaded_files.append(str(asset_dict))

        for f in downloaded_files:
            fp = Path(f)
            if fp.exists():
                total_size_mb += fp.stat().st_size / (1024 * 1024)

        processing_time = time.time() - start_time

        return DownloadResult(
            success=True,
            message=f"Downloaded {len(downloaded_files)} files from {source_enum.value}",
            output_files=downloaded_files,
            processing_time_seconds=processing_time,
            num_files=len(downloaded_files),
            total_size_mb=round(total_size_mb, 4),
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "imagery download")
    except Exception as e:
        logger.exception("Unexpected error in download_satellite_imagery")
        return _format_error_result(e, "imagery download")


@mcp.tool()
async def prepare_training_data(
    images_dir: str,
    labels_dir: str,
    tile_size: int = 256,
    overlap: int = 32,
    train_ratio: float = 0.7,
    augment: bool = True,
    output_subdir: Optional[str] = None,
) -> dict[str, Any]:
    """Prepare training datasets from images and labels.

    Creates tiled training data suitable for deep learning using GeoAI's
    GeoTIFF tile export utilities.

    Args:
        images_dir: Directory or image file containing source imagery.
        labels_dir: Directory or label file containing label masks/vectors.
        tile_size: Output tile size in pixels (64-2048).
        overlap: Tile overlap in pixels (must be less than tile_size).
        train_ratio: Training data proportion (0.1-0.95).
        augment: Whether to apply data augmentation (single-raster mode).
        output_subdir: Subdirectory for training data (optional).

    Returns:
        Dictionary with training data statistics and paths.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        PrepareTrainingDataInput(
            images_dir=images_dir,
            labels_dir=labels_dir,
            tile_size=tile_size,
            overlap=overlap,
            train_ratio=train_ratio,
            augment=augment,
            output_subdir=output_subdir,
        )
        if overlap >= tile_size:
            raise InputValidationError(
                message=f"overlap ({overlap}) must be less than tile_size ({tile_size})",
                parameter_name="overlap",
                received_value=overlap,
                expected=f"An integer less than {tile_size}",
            )

        images_path = get_safe_input_path(images_dir, cfg)
        labels_path = get_safe_input_path(labels_dir, cfg)

        output_dir = cfg.output_dir
        if output_subdir:
            output_dir = validate_output_subdir(output_subdir, cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing training data: {images_path} + {labels_path}")

        utils_module = _get_geoai_module("utils")
        stride = tile_size - overlap

        if images_path.is_dir():
            batch_kwargs: dict[str, Any] = {
                "images_folder": str(images_path),
                "output_folder": str(output_dir),
                "tile_size": tile_size,
                "stride": stride,
                "quiet": True,
            }
            if labels_path.is_dir():
                batch_kwargs["masks_folder"] = str(labels_path)
            else:
                batch_kwargs["masks_file"] = str(labels_path)

            result = await _run_blocking(
                utils_module.export_geotiff_tiles_batch, **batch_kwargs
            )
            num_tiles = (
                int(result.get("total_tiles", 0)) if isinstance(result, dict) else 0
            )
        else:
            await _run_blocking(
                utils_module.export_geotiff_tiles,
                in_raster=str(images_path),
                out_folder=str(output_dir),
                in_class_data=str(labels_path),
                tile_size=tile_size,
                stride=stride,
                quiet=True,
                apply_augmentation=augment,
            )
            images_out = output_dir / "images"
            if images_out.is_dir():
                num_tiles = sum(1 for p in images_out.iterdir() if p.is_file())
            else:
                num_tiles = sum(1 for p in output_dir.rglob("*.tif") if p.is_file())

        num_train = int(num_tiles * train_ratio)
        num_val = num_tiles - num_train
        processing_time = time.time() - start_time

        return TrainingDataResult(
            success=True,
            message=f"Created {num_tiles} training tiles",
            output_files=[str(output_dir)],
            processing_time_seconds=processing_time,
            num_tiles=num_tiles,
            num_train=num_train,
            num_val=num_val,
            tile_size=tile_size,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "training data preparation")
    except Exception as e:
        logger.exception("Unexpected error in prepare_training_data")
        return _format_error_result(e, "training data preparation")


# =============================================================================
# FOUNDATION MODEL TOOLS
# =============================================================================


@mcp.tool()
async def extract_features_with_foundation_model(
    image_path: str,
    model: str = "dinov3",
    output_type: str = "embeddings",
    reference_x: Optional[int] = None,
    reference_y: Optional[int] = None,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Extract deep features using geospatial foundation models.

    Uses DINOv3 or Prithvi to extract rich visual features for downstream
    tasks like similarity search, clustering, or transfer learning.

    Args:
        image_path: Path to input image.
        model: Foundation model: "dinov3" or "prithvi".
        output_type: "embeddings" for raw features, "similarity_map" for similarity analysis.
        reference_x: X coordinate for similarity reference point (pair with reference_y).
        reference_y: Y coordinate for similarity reference point (pair with reference_x).
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with extracted features or similarity map.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        try:
            fm = FoundationModel(model)
        except ValueError as exc:
            valid = ", ".join(m.value for m in FoundationModel)
            raise InputValidationError(
                message=f"Unknown model '{model}'. Expected one of: {valid}",
                parameter_name="model",
                received_value=model,
                expected=valid,
            ) from exc

        if output_type not in ("embeddings", "similarity_map"):
            raise InputValidationError(
                message=f"Invalid output_type '{output_type}'. Use 'embeddings' or 'similarity_map'.",
                parameter_name="output_type",
                received_value=output_type,
                expected="'embeddings' or 'similarity_map'",
            )

        if (reference_x is None) != (reference_y is None):
            raise InputValidationError(
                message="reference_x and reference_y must both be provided or both be omitted",
                parameter_name="reference_x/reference_y",
                received_value=f"reference_x={reference_x}, reference_y={reference_y}",
                expected="Both integer coordinates or both None",
            )

        if output_type == "similarity_map" and reference_x is None:
            raise InputValidationError(
                message="Both reference_x and reference_y are required for output_type='similarity_map'",
                parameter_name="reference_x/reference_y",
                expected="Both reference_x and reference_y coordinates",
            )

        full_input_path = get_safe_input_path(image_path, cfg)

        out_name = output_filename or generate_output_filename(
            image_path, f"features_{fm.value}", "npy"
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Extracting features from {full_input_path} using {fm.value}")

        feature_dims: tuple[int, ...] = ()
        output_files: list[str] = [str(output_path)]

        if fm == FoundationModel.DINOV3:
            dino_module = _get_geoai_module("dinov3")

            if output_type == "similarity_map":
                sim_dir = output_path.with_suffix("")
                sim_dir.mkdir(parents=True, exist_ok=True)
                result = await _run_blocking(
                    dino_module.create_similarity_map,
                    input_image=str(full_input_path),
                    query_coords=(float(reference_x), float(reference_y)),
                    output_dir=str(sim_dir),
                )
                if isinstance(result, dict):
                    sim_data = result.get("similarity_map")
                    if sim_data is None and "image_dict" in result:
                        img_dict = result["image_dict"]
                        if isinstance(img_dict, dict):
                            sim_data = img_dict.get("image")
                    if sim_data is not None and hasattr(sim_data, "shape"):
                        feature_dims = tuple(int(d) for d in sim_data.shape)
                generated = sorted(str(p) for p in sim_dir.iterdir() if p.is_file())
                output_files = generated or [str(sim_dir)]
            else:
                import numpy as np

                def _extract_dinov3():
                    processor = dino_module.DINOv3GeoProcessor()
                    data, _meta = processor.load_image(str(full_input_path))
                    pil_img = processor.preprocess_image_for_dinov3(data)
                    features, _h, _w = processor.extract_features(pil_img)
                    arr = (
                        features.detach().cpu().numpy()
                        if hasattr(features, "detach")
                        else np.asarray(features)
                    )
                    np.save(str(output_path), arr)
                    return arr

                arr = await _run_blocking(_extract_dinov3)
                if hasattr(arr, "shape"):
                    feature_dims = tuple(int(d) for d in arr.shape)

        elif fm == FoundationModel.PRITHVI:
            import numpy as np

            prithvi_module = _get_geoai_module("prithvi")

            def _extract_prithvi():
                processor = prithvi_module.PrithviProcessor()
                _input_tensor, rec_tensor, _mask_tensor = processor.process_images(
                    file_paths=[str(full_input_path)]
                )
                arr = (
                    rec_tensor.detach().cpu().numpy()
                    if hasattr(rec_tensor, "detach")
                    else np.asarray(rec_tensor)
                )
                np.save(str(output_path), arr)
                return arr

            arr = await _run_blocking(_extract_prithvi)
            if hasattr(arr, "shape"):
                feature_dims = tuple(int(d) for d in arr.shape)

        processing_time = time.time() - start_time

        return FeatureExtractionResult(
            success=True,
            message=f"Feature extraction complete using {fm.value}",
            output_files=output_files,
            processing_time_seconds=processing_time,
            feature_dimensions=feature_dims,
            model_used=fm.value,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "feature extraction")
    except Exception as e:
        logger.exception("Unexpected error in extract_features_with_foundation_model")
        return _format_error_result(e, "feature extraction")


@mcp.tool()
async def estimate_canopy_height(
    image_path: str,
    output_format: str = "geotiff",
    include_statistics: bool = True,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Estimate canopy/vegetation height from RGB imagery.

    Uses deep learning to estimate tree and vegetation heights from
    standard RGB aerial/satellite imagery without requiring LiDAR.

    Args:
        image_path: Path to RGB input image.
        output_format: Output format for height map.
        include_statistics: Whether to calculate height statistics.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with height map and statistics.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        out_format = _parse_output_format(output_format)
        full_input_path = get_safe_input_path(image_path, cfg)

        out_name = output_filename or generate_output_filename(
            image_path, "canopy_height", out_format.extension
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Estimating canopy height for {full_input_path}")

        canopy_module = _get_geoai_module("canopy")

        heights = await _run_blocking(
            canopy_module.canopy_height_estimation,
            str(full_input_path),
            output_path=str(output_path),
        )

        min_height = 0.0
        max_height = 0.0
        mean_height = 0.0
        coverage = 0.0

        if include_statistics:
            if hasattr(heights, "statistics") and isinstance(heights.statistics, dict):
                stats = heights.statistics
                min_height = float(stats.get("min_height", 0.0))
                max_height = float(stats.get("max_height", 0.0))
                mean_height = float(stats.get("mean_height", 0.0))
                coverage = float(stats.get("forest_coverage", 0.0))
            elif heights is not None:
                import numpy as np

                arr = np.asarray(heights, dtype=float)
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    min_height = round(float(np.min(valid)), 3)
                    max_height = round(float(np.max(valid)), 3)
                    mean_height = round(float(np.mean(valid)), 3)
                    coverage = round(float(np.mean(valid > 0)) * 100.0, 2)

        processing_time = time.time() - start_time

        return CanopyHeightResult(
            success=True,
            message=f"Canopy height estimation complete. Mean height: {mean_height:.1f}m",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            min_height=min_height,
            max_height=max_height,
            mean_height=mean_height,
            forest_coverage_percent=coverage,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "canopy height estimation")
    except Exception as e:
        logger.exception("Unexpected error in estimate_canopy_height")
        return _format_error_result(e, "canopy height estimation")


@mcp.tool()
async def analyze_with_vision_language_model(
    image_path: str,
    task: str = "caption",
    query: Optional[str] = None,
    detect_target: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Analyze imagery using vision-language models (Moondream).

    Use natural language to caption, query, detect, or point at objects in imagery.
    Ideal for understanding image content without predefined categories.

    Args:
        image_path: Path to input image.
        task: Analysis task: "caption", "query", "detect", or "point".
        query: Question about the image (required for "query" task).
        detect_target: Object description (required for "detect" and "point" tasks).
        output_filename: Output filename for detection/point results (optional).

    Returns:
        Dictionary with analysis results (caption, answer, or detections).
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        valid_tasks = ("caption", "query", "detect", "point")
        if task not in valid_tasks:
            raise InputValidationError(
                message=f"Invalid VLM task '{task}'. Must be one of: {', '.join(valid_tasks)}",
                parameter_name="task",
                received_value=task,
                expected=", ".join(valid_tasks),
            )

        if task == "query" and not query:
            raise InputValidationError(
                message="query is required for 'query' task",
                parameter_name="query",
                expected="A non-empty question string",
            )
        if task in ("detect", "point") and not detect_target:
            raise InputValidationError(
                message=f"detect_target is required for '{task}' task",
                parameter_name="detect_target",
                expected="A non-empty object description string",
            )

        full_input_path = get_safe_input_path(image_path, cfg)

        logger.info(f"Analyzing {full_input_path} with VLM, task: {task}")

        moondream_module = _get_geoai_module("moondream")

        caption: Optional[str] = None
        answer: Optional[str] = None
        detections: list[dict[str, Any]] = []
        output_files: list[str] = []

        if task == "caption":
            caption = await _run_blocking(
                moondream_module.moondream_caption,
                source=str(full_input_path),
            )

        elif task == "query":
            answer = await _run_blocking(
                moondream_module.moondream_query,
                question=query,
                source=str(full_input_path),
            )

        elif task == "detect":
            out_name = output_filename or generate_output_filename(
                image_path, "vlm_detections", "geojson"
            )
            output_path = validate_output_path(out_name, cfg)

            raw_result = await _run_blocking(
                moondream_module.moondream_detect,
                source=str(full_input_path),
                object_type=detect_target,
                output_path=str(output_path),
            )
            if isinstance(raw_result, dict):
                objects = raw_result.get("objects", [])
                detections = objects if isinstance(objects, list) else []
            elif isinstance(raw_result, list):
                detections = raw_result
            if output_path.exists():
                output_files = [str(output_path)]

        elif task == "point":
            out_name = output_filename or generate_output_filename(
                image_path, "vlm_points", "geojson"
            )
            output_path = validate_output_path(out_name, cfg)

            raw_result = await _run_blocking(
                moondream_module.moondream_point,
                source=str(full_input_path),
                object_description=detect_target,
                output_path=str(output_path),
            )
            if isinstance(raw_result, dict):
                points = raw_result.get("points", [])
                detections = points if isinstance(points, list) else []
            elif isinstance(raw_result, list):
                detections = raw_result
            if output_path.exists():
                output_files = [str(output_path)]

        processing_time = time.time() - start_time

        if caption is not None:
            message = f"Caption: {caption}"
        elif answer is not None:
            message = f"Answer: {answer}"
        else:
            message = f"Found {len(detections)} instances of '{detect_target}'"

        return VLMResult(
            success=True,
            message=message,
            output_files=output_files,
            processing_time_seconds=processing_time,
            task=task,
            caption=caption,
            answer=answer,
            detections=detections,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "VLM analysis")
    except Exception as e:
        logger.exception("Unexpected error in analyze_with_vision_language_model")
        return _format_error_result(e, "VLM analysis")


# =============================================================================
# UTILITY & INSPECTION TOOLS
# =============================================================================


@mcp.tool()
async def clean_segmentation_results(
    input_path: str,
    operation: str = "all",
    min_size: int = 100,
    regularize_buildings: bool = False,
    output_filename: Optional[str] = None,
    source: str = "input",
) -> dict[str, Any]:
    """Clean and post-process segmentation/detection vector results.

    Removes small noise polygons, fixes invalid geometries, smooths boundaries,
    and optionally regularizes building footprints into clean shapes.

    Args:
        input_path: Path to segmentation result vector file.
        operation: Operation: "remove_noise", "fill_holes", "regularize",
                  "smooth", "all".
        min_size: Minimum polygon area (in square meters when CRS is geographic
                 or CRS units when projected).
        regularize_buildings: Whether to apply building footprint regularization.
        output_filename: Custom output filename (optional).
        source: Directory to read input_path from: "input" or "output" (default "input").

    Returns:
        Dictionary with cleaning statistics and output path.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        valid_ops = ("remove_noise", "fill_holes", "regularize", "smooth", "all")
        if operation not in valid_ops:
            raise InputValidationError(
                message=f"Invalid operation '{operation}'. Expected one of: {', '.join(valid_ops)}",
                parameter_name="operation",
                received_value=operation,
                expected=", ".join(valid_ops),
            )

        full_input_path = get_safe_file_path(input_path, source=source, config=cfg)

        out_ext = full_input_path.suffix.lstrip(".") or "geojson"
        out_name = output_filename or generate_output_filename(
            input_path, "cleaned", out_ext
        )
        output_path = validate_output_path(out_name, cfg)

        logger.info(f"Cleaning segmentation results: {full_input_path}")

        def _run_clean():
            import geopandas as gpd

            try:
                gdf = gpd.read_file(str(full_input_path))
            except Exception as exc:
                raise ProcessingError(
                    message=f"Failed to read vector file '{input_path}': {exc}",
                    stage="read",
                ) from exc

            original_count = len(gdf)
            result_gdf = gdf.copy()
            objects_removed = 0
            objects_modified = 0

            # Project geographic CRS to a metric UTM CRS before area/distance operations
            original_crs = result_gdf.crs
            reprojected = False
            if (
                original_crs is not None
                and original_crs.is_geographic
                and len(result_gdf) > 0
            ):
                metric_crs = result_gdf.estimate_utm_crs()
                if metric_crs is not None:
                    result_gdf = result_gdf.to_crs(metric_crs)
                    reprojected = True

            if operation in ("all", "remove_noise") and len(result_gdf) > 0:
                result_gdf = result_gdf[result_gdf.geometry.area >= min_size].copy()
                objects_removed = original_count - len(result_gdf)

            if operation in ("all", "fill_holes") and len(result_gdf) > 0:
                result_gdf["geometry"] = result_gdf.geometry.buffer(0)

            if operation in ("all", "smooth") and len(result_gdf) > 0:
                result_gdf["geometry"] = result_gdf.geometry.simplify(1.0)
                objects_modified += len(result_gdf)

            saved_by_regularize = False
            if regularize_buildings or operation == "regularize":
                utils_module = _get_geoai_module("utils")
                if reprojected and original_crs is not None:
                    result_gdf = result_gdf.to_crs(original_crs)
                    reprojected = False
                reg_gdf = utils_module.regularize(
                    result_gdf,
                    output_path=str(output_path),
                )
                if reg_gdf is not None:
                    result_gdf = reg_gdf
                    objects_modified += len(result_gdf)
                saved_by_regularize = output_path.exists()

            if reprojected and original_crs is not None:
                result_gdf = result_gdf.to_crs(original_crs)

            if not saved_by_regularize:
                result_gdf.to_file(str(output_path))

            return original_count, len(result_gdf), objects_removed, objects_modified

        original_count, final_count, objects_removed, objects_modified = (
            await _run_blocking(_run_clean)
        )

        processing_time = time.time() - start_time

        return CleanResultsResult(
            success=True,
            message=f"Cleaned results: {objects_removed} removed, {objects_modified} modified",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            objects_removed=objects_removed,
            objects_modified=objects_modified,
            original_count=original_count,
            final_count=final_count,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "result cleaning")
    except Exception as e:
        logger.exception("Unexpected error in clean_segmentation_results")
        return _format_error_result(e, "result cleaning")


@mcp.tool()
async def list_available_files(
    directory: str = "input",
    pattern: str = "*",
    include_metadata: bool = True,
    max_files: int = 500,
) -> dict[str, Any]:
    """List available files in input or output directories.

    Shows files available for processing or previously generated results.
    Useful for discovering available data before running analysis.

    Args:
        directory: Which directory: "input" or "output".
        pattern: Glob pattern to filter files (e.g., "*.tif", "*.geojson").
        include_metadata: Whether to include file size and modification time.
        max_files: Maximum number of files to return (1-500).

    Returns:
        Dictionary with list of files and their metadata.
    """
    cfg = _get_config()

    try:
        if directory == "input":
            base_dir = cfg.input_dir
        elif directory == "output":
            base_dir = cfg.output_dir
        else:
            raise InputValidationError(
                f"Invalid directory: {directory}. Use 'input' or 'output'."
            )

        capped_max = max(1, min(int(max_files), 500))
        files = list_input_files(base_dir, pattern, max_files=capped_max)

        file_list = []
        total_size_mb = 0.0

        for f in files:
            file_info: dict[str, Any] = {
                "name": f.name,
                "path": str(f.resolve().relative_to(base_dir.resolve())),
            }

            if include_metadata:
                stat = f.stat()
                size_mb = stat.st_size / (1024 * 1024)
                total_size_mb += size_mb
                file_info["size_mb"] = round(size_mb, 2)
                file_info["modified"] = stat.st_mtime

            file_list.append(file_info)

        return ListFilesResult(
            success=True,
            message=f"Found {len(file_list)} files in {directory} directory",
            output_files=[],
            files=file_list,
            total_count=len(file_list),
            total_size_mb=round(total_size_mb, 2),
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "file listing")
    except Exception as e:
        logger.exception("Unexpected error in list_available_files")
        return _format_error_result(e, "file listing")


@mcp.tool()
async def inspect_raster(
    image_path: str,
    source: str = "input",
    include_stats: bool = False,
) -> dict[str, Any]:
    """Inspect metadata and optional band statistics for a raster file (read-only).

    Args:
        image_path: Path to the raster file relative to the selected workspace directory.
        source: Directory to read from: "input" or "output" (default "input").
        include_stats: Whether to compute per-band min/max/mean/std statistics.

    Returns:
        Dictionary with raster metadata and optional band statistics.
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        full_path = get_safe_file_path(image_path, source=source, config=cfg)

        def _read_raster_meta():
            import numpy as np
            import rasterio

            with rasterio.open(full_path) as src:
                metadata = {
                    "path": str(full_path),
                    "driver": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtypes": [str(d) for d in src.dtypes],
                    "crs": str(src.crs) if src.crs else None,
                    "bounds": {
                        "left": _sanitize_float(src.bounds.left),
                        "bottom": _sanitize_float(src.bounds.bottom),
                        "right": _sanitize_float(src.bounds.right),
                        "top": _sanitize_float(src.bounds.top),
                    },
                    "resolution": {
                        "x": _sanitize_float(src.res[0]),
                        "y": _sanitize_float(src.res[1]),
                    },
                    "nodata": _sanitize_float(src.nodata),
                    "file_size_mb": round(full_path.stat().st_size / (1024 * 1024), 4),
                }

                band_stats: list[dict[str, Any]] = []
                if include_stats:
                    nodata = src.nodata
                    for b in range(1, src.count + 1):
                        arr = src.read(b)
                        mask = np.isfinite(arr)
                        if nodata is not None and not (
                            isinstance(nodata, float) and math.isnan(nodata)
                        ):
                            mask &= arr != nodata
                        valid = arr[mask]
                        band_stats.append(
                            {
                                "band": b,
                                "min": (
                                    _sanitize_float(np.min(valid))
                                    if valid.size > 0
                                    else None
                                ),
                                "max": (
                                    _sanitize_float(np.max(valid))
                                    if valid.size > 0
                                    else None
                                ),
                                "mean": (
                                    _sanitize_float(np.mean(valid))
                                    if valid.size > 0
                                    else None
                                ),
                                "std": (
                                    _sanitize_float(np.std(valid))
                                    if valid.size > 0
                                    else None
                                ),
                                "valid_pixels": int(valid.size),
                                "total_pixels": int(arr.size),
                            }
                        )
                return metadata, band_stats

        metadata, band_stats = await _run_blocking(_read_raster_meta)
        processing_time = time.time() - start_time

        return InspectRasterResult(
            success=True,
            message=(
                f"Inspected raster {full_path.name}: "
                f"{metadata['width']}x{metadata['height']}, {metadata['bands']} band(s)"
            ),
            output_files=[],
            processing_time_seconds=processing_time,
            metadata=metadata,
            band_stats=band_stats,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "raster inspection")
    except Exception as e:
        logger.exception("Unexpected error in inspect_raster")
        return _format_error_result(e, "raster inspection")


@mcp.tool()
async def inspect_vector(
    vector_path: str,
    source: str = "input",
) -> dict[str, Any]:
    """Inspect metadata for a vector file (read-only).

    Args:
        vector_path: Path to the vector file relative to the selected workspace directory.
        source: Directory to read from: "input" or "output" (default "input").

    Returns:
        Dictionary with vector metadata (feature count, CRS, geometry types, bounds, columns).
    """
    start_time = time.time()
    cfg = _get_config()

    try:
        full_path = get_safe_file_path(vector_path, source=source, config=cfg)

        def _read_vector_meta():
            import geopandas as gpd

            gdf = gpd.read_file(str(full_path))
            bounds = gdf.total_bounds if len(gdf) > 0 else [None, None, None, None]
            return {
                "path": str(full_path),
                "feature_count": int(len(gdf)),
                "crs": str(gdf.crs) if gdf.crs else None,
                "geometry_types": (
                    [str(t) for t in gdf.geometry.geom_type.dropna().unique()]
                    if len(gdf) > 0
                    else []
                ),
                "bounds": {
                    "minx": _sanitize_float(bounds[0]),
                    "miny": _sanitize_float(bounds[1]),
                    "maxx": _sanitize_float(bounds[2]),
                    "maxy": _sanitize_float(bounds[3]),
                },
                "columns": [str(c) for c in gdf.columns if c != "geometry"],
                "file_size_mb": round(full_path.stat().st_size / (1024 * 1024), 4),
            }

        metadata = await _run_blocking(_read_vector_meta)
        processing_time = time.time() - start_time

        return InspectVectorResult(
            success=True,
            message=(
                f"Inspected vector {full_path.name}: "
                f"{metadata['feature_count']} feature(s)"
            ),
            output_files=[],
            processing_time_seconds=processing_time,
            metadata=metadata,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "vector inspection")
    except Exception as e:
        logger.exception("Unexpected error in inspect_vector")
        return _format_error_result(e, "vector inspection")


@mcp.tool()
async def search_stac_imagery(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    collection: str = "sentinel-2-l2a",
    date_range: Optional[str] = None,
    max_cloud_cover: Optional[float] = 20.0,
    max_items: int = 10,
) -> dict[str, Any]:
    """Search for satellite imagery via the Planetary Computer STAC API (read-only).

    Discovers available scenes and asset metadata without downloading files.

    Args:
        min_lon: Minimum longitude (west boundary).
        min_lat: Minimum latitude (south boundary).
        max_lon: Maximum longitude (east boundary).
        max_lat: Maximum latitude (north boundary).
        collection: STAC collection ID (e.g., "sentinel-2-l2a", "landsat-c2-l2", "naip").
        date_range: Date range as "YYYY-MM-DD/YYYY-MM-DD" (optional).
        max_cloud_cover: Maximum cloud cover percentage (0-100, optional).
        max_items: Maximum number of STAC items to return (1-100).

    Returns:
        Dictionary with matching STAC item metadata.
    """
    start_time = time.time()

    try:
        bbox = BoundingBox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        )
        if not 1 <= max_items <= 100:
            raise InputValidationError(
                message=f"max_items must be between 1 and 100, got {max_items}",
                parameter_name="max_items",
                received_value=max_items,
                expected="An integer between 1 and 100",
            )
        if max_cloud_cover is not None and not 0 <= max_cloud_cover <= 100:
            raise InputValidationError(
                message=f"max_cloud_cover must be between 0 and 100, got {max_cloud_cover}",
                parameter_name="max_cloud_cover",
                received_value=max_cloud_cover,
                expected="A number between 0 and 100",
            )

        query = (
            {"eo:cloud_cover": {"lt": max_cloud_cover}}
            if max_cloud_cover is not None
            else None
        )

        download_module = _get_geoai_module("download")
        raw_items = await _run_blocking(
            download_module.pc_stac_search,
            collection=collection,
            bbox=list(bbox.to_tuple()),
            time_range=date_range,
            query=query,
            max_items=max_items,
            quiet=True,
        )

        items_summary: list[dict[str, Any]] = []
        for item in raw_items or []:
            entry: dict[str, Any] = {
                "id": getattr(item, "id", str(item)),
                "datetime": str(getattr(item, "datetime", "") or ""),
            }
            item_bbox = getattr(item, "bbox", None)
            if item_bbox is not None:
                entry["bbox"] = [_sanitize_float(v) for v in item_bbox]
            props = getattr(item, "properties", None)
            if isinstance(props, dict):
                entry["cloud_cover"] = _sanitize_float(props.get("eo:cloud_cover"))
            assets = getattr(item, "assets", None)
            if isinstance(assets, dict):
                entry["assets"] = list(assets.keys())[:20]
            items_summary.append(entry)

        processing_time = time.time() - start_time

        return StacSearchResult(
            success=True,
            message=f"Found {len(items_summary)} STAC items in {collection}",
            output_files=[],
            processing_time_seconds=processing_time,
            collection=collection,
            total_items=len(items_summary),
            items=items_summary,
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "STAC search")
    except Exception as e:
        logger.exception("Unexpected error in search_stac_imagery")
        return _format_error_result(e, "STAC search")


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================


def main():
    """Run the GeoAI MCP Server."""
    global config
    config = load_config()
    setup_logging(config)

    logger.info("Starting GeoAI MCP Server...")
    logger.info(f"Input directory: {config.input_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Log level: {config.log_level}")

    # Run with stdio transport (default for Claude Desktop)
    mcp.run()


if __name__ == "__main__":
    main()
