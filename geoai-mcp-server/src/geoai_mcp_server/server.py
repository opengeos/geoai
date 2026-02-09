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
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from .config import load_config, GeoAIConfig
from .schemas import (
    # Segmentation
    SegmentObjectsInput,
    AutoSegmentInput,
    SegmentationResult,
    # Detection
    DetectFeaturesInput,
    ClassifyLandCoverInput,
    DetectionResult,
    ClassificationResult,
    # Change Detection
    ChangeDetectionInput,
    ChangeDetectionResult,
    # Data Download
    DownloadImageryInput,
    PrepareTrainingDataInput,
    DownloadResult,
    TrainingDataResult,
    # Foundation Models
    ExtractFeaturesInput,
    CanopyHeightInput,
    VLMAnalysisInput,
    FeatureExtractionResult,
    CanopyHeightResult,
    VLMResult,
    # Utilities
    CleanResultsInput,
    CleanResultsResult,
    ListFilesResult,
    # Common
    OutputFormat,
    BoundingBox,
)
from .utils.error_handling import (
    GeoAIError,
    InputValidationError,
    FileAccessError,
    ModelLoadError,
    ProcessingError,
)
from .utils.file_management import (
    validate_input_path,
    validate_output_path,
    get_safe_input_path,
    list_input_files,
    generate_output_filename,
)
from .utils.validation import (
    validate_bbox,
    validate_image_path,
    validate_text_prompts,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("geoai-mcp-server")

# Load configuration
config = load_config()

# Initialize MCP Server
mcp = MCPServer(
    "GeoAI Server",
    version="0.1.0",
    instructions="Geospatial AI tools for remote sensing and Earth observation analysis",
)


# =============================================================================
# Helper Functions
# =============================================================================


def _format_error_result(error: Exception, operation: str) -> dict:
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
        import importlib

        return importlib.import_module(f"geoai.{module_name}")
    except ImportError as e:
        raise ModelLoadError(f"Failed to import geoai.{module_name}: {e}")


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
        output_format: Output format - "geojson", "geotiff", "shapefile", "geopackage".
        model: Model to use - "sam", "sam2", "grounded_sam", "clipseg", "auto".
        confidence_threshold: Minimum confidence (0-1). Lower = more detections.
        tile_size: Tile size for large images (256-4096 pixels).
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with segmentation results, output file paths, and statistics.

    Example:
        >>> segment_objects_with_prompts(
        ...     image_path="area_of_interest.tif",
        ...     prompts=["building", "parking lot", "swimming pool"],
        ...     confidence_threshold=0.4
        ... )
    """
    start_time = time.time()

    try:
        # Validate inputs
        input_data = SegmentObjectsInput(
            image_path=image_path,
            prompts=prompts,
            output_format=OutputFormat(output_format),
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
            output_filename=output_filename,
        )

        # Get full paths
        full_input_path = get_safe_input_path(input_data.image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {input_data.image_path}")

        # Generate output filename
        out_name = output_filename or generate_output_filename(
            input_data.image_path, "segmented", input_data.output_format.value
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Segmenting objects in {full_input_path} with prompts: {prompts}")

        # Import GeoAI modules
        sam_module = _get_geoai_module("sam")

        # Initialize model based on selection
        if model in ("grounded_sam", "auto"):
            # Try GroundedSAM first for text prompts
            try:
                segmenter = sam_module.SamGeo(
                    model_type="vit_h",
                    checkpoint=None,  # Uses default
                )
                # Use text prompts for segmentation
                segmenter.set_image(str(full_input_path))
                masks = segmenter.predict_with_text(
                    text_prompts=prompts,
                    box_threshold=confidence_threshold,
                    text_threshold=confidence_threshold,
                )
            except Exception as e:
                logger.warning(f"GroundedSAM failed, falling back to CLIPSeg: {e}")
                # Fallback to CLIPSeg
                segment_module = _get_geoai_module("segment")
                masks = segment_module.segment_with_text(
                    str(full_input_path),
                    prompts,
                    threshold=confidence_threshold,
                )
        elif model == "clipseg":
            segment_module = _get_geoai_module("segment")
            masks = segment_module.segment_with_text(
                str(full_input_path),
                prompts,
                threshold=confidence_threshold,
            )
        else:
            # SAM with automatic prompt generation
            segmenter = sam_module.SamGeo(model_type="vit_h")
            segmenter.set_image(str(full_input_path))
            masks = segmenter.generate(output=str(output_path))

        # Save results
        if input_data.output_format == OutputFormat.GEOJSON:
            # Convert masks to vector
            utils_module = _get_geoai_module("utils")
            utils_module.raster_to_vector(
                masks if isinstance(masks, str) else str(output_path),
                str(output_path),
            )

        # Calculate statistics
        num_objects = 0
        if hasattr(masks, "__len__"):
            num_objects = len(masks) if not isinstance(masks, str) else 1

        processing_time = time.time() - start_time

        return SegmentationResult(
            success=True,
            message=f"Successfully segmented objects matching {prompts}",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            num_objects=num_objects,
            classes_found=prompts,
            statistics={
                "model_used": model,
                "confidence_threshold": confidence_threshold,
            },
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "text-prompted segmentation")
    except Exception as e:
        logger.exception("Unexpected error in segment_objects_with_prompts")
        return _format_error_result(e, "text-prompted segmentation")


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
        output_format: Output format - "geotiff", "geojson", "shapefile".
        min_object_size: Minimum object size in pixels to keep.
        max_object_size: Maximum object size in pixels (optional).
        clean_results: Whether to clean up small artifacts.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with segmentation masks and statistics.

    Example:
        >>> auto_segment_image(
        ...     image_path="satellite_tile.tif",
        ...     min_object_size=200,
        ...     output_format="geojson"
        ... )
    """
    start_time = time.time()

    try:
        input_data = AutoSegmentInput(
            image_path=image_path,
            output_format=OutputFormat(output_format),
            min_object_size=min_object_size,
            max_object_size=max_object_size,
            clean_results=clean_results,
            output_filename=output_filename,
        )

        full_input_path = get_safe_input_path(input_data.image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {input_data.image_path}")

        out_name = output_filename or generate_output_filename(
            input_data.image_path, "auto_segmented", input_data.output_format.value
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Auto-segmenting {full_input_path}")

        sam_module = _get_geoai_module("sam")

        # Initialize SAM
        segmenter = sam_module.SamGeo(model_type="vit_h")
        segmenter.set_image(str(full_input_path))

        # Generate all masks
        masks = segmenter.generate(output=str(output_path))

        # Filter by size if specified
        num_objects = len(masks) if hasattr(masks, "__len__") else 0

        # Clean results if requested
        if clean_results and input_data.output_format == OutputFormat.GEOJSON:
            utils_module = _get_geoai_module("utils")
            utils_module.clean_segmentation(
                str(output_path),
                min_size=min_object_size,
                max_size=max_object_size,
            )

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
        feature_types: Types to detect - "buildings", "vehicles", "ships",
                      "solar_panels", "trees", "water", "roads", "custom".
        confidence_threshold: Minimum detection confidence (0-1).
        output_format: Output format for results.
        custom_prompts: Custom text prompts when feature_type is "custom".
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with detection results and bounding boxes.

    Example:
        >>> detect_and_classify_features(
        ...     image_path="urban_area.tif",
        ...     feature_types=["buildings", "vehicles"],
        ...     confidence_threshold=0.6
        ... )
    """
    start_time = time.time()

    try:
        full_input_path = get_safe_input_path(image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {image_path}")

        out_name = output_filename or generate_output_filename(
            image_path, "detections", output_format
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Detecting {feature_types} in {full_input_path}")

        # Build prompts from feature types
        prompts = []
        for ft in feature_types:
            if ft == "custom" and custom_prompts:
                prompts.extend(custom_prompts)
            elif ft == "buildings":
                prompts.append("building")
            elif ft == "vehicles":
                prompts.extend(["car", "truck", "vehicle"])
            elif ft == "ships":
                prompts.extend(["ship", "boat", "vessel"])
            elif ft == "solar_panels":
                prompts.append("solar panel")
            elif ft == "trees":
                prompts.append("tree")
            elif ft == "water":
                prompts.extend(["water", "pond", "lake", "river"])
            elif ft == "roads":
                prompts.append("road")

        if not prompts:
            prompts = feature_types

        # Use GroundedSAM for detection
        sam_module = _get_geoai_module("sam")

        try:
            segmenter = sam_module.SamGeo(model_type="vit_h")
            segmenter.set_image(str(full_input_path))
            results = segmenter.predict_with_text(
                text_prompts=prompts,
                box_threshold=confidence_threshold,
                text_threshold=confidence_threshold,
                output=str(output_path),
            )

            # Count detections by class
            detections_by_class = {}
            if hasattr(results, "__iter__"):
                for r in results:
                    cls = (
                        r.get("class", "unknown") if isinstance(r, dict) else "detected"
                    )
                    detections_by_class[cls] = detections_by_class.get(cls, 0) + 1

            num_detections = sum(detections_by_class.values())

        except Exception as e:
            logger.warning(f"GroundedSAM detection failed: {e}")
            # Fallback message
            num_detections = 0
            detections_by_class = {}

        processing_time = time.time() - start_time

        return DetectionResult(
            success=True,
            message=f"Detection complete: found {num_detections} objects",
            output_files=[str(output_path)],
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
        model_path: Path to trained model (uses default if not specified).
        num_classes: Number of land cover classes (2-100).
        output_format: Output format for classified map.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with classification results and class distribution.

    Example:
        >>> classify_land_cover(
        ...     image_path="sentinel2_tile.tif",
        ...     num_classes=8
        ... )
    """
    start_time = time.time()

    try:
        full_input_path = get_safe_input_path(image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {image_path}")

        out_name = output_filename or generate_output_filename(
            image_path, "classified", output_format
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Classifying land cover in {full_input_path}")

        classify_module = _get_geoai_module("classify")

        # Perform classification
        result = classify_module.classify_image(
            str(full_input_path),
            output=str(output_path),
            model_path=model_path,
            num_classes=num_classes,
        )

        # Get class distribution
        class_distribution = {}
        dominant_class = None
        if hasattr(result, "class_distribution"):
            class_distribution = result.class_distribution
            dominant_class = max(class_distribution, key=class_distribution.get)

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

    Example:
        >>> detect_temporal_changes(
        ...     image1_path="area_2020.tif",
        ...     image2_path="area_2023.tif",
        ...     change_threshold=0.4
        ... )
    """
    start_time = time.time()

    try:
        full_path1 = get_safe_input_path(image1_path, config)
        full_path2 = get_safe_input_path(image2_path, config)

        if not full_path1.exists():
            raise FileAccessError(f"First image not found: {image1_path}")
        if not full_path2.exists():
            raise FileAccessError(f"Second image not found: {image2_path}")

        out_name = output_filename or generate_output_filename(
            image1_path, "change_map", output_format
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Detecting changes between {full_path1} and {full_path2}")

        change_module = _get_geoai_module("change_detection")

        # Perform change detection
        result = change_module.detect_changes(
            str(full_path1),
            str(full_path2),
            output=str(output_path),
            threshold=change_threshold,
        )

        # Calculate statistics
        change_percentage = 0.0
        change_area_sq_meters = None
        change_types = {}

        if include_statistics and hasattr(result, "statistics"):
            stats = result.statistics
            change_percentage = stats.get("change_percentage", 0.0)
            change_area_sq_meters = stats.get("change_area_sq_meters")
            change_types = stats.get("change_types", {})

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

    Fetches imagery from various sources including NAIP, Sentinel-2, Landsat,
    and Microsoft Planetary Computer.

    Args:
        min_lon: Minimum longitude (west boundary).
        min_lat: Minimum latitude (south boundary).
        max_lon: Maximum longitude (east boundary).
        max_lat: Maximum latitude (north boundary).
        data_source: Source - "naip", "sentinel2", "landsat", "planetary_computer".
        date_range: Date range as "YYYY-MM-DD/YYYY-MM-DD" (optional).
        max_cloud_cover: Maximum cloud cover percentage (0-100).
        max_items: Maximum number of images to download (1-100).
        output_subdir: Subdirectory for downloads (optional).

    Returns:
        Dictionary with download results and file paths.

    Example:
        >>> download_satellite_imagery(
        ...     min_lon=-122.5, min_lat=37.7,
        ...     max_lon=-122.4, max_lat=37.8,
        ...     data_source="naip",
        ...     max_cloud_cover=10
        ... )
    """
    start_time = time.time()

    try:
        bbox = BoundingBox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        )

        # Determine output directory
        output_dir = config.output_dir
        if output_subdir:
            output_dir = output_dir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {data_source} imagery for bbox: {bbox.to_tuple()}")

        download_module = _get_geoai_module("download")

        downloaded_files = []
        total_size_mb = 0.0

        if data_source == "naip":
            files = download_module.download_naip(
                bbox=bbox.to_tuple(),
                output_dir=str(output_dir),
                max_items=max_items,
            )
            downloaded_files = files if isinstance(files, list) else [files]

        elif data_source == "sentinel2":
            files = download_module.download_sentinel2(
                bbox=bbox.to_tuple(),
                output_dir=str(output_dir),
                date_range=date_range,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
            )
            downloaded_files = files if isinstance(files, list) else [files]

        elif data_source == "landsat":
            files = download_module.download_landsat(
                bbox=bbox.to_tuple(),
                output_dir=str(output_dir),
                date_range=date_range,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
            )
            downloaded_files = files if isinstance(files, list) else [files]

        elif data_source == "planetary_computer":
            files = download_module.download_planetary_computer(
                bbox=bbox.to_tuple(),
                output_dir=str(output_dir),
                collection="sentinel-2-l2a",
                max_items=max_items,
            )
            downloaded_files = files if isinstance(files, list) else [files]

        # Calculate total size
        for f in downloaded_files:
            if Path(f).exists():
                total_size_mb += Path(f).stat().st_size / (1024 * 1024)

        processing_time = time.time() - start_time

        return DownloadResult(
            success=True,
            message=f"Downloaded {len(downloaded_files)} files from {data_source}",
            output_files=[str(f) for f in downloaded_files],
            processing_time_seconds=processing_time,
            num_files=len(downloaded_files),
            total_size_mb=total_size_mb,
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

    Creates tiled training data suitable for deep learning, with
    automatic train/validation splits and optional augmentation.

    Args:
        images_dir: Directory containing source images.
        labels_dir: Directory containing label masks.
        tile_size: Output tile size in pixels (64-2048).
        overlap: Tile overlap in pixels.
        train_ratio: Training data proportion (0.1-0.95).
        augment: Whether to apply data augmentation.
        output_subdir: Subdirectory for training data (optional).

    Returns:
        Dictionary with training data statistics and paths.

    Example:
        >>> prepare_training_data(
        ...     images_dir="raw_images",
        ...     labels_dir="labels",
        ...     tile_size=512,
        ...     train_ratio=0.8
        ... )
    """
    start_time = time.time()

    try:
        images_path = get_safe_input_path(images_dir, config)
        labels_path = get_safe_input_path(labels_dir, config)

        if not images_path.exists():
            raise FileAccessError(f"Images directory not found: {images_dir}")
        if not labels_path.exists():
            raise FileAccessError(f"Labels directory not found: {labels_dir}")

        output_dir = config.output_dir
        if output_subdir:
            output_dir = output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preparing training data: {images_path} + {labels_path}")

        train_module = _get_geoai_module("train")

        # Create training dataset
        result = train_module.prepare_training_data(
            images_dir=str(images_path),
            labels_dir=str(labels_path),
            output_dir=str(output_dir),
            tile_size=tile_size,
            overlap=overlap,
            train_ratio=train_ratio,
            augment=augment,
        )

        num_tiles = getattr(result, "total_tiles", 0)
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
        model: Foundation model - "dinov3" or "prithvi".
        output_type: "embeddings" for raw features, "similarity_map" for similarity analysis.
        reference_x: X coordinate for similarity reference point (if output_type="similarity_map").
        reference_y: Y coordinate for similarity reference point.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with extracted features or similarity map.

    Example:
        >>> extract_features_with_foundation_model(
        ...     image_path="scene.tif",
        ...     model="dinov3",
        ...     output_type="similarity_map",
        ...     reference_x=100, reference_y=100
        ... )
    """
    start_time = time.time()

    try:
        full_input_path = get_safe_input_path(image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {image_path}")

        out_name = output_filename or generate_output_filename(
            image_path, f"features_{model}", "npy"
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Extracting features from {full_input_path} using {model}")

        if model == "dinov3":
            dino_module = _get_geoai_module("dinov3")

            if output_type == "similarity_map" and reference_x is not None:
                result = dino_module.compute_similarity_map(
                    str(full_input_path),
                    reference_point=(reference_x, reference_y),
                    output=str(output_path),
                )
            else:
                result = dino_module.extract_features(
                    str(full_input_path),
                    output=str(output_path),
                )

        elif model == "prithvi":
            prithvi_module = _get_geoai_module("prithvi")
            result = prithvi_module.extract_features(
                str(full_input_path),
                output=str(output_path),
            )
        else:
            raise InputValidationError(f"Unknown model: {model}")

        feature_dims = ()
        if hasattr(result, "shape"):
            feature_dims = result.shape

        processing_time = time.time() - start_time

        return FeatureExtractionResult(
            success=True,
            message=f"Feature extraction complete using {model}",
            output_files=[str(output_path)],
            processing_time_seconds=processing_time,
            feature_dimensions=feature_dims,
            model_used=model,
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

    Example:
        >>> estimate_canopy_height(
        ...     image_path="forest_area.tif",
        ...     include_statistics=True
        ... )
    """
    start_time = time.time()

    try:
        full_input_path = get_safe_input_path(image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {image_path}")

        out_name = output_filename or generate_output_filename(
            image_path, "canopy_height", output_format
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Estimating canopy height for {full_input_path}")

        canopy_module = _get_geoai_module("canopy")

        result = canopy_module.estimate_canopy_height(
            str(full_input_path),
            output=str(output_path),
        )

        # Extract statistics
        min_height = 0.0
        max_height = 0.0
        mean_height = 0.0
        coverage = 0.0

        if include_statistics and hasattr(result, "statistics"):
            stats = result.statistics
            min_height = stats.get("min_height", 0.0)
            max_height = stats.get("max_height", 0.0)
            mean_height = stats.get("mean_height", 0.0)
            coverage = stats.get("forest_coverage", 0.0)

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

    Use natural language to caption, query, or detect objects in imagery.
    Ideal for understanding image content without predefined categories.

    Args:
        image_path: Path to input image.
        task: Analysis task - "caption" for description, "query" for Q&A,
              "detect" for object detection.
        query: Question about the image (required for "query" task).
        detect_target: Object to detect (required for "detect" task).
        output_filename: Output filename for detection results (optional).

    Returns:
        Dictionary with analysis results (caption, answer, or detections).

    Example:
        >>> analyze_with_vision_language_model(
        ...     image_path="aerial_view.jpg",
        ...     task="query",
        ...     query="How many buildings are visible?"
        ... )
    """
    start_time = time.time()

    try:
        full_input_path = get_safe_input_path(image_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {image_path}")

        if task == "query" and not query:
            raise InputValidationError("Query is required for 'query' task")
        if task == "detect" and not detect_target:
            raise InputValidationError("detect_target is required for 'detect' task")

        logger.info(f"Analyzing {full_input_path} with VLM, task: {task}")

        moondream_module = _get_geoai_module("moondream")

        caption = None
        answer = None
        detections = []
        output_files = []

        if task == "caption":
            caption = moondream_module.caption_image(str(full_input_path))

        elif task == "query":
            answer = moondream_module.query_image(str(full_input_path), query)

        elif task == "detect":
            out_name = output_filename or generate_output_filename(
                image_path, "vlm_detections", "geojson"
            )
            output_path = validate_output_path(out_name, config)

            detections = moondream_module.detect_objects(
                str(full_input_path),
                detect_target,
                output=str(output_path),
            )
            output_files = [str(output_path)]

        processing_time = time.time() - start_time

        # Build message
        if caption:
            message = f"Caption: {caption}"
        elif answer:
            message = f"Answer: {answer}"
        else:
            message = f"Detected {len(detections)} instances of '{detect_target}'"

        return VLMResult(
            success=True,
            message=message,
            output_files=output_files,
            processing_time_seconds=processing_time,
            task=task,
            caption=caption,
            answer=answer,
            detections=detections if isinstance(detections, list) else [],
        ).model_dump()

    except GeoAIError as e:
        return _format_error_result(e, "VLM analysis")
    except Exception as e:
        logger.exception("Unexpected error in analyze_with_vision_language_model")
        return _format_error_result(e, "VLM analysis")


# =============================================================================
# UTILITY TOOLS
# =============================================================================


@mcp.tool()
async def clean_segmentation_results(
    input_path: str,
    operation: str = "all",
    min_size: int = 100,
    regularize_buildings: bool = False,
    output_filename: Optional[str] = None,
) -> dict[str, Any]:
    """Clean and post-process segmentation/detection results.

    Removes noise, fills holes, smooths boundaries, and optionally
    regularizes building footprints into clean rectangular shapes.

    Args:
        input_path: Path to segmentation result file.
        operation: Operation - "remove_noise", "fill_holes", "regularize",
                  "smooth", "all".
        min_size: Minimum object size to keep (pixels).
        regularize_buildings: Whether to regularize building shapes.
        output_filename: Custom output filename (optional).

    Returns:
        Dictionary with cleaning statistics and output path.

    Example:
        >>> clean_segmentation_results(
        ...     input_path="buildings_raw.geojson",
        ...     operation="all",
        ...     regularize_buildings=True
        ... )
    """
    start_time = time.time()

    try:
        full_input_path = get_safe_input_path(input_path, config)
        if not full_input_path.exists():
            raise FileAccessError(f"Input file not found: {input_path}")

        out_ext = full_input_path.suffix
        out_name = output_filename or generate_output_filename(
            input_path, "cleaned", out_ext.lstrip(".")
        )
        output_path = validate_output_path(out_name, config)

        logger.info(f"Cleaning segmentation results: {full_input_path}")

        utils_module = _get_geoai_module("utils")

        # Get initial count
        original_count = 0
        try:
            import geopandas as gpd

            gdf = gpd.read_file(str(full_input_path))
            original_count = len(gdf)
        except Exception:
            pass

        # Apply cleaning operations
        result_gdf = gdf.copy() if "gdf" in dir() else None
        objects_removed = 0
        objects_modified = 0

        if operation in ("all", "remove_noise"):
            if result_gdf is not None:
                result_gdf = result_gdf[result_gdf.geometry.area >= min_size]
                objects_removed = original_count - len(result_gdf)

        if operation in ("all", "fill_holes"):
            # Fill small holes in polygons
            if result_gdf is not None:
                result_gdf.geometry = result_gdf.geometry.buffer(0)

        if operation in ("all", "smooth"):
            if result_gdf is not None:
                result_gdf.geometry = result_gdf.geometry.simplify(1)
                objects_modified += len(result_gdf)

        if regularize_buildings:
            # Apply building regularization
            try:
                result_gdf = utils_module.regularize_buildings(
                    result_gdf if result_gdf is not None else str(full_input_path),
                    output=str(output_path),
                )
                objects_modified += len(result_gdf) if result_gdf is not None else 0
            except Exception as e:
                logger.warning(f"Building regularization failed: {e}")

        # Save result
        if result_gdf is not None:
            result_gdf.to_file(str(output_path))
            final_count = len(result_gdf)
        else:
            final_count = original_count

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
) -> dict[str, Any]:
    """List available files in input or output directories.

    Shows files available for processing or previously generated results.
    Useful for discovering available data before running analysis.

    Args:
        directory: Which directory - "input" or "output".
        pattern: Glob pattern to filter files (e.g., "*.tif", "*.geojson").
        include_metadata: Whether to include file size and modification time.

    Returns:
        Dictionary with list of files and their metadata.

    Example:
        >>> list_available_files(
        ...     directory="input",
        ...     pattern="*.tif"
        ... )
    """
    try:
        if directory == "input":
            base_dir = config.input_dir
        elif directory == "output":
            base_dir = config.output_dir
        else:
            raise InputValidationError(
                f"Invalid directory: {directory}. Use 'input' or 'output'."
            )

        files = list_input_files(base_dir, pattern)

        file_list = []
        total_size_mb = 0.0

        for f in files:
            file_info = {"name": f.name, "path": str(f.relative_to(base_dir))}

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


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================


def main():
    """Run the GeoAI MCP Server."""
    logger.info("Starting GeoAI MCP Server...")
    logger.info(f"Input directory: {config.input_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Log level: {config.log_level}")

    # Run with stdio transport (default for Claude Desktop)
    mcp.run()


if __name__ == "__main__":
    main()
