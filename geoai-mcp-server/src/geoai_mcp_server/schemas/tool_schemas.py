"""Pydantic schemas for GeoAI MCP Server tool inputs and outputs.

These schemas define the structure of inputs and outputs for all tools,
enabling automatic validation and documentation generation.
"""

from enum import Enum
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field

# =============================================================================
# Common Enums and Types
# =============================================================================


class OutputFormat(str, Enum):
    """Supported output formats for geospatial data."""

    GEOTIFF = "geotiff"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    GEOPACKAGE = "geopackage"


class DataSource(str, Enum):
    """Supported satellite/aerial imagery data sources."""

    NAIP = "naip"
    SENTINEL2 = "sentinel2"
    LANDSAT = "landsat"
    PLANETARY_COMPUTER = "planetary_computer"


class SegmentationModel(str, Enum):
    """Available segmentation model types."""

    SAM = "sam"
    SAM2 = "sam2"
    GROUNDED_SAM = "grounded_sam"
    CLIPSEG = "clipseg"
    AUTO = "auto"


class DetectionTarget(str, Enum):
    """Types of features that can be detected."""

    BUILDINGS = "buildings"
    VEHICLES = "vehicles"
    SHIPS = "ships"
    SOLAR_PANELS = "solar_panels"
    TREES = "trees"
    WATER = "water"
    ROADS = "roads"
    CUSTOM = "custom"


class FoundationModel(str, Enum):
    """Available foundation models for feature extraction."""

    DINOV3 = "dinov3"
    PRITHVI = "prithvi"


class ModelType(str, Enum):
    """General model type categories."""

    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    CHANGE_DETECTION = "change_detection"


# =============================================================================
# Common Base Schemas
# =============================================================================


class BoundingBox(BaseModel):
    """Geographic bounding box in WGS84 coordinates."""

    min_lon: float = Field(..., description="Minimum longitude (west)", ge=-180, le=180)
    min_lat: float = Field(..., description="Minimum latitude (south)", ge=-90, le=90)
    max_lon: float = Field(..., description="Maximum longitude (east)", ge=-180, le=180)
    max_lat: float = Field(..., description="Maximum latitude (north)", ge=-90, le=90)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple format."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


class BaseResult(BaseModel):
    """Base schema for all tool results."""

    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Human-readable summary of the result")
    output_files: List[str] = Field(
        default_factory=list, description="List of generated output files"
    )
    processing_time_seconds: Optional[float] = Field(
        None, description="Time taken to process"
    )


# =============================================================================
# Segmentation Schemas
# =============================================================================


class SegmentObjectsInput(BaseModel):
    """Input schema for text-prompted object segmentation."""

    image_path: str = Field(
        ...,
        description="Path to the input image file (relative to input directory). "
        "Supports GeoTIFF, PNG, JPEG formats.",
    )
    prompts: List[str] = Field(
        ...,
        description="Text descriptions of objects to segment (e.g., ['building', 'road', 'tree']). "
        "The model will find and segment all instances matching these descriptions.",
        min_length=1,
        max_length=20,
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOJSON,
        description="Format for output data. GeoJSON recommended for vector output, "
        "GeoTIFF for raster masks.",
    )
    model: SegmentationModel = Field(
        default=SegmentationModel.AUTO,
        description="Segmentation model to use. 'auto' selects the best model based on the task.",
    )
    confidence_threshold: float = Field(
        default=0.3,
        description="Minimum confidence score for detections (0-1). Lower values find more objects "
        "but may include false positives.",
        ge=0,
        le=1,
    )
    tile_size: int = Field(
        default=1024,
        description="Size of tiles for processing large images. Larger tiles use more memory but "
        "may produce better results.",
        ge=256,
        le=4096,
    )
    output_filename: Optional[str] = Field(
        None,
        description="Custom output filename. If not provided, a name will be generated.",
    )


class AutoSegmentInput(BaseModel):
    """Input schema for automatic (promptless) segmentation."""

    image_path: str = Field(
        ..., description="Path to the input image file (relative to input directory)."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOTIFF, description="Format for output data."
    )
    min_object_size: int = Field(
        default=100,
        description="Minimum object size in pixels. Smaller objects will be filtered out.",
        ge=0,
    )
    max_object_size: Optional[int] = Field(
        None,
        description="Maximum object size in pixels. Larger objects will be filtered out.",
    )
    clean_results: bool = Field(
        default=True,
        description="Whether to apply post-processing to clean up segmentation results.",
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class SegmentationResult(BaseResult):
    """Result schema for segmentation operations."""

    num_objects: int = Field(0, description="Number of objects segmented")
    classes_found: List[str] = Field(
        default_factory=list, description="List of object classes found"
    )
    coverage_percent: Optional[float] = Field(
        None, description="Percentage of image covered by segmentation"
    )
    statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Additional statistics"
    )


# =============================================================================
# Detection and Classification Schemas
# =============================================================================


class DetectFeaturesInput(BaseModel):
    """Input schema for object detection."""

    image_path: str = Field(..., description="Path to the input image file.")
    feature_types: List[DetectionTarget] = Field(
        ...,
        description="Types of features to detect (e.g., ['buildings', 'vehicles']).",
        min_length=1,
    )
    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence score for detections.", ge=0, le=1
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOJSON, description="Output format for detection results."
    )
    custom_prompts: Optional[List[str]] = Field(
        None,
        description="Custom text prompts for detection (when feature_type is 'custom').",
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class ClassifyLandCoverInput(BaseModel):
    """Input schema for land cover classification."""

    image_path: str = Field(..., description="Path to the input image file.")
    model_path: Optional[str] = Field(
        None,
        description="Path to a trained classification model. If not provided, uses default model.",
    )
    num_classes: int = Field(
        default=10, description="Number of land cover classes.", ge=2, le=100
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOTIFF, description="Output format for classified image."
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class DetectionResult(BaseResult):
    """Result schema for detection operations."""

    num_detections: int = Field(0, description="Total number of detections")
    detections_by_class: Dict[str, int] = Field(
        default_factory=dict, description="Number of detections per class"
    )
    average_confidence: Optional[float] = Field(
        None, description="Average confidence score"
    )


class ClassificationResult(BaseResult):
    """Result schema for classification operations."""

    class_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of classes (class name -> percentage)",
    )
    dominant_class: Optional[str] = Field(None, description="Most common class")


# =============================================================================
# Change Detection Schemas
# =============================================================================


class ChangeDetectionInput(BaseModel):
    """Input schema for temporal change detection."""

    image1_path: str = Field(..., description="Path to the first (earlier) image.")
    image2_path: str = Field(..., description="Path to the second (later) image.")
    change_threshold: float = Field(
        default=0.5,
        description="Threshold for detecting changes (0-1). Higher values = stricter detection.",
        ge=0,
        le=1,
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOTIFF, description="Output format for change map."
    )
    include_statistics: bool = Field(
        default=True, description="Whether to include detailed change statistics."
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class ChangeDetectionResult(BaseResult):
    """Result schema for change detection."""

    change_percentage: float = Field(
        0, description="Percentage of area with detected changes"
    )
    change_area_sq_meters: Optional[float] = Field(
        None, description="Area of change in square meters"
    )
    change_types: Dict[str, float] = Field(
        default_factory=dict, description="Types of changes detected and their areas"
    )


# =============================================================================
# Data Download Schemas
# =============================================================================


class DownloadImageryInput(BaseModel):
    """Input schema for downloading satellite/aerial imagery."""

    bbox: BoundingBox = Field(
        ..., description="Geographic bounding box for the area of interest."
    )
    data_source: DataSource = Field(
        ..., description="Data source to download from (e.g., 'naip', 'sentinel2')."
    )
    date_range: Optional[str] = Field(
        None,
        description="Date range in format 'YYYY-MM-DD/YYYY-MM-DD'. If not specified, uses latest available.",
    )
    max_cloud_cover: float = Field(
        default=20,
        description="Maximum cloud cover percentage for satellite imagery.",
        ge=0,
        le=100,
    )
    max_items: int = Field(
        default=10, description="Maximum number of images to download.", ge=1, le=100
    )
    output_subdir: Optional[str] = Field(
        None, description="Subdirectory within output folder to save downloads."
    )


class PrepareTrainingDataInput(BaseModel):
    """Input schema for preparing training datasets."""

    images_dir: str = Field(..., description="Directory containing source images.")
    labels_dir: str = Field(..., description="Directory containing label masks.")
    tile_size: int = Field(
        default=256, description="Size of output tiles in pixels.", ge=64, le=2048
    )
    overlap: int = Field(
        default=32, description="Overlap between tiles in pixels.", ge=0
    )
    train_ratio: float = Field(
        default=0.7,
        description="Proportion of data for training (vs validation).",
        ge=0.1,
        le=0.95,
    )
    augment: bool = Field(
        default=True, description="Whether to apply data augmentation."
    )
    output_subdir: Optional[str] = Field(
        None, description="Subdirectory for training data output."
    )


class DownloadResult(BaseResult):
    """Result schema for download operations."""

    num_files: int = Field(0, description="Number of files downloaded")
    total_size_mb: float = Field(0, description="Total size of downloaded files in MB")
    coverage_percent: Optional[float] = Field(
        None, description="Percentage of requested area covered"
    )


class TrainingDataResult(BaseResult):
    """Result schema for training data preparation."""

    num_tiles: int = Field(0, description="Total number of tiles created")
    num_train: int = Field(0, description="Number of training tiles")
    num_val: int = Field(0, description="Number of validation tiles")
    tile_size: int = Field(0, description="Tile size in pixels")


# =============================================================================
# Foundation Model Schemas
# =============================================================================


class ExtractFeaturesInput(BaseModel):
    """Input schema for foundation model feature extraction."""

    image_path: str = Field(..., description="Path to the input image.")
    model: FoundationModel = Field(
        default=FoundationModel.DINOV3,
        description="Foundation model to use for feature extraction.",
    )
    output_type: str = Field(
        default="embeddings",
        description="Type of output: 'embeddings' for raw features, 'similarity_map' for similarity analysis.",
    )
    reference_point: Optional[tuple[int, int]] = Field(
        None, description="Reference point (x, y) for similarity map generation."
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class CanopyHeightInput(BaseModel):
    """Input schema for canopy height estimation."""

    image_path: str = Field(..., description="Path to the RGB input image.")
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOTIFF, description="Output format for height map."
    )
    include_statistics: bool = Field(
        default=True, description="Whether to include height statistics."
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class VLMAnalysisInput(BaseModel):
    """Input schema for vision-language model analysis."""

    image_path: str = Field(..., description="Path to the input image.")
    task: str = Field(
        default="caption",
        description="Analysis task: 'caption' for description, 'query' for Q&A, 'detect' for object detection.",
    )
    query: Optional[str] = Field(
        None, description="Question to ask about the image (required for 'query' task)."
    )
    detect_target: Optional[str] = Field(
        None, description="Object description to detect (required for 'detect' task)."
    )
    output_filename: Optional[str] = Field(
        None, description="Custom output filename for detection results."
    )


class FeatureExtractionResult(BaseResult):
    """Result schema for feature extraction."""

    feature_dimensions: tuple[int, ...] = Field(
        default_factory=tuple, description="Shape of extracted features"
    )
    model_used: str = Field("", description="Model that was used")


class CanopyHeightResult(BaseResult):
    """Result schema for canopy height estimation."""

    min_height: float = Field(0, description="Minimum canopy height in meters")
    max_height: float = Field(0, description="Maximum canopy height in meters")
    mean_height: float = Field(0, description="Mean canopy height in meters")
    forest_coverage_percent: float = Field(
        0, description="Percentage of area with detected vegetation"
    )


class VLMResult(BaseResult):
    """Result schema for VLM analysis."""

    task: str = Field("", description="Task that was performed")
    caption: Optional[str] = Field(
        None, description="Generated caption (for 'caption' task)"
    )
    answer: Optional[str] = Field(
        None, description="Answer to query (for 'query' task)"
    )
    detections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected objects (for 'detect' task)"
    )


# =============================================================================
# Utility Schemas
# =============================================================================


class CleanResultsInput(BaseModel):
    """Input schema for cleaning/post-processing results."""

    input_path: str = Field(
        ..., description="Path to the segmentation/detection result to clean."
    )
    operation: str = Field(
        default="all",
        description="Cleaning operation: 'remove_noise', 'fill_holes', 'regularize', 'smooth', 'all'.",
    )
    min_size: int = Field(
        default=100,
        description="Minimum object size to keep (smaller objects are removed).",
        ge=0,
    )
    regularize_buildings: bool = Field(
        default=False, description="Whether to apply building footprint regularization."
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename.")


class CleanResultsResult(BaseResult):
    """Result schema for cleaning operations."""

    objects_removed: int = Field(
        0, description="Number of objects removed during cleaning"
    )
    objects_modified: int = Field(0, description="Number of objects modified")
    original_count: int = Field(0, description="Original number of objects")
    final_count: int = Field(0, description="Final number of objects after cleaning")


class ListFilesResult(BaseResult):
    """Result schema for listing files."""

    files: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of files with metadata"
    )
    total_count: int = Field(0, description="Total number of files")
    total_size_mb: float = Field(0, description="Total size in megabytes")
