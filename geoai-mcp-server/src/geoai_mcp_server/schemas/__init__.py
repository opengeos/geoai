"""Pydantic schemas for GeoAI MCP Server tool inputs and outputs."""

from .tool_schemas import (
    # Common schemas
    BoundingBox,
    OutputFormat,
    DataSource,
    ModelType,
    # Segmentation schemas
    SegmentObjectsInput,
    AutoSegmentInput,
    SegmentationResult,
    # Detection schemas
    DetectFeaturesInput,
    ClassifyLandCoverInput,
    DetectionResult,
    ClassificationResult,
    # Change detection schemas
    ChangeDetectionInput,
    ChangeDetectionResult,
    # Data download schemas
    DownloadImageryInput,
    PrepareTrainingDataInput,
    DownloadResult,
    TrainingDataResult,
    # Foundation model schemas
    ExtractFeaturesInput,
    CanopyHeightInput,
    VLMAnalysisInput,
    FeatureExtractionResult,
    CanopyHeightResult,
    VLMResult,
    # Utility schemas
    CleanResultsInput,
    CleanResultsResult,
    ListFilesResult,
)

__all__ = [
    "BoundingBox",
    "OutputFormat",
    "DataSource",
    "ModelType",
    "SegmentObjectsInput",
    "AutoSegmentInput",
    "SegmentationResult",
    "DetectFeaturesInput",
    "ClassifyLandCoverInput",
    "DetectionResult",
    "ClassificationResult",
    "ChangeDetectionInput",
    "ChangeDetectionResult",
    "DownloadImageryInput",
    "PrepareTrainingDataInput",
    "DownloadResult",
    "TrainingDataResult",
    "ExtractFeaturesInput",
    "CanopyHeightInput",
    "VLMAnalysisInput",
    "FeatureExtractionResult",
    "CanopyHeightResult",
    "VLMResult",
    "CleanResultsInput",
    "CleanResultsResult",
    "ListFilesResult",
]
