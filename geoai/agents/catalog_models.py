"""Structured output models for catalog search results."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CatalogDatasetInfo(BaseModel):
    """Information about a catalog dataset."""

    id: str = Field(..., description="Dataset identifier")
    title: str = Field(..., description="Dataset title")
    type: Optional[str] = Field(
        None, description="Dataset type (e.g., image, image_collection, table)"
    )
    provider: Optional[str] = Field(None, description="Data provider")
    description: Optional[str] = Field(None, description="Dataset description")
    keywords: Optional[str] = Field(None, description="Keywords/tags")
    snippet: Optional[str] = Field(
        None, description="Code snippet to access the dataset"
    )
    start_date: Optional[str] = Field(None, description="Start date of coverage")
    end_date: Optional[str] = Field(None, description="End date of coverage")
    bbox: Optional[str] = Field(None, description="Bounding box")
    license: Optional[str] = Field(None, description="License information")
    url: Optional[str] = Field(None, description="Documentation URL")
    catalog: Optional[str] = Field(None, description="Catalog URL")
    deprecated: Optional[str] = Field(None, description="Deprecated status")


class CatalogSearchResult(BaseModel):
    """Container for catalog search results."""

    query: str = Field(..., description="Original search query")
    dataset_count: int = Field(..., description="Number of datasets found")
    datasets: List[CatalogDatasetInfo] = Field(
        default_factory=list, description="List of catalog datasets"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Filters applied to search"
    )


class LocationInfo(BaseModel):
    """Geographic location information."""

    name: str = Field(..., description="Location name")
    bbox: List[float] = Field(
        ..., description="Bounding box [west, south, east, north]"
    )
    center: List[float] = Field(..., description="Center coordinates [lon, lat]")
