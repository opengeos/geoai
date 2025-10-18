"""Structured output models for STAC catalog search results."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class STACCollectionInfo(BaseModel):
    """Information about a STAC collection."""

    id: str = Field(..., description="Collection identifier")
    title: str = Field(..., description="Collection title")
    description: Optional[str] = Field(None, description="Collection description")
    license: Optional[str] = Field(None, description="License information")
    temporal_extent: Optional[str] = Field(
        None, description="Temporal extent (start/end dates)"
    )
    spatial_extent: Optional[str] = Field(None, description="Spatial bounding box")
    providers: Optional[str] = Field(None, description="Data providers")
    keywords: Optional[str] = Field(None, description="Keywords")


class STACAssetInfo(BaseModel):
    """Information about a STAC item asset."""

    key: str = Field(..., description="Asset key/identifier")
    title: Optional[str] = Field(None, description="Asset title")


class STACItemInfo(BaseModel):
    """Information about a STAC item."""

    id: str = Field(..., description="Item identifier")
    collection: str = Field(..., description="Collection ID")
    datetime: Optional[str] = Field(None, description="Acquisition datetime")
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box [west, south, east, north]"
    )
    assets: List[STACAssetInfo] = Field(
        default_factory=list, description="Available assets"
    )
    # properties: Optional[Dict[str, Any]] = Field(
    #     None, description="Additional metadata properties"
    # )


class STACSearchResult(BaseModel):
    """Container for STAC search results."""

    query: str = Field(..., description="Original search query")
    collection: Optional[str] = Field(None, description="Collection searched")
    item_count: int = Field(..., description="Number of items found")
    items: List[STACItemInfo] = Field(
        default_factory=list, description="List of STAC items"
    )
    bbox: Optional[List[float]] = Field(None, description="Search bounding box used")
    time_range: Optional[str] = Field(None, description="Time range used for search")


class LocationInfo(BaseModel):
    """Geographic location information."""

    name: str = Field(..., description="Location name")
    bbox: List[float] = Field(
        ..., description="Bounding box [west, south, east, north]"
    )
    center: List[float] = Field(..., description="Center coordinates [lon, lat]")
