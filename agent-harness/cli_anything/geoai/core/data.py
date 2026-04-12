"""Data discovery, download, and preparation.

Wraps geoai data functions: NAIP download, STAC search,
Overture Maps, and tile generation for CLI consumption.
"""

import os
from typing import Any, Dict, List, Optional, Tuple


DATA_SOURCES = ["naip", "overture", "stac"]


def search_stac(
    bbox: Tuple[float, float, float, float],
    collection: str = "sentinel-2-l2a",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_items: int = 10,
) -> Dict[str, Any]:
    """Search for satellite imagery via STAC API.

    Args:
        bbox: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326.
        collection: STAC collection ID.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        max_items: Maximum number of results.

    Returns:
        Dict with search results.
    """
    from geoai import pc_stac_search

    kwargs = {
        "bbox": list(bbox),
        "collections": [collection],
        "max_items": max_items,
    }
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date

    items = pc_stac_search(**kwargs)

    results = []
    for item in items:
        entry = {
            "id": item.id if hasattr(item, "id") else str(item),
            "datetime": str(item.datetime) if hasattr(item, "datetime") else "",
        }
        if hasattr(item, "properties"):
            props = item.properties
            entry["cloud_cover"] = props.get("eo:cloud_cover")
        if hasattr(item, "assets"):
            entry["asset_count"] = len(item.assets)
            entry["assets"] = list(item.assets.keys())[:10]
        results.append(entry)

    return {
        "collection": collection,
        "bbox": list(bbox),
        "start_date": start_date,
        "end_date": end_date,
        "total_results": len(results),
        "items": results,
    }


def download_naip(
    bbox: Tuple[float, float, float, float],
    output: str,
    year: Optional[int] = None,
    max_items: int = 10,
) -> Dict[str, Any]:
    """Download NAIP aerial imagery.

    Args:
        bbox: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326.
        output: Output file path.
        year: Optional specific year.
        max_items: Maximum number of items to download.

    Returns:
        Dict with download result and file paths.
    """
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    from geoai.download import download_naip as _download_naip

    kwargs = {"bbox": list(bbox), "output": output, "max_items": max_items}
    if year is not None:
        kwargs["year"] = year

    result = _download_naip(**kwargs)

    files = result if isinstance(result, list) else [result]
    return {
        "source": "naip",
        "bbox": list(bbox),
        "year": year,
        "output": output,
        "files": [str(f) for f in files],
        "file_count": len(files),
    }


def download_overture(
    bbox: Tuple[float, float, float, float],
    output: str,
    data_type: str = "building",
    output_format: str = "geojson",
) -> Dict[str, Any]:
    """Download Overture Maps data.

    Args:
        bbox: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326.
        output: Output file path.
        data_type: Data type (building, transportation, place).
        output_format: Output format (geojson, parquet).

    Returns:
        Dict with download result.
    """
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    from geoai import download_overture_buildings

    result = download_overture_buildings(
        bbox=list(bbox),
        output=output,
    )

    out = {
        "source": "overture",
        "data_type": data_type,
        "bbox": list(bbox),
        "output": str(result) if result else output,
    }

    target = str(result) if result else output
    if os.path.isfile(target):
        out["file_size_bytes"] = os.path.getsize(target)

    return out


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse a comma-separated bounding box string.

    Args:
        bbox_str: Bounding box as "minx,miny,maxx,maxy".

    Returns:
        Tuple of (minx, miny, maxx, maxy).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError("Expected 4 values")
        return (parts[0], parts[1], parts[2], parts[3])
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid bbox format: '{bbox_str}'. "
            "Expected: minx,miny,maxx,maxy (e.g., -84.0,35.9,-83.9,36.0)"
        ) from e


def list_sources() -> List[Dict[str, str]]:
    """List available data sources.

    Returns:
        List of source info dicts.
    """
    descriptions = {
        "naip": "USDA National Agriculture Imagery Program (aerial, US only)",
        "overture": "Overture Maps Foundation (buildings, roads, POIs, global)",
        "stac": "Planetary Computer STAC API (Sentinel-2, Landsat, etc.)",
    }
    return [
        {"name": s, "description": descriptions.get(s, "")}
        for s in DATA_SOURCES
    ]
