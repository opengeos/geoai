"""Road network processing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_geopandas():
    """Import geopandas with a helpful error message."""
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - dependency is required by geoai
        raise ImportError(
            "geopandas is required for road network processing. "
            "Install it with `pip install geopandas`."
        ) from exc
    return gpd


def _import_neatnet():
    """Import neatnet with a helpful error message."""
    try:
        import neatnet
    except ImportError as exc:
        raise ImportError(
            "neatnet is required to simplify road networks. Install the optional "
            "network dependencies with `pip install geoai-py[networks]` or install "
            "neatnet directly with `pip install neatnet`."
        ) from exc
    return neatnet


def simplify_road_network(
    roads: Any,
    output: str | Path | None = None,
    exclusion_mask: Any | None = None,
    **kwargs: Any,
):
    """Simplify a road or street network with ``neatnet``.

    This function wraps :func:`neatnet.neatify` to convert transportation-oriented
    street geometries, such as dual carriageways and roundabouts, into a simpler
    morphological network that better represents street centerlines. It is useful
    for post-processing road networks derived from deep learning segmentation or
    vectorized masks.

    Args:
        roads: Input road network as a GeoDataFrame or a path to any vector file
            supported by GeoPandas, such as GeoPackage, GeoJSON, or Shapefile.
        output: Optional path where the simplified network will be written. The
            driver is inferred from the file extension by GeoPandas/Fiona.
        exclusion_mask: Optional GeoSeries, GeoDataFrame, geometry sequence, or
            vector file path passed to ``neatnet.neatify`` as ``exclusion_mask``.
            Building footprints are commonly used to preserve real street blocks.
        **kwargs: Additional keyword arguments forwarded to
            :func:`neatnet.neatify`.

    Returns:
        geopandas.GeoDataFrame: The simplified road network.

    Raises:
        ImportError: If ``neatnet`` is not installed.

    Examples:
        >>> import geoai
        >>> simplified = geoai.simplify_road_network("roads.gpkg")
        >>> simplified.to_file("roads_simplified.gpkg")
    """
    gpd = _import_geopandas()
    neatnet = _import_neatnet()

    if isinstance(roads, (str, Path)):
        roads_gdf = gpd.read_file(roads)
    else:
        roads_gdf = roads

    neatify_kwargs = dict(kwargs)
    if exclusion_mask is not None:
        if isinstance(exclusion_mask, (str, Path)):
            exclusion_data = gpd.read_file(exclusion_mask)
            neatify_kwargs["exclusion_mask"] = exclusion_data.geometry
        else:
            neatify_kwargs["exclusion_mask"] = exclusion_mask

    simplified = neatnet.neatify(roads_gdf, **neatify_kwargs)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simplified.to_file(output_path)

    return simplified
