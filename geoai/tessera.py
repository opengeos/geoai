"""TESSERA module for accessing geospatial foundation model embeddings.

This module provides tools for working with TESSERA (Temporal Embeddings of
Surface Spectra for Earth Representation and Analysis) embeddings via the
GeoTessera library. TESSERA is a foundation model developed at the University
of Cambridge that processes time-series Sentinel-1 and Sentinel-2 satellite
imagery to generate 128-channel representation maps at 10m resolution globally.

Reference:
    Feng et al., "TESSERA: Temporal Embeddings of Surface Spectra for Earth
    Representation and Analysis," ArXiv preprint, 2025.
    https://arxiv.org/abs/2506.20380

    Repository: https://github.com/ucam-eo/tessera
    GeoTessera library: https://github.com/ucam-eo/geotessera
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _check_geotessera():
    """Check if geotessera is installed and raise informative error if not."""
    try:
        import geotessera

        return geotessera
    except ImportError:
        raise ImportError(
            "The geotessera package is required for TESSERA support. "
            "Install it with: pip install geotessera\n"
            "Or with conda: conda install -c conda-forge geotessera"
        )


def tessera_download(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    year: int = 2024,
    output_dir: str = "./tessera_output",
    output_format: str = "tiff",
    bands: Optional[List[int]] = None,
    compress: str = "lzw",
    region_file: Optional[str] = None,
    dataset_version: str = "v1",
    **kwargs,
) -> List[str]:
    """Download TESSERA embeddings for a geographic region.

    Downloads pre-computed TESSERA foundation model embeddings at 10m resolution.
    Embeddings are 128-channel representations that compress a full year of
    Sentinel-1 and Sentinel-2 temporal-spectral features.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat). Either bbox
            or lon/lat must be provided.
        lon: Longitude for a single tile download. Used with lat.
        lat: Latitude for a single tile download. Used with lon.
        year: Year of embeddings to download. Available years: 2017-2024.
            Defaults to 2024.
        output_dir: Directory to save downloaded files. Defaults to
            "./tessera_output".
        output_format: Output format, either "tiff" (georeferenced GeoTIFF)
            or "npy" (raw numpy arrays with metadata JSON). Defaults to "tiff".
        bands: List of specific band indices to download (0-127). If None,
            all 128 bands are downloaded. Defaults to None.
        compress: Compression method for GeoTIFF output. Options: "lzw",
            "deflate", "zstd", "none". Defaults to "lzw".
        region_file: Path to a GeoJSON or Shapefile to define the download
            region. If provided, overrides bbox.
        dataset_version: TESSERA dataset version. Defaults to "v1".
        **kwargs: Additional keyword arguments passed to GeoTessera constructor.

    Returns:
        List of file paths for downloaded files.

    Raises:
        ImportError: If geotessera package is not installed.
        ValueError: If neither bbox, lon/lat, nor region_file is provided.

    Example:
        >>> import geoai
        >>> # Download embeddings for a bounding box
        >>> files = geoai.tessera_download(
        ...     bbox=(-0.2, 51.4, 0.1, 51.6),
        ...     year=2024,
        ...     output_dir="./london_embeddings"
        ... )
        >>> # Download a single tile
        >>> files = geoai.tessera_download(
        ...     lon=0.15, lat=52.05,
        ...     year=2024,
        ...     output_dir="./cambridge_tile"
        ... )
        >>> # Download specific bands only
        >>> files = geoai.tessera_download(
        ...     bbox=(-0.2, 51.4, 0.1, 51.6),
        ...     bands=[0, 1, 2],
        ...     output_dir="./london_rgb"
        ... )
    """
    _check_geotessera()
    from geotessera import GeoTessera

    gt = GeoTessera(dataset_version=dataset_version, **kwargs)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine tiles to fetch
    if region_file is not None:
        import geopandas as gpd

        gdf = gpd.read_file(region_file)
        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
    elif bbox is not None:
        pass  # Use bbox directly
    elif lon is not None and lat is not None:
        # Single tile mode - create a small bbox around the point
        bbox = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
    else:
        raise ValueError("Must provide one of: bbox, lon/lat pair, or region_file")

    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
    num_tiles = len(tiles_to_fetch)
    logger.info(f"Found {num_tiles} tiles to download for year {year}")

    if num_tiles == 0:
        logger.warning(
            f"No tiles available for the specified region and year {year}. "
            "Use tessera_coverage() to check data availability."
        )
        return []

    created_files = []

    if output_format.lower() == "tiff":
        files = gt.export_embedding_geotiffs(
            tiles_to_fetch=tiles_to_fetch,
            output_dir=str(output_path),
            bands=bands,
            compress=compress,
        )
        created_files = [str(f) for f in files]
        logger.info(f"Exported {len(created_files)} GeoTIFF files to {output_dir}")
    elif output_format.lower() == "npy":
        import json

        metadata = {
            "year": year,
            "bbox": list(bbox) if bbox else None,
            "bands": bands,
            "version": dataset_version,
            "tiles": [],
        }
        for yr, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(
            tiles_to_fetch
        ):
            if bands is not None:
                embedding = embedding[:, :, bands]

            filename = f"grid_{tile_lon:.2f}_{tile_lat:.2f}_{yr}.npy"
            filepath = output_path / filename
            np.save(str(filepath), embedding)
            created_files.append(str(filepath))

            metadata["tiles"].append(
                {
                    "file": filename,
                    "lon": tile_lon,
                    "lat": tile_lat,
                    "year": yr,
                    "shape": list(embedding.shape),
                    "crs": str(crs) if crs else None,
                }
            )

        # Save metadata
        meta_path = output_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        created_files.append(str(meta_path))
        logger.info(f"Saved {len(created_files) - 1} numpy arrays to {output_dir}")
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'tiff' or 'npy'.")

    return created_files


def tessera_fetch_embeddings(
    bbox: Tuple[float, float, float, float],
    year: int = 2024,
    bands: Optional[List[int]] = None,
    dataset_version: str = "v1",
    **kwargs,
) -> list:
    """Fetch TESSERA embeddings as numpy arrays without saving to disk.

    This function retrieves embeddings directly into memory, useful for
    immediate analysis without file I/O overhead.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        year: Year of embeddings. Defaults to 2024.
        bands: List of specific band indices to extract (0-127). If None,
            all 128 bands are returned. Defaults to None.
        dataset_version: TESSERA dataset version. Defaults to "v1".
        **kwargs: Additional keyword arguments passed to GeoTessera constructor.

    Returns:
        List of dictionaries, each containing:
            - "embedding": numpy array of shape (H, W, C)
            - "lon": tile center longitude
            - "lat": tile center latitude
            - "year": tile year
            - "crs": coordinate reference system
            - "transform": affine transform

    Example:
        >>> import geoai
        >>> tiles = geoai.tessera_fetch_embeddings(
        ...     bbox=(-0.2, 51.4, 0.1, 51.6),
        ...     year=2024
        ... )
        >>> for tile in tiles:
        ...     print(f"Tile ({tile['lon']}, {tile['lat']}): {tile['embedding'].shape}")
    """
    _check_geotessera()
    from geotessera import GeoTessera

    gt = GeoTessera(dataset_version=dataset_version, **kwargs)
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)

    results = []
    for yr, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(
        tiles_to_fetch
    ):
        if bands is not None:
            embedding = embedding[:, :, bands]

        results.append(
            {
                "embedding": embedding,
                "lon": tile_lon,
                "lat": tile_lat,
                "year": yr,
                "crs": crs,
                "transform": transform,
            }
        )

    return results


def tessera_coverage(
    year: Optional[int] = None,
    output_path: str = "tessera_coverage.png",
    region_bbox: Optional[Tuple[float, float, float, float]] = None,
    region_file: Optional[str] = None,
    tile_color: str = "red",
    tile_alpha: float = 0.6,
    width_pixels: int = 2000,
    show_countries: bool = True,
    dataset_version: str = "v1",
    **kwargs,
) -> str:
    """Generate a coverage map showing TESSERA data availability.

    Creates a PNG map showing which tiles have embeddings available for the
    specified year and region. This is the recommended first step before
    downloading data to verify availability.

    Args:
        year: Specific year to visualize coverage for. If None, shows all
            available years with color coding. Defaults to None.
        output_path: Path for the output PNG file. Defaults to
            "tessera_coverage.png".
        region_bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat)
            to focus on a specific region.
        region_file: Optional path to a GeoJSON/Shapefile to focus on a region.
        tile_color: Color for tile rectangles. Defaults to "red".
        tile_alpha: Transparency of tile rectangles (0-1). Defaults to 0.6.
        width_pixels: Width of output image in pixels. Defaults to 2000.
        show_countries: Whether to show country boundaries. Defaults to True.
        dataset_version: TESSERA dataset version. Defaults to "v1".
        **kwargs: Additional keyword arguments passed to GeoTessera constructor.

    Returns:
        Path to the created coverage map PNG file.

    Example:
        >>> import geoai
        >>> # Check global coverage for 2024
        >>> geoai.tessera_coverage(year=2024)
        >>> # Check coverage for a specific region
        >>> geoai.tessera_coverage(
        ...     year=2024,
        ...     region_bbox=(-10, 35, 40, 60),
        ...     output_path="europe_coverage.png"
        ... )
    """
    _check_geotessera()
    from geotessera import GeoTessera
    from geotessera.visualization import visualize_global_coverage

    gt = GeoTessera(dataset_version=dataset_version, **kwargs)

    result = visualize_global_coverage(
        tessera_client=gt,
        output_path=output_path,
        year=year,
        width_pixels=width_pixels,
        show_countries=show_countries,
        tile_color=tile_color,
        tile_alpha=tile_alpha,
        region_bbox=region_bbox,
        region_file=region_file,
    )

    logger.info(f"Coverage map saved to {result}")
    return result


def tessera_visualize_rgb(
    geotiff_dir: str,
    bands: Tuple[int, int, int] = (0, 1, 2),
    output_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """Visualize TESSERA embeddings as an RGB composite image.

    Creates a false-color RGB visualization from three selected embedding
    bands. This helps with visual inspection and understanding of the
    embedding spatial patterns.

    Args:
        geotiff_dir: Directory containing TESSERA GeoTIFF files or path to
            a single GeoTIFF file.
        bands: Tuple of three band indices to use as (R, G, B). Defaults
            to (0, 1, 2).
        output_path: Optional path to save the visualization. If None,
            displays with matplotlib. Defaults to None.
        normalize: Whether to normalize band values to 0-1 range using
            percentile stretching. Defaults to True.
        figsize: Figure size as (width, height) in inches. Defaults to
            (12, 8).
        title: Optional title for the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib imshow.

    Returns:
        Path to saved image if output_path is provided, otherwise None.

    Example:
        >>> import geoai
        >>> # Download embeddings first
        >>> files = geoai.tessera_download(
        ...     bbox=(-0.2, 51.4, 0.1, 51.6),
        ...     output_dir="./london"
        ... )
        >>> # Visualize with default bands
        >>> geoai.tessera_visualize_rgb("./london")
        >>> # Use different band combination
        >>> geoai.tessera_visualize_rgb("./london", bands=(30, 60, 90))
    """
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.merge import merge

    geotiff_path = Path(geotiff_dir)

    # Collect GeoTIFF files
    if geotiff_path.is_file():
        tiff_files = [geotiff_path]
    elif geotiff_path.is_dir():
        tiff_files = sorted(geotiff_path.glob("*.tif")) + sorted(
            geotiff_path.glob("*.tiff")
        )
    else:
        raise FileNotFoundError(f"Path not found: {geotiff_dir}")

    if not tiff_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {geotiff_dir}")

    # Read and merge if multiple files
    if len(tiff_files) == 1:
        with rasterio.open(tiff_files[0]) as src:
            # Read the three specified bands (1-indexed in rasterio)
            rgb = np.stack([src.read(b + 1) for b in bands], axis=-1).astype(np.float32)
    else:
        # Merge multiple tiles
        datasets = [rasterio.open(f) for f in tiff_files]
        try:
            merged, _ = merge(datasets, indexes=[b + 1 for b in bands])
            rgb = np.moveaxis(merged, 0, -1).astype(np.float32)
        finally:
            for ds in datasets:
                ds.close()

    # Normalize for display
    if normalize:
        for i in range(3):
            band = rgb[:, :, i]
            valid = band[~np.isnan(band)]
            if len(valid) > 0:
                vmin = np.percentile(valid, 2)
                vmax = np.percentile(valid, 98)
                if vmax > vmin:
                    rgb[:, :, i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
                else:
                    rgb[:, :, i] = 0

    # Handle NaN values
    rgb = np.nan_to_num(rgb, nan=0.0)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb, **kwargs)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(
            f"TESSERA Embedding RGB (bands {bands[0]}, {bands[1]}, {bands[2]})",
            fontsize=14,
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Visualization saved to {output_path}")
        return output_path
    else:
        plt.show()
        return None


def tessera_tile_count(
    bbox: Tuple[float, float, float, float],
    year: int = 2024,
    dataset_version: str = "v1",
    **kwargs,
) -> int:
    """Get the number of available TESSERA tiles in a bounding box.

    Useful for estimating download size before fetching data.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        year: Year to check. Defaults to 2024.
        dataset_version: TESSERA dataset version. Defaults to "v1".
        **kwargs: Additional keyword arguments passed to GeoTessera constructor.

    Returns:
        Number of available tiles.

    Example:
        >>> import geoai
        >>> count = geoai.tessera_tile_count(
        ...     bbox=(-0.2, 51.4, 0.1, 51.6),
        ...     year=2024
        ... )
        >>> print(f"{count} tiles available")
    """
    _check_geotessera()
    from geotessera import GeoTessera

    gt = GeoTessera(dataset_version=dataset_version, **kwargs)
    return gt.embeddings_count(bbox=bbox, year=year)


def tessera_available_years(
    dataset_version: str = "v1",
    **kwargs,
) -> List[int]:
    """Get list of years with available TESSERA embeddings.

    Args:
        dataset_version: TESSERA dataset version. Defaults to "v1".
        **kwargs: Additional keyword arguments passed to GeoTessera constructor.

    Returns:
        List of available years sorted in ascending order.

    Example:
        >>> import geoai
        >>> years = geoai.tessera_available_years()
        >>> print(f"Available years: {years}")
    """
    _check_geotessera()
    from geotessera import GeoTessera

    gt = GeoTessera(dataset_version=dataset_version, **kwargs)
    return gt.registry.get_available_years()


def tessera_sample_points(
    points: Union[str, "gpd.GeoDataFrame"],
    year: int = 2024,
    embeddings_dir: Optional[str] = None,
    auto_download: bool = True,
    dataset_version: str = "v1",
    **kwargs,
) -> "gpd.GeoDataFrame":
    """Sample TESSERA embeddings at specific point locations.

    Extracts 128-dimensional embedding vectors at given geographic point
    locations. Useful for generating features for downstream tasks such
    as classification, regression, or clustering.

    Args:
        points: GeoDataFrame with point geometries or path to a file
            (GeoJSON, Shapefile, etc.) containing point locations.
        year: Year of embeddings to sample. Defaults to 2024.
        embeddings_dir: Directory containing pre-downloaded embedding tiles.
            If None, uses current directory. Tiles are downloaded automatically
            if auto_download is True.
        auto_download: Whether to automatically download missing tiles.
            Defaults to True.
        dataset_version: TESSERA dataset version. Defaults to "v1".
        **kwargs: Additional keyword arguments passed to GeoTessera constructor.

    Returns:
        GeoDataFrame with the original columns plus 128 new columns
        (tessera_0 through tessera_127) containing embedding values.

    Example:
        >>> import geoai
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> # Create sample points
        >>> points = gpd.GeoDataFrame(
        ...     {"id": [1, 2]},
        ...     geometry=[Point(0.15, 52.05), Point(0.25, 52.15)],
        ...     crs="EPSG:4326"
        ... )
        >>> # Sample embeddings
        >>> result = geoai.tessera_sample_points(points, year=2024)
        >>> print(result.columns.tolist())
    """
    _check_geotessera()
    import geopandas as gpd
    from geotessera import GeoTessera

    # Load points if path is provided
    if isinstance(points, (str, Path)):
        gdf = gpd.read_file(points)
    else:
        gdf = points.copy()

    # Ensure CRS is WGS84
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gt_kwargs = dict(dataset_version=dataset_version, **kwargs)
    if embeddings_dir is not None:
        gt_kwargs["embeddings_dir"] = embeddings_dir

    gt = GeoTessera(**gt_kwargs)

    # Get bounding box of all points
    bounds = gdf.total_bounds
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

    # Fetch tiles for the region
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)

    # Build a spatial index of tiles
    tile_data = {}
    for yr, tile_lon, tile_lat, embedding, crs, transform in gt.fetch_embeddings(
        tiles_to_fetch
    ):
        tile_data[(tile_lon, tile_lat)] = {
            "embedding": embedding,
            "crs": crs,
            "transform": transform,
        }

    # Sample each point
    embedding_values = []
    for _, row in gdf.iterrows():
        point_lon = row.geometry.x
        point_lat = row.geometry.y

        # Find the tile containing this point
        tile_lon = round(round(point_lon / 0.1) * 0.1 + 0.05, 2)
        tile_lat = round(round(point_lat / 0.1) * 0.1 + 0.05, 2)

        tile = tile_data.get((tile_lon, tile_lat))
        if tile is None:
            # Try nearby tiles
            found = False
            for (tlon, tlat), tdata in tile_data.items():
                if abs(tlon - point_lon) <= 0.1 and abs(tlat - point_lat) <= 0.1:
                    tile = tdata
                    found = True
                    break
            if not found:
                embedding_values.append([np.nan] * 128)
                continue

        # Convert point to pixel coordinates using the transform
        try:
            import rasterio
            from rasterio.warp import transform as transform_coords

            if tile["crs"] and str(tile["crs"]) != "EPSG:4326":
                xs, ys = transform_coords(
                    "EPSG:4326", tile["crs"], [point_lon], [point_lat]
                )
                px_x, px_y = ~tile["transform"] * (xs[0], ys[0])
            else:
                px_x, px_y = ~tile["transform"] * (point_lon, point_lat)

            px_x, px_y = int(px_x), int(px_y)
            h, w = tile["embedding"].shape[:2]

            if 0 <= px_x < w and 0 <= px_y < h:
                values = tile["embedding"][px_y, px_x, :].tolist()
                embedding_values.append(values)
            else:
                embedding_values.append([np.nan] * 128)
        except Exception as e:
            logger.warning(f"Error sampling point ({point_lon}, {point_lat}): {e}")
            embedding_values.append([np.nan] * 128)

    # Add embedding columns to the GeoDataFrame
    embedding_cols = [f"tessera_{i}" for i in range(128)]
    embedding_df = gpd.pd.DataFrame(embedding_values, columns=embedding_cols)
    embedding_df.index = gdf.index

    result = gpd.pd.concat([gdf, embedding_df], axis=1)
    result = gpd.GeoDataFrame(result, geometry=gdf.geometry.name, crs=gdf.crs)

    return result
