"""Water body segmentation using OmniWaterMask.

This module provides a high-level interface for detecting water bodies in
satellite and aerial imagery using the OmniWaterMask library. It supports
a wide range of sensors (Sentinel-2, NAIP, Landsat, etc.) and resolutions
(0.2m to 50m) by combining a sensor-agnostic deep learning model with NDWI
calculations and OpenStreetMap reference data.

Reference:
    https://github.com/DPIRD-DMA/OmniWaterMask
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

BAND_ORDER_PRESETS: Dict[str, List[int]] = {
    "naip": [1, 2, 3, 4],
    "sentinel2": [3, 2, 1, 4],
    "landsat": [4, 3, 2, 5],
}
"""Predefined band order mappings for common sensors.

Each preset maps to a list of 1-based band indices in the order
[Red, Green, Blue, NIR] as used by rasterio.

- ``"naip"``: R, G, B, NIR (bands 1-4)
- ``"sentinel2"``: For 6-band composites (B2, B3, B4, B8, B11, B12),
  maps Red=B4 (band 3), Green=B3 (band 2), Blue=B2 (band 1),
  NIR=B8 (band 4)
- ``"landsat"``: For Landsat 8/9 (B1-B7), maps Red=B4 (band 4),
  Green=B3 (band 3), Blue=B2 (band 2), NIR=B5 (band 5)
"""


def _extract_water_band(src_path: str, dst_path: str) -> None:
    """Extract the water mask band from a multi-band OmniWaterMask output.

    OmniWaterMask outputs 2 bands: band 1 = water predictions,
    band 2 = no data mask. This function saves only band 1.

    Args:
        src_path: Path to the multi-band OmniWaterMask output GeoTIFF.
        dst_path: Path to save the single-band water mask GeoTIFF.
    """
    import rasterio

    with rasterio.open(src_path) as src:
        water_band = src.read(1)
        profile = src.profile.copy()
        profile.update(count=1)
        description = src.descriptions[0] if src.descriptions else "Water predictions"

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(water_band, 1)
        dst.set_band_description(1, description)


def _vectorize_mask(
    raster_path: str,
    output_vector: str,
    min_area: float,
    smooth: bool,
    smooth_iterations: int,
    overwrite: bool,
    verbose: bool,
) -> "gpd.GeoDataFrame":
    """Vectorize a water mask raster to polygons with optional smoothing.

    Args:
        raster_path: Path to the water mask GeoTIFF.
        output_vector: Path to save the output vector file.
        min_area: Minimum polygon area in square map units.
        smooth: Whether to smooth polygons.
        smooth_iterations: Number of smoothing iterations.
        overwrite: Whether to overwrite existing files.
        verbose: Whether to print progress messages.

    Returns:
        GeoDataFrame with vectorized water body polygons.
    """
    from .utils import add_geometric_properties, raster_to_vector, smooth_vector

    if not overwrite and Path(output_vector).exists():
        raise FileExistsError(
            f"Output vector '{output_vector}' already exists and "
            "overwrite is set to False."
        )

    if verbose:
        print("Converting water mask to vector polygons...")

    gdf = raster_to_vector(
        raster_path=raster_path,
        output_path=None,
        min_area=min_area,
    )

    if smooth and len(gdf) > 0:
        if verbose:
            print(f"Smoothing polygons ({smooth_iterations} iterations)...")
        gdf = smooth_vector(
            gdf,
            smooth_iterations=smooth_iterations,
        )

    gdf = add_geometric_properties(gdf, area_unit="m2", length_unit="m")

    # Save to file
    output_vector = str(output_vector)
    ext = Path(output_vector).suffix.lower()
    if ext == ".geojson":
        gdf.to_file(output_vector, driver="GeoJSON")
    elif ext == ".gpkg":
        gdf.to_file(output_vector, driver="GPKG")
    elif ext in (".shp", ".shapefile"):
        gdf.to_file(output_vector, driver="ESRI Shapefile")
    else:
        gdf.to_file(output_vector)

    if verbose:
        print(f"Water body polygons saved to: {output_vector}")
        print(f"Total water bodies detected: {len(gdf)}")

    return gdf


def segment_water(
    input_path: Union[str, "Path", List[Union[str, "Path"]]],
    band_order: Union[List[int], str] = "naip",
    output_raster: Optional[str] = None,
    output_vector: Optional[str] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
    dtype: str = "float32",
    no_data_value: int = 0,
    patch_size: int = 1000,
    overlap_size: int = 300,
    use_osm_water: bool = True,
    use_osm_building: bool = True,
    use_osm_roads: bool = True,
    cache_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    overwrite: bool = True,
    min_area: float = 10,
    smooth: bool = True,
    smooth_iterations: int = 3,
    verbose: bool = True,
    **kwargs: Any,
) -> Union[str, "gpd.GeoDataFrame", List[str], List["gpd.GeoDataFrame"]]:
    """Segment water bodies from satellite or aerial imagery using OmniWaterMask.

    Uses a sensor-agnostic deep learning model combined with NDWI and
    OpenStreetMap data to detect water bodies in imagery ranging from
    0.2m to 50m resolution. Supports Sentinel-2, NAIP, Landsat, and
    other multispectral sensors with Red, Green, Blue, and NIR bands.

    Args:
        input_path: Path to input GeoTIFF file(s). Can be a single path
            (string or Path) or a list of paths for batch processing.
            When a list is provided, each scene is processed and output
            files are named based on each input filename.
        band_order: Band indices for Red, Green, Blue, NIR channels
            (1-based, as used by rasterio). Can be a list of 4 integers
            or a string preset: ``"naip"`` ([1,2,3,4]),
            ``"sentinel2"`` ([3,2,1,4]), ``"landsat"`` ([4,3,2,5]).
            Defaults to ``"naip"``.
        output_raster: Path to save the output water mask GeoTIFF. If None,
            derives from input filename (e.g., ``input_water_mask.tif``).
            For multiple inputs, this is ignored and output names are
            derived from each input filename.
        output_vector: Path to save vectorized water body polygons (e.g.,
            GeoJSON, GPKG, Shapefile). If provided, the raster mask is
            converted to vector polygons. For multiple inputs, this is
            ignored and output names are derived from each input filename
            using the same extension. Defaults to None.
        batch_size: Number of scenes to process in parallel. Defaults to 4.
        device: Device for inference (e.g., ``"cuda"``, ``"cpu"``, ``"mps"``).
            If None, auto-selects the best available device. Defaults to None.
        dtype: Data type for model inference precision. One of
            ``"float32"``, ``"float16"``, or ``"bfloat16"``. Using
            ``"float16"`` is recommended as it is faster and uses less
            memory while the output is always a binary mask.
            Defaults to ``"float32"``.
        no_data_value: Value representing no-data pixels in the input imagery.
            Defaults to 0.
        patch_size: Size of patches for sliding-window inference in pixels.
            Defaults to 1000.
        overlap_size: Overlap between patches to reduce edge artifacts in
            pixels. Defaults to 300.
        use_osm_water: Include OpenStreetMap water features to improve
            accuracy. Defaults to True.
        use_osm_building: Include OSM building data to reduce false positives
            in built-up areas. Defaults to True.
        use_osm_roads: Include OSM road data to reduce false positives along
            roads. Defaults to True.
        cache_dir: Directory for caching intermediate results and OSM data.
            If None, uses ``OWM_cache`` in the current working directory.
            Defaults to None.
        model_dir: Custom directory to store/load OmniWaterMask model files.
            If None, uses the default location. Defaults to None.
        overwrite: Whether to overwrite existing output files.
            Defaults to True.
        min_area: Minimum polygon area in square map units to keep during
            vectorization. Defaults to 10.
        smooth: Whether to smooth vectorized polygons using the smoothify
            library. Only applies when ``output_vector`` is provided.
            Defaults to True.
        smooth_iterations: Number of smoothing iterations. Higher values
            produce smoother boundaries. Only applies when ``smooth=True``.
            Defaults to 3.
        verbose: Whether to print progress messages. Defaults to True.
        **kwargs: Additional keyword arguments passed to
            ``omniwatermask.make_water_mask()``.

    Returns:
        For a single input file:
            If ``output_vector`` is provided, returns a ``GeoDataFrame``
            with vectorized (and optionally smoothed) water body polygons.
            Otherwise, returns the file path (str) to the output water
            mask GeoTIFF.
        For multiple input files:
            Returns a list of results (list of str paths or list of
            GeoDataFrames), one per input file.

    Raises:
        ImportError: If omniwatermask is not installed.
        ValueError: If ``band_order`` is an unrecognized string preset or
            does not contain exactly 4 band indices.
        FileNotFoundError: If the input file(s) do not exist.

    Example:
        >>> import geoai
        >>> # NAIP imagery (R, G, B, NIR)
        >>> mask_path = geoai.segment_water(
        ...     "naip_scene.tif",
        ...     band_order="naip",
        ...     output_raster="water_mask.tif",
        ... )
        >>> # Sentinel-2 with vectorization and smoothing
        >>> gdf = geoai.segment_water(
        ...     "sentinel2_scene.tif",
        ...     band_order="sentinel2",
        ...     output_raster="water_mask.tif",
        ...     output_vector="water_bodies.geojson",
        ...     smooth=True,
        ...     smooth_iterations=3,
        ...     min_area=100,
        ... )
        >>> # Batch processing multiple files
        >>> results = geoai.segment_water(
        ...     ["scene1.tif", "scene2.tif"],
        ...     band_order="sentinel2",
        ... )
    """
    try:
        from omniwatermask import make_water_mask
    except ImportError:
        raise ImportError(
            "omniwatermask is required for water segmentation. "
            "Install it with: pip install omniwatermask "
            "or pip install geoai-py[extra]"
        )

    import torch

    # Resolve band order
    if isinstance(band_order, str):
        preset = band_order.lower()
        if preset not in BAND_ORDER_PRESETS:
            raise ValueError(
                f"Unknown band_order preset: {band_order!r}. "
                f"Available presets: {list(BAND_ORDER_PRESETS.keys())}. "
                f"Or pass a list of 4 integers for custom band order."
            )
        band_order = BAND_ORDER_PRESETS[preset]
    elif isinstance(band_order, (list, tuple)):
        if len(band_order) != 4:
            raise ValueError(
                f"band_order must contain exactly 4 band indices "
                f"(Red, Green, Blue, NIR), got {len(band_order)}."
            )
        if any(b < 1 for b in band_order):
            raise ValueError(
                f"band_order indices must be 1-based (>= 1), got {band_order}."
            )
    else:
        raise ValueError(
            f"band_order must be a string preset or list of integers, "
            f"got {type(band_order).__name__}."
        )

    # Normalize input path(s)
    if isinstance(input_path, (str, Path)):
        scene_paths = [Path(input_path)]
        multi_input = False
    else:
        scene_paths = [Path(p) for p in input_path]
        multi_input = len(scene_paths) > 1

    for p in scene_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # Determine output raster path(s)
    if multi_input:
        # For multiple inputs, derive output names from each input
        output_dir = Path.cwd() if output_raster is None else Path(output_raster)
        if output_raster is not None and not output_dir.suffix:
            # output_raster is treated as a directory for multi-input
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = Path.cwd()
        raster_paths = [
            str(output_dir / f"{p.stem}_water_mask.tif") for p in scene_paths
        ]
    else:
        if output_raster is None:
            stem = scene_paths[0].stem
            output_raster = str(scene_paths[0].parent / f"{stem}_water_mask.tif")
        raster_paths = [str(output_raster)]

    # Check overwrite for output files (single input only; multi checked per-file)
    if not multi_input and not overwrite:
        if Path(raster_paths[0]).exists():
            raise FileExistsError(
                f"Output raster '{raster_paths[0]}' already exists and "
                "overwrite is set to False."
            )
        if output_vector is not None and Path(output_vector).exists():
            raise FileExistsError(
                f"Output vector '{output_vector}' already exists and "
                "overwrite is set to False."
            )

    # Map dtype string to torch.dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(dtype, str):
        if dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype: {dtype!r}. Choose from: {list(dtype_map.keys())}"
            )
        inference_dtype = dtype_map[dtype]
    else:
        inference_dtype = dtype

    # Determine the output directory for OmniWaterMask
    owm_output_dir = Path(raster_paths[0]).parent

    # Build kwargs for make_water_mask
    owm_kwargs = {
        "scene_paths": scene_paths,
        "band_order": band_order,
        "batch_size": batch_size,
        "inference_dtype": inference_dtype,
        "no_data_value": no_data_value,
        "inference_patch_size": patch_size,
        "inference_overlap_size": overlap_size,
        "use_osm_water": use_osm_water,
        "use_osm_building": use_osm_building,
        "use_osm_roads": use_osm_roads,
        "overwrite": overwrite,
        "output_dir": owm_output_dir,
    }

    if cache_dir is not None:
        owm_kwargs["cache_dir"] = Path(cache_dir)

    if model_dir is not None:
        owm_kwargs["destination_model_dir"] = Path(model_dir)

    if device is not None:
        owm_kwargs["inference_device"] = device
        owm_kwargs["mosaic_device"] = device

    # Merge any extra kwargs
    owm_kwargs.update(kwargs)

    # Run OmniWaterMask inference
    if verbose:
        print("Running OmniWaterMask water segmentation...")

    result_paths = make_water_mask(**owm_kwargs)

    if not result_paths:
        raise RuntimeError("OmniWaterMask did not produce any output files.")

    # Process each result: extract water band and optionally vectorize
    output_rasters = []
    output_gdfs = []

    for i, result_path in enumerate(result_paths):
        dst_raster = (
            raster_paths[i]
            if i < len(raster_paths)
            else str(owm_output_dir / f"{scene_paths[i].stem}_water_mask.tif")
        )

        os.makedirs(Path(dst_raster).parent, exist_ok=True)
        _extract_water_band(str(result_path), dst_raster)

        # Clean up original multi-band file if it differs from output
        if Path(result_path).resolve() != Path(dst_raster).resolve():
            try:
                os.remove(str(result_path))
            except OSError:
                pass

        if verbose:
            print(f"Water mask saved to: {dst_raster}")

        output_rasters.append(dst_raster)

        # Vectorize if requested
        if output_vector is not None:
            if multi_input:
                # Derive per-file vector name from input stem
                vec_ext = Path(output_vector).suffix or ".geojson"
                vec_path = str(
                    Path(dst_raster).parent
                    / f"{scene_paths[i].stem}_water_bodies{vec_ext}"
                )
            else:
                vec_path = str(output_vector)

            gdf = _vectorize_mask(
                raster_path=dst_raster,
                output_vector=vec_path,
                min_area=min_area,
                smooth=smooth,
                smooth_iterations=smooth_iterations,
                overwrite=overwrite,
                verbose=verbose,
            )
            output_gdfs.append(gdf)

    # Return results
    if output_vector is not None:
        return output_gdfs if multi_input else output_gdfs[0]
    return output_rasters if multi_input else output_rasters[0]
