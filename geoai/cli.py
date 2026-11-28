"""Command-line interface for geoai."""

import json
import math
import os
import sys
from pathlib import Path

import click


def _to_json_value(value):
    """Recursively normalize values for strict JSON output."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if (
        hasattr(value, "left")
        and hasattr(value, "bottom")
        and hasattr(value, "right")
        and hasattr(value, "top")
    ):
        return {
            "left": _to_json_value(value.left),
            "bottom": _to_json_value(value.bottom),
            "right": _to_json_value(value.right),
            "top": _to_json_value(value.top),
        }
    if (
        hasattr(value, "a")
        and hasattr(value, "b")
        and hasattr(value, "c")
        and hasattr(value, "d")
        and hasattr(value, "e")
        and hasattr(value, "f")
        and hasattr(value, "to_gdal")
    ):
        return [_to_json_value(v) for v in value]
    if hasattr(value, "to_string") and type(value).__name__ == "CRS":
        return value.to_string()
    if type(value).__module__.startswith("numpy"):
        import numpy as np

        if isinstance(value, np.ma.core.MaskedConstant):
            return None
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            fval = float(value)
            return fval if math.isfinite(fval) else None
        if isinstance(value, np.ndarray):
            return [_to_json_value(v) for v in value.tolist()]
    if type(value).__name__ == "NAType":
        return None
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _emit_info(info_dict, as_json):
    """Print raster or vector info in plain text or JSON format."""
    if as_json:
        serializable = _to_json_value(info_dict)
        click.echo(json.dumps(serializable, indent=2, allow_nan=False))
    else:
        for key, value in info_dict.items():
            click.echo(f"{key}: {value}")


@click.group()
@click.version_option(package_name="geoai-py")
def main():
    """GeoAI - AI for Geospatial Data.

    A command-line tool for geospatial AI workflows including
    file inspection, data download, segmentation, and object detection.
    """
    pass


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output information as JSON.")
def info(filepath, as_json):
    """Display information about a raster or vector file.

    FILEPATH is the path to a raster (GeoTIFF, etc.) or vector
    (GeoJSON, Shapefile, GeoPackage, etc.) file.
    """
    filepath = os.path.abspath(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    vector_extensions = {
        ".geojson",
        ".json",
        ".shp",
        ".gpkg",
        ".parquet",
        ".geoparquet",
        ".fgb",
        ".kml",
    }
    raster_extensions = {".tif", ".tiff", ".img", ".jp2", ".vrt", ".nc", ".hdf"}

    if ext in vector_extensions:
        from geoai.utils import get_vector_info

        info_dict = get_vector_info(filepath)
        _emit_info(info_dict, as_json)
    elif ext in raster_extensions:
        from geoai.utils import get_raster_info

        info_dict = get_raster_info(filepath)
        _emit_info(info_dict, as_json)
    else:
        click.echo(
            f"Unknown file extension '{ext}'. Attempting to read as raster...",
            err=True,
        )
        try:
            from geoai.utils import get_raster_info

            info_dict = get_raster_info(filepath)
            _emit_info(info_dict, as_json)
        except Exception:
            click.echo(
                f"Could not read '{filepath}' as raster. "
                "Attempting to read as vector...",
                err=True,
            )
            try:
                from geoai.utils import get_vector_info

                info_dict = get_vector_info(filepath)
                _emit_info(info_dict, as_json)
            except Exception as e:
                click.echo(f"Error: Could not read file: {e}", err=True)
                sys.exit(1)


@main.command()
@click.argument(
    "source", type=click.Choice(["naip", "overture", "stac"], case_sensitive=False)
)
@click.option(
    "--bbox",
    required=True,
    help="Bounding box as minx,miny,maxx,maxy (EPSG:4326).",
)
@click.option("--output", "-o", required=True, help="Output file or directory path.")
@click.option("--year", default=None, type=int, help="Year of imagery (naip).")
@click.option(
    "--overture-type",
    default="building",
    help="Overture Maps data type (overture, e.g. building, place, segment).",
)
@click.option(
    "--collection",
    default="sentinel-2-l2a",
    help="STAC collection ID (stac).",
)
@click.option(
    "--time-range",
    default=None,
    help="STAC time range as 'start/end' (stac).",
)
@click.option(
    "--assets",
    default=None,
    help="Comma-separated STAC asset keys to download (stac).",
)
@click.option(
    "--max-items",
    default=1,
    type=click.IntRange(min=1),
    help="Maximum number of STAC items to download (stac).",
)
def download(
    source,
    bbox,
    output,
    year,
    overture_type,
    collection,
    time_range,
    assets,
    max_items,
):
    """Download geospatial data from supported sources.

    SOURCE is the data source to download from (naip, overture, or stac).
    """
    try:
        coords = [float(x.strip()) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError
    except ValueError:
        click.echo(
            "Error: --bbox must be four comma-separated numbers: minx,miny,maxx,maxy",
            err=True,
        )
        sys.exit(1)

    source = source.lower()
    if source == "naip":
        from geoai.download import download_naip

        kwargs = {}
        if year is not None:
            kwargs["year"] = year

        click.echo(f"Downloading NAIP imagery for bbox={coords}...")
        try:
            result = download_naip(bbox=coords, output_dir=output, **kwargs)
            click.echo(f"Downloaded: {result}")
        except Exception as e:
            click.echo(f"Error downloading NAIP data: {e}", err=True)
            sys.exit(1)
    elif source == "overture":
        from geoai.download import download_overture_buildings

        click.echo(f"Downloading Overture {overture_type} data for bbox={coords}...")
        try:
            download_overture_buildings(
                bbox=tuple(coords),
                output=output,
                overture_type=overture_type,
            )
            click.echo(f"Downloaded: {output}")
        except Exception as e:
            click.echo(f"Error downloading Overture data: {e}", err=True)
            sys.exit(1)
    elif source == "stac":
        from geoai.download import pc_stac_download, pc_stac_search

        asset_list = (
            [a.strip() for a in assets.split(",") if a.strip()] if assets else None
        )
        click.echo(f"Searching STAC collection '{collection}' for bbox={coords}...")
        try:
            items = pc_stac_search(
                collection=collection,
                bbox=coords,
                time_range=time_range,
                max_items=max_items,
            )
            if not items:
                click.echo("No STAC items found matching the query.", err=True)
                sys.exit(1)
            result = pc_stac_download(
                items=items,
                output_dir=output,
                assets=asset_list,
            )
            if not result or not any(result.values()):
                click.echo("Error: No STAC assets were downloaded.", err=True)
                sys.exit(1)
            click.echo(f"Downloaded: {result}")
        except Exception as e:
            click.echo(f"Error downloading STAC data: {e}", err=True)
            sys.exit(1)


@main.group()
def pipeline():
    """Run and manage batch processing pipelines."""
    pass


@pipeline.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--input-dir", "-i", help="Input directory (overrides config).")
@click.option("--output-dir", "-o", help="Output directory (overrides config).")
@click.option("--max-workers", "-w", type=int, help="Number of parallel workers.")
@click.option("--checkpoint-dir", "-c", help="Checkpoint directory for resume support.")
@click.option(
    "--on-error",
    type=click.Choice(["skip", "fail"]),
    default=None,
    help="Error handling policy.",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress bars.")
@click.option("--json", "as_json", is_flag=True, help="Output summary as JSON.")
def run(
    config_path,
    input_dir,
    output_dir,
    max_workers,
    checkpoint_dir,
    on_error,
    quiet,
    as_json,
):
    """Run a pipeline from a JSON or YAML config file.

    CONFIG_PATH is the path to a .json or .yaml pipeline definition.
    YAML support requires PyYAML (pip install pyyaml).

    \b
    Examples:
        geoai pipeline run segmentation.yaml -i ./data -o ./results
        geoai pipeline run batch_config.json --max-workers 4 -c ./checkpoints
    """
    from geoai.pipeline import load_pipeline

    overrides = {}
    if max_workers is not None:
        overrides["max_workers"] = max_workers
    if checkpoint_dir is not None:
        overrides["checkpoint_dir"] = checkpoint_dir
    if on_error is not None:
        overrides["on_error"] = on_error
    if quiet or as_json:
        overrides["quiet"] = True

    try:
        pipe = load_pipeline(config_path, **overrides)
    except Exception as e:
        click.echo(f"Error loading pipeline config: {e}", err=True)
        sys.exit(1)

    kwargs = {}
    if input_dir:
        kwargs["input_dir"] = input_dir
    if output_dir:
        kwargs["output_dir"] = output_dir

    try:
        result = pipe.run(**kwargs)
    except Exception as e:
        click.echo(f"Pipeline error: {e}", err=True)
        sys.exit(1)

    summary = result.summary
    if as_json:
        payload = {
            **summary,
            "checkpoint_path": result.checkpoint_path,
            "failed_items": [
                {"input_path": item.get("input_path"), "error": error}
                for item, error in result.failed
            ],
        }
        click.echo(json.dumps(_to_json_value(payload), indent=2, allow_nan=False))
        return

    click.echo("\nPipeline complete:")
    click.echo(f"  Completed: {summary['completed']}")
    click.echo(f"  Failed:    {summary['failed']}")
    click.echo(f"  Skipped:   {summary['skipped']}")
    click.echo(f"  Duration:  {summary['total_duration']}s")

    if result.checkpoint_path:
        click.echo(f"  Checkpoint: {result.checkpoint_path}")

    if result.failed:
        click.echo("\nFailed items:")
        for item, error in result.failed[:10]:
            click.echo(f"  - {item.get('input_path', '?')}: {error}")
        if len(result.failed) > 10:
            click.echo(f"  ... and {len(result.failed) - 10} more")


@pipeline.command()
@click.argument("config_path", type=click.Path(exists=True))
def show(config_path):
    """Display a pipeline configuration.

    CONFIG_PATH is the path to a .json or .yaml pipeline definition.
    """
    from geoai.pipeline import load_pipeline

    try:
        pipe = load_pipeline(config_path)
    except Exception as e:
        click.echo(f"Error loading pipeline config: {e}", err=True)
        sys.exit(1)

    click.echo(f"Pipeline: {pipe.name}")
    click.echo(f"Workers:  {pipe.max_workers}")
    click.echo(f"On error: {pipe.on_error.value}")
    click.echo(f"Steps ({len(pipe.steps)}):")
    for i, step in enumerate(pipe.steps, 1):
        click.echo(f"  {i}. {step}")


if __name__ == "__main__":
    main()
