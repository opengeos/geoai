"""Command-line interface for geoai."""

import os
import sys

import click


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
def info(filepath):
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
        for key, value in info_dict.items():
            click.echo(f"{key}: {value}")
    elif ext in raster_extensions:
        from geoai.utils import get_raster_info

        info_dict = get_raster_info(filepath)
        for key, value in info_dict.items():
            click.echo(f"{key}: {value}")
    else:
        click.echo(
            f"Unknown file extension '{ext}'. Attempting to read as raster...",
            err=True,
        )
        try:
            from geoai.utils import get_raster_info

            info_dict = get_raster_info(filepath)
            for key, value in info_dict.items():
                click.echo(f"{key}: {value}")
        except Exception:
            click.echo(
                f"Could not read '{filepath}' as raster. "
                "Attempting to read as vector...",
                err=True,
            )
            try:
                from geoai.utils import get_vector_info

                info_dict = get_vector_info(filepath)
                for key, value in info_dict.items():
                    click.echo(f"{key}: {value}")
            except Exception as e:
                click.echo(f"Error: Could not read file: {e}", err=True)
                sys.exit(1)


@main.command()
@click.argument("source", type=click.Choice(["naip"], case_sensitive=False))
@click.option(
    "--bbox",
    required=True,
    help="Bounding box as minx,miny,maxx,maxy (EPSG:4326).",
)
@click.option("--output", "-o", required=True, help="Output file path.")
@click.option("--year", default=None, type=int, help="Year of imagery.")
def download(source, bbox, output, year):
    """Download geospatial data from supported sources.

    SOURCE is the data source to download from (e.g., naip).
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

    if source == "naip":
        from geoai.download import download_naip

        kwargs = {}
        if year is not None:
            kwargs["year"] = year

        click.echo(f"Downloading NAIP imagery for bbox={coords}...")
        try:
            result = download_naip(bbox=coords, output=output, **kwargs)
            click.echo(f"Downloaded: {result}")
        except Exception as e:
            click.echo(f"Error downloading NAIP data: {e}", err=True)
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
def run(
    config_path, input_dir, output_dir, max_workers, checkpoint_dir, on_error, quiet
):
    """Run a pipeline from a YAML/JSON config file.

    CONFIG_PATH is the path to a .json or .yaml pipeline definition.

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
    if quiet:
        overrides["quiet"] = quiet

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
    click.echo(f"\nPipeline complete:")
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
