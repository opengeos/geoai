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

        get_vector_info(filepath)
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

                get_vector_info(filepath)
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


if __name__ == "__main__":
    main()
