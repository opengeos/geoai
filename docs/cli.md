# cli module

GeoAI provides a command-line interface (`geoai`) installed via the `geoai-py` package for inspecting raster and vector files, downloading geospatial data, and running batch processing pipelines.

## CLI Usage Guide

### Version and Help

Check the installed version or list available subcommands:

```bash
geoai --version
geoai --help
```

### Inspect Files (`geoai info`)

Display metadata for a raster (`.tif`, `.tiff`, `.img`, `.jp2`, `.vrt`, `.nc`, `.hdf`) or vector (`.geojson`, `.json`, `.shp`, `.gpkg`, `.parquet`, `.geoparquet`, `.fgb`, `.kml`) file:

```bash
geoai info path/to/image.tif
geoai info path/to/buildings.geojson --json
```

| Option | Description |
| ------ | ----------- |
| `--json` | Output file metadata as formatted JSON. |

If a file has an unrecognized extension, `geoai info` attempts to read it as a raster first and falls back to vector inspection.

### Download Geospatial Data (`geoai download`)

Download imagery or vector data from `naip`, `overture`, or `stac` (Microsoft Planetary Computer):

```bash
geoai download SOURCE --bbox MINX,MINY,MAXX,MAXY -o OUTPUT [OPTIONS]
```

#### Common Options

| Option | Required | Description |
| ------ | -------- | ----------- |
| `SOURCE` | Yes | Data source: `naip`, `overture`, or `stac` (case-insensitive). |
| `--bbox TEXT` | Yes | Bounding box as `minx,miny,maxx,maxy` in EPSG:4326. |
| `-o`, `--output TEXT` | Yes | Output file or directory path. |

#### Source-Specific Options

| Option | Source | Default | Description |
| ------ | ------ | ------- | ----------- |
| `--year INTEGER` | `naip` | `None` | Year of NAIP imagery to download. |
| `--overture-type TEXT` | `overture` | `building` | Overture Maps data type (for example `building`, `place`, `segment`). |
| `--collection TEXT` | `stac` | `sentinel-2-l2a` | STAC collection ID. |
| `--time-range TEXT` | `stac` | `None` | STAC time range as `start/end` (for example `2024-06-01/2024-08-31`). |
| `--assets TEXT` | `stac` | `None` | Comma-separated STAC asset keys to download (for example `B04,B03,B02`). |
| `--max-items INTEGER` | `stac` | `1` | Maximum number of STAC items to download (minimum `1`). |

#### Examples

```bash
# Download NAIP aerial imagery
geoai download naip --bbox -84.0,35.9,-83.9,36.0 -o ./naip_data --year 2023

# Download Overture Maps building footprints
geoai download overture --bbox -84.0,35.9,-83.9,36.0 -o buildings.geojson --overture-type building

# Download Sentinel-2 imagery via STAC
geoai download stac --bbox -84.0,35.9,-83.9,36.0 -o ./s2_data \
    --collection sentinel-2-l2a \
    --time-range 2024-06-01/2024-08-31 \
    --assets B04,B03,B02 \
    --max-items 2
```

### Batch Pipelines (`geoai pipeline`)

Run or inspect batch processing pipelines defined in JSON or YAML files. YAML configs require PyYAML (`pip install pyyaml`). See the [pipeline module](pipeline.md) guide for full configuration details.

#### Run a Pipeline (`geoai pipeline run`)

```bash
geoai pipeline run CONFIG_PATH [OPTIONS]
```

| Option | Description |
| ------ | ----------- |
| `-i`, `--input-dir TEXT` | Input directory (overrides config). |
| `-o`, `--output-dir TEXT` | Output directory (overrides config). |
| `-w`, `--max-workers INTEGER` | Number of parallel worker threads. |
| `-c`, `--checkpoint-dir TEXT` | Checkpoint directory for resume support. |
| `--on-error [skip\|fail]` | Error handling policy (`skip` continues on failed items; `fail` aborts on the first failure, cancelling pending tasks while already-running worker threads finish and may still write outputs). |
| `-q`, `--quiet` | Suppress progress bars. |
| `--json` | Output run summary (including `checkpoint_path` and `failed_items`) as JSON. |

```bash
geoai pipeline run segmentation.yaml -i ./data -o ./results -w 4 -c ./checkpoints
geoai pipeline run batch_config.json -i ./masks -o ./vectors --on-error skip --json
```

#### Inspect a Pipeline (`geoai pipeline show`)

Display the name, worker count, error policy, and ordered steps of a pipeline configuration file:

```bash
geoai pipeline show segmentation.yaml
```

## API Reference

::: geoai.cli
