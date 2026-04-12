---
name: "cli-anything-geoai"
description: "AI-powered geospatial analysis CLI. Segment satellite imagery, detect objects, classify images, detect changes, download data, and manage raster/vector operations -- all from the command line with JSON output for agent integration."
---

# cli-anything-geoai

CLI harness for GeoAI -- AI-powered geospatial analysis from the command line.

## Prerequisites

- **GeoAI** must be installed: `pip install geoai-py`
- **PyTorch** for model inference: `pip install torch torchvision`
- **GDAL/rasterio** for raster operations (installed with geoai)

## Installation

```bash
cd agent-harness && pip install -e .
```

Verify: `cli-anything-geoai --version`

## Global Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON (for agent consumption) |
| `--project PATH` | Load a project file for stateful operations |
| `--device cpu\|cuda\|mps\|auto` | Force compute device |
| `--version` | Show version |

## Command Groups

### project -- Workspace management

```bash
cli-anything-geoai project new -n "analysis" -o project.json
cli-anything-geoai --project project.json project add-file image.tif
cli-anything-geoai --project project.json project info
cli-anything-geoai --project project.json project save
cli-anything-geoai --project project.json project list-files
cli-anything-geoai --project project.json project list-results
```

### raster -- Raster inspection and operations

```bash
cli-anything-geoai raster info image.tif
cli-anything-geoai raster stats image.tif --band 1
cli-anything-geoai raster tile image.tif -o tiles/ -s 512 --overlap 64
cli-anything-geoai raster vectorize mask.tif -o polygons.geojson
```

### vector -- Vector inspection and operations

```bash
cli-anything-geoai vector info buildings.geojson
cli-anything-geoai vector rasterize labels.geojson -t template.tif -o labels.tif -a class
```

### data -- Data discovery and download

```bash
cli-anything-geoai data sources
cli-anything-geoai data search --bbox "-84.0,35.9,-83.9,36.0" -c sentinel-2-l2a
cli-anything-geoai data download naip --bbox "-84.0,35.9,-83.9,36.0" -o naip.tif
cli-anything-geoai data download overture --bbox "-84.0,35.9,-83.9,36.0" -o buildings.geojson
```

### segment -- Image segmentation

```bash
# SAM automatic segmentation
cli-anything-geoai segment sam image.tif -o mask.tif

# Text-prompted segmentation (GroundedSAM)
cli-anything-geoai segment grounded-sam image.tif -o mask.tif -p "buildings"

# Semantic segmentation with a trained model
cli-anything-geoai segment semantic image.tif -m model.pth -o output.tif -n 5

# Train a segmentation model
cli-anything-geoai segment train -i images/ -l labels/ -o model_output/ --arch unet --backbone resnet50 --epochs 20

# List available models and architectures
cli-anything-geoai segment list-models
cli-anything-geoai segment list-architectures
```

### detect -- Object detection

```bash
# Run detection
cli-anything-geoai detect run image.tif -m model.pth -n 5 -ov detections.geojson

# Train a detector
cli-anything-geoai detect train -i images/ -l labels/ -o model_output/ -n 5 --epochs 30

# List available models
cli-anything-geoai detect list-models
```

### classify -- Image classification

```bash
# Train a classifier
cli-anything-geoai classify train -t train_dir/ -v val_dir/ -o model_output/ -m resnet50

# Predict
cli-anything-geoai classify predict image.jpg -m model.pth
```

### change -- Change detection

```bash
# Detect changes between two images
cli-anything-geoai change detect before.tif after.tif -o changes.tif --method changestar

# List methods
cli-anything-geoai change list-methods
```

### pipeline -- Batch processing

```bash
cli-anything-geoai pipeline run config.yaml -i input/ -o output/ -w 4
cli-anything-geoai pipeline show config.yaml
```

### session -- State management

```bash
cli-anything-geoai session status
cli-anything-geoai session undo
cli-anything-geoai session redo
cli-anything-geoai session history
```

### system-info -- Diagnostics

```bash
cli-anything-geoai system-info
```

## Agent Usage Guide

### JSON Output

Always use `--json` for machine-readable output:

```bash
result=$(cli-anything-geoai --json raster info image.tif)
```

JSON output returns either:
- Success: `{"key": "value", ...}` (varies by command)
- Error: `{"error": "message", "type": "ErrorType"}`

### Typical Agent Workflow

```bash
# 1. Inspect available data
cli-anything-geoai --json raster info satellite.tif

# 2. Check statistics
cli-anything-geoai --json raster stats satellite.tif --band 1

# 3. Run segmentation
cli-anything-geoai --json segment sam satellite.tif -o masks.tif

# 4. Vectorize results
cli-anything-geoai --json raster vectorize masks.tif -o buildings.geojson

# 5. Check results
cli-anything-geoai --json vector info buildings.geojson
```

### Error Handling

All commands return clear error messages. In `--json` mode, errors are structured:

```json
{"error": "Raster file not found: /path/to/file.tif", "type": "FileNotFoundError"}
```

### Interactive REPL

Running `cli-anything-geoai` without arguments enters the REPL for interactive use:

```
> project new -n "my_analysis"
> raster info satellite.tif
> segment sam satellite.tif -o mask.tif
> session status
> quit
```
