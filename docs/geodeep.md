# GeoDeep - Object Detection & Segmentation

[GeoDeep](https://github.com/uav4geo/GeoDeep) is a lightweight library for applying ONNX AI models to geospatial imagery. It supports **object detection** (bounding boxes) and **semantic segmentation** (pixel masks) with built-in models for common remote sensing tasks.

- **CPU & GPU**: Runs on ONNX Runtime — CPU by default, NVIDIA CUDA when available
- Input: GeoTIFF rasters
- Output: GeoDataFrame, GeoJSON, GeoPackage, Shapefile, GeoTIFF masks

## Installation

Install GeoAI with the GeoDeep extra:

```bash
pip install geoai-py[geodeep]          # CPU only
pip install geoai-py[geodeep-gpu]      # GPU (CUDA) support
```

Or install GeoDeep standalone:

```bash
pip install geodeep                    # CPU
pip install geodeep onnxruntime-gpu    # GPU
```

## Available Models

| Model ID | Type | Description | Resolution |
|---|---|---|---|
| `cars` | Detection | Car detection (YOLOv7-m) | 10 cm/px |
| `trees` | Detection | Tree crown detection (RetinaNet) | 10 cm/px |
| `trees_yolov9` | Detection | Tree crown detection (YOLOv9) | 10 cm/px |
| `birds` | Detection | Bird detection (RetinaNet) | 2 cm/px |
| `planes` | Detection | Plane detection (YOLOv7-tiny) | 70 cm/px |
| `aerovision` | Detection | Multi-class aerial detection (YOLOv8) — vehicles, pools, fields, courts, bridges, etc. | 30 cm/px |
| `utilities` | Detection | Utility infrastructure (YOLOv8) — Gas, Manhole, Power, Sewer, Telecom, Water | 3 cm/px |
| `buildings` | Segmentation | Building footprint segmentation (XUNet) | 50 cm/px |
| `roads` | Segmentation | Road network segmentation | 21 cm/px |

Models are automatically downloaded and cached on first use.

## Quick Start

### Object Detection

```python
from geoai import GeoDeep

gd = GeoDeep("cars")
detections = gd.detect("aerial_image.tif")
print(f"Found {len(detections)} cars")
detections.head()
```

The result is a GeoDataFrame with `geometry` (bounding box polygons in EPSG:4326), `score` (confidence), and `class` (label) columns.

### Semantic Segmentation

```python
gd = GeoDeep("buildings")
result = gd.segment("satellite.tif", output_raster_path="buildings_mask.tif")
print(result["mask"].shape)  # (height, width) uint8 array
```

### Save Detection Results

```python
gd = GeoDeep("cars")
gd.detect("image.tif", output_path="detections.geojson")
```

Supported output formats: `.geojson`, `.gpkg`, `.shp`, `.parquet`.

## Usage Examples

### Detection with Confidence Filtering

```python
gd = GeoDeep("aerovision", conf_threshold=0.7)
detections = gd.detect(
    "image.tif",
    classes=["small-vehicle", "plane"],
)
```

### Batch Detection

Process multiple images at once with combined results:

```python
gd = GeoDeep("trees")
results = gd.detect_batch(
    ["area1.tif", "area2.tif", "area3.tif"],
    output_dir="results/",
)
print(f"Total detections: {len(results)}")
# results includes a 'source_file' column to track which image each detection came from
```

### Batch Segmentation

```python
gd = GeoDeep("roads")
results = gd.segment_batch(
    ["tile1.tif", "tile2.tif"],
    output_dir="masks/",
    output_format="raster",  # or "vector" or "both"
)
```

### Segmentation to Vector

Export segmentation results as vector polygons:

```python
gd = GeoDeep("buildings")
result = gd.segment(
    "city.tif",
    output_vector_path="buildings.gpkg",
)
buildings_gdf = result["gdf"]
print(f"Found {len(buildings_gdf)} building polygons")
```

### List Available Models

```python
from geoai import list_geodeep_models

for name, desc in list_geodeep_models().items():
    print(f"{name}: {desc}")
```

### Using Convenience Functions

For one-off calls without creating a class instance:

```python
from geoai import geodeep_detect, geodeep_segment

# Detection
detections = geodeep_detect("image.tif", model_id="planes")

# Segmentation
result = geodeep_segment(
    "image.tif",
    model_id="roads",
    output_raster_path="roads_mask.tif",
)
```

### Custom ONNX Models

You can use a custom ONNX model file instead of a built-in model:

```python
gd = GeoDeep("/path/to/custom_model.onnx")
detections = gd.detect("image.tif")
```

!!! example "Related Examples"
    - [GeoDeep Detection](examples/geodeep_detection.ipynb)

## API Reference

::: geoai.geodeep
