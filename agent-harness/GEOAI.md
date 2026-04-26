# GeoAI: Agent Harness SOP

## Software Overview

**GeoAI** is a Python package for integrating artificial intelligence with geospatial
data analysis. It provides high-level APIs for satellite imagery segmentation, object
detection, image classification, change detection, and geospatial data download/processing.

- **Package**: `geoai-py` (PyPI) / `geoai` (conda-forge)
- **Version**: 0.37.1+
- **Backend**: PyTorch, Transformers, segmentation_models_pytorch, rasterio, geopandas
- **Native formats**: GeoTIFF, GeoJSON, Shapefile, GeoPackage, COCO JSON, YOLO

## Backend Engine

GeoAI is itself the backend. Unlike GUI wrappers (GIMP, Blender), this CLI wraps a
Python library directly. The backend is invoked via Python imports, not subprocess calls.

**Core dependencies:**
- `torch` + `torchvision` -- model training and inference
- `rasterio` -- GeoTIFF I/O, windowed reading, CRS handling
- `geopandas` -- vector I/O (GeoJSON, Shapefile, GeoPackage)
- `transformers` -- HuggingFace models (SAM, CLIP, DINO)
- `segmentation_models_pytorch` -- UNet, DeepLabV3, FPN architectures
- `segment-anything` -- SAM model family

## GUI-to-CLI Action Mapping

| GUI/Notebook Action | CLI Command | Backend Function |
|---------------------|-------------|------------------|
| Load raster in map | `raster info <path>` | `geoai.utils.get_raster_info()` |
| Load vector layer | `vector info <path>` | `geoai.utils.get_vector_info()` |
| Run SAM segmentation | `segment sam <raster> -o <output>` | `SamGeo.generate()` |
| Text-prompted segment | `segment grounded-sam <raster> -p <prompt>` | `GroundedSAM.predict()` |
| Train segmentation | `segment train <config>` | `train_segmentation_model()` |
| Run semantic segmentation | `segment semantic <raster> -m <model>` | `semantic_segmentation()` |
| Train object detector | `detect train <images> <labels>` | `train_MaskRCNN_model()` |
| Run detection | `detect run <raster> -m <model>` | `multiclass_detection()` |
| Train classifier | `classify train <dir>` | `train_image_classifier()` |
| Predict classification | `classify predict <image> -m <model>` | `predict_with_timm()` |
| Change detection | `change detect <img1> <img2>` | `changestar_detect()` |
| Download NAIP | `data download naip --bbox <bbox>` | `download_naip()` |
| Search STAC | `data search --bbox <bbox> --collection <name>` | `pc_stac_search()` |
| Create tiles | `data tile <raster> -o <dir>` | `export_geotiff_tiles()` |
| Raster to vector | `raster vectorize <raster> -o <output>` | `raster_to_vector()` |
| Vector to raster | `vector rasterize <vector> -t <template>` | `vector_to_raster()` |
| Run pipeline | `pipeline run <config>` | `Pipeline.run()` |

## Data Model

### Project File (JSON)

```json
{
  "name": "my_project",
  "description": "",
  "created": "2026-04-11T10:00:00",
  "modified": "2026-04-11T12:00:00",
  "crs": "EPSG:4326",
  "files": [
    {
      "id": 0,
      "path": "/data/image.tif",
      "type": "raster",
      "metadata": {"bands": 4, "crs": "EPSG:32617", "shape": [1024, 1024]}
    }
  ],
  "results": [
    {
      "id": 0,
      "type": "segmentation",
      "input_file_id": 0,
      "output_path": "/results/masks.tif",
      "model": "facebook/sam-vit-huge",
      "parameters": {"automatic": true}
    }
  ],
  "models": [
    {
      "id": 0,
      "name": "sam-vit-huge",
      "type": "segmentation",
      "source": "facebook/sam-vit-huge"
    }
  ]
}
```

### Supported File Formats

**Raster input**: GeoTIFF (.tif/.tiff), IMG (.img), JPEG2000 (.jp2), VRT (.vrt)
**Vector input**: GeoJSON (.geojson/.json), Shapefile (.shp), GeoPackage (.gpkg),
                  Parquet (.parquet), FlatGeoBuf (.fgb), KML (.kml)
**Model checkpoints**: PyTorch (.pth/.pt), Lightning (.ckpt), ONNX (.onnx)
**Pipeline configs**: JSON (.json), YAML (.yaml/.yml)

## CLI Architecture

### Command Groups

1. **project** -- Workspace management (new, open, save, info, add-file, list-files)
2. **raster** -- Raster inspection and operations (info, stats, vectorize, tile)
3. **vector** -- Vector inspection and operations (info, rasterize)
4. **data** -- Data discovery and download (search, download, tile)
5. **segment** -- Image segmentation (sam, grounded-sam, semantic, train)
6. **detect** -- Object detection (run, train)
7. **classify** -- Image classification (train, predict)
8. **change** -- Change detection (detect)
9. **pipeline** -- Batch processing (run, show)
10. **session** -- State management (status, undo, redo, history)

### Global Options

- `--json` -- Machine-readable JSON output for all commands
- `--project <path>` -- Load a project file for stateful operations
- `--device <cpu|cuda|mps>` -- Force compute device selection

## Existing CLI

GeoAI ships a minimal Click CLI (`geoai.cli:main`) with only `info`, `download naip`,
`pipeline run`, and `pipeline show`. The agent harness extends this dramatically with
full segmentation, detection, classification, change detection, and project management.
