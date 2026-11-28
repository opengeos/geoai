# pipeline module

The `geoai.pipeline` module provides a batch processing framework for chaining geospatial AI steps (file discovery, inference, post-processing, and vectorization) with thread-based parallelism, YAML/JSON configuration files, and resume-from-checkpoint support.

## Quickstart

### Python API

```python
from geoai.pipeline import (
    GlobStep,
    Pipeline,
    RasterToVectorStep,
    SemanticSegmentationStep,
)

pipe = Pipeline(
    name="building_segmentation",
    steps=[
        GlobStep(extensions=[".tif", ".tiff"]),
        SemanticSegmentationStep(
            model_path="path/to/model.pth",
            architecture="unet",
            encoder_name="resnet34",
            num_channels=3,
            num_classes=2,
            window_size=512,
            overlap=256,
        ),
        RasterToVectorStep(output_format=".geojson", simplify_tolerance=1.0),
    ],
    max_workers=2,
    on_error="skip",
    checkpoint_dir="./checkpoints",
)

result = pipe.run(input_dir="./data/images", output_dir="./data/results")
print(result.summary)
```

For quick ad-hoc transformations, wrap any callable taking and returning an item `dict` with `FunctionStep`:

```python
from geoai.pipeline import FunctionStep, Pipeline


def tag_item(item):
    item["processed"] = True
    return item


pipe = Pipeline(steps=[FunctionStep("tag", tag_item)])
result = pipe.run(items=[{"input_path": "scene_01.tif"}])
```

You can also export a pipeline to YAML or JSON (`pipe.to_yaml("pipeline.yaml")`, `pipe.to_json("pipeline.json")`) and reload it later with `load_pipeline()`:

```python
from geoai.pipeline import load_pipeline

pipe = load_pipeline("semantic_segmentation.yaml", max_workers=4)
result = pipe.run(input_dir="./data/images", output_dir="./data/results")
```

### YAML and JSON Configuration Files

Pipelines composed of registered step types can be declared in YAML (requires PyYAML, `pip install pyyaml`) or JSON and executed from Python or the [CLI](cli.md).

#### YAML Example

```yaml
name: semantic_segmentation
max_workers: 2
executor_type: thread
on_error: skip

steps:
  - type: GlobStep
    name: find_images
    extensions:
      - ".tif"
      - ".tiff"

  - type: SemanticSegmentationStep
    name: segment
    model_path: "path/to/model.pth"
    architecture: unet
    encoder_name: resnet34
    num_channels: 3
    num_classes: 2
    window_size: 512
    overlap: 256
    batch_size: 4
    suffix: "_mask"

  - type: RasterToVectorStep
    name: vectorize
    output_format: ".geojson"
    input_key: output_path
    output_key: vector_path
```

#### JSON Example

```json
{
  "name": "batch_vectorize",
  "max_workers": 4,
  "executor_type": "thread",
  "on_error": "skip",
  "steps": [
    {
      "type": "GlobStep",
      "name": "find_masks",
      "extensions": [".tif"]
    },
    {
      "type": "RasterToVectorStep",
      "name": "vectorize",
      "output_format": ".gpkg",
      "input_key": "input_path",
      "output_key": "vector_path"
    }
  ]
}
```

Run or inspect config files from the command line:

```bash
geoai pipeline show semantic_segmentation.yaml
geoai pipeline run semantic_segmentation.yaml -i ./data/images -o ./data/results -c ./checkpoints -w 4
```

## Built-in Step Types

All six domain steps below are pre-registered in the config step registry and can be used directly in YAML/JSON configs by their class name:

| Step Class | Config `type` | Purpose |
| ---------- | ------------- | ------- |
| `GlobStep` | `GlobStep` | Expand `input_dir` or `input_pattern` into per-file work items (default extensions: `.tif`, `.tiff`, `.jp2`, `.img`). |
| `SemanticSegmentationStep` | `SemanticSegmentationStep` | Run sliding-window semantic segmentation using `geoai.train.semantic_segmentation`. |
| `RasterToVectorStep` | `RasterToVectorStep` | Convert raster masks to vector polygons (`.geojson`, `.gpkg`, etc.) with optional geometry simplification. |
| `ONNXInferenceStep` | `ONNXInferenceStep` | Load an ONNX model once in `setup()` and run tiled inference via `geoai.onnx.ONNXGeoModel`. |
| `CanopyHeightStep` | `CanopyHeightStep` | Load a canopy height model once in `setup()` and predict vegetation height via `geoai.canopy.CanopyHeightEstimation`. |
| `CleanMaskStep` | `CleanMaskStep` | Post-process segmentation masks with morphological cleaning via `geoai.tools.multiclean.clean_raster`. |

In Python code, you can also use `FunctionStep` for plain callables or subclass `PipelineStep` and decorate it with `@register_step` to make custom steps loadable from YAML/JSON configs.

## Checkpointing and Resume

Pass `checkpoint_dir` to `Pipeline` (or `-c` / `--checkpoint-dir` on the CLI) to enable resume support for long-running batch jobs:

- Progress is persisted atomically to `<checkpoint_dir>/<name>_checkpoint.json` every 10 processed items and when the run finishes or aborts.
- Each item is keyed by `item["input_path"]` by default (customizable via `item_key_fn`).
- On subsequent runs, items marked `completed` are skipped automatically and recorded in `result.skipped`.
- A hash of the pipeline name, step types, and step parameters is stored in the checkpoint. If the pipeline configuration changes, existing checkpoint progress is reset automatically so stale outputs are not reused.

## Error Handling and Execution

`Pipeline` executes items using a thread pool (`executor_type="thread"`, controlled by `max_workers`) so GPU models loaded in `setup()` can be shared safely without cross-process pickling. Two error policies are supported via `on_error` (or `--on-error`):

- **`"skip"` (`ErrorPolicy.SKIP`, default)**: Logs the error, records `(item, error_message)` in `result.failed`, marks the item as failed in the checkpoint, and continues processing the remaining items.
- **`"fail"` (`ErrorPolicy.FAIL`)**: Saves the checkpoint immediately and raises a `RuntimeError` on the first failed item.

## Example Pipeline Configurations

Ready-to-use sample configurations are available in the repository under [`docs/examples/pipelines/`](https://github.com/opengeos/geoai/tree/main/docs/examples/pipelines):

- [`semantic_segmentation.yaml`](https://github.com/opengeos/geoai/blob/main/docs/examples/pipelines/semantic_segmentation.yaml): Glob GeoTIFFs, run U-Net semantic segmentation, and export GeoJSON polygons.
- [`building_extraction.yaml`](https://github.com/opengeos/geoai/blob/main/docs/examples/pipelines/building_extraction.yaml): Multi-worker building footprint extraction with GeoPackage output and geometry simplification.
- [`batch_vectorize.json`](https://github.com/opengeos/geoai/blob/main/docs/examples/pipelines/batch_vectorize.json): Parallel raster-to-GeoPackage vectorization for pre-computed masks.
- [`glob_only.json`](https://github.com/opengeos/geoai/blob/main/docs/examples/pipelines/glob_only.json): Minimal file-discovery pipeline template.

## API Reference

::: geoai.pipeline
