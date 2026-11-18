# K-fold cross-validation

`geoai.cross_validation` runs a training callback for each reproducible fold,
stores its results, and pools out-of-fold (OOF) confusion counts. It depends only
on NumPy, so it works with custom tree-crown pipelines and different training
frameworks. `geoai.run_kfold` and `geoai.compute_fold_indices` are also available.

For an executable dataset walkthrough, open the
[K-fold cross-validation notebook](examples/kfold_cross_validation.ipynb). It
uses built-in U-Net training, performance plots, batch inference, and prediction
comparisons. Configure your image/mask paths and select a Python environment
with the local GeoAI checkout, PyTorch, and segmentation-models-pytorch installed.

## Built-in semantic segmentation training

Add `k_folds` to the usual training call; no callback is needed:

```python
import geoai
from pathlib import Path
import json

summary = geoai.train_segmentation_model(
    images_dir="data/img", labels_dir="data/mask", output_dir="runs/landslide_cv",
    architecture="unet", encoder_name="resnet34", encoder_weights="imagenet",
    num_channels=3, num_classes=2, batch_size=4, num_epochs=20,
    learning_rate=0.001, k_folds=5, monitor="val_loss", plot_curves=True,
)

for fold in summary["per_fold"]:
    directory = Path(fold["output_dir"])
    geoai.plot_performance_metrics(str(directory / "training_history.pth"))
    split = json.loads((directory / "split.json").read_text())
    geoai.semantic_segmentation_batch(
        input_dir="data/img", input_files=split["held_out_files"],
        output_dir=fold["prediction_dir"], model_path=fold["model_path"],
        architecture="unet", encoder_name="resnet34", num_channels=3, num_classes=2,
        skip_existing=True,
    )
```

Every fold initializes a fresh model, trains on K-1 folds, and validates on the
remaining fold. K determines the held-out fraction (approximately 1/K): five
folds use roughly 80% training / 20% validation, while two folds use 50% / 50%.
There is no additional inner holdout, and `val_split` is ignored in k-fold mode.
`fold_groups` keeps each supplied group in one fold; unequal group sizes can
produce unequal tile counts.
Directory-format inputs must have matching image/mask stems in sorted order.

Before training each fold, GeoAI prints the tile counts and percentages for
training and held-out validation/OOF evaluation. With `verbose=True`,
it also lists every tile's filename and zero-based dataset index under its role,
plus site/scene groups when supplied. Each fold's `split.json` saves the complete
image paths, paired mask paths, indices, and optional groups for both sets:
`fit_*` (training) and `held_out_*` (validation and OOF).
After held-out inference, verbose output reports each class's fold OOF metrics
and pixel confusion counts. The final summary still pools counts across folds.

`monitor="val_loss"` minimizes held-out validation loss; `monitor="val_iou"`
maximizes held-out validation IoU and remains the default. The selected criterion
also controls early stopping. Each fold's `model_selection.json` records the
criterion, score, and selected epoch; `best_model.pth` keeps the usual weights
format. Choose the criterion before training. Changing it requires a new run;
an existing IoU-selected checkpoint cannot be reclassified as loss-selected.

The same held-out fold selects the checkpoint, controls early stopping and the
learning-rate scheduler, and supplies OOF predictions. Its images are excluded
from weight updates, but the reported OOF metrics are CV validation results,
not an independent test of checkpoint selection. Use a separate test dataset
for final performance assessment. The manifest records `held_out_fold` as the
validation strategy; single-fold reruns cannot mix results from the previous
inner-holdout strategy with this one. Start a new experiment for the new strategy.

Training writes normal GeoAI checkpoints and histories in `fold_<i>/`, calls
`semantic_segmentation_batch` on only its held-out files, and scores saved masks
at native resolution. Binary OOF metrics are under `oof.micro.class_1` and
`oof.macro.class_1`; multiclass mode reports one-versus-rest counts for every
class. Binary scoring handles both 0/1 GeoTIFF predictions and 0/255 PNG
predictions. These differ from training-history metrics averaged over classes.
Held-out inference uses GeoAI's standard inference preprocessing; custom training
transforms must produce inputs consistent with that inference pipeline.

Without `k_folds`, the original random-split behavior and None return are
preserved. `only_fold=0, overwrite=True` reruns one compatible fold. Full training
state resumption is not supported in CV mode. Fold model paths are returned in
`summary["per_fold"]`; no single deployment model is selected automatically.

## Ensemble inference

After all folds complete, `summary["ensemble_model_paths"]` contains each fold's
selected checkpoint. Pass that list to either `semantic_segmentation` or
`semantic_segmentation_batch` to average class probabilities with uniform model
weights, then apply argmax or a probability threshold:

```python
geoai.semantic_segmentation_batch(
    input_dir="data/new_images", output_dir="runs/landslide_cv/ensemble_predictions",
    model_path=summary["ensemble_model_paths"],
    architecture="unet", encoder_name="resnet34", num_channels=3, num_classes=2,
    probability_threshold=0.5, raise_on_error=True,
)
```

This averages probabilities, not logits or thresholded masks. Single checkpoint
paths retain their previous behavior. All models must share architecture,
channels, classes, and input preprocessing. Models are loaded once and processed
sequentially per image batch; GPU memory includes every member's weights.

Partial CV runs return an empty ensemble list until all folds are complete.
OOF scoring continues to use only the model that held each image out. Applying
the full ensemble to CV training imagery includes models that trained on those
images, so those predictions must not replace OOF metrics. Evaluate the ensemble
on independent test imagery for an unbiased ensemble assessment.

Batch inference's `input_files` selects exact files. `skip_existing=True` reuses
known outputs; set it to False to regenerate them. `model_kwargs` forwards extra
architecture parameters. CV enables `raise_on_error=True` so a failed inference
cannot silently produce incomplete OOF results; normal batch inference retains
its log-and-continue default.

## Connect your training pipeline

Keep your image and target lists in a fixed order. The callback receives
zero-based fold indices, training indices, validation indices, and a separate
output directory. Initialize a fresh model and optimizer for every fold, train
on the supplied training indices, and predict each held-out sample once.

The following adapter illustrates the contract; `fit_model`, `predict_mask`,
and `read_mask` are functions supplied by your pipeline:

```python
from pathlib import Path
from geoai.cross_validation import run_kfold, update_counts, validate_paired_files

images = sorted(Path("data/images").glob("*.tif"))
masks = sorted(Path("data/masks").glob("*.tif"))
validate_paired_files(
    {"images": images, "masks": masks},
    index_fn=lambda path: Path(path).stem,
)

def train_fold(fold_idx, train_indices, val_indices, fold_dir):
    model = fit_model(
        [images[i] for i in train_indices],
        [masks[i] for i in train_indices],
        output_dir=fold_dir,
        seed=42 + fold_idx,
    )
    counts = {}
    for i in val_indices:
        probabilities = predict_mask(model, images[i])
        target = read_mask(masks[i])
        update_counts(counts, probabilities, target, threshold=0.5)
    return {"oof": {"crown_mask": counts}}

summary = run_kfold(
    n_samples=len(images), k=5, seed=42,
    experiment_dir="runs/tree_crowns_cv",
    train_fold=train_fold,
    sample_names=[str(path.resolve()) for path in images],
    metadata={"model": "unet", "threshold": 0.5, "training_seed": 42},
)
print(summary["oof"]["micro"]["crown_mask"])
```

The default `validate_paired_files` key is the last underscore-separated token
of the stem (`image_12.tif` pairs with `mask_12.tif`). Supply an `index_fn` for
your naming convention; duplicate keys and misordered pairs raise errors.

`run_kfold` orchestrates callbacks. `geoai.train_segmentation_model` accepts
`train_indices` and `val_indices` to bypass its internal random split; indices
refer to its discovered file lists (sorted for directory input). Both lists must
be non-empty, unique, in range, and disjoint. Positions in neither list are
excluded from training, allowing an independent held-out fold. Without explicit
indices or `k_folds`, its existing `val_split` behavior is preserved. A callback must honor
the supplied indices instead of making another random train/validation split.
For a custom PyTorch loop, create `torch.utils.data.Subset` objects over separate training and
validation dataset views, using augmentation only for the training view. Set
framework seeds inside the callback and save checkpoints under `fold_dir`.

## Keep sites or scenes together

Pass `groups`, a list of strings in the same order as the samples, to keep all
tiles from a site or source scene in one fold. For example:

```python
from geoai import compute_fold_indices

folds = compute_fold_indices(
    6, k=3, seed=42,
    groups=["site_a", "site_a", "site_b", "site_b", "site_c", "site_c"],
)
```

`run_kfold` accepts the same `groups` argument. Groups are balanced by group
count, so folds may have different sample counts. Grouping does not infer spatial
separation: choose identifiers that keep neighboring or overlapping imagery
together. Splits are not stratified by class.

## Pixel and instance metrics

`update_counts` accumulates binary pixel `tp`, `fp`, `fn`, and `tn`. Supply
probabilities, not logits, and targets encoded as 0/1. Targets above 0.5 are
positive. Non-finite targets are ignored; use `valid` to exclude finite nodata or
ignore-index pixels. CPU tensors are accepted; detach and move GPU tensors to
CPU first. For multiclass masks, accumulate a separate one-versus-rest count
dictionary per class.

For delineated tree instances, use integer label rasters with background 0:

```python
from geoai.cross_validation import add_counts, instance_counts_from_labels

instance_counts = {}
for gt_labels, predicted_labels in held_out_label_pairs:
    add_counts(instance_counts, instance_counts_from_labels(
        gt_labels, predicted_labels, iou_threshold=0.5,
    ))
# Include in the callback result:
result = {"oof": {"crowns": instance_counts}}
```

Instance IDs may be non-consecutive. Each object matches at most once; IoU
thresholds from 0.5 through 1 are supported. Instance `iou` is
`tp / (tp + fp + fn)` over objects. `mean_matched_iou` measures mask overlap
over matched objects. This is not COCO average precision or panoptic quality.

Every fold must return the same target names and count fields. Return raw counts,
not per-batch averages. NumPy scalar counts and metrics are converted to Python
values before validation and persistence; unsupported non-JSON objects still
raise an error. The summary reports:

- **Micro:** sum the held-out counts across folds, then calculate IoU, precision,
  recall, F1, and (for instances) mean matched IoU.
- **Macro:** average those metrics equally across folds.
- **Fold means:** average other numeric callback values, such as `best_val_loss`.

Metrics with a zero denominator are reported as zero. OOF scoring assumes the
callback evaluates precisely its held-out samples. If the same held-out fold
also selects checkpoints or hyperparameters, use nested CV or an independent
test set when an unbiased final evaluation is required.

## Results and individual fold reruns

Each experiment writes:

```text
kfold_manifest.json
fold_0/oof_metrics.json
fold_1/oof_metrics.json
...
kfold_summary.json
```

The manifest records assignments, optional sample names and groups, metadata,
and each completed result. JSON files are replaced atomically. If training
fails, completed results remain in the manifest and a stale summary is removed.

To train or retrain a single fold, use `only_fold=2` (zero-based). Reusing a
non-empty directory requires `overwrite=True`. Other folds' results are retained
only when the entire experiment identity matches: sample count, seed, fold
assignments, sample names, groups, and metadata. Record your training recipe,
thresholds, and dataset version in metadata, and supply stable, unique sample
names. Without those, the runner cannot detect changes to your inputs or model.

Built-in segmentation records custom transforms and losses using qualified
callable/type names and recursively serialized configuration, including array
and tensor values. Equivalent newly created objects can therefore be reused
across processes, while changed configuration is rejected. Use a new experiment
when changing callable implementations or other training code: qualified names
alone cannot detect source-code changes. Cyclic or unsupported configuration
raises an error instead of falling back to an object address.

With `only_fold=None`, `overwrite=True` reruns every fold and discards old
manifest results. Existing callback artifacts such as checkpoints are not
deleted; the callback controls their reuse. Concurrent writes to the same
experiment directory are unsupported. Partial summaries report `complete=False`
and list `folds_completed`; only those folds contribute to the metrics.

## API reference

::: geoai.cross_validation
