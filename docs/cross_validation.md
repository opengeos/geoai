# K-fold cross-validation

`geoai.cross_validation` runs a training callback for each reproducible fold,
stores its results, and pools out-of-fold (OOF) confusion counts. It depends only
on NumPy, so it works with custom tree-crown pipelines and different training
frameworks. `geoai.run_kfold` and `geoai.compute_fold_indices` are also available.

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

`run_kfold` orchestrates callbacks; existing GeoAI training functions keep their
current interfaces and internal random splits. A callback must honor the supplied
indices instead of making another random train/validation split. For a custom
PyTorch loop, create `torch.utils.data.Subset` objects over separate training and
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
not per-batch averages. The summary reports:

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

With `only_fold=None`, `overwrite=True` reruns every fold and discards old
manifest results. Existing callback artifacts such as checkpoints are not
deleted; the callback controls their reuse. Concurrent writes to the same
experiment directory are unsupported. Partial summaries report `complete=False`
and list `folds_completed`; only those folds contribute to the metrics.

## API reference

::: geoai.cross_validation
