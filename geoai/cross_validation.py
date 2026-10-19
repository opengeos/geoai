"""Generic k-fold cross-validation with pooled out-of-fold (OOF) metrics.

Provides reproducible folds, callback-based orchestration, and pooled pixel or
instance counts. Depends only on NumPy; model training stays in the callback.

How it works:

- ``compute_fold_indices`` shuffles ``range(n_samples)`` once with a seed and
  splits it into K contiguous validation chunks, so folds never overlap and the
  split is reproducible. Keep your sample list in a fixed (e.g. sorted) order.
- ``run_kfold`` trains one model per fold through your ``train_fold`` callback.
  The callback predicts each held-out sample exactly once and returns raw
  confusion counts. Counts, not ratios, are stored so they pool exactly.
- After every fold, the result is written into ``kfold_manifest.json``. That
  lets ``only_fold=i`` retrain a single fold and still produce a full summary
  from the folds already on disk.
- The summary reports micro OOF metrics (counts pooled across folds, then
  scored) and macro OOF metrics (the mean of per-fold scores).

Usage::

    from geoai.cross_validation import run_kfold, update_counts

    def train_fold(fold_idx, train_indices, val_indices, fold_dir):
        model = train(train_indices)                    # your code
        counts = {}
        for i in val_indices:                           # each held-out sample once
            update_counts(counts, predict(model, i), target(i), threshold=0.5)
        return {"oof": {"mask": counts}, "best_val_loss": 0.12}

    summary = run_kfold(n_samples, k=5, seed=1337, experiment_dir="runs/exp1",
                        train_fold=train_fold, sample_names=names)

Experiment directory layout::

    kfold_manifest.json       fold assignment, metadata, each finished fold's result
    fold_<i>/oof_metrics.json scored OOF metrics of fold i
    kfold_summary.json        pooled micro + macro OOF metrics, fold means
"""

from __future__ import annotations

import json
import logging
from numbers import Integral
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "compute_fold_indices",
    "sample_index",
    "validate_paired_files",
    "validate_threshold",
    "update_counts",
    "instance_counts_from_labels",
    "add_counts",
    "metrics_from_counts",
    "summarize_oof",
    "run_kfold",
]

MANIFEST_NAME = "kfold_manifest.json"
SUMMARY_NAME = "kfold_summary.json"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


def compute_fold_indices(
    n_samples: int, k: int, seed: int, *, groups: list[str] | None = None
) -> list[dict[str, list[int]]]:
    """Return reproducible train/validation indices covering every sample once.

    Args:
        n_samples: Number of samples in a fixed input order.
        k: Number of folds, between two and the number of samples or groups.
        seed: Seed for NumPy's local random generator.
        groups: Optional site/scene identifiers, one string per sample. All
            samples with the same identifier stay in one validation fold.
            Groups are shuffled and split evenly by group count, not tile count.

    Returns:
        A list of dictionaries with sorted ``train`` and ``val`` indices.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if k < 2:
        raise ValueError("k must be >= 2.")
    if k > n_samples:
        raise ValueError(f"k ({k}) cannot exceed number of samples ({n_samples}).")

    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(n_samples)
        fold_val_sets = [list(map(int, chunk)) for chunk in np.array_split(order, k)]
    else:
        if len(groups) != n_samples or any(not isinstance(g, str) for g in groups):
            raise ValueError("groups must contain one string per sample.")
        unique_groups = list(dict.fromkeys(groups))
        if k > len(unique_groups):
            raise ValueError("k cannot exceed the number of distinct groups.")
        fold_val_sets = []
        for chunk in np.array_split(rng.permutation(len(unique_groups)), k):
            held_out = {unique_groups[int(i)] for i in chunk}
            fold_val_sets.append(
                [i for i, group in enumerate(groups) if group in held_out]
            )

    folds: list[dict[str, list[int]]] = []
    for i in range(k):
        val_idx = fold_val_sets[i]
        val_set = set(val_idx)
        train_idx = [int(j) for j in range(n_samples) if j not in val_set]
        folds.append({"train": sorted(train_idx), "val": sorted(val_idx)})
    return folds


def sample_index(path: str | Path) -> str:
    """Default sample key: the last '_'-separated token of the file stem (tile_12.tif -> '12')."""
    return Path(path).stem.split("_")[-1]


def validate_paired_files(
    files_by_name: dict[str, list[str | Path]],
    reference: str | None = None,
    index_fn: Callable[[str | Path], str] = sample_index,
) -> None:
    """Check that parallel file lists (e.g. images, masks) pair up position by position.

    Every list must have the same length, the same set of sample keys, and the
    same order as the ``reference`` list (default: the first one). Fold
    indices refer to positions, so a misaligned list silently pairs the wrong
    target with an image.
    """
    if not files_by_name:
        raise ValueError("files_by_name must contain at least one file list.")
    if reference is not None and reference not in files_by_name:
        raise ValueError(f"Unknown reference file list: {reference!r}.")
    indices_by_name = {
        name: [index_fn(path) for path in files]
        for name, files in files_by_name.items()
    }
    counts = {name: len(indices) for name, indices in indices_by_name.items()}
    for name, indices in indices_by_name.items():
        if len(set(indices)) != len(indices):
            raise RuntimeError(
                f"Duplicate sample keys in {name}; use a unique index_fn."
            )
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            "Sample count mismatch across file lists: "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )

    reference_name = reference if reference is not None else next(iter(files_by_name))
    reference_indices = indices_by_name[reference_name]
    reference_set = set(reference_indices)
    problems = []
    for name, indices in indices_by_name.items():
        if name == reference_name:
            continue
        index_set = set(indices)
        missing = sorted(reference_set - index_set)
        extra = sorted(index_set - reference_set)
        if missing or extra:
            detail = [name]
            if missing:
                detail.append(f"missing indices {missing}")
            if extra:
                detail.append(f"extra indices {extra}")
            problems.append(": ".join([detail[0], "; ".join(detail[1:])]))
        if indices != reference_indices:
            first_bad = next(
                (
                    i
                    for i, (left, right) in enumerate(zip(reference_indices, indices))
                    if left != right
                ),
                None,
            )
            if first_bad is not None:
                problems.append(
                    f"{name}: first order mismatch at position {first_bad}: "
                    f"{reference_name} index {reference_indices[first_bad]} vs {name} index {indices[first_bad]}"
                )
    if problems:
        raise RuntimeError(
            "File lists are not paired by sample index. " + " | ".join(problems)
        )


# ---------------------------------------------------------------------------
# Counts and metrics
# ---------------------------------------------------------------------------


def validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold!r}.")
    return threshold


def update_counts(
    counts: dict, probabilities, targets, threshold: float = 0.5, valid=None
) -> dict:
    """Add the pixel confusion counts of one prediction to ``counts`` (start with {}).

    ``probabilities`` and ``targets`` are same-shaped arrays (numpy or CPU
    tensors). A target counts as positive when > 0.5. Non-finite target pixels,
    and pixels where ``valid`` is non-finite or <= 0.5, are ignored.
    """
    threshold = validate_threshold(threshold)
    probabilities = np.asarray(probabilities)
    pred = probabilities >= threshold
    targets = np.asarray(targets)
    if pred.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} != {targets.shape}."
        )
    true = targets > 0.5
    mask = np.isfinite(targets)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != targets.shape:
            raise ValueError(
                f"Valid-area and target shapes differ: {valid.shape} != {targets.shape}."
            )
        mask &= np.isfinite(valid) & (valid > 0.5)
    if not np.all(np.isfinite(probabilities[mask])):
        raise ValueError("Predictions must be finite on valid pixels.")
    for name, hits in (
        ("tp", mask & pred & true),
        ("fp", mask & pred & ~true),
        ("fn", mask & ~pred & true),
        ("tn", mask & ~pred & ~true),
    ):
        counts[name] = counts.get(name, 0) + int(hits.sum())
    return counts


def instance_counts_from_labels(
    gt_labels, pred_labels, iou_threshold: float = 0.5
) -> dict:
    """Match predicted instances to ground-truth instances by IoU.

    Both inputs are same-shaped nonnegative integer label arrays (0 =
    background). IDs may be non-consecutive. Each object is matched at most
    once, with IoU >= iou_threshold. Thresholds from 0.5 to 1 are supported;
    exact-0.5 ties are resolved greedily from highest IoU down. The ``iou``
    score derived from these counts measures object detection, not mask IoU.
    """
    if not np.isfinite(iou_threshold) or not 0.5 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0.5 and 1 inclusive.")
    gt = np.asarray(gt_labels)
    pred = np.asarray(pred_labels)
    if gt.shape != pred.shape:
        raise ValueError(
            f"GT and prediction label shapes differ: {gt.shape} != {pred.shape}."
        )

    for labels in (gt, pred):
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("Instance labels must be nonnegative integers.")

    def compact(labels):
        ids = np.unique(labels[labels > 0])
        return np.where(labels > 0, np.searchsorted(ids, labels) + 1, 0).ravel(), len(
            ids
        )

    gt, n_gt = compact(gt)
    pred, n_pred = compact(pred)
    if n_gt == 0 or n_pred == 0:
        return {"tp": 0, "fp": n_pred, "fn": n_gt, "matched_iou_sum": 0.0}

    gt_area = np.bincount(gt, minlength=n_gt + 1)
    pred_area = np.bincount(pred, minlength=n_pred + 1)

    # Pairwise intersection counts for pixels where both labels are non-zero.
    fg = (gt > 0) & (pred > 0)
    # Store only overlapping pairs, avoiding a dense objects-by-objects matrix.
    pairs, intersection = np.unique(
        np.column_stack((gt[fg], pred[fg])), axis=0, return_counts=True
    )
    union = gt_area[pairs[:, 0]] + pred_area[pairs[:, 1]] - intersection
    iou = intersection / union

    # Each object may be matched at most once. Assign greedily from the highest
    # IoU down; at threshold >= 0.5 this is optimal and also resolves exact-0.5
    # ties where one prediction overlaps two GT objects (a merge error).
    candidates = np.flatnonzero(iou >= iou_threshold)
    order = candidates[np.argsort(-iou[candidates], kind="stable")]
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    tp = 0
    matched_iou_sum = 0.0
    for k in order:
        g, p = map(int, pairs[k])
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        tp += 1
        matched_iou_sum += float(iou[k])
    fp = n_pred - len(used_pred)
    fn = n_gt - len(used_gt)
    # matched_iou_sum lets folds pool correctly: mean IoU = sum / tp.
    return {"tp": tp, "fp": fp, "fn": fn, "matched_iou_sum": matched_iou_sum}


def add_counts(total: dict, counts: dict) -> dict:
    """Add every count in ``counts`` into ``total`` (start with {})."""
    for name, value in counts.items():
        total[name] = total.get(name, 0) + value
    return total


def metrics_from_counts(counts: dict) -> dict:
    """Score a counts dict: IoU, precision, recall, F1 (and mean matched IoU for instances)."""
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {**counts, "iou": iou, "precision": precision, "recall": recall, "f1": f1}
    if "matched_iou_sum" in counts:
        # Mean IoU over matched (true-positive) pairs: the boundary quality of
        # the objects that were detected, which F1 says nothing about.
        out["mean_matched_iou"] = counts["matched_iou_sum"] / tp if tp else 0.0
    return out


def summarize_oof(fold_oofs: list[dict[str, dict]]) -> dict:
    """Pool per-fold OOF counts ({target: counts}) into micro and macro metrics."""
    targets = list(dict.fromkeys(t for oof in fold_oofs for t in oof))
    micro = {}
    macro = {}
    for target in targets:
        per_fold = [oof[target] for oof in fold_oofs if target in oof]
        pooled: dict = {}
        for counts in per_fold:
            add_counts(pooled, counts)
        micro[target] = metrics_from_counts(pooled)
        scored = [metrics_from_counts(c) for c in per_fold]
        names = ["iou", "precision", "recall", "f1"]
        if "mean_matched_iou" in micro[target]:
            names.append("mean_matched_iou")
        macro[target] = {
            name: float(np.mean([m[name] for m in scored if name in m]))
            for name in names
        }
    return {"micro": micro, "macro": macro}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _validate_split_indices(
    n_samples: int, train_indices: list[int], val_indices: list[int]
) -> tuple[list[int], list[int]]:
    """Validate explicit disjoint subsets; omitted samples remain unused."""
    cleaned = []
    for name, indices in (
        ("train_indices", train_indices),
        ("val_indices", val_indices),
    ):
        values = list(indices)
        if not values:
            raise ValueError(f"{name} must not be empty.")
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
            raise ValueError(f"{name} must contain integer indices.")
        if any(i < 0 or i >= n_samples for i in values):
            raise ValueError(f"{name} contains indices outside [0, {n_samples - 1}].")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} contains duplicate indices.")
        cleaned.append([int(i) for i in values])
    if set(cleaned[0]) & set(cleaned[1]):
        raise ValueError("train_indices and val_indices must be disjoint.")
    return cleaned[0], cleaned[1]


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Any) -> None:
    """Replace a JSON file atomically, preserving the old file on write failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as fh:
            temporary = Path(fh.name)
            json.dump(data, fh, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_oof(oof: Any) -> None:
    """Reject ratios, malformed counts, and non-finite callback results."""
    if not isinstance(oof, dict) or not oof:
        raise ValueError("oof must be a non-empty mapping of target names to counts.")
    for target, counts in oof.items():
        if not isinstance(target, str) or not isinstance(counts, dict):
            raise ValueError("Each oof target must map a string name to a counts dict.")
        if not {"tp", "fp", "fn"}.issubset(counts):
            raise ValueError(f"OOF target {target!r} must include tp, fp, and fn.")
        if set(counts) - {"tp", "fp", "fn", "tn", "matched_iou_sum"}:
            raise ValueError(f"OOF target {target!r} contains unsupported count names.")
        for name, value in counts.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
                or (name != "matched_iou_sum" and value != int(value))
            ):
                raise ValueError(f"Invalid OOF count {target}.{name}: {value!r}.")
        if counts.get("matched_iou_sum", 0) > counts["tp"]:
            raise ValueError("matched_iou_sum cannot exceed tp.")


def _mean(values: list) -> float | None:
    vals = [
        v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(vals) / len(vals) if vals else None


def run_kfold(
    n_samples: int,
    k: int,
    seed: int,
    experiment_dir: str | Path,
    train_fold: Callable[[int, list[int], list[int], Path], dict],
    *,
    only_fold: int | None = None,
    sample_names: list[str] | None = None,
    metadata: dict | None = None,
    overwrite: bool = False,
    groups: list[str] | None = None,
) -> dict:
    """Train and evaluate every fold, then write and return the CV summary.

    ``train_fold(fold_idx, train_indices, val_indices, fold_dir)`` must return a
    JSON-serialisable dict with an ``"oof"`` entry: ``{target: counts}``, where
    counts come from ``update_counts`` / ``instance_counts_from_labels`` summed
    over that fold's held-out samples. Other numeric entries (e.g.
    ``best_val_loss``) are averaged across folds in the summary.

    ``only_fold`` trains a single fold. Folds that an earlier run already
    finished with the same assignment are kept from the manifest, so the summary
    still covers them. ``sample_names`` (same order as the indices) are recorded
    per fold for auditing. ``metadata`` (e.g. the training recipe) goes into the
    manifest. Restoring results requires identical sample names, groups,
    metadata, and fold assignments. ``overwrite=True`` permits rerunning in a
    non-empty directory; it never authorizes mixing incompatible experiments.
    ``groups`` optionally keeps tiles from the same scene/site together (see
    ``compute_fold_indices``). Set framework seeds yourself, initialize a fresh
    model and optimizer per fold, and evaluate only the supplied held-out indices.
    This utility does not control model training or verify callback predictions.
    Calls sharing an experiment directory must run sequentially.
    """
    folds = compute_fold_indices(n_samples, k, seed, groups=groups)
    if sample_names is not None and (
        len(sample_names) != n_samples
        or any(not isinstance(name, str) for name in sample_names)
        or len(set(sample_names)) != n_samples
    ):
        raise ValueError("sample_names must contain one unique string per sample.")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    if only_fold is not None and (
        isinstance(only_fold, bool)
        or not isinstance(only_fold, int)
        or not 0 <= only_fold < k
    ):
        raise ValueError(f"only_fold {only_fold} out of range [0, {k - 1}].")

    experiment_dir = Path(experiment_dir).expanduser().resolve()
    manifest_path = experiment_dir / MANIFEST_NAME
    existing_manifest = _read_json(manifest_path)
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Experiment dir already exists and is not empty: {experiment_dir}. "
            "Pass overwrite=True to continue."
        )
    # Persist the exact fold assignment for reproducibility / auditing.
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "n_samples": n_samples,
        "folds": k,
        "fold_seed": seed,
        "metadata": metadata or {},
        "sample_names": sample_names,
        "groups": groups,
        "fold_assignment": {},
    }
    for i, fold in enumerate(folds):
        entry: dict[str, Any] = {
            "train_indices": fold["train"],
            "val_indices": fold["val"],
        }
        if sample_names is not None:
            entry["val_samples"] = [sample_names[j] for j in fold["val"]]
        manifest["fold_assignment"][f"fold_{i}"] = entry

    # Round-trip before any writes to validate JSON and compare persisted values.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    if only_fold is not None and existing_manifest is not None:
        for key in (
            "schema_version",
            "n_samples",
            "folds",
            "fold_seed",
            "metadata",
            "sample_names",
            "groups",
        ):
            if existing_manifest.get(key) != manifest[key]:
                raise ValueError(
                    f"Cannot reuse results: manifest {key} differs. Use a new experiment_dir."
                )
        old_assignment = {
            name: {key: value for key, value in entry.items() if key != "result"}
            for name, entry in existing_manifest.get("fold_assignment", {}).items()
        }
        if old_assignment != manifest["fold_assignment"]:
            raise ValueError("Cannot reuse results: fold assignments differ.")
        # Keep other folds' finished results when their assignment is unchanged.
        for fold_name, existing_entry in existing_manifest.get(
            "fold_assignment", {}
        ).items():
            if (
                fold_name not in manifest["fold_assignment"]
                or fold_name == f"fold_{only_fold}"
            ):
                continue
            new_entry = manifest["fold_assignment"][fold_name]
            if (
                existing_entry.get("train_indices") == new_entry["train_indices"]
                and existing_entry.get("val_indices") == new_entry["val_indices"]
            ):
                manifest["fold_assignment"][fold_name] = existing_entry

    experiment_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted rerun must not leave a stale successful summary behind.
    (experiment_dir / SUMMARY_NAME).unlink(missing_ok=True)
    _write_json(manifest_path, manifest)
    logger.info("Wrote fold manifest: %s", manifest_path)

    fold_range = range(k) if only_fold is None else [only_fold]
    overall_start = time.monotonic()

    for fold_idx in fold_range:
        fold = folds[fold_idx]
        fold_name = f"fold_{fold_idx}"
        fold_dir = experiment_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Fold %s/%s: %s train, %s validation samples",
            fold_idx + 1,
            k,
            len(fold["train"]),
            len(fold["val"]),
        )

        fold_start = time.monotonic()
        result = train_fold(fold_idx, list(fold["train"]), list(fold["val"]), fold_dir)
        if not isinstance(result, dict) or "oof" not in result:
            raise ValueError(
                f"train_fold must return a dict with an 'oof' entry (fold {fold_idx})."
            )
        _validate_oof(result["oof"])
        for entry in manifest["fold_assignment"].values():
            if "result" in entry:
                previous = entry["result"]["oof"]
                if {t: set(c) for t, c in previous.items()} != {
                    t: set(c) for t, c in result["oof"].items()
                }:
                    raise ValueError(
                        "All folds must report the same OOF targets and count names."
                    )
        result = {**result, "minutes": round((time.monotonic() - fold_start) / 60.0, 2)}
        json.dumps(result, allow_nan=False)

        _write_json(
            fold_dir / "oof_metrics.json",
            {
                target: metrics_from_counts(counts)
                for target, counts in result["oof"].items()
            },
        )
        manifest["fold_assignment"][fold_name]["result"] = result
        _write_json(manifest_path, manifest)
        logger.info("Fold %s done in %.1f min", fold_idx, result["minutes"])

    # Cross-validation summary across every fold with a result (trained now or restored).
    completed = [
        (i, manifest["fold_assignment"][f"fold_{i}"]["result"])
        for i in range(k)
        if "result" in manifest["fold_assignment"][f"fold_{i}"]
    ]
    scalar_names = list(
        dict.fromkeys(
            name
            for _, result in completed
            for name, value in result.items()
            if name != "oof"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    )
    summary = {
        "folds": k,
        "folds_completed": [i for i, _ in completed],
        "complete": len(completed) == k,
        "oof": summarize_oof([result["oof"] for _, result in completed]),
        "fold_means": {
            name: _mean([r.get(name) for _, r in completed]) for name in scalar_names
        },
        "per_fold": [{"fold": i, **result} for i, result in completed],
        "total_minutes": round((time.monotonic() - overall_start) / 60.0, 2),
    }
    summary_path = experiment_dir / SUMMARY_NAME
    _write_json(summary_path, summary)

    logger.info("Folds completed: %s of %s", summary["folds_completed"], k)
    for target, metrics in summary["oof"]["micro"].items():
        logger.info(
            f"OOF micro {target}: P={metrics['precision']:.6f}, R={metrics['recall']:.6f}, "
            f"IoU={metrics['iou']:.6f}, F1={metrics['f1']:.6f}"
        )
    logger.info("Summary: %s", summary_path)
    return summary
