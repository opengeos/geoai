"""Tests for reproducible folds, OOF metrics, and experiment persistence."""

import json

import numpy as np
import pytest

import geoai
from geoai.cross_validation import (
    _write_json,
    add_counts,
    compute_fold_indices,
    instance_counts_from_labels,
    metrics_from_counts,
    run_kfold,
    summarize_oof,
    update_counts,
    validate_paired_files,
)


def test_public_api():
    assert geoai.run_kfold is run_kfold
    assert geoai.compute_fold_indices is compute_fold_indices
    assert geoai.cross_validation.run_kfold is run_kfold


@pytest.mark.parametrize(
    "train,val",
    [
        ([], [1]),
        ([0], []),
        ([0, 0], [1]),
        ([0], [1, 1]),
        ([0], [0]),
        ([-1], [1]),
        ([0], [4]),
        ([True], [1]),
        ([0.5], [1]),
    ],
)
def test_invalid_explicit_split_indices(train, val):
    from geoai.cross_validation import _validate_split_indices

    with pytest.raises(ValueError):
        _validate_split_indices(4, train, val)


def test_explicit_split_allows_excluded_test_samples():
    from geoai.cross_validation import _validate_split_indices

    assert _validate_split_indices(5, [np.int64(3), 0], [2]) == ([3, 0], [2])


def test_fold_coverage_reproducibility_and_balance():
    folds = compute_fold_indices(11, 3, 42)
    assert folds == compute_fold_indices(11, 3, 42)
    assert folds != compute_fold_indices(11, 3, 43)
    assert sorted(i for fold in folds for i in fold["val"]) == list(range(11))
    assert sorted(len(fold["val"]) for fold in folds) == [3, 4, 4]
    for fold in folds:
        assert not set(fold["train"]) & set(fold["val"])
        assert sorted(fold["train"] + fold["val"]) == list(range(11))


@pytest.mark.parametrize("n,k", [(1, 2), (4, 1), (3, 4), (3.5, 2), (4, True)])
def test_invalid_folds(n, k):
    with pytest.raises(ValueError):
        compute_fold_indices(n, k, 42)


def test_grouped_folds():
    groups = ["site_a", "site_a", "site_b", "site_c", "site_c", "site_d"]
    folds = compute_fold_indices(6, 3, 42, groups=groups)
    assert folds == compute_fold_indices(6, 3, 42, groups=groups)
    assert sorted(i for fold in folds for i in fold["val"]) == list(range(6))
    for fold in folds:
        assert not {groups[i] for i in fold["train"]} & {groups[i] for i in fold["val"]}
    with pytest.raises(ValueError, match="distinct groups"):
        compute_fold_indices(6, 5, 42, groups=groups)
    with pytest.raises(ValueError, match="one string"):
        compute_fold_indices(6, 3, 42, groups=["a"])


def test_paired_files():
    validate_paired_files(
        {
            "images": ["image_1.tif", "image_a.tif"],
            "masks": ["mask_1.tif", "mask_a.tif"],
        }
    )
    with pytest.raises(RuntimeError, match="missing indices"):
        validate_paired_files(
            {
                "images": ["image_1.tif", "image_a.tif"],
                "masks": ["mask_2.tif", "mask_b.tif"],
            }
        )
    with pytest.raises(RuntimeError, match="order mismatch"):
        validate_paired_files(
            {
                "images": ["image_1.tif", "image_2.tif"],
                "masks": ["mask_2.tif", "mask_1.tif"],
            }
        )
    with pytest.raises(RuntimeError, match="Duplicate"):
        validate_paired_files({"images": ["tile_a_1.tif", "tile_b_1.tif"]})
    with pytest.raises(RuntimeError, match="count mismatch"):
        validate_paired_files({"images": ["image_1.tif"], "masks": []})
    with pytest.raises(ValueError):
        validate_paired_files({})
    with pytest.raises(ValueError, match="reference"):
        validate_paired_files({"images": []}, reference="masks")


def test_pixel_counts_ignore_invalid_and_accumulate():
    counts = update_counts({}, [0.5, 0.8, 0.2, 0.1, np.nan], [1, 0, 1, 0, np.nan])
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    update_counts(counts, [0.8, 0.8], [1, 0], valid=[1, 0])
    assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    with pytest.raises(ValueError, match="finite"):
        update_counts({}, [np.nan], [1])


@pytest.mark.parametrize("threshold", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_pixel_threshold(threshold):
    with pytest.raises(ValueError):
        update_counts({}, [0.5], [1], threshold=threshold)


def test_pixel_shapes():
    with pytest.raises(ValueError, match="shapes"):
        update_counts({}, [[0.5]], [1])
    with pytest.raises(ValueError, match="shapes"):
        update_counts({}, [0.5], [1], valid=[[1]])


def test_nonfinite_valid_pixels_are_excluded():
    counts = update_counts({}, [np.nan, 0.8], [1, 1], valid=[np.inf, 1])
    assert counts == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}


def test_instance_metrics_pool_matched_iou_by_object_count():
    summary = summarize_oof(
        [
            {"crowns": {"tp": 1, "fp": 0, "fn": 0, "matched_iou_sum": 1.0}},
            {"crowns": {"tp": 3, "fp": 0, "fn": 0, "matched_iou_sum": 1.5}},
        ]
    )
    assert summary["micro"]["crowns"]["mean_matched_iou"] == 0.625
    assert summary["macro"]["crowns"]["mean_matched_iou"] == 0.75


def test_resume_rejects_tampered_assignment(tmp_path):
    run_kfold(6, 3, 42, tmp_path, perfect_fold)
    path = tmp_path / "kfold_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["fold_assignment"]["fold_0"]["val_indices"] = [0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="assignments differ"):
        run_kfold(6, 3, 42, tmp_path, perfect_fold, only_fold=1, overwrite=True)


def test_instance_sparse_ids_empty_and_no_overlap():
    assert instance_counts_from_labels([7, 7, 0, 9000], [42, 42, 0, 1000000]) == {
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "matched_iou_sum": 2.0,
    }
    assert instance_counts_from_labels([0, 0], [100, 100])["fp"] == 1
    assert (
        instance_counts_from_labels(np.array([], dtype=int), np.array([], dtype=int))[
            "tp"
        ]
        == 0
    )
    result = instance_counts_from_labels([1, 0], [0, 2])
    assert result == {"tp": 0, "fp": 1, "fn": 1, "matched_iou_sum": 0.0}


def test_instance_merge_matches_once():
    result = instance_counts_from_labels([1, 1, 2, 2], [5, 5, 5, 5])
    assert result == {"tp": 1, "fp": 0, "fn": 1, "matched_iou_sum": 0.5}
    assert instance_counts_from_labels([1, 1], [2, 2], 1)["tp"] == 1


@pytest.mark.parametrize(
    "gt,pred,threshold",
    [
        ([1], [1], 0.4),
        ([1], [1], np.nan),
        ([1], [1], 1.1),
        ([-1], [1], 0.5),
        ([1.2], [1], 0.5),
        ([[1]], [1], 0.5),
    ],
)
def test_invalid_instance_inputs(gt, pred, threshold):
    with pytest.raises(ValueError):
        instance_counts_from_labels(gt, pred, threshold)


def test_metrics_pool_counts_before_scoring():
    summary = summarize_oof(
        [
            {"mask": {"tp": 1, "fp": 0, "fn": 0}},
            {"mask": {"tp": 0, "fp": 9, "fn": 9}},
        ]
    )
    assert summary["micro"]["mask"]["f1"] == pytest.approx(0.1)
    assert summary["macro"]["mask"]["f1"] == pytest.approx(0.5)
    total = add_counts({}, {"tp": 2, "matched_iou_sum": 1.5})
    assert metrics_from_counts(total)["mean_matched_iou"] == 0.75
    assert metrics_from_counts({})["f1"] == 0


def perfect_fold(fold, train, val, directory):
    assert directory.is_dir()
    assert not set(train) & set(val)
    return {"oof": {"mask": {"tp": len(val), "fp": 0, "fn": 0}}, "loss": fold + 1}


def test_run_writes_manifest_metrics_and_summary(tmp_path):
    names = [f"tile_{i}" for i in range(7)]
    summary = run_kfold(7, 3, 42, tmp_path, perfect_fold, sample_names=names)
    assert summary["complete"]
    assert summary["folds_completed"] == [0, 1, 2]
    assert summary["oof"]["micro"]["mask"]["tp"] == 7
    assert summary["fold_means"]["loss"] == 2
    assert json.loads((tmp_path / "kfold_summary.json").read_text()) == summary
    manifest = json.loads((tmp_path / "kfold_manifest.json").read_text())
    for entry in manifest["fold_assignment"].values():
        assert entry["val_samples"] == [names[i] for i in entry["val_indices"]]
    assert (
        json.loads((tmp_path / "fold_0/oof_metrics.json").read_text())["mask"]["f1"]
        == 1
    )
    with pytest.raises(FileExistsError):
        run_kfold(7, 3, 42, tmp_path, perfect_fold)


def test_only_fold_preserves_compatible_results(tmp_path):
    first = run_kfold(6, 3, 42, tmp_path, perfect_fold, only_fold=0)
    assert not first["complete"]
    second = run_kfold(6, 3, 42, tmp_path, perfect_fold, only_fold=1, overwrite=True)
    assert second["folds_completed"] == [0, 1]
    last = run_kfold(6, 3, 42, tmp_path, perfect_fold, only_fold=2, overwrite=True)
    assert last["complete"]
    assert last["oof"]["micro"]["mask"]["tp"] == 6


@pytest.mark.parametrize(
    "changed",
    [
        {"seed": 43},
        {"sample_names": ["c", "b", "a", "d"]},
        {"metadata": {"threshold": 0.6}},
        {"groups": ["x", "x", "y", "y"]},
    ],
)
def test_resume_rejects_changed_experiment_without_writes(tmp_path, changed):
    settings = dict(
        n_samples=4,
        k=2,
        seed=42,
        experiment_dir=tmp_path,
        train_fold=perfect_fold,
        sample_names=["a", "b", "c", "d"],
        metadata={"threshold": 0.5},
    )
    run_kfold(**settings)
    original = (tmp_path / "kfold_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Cannot reuse"):
        run_kfold(**{**settings, **changed}, only_fold=0, overwrite=True)
    assert (tmp_path / "kfold_manifest.json").read_bytes() == original
    assert (tmp_path / "kfold_summary.json").exists()


def test_failure_keeps_completed_results_and_removes_stale_summary(tmp_path):
    run_kfold(6, 3, 42, tmp_path, perfect_fold)

    def failing(fold, train, val, directory):
        if fold == 1:
            raise RuntimeError("training failed")
        return perfect_fold(fold, train, val, directory)

    with pytest.raises(RuntimeError, match="training failed"):
        run_kfold(6, 3, 42, tmp_path, failing, overwrite=True)
    manifest = json.loads((tmp_path / "kfold_manifest.json").read_text())
    assert "result" in manifest["fold_assignment"]["fold_0"]
    assert "result" not in manifest["fold_assignment"]["fold_1"]
    assert not (tmp_path / "kfold_summary.json").exists()


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"oof": {}},
        {"oof": {"mask": {"tp": -1, "fp": 0, "fn": 0}}},
        {"oof": {"mask": {"tp": 1.5, "fp": 0, "fn": 0}}},
        {"oof": {"mask": {"tp": 1, "fp": 0, "fn": 0, "iou": 1}}},
    ],
)
def test_invalid_callback_results(tmp_path, result):
    with pytest.raises(ValueError):
        run_kfold(4, 2, 42, tmp_path, lambda *args: result)


def test_inconsistent_targets_rejected(tmp_path):
    def callback(fold, *args):
        return {"oof": {str(fold): {"tp": 1, "fp": 0, "fn": 0}}}

    with pytest.raises(ValueError, match="same OOF targets"):
        run_kfold(4, 2, 42, tmp_path, callback)


def test_atomic_json_preserves_original_on_serialization_failure(tmp_path):
    path = tmp_path / "manifest.json"
    _write_json(path, {"original": True})
    with pytest.raises(ValueError):
        _write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 1},
        {"only_fold": 2},
        {"only_fold": 0.5},
        {"sample_names": ["a"]},
        {"sample_names": ["a"] * 4},
        {"metadata": {"bad": float("nan")}},
    ],
)
def test_invalid_run_arguments_do_not_create_outputs(tmp_path, kwargs):
    output = tmp_path / "new"
    settings = dict(
        n_samples=4, k=2, seed=42, experiment_dir=output, train_fold=perfect_fold
    )
    with pytest.raises(ValueError):
        run_kfold(**{**settings, **kwargs})
    assert not output.exists()
