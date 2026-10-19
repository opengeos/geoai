"""Built-in semantic CV trains on K-1 folds and validates on the held-out fold."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest

import geoai.train as training


@pytest.fixture
def paired_images(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for i in range(9):
        Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(
            images / f"tile_{i}.png"
        )
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[: i + 1] = 1
        Image.fromarray(mask).save(masks / f"tile_{i}.png")
    return images, masks


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("monitor", ["val_iou", "val_loss"])
@pytest.mark.parametrize("only_fold", [None, 0])
@pytest.mark.parametrize("verbose", [True, False])
def test_builtin_kfold_isolates_samples_and_pools_native_predictions(
    paired_images, tmp_path, num_classes, monitor, only_fold, verbose, capsys
):
    images, masks = paired_images
    original = training.train_segmentation_model
    calls = []

    def fake_train(**options):
        directory = Path(options["output_dir"])
        (directory / "best_model.pth").touch()
        calls.append(options)

    def fake_batch(**options):
        assert options["model_kwargs"] == {"decoder_channels": (32, 16)}
        assert options["raise_on_error"] is True
        output = Path(options["output_dir"])
        output.mkdir()
        for image in options["input_files"]:
            path = Path(image)
            with Image.open(masks / path.name) as src:
                src.save(output / f"{path.stem}_mask.png")

    with (
        patch.object(training, "train_segmentation_model", side_effect=fake_train),
        patch.object(training, "semantic_segmentation_batch", side_effect=fake_batch),
    ):
        summary = original(
            str(images),
            str(masks),
            str(tmp_path / "cv"),
            k_folds=3,
            encoder_weights=None,
            device="cpu",
            decoder_channels=(32, 16),
            num_classes=num_classes,
            monitor=monitor,
            only_fold=only_fold,
            verbose=verbose,
        )
    console = capsys.readouterr().out
    assert summary["complete"] == (only_fold is None)
    assert len(calls) == (3 if only_fold is None else 1)
    assert summary["monitor"] == monitor
    assert summary["validation_strategy"] == "held_out_fold"
    manifest = json.loads((tmp_path / "cv/kfold_manifest.json").read_text())
    assert manifest["metadata"]["validation_strategy"] == "held_out_fold"
    assert "val_split" not in manifest["metadata"]
    assert summary["ensemble_model_paths"] == (
        [fold["model_path"] for fold in summary["per_fold"]]
        if only_fold is None
        else []
    )
    assert (
        json.loads((tmp_path / "cv/kfold_summary.json").read_text())[
            "ensemble_model_paths"
        ]
        == summary["ensemble_model_paths"]
    )
    total_tp = 0
    for call in calls:
        directory = Path(call["output_dir"])
        split = json.loads((directory / "split.json").read_text())
        fold_number = int(directory.name.split("_")[-1]) + 1
        report = console.split(f"Fold {fold_number}/3 tile allocation:", 1)[1]
        report = report.split("Full image/mask paths and indices:", 1)[0]
        for role, title in (
            ("fit", "Training (weight updates)"),
            ("held_out", "Validation / OOF (checkpoint selection and evaluation)"),
        ):
            indices = split[f"{role}_indices"]
            assert split[f"{role}_files"] == [
                str((images / f"tile_{i}.png").resolve()) for i in indices
            ]
            assert split[f"{role}_label_files"] == [
                str((masks / f"tile_{i}.png").resolve()) for i in indices
            ]
            assert f"{title}: {len(indices)}/9 tiles" in report
            section = report.split(f"{title}:", 1)[1]
            section = section.split("\n  Validation / OOF", 1)[0]
            for i in range(9):
                assert (f"[{i}] tile_{i}.png" in section) == (verbose and i in indices)
        assert (f"Fold {fold_number}/3 OOF class_1:" in console) == verbose
        fit, validation, held_out = map(
            set, (call["train_indices"], call["val_indices"], split["held_out_indices"])
        )
        assert len(fit) == 6 and len(validation) == 3
        assert validation == held_out
        assert not fit & validation
        assert fit | held_out == set(range(9))
        assert set(split["fit_indices"]) == fit
        assert "monitor_indices" not in split
        assert call["decoder_channels"] == (32, 16)
        assert call["monitor"] == monitor
        for i in held_out:
            with Image.open(masks / f"tile_{i}.png") as src:
                total_tp += int(np.asarray(src).sum())
    assert summary["oof"]["micro"]["class_1"]["tp"] == total_tp
    assert summary["oof"]["micro"]["class_1"]["f1"] == 1
    if num_classes == 3:
        assert set(summary["oof"]["micro"]) == {"class_0", "class_1", "class_2"}


def test_builtin_kfold_keeps_groups_in_train_validation_split(paired_images, tmp_path):
    images, masks = paired_images
    original = training.train_segmentation_model
    # Two groups suffice for two folds; no extra group is needed for an inner split.
    groups = [f"scene_{i % 2}" for i in range(9)]
    captured = []

    def stop(**options):
        captured.append(options)
        raise RuntimeError("split inspected")

    with (
        patch.object(training, "train_segmentation_model", side_effect=stop),
        pytest.raises(RuntimeError, match="split inspected"),
    ):
        original(
            str(images),
            str(masks),
            str(tmp_path / "cv"),
            k_folds=2,
            fold_groups=groups,
            device="cpu",
        )
    call = captured[0]
    split = json.loads((Path(call["output_dir"]) / "split.json").read_text())
    for role in ("fit", "held_out"):
        assert split[f"{role}_groups"] == [groups[i] for i in split[f"{role}_indices"]]
    assert not {groups[i] for i in call["train_indices"]} & {
        groups[i] for i in call["val_indices"]
    }
    assert call["val_indices"] == split["held_out_indices"]
    assert sorted(call["train_indices"] + call["val_indices"]) == list(range(9))


@pytest.mark.parametrize(
    "k,train_size,val_size", [(2, 4, 5), (3, 6, 3), (5, 7, 2), (9, 8, 1)]
)
@pytest.mark.parametrize("val_split", [0.0, 0.2, 0.9])
def test_k_alone_controls_validation_size(
    paired_images, tmp_path, k, train_size, val_size, val_split
):
    images, masks = paired_images
    original = training.train_segmentation_model
    captured = []

    def stop(**options):
        captured.append(options)
        raise RuntimeError("split inspected")

    with (
        patch.object(training, "train_segmentation_model", side_effect=stop),
        pytest.raises(RuntimeError, match="split inspected"),
    ):
        original(
            str(images),
            str(masks),
            str(tmp_path / "cv"),
            k_folds=k,
            val_split=val_split,
            device="cpu",
            verbose=False,
        )
    call = captured[0]
    assert len(call["train_indices"]) == train_size
    assert len(call["val_indices"]) == val_size
    assert set(call["train_indices"]).isdisjoint(call["val_indices"])
    assert sorted(call["train_indices"] + call["val_indices"]) == list(range(9))


@pytest.mark.parametrize("legacy_inner_holdout", [False, True])
def test_fold_rerun_ignores_val_split_but_rejects_old_validation_strategy(
    paired_images, tmp_path, legacy_inner_holdout
):
    images, masks = paired_images
    original = training.train_segmentation_model
    captured = []

    def stop(**options):
        captured.append(options)
        raise RuntimeError("split inspected")

    with patch.object(training, "train_segmentation_model", side_effect=stop):
        with pytest.raises(RuntimeError, match="split inspected"):
            original(
                str(images),
                str(masks),
                str(tmp_path / "cv"),
                k_folds=3,
                only_fold=0,
                val_split=0.2,
                device="cpu",
                verbose=False,
            )
        manifest_path = tmp_path / "cv/kfold_manifest.json"
        if legacy_inner_holdout:
            manifest = json.loads(manifest_path.read_text())
            manifest["metadata"].pop("validation_strategy")
            manifest["metadata"]["selection"] = "best inner val_iou"
            manifest["metadata"]["val_split"] = 0.2
            manifest_path.write_text(json.dumps(manifest))
        error = ValueError if legacy_inner_holdout else RuntimeError
        message = (
            "manifest metadata differs" if legacy_inner_holdout else "split inspected"
        )
        with pytest.raises(error, match=message):
            original(
                str(images),
                str(masks),
                str(tmp_path / "cv"),
                k_folds=3,
                only_fold=1,
                overwrite=True,
                val_split=0.9,
                device="cpu",
                verbose=False,
            )
    assert len(captured) == (1 if legacy_inner_holdout else 2)


@pytest.mark.parametrize(
    "options",
    [
        {"k_folds": 3, "train_indices": [0], "val_indices": [1]},
        {"k_folds": 3, "resume_training": True},
        {"only_fold": 0},
        {"fold_groups": ["a"]},
        {"overwrite": True},
    ],
)
def test_invalid_mode_combinations(options, tmp_path):
    with pytest.raises(ValueError):
        training.train_segmentation_model(
            "missing", "missing", str(tmp_path), **options
        )


def test_batch_selects_explicit_inputs_and_keeps_multiclass_labels(
    paired_images, tmp_path
):
    images, _ = paired_images
    model = MagicMock()
    weights = tmp_path / "model.pth"
    weights.touch()
    with (
        patch.object(training, "get_smp_model", return_value=model) as create,
        patch.object(training.torch, "load", return_value={}),
        patch.object(training, "semantic_inference_on_image") as infer,
    ):
        training.semantic_segmentation_batch(
            str(images),
            str(tmp_path / "predictions"),
            str(weights),
            input_files=["tile_2.png", "tile_0.png"],
            num_classes=3,
            device="cpu",
            model_kwargs={"encoder_depth": 3},
        )
    assert [Path(c.kwargs["image_path"]).name for c in infer.call_args_list] == [
        "tile_2.png",
        "tile_0.png",
    ]
    assert all(not c.kwargs["binary_output"] for c in infer.call_args_list)
    assert create.call_args.kwargs["encoder_depth"] == 3


def test_batch_skips_existing_outputs(paired_images, tmp_path):
    images, _ = paired_images
    output = tmp_path / "predictions"
    output.mkdir()
    (output / "tile_0_mask.png").touch()
    weights = tmp_path / "model.pth"
    weights.touch()
    with (
        patch.object(training, "get_smp_model", return_value=MagicMock()),
        patch.object(training.torch, "load", return_value={}),
        patch.object(training, "semantic_inference_on_image") as infer,
    ):
        training.semantic_segmentation_batch(
            str(images),
            str(output),
            str(weights),
            input_files=["tile_0.png"],
            device="cpu",
            skip_existing=True,
        )
    infer.assert_not_called()


def test_batch_rejects_duplicate_input_paths(paired_images, tmp_path):
    images, _ = paired_images
    with pytest.raises(ValueError, match="duplicate"):
        training.semantic_segmentation_batch(
            str(images),
            str(tmp_path / "predictions"),
            "unused.pth",
            input_files=["tile_0.png", "tile_0.png"],
            device="cpu",
        )


@pytest.mark.parametrize("strict", [False, True])
def test_batch_failure_behavior_is_explicit(paired_images, tmp_path, strict):
    images, _ = paired_images
    weights = tmp_path / "model.pth"
    weights.touch()
    with (
        patch.object(training, "get_smp_model", return_value=MagicMock()),
        patch.object(training.torch, "load", return_value={}),
        patch.object(
            training,
            "semantic_inference_on_image",
            side_effect=RuntimeError("failed image"),
        ),
    ):
        if strict:
            with pytest.raises(RuntimeError, match="failed image"):
                training.semantic_segmentation_batch(
                    str(images),
                    str(tmp_path / "out"),
                    str(weights),
                    input_files=["tile_0.png"],
                    device="cpu",
                    raise_on_error=True,
                )
        else:
            training.semantic_segmentation_batch(
                str(images),
                str(tmp_path / "out"),
                str(weights),
                input_files=["tile_0.png"],
                device="cpu",
            )


def test_single_split_saves_first_zero_iou_checkpoint(paired_images, tmp_path):
    images, masks = paired_images
    model = training.torch.nn.Conv2d(3, 2, kernel_size=1)
    scores = {"loss": 1.0, "IoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}
    with (
        patch.object(training, "get_smp_model", return_value=model),
        patch.object(training, "train_semantic_one_epoch", return_value=1.0),
        patch.object(training, "evaluate_semantic", return_value=scores),
    ):
        result = training.train_segmentation_model(
            str(images),
            str(masks),
            str(tmp_path / "single"),
            num_epochs=1,
            target_size=(32, 32),
            num_workers=0,
            device="cpu",
            encoder_weights=None,
        )
    assert result is None
    assert (tmp_path / "single/best_model.pth").is_file()


def test_builtin_kfold_resolves_relative_inputs(paired_images, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = training.train_segmentation_model

    def fake_train(**options):
        (Path(options["output_dir"]) / "best_model.pth").touch()

    def inspect_batch(**options):
        assert all(
            Path(p).is_absolute() and Path(p).is_file() for p in options["input_files"]
        )
        raise RuntimeError("relative inputs resolved")

    with (
        patch.object(training, "train_segmentation_model", side_effect=fake_train),
        patch.object(
            training, "semantic_segmentation_batch", side_effect=inspect_batch
        ),
        pytest.raises(RuntimeError, match="relative inputs resolved"),
    ):
        original("images", "masks", "cv", k_folds=3, device="cpu")


@pytest.mark.parametrize("mask_mode", ["L", "RGB", "P"])
def test_oof_scores_actual_binary_png_inference(paired_images, tmp_path, mask_mode):
    images, masks = paired_images
    expected_tp = 0
    for path in masks.iterdir():
        with Image.open(path) as src:
            mask = np.asarray(src).copy()
        label = Image.fromarray(mask).convert(mask_mode)
        if mask_mode == "P":
            # Palette values differ from raw indices, as in the training reader.
            label.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
        label.save(path)
        expected_tp += int((np.asarray(label.convert("L")) > 0).sum())

    original_train = training.train_segmentation_model
    original_batch = training.semantic_segmentation_batch

    def save_foreground_model(**options):
        training.torch.save(
            _ConstantSegmentationModel(0.9).state_dict(),
            Path(options["output_dir"]) / "best_model.pth",
        )

    def small_window_batch(**options):
        original_batch(**options, window_size=8, overlap=4)

    with (
        patch.object(
            training, "train_segmentation_model", side_effect=save_foreground_model
        ),
        patch.object(
            training, "semantic_segmentation_batch", side_effect=small_window_batch
        ),
        patch.object(
            training,
            "get_smp_model",
            side_effect=lambda **kwargs: _ConstantSegmentationModel(),
        ),
    ):
        summary = original_train(
            str(images),
            str(masks),
            str(tmp_path / "cv"),
            k_folds=2,
            device="cpu",
            verbose=False,
        )
    counts = summary["oof"]["micro"]["class_1"]
    assert counts["tp"] == expected_tp
    assert counts["fp"] == 9 * 8 * 8 - expected_tp
    assert counts["fn"] == counts["tn"] == 0
    for fold in summary["per_fold"]:
        for path in Path(fold["prediction_dir"]).glob("*.png"):
            with Image.open(path) as src:
                assert np.all(np.asarray(src) == 255)


@pytest.mark.parametrize("num_epochs", [0, -1, True, 1.5])
def test_kfold_rejects_empty_training_before_reusing_artifacts(tmp_path, num_epochs):
    with pytest.raises(ValueError, match="num_epochs"):
        training.train_segmentation_model(
            "missing",
            "missing",
            str(tmp_path),
            k_folds=2,
            num_epochs=num_epochs,
            overwrite=True,
        )


class SplitReached(Exception):
    pass


def test_trainer_uses_only_explicit_pairs(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(5):
        (images / f"tile_{i}.tif").touch()
        (labels / f"tile_{i}.tif").touch()
    datasets = []

    def capture(image_files, label_files, **kwargs):
        datasets.append((image_files, label_files))
        if len(datasets) == 2:
            raise SplitReached
        return object()

    with (
        patch.object(training, "SMP_AVAILABLE", True),
        patch.object(training, "SemanticSegmentationDataset", side_effect=capture),
        patch.object(
            training, "train_test_split", side_effect=AssertionError("Must not resplit")
        ),
        pytest.raises(SplitReached),
    ):
        training.train_segmentation_model(
            str(images),
            str(labels),
            str(tmp_path / "run"),
            target_size=(32, 32),
            device="cpu",
            train_indices=[3, 0],
            val_indices=[2],
        )
    assert datasets == [
        (
            [str(images / "tile_3.tif"), str(images / "tile_0.tif")],
            [str(labels / "tile_3.tif"), str(labels / "tile_0.tif")],
        ),
        ([str(images / "tile_2.tif")], [str(labels / "tile_2.tif")]),
    ]


def test_trainer_requires_both_index_lists(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "tile.tif").touch()
    (labels / "tile.tif").touch()
    with (
        patch.object(training, "SMP_AVAILABLE", True),
        pytest.raises(ValueError, match="both"),
    ):
        training.train_segmentation_model(
            str(images),
            str(labels),
            str(tmp_path / "run"),
            device="cpu",
            train_indices=[0],
        )


@pytest.mark.parametrize(
    "monitor,epoch,score", [("val_loss", 2, 0.2), ("val_iou", 3, 0.9)]
)
def test_checkpoint_selection_uses_chosen_criterion(
    paired_images, tmp_path, monitor, epoch, score
):
    images, masks = paired_images
    model = training.torch.nn.Conv2d(3, 2, kernel_size=1)
    trained = 0

    def fake_epoch(*args, **kwargs):
        nonlocal trained
        trained += 1
        with training.torch.no_grad():
            model.weight.fill_(trained)
        return 1.0

    metrics = [
        {"loss": loss, "IoU": iou, "F1": 0.5, "Precision": 0.5, "Recall": 0.5}
        for loss, iou in [(0.5, 0.6), (0.2, 0.4), (0.3, 0.9)]
    ]
    output = tmp_path / monitor
    with (
        patch.object(training, "get_smp_model", return_value=model),
        patch.object(training, "train_semantic_one_epoch", side_effect=fake_epoch),
        patch.object(training, "evaluate_semantic", side_effect=metrics),
    ):
        training.train_segmentation_model(
            str(images),
            str(masks),
            str(output),
            num_epochs=3,
            target_size=(32, 32),
            num_workers=0,
            device="cpu",
            encoder_weights=None,
            monitor=monitor,
            save_best_only=False,
        )
    selected = training.torch.load(output / "best_model.pth", weights_only=True)
    assert training.torch.all(selected["weight"] == epoch)
    selection = json.loads((output / "model_selection.json").read_text())
    assert selection["monitor"] == monitor
    assert selection["best_epoch"] == epoch
    assert selection["best_score"] == score
    checkpoint = training.torch.load(
        output / "checkpoint_epoch_3.pth", weights_only=True
    )
    assert checkpoint["monitor"] == monitor
    assert checkpoint["best_score"] == score


@pytest.mark.parametrize("monitor,epochs", [("val_loss", 3), ("val_iou", 4)])
def test_early_stopping_uses_chosen_criterion(paired_images, tmp_path, monitor, epochs):
    images, masks = paired_images
    metrics = [
        {"loss": loss, "IoU": iou, "F1": 0.5, "Precision": 0.5, "Recall": 0.5}
        for loss, iou in [(0.5, 0.1), (0.4, 0.2), (0.6, 0.3), (0.7, 0.4)]
    ]
    output = tmp_path / monitor
    with (
        patch.object(
            training, "get_smp_model", return_value=training.torch.nn.Conv2d(3, 2, 1)
        ),
        patch.object(training, "train_semantic_one_epoch", return_value=1.0) as train,
        patch.object(training, "evaluate_semantic", side_effect=metrics),
    ):
        training.train_segmentation_model(
            str(images),
            str(masks),
            str(output),
            num_epochs=4,
            early_stopping_patience=1,
            target_size=(32, 32),
            num_workers=0,
            device="cpu",
            encoder_weights=None,
            monitor=monitor,
        )
    assert train.call_count == epochs


def test_invalid_checkpoint_criterion_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="monitor"):
        training.train_segmentation_model(
            "unused", "unused", str(tmp_path), monitor="accuracy"
        )


class _ConstantSegmentationModel(training.torch.nn.Module):
    def __init__(self, foreground=0.5):
        super().__init__()
        self.register_buffer(
            "logits",
            training.torch.tensor(
                [np.log(1 - foreground), np.log(foreground)],
                dtype=training.torch.float32,
            ),
        )

    def forward(self, images):
        return self.logits.view(1, 2, 1, 1).expand(
            images.shape[0], 2, *images.shape[-2:]
        )


def test_ensemble_averages_probabilities_instead_of_logits():
    models = [_ConstantSegmentationModel(p) for p in (0.99, 0.1, 0.1)]
    images = training.torch.zeros(1, 3, 8, 8)
    probabilities = training._ProbabilityEnsemble(models)(images).softmax(dim=1)
    assert training.torch.allclose(
        probabilities[:, 1], training.torch.full((1, 8, 8), 1.19 / 3)
    )
    assert not probabilities.argmax(dim=1).any()
    # Averaging logits would select the opposite class for this example.
    averaged_logits = sum(model(images) for model in models) / 3
    assert averaged_logits.argmax(dim=1).all()


@pytest.mark.parametrize("batch", [False, True])
def test_public_inference_accepts_checkpoint_ensemble(paired_images, tmp_path, batch):
    images, _ = paired_images
    paths = []
    for i, probability in enumerate((0.99, 0.1, 0.1)):
        path = tmp_path / f"model_{i}.pth"
        training.torch.save(_ConstantSegmentationModel(probability).state_dict(), path)
        paths.append(str(path))
    with patch.object(
        training,
        "get_smp_model",
        side_effect=lambda **kwargs: _ConstantSegmentationModel(),
    ) as create:
        if batch:
            output = tmp_path / "batch"
            training.semantic_segmentation_batch(
                str(images),
                str(output),
                paths,
                input_files=["tile_0.png"],
                device="cpu",
                window_size=8,
                overlap=4,
                probability_threshold=0.5,
                raise_on_error=True,
                quiet=True,
            )
            prediction = output / "tile_0_mask.png"
        else:
            prediction = tmp_path / "single_mask.png"
            training.semantic_segmentation(
                str(images / "tile_0.png"),
                str(prediction),
                paths,
                device="cpu",
                window_size=8,
                overlap=4,
                probability_threshold=0.5,
                quiet=True,
            )
    assert create.call_count == 3
    with Image.open(prediction) as src:
        assert not np.asarray(src).any()


@pytest.mark.parametrize("case", ["empty", "duplicate"])
def test_ensemble_rejects_invalid_checkpoint_lists(tmp_path, case):
    path = tmp_path / "model.pth"
    path.touch()
    paths = [] if case == "empty" else [str(path), str(path)]
    with pytest.raises(ValueError):
        training._load_segmentation_model(paths, "unet", "resnet34", 3, 2, "cpu")
