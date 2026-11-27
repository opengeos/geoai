"""IoU metrics of TimmSegmentationModel are pooled over the epoch."""

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("timm")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from geoai.timm_segment import TimmSegmentationModel  # noqa: E402


class FixedPredictions(TimmSegmentationModel):
    """Predicts the class stored in channel 0 of the input."""

    def forward(self, x):
        p = x[:, 0]
        logits = torch.stack([1 - p, p], dim=1) * 10
        # Keep a (zero) path to the parameters so training can backpropagate.
        return logits + 0 * next(self.model.parameters()).sum()


def _data():
    # Image 0: all background, predicted correctly.
    # Image 1: one class-1 pixel that the model predicts as background.
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    target[1, 0, 0] = 1
    pred = torch.zeros(2, 4, 4)
    x = torch.stack([pred, pred, pred], dim=1)
    return x, target


# Over both images: class 0 IoU 31/32, class 1 IoU 0/1.
POOLED_MIOU = (31 / 32 + 0) / 2
# The old per-batch IoU added a 1e-6 smoothing term, so compare loosely.
TOL = 1e-5


def _model():
    return FixedPredictions(
        encoder_name="resnet18", encoder_weights=None, num_classes=2
    )


def _trainer(**kwargs):
    return pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        **kwargs,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_val_iou_is_pooled_over_the_validation_set(batch_size):
    loader = DataLoader(TensorDataset(*_data()), batch_size=batch_size)
    out = _trainer().validate(_model(), loader, verbose=False)
    assert out[0]["val_iou"] == pytest.approx(POOLED_MIOU, abs=TOL)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_test_iou_is_pooled_over_the_test_set(batch_size):
    loader = DataLoader(TensorDataset(*_data()), batch_size=batch_size)
    out = _trainer().test(_model(), loader, verbose=False)
    assert out[0]["test_iou"] == pytest.approx(POOLED_MIOU, abs=TOL)


def test_train_iou_epoch_value_is_pooled():
    loader = DataLoader(TensorDataset(*_data()), batch_size=1)
    trainer = _trainer(max_epochs=1, num_sanity_val_steps=0)
    # The LR scheduler monitors val_loss, so a validation loader is needed.
    trainer.fit(_model(), loader, loader)
    assert trainer.callback_metrics["train_iou_epoch"].item() == pytest.approx(
        POOLED_MIOU, abs=TOL
    )


def test_validation_epochs_do_not_carry_over():
    model = _model()
    loader = DataLoader(TensorDataset(*_data()), batch_size=1)
    first = _trainer().validate(model, loader, verbose=False)[0]["val_iou"]
    assert first == pytest.approx(POOLED_MIOU, abs=TOL)

    # Only correctly predicted background: IoU is 1.0 unless the first
    # run's counts were kept (which would give 0.4921875).
    x, target = _data()
    clean = DataLoader(TensorDataset(x[:1], target[:1]), batch_size=1)
    second = _trainer().validate(model, clean, verbose=False)[0]["val_iou"]
    assert second == pytest.approx(1.0, abs=TOL)


def test_pooled_iou_matches_single_pass_and_skips_absent_classes():
    from geoai.timm_segment import _PooledIoU

    gen = torch.Generator().manual_seed(0)
    pred = torch.randint(0, 3, (6, 8, 8), generator=gen)
    target = torch.randint(0, 3, (6, 8, 8), generator=gen)
    pred[pred == 2] = 0  # class 3 never appears; class 2 only in targets
    target[target == 2] = 1

    batched = _PooledIoU(num_classes=4)
    for i in range(0, 6, 4):
        batched.update(pred[i : i + 4], target[i : i + 4])

    ious = []
    for cls in range(4):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union:
            ious.append(inter / union)
    assert batched.compute().item() == pytest.approx(sum(ious) / len(ious))


def test_pooled_iou_is_zero_before_any_update():
    from geoai.timm_segment import _PooledIoU

    assert _PooledIoU(num_classes=2).compute().item() == 0.0


def test_metric_state_is_not_saved_in_checkpoints():
    assert not any("iou" in key for key in _model().state_dict())


def test_pooled_iou_ignores_out_of_range_targets():
    from geoai.timm_segment import _PooledIoU

    pred = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 255, -1])
    metric = _PooledIoU(num_classes=2)
    metric.update(pred, target)
    # Class 0: inter 1, union 2 (pixel 3 predicted 0). Class 1: inter 1, union 2.
    assert metric.intersection.tolist() == [1, 1]
    assert metric.union.tolist() == [2, 2]
    assert metric.compute().item() == pytest.approx(0.5)
