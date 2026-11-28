"""ignore_index support in geoai.timm_segment."""

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("timm")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from geoai import timm_segment  # noqa: E402
from geoai.timm_segment import TimmSegmentationModel, _PooledIoU  # noqa: E402

NODATA = 255


class FixedPredictions(TimmSegmentationModel):
    """Predicts the class stored in channel 0 of the input (3 classes)."""

    def forward(self, x):
        logits = nn.functional.one_hot(x[:, 0].long(), 3).permute(0, 3, 1, 2)
        # Keep a (zero) path to the parameters so training can backpropagate.
        return logits.float() * 10 + 0 * next(self.model.parameters()).sum()


def _model(cls=TimmSegmentationModel, **kwargs):
    return cls(encoder_name="resnet18", encoder_weights=None, num_classes=3, **kwargs)


def _nodata_batch():
    # Labelled pixels are predicted correctly; nodata pixels are predicted as 1.
    target = torch.tensor([[[0, 0, 1, 1], [2, 2, NODATA, NODATA]]])
    pred = torch.where(target == NODATA, 1, target)
    x = pred[:, None].float().expand(-1, 3, -1, -1).contiguous()
    return x, target


def test_default_ignore_index_matches_pytorch():
    model = _model()
    assert model.hparams.ignore_index == -100
    assert model.loss_fn.ignore_index == -100


def test_default_loss_skips_nodata_pixels():
    torch.manual_seed(0)
    model = _model(ignore_index=NODATA).eval()
    x = torch.rand(2, 3, 32, 32)
    y = torch.randint(0, 3, (2, 32, 32))
    y[:, :, :8] = NODATA

    loss = model.training_step((x, y), 0)

    logits = model(x)
    keep = y != NODATA
    expected = nn.functional.cross_entropy(logits.permute(0, 2, 3, 1)[keep], y[keep])
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_nodata_without_ignore_index_still_fails():
    # Without ignore_index the value 255 is not a valid class.
    model = _model().eval()
    x = torch.rand(2, 3, 32, 32)
    y = torch.full((2, 32, 32), NODATA)
    with pytest.raises((IndexError, RuntimeError)):
        model.training_step((x, y), 0)


def test_class_weights_keep_ignore_index():
    weights = torch.tensor([1.0, 2.0, 3.0])
    model = _model(class_weights=weights, ignore_index=NODATA)
    assert model.loss_fn.ignore_index == NODATA
    assert torch.equal(model.loss_fn.weight, weights)


def test_custom_loss_fn_is_used_as_given():
    loss_fn = nn.CrossEntropyLoss(ignore_index=7)
    model = _model(loss_fn=loss_fn, ignore_index=NODATA)
    assert model.loss_fn is loss_fn
    assert model.val_iou.ignore_index == NODATA


def test_none_disables_ignore_index():
    model = _model(ignore_index=None)
    assert model.loss_fn.ignore_index == -100  # PyTorch's own default
    assert model.val_iou.ignore_index is None


def test_iou_does_not_count_predictions_on_ignored_pixels():
    _, target = _nodata_batch()
    pred = torch.where(target == NODATA, 1, target)

    metric = _PooledIoU(num_classes=3, ignore_index=NODATA)
    metric.update(pred, target)
    assert metric.compute().item() == pytest.approx(1.0)

    # Without ignore_index the nodata predictions are false positives for class 1.
    old = _PooledIoU(num_classes=3)
    old.update(pred, target)
    assert old.compute().item() == pytest.approx((1 + 2 / 4 + 1) / 3)


def test_in_range_ignore_index_is_left_out_of_the_mean():
    # Class 0 marks unlabelled pixels; classes 1 and 2 are predicted correctly.
    target = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    pred = torch.tensor([1, 2, 1, 2, 1, 1, 2, 2])

    metric = _PooledIoU(num_classes=3, ignore_index=0)
    metric.update(pred, target)
    assert metric.intersection.tolist() == [0, 2, 2]
    assert metric.union.tolist() == [0, 2, 2]
    assert metric.compute().item() == pytest.approx(1.0)


def test_in_range_ignore_index_errors_still_count_for_other_classes():
    # A labelled class-1 pixel predicted as the ignored class 0 is a miss for class 1.
    target = torch.tensor([1, 1, 2, 2])
    pred = torch.tensor([1, 0, 2, 2])
    metric = _PooledIoU(num_classes=3, ignore_index=0)
    metric.update(pred, target)
    assert metric.compute().item() == pytest.approx((1 / 2 + 1) / 2)


def test_all_pixels_ignored_gives_zero():
    metric = _PooledIoU(num_classes=3, ignore_index=NODATA)
    metric.update(torch.tensor([0, 1, 2]), torch.full((3,), NODATA))
    assert metric.compute().item() == 0.0


def test_val_iou_ignores_nodata_end_to_end():
    x, y = _nodata_batch()
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    trainer = pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    results = trainer.validate(
        _model(FixedPredictions, ignore_index=NODATA), loader, verbose=False
    )
    assert results[0]["val_iou"] == pytest.approx(1.0)


def test_train_timm_segmentation_passes_ignore_index(tmp_path):
    data = TensorDataset(torch.rand(2, 3, 32, 32), torch.full((2, 32, 32), NODATA))
    data.tensors[1][:, :16] = 1
    model = timm_segment.train_timm_segmentation(
        train_dataset=data,
        val_dataset=data,
        encoder_name="resnet18",
        num_classes=3,
        encoder_weights=None,
        output_dir=str(tmp_path),
        batch_size=2,
        num_epochs=1,
        num_workers=0,
        accelerator="cpu",
        ignore_index=NODATA,
        enable_progress_bar=False,
    )
    assert model.hparams.ignore_index == NODATA
    assert model.loss_fn.ignore_index == NODATA


def test_train_timm_segmentation_model_passes_ignore_index(tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(3):
        for folder, count, value in ((images, 3, 1), (labels, 1, NODATA)):
            with rasterio.open(
                folder / f"tile_{i}.tif",
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=count,
                dtype="uint8",
            ) as dst:
                dst.write(np.full((count, 8, 8), value, dtype="uint8"))

    seen = {}

    def fake_train(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(model=nn.Identity())

    monkeypatch.setattr(timm_segment, "train_timm_segmentation", fake_train)
    timm_segment.train_timm_segmentation_model(
        images_dir=str(images),
        labels_dir=str(labels),
        output_dir=str(tmp_path / "out"),
        num_classes=3,
        ignore_index=NODATA,
        verbose=False,
    )
    assert seen["ignore_index"] == NODATA
    param = inspect.signature(timm_segment.train_timm_segmentation_model).parameters
    assert param["ignore_index"].default == -100
