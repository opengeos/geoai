"""evaluate_semantic pools TP/FP/FN over the whole loader (dataset-level mIoU)."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from geoai.train import evaluate_semantic


class _Replay(torch.nn.Module):
    """Returns fixed one-hot logits for each batch, in order."""

    def __init__(self, preds, num_classes):
        super().__init__()
        self.logits = (
            torch.nn.functional.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
            * 10
        )
        self.i = 0

    def forward(self, x):
        out = self.logits[self.i : self.i + len(x)]
        self.i += len(x)
        return out


def _evaluate(preds, targets, num_classes=2, batch_size=2, ignore_index=None):
    loader = DataLoader(
        TensorDataset(torch.zeros(len(targets), 3, 8, 8), targets),
        batch_size=batch_size,
    )
    return evaluate_semantic(
        _Replay(preds, num_classes),
        loader,
        torch.device("cpu"),
        torch.nn.CrossEntropyLoss(
            ignore_index=-100 if ignore_index is None else ignore_index
        ),
        num_classes,
        ignore_index=ignore_index,
    )


class TestEvaluateSemanticPooled(unittest.TestCase):
    def test_pooled_value_hand_computed(self):
        targets = torch.zeros(3, 8, 8, dtype=torch.long)
        targets[0, :, :4] = 1  # 32 building pixels, tiles 1-2 empty
        preds = torch.zeros_like(targets)  # building never predicted
        out = _evaluate(preds, targets)
        # class 0: TP=160 FP=32 FN=0 -> 160/192; class 1: TP=0 -> 0
        self.assertAlmostEqual(out["IoU"], (160 / 192 + 0) / 2, places=5)
        self.assertAlmostEqual(out["Recall"], (1.0 + 0) / 2, places=5)

    def test_never_finds_object_scores_lower_than_finding_it(self):
        targets = torch.zeros(4, 8, 8, dtype=torch.long)
        targets[0, :, :4] = 1
        model_a = targets.clone()
        model_a[1:, 0, :2] = 1  # finds building, a few stray pixels
        model_b = torch.zeros_like(targets)
        self.assertGreater(
            _evaluate(model_a, targets)["IoU"], _evaluate(model_b, targets)["IoU"]
        )

    def test_independent_of_batch_size(self):
        torch.manual_seed(0)
        targets = torch.randint(0, 3, (5, 8, 8))
        preds = torch.randint(0, 3, (5, 8, 8))
        a = _evaluate(preds, targets, num_classes=3, batch_size=1)
        b = _evaluate(preds, targets, num_classes=3, batch_size=5)
        for k in ("IoU", "F1", "Precision", "Recall"):
            self.assertAlmostEqual(a[k], b[k], places=6)

    def test_all_ignore_tile_does_not_count(self):
        for ignore in (255, -100):
            targets = torch.zeros(3, 8, 8, dtype=torch.long)
            targets[2] = ignore
            preds = torch.zeros_like(targets)
            out = _evaluate(preds, targets, ignore_index=ignore)
            # only class 0 has pixels; it is perfect
            self.assertAlmostEqual(out["IoU"], 1.0, places=5)

    def test_in_range_ignore_index_class_not_averaged(self):
        # 3 classes, ignore_index=0: class 0 is excluded from the mean even
        # when the model predicts it on valid pixels.
        targets = torch.tensor([1, 1, 2, 2]).repeat(16).reshape(1, 8, 8)
        preds = torch.tensor([0, 1, 2, 2]).repeat(16).reshape(1, 8, 8)
        out = _evaluate(preds, targets, num_classes=3, batch_size=1, ignore_index=0)
        # class 1: tp=16, fn=16 -> IoU 0.5; class 2: IoU 1.0 -> mean 0.75
        self.assertAlmostEqual(out["IoU"], 0.75, places=5)


if __name__ == "__main__":
    unittest.main()
