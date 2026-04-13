#!/usr/bin/env python3
"""Train Point Transformer V3 (PTv3) on the DALES aerial LiDAR dataset.

A standalone training script with per-epoch metrics (train/val loss,
accuracy, mIoU, per-class IoU/precision/recall), multi-GPU DDP support,
mixed-precision training, and checkpoint management.

Requirements:
    pip install torch numpy laspy
    # Pointcept (for PTv3 backbone + CUDA serialisation extensions):
    git clone https://github.com/Pointcept/Pointcept
    cd Pointcept/libs/pointops && python setup.py install && cd ../..
    export PYTHONPATH=/path/to/Pointcept:$PYTHONPATH

Workflow:
    # 1  Preprocess DALES tiles into .npy blocks (run once)
    python scripts/preprocess_dales_ptv3.py --input_dir dales_las --output_dir data/dales_ptv3

    # 2  Train -- single GPU
    python scripts/train_ptv3_dales.py --data_root data/dales_ptv3

    # 3  Train -- multi-GPU
    torchrun --nproc_per_node=4 scripts/train_ptv3_dales.py \\
        --data_root data/dales_ptv3

    # 4  Evaluate a trained checkpoint
    python scripts/train_ptv3_dales.py --data_root data/dales_ptv3 \\
        --eval_only --checkpoint checkpoints_ptv3/ptv3_dales_best.pth

    # 5  Fine-tune from HuggingFace pre-trained checkpoint
    python scripts/train_ptv3_dales.py --data_root data/your_dataset \\
        --hf_pretrained --epochs 50 --lr 0.0001 --no_amp

Pre-trained checkpoint (HuggingFace):
    https://huggingface.co/jayakumarpujar/Ptv3

References:
    Wu et al., "Point Transformer V3: Simpler, Faster, Stronger," CVPR 2024.
    Varney et al., "DALES: A Large-scale Aerial LiDAR Data Set for Semantic
    Segmentation," CVPRW 2020.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional -- fall back to a no-op wrapper
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

logger = logging.getLogger(__name__)

# =====================================================================
# Constants
# =====================================================================

DALES_CLASSES: dict[int, str] = {
    0: "unknown",
    1: "ground",
    2: "vegetation",
    3: "cars",
    4: "trucks",
    5: "power_lines",
    6: "fences",
    7: "poles",
    8: "buildings",
}
NUM_CLASSES = 9
IGNORE_INDEX = 0

# Pre-trained checkpoint on Hugging Face (trained on DALES, 100 epochs)
HF_REPO_ID = "jayakumarpujar/Ptv3"
HF_CHECKPOINT_FILENAME = "base_model_ptv3_dales_.pth"

# =====================================================================
# PTv3 backbone configuration
# Matches nuScenes-base so that pre-trained weights transfer directly;
# only num_classes (in the seg head) differs.
# =====================================================================

PTV3_BACKBONE_DEFAULTS: dict = dict(
    in_channels=4,  # xyz + return_number (DALES has zero intensity)
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("DALES",),
)

# =====================================================================
# Lovasz-Softmax loss  (Berman et al., CVPR 2018)
# Inline implementation so the script is self-contained.
# =====================================================================


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension w.r.t. sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss for IoU optimisation.

    Parameters
    ----------
    probas : (N, C)  class probabilities **after** softmax
    labels : (N,)    ground-truth labels
    """
    valid = labels != ignore_index
    if valid.sum() == 0:
        return probas.sum() * 0.0  # differentiable zero
    probas, labels = probas[valid], labels[valid]

    losses: list[torch.Tensor] = []
    for c in labels.unique():
        fg = (labels == c).float()
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))

    return torch.stack(losses).mean() if losses else probas.sum() * 0.0


# =====================================================================
# Data transforms
# =====================================================================


class Compose:
    """Chain multiple point-cloud transforms."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for t in self.transforms:
            data = t(data)
        return data


class GridSample:
    """Voxel-grid sub-sampling: keep one representative per voxel."""

    def __init__(self, grid_size: float = 0.08) -> None:
        self.grid_size = grid_size

    def __call__(self, data: dict) -> dict:
        coord = data["coord"]
        n = len(coord)
        gc = np.floor(coord / self.grid_size).astype(np.int64)
        # hash-based dedup (much faster than np.unique on rows)
        keys = gc[:, 0] * 1_000_003 + gc[:, 1] * 1_000_033 + gc[:, 2]
        _, idx = np.unique(keys, return_index=True)
        return {
            k: v[idx] if isinstance(v, np.ndarray) and v.shape[0] == n else v
            for k, v in data.items()
        }


class RandomSample:
    """Randomly sub-sample to at most *max_points*."""

    def __init__(self, max_points: int = 80_000) -> None:
        self.max_points = max_points

    def __call__(self, data: dict) -> dict:
        n = data["coord"].shape[0]
        if n <= self.max_points:
            return data
        idx = np.random.choice(n, self.max_points, replace=False)
        return {
            k: v[idx] if isinstance(v, np.ndarray) and v.shape[0] == n else v
            for k, v in data.items()
        }


class CenterShift:
    """Subtract the coordinate mean so values stay near zero."""

    def __call__(self, data: dict) -> dict:
        data["coord"] = data["coord"] - data["coord"].mean(axis=0)
        return data


class RandomRotateZ:
    """Random rotation around the vertical (Z) axis."""

    def __call__(self, data: dict) -> dict:
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        data["coord"] = data["coord"] @ rot.T
        return data


class RandomScale:
    """Uniform random scaling."""

    def __init__(self, lo: float = 0.9, hi: float = 1.1) -> None:
        self.lo, self.hi = lo, hi

    def __call__(self, data: dict) -> dict:
        data["coord"] = data["coord"] * np.random.uniform(self.lo, self.hi)
        return data


class RandomFlip:
    """Independently flip X and/or Y with 50 % probability."""

    def __call__(self, data: dict) -> dict:
        if np.random.random() > 0.5:
            data["coord"][:, 0] *= -1
        if np.random.random() > 0.5:
            data["coord"][:, 1] *= -1
        return data


class RandomJitter:
    """Additive Gaussian noise on coordinates."""

    def __init__(self, sigma: float = 0.005, clip: float = 0.02) -> None:
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data: dict) -> dict:
        noise = np.clip(
            np.random.normal(0, self.sigma, data["coord"].shape),
            -self.clip,
            self.clip,
        ).astype(np.float32)
        data["coord"] = data["coord"] + noise
        return data


def build_train_transforms(grid_size: float, max_points: int) -> Compose:
    return Compose(
        [
            GridSample(grid_size),
            RandomSample(max_points),
            CenterShift(),
            RandomRotateZ(),
            RandomScale(0.9, 1.1),
            RandomFlip(),
            RandomJitter(0.005, 0.02),
        ]
    )


def build_eval_transforms(grid_size: float, max_points: int) -> Compose:
    return Compose(
        [
            GridSample(grid_size),
            RandomSample(max_points),
            CenterShift(),
        ]
    )


# =====================================================================
# Dataset & collation
# =====================================================================


class DALESDataset(Dataset):
    """Load pre-processed DALES blocks (coord / strength / segment .npy)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Compose | None = None,
        grid_size: float = 0.08,
    ) -> None:
        self.root = Path(root) / split
        self.transform = transform
        self.grid_size = grid_size

        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        self.block_dirs = sorted(
            d
            for d in self.root.iterdir()
            if d.is_dir() and (d / "coord.npy").exists()
        )
        if not self.block_dirs:
            raise FileNotFoundError(f"No valid blocks in {self.root}")

        logger.info("DALESDataset[%s]: %d blocks", split, len(self.block_dirs))

    def __len__(self) -> int:
        return len(self.block_dirs)

    def __getitem__(self, idx: int) -> dict:
        d = self.block_dirs[idx]
        coord = np.load(d / "coord.npy")  # (N, 3) float32
        strength = np.load(d / "strength.npy")  # (N, 1) float32
        segment = np.load(d / "segment.npy")  # (N,)   int32

        data: dict = {
            "coord": coord,
            "strength": strength,
            "segment": segment,
        }

        if self.transform is not None:
            data = self.transform(data)

        # Feature vector consumed by PTv3: [x, y, z, intensity]
        strength_col = data.get(
            "strength",
            np.zeros((len(data["coord"]), 1), dtype=np.float32),
        )
        data["feat"] = np.concatenate([data["coord"], strength_col], axis=1)
        data["grid_size"] = np.float32(self.grid_size)
        return data


def point_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-size point clouds into a single batch.

    PTv3 expects ``offset`` -- a (B,) tensor of cumulative point counts.
    """
    coords: list[torch.Tensor] = []
    feats: list[torch.Tensor] = []
    segments: list[torch.Tensor] = []
    offsets: list[int] = []
    count = 0

    for item in batch:
        n = item["coord"].shape[0]
        coords.append(torch.as_tensor(item["coord"], dtype=torch.float32))
        feats.append(torch.as_tensor(item["feat"], dtype=torch.float32))
        segments.append(torch.as_tensor(item["segment"], dtype=torch.long))
        count += n
        offsets.append(count)

    return {
        "coord": torch.cat(coords),
        "feat": torch.cat(feats),
        "segment": torch.cat(segments),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "grid_size": float(batch[0].get("grid_size", 0.08)),
    }


# =====================================================================
# Metrics
# =====================================================================


class SegmentationMetrics:
    """Confusion-matrix accumulator for OA / mIoU / per-class stats."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ------------------------------------------------------------------
    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        mask = target != self.ignore_index
        p, t = pred[mask], target[mask]
        valid = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        np.add.at(self.cm, (t[valid], p[valid]), 1)

    # ------------------------------------------------------------------
    @property
    def overall_accuracy(self) -> float:
        return float(np.diag(self.cm).sum()) / max(int(self.cm.sum()), 1)

    def per_class_iou(self) -> dict[int, float]:
        iou: dict[int, float] = {}
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            tp = self.cm[c, c]
            fp = self.cm[:, c].sum() - tp
            fn = self.cm[c, :].sum() - tp
            iou[c] = float(tp) / max(int(tp + fp + fn), 1)
        return iou

    def per_class_precision_recall(self) -> dict[int, tuple[float, float]]:
        result: dict[int, tuple[float, float]] = {}
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            tp = self.cm[c, c]
            fp = self.cm[:, c].sum() - tp
            fn = self.cm[c, :].sum() - tp
            prec = float(tp) / max(int(tp + fp), 1)
            rec = float(tp) / max(int(tp + fn), 1)
            result[c] = (prec, rec)
        return result

    @property
    def mean_iou(self) -> float:
        vals = list(self.per_class_iou().values())
        return float(np.mean(vals)) if vals else 0.0

    def reset(self) -> None:
        self.cm.fill(0)


# =====================================================================
# Model
# =====================================================================


def _stub_torch_scatter_if_broken() -> None:
    """Inject a no-op ``torch_scatter`` module if the real one is unloadable.

    Pointcept's ``models/__init__.py`` -> ``default.py`` does ``import
    torch_scatter`` at module load time.  If the installed torch_scatter wheel
    was built against a different PyTorch ABI, that import raises OSError and
    kills any attempt to reach PTv3.  PTv3 itself does not call torch_scatter,
    so we register a harmless stub in ``sys.modules`` **before** Pointcept is
    imported -- subsequent ``import torch_scatter`` statements find the stub
    and never touch the broken .so file.
    """
    import sys
    import types

    if "torch_scatter" in sys.modules:
        return
    try:
        import torch_scatter  # noqa: F401
        return  # Real one works -- no stub needed.
    except Exception:
        pass  # Fall through and install stub.

    stub = types.ModuleType("torch_scatter")

    def _unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "torch_scatter is stubbed -- the installed wheel has an ABI "
            "mismatch with PyTorch.  PTv3 does not need it, but this code "
            "path should not be reached.  Reinstall torch_scatter against "
            "your current torch version if you need it."
        )

    for name in (
        "scatter",
        "scatter_add",
        "scatter_mean",
        "scatter_max",
        "scatter_min",
        "scatter_sum",
        "scatter_mul",
        "segment_csr",
        "gather_csr",
    ):
        setattr(stub, name, _unavailable)

    sys.modules["torch_scatter"] = stub


def _import_ptv3_class() -> type:
    """Import PointTransformerV3 from Pointcept, bypassing broken deps.

    Pointcept's ``models/__init__.py`` pulls in ``torch_scatter`` and other
    heavy deps that PTv3 itself does not need.  We stub torch_scatter if its
    wheel is broken, then scan the ``point_transformer_v3`` package directory
    for any variant file (m1, m2, m3_utonia, the un-suffixed base, ...) and
    load the first one that exposes a ``PointTransformerV3`` class.
    """
    import importlib.util
    import sys

    _stub_torch_scatter_if_broken()

    # Preference order for variant filenames.  Older, well-tested variants
    # first; newer research variants (like m3_utonia) last.  Anything not
    # explicitly listed but discovered by the glob is appended after these.
    preferred_stems = (
        "point_transformer_v3m1_base",  # current stable variant (main branch)
        "point_transformer_v3m1",        # older name of the same variant
        "point_transformer_v3",          # legacy un-suffixed path
        "point_transformer_v3m2",
        "point_transformer_v3m2_sonata",
        "point_transformer_v3m3",
        "point_transformer_v3m3_utonia",
    )

    # ---- 1. Discover PTv3 variant files on disk --------------------------
    variant_files: list[Path] = []
    seen: set[Path] = set()
    for base in sys.path:
        pkg_dir = Path(base) / "pointcept" / "models" / "point_transformer_v3"
        if not pkg_dir.is_dir():
            continue
        for py_file in sorted(pkg_dir.glob("point_transformer_v3*.py")):
            if py_file.name == "__init__.py":
                continue
            if py_file in seen:
                continue
            seen.add(py_file)
            variant_files.append(py_file)

    # Sort: preferred stems first (in given order), then the rest.
    def _rank(p: Path) -> tuple[int, str]:
        stem = p.stem
        try:
            return (preferred_stems.index(stem), stem)
        except ValueError:
            return (len(preferred_stems), stem)

    variant_files.sort(key=_rank)

    # ---- 2. Direct file loading (bypasses __init__.py chain) -------------
    for filepath in variant_files:
        mod_name = f"_ptv3_direct_{filepath.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            continue
        cls = getattr(mod, "PointTransformerV3", None)
        if cls is not None:
            return cls

    # ---- 3. Standard import fallback (torch_scatter stubbed) -------------
    candidates = [
        "pointcept.models.point_transformer_v3",  # package re-export
        "pointcept.models.point_transformer_v3.point_transformer_v3m1",
        "pointcept.models.point_transformer_v3.point_transformer_v3",
    ]
    for mod_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=["PointTransformerV3"])
        except (ImportError, OSError, Exception):
            continue
        cls = getattr(mod, "PointTransformerV3", None)
        if cls is not None:
            return cls

    raise ImportError(
        "Pointcept is required for PTv3.  Install from source:\n"
        "  git clone https://github.com/Pointcept/Pointcept\n"
        "  cd Pointcept/libs/pointops && python setup.py install && cd ../..\n"
        "  export PYTHONPATH=/path/to/Pointcept:$PYTHONPATH\n"
        "\n"
        "If pointcept is already on PYTHONPATH, your checkout may lack any\n"
        "point_transformer_v3*.py file in pointcept/models/point_transformer_v3/."
    )


class PTv3Segmentor(nn.Module):
    """PTv3 encoder-decoder backbone with a linear segmentation head.

    The seg head uses ``nn.LazyLinear`` so the backbone output-channel count is
    inferred at the first forward pass.  This keeps the wrapper working with
    any PTv3 variant (m1, m2, m3_utonia, ...) without hard-coding a channel
    count, since different variants ship with different default decoder widths.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone_out_channels: int | None = None,  # kept for BC, unused
        backbone_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        del backbone_out_channels  # inferred lazily from backbone output
        ptv3_cls = _import_ptv3_class()
        cfg = copy.deepcopy(backbone_cfg or PTV3_BACKBONE_DEFAULTS)
        cfg = self._adapt_cfg_to_variant(ptv3_cls, cfg)
        self.backbone = ptv3_cls(**cfg)
        self.seg_head = nn.LazyLinear(num_classes)
        self.num_classes = num_classes

    @staticmethod
    def _adapt_cfg_to_variant(ptv3_cls: type, cfg: dict) -> dict:
        """Drop kwargs the variant does not accept, and adjust channels if
        the variant uses 3D RoPE (per-head dim must be divisible by 3).
        """
        import inspect

        # 1. Strip kwargs not in the variant's __init__ signature.
        try:
            accepted = set(inspect.signature(ptv3_cls.__init__).parameters)
        except (TypeError, ValueError):
            accepted = set()
        if accepted:
            cfg = {k: v for k, v in cfg.items() if k in accepted or k == "self"}

        # 2. 3D RoPE variants (m3_utonia) need per-head channels divisible
        #    by 3.  Detect via class source; m1_base / v1 variants do not.
        needs_3d_rope = False
        try:
            src_file = inspect.getfile(ptv3_cls)
            if "utonia" in src_file.lower() or "m3" in Path(src_file).stem:
                needs_3d_rope = True
        except (TypeError, OSError):
            pass

        if needs_3d_rope:
            if "enc_channels" in cfg:
                cfg["enc_channels"] = tuple(int(c * 1.5) for c in cfg["enc_channels"])
            if "dec_channels" in cfg:
                cfg["dec_channels"] = tuple(int(c * 1.5) for c in cfg["dec_channels"])

        return cfg

    def forward(self, data_dict: dict) -> torch.Tensor:
        """Return **(N, num_classes)** logits."""
        point = self.backbone(data_dict)
        return self.seg_head(point["feat"])


def load_pretrained_backbone(
    model: PTv3Segmentor,
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[list[str], list[str]]:
    """Load pre-trained *backbone* weights -- the seg-head is skipped.

    Handles checkpoint key prefixes from both full-model and DDP saves.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    backbone_state: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        for prefix in ("backbone.", "module.backbone."):
            if key.startswith(prefix):
                backbone_state[key[len(prefix) :]] = val
                break
        else:
            # Key without a known prefix -- include unless it belongs to
            # the seg head or loss criteria.
            if not any(key.startswith(p) for p in ("seg_head", "criteria")):
                backbone_state[key] = val

    return model.backbone.load_state_dict(backbone_state, strict=False)


# =====================================================================
# Training & evaluation loops
# =====================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    accum_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run a single training epoch with optional gradient accumulation.

    When ``accum_steps > 1`` gradients are accumulated over that many
    micro-batches before an optimiser step, keeping peak GPU memory low
    while achieving a larger effective batch size.
    """
    model.train()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    n_steps = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader,
        total=len(loader),
        desc="train",
        dynamic_ncols=True,
        leave=False,
    )
    for step_idx, batch in enumerate(pbar):
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            seg_logits = model(batch)
            segment = batch["segment"]
            ce_loss = ce_criterion(seg_logits, segment)
            lv_loss = lovasz_softmax(
                torch.softmax(seg_logits.float(), dim=1),
                segment,
                ignore_index=IGNORE_INDEX,
            )
            loss = (ce_loss + lv_loss) / accum_steps

        if not math.isfinite(loss.item()):
            logger.warning("NaN/Inf loss at step %d -- skipping", n_steps)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        # Optimiser step after accumulating enough micro-batches
        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            metrics.update(
                seg_logits.argmax(dim=1).cpu().numpy(),
                segment.cpu().numpy(),
            )
            total_loss += loss.item() * accum_steps  # undo the /accum_steps
            n_steps += 1

        # Live progress line -- cheap, one-line update per step.
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{total_loss / n_steps:.4f}",
                acc=f"{metrics.overall_accuracy:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return {
        "loss": total_loss / max(n_steps, 1),
        "acc": metrics.overall_accuracy,
        "miou": metrics.mean_iou,
        "steps": n_steps,
        "time": time.time() - t0,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Evaluate on val / test set.  Returns metrics dict."""
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    n_steps = 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc="val",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in pbar:
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            seg_logits = model(batch)
            segment = batch["segment"]
            ce_loss = ce_criterion(seg_logits, segment)
            lv_loss = lovasz_softmax(
                torch.softmax(seg_logits.float(), dim=1),
                segment,
                ignore_index=IGNORE_INDEX,
            )
            loss = ce_loss + lv_loss

        metrics.update(
            seg_logits.argmax(dim=1).cpu().numpy(),
            segment.cpu().numpy(),
        )
        total_loss += loss.item()
        n_steps += 1

    return {
        "loss": total_loss / max(n_steps, 1),
        "acc": metrics.overall_accuracy,
        "miou": metrics.mean_iou,
        "per_class_iou": metrics.per_class_iou(),
        "per_class_pr": metrics.per_class_precision_recall(),
        "confusion_matrix": metrics.cm.copy(),
    }


# =====================================================================
# Pretty-print helpers
# =====================================================================


def print_epoch_report(
    epoch: int,
    total_epochs: int,
    train_stats: dict[str, float],
    val_stats: dict | None,
    best_miou: float,
    best_epoch: int,
) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(
        f"Epoch {epoch:>3d}/{total_epochs}"
        f"{'':>50s}"
        f"lr: {train_stats['lr']:.2e}"
    )
    print(sep)
    print(
        f"  Train | loss: {train_stats['loss']:.4f} | "
        f"acc: {train_stats['acc'] * 100:5.2f}% | "
        f"mIoU: {train_stats['miou'] * 100:5.2f}% | "
        f"{train_stats['steps']} steps | {train_stats['time']:.1f}s"
    )

    if val_stats is not None:
        print(
            f"  Val   | loss: {val_stats['loss']:.4f} | "
            f"acc: {val_stats['acc'] * 100:5.2f}% | "
            f"mIoU: {val_stats['miou'] * 100:5.2f}%"
        )
        iou = val_stats["per_class_iou"]
        pr = val_stats["per_class_pr"]
        sep = "-" * 45
        print(f"\n  {'Class':<15s} {'IoU':>8s}  {'Prec':>8s}  {'Recall':>8s}")
        print(f"  {sep}")
        for c in sorted(iou):
            name = DALES_CLASSES.get(c, str(c))
            p, r = pr.get(c, (0.0, 0.0))
            print(
                f"  {name:<15s} {iou[c] * 100:7.2f}%  {p * 100:7.2f}%  {r * 100:7.2f}%"
            )

    marker = (
        f"* New best mIoU: {best_miou * 100:.2f}% (epoch {best_epoch})"
        if val_stats is not None and val_stats["miou"] >= best_miou
        else f"  Best mIoU so far: {best_miou * 100:.2f}% (epoch {best_epoch})"
    )
    print(f"\n  {marker}")
    print("-" * 78)


def print_eval_report(split: str, n_blocks: int, stats: dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"Evaluation -- {split} ({n_blocks} blocks)")
    print(sep)
    print(
        f"  OA: {stats['acc'] * 100:.2f}%  "
        f"mIoU: {stats['miou'] * 100:.2f}%  "
        f"loss: {stats['loss']:.4f}"
    )
    iou = stats["per_class_iou"]
    pr = stats["per_class_pr"]
    sep = "-" * 45
    print(f"\n  {'Class':<15s} {'IoU':>8s}  {'Prec':>8s}  {'Recall':>8s}")
    print(f"  {sep}")
    for c in sorted(iou):
        name = DALES_CLASSES.get(c, str(c))
        p, r = pr.get(c, (0.0, 0.0))
        print(
            f"  {name:<15s} {iou[c] * 100:7.2f}%  {p * 100:7.2f}%  {r * 100:7.2f}%"
        )


# =====================================================================
# Checkpoint save / load
# =====================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    path: Path,
) -> None:
    raw = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "num_classes": NUM_CLASSES,
            "class_names": DALES_CLASSES,
        },
        path,
    )
    logger.info("  Checkpoint saved: %s", path)


# =====================================================================
# CLI
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PTv3 on DALES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data_root", required=True, help="Preprocessed data dir")
    g.add_argument(
        "--grid_size", type=float, default=0.15, help="Voxel grid size in metres"
    )
    g.add_argument(
        "--max_points", type=int, default=40_000, help="Max points per sample"
    )

    # Model
    g = p.add_argument_group("model")
    g.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    g.add_argument("--pretrained", default=None, help="Pre-trained backbone ckpt")
    g.add_argument(
        "--hf_pretrained",
        action="store_true",
        help="Download pre-trained DALES checkpoint from HuggingFace and resume",
    )
    g.add_argument(
        "--no_flash_attn",
        action="store_true",
        default=True,
        help="Disable flash attention (default: disabled; use --flash_attn to enable)",
    )
    g.add_argument(
        "--flash_attn",
        action="store_true",
        help="Enable flash attention (requires flash-attn package, A100/H100)",
    )

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    g.add_argument("--lr", type=float, default=0.002, help="Peak learning rate")
    g.add_argument("--weight_decay", type=float, default=0.005)
    g.add_argument(
        "--warmup_pct", type=float, default=0.05, help="LR warmup fraction"
    )
    g.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (lower = more aggressive; 1.0 is safe default)",
    )
    g.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps * world_size)",
    )

    # GPU memory presets (defaults are already low_memory safe)
    g = p.add_argument_group("memory presets (override defaults for larger GPUs)")
    g.add_argument(
        "--low_memory",
        action="store_true",
        help="16-24 GB GPU: grid=0.15, batch=2, accum=4 (same as defaults)",
    )
    g.add_argument(
        "--medium_memory",
        action="store_true",
        help="32-40 GB GPU: grid=0.10, max_pts=60k, batch=3, accum=2",
    )
    g.add_argument(
        "--high_memory",
        action="store_true",
        help="80 GB GPU (A100): grid=0.06, max_pts=100k, batch=4, accum=1",
    )

    # Checkpointing
    g = p.add_argument_group("checkpointing")
    g.add_argument("--save_dir", default="checkpoints_ptv3")
    g.add_argument("--save_every", type=int, default=10, help="Save every N epochs")
    g.add_argument("--resume", default=None, help="Resume from checkpoint")

    # Evaluation
    g = p.add_argument_group("evaluation")
    g.add_argument("--eval_only", action="store_true")
    g.add_argument("--checkpoint", default=None, help="Checkpoint for --eval_only")

    args = p.parse_args()

    # --flash_attn overrides --no_flash_attn
    if args.flash_attn:
        args.no_flash_attn = False

    # Apply memory presets (defaults are already low_memory safe).
    # --low_memory is a no-op; medium/high scale up for bigger GPUs.
    if args.medium_memory:
        args.grid_size = 0.10
        args.max_points = 60_000
        args.batch_size = 3
        args.accum_steps = 2
    elif args.high_memory:
        args.grid_size = 0.06
        args.max_points = 100_000
        args.batch_size = 4
        args.accum_steps = 1
        args.no_flash_attn = False  # A100/H100 should use flash attention

    return args


# =====================================================================
# Distributed helpers
# =====================================================================


def setup_distributed() -> tuple[int, int, int]:
    """Return ``(rank, local_rank, world_size)``.

    When launched via ``torchrun`` the env vars are set automatically.
    For a plain ``python`` invocation this returns (0, 0, 1).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    use_amp = not args.no_amp

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    # -- metadata ------------------------------------------------------
    data_root = Path(args.data_root)
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {data_root}.  "
            "Run preprocess_dales_ptv3.py first."
        )
    with open(meta_path) as f:
        metadata = json.load(f)

    class_weights = torch.tensor(
        metadata["class_weights"], dtype=torch.float32
    ).to(device)

    # -- datasets & loaders --------------------------------------------
    train_tf = build_train_transforms(args.grid_size, args.max_points)
    eval_tf = build_eval_transforms(args.grid_size, args.max_points * 2)

    train_ds = DALESDataset(data_root, "train", train_tf, args.grid_size)
    val_ds = (
        DALESDataset(data_root, "val", eval_tf, args.grid_size)
        if (data_root / "val").exists()
        else None
    )

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False)
        if world_size > 1 and val_ds
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=point_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=point_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_ds
        else None
    )

    if is_main:
        logger.info(
            "Train: %d blocks,  Val: %d blocks",
            len(train_ds),
            len(val_ds) if val_ds else 0,
        )
        logger.info("Device: %s,  World size: %d", device, world_size)

    # -- model ---------------------------------------------------------
    backbone_cfg = copy.deepcopy(PTV3_BACKBONE_DEFAULTS)
    if args.no_flash_attn:
        backbone_cfg["enable_flash"] = False

    model = PTv3Segmentor(
        num_classes=args.num_classes,
        backbone_cfg=backbone_cfg,
    ).to(device)

    # Materialise LazyLinear seg-head so the optimizer sees its parameters.
    # PTv3 variants differ in output channel count; a dummy forward lets
    # LazyLinear infer the input size from the real backbone output.
    with torch.no_grad():
        dummy_sample = train_ds[0]
        dummy_batch = point_collate_fn([dummy_sample])
        dummy_batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in dummy_batch.items()
        }
        model.eval()
        _ = model(dummy_batch)
        model.train()
    if is_main:
        out_ch = model.seg_head.in_features
        logger.info("Seg-head input channels (inferred): %d", out_ch)

    # -- auto-download from HuggingFace --------------------------------
    if args.hf_pretrained and not args.resume and not args.pretrained:
        try:
            from huggingface_hub import hf_hub_download

            if is_main:
                logger.info(
                    "Downloading pre-trained checkpoint from %s ...", HF_REPO_ID
                )
            args.resume = hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_CHECKPOINT_FILENAME
            )
            if is_main:
                logger.info("  Downloaded: %s", args.resume)
        except ImportError:
            raise SystemExit(
                "huggingface_hub is required for --hf_pretrained. "
                "Install with: pip install huggingface_hub"
            )

    if args.pretrained:
        if is_main:
            logger.info("Loading pre-trained backbone: %s", args.pretrained)
        missing, unexpected = load_pretrained_backbone(
            model, args.pretrained, device=str(device)
        )
        if is_main and missing:
            logger.info("  Missing keys: %d (expected -- fresh seg_head)", len(missing))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info("Model: PTv3Segmentor -- %s parameters", f"{n_params:,}")

    # -- loss / optimiser / scheduler ----------------------------------
    ce_criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=IGNORE_INDEX
    )

    # Scale LR by sqrt(effective_batch / base_batch) for multi-GPU stability.
    # Base batch = 8 (single-GPU low_memory: 2 * 4 accum).
    eff_batch = args.batch_size * args.accum_steps * world_size
    base_batch = args.batch_size * args.accum_steps  # single-GPU reference
    scale_factor = math.sqrt(eff_batch / base_batch) if world_size > 1 else 1.0
    scaled_lr = args.lr * scale_factor

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=scaled_lr,
        total_steps=total_steps,
        pct_start=args.warmup_pct,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # -- resume --------------------------------------------------------
    start_epoch = 1
    best_miou = 0.0
    best_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        if is_main:
            logger.info(
                "Resumed from epoch %d  (best mIoU: %.2f%%)",
                start_epoch - 1,
                best_miou * 100,
            )

    # -- eval-only mode ------------------------------------------------
    if args.eval_only:
        ckpt_path = args.checkpoint or args.resume
        if ckpt_path is None:
            raise SystemExit("Provide --checkpoint or --resume for --eval_only")
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            raw = model.module if isinstance(model, DDP) else model
            raw.load_state_dict(ckpt["model_state_dict"])

        for split in ("val", "test"):
            if not (data_root / split).exists():
                continue
            ds = DALESDataset(data_root, split, eval_tf, args.grid_size)
            ldr = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=point_collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            stats = evaluate(model, ldr, ce_criterion, device, use_amp=use_amp)
            if is_main:
                print_eval_report(split, len(ds), stats)
                # Save confusion matrix
                cm_path = Path(args.save_dir) / f"cm_{split}.npy"
                np.save(cm_path, stats["confusion_matrix"])
                logger.info("  Confusion matrix saved: %s", cm_path)
        return

    # -- training loop -------------------------------------------------
    if is_main:
        logger.info(
            "\nTraining: %d epochs, batch=%d x %d accum x %d GPU = %d effective",
            args.epochs,
            args.batch_size,
            args.accum_steps,
            world_size,
            eff_batch,
        )
        logger.info(
            "grid=%.2fm, max_pts=%d, base_lr=%g, scaled_lr=%g (sqrt scaling, %d GPUs)",
            args.grid_size,
            args.max_points,
            args.lr,
            scaled_lr,
            world_size,
        )
        logger.info(
            "AMP: %s,  Flash attention: %s",
            "on" if use_amp else "off",
            "off" if args.no_flash_attn else "on",
        )
        if device.type == "cuda":
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
            logger.info("GPU: %s (%.1f GB)\n", torch.cuda.get_device_name(device), gpu_mem)
        else:
            logger.info(
                "WARNING: Running on CPU -- PTv3 requires CUDA for "
                "serialisation.  This will likely fail.\n"
            )

    training_log: list[dict] = []
    save_path = Path(args.save_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            train_loader,
            ce_criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp=use_amp,
            accum_steps=args.accum_steps,
            max_grad_norm=args.max_grad_norm,
        )

        val_stats = None
        if val_loader is not None:
            val_stats = evaluate(
                model, val_loader, ce_criterion, device, use_amp=use_amp
            )

        current_miou = val_stats["miou"] if val_stats else train_stats["miou"]
        is_best = current_miou > best_miou
        if is_best:
            best_miou = current_miou
            best_epoch = epoch

        if is_main:
            print_epoch_report(
                epoch, args.epochs, train_stats, val_stats, best_miou, best_epoch
            )

            if is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_miou,
                    best_epoch,
                    save_path / "ptv3_dales_best.pth",
                )
            if epoch % args.save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_miou,
                    best_epoch,
                    save_path / f"ptv3_dales_epoch_{epoch:03d}.pth",
                )

            entry = {"epoch": epoch, **{k: v for k, v in train_stats.items()}}
            if val_stats:
                entry.update(
                    {
                        f"val_{k}": v
                        for k, v in val_stats.items()
                        if k not in ("per_class_iou", "per_class_pr", "confusion_matrix")
                    }
                )
            training_log.append(entry)

    # -- finalise ------------------------------------------------------
    if is_main:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            args.epochs,
            best_miou,
            best_epoch,
            save_path / "ptv3_dales_final.pth",
        )
        with open(save_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, default=float)
        logger.info("\nTraining complete!")
        logger.info("  Best mIoU: %.2f%% (epoch %d)", best_miou * 100, best_epoch)
        logger.info("  Checkpoints: %s/", args.save_dir)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
