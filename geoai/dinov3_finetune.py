"""Module for finetuning DINOv3 for semantic segmentation of geospatial imagery.

This module provides a DPT (Dense Prediction Transformer) decoder on top of a
frozen DINOv3 backbone with optional LoRA adaptation.  It follows the same
PyTorch Lightning conventions used elsewhere in geoai (see ``timm_segment.py``).

Key components:

* :class:`DPTSegmentationHead` -- lightweight multi-scale decoder.
* :class:`LoRALinear` -- low-rank adaptation applied to ViT attention layers.
* :class:`DINOv3Segmenter` -- Lightning module that wires backbone + decoder.
* :class:`DINOv3SegmentationDataset` -- dataset for image/mask pairs (GeoTIFF).
* :func:`train_dinov3_segmentation` -- end-to-end training entry point.
* :func:`dinov3_segment_geotiff` -- sliding-window inference on a GeoTIFF.
"""

import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "DINOv3Segmenter",
    "DINOv3SegmentationDataset",
    "train_dinov3_segmentation",
    "dinov3_segment_geotiff",
]

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    LIGHTNING_AVAILABLE = False

# Resolve the base class for DINOv3Segmenter at import time so the
# class definition never fails even when Lightning is absent.
_LightningBase: type = pl.LightningModule if LIGHTNING_AVAILABLE else nn.Module  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------

# Layer indices for extracting multi-scale features from DINOv3 ViTs.
# Keyed by number of transformer blocks.
_EXTRACTION_LAYERS: Dict[int, List[int]] = {
    12: [2, 5, 8, 11],  # ViT-S, ViT-B
    24: [5, 11, 17, 23],  # ViT-L
    32: [7, 15, 23, 31],  # ViT-H / 7B
    40: [9, 19, 29, 39],  # ViT-g
}


def _get_extraction_layers(num_blocks: int) -> List[int]:
    """Return the intermediate layer indices for a given ViT depth.

    Args:
        num_blocks: Total number of transformer blocks in the backbone.

    Returns:
        List of four layer indices (0-indexed) used for multi-scale feature
        extraction.

    Raises:
        ValueError: If *num_blocks* is not in the supported set.
    """
    if num_blocks in _EXTRACTION_LAYERS:
        return _EXTRACTION_LAYERS[num_blocks]
    raise ValueError(
        f"Unsupported number of transformer blocks: {num_blocks}. "
        f"Supported values: {sorted(_EXTRACTION_LAYERS.keys())}."
    )


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer applied to an existing ``nn.Linear``.

    Only the low-rank matrices A and B are trainable; the original weight
    is kept frozen.  During ``forward``, the output is::

        out = original_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    Args:
        original: The ``nn.Linear`` layer to adapt.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor (defaults to *rank*).
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 4,
        alpha: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the original weight.
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + lora_out * (self.alpha / self.rank)


def _apply_lora_to_backbone(
    backbone: nn.Module,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Tuple[str, ...] = ("qkv",),
) -> None:
    """Replace targeted ``nn.Linear`` modules with :class:`LoRALinear` in-place.

    Args:
        backbone: The ViT backbone to modify.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Tuple of substrings identifying which ``nn.Linear``
            modules to replace.  Only leaves whose name ends with one of
            these strings are adapted.
    """
    for name, module in list(backbone.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = backbone
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha))


class DPTSegmentationHead(nn.Module):
    """Dense Prediction Transformer head for per-pixel segmentation.

    Extracts features at four intermediate depths from a ViT backbone,
    projects each to a common channel dimension, progressively fuses them
    with bilinear upsampling, and produces per-pixel class logits.

    Args:
        embed_dim: Hidden dimension of the backbone.
        num_classes: Number of output segmentation classes.
        features: Internal feature dimension used throughout the decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        features: int = 256,
    ) -> None:
        super().__init__()
        # Project each backbone layer to *features* channels.
        self.projects = nn.ModuleList(
            [nn.Conv2d(embed_dim, features, kernel_size=1) for _ in range(4)]
        )
        # Refine each projected feature map.
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        # Fuse all four scales.
        self.fuse = nn.Sequential(
            nn.Conv2d(features * 4, features, kernel_size=1),
            nn.BatchNorm2d(features),
            nn.GELU(),
        )
        self.head = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(
        self, multi_scale_features: List[torch.Tensor], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Produce logits from multi-scale ViT features.

        Args:
            multi_scale_features: Four tensors of shape ``(B, C, H_p, W_p)``
                from the backbone intermediate layers.
            target_size: ``(H, W)`` of the input image -- logits are
                interpolated to this resolution.

        Returns:
            Tensor of shape ``(B, num_classes, H, W)``.
        """
        refined = []
        for feat, proj, ref in zip(multi_scale_features, self.projects, self.refine):
            refined.append(ref(proj(feat)))

        # Upsample all to the spatial size of the largest (first) feature map.
        target_h, target_w = refined[0].shape[2], refined[0].shape[3]
        upsampled = [refined[0]]
        for r in refined[1:]:
            upsampled.append(
                F.interpolate(
                    r, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            )

        fused = self.fuse(torch.cat(upsampled, dim=1))
        logits = self.head(fused)
        # Final interpolation to input resolution.
        logits = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
        return logits


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class DINOv3Segmenter(_LightningBase):
    """PyTorch Lightning module for DINOv3-based semantic segmentation.

    The backbone is loaded from ``torch.hub`` (or a local clone) and is
    **frozen** by default.  Optional LoRA adaptation can be enabled to
    cheaply adapt the backbone while keeping memory usage low.  The DPT
    decoder is always trainable.

    Args:
        model_name: DINOv3 hub model identifier (e.g. ``"dinov3_vitl16"``).
        weights_path: Path to pretrained backbone weights.  When ``None``
            the SAT-493M checkpoint is downloaded from Hugging Face.
        num_classes: Number of segmentation classes (>= 2).
        decoder_features: Hidden dimension of the DPT decoder.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        freeze_backbone: Whether to freeze all backbone parameters.
        use_lora: Apply LoRA adaptation to the backbone attention layers.
        lora_rank: Rank of the LoRA decomposition.
        lora_alpha: LoRA scaling factor (defaults to *lora_rank*).
        loss_fn: Custom loss function.  Defaults to ``CrossEntropyLoss``.
        class_weights: Optional per-class weights for the loss.
        ignore_index: Label index to ignore in the loss (e.g. 255 for nodata).
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        weights_path: Optional[str] = None,
        num_classes: int = 2,
        decoder_features: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: Optional[float] = None,
        loss_fn: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ) -> None:
        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "PyTorch Lightning is required. Install it with: pip install lightning"
            )
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        # --- backbone ---
        self.backbone = self._load_backbone(model_name, weights_path)
        self.patch_size: int = self.backbone.patch_size
        self.embed_dim: int = self.backbone.embed_dim
        num_blocks = len(self.backbone.blocks)
        self.extraction_layers = _get_extraction_layers(num_blocks)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if use_lora:
            _apply_lora_to_backbone(self.backbone, rank=lora_rank, alpha=lora_alpha)

        # --- decoder ---
        self.decoder = DPTSegmentationHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            features=decoder_features,
        )

        # --- loss ---
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=ignore_index
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # -- backbone loading (mirrors DINOv3GeoProcessor) --

    @staticmethod
    def _load_backbone(
        model_name: str,
        weights_path: Optional[str],
    ) -> nn.Module:
        """Load and return a DINOv3 ViT backbone.

        When *weights_path* is ``None`` the SAT-493M checkpoint is
        downloaded from Hugging Face via ``huggingface_hub``.
        """
        from huggingface_hub import hf_hub_download

        dinov3_location = os.getenv("DINOV3_LOCATION", "facebookresearch/dinov3")
        source = "local" if dinov3_location != "facebookresearch/dinov3" else "github"

        if (
            dinov3_location != "facebookresearch/dinov3"
            and dinov3_location not in sys.path
        ):
            sys.path.append(dinov3_location)

        model = torch.hub.load(
            repo_or_dir=dinov3_location,
            model=model_name,
            source=source,
            pretrained=False,
            weights=None,
            trust_repo=True,
            skip_validation=True,
        )

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            if weights_path:
                logger.warning(
                    "weights_path '%s' not found; downloading SAT-493M defaults",
                    weights_path,
                )
            weights_file = hf_hub_download(
                repo_id="giswqs/geoai", filename="dinov3_vitl16_sat493m.pth"
            )
            state_dict = torch.load(weights_file, map_location="cpu")

        model.load_state_dict(state_dict, strict=False)
        return model

    # -- forward / steps --

    def _extract_multi_scale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return four intermediate feature maps from the backbone.

        Each tensor has shape ``(B, embed_dim, H_p, W_p)`` where
        ``H_p = H / patch_size`` and ``W_p = W / patch_size``.

        We pass the explicit list of layer indices (e.g. ``[2, 5, 8, 11]``)
        rather than an integer ``n`` so we extract features at evenly-spaced
        depths instead of only from the last ``n`` blocks.
        """
        features = self.backbone.get_intermediate_layers(
            x, n=self.extraction_layers, reshape=True, norm=True
        )
        return list(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce per-pixel class logits.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)`` where ``H`` and ``W``
               are divisible by *patch_size*.

        Returns:
            Logits of shape ``(B, num_classes, H, W)``.
        """
        target_size = (x.shape[2], x.shape[3])
        multi_scale = self._extract_multi_scale(x)
        return self.decoder(multi_scale, target_size)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        iou = self._compute_miou(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        iou = self._compute_miou(pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        iou = self._compute_miou(pred, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_iou", iou, on_epoch=True)
        return loss

    def _compute_miou(
        self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
    ) -> torch.Tensor:
        """Compute mean Intersection over Union across all classes."""
        num_classes = self.hparams.num_classes
        ignore_index = self.hparams.ignore_index
        ious: List[torch.Tensor] = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            pred_cls = pred == cls
            target_cls = target == cls
            # Exclude ignored pixels from the computation.
            valid = target != ignore_index
            pred_cls = pred_cls & valid
            target_cls = target_cls & valid
            intersection = (pred_cls & target_cls).float().sum()
            union = (pred_cls | target_cls).float().sum()
            if union == 0:
                continue
            ious.append((intersection + smooth) / (union + smooth))
        if not ious:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(ious).mean()

    def configure_optimizers(self) -> Dict[str, Any]:
        # Only pass parameters that require grad.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"predictions": preds, "probabilities": probs}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DINOv3SegmentationDataset(Dataset):
    """Dataset for image/mask pairs stored as GeoTIFF files.

    Images are read with ``rasterio``, selecting the first three bands if
    more are available, and normalised to ``[0, 1]``.  Masks are single-band
    integer arrays.

    Args:
        image_paths: List of paths to image GeoTIFF files.
        mask_paths: List of paths to mask GeoTIFF files.
        patch_size: ViT patch size -- spatial dimensions are padded to a
            multiple of this value.
        target_size: If given, images and masks are resized to
            ``(target_size, target_size)`` before patching.
        num_channels: Number of image channels to use.
        transform: Optional callable ``(image, mask) -> (image, mask)``
            applied after loading and normalising.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        patch_size: int = 16,
        target_size: Optional[int] = None,
        num_channels: int = 3,
        transform: Optional[Any] = None,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Number of images ({len(image_paths)}) does not match "
                f"number of masks ({len(mask_paths)})"
            )
        if len(image_paths) == 0:
            raise ValueError("image_paths must not be empty")

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.target_size = target_size
        self.num_channels = num_channels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import rasterio

        # --- image ---
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read().astype(np.float32)  # (C, H, W)

        # Channel selection / padding.
        if image.shape[0] > self.num_channels:
            image = image[: self.num_channels]
        elif image.shape[0] < self.num_channels:
            pad = np.zeros(
                (self.num_channels, image.shape[1], image.shape[2]), dtype=np.float32
            )
            pad[: image.shape[0]] = image
            image = pad

        # Normalise to [0, 1].
        img_max = image.max()
        if img_max > 1.0:
            image = image / 255.0

        # --- mask ---
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # Optional resize.
        if self.target_size is not None:
            from PIL import Image as PILImage

            # Resize image (C, H, W) -> (C, target_size, target_size).
            img_pil = PILImage.fromarray(
                (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)
            )
            img_pil = img_pil.resize(
                (self.target_size, self.target_size), PILImage.BILINEAR
            )
            image = np.transpose(
                np.array(img_pil).astype(np.float32) / 255.0, (2, 0, 1)
            )

            mask_pil = PILImage.fromarray(mask.astype(np.uint8))
            mask_pil = mask_pil.resize(
                (self.target_size, self.target_size), PILImage.NEAREST
            )
            mask = np.array(mask_pil).astype(np.int64)

        # Pad to multiple of patch_size.
        _, h, w = image.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            mask = np.pad(
                mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255
            )

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_dinov3_segmentation(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    model_name: str = "dinov3_vitl16",
    weights_path: Optional[str] = None,
    num_classes: int = 2,
    decoder_features: int = 256,
    output_dir: str = "output",
    batch_size: int = 4,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    freeze_backbone: bool = True,
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: Optional[float] = None,
    class_weights: Optional[List[float]] = None,
    ignore_index: int = 255,
    accelerator: str = "auto",
    devices: str = "auto",
    monitor_metric: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 1,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> "DINOv3Segmenter":
    """Train a DINOv3-based semantic segmentation model.

    This is the main entry point for finetuning.  It creates a
    :class:`DINOv3Segmenter`, sets up data loaders, callbacks, and a
    Lightning trainer, then runs ``trainer.fit()``.

    Args:
        train_dataset: Training dataset returning ``(image, mask)`` tuples.
        val_dataset: Optional validation dataset.
        test_dataset: Optional test dataset.
        model_name: DINOv3 model name (``"dinov3_vitl16"``, etc.).
        weights_path: Optional path to pretrained backbone weights.
        num_classes: Number of segmentation classes.
        decoder_features: Hidden dim of the DPT decoder.
        output_dir: Directory for checkpoints and logs.
        batch_size: Training batch size.
        num_epochs: Maximum training epochs.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        num_workers: DataLoader workers.
        freeze_backbone: Freeze backbone parameters.
        use_lora: Apply LoRA to backbone attention layers.
        lora_rank: LoRA rank.
        lora_alpha: LoRA scaling factor.
        class_weights: Per-class loss weights.
        ignore_index: Label value to ignore in loss (default 255).
        accelerator: Lightning accelerator.
        devices: Lightning devices.
        monitor_metric: Metric to monitor for checkpointing.
        mode: ``"min"`` or ``"max"`` for the monitored metric.
        patience: Early-stopping patience.
        save_top_k: Number of best checkpoints to keep.
        checkpoint_path: Path to resume training from.
        **kwargs: Extra arguments forwarded to ``pl.Trainer``.

    Returns:
        The trained :class:`DINOv3Segmenter` model.
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "PyTorch Lightning is required. Install it with: pip install lightning"
        )

    output_dir = os.path.abspath(output_dir)
    parent_dir = os.path.dirname(output_dir)
    if parent_dir and not os.path.isdir(parent_dir):
        raise FileNotFoundError(
            f"Parent directory of output_dir does not exist: {parent_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    model = DINOv3Segmenter(
        model_name=model_name,
        weights_path=weights_path,
        num_classes=num_classes,
        decoder_features=decoder_features,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        class_weights=weight_tensor,
        ignore_index=ignore_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"dinov3_seg_{{epoch:02d}}_{{val_loss:.4f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    early_stop = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
        verbose=True,
    )
    callbacks.append(early_stop)

    csv_logger = CSVLogger(model_dir, name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=10,
        **kwargs,
    )

    logger.info("Training DINOv3 segmentation for %d epochs...", num_epochs)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("Testing model on test set...")
        trainer.test(model, dataloaders=test_loader)

    logger.info("Best model saved at: %s", checkpoint_callback.best_model_path)
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def dinov3_segment_geotiff(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    model_name: str = "dinov3_vitl16",
    weights_path: Optional[str] = None,
    num_classes: int = 2,
    decoder_features: int = 256,
    window_size: int = 512,
    overlap: int = 256,
    batch_size: int = 4,
    device: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """Run sliding-window segmentation inference on a GeoTIFF.

    Args:
        input_path: Path to input GeoTIFF.
        output_path: Path for the output segmentation mask GeoTIFF.
        checkpoint_path: Path to a Lightning ``.ckpt`` or plain ``.pth``
            file containing the trained :class:`DINOv3Segmenter`.
        model_name: DINOv3 model name used during training.
        weights_path: Backbone weights path (only needed for ``.pth`` files).
        num_classes: Number of segmentation classes used during training.
        decoder_features: Decoder hidden dim used during training.
        window_size: Spatial size of the sliding window (pixels).
        overlap: Overlap between adjacent windows (pixels).
        batch_size: Number of windows to process at once.
        device: Device string (auto-detected if ``None``).
        quiet: Suppress progress output.
    """
    import rasterio
    from rasterio.windows import Window
    from tqdm import tqdm

    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than window_size ({window_size})"
        )

    if device is None:
        from .utils.device import get_device

        dev = get_device()
    else:
        dev = torch.device(device)

    # Load model.
    if checkpoint_path.endswith(".ckpt"):
        model_module = DINOv3Segmenter.load_from_checkpoint(
            checkpoint_path, map_location=dev
        )
    else:
        model_module = DINOv3Segmenter(
            model_name=model_name,
            weights_path=weights_path,
            num_classes=num_classes,
            decoder_features=decoder_features,
        )
        state = torch.load(checkpoint_path, map_location=dev)
        model_module.load_state_dict(state, strict=False)

    model_module.eval()
    model_module = model_module.to(dev)
    patch_size = model_module.patch_size

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.shape
        num_channels = min(src.count, 3)

        stride = window_size - overlap
        n_rows = max(1, int(np.ceil((height - overlap) / stride)))
        n_cols = max(1, int(np.ceil((width - overlap) / stride)))

        if not quiet:
            logger.info(
                "Processing %d x %d = %d windows", n_rows, n_cols, n_rows * n_cols
            )

        # Accumulate per-class votes for overlapping windows.
        votes = np.zeros((num_classes, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        # Precompute the padded tensor size so every window in a batch
        # has the same spatial dimensions.
        padded_h = window_size + (patch_size - window_size % patch_size) % patch_size
        padded_w = padded_h  # square

        def _prepare_window(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
            """Normalise, channel-select, and pad a raw window."""
            if img.shape[0] > num_channels:
                img = img[:num_channels]
            elif img.shape[0] < num_channels:
                pad_arr = np.zeros(
                    (num_channels, img.shape[1], img.shape[2]), dtype=np.float32
                )
                pad_arr[: img.shape[0]] = img
                img = pad_arr

            if img.max() > 1.0:
                img = img / 255.0

            h, w = img.shape[1], img.shape[2]
            if h < padded_h or w < padded_w:
                padded = np.zeros((num_channels, padded_h, padded_w), dtype=np.float32)
                padded[:, :h, :w] = img
                img = padded
            return img, h, w

        def _flush_batch(
            batch_imgs: List[np.ndarray],
            batch_meta: List[Tuple[int, int, int, int, int, int]],
        ) -> None:
            """Run inference on a collected batch and accumulate votes."""
            tensor = torch.from_numpy(np.stack(batch_imgs)).to(dev)
            logits = model_module(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for k, (rs, re, cs, ce, h, w) in enumerate(batch_meta):
                votes[:, rs:re, cs:ce] += probs[k, :, :h, :w]
                count[rs:re, cs:ce] += 1.0

        with torch.no_grad():
            batch_imgs: List[np.ndarray] = []
            batch_meta: List[Tuple[int, int, int, int, int, int]] = []

            total_windows = n_rows * n_cols
            pbar = tqdm(total=total_windows, disable=quiet, desc="Windows")

            for i in range(n_rows):
                for j in range(n_cols):
                    row_start = i * stride
                    col_start = j * stride
                    row_end = min(row_start + window_size, height)
                    col_end = min(col_start + window_size, width)

                    win = Window(
                        col_start,
                        row_start,
                        col_end - col_start,
                        row_end - row_start,
                    )
                    raw = src.read(window=win).astype(np.float32)
                    img, h, w = _prepare_window(raw)

                    batch_imgs.append(img)
                    batch_meta.append((row_start, row_end, col_start, col_end, h, w))

                    if len(batch_imgs) == batch_size:
                        _flush_batch(batch_imgs, batch_meta)
                        pbar.update(len(batch_imgs))
                        batch_imgs.clear()
                        batch_meta.clear()

            # Flush remaining windows.
            if batch_imgs:
                _flush_batch(batch_imgs, batch_meta)
                pbar.update(len(batch_imgs))

            pbar.close()

        # Majority vote (avoid division by zero).
        count = np.maximum(count, 1.0)
        votes /= count[np.newaxis, :, :]
        output = np.argmax(votes, axis=0).astype(np.uint8)

    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(output, 1)

    if not quiet:
        logger.info("Segmentation saved to %s", output_path)
