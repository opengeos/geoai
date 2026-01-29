"""Module for training pixel-level regression models using timm encoders with PyTorch Lightning.

This module provides tools for remote sensing regression tasks like predicting NDVI,
biomass, temperature, or other continuous values at the pixel level.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import CSVLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


class _CompactProgressBar(TQDMProgressBar):
    """Progress bar that shows key metrics in the postfix, updated in place."""

    def get_metrics(self, trainer, pl_module):
        # Don't let Lightning set the postfix — we control it
        return {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.train_progress_bar is not None:
            self.train_progress_bar.set_postfix_str("")

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        metrics = trainer.callback_metrics
        if metrics and self.train_progress_bar is not None:
            keys = [
                "train_loss_epoch",
                "train_r2",
                "val_loss",
                "val_r2",
                "val_rmse",
            ]
            parts = []
            for k in keys:
                v = metrics.get(k)
                if v is not None:
                    val = v.item() if hasattr(v, "item") else v
                    if isinstance(val, float):
                        parts.append(f"{k}={val:.4g}")
            if parts:
                self.train_progress_bar.set_postfix_str(", ".join(parts))


def _prepare_normalization_stats(
    mean: List[float], std: List[float], num_channels: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare mean/std arrays that match the number of channels."""
    if len(mean) == num_channels and len(std) == num_channels:
        mean_vals = mean
        std_vals = std
    elif len(mean) == 3 and len(std) == 3 and num_channels > 3:
        mean_pad = [float(np.mean(mean))] * (num_channels - 3)
        std_pad = [float(np.mean(std))] * (num_channels - 3)
        mean_vals = list(mean) + mean_pad
        std_vals = list(std) + std_pad
    elif len(mean) == 1 and len(std) == 1:
        mean_vals = list(mean) * num_channels
        std_vals = list(std) * num_channels
    elif len(mean) >= num_channels and len(std) >= num_channels:
        mean_vals = list(mean)[:num_channels]
        std_vals = list(std)[:num_channels]
    else:
        raise ValueError(
            "Normalization stats length must match channels. "
            f"Got mean={len(mean)}, std={len(std)}, channels={num_channels}."
        )

    mean_arr = np.array(mean_vals, dtype=np.float32)[:, None, None]
    std_arr = np.array(std_vals, dtype=np.float32)[:, None, None]
    return mean_arr, std_arr


def _infer_preprocessing_params(
    encoder_name: str, encoder_weights: Optional[str]
) -> Optional[Dict[str, Any]]:
    if encoder_weights is None:
        return None
    if not SMP_AVAILABLE:
        return None
    try:
        return smp.encoders.get_preprocessing_params(
            encoder_name, pretrained=encoder_weights
        )
    except Exception:
        return None


class PixelRegressionModel(pl.LightningModule):
    """
    PyTorch Lightning module for pixel-level regression using encoder-decoder architectures.

    Uses segmentation-models-pytorch (SMP) with timm encoders but configured for
    regression (single channel output with continuous values).
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        architecture: str = "unet",
        in_channels: int = 3,
        encoder_weights: str = "imagenet",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_encoder: bool = False,
        loss_fn: Optional[nn.Module] = None,
        loss_type: str = "mse",
        **decoder_kwargs: Any,
    ):
        """
        Initialize PixelRegressionModel.

        Args:
            encoder_name (str): Name of timm encoder (e.g., 'resnet50', 'efficientnet_b0').
            architecture (str): Segmentation architecture ('unet', 'unetplusplus', 'deeplabv3',
                'deeplabv3plus', 'fpn', 'pspnet', 'linknet', 'manet', 'pan').
            in_channels (int): Number of input channels (3 for RGB, 4 for RGBN, etc.).
            encoder_weights (str): Pretrained weights for encoder ('imagenet', None).
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            freeze_encoder (bool): Freeze encoder weights during training.
            loss_fn (nn.Module, optional): Custom loss function.
            loss_type (str): Type of loss if loss_fn is None ('mse', 'l1', 'mae', 'huber').
            **decoder_kwargs: Additional arguments for decoder.
        """
        super().__init__()

        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation-models-pytorch is required. "
                "Install it with: pip install segmentation-models-pytorch"
            )

        self.save_hyperparameters()

        # Create segmentation model with 1 output class for regression
        try:
            self.model = smp.create_model(
                arch=architecture,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,  # Single channel for regression
                **decoder_kwargs,
            )
        except Exception as e:
            available_archs = [
                "unet",
                "unetplusplus",
                "manet",
                "linknet",
                "fpn",
                "pspnet",
                "deeplabv3",
                "deeplabv3plus",
                "pan",
                "upernet",
            ]
            raise ValueError(
                f"Failed to create model with architecture '{architecture}' and encoder '{encoder_name}'. "
                f"Error: {str(e)}. "
                f"Available architectures: {', '.join(available_archs)}."
            )

        if freeze_encoder:
            self._freeze_encoder()

        # Set up loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            loss_type = loss_type.lower()
            if loss_type == "mse":
                self.loss_fn = nn.MSELoss()
            elif loss_type in ["l1", "mae"]:
                self.loss_fn = nn.L1Loss()
            elif loss_type in ["huber", "smooth_l1"]:
                self.loss_fn = nn.SmoothL1Loss()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _freeze_encoder(self):
        """Freeze encoder weights."""
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x).squeeze(1)  # Remove channel dim: (B, 1, H, W) -> (B, H, W)

    def _compute_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute pixel-wise regression metrics."""
        # Flatten for metrics computation
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # MSE and RMSE
        mse = torch.mean((preds_flat - targets_flat) ** 2)
        rmse = torch.sqrt(mse)

        # MAE
        mae = torch.mean(torch.abs(preds_flat - targets_flat))

        # R² (coefficient of determination)
        ss_res = torch.sum((targets_flat - preds_flat) ** 2)
        ss_tot = torch.sum((targets_flat - targets_flat.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        metrics = self._compute_metrics(preds, y)

        pb = getattr(self, "_prog_bar_metrics", True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=pb)
        self.log("train_rmse", metrics["rmse"], on_step=False, on_epoch=True)
        self.log("train_r2", metrics["r2"], on_step=False, on_epoch=True, prog_bar=pb)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        metrics = self._compute_metrics(preds, y)

        pb = getattr(self, "_prog_bar_metrics", True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=pb)
        self.log("val_rmse", metrics["rmse"], on_epoch=True, prog_bar=pb)
        self.log("val_mae", metrics["mae"], on_epoch=True)
        self.log("val_r2", metrics["r2"], on_epoch=True, prog_bar=pb)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        metrics = self._compute_metrics(preds, y)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_rmse", metrics["rmse"], on_epoch=True)
        self.log("test_mae", metrics["mae"], on_epoch=True)
        self.log("test_r2", metrics["r2"], on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)


class PixelRegressionDataset(Dataset):
    """
    Dataset for pixel-level regression from paired image and target rasters.

    Loads image patches and corresponding target patches for training
    pixel-wise regression models.
    """

    def __init__(
        self,
        image_paths: List[str],
        target_paths: List[str],
        input_bands: Optional[List[int]] = None,
        target_band: int = 1,
        transform: Optional[Callable] = None,
        normalize_input: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        target_nodata: Optional[float] = None,
    ):
        """
        Initialize PixelRegressionDataset.

        Args:
            image_paths (List[str]): List of paths to input image patches.
            target_paths (List[str]): List of paths to target raster patches.
            input_bands (List[int], optional): Band indices to use (1-indexed). If None, uses all.
            target_band (int): Band index for target raster (1-indexed).
            transform (callable, optional): Transform to apply to images.
        normalize_input (bool): Whether to normalize input to [0, 1].
        image_mean (List[float], optional): Per-channel mean for normalization.
        image_std (List[float], optional): Per-channel std for normalization.
        target_nodata (float, optional): NoData value for targets.
        """
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.input_bands = input_bands
        self.target_band = target_band
        self.transform = transform
        self.normalize_input = normalize_input
        self.image_mean = image_mean
        self.image_std = image_std
        self.target_nodata = target_nodata
        self._mean_array = None
        self._std_array = None

        if len(image_paths) != len(target_paths):
            raise ValueError("Number of images must match number of targets")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import rasterio

        # Load input image
        with rasterio.open(self.image_paths[idx]) as src:
            if self.input_bands is not None:
                image = src.read(self.input_bands)
            else:
                image = src.read()

        # Load target
        with rasterio.open(self.target_paths[idx]) as src:
            target = src.read(self.target_band)

        # Handle NaN
        image = np.nan_to_num(image, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        # Normalize input
        image = image.astype(np.float32)
        if self.normalize_input:
            data_max = np.abs(image).max()
            if data_max <= 1.5:
                image = np.clip(image, 0, 1)
            else:
                image = np.clip(image, 0, 10000) / 10000.0

        if self.image_mean is not None and self.image_std is not None:
            mean, std = _prepare_normalization_stats(
                self.image_mean, self.image_std, image.shape[0]
            )
            image = (image - mean) / std

        target = target.astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image)
        target = torch.from_numpy(target)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def create_regression_tiles(
    input_raster: str,
    target_raster: str,
    output_dir: str,
    tile_size: int = 256,
    stride: Optional[int] = None,
    input_bands: Optional[List[int]] = None,
    target_band: int = 1,
    min_valid_ratio: float = 0.8,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """
    Create paired image and target tiles from input and target rasters.

    Args:
        input_raster (str): Path to input raster (e.g., Landsat imagery).
        target_raster (str): Path to target raster (e.g., NDVI).
        output_dir (str): Directory to save tiles.
        tile_size (int): Size of each tile (tile_size x tile_size pixels).
        stride (int, optional): Stride between tiles. Defaults to tile_size (no overlap).
        input_bands (List[int], optional): Band indices to use (1-indexed).
        target_band (int): Band index for target raster (1-indexed).
        min_valid_ratio (float): Minimum ratio of valid pixels in tile.
        target_min (float, optional): Minimum valid target value.
        target_max (float, optional): Maximum valid target value.

    Returns:
        Tuple of (image_paths, target_paths): Lists of tile paths.
    """
    import rasterio

    image_dir = os.path.join(output_dir, "images")
    target_dir = os.path.join(output_dir, "targets")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    if stride is None:
        stride = tile_size

    image_paths = []
    target_paths = []

    with rasterio.open(input_raster) as src_input:
        with rasterio.open(target_raster) as src_target:
            height = src_input.height
            width = src_input.width

            if input_bands is None:
                input_bands = list(range(1, src_input.count + 1))

            n_tiles_y = (height - tile_size) // stride + 1
            n_tiles_x = (width - tile_size) // stride + 1

            print(f"Input raster: {width}x{height}, {src_input.count} bands")
            print(f"Target raster: {src_target.width}x{src_target.height}")
            print(f"Tile size: {tile_size}x{tile_size}, stride: {stride}")
            print(
                f"Expected tiles: {n_tiles_y} x {n_tiles_x} = {n_tiles_y * n_tiles_x}"
            )

            tile_idx = 0
            valid_tiles = 0
            skipped_nodata = 0
            skipped_range = 0

            for row in tqdm(
                range(0, height - tile_size + 1, stride), desc="Creating tiles"
            ):
                for col in range(0, width - tile_size + 1, stride):
                    window = rasterio.windows.Window(col, row, tile_size, tile_size)

                    # Read tiles
                    input_tile = src_input.read(input_bands, window=window)
                    target_tile = src_target.read(target_band, window=window)

                    # Check for valid pixels
                    valid_mask = ~np.isnan(input_tile).any(axis=0) & ~np.isnan(
                        target_tile
                    )
                    valid_ratio = valid_mask.sum() / (tile_size * tile_size)

                    if valid_ratio < min_valid_ratio:
                        tile_idx += 1
                        skipped_nodata += 1
                        continue

                    # Check target range - skip tiles with >5% out-of-range values
                    valid_target = target_tile[valid_mask]
                    out_of_range_ratio = 0.0
                    if target_min is not None or target_max is not None:
                        out_of_range = np.zeros_like(valid_target, dtype=bool)
                        if target_min is not None:
                            out_of_range |= valid_target < target_min
                        if target_max is not None:
                            out_of_range |= valid_target > target_max
                        out_of_range_ratio = out_of_range.sum() / len(valid_target)

                        # Skip if more than 5% of pixels are out of range
                        if out_of_range_ratio > 0.05:
                            tile_idx += 1
                            skipped_range += 1
                            continue

                    # Replace NaN with 0
                    input_tile = np.nan_to_num(input_tile, nan=0.0)
                    target_tile = np.nan_to_num(target_tile, nan=0.0)

                    # Clip target values to valid range (important!)
                    if target_min is not None or target_max is not None:
                        target_tile = np.clip(
                            target_tile,
                            target_min if target_min is not None else -np.inf,
                            target_max if target_max is not None else np.inf,
                        )

                    # Save tiles
                    image_path = os.path.join(image_dir, f"tile_{tile_idx:06d}.tif")
                    target_path = os.path.join(target_dir, f"tile_{tile_idx:06d}.tif")

                    # Save input tile
                    profile = src_input.profile.copy()
                    profile.update(
                        width=tile_size,
                        height=tile_size,
                        count=len(input_bands),
                        dtype=input_tile.dtype,
                        transform=rasterio.windows.transform(
                            window, src_input.transform
                        ),
                        tiled=False,  # Disable tiling for small tiles
                    )
                    # Remove block size settings that cause warnings
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)
                    with rasterio.open(image_path, "w", **profile) as dst:
                        dst.write(input_tile)

                    # Save target tile
                    profile = src_target.profile.copy()
                    profile.update(
                        width=tile_size,
                        height=tile_size,
                        count=1,
                        dtype=target_tile.dtype,
                        transform=rasterio.windows.transform(
                            window, src_target.transform
                        ),
                        tiled=False,
                    )
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)
                    with rasterio.open(target_path, "w", **profile) as dst:
                        dst.write(target_tile[np.newaxis, :, :])

                    image_paths.append(image_path)
                    target_paths.append(target_path)
                    valid_tiles += 1
                    tile_idx += 1

    print(f"\nCreated {valid_tiles} valid tiles out of {tile_idx} total")
    print(f"Skipped due to nodata: {skipped_nodata}")
    print(f"Skipped due to target range: {skipped_range}")

    return image_paths, target_paths


def train_pixel_regressor(
    train_image_paths: List[str],
    train_target_paths: List[str],
    val_image_paths: Optional[List[str]] = None,
    val_target_paths: Optional[List[str]] = None,
    encoder_name: str = "resnet50",
    architecture: str = "unet",
    in_channels: int = 3,
    encoder_weights: str = "imagenet",
    output_dir: str = "output",
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    freeze_encoder: bool = False,
    loss_type: str = "mse",
    normalize_input: bool = True,
    accelerator: str = "auto",
    devices: int = 1,
    monitor_metric: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 1,
    checkpoint_path: Optional[str] = None,
    input_bands: Optional[List[int]] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> PixelRegressionModel:
    """
    Train a pixel-level regression model.

    Args:
        train_image_paths: List of training image paths.
        train_target_paths: List of training target paths.
        val_image_paths: List of validation image paths.
        val_target_paths: List of validation target paths.
        encoder_name: Name of timm encoder.
        architecture: Segmentation architecture ('unet', 'unetplusplus', 'deeplabv3plus', etc.).
        in_channels: Number of input channels.
        encoder_weights: Pretrained weights for encoder.
        output_dir: Directory to save outputs.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        num_workers: Number of data loading workers.
        freeze_encoder: Freeze encoder during training.
        loss_type: Loss function type ('mse', 'l1', 'huber').
        normalize_input: Normalize input tiles to expected range.
        accelerator: Accelerator type.
        devices: Number of devices.
        monitor_metric: Metric to monitor for checkpointing.
        mode: 'min' or 'max' for monitor_metric.
        patience: Early stopping patience.
        save_top_k: Number of best models to save.
        checkpoint_path: Path to checkpoint to resume from.
        input_bands: Band indices to use (1-indexed).
        verbose: Whether to show detailed training logs, progress bars,
            and callback messages. Defaults to True.
        **kwargs: Additional arguments for Trainer.

    Returns:
        PixelRegressionModel: Trained model.
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "PyTorch Lightning is required. Install with: pip install lightning"
        )

    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Create model
    model = PixelRegressionModel(
        encoder_name=encoder_name,
        architecture=architecture,
        in_channels=in_channels,
        encoder_weights=encoder_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_encoder=freeze_encoder,
        loss_type=loss_type,
    )

    preprocessing = _infer_preprocessing_params(encoder_name, encoder_weights)
    image_mean = None
    image_std = None
    if preprocessing is not None:
        pp_mean = preprocessing.get("mean")
        pp_std = preprocessing.get("std")
        # Only apply encoder preprocessing when channel count matches
        # (e.g. 3-band RGB with ImageNet weights).  For multi-spectral
        # inputs the ImageNet statistics are inappropriate; the
        # normalize_input flag already scales values to [0, 1].
        if pp_mean is not None and pp_std is not None and len(pp_mean) == in_channels:
            image_mean = pp_mean
            image_std = pp_std

    # Create datasets
    train_dataset = PixelRegressionDataset(
        train_image_paths,
        train_target_paths,
        input_bands=input_bands,
        normalize_input=normalize_input,
        image_mean=image_mean,
        image_std=image_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_image_paths is not None and val_target_paths is not None:
        val_dataset = PixelRegressionDataset(
            val_image_paths,
            val_target_paths,
            input_bands=input_bands,
            normalize_input=normalize_input,
            image_mean=image_mean,
            image_std=image_std,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{encoder_name}_{architecture}_{{epoch:02d}}_{{val_loss:.4f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=verbose,
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
        verbose=verbose,
    )
    callbacks.append(early_stop_callback)

    if not verbose:
        import logging

        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        callbacks.append(_CompactProgressBar())
        model._prog_bar_metrics = False

    logger = CSVLogger(model_dir, name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        enable_model_summary=verbose,
        **kwargs,
    )

    if verbose:
        print(
            f"Training {architecture} with {encoder_name} encoder"
            f" for {num_epochs} epochs..."
        )
        print(f"Loss function: {loss_type.upper()}")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        if verbose:
            print(f"\nBest model saved at: {best_model_path}")
        model = PixelRegressionModel.load_from_checkpoint(best_model_path)
        model.best_model_path = best_model_path
    else:
        if verbose:
            print("\nBest model path not found; returning last epoch model.")

    return model


def predict_raster(
    model: Union[PixelRegressionModel, nn.Module],
    input_raster: str,
    output_raster: str,
    tile_size: int = 256,
    overlap: int = 64,
    input_bands: Optional[List[int]] = None,
    batch_size: int = 4,
    device: Optional[str] = None,
    output_nodata: float = -9999.0,
    clip_range: Optional[Tuple[float, float]] = None,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    use_model_preprocessing: bool = True,
) -> str:
    """
    Run pixel-level inference on a raster.

    Uses sliding window with overlap and blending for smooth predictions.
    Output dimensions match input dimensions exactly.

    Args:
        model: Trained pixel regression model.
        input_raster: Path to input raster.
        output_raster: Path to save output raster.
        tile_size: Size of tiles for inference.
        overlap: Overlap between tiles for blending.
        input_bands: Band indices to use (1-indexed).
        batch_size: Batch size for inference.
        device: Device to use.
        output_nodata: NoData value for output.
        clip_range: Optional tuple (min, max) to clip output values.
        image_mean: Optional per-channel mean for normalization.
        image_std: Optional per-channel std for normalization.
        use_model_preprocessing: Use encoder preprocessing params if available.

    Returns:
        str: Path to output raster.
    """
    import rasterio

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model = model.to(device)

    stride = tile_size - overlap

    with rasterio.open(input_raster) as src:
        height = src.height
        width = src.width

        if input_bands is None:
            input_bands = list(range(1, src.count + 1))

        print(f"Input raster: {width}x{height}")
        print(f"Tile size: {tile_size}, overlap: {overlap}, stride: {stride}")

        # Initialize output arrays
        output_sum = np.zeros((height, width), dtype=np.float64)
        weight_sum = np.zeros((height, width), dtype=np.float64)

        # Create weight mask for blending (higher weight in center)
        weight_mask = _create_weight_mask(tile_size, overlap)

        # Read full input for nodata mask
        full_input = src.read(input_bands)
        nodata_mask = np.any(np.isnan(full_input), axis=0)

        if use_model_preprocessing and image_mean is None and image_std is None:
            encoder_name = getattr(
                getattr(model, "hparams", None), "encoder_name", None
            )
            encoder_weights = getattr(
                getattr(model, "hparams", None), "encoder_weights", None
            )
            model_in_channels = getattr(
                getattr(model, "hparams", None),
                "in_channels",
                len(input_bands),
            )
            if encoder_name and encoder_weights:
                preprocessing = _infer_preprocessing_params(
                    encoder_name, encoder_weights
                )
                if preprocessing is not None:
                    pp_mean = preprocessing.get("mean")
                    pp_std = preprocessing.get("std")
                    if (
                        pp_mean is not None
                        and pp_std is not None
                        and len(pp_mean) == model_in_channels
                    ):
                        image_mean = pp_mean
                        image_std = pp_std

        # Collect tiles
        tiles = []
        positions = []

        for row in range(0, height, stride):
            for col in range(0, width, stride):
                # Calculate tile bounds
                row_end = min(row + tile_size, height)
                col_end = min(col + tile_size, width)
                row_start = row_end - tile_size
                col_start = col_end - tile_size

                # Clamp to valid range
                row_start = max(0, row_start)
                col_start = max(0, col_start)

                tiles.append((row_start, col_start, row_end, col_end))
                positions.append((row_start, col_start))

        print(f"Total tiles: {len(tiles)}")

        # Process in batches
        for batch_start in tqdm(
            range(0, len(tiles), batch_size), desc="Running inference"
        ):
            batch_end = min(batch_start + batch_size, len(tiles))
            batch_tiles = tiles[batch_start:batch_end]

            # Load batch
            batch_images = []
            for row_start, col_start, row_end, col_end in batch_tiles:
                window = rasterio.windows.Window(
                    col_start, row_start, col_end - col_start, row_end - row_start
                )
                tile = src.read(input_bands, window=window).astype(np.float32)

                # Handle non-square tiles at edges
                if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                    padded = np.zeros(
                        (len(input_bands), tile_size, tile_size), dtype=np.float32
                    )
                    padded[:, : tile.shape[1], : tile.shape[2]] = tile
                    tile = padded

                # Normalize
                tile = np.nan_to_num(tile, nan=0.0)
                data_max = np.abs(tile).max()
                if data_max <= 1.5:
                    tile = np.clip(tile, 0, 1)
                else:
                    tile = np.clip(tile, 0, 10000) / 10000.0

                if image_mean is not None and image_std is not None:
                    mean, std = _prepare_normalization_stats(
                        image_mean, image_std, tile.shape[0]
                    )
                    tile = (tile - mean) / std

                batch_images.append(tile)

            batch_tensor = torch.from_numpy(np.stack(batch_images)).to(device)

            # Inference
            with torch.no_grad():
                preds = model(batch_tensor).cpu().numpy()

            # Apply predictions with blending
            for i, (row_start, col_start, row_end, col_end) in enumerate(batch_tiles):
                pred = preds[i]
                h = row_end - row_start
                w = col_end - col_start

                # Get the relevant portion of prediction and weight
                pred_crop = pred[:h, :w]
                weight_crop = weight_mask[:h, :w]

                # Accumulate
                output_sum[row_start:row_end, col_start:col_end] += (
                    pred_crop * weight_crop
                )
                weight_sum[row_start:row_end, col_start:col_end] += weight_crop

        # Normalize by weights
        valid_weights = weight_sum > 0
        output_array = np.full((height, width), output_nodata, dtype=np.float32)
        output_array[valid_weights] = (
            output_sum[valid_weights] / weight_sum[valid_weights]
        )

        # Apply nodata mask
        output_array[nodata_mask] = output_nodata

        # Clip output to valid range if specified
        if clip_range is not None:
            valid_data_mask = ~nodata_mask & valid_weights
            output_array[valid_data_mask] = np.clip(
                output_array[valid_data_mask], clip_range[0], clip_range[1]
            )

        # Save output
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=np.float32,
            nodata=output_nodata,
        )

        output_dir = os.path.dirname(os.path.abspath(output_raster))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(output_array, 1)

    valid_data = output_array[~nodata_mask & valid_weights]
    print(f"\nOutput saved to: {output_raster}")
    print(f"Output dimensions: {width}x{height} (same as input)")
    if len(valid_data) > 0:
        print(f"Prediction range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")

    return output_raster


def _create_weight_mask(tile_size: int, overlap: int) -> np.ndarray:
    """Create a weight mask for blending overlapping tiles."""
    if overlap == 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    # Create 1D ramp
    ramp = np.ones(tile_size, dtype=np.float32)
    ramp[:overlap] = np.linspace(0, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0, overlap)

    # Create 2D weight mask
    weight_mask = np.outer(ramp, ramp)
    return weight_mask


# ============================================================================
# Evaluation and Visualization Functions
# ============================================================================


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    print_results: bool = True,
) -> Dict[str, float]:
    """
    Evaluate regression predictions with multiple metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        mask: Optional mask of valid pixels.
        print_results: Whether to print results.

    Returns:
        Dictionary of metrics: MSE, RMSE, MAE, R².
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if mask is not None:
        mask = np.array(mask).flatten()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    if print_results:
        print("=" * 50)
        print("Regression Evaluation Metrics")
        print("=" * 50)
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.4f}")
        print("=" * 50)

    return metrics


def plot_regression_comparison(
    true_raster: str,
    pred_raster: str,
    title: str = "Regression Results",
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    valid_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
):
    """
    Plot comparison of ground truth, prediction, and difference.

    Args:
        true_raster: Path to ground truth raster.
        pred_raster: Path to prediction raster.
        title: Title for the plot.
        cmap: Colormap for visualization.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        valid_range: Tuple of (min, max) valid values for filtering outliers.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Tuple of (figure, metrics_dict).
    """
    import matplotlib.pyplot as plt
    import rasterio

    with rasterio.open(true_raster) as src:
        true_data = src.read(1)
        true_nodata = src.nodata

    with rasterio.open(pred_raster) as src:
        pred_data = src.read(1)
        pred_nodata = src.nodata

    # Create valid mask
    valid_mask = np.ones_like(true_data, dtype=bool)
    if true_nodata is not None:
        valid_mask &= true_data != true_nodata
    if pred_nodata is not None:
        valid_mask &= pred_data != pred_nodata
    valid_mask &= ~np.isnan(true_data) & ~np.isnan(pred_data)

    # Filter by valid range (important for NDVI which should be [-1, 1])
    if valid_range is not None:
        valid_mask &= (true_data >= valid_range[0]) & (true_data <= valid_range[1])
        valid_mask &= (pred_data >= valid_range[0]) & (pred_data <= valid_range[1])

    # Calculate metrics
    metrics = evaluate_regression(
        true_data[valid_mask], pred_data[valid_mask], print_results=False
    )

    # Auto-determine vmin/vmax if not specified
    if vmin is None:
        vmin = np.percentile(true_data[valid_mask], 2)
    if vmax is None:
        vmax = np.percentile(true_data[valid_mask], 98)

    # Create masked arrays for display
    true_masked = np.ma.masked_where(~valid_mask, true_data)
    pred_masked = np.ma.masked_where(~valid_mask, pred_data)
    diff = pred_data - true_data
    diff_masked = np.ma.masked_where(~valid_mask, diff)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im1 = axes[0].imshow(true_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(pred_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Prediction (R²={metrics['r2']:.4f})", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    diff_range = max(
        abs(np.percentile(diff[valid_mask], 5)),
        abs(np.percentile(diff[valid_mask], 95)),
    )
    im3 = axes[2].imshow(diff_masked, cmap="RdBu_r", vmin=-diff_range, vmax=diff_range)
    axes[2].set_title(f"Difference (RMSE={metrics['rmse']:.4f})", fontsize=14)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, metrics


def plot_scatter(
    true_raster: str,
    pred_raster: str,
    sample_size: int = 10000,
    title: str = "Predicted vs Actual",
    valid_range: Optional[Tuple[float, float]] = None,
    fit_line: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot scatter plot of predicted vs actual values with optional trend line.

    Args:
        true_raster: Path to ground truth raster.
        pred_raster: Path to prediction raster.
        sample_size: Number of points to plot (sampled if needed).
        title: Title for the plot.
        valid_range: Tuple of (min, max) valid values for filtering outliers.
        fit_line: Whether to show a linear regression trend line.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Tuple of (figure, metrics_dict).
    """
    import matplotlib.pyplot as plt
    import rasterio
    from sklearn.metrics import r2_score

    with rasterio.open(true_raster) as src:
        true_data = src.read(1)
        true_nodata = src.nodata

    with rasterio.open(pred_raster) as src:
        pred_data = src.read(1)
        pred_nodata = src.nodata

    # Create valid mask
    valid_mask = np.ones_like(true_data, dtype=bool)
    if true_nodata is not None:
        valid_mask &= true_data != true_nodata
    if pred_nodata is not None:
        valid_mask &= pred_data != pred_nodata
    valid_mask &= ~np.isnan(true_data) & ~np.isnan(pred_data)

    # Filter by valid range
    if valid_range is not None:
        valid_mask &= (true_data >= valid_range[0]) & (true_data <= valid_range[1])
        valid_mask &= (pred_data >= valid_range[0]) & (pred_data <= valid_range[1])

    y_true = true_data[valid_mask]
    y_pred = pred_data[valid_mask]

    # Sample if too many points
    if len(y_true) > sample_size:
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    # Calculate metrics on full data
    metrics = evaluate_regression(y_true, y_pred, print_results=False)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=5, edgecolors="none")

    # Add 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 Line")

    # Add linear regression trend line
    if fit_line:
        coeffs = np.polyfit(y_true, y_pred, 1)
        slope, intercept = coeffs
        fit_x = np.array([min_val, max_val])
        fit_y = slope * fit_x + intercept
        ax.plot(
            fit_x,
            fit_y,
            "b-",
            lw=2,
            label=f"Fit: y = {slope:.3f}x + {intercept:.3f}",
        )
        metrics["slope"] = float(slope)
        metrics["intercept"] = float(intercept)

    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(
        f"{title}\nR² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, metrics


def plot_training_history(
    log_dir: str,
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    tail: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """
    Plot training history curves from Lightning CSV logs.

    Reads the ``metrics.csv`` file produced by :class:`CSVLogger` and plots
    the requested training and validation metrics over epochs.

    Args:
        log_dir: Path to the model output directory (the same ``output_dir``
            passed to :func:`train_pixel_regressor`).  The function searches
            for ``lightning_logs/version_*/metrics.csv`` inside a ``models``
            sub-directory (or directly under *log_dir*).
        metrics: List of metric names to plot.  Each name is matched against
            the CSV columns; both the ``train_`` and ``val_`` variants are
            plotted when available.  Defaults to ``["loss", "r2"]``.
        figsize: Figure size as ``(width, height)``.  Defaults to
            ``(6 * n_metrics, 5)``.
        tail: If given, only plot the last *tail* epochs.  Useful for
            skipping early warm-up instability. By default the function
            automatically skips early epochs when extreme outliers would
            compress the y-axis (more than 10× the stable range).
        save_path: If given, save the figure to this path.

    Returns:
        Tuple of (figure, pandas.DataFrame of the loaded metrics).
    """
    import glob

    import matplotlib.pyplot as plt

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for plot_training_history")

    if metrics is None:
        metrics = ["loss", "r2"]

    # Locate metrics.csv
    search_paths = [
        os.path.join(log_dir, "models", "lightning_logs", "version_*", "metrics.csv"),
        os.path.join(log_dir, "lightning_logs", "version_*", "metrics.csv"),
        os.path.join(log_dir, "version_*", "metrics.csv"),
    ]

    csv_path = None
    for pattern in search_paths:
        matches = sorted(glob.glob(pattern))
        if matches:
            csv_path = matches[-1]  # latest version
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"No metrics.csv found under '{log_dir}'. "
            "Looked for lightning_logs/version_*/metrics.csv"
        )

    df = pd.read_csv(csv_path)
    _n_epochs = df["epoch"].nunique() if "epoch" in df.columns else len(df)
    print(f"Reading logs: {csv_path} ({_n_epochs} epochs)")

    # Group rows by epoch – Lightning logs multiple rows per epoch (one per
    # step plus validation).  Use ``last()`` with ``skipna`` so we keep the
    # last non-null value for every column within each epoch.
    if "epoch" in df.columns:
        try:
            df_epoch = df.groupby("epoch").last(skipna=True).reset_index()
        except TypeError:
            # older pandas without skipna
            df_epoch = df.groupby("epoch").last().reset_index()
    else:
        df_epoch = df

    # Apply tail filter
    if tail is not None:
        df_epoch = df_epoch.tail(tail).reset_index(drop=True)

    n_metrics = len(metrics)
    if figsize is None:
        figsize = (6 * n_metrics, 5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_col = (
            f"train_{metric}_epoch"
            if f"train_{metric}_epoch" in df_epoch.columns
            else f"train_{metric}"
        )
        val_col = f"val_{metric}"

        has_train = train_col in df_epoch.columns
        has_val = val_col in df_epoch.columns

        if not has_train and not has_val:
            ax.set_title(f"{metric} (no data)")
            continue

        x = df_epoch["epoch"] if "epoch" in df_epoch.columns else df_epoch.index

        if has_train:
            train_data = df_epoch[train_col].dropna()
            ax.plot(
                x[train_data.index],
                train_data.values,
                label=f"Train {metric}",
                linewidth=2,
            )
        if has_val:
            val_data = df_epoch[val_col].dropna()
            ax.plot(
                x[val_data.index],
                val_data.values,
                label=f"Val {metric}",
                linewidth=2,
            )

        # Auto-zoom: if early outliers compress the view, clip y-axis to
        # the range of the stable second half of training.
        if tail is None:
            n_epochs = len(df_epoch)
            if n_epochs >= 10:
                half = n_epochs // 2
                second_half_vals = []
                all_vals = []
                if has_train:
                    col_data = df_epoch[train_col].dropna()
                    all_vals.extend(col_data.values)
                    second_half_vals.extend(
                        df_epoch[train_col].iloc[half:].dropna().values
                    )
                if has_val:
                    col_data = df_epoch[val_col].dropna()
                    all_vals.extend(col_data.values)
                    second_half_vals.extend(
                        df_epoch[val_col].iloc[half:].dropna().values
                    )
                if second_half_vals and all_vals:
                    sh_arr = np.array(second_half_vals)
                    all_arr = np.array(all_vals)
                    sh_min, sh_max = sh_arr.min(), sh_arr.max()
                    sh_range = sh_max - sh_min if sh_max != sh_min else 1.0
                    full_range = all_arr.max() - all_arr.min()
                    if full_range == 0:
                        full_range = 1.0
                    # If full range is >5× the stable range, zoom in
                    if full_range > 5 * sh_range:
                        margin = sh_range * 0.3
                        ax.set_ylim(sh_min - margin, sh_max + margin)

        label = metric.upper() if len(metric) <= 4 else metric.replace("_", " ").title()
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"Training & Validation {label}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, df_epoch


def visualize_prediction(
    input_raster: str,
    pred_raster: str,
    rgb_bands: List[int] = [1, 2, 3],
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
):
    """
    Visualize input RGB and prediction side by side.

    Args:
        input_raster: Path to input raster.
        pred_raster: Path to prediction raster.
        rgb_bands: Band indices for RGB display (1-indexed).
        cmap: Colormap for prediction.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Figure object.
    """
    import matplotlib.pyplot as plt
    import rasterio

    with rasterio.open(input_raster) as src:
        rgb = src.read(rgb_bands).astype(np.float64)
        # Per-band 2–98 percentile stretch for proper RGB display
        for i in range(rgb.shape[0]):
            band = rgb[i]
            valid = band[
                np.isfinite(band) & (band != src.nodata if src.nodata else True)
            ]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    rgb[i] = (band - p2) / (p98 - p2)
                else:
                    rgb[i] = band / p98 if p98 > 0 else band
        rgb = np.clip(rgb, 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))

    with rasterio.open(pred_raster) as src:
        pred = src.read(1)
        pred_nodata = src.nodata

    # Mask
    valid_mask = np.ones_like(pred, dtype=bool)
    if pred_nodata is not None:
        valid_mask &= pred != pred_nodata
    valid_mask &= ~np.isnan(pred)
    pred_masked = np.ma.masked_where(~valid_mask, pred)

    if vmin is None:
        vmin = np.percentile(pred[valid_mask], 2)
    if vmax is None:
        vmax = np.percentile(pred[valid_mask], 98)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB", fontsize=14)
    axes[0].axis("off")

    im = axes[1].imshow(pred_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ============================================================================
# Backward compatibility aliases
# ============================================================================

# Aliases for backward compatibility
TimmRegressor = PixelRegressionModel
RegressionDataset = PixelRegressionDataset
create_regression_patches = create_regression_tiles
train_timm_regressor = train_pixel_regressor


def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Results",
    fit_line: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Plot regression results: scatter plot with trend line and residual plot.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        title: Title for the plots.
        fit_line: Whether to show a linear regression trend line.
        figsize: Figure size.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=20)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 Line")

    # Add linear regression trend line
    if fit_line:
        coeffs = np.polyfit(y_true, y_pred, 1)
        slope, intercept = coeffs
        fit_x = np.array([min_val, max_val])
        fit_y = slope * fit_x + intercept
        ax1.plot(
            fit_x,
            fit_y,
            "b-",
            lw=2,
            label=f"Fit: y = {slope:.3f}x + {intercept:.3f}",
        )

    r2 = r2_score(y_true, y_pred)
    ax1.set_xlabel("Actual Values", fontsize=12)
    ax1.set_ylabel("Predicted Values", fontsize=12)
    ax1.set_title(f"Predicted vs Actual (R² = {r2:.4f})", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residual plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors="none", s=20)
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)

    ax2.set_xlabel("Predicted Values", fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.set_title("Residual Plot", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig


def predict_with_timm_regressor(*args, **kwargs):
    """Deprecated: Use predict_raster instead."""
    raise NotImplementedError(
        "predict_with_timm_regressor is deprecated. "
        "Use predict_raster for pixel-level predictions."
    )


def get_timm_regression_model(*args, **kwargs):
    """Deprecated: Use PixelRegressionModel instead."""
    raise NotImplementedError(
        "get_timm_regression_model is deprecated. "
        "Use PixelRegressionModel for pixel-level regression."
    )
