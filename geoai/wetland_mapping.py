"""
Wetland Mapping with Foundation Models
=====================================

This module provides wetland mapping functionality using Prithvi foundation models,
NAIP imagery, and National Wetlands Inventory (NWI) data.

Key Components:
- Data access: NAIP (Planetary Computer) + NWI (FWS Wetlands Mapper)
- Foundation model: Prithvi-EO-2.0 (IBM/NASA)
- Training pipeline: TIMM segmentation with PyTorch Lightning
- Inference: Large image processing with sliding windows

Author: Built on opengeos/geoai infrastructure
"""

import os
import logging
from typing import List, Optional, Tuple, Dict, Union, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import existing GeoAI functionality
from .download import download_naip
from .prithvi import PrithviProcessor, load_prithvi_model
from .timm_segment import TimmSegmentationModel, SegmentationDataset

# Try importing optional dependencies
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available. Training functionality limited.")

try:
    import leafmap

    LEAFMAP_AVAILABLE = True
except ImportError:
    LEAFMAP_AVAILABLE = False
    print("Warning: leafmap not available. NWI data access limited.")

try:
    import satlaspretrain_models

    SATLAS_AVAILABLE = True
except ImportError:
    SATLAS_AVAILABLE = False
    print(
        "Warning: satlaspretrain_models not available. Satlas backbone not supported."
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Wetland classification mapping
WETLAND_CLASSES = {
    "background": 0,
    "freshwater_emergent": 1,
    "freshwater_forested": 2,
    "freshwater_pond": 3,
    "estuarine": 4,
    "other_wetland": 5,
}


# ---------------------------------------------------------------------------
# Loss functions for imbalanced segmentation
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) — down-weights easy examples so the
    model focuses on hard, misclassified pixels (e.g. rare wetland classes)."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("alpha", alpha)  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal


class DiceLoss(nn.Module):
    """Soft Dice Loss — directly optimises overlap, which is more
    appropriate for segmentation than pixel-wise cross-entropy alone."""

    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        probs = torch.softmax(inputs, dim=1)
        targets_oh = (
            nn.functional.one_hot(targets.clamp(0, num_classes - 1), num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_oh = targets_oh * mask

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Focal + Dice — best-practice combo for imbalanced segmentation."""

    def __init__(self, class_weights=None, focal_gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.dice_weight * self.dice(
            inputs, targets
        )


class WetlandDatasetBuilder:
    """Build wetland training datasets from NAIP imagery and NWI polygons."""

    def __init__(self, cache_dir: str = "wetland_data_cache"):
        """Initialize the dataset builder.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.wetland_classes = WETLAND_CLASSES

    def download_naip_for_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: Optional[int] = None,
        max_items: int = 50,
    ) -> List[str]:
        """Download NAIP imagery for a region using existing GeoAI function.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            year: NAIP year (e.g., 2020). If None, gets latest available
            max_items: Maximum number of tiles to download

        Returns:
            List of downloaded NAIP file paths
        """
        output_dir = self.cache_dir / "naip"

        logger.info(f"Downloading NAIP imagery for bbox {bbox}, year {year}")

        # Use existing GeoAI download function
        naip_files = download_naip(
            bbox=bbox,
            output_dir=str(output_dir),
            year=year,
            max_items=max_items,
            overwrite=False,
        )

        logger.info(f"Downloaded {len(naip_files)} NAIP files")
        return naip_files

    def get_nwi_data_for_region(
        self, bbox: Tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """Get NWI wetland data for a region using leafmap.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns:
            GeoDataFrame with wetland polygons and classifications
        """
        if not LEAFMAP_AVAILABLE:
            raise ImportError(
                "leafmap required for NWI data access. Install with: pip install leafmap"
            )

        logger.info(f"Fetching NWI data for bbox {bbox}")

        # Convert bbox to geodataframe
        bbox_geometry = leafmap.bbox_to_gdf(bbox)

        # Get NWI data using leafmap
        nwi_gdf = leafmap.get_nwi(bbox_geometry)

        if nwi_gdf is None or len(nwi_gdf) == 0:
            logger.warning("No NWI data found for this region")
            return gpd.GeoDataFrame()

        # Simplify wetland types for foundation model training
        nwi_gdf["simplified_class"] = nwi_gdf["WETLAND_TY"].apply(
            self._simplify_wetland_type
        )

        logger.info(f"Found {len(nwi_gdf)} wetland features")
        return nwi_gdf

    def _simplify_wetland_type(self, wetland_code: str) -> str:
        """Simplify NWI wetland codes to main categories."""
        if pd.isna(wetland_code):
            return "other_wetland"

        code = str(wetland_code).upper()

        # Map common NWI codes to simplified classes
        if any(x in code for x in ["PEM", "PAB"]):  # Palustrine emergent/aquatic bed
            return "freshwater_emergent"
        elif any(x in code for x in ["PSS", "PFO"]):  # Palustrine scrub-shrub/forested
            return "freshwater_forested"
        elif (
            "PUB" in code or "POW" in code
        ):  # Palustrine unconsolidated bottom/open water
            return "freshwater_pond"
        elif any(x in code for x in ["E1", "E2", "M1", "M2"]):  # Estuarine/marine
            return "estuarine"
        else:
            return "other_wetland"

    def create_training_tiles(
        self,
        naip_files: List[str],
        nwi_gdf: gpd.GeoDataFrame,
        output_dir: str,
        tile_size: int = 512,
        stride: int = 256,
        min_wetland_pixels: int = 100,
    ) -> Dict[str, int]:
        """Create training tiles from NAIP imagery and NWI masks.

        Args:
            naip_files: List of NAIP GeoTIFF file paths
            nwi_gdf: NWI wetland polygons
            output_dir: Output directory for tiles
            tile_size: Size of output tiles (pixels)
            stride: Stride between tiles (pixels)
            min_wetland_pixels: Minimum wetland pixels per tile to include

        Returns:
            Dictionary with tile creation statistics
        """
        from rasterio.features import rasterize

        output_path = Path(output_dir)
        images_dir = output_path / "images"
        masks_dir = output_path / "masks"

        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        tile_count = 0
        wetland_tile_count = 0
        stats = {"total_tiles": 0, "wetland_tiles": 0, "files_processed": 0}

        for naip_file in naip_files:
            logger.info(f"Processing {naip_file}")

            try:
                # Read NAIP imagery
                with rasterio.open(naip_file) as src:
                    bounds = src.bounds
                    crs = src.crs
                    transform = src.transform
                    width, height = src.width, src.height

                    # Create NAIP bounds geometry for proper intersection
                    from shapely.geometry import box

                    naip_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                    naip_gdf = gpd.GeoDataFrame([1], geometry=[naip_box], crs=crs)

                    # Convert NWI to same CRS as NAIP first
                    if nwi_gdf.crs != crs:
                        logger.info(f"Converting NWI from {nwi_gdf.crs} to {crs}")
                        nwi_proj = nwi_gdf.to_crs(crs)
                    else:
                        nwi_proj = nwi_gdf

                    # Find wetlands that intersect with NAIP bounds using geometric intersection
                    nwi_clip = gpd.overlay(nwi_proj, naip_gdf, how="intersection")

                    if len(nwi_clip) == 0:
                        logger.info(f"No wetlands found in {naip_file}")
                        continue

                    # Create shapes list with class values
                    shapes = [
                        (geom, self.wetland_classes[cls])
                        for geom, cls in zip(
                            nwi_clip.geometry, nwi_clip["simplified_class"]
                        )
                    ]

                    # Rasterize wetland polygons
                    wetland_mask = rasterize(
                        shapes=shapes,
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                    )

                    file_tile_count = 0
                    file_wetland_count = 0

                    # Generate tiles
                    y_range = range(0, height - tile_size + 1, stride)
                    x_range = range(0, width - tile_size + 1, stride)
                    logger.info(f"Generating {len(x_range) * len(y_range)} tiles...")

                    for y in y_range:
                        for x in x_range:
                            try:
                                # Read image tile
                                window = Window(x, y, tile_size, tile_size)
                                image_tile = src.read(
                                    window=window
                                )  # Shape: (bands, height, width)

                                # Extract corresponding mask tile
                                mask_tile = wetland_mask[
                                    y : y + tile_size, x : x + tile_size
                                ]

                                # Check if tile has enough wetland pixels
                                wetland_pixels = np.sum(mask_tile > 0)

                                # Save tile if it meets criteria or random background sample
                                if (
                                    wetland_pixels >= min_wetland_pixels
                                    or np.random.random() < 0.1
                                ):
                                    tile_id = f"{Path(naip_file).stem}_{y}_{x}"

                                    # Get tile transform
                                    tile_transform = rasterio.windows.transform(
                                        window, transform
                                    )

                                    # Create proper profiles for image and mask
                                    image_profile = {
                                        "driver": "GTiff",
                                        "height": tile_size,
                                        "width": tile_size,
                                        "count": image_tile.shape[0],
                                        "dtype": image_tile.dtype,
                                        "crs": crs,
                                        "transform": tile_transform,
                                        "compress": "lzw",
                                    }

                                    mask_profile = {
                                        "driver": "GTiff",
                                        "height": tile_size,
                                        "width": tile_size,
                                        "count": 1,
                                        "dtype": "uint8",
                                        "crs": crs,
                                        "transform": tile_transform,
                                        "compress": "lzw",
                                    }

                                    # Save image tile
                                    image_path = images_dir / f"{tile_id}.tif"
                                    with rasterio.open(
                                        image_path, "w", **image_profile
                                    ) as dst:
                                        dst.write(image_tile)

                                    # Save mask tile
                                    mask_path = masks_dir / f"{tile_id}.tif"
                                    with rasterio.open(
                                        mask_path, "w", **mask_profile
                                    ) as dst:
                                        dst.write(mask_tile, 1)

                                    if wetland_pixels >= min_wetland_pixels:
                                        file_wetland_count += 1

                                    file_tile_count += 1

                                    # Log progress every 100 tiles
                                    if file_tile_count % 100 == 0:
                                        logger.info(
                                            f"Created {file_tile_count} tiles ({file_wetland_count} wetland)"
                                        )

                            except Exception as tile_error:
                                logger.error(
                                    f"Error creating tile at ({x}, {y}): {tile_error}"
                                )
                                continue

                    tile_count += file_tile_count
                    wetland_tile_count += file_wetland_count
                    stats["files_processed"] += 1

                    logger.info(
                        f"  Created {file_wetland_count} wetland tiles from {file_tile_count} total tiles"
                    )

            except Exception as e:
                logger.error(f"Error processing {naip_file}: {e}")
                continue

        stats["total_tiles"] = tile_count
        stats["wetland_tiles"] = wetland_tile_count

        logger.info(
            f"Dataset creation complete: {wetland_tile_count} wetland tiles from {tile_count} total tiles"
        )
        return stats


class WetlandPrithviModel(pl.LightningModule if LIGHTNING_AVAILABLE else nn.Module):
    """Wetland segmentation model using Prithvi foundation model backbone."""

    def __init__(
        self,
        prithvi_model_name: str = "Prithvi-EO-2.0-300M-TL",
        num_wetland_classes: int = 6,
        freeze_backbone_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """Initialize wetland Prithvi model.

        Args:
            prithvi_model_name: Name of Prithvi model variant
            num_wetland_classes: Number of wetland classes (including background)
            freeze_backbone_epochs: Epochs to freeze Prithvi backbone
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()

        if LIGHTNING_AVAILABLE:
            self.save_hyperparameters()

        self.num_classes = num_wetland_classes
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prithvi_model_name = prithvi_model_name

        # Load Prithvi foundation model
        self.prithvi_processor = load_prithvi_model(
            model_name=prithvi_model_name,
            device="cpu",  # Will be moved to GPU during training
        )

        # Get Prithvi encoder features
        self.backbone = self.prithvi_processor.model.encoder

        # Add channel adaptation for NAIP (4 channels) to Prithvi (6 channels)
        # Prithvi expects 6 bands from HLS (Landsat/Sentinel-2)
        # NAIP has 4 bands (RGB + NIR)
        self.channel_adapter = nn.Conv2d(4, 6, kernel_size=1, bias=False)
        # Initialize to pass through first 4 channels and zero-pad last 2
        with torch.no_grad():
            self.channel_adapter.weight[:4, :, 0, 0] = torch.eye(4)
            self.channel_adapter.weight[4:, :, 0, 0] = 0

        # Add segmentation head
        self.decoder = self._create_decoder()

        # Loss function — Focal-only (no Dice).  Dice loss amplifies the
        # minority-class signal on top of Focal, causing the model to
        # oscillate between all-background and all-wetland.  Focal loss
        # alone with moderate inverse-frequency weights provides a stable
        # gradient signal.
        class_weights = torch.tensor([0.5, 3.0, 3.0, 4.0, 3.5, 3.0])
        self.criterion = CombinedSegmentationLoss(
            class_weights=class_weights,
            focal_gamma=2.0,
            dice_weight=0.0,
        )

        # Initially freeze backbone
        self.freeze_backbone()

    def _create_decoder(self) -> nn.Module:
        """Create segmentation decoder."""
        embed_dim = self.prithvi_processor.config["embed_dim"]

        decoder = nn.Sequential(
            # Upsample from patch embeddings to pixel predictions
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final prediction layer
            nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1),
        )

        return decoder

    def freeze_backbone(self):
        """Freeze Prithvi backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze Prithvi backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass through model."""
        batch_size = x.shape[0]
        original_height, original_width = x.shape[-2:]

        # Normalize input to [0, 1] range (NAIP uint8 values are 0-255)
        if x.max() > 1.0:
            x = x / 255.0

        # Resize input to match Prithvi's expected spatial size (224x224)
        prithvi_size = self.prithvi_processor.config["img_size"]  # 224
        if x.shape[-1] != prithvi_size or x.shape[-2] != prithvi_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(prithvi_size, prithvi_size),
                mode="bilinear",
                align_corners=False,
            )

        # Adapt NAIP channels (4) to Prithvi expected channels (6)
        if x.shape[1] == 4:  # NAIP format
            x = self.channel_adapter(x)  # 4 channels -> 6 channels

        # Adapt to Prithvi temporal format (B,C,T,H,W)
        num_frames = self.prithvi_processor.config.get("num_frames", 1)
        if x.dim() == 4:  # (B, C, H, W)
            x = x.unsqueeze(2)  # Add temporal dimension: (B, C, 1, H, W)
            x = x.repeat(1, 1, num_frames, 1, 1)  # Repeat to match expected frames

        # Extract features using Prithvi encoder.
        # When backbone is frozen its params already have requires_grad=False,
        # so autograd naturally skips them — no need for a no_grad() context
        # (which would also block gradients to the channel_adapter).
        features, _, _ = self.backbone(x, mask_ratio=0.0)

        # Remove CLS token
        patch_features = features[:, 1:, :]  # (B, T*H_p*W_p, D)
        embed_dim = patch_features.shape[-1]

        # Compute spatial grid size from config (not from sqrt of total patches)
        config_patch_size = self.prithvi_processor.config["patch_size"]
        if isinstance(config_patch_size, (list, tuple)):
            spatial_patch_size = config_patch_size[-1]
        else:
            spatial_patch_size = config_patch_size
        grid_h = prithvi_size // spatial_patch_size  # e.g. 224 // 16 = 14
        grid_w = grid_h

        # Reshape separating temporal and spatial: (B, T*H_p*W_p, D) → (B, T, H_p, W_p, D)
        patch_features = patch_features.reshape(
            batch_size, num_frames, grid_h, grid_w, embed_dim
        )
        # Average across temporal dimension to get spatial-only features
        spatial_features = patch_features.mean(dim=1)  # (B, H_p, W_p, D)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, D, H_p, W_p)

        # Decode to segmentation map
        output = self.decoder(spatial_features)

        # Resize output to match original input size
        if output.shape[-1] != original_width or output.shape[-2] != original_height:
            output = torch.nn.functional.interpolate(
                output,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

        return output

    def training_step(self, batch, batch_idx):
        """Training step with class-distribution diagnostics."""
        if not LIGHTNING_AVAILABLE:
            raise RuntimeError("PyTorch Lightning required for training")

        images, masks = batch
        predictions = self(images)

        loss = self.criterion(predictions, masks.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log predicted class distribution periodically so we can detect
        # the "all-background" collapse early.
        if batch_idx % 20 == 0:
            with torch.no_grad():
                pred_cls = torch.argmax(predictions, dim=1)
                for c in range(self.num_classes):
                    frac = (pred_cls == c).float().mean()
                    self.log(
                        f"train_pred_class_{c}", frac, on_step=True, on_epoch=False
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with per-class IoU logging."""
        if not LIGHTNING_AVAILABLE:
            raise RuntimeError("PyTorch Lightning required for validation")

        images, masks = batch
        predictions = self(images)

        loss = self.criterion(predictions, masks.long())
        pred_classes = torch.argmax(predictions, dim=1)

        # Overall accuracy
        accuracy = (pred_classes == masks).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Per-class accuracy (excluding classes absent from this batch)
        for c in range(self.num_classes):
            mask_c = masks == c
            if mask_c.any():
                acc_c = (pred_classes[mask_c] == c).float().mean()
                self.log(f"val_acc_class_{c}", acc_c, on_step=False, on_epoch=True)

        # Fraction of predictions that are non-background
        non_bg = (pred_classes > 0).float().mean()
        self.log("val_non_bg_frac", non_bg, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers with separate LR groups.

        The decoder (randomly initialised) needs a higher learning rate than
        the pre-trained backbone to converge in a reasonable number of epochs.
        """
        if not LIGHTNING_AVAILABLE:
            raise RuntimeError("PyTorch Lightning required for training")

        # Separate parameter groups: backbone gets base LR, decoder gets 10×
        backbone_params = list(self.backbone.parameters()) + list(
            self.channel_adapter.parameters()
        )
        decoder_params = list(self.decoder.parameters())

        param_groups = [
            {"params": backbone_params, "lr": self.learning_rate},
            {"params": decoder_params, "lr": self.learning_rate * 10},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 50,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self):
        """Unfreeze backbone at the right epoch (checked at *start* so the
        first unfrozen epoch actually trains the backbone)."""
        if not LIGHTNING_AVAILABLE:
            return
        if self.current_epoch >= self.freeze_backbone_epochs:
            self.unfreeze_backbone()
            if self.current_epoch == self.freeze_backbone_epochs:
                logger.info(
                    f"Epoch {self.current_epoch}: backbone unfrozen — "
                    "all parameters now trainable."
                )


class WetlandSatlasModel(pl.LightningModule if LIGHTNING_AVAILABLE else nn.Module):
    """Wetland segmentation model using Satlas Aerial backbone.

    Architecture: Swin-v2-Base (pre-trained on NAIP aerial imagery) →
    Feature Pyramid Network → Upsample → lightweight conv head that
    outputs raw logits for ``num_wetland_classes`` classes.

    We intentionally do **not** use the Satlas built-in ``SimpleHead``
    because it applies softmax internally and returns ``(probs, loss)``,
    which is incompatible with our ``CombinedSegmentationLoss`` (Focal +
    Dice) that expects raw logits.
    """

    def __init__(
        self,
        num_wetland_classes: int = 6,
        freeze_backbone_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        if not SATLAS_AVAILABLE:
            raise ImportError(
                "satlaspretrain_models required. "
                "Install with: pip install satlaspretrain-models"
            )

        if LIGHTNING_AVAILABLE:
            self.save_hyperparameters()

        self.num_classes = num_wetland_classes
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Load Satlas backbone + FPN (NO head).
        # With fpn=True the model returns a list of feature maps.
        # After the built-in Upsample the first element is the
        # full-resolution (1×) 128-channel feature map.
        self.weights_manager = satlaspretrain_models.Weights()
        self.backbone_fpn = self.weights_manager.get_pretrained_model(
            "Aerial_SwinB_SI",
            fpn=True,
            # head is NOT specified → no built-in prediction head
        )

        # Lightweight segmentation head: raw logits (no softmax)
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_wetland_classes, 1),
        )

        # Loss — Weighted cross-entropy with inverse-frequency weights
        # computed for this dataset: 95% background, 5% class-5 wetland.
        # Classes 1-4 are absent but get a small weight to penalise
        # spurious predictions.  Inverse-freq: bg=1/0.95≈1.05,
        # wetland=1/0.05=20.  This makes the gradient contribution from
        # bg and wetland pixels roughly equal.
        class_weights = torch.tensor([1.05, 0.1, 0.1, 0.1, 0.1, 20.0])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Initially freeze backbone + FPN
        self.freeze_backbone()

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze Satlas backbone + FPN parameters."""
        for param in self.backbone_fpn.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze Satlas backbone + FPN parameters."""
        for param in self.backbone_fpn.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) — NAIP image (3 or 4 channels, uint8 or float).

        Returns:
            Raw logits (B, num_classes, H, W).
        """
        # Normalise uint8 → [0, 1]
        if x.max() > 1.0:
            x = x / 255.0

        # Satlas expects RGB only (3 channels)
        if x.shape[1] > 3:
            x = x[:, :3, :, :]

        # Backbone + FPN + Upsample → list of feature maps
        # features[0] is the full-resolution 128-ch map
        features = self.backbone_fpn(x)
        full_res = features[0]  # (B, 128, H, W)

        # Segmentation head → raw logits
        logits = self.seg_head(full_res)  # (B, num_classes, H, W)
        return logits

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if not LIGHTNING_AVAILABLE:
            raise RuntimeError("PyTorch Lightning required for training")

        images, masks = batch
        logits = self(images)

        loss = self.criterion(logits, masks.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % 20 == 0:
            with torch.no_grad():
                pred_cls = torch.argmax(logits, dim=1)
                for c in range(self.num_classes):
                    frac = (pred_cls == c).float().mean()
                    self.log(
                        f"train_pred_class_{c}", frac, on_step=True, on_epoch=False
                    )
        return loss

    def validation_step(self, batch, batch_idx):
        if not LIGHTNING_AVAILABLE:
            raise RuntimeError("PyTorch Lightning required for validation")

        images, masks = batch
        logits = self(images)

        loss = self.criterion(logits, masks.long())
        pred_classes = torch.argmax(logits, dim=1)

        accuracy = (pred_classes == masks).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        for c in range(self.num_classes):
            mask_c = masks == c
            if mask_c.any():
                acc_c = (pred_classes[mask_c] == c).float().mean()
                self.log(f"val_acc_class_{c}", acc_c, on_step=False, on_epoch=True)

        non_bg = (pred_classes > 0).float().mean()
        self.log("val_non_bg_frac", non_bg, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers with separate LR groups."""
        if not LIGHTNING_AVAILABLE:
            raise RuntimeError("PyTorch Lightning required for training")

        # backbone+FPN gets base LR, seg_head gets 10×
        param_groups = [
            {"params": self.backbone_fpn.parameters(), "lr": self.learning_rate},
            {"params": self.seg_head.parameters(), "lr": self.learning_rate * 10},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 50,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self):
        """Unfreeze backbone at the right epoch."""
        if not LIGHTNING_AVAILABLE:
            return
        if self.current_epoch >= self.freeze_backbone_epochs:
            self.unfreeze_backbone()
            if self.current_epoch == self.freeze_backbone_epochs:
                logger.info(
                    f"Epoch {self.current_epoch}: backbone unfrozen — "
                    "all parameters now trainable."
                )


class WetlandDataset(Dataset):
    """PyTorch dataset for wetland training data."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        num_channels: int = 4,
        transform=None,
        image_suffix: str = ".tif",
        mask_suffix: str = ".tif",
    ):
        """Initialize dataset.

        Args:
            images_dir: Directory containing image tiles
            masks_dir: Directory containing mask tiles
            num_channels: Number of channels to use (3 for RGB, 4 for RGBNIR)
            transform: Optional transforms to apply
            image_suffix: File suffix for images
            mask_suffix: File suffix for masks
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_channels = num_channels
        self.transform = transform

        # Get list of image files
        self.image_files = sorted(list(self.images_dir.glob(f"*{image_suffix}")))

        # Verify corresponding masks exist
        valid_files = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / img_file.name.replace(
                image_suffix, mask_suffix
            )
            if mask_file.exists():
                valid_files.append(img_file)

        self.image_files = valid_files
        logger.info(f"Found {len(self.image_files)} valid image-mask pairs")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Read image
        with rasterio.open(img_path) as src:
            if self.num_channels == 3:
                # Read only RGB bands (first 3 channels)
                image = src.read([1, 2, 3])  # Shape: (3, H, W)
            else:
                # Read all channels (backward compatibility)
                image = src.read()  # Shape: (C, H, W)

        # Read mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Shape: (H, W)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def train_wetland_model(
    dataset_dir: str,
    output_dir: str = "wetland_model_output",
    backbone: str = "satlas",
    prithvi_model: str = "Prithvi-EO-2.0-300M-TL",
    batch_size: int = 4,
    max_epochs: int = 50,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    freeze_backbone_epochs: int = 2,
    resume_from: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    """Train wetland segmentation model using either Satlas or Prithvi backbone.

    Supports resuming from a previous checkpoint so training does not need
    to start from scratch each time.

    Args:
        dataset_dir: Directory containing 'images' and 'masks' subdirectories
        output_dir: Directory to save model outputs
        backbone: Backbone type to use ("satlas" or "prithvi")
        prithvi_model: Prithvi model variant to use (only used when backbone="prithvi")
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate
        val_split: Validation split fraction
        freeze_backbone_epochs: Epochs to freeze backbone (default 2)
        resume_from: Path to a checkpoint to resume training from. Accepts:
            - A specific checkpoint file path
            - "last" to auto-detect the last checkpoint in output_dir
            - None to train from scratch (default)
        **kwargs: Additional arguments

    Returns:
        Dictionary with paths to saved model and training logs
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "PyTorch Lightning required for training. Install with: pip install lightning"
        )

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Setup datasets
    images_dir = Path(dataset_dir) / "images"
    masks_dir = Path(dataset_dir) / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(
            f"Dataset directory must contain 'images' and 'masks' subdirectories"
        )

    # Determine number of channels based on backbone
    num_channels = 3 if backbone == "satlas" else 4

    full_dataset = WetlandDataset(
        str(images_dir), str(masks_dir), num_channels=num_channels
    )

    if len(full_dataset) == 0:
        raise ValueError("No training data found in dataset directory")

    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Resolve resume checkpoint path
    ckpt_path = None
    if resume_from is not None:
        if resume_from == "last":
            # Auto-detect last.ckpt saved by ModelCheckpoint(save_last=True)
            last_ckpt = output_path / "checkpoints" / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
                logger.info(f"Resuming from last checkpoint: {ckpt_path}")
            else:
                logger.warning(
                    f"No last.ckpt found in {output_path / 'checkpoints'}. "
                    "Training from scratch."
                )
        elif Path(resume_from).is_file():
            ckpt_path = str(resume_from)
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
        else:
            logger.warning(
                f"Checkpoint not found: {resume_from}. Training from scratch."
            )

    # Initialize model based on backbone type
    if backbone == "satlas":
        model = WetlandSatlasModel(
            learning_rate=learning_rate,
            freeze_backbone_epochs=freeze_backbone_epochs,
            **kwargs,
        )
    elif backbone == "prithvi":
        model = WetlandPrithviModel(
            prithvi_model_name=prithvi_model,
            learning_rate=learning_rate,
            freeze_backbone_epochs=freeze_backbone_epochs,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. Choose 'satlas' or 'prithvi'"
        )

    # If resuming, validate that the checkpoint is compatible with the
    # current model.  Old checkpoints may use a different loss function
    # (e.g. plain CrossEntropyLoss with `criterion.weight` instead of
    # CombinedSegmentationLoss with `criterion.focal.alpha`).  When the
    # state dict is incompatible we load the *model weights* leniently
    # (strict=False) and start a fresh training run instead of crashing.
    if ckpt_path:
        try:
            _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _sd = _ckpt.get("state_dict", _ckpt)
            model.load_state_dict(_sd, strict=True)
            # Strict load succeeded — checkpoint is fully compatible.
            # Re-init model so trainer.fit(ckpt_path=...) handles everything
            # (optimizer state, epoch counter, scheduler, etc.).
            if backbone == "satlas":
                model = WetlandSatlasModel(
                    learning_rate=learning_rate,
                    freeze_backbone_epochs=freeze_backbone_epochs,
                    **kwargs,
                )
            else:  # prithvi
                model = WetlandPrithviModel(
                    prithvi_model_name=prithvi_model,
                    learning_rate=learning_rate,
                    freeze_backbone_epochs=freeze_backbone_epochs,
                    **kwargs,
                )
        except RuntimeError as e:
            logger.warning(
                f"Checkpoint state dict incompatible ({e}). "
                "Loading model weights leniently (strict=False) — "
                "optimizer/scheduler state will NOT be restored."
            )
            missing, unexpected = model.load_state_dict(_sd, strict=False)
            if missing:
                logger.info(f"Missing keys (loss buffers, OK): {missing}")
            if unexpected:
                logger.info(f"Unexpected keys (old loss, OK): {unexpected}")
            # Cannot pass ckpt_path to trainer.fit — the strict reload would
            # fail again.  Training restarts from epoch 0 but with pretrained
            # encoder/decoder weights.
            ckpt_path = None

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="wetland-prithvi-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    # Setup trainer — use "auto" strategy (picks DDP only when multi-GPU).
    # Frozen backbone params are handled naturally by autograd (requires_grad=False)
    # so find_unused_parameters is not needed.
    # Use float32 precision to avoid NaN from large loss values in fp16.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,  # single-GPU to avoid DDP complexity
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        default_root_dir=str(output_path),
    )

    # Train model (resume from checkpoint if provided)
    if ckpt_path:
        logger.info(
            f"Resuming training from epoch saved in {ckpt_path} "
            f"up to max_epochs={max_epochs}"
        )
    else:
        logger.info(
            f"Starting training from scratch with {len(train_dataset)} "
            f"training samples, {len(val_dataset)} validation samples"
        )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Save final model as a proper Lightning checkpoint
    best_model_path = output_path / "best_wetland_model.ckpt"
    trainer.save_checkpoint(best_model_path)

    return {
        "best_model_path": str(best_model_path),
        "checkpoint_path": checkpoint_callback.best_model_path,
        "output_dir": str(output_path),
        "training_completed": True,
        "resumed_from": ckpt_path,
    }


def predict_wetlands_large_image(
    model_path: str,
    input_raster: str,
    output_path: str,
    tile_size: int = 512,
    overlap: int = 64,
    backbone: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """Run wetland prediction on a large raster using trained model.

    Args:
        model_path: Path to trained model checkpoint
        input_raster: Path to input NAIP raster
        output_path: Output path for wetland predictions
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles
        backbone: Backbone type ("satlas" or "prithvi"). If None, auto-detect from checkpoint
        device: Device to use for inference

    Returns:
        Path to output wetland prediction raster
    """
    logger.info(f"Running wetland prediction on {input_raster}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect backbone type if not specified
    if backbone is None:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})
        state_dict = ckpt.get("state_dict", ckpt)

        # Check for Satlas-specific keys in state_dict or hparams
        is_satlas = (
            any(k.startswith(("backbone_fpn.", "seg_head.")) for k in state_dict)
            or "prithvi_model_name" not in hparams
        )
        backbone = "satlas" if is_satlas else "prithvi"
        logger.info(f"Auto-detected backbone: {backbone}")

    # Load trained model based on backbone type
    try:
        if backbone == "satlas":
            model = WetlandSatlasModel.load_from_checkpoint(
                model_path, map_location=device
            )
        else:  # prithvi
            model = WetlandPrithviModel.load_from_checkpoint(
                model_path, map_location=device
            )
    except (KeyError, RuntimeError) as load_err:
        logger.warning(
            f"Strict load_from_checkpoint failed ({load_err}); "
            "falling back to lenient loading with strict=False."
        )
        # Build a fresh model with the saved hyper-parameters (if available)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})

        if backbone == "satlas":
            model = WetlandSatlasModel(
                learning_rate=hparams.get("learning_rate", 1e-4),
                freeze_backbone_epochs=hparams.get("freeze_backbone_epochs", 2),
            )
        else:  # prithvi
            model = WetlandPrithviModel(
                prithvi_model_name=hparams.get(
                    "prithvi_model_name", "Prithvi-EO-2.0-300M-TL"
                ),
                learning_rate=hparams.get("learning_rate", 1e-4),
                freeze_backbone_epochs=hparams.get("freeze_backbone_epochs", 2),
            )

        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"Missing keys (OK for loss buffers): {missing}")
        if unexpected:
            logger.info(f"Unexpected keys (OK for old loss): {unexpected}")

    model.eval()
    model = model.to(device)

    with rasterio.open(input_raster) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()

        # Update profile for output
        profile.update(dtype="uint8", count=1, compress="lzw")

        # Initialize output array
        prediction = np.zeros((height, width), dtype=np.uint8)

        # Process in tiles
        stride = tile_size - overlap

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Calculate actual tile size (handle edges)
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                actual_height = y_end - y
                actual_width = x_end - x

                # Read tile
                window = Window(x, y, actual_width, actual_height)

                # Read appropriate bands based on model type
                if backbone == "satlas":
                    # Satlas expects RGB only (3 channels)
                    tile_data = src.read([1, 2, 3], window=window)  # RGB bands
                    # Normalize to [0,1] range for Satlas
                    tile_data = tile_data.astype(np.float32) / 255.0
                else:
                    # Prithvi expects all channels (RGBNIR)
                    tile_data = src.read(window=window)  # All channels

                # Pad tile if needed
                if actual_height != tile_size or actual_width != tile_size:
                    pad_h = tile_size - actual_height
                    pad_w = tile_size - actual_width
                    tile_data = np.pad(
                        tile_data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
                    )

                # Convert to tensor and add batch dimension
                tile_tensor = (
                    torch.from_numpy(tile_data).float().unsqueeze(0).to(device)
                )

                # Run inference
                with torch.no_grad():
                    pred_logits = model(tile_tensor)
                    pred_classes = torch.argmax(pred_logits, dim=1)

                # Convert back to numpy
                pred_np = pred_classes.cpu().numpy()[0]

                # Remove padding if it was added
                if actual_height != tile_size or actual_width != tile_size:
                    pred_np = pred_np[:actual_height, :actual_width]

                # Handle overlapping regions (take max prediction)
                prediction[y:y_end, x:x_end] = np.maximum(
                    prediction[y:y_end, x:x_end], pred_np
                )

        # Save prediction
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prediction, 1)

    logger.info(f"Wetland prediction saved to {output_path}")
    return output_path


# Convenience functions for integration
def create_wetland_dataset(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    year: int = 2020,
    max_naip_items: int = 50,
    tile_size: int = 512,
    min_wetland_pixels: int = 100,
) -> Dict[str, Any]:
    """Create a complete wetland training dataset for a region.

    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        output_dir: Output directory for training tiles
        year: NAIP imagery year
        max_naip_items: Maximum NAIP tiles to download
        tile_size: Training tile size
        min_wetland_pixels: Minimum wetland pixels per tile

    Returns:
        Dictionary with dataset statistics
    """
    # Initialize builder
    builder = WetlandDatasetBuilder()

    # Download NAIP data
    naip_files = builder.download_naip_for_region(
        bbox=bbox, year=year, max_items=max_naip_items
    )

    # Get NWI data
    nwi_data = builder.get_nwi_data_for_region(bbox=bbox)

    if len(nwi_data) == 0:
        raise ValueError("No wetland data found in the specified region")

    # Create training tiles
    stats = builder.create_training_tiles(
        naip_files=naip_files,
        nwi_gdf=nwi_data,
        output_dir=output_dir,
        tile_size=tile_size,
        min_wetland_pixels=min_wetland_pixels,
    )

    # Add summary info
    stats.update(
        {
            "naip_files_downloaded": len(naip_files),
            "wetland_features_found": len(nwi_data),
            "dataset_dir": output_dir,
            "bbox": bbox,
            "year": year,
        }
    )

    return stats


def get_wetland_classes() -> Dict[str, int]:
    """Get wetland class mapping."""
    return WETLAND_CLASSES.copy()


def visualize_wetland_predictions(
    prediction_path: str,
    naip_path: Optional[str] = None,
    center: Optional[List[float]] = None,
):
    """Visualize wetland predictions using leafmap.

    Args:
        prediction_path: Path to wetland prediction raster
        naip_path: Optional path to original NAIP imagery
        center: Optional map center [lat, lon]
    """
    if not LEAFMAP_AVAILABLE:
        raise ImportError(
            "leafmap required for visualization. Install with: pip install leafmap"
        )

    from matplotlib.colors import ListedColormap

    # Create map
    m = leafmap.Map(center=center or [40, -100], zoom=12)

    # Add NAIP imagery if provided
    if naip_path and os.path.exists(naip_path):
        m.add_raster(naip_path, bands=[4, 1, 2], layer_name="NAIP (NIR-R-G)")

    # Build a proper ListedColormap for the 6 wetland classes
    wetland_color_list = [
        "#ffffff",  # 0: background
        "#1f77b4",  # 1: freshwater_emergent - blue
        "#ff7f0e",  # 2: freshwater_forested - orange
        "#2ca02c",  # 3: freshwater_pond - green
        "#d62728",  # 4: estuarine - red
        "#9467bd",  # 5: other_wetland - purple
    ]
    cmap = ListedColormap(wetland_color_list, name="wetland_classes")

    m.add_raster(
        prediction_path,
        layer_name="Wetland Predictions",
        colormap=cmap,
        vmin=0,
        vmax=5,
        opacity=0.7,
    )

    # Add legend
    legend_dict = {
        "Background": "#ffffff",
        "Freshwater Emergent": "#1f77b4",
        "Freshwater Forested": "#ff7f0e",
        "Freshwater Pond": "#2ca02c",
        "Estuarine": "#d62728",
        "Other Wetland": "#9467bd",
    }
    m.add_legend(legend_dict=legend_dict, title="Wetland Classes")

    return m
