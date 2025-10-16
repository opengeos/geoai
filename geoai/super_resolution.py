"""Super-resolution module for enhancing geospatial imagery resolution using deep learning."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block for ESRGAN generator."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out


class ESRGANGenerator(nn.Module):
    """ESRGAN-inspired generator network for super-resolution."""

    def __init__(self, upscale_factor: int = 4, num_channels: int = 3):
        super(ESRGANGenerator, self).__init__()
        self.upscale_factor = upscale_factor

        # Initial convolution
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])

        # Second convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling layers
        self.upsample = nn.Sequential()
        for _ in range(int(np.log2(upscale_factor))):
            self.upsample.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            self.upsample.append(nn.PixelShuffle(2))
            self.upsample.append(nn.ReLU(inplace=True))

        # Final convolution
        self.conv3 = nn.Conv2d(64, num_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = F.relu(self.conv1(x))
        out = self.residual_blocks(out1)
        out2 = self.bn2(self.conv2(out))
        out = out1 + out2  # Skip connection
        out = self.upsample(out)
        out = self.conv3(out)
        return out


class SRCNN(nn.Module):
    """SRCNN (Super-Resolution CNN) implementation."""

    def __init__(self, upscale_factor: int = 2, num_channels: int = 3):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor

        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic upsampling first
        x = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )
        # Then refine with CNN
        x = self.layers(x)
        return x


class GeospatialSRDataset(Dataset):
    """Dataset for geospatial super-resolution training."""

    def __init__(self, image_dir: str, upscale_factor: int = 4, patch_size: int = 64):
        self.image_dir = Path(image_dir)
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.image_files = list(self.image_dir.glob("*.tif")) + list(
            self.image_dir.glob("*.tiff")
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with rasterio.open(self.image_files[idx]) as src:
            # Read RGB bands
            if src.count >= 3:
                image = src.read([1, 2, 3])
            else:
                image = src.read(1)
                image = np.stack([image, image, image])  # Convert to 3-channel

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Create low-resolution version
            hr_image = torch.from_numpy(image)
            lr_image = F.interpolate(
                hr_image.unsqueeze(0),
                scale_factor=1 / self.upscale_factor,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

            # Random crop to patch size
            if hr_image.shape[-1] > self.patch_size:
                i = np.random.randint(0, hr_image.shape[-2] - self.patch_size)
                j = np.random.randint(0, hr_image.shape[-1] - self.patch_size)
                hr_image = hr_image[:, i : i + self.patch_size, j : j + self.patch_size]
                # Scale indices and patch size for LR image
                lr_i = i // self.upscale_factor
                lr_j = j // self.upscale_factor
                lr_patch_size = self.patch_size // self.upscale_factor
                lr_image = lr_image[
                    :, lr_i : lr_i + lr_patch_size, lr_j : lr_j + lr_patch_size
                ]

            return lr_image, hr_image


class SuperResolutionModel:
    """Super-resolution model for geospatial imagery enhancement."""

    def __init__(
        self,
        model_type: str = "esrgan",
        upscale_factor: int = 4,
        device: Optional[str] = None,
        num_channels: int = 3,
    ):
        """
        Initialize super-resolution model.

        Args:
            model_type: Type of model ('esrgan', 'srcnn')
            upscale_factor: Upscaling factor (2, 4, 8)
            device: Computing device ('cuda', 'cpu', 'mps')
            num_channels: Number of input channels
        """
        self.model_type = model_type
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)

        # Loss functions
        self.criterion = nn.MSELoss()

        logger.info(
            f"Initialized {model_type} model with {upscale_factor}x upscaling on {self.device}"
        )

    def _create_model(self) -> nn.Module:
        """Create the appropriate model architecture."""
        if self.model_type == "esrgan":
            return ESRGANGenerator(self.upscale_factor, self.num_channels)
        elif self.model_type == "srcnn":
            # SRCNN implementation with upsampling
            return SRCNN(self.upscale_factor, self.num_channels)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def load_model(self, model_path: str) -> None:
        """Load pre-trained model weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logger.info(f"Loaded model from {model_path}")

    def save_model(self, model_path: str) -> None:
        """Save model weights."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_type": self.model_type,
                "upscale_factor": self.upscale_factor,
            },
            model_path,
        )
        logger.info(f"Saved model to {model_path}")

    def train(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the super-resolution model."""

        # Create datasets
        train_dataset = GeospatialSRDataset(train_dir, self.upscale_factor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_dir:
            val_dataset = GeospatialSRDataset(val_dir, self.upscale_factor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        history = {"train_loss": [], "val_loss": []}

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for lr_images, hr_images in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}"
            ):
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                optimizer.zero_grad()
                sr_images = self.model(lr_images)
                loss = self.criterion(sr_images, hr_images)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                logger.info(
                    f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

            # Save model
            if save_path and (epoch + 1) % 10 == 0:
                self.save_model(f"{save_path}_epoch_{epoch+1}.pth")

        if save_path:
            self.save_model(f"{save_path}_final.pth")

        return history

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                sr_images = self.model(lr_images)
                loss = self.criterion(sr_images, hr_images)
                val_loss += loss.item()

        self.model.train()
        return val_loss / len(val_loader)

    def enhance_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        tile_size: int = 512,
        overlap: int = 32,
    ) -> Union[str, np.ndarray]:
        """
        Enhance image resolution.

        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image (optional)
            tile_size: Size of tiles for processing large images
            overlap: Overlap between tiles

        Returns:
            Path to output image or enhanced image array
        """

        with rasterio.open(input_path) as src:
            # Read image data
            if src.count >= 3:
                image = src.read([1, 2, 3])
            else:
                image = src.read(1)
                image = np.stack([image] * 3)  # Convert to 3-channel

            # Get metadata
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Process in tiles if image is large
            height, width = image.shape[1], image.shape[2]
            if height > tile_size or width > tile_size:
                enhanced = self._process_tiled(image, tile_size, overlap)
            else:
                enhanced = self._enhance_single_tile(image)

            # Denormalize
            enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)

            if output_path:
                # Update metadata for enhanced image
                new_height, new_width = enhanced.shape[1], enhanced.shape[2]
                new_transform = from_bounds(
                    transform.c,  # left
                    transform.f
                    - (transform.e * height * self.upscale_factor),  # bottom
                    transform.c + (transform.a * width * self.upscale_factor),  # right
                    transform.f,  # top
                    new_width,
                    new_height,
                )

                meta.update(
                    {
                        "height": new_height,
                        "width": new_width,
                        "transform": new_transform,
                        "count": 3,
                    }
                )

                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(enhanced)

                return output_path
            else:
                return enhanced

    def _enhance_single_tile(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single image tile."""
        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            enhanced = self.model(tensor)
            enhanced = enhanced.squeeze(0).cpu().numpy()

        return enhanced

    def _process_tiled(
        self, image: np.ndarray, tile_size: int, overlap: int
    ) -> np.ndarray:
        """Process large images in tiles."""
        _, height, width = image.shape
        enhanced_height = height * self.upscale_factor
        enhanced_width = width * self.upscale_factor

        enhanced = np.zeros((3, enhanced_height, enhanced_width), dtype=np.float32)

        stride = tile_size - overlap
        enhanced_stride = stride * self.upscale_factor

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Extract tile with overlap
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)

                tile = image[:, y:y_end, x:x_end]

                # Pad if necessary
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                    padded[:, : tile.shape[1], : tile.shape[2]] = tile
                    tile = padded

                # Enhance tile
                enhanced_tile = self._enhance_single_tile(tile)

                # Remove padding
                enhanced_tile = enhanced_tile[
                    :,
                    : min(tile_size, y_end - y) * self.upscale_factor,
                    : min(tile_size, x_end - x) * self.upscale_factor,
                ]

                # Place in output
                ey_start = y * self.upscale_factor
                ex_start = x * self.upscale_factor
                ey_end = ey_start + enhanced_tile.shape[1]
                ex_end = ex_start + enhanced_tile.shape[2]

                enhanced[:, ey_start:ey_end, ex_start:ex_end] = enhanced_tile

        return enhanced

    def evaluate(
        self, test_dir: str, metrics: List[str] = ["psnr", "ssim"]
    ) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        test_dataset = GeospatialSRDataset(test_dir, self.upscale_factor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        results = {metric: [] for metric in metrics}

        self.model.eval()
        with torch.no_grad():
            for lr_images, hr_images in test_loader:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                sr_images = self.model(lr_images)

                # Convert to numpy
                sr_np = sr_images.cpu().numpy().squeeze(0)
                hr_np = hr_images.cpu().numpy().squeeze(0)

                # Calculate metrics
                if "psnr" in metrics:
                    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
                    results["psnr"].append(psnr)

                if "ssim" in metrics:
                    ssim = structural_similarity(
                        hr_np, sr_np, data_range=1.0, channel_axis=0
                    )
                    results["ssim"].append(ssim)

        # Average results
        for metric in results:
            results[metric] = np.mean(results[metric])

        return results


def create_super_resolution_model(
    model_type: str = "esrgan", upscale_factor: int = 4, **kwargs
) -> SuperResolutionModel:
    """Convenience function to create a super-resolution model."""
    return SuperResolutionModel(
        model_type=model_type, upscale_factor=upscale_factor, **kwargs
    )
