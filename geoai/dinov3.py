"""DINOv3 module for patch similarity analysis with GeoTIFF support.

This module provides tools for computing patch similarity using DINOv3 features
on geospatial imagery stored in GeoTIFF format.
"""

import json
import math
import os
import sys
from typing import Tuple, Optional, Dict, List, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import rasterio
from rasterio.windows import Window
from rasterio.io import DatasetReader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from huggingface_hub import hf_hub_download

from .utils import get_device, coords_to_xy, dict_to_image, dict_to_rioxarray


class DINOv3GeoProcessor:
    """DINOv3 processor with GeoTIFF input/output support.
    https://github.com/facebookresearch/dinov3
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize DINOv3 processor.

        Args:
            model_name: Name of the DINOv3 model. Can be "dinov3_vits16", "dinov3_vits16plus",
                "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16", "dinov3_convnext_tiny",
                "dinov3_convnext_small", "dinov3_convnext_base", "dinov3_convnext_large",
                "dinov3dinov3_vitl16", and "dinov3_vit7b16".
                 See https://github.com/facebookresearch/dinov3 for more details.
            weights_path: Path to model weights (optional)
            device: Torch device to use
            dinov3_location: Path to DINOv3 repository
        """

        dinov3_github_location = "facebookresearch/dinov3"

        if os.getenv("DINOV3_LOCATION") is not None:
            dinov3_location = os.getenv("DINOV3_LOCATION")
        else:
            dinov3_location = dinov3_github_location

        self.dinov3_location = dinov3_location
        self.dinov3_source = (
            "local" if dinov3_location != dinov3_github_location else "github"
        )

        self.device = device or get_device()
        self.model_name = model_name

        # Add DINOv3 to path if needed
        if dinov3_location != "facebookresearch/dinov3" and (
            dinov3_location not in sys.path
        ):
            sys.path.append(dinov3_location)

        # Load model
        self.model = self._load_model(weights_path)
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim

        # Image transforms - satellite imagery normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.430, 0.411, 0.296),  # SAT-493M normalization
                    std=(0.213, 0.156, 0.143),
                ),
            ]
        )

    def _download_model_from_hf(
        self, model_path: Optional[str] = None, repo_id: Optional[str] = None
    ) -> str:
        """
        Download the object detection model from Hugging Face.

        Args:
            model_path: Path to the model file.
            repo_id: Hugging Face repository ID.

        Returns:
            Path to the downloaded model file
        """
        try:

            # Define the repository ID and model filename
            if repo_id is None:
                repo_id = "giswqs/geoai"

            if model_path is None:
                model_path = "dinov3_vitl16_sat493m.pth"

            # Download the model
            model_path = hf_hub_download(repo_id=repo_id, filename=model_path)

            return model_path

        except Exception as e:
            print(f"Error downloading model from Hugging Face: {e}")
            print("Please specify a local model path or ensure internet connectivity.")
            raise

    def _load_model(self, weights_path: Optional[str] = None) -> torch.nn.Module:
        """Load DINOv3 model."""
        try:
            if weights_path and os.path.exists(weights_path):
                # Load with custom weights
                model = torch.hub.load(
                    repo_or_dir=self.dinov3_location,
                    model=self.model_name,
                    source=self.dinov3_source,
                )
                # Load state dict manually
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
            else:
                # Download weights and load manually
                weights_path = self._download_model_from_hf()
                model = torch.hub.load(
                    repo_or_dir=self.dinov3_location,
                    model=self.model_name,
                    source=self.dinov3_source,
                )
                # Load state dict manually
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)

            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv3 model: {e}") from e

    def load_regular_image(
        self,
        image_path: str,
    ) -> Tuple[np.ndarray, dict]:
        """Load regular image file (PNG, JPG, etc.).

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (image array, metadata)
        """
        try:
            # Load image using PIL
            image = Image.open(image_path).convert("RGB")

            # Convert to numpy array (H, W, C)
            img_array = np.array(image)

            # Convert to (C, H, W) format to match GeoTIFF format
            data = np.transpose(img_array, (2, 0, 1)).astype(np.uint8)

            # Create basic metadata
            height, width = img_array.shape[:2]
            metadata = {
                "profile": {
                    "driver": "PNG",
                    "dtype": "uint8",
                    "nodata": None,
                    "width": width,
                    "height": height,
                    "count": 3,
                    "crs": None,
                    "transform": None,
                },
                "crs": None,
                "transform": None,
                "bounds": (0, 0, width, height),
            }

            return data, metadata

        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

    def load_geotiff(
        self,
        source: Union[str, DatasetReader],
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Load GeoTIFF file.

        Args:
            source: Path to GeoTIFF file (str) or an open rasterio.DatasetReader
            window: Rasterio window for reading subset
            bands: List of bands to read (1-indexed)

        Returns:
            Tuple of (image array, metadata)
        """
        # Flag to determine if we need to close the dataset afterwards
        should_close = False
        if isinstance(source, str):
            src = rasterio.open(source)
            should_close = True
        elif isinstance(source, DatasetReader):
            src = source
        else:
            raise TypeError("source must be a str path or a rasterio.DatasetReader")

        try:
            # Read specified bands or all bands
            if bands:
                data = src.read(bands, window=window)
            else:
                data = src.read(window=window)

            # Get metadata
            profile = src.profile.copy()
            if window:
                profile.update(
                    {
                        "height": window.height,
                        "width": window.width,
                        "transform": src.window_transform(window),
                    }
                )

            metadata = {
                "profile": profile,
                "crs": src.crs,
                "transform": profile["transform"],
                "bounds": (
                    src.bounds
                    if not window
                    else rasterio.windows.bounds(window, src.transform)
                ),
            }
        finally:
            if should_close:
                src.close()

        return data, metadata

    def load_image(
        self,
        source: Union[str, DatasetReader],
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Load image file (GeoTIFF or regular image).

        Args:
            source: Path to image file (str) or an open rasterio.DatasetReader
            window: Rasterio window for reading subset (only applies to GeoTIFF)
            bands: List of bands to read (only applies to GeoTIFF)

        Returns:
            Tuple of (image array, metadata)
        """
        if isinstance(source, str):
            # Check if it's a GeoTIFF file
            try:
                # Try to open with rasterio first
                with rasterio.open(source) as src:
                    # If successful and has CRS, treat as GeoTIFF
                    if src.crs is not None:
                        return self.load_geotiff(source, window, bands)
                    # If no CRS, it might be a regular image opened by rasterio
                    else:
                        # Check file extension
                        file_ext = source.lower().split(".")[-1]
                        if file_ext in ["tif", "tiff"]:
                            return self.load_geotiff(source, window, bands)
                        else:
                            return self.load_regular_image(source)
            except (rasterio.RasterioIOError, rasterio.errors.RasterioIOError):
                # If rasterio fails, try as regular image
                return self.load_regular_image(source)
        elif isinstance(source, DatasetReader):
            # Already opened rasterio dataset
            return self.load_geotiff(source, window, bands)
        else:
            raise TypeError("source must be a str path or a rasterio.DatasetReader")

    def save_geotiff(
        self, data: np.ndarray, output_path: str, metadata: dict, dtype: str = "float32"
    ) -> None:
        """Save array as GeoTIFF.

        Args:
            data: Array to save
            output_path: Output file path
            metadata: Metadata from original file
            dtype: Output data type
        """
        profile = metadata["profile"].copy()
        profile.update(
            {
                "dtype": dtype,
                "count": data.shape[0] if data.ndim == 3 else 1,
                "height": data.shape[-2] if data.ndim >= 2 else data.shape[0],
                "width": data.shape[-1] if data.ndim >= 2 else 1,
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                dst.write(data)

    def save_similarity_as_image(
        self, similarity_data: np.ndarray, output_path: str, colormap: str = "turbo"
    ) -> None:
        """Save similarity array as PNG image with colormap.

        Args:
            similarity_data: 2D similarity array
            output_path: Output file path
            colormap: Matplotlib colormap name
        """
        import matplotlib.pyplot as plt

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored_data = cmap(similarity_data)

        # Convert to uint8 image (remove alpha channel)
        img_data = (colored_data[..., :3] * 255).astype(np.uint8)

        # Save as PNG
        img = Image.fromarray(img_data)
        img.save(output_path)

    def preprocess_image_for_dinov3(
        self,
        data: np.ndarray,
        target_size: int = 896,
        normalize_percentile: bool = True,
    ) -> Image.Image:
        """Preprocess image data for DINOv3.

        Args:
            data: Input array (C, H, W) or (H, W)
            target_size: Target size for resizing
            normalize_percentile: Whether to normalize using percentiles

        Returns:
            PIL Image ready for DINOv3
        """
        # Handle different input shapes
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # Add channel dimension
        elif data.ndim == 3 and data.shape[0] > 3:
            # Take first 3 bands if more than 3 channels
            data = data[:3, :, :]

        # Normalize data
        if normalize_percentile:
            # Normalize each band using percentiles
            normalized_data = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[0]):
                band = data[i]
                p2, p98 = np.percentile(band, [2, 98])
                normalized_data[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            # Simple min-max normalization
            normalized_data = (data - data.min()) / (data.max() - data.min())

        # Convert to PIL Image
        if normalized_data.shape[0] == 1:
            # Grayscale - repeat to 3 channels
            img_array = np.repeat(normalized_data[0], 3, axis=0)
        else:
            img_array = normalized_data

        # Transpose to HWC format and convert to uint8
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = (img_array * 255).astype(np.uint8)

        # Create PIL Image
        image = Image.fromarray(img_array)

        # Resize to patch-aligned dimensions
        return self.resize_to_patch_aligned(image, target_size)

    def resize_to_patch_aligned(
        self, image: Image.Image, target_size: int = 896
    ) -> Image.Image:
        """Resize image to be aligned with patch grid."""
        w, h = image.size

        # Calculate new dimensions that are multiples of patch_size
        if w > h:
            new_h = target_size
            new_w = int((w * target_size) / h)
        else:
            new_w = target_size
            new_h = int((h * target_size) / w)

        # Round to nearest multiple of patch_size
        new_h = ((new_h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((new_w + self.patch_size - 1) // self.patch_size) * self.patch_size

        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, int, int]:
        """Extract patch features from image."""

        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features from last layer
            features = self.model.get_intermediate_layers(
                img_tensor, n=1, reshape=True, norm=True
            )[
                0
            ]  # Shape: [1, embed_dim, h_patches, w_patches]

        # Rearrange to [h_patches, w_patches, embed_dim]
        features = features.squeeze(0).permute(1, 2, 0)
        h_patches, w_patches = features.shape[:2]

        return features, h_patches, w_patches

    def compute_patch_similarity(
        self, features: torch.Tensor, patch_x: int, patch_y: int
    ) -> torch.Tensor:
        """Compute cosine similarity between selected patch and all patches."""
        h_patches, w_patches, embed_dim = features.shape

        # Get query patch feature
        query_feature = features[patch_y, patch_x]  # Shape: [embed_dim]

        # Reshape features for batch computation
        all_features = features.view(
            -1, embed_dim
        )  # Shape: [h_patches * w_patches, embed_dim]

        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_feature.unsqueeze(0),  # Shape: [1, embed_dim]
            all_features,  # Shape: [h_patches * w_patches, embed_dim]
            dim=1,
        )

        # Reshape back to patch grid
        similarities = similarities.view(h_patches, w_patches)

        # Normalize to 0-1 range
        similarities = (similarities + 1) / 2

        return similarities

    def compute_similarity(
        self,
        source: str = None,
        features: torch.Tensor = None,
        query_coords: Tuple[float, float] = None,
        output_dir: str = None,
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
        target_size: int = 896,
        save_features: bool = False,
        coord_crs: str = None,
        use_interpolation: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Process GeoTIFF for patch similarity analysis.

        Args:
            source: Path to input GeoTIFF or rasterio dataset
            features: Pre-extracted features (h_patches, w_patches, embed_dim)
            query_coords: (x, y) coordinates in image pixel space or (lon, lat) in geographic space
            output_dir: Output directory for results
            window: Optional window for reading subset
            bands: Optional list of bands to use
            target_size: Target size for processing
            save_features: Whether to save extracted features
            coord_crs: Coordinate CRS of the query coordinates
            use_interpolation: Whether to use interpolation when resizing similarity map

        Returns:
            Dictionary containing similarity results and metadata
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load image (GeoTIFF or regular image)
        data, metadata = self.load_image(source, window, bands)
        raw_img_w, raw_img_h = data.shape[-1], data.shape[-2]

        # Preprocess for DINOv3
        image = self.preprocess_image_for_dinov3(data, target_size)

        # Extract features
        if features is None:
            features, h_patches, w_patches = self.extract_features(image)
        else:
            h_patches, w_patches = features.shape[:2]

        # Convert coordinates to patch space
        img_w, img_h = image.size
        if len(query_coords) == 2:
            # Assume pixel coordinates for now
            if coord_crs is not None:
                [query_coords] = coords_to_xy(source, [query_coords], coord_crs)

            new_x = math.floor(query_coords[0] / raw_img_w * img_w)
            new_y = math.floor(query_coords[1] / raw_img_h * img_h)
            query_coords = [new_x, new_y]

            x_pixel, y_pixel = query_coords
            patch_x = math.floor((x_pixel / img_w) * w_patches)
            patch_y = math.floor((y_pixel / img_h) * h_patches)

            # Clamp to valid range
            patch_x = max(0, min(w_patches - 1, patch_x))
            patch_y = max(0, min(h_patches - 1, patch_y))

        # Compute similarity
        similarities = self.compute_patch_similarity(features, patch_x, patch_y)

        # Prepare results
        results = {
            "similarities": similarities.cpu().numpy(),
            "patch_coords": (patch_x, patch_y),
            "patch_grid_size": (h_patches, w_patches),
            "image_size": (img_w, img_h),
            "metadata": metadata,
        }

        # Save similarity as GeoTIFF
        sim_array = similarities.cpu().numpy()

        # Resize similarity to original data dimensions
        if use_interpolation:
            try:
                from skimage.transform import resize

                sim_resized = resize(
                    sim_array,
                    (data.shape[-2], data.shape[-1]),
                    preserve_range=True,
                    anti_aliasing=True,
                )
            except ImportError:
                # Fallback to PIL if scikit-image not available
                from PIL import Image as PILImage

                sim_pil = PILImage.fromarray((sim_array * 255).astype(np.uint8))
                sim_pil = sim_pil.resize(
                    (data.shape[-1], data.shape[-2]), PILImage.LANCZOS
                )
                sim_resized = np.array(sim_pil, dtype=np.float32) / 255.0
        else:
            # Resize without interpolation (nearest neighbor)
            try:
                from skimage.transform import resize

                sim_resized = resize(
                    sim_array,
                    (data.shape[-2], data.shape[-1]),
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,  # Nearest neighbor interpolation
                )
            except ImportError:
                # Fallback to PIL with nearest neighbor
                from PIL import Image as PILImage

                sim_pil = PILImage.fromarray((sim_array * 255).astype(np.uint8))
                sim_pil = sim_pil.resize(
                    (data.shape[-1], data.shape[-2]), PILImage.NEAREST
                )
                sim_resized = np.array(sim_pil, dtype=np.float32) / 255.0

        # Save similarity map
        if metadata["crs"] is not None:
            # Save as GeoTIFF for georeferenced data
            similarity_path = os.path.join(
                output_dir, f"similarity_patch_{patch_x}_{patch_y}.tif"
            )
            self.save_geotiff(
                sim_resized[np.newaxis, :, :],
                similarity_path,
                metadata,
                dtype="float32",
            )
        else:
            # Save as PNG for regular images
            similarity_path = os.path.join(
                output_dir, f"similarity_patch_{patch_x}_{patch_y}.png"
            )
            self.save_similarity_as_image(sim_resized, similarity_path)

        image_dict = {
            "crs": metadata["crs"],
            "bounds": metadata["bounds"],
            "image": sim_resized[np.newaxis, :, :],
        }
        results["image_dict"] = image_dict

        # Save features if requested
        if save_features:
            features_np = features.cpu().numpy()
            features_path = os.path.join(
                output_dir, f"features_patch_{patch_x}_{patch_y}.npy"
            )
            np.save(features_path, features_np)

        # Save metadata
        metadata_dict = {
            "input_path": source,
            "query_coords": query_coords,
            "patch_coords": (patch_x, patch_y),
            "patch_grid_size": (h_patches, w_patches),
            "image_size": (img_w, img_h),
            "similarity_stats": {
                "max": float(sim_array.max()),
                "min": float(sim_array.min()),
                "mean": float(sim_array.mean()),
                "std": float(sim_array.std()),
            },
        }

        if save_features:
            metadata_path = os.path.join(
                output_dir, f"metadata_patch_{patch_x}_{patch_y}.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2)

            results["output_paths"] = {
                "similarity": similarity_path,
                "metadata": metadata_path,
                "features": features_path if save_features else None,
            }

        return results

    def visualize_similarity(
        self,
        source: str,
        similarity_data: np.ndarray,
        query_coords: Tuple[float, float] = None,
        patch_coords: Tuple[int, int] = None,
        figsize: Tuple[int, int] = (15, 6),
        colormap: str = "turbo",
        alpha: float = 0.7,
        save_path: str = None,
        show_query_point: bool = True,
        overlay: bool = False,
    ) -> plt.Figure:
        """Visualize original image and similarity map side by side or as overlay.

        Args:
            source: Path to original image
            similarity_data: 2D similarity array
            query_coords: Query coordinates in pixel space (x, y)
            patch_coords: Patch coordinates (patch_x, patch_y) for marking query patch
            figsize: Figure size for visualization
            colormap: Colormap for similarity visualization
            alpha: Transparency for overlay mode
            save_path: Optional path to save the visualization
            show_query_point: Whether to show the query point marker
            overlay: If True, overlay similarity on original image; if False, show side by side

        Returns:
            Matplotlib figure object
        """
        # Load original image
        data, metadata = self.load_image(source)

        # Convert image data to displayable format
        if data.ndim == 3:
            if data.shape[0] <= 3:
                # Standard RGB/grayscale image (C, H, W)
                display_img = np.transpose(data, (1, 2, 0))
            else:
                # Multi-band image, take first 3 bands
                display_img = np.transpose(data[:3], (1, 2, 0))
        else:
            # Single band image
            display_img = data

        # Normalize image for display
        if display_img.dtype != np.uint8:
            # Normalize using percentiles
            if display_img.ndim == 3:
                normalized_img = np.zeros_like(display_img, dtype=np.float32)
                for i in range(display_img.shape[2]):
                    band = display_img[:, :, i]
                    p2, p98 = np.percentile(band, [2, 98])
                    normalized_img[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                p2, p98 = np.percentile(display_img, [2, 98])
                normalized_img = np.clip((display_img - p2) / (p98 - p2), 0, 1)
            display_img = normalized_img
        else:
            display_img = display_img / 255.0

        # Ensure similarity data matches image dimensions
        if similarity_data.shape != display_img.shape[:2]:
            from PIL import Image as PILImage

            sim_pil = PILImage.fromarray((similarity_data * 255).astype(np.uint8))
            sim_pil = sim_pil.resize(
                (display_img.shape[1], display_img.shape[0]), PILImage.LANCZOS
            )
            similarity_data = np.array(sim_pil, dtype=np.float32) / 255.0

        if overlay:
            # Single plot with overlay
            fig, ax = plt.subplots(1, 1, figsize=(figsize[1], figsize[1]))

            # Show original image
            if display_img.ndim == 2:
                ax.imshow(display_img, cmap="gray")
            else:
                ax.imshow(display_img)

            # Overlay similarity map
            im_sim = ax.imshow(
                similarity_data, cmap=colormap, alpha=alpha, vmin=0, vmax=1
            )

            # Add colorbar for similarity
            cbar = plt.colorbar(im_sim, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Similarity", rotation=270, labelpad=20)

            ax.set_title("Image with Similarity Overlay")

        else:
            # Side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Original image
            if display_img.ndim == 2:
                ax1.imshow(display_img, cmap="gray")
            else:
                ax1.imshow(display_img)
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Similarity map
            im_sim = ax2.imshow(similarity_data, cmap=colormap, vmin=0, vmax=1)
            ax2.set_title("Similarity Map")
            ax2.axis("off")

            # Add colorbar
            cbar = plt.colorbar(im_sim, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("Similarity", rotation=270, labelpad=20)

        # Mark query point if provided
        if show_query_point and query_coords is not None:
            x, y = query_coords
            if overlay:
                ax.plot(
                    x,
                    y,
                    "r*",
                    markersize=15,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )
                ax.plot(x, y, "r*", markersize=12)
            else:
                ax1.plot(
                    x,
                    y,
                    "r*",
                    markersize=15,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )
                ax1.plot(x, y, "r*", markersize=12)
                ax2.plot(
                    x,
                    y,
                    "r*",
                    markersize=15,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )
                ax2.plot(x, y, "r*", markersize=12)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def visualize_patches(
        self,
        image: Image.Image,
        features: torch.Tensor,
        patch_coords: Tuple[int, int],
        add_text: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = None,
    ) -> plt.Figure:
        """Visualize image with patch grid and highlight selected patch.

        Args:
            image: PIL Image
            features: Feature tensor (h_patches, w_patches, embed_dim)
            patch_coords: Selected patch coordinates (patch_x, patch_y)
            add_text: Whether to add text to the patch
            figsize: Figure size
            save_path: Optional path to save visualization

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Display image
        ax.imshow(image)
        ax.set_title("Image with Patch Grid")
        ax.axis("off")

        # Get dimensions
        img_w, img_h = image.size
        h_patches, w_patches = features.shape[:2]
        patch_x, patch_y = patch_coords

        # Calculate patch size in pixels
        patch_w = img_w / w_patches
        patch_h = img_h / h_patches

        # Draw patch grid
        for i in range(w_patches + 1):
            x = i * patch_w
            ax.axvline(x=x, color="white", alpha=0.3, linewidth=0.5)

        for i in range(h_patches + 1):
            y = i * patch_h
            ax.axhline(y=y, color="white", alpha=0.3, linewidth=0.5)

        # Highlight selected patch
        rect_x = patch_x * patch_w
        rect_y = patch_y * patch_h
        rect = patches.Rectangle(
            (rect_x, rect_y),
            patch_w,
            patch_h,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add patch coordinate text
        if add_text:
            ax.text(
                rect_x + patch_w / 2,
                rect_y + patch_h / 2,
                f"({patch_x}, {patch_y})",
                color="red",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def create_similarity_overlay(
        self,
        source: str,
        similarity_data: np.ndarray,
        colormap: str = "turbo",
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Create an overlay of similarity map on original image.

        Args:
            source: Path to original image
            similarity_data: 2D similarity array
            colormap: Colormap for similarity visualization
            alpha: Transparency for overlay

        Returns:
            RGB overlay image as numpy array
        """
        # Load original image
        data, _ = self.load_image(source)

        # Convert to display format
        if data.ndim == 3:
            if data.shape[0] <= 3:
                display_img = np.transpose(data, (1, 2, 0))
            else:
                display_img = np.transpose(data[:3], (1, 2, 0))
        else:
            display_img = data

        # Normalize image
        if display_img.dtype != np.uint8:
            if display_img.ndim == 3:
                normalized_img = np.zeros_like(display_img, dtype=np.float32)
                for i in range(display_img.shape[2]):
                    band = display_img[:, :, i]
                    p2, p98 = np.percentile(band, [2, 98])
                    normalized_img[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                p2, p98 = np.percentile(display_img, [2, 98])
                normalized_img = np.clip((display_img - p2) / (p98 - p2), 0, 1)
            base_img = normalized_img
        else:
            base_img = display_img / 255.0

        # Convert grayscale to RGB if needed
        if base_img.ndim == 2:
            base_img = np.stack([base_img] * 3, axis=2)

        # Resize similarity data to match image
        if similarity_data.shape != base_img.shape[:2]:
            from PIL import Image as PILImage

            sim_pil = PILImage.fromarray((similarity_data * 255).astype(np.uint8))
            sim_pil = sim_pil.resize(
                (base_img.shape[1], base_img.shape[0]), PILImage.LANCZOS
            )
            similarity_data = np.array(sim_pil, dtype=np.float32) / 255.0

        # Apply colormap to similarity data
        cmap = plt.get_cmap(colormap)
        colored_similarity = cmap(similarity_data)[:, :, :3]  # Remove alpha channel

        # Blend images
        overlay_img = (1 - alpha) * base_img + alpha * colored_similarity

        return np.clip(overlay_img, 0, 1)

    def batch_similarity_analysis(
        self,
        input_path: str,
        query_points: List[Tuple[float, float]],
        output_dir: str,
        window: Optional[Window] = None,
        bands: Optional[List[int]] = None,
        target_size: int = 896,
    ) -> List[Dict[str, np.ndarray]]:
        """Process multiple query points for similarity analysis.

        Args:
            input_path: Path to input GeoTIFF
            query_points: List of (x, y) coordinates
            output_dir: Output directory for results
            window: Optional window for reading subset
            bands: Optional list of bands to use
            target_size: Target size for processing

        Returns:
            List of result dictionaries
        """
        results = []
        for i, coords in enumerate(query_points):
            point_output_dir = os.path.join(output_dir, f"point_{i}")
            result = self.compute_similarity(
                source=input_path,
                query_coords=coords,
                output_dir=point_output_dir,
                window=window,
                bands=bands,
                target_size=target_size,
            )
            results.append(result)

        return results


def create_similarity_map(
    input_image: str,
    query_coords: Tuple[float, float],
    output_dir: str,
    model_name: str = "dinov3_vitl16",
    weights_path: Optional[str] = None,
    window: Optional[Window] = None,
    bands: Optional[List[int]] = None,
    target_size: int = 896,
    save_features: bool = False,
    coord_crs: str = None,
    use_interpolation: bool = True,
) -> Dict[str, np.ndarray]:
    """Convenience function to create similarity map from image file.

    Args:
        input_image: Path to input image file (GeoTIFF, PNG, JPG, etc.)
        query_coords: Query coordinates (x, y) in pixel space
        output_dir: Output directory
        model_name: DINOv3 model name
        weights_path: Optional path to model weights
        window: Optional rasterio window (only applies to GeoTIFF)
        bands: Optional list of bands to use (only applies to GeoTIFF)
        target_size: Target size for processing
        save_features: Whether to save extracted features
        coord_crs: Coordinate CRS of the query coordinates (only applies to GeoTIFF)
        use_interpolation: Whether to use interpolation when resizing similarity map

    Returns:
        Dictionary containing results
    """
    processor = DINOv3GeoProcessor(model_name=model_name, weights_path=weights_path)

    return processor.compute_similarity(
        source=input_image,
        query_coords=query_coords,
        output_dir=output_dir,
        window=window,
        bands=bands,
        target_size=target_size,
        save_features=save_features,
        coord_crs=coord_crs,
        use_interpolation=use_interpolation,
    )


def analyze_image_patches(
    input_image: str,
    query_points: List[Tuple[float, float]],
    output_dir: str,
    model_name: str = "dinov3_vitl16",
    weights_path: Optional[str] = None,
) -> List[Dict[str, np.ndarray]]:
    """Analyze multiple patches in an image file.

    Args:
        input_image: Path to input image file (GeoTIFF, PNG, JPG, etc.)
        query_points: List of query coordinates
        output_dir: Output directory
        model_name: DINOv3 model name
        weights_path: Optional path to model weights

    Returns:
        List of result dictionaries
    """
    processor = DINOv3GeoProcessor(model_name=model_name, weights_path=weights_path)

    return processor.batch_similarity_analysis(input_image, query_points, output_dir)


def visualize_similarity_results(
    input_image: str,
    query_coords: Tuple[float, float],
    output_dir: str = None,
    model_name: str = "dinov3_vitl16",
    weights_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 6),
    colormap: str = "turbo",
    alpha: float = 0.7,
    save_path: str = None,
    show_query_point: bool = True,
    overlay: bool = False,
    target_size: int = 896,
    coord_crs: str = None,
    use_interpolation: bool = True,
) -> Dict:
    """Create similarity map and visualize results in one function.

    Args:
        input_image: Path to input image file (GeoTIFF, PNG, JPG, etc.)
        query_coords: Query coordinates (x, y) in pixel space
        output_dir: Output directory for similarity map files (optional)
        model_name: DINOv3 model name
        weights_path: Optional path to model weights
        figsize: Figure size for visualization
        colormap: Colormap for similarity visualization
        alpha: Transparency for overlay mode
        save_path: Optional path to save the visualization
        show_query_point: Whether to show the query point marker
        overlay: If True, overlay similarity on original image; if False, show side by side
        target_size: Target size for processing
        coord_crs: Coordinate CRS of the query coordinates
        use_interpolation: Whether to use interpolation when resizing similarity map

    Returns:
        Dictionary containing similarity results, metadata, and matplotlib figure
    """
    processor = DINOv3GeoProcessor(model_name=model_name, weights_path=weights_path)

    # Create temporary output directory if not provided
    if output_dir is None:
        import tempfile

        output_dir = tempfile.mkdtemp(prefix="dinov3_similarity_")

    # Compute similarity
    results = processor.compute_similarity(
        source=input_image,
        query_coords=query_coords,
        output_dir=output_dir,
        target_size=target_size,
        coord_crs=coord_crs,
        use_interpolation=use_interpolation,
    )

    # Get similarity data from results
    similarity_data = results["image_dict"]["image"][0]  # Remove channel dimension

    # Create visualization
    fig = processor.visualize_similarity(
        source=input_image,
        similarity_data=similarity_data,
        query_coords=query_coords,
        patch_coords=results["patch_coords"],
        figsize=figsize,
        colormap=colormap,
        alpha=alpha,
        save_path=save_path,
        show_query_point=show_query_point,
        overlay=overlay,
    )

    # Add figure to results
    results["visualization"] = fig

    return results
