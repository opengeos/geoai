"""
Clay foundation model wrapper for geospatial embeddings.

This module provides a simplified wrapper around the Clay foundation model for generating
rich spectral embeddings from geospatial imagery.
"""

import os
import math
import datetime
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, Union


from claymodel.module import ClayMAEModule
from .utils import download_file
from torchvision.transforms import v2
import yaml
from box import Box


def normalize_timestamp(date):
    """Normaize the timestamp for clay. Taken from https://github.com/Clay-foundation/stacchip/blob/main/stacchip/processors/prechip.py"""
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(bounds):
    """Normalize latitude and longitude for clay. Taken from https://github.com/Clay-foundation/stacchip/blob/main/stacchip/processors/prechip.py"""
    lon = bounds[0] + (bounds[2] - bounds[0]) / 2
    lat = bounds[1] + (bounds[3] - bounds[1]) / 2

    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validate metadata structure for custom sensor configurations.

    Args:
        metadata: Metadata dictionary to validate

    Raises:
        ValueError: If metadata structure is invalid
    """
    required_keys = ["band_order", "gsd", "bands"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required key: {key}")

    band_order = metadata["band_order"]
    if not isinstance(band_order, list) or not band_order:
        raise ValueError("band_order must be a non-empty list")

    num_bands = len(band_order)

    # Validate bands structure
    bands = metadata["bands"]
    required_band_keys = ["mean", "std", "wavelength"]
    for band_key in required_band_keys:
        if band_key not in bands:
            raise ValueError(f"Missing required bands key: {band_key}")

        band_dict = bands[band_key]
        if not isinstance(band_dict, dict):
            raise ValueError(f"bands.{band_key} must be a dictionary")

        # Check all bands are present
        for band in band_order:
            if band not in band_dict:
                raise ValueError(f"Missing {band_key} value for band: {band}")

        # Check no extra bands
        if len(band_dict) != num_bands:
            raise ValueError(
                f"bands.{band_key} has {len(band_dict)} entries but expected {num_bands}"
            )


def load_metadata(
    sensor_name: Optional[str] = None, custom_metadata: Optional[Dict[str, Any]] = None
) -> Box:
    """
    Load sensor metadata from config file or validate custom metadata.

    Args:
        sensor_name: Name of sensor to load from config file
        custom_metadata: Custom metadata dictionary (takes priority over sensor_name)

    Returns:
        Box object containing sensor metadata

    Raises:
        ValueError: If neither parameter provided or metadata invalid
    """
    # Load from config file
    config_path = os.path.join(
        os.path.dirname(__file__), "config", "clay_metadata.yaml"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Sensor metadata config file not found: {config_path}")

    if custom_metadata is not None:
        validate_metadata(custom_metadata)
        return config_path, Box(custom_metadata)

    if sensor_name is None:
        raise ValueError("Must provide either sensor_name or custom_metadata")

    with open(config_path, "r") as f:
        all_metadata = yaml.safe_load(f)

    if sensor_name not in all_metadata:
        available_sensors = list(all_metadata.keys())
        raise ValueError(
            f"Unknown sensor: {sensor_name}. Available sensors: {available_sensors}"
        )

    return config_path, Box(all_metadata[sensor_name])


class Clay:
    """
    Clay foundation model wrapper for generating geospatial embeddings.

    This class provides a simplified interface to generate rich spectral embeddings from
    geospatial imagery using the Clay foundation model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_size: str = "large",
        sensor_name: str = "",
        custom_metadata: Optional[Dict[str, Any]] = None,
        device: str = "auto",
    ):
        """
        Initialize Clay embeddings model.

        Args:
            checkpoint_path: Path to Clay model checkpoint (defaults to ~/.cache/clay/clay-v1.5.ckpt)
            model_size: Model size ('tiny', 'small', 'base', 'large')
            sensor_name: Name of sensor to load metadata for
            custom_metadata: Custom metadata dictionary (takes priority over sensor_name)
            device: Device to run model on ('auto', 'cuda', 'cpu')
        """

        # Set default checkpoint path if not provided
        if checkpoint_path is None:
            cache_dir = os.path.expanduser("~/.cache/clay")
            checkpoint_path = os.path.join(cache_dir, "clay-v1.5.ckpt")

        self.checkpoint_path = checkpoint_path
        self.model_size = model_size

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Handle metadata loading with warnings
        if custom_metadata is not None and sensor_name != "":
            print(
                f"Warning: Using custom metadata for processing, but sensor name '{sensor_name}' will be retained in saved embeddings"
            )

        if custom_metadata is None and sensor_name == "":
            sensor_name = "sentinel-2-l2a"
            print(
                "Warning: No sensor name or custom metadata provided, defaulting to 'sentinel-2-l2a'"
            )

        self.sensor_name = sensor_name
        self.config_path, self.metadata = load_metadata(
            sensor_name if sensor_name else None, custom_metadata
        )

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the Clay model from checkpoint."""
        # Check if checkpoint exists, if not download from HuggingFace
        if not os.path.exists(self.checkpoint_path):
            print(
                f"Checkpoint not found at {self.checkpoint_path}, downloading from HuggingFace..."
            )
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            url = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
            self.checkpoint_path = download_file(
                url, self.checkpoint_path, overwrite=False, unzip=False
            )

        torch.set_default_device(self.device)
        self.module = ClayMAEModule.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model_size=self.model_size,
            metadata_path=self.config_path,
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
        self.module.eval()

    def prepare_datacube(
        self,
        image: Union[np.ndarray, torch.Tensor],
        bounds: Optional[
            Union[
                Tuple[float, float, float, float],
                List[Tuple[float, float, float, float]],
            ]
        ] = None,
        date: Optional[Union[datetime.datetime, List[datetime.datetime]]] = None,
        gsd: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare datacube for Clay model input (single image or batch).

        Args:
            image: Input image array [H, W, C] for single image or [B, H, W, C] for batch
            bounds: Image bounds as (min_lon, min_lat, max_lon, max_lat) in WGS84, or list of bounds for batch
            date: Image acquisition date, or list of dates for batch
            gsd: Ground sample distance override (same for all images in batch)

        Returns:
            Datacube dictionary for Clay model
        """
        # Get sensor metadata
        band_order = self.metadata.band_order
        gsd = gsd if gsd is not None else self.metadata.gsd

        # Extract normalization parameters
        means = [self.metadata.bands.mean[band] for band in band_order]
        stds = [self.metadata.bands.std[band] for band in band_order]
        wavelengths = [self.metadata.bands.wavelength[band] for band in band_order]

        # Determine if this is a batch
        if isinstance(image, torch.Tensor):
            is_batch = image.dim() == 4
        else:
            is_batch = len(image.shape) == 4

        # Convert to tensor and handle dimensions
        if isinstance(image, torch.Tensor):
            pixels = image.float()
            if is_batch:
                if pixels.shape[-1] != pixels.shape[1]:  # [B, H, W, C] format
                    pixels = pixels.permute(0, 3, 1, 2)  # -> [B, C, H, W]
            else:
                if (
                    pixels.dim() == 3 and pixels.shape[-1] != pixels.shape[0]
                ):  # [H, W, C] format
                    pixels = pixels.permute(2, 0, 1)  # -> [C, H, W]
                pixels = pixels.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        else:
            if is_batch:
                pixels = torch.from_numpy(image.astype(np.float32)).permute(
                    0, 3, 1, 2
                )  # [B, H, W, C] -> [B, C, H, W]
            else:
                pixels = (
                    torch.from_numpy(image.astype(np.float32))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )  # [H, W, C] -> [1, C, H, W]

        # Normalize
        transform = v2.Compose([v2.Normalize(mean=means, std=stds)])
        pixels = transform(pixels)

        batch_size = pixels.shape[0]

        # Prepare temporal encoding
        if date is not None:
            if is_batch:
                if not isinstance(date, list):
                    raise ValueError(
                        "For batch processing, date must be a list of datetime objects"
                    )
                if len(date) != batch_size:
                    raise ValueError(
                        f"Number of dates ({len(date)}) must match batch size ({batch_size})"
                    )

                times = [normalize_timestamp(d) for d in date]
                week_cos_sin = torch.tensor(
                    [t[0] for t in times], dtype=torch.float32, device=self.device
                )  # [B, 2]
                hour_cos_sin = torch.tensor(
                    [t[1] for t in times], dtype=torch.float32, device=self.device
                )  # [B, 2]
                time_tensor = torch.cat([week_cos_sin, hour_cos_sin], dim=1)  # [B, 4]
            else:
                week_norm, hour_norm = normalize_timestamp(date)
                time_tensor = torch.tensor(
                    week_norm + hour_norm,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
        else:
            time_tensor = torch.zeros(
                batch_size, 4, dtype=torch.float32, device=self.device
            )

        # Prepare spatial encoding
        if bounds is not None:
            if is_batch:
                if not isinstance(bounds, list):
                    raise ValueError(
                        "For batch processing, bounds must be a list of bound tuples"
                    )
                if len(bounds) != batch_size:
                    raise ValueError(
                        f"Number of bounds ({len(bounds)}) must match batch size ({batch_size})"
                    )

                latlons = [normalize_latlon(b) for b in bounds]
                lat_cos_sin = torch.tensor(
                    [ll[0] for ll in latlons], dtype=torch.float32, device=self.device
                )  # [B, 2]
                lon_cos_sin = torch.tensor(
                    [ll[1] for ll in latlons], dtype=torch.float32, device=self.device
                )  # [B, 2]
                latlon_tensor = torch.cat([lat_cos_sin, lon_cos_sin], dim=1)  # [B, 4]
            else:
                lat_norm, lon_norm = normalize_latlon(bounds)
                latlon_tensor = torch.tensor(
                    lat_norm + lon_norm,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
        else:
            latlon_tensor = torch.zeros(
                batch_size, 4, dtype=torch.float32, device=self.device
            )

        # Create datacube
        datacube = {
            "pixels": pixels.to(self.device),
            "time": time_tensor,
            "latlon": latlon_tensor,
            "gsd": torch.full(
                (batch_size,), gsd, dtype=torch.float32, device=self.device
            ),
            "waves": torch.tensor(wavelengths, dtype=torch.float32, device=self.device),
        }

        return datacube

    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor],
        bounds: Optional[
            Union[
                Tuple[float, float, float, float],
                List[Tuple[float, float, float, float]],
            ]
        ] = None,
        date: Optional[Union[datetime.datetime, List[datetime.datetime]]] = None,
        gsd: Optional[float] = None,
        only_cls_token: bool = False,
    ) -> torch.Tensor:
        """
        Generate embeddings for single image or batch of images.

        Args:
            image: Input image array [H, W, C] for single image or [B, H, W, C] for batch
            bounds: Image bounds as (min_lon, min_lat, max_lon, max_lat) in WGS84, or list of bounds for batch
            date: Image acquisition date, or list of dates for batch
            gsd: Ground sample distance override (same for all images in batch)
            only_cls_token: If True, return only CLS token embeddings

        Returns:
            Embedding array [1, seq_len, embed_dim] for single image or [B, seq_len, embed_dim] for batch
        """
        datacube = self.prepare_datacube(image, bounds, date, gsd)

        with torch.no_grad():
            encoded_patches, _, _, _ = self.module.model.encoder(datacube)
            if only_cls_token:
                # Extract only class token (global embedding)
                embedding = encoded_patches[:, 0, :]
            else:
                # Return full sequence
                embedding = encoded_patches

        return embedding

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        bounds: Tuple[float, float, float, float],
        date: datetime.datetime,
        image_shape: Tuple[int, int],
        output_path: str,
        format: str = "npz",
    ):
        """
        Save embeddings to file.

        Args:
            embeddings: Embedding array
            bounds: Image bounds
            date: Image date
            image_shape: Shape of original image
            output_path: Output file path
            format: Output format ('npz', 'pt')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "npz":
            np.savez_compressed(
                output_path,
                embeddings=embeddings,
                bounds=np.array(bounds),
                date=date.isoformat(),
                image_shape=np.array(image_shape),
                sensor_type=self.sensor_name,
            )
        elif format == "pt":
            data = {
                "embeddings": torch.from_numpy(embeddings),
                "bounds": bounds,
                "date": date.isoformat(),
                "image_shape": image_shape,
                "sensor_type": self.sensor_name,
            }
            torch.save(data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Load embeddings from file.

    Args:
        file_path: Path to embeddings file

    Returns:
        Embeddings dictionary
    """
    if file_path.endswith(".npz"):
        data = np.load(file_path, allow_pickle=True)
        # Handle both old format (lat/lon) and new format (bounds)
        if "bounds" in data:
            return {
                "embeddings": data["embeddings"],
                "tile_coords": data["tile_coords"].tolist(),
                "image_shape": tuple(data["image_shape"]),
                "sensor_type": str(data["sensor_type"]),
                "bounds": tuple(data["bounds"]),
                "date": str(data["date"]) if data["date"] != "None" else None,
                "num_tiles": int(data["num_tiles"]),
            }
        else:
            # Legacy format with lat/lon
            return {
                "embeddings": data["embeddings"],
                "tile_coords": data["tile_coords"].tolist(),
                "image_shape": tuple(data["image_shape"]),
                "sensor_type": str(data["sensor_type"]),
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
                "date": str(data["date"]) if data["date"] != "None" else None,
                "num_tiles": int(data["num_tiles"]),
            }
    elif file_path.endswith(".pt"):
        return torch.load(file_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
