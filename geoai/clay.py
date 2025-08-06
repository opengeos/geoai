"""
Clay foundation model wrapper for geospatial embeddings.

This module provides a wrapper around the Clay foundation model for generating
rich spectral embeddings from geospatial imagery. It integrates with the 
segment-geospatial library's raster I/O infrastructure.
"""

import os
import math
import datetime
import numpy as np
import torch
import cv2
import rasterio
import warnings
from typing import Optional, Union, Tuple, Dict, Any

try:
    from claymodel.model import ClayMAEModule
    from torchvision.transforms import v2
    import yaml
    from box import Box
    CLAY_AVAILABLE = True
except ImportError:
    CLAY_AVAILABLE = False

from .utils import (
    download_file,
)


# Default metadata for common sensors
DEFAULT_METADATA = {
    'sentinel-2-l2a': {
        'band_order': ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'swir16', 'swir22'],
        'rgb_indices': [2, 1, 0],
        'gsd': 10,
        'bands': {
            'mean': {'blue': 1105., 'green': 1355., 'red': 1552., 'rededge1': 1887., 'rededge2': 2422., 'rededge3': 2630., 'nir': 2743., 'nir08': 2785., 'swir16': 2388., 'swir22': 1835.},
            'std': {'blue': 1809., 'green': 1757., 'red': 1888., 'rededge1': 1870., 'rededge2': 1732., 'rededge3': 1697., 'nir': 1742., 'nir08': 1648., 'swir16': 1470., 'swir22': 1379.},
            'wavelength': {'blue': 0.493, 'green': 0.56, 'red': 0.665, 'rededge1': 0.704, 'rededge2': 0.74, 'rededge3': 0.783, 'nir': 0.842, 'nir08': 0.865, 'swir16': 1.61, 'swir22': 2.19}
        }
    },
    'landsat-c2l2-sr': {
        'band_order': ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22'],
        'rgb_indices': [0, 1, 2],
        'gsd': 30,
        'bands': {
            'mean': {'red': 13705., 'green': 13310., 'blue': 12474., 'nir08': 17801., 'swir16': 14615., 'swir22': 12701.},
            'std': {'red': 9578., 'green': 9408., 'blue': 10144., 'nir08': 8277., 'swir16': 5300., 'swir22': 4522.},
            'wavelength': {'red': 0.65, 'green': 0.56, 'blue': 0.48, 'nir08': 0.86, 'swir16': 1.6, 'swir22': 2.2}
        }
    },
    'naip': {
        'band_order': ['red', 'green', 'blue', 'nir'],
        'rgb_indices': [0, 1, 2],
        'gsd': 1.0,
        'bands': {
            'mean': {'red': 110.16, 'green': 115.41, 'blue': 98.15, 'nir': 139.04},
            'std': {'red': 47.23, 'green': 39.82, 'blue': 35.43, 'nir': 49.86},
            'wavelength': {'red': 0.65, 'green': 0.56, 'blue': 0.48, 'nir': 0.842}
        }
    }
}



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


class Clay:
    """
    Clay foundation model wrapper for generating geospatial embeddings.
    
    This class provides an interface to generate rich spectral embeddings from
    geospatial imagery using the Clay foundation model.
    """
    
    
    
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = 'large',
        metadata_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize Clay embeddings model.
        
        Args:
            checkpoint_path: Path to Clay model checkpoint
            metadata_path: Path to Clay metadata YAML file (optional)
            device: Device to run model on ('auto', 'cuda', 'cpu')
            mask_ratio: Masking ratio for model (0.0 for inference)
            shuffle: Whether to shuffle patches
        """
        if not CLAY_AVAILABLE:
            raise ImportError(
                "Clay model dependencies not available. "
                "Please install: pip install claymodel torch torchvision pyyaml python-box"
            )
        
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load metadata
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = Box(yaml.safe_load(f))
        else:
            self.metadata = Box(self.DEFAULT_METADATA)
            if metadata_path:
                warnings.warn(f"Metadata file not found: {metadata_path}. Using defaults.")
        
        
        self.model_size = model_size
        if self.model_size not in ['tiny','small','base','large']:
            raise ValueError(f"model_size must be one of: {['tiny','small','base','large']}")

        

        # Load model
        self._load_model()
        
        # Image processing attributes
        self.image = None
        self.source = None
        self.sensor_type = None
        self.raster_profile = None
        
    def _load_model(self):
        """Load the Clay model from checkpoint."""
        try:
            torch.set_default_device(self.device)
            self.module = ClayMAEModule.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model_size=self.model_size,
                dolls=[16, 32, 64, 128, 256, 768, 1024],
                doll_weights=[1, 1, 1, 1, 1, 1, 1],
                mask_ratio=0.0,
                shuffle=False,
            )
            self.module.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Clay model: {e}")
    
    def _detect_sensor_type(
        self, 
        src: rasterio.DatasetReader, 
        source_path: Optional[str] = None
    ) -> str:
        """
        Detect sensor type from raster metadata and characteristics.
        
        Args:
            src: Rasterio dataset reader
            source_path: Optional source file path for filename-based detection
            
        Returns:
            Detected sensor type string
        """
        band_count = src.count
        resolution = abs(src.transform[0])  # Pixel size
        
        # Try filename-based detection first
        if source_path:
            filename = os.path.basename(source_path).lower()
            if 'sentinel' in filename or 's2' in filename:
                return 'sentinel-2-l2a'
            elif 'landsat' in filename or 'l8' in filename or 'l9' in filename:
                return 'landsat-c2l2-sr'
            elif 'naip' in filename:
                return 'naip'
        
        # Fallback to resolution and band count heuristics
        if band_count == 4 and resolution <= 5:
            return 'naip'  # High-res 4-band imagery
        elif band_count >= 6 and 25 <= resolution <= 35:
            return 'landsat-c2l2-sr'  # Landsat resolution
        elif band_count >= 10 and 8 <= resolution <= 12:
            return 'sentinel-2-l2a'  # Sentinel-2 resolution
        elif band_count == 4:
            return 'naip'  # Default 4-band to NAIP
        else:
            # Default fallback
            warnings.warn(
                f"Could not detect sensor type (bands: {band_count}, "
                f"resolution: {resolution:.1f}m). Defaulting to NAIP."
            )
            return 'naip'
    
    def _get_raster_bounds_wgs84(self, src: rasterio.DatasetReader) -> Tuple[float, float, float, float]:
        """Get the WGS84 bounds of the raster."""
        bounds = src.bounds
        
        # Transform to WGS84 if needed
        if src.crs != 'EPSG:4326':
            from rasterio.warp import transform as transform_coords
            # Transform all four corners
            xs = [bounds.left, bounds.right, bounds.left, bounds.right]
            ys = [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
            transformed_xs, transformed_ys = transform_coords(src.crs, 'EPSG:4326', xs, ys)
            
            min_lon, max_lon = min(transformed_xs), max(transformed_xs)
            min_lat, max_lat = min(transformed_ys), max(transformed_ys)
            
            return min_lon, min_lat, max_lon, max_lat
        else:
            return bounds.left, bounds.bottom, bounds.right, bounds.top
    
    def _prepare_datacube(
        self, 
        image: np.ndarray, 
        sensor_type: str,
        bounds: Tuple[float, float, float, float], 
        date: Optional[datetime.datetime] = None,
        gsd_override: Optional[float] = None,
        add_batch_dim: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare datacube for Clay model input.
        
        Args:
            image: Input image array [H, W, C]
            sensor_type: Detected sensor type
            bounds: Image bounds as (min_lon, min_lat, max_lon, max_lat) in WGS84
            date: Image acquisition date
            gsd_override: Override GSD value
            add_batch_dim: Whether to add batch dimension
            
        Returns:
            Datacube dictionary for Clay model
        """
        if date is None:
            date = datetime.datetime.now()
        
        # Get sensor metadata
        sensor_meta = self.metadata[sensor_type]
        band_order = sensor_meta.band_order
        gsd = gsd_override if gsd_override is not None else sensor_meta.gsd
        
        # Extract normalization parameters
        means = [sensor_meta.bands.mean[band] for band in band_order]
        stds = [sensor_meta.bands.std[band] for band in band_order]
        wavelengths = [sensor_meta.bands.wavelength[band] for band in band_order]
        
        # Convert image to torch tensor and normalize
        # Ensure we have the right number of bands
        if image.shape[2] != len(band_order):
            warnings.warn(
                f"Image has {image.shape[2]} bands but sensor {sensor_type} "
                f"expects {len(band_order)} bands. Using available bands."
            )
            # Take only the available bands
            num_bands = min(image.shape[2], len(band_order))
            image = image[:, :, :num_bands]
            means = means[:num_bands]
            stds = stds[:num_bands]
            wavelengths = wavelengths[:num_bands]
        
        # Convert to tensor and transpose to [C, H, W]
        pixels = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        
        # Normalize
        transform = v2.Compose([v2.Normalize(mean=means, std=stds)])
        pixels = transform(pixels)
        if add_batch_dim:
            pixels = pixels.unsqueeze(0)  # Add batch dimension
        
        # Prepare temporal encoding
        time_norm = normalize_timestamp(date)
        
        # Prepare spatial encoding
        lat_norm, lon_norm = normalize_latlon(bounds)
        
        # Create temporal and spatial tensors
        time_tensor = torch.tensor(
            time_norm + time_norm,  # Clay expects 4 elements: [week, hour, week, hour]
            dtype=torch.float32,
            device=self.device
        )
        latlon_tensor = torch.tensor(
            lat_norm + lon_norm,  # Clay expects 4 elements: [sin_lat, cos_lat, sin_lon, cos_lon]
            dtype=torch.float32,
            device=self.device
        )
        
        if add_batch_dim:
            time_tensor = time_tensor.unsqueeze(0)
            latlon_tensor = latlon_tensor.unsqueeze(0)
        
        # Create datacube
        datacube = {
            'pixels': pixels.to(self.device),
            'time': time_tensor,
            'latlon': latlon_tensor,
            'gsd': torch.tensor(gsd, device=self.device),
            'waves': torch.tensor(wavelengths, device=self.device)
        }
        
        return datacube
    
    def set_image(
        self, 
        source: Union[str, np.ndarray],
        sensor_type: Optional[str] = None,
        date: Optional[Union[str, datetime.datetime]] = None,
        gsd_override: Optional[float] = None
    ):
        """
        Set the input image for embedding generation.
        
        Args:
            source: Path to image file or numpy array
            sensor_type: Optional sensor type override
            date: Image acquisition date
            gsd_override: Override GSD value
        """
        if isinstance(source, str):
            if source.startswith("http"):
                source = download_file(source)
            
            if not os.path.exists(source):
                raise ValueError(f"Input path {source} does not exist.")
            
            # Read with rasterio for geospatial images
            try:
                with rasterio.open(source) as src:
                    # Read all bands
                    image = src.read()  # Shape: [C, H, W]
                    image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]
                    
                    # Store raster metadata
                    self.raster_profile = src.profile
                    
                    # Detect sensor type
                    if sensor_type is None:
                        sensor_type = self._detect_sensor_type(src, source)
                    
                    # Get image bounds
                    bounds = self._get_raster_bounds_wgs84(src)
                    
            except Exception:
                # Fallback to OpenCV for regular images
                image = cv2.imread(source)
                if image is None:
                    raise ValueError(f"Could not read image: {source}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Use defaults for non-geospatial images
                sensor_type = sensor_type or 'naip'
                bounds = (0.0, 0.0, 0.01, 0.01)  # Default small bounds
                self.raster_profile = None
                
        elif isinstance(source, np.ndarray):
            image = source
            sensor_type = sensor_type or 'naip'
            bounds = (0.0, 0.0, 0.01, 0.01)  # Default small bounds
            self.raster_profile = None
            
        else:
            raise ValueError("Source must be a file path or numpy array")
        
        # Parse date if string
        if isinstance(date, str):
            try:
                date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00'))
            except ValueError:
                date = datetime.datetime.now()
                warnings.warn(f"Could not parse date: {date}. Using current time.")
        elif date is None:
            date = datetime.datetime.now()
        
        # Store image and metadata
        self.source = source if isinstance(source, str) else None
        self.image = image
        self.sensor_type = sensor_type
        self.bounds = bounds
        self.date = date
        self.gsd_override = gsd_override
        
        # Calculate center for display
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        print(f"Set image: shape={image.shape}, sensor={sensor_type}, "
              f"bounds={bounds}, center_lat={center_lat:.4f}, center_lon={center_lon:.4f}")
    
    def predict(
        self, 
        tile_size: int = 256,
        overlap: float = 0.0,
        only_cls_token: bool = False
    ) -> Dict[str, Any]:
        """
        Generate embeddings for the loaded image.
        
        Args:
            tile_size: Size of tiles for processing large images
            overlap: Overlap fraction between tiles (0.0 to 1.0)
            only_cls_token: If True, return only CLS token embeddings (first token)
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        if self.image is None:
            raise ValueError("No image loaded. Call set_image() first.")
        
        image = self.image
        h, w = image.shape[:2]
        
        # If image is smaller than tile_size, process as single tile
        if h <= tile_size and w <= tile_size:
            # Pad image to tile_size if needed
            if h < tile_size or w < tile_size:
                pad_h = max(0, tile_size - h)
                pad_w = max(0, tile_size - w)
                image = np.pad(
                    image, 
                    ((0, pad_h), (0, pad_w), (0, 0)), 
                    mode='reflect'
                )
            
            # Generate single embedding
            datacube = self._prepare_datacube(
                image, self.sensor_type, self.bounds, 
                self.date, self.gsd_override, add_batch_dim=True
            )
            
            with torch.no_grad():
                encoded_patches, _, _, _ = self.module.model.encoder(datacube)
                if only_cls_token:
                    # Extract only class token (global embedding)
                    embedding = encoded_patches[:, 0, :].cpu().numpy()
                else:
                    # Return full sequence
                    embedding = encoded_patches.cpu().numpy()
            
            return {
                'embeddings': embedding,
                'tile_coords': [(0, 0, h, w)],
                'image_shape': (h, w),
                'sensor_type': self.sensor_type,
                'bounds': self.bounds,
                'date': self.date.isoformat() if self.date else None,
                'num_tiles': 1
            }
        
        else:
            # Process as overlapping tiles
            step_size = int(tile_size * (1 - overlap))
            embeddings = []
            tile_coords = []
            
            for y in range(0, h - tile_size + 1, step_size):
                for x in range(0, w - tile_size + 1, step_size):
                    # Extract tile
                    tile = image[y:y+tile_size, x:x+tile_size]
                    
                    # Prepare datacube for this tile
                    datacube = self._prepare_datacube(
                        tile, self.sensor_type, self.bounds,
                        self.date, self.gsd_override, add_batch_dim=True
                    )
                    
                    # Generate embedding
                    with torch.no_grad():
                        encoded_patches, _, _, _ = self.module.model.encoder(datacube)
                        if only_cls_token:
                            # Extract only class token (global embedding)
                            embedding = encoded_patches[:, 0, :].cpu().numpy()
                        else:
                            # Return full sequence
                            embedding = encoded_patches.cpu().numpy()
                    
                    embeddings.append(embedding)
                    tile_coords.append((x, y, x+tile_size, y+tile_size))
            
            return {
                'embeddings': np.vstack(embeddings),
                'tile_coords': tile_coords,
                'image_shape': (h, w),
                'sensor_type': self.sensor_type,
                'bounds': self.bounds,
                'date': self.date.isoformat() if self.date else None,
                'num_tiles': len(embeddings)
            }
    
    def save_embeddings(
        self, 
        embeddings_result: Dict[str, Any], 
        output_path: str,
        format: str = 'npz'
    ):
        """
        Save embeddings to file.
        
        Args:
            embeddings_result: Result from generate_embeddings()
            output_path: Output file path
            format: Output format ('npz', 'pt')
        """
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'npz':
            np.savez_compressed(
                output_path,
                embeddings=embeddings_result['embeddings'],
                tile_coords=np.array(embeddings_result['tile_coords']),
                image_shape=np.array(embeddings_result['image_shape']),
                sensor_type=embeddings_result['sensor_type'],
                bounds=np.array(embeddings_result['bounds']),
                date=embeddings_result['date'],
                num_tiles=embeddings_result['num_tiles']
            )
        elif format == 'pt':
            torch.save(embeddings_result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved embeddings to {output_path}")

    def _prepare_batch_datacube(
        self, 
        images: list, 
        sensor_types: list,
        bounds_list: list, 
        dates: list,
        gsd_overrides: list
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch datacube for Clay model input.
        
        Args:
            images: List of image arrays [H, W, C]
            sensor_types: List of sensor types
            bounds_list: List of image bounds
            dates: List of acquisition dates
            gsd_overrides: List of GSD overrides
            
        Returns:
            Batched datacube dictionary for Clay model
        """
        batch_pixels = []
        batch_times = []
        batch_latlons = []
        batch_gsds = []
        batch_waves = []
        
        for image, sensor_type, bounds, date, gsd_override in zip(
            images, sensor_types, bounds_list, dates, gsd_overrides
        ):
            datacube = self._prepare_datacube(
                image, sensor_type, bounds, date, gsd_override, add_batch_dim=False
            )
            
            batch_pixels.append(datacube['pixels'])
            batch_times.append(datacube['time'])
            batch_latlons.append(datacube['latlon'])
            batch_gsds.append(datacube['gsd'])
            batch_waves.append(datacube['waves'])
        
        # Stack tensors to create batch
        return {
            'pixels': torch.stack(batch_pixels),
            'time': torch.stack(batch_times),
            'latlon': torch.stack(batch_latlons),
            'gsd': torch.stack(batch_gsds),
            'waves': torch.stack(batch_waves)
        }
    
    def _process_batch_images(
        self,
        sources: list,
        sensor_types: list,
        dates: list,
        gsd_overrides: list
    ) -> Tuple[list, list, list, list, list]:
        """
        Process a batch of images and extract metadata.
        
        Returns:
            Tuple of (images, sensor_types, bounds_list, dates, gsd_overrides)
        """
        images = []
        processed_sensor_types = []
        bounds_list = []
        processed_dates = []
        processed_gsd_overrides = []
        
        for source, sensor_type, date, gsd_override in zip(
            sources, sensor_types, dates, gsd_overrides
        ):
            # Process each image similar to set_image but without storing state
            if isinstance(source, str):
                if source.startswith("http"):
                    source = download_file(source)
                
                if not os.path.exists(source):
                    raise ValueError(f"Input path {source} does not exist.")
                
                # Read with rasterio for geospatial images
                try:
                    with rasterio.open(source) as src:
                        # Read all bands
                        image = src.read()  # Shape: [C, H, W]
                        image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]
                        
                        # Detect sensor type
                        if sensor_type is None:
                            sensor_type = self._detect_sensor_type(src, source)
                        
                        # Get image bounds
                        bounds = self._get_raster_bounds_wgs84(src)
                        
                except Exception:
                    # Fallback to OpenCV for regular images
                    image = cv2.imread(source)
                    if image is None:
                        raise ValueError(f"Could not read image: {source}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Use defaults for non-geospatial images
                    sensor_type = sensor_type or 'naip'
                    bounds = (0.0, 0.0, 0.01, 0.01)  # Default small bounds
                    
            elif isinstance(source, np.ndarray):
                image = source
                sensor_type = sensor_type or 'naip'
                bounds = (0.0, 0.0, 0.01, 0.01)  # Default small bounds
                
            else:
                raise ValueError("Source must be a file path or numpy array")
            
            # Parse date if string
            if isinstance(date, str):
                try:
                    date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00'))
                except ValueError:
                    date = datetime.datetime.now()
                    warnings.warn(f"Could not parse date: {date}. Using current time.")
            elif date is None:
                date = datetime.datetime.now()
            
            images.append(image)
            processed_sensor_types.append(sensor_type)
            bounds_list.append(bounds)
            processed_dates.append(date)
            processed_gsd_overrides.append(gsd_override)
        
        return images, processed_sensor_types, bounds_list, processed_dates, processed_gsd_overrides
    
    def _needs_tiling(self, source: Union[str, np.ndarray], tile_size: int) -> bool:
        """Check if an image needs tiling based on its size."""
        try:
            if isinstance(source, np.ndarray):
                h, w = source.shape[:2]
                return h > tile_size or w > tile_size
            
            with rasterio.open(source) as src:
                return src.height > tile_size or src.width > tile_size
        except Exception:
            # If we can't determine size, assume it might need tiling
            return True
    
    def _generate_sequential(
        self,
        sources: list,
        output_dir: Optional[str],
        sensor_types: list,
        dates: list,
        gsd_overrides: list,
        tile_size: int,
        overlap: float,
        save: bool,
        format: str,
        only_cls_token: bool
    ) -> Union[Dict[str, Any], list]:
        """Fall back to sequential processing for complex cases."""
        results = []
        for i, (source, sensor_type, date, gsd_override) in enumerate(
            zip(sources, sensor_types, dates, gsd_overrides)
        ):
            # Set image
            self.set_image(source, sensor_type, date, gsd_override)
            
            # Generate embeddings
            result = self.predict(tile_size, overlap, only_cls_token)
            
            # Save if requested
            if save:
                if isinstance(source, str):
                    base_name = os.path.splitext(os.path.basename(source))[0]
                else:
                    base_name = f"image_{i:04d}"
                
                output_path = os.path.join(output_dir, f"{base_name}_embeddings.{format}")
                self.save_embeddings(result, output_path, format)
            
            results.append(result)
        
        # Return single result or list
        if len(results) == 1:
            return results[0]
        return results

    def generate(
        self,
        sources: Union[str, list],
        output_dir: Optional[str] = None,
        sensor_types: Optional[Union[str, list]] = None,
        dates: Optional[Union[str, datetime.datetime, list]] = None,
        gsd_overrides: Optional[Union[float, list]] = None,
        tile_size: int = 256,
        overlap: float = 0.0,
        save: bool = True,
        format: str = 'npz',
        only_cls_token: bool = False,
        batch_size: int = 8
    ) -> Union[Dict[str, Any], list]:
        """
        Generate embeddings for one or multiple images with true batch processing.
        
        Args:
            sources: Single image path/array or list of image paths/arrays
            output_dir: Directory to save embeddings (required if save=True)
            sensor_types: Single sensor type or list of sensor types
            dates: Single date or list of dates
            gsd_overrides: Single GSD override or list of GSD overrides
            tile_size: Size of tiles for processing large images
            overlap: Overlap fraction between tiles (0.0 to 1.0)
            save: Whether to save embeddings to disk
            format: Output format ('npz', 'pt')
            only_cls_token: If True, return only CLS token embeddings
            batch_size: Number of images to process simultaneously in model
            
        Returns:
            Single embeddings dict or list of embeddings dicts
        """
        # Normalize inputs to lists
        if not isinstance(sources, list):
            sources = [sources]
        
        if sensor_types is not None and not isinstance(sensor_types, list):
            sensor_types = [sensor_types] * len(sources)
        elif sensor_types is None:
            sensor_types = [None] * len(sources)
        
        if dates is not None and not isinstance(dates, list):
            dates = [dates] * len(sources)
        elif dates is None:
            dates = [None] * len(sources)
            
        if gsd_overrides is not None and not isinstance(gsd_overrides, list):
            gsd_overrides = [gsd_overrides] * len(sources)
        elif gsd_overrides is None:
            gsd_overrides = [None] * len(sources)
        
        # Validate parameters
        if save and output_dir is None:
            raise ValueError("output_dir is required when save=True")
        
        if save:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        # For single image or when any image needs tiling, fall back to sequential processing
        if len(sources) == 1 or any(self._needs_tiling(src, tile_size) for src in sources):
            return self._generate_sequential(
                sources, output_dir, sensor_types, dates, gsd_overrides,
                tile_size, overlap, save, format, only_cls_token
            )
        
        # Process in batches for true model-level batching
        all_results = []
        
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            batch_sensor_types = sensor_types[i:i+batch_size]
            batch_dates = dates[i:i+batch_size]
            batch_gsd_overrides = gsd_overrides[i:i+batch_size]
            
            # Process batch images
            images, proc_sensor_types, bounds_list, proc_dates, proc_gsd_overrides = self._process_batch_images(
                batch_sources, batch_sensor_types, batch_dates, batch_gsd_overrides
            )
            
            # Prepare batch datacube
            batch_datacube = self._prepare_batch_datacube(
                images, proc_sensor_types, bounds_list, proc_dates, proc_gsd_overrides
            )
            
            # Generate batch embeddings
            with torch.no_grad():
                encoded_patches, _, _, _ = self.module.model.encoder(batch_datacube)
                if only_cls_token:
                    # Extract only class tokens (global embeddings)
                    batch_embeddings = encoded_patches[:, 0, :].cpu().numpy()
                else:
                    # Return full sequences
                    batch_embeddings = encoded_patches.cpu().numpy()
            
            # Create individual results
            for j, (source, image, sensor_type, bounds, date) in enumerate(
                zip(batch_sources, images, proc_sensor_types, bounds_list, proc_dates)
            ):
                h, w = image.shape[:2]
                embedding = batch_embeddings[j:j+1]  # Keep batch dimension for consistency
                
                result = {
                    'embeddings': embedding,
                    'tile_coords': [(0, 0, h, w)],
                    'image_shape': (h, w),
                    'sensor_type': sensor_type,
                    'bounds': bounds,
                    'date': date.isoformat() if date else None,
                    'num_tiles': 1
                }
                
                # Save if requested
                if save:
                    if isinstance(source, str):
                        base_name = os.path.splitext(os.path.basename(source))[0]
                    else:
                        base_name = f"image_{i+j:04d}"
                    
                    output_path = os.path.join(output_dir, f"{base_name}_embeddings.{format}")
                    self.save_embeddings(result, output_path, format)
                
                all_results.append(result)
        
        # Return single result or list
        if len(all_results) == 1:
            return all_results[0]
        return all_results

    def generate_from_dir(
        self,
        image_dir: str,
        output_dir: str,
        metadata_dir: Optional[str] = None,
        batch_size: int = 1,
        tile_size: int = 256,
        overlap: float = 0.0,
        format: str = 'npz',
        only_cls_token: bool = False,
        sink_callback: Optional[callable] = None
    ):
        """
        Generate embeddings for all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save embeddings
            metadata_dir: Directory containing metadata files (optional)
            batch_size: Number of images to process at once
            tile_size: Size of tiles for processing large images
            overlap: Overlap fraction between tiles (0.0 to 1.0)
            format: Output format ('npz', 'pt')
            only_cls_token: If True, return only CLS token embeddings (first token)
            sink_callback: Optional callable(filename, embeddings) called after each batch
        """
        image_dir = os.path.abspath(image_dir)
        output_dir = os.path.abspath(output_dir)
        
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_sensor_types = []
            batch_dates = []
            batch_gsd_overrides = []
            
            # Load metadata for batch if metadata_dir provided
            for img_file in batch_files:
                sensor_type = None
                date = None
                gsd_override = None
                
                if metadata_dir:
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    metadata_file = os.path.join(metadata_dir, f"{base_name}.yaml")
                    
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = yaml.safe_load(f)
                                sensor_type = metadata.get('sensor_type')
                                date_str = metadata.get('date')
                                if date_str:
                                    try:
                                        date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                    except ValueError:
                                        pass
                                gsd_override = metadata.get('gsd')
                        except Exception as e:
                            print(f"Warning: Could not load metadata for {img_file}: {e}")
                
                batch_sensor_types.append(sensor_type)
                batch_dates.append(date)
                batch_gsd_overrides.append(gsd_override)
            
            # Process batch
            print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
            
            try:
                self.generate(
                    sources=batch_files,
                    output_dir=output_dir,
                    sensor_types=batch_sensor_types,
                    dates=batch_dates,
                    gsd_overrides=batch_gsd_overrides,
                    tile_size=tile_size,
                    overlap=overlap,
                    save=True,
                    format=format
                )
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Finished processing all images. Embeddings saved to {output_dir}")


def load_embeddings(file_path: str) -> Dict[str, Any]:
    """
    Load embeddings from file.
    
    Args:
        file_path: Path to embeddings file
        
    Returns:
        Embeddings dictionary
    """
    if file_path.endswith('.npz'):
        data = np.load(file_path, allow_pickle=True)
        # Handle both old format (lat/lon) and new format (bounds)
        if 'bounds' in data:
            return {
                'embeddings': data['embeddings'],
                'tile_coords': data['tile_coords'].tolist(),
                'image_shape': tuple(data['image_shape']),
                'sensor_type': str(data['sensor_type']),
                'bounds': tuple(data['bounds']),
                'date': str(data['date']) if data['date'] != 'None' else None,
                'num_tiles': int(data['num_tiles'])
            }
        else:
            # Legacy format with lat/lon
            return {
                'embeddings': data['embeddings'],
                'tile_coords': data['tile_coords'].tolist(),
                'image_shape': tuple(data['image_shape']),
                'sensor_type': str(data['sensor_type']),
                'lat': float(data['lat']),
                'lon': float(data['lon']),
                'date': str(data['date']) if data['date'] != 'None' else None,
                'num_tiles': int(data['num_tiles'])
            }
    elif file_path.endswith('.pt'):
        return torch.load(file_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")