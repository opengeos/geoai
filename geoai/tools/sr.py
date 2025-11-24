"""
Super-resolution utilities using OpenSR latent diffusion models.

This module provides functions to perform super-resolution on multispectral
GeoTIFF images using the latent diffusion models from the ESA OpenSR project:

    GitHub: https://github.com/ESAOpenSR/opensr-model.git
"""

import os
from typing import Tuple

import torch
import numpy as np
import rasterio
import requests
from rasterio.transform import Affine
from io import StringIO
from omegaconf import OmegaConf

try:
    import opensr_model

    OPENSR_MODEL_AVAILABLE = True
except ImportError:
    OPENSR_MODEL_AVAILABLE = False


def super_resolution(
    input_lr_path: str,
    output_sr_path: str,
    output_uncertainty_path: str,
    rgb_nir_bands: list[int] = [3, 2, 1, 4],  # Default example: R=3,G=2,B=1,NIR=4
    sampling_steps: int = 100,
    n_variations: int = 25,
    scale: int = 4,  # OpenSR scaling factor, e.g., 10m -> 2.5m
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform super-resolution on RGB+NIR bands of a multispectral GeoTIFF using OpenSR latent diffusion.

    Args:
        input_lr_path (str): Path to the input low-resolution GeoTIFF.
        output_sr_path (str): Path to save the super-resolution GeoTIFF.
        output_uncertainty_path (str): Path to save the uncertainty map GeoTIFF.
        rgb_nir_bands (list[int]): List of 4 band indices corresponding to [R, G, B, NIR].
        sampling_steps (int): Number of diffusion sampling steps. Default is 100.
        n_variations (int): Number of samples to compute uncertainty. Default is 25.
        scale (int, optional): Super-resolution scale factor. Default is 4.
            This adjusts the affine transform to ensure georeference matches
            the original image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - sr_image: Super-resolution image as a NumPy array (4, H, W)
            - uncertainty: Uncertainty map as a NumPy array (H, W)
    """
    if len(rgb_nir_bands) != 4:
        raise ValueError("rgb_nir_bands must be a list of 4 integers: [R, G, B, NIR]")
    if not all(isinstance(b, int) for b in rgb_nir_bands):
        raise ValueError(
            "All elements of rgb_nir_bands must be integers. Received: {}".format(
                rgb_nir_bands
            )
        )
    if not OPENSR_MODEL_AVAILABLE:
        raise ImportError(
            "The 'opensr-model' package is required for super-resolution. "
            "Please install it using: pip install opensr-model\n"
            "Or install GeoAI with the sr optional dependency: pip install geoai-py[sr]"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download configuration YAML from GitHub
    config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"
    print("Downloading model configuration from:", config_url)
    try:
        response = requests.get(config_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error downloading model configuration: {e}")
        raise
    config = OmegaConf.load(StringIO(response.text))

    # Initialize latent diffusion model and load pretrained weights
    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)

    # Load only the specified RGB+NIR bands
    lr_tensor, profile = load_image_tensor(
        image_path=input_lr_path, device=device, bands=rgb_nir_bands
    )

    # Generate super-resolution tensor
    sr_tensor = model.forward(lr_tensor, sampling_steps=sampling_steps)
    sr_image = sr_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    save_geotiff(sr_image, profile, output_sr_path, scale)
    print("Saved super-resolution image to:", output_sr_path)

    # Compute uncertainty map
    unc_tensor = model.uncertainty_map(lr_tensor, n_variations=n_variations)
    uncertainty = unc_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    save_geotiff(uncertainty, profile, output_uncertainty_path, scale)
    print("Saved uncertainty map to:", output_uncertainty_path)

    return sr_image, uncertainty


def save_geotiff(
    data: np.ndarray, reference_profile: dict, output_path: str, scale: int = 4
):
    """
    Save a 2D or 3D NumPy array as a GeoTIFF with super-resolution scaling
    and corrected georeference.

    Args:
        data (np.ndarray): Image array to save. Can be:
            - 2D array (H, W) for a single-band image
            - 3D array (C, H, W) for multi-band images (e.g., RGB+NIR)
        reference_profile (dict): Rasterio metadata from a reference GeoTIFF.
            Used to preserve CRS, transform, and other metadata.
        output_path (str): Path to save the output GeoTIFF.
        scale (int, optional): Super-resolution scale factor. Default is 4.
            This adjusts the affine transform to ensure georeference matches
            the original image.

    Returns:
        None

    Note:
        Writes the image to disk at the specified output path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    # Update profile and transform
    profile = reference_profile.copy()
    old_transform = profile["transform"]
    new_transform = Affine(
        old_transform.a / scale,
        old_transform.b,
        old_transform.c,
        old_transform.d,
        old_transform.e / scale,
        old_transform.f,
    )
    profile.update(
        dtype=rasterio.float32,
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        compress="lzw",
        transform=new_transform,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def load_image_tensor(
    image_path: str, device: str, bands: list[int]
) -> Tuple[torch.Tensor, dict]:
    """
    Load only specified bands of a multispectral GeoTIFF as a PyTorch tensor.

    Args:
        image_path (str): Path to input GeoTIFF.
        device (str): Device to move the tensor to ('cpu' or 'cuda').
        bands (list[int]): List of 1-based band indices to read.

    Returns:
        Tuple[torch.Tensor, dict]: Tensor (1, C, H, W) and rasterio profile.

    Raises:
        FileNotFoundError: If input image does not exist.
        ValueError: If any band index is out of range.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    with rasterio.open(image_path) as src:
        n_bands = src.count
        if min(bands) < 1 or max(bands) > n_bands:
            raise ValueError(
                f"Input image has {n_bands} bands, requested bands {bands} out of range."
            )
        image = src.read(bands)  # shape: (4, H, W)
        profile = src.profile

    image = image.astype(np.float32)
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    return tensor, profile
