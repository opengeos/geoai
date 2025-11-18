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
from io import StringIO
from omegaconf import OmegaConf
import opensr_model


def super_resolution(
    input_lr_path: str,
    output_sr_path: str,
    output_uncertainty_path: str,
    sampling_steps: int = 100,
    n_variations: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform super-resolution on a multispectral GeoTIFF using OpenSR latent diffusion.

    Args:
        input_lr_path (str): Path to the input low-resolution GeoTIFF.
        output_sr_path (str): Path to save the super-resolution GeoTIFF.
        output_uncertainty_path (str): Path to save the uncertainty map GeoTIFF.
        sampling_steps (int): Number of diffusion sampling steps. Default is 100.
        n_variations (int): Number of samples to compute uncertainty. Default is 25.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - sr_image: Super-resolution image as a NumPy array (C, H, W).
            - uncertainty: Uncertainty map as a NumPy array (H, W).
    """
    # Determine computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download configuration YAML from GitHub
    config_url = (
        "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/"
        "opensr_model/configs/config_10m.yaml"
    )
    print("Downloading model configuration from:", config_url)
    response = requests.get(config_url)
    config = OmegaConf.load(StringIO(response.text))

    # Initialize latent diffusion model and load pretrained weights
    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)

    # Load low-resolution image as tensor
    lr_tensor, profile = load_image_tensor(
        image_path=input_lr_path, device=device, max_bands=4
    )

    # Generate super-resolution tensor
    sr_tensor = model.forward(lr_tensor, sampling_steps=sampling_steps)

    # Convert tensor to NumPy array for saving
    sr_image = sr_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    save_geotiff(sr_image, profile, output_sr_path)
    print("Saved super-resolution image to:", output_sr_path)

    # Compute uncertainty map
    unc_tensor = model.uncertainty_map(lr_tensor, n_variations=n_variations)
    uncertainty = unc_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    save_geotiff(uncertainty, profile, output_uncertainty_path)
    print("Saved uncertainty map to:", output_uncertainty_path)

    return sr_image, uncertainty


def save_geotiff(data: np.ndarray, reference_profile: dict, output_path: str):
    """
    Save a 2D or 3D NumPy array as a GeoTIFF using a reference metadata profile.

    Args:
        data (np.ndarray): Image array to save. Shape can be (C, H, W) or (H, W).
        reference_profile (dict): Rasterio metadata from a reference image.
        output_path (str): Path to save the output GeoTIFF.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Add channel dimension if missing (for 2D arrays)
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    # Copy and update raster metadata
    profile = reference_profile.copy()
    profile.update(
        dtype=rasterio.float32,
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        compress="lzw",
    )

    # Write data to GeoTIFF
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def load_image_tensor(image_path: str, device: str, max_bands: int) -> Tuple[torch.Tensor, dict]:
    """
    Load a multispectral GeoTIFF as a PyTorch tensor with a batch dimension.

    Args:
        image_path (str): Path to input GeoTIFF.
        device (str): Device to move the tensor to ('cpu' or 'cuda').
        max_bands (int): Number of bands to read from the image.

    Returns:
        Tuple[torch.Tensor, dict]: Tuple containing:
            - Tensor with shape (1, C, H, W)
            - Rasterio profile dictionary

    Raises:
        FileNotFoundError: If input image does not exist.
        ValueError: If input image has fewer bands than max_bands.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    # Read image and profile using rasterio
    with rasterio.open(image_path) as src:
        image = src.read()  # shape: (C, H, W)
        profile = src.profile

    # Check band count
    if image.shape[0] < max_bands:
        raise ValueError(
            f"Input image has {image.shape[0]} bands, but {max_bands} required."
        )

    # Select first max_bands bands and convert to float32
    image = image[:max_bands].astype(np.float32)

    # Convert to PyTorch tensor and add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    return tensor, profile
