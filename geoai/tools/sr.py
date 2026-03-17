"""
Super-resolution utilities using OpenSR latent diffusion models.

This module provides functions to perform super-resolution on multispectral
GeoTIFF images using the latent diffusion models from the ESA OpenSR project:

    GitHub: https://github.com/ESAOpenSR/opensr-model.git
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import requests
import torch
from io import StringIO
from omegaconf import OmegaConf
from rasterio.transform import Affine
from rasterio.windows import Window

try:
    import opensr_model

    OPENSR_MODEL_AVAILABLE = True
except ImportError:
    OPENSR_MODEL_AVAILABLE = False


_CKPT_MIN_SIZE = 1_000_000  # 1 MB; real checkpoint is ~1.1 GB


def _get_cached_checkpoint(ckpt_name: str) -> str:
    """Return the path to a cached OpenSR checkpoint, downloading it if
    necessary.

    The checkpoint is stored in the standard torch hub cache directory
    (``~/.cache/torch/hub/checkpoints/`` by default). If the file exists
    but is smaller than ``_CKPT_MIN_SIZE`` bytes (e.g. a truncated
    download), it is deleted and re-downloaded.

    Args:
        ckpt_name (str): Plain filename of the checkpoint
            (e.g. ``"opensr-ldsrs2_v1_0_0.ckpt"``).

    Returns:
        str: Absolute path to the cached checkpoint file.
    """
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    ckpt_path = os.path.join(cache_dir, ckpt_name)

    needs_download = not os.path.isfile(ckpt_path) or (
        os.path.getsize(ckpt_path) < _CKPT_MIN_SIZE
    )
    if needs_download:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        hf_url = (
            "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/" + ckpt_name
        )
        print("Downloading pretrained weights to:", ckpt_path)
        torch.hub.download_url_to_file(hf_url, ckpt_path)
    return ckpt_path


def super_resolution(
    input_lr_path: str,
    output_sr_path: str,
    output_uncertainty_path: Optional[str] = None,
    rgb_nir_bands: list[int] = [1, 2, 3, 4],
    sampling_steps: int = 100,
    n_variations: int = 25,
    scale: int = 4,
    compute_uncertainty: bool = False,
    window: Optional[tuple] = None,
    scale_factor: float = 10000.0,
    patch_size: int = 128,
    overlap: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Perform super-resolution on RGB+NIR bands of a multispectral GeoTIFF
    using OpenSR latent diffusion.

    The model enhances Sentinel-2 imagery from 10m to 2.5m spatial resolution
    (4x upsampling) using the LDSR-S2 latent diffusion model from the ESA
    OpenSR project. For images larger than ``patch_size``, the input is
    automatically tiled into overlapping patches, each patch is super-resolved,
    and the results are stitched back together with linear blending.

    Args:
        input_lr_path (str): Path to the input low-resolution GeoTIFF.
        output_sr_path (str): Path to save the super-resolution GeoTIFF.
        output_uncertainty_path (str, optional): Path to save the uncertainty
            map GeoTIFF. Required when ``compute_uncertainty`` is True.
        rgb_nir_bands (list[int]): List of 4 one-based band indices
            corresponding to [R, G, B, NIR] in the input file. Default is
            ``[1, 2, 3, 4]``.
        sampling_steps (int): Number of diffusion sampling steps. Higher
            values produce better results but are slower. Default is 100.
        n_variations (int): Number of stochastic forward passes used to
            estimate uncertainty. Default is 25.
        scale (int): Super-resolution scale factor. Default is 4.
        compute_uncertainty (bool): Whether to compute an uncertainty map
            via multiple stochastic forward passes. Default is False.
        window (tuple, optional): Region of interest as
            ``(row_off, col_off, height, width)`` to read a subset of the
            input image. If None, the entire image is processed.
        scale_factor (float): Divisor to normalize pixel values to the
            [0, 1] range. For Sentinel-2 L2A BOA reflectance, use 10000.0
            (the default). Set to 1.0 if the data is already normalized.
        patch_size (int): Tile size for patched inference. The model expects
            128x128 input patches. Default is 128.
        overlap (int): Number of overlapping pixels between adjacent patches
            to reduce tiling artifacts. Default is 16.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Tuple containing:
            - sr_image: Super-resolution image as a NumPy array (4, H, W).
            - uncertainty: Uncertainty map as a NumPy array (H, W), or None
              if ``compute_uncertainty`` is False.

    Raises:
        ValueError: If ``rgb_nir_bands`` does not contain exactly 4 integers,
            or if ``compute_uncertainty`` is True but no output path is given.
        ImportError: If the ``opensr-model`` package is not installed.

    Example:
        >>> import geoai
        >>> sr_image, uncertainty = geoai.super_resolution(
        ...     input_lr_path="sentinel2.tif",
        ...     output_sr_path="sr_output.tif",
        ...     rgb_nir_bands=[1, 2, 3, 4],
        ...     window=(500, 500, 128, 128),
        ...     sampling_steps=50,
        ... )
    """
    if len(rgb_nir_bands) != 4:
        raise ValueError("rgb_nir_bands must be a list of 4 integers: [R, G, B, NIR]")
    if not all(isinstance(b, int) for b in rgb_nir_bands):
        raise ValueError(
            "All elements of rgb_nir_bands must be integers. Received: {}".format(
                rgb_nir_bands
            )
        )
    if output_uncertainty_path is not None and not compute_uncertainty:
        compute_uncertainty = True
    if compute_uncertainty and output_uncertainty_path is None:
        raise ValueError(
            "output_uncertainty_path must be provided when compute_uncertainty is True."
        )
    if compute_uncertainty and n_variations <= 3:
        raise ValueError(
            "n_variations must be greater than 3 to compute uncertainty. "
            f"Received: {n_variations}"
        )
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive. Received: {scale_factor}")
    if patch_size <= 0 or overlap < 0 or overlap >= patch_size:
        raise ValueError(
            f"Requires patch_size > 0 and 0 <= overlap < patch_size. "
            f"Received patch_size={patch_size}, overlap={overlap}."
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

    # Initialize latent diffusion model and load pretrained weights.
    # Download checkpoint to the torch hub cache directory instead of cwd.
    model = opensr_model.SRLatentDiffusion(config, device=device)
    ckpt_name = os.path.basename(config.ckpt_version)
    if not ckpt_name or ckpt_name != config.ckpt_version:
        raise ValueError(
            f"Invalid checkpoint name in config: {config.ckpt_version!r}. "
            "Expected a plain filename without path separators."
        )
    ckpt_path = _get_cached_checkpoint(ckpt_name)
    model.load_pretrained(ckpt_path)

    # Load only the specified RGB+NIR bands
    lr_tensor, profile = load_image_tensor(
        image_path=input_lr_path,
        device=device,
        bands=rgb_nir_bands,
        window=window,
        scale_factor=scale_factor,
    )

    # Determine whether patched inference is needed
    _, _, h, w = lr_tensor.shape
    if h > patch_size or w > patch_size:
        sr_image = _process_patched(
            model=model,
            lr_tensor=lr_tensor,
            patch_size=patch_size,
            overlap=overlap,
            scale=scale,
            sampling_steps=sampling_steps,
        )
    else:
        sr_tensor = model.forward(lr_tensor, sampling_steps=sampling_steps)
        sr_image = sr_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    save_geotiff(sr_image, profile, output_sr_path, scale)
    print("Saved super-resolution image to:", output_sr_path)

    # Compute uncertainty map if requested
    uncertainty = None
    if compute_uncertainty:
        if h > patch_size or w > patch_size:
            uncertainty = _process_patched_uncertainty(
                model=model,
                lr_tensor=lr_tensor,
                patch_size=patch_size,
                overlap=overlap,
                scale=scale,
                n_variations=n_variations,
                sampling_steps=sampling_steps,
            )
        else:
            unc_tensor = model.uncertainty_map(
                lr_tensor,
                n_variations=n_variations,
                sampling_steps=sampling_steps,
            )
            uncertainty = unc_tensor.squeeze().cpu().numpy().astype(np.float32)
        save_geotiff(uncertainty, profile, output_uncertainty_path, scale)
        print("Saved uncertainty map to:", output_uncertainty_path)

    return sr_image, uncertainty


def save_geotiff(
    data: np.ndarray, reference_profile: dict, output_path: str, scale: int = 4
):
    """Save a 2D or 3D NumPy array as a GeoTIFF with super-resolution scaling
    and corrected georeference.

    Args:
        data (np.ndarray): Image array to save. Can be:
            - 2D array (H, W) for a single-band image
            - 3D array (C, H, W) for multi-band images (e.g., RGB+NIR)
        reference_profile (dict): Rasterio metadata from a reference GeoTIFF.
            Used to preserve CRS, transform, and other metadata.
        output_path (str): Path to save the output GeoTIFF.
        scale (int): Super-resolution scale factor. Default is 4. This adjusts
            the affine transform to ensure georeference matches the original
            image.
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
    image_path: str,
    device: str,
    bands: list[int],
    window: Optional[tuple] = None,
    scale_factor: float = 10000.0,
) -> Tuple[torch.Tensor, dict]:
    """Load specified bands of a multispectral GeoTIFF as a PyTorch tensor.

    The pixel values are divided by ``scale_factor`` to normalize them to the
    [0, 1] range expected by the OpenSR model.

    Args:
        image_path (str): Path to input GeoTIFF.
        device (str): Device to move the tensor to ('cpu' or 'cuda').
        bands (list[int]): List of 1-based band indices to read.
        window (tuple, optional): Region of interest as
            ``(row_off, col_off, height, width)``. If None, the full image
            is read.
        scale_factor (float): Divisor to normalize pixel values to [0, 1].
            Default is 10000.0 for Sentinel-2 L2A BOA reflectance.

    Returns:
        Tuple[torch.Tensor, dict]: Tensor of shape (1, C, H, W) and rasterio
        profile adjusted for the window (if provided).

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

        rio_window = None
        if window is not None:
            if len(window) != 4:
                raise ValueError(
                    f"window must be a 4-tuple (row_off, col_off, height, width). "
                    f"Received {len(window)} elements."
                )
            row_off, col_off, win_h, win_w = window
            if row_off < 0 or col_off < 0 or win_h <= 0 or win_w <= 0:
                raise ValueError(
                    f"Window offsets must be >= 0 and dimensions must be > 0. "
                    f"Received row_off={row_off}, col_off={col_off}, "
                    f"height={win_h}, width={win_w}."
                )
            if row_off + win_h > src.height or col_off + win_w > src.width:
                raise ValueError(
                    f"Window (row_off={row_off}, col_off={col_off}, "
                    f"height={win_h}, width={win_w}) exceeds image "
                    f"dimensions ({src.height} x {src.width})."
                )
            rio_window = Window(col_off, row_off, win_w, win_h)

        image = src.read(bands, window=rio_window)  # shape: (C, H, W)
        profile = src.profile.copy()

        if rio_window is not None:
            profile["transform"] = src.window_transform(rio_window)
            profile["height"] = image.shape[1]
            profile["width"] = image.shape[2]

    image = image.astype(np.float32) / scale_factor
    image = np.nan_to_num(image, nan=0.0)
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    return tensor, profile


def _create_patches(
    tensor: torch.Tensor, patch_size: int, overlap: int
) -> List[Tuple[torch.Tensor, int, int]]:
    """Tile a tensor into overlapping patches.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, C, H, W).
        patch_size (int): Size of each square patch.
        overlap (int): Number of overlapping pixels between adjacent patches.

    Returns:
        List[Tuple[torch.Tensor, int, int]]: List of (patch, row_start,
        col_start) tuples.
    """
    _, _, h, w = tensor.shape
    stride = patch_size - overlap
    patches = []

    row_starts = list(range(0, h - patch_size + 1, stride))
    if not row_starts or row_starts[-1] + patch_size < h:
        row_starts.append(max(0, h - patch_size))

    col_starts = list(range(0, w - patch_size + 1, stride))
    if not col_starts or col_starts[-1] + patch_size < w:
        col_starts.append(max(0, w - patch_size))

    # Deduplicate while preserving order
    row_starts = list(dict.fromkeys(row_starts))
    col_starts = list(dict.fromkeys(col_starts))

    for r in row_starts:
        for c in col_starts:
            patch = tensor[:, :, r : r + patch_size, c : c + patch_size]
            patches.append((patch, r, c))

    return patches


def _create_blend_weights(size: int, overlap: int, scale: int) -> np.ndarray:
    """Create 1D linear blending weights for patch stitching.

    Args:
        size (int): Length of the 1D weight array (in SR pixel space).
        overlap (int): Overlap region in LR pixel space.
        scale (int): Super-resolution scale factor.

    Returns:
        np.ndarray: 1D array of blending weights in [0, 1].
    """
    sr_overlap = overlap * scale
    weights = np.ones(size, dtype=np.float32)
    if sr_overlap > 0:
        ramp = np.linspace(0, 1, sr_overlap, dtype=np.float32)
        weights[:sr_overlap] = ramp
        weights[-sr_overlap:] = ramp[::-1]
    return weights


def _stitch_patches(
    patches: List[Tuple[np.ndarray, int, int]],
    h: int,
    w: int,
    channels: int,
    scale: int,
    overlap: int,
    patch_size: int,
) -> np.ndarray:
    """Stitch super-resolved patches back into a full image with linear
    blending in overlap regions.

    Args:
        patches (list): List of (sr_patch, lr_row_start, lr_col_start) tuples.
            Each sr_patch has shape (C, H_sr, W_sr).
        h (int): Height of the original LR image.
        w (int): Width of the original LR image.
        channels (int): Number of output channels.
        scale (int): Super-resolution scale factor.
        overlap (int): Overlap in LR pixel space.
        patch_size (int): Patch size in LR pixel space.

    Returns:
        np.ndarray: Stitched image of shape (C, h*scale, w*scale) or
        (h*scale, w*scale) for single-channel data.
    """
    sr_h = h * scale
    sr_w = w * scale
    sr_patch_size = patch_size * scale

    output = np.zeros((channels, sr_h, sr_w), dtype=np.float32)
    weight_map = np.zeros((1, sr_h, sr_w), dtype=np.float32)

    # Build 2D blending weights
    w1d = _create_blend_weights(sr_patch_size, overlap, scale)
    weight_2d = w1d[np.newaxis, :] * w1d[:, np.newaxis]  # (ps, ps)

    for sr_patch, lr_r, lr_c in patches:
        sr_r = lr_r * scale
        sr_c = lr_c * scale
        ph = sr_patch.shape[-2]
        pw = sr_patch.shape[-1]

        # Trim weight_2d to match actual patch dimensions
        w2d = weight_2d[:ph, :pw]

        output[:, sr_r : sr_r + ph, sr_c : sr_c + pw] += sr_patch * w2d[np.newaxis]
        weight_map[:, sr_r : sr_r + ph, sr_c : sr_c + pw] += w2d[np.newaxis]

    # Avoid division by zero
    weight_map = np.maximum(weight_map, 1e-8)
    output /= weight_map

    if channels == 1:
        return output[0]
    return output


def _process_patched(
    model,
    lr_tensor: torch.Tensor,
    patch_size: int,
    overlap: int,
    scale: int,
    sampling_steps: int,
) -> np.ndarray:
    """Run super-resolution on a large image using patched inference.

    Args:
        model: The OpenSR SRLatentDiffusion model.
        lr_tensor (torch.Tensor): Input tensor of shape (1, C, H, W).
        patch_size (int): Tile size in pixels.
        overlap (int): Overlap between adjacent tiles in pixels.
        scale (int): Super-resolution scale factor.
        sampling_steps (int): Number of diffusion sampling steps.

    Returns:
        np.ndarray: Stitched super-resolution image of shape (C, H*scale,
        W*scale).
    """
    _, channels, h, w = lr_tensor.shape
    patches = _create_patches(lr_tensor, patch_size, overlap)
    total = len(patches)
    print(f"Processing {total} patches ({patch_size}x{patch_size}, overlap={overlap})")

    sr_patches = []
    for i, (patch, r, c) in enumerate(patches):
        print(f"  Patch {i + 1}/{total} at ({r}, {c})")
        sr_tensor = model.forward(patch, sampling_steps=sampling_steps)
        sr_patch = sr_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        sr_patches.append((sr_patch, r, c))

    return _stitch_patches(sr_patches, h, w, channels, scale, overlap, patch_size)


def _process_patched_uncertainty(
    model,
    lr_tensor: torch.Tensor,
    patch_size: int,
    overlap: int,
    scale: int,
    n_variations: int,
    sampling_steps: int,
) -> np.ndarray:
    """Compute uncertainty maps using patched inference.

    Args:
        model: The OpenSR SRLatentDiffusion model.
        lr_tensor (torch.Tensor): Input tensor of shape (1, C, H, W).
        patch_size (int): Tile size in pixels.
        overlap (int): Overlap between adjacent tiles in pixels.
        scale (int): Super-resolution scale factor.
        n_variations (int): Number of stochastic forward passes.
        sampling_steps (int): Number of diffusion sampling steps.

    Returns:
        np.ndarray: Stitched uncertainty map of shape (H*scale, W*scale).
    """
    _, _, h, w = lr_tensor.shape
    patches = _create_patches(lr_tensor, patch_size, overlap)
    total = len(patches)
    print(f"Computing uncertainty for {total} patches")

    unc_patches = []
    for i, (patch, r, c) in enumerate(patches):
        print(f"  Uncertainty patch {i + 1}/{total} at ({r}, {c})")
        unc_tensor = model.uncertainty_map(
            patch, n_variations=n_variations, sampling_steps=sampling_steps
        )
        unc_patch = unc_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        unc_patches.append((unc_patch, r, c))

    return _stitch_patches(unc_patches, h, w, 1, scale, overlap, patch_size)


def plot_sr_comparison(
    lr_path: str,
    sr_path: str,
    bands: list[int] = [1, 2, 3],
    lr_vmax: Optional[float] = None,
    sr_vmax: Optional[float] = None,
    figsize: tuple = (14, 7),
    **kwargs,
):
    """Plot a side-by-side comparison of low-resolution and super-resolution
    images.

    Displays RGB composites of the LR input and SR output.

    Args:
        lr_path (str): Path to the low-resolution GeoTIFF.
        sr_path (str): Path to the super-resolution GeoTIFF.
        bands (list[int]): Three 1-based band indices for the RGB composite.
            Default is ``[1, 2, 3]``.
        lr_vmax (float, optional): Maximum value for LR image contrast
            stretch. If None, the 98th percentile is used.
        sr_vmax (float, optional): Maximum value for SR image contrast
            stretch. If None, the 98th percentile is used.
        figsize (tuple): Figure size. Default is ``(14, 7)``.
        **kwargs: Additional keyword arguments passed to
            ``matplotlib.pyplot.subplots``.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    with rasterio.open(sr_path) as sr_src:
        sr = sr_src.read(bands).astype(np.float32)
        sr_res = abs(sr_src.transform.a)
        sr_bounds = sr_src.bounds

    with rasterio.open(lr_path) as lr_src:
        lr_res = abs(lr_src.transform.a)
        # Read only the region matching the SR output extent
        lr_window = lr_src.window(*sr_bounds)
        lr_window = lr_window.round_offsets().round_lengths()
        lr_window = lr_window.intersection(Window(0, 0, lr_src.width, lr_src.height))
        lr = lr_src.read(bands, window=lr_window).astype(np.float32)

    def _stretch(img, vmax=None):
        out = np.zeros_like(img)
        for i in range(img.shape[0]):
            band = img[i]
            vmin = np.percentile(band[band > 0], 2) if (band > 0).any() else 0
            if vmax is None:
                vm = np.percentile(band[band > 0], 98) if (band > 0).any() else 1
            else:
                vm = vmax
            out[i] = (band - vmin) / (vm - vmin + 1e-10)
        return np.clip(out, 0, 1).transpose(1, 2, 0)

    lr_rgb = _stretch(lr, lr_vmax)
    sr_rgb = _stretch(sr, sr_vmax)

    fig, axes = plt.subplots(1, 2, figsize=figsize, **kwargs)

    axes[0].imshow(lr_rgb)
    axes[0].set_title(f"Low Resolution ({lr_res:.1f} m)")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(sr_rgb)
    axes[1].set_title(f"Super Resolution ({sr_res:.2f} m)")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    plt.tight_layout()
    return fig


def plot_sr_uncertainty(
    uncertainty_path: str,
    cmap: str = "RdYlGn_r",
    normalize: bool = True,
    figsize: tuple = (8, 8),
    **kwargs,
):
    """Plot the uncertainty map from super-resolution inference.

    Args:
        uncertainty_path (str): Path to the uncertainty GeoTIFF.
        cmap (str): Matplotlib colormap name. Default is ``'RdYlGn_r'``.
        normalize (bool): Whether to normalize values to [0, 1]. Default
            is True.
        figsize (tuple): Figure size. Default is ``(8, 8)``.
        **kwargs: Additional keyword arguments passed to
            ``matplotlib.pyplot.subplots``.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    with rasterio.open(uncertainty_path) as src:
        unc = src.read(1).astype(np.float32)
        res = abs(src.transform.a)

    if normalize and unc.max() > unc.min():
        unc = (unc - unc.min()) / (unc.max() - unc.min())
        label = "Uncertainty (Normalized)"
    else:
        label = "Uncertainty"

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    im = ax.imshow(unc, cmap=cmap)
    ax.set_title(f"{label} ({res:.2f} m)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig
