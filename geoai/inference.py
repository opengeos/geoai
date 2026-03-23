"""Memory-efficient tiled inference with blending and test-time augmentation.

Provides a generic sliding-window inference pipeline that works with any
PyTorch segmentation or regression model on GeoTIFF rasters.  Key features:

*   **Windowed I/O** -- reads tiles directly via rasterio windows,
    avoiding full-image input memory allocation.
*   **Multiple blending strategies** -- linear ramp, raised cosine, and spline
    windows for seamless tile stitching.
*   **D4 test-time augmentation** -- optional 8-fold augmentation using the
    dihedral group (identity, 3 rotations, horizontal flip, vertical flip,
    2 diagonal flips).

References:
    - Spline blending: https://github.com/Vooban/Smoothly-Blend-Image-Patches
    - GitHub issue: https://github.com/opengeos/geoai/issues/87
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BlendMode",
    "create_weight_mask",
    "predict_geotiff",
    "d4_forward",
    "d4_inverse",
    "d4_tta_forward",
]


class BlendMode(str, Enum):
    """Blending strategy for overlapping tile predictions.

    Attributes:
        NONE: Uniform averaging -- all pixels are weighted equally (1.0),
            so overlapping tiles are simply averaged without tapering.
        LINEAR: Linear ramp from 0 at edges to 1 at center.
        COSINE: Raised-cosine (Hann) taper in the overlap region.
        SPLINE: Powered raised-cosine taper for smooth transitions in
            the overlap zones. Requires ``overlap <= tile_size // 2``.
    """

    NONE = "none"
    LINEAR = "linear"
    COSINE = "cosine"
    SPLINE = "spline"


def _spline_window_1d(window_size: int, overlap: int, power: int = 2) -> np.ndarray:
    """Create a 1D spline window that tapers only in the overlap zones.

    The window is 1.0 in the non-overlapping centre and smoothly tapers
    to 0 at the edges using a raised half-cosine shaped by *power*.

    Args:
        window_size: Length of the window in pixels.
        overlap: Number of pixels in the overlap zone at each edge.
        power: Exponent applied to the taper. Higher values produce
            steeper roll-off near the edges.

    Returns:
        numpy.ndarray: 1D float64 array of shape ``(window_size,)``.
    """
    if overlap <= 0:
        return np.ones(window_size, dtype=np.float64)

    window = np.ones(window_size, dtype=np.float64)
    # Left taper: 0 .. overlap  ->  ramp from 0 to 1
    # Use endpoint=True so the taper reaches exactly 1.0, avoiding a
    # discontinuity at the boundary with the flat centre region.
    ramp = np.linspace(0.0, 1.0, overlap + 1, endpoint=True)[1:]
    # Apply half-cosine shaping then power for smoothness
    ramp = (0.5 * (1.0 - np.cos(np.pi * ramp))) ** power
    window[:overlap] = ramp
    # Right taper: mirror of left
    window[-overlap:] = ramp[::-1]
    return window


def create_weight_mask(
    tile_size: int,
    overlap: int,
    mode: Union[str, BlendMode] = BlendMode.SPLINE,
    power: int = 2,
) -> np.ndarray:
    """Create a 2D weight mask for blending overlapping tiles.

    Args:
        tile_size: Size of each square tile in pixels.
        overlap: Number of pixels of overlap between adjacent tiles.
        mode: Blending strategy. One of ``"none"``, ``"linear"``,
            ``"cosine"``, or ``"spline"``.
        power: Exponent for spline mode. Higher values concentrate
            weight toward the center. Ignored for other modes.

    Returns:
        numpy.ndarray: Float32 array of shape ``(tile_size, tile_size)``
            with values in ``[0, 1]``.

    Raises:
        ValueError: If *mode* is not a recognized blending strategy, or
            if *overlap* is negative or >= *tile_size*.

    Example:
        >>> from geoai.inference import create_weight_mask
        >>> mask = create_weight_mask(256, 64, mode="spline")
        >>> mask.shape
        (256, 256)
    """
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"overlap must be >= 0 and < tile_size ({tile_size}), got {overlap}"
        )

    mode = BlendMode(mode)

    if overlap == 0 or mode == BlendMode.NONE:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    if mode == BlendMode.LINEAR:
        # Compute per-pixel distance from nearest edge, capped at overlap.
        # np.minimum avoids write-order corruption when overlap > tile_size // 2.
        left = np.arange(tile_size, dtype=np.float32)
        right = np.arange(tile_size - 1, -1, -1, dtype=np.float32)
        ramp = np.minimum(np.minimum(left, right), overlap) / overlap
        return np.outer(ramp, ramp)

    if mode == BlendMode.COSINE:
        # Same edge-distance approach, then apply raised-cosine taper.
        left = np.arange(tile_size, dtype=np.float32)
        right = np.arange(tile_size - 1, -1, -1, dtype=np.float32)
        dist = np.minimum(np.minimum(left, right), overlap) / overlap
        w = (0.5 * (1.0 - np.cos(np.pi * dist))).astype(np.float32)
        return np.outer(w, w)

    if mode == BlendMode.SPLINE:
        if overlap > tile_size // 2:
            raise ValueError(
                f"For spline blending, overlap must be <= tile_size // 2 "
                f"({tile_size // 2}), got {overlap}. Use 'linear' or "
                f"'cosine' blending for larger overlaps."
            )
        w1d = _spline_window_1d(tile_size, overlap, power=power).astype(np.float32)
        return np.outer(w1d, w1d)

    raise ValueError(f"Unknown blend mode: {mode!r}")


# ---------------------------------------------------------------------------
# D4 dihedral group test-time augmentation
# ---------------------------------------------------------------------------


def d4_forward(tensor: "torch.Tensor") -> List["torch.Tensor"]:
    """Apply all 8 D4 dihedral group transforms to a batch of images.

    The D4 group consists of the identity, three 90-degree rotations,
    horizontal flip, vertical flip, and two diagonal flips.

    Args:
        tensor: Input tensor of shape ``(B, C, H, W)``.

    Returns:
        list: List of 8 tensors, each of shape ``(B, C, H, W)``.
    """
    import torch  # noqa: F811

    return [
        tensor,  # identity
        torch.rot90(tensor, k=1, dims=[-2, -1]),
        torch.rot90(tensor, k=2, dims=[-2, -1]),
        torch.rot90(tensor, k=3, dims=[-2, -1]),
        torch.flip(tensor, dims=[-1]),  # horizontal flip
        torch.flip(tensor, dims=[-2]),  # vertical flip
        torch.flip(torch.rot90(tensor, k=1, dims=[-2, -1]), dims=[-1]),
        torch.flip(torch.rot90(tensor, k=1, dims=[-2, -1]), dims=[-2]),
    ]


def d4_inverse(tensors: List["torch.Tensor"]) -> List["torch.Tensor"]:
    """Apply the inverse D4 transforms to undo :func:`d4_forward`.

    Args:
        tensors: List of 8 tensors from :func:`d4_forward`, each of
            shape ``(B, C, H, W)``.

    Returns:
        list: List of 8 tensors, each transformed back to the original
            orientation.
    """
    import torch  # noqa: F811

    return [
        tensors[0],  # identity
        torch.rot90(tensors[1], k=3, dims=[-2, -1]),
        torch.rot90(tensors[2], k=2, dims=[-2, -1]),
        torch.rot90(tensors[3], k=1, dims=[-2, -1]),
        torch.flip(tensors[4], dims=[-1]),
        torch.flip(tensors[5], dims=[-2]),
        torch.rot90(torch.flip(tensors[6], dims=[-1]), k=3, dims=[-2, -1]),
        torch.rot90(torch.flip(tensors[7], dims=[-2]), k=3, dims=[-2, -1]),
    ]


def d4_tta_forward(
    model: "torch.nn.Module",
    batch: "torch.Tensor",
) -> "torch.Tensor":
    """Run inference with D4 test-time augmentation and average results.

    Applies all 8 D4 transforms, runs the model on each, inverts the
    transforms, and averages the predictions.  This can improve
    prediction quality at the cost of 8x compute.

    Args:
        model: PyTorch model that accepts ``(B, C, H, W)`` input and
            returns ``(B, num_classes, H, W)`` output.
        batch: Input tensor of shape ``(B, C, H, W)``.

    Returns:
        torch.Tensor: Averaged prediction tensor of shape
            ``(B, num_classes, H, W)``.
    """
    import torch  # noqa: F811

    augmented = d4_forward(batch)
    outputs = [model(aug) for aug in augmented]
    restored = d4_inverse(outputs)
    return torch.stack(restored).mean(dim=0)


# ---------------------------------------------------------------------------
# Default preprocessing
# ---------------------------------------------------------------------------


def _default_preprocess(tile: np.ndarray) -> np.ndarray:
    """Default preprocessing: cast to float32, scale uint8 to [0, 1].

    Args:
        tile: Array of shape ``(C, H, W)``.

    Returns:
        numpy.ndarray: Float32 array of shape ``(C, H, W)``.
    """
    tile = tile.astype(np.float32)
    if tile.max() > 1.5:
        tile = tile / 255.0
    return np.nan_to_num(tile, nan=0.0)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


def predict_geotiff(
    model: "torch.nn.Module",
    input_raster: str,
    output_raster: str,
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 4,
    input_bands: Optional[List[int]] = None,
    num_classes: int = 1,
    output_dtype: str = "float32",
    output_nodata: float = -9999.0,
    blend_mode: Union[str, BlendMode] = "spline",
    blend_power: int = 2,
    tta: bool = False,
    preprocess_fn: Optional[Callable[..., np.ndarray]] = None,
    postprocess_fn: Optional[Callable[..., np.ndarray]] = None,
    device: Optional[str] = None,
    compress: str = "lzw",
    verbose: bool = True,
) -> str:
    """Run tiled inference on a GeoTIFF with blending and optional TTA.

    Reads tiles from *input_raster* using rasterio windowed I/O, runs
    each batch through *model*, blends overlapping predictions with the
    chosen weight mask, and writes results to *output_raster*.  Memory
    usage is proportional to ``batch_size * tile_size**2`` for input
    reads rather than the full image.

    Args:
        model: PyTorch model accepting ``(B, C, H, W)`` float tensors
            and returning ``(B, num_classes, H, W)`` predictions.
        input_raster: Path to the input GeoTIFF file.
        output_raster: Path to save the output GeoTIFF.
        tile_size: Size of square tiles in pixels.
        overlap: Overlap between adjacent tiles in pixels. Using overlap
            with blending weights eliminates tile-boundary artefacts.
            Higher values give smoother results at the cost of more
            computation.
        batch_size: Number of tiles per forward pass.
        input_bands: 1-based band indices to read. If None, reads all
            bands.
        num_classes: Number of output channels/classes from the model.
            Use 1 for regression or binary segmentation.
        output_dtype: NumPy dtype string for the output raster (e.g.,
            ``"float32"``, ``"uint8"``).
        output_nodata: NoData value for the output raster.
        blend_mode: Blending strategy: ``"none"``, ``"linear"``,
            ``"cosine"``, or ``"spline"``.
        blend_power: Exponent for spline blending (ignored for other
            modes).
        tta: If True, apply D4 test-time augmentation. Increases
            compute by 8x but can improve prediction quality.
        preprocess_fn: Optional callable ``(np.ndarray) -> np.ndarray``
            applied to each tile after reading (e.g., normalization).
            Input shape is ``(C, H, W)``.  If None, tiles are cast to
            float32 and divided by 255 when values exceed 1.5.
        postprocess_fn: Optional callable ``(np.ndarray) -> np.ndarray``
            applied to the final blended output array of shape
            ``(num_classes, H, W)`` before writing (e.g., argmax for
            classification).  If None, no post-processing is applied.
        device: PyTorch device string (e.g., ``"cuda"``, ``"cpu"``).
            Auto-detected if None.
        compress: Compression for the output GeoTIFF.
        verbose: Print progress information.

    Returns:
        str: Path to the output raster.

    Raises:
        FileNotFoundError: If *input_raster* does not exist.
        ValueError: If *overlap* >= *tile_size* or *overlap* < 0.

    Example:
        >>> from geoai.inference import predict_geotiff
        >>> predict_geotiff(
        ...     model=my_model,
        ...     input_raster="input.tif",
        ...     output_raster="output.tif",
        ...     tile_size=256,
        ...     overlap=64,
        ...     blend_mode="spline",
        ...     tta=False,
        ... )
        'output.tif'
    """
    import torch
    import rasterio
    from rasterio.windows import Window
    from tqdm.auto import tqdm

    from geoai.utils import get_device

    # ---- validation ----
    if not os.path.exists(input_raster):
        raise FileNotFoundError(f"Input raster not found: {input_raster}")

    if overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"overlap must be >= 0 and < tile_size ({tile_size}), got {overlap}"
        )

    # Validate output_nodata fits the chosen output_dtype
    out_dt = np.dtype(output_dtype)
    if np.issubdtype(out_dt, np.integer):
        info = np.iinfo(out_dt)
        if not (info.min <= output_nodata <= info.max):
            raise ValueError(
                f"output_nodata={output_nodata} is outside the valid range "
                f"[{info.min}, {info.max}] for output_dtype='{output_dtype}'. "
                f"Choose a nodata value that fits the dtype (e.g., 0 or 255 "
                f"for uint8)."
            )

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    preprocess = preprocess_fn if preprocess_fn is not None else _default_preprocess
    stride = tile_size - overlap

    # ---- compute weight mask once ----
    weight_mask = create_weight_mask(
        tile_size, overlap, mode=blend_mode, power=blend_power
    )

    model.to(device)
    model.eval()

    with rasterio.open(input_raster) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()

        if input_bands is None:
            input_bands = list(range(1, src.count + 1))

        if verbose:
            logger.info(
                "Input raster: %dx%d, %d bands", width, height, len(input_bands)
            )
            logger.info(
                "Tile size: %d, overlap: %d, stride: %d", tile_size, overlap, stride
            )

        # ---- allocate output accumulators ----
        output_sum = np.zeros((num_classes, height, width), dtype=np.float64)
        weight_sum = np.zeros((1, height, width), dtype=np.float64)

        # ---- build tile grid ----
        tiles: List[Tuple[int, int, int, int]] = []
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                row_end = min(row + tile_size, height)
                col_end = min(col + tile_size, width)
                # Pull start back so the tile is tile_size when possible
                row_start = max(0, row_end - tile_size)
                col_start = max(0, col_end - tile_size)
                tiles.append((row_start, col_start, row_end, col_end))

        # Deduplicate tiles that map to the same position at boundaries
        tiles = list(dict.fromkeys(tiles))

        if verbose:
            logger.info("Total tiles: %d", len(tiles))

        # ---- process in batches ----
        iterator = range(0, len(tiles), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Running inference")

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(tiles))
            batch_tiles = tiles[batch_start:batch_end]

            # Read tiles via windowed I/O
            batch_images = []
            batch_actual_sizes: List[Tuple[int, int]] = []

            for row_start, col_start, row_end, col_end in batch_tiles:
                actual_h = row_end - row_start
                actual_w = col_end - col_start
                batch_actual_sizes.append((actual_h, actual_w))

                window = Window(col_start, row_start, actual_w, actual_h)
                tile_data = src.read(input_bands, window=window).astype(np.float32)

                # Pad undersized edge tiles
                if actual_h != tile_size or actual_w != tile_size:
                    padded = np.zeros(
                        (len(input_bands), tile_size, tile_size),
                        dtype=np.float32,
                    )
                    padded[:, :actual_h, :actual_w] = tile_data
                    tile_data = padded

                tile_data = preprocess(tile_data)
                batch_images.append(tile_data)

            batch_tensor = torch.from_numpy(np.stack(batch_images)).to(device)

            # Inference
            with torch.no_grad():
                if tta:
                    preds = d4_tta_forward(model, batch_tensor)
                else:
                    preds = model(batch_tensor)

            preds = preds.cpu().numpy()

            # Validate model output shape on first batch
            if batch_start == 0:
                pred_channels = preds.shape[1] if preds.ndim == 4 else 1
                if pred_channels != num_classes:
                    raise ValueError(
                        f"Model output has {pred_channels} channels but "
                        f"num_classes={num_classes}. Set num_classes to "
                        f"match the model output."
                    )

            # Handle models that return (B, H, W) instead of (B, 1, H, W)
            if preds.ndim == 3:
                preds = preds[:, np.newaxis, :, :]

            # Accumulate with blending
            for i, (row_start, col_start, row_end, col_end) in enumerate(batch_tiles):
                actual_h, actual_w = batch_actual_sizes[i]
                pred = preds[i]  # (num_classes, tile_size, tile_size)

                # Crop to actual tile dimensions
                pred_crop = pred[:, :actual_h, :actual_w]
                weight_crop = weight_mask[:actual_h, :actual_w]

                # Accumulate weighted prediction and weights
                output_sum[:, row_start:row_end, col_start:col_end] += (
                    pred_crop * weight_crop[np.newaxis, :, :]
                )
                weight_sum[:, row_start:row_end, col_start:col_end] += weight_crop[
                    np.newaxis, :, :
                ]

    # ---- normalize by weights ----
    # valid mask is (1, H, W); NumPy broadcasts it against (num_classes, H, W)
    valid = weight_sum > 0  # (1, H, W)
    output_array = np.where(
        valid,
        output_sum / (weight_sum + 1e-8),
        output_nodata,
    ).astype(np.float32)

    # ---- postprocess ----
    if postprocess_fn is not None:
        output_array = postprocess_fn(output_array)

    # ---- write output ----
    output_dir = os.path.dirname(os.path.abspath(output_raster))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out_count = output_array.shape[0] if output_array.ndim == 3 else 1
    profile.update(
        count=out_count,
        dtype=output_dtype,
        nodata=output_nodata,
        compress=compress,
    )

    with rasterio.open(output_raster, "w", **profile) as dst:
        if output_array.ndim == 3:
            for band_idx in range(out_count):
                dst.write(output_array[band_idx].astype(output_dtype), band_idx + 1)
        else:
            dst.write(output_array.astype(output_dtype), 1)

    if verbose:
        logger.info("Output saved to: %s", output_raster)
        logger.info("Output dimensions: %dx%d", width, height)

    return output_raster
