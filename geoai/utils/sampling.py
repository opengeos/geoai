"""TorchGeo-style sampling utilities for GeoAI vision workflows.

This module centralizes common TorchGeo dataset, sampler, and dataloader
construction patterns while keeping TorchGeo imports lazy. Importing
``geoai.utils`` should not require TorchGeo; these helpers only require it
when they are called.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "create_raster_dataset",
    "create_segmentation_dataset",
    "create_geo_sampler",
    "create_geo_dataloader",
    "create_geo_dataloaders",
    "create_torchgeo_segmentation_dataloaders",
    "geo_sample_to_tuple",
    "predict_torchgeo_segmentation_batch",
    "plot_torchgeo_segmentation_predictions",
    "train_torchgeo_segmentation_model",
]


def _import_torchgeo() -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Import TorchGeo objects with a clear error message."""
    try:
        from torch.utils.data import DataLoader
        from torchgeo.datasets import GeoDataset, RasterDataset, stack_samples
        from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
    except ImportError as exc:
        raise ImportError(
            "TorchGeo is required for GeoAI sampling utilities. "
            "Install it with: pip install torchgeo"
        ) from exc

    return (
        DataLoader,
        GeoDataset,
        RasterDataset,
        stack_samples,
        RandomGeoSampler,
        GridGeoSampler,
    )


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only keyword arguments accepted by ``callable_obj``."""
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return {key: value for key, value in kwargs.items() if value is not None}

    return {
        key: value
        for key, value in kwargs.items()
        if key in params
        and (value is not None or params[key].default is inspect.Parameter.empty)
    }


def _to_units(units: Any) -> Any:
    """Convert a string units value to TorchGeo's ``Units`` enum if available."""
    if not isinstance(units, str):
        return units

    try:
        from torchgeo.samplers import Units
    except ImportError:
        return units

    value = units.lower()
    if value in {"pixel", "pixels"}:
        return Units.PIXELS
    if value in {"crs", "map", "meters", "metres"}:
        return Units.CRS
    raise ValueError("units must be 'pixels' or 'crs'")


def create_raster_dataset(
    paths: Any = "data",
    *,
    is_image: bool = True,
    filename_glob: str = "*.tif",
    separate_files: bool = False,
    filename_regex: str = ".*",
    date_format: str = "%Y%m%d",
    crs: Any = None,
    res: Any = None,
    bands: Any = None,
    transforms: Any = None,
    cache: bool = True,
    time_series: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a TorchGeo ``RasterDataset`` for imagery or masks.

    Args:
        paths: File, directory, or iterable of paths accepted by TorchGeo.
        is_image: Whether the raster should be loaded as imagery.
        filename_glob: Glob pattern used by TorchGeo to discover files.
        separate_files: Whether bands are stored in separate files.
        filename_regex: Regex used by TorchGeo for date/band extraction.
        date_format: Date format used with ``filename_regex``.
        crs: Optional target CRS.
        res: Optional target resolution.
        bands: Optional band names.
        transforms: Optional TorchGeo sample transform.
        cache: Whether TorchGeo should cache file handles/indexing.
        time_series: Whether to build a time-series raster dataset.
        **kwargs: Additional keyword arguments forwarded to ``RasterDataset``.

    Returns:
        A TorchGeo ``RasterDataset`` instance.
    """
    _, _, RasterDataset, _, _, _ = _import_torchgeo()

    class GeoAIRasterDataset(RasterDataset):  # type: ignore[misc, valid-type]
        pass

    GeoAIRasterDataset.filename_glob = filename_glob
    GeoAIRasterDataset.is_image = is_image
    GeoAIRasterDataset.separate_files = separate_files
    GeoAIRasterDataset.filename_regex = filename_regex
    GeoAIRasterDataset.date_format = date_format

    init_kwargs = {
        "paths": paths,
        "crs": crs,
        "res": res,
        "bands": bands,
        "transforms": transforms,
        "cache": cache,
        "time_series": time_series,
        **kwargs,
    }
    return GeoAIRasterDataset(**_filter_kwargs(RasterDataset, init_kwargs))


def create_segmentation_dataset(
    image_paths_or_dataset: Any,
    mask_paths_or_dataset: Any,
    *,
    image_filename_glob: str = "*.tif",
    mask_filename_glob: str = "*.tif",
    image_transforms: Any = None,
    mask_transforms: Any = None,
    crs: Any = None,
    res: Any = None,
    bands: Any = None,
    cache: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an intersected TorchGeo image/mask dataset.

    Existing ``GeoDataset`` instances are used as-is. Path inputs are wrapped
    in ``RasterDataset`` subclasses and combined with TorchGeo's ``&`` operator.
    """
    _, GeoDataset, _, _, _, _ = _import_torchgeo()

    if isinstance(image_paths_or_dataset, GeoDataset):
        images = image_paths_or_dataset
    else:
        images = create_raster_dataset(
            image_paths_or_dataset,
            is_image=True,
            filename_glob=image_filename_glob,
            transforms=image_transforms,
            crs=crs,
            res=res,
            bands=bands,
            cache=cache,
            **kwargs,
        )

    if isinstance(mask_paths_or_dataset, GeoDataset):
        masks = mask_paths_or_dataset
    else:
        masks = create_raster_dataset(
            mask_paths_or_dataset,
            is_image=False,
            filename_glob=mask_filename_glob,
            transforms=mask_transforms,
            crs=crs,
            res=res,
            cache=cache,
            **kwargs,
        )

    return images & masks


def create_geo_sampler(
    dataset: Any,
    sampler: Any = "random",
    *,
    size: Any = 256,
    stride: Any = None,
    length: Optional[int] = None,
    roi: Any = None,
    toi: Any = None,
    units: Any = "pixels",
    generator: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a TorchGeo sampler or return an already-created sampler.

    Args:
        dataset: TorchGeo ``GeoDataset`` to sample.
        sampler: ``"random"``, ``"grid"``, or a sampler instance.
        size: Patch size in pixels or CRS units.
        stride: Grid stride. Defaults to ``size`` for grid sampling.
        length: Number of random samples per epoch.
        roi: Optional region of interest.
        toi: Optional time of interest for TorchGeo versions that support it.
        units: ``"pixels"``, ``"crs"``, or a TorchGeo ``Units`` value.
        generator: Optional random generator for supported TorchGeo versions.
        **kwargs: Extra sampler keyword arguments.

    Returns:
        A TorchGeo sampler instance.
    """
    _, _, _, _, RandomGeoSampler, GridGeoSampler = _import_torchgeo()

    if not isinstance(sampler, str):
        return sampler

    sampler_name = sampler.lower()
    units_value = _to_units(units)

    if sampler_name in {"random", "random_geo", "randomgeosampler"}:
        sampler_kwargs = {
            "dataset": dataset,
            "size": size,
            "length": length,
            "roi": roi,
            "toi": toi,
            "units": units_value,
            "generator": generator,
            **kwargs,
        }
        return RandomGeoSampler(**_filter_kwargs(RandomGeoSampler, sampler_kwargs))

    if sampler_name in {"grid", "grid_geo", "gridgeosampler"}:
        sampler_kwargs = {
            "dataset": dataset,
            "size": size,
            "stride": size if stride is None else stride,
            "roi": roi,
            "toi": toi,
            "units": units_value,
            **kwargs,
        }
        return GridGeoSampler(**_filter_kwargs(GridGeoSampler, sampler_kwargs))

    raise ValueError("sampler must be 'random', 'grid', or a sampler instance")


def create_geo_dataloader(
    dataset: Any,
    *,
    sampler: Any = None,
    sampler_type: str = "random",
    size: Any = 256,
    stride: Any = None,
    length: Optional[int] = None,
    roi: Any = None,
    toi: Any = None,
    units: Any = "pixels",
    generator: Any = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Any = None,
    **dataloader_kwargs: Any,
) -> Any:
    """Create a PyTorch ``DataLoader`` for a TorchGeo ``GeoDataset``."""
    DataLoader, _, _, stack_samples, _, _ = _import_torchgeo()

    if sampler is None:
        sampler = create_geo_sampler(
            dataset,
            sampler=sampler_type,
            size=size,
            stride=stride,
            length=length,
            roi=roi,
            toi=toi,
            units=units,
            generator=generator,
        )

    if collate_fn is None:
        collate_fn = stack_samples

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **dataloader_kwargs,
    )


def create_geo_dataloaders(
    train_dataset: Any,
    val_dataset: Any = None,
    test_dataset: Any = None,
    *,
    size: Any = 256,
    stride: Any = None,
    length: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    train_sampler: Any = "random",
    eval_sampler: Any = "grid",
    collate_fn: Any = None,
    **dataloader_kwargs: Any,
) -> Dict[str, Any]:
    """Create train/validation/test TorchGeo dataloaders.

    Returns a dictionary with ``"train"``, ``"val"``, and ``"test"`` keys.
    Missing validation or test datasets map to ``None``.
    """
    loaders: Dict[str, Any] = {
        "train": create_geo_dataloader(
            train_dataset,
            sampler=train_sampler if not isinstance(train_sampler, str) else None,
            sampler_type=train_sampler if isinstance(train_sampler, str) else "random",
            size=size,
            length=length,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        ),
        "val": None,
        "test": None,
    }

    if val_dataset is not None:
        loaders["val"] = create_geo_dataloader(
            val_dataset,
            sampler=eval_sampler if not isinstance(eval_sampler, str) else None,
            sampler_type=eval_sampler if isinstance(eval_sampler, str) else "grid",
            size=size,
            stride=stride,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    if test_dataset is not None:
        loaders["test"] = create_geo_dataloader(
            test_dataset,
            sampler=eval_sampler if not isinstance(eval_sampler, str) else None,
            sampler_type=eval_sampler if isinstance(eval_sampler, str) else "grid",
            size=size,
            stride=stride,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    return loaders


def _as_tensor(value: Any) -> Any:
    """Convert array-like values to tensors while preserving tensors."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to convert GeoAI samples to tensors."
        ) from exc

    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _coerce_channels(image: Any, num_channels: Optional[int]) -> Any:
    """Truncate or zero-pad the channel dimension to ``num_channels``.

    When the input has more channels than ``num_channels``, the first
    ``num_channels`` bands are kept and a warning is emitted because dropped
    bands may carry meaningful information (NIR/SWIR, etc.). Callers that
    need explicit band selection should subset the image before calling.
    """
    if num_channels is None:
        return image

    import torch

    if image.ndim < 3:
        raise ValueError("image must have shape (C, H, W) or (B, C, H, W)")

    channel_dim = -3
    channels = image.shape[channel_dim]
    if channels == num_channels:
        return image
    if channels > num_channels:
        warnings.warn(
            f"Dropping {channels - num_channels} band(s) to match num_channels="
            f"{num_channels}; keeping the first {num_channels} band(s). Subset "
            "the image yourself to control which bands are retained.",
            stacklevel=2,
        )
        index = [slice(None)] * image.ndim
        index[channel_dim] = slice(0, num_channels)
        return image[tuple(index)]

    pad_shape = list(image.shape)
    pad_shape[channel_dim] = num_channels - channels
    padding = torch.zeros(
        pad_shape, dtype=image.dtype, device=image.device, layout=image.layout
    )
    return torch.cat([image, padding], dim=channel_dim)


def geo_sample_to_tuple(
    batch: Any,
    *,
    image_key: str = "image",
    target_key: Optional[str] = "mask",
    num_channels: Optional[int] = None,
    normalize: bool = False,
    squeeze_target: bool = True,
) -> Tuple[Any, Any]:
    """Convert a TorchGeo dict batch to an ``(image, target)`` tuple.

    Tuple/list batches are also accepted for compatibility with standard
    PyTorch datasets.
    """
    if isinstance(batch, dict):
        if image_key not in batch:
            raise KeyError(f"Missing image key: {image_key!r}")
        image = batch[image_key]
        target = batch[target_key] if target_key is not None else None
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("batch must not be empty")
        image = batch[0]
        target = batch[1] if len(batch) > 1 and target_key is not None else None
    else:
        image = batch
        target = None

    image = _as_tensor(image)
    image = _coerce_channels(image, num_channels)
    if normalize:
        # Only divide by 255 when the input is an integer dtype; float tensors
        # are assumed to be already model-ready (e.g. ImageNet-normalized) so
        # that callers who pre-normalize are not silently re-scaled.
        if not image.is_floating_point():
            image = image.float() / 255.0
        else:
            image = image.float()

    if target is None:
        return image, None

    target = _as_tensor(target)
    if squeeze_target:
        if image.ndim == 3 and target.ndim == 3 and target.shape[0] == 1:
            target = target.squeeze(0)
        elif image.ndim == 4 and target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
    return image, target.long()


def create_torchgeo_segmentation_dataloaders(
    image_path: Any,
    mask_path: Any,
    *,
    chip_size: Any = 256,
    stride: Any = None,
    train_length: int = 128,
    val_length: int = 32,
    batch_size: int = 4,
    num_workers: int = 0,
    val_sampler: str = "random",
    include_grid_loader: bool = True,
    image_filename_glob: str = "*.tif",
    mask_filename_glob: str = "*.tif",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create dataset and dataloaders for TorchGeo semantic segmentation.

    This is a higher-level convenience wrapper around
    :func:`create_segmentation_dataset` and :func:`create_geo_dataloader`.
    """
    dataset = create_segmentation_dataset(
        image_path,
        mask_path,
        image_filename_glob=image_filename_glob,
        mask_filename_glob=mask_filename_glob,
        **kwargs,
    )

    loaders: Dict[str, Any] = {
        "dataset": dataset,
        "train": create_geo_dataloader(
            dataset,
            sampler_type="random",
            size=chip_size,
            length=train_length,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    }

    if val_sampler == "random":
        loaders["val"] = create_geo_dataloader(
            dataset,
            sampler_type="random",
            size=chip_size,
            length=val_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        loaders["val"] = create_geo_dataloader(
            dataset,
            sampler_type=val_sampler,
            size=chip_size,
            stride=chip_size if stride is None else stride,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    if include_grid_loader:
        loaders["grid"] = create_geo_dataloader(
            dataset,
            sampler_type="grid",
            size=chip_size,
            stride=chip_size if stride is None else stride,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return loaders


def _default_segmentation_model(num_channels: int, num_classes: int) -> Any:
    """Create a small CPU-friendly segmentation model."""
    import torch.nn as nn

    class SmallSegNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(32, num_classes, kernel_size=1),
            )

        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    return SmallSegNet()


def _prepare_segmentation_batch(
    batch: Any,
    device: Any,
    *,
    num_channels: int,
    normalize: bool,
    binary: bool,
    mask_threshold: int,
) -> Tuple[Any, Any]:
    """Prepare a TorchGeo batch for semantic segmentation training."""
    images, masks = geo_sample_to_tuple(
        batch,
        num_channels=num_channels,
        normalize=normalize,
    )
    if masks is None:
        raise ValueError("Segmentation batches must include masks")
    if binary:
        masks = (masks > mask_threshold).long()
    return images.to(device), masks.to(device)


def _segmentation_mean_iou(logits: Any, masks: Any, num_classes: int) -> float:
    """Compute mean IoU for segmentation logits."""
    preds = logits.argmax(dim=1)
    scores = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls
        union = (pred_cls | mask_cls).sum()
        if union == 0:
            continue
        intersection = (pred_cls & mask_cls).sum()
        scores.append((intersection.float() / union.float()).item())
    return sum(scores) / len(scores) if scores else 0.0


def train_torchgeo_segmentation_model(
    image_path: Any = None,
    mask_path: Any = None,
    *,
    train_dataloader: Any = None,
    val_dataloader: Any = None,
    model: Any = None,
    output_dir: Optional[str] = None,
    num_channels: int = 3,
    num_classes: int = 2,
    chip_size: Any = 256,
    stride: Any = None,
    train_length: int = 128,
    val_length: int = 32,
    batch_size: int = 4,
    num_workers: int = 0,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_fn: Any = None,
    optimizer: Any = None,
    device: Any = None,
    normalize: bool = True,
    binary: bool = True,
    mask_threshold: int = 0,
    verbose: bool = True,
    **dataset_kwargs: Any,
) -> Dict[str, Any]:
    """Train a simple semantic segmentation model from TorchGeo samples.

    This convenience function provides a compact, chip-free training workflow
    for aligned image and mask GeoTIFFs. It is intended for examples,
    baselines, and quick experiments. For larger production models, pass a
    custom ``model``, ``loss_fn``, and/or prebuilt dataloaders.
    """
    import os

    import torch
    import torch.nn as nn

    if train_dataloader is None:
        if image_path is None or mask_path is None:
            raise ValueError(
                "Provide image_path and mask_path or a prebuilt train_dataloader"
            )
        data = create_torchgeo_segmentation_dataloaders(
            image_path,
            mask_path,
            chip_size=chip_size,
            stride=stride,
            train_length=train_length,
            val_length=val_length,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )
        train_dataloader = data["train"]
        if val_dataloader is None:
            val_dataloader = data["val"]
    else:
        data = {"train": train_dataloader, "val": val_dataloader}

    if device is None:
        from .device import get_device

        device = get_device()
    else:
        device = torch.device(device)

    if model is None:
        model = _default_segmentation_model(num_channels, num_classes)
    model = model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    best_val_loss = float("inf")
    best_model_path = None
    history = []

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, "best_model.pth")

    def run_epoch(loader: Any, training: bool) -> Tuple[float, float]:
        model.train(training)
        total_loss = 0.0
        total_iou = 0.0
        count = 0

        for batch in loader:
            images, masks = _prepare_segmentation_batch(
                batch,
                device,
                num_channels=num_channels,
                normalize=normalize,
                binary=binary,
                mask_threshold=mask_threshold,
            )

            with torch.set_grad_enabled(training):
                logits = model(images)
                loss = loss_fn(logits, masks)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            total_iou += _segmentation_mean_iou(logits.detach(), masks, num_classes)
            count += 1

        if count == 0:
            raise ValueError(
                "Dataloader yielded no batches; cannot compute epoch metrics. "
                "Check that the dataset and sampler produce at least one sample."
            )
        return total_loss / count, total_iou / count

    for epoch in range(num_epochs):
        train_loss, train_iou = run_epoch(train_dataloader, training=True)
        if val_dataloader is not None:
            val_loss, val_iou = run_epoch(val_dataloader, training=False)
        else:
            val_loss, val_iou = train_loss, train_iou

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "best_val_loss": best_val_loss,
        }
        history.append(row)

        if verbose:
            suffix = " *" if improved else ""
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.3f}, train_iou={train_iou:.3f}, "
                f"val_loss={val_loss:.3f}, val_iou={val_iou:.3f}, "
                f"best_val_loss={best_val_loss:.3f}{suffix}"
            )

    history_path = None
    if output_dir is not None:
        history_path = os.path.join(output_dir, "training_history.pth")
        torch.save(
            {
                "train_loss": [row["train_loss"] for row in history],
                "val_loss": [row["val_loss"] for row in history],
                "val_iou": [row["val_iou"] for row in history],
                "train_iou": [row["train_iou"] for row in history],
                "best_val_loss": [row["best_val_loss"] for row in history],
            },
            history_path,
        )

    return {
        "model": model,
        "history": history,
        "history_path": history_path,
        "train_loader": train_dataloader,
        "val_loader": val_dataloader,
        "dataset": data.get("dataset"),
        "best_model_path": best_model_path,
        "device": device,
    }


def predict_torchgeo_segmentation_batch(
    model: Any,
    dataloader_or_batch: Any,
    *,
    device: Any = None,
    num_channels: int = 3,
    normalize: bool = True,
    binary: bool = True,
    mask_threshold: int = 0,
) -> Dict[str, Any]:
    """Predict one TorchGeo segmentation batch."""
    import torch

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if isinstance(dataloader_or_batch, dict) or isinstance(
        dataloader_or_batch, (list, tuple)
    ):
        batch = dataloader_or_batch
    else:
        batch = next(iter(dataloader_or_batch))

    images, masks = _prepare_segmentation_batch(
        batch,
        device,
        num_channels=num_channels,
        normalize=normalize,
        binary=binary,
        mask_threshold=mask_threshold,
    )

    model.eval()
    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1)

    return {
        "images": images.detach().cpu(),
        "masks": masks.detach().cpu(),
        "predictions": predictions.detach().cpu(),
        "logits": logits.detach().cpu(),
    }


def plot_torchgeo_segmentation_predictions(
    model: Any,
    dataloader_or_batch: Any,
    *,
    n: int = 4,
    figsize: Tuple[int, int] = (9, 12),
    cmap: str = "Blues",
    **predict_kwargs: Any,
) -> Any:
    """Plot image, mask, and prediction panels for a TorchGeo batch."""
    import matplotlib.pyplot as plt

    result = predict_torchgeo_segmentation_batch(
        model, dataloader_or_batch, **predict_kwargs
    )
    images = result["images"]
    masks = result["masks"]
    predictions = result["predictions"]

    count = min(n, images.shape[0])
    fig, axes = plt.subplots(count, 3, figsize=figsize)
    if count == 1:
        axes = axes[None, :]

    for i in range(count):
        rgb = images[i].permute(1, 2, 0).clamp(0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(masks[i], cmap=cmap, vmin=0, vmax=1)
        axes[i, 1].set_title("Mask")
        axes[i, 2].imshow(predictions[i], cmap=cmap, vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    return fig
