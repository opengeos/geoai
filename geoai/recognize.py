"""Module for image recognition (classification) on ImageFolder-style datasets.

This module provides a high-level API for training and evaluating image
classification models on datasets organized as class-named subdirectories.
It supports both standard image formats (JPEG, PNG) and multi-band GeoTIFFs,
and reuses :class:`~geoai.timm_train.TimmClassifier` for training.
"""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from .timm_train import TimmClassifier

# File extensions recognized as GeoTIFF (loaded with rasterio)
_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}

# All supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageDataset(Dataset):
    """Dataset for image classification supporting both standard images and GeoTIFFs.

    Automatically detects the file format and uses the appropriate loader:
    PIL for standard images (JPEG, PNG, BMP) and rasterio for GeoTIFFs
    (supporting multi-band imagery with more than 3 channels).
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        image_size: int = 64,
        in_channels: Optional[int] = None,
    ):
        """Initialize ImageDataset.

        Args:
            image_paths: List of paths to image files (JPEG, PNG, or GeoTIFF).
            labels: List of integer labels corresponding to images.
            transform: Optional transform to apply to images. If ``None``, a
                default transform (Resize + ToTensor + Normalize) is used.
            image_size: Target size to resize images to (height and width).
            in_channels: Expected number of channels. When ``None``, channels
                are inferred from the file (3 for standard images, all bands
                for GeoTIFF). When specified, images are padded or truncated
                to match.
        """
        if len(image_paths) != len(labels):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match "
                f"number of labels ({len(labels)})"
            )

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_standard_image(self, path: str) -> np.ndarray:
        """Load a standard image (JPEG, PNG, BMP) via PIL.

        Args:
            path: Path to the image file.

        Returns:
            Image array with shape ``(C, H, W)`` as float32 in ``[0, 1]``.

        Raises:
            ImportError: If PIL (Pillow) is not installed.
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required to load standard images. "
                "Install with: pip install Pillow"
            )
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        return arr

    def _load_geotiff(self, path: str) -> np.ndarray:
        """Load a GeoTIFF image via rasterio.

        Args:
            path: Path to the GeoTIFF file.

        Returns:
            Image array with shape ``(C, H, W)`` as float32 in ``[0, 1]``.

        Raises:
            ImportError: If rasterio is not installed.
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError(
                "rasterio is required to load GeoTIFF images. "
                "Install with: pip install rasterio"
            )
        with rasterio.open(path) as src:
            # Read all bands: shape (C, H, W)
            arr = src.read().astype(np.float32)

        # Resize if needed (use scipy for float-aware resizing, PIL fallback)
        if arr.shape[1] != self.image_size or arr.shape[2] != self.image_size:
            try:
                import scipy.ndimage as _ndi

                zoom_factors = (
                    1.0,
                    float(self.image_size) / float(arr.shape[1]),
                    float(self.image_size) / float(arr.shape[2]),
                )
                arr = _ndi.zoom(arr, zoom=zoom_factors, order=1).astype(np.float32)
            except ImportError:
                from PIL import Image as _PILImage

                channels = []
                for c in range(arr.shape[0]):
                    band = arr[c]
                    bmin, bmax = float(band.min()), float(band.max())
                    if bmax > bmin:
                        band_norm = (band - bmin) / (bmax - bmin)
                    else:
                        band_norm = np.zeros_like(band, dtype=np.float32)
                    band_img = _PILImage.fromarray((band_norm * 255.0).astype(np.uint8))
                    band_img = band_img.resize(
                        (self.image_size, self.image_size), _PILImage.BILINEAR
                    )
                    band_resized = np.array(band_img, dtype=np.float32) / 255.0
                    # Rescale back to original range
                    channels.append(band_resized * (bmax - bmin) + bmin)
                arr = np.stack(channels, axis=0)

        # Normalize to [0, 1].
        # Values in (0, 1] are assumed already normalized.
        # Values > 1 and <= 255 are treated as uint8-range.
        # Values > 255 are rescaled by the per-array maximum.
        arr_max = float(arr.max())
        if arr_max > 1.0:
            arr = arr / 255.0 if arr_max <= 255.0 else arr / arr_max

        return arr

    def _adjust_channels(self, arr: np.ndarray) -> np.ndarray:
        """Pad or truncate the channel dimension to match ``in_channels``.

        Args:
            arr: Image array with shape ``(C, H, W)``.

        Returns:
            Adjusted array with shape ``(in_channels, H, W)``.
        """
        if self.in_channels is None:
            return arr

        current_channels = arr.shape[0]
        if current_channels == self.in_channels:
            return arr
        elif current_channels > self.in_channels:
            return arr[: self.in_channels]
        else:
            padded = np.zeros(
                (self.in_channels, arr.shape[1], arr.shape[2]), dtype=arr.dtype
            )
            padded[:current_channels] = arr
            return padded

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        ext = os.path.splitext(path)[1].lower()

        if ext in _GEOTIFF_EXTENSIONS and RASTERIO_AVAILABLE:
            arr = self._load_geotiff(path)
        else:
            arr = self._load_standard_image(path)

        arr = self._adjust_channels(arr)

        image = torch.from_numpy(arr)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_image_dataset(
    data_dir: str,
    extensions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Scan an ImageFolder-style directory to discover classes and images.

    The directory should have one subdirectory per class, each containing
    images of that class::

        data_dir/
            ClassA/
                img1.jpg
                img2.jpg
            ClassB/
                img3.tif

    Args:
        data_dir: Root directory of the dataset.
        extensions: List of file extensions to include (without dot).
            Defaults to ``["jpg", "jpeg", "png", "bmp", "tif", "tiff"]``.

    Returns:
        Dictionary with keys:
            - ``image_paths`` (list[str]): Paths to all discovered images.
            - ``labels`` (list[int]): Integer label for each image.
            - ``class_names`` (list[str]): Sorted list of class names.
            - ``class_to_idx`` (dict[str, int]): Mapping from class name to index.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
        ValueError: If no classes or no images are found.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]

    # Discover classes (sorted for reproducibility)
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    if not class_names:
        raise ValueError(f"No class subdirectories found in {data_dir}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths: List[str] = []
    labels: List[int] = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for ext in extensions:
            pattern = os.path.join(class_dir, f"*.{ext}")
            paths = sorted(glob.glob(pattern))
            image_paths.extend(paths)
            labels.extend([class_to_idx[class_name]] * len(paths))

    if not image_paths:
        # Auto-detect nested directory (e.g., ZIP extraction produces
        # data_dir/wrapper/ClassA/... instead of data_dir/ClassA/...).
        # Walk up to 3 levels deep to find an ImageFolder-style root.
        max_depth = 3
        for root, dirs, _files in os.walk(data_dir):
            rel_path = os.path.relpath(root, data_dir)
            depth = 0 if rel_path == os.curdir else rel_path.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue

            if not dirs:
                continue

            # Check if any subdirectory of *root* contains matching images
            has_images = False
            for d in dirs:
                candidate = os.path.join(root, d)
                for ext in extensions:
                    if glob.glob(os.path.join(candidate, f"*.{ext}")):
                        has_images = True
                        break
                if has_images:
                    break

            if has_images and root != data_dir:
                print(f"Auto-detected nested dataset at: {root}")
                return load_image_dataset(root, extensions)

        raise ValueError(f"No images found in {data_dir} with extensions {extensions}")

    print(f"Found {len(image_paths)} images in {len(class_names)} classes")
    for name in class_names:
        count = labels.count(class_to_idx[name])
        print(f"  {name}: {count}")

    return {
        "image_paths": image_paths,
        "labels": labels,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
    }


def train_image_classifier(
    data_dir: str,
    model_name: str = "resnet50",
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 64,
    in_channels: int = 3,
    test_size: float = 0.2,
    val_size: float = 0.2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    output_dir: str = "output",
    num_workers: int = 4,
    seed: int = 42,
    accelerator: str = "auto",
    devices: str = "auto",
    patience: int = 10,
    extensions: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train an image classifier on an ImageFolder-style dataset.

    This is a single-function API: point it at a directory of images organized
    by class and get back a trained model with evaluation results.

    Args:
        data_dir: Root directory with one subdirectory per class.
        model_name: Name of the timm model (e.g., ``"resnet50"``,
            ``"efficientnet_b0"``, ``"vit_tiny_patch16_224"``).
        num_epochs: Maximum number of training epochs.
        batch_size: Batch size for training and evaluation.
        learning_rate: Learning rate for the AdamW optimizer.
        weight_decay: Weight decay for the optimizer.
        image_size: Target image size (height and width).
        in_channels: Number of input channels (3 for RGB, 4+ for multispectral).
        test_size: Fraction of data reserved for testing.
        val_size: Fraction of remaining data reserved for validation.
        pretrained: Whether to use pretrained weights.
        freeze_backbone: If ``True``, freeze backbone and only train the head.
        output_dir: Directory to save model checkpoints and logs.
        num_workers: Number of data-loading workers.
        seed: Random seed for reproducibility.
        accelerator: PyTorch Lightning accelerator (``"auto"``, ``"gpu"``,
            ``"cpu"``).
        devices: Devices to use (``"auto"`` for automatic selection).
        patience: Early-stopping patience (epochs without improvement).
        extensions: Image file extensions to include.
        **kwargs: Extra arguments forwarded to ``lightning.pytorch.Trainer``.

    Returns:
        Dictionary with keys:
            - ``model``: Trained :class:`~geoai.timm_train.TimmClassifier`.
            - ``trainer``: The PyTorch Lightning ``Trainer`` instance.
            - ``class_names``: List of class name strings.
            - ``train_dataset``: Training :class:`ImageDataset`.
            - ``val_dataset``: Validation :class:`ImageDataset`.
            - ``test_dataset``: Test :class:`ImageDataset`.
            - ``checkpoint_path``: Path to the best model checkpoint.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
        ImportError: If PyTorch Lightning is not installed.
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "PyTorch Lightning is required. Install with: pip install lightning"
        )
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )

    # Load dataset
    dataset_info = load_image_dataset(data_dir, extensions=extensions)
    image_paths = dataset_info["image_paths"]
    labels = dataset_info["labels"]
    class_names = dataset_info["class_names"]
    num_classes = len(class_names)

    # Split into train+val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    # Split train+val into train and val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    print(
        f"\nSplit: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test"
    )

    # Create datasets
    train_dataset = ImageDataset(
        train_paths, train_labels, image_size=image_size, in_channels=in_channels
    )
    val_dataset = ImageDataset(
        val_paths, val_labels, image_size=image_size, in_channels=in_channels
    )
    test_dataset = ImageDataset(
        test_paths, test_labels, image_size=image_size, in_channels=in_channels
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    model = TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
    )

    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_{{epoch:02d}}_{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=patience,
        mode="max",
        verbose=True,
    )

    logger = CSVLogger(model_dir, name="lightning_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        **kwargs,
    )

    # Train
    print(f"\nTraining {model_name} for up to {num_epochs} epochs...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print("\nEvaluating on test set...")
    trainer.test(model, dataloaders=test_loader)

    best_path = checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_path}")

    return {
        "model": model,
        "trainer": trainer,
        "class_names": class_names,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "checkpoint_path": best_path,
    }


def predict_images(
    model: Union[TimmClassifier, nn.Module],
    image_paths: List[str],
    class_names: Optional[List[str]] = None,
    image_size: int = 64,
    in_channels: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference on a list of images.

    Args:
        model: Trained model (typically a :class:`~geoai.timm_train.TimmClassifier`).
        image_paths: Paths to images to classify.
        class_names: Optional list of class names for labelling predictions.
        image_size: Image size that the model was trained with.
        in_channels: Number of channels the model expects.
        batch_size: Inference batch size.
        num_workers: Data-loading workers.
        device: Device string (``"cuda"``, ``"cpu"``). Auto-detected when ``None``.

    Returns:
        Dictionary with keys:
            - ``predictions``: Array of predicted class indices.
            - ``probabilities``: Array of class probabilities.
            - ``predicted_classes``: List of class name strings (if
              ``class_names`` provided).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_labels = [0] * len(image_paths)
    dataset = ImageDataset(
        image_paths, dummy_labels, image_size=image_size, in_channels=in_channels
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    model = model.to(device)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)

    result: Dict[str, Any] = {
        "predictions": predictions,
        "probabilities": probabilities,
    }

    if class_names is not None:
        result["predicted_classes"] = [class_names[p] for p in predictions]

    return result


def evaluate_classifier(
    model: Union[TimmClassifier, nn.Module],
    dataset: Dataset,
    class_names: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained classifier on a dataset.

    Args:
        model: Trained model.
        dataset: An :class:`ImageDataset` (or compatible) with ground-truth labels.
        class_names: List of class name strings.
        batch_size: Evaluation batch size.
        num_workers: Data-loading workers.
        device: Device string. Auto-detected when ``None``.

    Returns:
        Dictionary with keys:
            - ``accuracy`` (float): Overall accuracy.
            - ``classification_report`` (str): sklearn classification report.
            - ``confusion_matrix`` (np.ndarray): Confusion matrix.
            - ``per_class_accuracy`` (dict): Accuracy per class name.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() > 0:
            per_class_acc[name] = float((y_pred[mask] == idx).mean())
        else:
            per_class_acc[name] = 0.0

    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"\n{report}")

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
    }


def plot_training_history(
    log_dir: str,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot training and validation loss/accuracy from Lightning CSV logs.

    Args:
        log_dir: Path to the Lightning CSV logger output directory. The
            function searches for ``metrics.csv`` inside this directory tree.
        figsize: Figure size as ``(width, height)``.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.

    Raises:
        FileNotFoundError: If no ``metrics.csv`` is found.
    """
    import pandas as pd

    # Find metrics.csv
    csv_path = None
    for root, _dirs, files in os.walk(log_dir):
        if "metrics.csv" in files:
            csv_path = os.path.join(root, "metrics.csv")
            break

    if csv_path is None:
        raise FileNotFoundError(f"No metrics.csv found under {log_dir}")

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    ax = axes[0]
    if "train_loss_epoch" in df.columns:
        loss_df = df.dropna(subset=["train_loss_epoch"])
        ax.plot(loss_df["epoch"], loss_df["train_loss_epoch"], label="Train Loss")
    if "val_loss" in df.columns:
        loss_df = df.dropna(subset=["val_loss"])
        ax.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if "train_acc_epoch" in df.columns:
        acc_df = df.dropna(subset=["train_acc_epoch"])
        ax.plot(acc_df["epoch"], acc_df["train_acc_epoch"], label="Train Acc")
    if "val_acc" in df.columns:
        acc_df = df.dropna(subset=["val_acc"])
        ax.plot(acc_df["epoch"], acc_df["val_acc"], label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "Blues",
    normalize: bool = False,
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array from :func:`evaluate_classifier`.
        class_names: List of class name strings.
        figsize: Figure size. Auto-scaled from class count when ``None``.
        cmap: Matplotlib colormap name.
        normalize: If ``True``, show percentages instead of counts.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".1%"
    else:
        cm_plot = cm
        fmt = "d"

    n = len(class_names)
    if figsize is None:
        size = max(8, n * 0.8)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    cm_max = cm_plot.max()
    thresh = cm_max / 2.0 if cm_max > 0 else 0.5
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    return fig


def plot_predictions(
    image_paths: List[str],
    predictions: np.ndarray,
    true_labels: List[int],
    class_names: List[str],
    num_images: int = 20,
    ncols: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    probabilities: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Visualize model predictions on a grid of images.

    Args:
        image_paths: Paths to the images.
        predictions: Predicted class indices.
        true_labels: Ground-truth class indices.
        class_names: List of class name strings.
        num_images: Maximum number of images to display.
        ncols: Number of columns in the grid.
        figsize: Figure size. Auto-scaled when ``None``.
        probabilities: Optional array of class probabilities for showing
            confidence values.

    Returns:
        Matplotlib :class:`~matplotlib.figure.Figure`.

    Raises:
        ImportError: If PIL (Pillow) is not installed.
    """
    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required to display images. " "Install with: pip install Pillow"
        )

    num_images = min(num_images, len(image_paths))
    nrows = (num_images + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx >= num_images:
            ax.axis("off")
            continue

        # Load image for display
        img = Image.open(image_paths[idx]).convert("RGB")
        ax.imshow(img)

        pred = int(predictions[idx])
        true = int(true_labels[idx])
        pred_name = class_names[pred]
        true_name = class_names[true]
        color = "green" if pred == true else "red"

        title = f"Pred: {pred_name}\nTrue: {true_name}"
        if probabilities is not None:
            conf = probabilities[idx][pred] * 100
            title += f"\n({conf:.1f}%)"

        ax.set_title(title, color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig
