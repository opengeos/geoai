"""Module for training semantic segmentation models using timm encoders with PyTorch Lightning."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


class TimmSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for semantic segmentation using timm encoders with SMP decoders,
    or pure timm models from Hugging Face Hub.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        architecture: str = "unet",
        num_classes: int = 2,
        in_channels: int = 3,
        encoder_weights: str = "imagenet",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_encoder: bool = False,
        loss_fn: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_timm_model: bool = False,
        timm_model_name: Optional[str] = None,
        **decoder_kwargs: Any,
    ):
        """
        Initialize TimmSegmentationModel.

        Args:
            encoder_name (str): Name of encoder (e.g., 'resnet50', 'efficientnet_b0').
            architecture (str): Segmentation architecture ('unet', 'unetplusplus', 'deeplabv3',
                'deeplabv3plus', 'fpn', 'pspnet', 'linknet', 'manet', 'pan').
                Ignored if use_timm_model=True.
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
            encoder_weights (str): Pretrained weights for encoder ('imagenet', 'ssl', 'swsl', None).
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            freeze_encoder (bool): Freeze encoder weights during training.
            loss_fn (nn.Module, optional): Custom loss function. Defaults to CrossEntropyLoss.
            class_weights (torch.Tensor, optional): Class weights for loss function.
            use_timm_model (bool): If True, load a complete segmentation model from timm/HF Hub
                instead of using SMP architecture. Defaults to False.
            timm_model_name (str, optional): Name or path of timm model from HF Hub
                (e.g., 'hf-hub:timm/segformer_b0.ade_512x512' or 'nvidia/mit-b0').
                Only used if use_timm_model=True.
            **decoder_kwargs: Additional arguments for decoder (only used with SMP).
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install it with: pip install timm")

        self.save_hyperparameters()

        # Check if using a pure timm model from HF Hub
        if use_timm_model:
            if timm_model_name is None:
                timm_model_name = encoder_name

            # Load model from timm (supports HF Hub with 'hf-hub:' prefix)
            try:
                self.model = timm.create_model(
                    timm_model_name,
                    pretrained=True if encoder_weights else False,
                    num_classes=num_classes,
                    in_chans=in_channels,
                )
                print(f"Loaded timm model: {timm_model_name}")
            except Exception as e:
                raise ValueError(
                    f"Failed to load timm model '{timm_model_name}'. "
                    f"Error: {str(e)}. "
                    f"For HF Hub models, use format 'hf-hub:username/model-name' or 'hf_hub:username/model-name'."
                )
        else:
            # Use SMP architecture with timm encoder
            if not SMP_AVAILABLE:
                raise ImportError(
                    "segmentation-models-pytorch is required. "
                    "Install it with: pip install segmentation-models-pytorch"
                )

            # Create segmentation model with timm encoder using smp.create_model
            try:
                self.model = smp.create_model(
                    arch=architecture,
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=num_classes,
                    **decoder_kwargs,
                )
            except Exception as e:
                # Provide helpful error message
                available_archs = [
                    "unet",
                    "unetplusplus",
                    "manet",
                    "linknet",
                    "fpn",
                    "pspnet",
                    "deeplabv3",
                    "deeplabv3plus",
                    "pan",
                    "upernet",
                ]
                raise ValueError(
                    f"Failed to create model with architecture '{architecture}' and encoder '{encoder_name}'. "
                    f"Error: {str(e)}. "
                    f"Available architectures include: {', '.join(available_archs)}. "
                    f"Please check the segmentation-models-pytorch documentation for supported combinations."
                )

        if freeze_encoder:
            self._freeze_encoder()

        # Set up loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _freeze_encoder(self):
        """Freeze encoder weights."""
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            # For pure timm models without separate encoder
            if not self.hparams.use_timm_model:
                raise ValueError("Model does not have an encoder attribute to freeze")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate IoU
        pred = torch.argmax(logits, dim=1)
        iou = self._compute_iou(pred, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate IoU
        pred = torch.argmax(logits, dim=1)
        iou = self._compute_iou(pred, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate IoU
        pred = torch.argmax(logits, dim=1)
        iou = self._compute_iou(pred, y)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_iou", iou, on_epoch=True)

        return loss

    def _compute_iou(self, pred, target, smooth=1e-6):
        """Compute mean IoU across all classes."""
        num_classes = self.hparams.num_classes
        ious = []

        for cls in range(num_classes):
            pred_cls = pred == cls
            target_cls = target == cls

            intersection = (pred_cls & target_cls).float().sum()
            union = (pred_cls | target_cls).float().sum()

            if union == 0:
                continue

            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou)

        return (
            torch.stack(ious).mean() if ious else torch.tensor(0.0, device=pred.device)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"predictions": preds, "probabilities": probs}


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation with remote sensing imagery.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable] = None,
        num_channels: Optional[int] = None,
    ):
        """
        Initialize SegmentationDataset.

        Args:
            image_paths (List[str]): List of paths to image files.
            mask_paths (List[str]): List of paths to mask files.
            transform (callable, optional): Transform to apply to images and masks.
            num_channels (int, optional): Number of channels to use. If None, uses all.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_channels = num_channels

        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images must match number of masks")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import rasterio

        # Load image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()  # Shape: (C, H, W)

            # Handle channel selection
            if self.num_channels is not None and image.shape[0] != self.num_channels:
                if image.shape[0] > self.num_channels:
                    image = image[: self.num_channels]
                else:
                    # Pad with zeros if needed
                    padded = np.zeros(
                        (self.num_channels, image.shape[1], image.shape[2])
                    )
                    padded[: image.shape[0]] = image
                    image = padded

            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

            image = image.astype(np.float32)

        # Load mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)  # Shape: (H, W)
            mask = mask.astype(np.int64)

        # Convert to tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


def train_timm_segmentation(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    encoder_name: str = "resnet50",
    architecture: str = "unet",
    num_classes: int = 2,
    in_channels: int = 3,
    encoder_weights: str = "imagenet",
    output_dir: str = "output",
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    freeze_encoder: bool = False,
    class_weights: Optional[List[float]] = None,
    accelerator: str = "auto",
    devices: str = "auto",
    monitor_metric: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 1,
    checkpoint_path: Optional[str] = None,
    use_timm_model: bool = False,
    timm_model_name: Optional[str] = None,
    **kwargs: Any,
) -> TimmSegmentationModel:
    """
    Train a semantic segmentation model using timm encoder.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset, optional): Validation dataset.
        test_dataset (Dataset, optional): Test dataset.
        encoder_name (str): Name of timm encoder.
        architecture (str): Segmentation architecture.
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        encoder_weights (str): Pretrained weights for encoder.
        output_dir (str): Directory to save outputs.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        num_workers (int): Number of data loading workers.
        freeze_encoder (bool): Freeze encoder during training.
        class_weights (List[float], optional): Class weights for loss.
        accelerator (str): Accelerator type ('auto', 'gpu', 'cpu').
        devices (str): Devices to use.
        monitor_metric (str): Metric to monitor for checkpointing.
        mode (str): 'min' or 'max' for monitor_metric.
        patience (int): Early stopping patience.
        save_top_k (int): Number of best models to save.
        checkpoint_path (str, optional): Path to checkpoint to resume from.
        use_timm_model (bool): Load complete segmentation model from timm/HF Hub.
        timm_model_name (str, optional): Model name from HF Hub (e.g., 'hf-hub:nvidia/mit-b0').
        **kwargs: Additional arguments for PyTorch Lightning Trainer.

    Returns:
        TimmSegmentationModel: Trained model.
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "PyTorch Lightning is required. Install it with: pip install lightning"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Convert class weights to tensor if provided
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Create model
    model = TimmSegmentationModel(
        encoder_name=encoder_name,
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        encoder_weights=encoder_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_encoder=freeze_encoder,
        class_weights=weight_tensor,
        use_timm_model=use_timm_model,
        timm_model_name=timm_model_name,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Set up callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{encoder_name}_{architecture}_{{epoch:02d}}_{{val_loss:.4f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    # Set up logger
    logger = CSVLogger(model_dir, name="lightning_logs")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        **kwargs,
    )

    # Train model
    print(f"Training {encoder_name} {architecture} for {num_epochs} epochs...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )

    # Test if test dataset provided
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print("\nTesting model on test set...")
        trainer.test(model, dataloaders=test_loader)

    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")

    # Save training history in compatible format
    metrics = trainer.logged_metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "epochs": [],
    }

    # Extract metrics from logger
    import pandas as pd
    import glob

    csv_files = glob.glob(
        os.path.join(model_dir, "lightning_logs", "**", "metrics.csv"), recursive=True
    )
    if csv_files:
        df = pd.read_csv(csv_files[0])

        # Group by epoch to get epoch-level metrics
        epoch_data = df.groupby("epoch").last().reset_index()

        if "train_loss_epoch" in epoch_data.columns:
            history["train_loss"] = epoch_data["train_loss_epoch"].dropna().tolist()
        if "val_loss" in epoch_data.columns:
            history["val_loss"] = epoch_data["val_loss"].dropna().tolist()
        if "val_iou" in epoch_data.columns:
            history["val_iou"] = epoch_data["val_iou"].dropna().tolist()
        if "epoch" in epoch_data.columns:
            history["epochs"] = epoch_data["epoch"].dropna().tolist()

    # Save history
    history_path = os.path.join(model_dir, "training_history.pth")
    torch.save(history, history_path)
    print(f"Training history saved to: {history_path}")

    return model


def predict_segmentation(
    model: Union[TimmSegmentationModel, nn.Module],
    image_paths: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    device: Optional[str] = None,
    return_probabilities: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions on images using a trained segmentation model.

    Args:
        model: Trained model.
        image_paths: List of paths to images.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        device: Device to use ('cuda', 'cpu', etc.). Auto-detected if None.
        return_probabilities: If True, return both predictions and probabilities.

    Returns:
        predictions: Array of predicted segmentation masks.
        probabilities (optional): Array of class probabilities if return_probabilities=True.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy masks for dataset
    dummy_masks = image_paths  # Use image paths as placeholders
    dataset = SegmentationDataset(image_paths, dummy_masks)

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
        for images, _ in tqdm(loader, desc="Making predictions"):
            images = images.to(device)

            if isinstance(model, TimmSegmentationModel):
                logits = model(images)
            else:
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            if return_probabilities:
                all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)

    if return_probabilities:
        probabilities = np.concatenate(all_probs)
        return predictions, probabilities

    return predictions


def train_timm_segmentation_model(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    input_format: str = "directory",
    encoder_name: str = "resnet50",
    architecture: str = "unet",
    encoder_weights: str = "imagenet",
    num_channels: int = 3,
    num_classes: int = 2,
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    freeze_encoder: bool = False,
    monitor_metric: str = "val_iou",
    mode: str = "max",
    patience: int = 10,
    save_top_k: int = 1,
    verbose: bool = True,
    device: Optional[str] = None,
    use_timm_model: bool = False,
    timm_model_name: Optional[str] = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """
    Train a semantic segmentation model using timm encoder (simplified interface).

    This is a simplified function that takes image and label directories and handles
    the dataset creation automatically, similar to train_segmentation_model.

    Args:
        images_dir (str): Directory containing image GeoTIFF files (for 'directory' format),
            or root directory containing images/ subdirectory (for 'yolo' format),
            or directory containing images (for 'coco' format).
        labels_dir (str): Directory containing label GeoTIFF files (for 'directory' format),
            or path to COCO annotations JSON file (for 'coco' format),
            or not used (for 'yolo' format - labels are in images_dir/labels/).
        output_dir (str): Directory to save model checkpoints and results.
        input_format (str): Input data format - 'directory' (default), 'coco', or 'yolo'.
            - 'directory': Standard directory structure with separate images_dir and labels_dir
            - 'coco': COCO JSON format (labels_dir should be path to instances.json)
            - 'yolo': YOLO format (images_dir is root with images/ and labels/ subdirectories)
        encoder_name (str): Name of timm encoder (e.g., 'resnet50', 'efficientnet_b3').
        architecture (str): Segmentation architecture ('unet', 'unetplusplus', 'deeplabv3',
            'deeplabv3plus', 'fpn', 'pspnet', 'linknet', 'manet', 'pan').
        encoder_weights (str): Pretrained weights ('imagenet', 'ssl', 'swsl', None).
        num_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        val_split (float): Validation split ratio (0-1).
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of data loading workers.
        freeze_encoder (bool): Freeze encoder during training.
        monitor_metric (str): Metric to monitor ('val_loss' or 'val_iou').
        mode (str): 'min' for loss, 'max' for metrics.
        patience (int): Early stopping patience.
        save_top_k (int): Number of best models to save.
        verbose (bool): Print training progress.
        device (str, optional): Device to use. Auto-detected if None.
        use_timm_model (bool): Load complete segmentation model from timm/HF Hub.
        timm_model_name (str, optional): Model name from HF Hub (e.g., 'hf-hub:nvidia/mit-b0').
        **kwargs: Additional arguments for training.

    Returns:
        torch.nn.Module: Trained model.
    """
    import glob
    from sklearn.model_selection import train_test_split
    from .train import parse_coco_annotations, parse_yolo_annotations

    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "PyTorch Lightning is required. Install it with: pip install lightning"
        )

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get image and label paths based on input format
    if input_format.lower() == "coco":
        # Parse COCO format annotations
        if verbose:
            print(f"Loading COCO format annotations from {labels_dir}")
        # For COCO format, labels_dir is path to instances.json
        # Labels are typically in a "labels" directory parallel to "annotations"
        coco_root = os.path.dirname(os.path.dirname(labels_dir))  # Go up two levels
        labels_directory = os.path.join(coco_root, "labels")
        image_paths, label_paths = parse_coco_annotations(
            labels_dir, images_dir, labels_directory
        )
    elif input_format.lower() == "yolo":
        # Parse YOLO format annotations
        if verbose:
            print(f"Loading YOLO format data from {images_dir}")
        image_paths, label_paths = parse_yolo_annotations(images_dir)
    else:
        # Default: directory format
        image_paths = sorted(
            glob.glob(os.path.join(images_dir, "*.tif"))
            + glob.glob(os.path.join(images_dir, "*.tiff"))
        )
        label_paths = sorted(
            glob.glob(os.path.join(labels_dir, "*.tif"))
            + glob.glob(os.path.join(labels_dir, "*.tiff"))
        )

    if len(image_paths) == 0:
        raise ValueError(f"No images found")
    if len(label_paths) == 0:
        raise ValueError(f"No labels found")
    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"Number of images ({len(image_paths)}) doesn't match "
            f"number of labels ({len(label_paths)})"
        )

    if verbose:
        print(f"Found {len(image_paths)} image-label pairs")

    # Split into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths, test_size=val_split, random_state=seed
    )

    if verbose:
        print(f"Training samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")

    # Create datasets
    train_dataset = SegmentationDataset(
        image_paths=train_images,
        mask_paths=train_labels,
        num_channels=num_channels,
    )

    val_dataset = SegmentationDataset(
        image_paths=val_images,
        mask_paths=val_labels,
        num_channels=num_channels,
    )

    # Train model
    model = train_timm_segmentation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        encoder_name=encoder_name,
        architecture=architecture,
        num_classes=num_classes,
        in_channels=num_channels,
        encoder_weights=encoder_weights,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        freeze_encoder=freeze_encoder,
        accelerator="auto" if device is None else device,
        monitor_metric=monitor_metric,
        mode=mode,
        patience=patience,
        save_top_k=save_top_k,
        use_timm_model=use_timm_model,
        timm_model_name=timm_model_name,
        **kwargs,
    )

    if verbose:
        print(f"\nTraining completed. Model saved to {output_dir}")

    return model.model  # Return the underlying model


def timm_semantic_segmentation(
    input_path: str,
    output_path: str,
    model_path: str,
    encoder_name: str = "resnet50",
    architecture: str = "unet",
    num_channels: int = 3,
    num_classes: int = 2,
    window_size: int = 512,
    overlap: int = 256,
    batch_size: int = 4,
    device: Optional[str] = None,
    quiet: bool = False,
    use_timm_model: bool = False,
    timm_model_name: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Perform semantic segmentation on a raster using a trained timm model.

    This function performs inference on a GeoTIFF using a sliding window approach
    and saves the result as a georeferenced raster.

    Args:
        input_path (str): Path to input GeoTIFF file.
        output_path (str): Path to save output mask.
        model_path (str): Path to trained model checkpoint (.ckpt or .pth).
        encoder_name (str): Name of timm encoder used in training.
        architecture (str): Segmentation architecture used in training.
        num_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        window_size (int): Size of sliding window for inference.
        overlap (int): Overlap between adjacent windows.
        batch_size (int): Batch size for inference.
        device (str, optional): Device to use. Auto-detected if None.
        quiet (bool): If True, suppress progress messages.
        use_timm_model (bool): If True, model was trained with timm model from HF Hub.
        timm_model_name (str, optional): Model name from HF Hub used during training.
        **kwargs: Additional arguments.
    """
    import rasterio
    from rasterio.windows import Window

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if model_path.endswith(".ckpt"):
        model = TimmSegmentationModel.load_from_checkpoint(
            model_path,
            encoder_name=encoder_name,
            architecture=architecture,
            num_classes=num_classes,
            in_channels=num_channels,
            use_timm_model=use_timm_model,
            timm_model_name=timm_model_name,
        )
        model = model.model  # Get underlying model
    else:
        # Load state dict
        if use_timm_model:
            # Load pure timm model
            if timm_model_name is None:
                timm_model_name = encoder_name

            model = timm.create_model(
                timm_model_name,
                pretrained=False,
                num_classes=num_classes,
                in_chans=num_channels,
            )
        else:
            # Load SMP model
            import segmentation_models_pytorch as smp

            try:
                model = smp.create_model(
                    arch=architecture,
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=num_channels,
                    classes=num_classes,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create model with architecture '{architecture}' and encoder '{encoder_name}'. "
                    f"Error: {str(e)}"
                )

        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model = model.to(device)

    # Read input raster
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        height, width = src.shape

        # Calculate number of windows
        stride = window_size - overlap
        n_rows = int(np.ceil((height - overlap) / stride))
        n_cols = int(np.ceil((width - overlap) / stride))

        if not quiet:
            print(f"Processing {n_rows} x {n_cols} = {n_rows * n_cols} windows")

        # Initialize output array (use int32 to avoid overflow during accumulation)
        output = np.zeros((height, width), dtype=np.int32)
        count = np.zeros((height, width), dtype=np.int32)

        # Process windows
        with torch.no_grad():
            for i in tqdm(range(n_rows), disable=quiet, desc="Processing rows"):
                for j in range(n_cols):
                    # Calculate window bounds
                    row_start = i * stride
                    col_start = j * stride
                    row_end = min(row_start + window_size, height)
                    col_end = min(col_start + window_size, width)

                    # Read window
                    window = Window(
                        col_start, row_start, col_end - col_start, row_end - row_start
                    )
                    img = src.read(window=window)

                    # Handle channel selection
                    if img.shape[0] > num_channels:
                        img = img[:num_channels]
                    elif img.shape[0] < num_channels:
                        padded = np.zeros((num_channels, img.shape[1], img.shape[2]))
                        padded[: img.shape[0]] = img
                        img = padded

                    # Normalize
                    if img.max() > 1.0:
                        img = img / 255.0
                    img = img.astype(np.float32)

                    # Pad if necessary
                    h, w = img.shape[1], img.shape[2]
                    if h < window_size or w < window_size:
                        padded = np.zeros(
                            (num_channels, window_size, window_size), dtype=np.float32
                        )
                        padded[:, :h, :w] = img
                        img = padded

                    # Predict
                    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
                    logits = model(img_tensor)
                    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                    # Crop to actual size
                    pred = pred[:h, :w]

                    # Add to output
                    output[row_start:row_end, col_start:col_end] += pred
                    count[row_start:row_end, col_start:col_end] += 1

        # Average overlapping predictions
        output = (output / np.maximum(count, 1)).astype(np.uint8)

    # Save output
    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(output, 1)

    if not quiet:
        print(f"Segmentation saved to {output_path}")


def push_timm_model_to_hub(
    model_path: str,
    repo_id: str,
    encoder_name: str = "resnet50",
    architecture: str = "unet",
    num_channels: int = 3,
    num_classes: int = 2,
    use_timm_model: bool = False,
    timm_model_name: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Push a trained timm segmentation model to Hugging Face Hub.

    Args:
        model_path (str): Path to trained model checkpoint (.ckpt or .pth).
        repo_id (str): Repository ID on HF Hub (e.g., 'username/model-name').
        encoder_name (str): Name of timm encoder used in training.
        architecture (str): Segmentation architecture used in training.
        num_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        use_timm_model (bool): If True, model was trained with pure timm model.
        timm_model_name (str, optional): Model name from HF Hub used during training.
        commit_message (str, optional): Commit message for the upload.
        private (bool): Whether to make the repository private.
        token (str, optional): HuggingFace API token. If None, uses logged-in token.
        **kwargs: Additional arguments for push_to_hub.

    Returns:
        str: URL of the uploaded model on HF Hub.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to push models. "
            "Install it with: pip install huggingface-hub"
        )

    # Load model
    if model_path.endswith(".ckpt"):
        lightning_model = TimmSegmentationModel.load_from_checkpoint(
            model_path,
            encoder_name=encoder_name,
            architecture=architecture,
            num_classes=num_classes,
            in_channels=num_channels,
            use_timm_model=use_timm_model,
            timm_model_name=timm_model_name,
        )
        model = lightning_model.model
    else:
        # Load state dict
        if use_timm_model:
            if timm_model_name is None:
                timm_model_name = encoder_name

            model = timm.create_model(
                timm_model_name,
                pretrained=False,
                num_classes=num_classes,
                in_chans=num_channels,
            )
        else:
            import segmentation_models_pytorch as smp

            model = smp.create_model(
                arch=architecture,
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=num_channels,
                classes=num_classes,
            )

        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Create repository if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repository creation note: {e}")

    # Save model configuration
    config = {
        "encoder_name": encoder_name,
        "architecture": architecture,
        "num_channels": num_channels,
        "num_classes": num_classes,
        "use_timm_model": use_timm_model,
        "timm_model_name": timm_model_name,
        "model_type": "timm_segmentation",
    }

    # Save model state dict to temporary file
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model state dict
        model_save_path = os.path.join(tmpdir, "model.pth")
        torch.save(model.state_dict(), model_save_path)

        # Save config
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Upload files
        if commit_message is None:
            commit_message = f"Upload {architecture} with {encoder_name} encoder"

        api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
            **kwargs,
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Model successfully pushed to: {url}")
    return url
