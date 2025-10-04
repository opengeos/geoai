"""Module for training and fine-tuning models using timm (PyTorch Image Models) with remote sensing imagery."""

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
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


def get_timm_model(
    model_name: str = "resnet50",
    num_classes: int = 10,
    in_channels: int = 3,
    pretrained: bool = True,
    features_only: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Create a timm model with custom input channels for remote sensing imagery.

    Args:
        model_name (str): Name of the timm model (e.g., 'resnet50', 'efficientnet_b0',
            'vit_base_patch16_224', 'convnext_base').
        num_classes (int): Number of output classes for classification.
        in_channels (int): Number of input channels (3 for RGB, 4 for RGBN, etc.).
        pretrained (bool): Whether to use pretrained weights.
        features_only (bool): If True, return feature extraction model without classifier.
        **kwargs: Additional arguments to pass to timm.create_model.

    Returns:
        nn.Module: Configured timm model.

    Raises:
        ImportError: If timm is not installed.
        ValueError: If model_name is not available in timm.
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required. Install it with: pip install timm")

    # Check if model exists
    if model_name not in timm.list_models():
        available_models = timm.list_models(pretrained=True)[:10]
        raise ValueError(
            f"Model '{model_name}' not found in timm. "
            f"First 10 available models: {available_models}. "
            f"See all models at: https://github.com/huggingface/pytorch-image-models"
        )

    # Create base model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes if not features_only else 0,
        in_chans=in_channels,
        features_only=features_only,
        **kwargs,
    )

    return model


def modify_first_conv_for_channels(
    model: nn.Module,
    in_channels: int,
    pretrained_channels: int = 3,
) -> nn.Module:
    """
    Modify the first convolutional layer of a model to accept different number of input channels.

    This is useful when you have a pretrained model with 3 input channels but want to use
    imagery with more channels (e.g., 4 for RGBN, or more for multispectral).

    Args:
        model (nn.Module): PyTorch model to modify.
        in_channels (int): Desired number of input channels.
        pretrained_channels (int): Number of channels in pretrained weights (usually 3).

    Returns:
        nn.Module: Modified model with updated first conv layer.
    """
    if in_channels == pretrained_channels:
        return model

    # Find the first conv layer (different models have different architectures)
    first_conv_name = None
    first_conv = None

    # Common patterns for first conv layers
    possible_names = ["conv1", "conv_stem", "patch_embed.proj", "stem.conv1"]

    for name in possible_names:
        try:
            parts = name.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
            if isinstance(module, nn.Conv2d):
                first_conv_name = name
                first_conv = module
                break
        except AttributeError:
            continue

    if first_conv is None:
        # Fallback: search recursively
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv_name = name
                first_conv = module
                break

    if first_conv is None:
        raise ValueError("Could not find first convolutional layer in model")

    # Create new conv layer with desired input channels
    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )

    # Initialize weights
    with torch.no_grad():
        if pretrained_channels == 3 and in_channels > 3:
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = first_conv.weight

            # Initialize additional channels with mean of RGB weights
            mean_weight = first_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i : i + 1, :, :] = mean_weight
        else:
            # Generic initialization
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="relu"
            )

        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias

    # Replace the first conv layer
    parts = first_conv_name.split(".")
    if len(parts) == 1:
        setattr(model, first_conv_name, new_conv)
    else:
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    return model


class TimmClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for image classification using timm models.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        in_channels: int = 3,
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        loss_fn: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
        **model_kwargs: Any,
    ):
        """
        Initialize TimmClassifier.

        Args:
            model_name (str): Name of timm model.
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
            pretrained (bool): Use pretrained weights.
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for optimizer.
            freeze_backbone (bool): Freeze backbone weights during training.
            loss_fn (nn.Module, optional): Custom loss function. Defaults to CrossEntropyLoss.
            class_weights (torch.Tensor, optional): Class weights for loss function.
            **model_kwargs: Additional arguments for timm model.
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install it with: pip install timm")

        self.save_hyperparameters()

        self.model = get_timm_model(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            **model_kwargs,
        )

        if freeze_backbone:
            self._freeze_backbone()

        # Set up loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.model.named_parameters():
            if "fc" not in name and "head" not in name and "classifier" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

        return loss

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


class RemoteSensingDataset(Dataset):
    """
    Dataset for remote sensing imagery classification.

    This dataset handles loading raster images and their corresponding labels
    for training classification models.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        num_channels: Optional[int] = None,
    ):
        """
        Initialize RemoteSensingDataset.

        Args:
            image_paths (List[str]): List of paths to image files.
            labels (List[int]): List of integer labels corresponding to images.
            transform (callable, optional): Transform to apply to images.
            num_channels (int, optional): Number of channels to use. If None, uses all.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.num_channels = num_channels

        if len(image_paths) != len(labels):
            raise ValueError("Number of images must match number of labels")

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

        # Get label
        label = self.labels[idx]

        # Convert to tensor
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def train_timm_classifier(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    model_name: str = "resnet50",
    num_classes: int = 10,
    in_channels: int = 3,
    pretrained: bool = True,
    output_dir: str = "output",
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    freeze_backbone: bool = False,
    class_weights: Optional[List[float]] = None,
    accelerator: str = "auto",
    devices: str = "auto",
    monitor_metric: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 1,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> TimmClassifier:
    """
    Train a timm-based classifier on remote sensing imagery.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset, optional): Validation dataset.
        test_dataset (Dataset, optional): Test dataset.
        model_name (str): Name of timm model to use.
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        pretrained (bool): Use pretrained weights.
        output_dir (str): Directory to save outputs.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        num_workers (int): Number of data loading workers.
        freeze_backbone (bool): Freeze backbone during training.
        class_weights (List[float], optional): Class weights for loss.
        accelerator (str): Accelerator type ('auto', 'gpu', 'cpu').
        devices (str): Devices to use.
        monitor_metric (str): Metric to monitor for checkpointing.
        mode (str): 'min' or 'max' for monitor_metric.
        patience (int): Early stopping patience.
        save_top_k (int): Number of best models to save.
        checkpoint_path (str, optional): Path to checkpoint to resume from.
        **kwargs: Additional arguments for PyTorch Lightning Trainer.

    Returns:
        TimmClassifier: Trained model.

    Raises:
        ImportError: If PyTorch Lightning is not installed.
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
    model = TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        class_weights=weight_tensor,
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
        filename=f"{model_name}_{{epoch:02d}}_{{val_loss:.4f}}",
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
    print(f"Training {model_name} for {num_epochs} epochs...")
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

    return model


def predict_with_timm(
    model: Union[TimmClassifier, nn.Module],
    image_paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
    return_probabilities: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions on images using a trained timm model.

    Args:
        model: Trained model (TimmClassifier or nn.Module).
        image_paths: List of paths to images.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        device: Device to use ('cuda', 'cpu', etc.). Auto-detected if None.
        return_probabilities: If True, return both predictions and probabilities.

    Returns:
        predictions: Array of predicted class indices.
        probabilities (optional): Array of class probabilities if return_probabilities=True.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy labels for dataset
    dummy_labels = [0] * len(image_paths)
    dataset = RemoteSensingDataset(image_paths, dummy_labels)

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

            if isinstance(model, TimmClassifier):
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


def list_timm_models(
    filter: str = "",
    pretrained: bool = False,
    limit: Optional[int] = None,
) -> List[str]:
    """
    List available timm models.

    Args:
        filter (str): Filter models by name pattern (e.g., 'resnet', 'efficientnet').
            The filter supports wildcards. If no wildcards are provided, '*' is added automatically.
        pretrained (bool): Only show models with pretrained weights.
        limit (int, optional): Maximum number of models to return.

    Returns:
        List of model names.

    Raises:
        ImportError: If timm is not installed.
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required. Install it with: pip install timm")

    # Add wildcards if not present in filter
    if filter and "*" not in filter:
        filter = f"*{filter}*"

    models = timm.list_models(filter=filter, pretrained=pretrained)

    if limit is not None:
        models = models[:limit]

    return models
