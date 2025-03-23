"""The module for training semantic segmentation models for classifying remote sensing imagery."""


def train_classifier(
    image_root,
    label_root,
    output_dir="output",
    in_channels=4,
    num_classes=14,
    epochs=20,
    img_size=256,
    batch_size=8,
    sample_size=500,
    model="unet",
    backbone="resnet50",
    weights=True,
    num_filters=3,
    loss="ce",
    class_weights=None,
    ignore_index=None,
    lr=0.001,
    patience=10,
    freeze_backbone=False,
    freeze_decoder=False,
    transforms=None,
    use_augmentation=False,
    seed=42,
    train_val_test_split=(0.6, 0.2, 0.2),
    accelerator="auto",
    devices="auto",
    logger=None,
    callbacks=None,
    log_every_n_steps=10,
    use_distributed_sampler=False,
    monitor_metric="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    checkpoint_filename="best_model_{epoch:02d}_{val_loss:.4f}",
    every_n_epochs=1,
    **kwargs,
):
    """Train a semantic segmentation model on geospatial imagery.

    This function sets up datasets, model, trainer, and executes the training process
    for semantic segmentation tasks using geospatial data. It supports training
    from scratch or resuming from a checkpoint if available.

    Args:
        image_root (str): Path to directory containing imagery.
        label_root (str): Path to directory containing land cover labels.
        output_dir (str, optional): Directory to save model outputs and checkpoints.
            Defaults to "output".
        in_channels (int, optional): Number of input channels in the imagery.
            Defaults to 4.
        num_classes (int, optional): Number of classes in the segmentation task.
            Defaults to 14.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        img_size (int, optional): Size of image patches for training. Defaults to 256.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        sample_size (int, optional): Number of samples per epoch. Defaults to 500.
        model (str, optional): Model architecture to use. Defaults to "unet".
        backbone (str, optional): Backbone network for the model. Defaults to "resnet50".
        weights (bool, optional): Whether to use pretrained weights. Defaults to True.
        num_filters (int, optional): Number of filters for the model. Defaults to 3.
        loss (str, optional): Loss function to use ('ce', 'jaccard', or 'focal').
            Defaults to "ce".
        class_weights (list, optional): Class weights for loss function. Defaults to None.
        ignore_index (int, optional): Index to ignore in loss calculation. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 0.001.
        patience (int, optional): Number of epochs with no improvement after which
            training will stop. Defaults to 10.
        freeze_backbone (bool, optional): Whether to freeze backbone. Defaults to False.
        freeze_decoder (bool, optional): Whether to freeze decoder. Defaults to False.
        transforms (callable, optional): Transforms to apply to the data. Defaults to None.
        use_augmentation (bool, optional): Whether to apply data augmentation.
            Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        train_val_test_split (list, optional): Proportions for train/val/test split.
            Defaults to [0.6, 0.2, 0.2].
        accelerator (str, optional): Accelerator to use for training ('cpu', 'gpu', etc.).
            Defaults to "auto".
        devices (str, optional): Number of devices to use for training. Defaults to "auto".
        logger (object, optional): Logger for tracking training progress. Defaults to None.
        callbacks (list, optional): List of callbacks for the trainer. Defaults to None.
        log_every_n_steps (int, optional): Frequency of logging training progress.
            Defaults to 10.
        use_distributed_sampler (bool, optional): Whether to use distributed sampling.
            Defaults to False.
        monitor_metric (str, optional): Metric to monitor for saving best model.
            Defaults to "val_loss".
        mode (str, optional): Mode for monitoring metric ('min' or 'max').
            Use 'min' for losses and 'max' for metrics like accuracy.
            Defaults to "min".
        save_top_k (int, optional): Number of best models to save.
            Defaults to 1.
        save_last (bool, optional): Whether to save the model from the last epoch.
            Defaults to True.
        checkpoint_filename (str, optional): Filename pattern for saved checkpoints.
            Defaults to "best_model_{epoch:02d}_{val_loss:.4f}".
        every_n_epochs (int, optional): Save a checkpoint every N epochs.
            Defaults to 1.
        **kwargs: Additional keyword arguments to pass to the datasets.

    Returns:
        object: Trained SemanticSegmentationTask model.
    """
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader
    from torchgeo.datasets import stack_samples, RasterDataset
    from torchgeo.datasets.splits import random_bbox_assignment
    from torchgeo.samplers import (
        RandomGeoSampler,
        RandomBatchGeoSampler,
        GridGeoSampler,
    )
    import os
    import torch
    import multiprocessing as mp
    import timeit
    import albumentations as A
    from torchgeo.datamodules import GeoDataModule
    from torchgeo.trainers import SemanticSegmentationTask
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    # Create a wrapper class for albumentations to work with TorchGeo format
    class AlbumentationsWrapper:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, sample):
            # Extract image and mask from TorchGeo sample format
            if "image" not in sample or "mask" not in sample:
                return sample

            image = sample["image"]
            mask = sample["mask"]

            # Albumentations expects channels last, but TorchGeo uses channels first
            # Convert (C, H, W) to (H, W, C) for image
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.squeeze(0).numpy() if mask.dim() > 2 else mask.numpy()

            # Apply transformation with named arguments
            transformed = self.transform(image=image_np, mask=mask_np)

            # Convert back to PyTorch tensors with channels first
            transformed_image = torch.from_numpy(transformed["image"]).permute(2, 0, 1)
            transformed_mask = torch.from_numpy(transformed["mask"]).unsqueeze(0)

            # Update the sample dictionary
            result = sample.copy()
            result["image"] = transformed_image
            result["mask"] = transformed_mask

            return result

    # Set up data augmentation if requested
    if use_augmentation:
        aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45
                ),
                A.RandomBrightnessContrast(
                    p=0.5, brightness_limit=0.2, contrast_limit=0.2
                ),
                A.GaussianBlur(p=0.3),
                A.GaussNoise(p=0.3),
                A.CoarseDropout(p=0.3, max_holes=8, max_height=32, max_width=32),
            ]
        )
        # Wrap the albumentations transforms
        transforms = AlbumentationsWrapper(aug_transforms)

    # # Set up device configuration
    # device, num_devices = (
    #     ("cuda", torch.cuda.device_count())
    #     if torch.cuda.is_available()
    #     else ("cpu", mp.cpu_count())
    # )
    workers = mp.cpu_count()
    # print(f"Running on {num_devices} {device}(s)")

    # Define datasets
    class ImageDatasetClass(RasterDataset):
        filename_glob = "*.tif"
        is_image = True
        separate_files = False

    class LabelDatasetClass(RasterDataset):
        filename_glob = "*.tif"
        is_image = False
        separate_files = False

    # Prepare output directory
    test_dir = os.path.join(output_dir, "models")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Set up logger and checkpoint callback
    if logger is None:
        logger = CSVLogger(test_dir, name="lightning_logs")

    if callbacks is None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=test_dir,
            filename=checkpoint_filename,
            save_top_k=save_top_k,
            monitor=monitor_metric,
            mode=mode,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            verbose=True,
        )
        callbacks = [checkpoint_callback]

    # Initialize the segmentation task
    task = SemanticSegmentationTask(
        model=model,
        backbone=backbone,
        weights=weights,
        in_channels=in_channels,
        num_classes=num_classes,
        num_filters=num_filters,
        loss=loss,
        class_weights=class_weights,
        ignore_index=ignore_index,
        lr=lr,
        patience=patience,
        freeze_backbone=freeze_backbone,
        freeze_decoder=freeze_decoder,
    )

    # Set up trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        use_distributed_sampler=use_distributed_sampler,
        **kwargs,  # Pass any additional kwargs to the trainer
    )

    # Load datasets with transforms if augmentation is enabled

    if isinstance(image_root, RasterDataset):
        images = image_root
    else:
        images = ImageDatasetClass(paths=image_root, transforms=transforms, **kwargs)

    if isinstance(label_root, RasterDataset):
        labels = label_root
    else:
        labels = LabelDatasetClass(paths=label_root, **kwargs)

    # Create intersection dataset
    dataset = images & labels

    # Define custom datamodule for training
    class CustomGeoDataModule(GeoDataModule):
        def setup(self, stage: str) -> None:
            """Set up datasets.

            Args:
                stage: Either 'fit', 'validate', 'test', or 'predict'.
            """
            self.dataset = self.dataset_class(**self.kwargs)

            generator = torch.Generator().manual_seed(seed)
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = random_bbox_assignment(dataset, train_val_test_split, generator)

            if stage in ["fit"]:
                self.train_batch_sampler = RandomBatchGeoSampler(
                    self.train_dataset, self.patch_size, self.batch_size, self.length
                )
            if stage in ["fit", "validate"]:
                self.val_sampler = GridGeoSampler(
                    self.val_dataset, self.patch_size, self.patch_size
                )
            if stage in ["test"]:
                self.test_sampler = GridGeoSampler(
                    self.test_dataset, self.patch_size, self.patch_size
                )

    # Create datamodule
    datamodule = CustomGeoDataModule(
        dataset_class=type(dataset),
        batch_size=batch_size,
        patch_size=img_size,
        length=sample_size,
        num_workers=workers,
        dataset1=images,
        dataset2=labels,
        collate_fn=stack_samples,
    )

    # Start training timer
    start = timeit.default_timer()

    # Check for existing checkpoint
    checkpoint_file = os.path.join(test_dir, "torchgeo_trained.ckpt")

    if os.path.isfile(checkpoint_file):
        print("Resuming training from previous checkpoint...")
        trainer.fit(model=task, datamodule=datamodule, ckpt_path=checkpoint_file)
    else:
        print("Starting training from scratch...")
        trainer.fit(
            model=task,
            datamodule=datamodule,
        )

    training_time = timeit.default_timer() - start
    print(f"The time taken to train was: {training_time:.2f} seconds")

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Test the model
    trainer.test(model=task, datamodule=datamodule)

    return task
