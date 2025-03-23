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
    checkpoint_filename="best_model",
    checkpoint_path=None,
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
        checkpoint_path (str, optional): Path to a checkpoint file to resume training.
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
    if checkpoint_path is not None:
        checkpoint_file = os.path.abspath(checkpoint_path)
    else:
        checkpoint_file = os.path.join(test_dir, "last.ckpt")

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


def classify_image(
    image_path, model_path, output_path=None, chip_size=1024, batch_size=4
):
    """
    Classify a geospatial image using a trained semantic segmentation model.

    This function handles the full image classification pipeline:
    1. Loads the image and model
    2. Cuts the image into tiles/chips
    3. Makes predictions on each chip
    4. Georeferences each prediction
    5. Merges all predictions into a single georeferenced output

    Parameters:
        image_path (str): Path to the input GeoTIFF image.
        model_path (str): Path to the trained model checkpoint.
        output_path (str, optional): Path to save the output classified image.
                                    Defaults to "classified_output.tif".
        chip_size (int, optional): Size of chips for processing. Defaults to 1024.
        batch_size (int, optional): Batch size for inference. Defaults to 4.

    Returns:
        str: Path to the saved classified image.
    """
    import os
    import numpy as np
    import timeit
    from tqdm import tqdm

    import torch
    from torch.utils.data import DataLoader

    from torchgeo.datasets import RasterDataset, stack_samples
    from torchgeo.samplers import GridGeoSampler
    from torchgeo.trainers import SemanticSegmentationTask

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.io import MemoryFile
    from rasterio.merge import merge

    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_classified.tif"

    # Make sure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the model
    print(f"Loading model from {model_path}...")
    task = SemanticSegmentationTask.load_from_checkpoint(model_path)
    task.model.eval()
    task.model.cuda()

    # Set up dataset and sampler
    print(f"Loading image from {image_path}...")
    dataset = RasterDataset(paths=image_path)

    # Get the bounds and resolution of the dataset
    original_bounds = dataset.bounds
    pixel_size = dataset.res
    crs = dataset.crs.to_epsg()

    # Use a GridGeoSampler to sample the image in tiles
    sampler = GridGeoSampler(dataset, chip_size, chip_size)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=stack_samples,
        num_workers=2,
    )

    print(f"Processing image in {len(dataloader)} batches...")

    # Helper function to create in-memory geotiffs for chips
    def create_in_memory_geochip(predicted_chip, geotransform, crs):
        """Create in-memory georeferenced chips."""
        photometric = "MINISBLACK"

        # Ensure predicted_chip has shape (bands, height, width)
        if len(predicted_chip.shape) == 2:
            predicted_chip = predicted_chip[np.newaxis, :, :]

        memfile = MemoryFile()
        dataset = memfile.open(
            driver="GTiff",
            height=predicted_chip.shape[1],
            width=predicted_chip.shape[2],
            count=predicted_chip.shape[0],  # Number of bands
            dtype=np.uint8,
            crs=crs,
            transform=geotransform,
            photometric=photometric,
        )

        # Write all bands
        for band_idx in range(predicted_chip.shape[0]):
            dataset.write(
                predicted_chip[band_idx], band_idx + 1
            )  # Band indices are 1-based in rasterio

        return dataset

    # Helper function to clip to original bounds
    def clip_to_original_bounds(tif_path, original_bounds):
        """Clip a GeoTIFF to match original bounds."""
        with rasterio.open(tif_path) as src:
            # Create a window that matches the original bounds
            window = rasterio.windows.from_bounds(
                original_bounds.minx,
                original_bounds.miny,
                original_bounds.maxx,
                original_bounds.maxy,
                transform=src.transform,
            )

            # Read data within the window
            data = src.read(window=window)

            # Update the transform
            transform = rasterio.windows.transform(window, src.transform)

            # Create new metadata
            meta = src.meta.copy()
            meta.update(
                {"height": window.height, "width": window.width, "transform": transform}
            )

        # Write the clipped data to the same file
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(data)

    # Run inference on all chips
    start_time = timeit.default_timer()
    georref_chips_list = []

    # Progress bar for processing chips
    progress_bar = tqdm(total=len(dataloader), desc="Processing tiles", unit="batch")

    for batch in dataloader:
        # Get images and bounds
        images = batch["image"]
        bounds_list = batch["bounds"]

        # Normalize images
        images = images / 255.0

        # Make predictions
        with torch.no_grad():
            predictions = task.model.predict(images.cuda())
            predictions = torch.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)

        # Process each prediction in the batch
        for i in range(len(predictions)):
            # Get the bounds for this chip
            bounds = bounds_list[i]

            # Create geotransform
            geotransform = from_origin(bounds.minx, bounds.maxy, pixel_size, pixel_size)

            # Convert prediction to numpy array
            pred = predictions[i].cpu().numpy().astype(np.uint8)
            if len(pred.shape) == 2:
                pred = pred[np.newaxis, :, :]

            # Create georeferenced chip
            georref_chips_list.append(create_in_memory_geochip(pred, geotransform, crs))

        # Update progress bar
        progress_bar.update(1)

    progress_bar.close()

    prediction_time = timeit.default_timer() - start_time
    print(f"Prediction complete in {prediction_time:.2f} seconds")
    print(f"Produced {len(georref_chips_list)} georeferenced chips")

    # Merge all georeferenced chips into a single output
    print("Merging predictions...")
    merge_start = timeit.default_timer()

    # Merge the chips using Rasterio's merge function
    merged, merged_transform = merge(georref_chips_list)

    # Calculate the number of rows and columns for the merged output
    rows, cols = merged.shape[1], merged.shape[2]

    # Update the metadata of the merged dataset
    merged_metadata = georref_chips_list[0].meta
    merged_metadata.update(
        {"height": rows, "width": cols, "transform": merged_transform}
    )

    # Write the merged array to the output file
    with rasterio.open(output_path, "w", **merged_metadata) as dst:
        dst.write(merged)

    # Clip to original bounds
    print("Clipping to original image bounds...")
    clip_to_original_bounds(output_path, original_bounds)

    # Close all chip datasets
    for chip in tqdm(georref_chips_list, desc="Cleaning up", unit="chip"):
        chip.close()

    merge_time = timeit.default_timer() - merge_start
    total_time = timeit.default_timer() - start_time

    print(f"Merge and save complete in {merge_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully saved classified image to {output_path}")

    return output_path
