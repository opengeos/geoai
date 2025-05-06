"""The module for training semantic segmentation models for classifying remote sensing imagery."""

import os

import numpy as np


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
    import multiprocessing as mp
    import timeit

    import albumentations as A
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from torchgeo.datamodules import GeoDataModule
    from torchgeo.datasets import RasterDataset, stack_samples
    from torchgeo.datasets.splits import random_bbox_assignment
    from torchgeo.samplers import (
        GridGeoSampler,
        RandomBatchGeoSampler,
        RandomGeoSampler,
    )
    from torchgeo.trainers import SemanticSegmentationTask

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


def _classify_image(
    image_path,
    model_path,
    output_path=None,
    chip_size=1024,
    batch_size=4,
    colormap=None,
    num_workers=2,
    **kwargs,
):
    """
    Classify a geospatial image using a trained semantic segmentation model. The version has
        tile edge artifacts.

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
        colormap (dict, optional): Colormap to apply to the output image.
                                    Defaults to None.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 2.
        **kwargs: Additional keyword arguments for DataLoader.

    Returns:
        str: Path to the saved classified image.
    """
    import timeit

    import numpy as np
    import rasterio
    import torch
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from torch.utils.data import DataLoader
    from torchgeo.datasets import RasterDataset, stack_samples
    from torchgeo.samplers import GridGeoSampler
    from torchgeo.trainers import SemanticSegmentationTask
    from tqdm import tqdm

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
        num_workers=num_workers,
        **kwargs,
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
    def clip_to_original_bounds(tif_path, original_bounds, colormap=None):
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
                {
                    "height": window.height,
                    "width": window.width,
                    "transform": transform,
                    "compress": "deflate",
                }
            )

        # Write the clipped data to the same file
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(data)
            if isinstance(colormap, dict):
                dst.write_colormap(1, colormap)

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
        # if isinstance(colormap, dict):
        #     dst.write_colormap(1, colormap)

    # Clip to original bounds
    print("Clipping to original image bounds...")
    clip_to_original_bounds(output_path, original_bounds, colormap)

    # Close all chip datasets
    for chip in tqdm(georref_chips_list, desc="Cleaning up", unit="chip"):
        chip.close()

    merge_time = timeit.default_timer() - merge_start
    total_time = timeit.default_timer() - start_time

    print(f"Merge and save complete in {merge_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully saved classified image to {output_path}")

    return output_path


def classify_image(
    image_path,
    model_path,
    output_path=None,
    chip_size=1024,
    overlap=256,
    batch_size=4,
    colormap=None,
    **kwargs,
):
    """
    Classify a geospatial image using a trained semantic segmentation model.

    This function handles the full image classification pipeline with special
    attention to edge handling:
    1. Process the image in a grid pattern with overlapping tiles
    2. Use central regions of tiles for interior parts
    3. Special handling for edges to ensure complete coverage
    4. Merge results into a single georeferenced output

    Parameters:
        image_path (str): Path to the input GeoTIFF image.
        model_path (str): Path to the trained model checkpoint.
        output_path (str, optional): Path to save the output classified image.
                                    Defaults to "[input_name]_classified.tif".
        chip_size (int, optional): Size of chips for processing. Defaults to 1024.
        overlap (int, optional): Overlap size between adjacent tiles. Defaults to 256.
        batch_size (int, optional): Batch size for inference. Defaults to 4.
        colormap (dict, optional): Colormap to apply to the output image.
                                   Defaults to None.
        **kwargs: Additional keyword arguments for DataLoader.

    Returns:
        str: Path to the saved classified image.
    """
    import timeit
    import warnings

    import rasterio
    import torch
    from rasterio.errors import NotGeoreferencedWarning
    from torchgeo.trainers import SemanticSegmentationTask

    # Disable specific GDAL/rasterio warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="rasterio._.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    # Also suppress GDAL error reports
    import logging

    logging.getLogger("rasterio").setLevel(logging.ERROR)

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

    # Process the image using a modified tiling approach
    with rasterio.open(image_path) as src:
        # Get image dimensions and metadata
        height = src.height
        width = src.width
        profile = src.profile.copy()

        # Prepare output array for the final result
        output_image = np.zeros((height, width), dtype=np.uint8)
        confidence_map = np.zeros((height, width), dtype=np.float32)

        # Calculate number of tiles needed with overlap
        # Ensure we have tiles that specifically cover the edges
        effective_stride = chip_size - overlap

        # Calculate x positions ensuring leftmost and rightmost edges are covered
        x_positions = []
        # Always include the leftmost position
        x_positions.append(0)
        # Add regular grid positions
        for x in range(effective_stride, width - chip_size, effective_stride):
            x_positions.append(x)
        # Always include rightmost position that still fits
        if width > chip_size and x_positions[-1] + chip_size < width:
            x_positions.append(width - chip_size)

        # Calculate y positions ensuring top and bottom edges are covered
        y_positions = []
        # Always include the topmost position
        y_positions.append(0)
        # Add regular grid positions
        for y in range(effective_stride, height - chip_size, effective_stride):
            y_positions.append(y)
        # Always include bottommost position that still fits
        if height > chip_size and y_positions[-1] + chip_size < height:
            y_positions.append(height - chip_size)

        # Create list of all tile positions
        tile_positions = []
        for y in y_positions:
            for x in x_positions:
                y_end = min(y + chip_size, height)
                x_end = min(x + chip_size, width)
                tile_positions.append((y, x, y_end, x_end))

        # Print information about the tiling
        print(
            f"Processing {len(tile_positions)} patches covering an image of size {height}x{width}..."
        )
        start_time = timeit.default_timer()

        # Process tiles in batches
        for batch_start in range(0, len(tile_positions), batch_size):
            batch_end = min(batch_start + batch_size, len(tile_positions))
            batch_positions = tile_positions[batch_start:batch_end]
            batch_data = []

            # Load data for current batch
            for y_start, x_start, y_end, x_end in batch_positions:
                # Calculate actual tile size
                actual_height = y_end - y_start
                actual_width = x_end - x_start

                # Read the tile data
                tile_data = src.read(window=((y_start, y_end), (x_start, x_end)))

                # Handle different sized tiles by padding if necessary
                if tile_data.shape[1] != chip_size or tile_data.shape[2] != chip_size:
                    padded_data = np.zeros(
                        (tile_data.shape[0], chip_size, chip_size),
                        dtype=tile_data.dtype,
                    )
                    padded_data[:, : tile_data.shape[1], : tile_data.shape[2]] = (
                        tile_data
                    )
                    tile_data = padded_data

                # Convert to tensor

                tile_tensor = torch.from_numpy(tile_data).float() / 255.0
                batch_data.append(tile_tensor)

            # Convert batch to tensor
            batch_tensor = torch.stack(batch_data)

            # Run inference
            with torch.no_grad():
                logits = task.model.predict(batch_tensor.cuda())
                probs = torch.softmax(logits, dim=1)
                confidence, predictions = torch.max(probs, dim=1)
                predictions = predictions.cpu().numpy()
                confidence = confidence.cpu().numpy()

            # Process each prediction
            for idx, (y_start, x_start, y_end, x_end) in enumerate(batch_positions):
                pred = predictions[idx]
                conf = confidence[idx]

                # Calculate actual tile size
                actual_height = y_end - y_start
                actual_width = x_end - x_start

                # Get the actual prediction (removing padding if needed)
                valid_pred = pred[:actual_height, :actual_width]
                valid_conf = conf[:actual_height, :actual_width]

                # Create confidence weights that favor central parts of tiles
                # but still allow edge tiles to contribute fully at the image edges
                is_edge_x = (x_start == 0) or (x_end == width)
                is_edge_y = (y_start == 0) or (y_end == height)

                # Create a mask that gives higher weight to central regions
                # but ensures proper edge handling for boundary tiles
                weight_mask = np.ones((actual_height, actual_width), dtype=np.float32)

                # Only apply central weighting if not at an image edge
                border = overlap // 2
                if not is_edge_x and actual_width > 2 * border:
                    # Apply horizontal edge falloff (linear)
                    for i in range(border):
                        # Left edge
                        weight_mask[:, i] = (i + 1) / (border + 1)
                        # Right edge (if not at image edge)
                        if i < actual_width - border:
                            weight_mask[:, actual_width - i - 1] = (i + 1) / (
                                border + 1
                            )

                if not is_edge_y and actual_height > 2 * border:
                    # Apply vertical edge falloff (linear)
                    for i in range(border):
                        # Top edge
                        weight_mask[i, :] = (i + 1) / (border + 1)
                        # Bottom edge (if not at image edge)
                        if i < actual_height - border:
                            weight_mask[actual_height - i - 1, :] = (i + 1) / (
                                border + 1
                            )

                # Combine with prediction confidence
                final_weight = weight_mask * valid_conf

                # Update the output image based on confidence
                current_conf = confidence_map[y_start:y_end, x_start:x_end]
                update_mask = final_weight > current_conf

                if np.any(update_mask):
                    # Update only pixels where this prediction has higher confidence
                    output_image[y_start:y_end, x_start:x_end][update_mask] = (
                        valid_pred[update_mask]
                    )
                    confidence_map[y_start:y_end, x_start:x_end][update_mask] = (
                        final_weight[update_mask]
                    )

        # Update profile for output
        profile.update({"count": 1, "dtype": "uint8", "nodata": 0})

        # Save the result
        print(f"Saving classified image to {output_path}...")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output_image[np.newaxis, :, :])
            if isinstance(colormap, dict):
                dst.write_colormap(1, colormap)

        # Calculate timing
        total_time = timeit.default_timer() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Successfully saved classified image to {output_path}")

    return output_path


def classify_images(
    image_paths,
    model_path,
    output_dir=None,
    chip_size=1024,
    batch_size=4,
    colormap=None,
    file_extension=".tif",
    **kwargs,
):
    """
    Classify multiple geospatial images using a trained semantic segmentation model.

    This function accepts either a list of image paths or a directory containing images
    and applies the classify_image function to each image, saving the results in the
    specified output directory.

    Parameters:
        image_paths (str or list): Either a directory path containing images or a list
            of paths to input GeoTIFF images.
        model_path (str): Path to the trained model checkpoint.
        output_dir (str, optional): Directory to save the output classified images.
            Defaults to None (same directory as input images for a list, or a new
            "classified" subdirectory for a directory input).
        chip_size (int, optional): Size of chips for processing. Defaults to 1024.
        batch_size (int, optional): Batch size for inference. Defaults to 4.
        colormap (dict, optional): Colormap to apply to the output images.
            Defaults to None.
        file_extension (str, optional): File extension to filter by when image_paths
            is a directory. Defaults to ".tif".
        **kwargs: Additional keyword arguments for the classify_image function.

    Returns:
        list: List of paths to the saved classified images.
    """
    # Import required libraries
    import glob

    from tqdm import tqdm

    # Process directory input
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(image_paths, "classified")

        # Get all images with the specified extension
        image_path_list = glob.glob(os.path.join(image_paths, f"*{file_extension}"))

        # Check if any images were found
        if not image_path_list:
            print(f"No files with extension '{file_extension}' found in {image_paths}")
            return []

        print(f"Found {len(image_path_list)} images in directory {image_paths}")

    # Process list input
    elif isinstance(image_paths, list):
        image_path_list = image_paths

        # Set default output directory if not provided
        if output_dir is None and len(image_path_list) > 0:
            output_dir = os.path.dirname(image_path_list[0])

    # Invalid input
    else:
        raise ValueError(
            "image_paths must be either a directory path or a list of file paths"
        )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classified_image_paths = []

    # Create progress bar
    for image_path in tqdm(image_path_list, desc="Classifying images", unit="image"):
        try:
            # Get just the filename without extension
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            # Create output path within output_dir
            output_path = os.path.join(
                output_dir, f"{base_filename}_classified{file_extension}"
            )

            # Perform classification
            classified_image_path = classify_image(
                image_path,
                model_path,
                output_path=output_path,
                chip_size=chip_size,
                batch_size=batch_size,
                colormap=colormap,
                **kwargs,
            )
            classified_image_paths.append(classified_image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    print(
        f"Classification complete. Processed {len(classified_image_paths)} images successfully."
    )
    return classified_image_paths
