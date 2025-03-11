"""This module contains utility functions for working with Hugging Face models."""

import csv
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedImageModeling, pipeline


def get_model_config(model_id):
    """
    Get the model configuration for a Hugging Face model.

    Args:
        model_id (str): The Hugging Face model ID.

    Returns:
        transformers.configuration_utils.PretrainedConfig: The model configuration.
    """
    return AutoConfig.from_pretrained(model_id)


def get_model_input_channels(model_id):
    """
    Check the number of input channels supported by a Hugging Face model.

    Args:
        model_id (str): The Hugging Face model ID.

    Returns:
        int: The number of input channels the model accepts.

    Raises:
        ValueError: If unable to determine the number of input channels.
    """
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_id)

    # For Mask2Former models
    if hasattr(config, "backbone_config"):
        if hasattr(config.backbone_config, "num_channels"):
            return config.backbone_config.num_channels

    # Try to load the model and inspect its architecture
    try:
        model = AutoModelForMaskedImageModeling.from_pretrained(model_id)

        # For Swin Transformer-based models like Mask2Former
        if hasattr(model, "backbone") and hasattr(model.backbone, "embeddings"):
            if hasattr(model.backbone.embeddings, "patch_embeddings"):
                # Swin models typically have patch embeddings that indicate channel count
                return model.backbone.embeddings.patch_embeddings.in_channels
    except Exception as e:
        print(f"Couldn't inspect model architecture: {e}")

    # Default for most vision models
    return 3


def image_segmentation(
    tif_path,
    output_path,
    labels_to_extract=None,
    dtype="uint8",
    model_name=None,
    segmenter_args=None,
    **kwargs,
):
    """
    Segments an image with a Hugging Face segmentation model and saves the results
    as a single georeferenced image where each class has a unique integer value.

    Args:
        tif_path (str): Path to the input georeferenced TIF file.
        output_path (str): Path where the output georeferenced segmentation will be saved.
        labels_to_extract (list, optional): List of labels to extract. If None, extracts all labels.
        dtype (str, optional): Data type to use for the output mask. Defaults to "uint8".
        model_name (str, optional): Name of the Hugging Face model to use for segmentation,
            such as "facebook/mask2former-swin-large-cityscapes-semantic". Defaults to None.
            See https://huggingface.co/models?pipeline_tag=image-segmentation&sort=trending for options.
        segmenter_args (dict, optional): Additional arguments to pass to the segmenter.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the segmentation pipeline

    Returns:
        tuple: (Path to saved image, dictionary mapping label names to their assigned values,
            dictionary mapping label names to confidence scores)
    """
    # Load the original georeferenced image to extract metadata
    with rasterio.open(tif_path) as src:
        # Save the metadata for later use
        meta = src.meta.copy()
        # Get the dimensions
        height = src.height
        width = src.width
        # Get the transform and CRS for georeferencing
        # transform = src.transform
        # crs = src.crs

    # Initialize the segmentation pipeline
    if model_name is None:
        model_name = "facebook/mask2former-swin-large-cityscapes-semantic"

    kwargs["task"] = "image-segmentation"

    segmenter = pipeline(model=model_name, **kwargs)

    # Run the segmentation on the GeoTIFF
    if segmenter_args is None:
        segmenter_args = {}

    segments = segmenter(tif_path, **segmenter_args)

    # If no specific labels are requested, extract all available ones
    if labels_to_extract is None:
        labels_to_extract = [segment["label"] for segment in segments]

    # Create an empty mask to hold all the labels
    # Using uint8 for up to 255 classes, switch to uint16 for more
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Create a dictionary to map labels to values and store scores
    label_to_value = {}
    label_to_score = {}

    # Process each segment we want to keep
    for i, segment in enumerate(
        [s for s in segments if s["label"] in labels_to_extract]
    ):
        # Assign a unique value to each label (starting from 1)
        value = i + 1
        label = segment["label"]
        score = segment["score"]

        label_to_value[label] = value
        label_to_score[label] = score

        # Convert PIL image to numpy array
        mask = np.array(segment["mask"])

        # Apply a threshold if it's a probability mask (not binary)
        if mask.dtype == float:
            mask = (mask > 0.5).astype(np.uint8)

        # Resize if needed to match original dimensions
        if mask.shape != (height, width):
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize((width, height))
            mask = np.array(mask_img)

        # Add this class to the combined mask
        # Only overwrite if the pixel isn't already assigned to another class
        # This handles overlapping segments by giving priority to earlier segments
        combined_mask = np.where(
            (mask > 0) & (combined_mask == 0), value, combined_mask
        )

    # Update metadata for the output raster
    meta.update(
        {
            "count": 1,  # One band for the mask
            "dtype": dtype,  # Use uint8 for up to 255 classes
            "nodata": 0,  # 0 represents no class
        }
    )

    # Save the mask as a new georeferenced GeoTIFF
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(combined_mask[np.newaxis, :, :])  # Add channel dimension

    # Create a CSV colormap file with scores included
    csv_path = os.path.splitext(output_path)[0] + "_colormap.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["ClassValue", "ClassName", "ConfidenceScore"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for label, value in label_to_value.items():
            writer.writerow(
                {
                    "ClassValue": value,
                    "ClassName": label,
                    "ConfidenceScore": f"{label_to_score[label]:.4f}",
                }
            )

    return output_path, label_to_value, label_to_score


def mask_generation(
    input_path: str,
    output_mask_path: str,
    output_csv_path: str,
    model: str = "facebook/sam-vit-base",
    confidence_threshold: float = 0.5,
    points_per_side: int = 32,
    crop_size: Optional[int] = None,
    batch_size: int = 1,
    band_indices: Optional[List[int]] = None,
    min_object_size: int = 0,
    generator_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Tuple[str, str]:
    """
    Process a GeoTIFF using SAM mask generation and save results as a GeoTIFF and CSV.

    The function reads a GeoTIFF image, applies the SAM mask generator from the
    Hugging Face transformers pipeline, rasterizes the resulting masks to create
    a labeled mask GeoTIFF, and saves mask scores and geometries to a CSV file.

    Args:
        input_path: Path to the input GeoTIFF image.
        output_mask_path: Path where the output mask GeoTIFF will be saved.
        output_csv_path: Path where the mask scores CSV will be saved.
        model: HuggingFace model checkpoint for the SAM model.
        confidence_threshold: Minimum confidence score for masks to be included.
        points_per_side: Number of points to sample along each side of the image.
        crop_size: Size of image crops for processing. If None, process the full image.
        band_indices: List of band indices to use. If None, use all bands.
        batch_size: Batch size for inference.
        min_object_size: Minimum size in pixels for objects to be included. Smaller masks will be filtered out.
        generator_kwargs: Additional keyword arguments to pass to the mask generator.

    Returns:
        Tuple containing the paths to the saved mask GeoTIFF and CSV file.

    Raises:
        ValueError: If the input file cannot be opened or processed.
        RuntimeError: If mask generation fails.
    """
    # Set up the mask generator
    print("Setting up mask generator...")
    mask_generator = pipeline(model=model, task="mask-generation", **kwargs)

    # Open the GeoTIFF file
    try:
        print(f"Reading input GeoTIFF: {input_path}")
        with rasterio.open(input_path) as src:
            # Read metadata
            profile = src.profile
            # transform = src.transform
            # crs = src.crs

            # Read the image data
            if band_indices is not None:
                print(f"Using specified bands: {band_indices}")
                image_data = np.stack([src.read(i + 1) for i in band_indices])
            else:
                print("Using all bands")
                image_data = src.read()

            # Handle image with more than 3 bands (convert to RGB for visualization)
            if image_data.shape[0] > 3:
                print(
                    f"Converting {image_data.shape[0]} bands to RGB (using first 3 bands)"
                )
                # Select first three bands or perform other band combination
                image_data = image_data[:3]
            elif image_data.shape[0] == 1:
                print("Duplicating single band to create 3-band image")
                # Duplicate single band to create a 3-band image
                image_data = np.vstack([image_data] * 3)

            # Transpose to HWC format for the model
            image_data = np.transpose(image_data, (1, 2, 0))

            # Normalize the image if needed
            if image_data.dtype != np.uint8:
                print(f"Normalizing image from {image_data.dtype} to uint8")
                image_data = (image_data / image_data.max() * 255).astype(np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to open or process input GeoTIFF: {e}")

    # Process the image with the mask generator
    try:
        # Convert numpy array to PIL Image for the pipeline
        # Ensure the array is in the right format (HWC and uint8)
        if image_data.dtype != np.uint8:
            image_data = (image_data / image_data.max() * 255).astype(np.uint8)

        # Create a PIL Image from the numpy array
        print("Converting to PIL Image for mask generation")
        pil_image = Image.fromarray(image_data)

        # Use the SAM pipeline for mask generation
        if generator_kwargs is None:
            generator_kwargs = {}

        print("Running mask generation...")
        mask_results = mask_generator(
            pil_image,
            points_per_side=points_per_side,
            crop_n_points_downscale_factor=1 if crop_size is None else 2,
            point_grids=None,
            pred_iou_thresh=confidence_threshold,
            stability_score_thresh=confidence_threshold,
            crops_n_layers=0 if crop_size is None else 1,
            crop_overlap_ratio=0.5,
            batch_size=batch_size,
            **generator_kwargs,
        )

        print(
            f"Number of initial masks: {len(mask_results['masks']) if isinstance(mask_results, dict) and 'masks' in mask_results else len(mask_results)}"
        )

    except Exception as e:
        raise RuntimeError(f"Mask generation failed: {e}")

    # Create a mask raster with unique IDs for each mask
    mask_raster = np.zeros((image_data.shape[0], image_data.shape[1]), dtype=np.uint32)
    mask_records = []

    # Process each mask based on the structure of mask_results
    if (
        isinstance(mask_results, dict)
        and "masks" in mask_results
        and "scores" in mask_results
    ):
        # Handle dictionary with 'masks' and 'scores' lists
        print("Processing masks...")
        total_masks = len(mask_results["masks"])

        # Create progress bar
        for i, (mask_data, score) in enumerate(
            tqdm(
                zip(mask_results["masks"], mask_results["scores"]),
                total=total_masks,
                desc="Processing masks",
            )
        ):
            mask_id = i + 1  # Start IDs at 1

            # Convert to numpy if not already
            if not isinstance(mask_data, np.ndarray):
                # Try to convert from tensor or other format if needed
                try:
                    mask_data = np.array(mask_data)
                except:
                    print(f"Could not convert mask at index {i} to numpy array")
                    continue

            mask_binary = mask_data.astype(bool)
            area_pixels = np.sum(mask_binary)

            # Skip if mask is smaller than the minimum size
            if area_pixels < min_object_size:
                continue

            # Add the mask to the raster with a unique ID
            mask_raster[mask_binary] = mask_id

            # Create a record for the CSV - without geometry calculation
            mask_records.append(
                {"mask_id": mask_id, "score": float(score), "area_pixels": area_pixels}
            )
    elif isinstance(mask_results, list):
        # Handle list of dictionaries format (SAM original format)
        print("Processing masks...")
        total_masks = len(mask_results)

        # Create progress bar
        for i, mask_result in enumerate(tqdm(mask_results, desc="Processing masks")):
            mask_id = i + 1  # Start IDs at 1

            # Try different possible key names for masks and scores
            mask_data = None
            score = None

            if isinstance(mask_result, dict):
                # Try to find mask data
                if "segmentation" in mask_result:
                    mask_data = mask_result["segmentation"]
                elif "mask" in mask_result:
                    mask_data = mask_result["mask"]

                # Try to find score
                if "score" in mask_result:
                    score = mask_result["score"]
                elif "predicted_iou" in mask_result:
                    score = mask_result["predicted_iou"]
                elif "stability_score" in mask_result:
                    score = mask_result["stability_score"]
                else:
                    score = 1.0  # Default score if none found
            else:
                # If mask_result is not a dict, it might be the mask directly
                try:
                    mask_data = np.array(mask_result)
                    score = 1.0  # Default score
                except:
                    print(f"Could not process mask at index {i}")
                    continue

            if mask_data is not None:
                # Convert to numpy if not already
                if not isinstance(mask_data, np.ndarray):
                    try:
                        mask_data = np.array(mask_data)
                    except:
                        print(f"Could not convert mask at index {i} to numpy array")
                        continue

                mask_binary = mask_data.astype(bool)
                area_pixels = np.sum(mask_binary)

                # Skip if mask is smaller than the minimum size
                if area_pixels < min_object_size:
                    continue

                # Add the mask to the raster with a unique ID
                mask_raster[mask_binary] = mask_id

                # Create a record for the CSV - without geometry calculation
                mask_records.append(
                    {
                        "mask_id": mask_id,
                        "score": float(score),
                        "area_pixels": area_pixels,
                    }
                )
    else:
        # If we couldn't figure out the format, raise an error
        raise ValueError(f"Unexpected format for mask_results: {type(mask_results)}")

    print(f"Number of final masks (after size filtering): {len(mask_records)}")

    # Save the mask raster as a GeoTIFF
    print(f"Saving mask GeoTIFF to {output_mask_path}")
    output_profile = profile.copy()
    output_profile.update(dtype=rasterio.uint32, count=1, compress="lzw", nodata=0)

    with rasterio.open(output_mask_path, "w", **output_profile) as dst:
        dst.write(mask_raster.astype(rasterio.uint32), 1)

    # Save the mask data as a CSV
    print(f"Saving mask metadata to {output_csv_path}")
    mask_df = pd.DataFrame(mask_records)
    mask_df.to_csv(output_csv_path, index=False)

    print("Processing complete!")
    return output_mask_path, output_csv_path
