"""This module provides functionality for segmenting high-resolution satellite imagery using vision-language models."""

import os

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class CLIPSegmentation:
    """
    A class for segmenting high-resolution satellite imagery using text prompts with CLIP-based models.

    This segmenter utilizes the CLIP-Seg model to perform semantic segmentation based on text prompts.
    It can process large GeoTIFF files by tiling them and handles proper georeferencing in the output.

    Args:
        model_name (str): Name of the CLIP-Seg model to use. Defaults to "CIDAS/clipseg-rd64-refined".
        device (str): Device to run the model on ('cuda', 'cpu'). If None, will use CUDA if available.
        tile_size (int): Size of tiles to process the image in chunks. Defaults to 352.
        overlap (int): Overlap between tiles to avoid edge artifacts. Defaults to 16.

    Attributes:
        processor (CLIPSegProcessor): The processor for the CLIP-Seg model.
        model (CLIPSegForImageSegmentation): The CLIP-Seg model for segmentation.
        device (str): The device being used ('cuda' or 'cpu').
        tile_size (int): Size of tiles for processing.
        overlap (int): Overlap between tiles.
    """

    def __init__(
        self,
        model_name="CIDAS/clipseg-rd64-refined",
        device=None,
        tile_size=512,
        overlap=32,
    ):
        """
        Initialize the ImageSegmenter with the specified model and settings.

        Args:
            model_name (str): Name of the CLIP-Seg model to use. Defaults to "CIDAS/clipseg-rd64-refined".
            device (str): Device to run the model on ('cuda', 'cpu'). If None, will use CUDA if available.
            tile_size (int): Size of tiles to process the image in chunks. Defaults to 512.
            overlap (int): Overlap between tiles to avoid edge artifacts. Defaults to 32.
        """
        self.tile_size = tile_size
        self.overlap = overlap

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and processor
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(
            self.device
        )

        print(f"Model loaded on {self.device}")

    def segment_image(
        self, input_path, output_path, text_prompt, threshold=0.5, smoothing_sigma=1.0
    ):
        """
        Segment a GeoTIFF image using the provided text prompt.

        The function processes the image in tiles and saves the result as a GeoTIFF with two bands:
        - Band 1: Binary segmentation mask (0 or 1)
        - Band 2: Probability scores (0.0 to 1.0)

        Args:
            input_path (str): Path to the input GeoTIFF file.
            output_path (str): Path where the output GeoTIFF will be saved.
            text_prompt (str): Text description of what to segment (e.g., "water", "buildings").
            threshold (float): Threshold for binary segmentation (0.0 to 1.0). Defaults to 0.5.
            smoothing_sigma (float): Sigma value for Gaussian smoothing to reduce blockiness. Defaults to 1.0.

        Returns:
            str: Path to the saved output file.
        """
        # Open the input GeoTIFF
        with rasterio.open(input_path) as src:
            # Get metadata
            meta = src.meta
            height = src.height
            width = src.width

            # Create output metadata
            out_meta = meta.copy()
            out_meta.update({"count": 2, "dtype": "float32", "nodata": None})

            # Create arrays for results
            segmentation = np.zeros((height, width), dtype=np.float32)
            probabilities = np.zeros((height, width), dtype=np.float32)

            # Calculate effective tile size (accounting for overlap)
            effective_tile_size = self.tile_size - 2 * self.overlap

            # Calculate number of tiles
            n_tiles_x = max(1, int(np.ceil(width / effective_tile_size)))
            n_tiles_y = max(1, int(np.ceil(height / effective_tile_size)))
            total_tiles = n_tiles_x * n_tiles_y

            # Process tiles with tqdm progress bar
            with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
                # Iterate through tiles
                for y in range(n_tiles_y):
                    for x in range(n_tiles_x):
                        # Calculate tile coordinates with overlap
                        x_start = max(0, x * effective_tile_size - self.overlap)
                        y_start = max(0, y * effective_tile_size - self.overlap)
                        x_end = min(width, (x + 1) * effective_tile_size + self.overlap)
                        y_end = min(
                            height, (y + 1) * effective_tile_size + self.overlap
                        )

                        tile_width = x_end - x_start
                        tile_height = y_end - y_start

                        # Read the tile
                        window = Window(x_start, y_start, tile_width, tile_height)
                        tile_data = src.read(window=window)

                        # Process the tile
                        try:
                            # Convert to RGB if necessary (handling different satellite bands)
                            if tile_data.shape[0] > 3:
                                # Use first three bands for RGB representation
                                rgb_tile = tile_data[:3].transpose(1, 2, 0)
                                # Normalize data to 0-255 range if needed
                                if rgb_tile.max() > 0:
                                    rgb_tile = (
                                        (rgb_tile - rgb_tile.min())
                                        / (rgb_tile.max() - rgb_tile.min())
                                        * 255
                                    ).astype(np.uint8)
                            elif tile_data.shape[0] == 1:
                                # Create RGB from grayscale
                                rgb_tile = np.repeat(
                                    tile_data[0][:, :, np.newaxis], 3, axis=2
                                )
                                # Normalize if needed
                                if rgb_tile.max() > 0:
                                    rgb_tile = (
                                        (rgb_tile - rgb_tile.min())
                                        / (rgb_tile.max() - rgb_tile.min())
                                        * 255
                                    ).astype(np.uint8)
                            else:
                                # Already 3-channel, assume RGB
                                rgb_tile = tile_data.transpose(1, 2, 0)
                                # Normalize if needed
                                if rgb_tile.max() > 0:
                                    rgb_tile = (
                                        (rgb_tile - rgb_tile.min())
                                        / (rgb_tile.max() - rgb_tile.min())
                                        * 255
                                    ).astype(np.uint8)

                            # Convert to PIL Image
                            pil_image = Image.fromarray(rgb_tile)

                            # Resize if needed to match model's requirements
                            if (
                                pil_image.width > self.tile_size
                                or pil_image.height > self.tile_size
                            ):
                                # Keep aspect ratio
                                pil_image.thumbnail(
                                    (self.tile_size, self.tile_size), Image.LANCZOS
                                )

                            # Process with CLIP-Seg
                            inputs = self.processor(
                                text=text_prompt, images=pil_image, return_tensors="pt"
                            ).to(self.device)

                            # Forward pass
                            with torch.no_grad():
                                outputs = self.model(**inputs)

                            # Get logits and resize to original tile size
                            logits = outputs.logits[0]

                            # Convert logits to probabilities with sigmoid
                            probs = torch.sigmoid(logits).cpu().numpy()

                            # Resize back to original tile size if needed
                            if probs.shape != (tile_height, tile_width):
                                # Use bicubic interpolation for smoother results
                                probs_resized = np.array(
                                    Image.fromarray(probs).resize(
                                        (tile_width, tile_height), Image.BICUBIC
                                    )
                                )
                            else:
                                probs_resized = probs

                            # Apply gaussian blur to reduce blockiness
                            try:
                                from scipy.ndimage import gaussian_filter

                                probs_resized = gaussian_filter(
                                    probs_resized, sigma=smoothing_sigma
                                )
                            except ImportError:
                                pass  # Continue without smoothing if scipy is not available

                            # Store results in the full arrays
                            # Only store the non-overlapping part (except at edges)
                            valid_x_start = self.overlap if x > 0 else 0
                            valid_y_start = self.overlap if y > 0 else 0
                            valid_x_end = (
                                tile_width - self.overlap
                                if x < n_tiles_x - 1
                                else tile_width
                            )
                            valid_y_end = (
                                tile_height - self.overlap
                                if y < n_tiles_y - 1
                                else tile_height
                            )

                            dest_x_start = x_start + valid_x_start
                            dest_y_start = y_start + valid_y_start
                            dest_x_end = x_start + valid_x_end
                            dest_y_end = y_start + valid_y_end

                            # Store probabilities
                            probabilities[
                                dest_y_start:dest_y_end, dest_x_start:dest_x_end
                            ] = probs_resized[
                                valid_y_start:valid_y_end, valid_x_start:valid_x_end
                            ]

                        except Exception as e:
                            print(f"Error processing tile at ({x}, {y}): {str(e)}")
                            # Continue with next tile

                        # Update progress bar
                        pbar.update(1)

            # Create binary segmentation from probabilities
            segmentation = (probabilities >= threshold).astype(np.float32)

            # Write the output GeoTIFF
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(segmentation, 1)
                dst.write(probabilities, 2)

                # Add descriptions to bands
                dst.set_band_description(1, "Binary Segmentation")
                dst.set_band_description(2, "Probability Scores")

            print(f"Segmentation saved to {output_path}")
            return output_path

    def segment_image_batch(
        self,
        input_paths,
        output_dir,
        text_prompt,
        threshold=0.5,
        smoothing_sigma=1.0,
        suffix="_segmented",
    ):
        """
        Segment multiple GeoTIFF images using the provided text prompt.

        Args:
            input_paths (list): List of paths to input GeoTIFF files.
            output_dir (str): Directory where output GeoTIFFs will be saved.
            text_prompt (str): Text description of what to segment.
            threshold (float): Threshold for binary segmentation. Defaults to 0.5.
            smoothing_sigma (float): Sigma value for Gaussian smoothing to reduce blockiness. Defaults to 1.0.
            suffix (str): Suffix to add to output filenames. Defaults to "_segmented".

        Returns:
            list: Paths to all saved output files.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []

        # Process each input file
        for input_path in tqdm(input_paths, desc="Processing files"):
            # Generate output path
            filename = os.path.basename(input_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")

            # Segment the image
            result_path = self.segment_image(
                input_path, output_path, text_prompt, threshold, smoothing_sigma
            )
            output_paths.append(result_path)

        return output_paths
