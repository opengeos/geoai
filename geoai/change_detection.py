"""Change detection module for remote sensing imagery using torchange."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from skimage.transform import resize

try:
    from torchange.models.segment_any_change import AnyChange, show_change_masks
except ImportError:
    AnyChange = None
    show_change_masks = None

from .utils import download_file


class ChangeDetection:
    """A class for change detection on geospatial imagery using torchange and SAM."""

    def __init__(self, sam_model_type="vit_h", sam_checkpoint=None):
        """
        Initialize the ChangeDetection class.

        Args:
            sam_model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
            sam_checkpoint (str): Path to SAM checkpoint file
        """
        self.sam_model_type = sam_model_type
        self.sam_checkpoint = sam_checkpoint
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize the AnyChange model."""
        if AnyChange is None:
            raise ImportError(
                "The 'torchange' package is required for change detection. "
                "Please install it using: pip install torchange\n"
                "Note: torchange requires Python 3.11 or higher."
            )

        if self.sam_checkpoint is None:
            self.sam_checkpoint = download_checkpoint(self.sam_model_type)

        self.model = AnyChange(self.sam_model_type, sam_checkpoint=self.sam_checkpoint)

        # Set default hyperparameters
        self.model.make_mask_generator(
            points_per_side=32,
            stability_score_thresh=0.95,
        )
        self.model.set_hyperparameters(
            change_confidence_threshold=145,
            use_normalized_feature=True,
            bitemporal_match=True,
        )

    def set_hyperparameters(
        self,
        change_confidence_threshold: int = 155,
        auto_threshold: bool = False,
        use_normalized_feature: bool = True,
        area_thresh: float = 0.8,
        match_hist: bool = False,
        object_sim_thresh: int = 60,
        bitemporal_match: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Set hyperparameters for the change detection model.

        Args:
            change_confidence_threshold (int): Change confidence threshold for SAM
            auto_threshold (bool): Whether to use auto threshold for SAM
            use_normalized_feature (bool): Whether to use normalized feature for SAM
            area_thresh (float): Area threshold for SAM
            match_hist (bool): Whether to use match hist for SAM
            object_sim_thresh (int): Object similarity threshold for SAM
            bitemporal_match (bool): Whether to use bitemporal match for SAM
            **kwargs: Keyword arguments for model hyperparameters
        """
        if self.model:
            self.model.set_hyperparameters(
                change_confidence_threshold=change_confidence_threshold,
                auto_threshold=auto_threshold,
                use_normalized_feature=use_normalized_feature,
                area_thresh=area_thresh,
                match_hist=match_hist,
                object_sim_thresh=object_sim_thresh,
                bitemporal_match=bitemporal_match,
                **kwargs,
            )

    def set_mask_generator_params(
        self,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        point_grids: Optional[List] = None,
        min_mask_region_area: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Set mask generator parameters.

        Args:
            points_per_side (int): Number of points per side for SAM
            points_per_batch (int): Number of points per batch for SAM
            pred_iou_thresh (float): IoU threshold for SAM
            stability_score_thresh (float): Stability score threshold for SAM
            stability_score_offset (float): Stability score offset for SAM
            box_nms_thresh (float): NMS threshold for SAM
            point_grids (list): Point grids for SAM
            min_mask_region_area (int): Minimum mask region area for SAM
            **kwargs: Keyword arguments for mask generator
        """
        if self.model:
            self.model.make_mask_generator(
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                box_nms_thresh=box_nms_thresh,
                point_grids=point_grids,
                min_mask_region_area=min_mask_region_area,
                **kwargs,
            )

    def _read_and_align_images(self, image1_path, image2_path, target_size=1024):
        """
        Read and align two GeoTIFF images, handling different extents and projections.

        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            target_size (int): Target size for processing (default 1024 for torchange)

        Returns:
            tuple: (aligned_img1, aligned_img2, transform, crs, bounds)
        """
        with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
            # Get the intersection of bounds
            bounds1 = src1.bounds
            bounds2 = src2.bounds

            # Calculate intersection bounds
            left = max(bounds1.left, bounds2.left)
            bottom = max(bounds1.bottom, bounds2.bottom)
            right = min(bounds1.right, bounds2.right)
            top = min(bounds1.top, bounds2.top)

            if left >= right or bottom >= top:
                raise ValueError("Images do not overlap")

            intersection_bounds = (left, bottom, right, top)

            # Read the intersecting area from both images
            window1 = from_bounds(*intersection_bounds, src1.transform)
            window2 = from_bounds(*intersection_bounds, src2.transform)

            # Read data
            img1_data = src1.read(window=window1)
            img2_data = src2.read(window=window2)

            # Get transform for the intersecting area
            transform = src1.window_transform(window1)
            crs = src1.crs

            # Convert from (bands, height, width) to (height, width, bands)
            img1_data = np.transpose(img1_data, (1, 2, 0))
            img2_data = np.transpose(img2_data, (1, 2, 0))

            # Use only RGB bands (first 3 channels) for torchange
            if img1_data.shape[2] >= 3:
                img1_data = img1_data[:, :, :3]
            if img2_data.shape[2] >= 3:
                img2_data = img2_data[:, :, :3]

            # Normalize to 0-255 range if needed
            if img1_data.dtype != np.uint8:
                img1_data = (
                    (img1_data - img1_data.min())
                    / (img1_data.max() - img1_data.min())
                    * 255
                ).astype(np.uint8)
            if img2_data.dtype != np.uint8:
                img2_data = (
                    (img2_data - img2_data.min())
                    / (img2_data.max() - img2_data.min())
                    * 255
                ).astype(np.uint8)

            # Store original size for later use
            original_shape = img1_data.shape[:2]

            # Resize to target size for torchange processing
            if img1_data.shape[0] != target_size or img1_data.shape[1] != target_size:
                img1_resized = resize(
                    img1_data, (target_size, target_size), preserve_range=True
                ).astype(np.uint8)
                img2_resized = resize(
                    img2_data, (target_size, target_size), preserve_range=True
                ).astype(np.uint8)
            else:
                img1_resized = img1_data
                img2_resized = img2_data

            return (img1_resized, img2_resized, transform, crs, original_shape)

    def detect_changes(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None,
        target_size: int = 1024,
        return_results: bool = True,
        export_probability: bool = False,
        probability_output_path: Optional[str] = None,
        export_instance_masks: bool = False,
        instance_masks_output_path: Optional[str] = None,
        return_detailed_results: bool = False,
    ) -> Union[Tuple[Any, np.ndarray, np.ndarray], Dict[str, Any], None]:
        """
        Detect changes between two GeoTIFF images with instance segmentation.

        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            output_path (str): Optional path to save binary change mask as GeoTIFF
            target_size (int): Target size for processing
            return_results (bool): Whether to return results
            export_probability (bool): Whether to export probability mask
            probability_output_path (str): Path to save probability mask (required if export_probability=True)
            export_instance_masks (bool): Whether to export instance segmentation masks
            instance_masks_output_path (str): Path to save instance masks (required if export_instance_masks=True)
            return_detailed_results (bool): Whether to return detailed mask information

        Returns:
            tuple: (change_masks, img1, img2) if return_results=True
            dict: Detailed results if return_detailed_results=True
        """
        # Read and align images
        img1, img2, transform, crs, original_shape = self._read_and_align_images(
            image1_path, image2_path, target_size
        )

        # Detect changes
        change_masks, _, _ = self.model.forward(img1, img2)

        # If output path specified, save binary mask as GeoTIFF
        if output_path:
            self._save_change_mask(
                change_masks, output_path, transform, crs, original_shape, target_size
            )

        # If probability export requested, save probability mask
        if export_probability:
            if probability_output_path is None:
                raise ValueError(
                    "probability_output_path must be specified when export_probability=True"
                )
            self._save_probability_mask(
                change_masks,
                probability_output_path,
                transform,
                crs,
                original_shape,
                target_size,
            )

        # If instance masks export requested, save instance segmentation masks
        if export_instance_masks:
            if instance_masks_output_path is None:
                raise ValueError(
                    "instance_masks_output_path must be specified when export_instance_masks=True"
                )
            num_instances = self._save_instance_segmentation_masks(
                change_masks,
                instance_masks_output_path,
                transform,
                crs,
                original_shape,
                target_size,
            )

            # Also save instance scores if requested
            scores_path = instance_masks_output_path.replace(".tif", "_scores.tif")
            self._save_instance_scores_mask(
                change_masks,
                scores_path,
                transform,
                crs,
                original_shape,
                target_size,
            )

        # Return detailed results if requested
        if return_detailed_results:
            return self._extract_detailed_results(
                change_masks, transform, crs, original_shape, target_size
            )

        if return_results:
            return change_masks, img1, img2

    def _save_change_mask(
        self, change_masks, output_path, transform, crs, original_shape, target_size
    ):
        """
        Save change masks as a GeoTIFF with proper georeference.

        Args:
            change_masks: Change detection masks (MaskData object)
            output_path (str): Output file path
            transform: Rasterio transform
            crs: Coordinate reference system
            original_shape (tuple): Original image shape
            target_size (int): Processing target size
        """
        # Convert MaskData to binary mask by decoding RLE masks
        combined_mask = np.zeros((target_size, target_size), dtype=bool)

        # Extract RLE masks from MaskData object
        mask_items = dict(change_masks.items())
        if "rles" in mask_items:
            rles = mask_items["rles"]
            for rle in rles:
                if isinstance(rle, dict) and "size" in rle and "counts" in rle:
                    try:
                        # Decode RLE to binary mask
                        size = rle["size"]
                        counts = rle["counts"]

                        # Create binary mask from RLE counts
                        mask = np.zeros(size[0] * size[1], dtype=np.uint8)
                        pos = 0
                        value = 0

                        for count in counts:
                            if pos + count <= len(mask):
                                if value == 1:
                                    mask[pos : pos + count] = 1
                                pos += count
                                value = 1 - value  # Toggle between 0 and 1
                            else:
                                break

                        # RLE is column-major, reshape and transpose
                        mask = mask.reshape(size).T
                        if mask.shape == (target_size, target_size):
                            combined_mask = np.logical_or(
                                combined_mask, mask.astype(bool)
                            )

                    except Exception as e:
                        print(f"Warning: Failed to decode RLE mask: {e}")
                        continue

        # Convert to uint8 first, then resize if needed
        combined_mask_uint8 = combined_mask.astype(np.uint8) * 255

        # Resize back to original shape if needed
        if original_shape != (target_size, target_size):
            # Use precise resize
            combined_mask_resized = resize(
                combined_mask_uint8.astype(np.float32),
                original_shape,
                preserve_range=True,
                anti_aliasing=False,
                order=0,
            )
            combined_mask = (combined_mask_resized > 127).astype(np.uint8) * 255
        else:
            combined_mask = combined_mask_uint8

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=combined_mask.shape[0],
            width=combined_mask.shape[1],
            count=1,
            dtype=combined_mask.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(combined_mask, 1)

    def _save_probability_mask(
        self, change_masks, output_path, transform, crs, original_shape, target_size
    ):
        """
        Save probability masks as a GeoTIFF with proper georeference.

        Args:
            change_masks: Change detection masks (MaskData object)
            output_path (str): Output file path
            transform: Rasterio transform
            crs: Coordinate reference system
            original_shape (tuple): Original image shape
            target_size (int): Processing target size
        """
        # Extract mask components for probability calculation
        mask_items = dict(change_masks.items())
        rles = mask_items.get("rles", [])
        iou_preds = mask_items.get("iou_preds", None)
        stability_scores = mask_items.get("stability_score", None)
        change_confidence = mask_items.get("change_confidence", None)
        areas = mask_items.get("areas", None)

        # Convert tensors to numpy if needed
        if iou_preds is not None:
            iou_preds = iou_preds.detach().cpu().numpy()
        if stability_scores is not None:
            stability_scores = stability_scores.detach().cpu().numpy()
        if change_confidence is not None:
            change_confidence = change_confidence.detach().cpu().numpy()
        if areas is not None:
            areas = areas.detach().cpu().numpy()

        # Create probability mask
        probability_mask = np.zeros((target_size, target_size), dtype=np.float32)

        # Process each mask with probability weighting
        for i, rle in enumerate(rles):
            if isinstance(rle, dict) and "size" in rle and "counts" in rle:
                try:
                    # Decode RLE to binary mask
                    size = rle["size"]
                    counts = rle["counts"]

                    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
                    pos = 0
                    value = 0

                    for count in counts:
                        if pos + count <= len(mask):
                            if value == 1:
                                mask[pos : pos + count] = 1
                            pos += count
                            value = 1 - value
                        else:
                            break

                    mask = mask.reshape(size).T
                    if mask.shape != (target_size, target_size):
                        continue

                    mask_bool = mask.astype(bool)

                    # Calculate probability using multiple factors
                    prob_components = []

                    # IoU prediction (0-1, higher is better)
                    if iou_preds is not None and i < len(iou_preds):
                        iou_score = float(iou_preds[i])
                        prob_components.append(("iou", iou_score))
                    else:
                        prob_components.append(("iou", 0.8))

                    # Stability score (0-1, higher is better)
                    if stability_scores is not None and i < len(stability_scores):
                        stability = float(stability_scores[i])
                        prob_components.append(("stability", stability))
                    else:
                        prob_components.append(("stability", 0.8))

                    # Change confidence (normalize based on threshold)
                    if change_confidence is not None and i < len(change_confidence):
                        conf = float(change_confidence[i])
                        # Normalize confidence: threshold is 145, values above indicate higher confidence
                        if conf >= 145:
                            conf_normalized = 0.5 + min(0.5, (conf - 145) / 145)
                        else:
                            conf_normalized = max(0.0, conf / 145 * 0.5)
                        prob_components.append(("confidence", conf_normalized))
                    else:
                        prob_components.append(("confidence", 0.5))

                    # Area-based weight (normalize using log scale)
                    if areas is not None and i < len(areas):
                        area = float(areas[i])
                        area_normalized = 0.2 + 0.8 * min(1.0, np.log(area + 1) / 15.0)
                        prob_components.append(("area", area_normalized))
                    else:
                        prob_components.append(("area", 0.6))

                    # Calculate weighted probability
                    weights = {
                        "iou": 0.3,
                        "stability": 0.3,
                        "confidence": 0.35,
                        "area": 0.05,
                    }
                    prob_weight = sum(
                        weights[name] * value for name, value in prob_components
                    )
                    prob_weight = np.clip(prob_weight, 0.0, 1.0)

                    # Add to probability mask (take maximum where masks overlap)
                    current_prob = probability_mask[mask_bool]
                    new_prob = np.maximum(current_prob, prob_weight)
                    probability_mask[mask_bool] = new_prob

                except Exception as e:
                    print(f"Warning: Failed to process probability mask {i}: {e}")
                    continue

        # Resize back to original shape if needed
        if original_shape != (target_size, target_size):
            prob_resized = resize(
                probability_mask,
                original_shape,
                preserve_range=True,
                anti_aliasing=True,
                order=1,
            )
            prob_final = np.clip(prob_resized, 0.0, 1.0)
        else:
            prob_final = probability_mask

        # Save as float32 GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=prob_final.shape[0],
            width=prob_final.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(prob_final.astype(np.float32), 1)

    def visualize_changes(
        self, image1_path: str, image2_path: str, figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Visualize change detection results.

        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if show_change_masks is None:
            raise ImportError(
                "The 'torchange' package is required for change detection visualization. "
                "Please install it using: pip install torchange\n"
                "Note: torchange requires Python 3.11 or higher."
            )

        change_masks, img1, img2 = self.detect_changes(
            image1_path, image2_path, return_results=True
        )

        # Use torchange's visualization function
        fig, _ = show_change_masks(img1, img2, change_masks)
        fig.set_size_inches(figsize)

        return fig

    def visualize_results(
        self,
        image1_path,
        image2_path,
        binary_path,
        prob_path,
        title1="Earlier Image",
        title2="Later Image",
    ):
        """Create enhanced visualization with probability analysis."""

        # Load data
        with rasterio.open(image1_path) as src:
            img1 = src.read([1, 2, 3])
            img1 = np.transpose(img1, (1, 2, 0))

        with rasterio.open(image2_path) as src:
            img2 = src.read([1, 2, 3])
            img2 = np.transpose(img2, (1, 2, 0))

        with rasterio.open(binary_path) as src:
            binary_mask = src.read(1)

        with rasterio.open(prob_path) as src:
            prob_mask = src.read(1)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # Crop for better visualization
        h, w = img1.shape[:2]
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4

        img1_crop = img1[y1:y2, x1:x2]
        img2_crop = img2[y1:y2, x1:x2]
        binary_crop = binary_mask[y1:y2, x1:x2]
        prob_crop = prob_mask[y1:y2, x1:x2]

        # Row 1: Original and overlays
        axes[0, 0].imshow(img1_crop)
        axes[0, 0].set_title(title1, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(img2_crop)
        axes[0, 1].set_title(title2, fontweight="bold")
        axes[0, 1].axis("off")

        # Binary overlay
        overlay_binary = img2_crop.copy()
        overlay_binary[binary_crop > 0] = [255, 0, 0]
        axes[0, 2].imshow(overlay_binary)
        axes[0, 2].set_title("Binary Changes\n(Red = Change)", fontweight="bold")
        axes[0, 2].axis("off")

        # Probability heatmap
        im1 = axes[0, 3].imshow(prob_crop, cmap="hot", vmin=0, vmax=1)
        axes[0, 3].set_title(
            "Probability Heatmap\n(White = High Confidence)", fontweight="bold"
        )
        axes[0, 3].axis("off")
        plt.colorbar(im1, ax=axes[0, 3], shrink=0.8)

        # Row 2: Detailed probability analysis
        # Confidence levels overlay
        overlay_conf = img2_crop.copy()
        high_conf = prob_crop > 0.7
        med_conf = (prob_crop > 0.4) & (prob_crop <= 0.7)
        low_conf = (prob_crop > 0.1) & (prob_crop <= 0.4)

        overlay_conf[high_conf] = [255, 0, 0]  # Red for high
        overlay_conf[med_conf] = [255, 165, 0]  # Orange for medium
        overlay_conf[low_conf] = [255, 255, 0]  # Yellow for low

        axes[1, 0].imshow(overlay_conf)
        axes[1, 0].set_title(
            "Confidence Levels\n(Red>0.7, Orange>0.4, Yellow>0.1)", fontweight="bold"
        )
        axes[1, 0].axis("off")

        # Thresholded probability (>0.5)
        overlay_thresh = img2_crop.copy()
        high_prob = prob_crop > 0.5
        overlay_thresh[high_prob] = [255, 0, 0]
        axes[1, 1].imshow(overlay_thresh)
        axes[1, 1].set_title(
            "High Confidence Only\n(Probability > 0.5)", fontweight="bold"
        )
        axes[1, 1].axis("off")

        # Probability histogram
        prob_values = prob_crop[prob_crop > 0]
        if len(prob_values) > 0:
            axes[1, 2].hist(
                prob_values, bins=50, alpha=0.7, color="red", edgecolor="black"
            )
            axes[1, 2].axvline(
                x=0.5, color="blue", linestyle="--", label="0.5 threshold"
            )
            axes[1, 2].axvline(
                x=0.7, color="green", linestyle="--", label="0.7 threshold"
            )
            axes[1, 2].set_xlabel("Change Probability")
            axes[1, 2].set_ylabel("Pixel Count")
            axes[1, 2].set_title(
                f"Probability Distribution\n({len(prob_values):,} pixels)"
            )
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        # Statistics text
        stats_text = f"""Probability Statistics:
    Min: {np.min(prob_values):.3f}
    Max: {np.max(prob_values):.3f}
    Mean: {np.mean(prob_values):.3f}
    Median: {np.median(prob_values):.3f}

    Confidence Levels:
    High (>0.7): {np.sum(prob_crop > 0.7):,}
    Med (0.4-0.7): {np.sum((prob_crop > 0.4) & (prob_crop <= 0.7)):,}
    Low (0.1-0.4): {np.sum((prob_crop > 0.1) & (prob_crop <= 0.4)):,}"""

        axes[1, 3].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 3].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 3].set_xlim(0, 1)
        axes[1, 3].set_ylim(0, 1)
        axes[1, 3].axis("off")
        axes[1, 3].set_title("Statistics Summary", fontweight="bold")

        plt.tight_layout()
        plt.suptitle(
            "Enhanced Probability-Based Change Detection",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.savefig("enhanced_probability_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("💾 Enhanced visualization saved as 'enhanced_probability_results.png'")

    def create_split_comparison(
        self,
        image1_path,
        image2_path,
        binary_path,
        prob_path,
        title1="Earlier Image",
        title2="Later Image",
        output_path="split_comparison.png",
    ):
        """Create a split comparison visualization showing before/after with change overlay."""
        import cv2  # Lazy import to avoid QGIS opencv conflicts

        # Load data
        with rasterio.open(image1_path) as src:
            img1 = src.read([1, 2, 3])
            img1 = np.transpose(img1, (1, 2, 0))
            if img1.dtype != np.uint8:
                img1 = ((img1 - img1.min()) / (img1.max() - img1.min()) * 255).astype(
                    np.uint8
                )

        with rasterio.open(image2_path) as src:
            img2 = src.read([1, 2, 3])
            img2 = np.transpose(img2, (1, 2, 0))
            if img2.dtype != np.uint8:
                img2 = ((img2 - img2.min()) / (img2.max() - img2.min()) * 255).astype(
                    np.uint8
                )

        with rasterio.open(prob_path) as src:
            prob_mask = src.read(1)

        # Ensure all arrays have the same shape
        h, w = img1.shape[:2]
        if prob_mask.shape != (h, w):
            prob_mask = resize(
                prob_mask, (h, w), preserve_range=True, anti_aliasing=True, order=1
            )

        # Create split comparison
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Create combined image - left half is earlier, right half is later
        combined_img = np.zeros_like(img1)
        combined_img[:, : w // 2] = img1[:, : w // 2]
        combined_img[:, w // 2 :] = img2[:, w // 2 :]

        # Create overlay with changes - ensure prob_mask is 2D and matches image dimensions
        overlay = combined_img.copy()
        high_conf_changes = prob_mask > 0.5

        # Apply overlay only where changes are detected
        if len(overlay.shape) == 3:  # RGB image
            overlay[high_conf_changes] = [255, 0, 0]  # Red for high confidence changes

        # Blend overlay with original
        blended = cv2.addWeighted(combined_img, 0.7, overlay, 0.3, 0)

        ax.imshow(blended)
        ax.axvline(x=w // 2, color="white", linewidth=3, linestyle="--", alpha=0.8)
        ax.text(
            w // 4,
            50,
            title1,
            fontsize=20,
            color="white",
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.8},
        )
        ax.text(
            3 * w // 4,
            50,
            title2,
            fontsize=20,
            color="white",
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.8},
        )

        ax.set_title(
            "Split Comparison with Change Detection\n(Red = High Confidence Changes)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"💾 Split comparison saved as '{output_path}'")

    def analyze_instances(
        self, instance_mask_path, scores_path, output_path="instance_analysis.png"
    ):
        """Analyze and visualize instance segmentation results."""

        # Load instance mask and scores
        with rasterio.open(instance_mask_path) as src:
            instance_mask = src.read(1)

        with rasterio.open(scores_path) as src:
            scores_mask = src.read(1)

        # Get unique instances (excluding background)
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]

        # Calculate statistics for each instance
        instance_stats = []
        for instance_id in unique_instances:
            mask = instance_mask == instance_id
            area = np.sum(mask)
            score = np.mean(scores_mask[mask])
            instance_stats.append({"id": instance_id, "area": area, "score": score})

        # Sort by score
        instance_stats.sort(key=lambda x: x["score"], reverse=True)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Instance segmentation visualization
        colored_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_instances)))

        for i, instance_id in enumerate(unique_instances):
            mask = instance_mask == instance_id
            colored_mask[mask] = (colors[i][:3] * 255).astype(np.uint8)

        axes[0, 0].imshow(colored_mask)
        axes[0, 0].set_title(
            f"Instance Segmentation\n({len(unique_instances)} instances)",
            fontweight="bold",
        )
        axes[0, 0].axis("off")

        # 2. Scores heatmap
        im = axes[0, 1].imshow(scores_mask, cmap="viridis", vmin=0, vmax=1)
        axes[0, 1].set_title("Instance Confidence Scores", fontweight="bold")
        axes[0, 1].axis("off")
        plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

        # 3. Score distribution
        all_scores = [stat["score"] for stat in instance_stats]
        axes[1, 0].hist(
            all_scores, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[1, 0].axvline(
            x=np.mean(all_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(all_scores):.3f}",
        )
        axes[1, 0].set_xlabel("Confidence Score")
        axes[1, 0].set_ylabel("Instance Count")
        axes[1, 0].set_title("Score Distribution", fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Top instances by score
        top_instances = instance_stats[:10]
        instance_ids = [stat["id"] for stat in top_instances]
        scores = [stat["score"] for stat in top_instances]
        areas = [stat["area"] for stat in top_instances]

        bars = axes[1, 1].bar(
            range(len(top_instances)), scores, color="coral", alpha=0.7
        )
        axes[1, 1].set_xlabel("Top 10 Instances")
        axes[1, 1].set_ylabel("Confidence Score")
        axes[1, 1].set_title("Top Instances by Confidence", fontweight="bold")
        axes[1, 1].set_xticks(range(len(top_instances)))
        axes[1, 1].set_xticklabels([f"#{id}" for id in instance_ids], rotation=45)

        # Add area info as text on bars
        for i, (bar, area) in enumerate(zip(bars, areas)):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{area}px",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

        # Print summary statistics
        print(f"\n📊 Instance Analysis Summary:")
        print(f"   Total instances: {len(unique_instances)}")
        print(f"   Average confidence: {np.mean(all_scores):.3f}")
        print(f"   Score range: {np.min(all_scores):.3f} - {np.max(all_scores):.3f}")
        print(f"   Total change area: {sum(areas):,} pixels")

        print(f"\n💾 Instance analysis saved as '{output_path}'")

        return instance_stats

    def create_comprehensive_report(
        self, results_dict, output_path="comprehensive_report.png"
    ):
        """Create a comprehensive visualization report from detailed results."""

        if not results_dict or "masks" not in results_dict:
            print("❌ No detailed results provided")
            return

        masks = results_dict["masks"]
        stats = results_dict["statistics"]

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Score distributions
        if "iou_predictions" in stats:
            iou_scores = [
                mask["iou_pred"] for mask in masks if mask["iou_pred"] is not None
            ]
            axes[0, 0].hist(
                iou_scores, bins=20, alpha=0.7, color="lightblue", edgecolor="black"
            )
            axes[0, 0].axvline(
                x=stats["iou_predictions"]["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats['iou_predictions']['mean']:.3f}",
            )
            axes[0, 0].set_xlabel("IoU Score")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].set_title("IoU Predictions Distribution", fontweight="bold")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Stability scores
        if "stability_scores" in stats:
            stability_scores = [
                mask["stability_score"]
                for mask in masks
                if mask["stability_score"] is not None
            ]
            axes[0, 1].hist(
                stability_scores,
                bins=20,
                alpha=0.7,
                color="lightgreen",
                edgecolor="black",
            )
            axes[0, 1].axvline(
                x=stats["stability_scores"]["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats['stability_scores']['mean']:.3f}",
            )
            axes[0, 1].set_xlabel("Stability Score")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("Stability Scores Distribution", fontweight="bold")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Change confidence
        if "change_confidence" in stats:
            change_conf = [
                mask["change_confidence"]
                for mask in masks
                if mask["change_confidence"] is not None
            ]
            axes[0, 2].hist(
                change_conf, bins=20, alpha=0.7, color="lightyellow", edgecolor="black"
            )
            axes[0, 2].axvline(
                x=stats["change_confidence"]["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats['change_confidence']['mean']:.1f}",
            )
            axes[0, 2].set_xlabel("Change Confidence")
            axes[0, 2].set_ylabel("Count")
            axes[0, 2].set_title("Change Confidence Distribution", fontweight="bold")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Area distribution
        if "areas" in stats:
            areas = [mask["area"] for mask in masks if mask["area"] is not None]
            axes[1, 0].hist(
                areas, bins=20, alpha=0.7, color="lightcoral", edgecolor="black"
            )
            axes[1, 0].axvline(
                x=stats["areas"]["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats['areas']['mean']:.1f}",
            )
            axes[1, 0].set_xlabel("Area (pixels)")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Area Distribution", fontweight="bold")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Combined confidence vs area scatter
        combined_conf = [
            mask["combined_confidence"]
            for mask in masks
            if "combined_confidence" in mask
        ]
        areas_for_scatter = [
            mask["area"]
            for mask in masks
            if "combined_confidence" in mask and mask["area"] is not None
        ]

        if combined_conf and areas_for_scatter:
            scatter = axes[1, 1].scatter(
                areas_for_scatter,
                combined_conf,
                alpha=0.6,
                c=combined_conf,
                cmap="viridis",
                s=50,
            )
            axes[1, 1].set_xlabel("Area (pixels)")
            axes[1, 1].set_ylabel("Combined Confidence")
            axes[1, 1].set_title("Confidence vs Area", fontweight="bold")
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], shrink=0.8)

        # 6. Summary statistics text
        summary_text = f"""Detection Summary:
Total Instances: {len(masks)}
Processing Size: {results_dict['summary']['target_size']}
Original Shape: {results_dict['summary']['original_shape']}

Quality Metrics:"""

        if "iou_predictions" in stats:
            summary_text += f"""
IoU Predictions:
  Mean: {stats['iou_predictions']['mean']:.3f}
  Range: {stats['iou_predictions']['min']:.3f} - {stats['iou_predictions']['max']:.3f}"""

        if "stability_scores" in stats:
            summary_text += f"""
Stability Scores:
  Mean: {stats['stability_scores']['mean']:.3f}
  Range: {stats['stability_scores']['min']:.3f} - {stats['stability_scores']['max']:.3f}"""

        if "change_confidence" in stats:
            summary_text += f"""
Change Confidence:
  Mean: {stats['change_confidence']['mean']:.1f}
  Range: {stats['change_confidence']['min']:.1f} - {stats['change_confidence']['max']:.1f}"""

        if "areas" in stats:
            summary_text += f"""
Areas:
  Mean: {stats['areas']['mean']:.1f}
  Total: {stats['areas']['total']:,.0f} pixels"""

        axes[1, 2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis("off")
        axes[1, 2].set_title("Summary Statistics", fontweight="bold")

        plt.tight_layout()
        plt.suptitle(
            "Comprehensive Change Detection Report",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"💾 Comprehensive report saved as '{output_path}'")

    def run_complete_analysis(
        self, image1_path, image2_path, output_dir="change_detection_results"
    ):
        """Run complete change detection analysis with all outputs and visualizations."""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define output paths
        binary_path = os.path.join(output_dir, "binary_mask.tif")
        prob_path = os.path.join(output_dir, "probability_mask.tif")
        instance_path = os.path.join(output_dir, "instance_masks.tif")

        print("🔍 Running complete change detection analysis...")

        # Run detection with all outputs
        results = self.detect_changes(
            image1_path,
            image2_path,
            output_path=binary_path,
            export_probability=True,
            probability_output_path=prob_path,
            export_instance_masks=True,
            instance_masks_output_path=instance_path,
            return_detailed_results=True,
            return_results=False,
        )

        print("📊 Creating visualizations...")

        # Create all visualizations
        self.visualize_results(image1_path, image2_path, binary_path, prob_path)

        self.create_split_comparison(
            image1_path,
            image2_path,
            binary_path,
            prob_path,
            os.path.join(output_dir, "split_comparison.png"),
        )

        scores_path = instance_path.replace(".tif", "_scores.tif")
        self.analyze_instances(
            instance_path,
            scores_path,
            os.path.join(output_dir, "instance_analysis.png"),
        )

        self.create_comprehensive_report(
            results, os.path.join(output_dir, "comprehensive_report.png")
        )

        print(f"✅ Complete analysis finished! Results saved to: {output_dir}")
        return results

    def _save_instance_segmentation_masks(
        self, change_masks, output_path, transform, crs, original_shape, target_size
    ):
        """
        Save instance segmentation masks as a single GeoTIFF where each instance has a unique ID.

        Args:
            change_masks: Change detection masks (MaskData object)
            output_path (str): Output path for instance segmentation GeoTIFF
            transform: Rasterio transform
            crs: Coordinate reference system
            original_shape (tuple): Original image shape
            target_size (int): Processing target size
        """
        # Extract mask components
        mask_items = dict(change_masks.items())
        rles = mask_items.get("rles", [])

        # Create instance segmentation mask (each instance gets unique ID)
        instance_mask = np.zeros((target_size, target_size), dtype=np.uint16)

        # Process each mask and assign unique instance ID
        for instance_id, rle in enumerate(rles, start=1):
            if isinstance(rle, dict) and "size" in rle and "counts" in rle:
                try:
                    # Decode RLE to binary mask
                    size = rle["size"]
                    counts = rle["counts"]

                    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
                    pos = 0
                    value = 0

                    for count in counts:
                        if pos + count <= len(mask):
                            if value == 1:
                                mask[pos : pos + count] = 1
                            pos += count
                            value = 1 - value
                        else:
                            break

                    # RLE is column-major, reshape and transpose
                    mask = mask.reshape(size).T
                    if mask.shape != (target_size, target_size):
                        continue

                    # Assign instance ID to this mask
                    instance_mask[mask.astype(bool)] = instance_id

                except Exception as e:
                    print(f"Warning: Failed to process mask {instance_id}: {e}")
                    continue

        # Resize back to original shape if needed
        if original_shape != (target_size, target_size):
            instance_mask_resized = resize(
                instance_mask.astype(np.float32),
                original_shape,
                preserve_range=True,
                anti_aliasing=False,
                order=0,
            )
            instance_mask_final = np.round(instance_mask_resized).astype(np.uint16)
        else:
            instance_mask_final = instance_mask

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=instance_mask_final.shape[0],
            width=instance_mask_final.shape[1],
            count=1,
            dtype=instance_mask_final.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(instance_mask_final, 1)

            # Add metadata
            dst.update_tags(
                description="Instance segmentation mask with unique IDs for each change object",
                total_instances=str(len(rles)),
                background_value="0",
                instance_range=f"1-{len(rles)}",
            )

        print(
            f"Saved instance segmentation mask with {len(rles)} instances to {output_path}"
        )
        return len(rles)

    def _save_instance_scores_mask(
        self, change_masks, output_path, transform, crs, original_shape, target_size
    ):
        """
        Save instance scores/probability mask as a GeoTIFF where each instance has its confidence score.

        Args:
            change_masks: Change detection masks (MaskData object)
            output_path (str): Output path for instance scores GeoTIFF
            transform: Rasterio transform
            crs: Coordinate reference system
            original_shape (tuple): Original image shape
            target_size (int): Processing target size
        """
        # Extract mask components
        mask_items = dict(change_masks.items())
        rles = mask_items.get("rles", [])
        iou_preds = mask_items.get("iou_preds", None)
        stability_scores = mask_items.get("stability_score", None)
        change_confidence = mask_items.get("change_confidence", None)

        # Convert tensors to numpy if needed
        if iou_preds is not None:
            iou_preds = iou_preds.detach().cpu().numpy()
        if stability_scores is not None:
            stability_scores = stability_scores.detach().cpu().numpy()
        if change_confidence is not None:
            change_confidence = change_confidence.detach().cpu().numpy()

        # Create instance scores mask
        scores_mask = np.zeros((target_size, target_size), dtype=np.float32)

        # Process each mask and assign confidence score
        for instance_id, rle in enumerate(rles):
            if isinstance(rle, dict) and "size" in rle and "counts" in rle:
                try:
                    # Decode RLE to binary mask
                    size = rle["size"]
                    counts = rle["counts"]

                    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
                    pos = 0
                    value = 0

                    for count in counts:
                        if pos + count <= len(mask):
                            if value == 1:
                                mask[pos : pos + count] = 1
                            pos += count
                            value = 1 - value
                        else:
                            break

                    # RLE is column-major, reshape and transpose
                    mask = mask.reshape(size).T
                    if mask.shape != (target_size, target_size):
                        continue

                    # Calculate combined confidence score
                    confidence_score = 0.5  # Default
                    if iou_preds is not None and instance_id < len(iou_preds):
                        iou_score = float(iou_preds[instance_id])

                        if stability_scores is not None and instance_id < len(
                            stability_scores
                        ):
                            stability_score = float(stability_scores[instance_id])

                            if change_confidence is not None and instance_id < len(
                                change_confidence
                            ):
                                change_conf = float(change_confidence[instance_id])
                                # Normalize change confidence (typically around 145 threshold)
                                change_conf_norm = max(
                                    0.0, min(1.0, abs(change_conf) / 200.0)
                                )

                                # Weighted combination of scores
                                confidence_score = (
                                    0.35 * iou_score
                                    + 0.35 * stability_score
                                    + 0.3 * change_conf_norm
                                )
                            else:
                                confidence_score = 0.5 * (iou_score + stability_score)
                        else:
                            confidence_score = iou_score

                    # Assign confidence score to this mask
                    scores_mask[mask.astype(bool)] = confidence_score

                except Exception as e:
                    print(
                        f"Warning: Failed to process scores for mask {instance_id}: {e}"
                    )
                    continue

        # Resize back to original shape if needed
        if original_shape != (target_size, target_size):
            scores_mask_resized = resize(
                scores_mask,
                original_shape,
                preserve_range=True,
                anti_aliasing=True,
                order=1,
            )
            scores_mask_final = np.clip(scores_mask_resized, 0.0, 1.0).astype(
                np.float32
            )
        else:
            scores_mask_final = scores_mask

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=scores_mask_final.shape[0],
            width=scores_mask_final.shape[1],
            count=1,
            dtype=scores_mask_final.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(scores_mask_final, 1)

            # Add metadata
            dst.update_tags(
                description="Instance scores mask with confidence values for each change object",
                total_instances=str(len(rles)),
                background_value="0.0",
                score_range="0.0-1.0",
            )

        print(f"Saved instance scores mask with {len(rles)} instances to {output_path}")
        return len(rles)

    def _extract_detailed_results(
        self, change_masks, transform, crs, original_shape, target_size
    ):
        """
        Extract detailed results from change masks.

        Args:
            change_masks: Change detection masks (MaskData object)
            transform: Rasterio transform
            crs: Coordinate reference system
            original_shape (tuple): Original image shape
            target_size (int): Processing target size

        Returns:
            dict: Detailed results with mask information and statistics
        """
        # Extract mask components
        mask_items = dict(change_masks.items())
        rles = mask_items.get("rles", [])
        iou_preds = mask_items.get("iou_preds", None)
        stability_scores = mask_items.get("stability_score", None)
        change_confidence = mask_items.get("change_confidence", None)
        areas = mask_items.get("areas", None)
        boxes = mask_items.get("boxes", None)
        points = mask_items.get("points", None)

        # Convert tensors to numpy if needed
        if iou_preds is not None:
            iou_preds = iou_preds.detach().cpu().numpy()
        if stability_scores is not None:
            stability_scores = stability_scores.detach().cpu().numpy()
        if change_confidence is not None:
            change_confidence = change_confidence.detach().cpu().numpy()
        if areas is not None:
            areas = areas.detach().cpu().numpy()
        if boxes is not None:
            boxes = boxes.detach().cpu().numpy()
        if points is not None:
            points = points.detach().cpu().numpy()

        # Calculate statistics
        results = {
            "summary": {
                "total_masks": len(rles),
                "target_size": target_size,
                "original_shape": original_shape,
                "crs": str(crs),
                "transform": transform.to_gdal(),
            },
            "statistics": {},
            "masks": [],
        }

        # Calculate statistics for each metric
        if iou_preds is not None and len(iou_preds) > 0:
            results["statistics"]["iou_predictions"] = {
                "mean": float(np.mean(iou_preds)),
                "std": float(np.std(iou_preds)),
                "min": float(np.min(iou_preds)),
                "max": float(np.max(iou_preds)),
                "median": float(np.median(iou_preds)),
            }

        if stability_scores is not None and len(stability_scores) > 0:
            results["statistics"]["stability_scores"] = {
                "mean": float(np.mean(stability_scores)),
                "std": float(np.std(stability_scores)),
                "min": float(np.min(stability_scores)),
                "max": float(np.max(stability_scores)),
                "median": float(np.median(stability_scores)),
            }

        if change_confidence is not None and len(change_confidence) > 0:
            results["statistics"]["change_confidence"] = {
                "mean": float(np.mean(change_confidence)),
                "std": float(np.std(change_confidence)),
                "min": float(np.min(change_confidence)),
                "max": float(np.max(change_confidence)),
                "median": float(np.median(change_confidence)),
            }

        if areas is not None and len(areas) > 0:
            results["statistics"]["areas"] = {
                "mean": float(np.mean(areas)),
                "std": float(np.std(areas)),
                "min": float(np.min(areas)),
                "max": float(np.max(areas)),
                "median": float(np.median(areas)),
                "total": float(np.sum(areas)),
            }

        # Extract individual mask details
        for i in range(len(rles)):
            mask_info = {
                "mask_id": i,
                "iou_pred": (
                    float(iou_preds[i])
                    if iou_preds is not None and i < len(iou_preds)
                    else None
                ),
                "stability_score": (
                    float(stability_scores[i])
                    if stability_scores is not None and i < len(stability_scores)
                    else None
                ),
                "change_confidence": (
                    float(change_confidence[i])
                    if change_confidence is not None and i < len(change_confidence)
                    else None
                ),
                "area": int(areas[i]) if areas is not None and i < len(areas) else None,
                "bbox": (
                    boxes[i].tolist() if boxes is not None and i < len(boxes) else None
                ),
                "center_point": (
                    points[i].tolist()
                    if points is not None and i < len(points)
                    else None
                ),
            }

            # Calculate combined confidence score
            if all(
                v is not None
                for v in [
                    mask_info["iou_pred"],
                    mask_info["stability_score"],
                    mask_info["change_confidence"],
                ]
            ):
                # Normalize change confidence (145 is typical threshold)
                conf_norm = max(0.0, min(1.0, mask_info["change_confidence"] / 145.0))
                combined_score = (
                    0.3 * mask_info["iou_pred"]
                    + 0.3 * mask_info["stability_score"]
                    + 0.4 * conf_norm
                )
                mask_info["combined_confidence"] = float(combined_score)

            results["masks"].append(mask_info)

        # Sort masks by combined confidence if available
        if results["masks"] and "combined_confidence" in results["masks"][0]:
            results["masks"].sort(key=lambda x: x["combined_confidence"], reverse=True)

        return results


def download_checkpoint(
    model_type: str = "vit_h", checkpoint_dir: Optional[str] = None
) -> str:
    """Download the SAM model checkpoint.

    Args:
        model_type (str, optional): The model type. Can be one of ['vit_h', 'vit_l', 'vit_b'].
            Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
        checkpoint_dir (str, optional): The checkpoint_dir directory. Defaults to None,
            which uses "~/.cache/torch/hub/checkpoints".
    """

    model_types = {
        "vit_h": {
            "name": "sam_vit_h_4b8939.pth",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        },
        "vit_l": {
            "name": "sam_vit_l_0b3195.pth",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        },
        "vit_b": {
            "name": "sam_vit_b_01ec64.pth",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        },
    }

    if model_type not in model_types:
        raise ValueError(
            f"Invalid model_type: {model_type}. It must be one of {', '.join(model_types)}"
        )

    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get(
            "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
        )

    checkpoint = os.path.join(checkpoint_dir, model_types[model_type]["name"])
    if not os.path.exists(checkpoint):
        print(f"Model checkpoint for {model_type} not found.")
        url = model_types[model_type]["url"]
        if isinstance(url, str):
            download_file(url, checkpoint)

    return checkpoint


# Available ChangeStar model variants
CHANGESTAR_MODELS = {
    "s0_s1c1_vitb": "s0_init_s1c1_changestar_vitb_1x256",
    "s0_s1c5_vitb": "s0_init_s1c5_changestar_vitb_1x256",
    "s0_s9c1_vitb": "s0_init_s9c1_changestar_vitb_1x256",
    "s0_xview2_s1c5_vitb": "s0_init_xView2_ft_s1c5_changestar_vitb_1x256",
    "s1_s1c1_vitb": "s1_init_s1c1_changestar_vitb_1x256",
    "s1_s1c1_vitl": "s1_init_s1c1_changestar_vitl_1x256",
    "s9_s9c1_vitb": "s9_init_s9c1_changestar_vitb_1x256",
}


def list_changestar_models() -> Dict[str, str]:
    """List available ChangeStar model variants.

    Returns:
        Dict[str, str]: A dictionary mapping short model names to their
            full function names in torchange.

    Example:
        >>> from geoai.change_detection import list_changestar_models
        >>> models = list_changestar_models()
        >>> for short_name, full_name in models.items():
        ...     print(f"{short_name}: {full_name}")
    """
    return dict(CHANGESTAR_MODELS)


class ChangeStarDetection:
    """Change detection using ChangeStar models from the torchange package.

    ChangeStar is a building change detection model that uses Changen2
    pre-trained weights for zero-shot or fine-tuned building change detection.
    It takes two images (before/after) and outputs change maps and semantic
    segmentation maps.

    This class wraps the torchange ChangeStar models with GeoTIFF I/O
    support, tiled processing for large rasters, and vector output
    capabilities.

    Args:
        model_name (str): Short name of the ChangeStar model variant.
            Use :func:`list_changestar_models` to see available options.
            Defaults to ``"s1_s1c1_vitb"``.
        device (str or None): Device to use for inference (``"cuda"``,
            ``"cpu"``, or ``None`` for auto-detection). Defaults to ``None``.

    Raises:
        ImportError: If ``torchange`` or ``albumentations`` is not installed.
        ValueError: If an invalid ``model_name`` is provided.

    Example:
        >>> from geoai.change_detection import ChangeStarDetection
        >>> detector = ChangeStarDetection(model_name="s1_s1c1_vitb")
        >>> result = detector.predict("before.tif", "after.tif")
    """

    def __init__(
        self,
        model_name: str = "s1_s1c1_vitb",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._preprocess = None
        self._init_model()

    def _init_model(self):
        """Initialize the ChangeStar model and preprocessing pipeline."""
        try:
            import torch
            from torchange.models import changen2
            import albumentations as A
            import albumentations.pytorch
        except ImportError as e:
            raise ImportError(
                "The 'torchange' and 'albumentations' packages are required "
                "for ChangeStar change detection. "
                "Please install them using: pip install torchange albumentations\n"
                f"Original error: {e}"
            ) from e

        if self.model_name not in CHANGESTAR_MODELS:
            raise ValueError(
                f"Invalid model_name: '{self.model_name}'. "
                f"Available models: {list(CHANGESTAR_MODELS.keys())}"
            )

        # Get the model constructor function
        func_name = CHANGESTAR_MODELS[self.model_name]
        model_fn = getattr(changen2, func_name)
        self.model = model_fn()
        self.model.eval()

        # Set device
        if self.device is None:
            try:
                import ever as er

                self._device = er.auto_device()
            except ImportError:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self._device = torch.device(self.device)

        self.model = self.model.to(self._device)

        # Set up preprocessing
        self._preprocess = A.Compose(
            [A.Normalize(), A.pytorch.ToTensorV2()],
            additional_targets={"image2": "image"},
        )

    def _read_geotiff(
        self, path: str, window=None
    ) -> Tuple[np.ndarray, Any, Any, Tuple[int, int]]:
        """Read a GeoTIFF file and return image data with georeference info.

        Args:
            path (str): Path to the GeoTIFF file.
            window: Optional rasterio window for reading a subset.

        Returns:
            Tuple of (image_array, transform, crs, shape):
                - image_array: numpy array with shape (H, W, 3) in uint8.
                - transform: rasterio affine transform.
                - crs: coordinate reference system.
                - shape: original (height, width) of the read region.
        """
        with rasterio.open(path) as src:
            if window is not None:
                img_data = src.read(window=window)
                transform = src.window_transform(window)
            else:
                img_data = src.read()
                transform = src.transform
            crs = src.crs

            # Convert from (bands, height, width) to (height, width, bands)
            img_data = np.transpose(img_data, (1, 2, 0))

            # Use only RGB bands (first 3 channels)
            if img_data.shape[2] >= 3:
                img_data = img_data[:, :, :3]

            # Normalize to 0-255 range if needed
            if img_data.dtype != np.uint8:
                img_min = img_data.min()
                img_max = img_data.max()
                if img_max > img_min:
                    img_data = (
                        (img_data - img_min) / (img_max - img_min) * 255
                    ).astype(np.uint8)
                else:
                    img_data = np.zeros_like(img_data, dtype=np.uint8)

            shape = img_data.shape[:2]
            return img_data, transform, crs, shape

    def _read_and_align_images(
        self, image1_path: str, image2_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Any, Any, Tuple[int, int]]:
        """Read and align two GeoTIFF images by their spatial overlap.

        Args:
            image1_path (str): Path to the first (before) image.
            image2_path (str): Path to the second (after) image.

        Returns:
            Tuple of (img1, img2, transform, crs, shape):
                - img1: first image as (H, W, 3) uint8 array.
                - img2: second image as (H, W, 3) uint8 array.
                - transform: affine transform for the overlap area.
                - crs: coordinate reference system.
                - shape: (height, width) of the overlap area.

        Raises:
            ValueError: If the images do not overlap.
        """
        with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
            bounds1 = src1.bounds
            bounds2 = src2.bounds

            left = max(bounds1.left, bounds2.left)
            bottom = max(bounds1.bottom, bounds2.bottom)
            right = min(bounds1.right, bounds2.right)
            top = min(bounds1.top, bounds2.top)

            if left >= right or bottom >= top:
                raise ValueError("Images do not overlap")

            intersection_bounds = (left, bottom, right, top)

            window1 = from_bounds(*intersection_bounds, src1.transform)
            window2 = from_bounds(*intersection_bounds, src2.transform)

            img1_data = src1.read(window=window1)
            img2_data = src2.read(window=window2)

            transform = src1.window_transform(window1)
            crs = src1.crs

            img1_data = np.transpose(img1_data, (1, 2, 0))
            img2_data = np.transpose(img2_data, (1, 2, 0))

            # Use only RGB bands
            if img1_data.shape[2] >= 3:
                img1_data = img1_data[:, :, :3]
            if img2_data.shape[2] >= 3:
                img2_data = img2_data[:, :, :3]

            # Normalize to uint8
            for arr_name in ["img1_data", "img2_data"]:
                arr = locals()[arr_name]
                if arr.dtype != np.uint8:
                    arr_min = arr.min()
                    arr_max = arr.max()
                    if arr_max > arr_min:
                        arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(
                            np.uint8
                        )
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                    if arr_name == "img1_data":
                        img1_data = arr
                    else:
                        img2_data = arr

            # Ensure same size by resizing img2 to match img1
            if img1_data.shape[:2] != img2_data.shape[:2]:
                img2_data = resize(
                    img2_data,
                    img1_data.shape[:2],
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.uint8)

            shape = img1_data.shape[:2]
            return img1_data, img2_data, transform, crs, shape

    def _preprocess_pair(self, img1: np.ndarray, img2: np.ndarray):
        """Preprocess a pair of images for model input.

        Args:
            img1 (np.ndarray): First image as (H, W, 3) uint8 array.
            img2 (np.ndarray): Second image as (H, W, 3) uint8 array.

        Returns:
            torch.Tensor: Concatenated and normalized tensor of shape
                (1, 6, H, W).
        """
        import torch

        data = self._preprocess(image=img1, image2=img2)
        img = torch.cat([data["image"], data["image2"]], dim=0)  # (6, H, W)
        return img.unsqueeze(0)  # (1, 6, H, W)

    def _predict_tile(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """Run model inference on a single tile pair.

        Args:
            img1 (np.ndarray): First image tile as (H, W, 3) uint8 array.
            img2 (np.ndarray): Second image tile as (H, W, 3) uint8 array.

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - ``"change_map"``: binary change map as (H, W) uint8 array.
                - ``"change_prob"``: change probability as (H, W) float32 array.
                - ``"t1_semantic"``: semantic segmentation for t1 as
                  (H, W) uint8 array.
                - ``"t2_semantic"``: semantic segmentation for t2 as
                  (H, W) uint8 array.
                - ``"t1_semantic_prob"``: semantic probability for t1 as
                  (H, W) float32 array.
                - ``"t2_semantic_prob"``: semantic probability for t2 as
                  (H, W) float32 array.
        """
        import torch

        input_tensor = self._preprocess_pair(img1, img2)
        input_tensor = input_tensor.to(self._device)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Move all tensors to CPU
        prediction = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in prediction.items()
        }

        # Extract predictions
        change_prob = torch.sigmoid(prediction["change_prediction"].squeeze()).numpy()
        change_map = (change_prob > 0.5).astype(np.uint8)

        t1_sem_prob = torch.sigmoid(
            prediction["t1_semantic_prediction"].squeeze()
        ).numpy()
        t1_semantic = (t1_sem_prob > 0.5).astype(np.uint8)

        t2_sem_prob = torch.sigmoid(
            prediction["t2_semantic_prediction"].squeeze()
        ).numpy()
        t2_semantic = (t2_sem_prob > 0.5).astype(np.uint8)

        return {
            "change_map": change_map,
            "change_prob": change_prob.astype(np.float32),
            "t1_semantic": t1_semantic,
            "t2_semantic": t2_semantic,
            "t1_semantic_prob": t1_sem_prob.astype(np.float32),
            "t2_semantic_prob": t2_sem_prob.astype(np.float32),
        }

    def predict(
        self,
        image1_path: str,
        image2_path: str,
        output_change: Optional[str] = None,
        output_t1_semantic: Optional[str] = None,
        output_t2_semantic: Optional[str] = None,
        output_vector: Optional[str] = None,
        tile_size: int = 1024,
        overlap: int = 64,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Run change detection on two GeoTIFF images.

        Supports tiled processing for large rasters. When the image is
        larger than ``tile_size``, it is split into overlapping tiles,
        each tile is processed independently, and results are stitched
        back together.

        Args:
            image1_path (str): Path to the first (before) GeoTIFF image.
            image2_path (str): Path to the second (after) GeoTIFF image.
            output_change (str or None): Path to save the binary change
                mask as a GeoTIFF. Defaults to ``None``.
            output_t1_semantic (str or None): Path to save the t1 semantic
                segmentation as a GeoTIFF. Defaults to ``None``.
            output_t2_semantic (str or None): Path to save the t2 semantic
                segmentation as a GeoTIFF. Defaults to ``None``.
            output_vector (str or None): Path to save change polygons as
                a vector file (e.g., GeoJSON, GPKG). Defaults to ``None``.
            tile_size (int): Size of tiles for processing large images.
                Defaults to ``1024``.
            overlap (int): Overlap between tiles in pixels. Defaults to
                ``64``.
            threshold (float): Probability threshold for binary change map.
                Defaults to ``0.5``.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            Dict[str, np.ndarray]: Dictionary with keys:
                - ``"change_map"``: binary change map as (H, W) uint8 array.
                - ``"change_prob"``: change probability as (H, W)
                  float32 array.
                - ``"t1_semantic"``: semantic segmentation for t1 as
                  (H, W) uint8 array.
                - ``"t2_semantic"``: semantic segmentation for t2 as
                  (H, W) uint8 array.
                - ``"t1_semantic_prob"``: semantic probability for t1 as
                  (H, W) float32 array.
                - ``"t2_semantic_prob"``: semantic probability for t2 as
                  (H, W) float32 array.

        Example:
            >>> detector = ChangeStarDetection(model_name="s1_s1c1_vitb")
            >>> result = detector.predict(
            ...     "before.tif",
            ...     "after.tif",
            ...     output_change="change_map.tif",
            ...     output_vector="changes.gpkg",
            ... )
        """
        # Read and align images
        img1, img2, transform, crs, shape = self._read_and_align_images(
            image1_path, image2_path
        )
        h, w = shape

        # Decide whether to use tiled processing
        if h <= tile_size and w <= tile_size:
            result = self._predict_tile(img1, img2)
        else:
            result = self._predict_tiled(
                img1, img2, tile_size=tile_size, overlap=overlap
            )

        # Apply threshold
        result["change_map"] = (result["change_prob"] > threshold).astype(np.uint8)
        result["t1_semantic"] = (result["t1_semantic_prob"] > threshold).astype(
            np.uint8
        )
        result["t2_semantic"] = (result["t2_semantic_prob"] > threshold).astype(
            np.uint8
        )

        # Save outputs
        if output_change:
            self._save_raster(
                result["change_map"], output_change, transform, crs, dtype=np.uint8
            )

        if output_t1_semantic:
            self._save_raster(
                result["t1_semantic"],
                output_t1_semantic,
                transform,
                crs,
                dtype=np.uint8,
            )

        if output_t2_semantic:
            self._save_raster(
                result["t2_semantic"],
                output_t2_semantic,
                transform,
                crs,
                dtype=np.uint8,
            )

        if output_vector:
            self._save_vector(result["change_map"], output_vector, transform, crs)

        return result

    def _predict_tiled(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 64,
    ) -> Dict[str, np.ndarray]:
        """Process large images in overlapping tiles and merge results.

        Args:
            img1 (np.ndarray): First image as (H, W, 3) uint8 array.
            img2 (np.ndarray): Second image as (H, W, 3) uint8 array.
            tile_size (int): Size of each tile. Defaults to ``1024``.
            overlap (int): Number of pixels to overlap between tiles.
                Defaults to ``64``.

        Returns:
            Dict[str, np.ndarray]: Merged prediction results.
        """
        h, w = img1.shape[:2]
        stride = tile_size - overlap

        # Initialize output arrays
        change_prob = np.zeros((h, w), dtype=np.float32)
        t1_sem_prob = np.zeros((h, w), dtype=np.float32)
        t2_sem_prob = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        # Process tiles
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate tile bounds
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile1 = img1[y_start:y_end, x_start:x_end]
                tile2 = img2[y_start:y_end, x_start:x_end]

                # Skip empty tiles
                if tile1.max() == 0 and tile2.max() == 0:
                    continue

                # Pad if needed
                th, tw = tile1.shape[:2]
                if th < tile_size or tw < tile_size:
                    pad_h = tile_size - th
                    pad_w = tile_size - tw
                    tile1 = np.pad(
                        tile1,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="reflect",
                    )
                    tile2 = np.pad(
                        tile2,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="reflect",
                    )

                tile_result = self._predict_tile(tile1, tile2)

                # Crop padding if applied
                tile_change = tile_result["change_prob"][:th, :tw]
                tile_t1 = tile_result["t1_semantic_prob"][:th, :tw]
                tile_t2 = tile_result["t2_semantic_prob"][:th, :tw]

                # Accumulate (average overlapping regions)
                change_prob[y_start:y_end, x_start:x_end] += tile_change
                t1_sem_prob[y_start:y_end, x_start:x_end] += tile_t1
                t2_sem_prob[y_start:y_end, x_start:x_end] += tile_t2
                count[y_start:y_end, x_start:x_end] += 1

        # Average overlapping regions
        mask = count > 0
        change_prob[mask] /= count[mask]
        t1_sem_prob[mask] /= count[mask]
        t2_sem_prob[mask] /= count[mask]

        return {
            "change_map": (change_prob > 0.5).astype(np.uint8),
            "change_prob": change_prob,
            "t1_semantic": (t1_sem_prob > 0.5).astype(np.uint8),
            "t2_semantic": (t2_sem_prob > 0.5).astype(np.uint8),
            "t1_semantic_prob": t1_sem_prob,
            "t2_semantic_prob": t2_sem_prob,
        }

    def _save_raster(
        self,
        data: np.ndarray,
        output_path: str,
        transform: Any,
        crs: Any,
        dtype=np.uint8,
    ) -> None:
        """Save a 2D array as a GeoTIFF.

        Args:
            data (np.ndarray): 2D array to save.
            output_path (str): Path for the output GeoTIFF.
            transform: Rasterio affine transform.
            crs: Coordinate reference system.
            dtype: Output data type. Defaults to ``np.uint8``.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(data.astype(dtype), 1)

    def _save_vector(
        self,
        change_map: np.ndarray,
        output_path: str,
        transform: Any,
        crs: Any,
    ) -> None:
        """Convert a binary change map to vector polygons and save.

        Args:
            change_map (np.ndarray): Binary change map as (H, W) uint8.
            output_path (str): Path for the output vector file
                (GeoJSON, GPKG, etc.).
            transform: Rasterio affine transform.
            crs: Coordinate reference system.
        """
        try:
            import geopandas as gpd
            from rasterio.features import shapes
            from shapely.geometry import shape as shapely_shape
        except ImportError as e:
            print(
                f"Warning: Cannot save vector output. "
                f"Missing dependency: {e}. "
                f"Install geopandas and shapely."
            )
            return

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Extract polygons from the binary mask
        mask = change_map.astype(np.uint8)
        geometries = []
        values = []

        for geom, val in shapes(mask, transform=transform):
            if val == 1:  # Only keep change regions
                geometries.append(shapely_shape(geom))
                values.append(int(val))

        if not geometries:
            print("No change polygons found.")
            return

        gdf = gpd.GeoDataFrame(
            {"change": values},
            geometry=geometries,
            crs=crs,
        )

        # Determine driver from extension
        ext = os.path.splitext(output_path)[1].lower()
        driver_map = {
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
            ".shp": "ESRI Shapefile",
        }
        driver = driver_map.get(ext, "GPKG")

        gdf.to_file(output_path, driver=driver)

    def visualize(
        self,
        image1_path: str,
        image2_path: str,
        result: Optional[Dict[str, np.ndarray]] = None,
        figsize: Tuple[int, int] = (20, 5),
        title1: str = "Before (T1)",
        title2: str = "After (T2)",
        **kwargs: Any,
    ) -> plt.Figure:
        """Visualize change detection results.

        Shows the before image, after image, change map, and optionally
        the semantic segmentation maps side by side.

        Args:
            image1_path (str): Path to the first (before) image.
            image2_path (str): Path to the second (after) image.
            result (Dict or None): Pre-computed prediction result from
                :meth:`predict`. If ``None``, prediction is run first.
                Defaults to ``None``.
            figsize (Tuple[int, int]): Figure size. Defaults to
                ``(20, 5)``.
            title1 (str): Title for the before image. Defaults to
                ``"Before (T1)"``.
            title2 (str): Title for the after image. Defaults to
                ``"After (T2)"``.
            **kwargs: Additional keyword arguments passed to
                :meth:`predict`.

        Returns:
            matplotlib.figure.Figure: The figure object.

        Example:
            >>> detector = ChangeStarDetection()
            >>> fig = detector.visualize("before.tif", "after.tif")
        """
        if result is None:
            result = self.predict(image1_path, image2_path, **kwargs)

        img1, img2, _, _, _ = self._read_and_align_images(image1_path, image2_path)

        fig, axes = plt.subplots(1, 5, figsize=figsize)

        axes[0].imshow(img1)
        axes[0].set_title(title1)
        axes[0].axis("off")

        axes[1].imshow(img2)
        axes[1].set_title(title2)
        axes[1].axis("off")

        axes[2].imshow(result["change_map"], cmap="gray")
        axes[2].set_title("Change Map")
        axes[2].axis("off")

        axes[3].imshow(result["t1_semantic"], cmap="gray")
        axes[3].set_title("T1 Buildings")
        axes[3].axis("off")

        axes[4].imshow(result["t2_semantic"], cmap="gray")
        axes[4].set_title("T2 Buildings")
        axes[4].axis("off")

        plt.tight_layout()
        return fig

    def visualize_overlay(
        self,
        image1_path: str,
        image2_path: str,
        result: Optional[Dict[str, np.ndarray]] = None,
        figsize: Tuple[int, int] = (15, 5),
        alpha: float = 0.4,
        title1: str = "Before (T1)",
        title2: str = "After (T2)",
        **kwargs: Any,
    ) -> plt.Figure:
        """Visualize results with change map overlaid on the images.

        Args:
            image1_path (str): Path to the first (before) image.
            image2_path (str): Path to the second (after) image.
            result (Dict or None): Pre-computed prediction result from
                :meth:`predict`. If ``None``, prediction is run first.
                Defaults to ``None``.
            figsize (Tuple[int, int]): Figure size. Defaults to
                ``(15, 5)``.
            alpha (float): Transparency for the overlay. Defaults to
                ``0.4``.
            title1 (str): Title for the before image. Defaults to
                ``"Before (T1)"``.
            title2 (str): Title for the after image. Defaults to
                ``"After (T2)"``.
            **kwargs: Additional keyword arguments passed to
                :meth:`predict`.

        Returns:
            matplotlib.figure.Figure: The figure object.

        Example:
            >>> detector = ChangeStarDetection()
            >>> fig = detector.visualize_overlay("before.tif", "after.tif")
        """
        if result is None:
            result = self.predict(image1_path, image2_path, **kwargs)

        img1, img2, _, _, _ = self._read_and_align_images(image1_path, image2_path)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Before image with T1 buildings overlay
        axes[0].imshow(img1)
        change_overlay_t1 = np.zeros((*result["t1_semantic"].shape, 4))
        change_overlay_t1[result["t1_semantic"] == 1] = [0, 0, 1, alpha]
        axes[0].imshow(change_overlay_t1)
        axes[0].set_title(f"{title1}\n(Blue = Buildings)")
        axes[0].axis("off")

        # After image with T2 buildings overlay
        axes[1].imshow(img2)
        change_overlay_t2 = np.zeros((*result["t2_semantic"].shape, 4))
        change_overlay_t2[result["t2_semantic"] == 1] = [0, 0, 1, alpha]
        axes[1].imshow(change_overlay_t2)
        axes[1].set_title(f"{title2}\n(Blue = Buildings)")
        axes[1].axis("off")

        # After image with change overlay
        axes[2].imshow(img2)
        change_overlay = np.zeros((*result["change_map"].shape, 4))
        change_overlay[result["change_map"] == 1] = [1, 0, 0, alpha]
        axes[2].imshow(change_overlay)
        axes[2].set_title("Change Detection\n(Red = Change)")
        axes[2].axis("off")

        plt.tight_layout()
        return fig


def changestar_detect(
    image1_path: str,
    image2_path: str,
    model_name: str = "s1_s1c1_vitb",
    output_change: Optional[str] = None,
    output_t1_semantic: Optional[str] = None,
    output_t2_semantic: Optional[str] = None,
    output_vector: Optional[str] = None,
    tile_size: int = 1024,
    overlap: int = 64,
    threshold: float = 0.5,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Convenience function for ChangeStar building change detection.

    Creates a :class:`ChangeStarDetection` instance and runs prediction
    in one step. For repeated use, instantiate :class:`ChangeStarDetection`
    directly to avoid reloading the model.

    Args:
        image1_path (str): Path to the first (before) GeoTIFF image.
        image2_path (str): Path to the second (after) GeoTIFF image.
        model_name (str): Short name of the ChangeStar model variant.
            Use :func:`list_changestar_models` to see options.
            Defaults to ``"s1_s1c1_vitb"``.
        output_change (str or None): Path to save the binary change mask
            as a GeoTIFF. Defaults to ``None``.
        output_t1_semantic (str or None): Path to save the t1 semantic
            segmentation as a GeoTIFF. Defaults to ``None``.
        output_t2_semantic (str or None): Path to save the t2 semantic
            segmentation as a GeoTIFF. Defaults to ``None``.
        output_vector (str or None): Path to save change polygons as a
            vector file. Defaults to ``None``.
        tile_size (int): Size of tiles for processing large images.
            Defaults to ``1024``.
        overlap (int): Overlap between tiles in pixels. Defaults to
            ``64``.
        threshold (float): Probability threshold for binary change map.
            Defaults to ``0.5``.
        device (str or None): Device for inference. Defaults to ``None``
            (auto-detect).
        **kwargs: Additional keyword arguments passed to
            :meth:`ChangeStarDetection.predict`.

    Returns:
        Dict[str, np.ndarray]: Dictionary with prediction results.

    Example:
        >>> from geoai import changestar_detect
        >>> result = changestar_detect(
        ...     "before.tif",
        ...     "after.tif",
        ...     output_change="change_map.tif",
        ...     output_vector="changes.gpkg",
        ... )
    """
    detector = ChangeStarDetection(model_name=model_name, device=device)
    return detector.predict(
        image1_path,
        image2_path,
        output_change=output_change,
        output_t1_semantic=output_t1_semantic,
        output_t2_semantic=output_t2_semantic,
        output_vector=output_vector,
        tile_size=tile_size,
        overlap=overlap,
        threshold=threshold,
        **kwargs,
    )
