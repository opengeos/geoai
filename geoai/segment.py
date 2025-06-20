"""This module provides functionality for segmenting high-resolution satellite imagery using vision-language models."""

import os
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import numpy as np
import rasterio
import torch
import geopandas as gpd
from PIL import Image
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from shapely.geometry import box, Polygon
from tqdm import tqdm
from transformers import (
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    AutoModelForMaskGeneration,
    AutoProcessor,
    pipeline,
)


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    """Represents a detection result with score, label, bounding box, and optional mask."""

    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


class GroundedSAM:
    """
    A class for segmenting remote sensing imagery using text prompts with Grounding DINO + SAM.

    This class combines Grounding DINO for object detection and Segment Anything Model (SAM) for
    precise segmentation based on text prompts. It can process large GeoTIFF files by tiling them
    and handles proper georeferencing in the outputs.

    Args:
        detector_id (str): Hugging Face model ID for Grounding DINO. Defaults to "IDEA-Research/grounding-dino-tiny".
        segmenter_id (str): Hugging Face model ID for SAM. Defaults to "facebook/sam-vit-base".
        device (str): Device to run the models on ('cuda', 'cpu'). If None, will use CUDA if available.
        tile_size (int): Size of tiles to process the image in chunks. Defaults to 1024.
        overlap (int): Overlap between tiles to avoid edge artifacts. Defaults to 128.
        threshold (float): Detection threshold for Grounding DINO. Defaults to 0.3.

    Attributes:
        detector_id (str): The Grounding DINO model ID.
        segmenter_id (str): The SAM model ID.
        device (str): The device being used ('cuda' or 'cpu').
        tile_size (int): Size of tiles for processing.
        overlap (int): Overlap between tiles.
        threshold (float): Detection threshold.
        object_detector: The Grounding DINO pipeline.
        segmentator: The SAM model.
        processor: The SAM processor.
    """

    def __init__(
        self,
        detector_id="IDEA-Research/grounding-dino-tiny",
        segmenter_id="facebook/sam-vit-base",
        device=None,
        tile_size=1024,
        overlap=128,
        threshold=0.3,
    ):
        """
        Initialize the GroundedSAM with the specified models and settings.

        Args:
            detector_id (str): Hugging Face model ID for Grounding DINO.
            segmenter_id (str): Hugging Face model ID for SAM.
            device (str): Device to run the models on ('cuda', 'cpu').
            tile_size (int): Size of tiles to process the image in chunks.
            overlap (int): Overlap between tiles to avoid edge artifacts.
            threshold (float): Detection threshold for Grounding DINO.
        """
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load models
        self._load_models()

        print(f"GroundedSAM initialized on {self.device}")

    def _load_models(self):
        """Load the Grounding DINO and SAM models."""
        # Load Grounding DINO
        self.object_detector = pipeline(
            model=self.detector_id,
            task="zero-shot-object-detection",
            device=self.device,
        )

        # Load SAM
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(
            self.segmenter_id
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.segmenter_id)

    def _detect(self, image: Image.Image, labels: List[str]) -> List[DetectionResult]:
        """
        Use Grounding DINO to detect objects in an image.

        Args:
            image (Image.Image): PIL image to detect objects in.
            labels (List[str]): List of text labels to detect.

        Returns:
            List[DetectionResult]: List of detection results.
        """
        # Ensure labels end with periods
        labels = [label if label.endswith(".") else label + "." for label in labels]

        results = self.object_detector(
            image, candidate_labels=labels, threshold=self.threshold
        )
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def _apply_nms(
        self, detections: List[DetectionResult], iou_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections (List[DetectionResult]): List of detection results.
            iou_threshold (float): IoU threshold for NMS.

        Returns:
            List[DetectionResult]: Filtered detection results.
        """
        if not detections:
            return detections

        # Convert to format for NMS
        boxes = []
        scores = []

        for detection in detections:
            boxes.append(
                [
                    detection.box.xmin,
                    detection.box.ymin,
                    detection.box.xmax,
                    detection.box.ymax,
                ]
            )
            scores.append(detection.score)

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, iou_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []

    def _get_boxes(self, results: List[DetectionResult]) -> List[List[List[float]]]:
        """Extract bounding boxes from detection results."""
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)
        return [boxes]

    def _refine_masks(
        self, masks: torch.BoolTensor, polygon_refinement: bool = False
    ) -> List[np.ndarray]:
        """Refine masks from SAM output."""
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self._mask_to_polygon(mask)
                if polygon:
                    mask = self._polygon_to_mask(polygon, shape)
                    masks[idx] = mask

        return masks

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Convert mask to polygon coordinates."""
        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def _polygon_to_mask(
        self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Convert polygon to mask."""
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask

    def _separate_instances(
        self, mask: np.ndarray, min_area: int = 50
    ) -> List[np.ndarray]:
        """
        Separate individual instances from a combined mask using connected components.

        Args:
            mask (np.ndarray): Combined binary mask.
            min_area (int): Minimum area threshold for valid instances.

        Returns:
            List[np.ndarray]: List of individual instance masks.
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        instances = []
        for i in range(1, num_labels):  # Skip background (label 0)
            # Get area of the component
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by minimum area
            if area >= min_area:
                # Create mask for this instance
                instance_mask = (labels == i).astype(np.uint8) * 255
                instances.append(instance_mask)

        return instances

    def _mask_to_polygons(
        self,
        mask: np.ndarray,
        transform,
        x_offset: int = 0,
        y_offset: int = 0,
        min_area: int = 50,
        simplify_tolerance: float = 1.0,
    ) -> List[Dict]:
        """
        Convert mask to individual polygons with geospatial coordinates.

        Args:
            mask (np.ndarray): Binary mask.
            transform: Rasterio transform object.
            x_offset (int): X offset for tile position.
            y_offset (int): Y offset for tile position.
            min_area (int): Minimum area threshold for valid polygons.
            simplify_tolerance (float): Tolerance for polygon simplification.

        Returns:
            List[Dict]: List of polygon dictionaries with geometry and properties.
        """
        polygons = []

        # Get individual instances
        instances = self._separate_instances(mask, min_area)

        for instance_mask in instances:
            # Find contours
            contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Filter by minimum area
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                # Simplify contour
                epsilon = simplify_tolerance
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

                # Convert to pixel coordinates (add offsets)
                pixel_coords = simplified_contour.reshape(-1, 2)
                pixel_coords = pixel_coords + [x_offset, y_offset]

                # Convert to geographic coordinates
                geo_coords = []
                for x, y in pixel_coords:
                    geo_x, geo_y = transform * (x, y)
                    geo_coords.append([geo_x, geo_y])

                # Close the polygon if needed
                if len(geo_coords) > 2:
                    if geo_coords[0] != geo_coords[-1]:
                        geo_coords.append(geo_coords[0])

                    # Create Shapely polygon
                    try:
                        polygon = Polygon(geo_coords)
                        if polygon.is_valid and polygon.area > 0:
                            polygons.append({"geometry": polygon, "area_pixels": area})
                    except Exception as e:
                        print(f"Error creating polygon: {e}")
                        continue

        return polygons

    def _segment(
        self,
        image: Image.Image,
        detection_results: List[DetectionResult],
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """
        Use SAM to generate masks for detected objects.

        Args:
            image (Image.Image): PIL image.
            detection_results (List[DetectionResult]): Detection results from Grounding DINO.
            polygon_refinement (bool): Whether to refine masks using polygon fitting.

        Returns:
            List[DetectionResult]: Detection results with masks.
        """
        if not detection_results:
            return detection_results

        boxes = self._get_boxes(detection_results)
        inputs = self.processor(
            images=image, input_boxes=boxes, return_tensors="pt"
        ).to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        masks = self._refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def segment_image(
        self,
        input_path,
        output_path,
        text_prompts,
        polygon_refinement=False,
        export_boxes=False,
        export_polygons=True,
        smoothing_sigma=1.0,
        nms_threshold=0.5,
        min_polygon_area=50,
        simplify_tolerance=2.0,
    ):
        """
        Segment a GeoTIFF image using text prompts with improved instance segmentation.

        Args:
            input_path (str): Path to the input GeoTIFF file.
            output_path (str): Path where the output GeoTIFF will be saved.
            text_prompts (Union[str, List[str]]): Text prompt(s) describing what to segment.
            polygon_refinement (bool): Whether to refine masks using polygon fitting.
            export_boxes (bool): Whether to export bounding boxes as a separate vector file.
            export_polygons (bool): Whether to export segmentation polygons as vector file.
            smoothing_sigma (float): Sigma value for Gaussian smoothing to reduce blockiness.
            nms_threshold (float): Non-maximum suppression threshold for removing overlapping detections.
            min_polygon_area (int): Minimum area in pixels for valid polygons.
            simplify_tolerance (float): Tolerance for polygon simplification.

        Returns:
            Dict: Dictionary containing paths to output files.
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # Open the input GeoTIFF
        with rasterio.open(input_path) as src:
            # Get metadata
            meta = src.meta
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs

            # Create output metadata for segmentation masks
            out_meta = meta.copy()
            out_meta.update(
                {"count": len(text_prompts) + 1, "dtype": "uint8", "nodata": 0}
            )

            # Create arrays for results
            all_masks = np.zeros((len(text_prompts), height, width), dtype=np.uint8)
            all_boxes = []
            all_polygons = []

            # Calculate effective tile size (accounting for overlap)
            effective_tile_size = self.tile_size - 2 * self.overlap

            # Calculate number of tiles
            n_tiles_x = max(1, int(np.ceil(width / effective_tile_size)))
            n_tiles_y = max(1, int(np.ceil(height / effective_tile_size)))
            total_tiles = n_tiles_x * n_tiles_y

            print(f"Processing {total_tiles} tiles ({n_tiles_x}x{n_tiles_y})")

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
                            # Convert to RGB format for processing
                            if tile_data.shape[0] >= 3:
                                # Use first three bands for RGB representation
                                rgb_tile = tile_data[:3].transpose(1, 2, 0)
                            elif tile_data.shape[0] == 1:
                                # Create RGB from grayscale
                                rgb_tile = np.repeat(
                                    tile_data[0][:, :, np.newaxis], 3, axis=2
                                )
                            else:
                                print(
                                    f"Unsupported number of bands: {tile_data.shape[0]}"
                                )
                                continue

                            # Normalize to 0-255 range if needed
                            if rgb_tile.max() > 0:
                                rgb_tile = (
                                    (rgb_tile - rgb_tile.min())
                                    / (rgb_tile.max() - rgb_tile.min())
                                    * 255
                                ).astype(np.uint8)

                            # Convert to PIL Image
                            pil_image = Image.fromarray(rgb_tile)

                            # Detect objects
                            detections = self._detect(pil_image, text_prompts)

                            if detections:
                                # Apply Non-Maximum Suppression to reduce overlapping detections
                                detections = self._apply_nms(detections, nms_threshold)

                                if detections:
                                    # Segment objects
                                    detections = self._segment(
                                        pil_image, detections, polygon_refinement
                                    )

                                    # Process results
                                    for i, prompt in enumerate(text_prompts):
                                        prompt_polygons = []
                                        prompt_mask = np.zeros(
                                            (tile_height, tile_width), dtype=np.uint8
                                        )

                                        for detection in detections:
                                            if (
                                                detection.label.replace(".", "")
                                                .strip()
                                                .lower()
                                                == prompt.lower()
                                            ):
                                                if detection.mask is not None:
                                                    # Apply gaussian blur to reduce blockiness
                                                    try:
                                                        from scipy.ndimage import (
                                                            gaussian_filter,
                                                        )

                                                        smoothed_mask = gaussian_filter(
                                                            detection.mask.astype(
                                                                float
                                                            ),
                                                            sigma=smoothing_sigma,
                                                        )
                                                        detection.mask = (
                                                            smoothed_mask > 0.5
                                                        ).astype(np.uint8)
                                                    except ImportError:
                                                        pass

                                                    # Add to combined mask for this prompt
                                                    prompt_mask = np.maximum(
                                                        prompt_mask, detection.mask
                                                    )

                                                # Store bounding box with geospatial coordinates
                                                if export_boxes:
                                                    bbox = detection.box
                                                    x_geo_min, y_geo_min = transform * (
                                                        x_start + bbox.xmin,
                                                        y_start + bbox.ymin,
                                                    )
                                                    x_geo_max, y_geo_max = transform * (
                                                        x_start + bbox.xmax,
                                                        y_start + bbox.ymax,
                                                    )

                                                    geo_box = {
                                                        "label": detection.label,
                                                        "score": detection.score,
                                                        "prompt": prompt,
                                                        "geometry": box(
                                                            x_geo_min,
                                                            y_geo_max,
                                                            x_geo_max,
                                                            y_geo_min,
                                                        ),
                                                    }
                                                    all_boxes.append(geo_box)

                                        # Convert masks to individual polygons
                                        if export_polygons and np.any(prompt_mask):
                                            tile_polygons = self._mask_to_polygons(
                                                prompt_mask,
                                                transform,
                                                x_start,
                                                y_start,
                                                min_polygon_area,
                                                simplify_tolerance,
                                            )

                                            # Add metadata to polygons
                                            for poly_data in tile_polygons:
                                                poly_data.update(
                                                    {
                                                        "label": prompt,
                                                        "score": max(
                                                            [
                                                                d.score
                                                                for d in detections
                                                                if d.label.replace(
                                                                    ".", ""
                                                                )
                                                                .strip()
                                                                .lower()
                                                                == prompt.lower()
                                                            ],
                                                            default=0.0,
                                                        ),
                                                        "tile_x": x,
                                                        "tile_y": y,
                                                    }
                                                )
                                                all_polygons.append(poly_data)

                                        # Store mask in the global array
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

                                        mask_slice = prompt_mask[
                                            valid_y_start:valid_y_end,
                                            valid_x_start:valid_x_end,
                                        ]
                                        all_masks[
                                            i,
                                            dest_y_start:dest_y_end,
                                            dest_x_start:dest_x_end,
                                        ] = np.maximum(
                                            all_masks[
                                                i,
                                                dest_y_start:dest_y_end,
                                                dest_x_start:dest_x_end,
                                            ],
                                            mask_slice,
                                        )

                        except Exception as e:
                            print(f"Error processing tile at ({x}, {y}): {str(e)}")
                            continue

                        # Update progress bar
                        pbar.update(1)

            # Create combined mask (union of all individual masks)
            combined_mask = np.any(all_masks, axis=0).astype(np.uint8)

            # Write the output GeoTIFF
            with rasterio.open(output_path, "w", **out_meta) as dst:
                # Write combined mask as first band
                dst.write(combined_mask, 1)

                # Write individual masks for each prompt
                for i, mask in enumerate(all_masks):
                    dst.write(mask, i + 2)

                # Add descriptions to bands
                dst.set_band_description(1, "Combined Segmentation")
                for i, prompt in enumerate(text_prompts):
                    dst.set_band_description(i + 2, f"Segmentation: {prompt}")

            result_files = {"segmentation": output_path}

            # Export bounding boxes if requested
            if export_boxes and all_boxes:
                boxes_path = output_path.replace(".tif", "_boxes.geojson")
                gdf = gpd.GeoDataFrame(all_boxes, crs=crs)
                gdf.to_file(boxes_path, driver="GeoJSON")
                result_files["boxes"] = boxes_path
                print(f"Exported {len(all_boxes)} bounding boxes to {boxes_path}")

            # Export instance polygons if requested
            if export_polygons and all_polygons:
                polygons_path = output_path.replace(".tif", "_polygons.geojson")
                gdf = gpd.GeoDataFrame(all_polygons, crs=crs)
                gdf.to_file(polygons_path, driver="GeoJSON")
                result_files["polygons"] = polygons_path
                print(
                    f"Exported {len(all_polygons)} instance polygons to {polygons_path}"
                )

            print(f"Segmentation saved to {output_path}")
            print(
                f"Found {len(all_polygons)} individual building instances"
                if export_polygons
                else ""
            )

            return result_files


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
                                # Keep aspect ratio - use LANCZOS resampling instead of deprecated constant
                                pil_image.thumbnail(
                                    (self.tile_size, self.tile_size),
                                    Image.Resampling.LANCZOS,
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
                                        (tile_width, tile_height),
                                        Image.Resampling.BICUBIC,
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
