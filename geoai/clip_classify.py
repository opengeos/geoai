"""Zero-shot CLIP classification for geospatial vector features.

This module provides image-level zero-shot classification using OpenAI's CLIP
model. Given a GeoDataFrame of polygon features and a raster image, it extracts
the bounding-box image chip for each polygon, runs it through CLIP, and assigns
the best-matching category label from a user-provided list of candidate labels.

Key Features:
- Zero-shot classification: no training data required, just candidate labels
- Automatic CRS alignment between vector and raster inputs
- Batch processing with GPU acceleration
- Pre-computed text embeddings for efficient large-scale inference
- Handles multi-band, single-band, and float rasters automatically

Reference: https://github.com/opengeos/geoai/issues/129
"""

import logging
import os
from typing import Any, List, Optional, Union

import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import from_bounds as window_from_bounds
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "CLIPVectorClassifier",
    "clip_classify_vector",
]


def _to_rgb_uint8(tile_data: np.ndarray) -> Optional[np.ndarray]:
    """Convert rasterio tile data (bands, H, W) to RGB uint8 (H, W, 3).

    Args:
        tile_data: Array with shape (bands, H, W) as returned by
            ``rasterio.DatasetReader.read()``.

    Returns:
        Array with shape (H, W, 3) and dtype uint8, or None if the tile
        is entirely empty (all zeros).
    """
    if tile_data.size == 0:
        return None

    bands = tile_data.shape[0]
    if bands >= 3:
        rgb = tile_data[:3].transpose(1, 2, 0)
    elif bands == 1:
        rgb = np.repeat(tile_data[0][:, :, np.newaxis], 3, axis=2)
    else:
        # 2-band: use first band replicated
        rgb = np.repeat(tile_data[0][:, :, np.newaxis], 3, axis=2)

    rgb = rgb.astype(np.float64)
    vmin, vmax = rgb.min(), rgb.max()
    if vmax > vmin:
        rgb = (rgb - vmin) / (vmax - vmin) * 255.0
    elif vmax > 0:
        rgb = np.full_like(rgb, 128.0)
    else:
        return None

    return rgb.astype(np.uint8)


class CLIPVectorClassifier:
    """Zero-shot classification of vector polygon features using CLIP.

    Loads a CLIP model from Hugging Face and classifies image chips
    extracted from a raster at the bounding-box locations defined by
    polygon geometries.

    Args:
        model_name: Hugging Face model ID for the CLIP model.
            Default: ``"openai/clip-vit-base-patch32"``.
        device: Device for inference (``"cuda"``, ``"cpu"``).
            If None, auto-selects CUDA when available.

    Example:
        >>> classifier = CLIPVectorClassifier()
        >>> result = classifier.classify(
        ...     vector_data="polygons.geojson",
        ...     raster_path="image.tif",
        ...     labels=["forest", "water", "urban", "agriculture"],
        ... )
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"CLIP model loaded on {self.device}")

    def classify(
        self,
        vector_data: Union[str, gpd.GeoDataFrame],
        raster_path: str,
        labels: List[str],
        label_prefix: str = "a satellite image of ",
        top_k: int = 1,
        batch_size: int = 16,
        min_chip_size: int = 10,
        output_path: Optional[str] = None,
        quiet: bool = False,
    ) -> gpd.GeoDataFrame:
        """Classify each polygon feature using zero-shot CLIP inference.

        For every polygon in *vector_data*, extracts the bounding-box image
        chip from *raster_path*, encodes it with CLIP, and assigns the
        best-matching label from *labels* via cosine similarity.

        Args:
            vector_data: Path to a vector file or a GeoDataFrame.
            raster_path: Path to the raster image (GeoTIFF, etc.).
            labels: Candidate category labels for zero-shot classification.
            label_prefix: Text prefix prepended to each label for CLIP
                text encoding. Default: ``"a satellite image of "``.
            top_k: Number of top predictions per polygon. Default: 1.
            batch_size: Images per inference batch. Default: 16.
            min_chip_size: Minimum chip dimension in pixels. Polygons
                yielding smaller chips are skipped. Default: 10.
            output_path: Optional path to save the annotated GeoDataFrame
                (``.geojson``, ``.parquet``, ``.gpkg``).
            quiet: Suppress progress bar. Default: False.

        Returns:
            GeoDataFrame with added columns:

            - ``clip_label``: top-1 predicted label (str or None).
            - ``clip_confidence``: top-1 softmax score (float or NaN).
            - ``clip_top_k_labels``: list of top-k labels (only if
              *top_k* > 1).
            - ``clip_top_k_scores``: list of top-k scores (only if
              *top_k* > 1).

        Raises:
            ValueError: If *labels* is empty.
            FileNotFoundError: If *raster_path* does not exist.
            FileNotFoundError: If *vector_data* is a path that does not exist.
        """
        # --- Validate ---
        if not labels:
            raise ValueError("labels must be a non-empty list of category names.")

        if not os.path.isfile(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")

        top_k = min(top_k, len(labels))

        # --- Load vector data ---
        gdf = self._load_vector(vector_data)

        if len(gdf) == 0:
            gdf["clip_label"] = []
            gdf["clip_confidence"] = []
            if top_k > 1:
                gdf["clip_top_k_labels"] = []
                gdf["clip_top_k_scores"] = []
            return gdf

        # --- Open raster and align CRS ---
        with rasterio.open(raster_path) as src:
            if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
                logger.info(
                    "Reprojecting vector data from %s to %s",
                    gdf.crs,
                    src.crs,
                )
                gdf = gdf.to_crs(src.crs)

            # --- Pre-compute text embeddings ---
            text_embeds = self._encode_text(labels, label_prefix)

            # --- Extract chips and classify in batches ---
            result_labels = [None] * len(gdf)
            result_confs = [float("nan")] * len(gdf)
            result_top_k_labels = [None] * len(gdf) if top_k > 1 else None
            result_top_k_scores = [None] * len(gdf) if top_k > 1 else None

            batch_images: List[Image.Image] = []
            batch_indices: List[int] = []

            iterator = enumerate(gdf.geometry)
            if not quiet:
                iterator = tqdm(
                    list(iterator),
                    desc="Classifying features",
                    unit="feature",
                )

            for idx, geom in iterator:
                chip = self._extract_chip(src, geom, min_chip_size)
                if chip is None:
                    continue

                batch_images.append(chip)
                batch_indices.append(idx)

                if len(batch_images) >= batch_size:
                    self._process_batch(
                        batch_images,
                        batch_indices,
                        text_embeds,
                        labels,
                        top_k,
                        result_labels,
                        result_confs,
                        result_top_k_labels,
                        result_top_k_scores,
                    )
                    batch_images.clear()
                    batch_indices.clear()

            # Flush remaining
            if batch_images:
                self._process_batch(
                    batch_images,
                    batch_indices,
                    text_embeds,
                    labels,
                    top_k,
                    result_labels,
                    result_confs,
                    result_top_k_labels,
                    result_top_k_scores,
                )

        # --- Assign columns ---
        gdf = gdf.copy()
        gdf["clip_label"] = result_labels
        gdf["clip_confidence"] = result_confs
        if top_k > 1:
            gdf["clip_top_k_labels"] = result_top_k_labels
            gdf["clip_top_k_scores"] = result_top_k_scores

        # --- Save ---
        if output_path is not None:
            ext = os.path.splitext(output_path)[1].lower()
            if ext == ".parquet":
                gdf.to_parquet(output_path)
            elif ext == ".gpkg":
                gdf.to_file(output_path, driver="GPKG")
            else:
                gdf.to_file(output_path, driver="GeoJSON")

        return gdf

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_vector(
        self, vector_data: Union[str, gpd.GeoDataFrame]
    ) -> gpd.GeoDataFrame:
        """Load and validate vector data."""
        if isinstance(vector_data, gpd.GeoDataFrame):
            return vector_data.copy()

        if not isinstance(vector_data, str):
            raise TypeError(
                f"vector_data must be a file path or GeoDataFrame, "
                f"got {type(vector_data).__name__}"
            )

        if not os.path.isfile(vector_data):
            raise FileNotFoundError(f"Vector file not found: {vector_data}")

        ext = os.path.splitext(vector_data)[1].lower()
        if ext == ".parquet":
            return gpd.read_parquet(vector_data)
        return gpd.read_file(vector_data)

    def _extract_chip(
        self,
        src: rasterio.DatasetReader,
        geom: Any,
        min_chip_size: int,
    ) -> Optional[Image.Image]:
        """Extract an RGB PIL image chip for a geometry's bounding box."""
        if geom is None or geom.is_empty:
            return None

        minx, miny, maxx, maxy = geom.bounds
        rb = src.bounds

        # Check overlap
        minx = max(minx, rb.left)
        miny = max(miny, rb.bottom)
        maxx = min(maxx, rb.right)
        maxy = min(maxy, rb.top)

        if minx >= maxx or miny >= maxy:
            return None

        try:
            window = window_from_bounds(minx, miny, maxx, maxy, src.transform)
        except Exception:
            return None

        # Round to integer pixels
        col_off = max(0, int(round(window.col_off)))
        row_off = max(0, int(round(window.row_off)))
        width = max(1, int(round(window.width)))
        height = max(1, int(round(window.height)))

        # Clamp to raster dimensions
        width = min(width, src.width - col_off)
        height = min(height, src.height - row_off)

        if width < 1 or height < 1:
            return None

        if width < min_chip_size and height < min_chip_size:
            return None

        from rasterio.windows import Window

        pixel_window = Window(col_off, row_off, width, height)
        tile_data = src.read(window=pixel_window)

        rgb = _to_rgb_uint8(tile_data)
        if rgb is None:
            return None

        return Image.fromarray(rgb)

    @torch.no_grad()
    def _encode_text(self, labels: List[str], label_prefix: str) -> torch.Tensor:
        """Encode label texts and return normalized embeddings."""
        texts = [label_prefix + label for label in labels]
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_embeds = self.model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    @torch.no_grad()
    def _process_batch(
        self,
        images: List[Image.Image],
        indices: List[int],
        text_embeds: torch.Tensor,
        labels: List[str],
        top_k: int,
        result_labels: list,
        result_confs: list,
        result_top_k_labels: Optional[list],
        result_top_k_scores: Optional[list],
    ) -> None:
        """Classify a batch of images and write results into output lists."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = (image_embeds @ text_embeds.T) * logit_scale
        probs = logits.softmax(dim=-1).cpu().numpy()

        for i, idx in enumerate(indices):
            sorted_ids = np.argsort(probs[i])[::-1]
            result_labels[idx] = labels[sorted_ids[0]]
            result_confs[idx] = float(probs[i][sorted_ids[0]])

            if top_k > 1 and result_top_k_labels is not None:
                top_ids = sorted_ids[:top_k]
                result_top_k_labels[idx] = [labels[j] for j in top_ids]
                result_top_k_scores[idx] = [float(probs[i][j]) for j in top_ids]


def clip_classify_vector(
    vector_data: Union[str, gpd.GeoDataFrame],
    raster_path: str,
    labels: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
    label_prefix: str = "a satellite image of ",
    top_k: int = 1,
    batch_size: int = 16,
    min_chip_size: int = 10,
    output_path: Optional[str] = None,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Classify vector polygon features using zero-shot CLIP inference.

    Convenience function that creates a :class:`CLIPVectorClassifier` and
    runs classification in a single call.

    For each polygon in *vector_data*, the bounding-box image chip is
    extracted from *raster_path*, encoded with CLIP, and matched against
    *labels* via cosine similarity to assign the best category.

    Args:
        vector_data: Path to a vector file or a GeoDataFrame.
        raster_path: Path to the raster image (GeoTIFF, etc.).
        labels: Candidate category labels for zero-shot classification.
        model_name: Hugging Face model ID for the CLIP model.
            Default: ``"openai/clip-vit-base-patch32"``.
        device: Device for inference. If None, auto-selects.
        label_prefix: Text prefix prepended to each label.
            Default: ``"a satellite image of "``.
        top_k: Number of top predictions per polygon. Default: 1.
        batch_size: Images per inference batch. Default: 16.
        min_chip_size: Minimum chip dimension in pixels. Default: 10.
        output_path: Optional path to save the result.
        quiet: Suppress progress bar. Default: False.

    Returns:
        GeoDataFrame with added ``clip_label`` and ``clip_confidence``
        columns. If *top_k* > 1, also includes ``clip_top_k_labels``
        and ``clip_top_k_scores``.

    Raises:
        ValueError: If *labels* is empty.
        FileNotFoundError: If *raster_path* does not exist.

    Example:
        >>> from geoai import clip_classify_vector
        >>> result = clip_classify_vector(
        ...     "buildings.geojson",
        ...     "naip_image.tif",
        ...     labels=["residential", "commercial", "industrial"],
        ... )
        >>> print(result[["clip_label", "clip_confidence"]].head())
    """
    classifier = CLIPVectorClassifier(model_name=model_name, device=device)
    return classifier.classify(
        vector_data=vector_data,
        raster_path=raster_path,
        labels=labels,
        label_prefix=label_prefix,
        top_k=top_k,
        batch_size=batch_size,
        min_chip_size=min_chip_size,
        output_path=output_path,
        quiet=quiet,
    )
