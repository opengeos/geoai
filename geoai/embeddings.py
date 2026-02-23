"""Module for working with geospatial embedding datasets from TorchGeo.

This module provides a unified, easy-to-use interface for loading, exploring,
visualizing, and analyzing pre-computed Earth embedding datasets introduced
in TorchGeo v0.9.0. These embeddings are pre-computed representations from
geospatial foundation models that encode satellite imagery into compact
vector representations.

Two types of embedding datasets are supported:

- **Patch-based (NonGeoDataset)**: Each sample is a single embedding vector
  for a geographic patch (e.g., ClayEmbeddings, MajorTOMEmbeddings).
  Data is stored in GeoParquet files.

- **Pixel-based (RasterDataset)**: Each pixel contains an embedding vector,
  stored as multi-band GeoTIFF rasters (e.g., GoogleSatelliteEmbedding,
  TesseraEmbeddings).

Example usage:

    >>> import geoai
    >>> # List available embedding datasets
    >>> geoai.list_embedding_datasets()
    >>> # Load a patch-based dataset
    >>> ds = geoai.load_embedding_dataset("clay", root="path/to/data.parquet")
    >>> # Load a pixel-based dataset
    >>> ds = geoai.load_embedding_dataset("google_satellite", paths="path/to/data")
"""

import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "list_embedding_datasets",
    "load_embedding_dataset",
    "get_embedding_info",
    "extract_patch_embeddings",
    "extract_pixel_embeddings",
    "visualize_embeddings",
    "plot_embedding_vector",
    "plot_embedding_raster",
    "cluster_embeddings",
    "embedding_similarity",
    "train_embedding_classifier",
    "compare_embeddings",
    "embedding_to_geotiff",
    "EMBEDDING_DATASETS",
]

# ---------------------------------------------------------------------------
# Embedding dataset registry
# ---------------------------------------------------------------------------

EMBEDDING_DATASETS: Dict[str, Dict[str, Any]] = {
    "clay": {
        "class_name": "ClayEmbeddings",
        "kind": "patch",
        "base": "NonGeoDataset",
        "spatial_extent": "Global*",
        "spatial_resolution": "5.12 km",
        "temporal_extent": "2018-2023*",
        "dimensions": 768,
        "dtype": "float32",
        "license": "ODC-By-1.0",
        "description": (
            "Clay v0 and v1.5 embeddings from the Clay Foundation Model. "
            "Stored as GeoParquet files."
        ),
        "paper": "https://clay-foundation.github.io/model/",
        "data_source": "https://source.coop/clay/clay-model-v0-embeddings",
    },
    "major_tom": {
        "class_name": "MajorTOMEmbeddings",
        "kind": "patch",
        "base": "NonGeoDataset",
        "spatial_extent": "Global",
        "spatial_resolution": "2.14-3.56 km",
        "temporal_extent": "2015-2024*",
        "dimensions": 2048,
        "dtype": "float32",
        "license": "CC-BY-SA-4.0",
        "description": (
            "Major TOM (Terrestrial Observation Metaset) embeddings created "
            "from Major TOM Core using foundation models."
        ),
        "paper": "https://arxiv.org/abs/2412.05600",
        "data_source": "https://huggingface.co/Major-TOM",
    },
    "earth_index": {
        "class_name": "EarthIndexEmbeddings",
        "kind": "patch",
        "base": "NonGeoDataset",
        "spatial_extent": "Global",
        "spatial_resolution": "320 m",
        "temporal_extent": "2024",
        "dimensions": 384,
        "dtype": "float32",
        "license": "CC-BY-4.0",
        "description": (
            "Earth Index v2 embeddings from Sentinel-2 mosaics using the "
            "SoftCon model. Stored as GeoParquet."
        ),
        "paper": "https://source.coop/earthgenome/earthindexembeddings",
        "data_source": "https://source.coop/earthgenome/earthindexembeddings",
    },
    "earth_embeddings": {
        "class_name": "EarthEmbeddings",
        "kind": "patch",
        "base": "NonGeoDataset",
        "spatial_extent": "Global*",
        "spatial_resolution": "2.24-3.84 km",
        "temporal_extent": "2015-2024*",
        "dimensions": "256-1152",
        "dtype": "float16, float32",
        "license": "CC-BY-SA-4.0",
        "description": (
            "Pre-computed embeddings of MajorTOM-Core-S2L2A imagery using "
            "SatCLIP, FarSLIP, DINOv2, and SigLIP models."
        ),
        "paper": "https://huggingface.co/datasets/ML4Sustain/EarthEmbeddings",
        "data_source": "https://huggingface.co/datasets/ML4Sustain/EarthEmbeddings",
    },
    "copernicus_embed": {
        "class_name": "CopernicusEmbed",
        "kind": "pixel",
        "base": "RasterDataset",
        "spatial_extent": "Global",
        "spatial_resolution": "0.25 deg",
        "temporal_extent": "2021",
        "dimensions": 768,
        "dtype": "float32",
        "license": "CC-BY-4.0",
        "description": (
            "Copernicus-Embed aggregates all available Copernicus satellite "
            "modalities into a 0.25 deg grid of 768-D embedding vectors."
        ),
        "paper": "https://arxiv.org/abs/2503.11849",
        "data_source": "https://github.com/zhu-xlab/Copernicus-FM",
    },
    "presto": {
        "class_name": "PrestoEmbeddings",
        "kind": "pixel",
        "base": "RasterDataset",
        "spatial_extent": "Togo",
        "spatial_resolution": "10 m",
        "temporal_extent": "2019-2020",
        "dimensions": 128,
        "dtype": "uint16",
        "license": "CC-BY-4.0",
        "description": (
            "Presto geospatial embeddings for Togo at 10m resolution, compressing "
            "a year of Sentinel-1, Sentinel-2, ERA5, and SRTM data into 128-D vectors."
        ),
        "paper": "https://arxiv.org/abs/2511.02923",
        "data_source": "https://huggingface.co/datasets/izvonkov/Togo_Presto_Embeddings",
    },
    "tessera": {
        "class_name": "TesseraEmbeddings",
        "kind": "pixel",
        "base": "RasterDataset",
        "spatial_extent": "Global",
        "spatial_resolution": "10 m",
        "temporal_extent": "2017-2025*",
        "dimensions": 128,
        "dtype": "int8 -> float32",
        "license": "CC0-1.0",
        "description": (
            "Tessera foundation model embeddings from Sentinel-1 and Sentinel-2 "
            "imagery. 128-channel representations at 10m resolution."
        ),
        "paper": "https://arxiv.org/abs/2506.20380",
        "data_source": "https://github.com/ucam-eo/geotessera",
    },
    "google_satellite": {
        "class_name": "GoogleSatelliteEmbedding",
        "kind": "pixel",
        "base": "RasterDataset",
        "spatial_extent": "Global",
        "spatial_resolution": "10 m",
        "temporal_extent": "2017-2025",
        "dimensions": 64,
        "dtype": "int8 -> float64",
        "license": "CC-BY-4.0",
        "description": (
            "Google / AlphaEarth Foundations satellite embeddings. "
            "Unit-length 64-D embeddings from optical, radar, LiDAR, and more."
        ),
        "paper": "https://arxiv.org/abs/2507.22291",
        "data_source": (
            "https://developers.google.com/earth-engine/datasets/catalog/"
            "GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL"
        ),
    },
    "embedded_seamless": {
        "class_name": "EmbeddedSeamlessData",
        "kind": "pixel",
        "base": "RasterDataset",
        "spatial_extent": "Global",
        "spatial_resolution": "30 m",
        "temporal_extent": "2000-2024",
        "dimensions": 12,
        "dtype": "uint16 -> float32",
        "license": "CC-BY-4.0",
        "description": (
            "Embedded Seamless Data (ESD) compresses Landsat, MODIS, and NASADEM "
            "observations into 12-D embeddings at 30m resolution."
        ),
        "paper": "https://arxiv.org/abs/2601.11183",
        "data_source": "https://data-starcloud.pcl.ac.cn/iearthdata/64",
    },
}


def list_embedding_datasets(
    kind: Optional[str] = None,
    as_dataframe: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """List all available embedding datasets from TorchGeo v0.9.0.

    Args:
        kind: Filter by dataset kind. One of ``"patch"`` (NonGeoDataset) or
            ``"pixel"`` (RasterDataset). If None, list all datasets.
        as_dataframe: If True, return a pandas DataFrame. Otherwise return
            a dictionary.
        verbose: If True, print a summary table to the console.

    Returns:
        A DataFrame or dictionary describing the available datasets.

    Example:
        >>> import geoai
        >>> df = geoai.list_embedding_datasets()
        >>> df = geoai.list_embedding_datasets(kind="patch")
    """
    datasets = EMBEDDING_DATASETS
    if kind is not None:
        kind = kind.lower()
        if kind not in ("patch", "pixel"):
            raise ValueError(f"kind must be 'patch' or 'pixel', got '{kind}'")
        datasets = {k: v for k, v in datasets.items() if v["kind"] == kind}

    if not as_dataframe:
        return datasets

    rows = []
    for name, info in datasets.items():
        rows.append(
            {
                "name": name,
                "class": info["class_name"],
                "kind": info["kind"],
                "spatial_extent": info["spatial_extent"],
                "resolution": info["spatial_resolution"],
                "temporal_extent": info["temporal_extent"],
                "dimensions": info["dimensions"],
                "dtype": info["dtype"],
                "license": info["license"],
            }
        )
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
    return df


def _get_dataset_class(name: str) -> Any:
    """Import and return the torchgeo dataset class by registry name.

    Args:
        name: Registry name (key from EMBEDDING_DATASETS).

    Returns:
        The torchgeo dataset class.

    Raises:
        ValueError: If name is not found in the registry.
        ImportError: If torchgeo >= 0.9.0 is not installed.
    """
    if name not in EMBEDDING_DATASETS:
        available = ", ".join(sorted(EMBEDDING_DATASETS.keys()))
        raise ValueError(f"Unknown embedding dataset '{name}'. Available: {available}")

    class_name = EMBEDDING_DATASETS[name]["class_name"]

    try:
        import torchgeo.datasets as tgd

        cls = getattr(tgd, class_name, None)
        if cls is None:
            raise ImportError(
                f"torchgeo.datasets.{class_name} not found. "
                "Please upgrade torchgeo to >= 0.9.0: pip install -U torchgeo"
            )
        return cls
    except ImportError:
        raise ImportError(
            "torchgeo >= 0.9.0 is required for embedding datasets. "
            "Install with: pip install -U torchgeo"
        )


def load_embedding_dataset(
    name: str,
    root: Optional[str] = None,
    paths: Optional[Union[str, List[str]]] = None,
    transforms: Optional[Callable] = None,
    **kwargs: Any,
) -> Any:
    """Load an embedding dataset by name.

    This is a unified factory function that instantiates the correct
    torchgeo embedding dataset class based on the name.

    For **patch-based** datasets (``kind="patch"``), pass the ``root``
    parameter pointing to a GeoParquet file. For **pixel-based** datasets
    (``kind="pixel"``), pass the ``paths`` parameter pointing to a
    directory of GeoTIFF files.

    Args:
        name: Registry name of the dataset (e.g., ``"clay"``,
            ``"google_satellite"``). Use :func:`list_embedding_datasets` to
            see all available names.
        root: Root directory or file path (used by patch-based datasets).
        paths: One or more directories containing GeoTIFF files (used by
            pixel-based datasets).
        transforms: Optional transform function applied to each sample.
        **kwargs: Additional keyword arguments passed to the dataset
            constructor (e.g., ``crs``, ``res``, ``bands``, ``cache``,
            ``download``, ``time_series``).

    Returns:
        A torchgeo dataset instance.

    Raises:
        ValueError: If required arguments are missing or name is unknown.

    Example:
        >>> import geoai
        >>> # Patch-based dataset
        >>> ds = geoai.load_embedding_dataset(
        ...     "clay", root="path/to/clay_embeddings.parquet"
        ... )
        >>> sample = ds[0]
        >>> print(sample["embedding"].shape)
        >>> # Pixel-based dataset
        >>> ds = geoai.load_embedding_dataset(
        ...     "google_satellite", paths="path/to/geotiffs/"
        ... )
    """
    if name not in EMBEDDING_DATASETS:
        available = ", ".join(sorted(EMBEDDING_DATASETS.keys()))
        raise ValueError(f"Unknown embedding dataset '{name}'. Available: {available}")

    info = EMBEDDING_DATASETS[name]
    cls = _get_dataset_class(name)
    kind = info["kind"]

    if kind == "patch":
        if root is None:
            raise ValueError(
                f"Patch-based dataset '{name}' requires the 'root' parameter "
                "pointing to a GeoParquet file."
            )
        return cls(root=root, transforms=transforms, **kwargs)
    else:
        if paths is None and root is not None:
            paths = root
        if paths is None:
            raise ValueError(
                f"Pixel-based dataset '{name}' requires the 'paths' parameter "
                "pointing to a directory of GeoTIFF files."
            )
        return cls(paths=paths, transforms=transforms, **kwargs)


def get_embedding_info(name: str) -> Dict[str, Any]:
    """Get detailed information about an embedding dataset.

    Args:
        name: Registry name of the dataset.

    Returns:
        Dictionary with dataset metadata.

    Raises:
        ValueError: If name is not found in the registry.

    Example:
        >>> import geoai
        >>> info = geoai.get_embedding_info("google_satellite")
        >>> print(info["description"])
    """
    if name not in EMBEDDING_DATASETS:
        available = ", ".join(sorted(EMBEDDING_DATASETS.keys()))
        raise ValueError(f"Unknown embedding dataset '{name}'. Available: {available}")
    return EMBEDDING_DATASETS[name].copy()


# ---------------------------------------------------------------------------
# Embedding extraction utilities
# ---------------------------------------------------------------------------


def extract_patch_embeddings(
    dataset: Any,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Extract embeddings, coordinates, and timestamps from a patch-based dataset.

    Iterates over a patch-based (NonGeoDataset) embedding dataset and
    collects all embedding vectors along with their spatial and temporal
    metadata into NumPy arrays.

    Args:
        dataset: A patch-based embedding dataset (e.g., ClayEmbeddings).
        max_samples: Maximum number of samples to extract. If None, extract
            all samples.
        device: Device for tensor operations (unused, kept for API symmetry).

    Returns:
        Dictionary with keys:

        - ``"embeddings"``: ndarray of shape ``(N, D)``
        - ``"x"``: ndarray of shape ``(N,)`` with longitudes
        - ``"y"``: ndarray of shape ``(N,)`` with latitudes
        - ``"t"``: ndarray of shape ``(N,)`` with timestamps (if available)

    Example:
        >>> import geoai
        >>> ds = geoai.load_embedding_dataset("clay", root="data.parquet")
        >>> data = geoai.extract_patch_embeddings(ds, max_samples=1000)
        >>> print(data["embeddings"].shape)
    """
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    embeddings = []
    xs = []
    ys = []
    ts = []

    for i in range(n):
        sample = dataset[i]
        embeddings.append(sample["embedding"].numpy())
        if "x" in sample:
            xs.append(sample["x"].item())
        if "y" in sample:
            ys.append(sample["y"].item())
        if "t" in sample:
            ts.append(sample["t"].item())

    result = {"embeddings": np.stack(embeddings)}
    if xs:
        result["x"] = np.array(xs)
    if ys:
        result["y"] = np.array(ys)
    if ts:
        result["t"] = np.array(ts)

    return result


def extract_pixel_embeddings(
    dataset: Any,
    sampler: Any = None,
    num_samples: int = 100,
    size: float = 256,
    flatten: bool = True,
) -> Dict[str, Any]:
    """Extract embeddings from a pixel-based (RasterDataset) embedding dataset.

    Uses a TorchGeo sampler to draw spatial patches and returns the
    embedding tensors. If ``flatten=True``, pixels are reshaped to
    ``(N_total_pixels, D)``.

    Args:
        dataset: A pixel-based embedding dataset (e.g., GoogleSatelliteEmbedding).
        sampler: A torchgeo sampler instance. If None, a RandomGeoSampler is
            created with the given ``num_samples`` and ``size``.
        num_samples: Number of random samples to draw (used only when
            ``sampler`` is None).
        size: Patch size in dataset CRS units (used only when ``sampler``
            is None).
        flatten: If True, flatten spatial dimensions so that the result
            has shape ``(N, D)`` where N is the total number of pixels.

    Returns:
        Dictionary with keys:

        - ``"embeddings"``: ndarray of shape ``(N, D)`` if flattened, or
          list of arrays of shape ``(C, H, W)``
        - ``"bounds"``: list of bounding box tensors for each sample

    Example:
        >>> import geoai
        >>> ds = geoai.load_embedding_dataset(
        ...     "google_satellite", paths="data/"
        ... )
        >>> data = geoai.extract_pixel_embeddings(ds, num_samples=50)
        >>> print(data["embeddings"].shape)
    """
    from torchgeo.samplers import RandomGeoSampler

    if sampler is None:
        sampler = RandomGeoSampler(dataset, size=size, length=num_samples)

    all_embeddings = []
    all_bounds = []

    for query in sampler:
        sample = dataset[query]
        image = sample["image"]  # (C, H, W)
        all_bounds.append(sample.get("bounds"))

        if flatten:
            # (C, H, W) -> (H*W, C)
            c, h, w = image.shape
            pixels = image.permute(1, 2, 0).reshape(-1, c)
            all_embeddings.append(pixels.numpy())
        else:
            all_embeddings.append(image.numpy())

    result: Dict[str, Any] = {"bounds": all_bounds}
    if flatten:
        result["embeddings"] = np.concatenate(all_embeddings, axis=0)
    else:
        result["embeddings"] = all_embeddings

    return result


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    method: str = "pca",
    n_components: int = 2,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "tab10",
    alpha: float = 0.6,
    s: int = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs: Any,
) -> plt.Figure:
    """Visualize high-dimensional embeddings in 2D using dimensionality reduction.

    Supports PCA, t-SNE, and UMAP for projecting embedding vectors into a
    2D scatter plot.

    Args:
        embeddings: Array of shape ``(N, D)`` containing embedding vectors.
        labels: Optional integer labels of shape ``(N,)`` for coloring points.
        label_names: Optional list of label names for the legend.
        method: Dimensionality reduction method. One of ``"pca"``, ``"tsne"``,
            or ``"umap"``.
        n_components: Number of components for the projection (2 or 3).
        figsize: Figure size in inches.
        cmap: Matplotlib colormap name.
        alpha: Point transparency.
        s: Point size.
        title: Plot title. If None, an automatic title is generated.
        save_path: If provided, save the figure to this path.
        **kwargs: Additional keyword arguments passed to the reducer
            (e.g., ``perplexity`` for t-SNE, ``n_neighbors`` for UMAP).

    Returns:
        A matplotlib Figure.

    Raises:
        ValueError: If an unsupported method is specified.
        ImportError: If required libraries are not installed.

    Example:
        >>> import geoai
        >>> fig = geoai.visualize_embeddings(
        ...     embeddings, labels=labels, method="pca"
        ... )
    """
    method = method.lower()
    if method not in ("pca", "tsne", "umap"):
        raise ValueError(f"method must be 'pca', 'tsne', or 'umap', got '{method}'")

    # Dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, whiten=True, **kwargs)
        reduced = reducer.fit_transform(embeddings)
        explained = reducer.explained_variance_ratio_.sum()
        default_title = f"PCA of Embeddings (explained variance: {explained:.1%})"
    elif method == "tsne":
        from sklearn.manifold import TSNE

        tsne_kwargs = {"n_components": n_components, "random_state": 42}
        tsne_kwargs.update(kwargs)
        reducer = TSNE(**tsne_kwargs)
        reduced = reducer.fit_transform(embeddings)
        default_title = "t-SNE of Embeddings"
    else:  # umap
        try:
            import umap

            umap_kwargs = {"n_components": n_components, "random_state": 42}
            umap_kwargs.update(kwargs)
            reducer = umap.UMAP(**umap_kwargs)
            reduced = reducer.fit_transform(embeddings)
            default_title = "UMAP of Embeddings"
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP visualization. "
                "Install with: pip install umap-learn"
            )

    title = title or default_title

    fig, ax = plt.subplots(figsize=figsize)
    scatter_kwargs = dict(s=s, alpha=alpha)
    if labels is not None:
        scatter_kwargs["c"] = labels
        scatter_kwargs["cmap"] = cmap
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        **scatter_kwargs,
    )

    if labels is not None:
        handles, _ = scatter.legend_elements()
        if label_names is not None:
            ax.legend(handles, label_names, title="Classes", loc="best")
        else:
            ax.legend(*scatter.legend_elements(), title="Classes", loc="best")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_embedding_vector(
    embedding: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = "Embedding Vector",
    figsize: Tuple[int, int] = (10, 3),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a single embedding vector as a line chart.

    Args:
        embedding: 1D array or tensor of shape ``(D,)``.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.

    Returns:
        A matplotlib Figure.

    Example:
        >>> import geoai
        >>> ds = geoai.load_embedding_dataset("clay", root="data.parquet")
        >>> sample = ds[0]
        >>> fig = geoai.plot_embedding_vector(sample["embedding"])
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(embedding)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_embedding_raster(
    image: Union[np.ndarray, torch.Tensor],
    method: str = "pca",
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = "Embedding Visualization",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize a pixel-based embedding raster using PCA to create an RGB image.

    Projects high-dimensional embedding bands into 3 principal components
    for RGB visualization.

    Args:
        image: Embedding tensor of shape ``(C, H, W)`` or ``(H, W, C)``.
        method: Projection method. Currently only ``"pca"`` is supported.
        figsize: Figure size.
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        A matplotlib Figure.

    Example:
        >>> import geoai
        >>> ds = geoai.load_embedding_dataset(
        ...     "google_satellite", paths="data/"
        ... )
        >>> from torchgeo.samplers import RandomGeoSampler
        >>> sampler = RandomGeoSampler(ds, size=256, length=1)
        >>> sample = ds[next(iter(sampler))]
        >>> fig = geoai.plot_embedding_raster(sample["image"])
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # Ensure (C, H, W) format
    if image.ndim == 3 and image.shape[2] < image.shape[0]:
        # Likely (H, W, C) format
        image = np.transpose(image, (2, 0, 1))

    c, h, w = image.shape
    # Reshape to (H*W, C) for PCA
    pixels = image.reshape(c, -1).T  # (H*W, C)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(pixels)

    # Normalize to [0, 1]
    rgb -= rgb.min(axis=0, keepdims=True)
    maxvals = rgb.max(axis=0, keepdims=True)
    maxvals[maxvals == 0] = 1
    rgb /= maxvals
    rgb = rgb.reshape(h, w, 3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    method: str = "kmeans",
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Cluster embedding vectors using unsupervised methods.

    Args:
        embeddings: Array of shape ``(N, D)``.
        n_clusters: Number of clusters.
        method: Clustering method. One of ``"kmeans"``, ``"spectral"``,
            or ``"dbscan"``.
        random_state: Random seed for reproducibility.
        **kwargs: Additional keyword arguments passed to the clustering
            algorithm.

    Returns:
        Dictionary with keys:

        - ``"labels"``: ndarray of cluster assignments of shape ``(N,)``
        - ``"model"``: the fitted clustering model
        - ``"n_clusters"``: effective number of clusters found

    Raises:
        ValueError: If an unsupported method is specified.

    Example:
        >>> import geoai
        >>> result = geoai.cluster_embeddings(embeddings, n_clusters=5)
        >>> labels = result["labels"]
    """
    method = method.lower()

    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs
        )
        labels = model.fit_predict(embeddings)
    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        model = SpectralClustering(
            n_clusters=n_clusters, random_state=random_state, **kwargs
        )
        labels = model.fit_predict(embeddings)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN

        model = DBSCAN(**kwargs)
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        raise ValueError(
            f"method must be 'kmeans', 'spectral', or 'dbscan', got '{method}'"
        )

    return {
        "labels": labels,
        "model": model,
        "n_clusters": n_clusters if method == "dbscan" else n_clusters,
    }


def embedding_similarity(
    query: np.ndarray,
    embeddings: np.ndarray,
    metric: str = "cosine",
    top_k: int = 10,
) -> Dict[str, np.ndarray]:
    """Find the most similar embeddings to a query vector.

    Args:
        query: Query embedding of shape ``(D,)`` or ``(1, D)``.
        embeddings: Database of embeddings of shape ``(N, D)``.
        metric: Similarity metric. One of ``"cosine"`` or ``"euclidean"``.
        top_k: Number of most similar results to return.

    Returns:
        Dictionary with keys:

        - ``"indices"``: indices of the top-k most similar embeddings
        - ``"scores"``: similarity scores (higher is more similar for
          cosine, lower for euclidean)

    Example:
        >>> import geoai
        >>> results = geoai.embedding_similarity(
        ...     query=embeddings[0], embeddings=embeddings, top_k=5
        ... )
        >>> print(results["indices"])
    """
    query = np.asarray(query)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    metric = metric.lower()
    if metric == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity(query, embeddings).ravel()
        indices = np.argsort(-scores)[:top_k]
        return {"indices": indices, "scores": scores[indices]}
    elif metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(query, embeddings).ravel()
        indices = np.argsort(distances)[:top_k]
        return {"indices": indices, "scores": distances[indices]}
    else:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got '{metric}'")


def train_embedding_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    method: str = "knn",
    label_names: Optional[List[str]] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a lightweight classifier on pre-computed embeddings.

    This function trains a simple classifier (k-NN, Random Forest, or
    Logistic Regression) on embedding vectors, providing a quick baseline
    without requiring GPU fine-tuning.

    Args:
        train_embeddings: Training embeddings of shape ``(N_train, D)``.
        train_labels: Training labels of shape ``(N_train,)``.
        val_embeddings: Optional validation embeddings of shape ``(N_val, D)``.
        val_labels: Optional validation labels of shape ``(N_val,)``.
        method: Classifier type. One of ``"knn"``, ``"random_forest"``,
            or ``"logistic_regression"``.
        label_names: Optional list of human-readable class names.
        verbose: If True, print classification report for validation set.
        **kwargs: Additional keyword arguments passed to the classifier
            (e.g., ``n_neighbors`` for k-NN).

    Returns:
        Dictionary with keys:

        - ``"model"``: the fitted classifier
        - ``"train_accuracy"``: training accuracy
        - ``"val_accuracy"``: validation accuracy (if val data provided)
        - ``"val_predictions"``: predictions on validation set
        - ``"classification_report"``: string report (if val data provided)

    Example:
        >>> import geoai
        >>> result = geoai.train_embedding_classifier(
        ...     train_embeddings, train_labels,
        ...     val_embeddings, val_labels,
        ...     method="knn", n_neighbors=5,
        ... )
        >>> print(f"Validation accuracy: {result['val_accuracy']:.2%}")
    """
    method = method.lower()

    if method == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        n_neighbors = kwargs.pop("n_neighbors", 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    elif method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        n_estimators = kwargs.pop("n_estimators", 100)
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, **kwargs
        )
    elif method == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    else:
        raise ValueError(
            f"method must be 'knn', 'random_forest', or "
            f"'logistic_regression', got '{method}'"
        )

    model.fit(train_embeddings, train_labels)

    result: Dict[str, Any] = {
        "model": model,
        "train_accuracy": model.score(train_embeddings, train_labels),
    }

    if val_embeddings is not None and val_labels is not None:
        val_preds = model.predict(val_embeddings)
        result["val_predictions"] = val_preds
        result["val_accuracy"] = model.score(val_embeddings, val_labels)

        from sklearn.metrics import classification_report

        report = classification_report(
            val_labels,
            val_preds,
            target_names=label_names,
            digits=3,
            zero_division=0,
        )
        result["classification_report"] = report

        if verbose:
            print(report)

    return result


def compare_embeddings(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise similarity between two sets of embeddings.

    Useful for change detection between embeddings from different time
    periods or different sensors.

    Args:
        embeddings_a: First set of embeddings of shape ``(N, D)``.
        embeddings_b: Second set of embeddings of shape ``(N, D)``.
            Must have the same number of samples as ``embeddings_a``.
        metric: Similarity metric. One of ``"cosine"``, ``"dot"``, or
            ``"euclidean"``.

    Returns:
        Array of shape ``(N,)`` with element-wise similarity scores.

    Example:
        >>> import geoai
        >>> similarity = geoai.compare_embeddings(emb_2020, emb_2024)
    """
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError(
            f"Shape mismatch: {embeddings_a.shape} vs {embeddings_b.shape}"
        )

    metric = metric.lower()
    if metric == "cosine":
        # Cosine similarity per row
        dot = np.sum(embeddings_a * embeddings_b, axis=1)
        norm_a = np.linalg.norm(embeddings_a, axis=1)
        norm_b = np.linalg.norm(embeddings_b, axis=1)
        return dot / (norm_a * norm_b + 1e-8)
    elif metric == "dot":
        return np.sum(embeddings_a * embeddings_b, axis=1)
    elif metric == "euclidean":
        return np.linalg.norm(embeddings_a - embeddings_b, axis=1)
    else:
        raise ValueError(
            f"metric must be 'cosine', 'dot', or 'euclidean', got '{metric}'"
        )


def embedding_to_geotiff(
    embeddings: np.ndarray,
    bounds: Tuple[float, float, float, float],
    output_path: str,
    crs: str = "EPSG:4326",
) -> str:
    """Save embedding vectors as a multi-band GeoTIFF.

    Each embedding dimension becomes a separate band in the output raster.

    Args:
        embeddings: Array of shape ``(H, W, D)`` or ``(D, H, W)``.
        bounds: Geographic bounds as ``(west, south, east, north)``.
        output_path: Path to save the GeoTIFF file.
        crs: Coordinate reference system string.

    Returns:
        The output file path.

    Example:
        >>> import geoai
        >>> geoai.embedding_to_geotiff(
        ...     embeddings, bounds=(-122, 37, -121, 38),
        ...     output_path="embeddings.tif"
        ... )
    """
    import rasterio
    from rasterio.transform import from_bounds

    if embeddings.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {embeddings.shape}")

    # Ensure (D, H, W) format
    if embeddings.shape[0] > embeddings.shape[2]:
        # Likely (H, W, D)
        embeddings = np.transpose(embeddings, (2, 0, 1))

    bands, height, width = embeddings.shape
    west, south, east, north = bounds

    transform = from_bounds(west, south, east, north, width, height)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=embeddings.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(embeddings)

    logger.info(f"Saved {bands}-band GeoTIFF to {output_path}")
    return output_path
