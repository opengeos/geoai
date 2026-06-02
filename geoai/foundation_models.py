"""Remote sensing foundation model catalog and discovery utilities.

This module provides a curated registry of remote sensing foundation models
drawn from the Awesome Remote Sensing Foundation Models list
(https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
and supports optional model loading via TerraTorch
(https://github.com/IBM/terratorch).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "FOUNDATION_MODELS",
    "list_foundation_models",
    "get_foundation_model_info",
    "check_terratorch_available",
    "load_foundation_model",
]

# ---------------------------------------------------------------------------
# Valid vocabulary sets — used for input validation and filter dispatch
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = frozenset(
    {"vision", "vision-language", "generative", "vision-location", "agents"}
)

_VALID_MODALITIES = frozenset(
    {
        "optical",
        "sar",
        "multispectral",
        "hyperspectral",
        "multimodal",
        "timeseries",
        "lidar",
    }
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FOUNDATION_MODELS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Vision foundation models
    # ------------------------------------------------------------------
    "prithvi-eo-2.0-300m": {
        "name": "Prithvi-EO-2.0-300M",
        "abbreviation": "Prithvi-EO-2.0",
        "category": "vision",
        "modality": "multispectral",
        "tasks": ["segmentation", "regression", "classification"],
        "backbone": "ViT-L",
        "publication": "Arxiv2024",
        "year": 2024,
        "paper_url": "https://arxiv.org/abs/2412.02732",
        "code_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "huggingface_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "license": "Apache-2.0",
        "terratorch_supported": True,
        "description": (
            "Prithvi EO 2.0 300M is a versatile multi-temporal vision transformer "
            "foundation model for Earth observation, pre-trained on 4.2M "
            "Sentinel-2 L1C patches across six continents."
        ),
    },
    "prithvi-eo-2.0-600m": {
        "name": "Prithvi-EO-2.0-600M",
        "abbreviation": "Prithvi-EO-2.0",
        "category": "vision",
        "modality": "multispectral",
        "tasks": ["segmentation", "regression", "classification"],
        "backbone": "ViT-H",
        "publication": "Arxiv2024",
        "year": 2024,
        "paper_url": "https://arxiv.org/abs/2412.02732",
        "code_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "huggingface_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "license": "Apache-2.0",
        "terratorch_supported": True,
        "description": (
            "Prithvi EO 2.0 600M is the large variant of the Prithvi EO 2.0 "
            "multi-temporal foundation model, offering higher capacity for "
            "complex Earth observation tasks."
        ),
    },
    "clay-v1": {
        "name": "Clay Foundation Model v1",
        "abbreviation": "Clay",
        "category": "vision",
        "modality": "multispectral",
        "tasks": ["segmentation", "classification", "embedding"],
        "backbone": "ViT-L",
        "publication": "Arxiv2024",
        "year": 2024,
        "paper_url": None,
        "code_url": "https://github.com/Clay-foundation/model",
        "huggingface_id": "made-with-clay/Clay-v1",
        "license": "Apache-2.0",
        "terratorch_supported": True,
        "description": (
            "Clay v1 is an open-source foundation model trained on multi-sensor "
            "satellite imagery (Sentinel-1, Sentinel-2, Landsat) supporting "
            "embeddings and fine-tuning for diverse Earth observation tasks."
        ),
    },
    "dofa-large": {
        "name": "DOFA Large",
        "abbreviation": "DOFA",
        "category": "vision",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification", "detection"],
        "backbone": "ViT-L",
        "publication": "Arxiv2024",
        "year": 2024,
        "paper_url": "https://arxiv.org/abs/2403.15356",
        "code_url": "https://github.com/zhu-xlab/DOFA",
        "huggingface_id": "XShadow/DOFA",
        "license": "MIT",
        "terratorch_supported": True,
        "description": (
            "DOFA (Dynamic One-For-All) is a neural plasticity-inspired multimodal "
            "foundation model for Earth observation that adapts dynamically to "
            "arbitrary sensor wavelengths."
        ),
    },
    "satmae-base": {
        "name": "SatMAE Base",
        "abbreviation": "SatMAE",
        "category": "vision",
        "modality": "multispectral",
        "tasks": ["segmentation", "classification"],
        "backbone": "ViT-B",
        "publication": "NeurIPS2022",
        "year": 2022,
        "paper_url": (
            "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
            "01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html"
        ),
        "code_url": "https://github.com/sustainlab-group/SatMAE",
        "huggingface_id": None,
        "license": "MIT",
        "terratorch_supported": True,
        "description": (
            "SatMAE pre-trains transformers for temporal and multi-spectral "
            "satellite imagery via masked autoencoders, achieving strong "
            "downstream performance on land cover and change detection tasks."
        ),
    },
    "scale-mae-large": {
        "name": "Scale-MAE Large",
        "abbreviation": "Scale-MAE",
        "category": "vision",
        "modality": "optical",
        "tasks": ["segmentation", "classification", "detection"],
        "backbone": "ViT-L",
        "publication": "ICCV2023",
        "year": 2023,
        "paper_url": "https://doi.org/10.1109/iccv51070.2023.00378",
        "code_url": "https://github.com/bair-climate-initiative/scale-mae",
        "huggingface_id": None,
        "license": "Apache-2.0",
        "terratorch_supported": True,
        "description": (
            "Scale-MAE is a scale-aware masked autoencoder for multiscale "
            "geospatial representation learning that explicitly encodes "
            "ground sampling distance during pre-training."
        ),
    },
    "ringmo": {
        "name": "RingMo",
        "abbreviation": "RingMo",
        "category": "vision",
        "modality": "optical",
        "tasks": ["segmentation", "classification", "detection"],
        "backbone": "Swin-B",
        "publication": "TGRS2022",
        "year": 2022,
        "paper_url": "https://ieeexplore.ieee.org/abstract/document/9844015",
        "code_url": "https://github.com/comeony/RingMo",
        "huggingface_id": None,
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "RingMo is a remote sensing foundation model trained with masked "
            "image modeling on two million optical satellite images, supporting "
            "a wide range of interpretation tasks."
        ),
    },
    "rvsa": {
        "name": "RVSA (Remote Sensing Vision Foundation Model)",
        "abbreviation": "RVSA",
        "category": "vision",
        "modality": "optical",
        "tasks": ["segmentation", "classification", "detection"],
        "backbone": "ViT-B+RVSA",
        "publication": "TGRS2022",
        "year": 2022,
        "paper_url": "https://ieeexplore.ieee.org/abstract/document/9956816",
        "code_url": "https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA",
        "huggingface_id": None,
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "RVSA advances plain vision transformers toward a remote sensing "
            "foundation model by introducing rotated varied-size window "
            "self-attention to handle arbitrary orientations and scales."
        ),
    },
    "satlas-pretrain": {
        "name": "SatlasPretrain",
        "abbreviation": "SatLas",
        "category": "vision",
        "modality": "multispectral",
        "tasks": ["segmentation", "classification", "detection", "regression"],
        "backbone": "Swin-B",
        "publication": "ICCV2023",
        "year": 2023,
        "paper_url": "https://doi.org/10.1109/iccv51070.2023.01538",
        "code_url": "https://github.com/allenai/satlas",
        "huggingface_id": "allenai/satlas-pretrain",
        "license": "Apache-2.0",
        "terratorch_supported": True,
        "description": (
            "SatlasPretrain is a large-scale dataset and pre-trained model for "
            "remote sensing image understanding across Sentinel-2 and NAIP "
            "imagery, supporting diverse geospatial tasks."
        ),
    },
    "croma": {
        "name": "CROMA",
        "abbreviation": "CROMA",
        "category": "vision",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification"],
        "backbone": "ViT-B",
        "publication": "NeurIPS2023",
        "year": 2023,
        "paper_url": "https://arxiv.org/abs/2311.00566",
        "code_url": "https://github.com/antofuller/CROMA",
        "huggingface_id": None,
        "license": "MIT",
        "terratorch_supported": False,
        "description": (
            "CROMA learns remote sensing representations via contrastive "
            "radar-optical masked autoencoders, leveraging complementary "
            "Sentinel-1 SAR and Sentinel-2 optical imagery."
        ),
    },
    "ssl4eo-s12": {
        "name": "SSL4EO-S12",
        "abbreviation": "SSL4EO-S12",
        "category": "vision",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification"],
        "backbone": "ViT-S",
        "publication": "TGRS2023",
        "year": 2023,
        "paper_url": "https://arxiv.org/abs/2211.07044",
        "code_url": "https://github.com/zhu-xlab/SSL4EO-S12",
        "huggingface_id": "wangyi111/SSL4EO-S12",
        "license": "MIT",
        "terratorch_supported": True,
        "description": (
            "SSL4EO-S12 provides a large-scale multi-modal self-supervised "
            "pre-training benchmark for Earth observation using Sentinel-1 "
            "and Sentinel-2 imagery with multiple SSL objectives."
        ),
    },
    "spectral-gpt": {
        "name": "SpectralGPT",
        "abbreviation": "SpectralGPT",
        "category": "vision",
        "modality": "multispectral",
        "tasks": ["segmentation", "classification"],
        "backbone": "ViT-B",
        "publication": "TPAMI2024",
        "year": 2024,
        "paper_url": "https://doi.org/10.1109/tpami.2024.3362475",
        "code_url": "https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT",
        "huggingface_id": None,
        "license": "MIT",
        "terratorch_supported": False,
        "description": (
            "SpectralGPT is a spectral remote sensing foundation model trained "
            "with a 3D token approach to handle progressive spectral and spatial "
            "masking across multi-spectral and hyperspectral imagery."
        ),
    },
    "hypersigma": {
        "name": "HyperSIGMA",
        "abbreviation": "HyperSIGMA",
        "category": "vision",
        "modality": "hyperspectral",
        "tasks": ["segmentation", "classification"],
        "backbone": "ViT-L",
        "publication": "TPAMI2025",
        "year": 2025,
        "paper_url": "https://arxiv.org/abs/2406.11519",
        "code_url": "https://github.com/WHU-Sigma/HyperSIGMA",
        "huggingface_id": "whu-sigma/HyperSIGMA",
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "HyperSIGMA is a hyperspectral intelligence comprehension foundation "
            "model that incorporates spatial-spectral decoupled attention and "
            "auxiliary data from diverse sensors for comprehensive scene "
            "understanding."
        ),
    },
    "presto": {
        "name": "Presto",
        "abbreviation": "Presto",
        "category": "vision",
        "modality": "timeseries",
        "tasks": ["classification", "regression"],
        "backbone": "Transformer",
        "publication": "Arxiv2023",
        "year": 2023,
        "paper_url": "https://arxiv.org/abs/2304.14065",
        "code_url": "https://github.com/nasaharvest/presto",
        "huggingface_id": "nasaharvest/presto-geobench",
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "Presto is a lightweight pre-trained transformer for remote sensing "
            "time-series data, designed for agricultural monitoring and crop "
            "type mapping at global scale with minimal compute."
        ),
    },
    "panopticon": {
        "name": "Panopticon",
        "abbreviation": "Panopticon",
        "category": "vision",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification", "detection"],
        "backbone": "ViT-L",
        "publication": "CVPRW2025",
        "year": 2025,
        "paper_url": "https://arxiv.org/abs/2503.10845",
        "code_url": "https://github.com/Panopticon-FM/panopticon",
        "huggingface_id": "panopticon-fm/panopticon",
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "Panopticon (CVPRW 2025 EarthVision Best Paper) advances any-sensor "
            "foundation models for Earth observation, handling arbitrary sensor "
            "combinations without modality-specific pre-training."
        ),
    },
    "dynamicvis": {
        "name": "DynamicVis",
        "abbreviation": "DynamicVis",
        "category": "vision",
        "modality": "optical",
        "tasks": ["segmentation", "classification", "detection"],
        "backbone": "Mamba",
        "publication": "Arxiv2025",
        "year": 2025,
        "paper_url": "https://arxiv.org/abs/2503.16426",
        "code_url": "https://github.com/KyanChen/DynamicVis",
        "huggingface_id": None,
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "DynamicVis employs dynamic visual perception with selective state "
            "space models (Mamba) for efficient remote sensing foundation models, "
            "achieving strong performance with significantly reduced FLOPs."
        ),
    },
    "fomo": {
        "name": "FoMo",
        "abbreviation": "FoMo",
        "category": "vision",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification", "regression"],
        "backbone": "ViT-B",
        "publication": "AAAI2025",
        "year": 2025,
        "paper_url": "https://doi.org/10.1609/aaai.v39i27.35002",
        "code_url": "https://github.com/RolnickLab/FoMo-Bench",
        "huggingface_id": None,
        "license": "MIT",
        "terratorch_supported": False,
        "description": (
            "FoMo is a multi-modal, multi-scale, and multi-task remote sensing "
            "foundation model purpose-built for global forest monitoring, "
            "integrating SAR, optical, and LiDAR modalities."
        ),
    },
    "mmearth": {
        "name": "MMEarth",
        "abbreviation": "MMEarth",
        "category": "vision",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification", "regression"],
        "backbone": "ConvNeXt-B",
        "publication": "ECCV2024",
        "year": 2024,
        "paper_url": "https://doi.org/10.1007/978-3-031-73039-9_10",
        "code_url": "https://vishalned.github.io/mmearth/",
        "huggingface_id": "vishalned/MMEarth-train",
        "license": "MIT",
        "terratorch_supported": False,
        "description": (
            "MMEarth explores multi-modal pretext tasks for geospatial "
            "representation learning using over one million globally distributed "
            "Sentinel-1/2 paired samples with 12 auxiliary modalities."
        ),
    },
    # ------------------------------------------------------------------
    # Vision-language foundation models
    # ------------------------------------------------------------------
    "skysense": {
        "name": "SkySense",
        "abbreviation": "SkySense",
        "category": "vision-language",
        "modality": "multimodal",
        "tasks": ["segmentation", "classification", "captioning", "vqa"],
        "backbone": "ViT-L",
        "publication": "CVPR2024",
        "year": 2024,
        "paper_url": (
            "https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_"
            "A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_"
            "Interpretation_CVPR_2024_paper.html"
        ),
        "code_url": "https://github.com/Jack-bo1220/SkySense",
        "huggingface_id": None,
        "license": None,
        "terratorch_supported": False,
        "description": (
            "SkySense is a multi-modal remote sensing foundation model for "
            "universal interpretation of Earth observation imagery, combining "
            "visual and language understanding across diverse sensor types."
        ),
    },
    "georsam": {
        "name": "GeoSAM / RSPrompter",
        "abbreviation": "GeoRSAM",
        "category": "vision-language",
        "modality": "optical",
        "tasks": ["segmentation", "detection"],
        "backbone": "SAM-ViT-H",
        "publication": "TGRS2024",
        "year": 2024,
        "paper_url": "https://arxiv.org/abs/2306.16269",
        "code_url": "https://github.com/ViTAE-Transformer/RSPrompter",
        "huggingface_id": None,
        "license": "Apache-2.0",
        "terratorch_supported": False,
        "description": (
            "GeoRSAM adapts the Segment Anything Model (SAM) to remote sensing "
            "by introducing prompt-based learning strategies for automatic and "
            "interactive geospatial object segmentation."
        ),
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_foundation_models(
    category: Optional[str] = None,
    modality: Optional[str] = None,
    task: Optional[str] = None,
    terratorch_only: bool = False,
    huggingface_only: bool = False,
    as_dataframe: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """List available remote sensing foundation models.

    Args:
        category: Filter by model category. One of ``"vision"``,
            ``"vision-language"``, ``"generative"``, ``"vision-location"``,
            or ``"agents"``. If None, all categories are returned.
        modality: Filter by sensor modality. One of ``"optical"``,
            ``"sar"``, ``"multispectral"``, ``"hyperspectral"``,
            ``"multimodal"``, ``"timeseries"``, or ``"lidar"``. If None,
            all modalities are returned.
        task: Filter by downstream task (e.g. ``"segmentation"``,
            ``"classification"``, ``"detection"``). Matches if the task
            string appears anywhere in the model's task list.
        terratorch_only: If True, return only models supported by
            TerraTorch for direct loading and fine-tuning.
        huggingface_only: If True, return only models with a
            HuggingFace Hub identifier.
        as_dataframe: If True, return a :class:`pandas.DataFrame`.
            Otherwise return the raw registry dictionary slice.
        verbose: If True, log the summary table to the console.

    Returns:
        A DataFrame or dictionary describing the matching models.

    Raises:
        ValueError: If ``category`` or ``modality`` is not a recognised value.

    Example:
        >>> import geoai
        >>> df = geoai.list_foundation_models()
        >>> df = geoai.list_foundation_models(category="vision", modality="multispectral")
        >>> df = geoai.list_foundation_models(task="segmentation", terratorch_only=True)
    """
    if category is not None:
        category = category.lower()
        if category not in _VALID_CATEGORIES:
            valid = ", ".join(sorted(_VALID_CATEGORIES))
            raise ValueError(
                f"Unknown category '{category}'. Valid options: {valid}"
            )

    if modality is not None:
        modality = modality.lower()
        if modality not in _VALID_MODALITIES:
            valid = ", ".join(sorted(_VALID_MODALITIES))
            raise ValueError(
                f"Unknown modality '{modality}'. Valid options: {valid}"
            )

    models = FOUNDATION_MODELS

    if category is not None:
        models = {k: v for k, v in models.items() if v["category"] == category}

    if modality is not None:
        models = {k: v for k, v in models.items() if v["modality"] == modality}

    if task is not None:
        task_lower = task.lower()
        models = {
            k: v for k, v in models.items() if any(task_lower in t for t in v["tasks"])
        }

    if terratorch_only:
        models = {k: v for k, v in models.items() if v["terratorch_supported"]}

    if huggingface_only:
        models = {k: v for k, v in models.items() if v["huggingface_id"] is not None}

    if not as_dataframe:
        return models

    rows = [
        {
            "name": v["name"],
            "abbreviation": v["abbreviation"],
            "category": v["category"],
            "modality": v["modality"],
            "tasks": ", ".join(v["tasks"]),
            "year": v["year"],
            "publication": v["publication"],
            "terratorch_supported": v["terratorch_supported"],
            "huggingface_id": v["huggingface_id"] or "",
        }
        for v in models.values()
    ]
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        logger.info("\n%s", df.to_string(index=False))
    return df


def get_foundation_model_info(name: str) -> Dict[str, Any]:
    """Get detailed metadata for a specific foundation model.

    Args:
        name: Registry key for the model (e.g. ``"prithvi-eo-2.0-300m"``).
            Use :func:`list_foundation_models` to discover available keys.

    Returns:
        A copy of the model's metadata dictionary.

    Raises:
        ValueError: If ``name`` is not found in the registry.

    Example:
        >>> import geoai
        >>> info = geoai.get_foundation_model_info("prithvi-eo-2.0-300m")
        >>> print(info["description"])
    """
    if name not in FOUNDATION_MODELS:
        available = ", ".join(sorted(FOUNDATION_MODELS.keys()))
        raise ValueError(
            f"Unknown foundation model '{name}'. Available models: {available}"
        )
    return FOUNDATION_MODELS[name].copy()


def check_terratorch_available() -> bool:
    """Check whether TerraTorch is installed in the current environment.

    Returns:
        True if ``terratorch`` can be imported, False otherwise.

    Example:
        >>> import geoai
        >>> if geoai.check_terratorch_available():
        ...     model = geoai.load_foundation_model("prithvi-eo-2.0-300m")
    """
    try:
        import terratorch  # noqa: F401

        return True
    except ImportError:
        return False


def load_foundation_model(name: str, **kwargs: Any) -> Any:
    """Load a foundation model backbone via TerraTorch.

    This function requires ``terratorch >= 1.0`` to be installed
    (``pip install terratorch``) and is limited to models whose
    ``terratorch_supported`` registry flag is True.

    Args:
        name: Registry key for the model (e.g. ``"prithvi-eo-2.0-300m"``).
        **kwargs: Additional keyword arguments forwarded to the TerraTorch
            backbone registry builder.

    Returns:
        A PyTorch model (``torch.nn.Module``) loaded from the TerraTorch
        backbone registry.

    Raises:
        ValueError: If ``name`` is not in the registry.
        NotImplementedError: If the model does not yet have TerraTorch support.
        ImportError: If TerraTorch is not installed.
        RuntimeError: If the TerraTorch registry lookup fails.

    Example:
        >>> import geoai
        >>> model = geoai.load_foundation_model("prithvi-eo-2.0-300m")
    """
    if name not in FOUNDATION_MODELS:
        available = ", ".join(sorted(FOUNDATION_MODELS.keys()))
        raise ValueError(
            f"Unknown foundation model '{name}'. Available models: {available}"
        )

    info = FOUNDATION_MODELS[name]

    if not info["terratorch_supported"]:
        raise NotImplementedError(
            f"'{name}' ({info['name']}) does not yet have TerraTorch support. "
            "Check https://github.com/IBM/terratorch for the latest supported "
            "models, or use the model's HuggingFace ID directly: "
            f"{info['huggingface_id'] or info['code_url']}"
        )

    if not check_terratorch_available():
        raise ImportError(
            "TerraTorch is required to load foundation models. "
            "Install it with: pip install terratorch"
        )

    try:
        from terratorch.models.backbones.registry import BACKBONE_REGISTRY

        model = BACKBONE_REGISTRY.build(name.replace("-", "_"), **kwargs)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"TerraTorch failed to load '{name}': {exc}. "
            "Ensure the model name matches a TerraTorch backbone key and "
            "that any required weights are downloadable."
        ) from exc
