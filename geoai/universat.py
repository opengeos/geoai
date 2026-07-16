"""UniverSat integration module for GeoAI."""

import os, sys, subprocess, logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np, torch, torch.nn as nn, rasterio
from sklearn.decomposition import PCA
from .utils import get_device

__all__ = [
    "load_universat_model",
    "UniverSatProcessor",
    "universat_inference",
    "get_tile_embedding",
    "get_pca_rgb",
    "universat_train",
]

# Auto-clone UniverSat repo and setup environment
UNIVERSAT_CACHE_DIR = os.path.expanduser("~/.cache/geoai/UniverSat")
_src = os.path.join(UNIVERSAT_CACHE_DIR, "src")
if not os.path.exists(UNIVERSAT_CACHE_DIR):
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/gastruc/UniverSat.git",
            UNIVERSAT_CACHE_DIR,
        ],
        check=True,
        timeout=300,
    )
sys.path = [UNIVERSAT_CACHE_DIR, _src] + [
    p for p in sys.path if p not in (UNIVERSAT_CACHE_DIR, _src)
]

try:
    import torch._dynamo

    torch._dynamo.config.disable = True
except ImportError:
    pass

from hubconf import UniverSat
import hubconf
from modality_registry import INPUT_RES, SUBPATCHES, WAVELENGTHS

TIME_SERIES_MODALITIES = {
    "s1",
    "s1-asc",
    "s1flair",
    "s2",
    "s2_4band",
    "s2l1c",
    "s2flair",
    "s2withaerosol",
    "s3",
    "l8",
    "l8forestnet",
    "l7",
    "alos",  # codespell:ignore alos
    "modis",
    "hls",
}


def load_universat_model(
    pretrained=True,
    size="base",
    device=None,
    eval_mode=True,
    model_name_or_path="g-astruc/UniverSat",
    **kwargs,
) -> nn.Module:
    """Load UniverSat model backbone."""
    model = (
        UniverSat.from_pretrained(model_name_or_path, **kwargs)
        if pretrained
        else hubconf.universat(pretrained=False, size=size, **kwargs)
    )
    model = model.to(device or get_device())
    return model.eval() if eval_mode else model


class UniverSatProcessor:
    """Processor class for UniverSat, handling GeoTIFFs, normalization, and inference."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_name_or_path: str = "g-astruc/UniverSat",
        device=None,
        eval_mode=True,
        pretrained: bool = True,
        size: str = "base",
    ):
        self.device = device or get_device()
        self.model = (
            (model.to(self.device).eval() if eval_mode else model.to(self.device))
            if model
            else load_universat_model(
                device=self.device,
                eval_mode=eval_mode,
                model_name_or_path=model_name_or_path,
                pretrained=pretrained,
                size=size,
            )
        )
        self.registry_wavelengths = WAVELENGTHS

    def read_geotiff(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        with rasterio.open(file_path) as src:
            return src.read(), src.meta

    def preprocess_image(
        self, img: np.ndarray, mod: str, scale: Optional[float] = None
    ) -> torch.Tensor:
        scale = scale or 1.0
        img_norm = img.astype(np.float32) / scale
        if mod in TIME_SERIES_MODALITIES and img_norm.ndim == 3:
            img_norm = np.expand_dims(img_norm, axis=0)
        return torch.tensor(img_norm, dtype=torch.float32)

    def _process_sample(
        self, sample: Dict[str, Any], mod: str, scale: Optional[float]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (val := sample.get(mod)) is None:
            raise ValueError(f"Missing modality '{mod}'")
        if isinstance(val, str):
            img, _ = self.read_geotiff(val)
            t_val = self.preprocess_image(img, mod, scale)
        elif isinstance(val, torch.Tensor):
            t_val = val.float() / (scale or 1.0)
            if mod in TIME_SERIES_MODALITIES and t_val.ndim == 3:
                t_val = t_val.unsqueeze(0)
        else:
            t_val = self.preprocess_image(val, mod, scale)
        d_val = (
            torch.tensor(
                (
                    sample.get(f"{mod}_dates")
                    if sample.get(f"{mod}_dates") is not None
                    else [0] * t_val.shape[0]
                ),
                dtype=torch.long,
            )
            if mod in TIME_SERIES_MODALITIES or sample.get(f"{mod}_dates") is not None
            else None
        )
        return t_val, d_val

    def format_batch(
        self, samples: List[Dict[str, Any]], scales: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        mods = list(set(k for s in samples for k in s if not str(k).endswith("_dates")))
        if not mods:
            raise ValueError(f"No recognized modalities in {samples}")

        batch = {}
        for mod in mods:
            processed = [
                self._process_sample(s, mod, (scales or {}).get(mod)) for s in samples
            ]
            batch[mod] = torch.stack([p[0] for p in processed]).to(self.device)
            if processed[0][1] is not None:
                batch[f"{mod}_dates"] = torch.stack([p[1] for p in processed]).to(
                    self.device
                )
        return batch

    def encode_raster(
        self,
        samples,
        patch_size=40.0,
        output_grid=None,
        normalize_scales=None,
        **kwargs,
    ):
        batch = self.format_batch(
            [samples] if isinstance(samples, dict) else samples, normalize_scales
        )
        return self.model.encode(
            batch, patch_size=patch_size, output_grid=output_grid, **kwargs
        )


def universat_inference(
    samples, patch_size=40.0, output_grid=None, device=None, **kwargs
):
    model_kwargs = {k: v for k, v in kwargs.items() if k in ["pretrained", "size"]}
    with torch.no_grad():
        return UniverSatProcessor(
            device=device, eval_mode=True, **model_kwargs
        ).encode_raster(
            samples,
            patch_size=patch_size,
            output_grid=output_grid,
            **{k: v for k, v in kwargs.items() if k not in model_kwargs},
        )


def get_tile_embedding(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensor, got {tokens.shape}")
    return tokens.mean(dim=1 if tokens.ndim == 3 else 0)


def get_pca_rgb(tokens: torch.Tensor) -> np.ndarray:
    t = (
        tokens.detach().cpu().numpy()
        if isinstance(tokens, torch.Tensor)
        else np.asarray(tokens)
    )

    def _proj(x, g):
        p = PCA(3).fit_transform(x.reshape(-1, x.shape[-1])).reshape(g, g, 3)
        return (
            (p - p.min()) / (p.max() - p.min())
            if p.max() > p.min()
            else np.zeros_like(p)
        )

    if t.ndim == 2:
        return _proj(t, int(t.shape[0] ** 0.5))
    if t.ndim == 3 and t.shape[0] == t.shape[1]:
        return _proj(t, t.shape[0])
    return np.stack(
        [_proj(s, int(s.shape[0] ** 0.5) if s.ndim == 2 else s.shape[0]) for s in t]
    )


def universat_train(
    experiment: str,
    overrides: Optional[List[str]] = None,
    project_root: Optional[str] = None,
):
    if not os.path.exists(p := UNIVERSAT_CACHE_DIR):
        raise FileNotFoundError("UniverSat repo not found.")
    subprocess.run(
        [sys.executable, os.path.join("src", "train.py"), f"exp={experiment}"]
        + (overrides or []),
        cwd=p,
        env={**os.environ, "PROJECT_ROOT": os.path.abspath(project_root or ".")},
        check=True,
    )
