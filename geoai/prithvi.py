"""Prithvi EO 2.0 module for geospatial foundation model inference.

This module provides tools for using NASA-IBM's Prithvi EO 2.0 geospatial foundation model
for masked autoencoding and feature extraction on multi-temporal satellite imagery.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block

from .utils import get_device

logger = logging.getLogger(__name__)

# Constants
NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99.9

# Available Prithvi models
AVAILABLE_MODELS = [
    "Prithvi-EO-2.0-tiny-TL",  # tiny transfer learning, embed_dim=192, depth=12, with coords
    "Prithvi-EO-2.0-100M-TL",  # 100M transfer learning, embed_dim=768, depth=12, with coords
    "Prithvi-EO-2.0-300M",  # 300M base model, embed_dim=1024, depth=24, no coords
    "Prithvi-EO-2.0-300M-TL",  # 300M transfer learning, embed_dim=768, depth=12, with coords
    "Prithvi-EO-2.0-600M",  # 600M base model, embed_dim=1280, depth=32, no coords
    "Prithvi-EO-2.0-600M-TL",  # 600M transfer learning, embed_dim=1280, depth=32, with coords
]


def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Create 3D sin/cos positional embeddings.

    Args:
        embed_dim (int): Embedding dimension.
        grid_size (tuple[int, int, int] | list[int]): The grid depth, height and width.
        add_cls_token (bool, optional): Whether or not to add a classification (CLS) token.

    Returns:
        Position embeddings (with or without cls token)
    """
    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sincos position embeddings."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
    """Modified torch version of get_1d_sincos_pos_embed_from_grid()."""
    assert embed_dim % 2 == 0
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def _init_weights(module):
    """Initialize the weights."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class PatchEmbed(nn.Module):
    """3D Patch Embedding."""

    def __init__(
        self,
        input_size: tuple[int, int, int] = (1, 224, 224),
        patch_size: tuple[int, int, int] = (1, 16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        assert all(
            g >= 1 for g in self.grid_size
        ), "Patch size is bigger than input size."
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class TemporalEncoder(nn.Module):
    """Temporal coordinate encoder."""

    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        if trainable_scale:
            self.scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.scale = 1.0

    def forward(
        self, temporal_coords: torch.Tensor, tokens_per_frame: int | None = None
    ):
        """
        temporal_coords: year and day-of-year info with shape (B, T, 2).
        """
        shape = temporal_coords.shape[:2] + (-1,)

        year = _get_1d_sincos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()
        ).reshape(shape)
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()
        ).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding


class LocationEncoder(nn.Module):
    """Location coordinate encoder."""

    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        if trainable_scale:
            self.scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.scale = 1.0

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)

        lat = _get_1d_sincos_embed_from_grid_torch(
            self.lat_embed_dim, location_coords[:, 0].flatten()
        ).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(
            self.lon_embed_dim, location_coords[:, 1].flatten()
        ).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding


class PrithviViT(nn.Module):
    """Prithvi ViT Encoder."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)

        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        coords_encoding = coords_encoding or []
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding

        if self.temporal_encoding:
            self.temporal_encoder = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_encoder = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def random_masking(self, sequence, mask_ratio, noise=None):
        N, L, D = sequence.shape
        len_keep = int(L * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(N, L, device=sequence.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, mask, ids_restore

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio=0.75,
    ):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            x = x + self.temporal_encoder(
                temporal_coords, x.shape[1] // self.num_frames
            )

        if self.location_encoding and location_coords is not None:
            x = x + self.location_encoder(location_coords)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """Transformer Decoder used in the Prithvi MAE."""

    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        grid_size: list[int] | tuple[int, int, int] = (3, 14, 14),
        in_chans: int = 3,
        encoder_embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = grid_size
        self.in_chans = in_chans
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.register_buffer(
            "decoder_pos_embed",
            torch.zeros(
                1, grid_size[0] * grid_size[1] * grid_size[2] + 1, decoder_embed_dim
            ),
        )

        coords_encoding = coords_encoding or []
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding

        if self.temporal_encoding:
            self.temporal_encoder = TemporalEncoder(
                decoder_embed_dim, coords_scale_learn
            )
        if self.location_encoding:
            self.location_encoder = LocationEncoder(
                decoder_embed_dim, coords_scale_learn
            )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size[0] * patch_size[1] * patch_size[2] * in_chans,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size, add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        input_size: list[int] = None,
    ):
        x = self.decoder_embed(hidden_states)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        if self.temporal_encoding and temporal_coords is not None:
            num_frames = temporal_coords.shape[1]
            tokens_per_frame = (x.shape[1] - 1) // num_frames
            temp_embed = self.temporal_encoder(temporal_coords, tokens_per_frame)
            x[:, 1:, :] = x[:, 1:, :] + temp_embed

        if self.location_encoding and location_coords is not None:
            x[:, 1:, :] = x[:, 1:, :] + self.location_encoder(location_coords)

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x


class PrithviMAE(nn.Module):
    """Prithvi Masked Autoencoder."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        num_frames: int = 4,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        mask_ratio: float = 0.75,
        **kwargs,
    ):
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (1, patch_size, patch_size)
        )
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss

        self.encoder = PrithviViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
        )

        self.decoder = MAEDecoder(
            patch_size=self.patch_size,
            grid_size=self.encoder.patch_embed.grid_size,
            in_chans=in_chans,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
        )

    def patchify(self, pixel_values):
        B, C, T, H, W = pixel_values.shape
        pH = H // self.patch_size[1]
        pW = W // self.patch_size[2]

        x = pixel_values.reshape(
            B,
            C,
            T // self.patch_size[0],
            self.patch_size[0],
            pH,
            self.patch_size[1],
            pW,
            self.patch_size[2],
        )
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        patchified_pixel_values = x.reshape(
            B,
            T // self.patch_size[0] * pH * pW,
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * C,
        )

        return patchified_pixel_values

    def unpatchify(
        self, patchified_pixel_values, image_size: tuple[int, int] | None = None
    ):
        if image_size is None:
            H, W = self.img_size
        else:
            H, W = image_size

        C = self.in_chans
        pH = H // self.patch_size[1]
        pW = W // self.patch_size[2]
        T = self.num_frames

        x = patchified_pixel_values.reshape(
            patchified_pixel_values.shape[0],
            T // self.patch_size[0],
            pH,
            pW,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            C,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        pixel_values = x.reshape(
            patchified_pixel_values.shape[0],
            C,
            T,
            pH * self.patch_size[1],
            pW * self.patch_size[2],
        )

        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        target = self.patchify(pixel_values)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio: float = None,
    ):
        mask_ratio = mask_ratio if mask_ratio is not None else 0.75

        latent, mask, ids_restore = self.encoder(
            pixel_values, temporal_coords, location_coords, mask_ratio
        )
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords)
        loss = self.forward_loss(pixel_values, pred, mask)

        return loss, pred, mask


class PrithviProcessor:
    """Prithvi EO 2.0 processor with GeoTIFF input/output support.

    Supports multiple model variants:
    - Prithvi-EO-2.0-tiny-TL (tiny transfer learning)
    - Prithvi-EO-2.0-100M-TL (100M transfer learning)
    - Prithvi-EO-2.0-300M (300M base model)
    - Prithvi-EO-2.0-300M-TL (300M transfer learning)
    - Prithvi-EO-2.0-600M (600M base model)
    - Prithvi-EO-2.0-600M-TL (600M transfer learning)

    References:
        - tiny-TL: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL
        - 100M-TL: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL
        - 300M: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
        - 300M-TL: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
        - 600M: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
        - 600M-TL: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL
        - GitHub: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
    """

    def __init__(
        self,
        model_name: str = "Prithvi-EO-2.0-300M-TL",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Prithvi processor.

        Args:
            model_name: Name of the Prithvi model to download from HuggingFace Hub.
                Options:
                - "Prithvi-EO-2.0-tiny-TL" (tiny, 192 dim, 12 layers)
                - "Prithvi-EO-2.0-100M-TL" (100M, 768 dim, 12 layers)
                - "Prithvi-EO-2.0-300M" (base, 1024 dim, 24 layers)
                - "Prithvi-EO-2.0-300M-TL" (default, 768 dim, 12 layers)
                - "Prithvi-EO-2.0-600M" (base, 1280 dim, 32 layers)
                - "Prithvi-EO-2.0-600M-TL" (1280 dim, 32 layers)
            config_path: Path to config file (optional, downloads if not provided)
            checkpoint_path: Path to checkpoint file (optional, downloads if not provided)
            device: Torch device to use
            cache_dir: Directory to cache downloaded files
        """
        self.device = device or get_device()
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Download or load config and checkpoint
        if config_path is None or checkpoint_path is None:
            config_path, checkpoint_path = self.download_model(model_name, cache_dir)

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Load config
        with open(config_path, "r") as f:
            config_data = json.load(f)
            self.config = config_data["pretrained_cfg"]

        # Extract parameters
        self.bands = self.config["bands"]
        self.mean = self.config["mean"]
        self.std = self.config["std"]
        self.img_size = self.config["img_size"]
        self.patch_size = self.config["patch_size"]
        self.mask_ratio = self.config["mask_ratio"]
        self.num_frames = self.config.get("num_frames", 4)
        self.coords_encoding = self.config.get("coords_encoding", [])

        # Load model
        self.model = self._load_model()

    @staticmethod
    def download_model(
        model_name: str = "Prithvi-EO-2.0-300M-TL", cache_dir: str = None
    ) -> Tuple[str, str]:
        """Download Prithvi model from HuggingFace Hub.

        Args:
            model_name: Name of the model. Options:
                - "Prithvi-EO-2.0-tiny-TL"
                - "Prithvi-EO-2.0-100M-TL"
                - "Prithvi-EO-2.0-300M" (base model)
                - "Prithvi-EO-2.0-300M-TL" (default)
                - "Prithvi-EO-2.0-600M" (base model)
                - "Prithvi-EO-2.0-600M-TL"
            cache_dir: Directory to cache files

        Returns:
            Tuple of (config_path, checkpoint_path)
        """
        repo_id = f"ibm-nasa-geospatial/{model_name}"

        try:
            # Download config
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                cache_dir=cache_dir,
            )

            # Download checkpoint
            # Model name format: Prithvi-EO-2.0-300M-TL -> Prithvi_EO_V2_300M_TL.pt
            checkpoint_filename = (
                model_name.replace("-", "_").replace("_2.0_", "_V2_") + ".pt"
            )
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=checkpoint_filename,
                cache_dir=cache_dir,
            )

            return config_path, checkpoint_path

        except Exception as e:
            raise RuntimeError(f"Failed to download model from HuggingFace Hub: {e}")

    def _load_model(self) -> PrithviMAE:
        """Load Prithvi MAE model."""
        try:
            # Convert patch_size to tuple if it's a list
            patch_size = self.config["patch_size"]
            if isinstance(patch_size, list):
                patch_size = tuple(patch_size)

            # Create model
            model = PrithviMAE(
                img_size=self.config["img_size"],
                patch_size=patch_size,
                num_frames=self.config["num_frames"],
                in_chans=self.config["in_chans"],
                embed_dim=self.config["embed_dim"],
                depth=self.config["depth"],
                num_heads=self.config["num_heads"],
                decoder_embed_dim=self.config["decoder_embed_dim"],
                decoder_depth=self.config["decoder_depth"],
                decoder_num_heads=self.config["decoder_num_heads"],
                mlp_ratio=self.config["mlp_ratio"],
                coords_encoding=self.coords_encoding,
                coords_scale_learn=self.config.get("coords_scale_learn", False),
                mask_ratio=self.mask_ratio,
                norm_pix_loss=self.config.get("norm_pix_loss", False),
            )

            # Load checkpoint
            state_dict = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )

            # Remove fixed pos_embed weights
            for k in list(state_dict.keys()):
                if "pos_embed" in k:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load Prithvi model: {e}")

    def read_geotiff(self, file_path: str) -> Tuple[np.ndarray, dict, Optional[Tuple]]:
        """Read GeoTIFF file.

        Args:
            file_path: Path to GeoTIFF file

        Returns:
            Tuple of (image array, metadata, coordinates)
        """
        with rasterio.open(file_path) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.tags()
            except:
                coords = None

        return img, meta, coords

    def preprocess_image(
        self,
        img: np.ndarray,
        indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            img: Image array with shape (C, H, W)
            indices: Optional band indices to select

        Returns:
            Preprocessed image
        """
        # Move channels to last dimension
        img = np.moveaxis(img, 0, -1)

        # Select bands if specified
        if indices is not None:
            img = img[..., indices]

        # Normalize (handle nodata)
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - self.mean) / self.std)

        return img

    def load_images(
        self,
        file_paths: List[str],
        indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[dict], List, List]:
        """Load and preprocess multiple images.

        Args:
            file_paths: List of GeoTIFF file paths
            indices: Optional band indices

        Returns:
            Tuple of (images, metadata, temporal_coords, location_coords)
        """
        # Check if we need to pad to num_frames
        if len(file_paths) < self.num_frames:
            # Pad file_paths by repeating the last file
            file_paths = list(file_paths) + [file_paths[-1]] * (
                self.num_frames - len(file_paths)
            )
        elif len(file_paths) > self.num_frames:
            file_paths = file_paths[: self.num_frames]

        imgs = []
        metas = []
        temporal_coords = []
        location_coords = []

        for file in file_paths:
            img, meta, coords = self.read_geotiff(file)

            # Preprocess
            img = self.preprocess_image(img, indices)

            imgs.append(img)
            metas.append(meta)

        # Stack images: (T, H, W, C)
        imgs = np.stack(imgs, axis=0)
        # Rearrange to: (C, T, H, W)
        imgs = np.moveaxis(imgs, -1, 0).astype("float32")
        # Add batch dimension: (1, C, T, H, W)
        imgs = np.expand_dims(imgs, axis=0)

        return imgs, metas, temporal_coords, location_coords

    def run_inference(
        self,
        input_data: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run model inference.

        Args:
            input_data: Input tensor with shape (B, C, T, H, W)
            temporal_coords: Optional temporal coordinates
            location_coords: Optional location coordinates
            mask_ratio: Mask ratio (default: from config)

        Returns:
            Tuple of (reconstructed_image, mask_image)
        """
        mask_ratio = mask_ratio or self.mask_ratio

        # Check if input dimensions match model expectations
        B, C, T, H, W = input_data.shape
        if H % self.img_size != 0 or W % self.img_size != 0:
            raise ValueError(
                f"Input spatial dimensions ({H}x{W}) must be divisible by model image size ({self.img_size}). "
                f"Use process_files() method which handles padding automatically, or pad your input to multiples of {self.img_size}."
            )

        with torch.no_grad():
            x = input_data.to(self.device)
            _, pred, mask = self.model(x, temporal_coords, location_coords, mask_ratio)

        # Create mask and prediction images
        mask_img = (
            self.model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]))
            .detach()
            .cpu()
        )
        pred_img = self.model.unpatchify(pred).detach().cpu()

        # Mix visible and predicted patches
        rec_img = input_data.clone()
        rec_img[mask_img == 1] = pred_img[mask_img == 1]

        # Invert mask for better visualization
        mask_img = (~(mask_img.to(torch.bool))).to(torch.float)

        return rec_img, mask_img

    def process_images(
        self,
        file_paths: List[str],
        mask_ratio: Optional[float] = None,
        indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process multiple GeoTIFF files and return tensors (without saving).

        This method handles large images using sliding windows and returns tensors
        for visualization, unlike process_files() which saves to disk.

        Args:
            file_paths: List of input file paths
            mask_ratio: Optional mask ratio
            indices: Optional band indices

        Returns:
            Tuple of (input_tensor, reconstructed_tensor, mask_tensor)
        """
        # Load images
        input_data, metas, temporal_coords, location_coords = self.load_images(
            file_paths, indices
        )

        # Handle padding
        original_h, original_w = input_data.shape[-2:]
        pad_h = (self.img_size - (original_h % self.img_size)) % self.img_size
        pad_w = (self.img_size - (original_w % self.img_size)) % self.img_size

        if pad_h > 0 or pad_w > 0:
            input_data = np.pad(
                input_data,
                ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )

        # Convert to tensor
        batch = torch.tensor(input_data, device="cpu")

        # Create sliding windows
        windows = batch.unfold(3, self.img_size, self.img_size).unfold(
            4, self.img_size, self.img_size
        )
        h1, w1 = windows.shape[3:5]
        windows = rearrange(
            windows,
            "b c t h1 w1 h w -> (b h1 w1) c t h w",
            h=self.img_size,
            w=self.img_size,
        )

        # Split into batches
        num_batches = max(1, windows.shape[0])
        windows_list = torch.tensor_split(windows, num_batches, dim=0)

        # Process each window
        rec_imgs = []
        mask_imgs = []

        for i, x in enumerate(windows_list):
            rec_img, mask_img = self.run_inference(x, None, None, mask_ratio)
            rec_imgs.append(rec_img)
            mask_imgs.append(mask_img)

        # Concatenate results
        rec_imgs = torch.cat(rec_imgs, dim=0)
        mask_imgs = torch.cat(mask_imgs, dim=0)

        # Rearrange patches back to image
        num_frames = len(file_paths)
        rec_imgs = rearrange(
            rec_imgs,
            "(b h1 w1) c t h w -> b c t (h1 h) (w1 w)",
            h=self.img_size,
            w=self.img_size,
            b=1,
            c=len(self.bands),
            t=num_frames,
            h1=h1,
            w1=w1,
        )
        mask_imgs = rearrange(
            mask_imgs,
            "(b h1 w1) c t h w -> b c t (h1 h) (w1 w)",
            h=self.img_size,
            w=self.img_size,
            b=1,
            c=len(self.bands),
            t=num_frames,
            h1=h1,
            w1=w1,
        )

        # Remove padding
        rec_imgs = rec_imgs[..., :original_h, :original_w]
        mask_imgs = mask_imgs[..., :original_h, :original_w]
        input_imgs = batch[..., :original_h, :original_w]

        return input_imgs, rec_imgs, mask_imgs

    def visualize_rgb(
        self,
        input_tensor: torch.Tensor,
        rec_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Extract RGB images from tensors for visualization.

        Args:
            input_tensor: Input tensor (B, C, T, H, W)
            rec_tensor: Reconstructed tensor (B, C, T, H, W)
            mask_tensor: Mask tensor (B, C, T, H, W)

        Returns:
            Tuple of (original_rgb, masked_rgb, reconstructed_rgb) lists
        """
        # Get RGB channel indices (B04=Red, B03=Green, B02=Blue)
        rgb_channels = [
            self.bands.index("B04"),
            self.bands.index("B03"),
            self.bands.index("B02"),
        ]

        # Remove batch dimension
        if input_tensor.dim() == 5:
            input_tensor = input_tensor[0]
        if rec_tensor.dim() == 5:
            rec_tensor = rec_tensor[0]
        if mask_tensor.dim() == 5:
            mask_tensor = mask_tensor[0]

        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)

        original_rgb = []
        masked_rgb = []
        reconstructed_rgb = []

        num_frames = input_tensor.shape[1]

        for t in range(num_frames):
            # Extract and denormalize original RGB
            rgb_orig = input_tensor[rgb_channels, t, :, :].clone()
            for i, c in enumerate(rgb_channels):
                rgb_orig[i] = rgb_orig[i] * std[c] + mean[c]
            rgb_orig_np = rgb_orig.numpy()
            rgb_orig_np = np.clip(rgb_orig_np, 0, 10000)
            rgb_orig_np = (rgb_orig_np / 10000 * 255).astype(np.uint8)
            rgb_orig_np = np.transpose(rgb_orig_np, (1, 2, 0))
            original_rgb.append(rgb_orig_np)

            # Extract and denormalize reconstructed RGB
            rgb_rec = rec_tensor[rgb_channels, t, :, :].clone()
            for i, c in enumerate(rgb_channels):
                rgb_rec[i] = rgb_rec[i] * std[c] + mean[c]
            rgb_rec_np = rgb_rec.numpy()
            rgb_rec_np = np.clip(rgb_rec_np, 0, 10000)
            rgb_rec_np = (rgb_rec_np / 10000 * 255).astype(np.uint8)
            rgb_rec_np = np.transpose(rgb_rec_np, (1, 2, 0))
            reconstructed_rgb.append(rgb_rec_np)

            # Create masked RGB (visible patches only)
            mask_t = mask_tensor[rgb_channels, t, :, :].numpy()
            masked_np = rgb_orig_np.astype(np.float32) * np.transpose(mask_t, (1, 2, 0))
            masked_rgb.append(masked_np.astype(np.uint8))

        return original_rgb, masked_rgb, reconstructed_rgb

    def process_files(
        self,
        file_paths: List[str],
        output_dir: str,
        mask_ratio: Optional[float] = None,
        indices: Optional[List[int]] = None,
    ):
        """Process multiple GeoTIFF files.

        Args:
            file_paths: List of input file paths
            output_dir: Output directory for results
            mask_ratio: Optional mask ratio
            indices: Optional band indices
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load images
        input_data, metas, temporal_coords, location_coords = self.load_images(
            file_paths, indices
        )

        # Handle padding
        original_h, original_w = input_data.shape[-2:]
        pad_h = self.img_size - (original_h % self.img_size)
        pad_w = self.img_size - (original_w % self.img_size)
        input_data = np.pad(
            input_data, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )

        # Convert to tensor
        batch = torch.tensor(input_data, device="cpu")

        # Create sliding windows
        windows = batch.unfold(3, self.img_size, self.img_size).unfold(
            4, self.img_size, self.img_size
        )
        h1, w1 = windows.shape[3:5]
        windows = rearrange(
            windows,
            "b c t h1 w1 h w -> (b h1 w1) c t h w",
            h=self.img_size,
            w=self.img_size,
        )

        # Split into batches
        num_batches = max(1, windows.shape[0])
        windows_list = torch.tensor_split(windows, num_batches, dim=0)

        # Process each window
        rec_imgs = []
        mask_imgs = []

        for i, x in enumerate(windows_list):
            rec_img, mask_img = self.run_inference(x, None, None, mask_ratio)
            rec_imgs.append(rec_img)
            mask_imgs.append(mask_img)

        # Concatenate results
        rec_imgs = torch.cat(rec_imgs, dim=0)
        mask_imgs = torch.cat(mask_imgs, dim=0)

        # Rearrange patches back to image
        num_frames = len(file_paths)
        rec_imgs = rearrange(
            rec_imgs,
            "(b h1 w1) c t h w -> b c t (h1 h) (w1 w)",
            h=self.img_size,
            w=self.img_size,
            b=1,
            c=len(self.bands),
            t=num_frames,
            h1=h1,
            w1=w1,
        )
        mask_imgs = rearrange(
            mask_imgs,
            "(b h1 w1) c t h w -> b c t (h1 h) (w1 w)",
            h=self.img_size,
            w=self.img_size,
            b=1,
            c=len(self.bands),
            t=num_frames,
            h1=h1,
            w1=w1,
        )

        # Remove padding
        rec_imgs = rec_imgs[..., :original_h, :original_w]
        mask_imgs = mask_imgs[..., :original_h, :original_w]

        # Save results
        self.save_results(rec_imgs[0], mask_imgs[0], metas, output_dir)

    def save_results(
        self,
        rec_img: torch.Tensor,
        mask_img: torch.Tensor,
        metas: List[dict],
        output_dir: str,
    ):
        """Save reconstruction results.

        Args:
            rec_img: Reconstructed image with shape (C, T, H, W)
            mask_img: Mask image with shape (C, T, H, W)
            metas: List of metadata dicts
            output_dir: Output directory
        """
        mean = torch.tensor(np.asarray(self.mean)[:, None, None])
        std = torch.tensor(np.asarray(self.std)[:, None, None])

        for t in range(rec_img.shape[1]):
            # Denormalize
            rec_img_t = ((rec_img[:, t, :, :] * std) + mean).to(torch.int16)
            mask_img_t = mask_img[:, t, :, :].to(torch.int16)

            # Update metadata
            meta = metas[t].copy()
            meta.update(compress="lzw", nodata=0)

            # Save files
            self._save_geotiff(
                rec_img_t.numpy(),
                os.path.join(output_dir, f"reconstructed_t{t}.tif"),
                meta,
            )
            self._save_geotiff(
                mask_img_t.numpy(),
                os.path.join(output_dir, f"mask_t{t}.tif"),
                meta,
            )

    @staticmethod
    def _save_geotiff(image: np.ndarray, output_path: str, meta: dict):
        """Save GeoTIFF file."""
        with rasterio.open(output_path, "w", **meta) as dest:
            for i in range(image.shape[0]):
                dest.write(image[i], i + 1)


def get_available_prithvi_models() -> List[str]:
    """Get list of available Prithvi model names.

    Returns:
        List of available model names

    Example:
        >>> models = get_available_prithvi_models()
        >>> print(models)
        ['Prithvi-EO-2.0-300M-TL', 'Prithvi-EO-2.0-600M-TL']
    """
    return AVAILABLE_MODELS.copy()


def load_prithvi_model(
    model_name: str = "Prithvi-EO-2.0-300M-TL",
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PrithviProcessor:
    """Load Prithvi model (convenience function).

    Args:
        model_name: Name of the model. Options:
            - "Prithvi-EO-2.0-tiny-TL"
            - "Prithvi-EO-2.0-100M-TL"
            - "Prithvi-EO-2.0-300M" (base)
            - "Prithvi-EO-2.0-300M-TL" (default)
            - "Prithvi-EO-2.0-600M" (base)
            - "Prithvi-EO-2.0-600M-TL"
        device: Device to use ('cuda' or 'cpu')
        cache_dir: Cache directory

    Returns:
        PrithviProcessor instance

    Example:
        >>> # Load tiny-TL model
        >>> processor = load_prithvi_model("Prithvi-EO-2.0-tiny-TL")
        >>> # Load 100M-TL model
        >>> processor = load_prithvi_model("Prithvi-EO-2.0-100M-TL")
        >>> # Load 300M base model
        >>> processor = load_prithvi_model("Prithvi-EO-2.0-300M")
        >>> # Load 300M-TL model
        >>> processor = load_prithvi_model("Prithvi-EO-2.0-300M-TL")
        >>> # Load 600M base model
        >>> processor = load_prithvi_model("Prithvi-EO-2.0-600M")
        >>> # Load 600M-TL model
        >>> processor = load_prithvi_model("Prithvi-EO-2.0-600M-TL")
    """
    if device is not None:
        device = torch.device(device)

    return PrithviProcessor(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )


def prithvi_inference(
    file_paths: List[str],
    output_dir: str = "output",
    model_name: str = "Prithvi-EO-2.0-300M-TL",
    mask_ratio: Optional[float] = None,
    device: Optional[str] = None,
):
    """Run Prithvi inference on GeoTIFF files (convenience function).

    Args:
        file_paths: List of input GeoTIFF files
        output_dir: Output directory
        model_name: Name of the model. Options:
            - "Prithvi-EO-2.0-tiny-TL"
            - "Prithvi-EO-2.0-100M-TL"
            - "Prithvi-EO-2.0-300M" (base)
            - "Prithvi-EO-2.0-300M-TL" (default)
            - "Prithvi-EO-2.0-600M" (base)
            - "Prithvi-EO-2.0-600M-TL"
        mask_ratio: Optional mask ratio
        device: Device to use

    Example:
        >>> # Use tiny-TL model
        >>> prithvi_inference(
        ...     file_paths=["img1.tif", "img2.tif", "img3.tif", "img4.tif"],
        ...     model_name="Prithvi-EO-2.0-tiny-TL",
        ...     output_dir="output_tiny"
        ... )
    """
    processor = load_prithvi_model(model_name, device)
    processor.process_files(file_paths, output_dir, mask_ratio)
