"""Canopy height estimation module using Meta's HighResCanopyHeight model.

This module provides canopy height estimation from RGB imagery using
DINOv2 backbone with DPT decoder, based on Meta's HighResCanopyHeight
research (https://github.com/facebookresearch/HighResCanopyHeight).

Reference:
    Tolan et al., "Very high resolution canopy height maps from RGB imagery
    using self-supervised vision transformer and convolutional decoder trained
    on Aerial Lidar," Remote Sensing of Environment, 2024.
    https://doi.org/10.1016/j.rse.2023.113888
"""

import math
import os
import warnings
from functools import partial
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required for canopy height estimation. "
        "Please install it: pip install torch torchvision"
    )

try:
    import rasterio
except ImportError:
    raise ImportError(
        "Rasterio is required for canopy height estimation. "
        "Please install it: pip install rasterio"
    )

__all__ = [
    "CanopyHeightEstimation",
    "canopy_height_estimation",
    "list_canopy_models",
    "MODEL_VARIANTS",
    "DEFAULT_CACHE_DIR",
]

# ---------------------------------------------------------------------------
# Model architecture components vendored from Meta's HighResCanopyHeight
# (Apache License 2.0). Adapted for standalone use without external deps.
# Source: https://github.com/facebookresearch/HighResCanopyHeight
# ---------------------------------------------------------------------------


def _resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=False,
):
    """Resize tensor using F.interpolate."""
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# -- Backbone (SSLVisionTransformer) --


class _Mlp(nn.Module):
    """MLP block with GELU activation and optional dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _LayerScale(nn.Module):
    """Per-channel learnable scaling layer."""

    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    """Transformer block with attention, MLP, and optional layer scale."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=_Attention,
        ffn_layer=_Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            _LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.ls2 = (
            _LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.sample_drop_ratio = drop_path

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patch embedding using 2D convolution."""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class _AdaptivePadding(nn.Module):
    """Adaptive padding to ensure input is divisible by kernel size."""

    def __init__(self, kernel_size=1, stride=1, padding="corner"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + kernel_h - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + kernel_w - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
        return x


class _DinoVisionTransformer(nn.Module):
    """DINOv2 Vision Transformer backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=1.0,
        norm_layer=None,
        act_layer=None,
        block_fn=_Block,
        ffn_layer="mlp",
        drop_path_uniform=False,
        **kwargs,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_cls = _Mlp
        else:
            ffn_cls = _Mlp

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_cls,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=True,
            recompute_scale_factor=True,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token + 0 * self.mask_token
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.interpolate_pos_encoding(x, w, h))
        return x

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class _SSLVisionTransformer(_DinoVisionTransformer):
    """SSL Vision Transformer with intermediate feature extraction for DPT head."""

    def __init__(
        self,
        out_indices=(4, 11, 17, 23),
        with_cls_token=True,
        output_cls_token=True,
        frozen_stages=100,
        pretrained=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_indices = list(out_indices)
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.frozen_stages = frozen_stages
        self.detach = False
        self.patch_size = self.patch_embed.patch_size
        self.adapad = _AdaptivePadding(
            kernel_size=self.patch_size, stride=self.patch_size, padding="same"
        )
        self._freeze_stages()

    def forward(self, x):
        with torch.set_grad_enabled(not self.detach):
            _, _, old_w, old_h = x.shape
            xx = self.adapad(x)
            x = F.pad(x, (0, xx.shape[-1] - x.shape[-1], 0, xx.shape[-2] - x.shape[-2]))
            B, nc, w, h = x.shape
            x = self.prepare_tokens(x)
            outs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.out_indices:
                    if self.with_cls_token:
                        out = x[:, 1:]
                    else:
                        out = x
                    B, _, C = out.shape
                    out = (
                        out.reshape(
                            B, w // self.patch_size[0], h // self.patch_size[1], C
                        )
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    if self.output_cls_token:
                        out = [out, x[:, 0]]
                    else:
                        out = [out]
                    if self.detach:
                        out = [o.detach() for o in out]
                    outs.append(out)
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for m in [self.patch_embed]:
                for param in m.parameters():
                    param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.mask_token.requires_grad = False
        if self.frozen_stages >= len(self.blocks) - 1:
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False
            self.detach = True
        for i, layer in enumerate(self.blocks):
            if i <= self.frozen_stages:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.detach = False
        self._freeze_stages()


# -- DPT Head --


def _kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class _ConvModule(nn.Module):
    """Convolution module with optional activation and Kaiming initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        act_cfg=None,
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        self.order = order
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = True
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_activation:
            self.activate = nn.ReLU()
        self.init_weights()

    @property
    def act_cfg(self):
        return {"type": "ReLU"}

    def init_weights(self):
        _kaiming_init(self.conv, a=0, nonlinearity="relu")

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class _Interpolate(nn.Module):
    """Interpolation module wrapping F.interpolate."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class _HeadDepth(nn.Module):
    """Depth prediction head with optional classification into bins."""

    def __init__(self, features, classify=False, n_bins=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            _Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                32, 1 if not classify else n_bins, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(self, x):
        return self.head(x)


class _ReassembleBlocks(nn.Module):
    """Reassemble multi-scale features from ViT for the DPT decoder."""

    def __init__(
        self,
        in_channels=1024,
        out_channels=None,
        readout_type="project",
        patch_size=16,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [128, 256, 512, 1024]
        self.readout_type = readout_type
        self.patch_size = patch_size
        self.projects = nn.ModuleList(
            [
                _ConvModule(
                    in_channels=in_channels,
                    out_channels=oc,
                    kernel_size=1,
                    act_cfg=None,
                )
                for oc in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                    for _ in range(len(self.projects))
                ]
            )

    def forward(self, inputs):
        out = []
        for i, x in enumerate(inputs):
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class _PreActResidualConvUnit(nn.Module):
    """Pre-activation residual convolution unit."""

    def __init__(self, in_channels, dilation=1):
        super().__init__()
        self.conv1 = _ConvModule(
            in_channels,
            in_channels,
            3,
            padding=dilation,
            dilation=dilation,
            act_cfg={"type": "ReLU"},
            bias=False,
            order=("act", "conv", "norm"),
        )
        self.conv2 = _ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            act_cfg={"type": "ReLU"},
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class _FeatureFusionBlock(nn.Module):
    """Feature fusion block with residual processing and 2x upsampling."""

    def __init__(self, in_channels, expand=False, align_corners=True):
        super().__init__()
        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = in_channels // 2 if expand else in_channels
        self.project = _ConvModule(
            self.in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True
        )
        self.res_conv_unit1 = _PreActResidualConvUnit(in_channels=self.in_channels)
        self.res_conv_unit2 = _PreActResidualConvUnit(in_channels=self.in_channels)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = _resize(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = _resize(
            x, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        x = self.project(x)
        return x


class _DPTHead(nn.Module):
    """Dense Prediction Transformer head for canopy height estimation."""

    def __init__(
        self,
        in_channels=(1024, 1024, 1024, 1024),
        channels=256,
        embed_dims=1024,
        post_process_channels=None,
        classify=False,
        n_bins=256,
    ):
        super().__init__()
        if post_process_channels is None:
            post_process_channels = [128, 256, 512, 1024]
        self.channels = channels
        self.min_depth = 0.001
        self.max_depth = 10
        self.n_bins = n_bins
        self.classify = classify
        self.reassemble_blocks = _ReassembleBlocks(
            in_channels=embed_dims, out_channels=post_process_channels
        )
        self.convs = nn.ModuleList(
            [
                _ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=None,
                    bias=False,
                )
                for channel in post_process_channels
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [_FeatureFusionBlock(self.channels) for _ in range(len(self.convs))]
        )
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = _ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            act_cfg={"type": "ReLU"},
        )
        self.conv_depth = _HeadDepth(self.channels, self.classify, self.n_bins)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        if self.classify:
            logit = self.conv_depth(out)
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=inputs[0][0].device
            )
            logit = torch.relu(logit)
            eps = 0.1
            logit = logit + eps
            logit = logit / logit.sum(dim=1, keepdim=True)
            out = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        else:
            out = self.relu(self.conv_depth(out)) + self.min_depth
        return out


class _SSLAE(nn.Module):
    """SSL Auto-Encoder combining backbone and DPT head."""

    def __init__(self, classify=True, n_bins=256, huge=False):
        super().__init__()
        if huge:
            self.backbone = _SSLVisionTransformer(
                embed_dim=1280,
                num_heads=20,
                out_indices=(9, 16, 22, 29),
                depth=32,
            )
            self.decode_head = _DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )
        else:
            self.backbone = _SSLVisionTransformer()
            self.decode_head = _DPTHead(classify=classify, n_bins=n_bins)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Available model variants and their S3 paths
MODEL_VARIANTS = {
    "compressed_SSLhuge": {
        "filename": "compressed_SSLhuge.pth",
        "url": "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/models/saved_checkpoints/compressed_SSLhuge.pth",
        "compressed": True,
        "huge": True,
        "description": "Compressed huge model (749M), CPU-friendly. Best general-purpose model.",
    },
    "SSLhuge_satellite": {
        "filename": "SSLhuge_satellite.pth",
        "url": "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/models/saved_checkpoints/SSLhuge_satellite.pth",
        "compressed": False,
        "huge": True,
        "description": "Full huge model (2.9G), GPU required. Best results on satellite imagery.",
    },
    "compressed_SSLlarge": {
        "filename": "compressed_SSLlarge.pth",
        "url": "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/models/saved_checkpoints/compressed_SSLlarge.pth",
        "compressed": True,
        "huge": False,
        "description": "Compressed large model (400M), CPU-friendly. Smaller ablation model.",
    },
    "compressed_SSLhuge_aerial": {
        "filename": "compressed_SSLhuge_aerial.pth",
        "url": "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/models/saved_checkpoints/compressed_SSLhuge_aerial.pth",
        "compressed": True,
        "huge": True,
        "description": "Compressed huge model trained on aerial imagery (749M), CPU-friendly.",
    },
}

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "geoai", "canopy")


def _download_checkpoint(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Download a model checkpoint from S3 if not already cached.

    Args:
        model_name: Name of the model variant.
        cache_dir: Directory to cache downloaded checkpoints.

    Returns:
        Path to the downloaded checkpoint file.
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model variant '{model_name}'. "
            f"Available variants: {list(MODEL_VARIANTS.keys())}"
        )

    info = MODEL_VARIANTS[model_name]
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, info["filename"])

    if os.path.exists(checkpoint_path):
        return checkpoint_path

    print(f"Downloading {model_name} checkpoint ({info['description']})...")
    print(f"URL: {info['url']}")
    print(f"Saving to: {checkpoint_path}")

    import urllib.request

    try:
        urllib.request.urlretrieve(info["url"], checkpoint_path)
    except Exception as e:
        # Clean up partial download
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        raise RuntimeError(
            f"Failed to download checkpoint from {info['url']}. "
            f"Error: {e}\n"
            f"You can manually download from: "
            f"s3://dataforgood-fb-data/forests/v1/models/saved_checkpoints/{info['filename']}"
        ) from e

    print("Download complete.")
    return checkpoint_path


def _load_model(
    model_name: str, checkpoint_path: str, device: str = "cpu"
) -> nn.Module:
    """Load a canopy height model from a checkpoint.

    Args:
        model_name: Name of the model variant.
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Loaded model ready for inference.
    """
    info = MODEL_VARIANTS[model_name]
    model = _SSLAE(classify=True, huge=info["huge"]).eval()

    if info["compressed"]:
        # weights_only=False is required for quantized checkpoints which
        # contain non-tensor objects (packed params, scales, zero points).
        # Checkpoints are only loaded from trusted Meta S3 URLs.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Compressed checkpoints store quantized state dicts.  Always load
        # into a quantized model on CPU first so keys match correctly.
        model_q = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d, nn.ConvTranspose2d},
            dtype=torch.qint8,
        )
        model_q.load_state_dict(ckpt, strict=False)
        if device == "cpu":
            model = model_q
        else:
            # Dequantize: extract float32 weights from quantized modules,
            # load into a fresh float model, and move to GPU.
            float_sd = {}
            for name, mod in model_q.named_modules():
                _qtypes = (
                    torch.ao.nn.quantized.dynamic.Linear,
                    torch.ao.nn.quantized.Linear,
                    torch.ao.nn.quantized.Conv2d,
                    torch.ao.nn.quantized.ConvTranspose2d,
                )
                if isinstance(mod, _qtypes):
                    w = mod.weight()
                    float_sd[name + ".weight"] = w.dequantize() if w.is_quantized else w
                    b = mod.bias()
                    if b is not None:
                        float_sd[name + ".bias"] = (
                            b.dequantize() if b.is_quantized else b
                        )
                elif hasattr(mod, "weight") and isinstance(mod.weight, nn.Parameter):
                    float_sd[name + ".weight"] = mod.weight.data
                    if hasattr(mod, "bias") and mod.bias is not None:
                        float_sd[name + ".bias"] = mod.bias.data
            for name, param in model_q.named_parameters():
                if name not in float_sd:
                    float_sd[name] = param.data
            model = _SSLAE(classify=True, huge=info["huge"]).eval()
            model.load_state_dict(float_sd, strict=False)
            model = model.to(device)
    else:
        # weights_only=False is needed because some checkpoints wrap
        # state_dict in a Lightning-style dict.  Only trusted Meta URLs.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Remove 'chm_module_.' prefix if present (from SSLModule wrapper)
        cleaned = {}
        for k, v in state_dict.items():
            new_k = k.replace("chm_module_.", "")
            cleaned[new_k] = v
        model.load_state_dict(cleaned, strict=False)
        model = model.to(device)

    return model


class CanopyHeightEstimation:
    """Estimate canopy height from RGB imagery using Meta's HighResCanopyHeight model.

    This class provides canopy height estimation from RGB aerial or satellite
    imagery using a DINOv2 backbone with DPT decoder architecture. The model
    predicts canopy height in meters for each pixel.

    The model was developed by Meta AI Research (FAIR) and is described in:
    Tolan et al., "Very high resolution canopy height maps from RGB imagery
    using self-supervised vision transformer and convolutional decoder trained
    on Aerial Lidar," Remote Sensing of Environment, 2023.

    Attributes:
        model_name (str): Name of the model variant being used.
        device (str): Device the model is running on.

    Example:
        >>> estimator = CanopyHeightEstimation()
        >>> estimator.predict("input.tif", output_path="canopy_height.tif")
    """

    def __init__(
        self,
        model_name: str = "compressed_SSLhuge",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """Initialize the CanopyHeightEstimation model.

        Args:
            model_name: Model variant to use. Options:
                - "compressed_SSLhuge" (default): Compressed huge model (749M),
                  CPU-friendly. Best general-purpose model.
                - "SSLhuge_satellite": Full huge model (2.9G), GPU required.
                  Best results on satellite imagery.
                - "compressed_SSLlarge": Compressed large model (400M), CPU-friendly.
                - "compressed_SSLhuge_aerial": Compressed huge model fine-tuned on
                  aerial imagery (749M), CPU-friendly.
            checkpoint_path: Optional path to a local checkpoint file.
                If None, the checkpoint will be downloaded automatically.
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects CUDA if available.  Compressed
                (quantized) models are loaded without quantization when placed
                on GPU, which is faster than CPU inference for large images.
            cache_dir: Directory to cache downloaded model checkpoints.
                Defaults to ~/.cache/geoai/canopy/.
        """
        self.model_name = model_name

        if model_name not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant '{model_name}'. "
                f"Available: {list(MODEL_VARIANTS.keys())}"
            )

        info = MODEL_VARIANTS[model_name]

        # Determine device – prefer CUDA when available for all model variants
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device

        # Download or use provided checkpoint
        if checkpoint_path is None:
            checkpoint_path = _download_checkpoint(model_name, cache_dir)
        self.checkpoint_path = checkpoint_path

        # Load model
        print(f"Loading {model_name} model on {device}...")
        self.model = _load_model(model_name, checkpoint_path, device)
        self.model.eval()
        print("Model loaded successfully.")

        # Image normalization (from Meta's inference.py)
        self._norm_mean = [0.420, 0.411, 0.296]
        self._norm_std = [0.213, 0.156, 0.143]

    def _normalize_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to an image tensor.

        Args:
            img_tensor: Image tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Normalized image tensor.
        """
        mean = torch.tensor(self._norm_mean, device=img_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(self._norm_std, device=img_tensor.device).view(1, 3, 1, 1)
        return (img_tensor - mean) / std

    @staticmethod
    def _make_weight_map(tile_size: int, overlap: int) -> np.ndarray:
        """Create a 2D raised-cosine weight map for smooth tile blending.

        Produces a weight map that is 1.0 in the non-overlapping centre and
        tapers smoothly to 0 at the edges using a cosine ramp over the
        overlap region.  When multiple overlapping tiles are accumulated
        with these weights the result is seamless.

        Args:
            tile_size: Size of each square tile.
            overlap: Number of pixels that overlap between adjacent tiles.

        Returns:
            2D float32 array of shape (tile_size, tile_size).
        """
        if overlap <= 0:
            return np.ones((tile_size, tile_size), dtype=np.float32)

        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))  # raised cosine

        w = np.ones(tile_size, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]

        return np.outer(w, w)

    def predict(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        tile_size: int = 256,
        overlap: int = 128,
        batch_size: int = 4,
        scale_factor: float = 10.0,
        **kwargs,
    ) -> np.ndarray:
        """Predict canopy height from a GeoTIFF image.

        Processes the input image in tiles of the specified size, runs
        inference on each tile, and assembles the results into a full
        canopy height map.  Adjacent tiles overlap and are blended using
        raised-cosine weights for seamless output.

        Args:
            input_path: Path to the input GeoTIFF file (RGB imagery).
            output_path: Optional path to save the output canopy height
                map as a GeoTIFF. If None, only returns the numpy array.
            tile_size: Size of tiles for processing (default: 256).
                The model expects 256x256 tiles.
            overlap: Number of pixels of overlap between tiles (default: 128).
                Using overlap with blending weights eliminates tile-boundary
                artefacts.  Higher values (up to tile_size // 2) give
                smoother results at the cost of more computation.
            batch_size: Number of tiles to process at once (default: 4).
                Larger values use more memory but process faster.
            scale_factor: Factor to multiply model output by to get height
                in meters (default: 10.0). This is set by the original model.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            numpy.ndarray: 2D array of canopy height values in meters.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is not a valid GeoTIFF.

        Example:
            >>> estimator = CanopyHeightEstimation()
            >>> heights = estimator.predict("aerial_image.tif",
            ...                              output_path="canopy_height.tif",
            ...                              overlap=128)
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if overlap < 0 or overlap >= tile_size:
            raise ValueError(
                f"overlap must be >= 0 and < tile_size ({tile_size}), got {overlap}"
            )

        with rasterio.open(input_path) as src:
            # Read the image
            img = src.read()  # (bands, height, width)
            profile = src.profile.copy()
            height, width = src.height, src.width

            # Handle different band counts
            if img.shape[0] >= 3:
                img = img[:3]  # Use first 3 bands (RGB)
            else:
                raise ValueError(
                    f"Input image has {img.shape[0]} bands, but at least 3 (RGB) are required."
                )

            # Convert to float32 in [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype in (np.float32, np.float64):
                img = img.astype(np.float32)
                # Clip to [0, 1] if needed
                img_max = img.max()
                if img_max > 1.0:
                    img = img / img_max
            else:
                img = img.astype(np.float32)
                img_max = img.max()
                if img_max > 0:
                    img = img / img_max

        # Calculate tile grid
        step = tile_size - overlap
        n_rows = max(1, math.ceil((height - overlap) / step))
        n_cols = max(1, math.ceil((width - overlap) / step))

        # Pad image to fit tile grid
        padded_h = n_rows * step + overlap
        padded_w = n_cols * step + overlap
        padded_img = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_img[:, :height, :width] = img

        # Blending weight map (raised-cosine taper in overlap regions)
        weight_map = self._make_weight_map(tile_size, overlap)

        # Output arrays
        output = np.zeros((padded_h, padded_w), dtype=np.float32)
        count = np.zeros((padded_h, padded_w), dtype=np.float32)

        # Process tiles in batches
        tiles = []
        positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                y = row * step
                x = col * step
                tile = padded_img[:, y : y + tile_size, x : x + tile_size]
                tiles.append(tile)
                positions.append((y, x))

                if len(tiles) == batch_size:
                    self._process_batch(
                        tiles, positions, output, count, scale_factor, weight_map
                    )
                    tiles = []
                    positions = []

        # Process remaining tiles
        if tiles:
            self._process_batch(
                tiles, positions, output, count, scale_factor, weight_map
            )

        # Average overlapping regions
        count[count == 0] = 1
        output = output / count

        # Crop back to original size
        result = output[:height, :width]

        # Save output if requested
        if output_path is not None:
            out_profile = profile.copy()
            out_profile.update(
                dtype="float32",
                count=1,
                compress="lzw",
                nodata=0,
            )
            # Remove alpha band related settings
            if "photometric" in out_profile:
                del out_profile["photometric"]

            with rasterio.open(output_path, "w", **out_profile) as dst:
                dst.write(result, 1)
            print(f"Canopy height map saved to: {output_path}")

        return result

    def _process_batch(
        self,
        tiles: list,
        positions: list,
        output: np.ndarray,
        count: np.ndarray,
        scale_factor: float,
        weight_map: Optional[np.ndarray] = None,
    ):
        """Process a batch of tiles through the model.

        Args:
            tiles: List of tile arrays (C, H, W).
            positions: List of (y, x) positions for each tile.
            output: Output array to accumulate predictions.
            count: Count array for averaging overlapping tiles.
            scale_factor: Scale factor for model output.
            weight_map: Optional 2D weight array for blending overlapping
                tiles.  If None, uniform weights (1.0) are used.
        """
        tile_size = tiles[0].shape[1]
        batch = torch.from_numpy(np.stack(tiles)).to(self.device)
        batch = self._normalize_image(batch)

        with torch.no_grad():
            pred = self.model(batch)
            pred = pred * scale_factor
            pred = pred.relu()

        pred = pred.cpu().numpy()

        if weight_map is None:
            weight_map = np.ones((tile_size, tile_size), dtype=np.float32)

        for i, (y, x) in enumerate(positions):
            output[y : y + tile_size, x : x + tile_size] += pred[i, 0] * weight_map
            count[y : y + tile_size, x : x + tile_size] += weight_map

    def visualize(
        self,
        input_path: str,
        height_map: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 6),
        cmap: str = "viridis",
        vmin: float = 0,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize the input image alongside the canopy height map.

        Args:
            input_path: Path to the input GeoTIFF image, or path to the
                canopy height GeoTIFF if height_map is None.
            height_map: Canopy height map array. If None, will attempt to
                read from input_path (assuming it's a height map).
            output_path: Optional path to save the figure.
            figsize: Figure size as (width, height).
            cmap: Colormap for the height map.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap. If None, uses the 98th
                percentile of the height values.
            title: Optional title for the figure.
            **kwargs: Additional keyword arguments for matplotlib.

        Returns:
            matplotlib.figure.Figure: The visualization figure.

        Example:
            >>> estimator = CanopyHeightEstimation()
            >>> heights = estimator.predict("input.tif")
            >>> fig = estimator.visualize("input.tif", heights)
        """
        if height_map is None:
            # Assume input_path is the height map
            with rasterio.open(input_path) as src:
                height_map = src.read(1)
            fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
            if vmax is None:
                vmax = (
                    np.percentile(height_map[height_map > 0], 98)
                    if np.any(height_map > 0)
                    else 1
                )
            im = ax.imshow(height_map, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title or "Canopy Height Map")
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Height (meters)")
            ax.axis("off")
        else:
            # Show input image and height map side by side
            with rasterio.open(input_path) as src:
                img = src.read()
                if img.shape[0] >= 3:
                    img = img[:3]
                if img.dtype == np.uint8:
                    img_display = np.moveaxis(img, 0, 2)
                elif img.dtype == np.uint16:
                    img_float = img.astype(np.float32)
                    img_max = img_float.max()
                    if img_max > 0:
                        img_float = img_float / img_max
                    img_display = np.moveaxis(img_float, 0, 2)
                else:
                    img_float = img.astype(np.float32)
                    img_max = img_float.max()
                    if img_max > 1.0:
                        img_float = img_float / img_max
                    img_display = np.moveaxis(img_float, 0, 2)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            ax1.imshow(img_display)
            ax1.set_title("Input Image")
            ax1.axis("off")

            if vmax is None:
                vmax = (
                    np.percentile(height_map[height_map > 0], 98)
                    if np.any(height_map > 0)
                    else 1
                )
            im = ax2.imshow(height_map, cmap=cmap, vmin=vmin, vmax=vmax)
            ax2.set_title("Canopy Height Map")
            ax2.axis("off")
            cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("Height (meters)")

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {output_path}")

        return fig


def canopy_height_estimation(
    input_path: str,
    output_path: Optional[str] = None,
    model_name: str = "compressed_SSLhuge",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    tile_size: int = 256,
    overlap: int = 128,
    batch_size: int = 4,
    cache_dir: str = DEFAULT_CACHE_DIR,
    **kwargs,
) -> np.ndarray:
    """Convenience function for canopy height estimation.

    Creates a CanopyHeightEstimation instance and runs prediction in one call.

    Args:
        input_path: Path to the input GeoTIFF file (RGB imagery).
        output_path: Optional path to save the output as a GeoTIFF.
        model_name: Model variant to use. Default: "compressed_SSLhuge".
            See CanopyHeightEstimation for available options.
        checkpoint_path: Optional path to a local checkpoint file.
        device: Device to run inference on. If None, auto-selects.
        tile_size: Size of tiles for processing (default: 256).
        overlap: Overlap between tiles in pixels (default: 128).
            Higher values give smoother results.
        batch_size: Number of tiles per batch (default: 4).
        cache_dir: Directory to cache model checkpoints.
        **kwargs: Additional keyword arguments passed to predict().

    Returns:
        numpy.ndarray: 2D array of canopy height values in meters.

    Example:
        >>> heights = canopy_height_estimation(
        ...     "aerial_image.tif",
        ...     output_path="canopy_height.tif",
        ...     model_name="compressed_SSLhuge",
        ... )
    """
    estimator = CanopyHeightEstimation(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        cache_dir=cache_dir,
    )
    return estimator.predict(
        input_path=input_path,
        output_path=output_path,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        **kwargs,
    )


def list_canopy_models() -> Dict[str, str]:
    """List available canopy height estimation model variants.

    Returns:
        Dictionary mapping model names to their descriptions.

    Example:
        >>> models = list_canopy_models()
        >>> for name, desc in models.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: info["description"] for name, info in MODEL_VARIANTS.items()}
