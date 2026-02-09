from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class CameraConditionConfig:
    cam_dim: int = 13
    embed_dim: int = 512
    latent_channels: int = 16
    context_dim: int = 4096
    mode: Literal["latent_bias", "context_token", "both"] = "both"


class CameraEncoder(nn.Module):
    """Encode camera parameters to a dense embedding."""

    def __init__(self, cam_dim: int = 13, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cam_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, camera_params: torch.Tensor) -> torch.Tensor:
        return self.net(camera_params)


class CameraConditioner(nn.Module):
    """Project camera embedding to latent-bias and/or context-token conditions."""

    def __init__(self, cfg: CameraConditionConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CameraEncoder(cam_dim=cfg.cam_dim, embed_dim=cfg.embed_dim)
        self.to_latent = nn.Linear(cfg.embed_dim, cfg.latent_channels)
        self.to_context = nn.Linear(cfg.embed_dim, cfg.context_dim)

    def forward(self, camera_params: torch.Tensor) -> torch.Tensor:
        return self.encoder(camera_params)

    def apply_to_latent(self, latent: torch.Tensor, cam_embed: torch.Tensor) -> torch.Tensor:
        """latent: [C,F,H,W], cam_embed: [D] for a single sample."""
        if self.cfg.mode not in {"latent_bias", "both"}:
            return latent
        bias = self.to_latent(cam_embed).view(-1, 1, 1, 1)
        return latent + bias

    def apply_to_context(self, context: torch.Tensor, cam_embed: torch.Tensor) -> torch.Tensor:
        """context: [L,C], returns [L+1,C] when context token is enabled."""
        if self.cfg.mode not in {"context_token", "both"}:
            return context
        cam_token = self.to_context(cam_embed).unsqueeze(0)
        return torch.cat([context, cam_token], dim=0)
