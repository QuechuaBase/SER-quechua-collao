"""Temporal pooling layers for frame-level speech representations."""

from __future__ import annotations

import torch
from torch import nn


def _valid_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    return mask.to(device=x.device, dtype=torch.bool)


class MeanPooling(nn.Module):
    output_multiplier = 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        valid = _valid_mask(x, mask).unsqueeze(-1)
        denominator = valid.sum(dim=1).clamp_min(1)
        return (x * valid).sum(dim=1) / denominator


class MeanStdPooling(nn.Module):
    output_multiplier = 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        valid = _valid_mask(x, mask).unsqueeze(-1)
        denominator = valid.sum(dim=1).clamp_min(1)
        mean = (x * valid).sum(dim=1) / denominator
        variance = ((x - mean.unsqueeze(1)).square() * valid).sum(dim=1) / denominator
        std = variance.clamp_min(1e-8).sqrt()
        return torch.cat([mean, std], dim=-1)


class AttentivePooling(nn.Module):
    output_multiplier = 1

    def __init__(self, input_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        valid = _valid_mask(x, mask)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


def build_pooling(name: str, input_dim: int) -> nn.Module:
    if name == "mean":
        return MeanPooling()
    if name == "mean_std":
        return MeanStdPooling()
    if name == "attentive":
        return AttentivePooling(input_dim)
    raise ValueError(f"Unsupported pooling: {name}")
