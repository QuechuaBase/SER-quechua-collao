"""Losses for categorical and dimensional SER."""

from __future__ import annotations

import torch


def ccc_score(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    pred_mean = prediction.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    covariance = (pred_centered * target_centered).mean(dim=0)
    pred_variance = pred_centered.square().mean(dim=0)
    target_variance = target_centered.square().mean(dim=0)
    return (2.0 * covariance) / (
        pred_variance
        + target_variance
        + (pred_mean - target_mean).square()
        + eps
    )


def ccc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ccc_score(prediction, target)


def mean_ccc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ccc_loss(prediction, target).mean()
