"""Log-Mel feature extraction for the SER-only baseline."""

from __future__ import annotations

import math

import torch
import torchaudio
from torch import nn


class LogMelFeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(
        self, waveform: torch.Tensor, valid_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform [samples], got {tuple(waveform.shape)}")
        features = torch.log(self.mel(waveform).clamp_min(self.eps)).transpose(0, 1)
        if valid_samples is None:
            valid_frames = features.shape[0]
        else:
            valid_frames = min(
                features.shape[0],
                max(1, math.ceil(valid_samples / self.hop_length) + 1),
            )
        mask = torch.arange(features.shape[0]) < valid_frames
        valid = features[mask]
        mean = valid.mean(dim=0, keepdim=True)
        std = valid.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        features = (features - mean) / std
        features[~mask] = 0.0
        return features, mask
