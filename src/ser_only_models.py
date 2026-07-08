"""CNN-BiLSTM models operating directly on log-Mel spectrograms."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .pooling import build_pooling


EMOTION_CLASSES = [
    "angry",
    "bored",
    "calm",
    "excited",
    "fear",
    "happy",
    "neutral",
    "sad",
    "sleepy",
]
VAD_OUTPUT_ORDER = ["arousal", "valence", "dominance"]


class LogMelSERModel(nn.Module):
    def __init__(
        self,
        task: str,
        n_mels: int = 80,
        hidden_dim: int = 256,
        pooling: str = "attentive",
    ) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for the bidirectional LSTM")
        if task not in {"vad_regression", "emotion_classification"}:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling
        self.convolution = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.temporal = nn.LSTM(
            input_size=192,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pooling = build_pooling(pooling, hidden_dim)
        pooled_dim = hidden_dim * getattr(self.pooling, "output_multiplier", 1)
        output_dim = 3 if task == "vad_regression" else len(EMOTION_CLASSES)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(
        self, log_mel: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if log_mel.ndim != 3:
            raise ValueError(f"Expected [B,T,n_mels], got {tuple(log_mel.shape)}")
        encoded = self.convolution(log_mel.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.temporal(encoded)
        downsampled_mask = None
        if mask is not None:
            downsampled_mask = functional.interpolate(
                mask.float().unsqueeze(1),
                size=encoded.shape[1],
                mode="nearest",
            ).squeeze(1).bool()
        return self.head(self.pooling(encoded, downsampled_mask))
