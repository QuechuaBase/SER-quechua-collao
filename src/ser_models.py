"""Small SER heads trained on frozen XLS-R frame representations."""

from __future__ import annotations

import torch
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


class _SERHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pooling: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pooling_name = pooling
        self.pooling = build_pooling(pooling, input_dim)
        pooled_dim = input_dim * getattr(self.pooling, "output_multiplier", 1)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.classifier(self.pooling(hidden_states, mask))


class VADRegressionHead(_SERHead):
    def __init__(self, input_dim: int = 1024, pooling: str = "attentive") -> None:
        super().__init__(input_dim, 3, pooling)


class EmotionClassificationHead(_SERHead):
    def __init__(self, input_dim: int = 1024, pooling: str = "attentive") -> None:
        super().__init__(input_dim, len(EMOTION_CLASSES), pooling)


def build_ser_model(task: str, input_dim: int, pooling: str) -> nn.Module:
    if task == "vad_regression":
        return VADRegressionHead(input_dim=input_dim, pooling=pooling)
    if task == "emotion_classification":
        return EmotionClassificationHead(input_dim=input_dim, pooling=pooling)
    raise ValueError(f"Unsupported task: {task}")
