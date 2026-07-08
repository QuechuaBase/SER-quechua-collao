"""Evaluation metrics for both SER tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from .ser_losses import ccc_score
from .ser_models import EMOTION_CLASSES, VAD_OUTPUT_ORDER


def vad_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    ccc = ccc_score(prediction_tensor, target_tensor).cpu().numpy()
    metrics = {
        f"ccc_{name}": float(ccc[index])
        for index, name in enumerate(VAD_OUTPUT_ORDER)
    }
    metrics.update(
        {
            "mean_ccc": float(np.mean(ccc)),
            "mae": float(mean_absolute_error(targets, predictions)),
            "mse": float(mean_squared_error(targets, predictions)),
        }
    )
    return metrics


def classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    names = class_names or EMOTION_CLASSES
    labels = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                targets, predictions, labels=labels, average="weighted", zero_division=0
            )
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
        "class_names": names,
    }
