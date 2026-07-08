"""Metrics for the classical eGeMAPS SER baselines."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


VAD_TARGETS = ["arousal", "valence", "dominance"]
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


def ccc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Lin's concordance correlation coefficient."""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.size == 0:
        raise ValueError("CCC requires at least one sample")

    mean_true = np.mean(true)
    mean_pred = np.mean(pred)
    variance_true = np.mean((true - mean_true) ** 2)
    variance_pred = np.mean((pred - mean_pred) ** 2)
    covariance = np.mean((true - mean_true) * (pred - mean_pred))
    denominator = (
        variance_true + variance_pred + (mean_true - mean_pred) ** 2
    )
    if denominator == 0:
        return 1.0 if np.allclose(true, pred) else 0.0
    return float(2.0 * covariance / denominator)


def vad_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] = VAD_TARGETS,
) -> dict[str, float]:
    """Compute per-dimension CCC and aggregate VAD regression metrics."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim == 1:
        true = true[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.shape[1] != len(target_names):
        raise ValueError(
            f"Expected {len(target_names)} VAD columns, found {true.shape[1]}"
        )

    ccc_values = [
        ccc_score(true[:, index], pred[:, index])
        for index in range(len(target_names))
    ]
    metrics = {
        f"ccc_{name}": float(ccc_values[index])
        for index, name in enumerate(target_names)
    }
    metrics.update(
        {
            "mean_ccc": float(np.mean(ccc_values)),
            "mae": float(mean_absolute_error(true, pred)),
            "mse": float(mean_squared_error(true, pred)),
        }
    )
    return metrics


def classification_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str] = EMOTION_CLASSES,
) -> np.ndarray:
    """Return a fixed-order confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(class_names))


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str] = EMOTION_CLASSES,
) -> dict[str, Any]:
    """Compute global and per-class emotion classification metrics."""
    names = list(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=names,
        zero_division=0,
    )
    matrix = classification_confusion_matrix(y_true, y_pred, names)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=names, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=names,
                average="weighted",
                zero_division=0,
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
        "confusion_matrix": matrix.tolist(),
        "class_names": names,
    }
