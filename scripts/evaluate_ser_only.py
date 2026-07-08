#!/usr/bin/env python3
"""Evaluate one trained SER-only fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.ser_only_dataset import SEROnlyDataset, collate_ser_only
from src.ser_only_models import EMOTION_CLASSES, LogMelSERModel
from train_ser_only import load_and_verify_fold, run_inference, select_device


def save_confusion(metrics: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    pd.DataFrame(matrix, index=EMOTION_CLASSES, columns=EMOTION_CLASSES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(9), EMOTION_CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(9), EMOTION_CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(9):
        for column in range(9):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best SER-only checkpoint.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = select_device(args.device)
    model_path = args.model_path.expanduser().resolve()
    fold_dir = args.fold_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    train_rows, _ = load_and_verify_fold(fold_dir)
    del train_rows
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    dataset = SEROnlyDataset(
        fold_dir / "valid.csv",
        checkpoint["task"],
        max_duration_seconds=float(config["max_duration_seconds"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ser_only,
    )
    model = LogMelSERModel(
        task=checkpoint["task"],
        hidden_dim=int(checkpoint["hidden_dim"]),
        pooling=checkpoint["pooling"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, predictions = run_inference(
        model, loader, checkpoint["task"], device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if checkpoint["task"] == "emotion_classification":
        save_confusion(metrics, output_dir)
    print(f"Evaluation saved to: {output_dir}")


if __name__ == "__main__":
    main()
