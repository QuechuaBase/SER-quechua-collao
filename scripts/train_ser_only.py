#!/usr/bin/env python3
"""Train a SER baseline directly from log-Mel spectrograms."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ser_losses import mean_ccc_loss
from src.ser_metrics import classification_metrics, vad_metrics
from src.ser_only_dataset import SEROnlyDataset, collate_ser_only
from src.ser_only_models import EMOTION_CLASSES, VAD_OUTPUT_ORDER, LogMelSERModel


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_verify_fold(fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(fold_dir / "train.csv")
    valid = pd.read_csv(fold_dir / "valid.csv")
    train_actors = set(train["actor"].astype(str))
    valid_actors = set(valid["actor"].astype(str))
    leakage = sorted(train_actors & valid_actors)
    if leakage:
        raise RuntimeError(f"Actor leakage detected: {leakage}")
    return train, valid


def inverse_class_weights(rows: pd.DataFrame) -> torch.Tensor:
    counts = rows["emotion"].value_counts()
    total = len(rows)
    return torch.tensor(
        [
            total / (len(EMOTION_CLASSES) * max(int(counts.get(name, 0)), 1))
            for name in EMOTION_CLASSES
        ],
        dtype=torch.float32,
    )


def run_inference(
    model: nn.Module, loader: DataLoader, task: str, device: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    outputs, targets = [], []
    utterances, actors, paths = [], [], []
    with torch.inference_mode():
        for batch in loader:
            prediction = model(
                batch["log_mel"].to(device), batch["mask"].to(device)
            )
            outputs.append(prediction.cpu())
            targets.append(batch["targets"].cpu())
            utterances.extend(batch["utterance_id"])
            actors.extend(batch["actor"])
            paths.extend(batch["audio_path"])
    output = torch.cat(outputs).numpy()
    target = torch.cat(targets).numpy()
    table: dict[str, Any] = {
        "utterance_id": utterances,
        "actor": actors,
        "audio_path": paths,
    }
    if task == "vad_regression":
        metrics = vad_metrics(output, target)
        for index, name in enumerate(VAD_OUTPUT_ORDER):
            table[f"true_{name}"] = target[:, index]
            table[f"pred_{name}"] = output[:, index]
    else:
        predicted = output.argmax(axis=1)
        metrics = classification_metrics(predicted, target)
        table["true_id"] = target
        table["pred_id"] = predicted
        table["true_emotion"] = [EMOTION_CLASSES[index] for index in target]
        table["pred_emotion"] = [EMOTION_CLASSES[index] for index in predicted]
    return metrics, pd.DataFrame(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a SER-only CNN-BiLSTM model from raw audio."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=("vad_regression", "emotion_classification"),
    )
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pooling", choices=("mean", "attentive"), default="attentive")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_duration_seconds", type=float, default=12.0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    fold_dir = args.fold_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train_rows, valid_rows = load_and_verify_fold(fold_dir)
    target_columns = (
        ["arousal", "valence", "dominance"]
        if args.task == "vad_regression"
        else ["emotion"]
    )
    if train_rows[target_columns].isna().any().any():
        raise ValueError("Training split contains missing labels")
    if valid_rows[target_columns].isna().any().any():
        raise ValueError("Validation split contains missing labels")

    train_dataset = SEROnlyDataset(
        fold_dir / "train.csv",
        args.task,
        max_duration_seconds=args.max_duration_seconds,
    )
    valid_dataset = SEROnlyDataset(
        fold_dir / "valid.csv",
        args.task,
        max_duration_seconds=args.max_duration_seconds,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ser_only,
        pin_memory=device == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ser_only,
        pin_memory=device == "cuda",
    )
    model = LogMelSERModel(
        task=args.task,
        hidden_dim=args.hidden_dim,
        pooling=args.pooling,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.task == "emotion_classification":
        weights = (
            inverse_class_weights(train_rows).to(device)
            if args.use_class_weights
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        selection_metric = "macro_f1"
    else:
        criterion = mean_ccc_loss
        selection_metric = "mean_ccc"

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "fold_dir": str(fold_dir),
        "output_dir": str(output_dir),
        "device_resolved": device,
        "sample_rate": 16000,
        "n_mels": 80,
        "n_fft": 400,
        "hop_length": 160,
        "win_length": 400,
        "selection_metric": selection_metric,
        "emotion_classes": EMOTION_CLASSES,
        "vad_output_order": VAD_OUTPUT_ORDER,
        "uses_asr": False,
        "uses_precomputed_embeddings": False,
        "actor_leakage_verified": True,
        "train_actors": sorted(train_rows["actor"].astype(str).unique().tolist()),
        "valid_actors": sorted(valid_rows["actor"].astype(str).unique().tolist()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    best_score = -float("inf")
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["log_mel"].to(device), batch["mask"].to(device)
            )
            loss = criterion(prediction, batch["targets"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _ = run_inference(model, valid_loader, args.task, device)
        score = float(metrics[selection_metric])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
        logs.append(row)
        payload = {
            "model_state_dict": model.state_dict(),
            "task": args.task,
            "pooling": args.pooling,
            "hidden_dim": args.hidden_dim,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        }
        torch.save(payload, output_dir / "last_model.pt")
        if score > best_score:
            best_score = score
            torch.save(payload, output_dir / "best_model.pt")
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"{selection_metric}={score:.6f}"
        )

    fields = sorted({key for row in logs for key in row})
    with (output_dir / "training_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(logs)
    try:
        best = torch.load(
            output_dir / "best_model.pt", map_location=device, weights_only=False
        )
    except TypeError:
        best = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    metrics, predictions = run_inference(model, valid_loader, args.task, device)
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Best {selection_metric}: {metrics[selection_metric]:.6f}")


if __name__ == "__main__":
    main()
