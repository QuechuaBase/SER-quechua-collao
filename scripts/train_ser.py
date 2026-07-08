#!/usr/bin/env python3
"""Train a small SER head on frozen precomputed ASR embeddings."""

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

from src.ser_dataset import (
    SEREmbeddingDataset,
    collate_ser_batch,
    merge_fold_with_embeddings,
)
from src.ser_losses import mean_ccc_loss
from src.ser_metrics import classification_metrics, vad_metrics
from src.ser_models import EMOTION_CLASSES, VAD_OUTPUT_ORDER, build_ser_model


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


def verify_no_actor_leakage(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    leakage = sorted(set(train["actor"].astype(str)) & set(valid["actor"].astype(str)))
    if leakage:
        raise RuntimeError(f"Actor leakage detected between train and valid: {leakage}")


def class_weights(rows: pd.DataFrame) -> torch.Tensor:
    counts = rows["emotion"].value_counts()
    total = len(rows)
    weights = [
        total / (len(EMOTION_CLASSES) * max(int(counts.get(name, 0)), 1))
        for name in EMOTION_CLASSES
    ]
    return torch.tensor(weights, dtype=torch.float32)


def run_inference(
    model: nn.Module, loader: DataLoader, task: str, device: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    outputs, targets = [], []
    utterance_ids, actors, audio_paths = [], [], []
    with torch.inference_mode():
        for batch in loader:
            predictions = model(
                batch["hidden_states"].to(device), batch["mask"].to(device)
            )
            outputs.append(predictions.cpu())
            targets.append(batch["targets"].cpu())
            utterance_ids.extend(batch["utterance_id"])
            actors.extend(batch["actor"])
            audio_paths.extend(batch["audio_path"])
    output = torch.cat(outputs).numpy()
    target = torch.cat(targets).numpy()
    base = {
        "utterance_id": utterance_ids,
        "actor": actors,
        "audio_path": audio_paths,
    }
    if task == "vad_regression":
        metrics = vad_metrics(output, target)
        for index, name in enumerate(VAD_OUTPUT_ORDER):
            base[f"true_{name}"] = target[:, index]
            base[f"pred_{name}"] = output[:, index]
    else:
        predicted_ids = output.argmax(axis=1)
        metrics = classification_metrics(predicted_ids, target)
        base["true_id"] = target
        base["pred_id"] = predicted_ids
        base["true_emotion"] = [EMOTION_CLASSES[index] for index in target]
        base["pred_emotion"] = [EMOTION_CLASSES[index] for index in predicted_ids]
    return metrics, pd.DataFrame(base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SER head on frozen embeddings.")
    parser.add_argument(
        "--task",
        required=True,
        choices=("vad_regression", "emotion_classification"),
    )
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--embeddings_metadata", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--pooling", choices=("mean", "mean_std", "attentive"), default="attentive"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    fold_dir = args.fold_dir.expanduser().resolve()
    metadata_path = args.embeddings_metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    train = merge_fold_with_embeddings(fold_dir / "train.csv", metadata_path)
    valid = merge_fold_with_embeddings(fold_dir / "valid.csv", metadata_path)
    verify_no_actor_leakage(train, valid)
    required_targets = (
        ["arousal", "valence", "dominance"]
        if args.task == "vad_regression"
        else ["emotion"]
    )
    if train[required_targets].isna().any().any() or valid[required_targets].isna().any().any():
        raise ValueError(f"Missing target values for task {args.task}")

    train_dataset = SEREmbeddingDataset(train, args.task)
    valid_dataset = SEREmbeddingDataset(valid, args.task)
    first_hidden = train_dataset[0]["hidden_states"]
    input_dim = int(first_hidden.shape[-1])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ser_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ser_batch,
    )
    model = build_ser_model(args.task, input_dim, args.pooling).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.task == "emotion_classification":
        weights = class_weights(train).to(device) if args.use_class_weights else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        selection_metric = "macro_f1"
    else:
        criterion = mean_ccc_loss
        selection_metric = "mean_ccc"

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "fold_dir": str(fold_dir),
        "embeddings_metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "device_resolved": device,
        "input_dim": input_dim,
        "selection_metric": selection_metric,
        "emotion_classes": EMOTION_CLASSES,
        "vad_output_order": VAD_OUTPUT_ORDER,
        "encoder_frozen": True,
        "actor_leakage_verified": True,
        "train_actors": sorted(train["actor"].astype(str).unique().tolist()),
        "valid_actors": sorted(valid["actor"].astype(str).unique().tolist()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    best_score = -float("inf")
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["hidden_states"].to(device), batch["mask"].to(device)
            )
            target = batch["targets"].to(device)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _ = run_inference(model, valid_loader, args.task, device)
        score = float(metrics[selection_metric])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        log_rows.append(row)
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "task": args.task,
            "pooling": args.pooling,
            "input_dim": input_dim,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        }
        torch.save(checkpoint_payload, output_dir / "last_model.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint_payload, output_dir / "best_model.pt")
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"{selection_metric}={score:.6f}"
        )

    all_fields = sorted({key for row in log_rows for key in row})
    with (output_dir / "training_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(log_rows)

    try:
        best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
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
