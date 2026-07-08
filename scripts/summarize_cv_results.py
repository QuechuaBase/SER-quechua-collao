#!/usr/bin/env python3
"""Aggregate metrics across six leave-one-actor-out folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_METRICS = {
    "vad_regression": [
        "ccc_arousal",
        "ccc_valence",
        "ccc_dominance",
        "mean_ccc",
        "mae",
        "mse",
    ],
    "emotion_classification": [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize six-fold SER CV results.")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=("vad_regression", "emotion_classification"),
    )
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    fold_metrics = []
    aggregate_confusion = None
    class_names = None
    for fold_index in range(1, 7):
        metrics_path = results_dir / f"fold_{fold_index}" / "metrics.json"
        if not metrics_path.is_file():
            parser.error(f"Missing fold metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"fold": fold_index}
        for name in TASK_METRICS[args.task]:
            if name not in metrics:
                parser.error(f"Metric {name} missing from {metrics_path}")
            row[name] = float(metrics[name])
        fold_metrics.append(row)
        if args.task == "emotion_classification":
            matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            aggregate_confusion = (
                matrix if aggregate_confusion is None else aggregate_confusion + matrix
            )
            class_names = metrics["class_names"]

    folds = pd.DataFrame(fold_metrics)
    summary_rows = []
    for metric in TASK_METRICS[args.task]:
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(folds[metric].mean()),
                "std": float(folds[metric].std(ddof=1)),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    folds.to_csv(output_path.with_name(f"{output_path.stem}_folds.csv"), index=False)
    if aggregate_confusion is not None:
        pd.DataFrame(
            aggregate_confusion, index=class_names, columns=class_names
        ).to_csv(output_path.with_name("emotion_cv_confusion_matrix.csv"))
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
