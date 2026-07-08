#!/usr/bin/env python3
"""Summarize six eGeMAPS actor-disjoint outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = {
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


def plot_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Aggregated emotion confusion matrix (6 folds)",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate mean and standard deviation over six eGeMAPS folds."
    )
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
    rows: list[dict[str, float | int]] = []
    aggregate_matrix: np.ndarray | None = None
    class_names: list[str] | None = None

    for fold in range(1, 7):
        metrics_path = results_dir / f"fold_{fold}" / "metrics.json"
        if not metrics_path.is_file():
            parser.error(f"Missing metrics file: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        missing = [name for name in METRICS[args.task] if name not in metrics]
        if missing:
            parser.error(f"{metrics_path} is missing metrics: {missing}")
        row: dict[str, float | int] = {"fold": fold}
        row.update(
            {name: float(metrics[name]) for name in METRICS[args.task]}
        )
        rows.append(row)

        if args.task == "emotion_classification":
            matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            names = list(metrics["class_names"])
            if aggregate_matrix is None:
                aggregate_matrix = matrix.copy()
                class_names = names
            else:
                if names != class_names or matrix.shape != aggregate_matrix.shape:
                    parser.error(f"Incompatible confusion matrix in {metrics_path}")
                aggregate_matrix += matrix

    folds = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": float(folds[metric].mean()),
                "std": float(folds[metric].std(ddof=1)),
            }
            for metric in METRICS[args.task]
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    folds.to_csv(output_path.with_name(f"{output_path.stem}_folds.csv"), index=False)

    if aggregate_matrix is not None and class_names is not None:
        matrix_csv = output_path.with_name(
            f"{output_path.stem}_confusion_matrix.csv"
        )
        pd.DataFrame(
            aggregate_matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(matrix_csv, index_label="true\\pred")
        plot_matrix(
            aggregate_matrix,
            class_names,
            output_path.with_name(f"{output_path.stem}_confusion_matrix.png"),
        )
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
