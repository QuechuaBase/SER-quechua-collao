#!/usr/bin/env python3
"""Summarize six SER-only leave-one-actor-out folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SER-only CV results.")
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=("vad_regression", "emotion_classification"),
    )
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    root = args.results_dir.expanduser().resolve()
    output = args.output_path.expanduser().resolve()
    rows = []
    aggregate = None
    class_names = None
    for fold in range(1, 7):
        path = root / f"fold_{fold}" / "metrics.json"
        if not path.is_file():
            parser.error(f"Missing metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {"fold": fold}
        for metric in METRICS[args.task]:
            row[metric] = float(metrics[metric])
        rows.append(row)
        if args.task == "emotion_classification":
            matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            aggregate = matrix if aggregate is None else aggregate + matrix
            class_names = metrics["class_names"]

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
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    folds.to_csv(output.with_name(f"{output.stem}_folds.csv"), index=False)
    if aggregate is not None:
        pd.DataFrame(aggregate, index=class_names, columns=class_names).to_csv(
            output.with_name("ser_only_emotion_confusion_matrix.csv")
        )
    print(f"Summary saved to: {output}")


if __name__ == "__main__":
    main()
