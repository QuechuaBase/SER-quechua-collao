#!/usr/bin/env python3
"""Create a compact comparison table across SER systems."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "ccc_arousal",
    "ccc_valence",
    "ccc_dominance",
    "mean_ccc",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]


def read_summary(path: Path) -> dict[str, float | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result summary: {path}")
    table = pd.read_csv(path)
    if not {"metric", "mean"}.issubset(table.columns):
        raise ValueError(f"Summary lacks metric/mean columns: {path}")
    return {str(row["metric"]): float(row["mean"]) for row in table.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing and vanilla XLS-R SER summaries.")
    parser.add_argument("--artifacts_results_dir", type=Path, default=Path("results"))
    parser.add_argument("--vanilla_results_dir", type=Path, default=Path("results/vanilla_xlsr"))
    parser.add_argument("--output_path", type=Path, default=Path("results/vanilla_xlsr/comparison_with_existing_systems.csv"))
    args = parser.parse_args()

    artifacts = args.artifacts_results_dir.expanduser().resolve()
    vanilla = args.vanilla_results_dir.expanduser().resolve()
    systems = [
        (
            "eGeMAPS",
            artifacts / "egemaps_vad_cv_summary.csv",
            artifacts / "egemaps_emotion_cv_summary.csv",
        ),
        (
            "SER-only neural",
            artifacts / "ser_only_vad_cv_summary.csv",
            artifacts / "ser_only_emotion_cv_summary.csv",
        ),
        (
            "vanilla XLS-R frozen",
            vanilla / "vad_cv_summary.csv",
            vanilla / "emotion_cv_summary.csv",
        ),
        (
            "Puno Quechua ASR-SER",
            artifacts / "vad_cv_summary.csv",
            artifacts / "emotion_cv_summary.csv",
        ),
    ]
    rows = []
    for system, vad_path, emotion_path in systems:
        values = {**read_summary(vad_path), **read_summary(emotion_path)}
        row = {"system": system}
        row.update({metric: values.get(metric) for metric in METRICS})
        rows.append(row)

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Comparison: {output_path}")


if __name__ == "__main__":
    main()
