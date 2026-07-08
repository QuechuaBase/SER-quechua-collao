#!/usr/bin/env python3
"""Create stable six-fold leave-one-actor-out splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leave-one-actor-out train/valid folds with leakage checks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    data = pd.read_csv(manifest_path)
    if "actor" not in data.columns:
        parser.error("Manifest must contain an actor column")
    actors = sorted(data["actor"].dropna().astype(str).unique().tolist())
    if len(actors) != 6:
        parser.error(
            f"Expected exactly 6 actors for six-fold LOAO, found {len(actors)}: {actors}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for index, validation_actor in enumerate(actors, start=1):
        fold_dir = output_dir / f"fold_{index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        valid = data[data["actor"].astype(str) == validation_actor].copy()
        train = data[data["actor"].astype(str) != validation_actor].copy()
        train_actors = set(train["actor"].astype(str))
        valid_actors = set(valid["actor"].astype(str))
        leakage = sorted(train_actors & valid_actors)
        if leakage or validation_actor in train_actors:
            raise RuntimeError(f"Actor leakage in fold {index}: {leakage}")
        train.to_csv(fold_dir / "train.csv", index=False)
        valid.to_csv(fold_dir / "valid.csv", index=False)
        folds.append(
            {
                "fold": index,
                "validation_actor": validation_actor,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "train_actors": sorted(train_actors),
                "valid_actors": sorted(valid_actors),
                "actor_leakage": False,
            }
        )
        print(
            f"fold_{index}: valid actor={validation_actor}, "
            f"train={len(train):,}, valid={len(valid):,}, leakage=False"
        )

    summary = {
        "protocol": "leave-one-actor-out",
        "actor_count": len(actors),
        "actors_stable_order": actors,
        "fold_count": len(folds),
        "folds": folds,
    }
    (output_dir / "folds_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
