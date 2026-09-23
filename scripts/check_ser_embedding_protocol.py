#!/usr/bin/env python3
"""Validate SER embedding metadata, labels, shapes, and actor-disjoint folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


REQUIRED_COLUMNS = {
    "embedding_path",
    "audio_path",
    "utterance_id",
    "actor",
    "emotion",
    "valence",
    "arousal",
    "dominance",
    "duration",
}


def load_hidden_shape(path: Path) -> tuple[int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    hidden = payload["hidden_states"] if isinstance(payload, dict) else payload
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    if hidden.ndim != 2:
        raise ValueError(f"Expected [T,D] hidden states in {path}, got {tuple(hidden.shape)}")
    return int(hidden.shape[0]), int(hidden.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SER embedding protocol checks.")
    parser.add_argument("--embeddings_metadata", type=Path, required=True)
    parser.add_argument("--folds_dir", type=Path, required=True)
    parser.add_argument("--expected_dim", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--output_path", type=Path, default=None)
    args = parser.parse_args()

    metadata_path = args.embeddings_metadata.expanduser().resolve()
    folds_dir = args.folds_dir.expanduser().resolve()
    if not metadata_path.is_file():
        parser.error(f"Metadata not found: {metadata_path}")
    if not folds_dir.is_dir():
        parser.error(f"Folds directory not found: {folds_dir}")
    if args.sample_size <= 0:
        parser.error("--sample_size must be positive")

    metadata = pd.read_csv(metadata_path)
    if missing := REQUIRED_COLUMNS - set(metadata.columns):
        parser.error(f"Metadata missing columns: {sorted(missing)}")
    if metadata["utterance_id"].duplicated().any():
        parser.error("Duplicate utterance_id values in embeddings metadata")
    label_columns = ["emotion", "valence", "arousal", "dominance"]
    if metadata[label_columns].isna().any().any():
        parser.error("Embeddings metadata has missing labels")

    shape_rows = []
    for row in metadata.head(args.sample_size).to_dict("records"):
        path = Path(str(row["embedding_path"]))
        if not path.is_file():
            parser.error(f"Embedding file not found: {path}")
        frames, dim = load_hidden_shape(path)
        if args.expected_dim is not None and dim != args.expected_dim:
            parser.error(f"{path} has dim={dim}, expected {args.expected_dim}")
        shape_rows.append(
            {
                "utterance_id": str(row["utterance_id"]),
                "frames": frames,
                "dim": dim,
            }
        )

    fold_rows = []
    for fold_index in range(1, 7):
        fold_dir = folds_dir / f"fold_{fold_index}"
        train_path = fold_dir / "train.csv"
        valid_path = fold_dir / "valid.csv"
        if not train_path.is_file() or not valid_path.is_file():
            parser.error(f"Missing train/valid split in {fold_dir}")
        train = pd.read_csv(train_path)
        valid = pd.read_csv(valid_path)
        leakage = sorted(set(train["actor"].astype(str)) & set(valid["actor"].astype(str)))
        if leakage:
            parser.error(f"Actor leakage in fold_{fold_index}: {leakage}")
        for split_name, split in (("train", train), ("valid", valid)):
            missing_embeddings = sorted(
                set(split["utterance_id"].astype(str))
                - set(metadata["utterance_id"].astype(str))
            )
            if missing_embeddings:
                parser.error(
                    f"Missing embeddings for {len(missing_embeddings)} {split_name} rows "
                    f"in fold_{fold_index}; examples={missing_embeddings[:5]}"
                )
        fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "train_actors": sorted(train["actor"].astype(str).unique().tolist()),
                "valid_actors": sorted(valid["actor"].astype(str).unique().tolist()),
                "actor_disjoint": True,
            }
        )

    report = {
        "embeddings_metadata": args.embeddings_metadata.as_posix(),
        "folds_dir": args.folds_dir.as_posix(),
        "utterances": int(len(metadata)),
        "sampled_shapes": shape_rows,
        "folds": fold_rows,
        "labels_present": True,
        "actor_leakage_detected": False,
    }
    if args.output_path is not None:
        output_path = args.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
