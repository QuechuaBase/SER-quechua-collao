"""Datasets and collation for precomputed frame-level SER embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .ser_models import EMOTION_CLASSES


class SEREmbeddingDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, task: str) -> None:
        self.rows = rows.reset_index(drop=True)
        self.task = task
        self.emotion_to_id = {name: index for index, name in enumerate(EMOTION_CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        try:
            payload = torch.load(
                Path(row["embedding_path"]), map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(Path(row["embedding_path"]), map_location="cpu")
        hidden = payload["hidden_states"] if isinstance(payload, dict) else payload
        hidden = hidden.float()
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        if hidden.ndim != 2:
            raise ValueError(
                f"Expected [T,D] embedding for {row['utterance_id']}, got {tuple(hidden.shape)}"
            )
        item = {
            "hidden_states": hidden,
            "utterance_id": str(row["utterance_id"]),
            "actor": str(row["actor"]),
            "audio_path": str(row["audio_path"]),
        }
        if self.task == "vad_regression":
            item["target"] = torch.tensor(
                [row["arousal"], row["valence"], row["dominance"]],
                dtype=torch.float32,
            )
        else:
            emotion = str(row["emotion"]).lower()
            if emotion not in self.emotion_to_id:
                raise ValueError(f"Unknown emotion label: {emotion}")
            item["target"] = torch.tensor(self.emotion_to_id[emotion], dtype=torch.long)
        return item


def collate_ser_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["hidden_states"].shape[0] for item in items])
    hidden = pad_sequence(
        [item["hidden_states"] for item in items], batch_first=True
    )
    time = torch.arange(hidden.shape[1]).unsqueeze(0)
    mask = time < lengths.unsqueeze(1)
    targets = torch.stack([item["target"] for item in items])
    return {
        "hidden_states": hidden,
        "mask": mask,
        "targets": targets,
        "utterance_id": [item["utterance_id"] for item in items],
        "actor": [item["actor"] for item in items],
        "audio_path": [item["audio_path"] for item in items],
    }


def merge_fold_with_embeddings(
    split_path: str | Path, embeddings_metadata_path: str | Path
) -> pd.DataFrame:
    split = pd.read_csv(split_path)
    embeddings = pd.read_csv(embeddings_metadata_path)
    if split["utterance_id"].duplicated().any():
        raise ValueError(f"Duplicate utterance_id in split: {split_path}")
    if embeddings["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance_id in embeddings metadata")
    merged = split.merge(
        embeddings[["utterance_id", "embedding_path"]],
        on="utterance_id",
        how="left",
        validate="one_to_one",
    )
    missing = merged["embedding_path"].isna()
    if missing.any():
        examples = merged.loc[missing, "utterance_id"].head().tolist()
        raise ValueError(f"Missing embeddings for {missing.sum()} utterances: {examples}")
    return merged
