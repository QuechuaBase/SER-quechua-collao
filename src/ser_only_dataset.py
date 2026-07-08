"""Raw-audio dataset for SER-only training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from .ser_only_features import LogMelFeatureExtractor
from .ser_only_models import EMOTION_CLASSES


def load_audio_safe(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio as float32 [channels, samples], with a soundfile fallback."""
    path = Path(audio_path)
    try:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.to(dtype=torch.float32)
    except Exception:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                "Audio loading failed with torchaudio and soundfile is not installed. "
                "Install requirements.txt to enable the fallback."
            ) from exc

        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            # soundfile returns [samples, channels].
            waveform = waveform.transpose(0, 1)
        else:
            raise ValueError(
                f"Unsupported audio shape from soundfile for {path}: "
                f"{tuple(waveform.shape)}"
            )

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(
            f"Expected audio shaped [channels, samples], got {tuple(waveform.shape)}"
        )
    return waveform.contiguous(), int(sample_rate)


class SEROnlyDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        task: str,
        max_duration_seconds: float = 12.0,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rows = pd.read_csv(self.csv_path)
        self.task = task
        self.sample_rate = sample_rate
        self.max_samples = int(round(max_duration_seconds * sample_rate))
        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=sample_rate, n_mels=n_mels
        )
        self.emotion_to_id = {name: index for index, name in enumerate(EMOTION_CLASSES)}
        required = {
            "audio_path", "utterance_id", "actor", "emotion",
            "valence", "arousal", "dominance",
        }
        if missing := required - set(self.rows.columns):
            raise ValueError(f"{self.csv_path} missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_waveform(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = load_audio_safe(audio_path)
        waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        valid_samples = min(waveform.numel(), self.max_samples)
        waveform = waveform[: self.max_samples]
        if waveform.numel() < self.max_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_samples - waveform.numel())
            )
        return waveform, valid_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        audio_path = Path(str(row["audio_path"])).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        waveform, valid_samples = self._load_waveform(audio_path)
        log_mel, mask = self.feature_extractor(waveform, valid_samples)
        if self.task == "vad_regression":
            target = torch.tensor(
                [row["arousal"], row["valence"], row["dominance"]],
                dtype=torch.float32,
            )
        else:
            emotion = str(row["emotion"]).strip().lower()
            if emotion not in self.emotion_to_id:
                raise ValueError(f"Unknown emotion: {emotion}")
            target = torch.tensor(self.emotion_to_id[emotion], dtype=torch.long)
        return {
            "log_mel": log_mel,
            "mask": mask,
            "target": target,
            "utterance_id": str(row["utterance_id"]),
            "actor": str(row["actor"]),
            "audio_path": str(audio_path),
        }


def collate_ser_only(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "log_mel": torch.stack([item["log_mel"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
        "targets": torch.stack([item["target"] for item in items]),
        "utterance_id": [item["utterance_id"] for item in items],
        "actor": [item["actor"] for item in items],
        "audio_path": [item["audio_path"] for item in items],
    }
