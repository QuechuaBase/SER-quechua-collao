"""Audio loading utilities shared by SER extraction scripts."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio_mono(path: str | Path, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(Path(path), always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        divisor = gcd(int(sample_rate), target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return audio.astype(np.float32, copy=False), target_sample_rate


def crop_or_pad(audio: np.ndarray, max_samples: int) -> tuple[np.ndarray, int]:
    original_samples = min(len(audio), max_samples)
    if len(audio) >= max_samples:
        return audio[:max_samples], original_samples
    return np.pad(audio, (0, max_samples - len(audio))), original_samples
