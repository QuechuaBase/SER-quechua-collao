#!/usr/bin/env python3
"""Build the merged categorical + dimensional Quechua Collao SER manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


EMOTION_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "boredom": "bored",
    "bored": "bored",
    "calmness": "calm",
    "calm": "calm",
    "excitement": "excited",
    "excited": "excited",
    "fearful": "fear",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "neutrality": "neutral",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "sleepiness": "sleepy",
    "sleepy": "sleepy",
}
EXPECTED_EMOTIONS = {
    "angry", "bored", "calm", "excited", "fear",
    "happy", "neutral", "sad", "sleepy",
}


def normalized_key(value: object) -> str:
    return Path(str(value).strip()).stem.casefold()


def normalize_emotion(value: object) -> str:
    key = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = EMOTION_ALIASES.get(key, key)
    if normalized not in EXPECTED_EMOTIONS:
        raise ValueError(f"Unsupported emotion label: {value!r} -> {normalized!r}")
    return normalized


def resolve_audio(corpus_root: Path, file_value: object, audio_value: object) -> Path | None:
    file_text = "" if pd.isna(file_value) else str(file_value).strip()
    audio_text = "" if pd.isna(audio_value) else str(audio_value).strip()
    candidates: list[Path] = []
    for value in (file_text, audio_text):
        if not value:
            continue
        supplied = Path(value)
        candidates.extend(
            [
                supplied if supplied.is_absolute() else corpus_root / supplied,
                corpus_root / "Data" / supplied,
                corpus_root / "Audio" / supplied,
                corpus_root / "Audios" / supplied,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    names = {Path(value).name for value in (file_text, audio_text) if value}
    stems = {Path(value).stem.casefold() for value in (file_text, audio_text) if value}
    matches = [
        path
        for path in corpus_root.rglob("*")
        if path.is_file()
        and (path.name in names or path.stem.casefold() in stems)
        and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ]
    return matches[0].resolve() if len(matches) == 1 else None


def safe_utterance_id(audio_value: object, index: int) -> str:
    base = Path(str(audio_value)).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
    return cleaned or f"utterance_{index:05d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Quechua Collao categorical and VAD metadata into one SER manifest."
    )
    parser.add_argument("--corpus_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    root = args.corpus_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    categorical_path = root / "Data" / "Data.xlsx"
    clean_labels = root / "Labels" / "Labels" / "Labels_clean.csv"
    fallback_labels = root / "Labels" / "Labels" / "Labels.csv"
    labels_path = clean_labels if clean_labels.is_file() else fallback_labels
    if not categorical_path.is_file():
        parser.error(f"Categorical metadata not found: {categorical_path}")
    if not labels_path.is_file():
        parser.error(f"Dimensional labels not found: {clean_labels} or {fallback_labels}")

    categorical = pd.read_excel(categorical_path, sheet_name="map")
    dimensional = pd.read_csv(labels_path)
    required_cat = {"Audio", "Emotion", "Actor", "File", "Duration (s)"}
    required_dim = {"Audio", "Valence", "Arousal", "Dominance"}
    if missing := required_cat - set(categorical.columns):
        parser.error(f"Missing columns in Data.xlsx/map: {sorted(missing)}")
    if missing := required_dim - set(dimensional.columns):
        parser.error(f"Missing columns in labels CSV: {sorted(missing)}")

    categorical = categorical.copy()
    dimensional = dimensional.copy()
    categorical["_audio_key"] = categorical["Audio"].map(normalized_key)
    dimensional["_audio_key"] = dimensional["Audio"].map(normalized_key)
    if categorical["_audio_key"].duplicated().any():
        parser.error("Duplicate Audio identifiers in Data.xlsx/map")
    if dimensional["_audio_key"].duplicated().any():
        parser.error("Duplicate Audio identifiers in dimensional labels")
    merged = categorical.merge(
        dimensional[["_audio_key", "Valence", "Arousal", "Dominance"]],
        on="_audio_key",
        how="left",
        validate="one_to_one",
    )

    records: list[dict[str, object]] = []
    missing_audio: list[str] = []
    for index, row in merged.iterrows():
        audio_path = resolve_audio(root, row["File"], row["Audio"])
        if audio_path is None:
            missing_audio.append(str(row["Audio"]))
            continue
        records.append(
            {
                "audio_path": str(audio_path),
                "utterance_id": safe_utterance_id(row["Audio"], index),
                "actor": str(row["Actor"]).strip(),
                "emotion": normalize_emotion(row["Emotion"]),
                "valence": pd.to_numeric(row["Valence"], errors="coerce"),
                "arousal": pd.to_numeric(row["Arousal"], errors="coerce"),
                "dominance": pd.to_numeric(row["Dominance"], errors="coerce"),
                "duration": pd.to_numeric(row["Duration (s)"], errors="coerce"),
            }
        )

    manifest = pd.DataFrame(records)
    if not manifest.empty and manifest["utterance_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["utterance_id"].duplicated(keep=False), "utterance_id"
        ].tolist()
        parser.error(f"Duplicate utterance IDs after normalization: {duplicates[:10]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    missing_counts = {
        column: int(manifest[column].isna().sum())
        for column in manifest.columns
    }
    summary = {
        "corpus_root": str(root),
        "categorical_metadata": str(categorical_path),
        "dimensional_metadata": str(labels_path),
        "total_audios": int(len(manifest)),
        "duration_total_seconds": float(manifest["duration"].fillna(0).sum()),
        "duration_total_hours": float(manifest["duration"].fillna(0).sum() / 3600),
        "distribution_by_actor": dict(Counter(manifest["actor"].astype(str))),
        "distribution_by_emotion": dict(Counter(manifest["emotion"].astype(str))),
        "missing_values": missing_counts,
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:50],
        "categorical_rows": int(len(categorical)),
        "dimensional_rows": int(len(dimensional)),
    }
    summary_path = output_path.with_name("ser_manifest_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {output_path} ({len(manifest):,} rows)")
    print(f"Summary: {summary_path}")
    print(f"Missing audio files: {len(missing_audio):,}")


if __name__ == "__main__":
    main()
