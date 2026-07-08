"""Data validation and merging helpers for eGeMAPS baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .egemaps_metrics import EMOTION_CLASSES, VAD_TARGETS


METADATA_COLUMNS = [
    "utterance_id",
    "audio_path",
    "actor",
    "emotion",
    "arousal",
    "valence",
    "dominance",
    "duration",
]

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


def validate_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    """Raise a useful error when required columns are absent."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalize_emotion_label(value: object) -> str:
    """Normalize known corpus emotion aliases to the canonical nine labels."""
    key = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = EMOTION_ALIASES.get(key, key)
    if normalized not in EMOTION_CLASSES:
        raise ValueError(f"Unsupported emotion label: {value!r} -> {normalized!r}")
    return normalized


def load_features(path: Path) -> pd.DataFrame:
    """Load and validate one-row-per-utterance eGeMAPS features."""
    features_path = path.expanduser().resolve()
    if not features_path.is_file():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")
    frame = pd.read_csv(features_path)
    validate_columns(frame, METADATA_COLUMNS, str(features_path))
    if frame["utterance_id"].isna().any():
        raise ValueError("Features CSV contains missing utterance_id values")
    if frame["utterance_id"].duplicated().any():
        duplicates = frame.loc[
            frame["utterance_id"].duplicated(keep=False), "utterance_id"
        ].astype(str).tolist()
        raise ValueError(f"Features CSV contains duplicate utterance IDs: {duplicates[:10]}")
    return frame


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select numeric eGeMAPS columns while excluding manifest metadata."""
    candidates = [column for column in frame.columns if column not in METADATA_COLUMNS]
    feature_columns = [
        column for column in candidates if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric eGeMAPS feature columns were found")
    non_numeric = sorted(set(candidates) - set(feature_columns))
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
    if frame[feature_columns].isna().any().any():
        bad = frame[feature_columns].columns[
            frame[feature_columns].isna().any()
        ].tolist()
        raise ValueError(f"eGeMAPS features contain missing values in: {bad[:10]}")
    return feature_columns


def merge_fold_with_features(
    fold_rows: pd.DataFrame,
    features: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Merge a fold split with extracted features by utterance ID."""
    validate_columns(fold_rows, ["utterance_id", "actor"], split_name)
    if fold_rows["utterance_id"].duplicated().any():
        raise ValueError(f"{split_name} contains duplicate utterance IDs")

    feature_columns = select_feature_columns(features)
    feature_subset = features[["utterance_id", *feature_columns]]
    merged = fold_rows.merge(
        feature_subset,
        on="utterance_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_ids = merged.loc[merged["_merge"] != "both", "utterance_id"].astype(str)
    if not missing_ids.empty:
        raise ValueError(
            f"{split_name} has {len(missing_ids)} utterances without features: "
            f"{missing_ids.head(10).tolist()}"
        )
    return merged.drop(columns="_merge")


def load_fold_data(
    fold_dir: Path,
    features_csv: Path,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load a fold, verify actor separation, and merge its eGeMAPS features."""
    directory = fold_dir.expanduser().resolve()
    train_path = directory / "train.csv"
    valid_path = directory / "valid.csv"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(
            f"Fold directory must contain train.csv and valid.csv: {directory}"
        )

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    required_targets = VAD_TARGETS if task == "vad_regression" else ["emotion"]
    validate_columns(train, ["utterance_id", "actor", *required_targets], str(train_path))
    validate_columns(valid, ["utterance_id", "actor", *required_targets], str(valid_path))

    train_actors = set(train["actor"].dropna().astype(str))
    valid_actors = set(valid["actor"].dropna().astype(str))
    leakage = sorted(train_actors & valid_actors)
    if leakage:
        raise ValueError(f"Actor leakage detected between train and valid: {leakage}")
    if len(train_actors) < 2:
        raise ValueError("At least two training actors are required for inner actor CV")
    if not valid_actors:
        raise ValueError("Validation split has no actors")

    if task == "emotion_classification":
        train = train.copy()
        valid = valid.copy()
        train["emotion"] = train["emotion"].map(normalize_emotion_label)
        valid["emotion"] = valid["emotion"].map(normalize_emotion_label)
    else:
        for split_name, frame in (("train", train), ("valid", valid)):
            converted = frame[VAD_TARGETS].apply(pd.to_numeric, errors="coerce")
            if converted.isna().any().any():
                raise ValueError(f"{split_name} contains missing or invalid VAD targets")
            frame[VAD_TARGETS] = converted

    features = load_features(features_csv)
    feature_columns = select_feature_columns(features)
    train_merged = merge_fold_with_features(train, features, "train split")
    valid_merged = merge_fold_with_features(valid, features, "valid split")
    return train_merged, valid_merged, feature_columns
