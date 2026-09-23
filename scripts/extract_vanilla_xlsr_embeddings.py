#!/usr/bin/env python3
"""Extract frozen frame-level embeddings from vanilla XLS-R 300M for SER."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_utils import load_audio_mono


MODEL_ID = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000


def select_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def model_commit_hash(model: Any, feature_extractor: Any) -> str | None:
    for owner in (model.config, feature_extractor):
        value = getattr(owner, "_commit_hash", None)
        if value:
            return str(value)
    return None


def valid_output_frames(model: Any, valid_samples: int, total_frames: int) -> int:
    output_lengths = model._get_feat_extract_output_lengths(  # noqa: SLF001
        valid_samples,
        add_adapter=False,
    )
    if hasattr(output_lengths, "item"):
        frames = int(output_lengths.item())
    else:
        frames = int(output_lengths)
    return max(1, min(total_frames, frames))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one frozen vanilla XLS-R 300M frame embedding file per SER utterance."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--revision", default="fdca614bc5b1534b850bccd3fabca0489ea723d7")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_duration_seconds", type=float, default=12.0)
    parser.add_argument("--max_utterances", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    if args.max_duration_seconds <= 0:
        parser.error("--max_duration_seconds must be positive")
    if args.max_utterances is not None and args.max_utterances <= 0:
        parser.error("--max_utterances must be positive when provided")

    try:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Model

        device = select_device(args.device, torch)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_id, revision=args.revision
        )
        model = Wav2Vec2Model.from_pretrained(
            args.model_id, revision=args.revision, use_safetensors=True
        )
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        hidden_size = int(model.config.hidden_size)
        manifest = pd.read_csv(manifest_path)
        required = {
            "audio_path",
            "utterance_id",
            "actor",
            "emotion",
            "valence",
            "arousal",
            "dominance",
            "duration",
        }
        if missing := required - set(manifest.columns):
            parser.error(f"Manifest missing columns: {sorted(missing)}")
        if manifest[list(required)].isna().any().any():
            missing_counts = manifest[list(required)].isna().sum()
            parser.error(
                "Manifest contains missing values: "
                + ", ".join(
                    f"{name}={int(count)}"
                    for name, count in missing_counts.items()
                    if int(count) > 0
                )
            )

        if args.max_utterances is not None:
            manifest = manifest.head(args.max_utterances).copy()

        output_dir.mkdir(parents=True, exist_ok=True)
        max_samples = int(round(args.max_duration_seconds * SAMPLE_RATE))
        metadata_rows: list[dict[str, Any]] = []
        observed_dims: set[int] = set()
        observed_frame_counts: list[int] = []

        for row in manifest.to_dict("records"):
            audio_path = Path(str(row["audio_path"])).expanduser()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            embedding_path = output_dir / f"{row['utterance_id']}.pt"
            if embedding_path.exists() and not args.overwrite:
                try:
                    payload = torch.load(
                        embedding_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    payload = torch.load(embedding_path, map_location="cpu")
                hidden = payload["hidden_states"] if isinstance(payload, dict) else payload
                observed_dims.add(int(hidden.shape[-1]))
                observed_frame_counts.append(int(hidden.shape[-2]))
                metadata_rows.append(
                    {**row, "embedding_path": str(embedding_path.resolve())}
                )
                continue

            audio, _ = load_audio_mono(audio_path, SAMPLE_RATE)
            audio = audio[:max_samples]
            valid_samples = int(len(audio))
            if valid_samples <= 0:
                raise RuntimeError(f"Audio has no samples after loading: {audio_path}")
            inputs = feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs["input_values"].to(device)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            with torch.inference_mode():
                output = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = output.last_hidden_state[0]
                frames = valid_output_frames(model, valid_samples, hidden.shape[0])
                hidden = hidden[:frames].cpu()
            if hidden.ndim != 2:
                raise RuntimeError(
                    f"Expected [T,D] hidden states for {row['utterance_id']}, got {tuple(hidden.shape)}"
                )
            if int(hidden.shape[-1]) != hidden_size:
                raise RuntimeError(
                    f"Config hidden_size={hidden_size}, observed embedding dim={hidden.shape[-1]}"
                )

            payload = {
                "hidden_states": hidden,
                "utterance_id": str(row["utterance_id"]),
                "audio_path": str(audio_path.resolve()),
                "duration": float(row["duration"]),
                "actor": str(row["actor"]),
                "emotion": str(row["emotion"]),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
                "model_id": args.model_id,
                "model_revision": args.revision,
                "sample_rate": SAMPLE_RATE,
                "layer": "last_hidden_state",
                "embedding_dim": hidden_size,
                "encoder_frozen": True,
            }
            torch.save(payload, embedding_path)
            observed_dims.add(int(hidden.shape[-1]))
            observed_frame_counts.append(int(hidden.shape[0]))
            metadata_rows.append({**payload, "embedding_path": str(embedding_path.resolve())})
            metadata_rows[-1].pop("hidden_states")
            print(f"{row['utterance_id']}: {tuple(hidden.shape)}")

        fields = [
            "embedding_path",
            "audio_path",
            "utterance_id",
            "actor",
            "emotion",
            "valence",
            "arousal",
            "dominance",
            "duration",
        ]
        with (output_dir / "embeddings_metadata.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [{field: row[field] for field in fields} for row in metadata_rows]
            )

        extraction_metadata = {
            "model_id": args.model_id,
            "model_revision": args.revision,
            "model_commit_hash": model_commit_hash(model, feature_extractor),
            "sample_rate": SAMPLE_RATE,
            "layer": "last_hidden_state",
            "embedding_dim": hidden_size,
            "encoder_frozen": True,
            "feature_extractor_class": feature_extractor.__class__.__name__,
            "model_class": model.__class__.__name__,
            "transformers_config": {
                "hidden_size": int(model.config.hidden_size),
                "num_hidden_layers": int(model.config.num_hidden_layers),
                "num_attention_heads": int(model.config.num_attention_heads),
                "feat_extract_norm": str(getattr(model.config, "feat_extract_norm", "")),
            },
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
            "utterances": len(metadata_rows),
            "observed_embedding_dims": sorted(observed_dims),
            "min_frames": min(observed_frame_counts) if observed_frame_counts else None,
            "max_frames": max(observed_frame_counts) if observed_frame_counts else None,
            "max_duration_seconds": args.max_duration_seconds,
            "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "extraction_metadata.json").write_text(
            json.dumps(extraction_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Metadata: {output_dir / 'embeddings_metadata.csv'}")
        print(f"Extraction metadata: {output_dir / 'extraction_metadata.json'}")
    except Exception as exc:
        print(f"ERROR: vanilla XLS-R extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
